"""
DPT Heatmap Head (Simplified)
==============================

Direct heatmap regression from multi-layer LLM visual tokens.

Architecture:
    1. Extract current-frame visual tokens from 4 intermediate LLM layers
    2. Per-layer: LayerNorm -> reshape to 2D (8x8) -> 1x1 Conv projection
    3. Concat all layers -> fusion conv
    4. 3-stage ConvTranspose upsampling: 8 -> 16 -> 32 -> 64
    5. 1x1 conv -> single-channel logits
    6. Training: KL divergence loss
    7. Inference: softmax -> probability map
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class DPTHeatmapConfig:
    """Configuration for DPTHeatmapHead."""
    dim_in: int = 4096
    heatmap_size: Tuple[int, int] = (64, 64)
    patch_h: int = 8
    patch_w: int = 8
    proj_dim: int = 256
    features: int = 256
    num_image_tokens: int = 64
    num_layers: int = 4


class UpsampleBlock(nn.Module):
    """ConvTranspose 2x upsample + Conv3x3 refinement."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DPTHeatmapHead(nn.Module):
    """
    Simplified heatmap head: concat multi-layer tokens + ConvTranspose decoder.

    Replaces the DPT RefineNet pyramid with a straightforward
    concat-and-upsample architecture. Same forward interface for
    drop-in replacement in the pipeline.
    """

    def __init__(self, config: DPTHeatmapConfig):
        super().__init__()
        self.config = config
        self.heatmap_size = config.heatmap_size
        self.patch_h = config.patch_h
        self.patch_w = config.patch_w
        self.num_image_tokens = config.num_image_tokens
        self._training_step_counter = 0

        dim_in = config.dim_in
        proj_dim = config.proj_dim
        features = config.features
        n_layers = config.num_layers

        # Per-layer: LayerNorm + 1x1 Conv projection
        self.norms = nn.ModuleList([nn.LayerNorm(dim_in) for _ in range(n_layers)])
        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, proj_dim, kernel_size=1) for _ in range(n_layers)
        ])

        # Fusion: concat all layers -> reduce channels
        self.fusion = nn.Sequential(
            nn.Conv2d(proj_dim * n_layers, features, kernel_size=1),
            nn.BatchNorm2d(features),
            nn.GELU(),
        )

        # Decoder: 3-stage upsample  8 -> 16 -> 32 -> 64
        self.decoder = nn.Sequential(
            UpsampleBlock(features, features),
            UpsampleBlock(features, features // 2),
            UpsampleBlock(features // 2, features // 4),
        )

        # Output head
        self.head = nn.Conv2d(features // 4, 1, kernel_size=1)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "DPTHeatmapHead(simplified): dim_in=%d, patch=%dx%d, "
            "proj_dim=%d, features=%d, heatmap=%s, params=%s",
            dim_in, config.patch_h, config.patch_w,
            proj_dim, features, config.heatmap_size, f"{total_params:,}",
        )

    def forward(
        self,
        multi_layer_vision_tokens: List[torch.Tensor],
        gt_heatmap: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        # Unused args kept for interface compatibility
        llm_tokens: Optional[torch.Tensor] = None,
        observation: Optional[torch.Tensor] = None,
        skip_inference: bool = False,
        direction_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        logits = self._decode(multi_layer_vision_tokens)

        if gt_heatmap is not None and return_loss:
            return self._compute_loss(logits, gt_heatmap)

        heatmap = self._logits_to_heatmap(logits)

        if return_loss:
            return {
                'heatmap': heatmap,
                'loss': torch.tensor(0.0, device=heatmap.device),
            }

        return heatmap

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _decode(self, multi_layer_vision_tokens: List[torch.Tensor]) -> torch.Tensor:
        B = multi_layer_vision_tokens[0].shape[0]
        ph, pw = self.patch_h, self.patch_w
        n_img = self.num_image_tokens

        parts = []
        for i, layer_tokens in enumerate(multi_layer_vision_tokens):
            x = layer_tokens[:, -n_img:]              # (B, 64, D)
            x = self.norms[i](x)
            x = x.permute(0, 2, 1).reshape(B, -1, ph, pw)  # (B, D, 8, 8)
            x = self.projects[i](x)                   # (B, proj_dim, 8, 8)
            parts.append(x)

        fused = torch.cat(parts, dim=1)               # (B, 4*proj_dim, 8, 8)
        fused = self.fusion(fused)                     # (B, features, 8, 8)
        fused = self.decoder(fused)                    # (B, features//4, 64, 64)
        logits = self.head(fused).squeeze(1)           # (B, 64, 64)
        return logits

    def _logits_to_heatmap(self, logits: torch.Tensor) -> torch.Tensor:
        B = logits.shape[0]
        probs = F.softmax(logits.view(B, -1), dim=-1)
        probs = probs.view(B, *self.heatmap_size)
        max_val = probs.flatten(1).max(dim=1).values.clamp(min=1e-8)
        return probs / max_val.view(B, 1, 1)

    def _normalize_gt(self, gt_heatmap: torch.Tensor) -> torch.Tensor:
        B = gt_heatmap.shape[0]
        gt_flat = gt_heatmap.view(B, -1).float()
        gt_sum = gt_flat.sum(dim=1, keepdim=True)

        is_negative = (gt_sum < 1e-6)
        uniform = torch.ones_like(gt_flat) / gt_flat.shape[1]

        return torch.where(
            is_negative.expand_as(gt_flat),
            uniform,
            gt_flat / gt_sum.clamp(min=1e-8),
        )

    def _compute_loss(
        self,
        logits: torch.Tensor,
        gt_heatmap: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B = logits.shape[0]
        Hm, Wm = self.heatmap_size

        gt = gt_heatmap
        if gt.dim() == 4:
            gt = gt.squeeze(1)
        if gt.shape[-2:] != (Hm, Wm):
            gt = F.interpolate(
                gt.unsqueeze(1), size=(Hm, Wm),
                mode='bilinear', align_corners=False,
            ).squeeze(1)

        log_q = F.log_softmax(logits.view(B, -1), dim=-1)
        target = self._normalize_gt(gt)
        kl_loss = F.kl_div(log_q, target, reduction="batchmean")

        with torch.no_grad():
            pred_heatmap = self._logits_to_heatmap(logits)

            sample_max = gt.flatten(1).max(dim=1).values
            is_positive = (sample_max > 0.01).float()
            n_pos = is_positive.sum().item()
            n_neg = B - n_pos

        self._training_step_counter += 1

        return {
            'loss': kl_loss,
            'heatmap': pred_heatmap.detach(),
            'dpt_kl_loss': kl_loss.item(),
            'dpt_pred_max': pred_heatmap.max().item(),
            'dpt_pred_mean': pred_heatmap.mean().item(),
            'dpt_n_pos': n_pos,
            'dpt_n_neg': n_neg,
        }

"""
DPT Heatmap Head
================

Dense Prediction Transformer for direct heatmap generation from LLM visual tokens.

Architecture (adapted from Depth-Anything-V2 / xyc):
    1. Extract visual tokens from 4 intermediate LLM layers
    2. Each layer: LayerNorm -> reshape to 2D -> 1x1 Conv projection -> resize
    3. RefineNet hierarchical fusion (deep-to-shallow)
    4. Bilinear upsample to target resolution -> 1-channel logits
    5. Training: KL divergence loss (heatmap as probability distribution)
    6. Inference: softmax -> probability map
"""

import logging
from dataclasses import dataclass, field
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
    out_channels: List[int] = field(default_factory=lambda: [256, 512, 1024, 1024])
    features: int = 256
    num_image_tokens: int = 64


# ============================================================================
# Building blocks
# ============================================================================

class ResidualConvUnit(nn.Module):
    """Residual convolution module for RefineNet."""

    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block for RefineNet."""

    def __init__(self, features: int, has_residual: bool = True):
        super().__init__()
        self.has_residual = has_residual
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, 1, bias=True)

    def forward(self, *xs, size=None) -> torch.Tensor:
        output = xs[0]
        if self.has_residual:
            output = output + self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)

        if size is None:
            modifier = {"scale_factor": 2}
        else:
            modifier = {"size": size}
        output = F.interpolate(output, **modifier, mode="bilinear", align_corners=True)
        output = self.out_conv(output)
        return output


# ============================================================================
# DPT Heatmap Head
# ============================================================================

class DPTHeatmapHead(nn.Module):
    """
    DPT-style head for heatmap prediction from multi-layer LLM visual tokens.

    Same forward interface as DiffusionHeatmapHead / DirectHeatmapHead
    for drop-in replacement in the pipeline.
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
        out_channels = config.out_channels
        features = config.features

        self.norm = nn.LayerNorm(dim_in)

        # 1x1 conv projections for each layer
        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, oc, kernel_size=1) for oc in out_channels
        ])

        # Resize layers to build multi-scale pyramid
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        # Scratch: project all scales to unified feature dim
        self.layer1_rn = nn.Conv2d(out_channels[0], features, 3, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(out_channels[1], features, 3, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(out_channels[2], features, 3, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(out_channels[3], features, 3, padding=1, bias=False)

        # RefineNet fusion blocks (deep to shallow)
        self.refinenet4 = FeatureFusionBlock(features, has_residual=False)
        self.refinenet3 = FeatureFusionBlock(features)
        self.refinenet2 = FeatureFusionBlock(features)
        self.refinenet1 = FeatureFusionBlock(features)

        # Output conv
        self.output_conv1 = nn.Conv2d(features, features, 3, padding=1)

        self.output_conv2 = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, 1, 1),
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "DPTHeatmapHead: dim_in=%d, patch=%dx%d, features=%d, "
            "heatmap=%s, params=%s",
            dim_in, config.patch_h, config.patch_w, features,
            config.heatmap_size, f"{total_params:,}",
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

        logits = self._decode(multi_layer_vision_tokens)  # (B, Hm, Wm)

        if gt_heatmap is not None and return_loss:
            return self._compute_loss(logits, gt_heatmap)

        heatmap = self._logits_to_heatmap(logits)

        if return_loss:
            return {
                'heatmap': heatmap,
                'loss': torch.tensor(0.0, device=heatmap.device),
            }

        return heatmap

    def _decode(self, multi_layer_vision_tokens: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode multi-layer visual tokens into heatmap logits.

        Args:
            multi_layer_vision_tokens: List[4] of (B, num_vision_tokens, D)

        Returns:
            (B, Hm, Wm) raw logits
        """
        B = multi_layer_vision_tokens[0].shape[0]
        patch_h, patch_w = self.patch_h, self.patch_w
        n_img = self.num_image_tokens

        out = []
        for i, layer_tokens in enumerate(multi_layer_vision_tokens):
            # Take the last n_img tokens (current frame's image tokens)
            x = layer_tokens[:, -n_img:]  # (B, n_img, D)
            x = self.norm(x)
            # Reshape to 2D spatial grid
            x = x.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w)  # (B, D, ph, pw)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        # RefineNet hierarchical fusion (deep to shallow)
        layer_1_rn = self.layer1_rn(out[0])
        layer_2_rn = self.layer2_rn(out[1])
        layer_3_rn = self.layer3_rn(out[2])
        layer_4_rn = self.layer4_rn(out[3])

        fused = self.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        fused = self.refinenet3(fused, layer_3_rn, size=layer_2_rn.shape[2:])
        fused = self.refinenet2(fused, layer_2_rn, size=layer_1_rn.shape[2:])
        fused = self.refinenet1(fused, layer_1_rn)

        fused = self.output_conv1(fused)

        Hm, Wm = self.heatmap_size
        fused = F.interpolate(fused, size=(Hm, Wm), mode="bilinear", align_corners=True)

        logits = self.output_conv2(fused).squeeze(1)  # (B, Hm, Wm)
        return logits

    def _logits_to_heatmap(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert raw logits to [0,1] heatmap via softmax + rescale."""
        B = logits.shape[0]
        probs = F.softmax(logits.view(B, -1), dim=-1)
        probs = probs.view(B, *self.heatmap_size)
        # Rescale so max=1 for visualization
        max_val = probs.flatten(1).max(dim=1).values.clamp(min=1e-8)
        heatmap = probs / max_val.view(B, 1, 1)
        return heatmap

    def _normalize_gt(self, gt_heatmap: torch.Tensor) -> torch.Tensor:
        """Normalize GT heatmap to a probability distribution (sum=1)."""
        B = gt_heatmap.shape[0]
        gt_flat = gt_heatmap.view(B, -1).float()
        gt_sum = gt_flat.sum(dim=1, keepdim=True)

        # Positive samples: normalize to sum=1
        # Negative samples (all zero): uniform distribution
        is_negative = (gt_sum < 1e-6)
        uniform = torch.ones_like(gt_flat) / gt_flat.shape[1]

        gt_normalized = torch.where(
            is_negative.expand_as(gt_flat),
            uniform,
            gt_flat / gt_sum.clamp(min=1e-8),
        )
        return gt_normalized  # (B, H*W)

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

        # KL divergence
        log_q = F.log_softmax(logits.view(B, -1), dim=-1)
        target = self._normalize_gt(gt)

        kl_loss = F.kl_div(log_q, target, reduction="batchmean")

        # Heatmap for diagnostics
        with torch.no_grad():
            pred_heatmap = self._logits_to_heatmap(logits)
            pred_max = pred_heatmap.max().item()
            pred_mean = pred_heatmap.mean().item()

            sample_max = gt.flatten(1).max(dim=1).values
            is_positive = (sample_max > 0.01).float()
            n_pos = is_positive.sum().item()
            n_neg = B - n_pos

            per_pixel_mse = (pred_heatmap - gt) ** 2
            mse_val = per_pixel_mse.mean().item()

        self._training_step_counter += 1

        return {
            'loss': kl_loss,
            'heatmap': pred_heatmap.detach(),
            'dpt_kl_loss': kl_loss.item(),
            'dpt_mse': mse_val,
            'dpt_pred_max': pred_max,
            'dpt_pred_mean': pred_mean,
            'dpt_n_pos': n_pos,
            'dpt_n_neg': n_neg,
        }

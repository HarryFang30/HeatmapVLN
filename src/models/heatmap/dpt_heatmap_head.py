"""
Spatial-Semantic Fusion Head
==============================

Heatmap regression via spatial-semantic fusion:
    - ViT multi-scale pre-merge features (16x16, 1152-dim) provide spatial precision.
    - LLM fused vision tokens (8x8, 4096-dim) provide instruction-aware semantics.
    - Cross-attention fuses "where to look" (LLM query) with
      "precise spatial detail" (ViT key/value).
    - 2-stage ConvTranspose decoder: 16 -> 32 -> 64.

Loss:  KL divergence  +  lambda * soft-argmax spatial MSE (positive samples only).
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
    """Configuration for SpatialSemanticFusionHead."""
    # LLM side
    dim_in: int = 4096           # LLM hidden size
    num_image_tokens: int = 64   # post-merge tokens per image (8x8)
    patch_h: int = 8             # post-merge spatial height
    patch_w: int = 8             # post-merge spatial width

    # ViT side
    vit_dim: int = 1152          # ViT hidden size (pre-merge)
    num_vit_layers: int = 4      # number of hooked ViT blocks
    spatial_merge_size: int = 2  # Qwen3-VL merge factor

    # Shared
    proj_dim: int = 256          # projection / channel dimension
    features: int = 256          # synonym kept for config compat
    n_cross_attn_heads: int = 4
    heatmap_size: Tuple[int, int] = (64, 64)

    # Loss
    lambda_spatial: float = 0.5

    # Legacy (unused, kept so old configs don't crash)
    num_layers: int = 4


# ------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------

class UpsampleBlock(nn.Module):
    """ConvTranspose 2x upsample + Conv3x3 refinement."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(self.up(x))))


# ------------------------------------------------------------------
# Head
# ------------------------------------------------------------------

class DPTHeatmapHead(nn.Module):
    """
    Spatial-Semantic Fusion Head.

    Despite the class name (kept for pipeline compatibility), this is *not*
    a classic DPT head.  It fuses frozen ViT spatial features with
    instruction-aware LLM tokens via cross-attention, then decodes to a
    64x64 heatmap.
    """

    def __init__(self, config: DPTHeatmapConfig):
        super().__init__()
        self.config = config
        self.heatmap_size = config.heatmap_size
        self.patch_h = config.patch_h
        self.patch_w = config.patch_w
        self.num_image_tokens = config.num_image_tokens
        self._training_step_counter = 0

        proj = config.proj_dim
        vit_dim = config.vit_dim
        llm_dim = config.dim_in
        n_vit = config.num_vit_layers

        # ---------- ViT multi-scale projection ----------
        self.vit_norms = nn.ModuleList([nn.LayerNorm(vit_dim) for _ in range(n_vit)])
        self.vit_projs = nn.ModuleList([
            nn.Conv2d(vit_dim, proj, kernel_size=1) for _ in range(n_vit)
        ])
        self.vit_fusion = nn.Sequential(
            nn.Conv2d(proj * n_vit, proj, kernel_size=1),
            nn.BatchNorm2d(proj),
            nn.GELU(),
        )

        # ---------- LLM semantic projection ----------
        self.llm_norm = nn.LayerNorm(llm_dim)
        self.llm_proj = nn.Conv2d(llm_dim, proj, kernel_size=1)

        # ---------- Cross-attention ----------
        self.cross_attn = nn.MultiheadAttention(
            proj, config.n_cross_attn_heads, batch_first=True,
        )
        self.post_attn_norm = nn.LayerNorm(proj)

        # ---------- Decoder: 16 -> 32 -> 64 ----------
        self.decoder = nn.Sequential(
            UpsampleBlock(proj, proj // 2),
            UpsampleBlock(proj // 2, proj // 4),
        )
        self.head = nn.Conv2d(proj // 4, 1, kernel_size=1)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "SpatialSemanticFusionHead: vit_dim=%d, llm_dim=%d, proj=%d, "
            "n_vit_layers=%d, attn_heads=%d, heatmap=%s, params=%s",
            vit_dim, llm_dim, proj, n_vit, config.n_cross_attn_heads,
            config.heatmap_size, f"{total_params:,}",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        vit_features: Optional[List[torch.Tensor]] = None,
        llm_tokens: Optional[torch.Tensor] = None,
        gt_heatmap: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        # Legacy args (ignored)
        multi_layer_vision_tokens: Optional[List[torch.Tensor]] = None,
        observation: Optional[torch.Tensor] = None,
        skip_inference: bool = False,
        direction_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        logits = self._decode(vit_features, llm_tokens)

        # Qwen3-VL processor may resize images dynamically, so the actual
        # spatial resolution from ViT hooks can differ from config.
        # Always resize logits to the target heatmap_size for consistency.
        Hm, Wm = self.heatmap_size
        if logits.shape[-2:] != (Hm, Wm):
            logits = F.interpolate(
                logits.unsqueeze(1), size=(Hm, Wm),
                mode='bilinear', align_corners=False,
            ).squeeze(1)

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

    def _decode(
        self,
        vit_features: Optional[List[torch.Tensor]],
        llm_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            vit_features: ``num_vit_layers`` tensors of ``(B, vit_dim, h_pre, w_pre)``.
            llm_tokens:   ``(B, N_vision, llm_dim)``  (fused LLM vision tokens).
        """
        # ---------- ViT path: project each layer then concat ----------
        B = vit_features[0].shape[0]
        h_pre = vit_features[0].shape[2]
        w_pre = vit_features[0].shape[3]

        parts = []
        for i, feat in enumerate(vit_features):
            # feat: (B, vit_dim, h_pre, w_pre)
            x = feat.permute(0, 2, 3, 1)                   # (B, h, w, D)
            x = self.vit_norms[i](x)
            x = x.permute(0, 3, 1, 2)                      # (B, D, h, w)
            x = self.vit_projs[i](x)                        # (B, proj, h, w)
            parts.append(x)

        vit_fused = torch.cat(parts, dim=1)                 # (B, n*proj, h, w)
        vit_fused = self.vit_fusion(vit_fused)              # (B, proj, h, w)

        # Flatten spatial for cross-attention K/V
        N_vit = h_pre * w_pre
        kv = vit_fused.flatten(2).permute(0, 2, 1)         # (B, N_vit, proj)

        # ---------- LLM path: project + upsample to ViT resolution ----------
        # Derive post-merge spatial dims from actual pre-merge resolution
        merge_s = self.config.spatial_merge_size
        ph = h_pre // merge_s
        pw = w_pre // merge_s
        n_img = ph * pw
        llm = llm_tokens[:, -n_img:]                        # (B, ph*pw, llm_dim)
        llm = self.llm_norm(llm)
        llm_2d = llm.permute(0, 2, 1).reshape(B, -1, ph, pw)  # (B, llm_dim, ph, pw)
        llm_2d = self.llm_proj(llm_2d)                     # (B, proj, 8, 8)
        llm_up = F.interpolate(
            llm_2d, size=(h_pre, w_pre),
            mode='bilinear', align_corners=False,
        )                                                    # (B, proj, h_pre, w_pre)
        q = llm_up.flatten(2).permute(0, 2, 1)             # (B, N_vit, proj)

        # ---------- Cross-attention: Q=LLM, K=V=ViT ----------
        attn_out, _ = self.cross_attn(q, kv, kv)           # (B, N_vit, proj)
        fused = self.post_attn_norm(attn_out + q)           # residual + LN

        # Reshape back to 2-D
        fused = fused.permute(0, 2, 1).reshape(B, -1, h_pre, w_pre)  # (B, proj, h, w)

        # ---------- Decode: 16 -> 32 -> 64 ----------
        fused = self.decoder(fused)                         # (B, proj//4, 64, 64)
        logits = self.head(fused).squeeze(1)                # (B, 64, 64)
        return logits

    # ------------------------------------------------------------------
    # Heatmap / loss utilities
    # ------------------------------------------------------------------

    def _logits_to_heatmap(self, logits: torch.Tensor) -> torch.Tensor:
        B = logits.shape[0]
        probs = F.softmax(logits.view(B, -1), dim=-1)
        probs = probs.view(B, *self.heatmap_size)
        max_val = probs.flatten(1).max(dim=1).values.clamp(min=1e-8)
        return probs / max_val.view(B, 1, 1)

    @staticmethod
    def _soft_argmax_2d(logits: torch.Tensor) -> torch.Tensor:
        """Differentiable peak coordinate extraction.

        Args:
            logits: ``(B, H, W)`` raw logits *or* normalised heatmap.
        Returns:
            ``(B, 2)``  — (y, x) coordinates.
        """
        B, H, W = logits.shape
        probs = F.softmax(logits.reshape(B, -1), dim=-1).view(B, H, W)
        dev = logits.device
        y_grid = torch.arange(H, device=dev, dtype=torch.float).view(1, -1, 1)
        x_grid = torch.arange(W, device=dev, dtype=torch.float).view(1, 1, -1)
        y = (probs * y_grid).sum(dim=[1, 2])
        x = (probs * x_grid).sum(dim=[1, 2])
        return torch.stack([y, x], dim=-1)

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

        # ---- KL divergence ----
        log_q = F.log_softmax(logits.view(B, -1), dim=-1)
        target = self._normalize_gt(gt)
        kl_loss = F.kl_div(log_q, target, reduction="batchmean")

        # ---- Spatial peak loss (positive samples only) ----
        with torch.no_grad():
            sample_max = gt.flatten(1).max(dim=1).values
            is_positive = (sample_max > 0.01).float()
            n_pos = is_positive.sum().item()
            n_neg = B - n_pos

        spatial_loss = torch.tensor(0.0, device=logits.device)
        if n_pos > 0:
            pred_coords = self._soft_argmax_2d(logits)  # (B, 2) differentiable
            with torch.no_grad():
                gt_coords = self._soft_argmax_2d(gt)    # (B, 2)
            # Normalise to [0, 1] then MSE; mask negatives
            diff = (pred_coords - gt_coords) / Hm
            sq_dist = (diff ** 2).sum(dim=-1)            # (B,)
            spatial_loss = (sq_dist * is_positive).sum() / max(n_pos, 1)

        total_loss = kl_loss + self.config.lambda_spatial * spatial_loss

        with torch.no_grad():
            pred_heatmap = self._logits_to_heatmap(logits)

        self._training_step_counter += 1

        return {
            'loss': total_loss,
            'heatmap': pred_heatmap.detach(),
            'dpt_kl_loss': kl_loss.item(),
            'dpt_spatial_loss': spatial_loss.item(),
            'dpt_pred_max': pred_heatmap.max().item(),
            'dpt_pred_mean': pred_heatmap.mean().item(),
            'dpt_n_pos': n_pos,
            'dpt_n_neg': n_neg,
        }

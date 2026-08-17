"""
Fine Localization Module
==========================

Produces 64x64 fine-grained heatmaps by fusing:
  - DPT-fused ViT features (16x16, C_fused) — high-resolution texture
  - Coarse spatial_out from TrajectoryGuidedAttention (V*H*W, d_attn) —
    per-position representations enriched with hist/traj context
  - Coarse heatmap (8x8) — scalar spatial attention prior

spatial_out replaces the old FiLM modulation: each spatial token already
carries position-specific "what to look for" information from the
self-attention, giving the decoder per-position conditioning instead of
a single global channel-weight vector.

Pipeline:
  1. Upsample spatial_out from 8x8 → 16x16 (bilinear)
  2. Upsample coarse heatmap 8x8 → 16x16 as scalar attention
  3. Concat [ViT_fused(C), spatial_out_up(C), coarse_attn(1)] → 2C+1 channels
  4. ConvTranspose decoder: 16x16 → 32x32 → 64x64

Reference: HeatmapVLN设计文档 Section 6.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FineLocalization(nn.Module):
    """
    Spatial-out guided fine heatmap regression.

    Args:
        c_fused: channel dim of DPT-fused ViT features (= d_attn of coarse).
    """

    def __init__(
        self,
        c_fused: int = 256,
        *,
        coarse_logit_residual: bool = False,
    ):
        super().__init__()
        self.c_fused = c_fused
        self.coarse_logit_residual = bool(coarse_logit_residual)

        self.refine = nn.Sequential(
            nn.ConvTranspose2d(c_fused * 2 + 1, 128, kernel_size=4, stride=2, padding=1),  # -> 32x32
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),                # -> 64x64
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),                                     # -> 64x64, 1ch
        )

    def forward(
        self,
        vit_fused: torch.Tensor,
        coarse_heatmap: torch.Tensor,
        spatial_out: torch.Tensor,
        *,
        return_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vit_fused:      ``(4, C, 16, 16)`` or ``(B, 4, C, 16, 16)``
            coarse_heatmap: ``(N, 4, H_c, W_c)`` or ``(B, N, 4, H_c, W_c)``
            spatial_out:    ``(N, V*H_c*W_c, d_attn)`` or ``(B, N, V*H_c*W_c, d_attn)``
        """
        if coarse_heatmap.dim() == 5:
            return self._forward_5d(
                vit_fused,
                coarse_heatmap,
                spatial_out,
                return_logits=return_logits,
            )
        return self._forward_4d(
            vit_fused,
            coarse_heatmap,
            spatial_out,
            return_logits=return_logits,
        )

    def _forward_4d(
        self,
        vit_fused: torch.Tensor,      # (V, C, 16, 16)
        coarse_heatmap: torch.Tensor,  # (N, V, H_c, W_c)
        spatial_out: torch.Tensor,     # (N, V*H_c*W_c, d_attn)
        *,
        return_logits: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        num_hist, num_views = coarse_heatmap.shape[:2]
        H_c, W_c = coarse_heatmap.shape[2], coarse_heatmap.shape[3]
        if num_hist == 0:
            empty = coarse_heatmap.new_empty((0, num_views, 64, 64))
            return (empty, empty) if return_logits else empty

        spatial_size = vit_fused.shape[2:]  # (16, 16)

        attn = F.interpolate(
            coarse_heatmap.reshape(num_hist * num_views, 1, H_c, W_c),
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )
        attn = torch.sigmoid(attn)  # (N*V, 1, 16, 16)

        so = spatial_out.reshape(num_hist, num_views, H_c, W_c, -1)
        so = so.permute(0, 1, 4, 2, 3).reshape(num_hist * num_views, -1, H_c, W_c)
        so_up = F.interpolate(so, size=spatial_size, mode="bilinear", align_corners=False)
        # (N*V, d_attn, 16, 16)

        vit_expanded = vit_fused.unsqueeze(0).expand(num_hist, -1, -1, -1, -1)
        vit_expanded = vit_expanded.reshape(num_hist * num_views, vit_fused.shape[1], *spatial_size)
        # (N*V, C, 16, 16)

        x = torch.cat([vit_expanded, so_up, attn], dim=1)  # (N*V, 2C+1, 16, 16)

        refine_logits = self.refine(x)
        if self.coarse_logit_residual:
            coarse_logits_up = F.interpolate(
                coarse_heatmap.reshape(num_hist * num_views, 1, H_c, W_c),
                size=refine_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            refine_logits = refine_logits + coarse_logits_up

        logits = refine_logits.squeeze(1).reshape(
            num_hist,
            num_views,
            refine_logits.shape[-2],
            refine_logits.shape[-1],
        )
        heatmaps = torch.sigmoid(logits)
        return (heatmaps, logits) if return_logits else heatmaps

    def _forward_5d(
        self,
        vit_fused: torch.Tensor,      # (B, V, C, 16, 16)
        coarse_heatmap: torch.Tensor,  # (B, N, V, H_c, W_c)
        spatial_out: torch.Tensor,     # (B, N, V*H_c*W_c, d_attn)
        *,
        return_logits: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_hist, num_views = coarse_heatmap.shape[:3]
        H_c, W_c = coarse_heatmap.shape[3], coarse_heatmap.shape[4]
        if num_hist == 0:
            empty = coarse_heatmap.new_empty((batch_size, 0, num_views, 64, 64))
            return (empty, empty) if return_logits else empty

        spatial_size = vit_fused.shape[-2:]  # (16, 16)
        BNV = batch_size * num_hist * num_views

        attn = F.interpolate(
            coarse_heatmap.reshape(BNV, 1, H_c, W_c),
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )
        attn = torch.sigmoid(attn)  # (B*N*V, 1, 16, 16)

        so = spatial_out.reshape(batch_size, num_hist, num_views, H_c, W_c, -1)
        so = so.permute(0, 1, 2, 5, 3, 4).reshape(BNV, -1, H_c, W_c)
        so_up = F.interpolate(so, size=spatial_size, mode="bilinear", align_corners=False)
        # (B*N*V, d_attn, 16, 16)

        vit_expanded = vit_fused[:, None, :, :, :, :].expand(-1, num_hist, -1, -1, -1, -1)
        vit_expanded = vit_expanded.reshape(BNV, vit_fused.shape[2], *spatial_size)
        # (B*N*V, C, 16, 16)

        x = torch.cat([vit_expanded, so_up, attn], dim=1)  # (B*N*V, 2C+1, 16, 16)

        refine_logits = self.refine(x)
        if self.coarse_logit_residual:
            coarse_logits_up = F.interpolate(
                coarse_heatmap.reshape(BNV, 1, H_c, W_c),
                size=refine_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            refine_logits = refine_logits + coarse_logits_up

        logits = refine_logits.squeeze(1).reshape(
            batch_size,
            num_hist,
            num_views,
            refine_logits.shape[-2],
            refine_logits.shape[-1],
        )
        heatmaps = torch.sigmoid(logits)
        return (heatmaps, logits) if return_logits else heatmaps

"""
Fine Localization Module
==========================

Produces 64x64 fine-grained heatmaps from:
  - DPT-fused ViT features (16x16, C_fused)
  - Coarse heatmap (8x8, spatial attention prior)
  - LLM text-token query vector (C_llm, FiLM modulation)

Pipeline:
  1. Upsample coarse heatmap 8x8 -> 16x16 as spatial attention
  2. FiLM modulation: project query_vector to C_fused, element-wise multiply
  3. Spatial attention weighting
  4. Concatenate attention channel
  5. ConvTranspose decoder: 16x16 -> 32x32 -> 64x64

Trainable parameters: ~1.5M (c_fused=256, c_llm=4096).

Reference: HeatmapVLN设计文档 Section 6.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FineLocalization(nn.Module):
    """
    Coarse-guided fine heatmap regression.

    Args:
        c_fused: channel dim of DPT-fused ViT features.
        c_llm:   hidden dim of LLM query vector.
    """

    def __init__(self, c_fused: int = 256, c_llm: int = 4096):
        super().__init__()
        self.query_proj = nn.Linear(c_llm, c_fused)

        self.refine = nn.Sequential(
            nn.ConvTranspose2d(c_fused + 1, 128, kernel_size=4, stride=2, padding=1),  # -> 32x32
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),            # -> 64x64
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),                                 # -> 64x64, 1ch
        )

    def forward(
        self,
        vit_fused: torch.Tensor,
        coarse_heatmap: torch.Tensor,
        query_vector: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vit_fused:      ``(1, C_fused, 16, 16)`` — DPT-fused ViT features.
            coarse_heatmap: ``(H_c, W_c)``           — coarse heatmap (e.g. 8x8).
            query_vector:   ``(C_llm,)``              — text-token hidden state.

        Returns:
            ``(64, 64)`` heatmap in [0, 1].
        """
        # Spatial attention from coarse heatmap
        attn = F.interpolate(
            coarse_heatmap[None, None],  # (1, 1, H_c, W_c)
            size=vit_fused.shape[2:],    # match ViT spatial dims
            mode="bilinear",
            align_corners=False,
        )
        attn = torch.sigmoid(attn)  # (1, 1, H, W)

        # FiLM modulation
        q = self.query_proj(query_vector)  # (C_fused,)
        modulated = vit_fused * q[None, :, None, None]  # (1, C_fused, H, W)

        # Spatial gating
        modulated = modulated * attn  # (1, C_fused, H, W)

        # Concatenate coarse attention as extra channel
        x = torch.cat([modulated, attn], dim=1)  # (1, C_fused+1, H, W)

        # Decode to 64x64
        out = self.refine(x)          # (1, 1, 64, 64)
        out = torch.sigmoid(out)

        return out.squeeze(0).squeeze(0)  # (64, 64)

    def forward_batched(
        self,
        vit_fused: torch.Tensor,
        coarse_heatmap: torch.Tensor,
        query_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Batched fine localization."""
        if coarse_heatmap.dim() == 5:
            batch_size, num_hist, num_views = coarse_heatmap.shape[:3]
            if num_hist == 0:
                return coarse_heatmap.new_empty((batch_size, 0, num_views, 64, 64))

            spatial_size = vit_fused.shape[-2:]
            attn = F.interpolate(
                coarse_heatmap.reshape(
                    batch_size * num_hist * num_views,
                    1,
                    coarse_heatmap.shape[-2],
                    coarse_heatmap.shape[-1],
                ),
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            )
            attn = torch.sigmoid(attn)

            q = self.query_proj(query_vector)[:, :, None, :].expand(-1, -1, num_views, -1)
            q = q.reshape(batch_size * num_hist * num_views, -1, 1, 1)

            vit_expanded = vit_fused[:, None, :, :, :, :].expand(-1, num_hist, -1, -1, -1, -1)
            vit_expanded = vit_expanded.reshape(
                batch_size * num_hist * num_views,
                vit_fused.shape[2],
                *spatial_size,
            )

            modulated = vit_expanded * q
            modulated = modulated * attn
            x = torch.cat([modulated, attn], dim=1)

            out = self.refine(x)
            out = torch.sigmoid(out)
            return out.squeeze(1).reshape(batch_size, num_hist, num_views, out.shape[-2], out.shape[-1])

        num_hist, num_views = coarse_heatmap.shape[:2]
        if num_hist == 0:
            return coarse_heatmap.new_empty((0, num_views, 64, 64))

        spatial_size = vit_fused.shape[2:]
        attn = F.interpolate(
            coarse_heatmap.reshape(num_hist * num_views, 1, coarse_heatmap.shape[-2], coarse_heatmap.shape[-1]),
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )
        attn = torch.sigmoid(attn)

        q = self.query_proj(query_vector)[:, None, :].expand(-1, num_views, -1)
        q = q.reshape(num_hist * num_views, -1, 1, 1)

        vit_expanded = vit_fused.unsqueeze(0).expand(num_hist, -1, -1, -1, -1)
        vit_expanded = vit_expanded.reshape(num_hist * num_views, vit_fused.shape[1], *spatial_size)

        modulated = vit_expanded * q
        modulated = modulated * attn
        x = torch.cat([modulated, attn], dim=1)

        out = self.refine(x)
        out = torch.sigmoid(out)
        return out.squeeze(1).reshape(num_hist, num_views, out.shape[-2], out.shape[-1])

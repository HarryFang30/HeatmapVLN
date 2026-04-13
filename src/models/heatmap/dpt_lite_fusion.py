"""
DPT-Lite Fusion Module
========================

Fuses multi-layer ViT intermediate features into a unified 16x16 feature map.

Each ViT layer provides (16, 16, C_vit) per image.  This module:
  1. Aligns each layer to C_fused channels via 1x1 conv
  2. Concatenates all layers
  3. Fuses with 3x3 conv -> GELU -> 3x3 conv

Trainable parameters: ~530K (with c_vit=1152, c_fused=256, n_layers=4).

Reference: HeatmapVLN设计文档 Section 6.1
"""


import torch
import torch.nn as nn


class DPTLiteFusion(nn.Module):
    """
    Multi-layer ViT feature fusion.

    Input:  list of ``n_layers`` tensors, each ``(B, C_vit, 16, 16)``
    Output: ``(B, C_fused, 16, 16)``
    """

    def __init__(self, c_vit: int = 1152, c_fused: int = 256, n_layers: int = 4):
        super().__init__()
        self.align = nn.ModuleList([
            nn.Conv2d(c_vit, c_fused, kernel_size=1) for _ in range(n_layers)
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(c_fused * n_layers, c_fused, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_fused, c_fused, kernel_size=3, padding=1),
        )

    def forward(self, multi_layer_feats: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            multi_layer_feats: list of ``(B, C_vit, H, W)`` tensors.
        Returns:
            ``(B, C_fused, H, W)``
        """
        aligned = [self.align[i](f) for i, f in enumerate(multi_layer_feats)]
        concat = torch.cat(aligned, dim=1)  # (B, C_fused * n_layers, H, W)
        return self.fuse(concat)             # (B, C_fused, H, W)

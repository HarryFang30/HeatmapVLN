"""
1D Convolution Building Blocks for Diffusion Policy

Migrated from DifNav/diffusion_policy for action generation.

Components:
- Downsample1d: Strided convolution for downsampling
- Upsample1d: Transposed convolution for upsampling
- Conv1dBlock: Conv1d -> GroupNorm -> Mish activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample1d(nn.Module):
    """1D downsampling using strided convolution."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """1D upsampling using transposed convolution."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish --> Dropout
    
    Standard convolution block with normalization, activation, and optional dropout.
    
    Args:
        inp_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        n_groups: Number of groups for GroupNorm
        dropout: Dropout rate (0.0 = no dropout)
    """

    def __init__(
        self, 
        inp_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        n_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = [
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


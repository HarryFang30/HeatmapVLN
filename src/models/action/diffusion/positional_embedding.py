"""
Sinusoidal Positional Embedding for Diffusion Timesteps

Migrated from DifNav/diffusion_policy for action generation.
"""

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timesteps.

    Converts scalar timestep values to high-dimensional embeddings
    using sinusoidal functions, similar to transformer positional encodings.

    Args:
        dim: Output embedding dimension
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B,) tensor of timestep values

        Returns:
            (B, dim) tensor of positional embeddings (same dtype as input)
        """
        device = x.device
        dtype = x.dtype
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


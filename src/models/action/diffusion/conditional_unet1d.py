"""
Conditional U-Net 1D for Diffusion Policy

Migrated from DifNav/diffusion_policy for action generation.

This is the core noise prediction network for diffusion-based action generation.
It uses a U-Net architecture with FiLM conditioning to predict noise given:
- Noisy action sample
- Diffusion timestep
- Global conditioning (e.g., LLM features)
- Optional local conditioning
"""

import itertools
import logging
from typing import Union

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .conv1d_components import Conv1dBlock
from .positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class ConditionalResidualBlock1D(nn.Module):
    """
    Conditional residual block with FiLM modulation.

    Uses Feature-wise Linear Modulation (FiLM) to condition the block
    on external information (timestep + global features).

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        cond_dim: Dimension of conditioning vector
        kernel_size: Convolution kernel size
        n_groups: Number of groups for GroupNorm
        cond_predict_scale: If True, predict both scale and bias for FiLM
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups, dropout=dropout),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups, dropout=dropout),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, horizon) input tensor
            cond: (batch_size, cond_dim) conditioning vector

        Returns:
            (batch_size, out_channels, horizon) output tensor
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """
    Conditional U-Net 1D for noise prediction in diffusion models.

    Architecture:
    - Encoder path: Series of conditional residual blocks with optional downsampling
    - Middle: Two conditional residual blocks
    - Decoder path: Series of conditional residual blocks with skip connections
    - Final conv: Project back to input dimension

    Conditioning:
    - Diffusion timestep: Encoded via sinusoidal embedding
    - Global condition: Concatenated with timestep embedding
    - Local condition: Added to features at specific layers

    Args:
        input_dim: Dimension of input (action dimension)
        local_cond_dim: Optional dimension for local conditioning
        global_cond_dim: Dimension of global conditioning (e.g., LLM features)
        diffusion_step_embed_dim: Dimension for timestep embedding
        down_dims: List of channel dimensions for each encoder level
        kernel_size: Convolution kernel size
        n_groups: Number of groups for GroupNorm
        cond_predict_scale: If True, use scale+bias FiLM, else just bias
        dropout: Dropout rate for regularization (applied in Conv1dBlocks)
    """

    def __init__(
        self,
        input_dim: int,
        local_cond_dim: int | None = None,
        global_cond_dim: int | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: list[int] | None = None,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
        dropout: float = 0.1,
    ):
        if down_dims is None:
            down_dims = [256, 512, 1024]
        super().__init__()
        all_dims = [input_dim, *list(down_dims)]
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(itertools.pairwise(all_dims))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, dropout=dropout),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, dropout=dropout)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale, dropout=dropout
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale, dropout=dropout
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, dropout=dropout),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, dropout=dropout),
                # Note: DifNav uses Identity instead of actual downsampling
                nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, dropout=dropout),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, dropout=dropout),
                # Note: DifNav uses Identity instead of actual upsampling
                nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "ConditionalUnet1D initialized with %e parameters",
            sum(p.numel() for p in self.parameters())
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for noise prediction.

        Args:
            sample: (B, pred_horizon, action_dim) noisy action sample
            timestep: (B,) or scalar diffusion timestep
            local_cond: Optional (B, pred_horizon, local_cond_dim) local conditioning
            global_cond: Optional (B, global_cond_dim) global conditioning

        Returns:
            (B, pred_horizon, action_dim) predicted noise
        """
        # Rearrange: (B, H, T) -> (B, T, H) for conv operations
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. Encode timestep
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=sample.dtype, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device=sample.device, dtype=sample.dtype)
        else:
            timesteps = timesteps.to(dtype=sample.dtype)
        # Broadcast to batch dimension
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        # 2. Concatenate global conditioning
        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        # 3. Encode local features (if any)
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)

        # 4. Encoder path
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # 5. Middle
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # 6. Decoder path with skip connections
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) - 1 and len(h_local) > 1:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        # 7. Final projection
        x = self.final_conv(x)

        # Rearrange back: (B, T, H) -> (B, H, T)
        x = einops.rearrange(x, 'b t h -> b h t')
        return x


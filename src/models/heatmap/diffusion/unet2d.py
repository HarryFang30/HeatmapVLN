"""
Lightweight Conditional UNet2D for Diffusion Heatmap Generation.

This module implements a compact 2D U-Net architecture for noise prediction
in diffusion-based heatmap generation. It uses FiLM conditioning to incorporate
global condition vectors (from LLM + observation encoding).

Architecture:
    - Encoder: Conv2d blocks with downsampling
    - Middle: Residual + optional Attention
    - Decoder: Conv2d blocks with upsampling + skip connections
    - Conditioning: FiLM (Feature-wise Linear Modulation)
"""

import math
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for diffusion timesteps.
    
    Maps scalar timesteps to high-dimensional vectors using sinusoidal encoding,
    following the original Transformer and DDPM papers.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (batch_size,) float timesteps (same dtype as model)
            
        Returns:
            (batch_size, dim) position embeddings
        """
        device = timesteps.device
        dtype = timesteps.dtype
        half_dim = self.dim // 2
        
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        return embeddings


class Conv2dBlock(nn.Module):
    """
    Basic Conv2d block with GroupNorm and activation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        
        # Ensure n_groups is valid
        n_groups = min(n_groups, out_channels)
        while out_channels % n_groups != 0:
            n_groups -= 1
        
        self.norm = nn.GroupNorm(n_groups, out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class ConditionalResidualBlock2D(nn.Module):
    """
    Conditional residual block with FiLM modulation.
    
    Uses Feature-wise Linear Modulation (FiLM) to condition the block
    on the timestep embedding and global condition.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Convolution blocks
        self.block1 = Conv2dBlock(in_channels, out_channels, kernel_size, n_groups, dropout)
        self.block2 = Conv2dBlock(out_channels, out_channels, kernel_size, n_groups, dropout)
        
        # FiLM modulation: predict scale and shift from condition
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels * 2),
        )
        
        # Residual connection
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features
            cond: (B, cond_dim) conditioning vector
            
        Returns:
            (B, out_channels, H, W) output features
        """
        residual = self.residual_conv(x)
        
        # First conv block
        h = self.block1(x)
        
        # FiLM modulation
        cond_out = self.cond_mlp(cond)  # (B, out_channels * 2)
        scale, shift = cond_out.chunk(2, dim=-1)  # Each (B, out_channels)
        scale = scale[:, :, None, None]  # (B, C, 1, 1)
        shift = shift[:, :, None, None]  # (B, C, 1, 1)
        
        h = h * (1 + scale) + shift
        
        # Second conv block
        h = self.block2(h)
        
        return h + residual


class Attention2D(nn.Module):
    """
    Self-attention block for 2D feature maps.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        head_dim: int = 32,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        
        self.norm = nn.GroupNorm(8, channels)
        self.to_qkv = nn.Conv2d(channels, inner_dim * 3, 1)
        self.to_out = nn.Conv2d(inner_dim, channels, 1)
        self.scale = head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        residual = x
        x = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.to_qkv(x)  # (B, inner_dim * 3, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each (B, heads, head_dim, H*W)
        
        # Transpose for attention
        q = q.permute(0, 1, 3, 2)  # (B, heads, H*W, head_dim)
        k = k.permute(0, 1, 3, 2)  # (B, heads, H*W, head_dim)
        v = v.permute(0, 1, 3, 2)  # (B, heads, H*W, head_dim)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, heads, H*W, H*W)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # (B, heads, H*W, head_dim)
        out = out.permute(0, 1, 3, 2)  # (B, heads, head_dim, H*W)
        out = out.reshape(B, -1, H, W)  # (B, inner_dim, H, W)
        out = self.to_out(out)
        
        return out + residual


class Downsample2D(nn.Module):
    """Downsampling block using strided convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2D(nn.Module):
    """Upsampling block using nearest neighbor + conv."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class ConditionalUnet2D(nn.Module):
    """
    Lightweight Conditional U-Net for 2D diffusion.
    
    This is a compact U-Net architecture designed for heatmap generation.
    It takes a noisy heatmap and conditioning vector, and predicts the noise.
    
    Architecture:
        - Encoder: ConditionalResidualBlock2D + Downsample
        - Middle: ConditionalResidualBlock2D + Attention
        - Decoder: ConditionalResidualBlock2D + Upsample + Skip connections
    
    Args:
        in_channels: Input channels (1 for heatmap)
        out_channels: Output channels (1 for noise)
        cond_dim: Conditioning vector dimension
        block_out_channels: Channel dimensions for each level
        layers_per_block: Number of residual blocks per level
        attention_levels: Which levels to add attention (0-indexed)
        n_groups: Groups for GroupNorm
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        cond_dim: int = 512,
        block_out_channels: Tuple[int, ...] = (64, 128, 256),
        layers_per_block: int = 2,
        attention_levels: Tuple[int, ...] = (2,),
        n_groups: int = 8,
        dropout: float = 0.0,
        timestep_embed_dim: int = 256,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)
        
        # ==================== Encoder ====================
        self.down_blocks = nn.ModuleList()
        self.down_attentions = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        in_ch = block_out_channels[0]
        for i, out_ch in enumerate(block_out_channels):
            # Residual blocks
            blocks = nn.ModuleList([
                ConditionalResidualBlock2D(
                    in_ch if j == 0 else out_ch,
                    out_ch,
                    cond_dim,
                    n_groups=n_groups,
                    dropout=dropout,
                )
                for j in range(layers_per_block)
            ])
            self.down_blocks.append(blocks)
            
            # Attention
            if i in attention_levels:
                self.down_attentions.append(Attention2D(out_ch))
            else:
                self.down_attentions.append(nn.Identity())
            
            # Downsampler (except for last level)
            if i < len(block_out_channels) - 1:
                self.downsamplers.append(Downsample2D(out_ch))
            else:
                self.downsamplers.append(nn.Identity())
            
            in_ch = out_ch
        
        # ==================== Middle ====================
        mid_channels = block_out_channels[-1]
        self.mid_block1 = ConditionalResidualBlock2D(
            mid_channels, mid_channels, cond_dim, n_groups=n_groups, dropout=dropout
        )
        self.mid_attn = Attention2D(mid_channels)
        self.mid_block2 = ConditionalResidualBlock2D(
            mid_channels, mid_channels, cond_dim, n_groups=n_groups, dropout=dropout
        )
        
        # ==================== Decoder ====================
        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        reversed_channels = list(reversed(block_out_channels))
        
        for i, out_ch in enumerate(reversed_channels):
            # Input includes skip connection
            skip_ch = reversed_channels[min(i, len(reversed_channels) - 1)]
            
            blocks = nn.ModuleList([
                ConditionalResidualBlock2D(
                    (in_ch + skip_ch) if j == 0 else out_ch,
                    out_ch,
                    cond_dim,
                    n_groups=n_groups,
                    dropout=dropout,
                )
                for j in range(layers_per_block)
            ])
            self.up_blocks.append(blocks)
            
            # Attention
            level_idx = len(block_out_channels) - 1 - i
            if level_idx in attention_levels:
                self.up_attentions.append(Attention2D(out_ch))
            else:
                self.up_attentions.append(nn.Identity())
            
            # Upsampler (except for last level)
            if i < len(reversed_channels) - 1:
                self.upsamplers.append(Upsample2D(out_ch))
            else:
                self.upsamplers.append(nn.Identity())
            
            in_ch = out_ch
        
        # ==================== Output ====================
        self.conv_out = nn.Sequential(
            nn.GroupNorm(n_groups, block_out_channels[0]),
            nn.SiLU(),
            nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1),
        )
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for noise prediction.
        
        Args:
            sample: (B, in_channels, H, W) noisy heatmap
            timestep: (B,) diffusion timesteps
            global_cond: (B, cond_dim) conditioning vector
            
        Returns:
            (B, out_channels, H, W) predicted noise
        """
        # Timestep embedding - convert to float dtype matching model weights
        timestep_float = timestep.to(dtype=sample.dtype)
        t_emb = self.time_embed(timestep_float)  # (B, cond_dim)
        
        # Combine timestep and global condition
        cond = t_emb + global_cond  # (B, cond_dim)
        
        # Initial conv
        h = self.conv_in(sample)  # (B, block_out_channels[0], H, W)
        
        # ==================== Encoder ====================
        skip_connections = []
        
        for blocks, attn, downsample in zip(
            self.down_blocks, self.down_attentions, self.downsamplers
        ):
            for block in blocks:
                h = block(h, cond)
            h = attn(h)
            skip_connections.append(h)
            h = downsample(h)
        
        # ==================== Middle ====================
        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)
        
        # ==================== Decoder ====================
        for blocks, attn, upsample in zip(
            self.up_blocks, self.up_attentions, self.upsamplers
        ):
            # Skip connection
            if skip_connections:
                skip = skip_connections.pop()
                # Handle size mismatch from downsampling
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')
                h = torch.cat([h, skip], dim=1)
            
            for block in blocks:
                h = block(h, cond)
            h = attn(h)
            h = upsample(h)
        
        # ==================== Output ====================
        return self.conv_out(h)


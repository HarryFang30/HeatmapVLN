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
    
    Supports 360° panorama images with circular padding in horizontal direction.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        dropout: float = 0.0,
        use_circular_padding: bool = False,
    ):
        super().__init__()
        
        self.use_circular_padding = use_circular_padding
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        if use_circular_padding:
            # 360° 全景图：水平方向 circular，垂直方向 replicate
            # Conv2d 不使用内置 padding，我们手动处理
            self.conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=0,  # 手动 padding
            )
        else:
            # 标准 padding
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
        if self.use_circular_padding:
            # 水平方向 circular padding (左右边界连续)
            x = F.pad(x, (self.padding, self.padding, 0, 0), mode='circular')
            # 垂直方向 replicate padding (上下边界复制)
            x = F.pad(x, (0, 0, self.padding, self.padding), mode='replicate')
        
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
    
    Supports 360° panorama images with circular padding.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        dropout: float = 0.0,
        use_circular_padding: bool = False,
    ):
        super().__init__()
        
        # Convolution blocks with optional circular padding for 360° panorama
        self.block1 = Conv2dBlock(
            in_channels, out_channels, kernel_size, n_groups, dropout,
            use_circular_padding=use_circular_padding
        )
        self.block2 = Conv2dBlock(
            out_channels, out_channels, kernel_size, n_groups, dropout,
            use_circular_padding=use_circular_padding
        )
        
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


class CrossAttention2D(nn.Module):
    """
    Cross-attention block for conditioning 2D feature maps.
    
    支持两种条件输入:
    - (B, cond_dim): 单向量条件 -> K/V 长度为 1 (向后兼容)
    - (B, seq_len, cond_dim): 序列条件 -> K/V 长度为 seq_len (序列级 cross-attention)
    
    序列模式下，每个空间位置可以 attend 到 LLM token 序列的不同部分，
    显著缓解单向量条件的信息瓶颈。
    """
    
    def __init__(
        self,
        channels: int,
        cond_dim: int,
        num_heads: int = 4,
        head_dim: int = 32,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        
        self.norm = nn.GroupNorm(8, channels)
        self.norm_cond = nn.LayerNorm(cond_dim)
        
        # Q 来自特征图，K/V 来自条件
        self.to_q = nn.Conv2d(channels, inner_dim, 1)
        self.to_k = nn.Linear(cond_dim, inner_dim)
        self.to_v = nn.Linear(cond_dim, inner_dim)
        self.to_out = nn.Conv2d(inner_dim, channels, 1)
        
        self.scale = head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 特征图
            cond: (B, cond_dim) 单向量条件 或 (B, seq_len, cond_dim) 序列条件
            
        Returns:
            (B, C, H, W) 条件增强后的特征图
        """
        B, C, H, W = x.shape
        
        residual = x
        x = self.norm(x)
        
        # 自适应处理: 将 (B, cond_dim) 统一为 (B, seq_len, cond_dim)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)  # (B, 1, cond_dim)
        
        cond = self.norm_cond(cond)  # (B, seq_len, cond_dim)
        seq_len = cond.shape[1]
        
        # Q from spatial features: (B, inner_dim, H, W) -> (B, heads, H*W, head_dim)
        q = self.to_q(x)  # (B, inner_dim, H, W)
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        q = q.permute(0, 1, 3, 2)  # (B, heads, H*W, head_dim)
        
        # K, V from condition: (B, seq_len, cond_dim) -> (B, heads, seq_len, head_dim)
        k = self.to_k(cond)  # (B, seq_len, inner_dim)
        v = self.to_v(cond)  # (B, seq_len, inner_dim)
        k = k.reshape(B, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # k, v: (B, heads, seq_len, head_dim)
        
        # Attention: 每个空间位置查询条件序列
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, heads, H*W, seq_len)
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
        use_circular_padding: bool = False,
        # Sequence cross-attention conditioning
        use_sequence_conditioning: bool = False,
        seq_cross_attn_heads: int = 8,
        seq_cross_attn_head_dim: int = 64,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim
        self.use_circular_padding = use_circular_padding
        self.use_sequence_conditioning = use_sequence_conditioning
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # Initial convolution (with optional circular padding for 360° panorama)
        if use_circular_padding:
            self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=0)
        else:
            self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)
        
        # ==================== Encoder ====================
        self.down_blocks = nn.ModuleList()
        self.down_attentions = nn.ModuleList()
        self.down_cross_attentions = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        in_ch = block_out_channels[0]
        for i, out_ch in enumerate(block_out_channels):
            # Residual blocks (with optional circular padding for 360° panorama)
            blocks = nn.ModuleList([
                ConditionalResidualBlock2D(
                    in_ch if j == 0 else out_ch,
                    out_ch,
                    cond_dim,
                    n_groups=n_groups,
                    dropout=dropout,
                    use_circular_padding=use_circular_padding,
                )
                for j in range(layers_per_block)
            ])
            self.down_blocks.append(blocks)
            
            # Self-Attention
            if i in attention_levels:
                self.down_attentions.append(Attention2D(out_ch))
            else:
                self.down_attentions.append(nn.Identity())
            
            # Sequence Cross-Attention (only at attention levels, when enabled)
            if use_sequence_conditioning and i in attention_levels:
                self.down_cross_attentions.append(CrossAttention2D(
                    channels=out_ch,
                    cond_dim=cond_dim,
                    num_heads=seq_cross_attn_heads,
                    head_dim=seq_cross_attn_head_dim,
                ))
            else:
                self.down_cross_attentions.append(nn.Identity())
            
            # Downsampler (except for last level)
            if i < len(block_out_channels) - 1:
                self.downsamplers.append(Downsample2D(out_ch))
            else:
                self.downsamplers.append(nn.Identity())
            
            in_ch = out_ch
        
        # ==================== Middle ====================
        mid_channels = block_out_channels[-1]
        self.mid_block1 = ConditionalResidualBlock2D(
            mid_channels, mid_channels, cond_dim, n_groups=n_groups, dropout=dropout,
            use_circular_padding=use_circular_padding
        )
        self.mid_attn = Attention2D(mid_channels)
        # Cross-Attention: 让每个空间位置都能查询条件
        # 序列模式下使用更大的 head 配置
        mid_cross_heads = seq_cross_attn_heads if use_sequence_conditioning else 4
        mid_cross_head_dim = seq_cross_attn_head_dim if use_sequence_conditioning else 32
        self.mid_cross_attn = CrossAttention2D(
            channels=mid_channels,
            cond_dim=cond_dim,
            num_heads=mid_cross_heads,
            head_dim=mid_cross_head_dim,
        )
        self.mid_block2 = ConditionalResidualBlock2D(
            mid_channels, mid_channels, cond_dim, n_groups=n_groups, dropout=dropout,
            use_circular_padding=use_circular_padding
        )
        
        # ==================== Decoder ====================
        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()
        self.up_cross_attentions = nn.ModuleList()
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
                    use_circular_padding=use_circular_padding,
                )
                for j in range(layers_per_block)
            ])
            self.up_blocks.append(blocks)
            
            # Self-Attention
            level_idx = len(block_out_channels) - 1 - i
            if level_idx in attention_levels:
                self.up_attentions.append(Attention2D(out_ch))
            else:
                self.up_attentions.append(nn.Identity())
            
            # Sequence Cross-Attention (mirror encoder)
            if use_sequence_conditioning and level_idx in attention_levels:
                self.up_cross_attentions.append(CrossAttention2D(
                    channels=out_ch,
                    cond_dim=cond_dim,
                    num_heads=seq_cross_attn_heads,
                    head_dim=seq_cross_attn_head_dim,
                ))
            else:
                self.up_cross_attentions.append(nn.Identity())
            
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
        seq_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for noise prediction.
        
        Args:
            sample: (B, in_channels, H, W) noisy heatmap
            timestep: (B,) diffusion timesteps
            global_cond: (B, cond_dim) conditioning vector (for FiLM)
            seq_cond: Optional (B, seq_len, cond_dim) sequence conditioning (for cross-attention)
                      If None, cross-attention uses global_cond (backward compatible)
            
        Returns:
            (B, out_channels, H, W) predicted noise
        """
        # Timestep embedding - convert to float dtype matching model weights
        timestep_float = timestep.to(dtype=sample.dtype)
        t_emb = self.time_embed(timestep_float)  # (B, cond_dim)
        
        # Combine timestep and global condition for FiLM
        cond = t_emb + global_cond  # (B, cond_dim)
        
        # Determine cross-attention condition: use seq_cond if available, else global_cond
        cross_cond = seq_cond if seq_cond is not None else global_cond
        
        # Initial conv (with optional circular padding for 360° panorama)
        if self.use_circular_padding:
            h = F.pad(sample, (1, 1, 0, 0), mode='circular')  # 水平 circular
            h = F.pad(h, (0, 0, 1, 1), mode='replicate')       # 垂直 replicate
            h = self.conv_in(h)
        else:
            h = self.conv_in(sample)  # (B, block_out_channels[0], H, W)
        
        # ==================== Encoder ====================
        skip_connections = []
        
        for blocks, attn, cross_attn, downsample in zip(
            self.down_blocks, self.down_attentions,
            self.down_cross_attentions, self.downsamplers
        ):
            for block in blocks:
                h = block(h, cond)  # FiLM conditioning
            h = attn(h)  # Self-attention
            # Sequence cross-attention (at attention_levels when enabled)
            if isinstance(cross_attn, CrossAttention2D):
                h = cross_attn(h, cross_cond)
            skip_connections.append(h)
            h = downsample(h)
        
        # ==================== Middle ====================
        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_cross_attn(h, cross_cond)  # Cross-Attention with sequence or global
        h = self.mid_block2(h, cond)
        
        # ==================== Decoder ====================
        for blocks, attn, cross_attn, upsample in zip(
            self.up_blocks, self.up_attentions,
            self.up_cross_attentions, self.upsamplers
        ):
            # Skip connection
            if skip_connections:
                skip = skip_connections.pop()
                # Handle size mismatch from downsampling
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')
                h = torch.cat([h, skip], dim=1)
            
            for block in blocks:
                h = block(h, cond)  # FiLM conditioning
            h = attn(h)  # Self-attention
            # Sequence cross-attention (mirror encoder)
            if isinstance(cross_attn, CrossAttention2D):
                h = cross_attn(h, cross_cond)
            h = upsample(h)
        
        # ==================== Output ====================
        return self.conv_out(h)


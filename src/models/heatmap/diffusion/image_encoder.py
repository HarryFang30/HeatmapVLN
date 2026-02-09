"""
Image Condition Encoder for Diffusion Heatmap Generation.

This module provides encoders for observation images that produce
conditioning vectors for the diffusion process.

Architecture options:
1. Lightweight CNN encoder (default)
2. ResNet-based encoder (optional)
"""

import math
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Attention Pooling ====================

class AttentionPooling(nn.Module):
    """
    Attention-based pooling for sequence features.
    
    使用可学习的 query 向量通过 attention 机制聚合序列特征，
    比 mean pooling 更好地保留重要信息。
    
    Args:
        dim: Feature dimension
        num_heads: Number of attention heads (default 4)
    """
    
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 可学习的 query 向量
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        # 投影层
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        
        # LayerNorm
        self.norm = nn.LayerNorm(dim)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, dim) sequence features
            
        Returns:
            (B, dim) pooled features
        """
        B, seq_len, dim = x.shape
        
        # Normalize input
        x = self.norm(x)
        
        # Expand query for batch
        query = self.query.expand(B, -1, -1)  # (B, 1, dim)
        
        # Compute key and value
        key = self.k_proj(x)    # (B, seq_len, dim)
        value = self.v_proj(x)  # (B, seq_len, dim)
        
        # Reshape for multi-head attention
        query = query.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # query: (B, num_heads, 1, head_dim)
        # key/value: (B, num_heads, seq_len, head_dim)
        
        # Attention
        attn = torch.matmul(query, key.transpose(-1, -2)) * self.scale  # (B, num_heads, 1, seq_len)
        attn = F.softmax(attn, dim=-1)
        
        # Aggregate
        out = torch.matmul(attn, value)  # (B, num_heads, 1, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, 1, dim)  # (B, 1, dim)
        out = self.out_proj(out)
        
        return out.squeeze(1)  # (B, dim)


def _get_num_groups(num_channels: int, max_groups: int = 32) -> int:
    """Calculate number of groups for GroupNorm, ensuring divisibility."""
    for g in [max_groups, 16, 8, 4, 2, 1]:
        if num_channels % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    """Basic convolutional block with GroupNorm and ReLU.
    
    Uses GroupNorm instead of BatchNorm for better stability with small batch sizes.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        # Use GroupNorm instead of BatchNorm for better stability with small batches
        num_groups = _get_num_groups(out_channels)
        self.bn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with two convolutions.
    
    Uses GroupNorm instead of BatchNorm for better stability with small batch sizes.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        num_groups = _get_num_groups(channels)
        self.bn1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class ImageConditionEncoder(nn.Module):
    """
    Lightweight CNN encoder for observation images.
    
    Encodes an observation image into a conditioning vector for the
    diffusion heatmap generator.
    
    Architecture:
        - Stem: Conv 3x3 stride 2 -> BN -> ReLU
        - Stages: [ConvBlock stride 2 + ResidualBlock] x N
        - Pool: Global Average Pooling
        - Project: Linear -> LayerNorm -> GELU -> Linear
    
    Args:
        in_channels: Input image channels (3 for RGB)
        out_dim: Output conditioning dimension
        hidden_channels: Channel dimensions for each stage
        image_size: Expected input image size (for validation)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 512,
        hidden_channels: List[int] = None,
        image_size: Tuple[int, int] = (224, 224),
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.image_size = image_size

        # ==================== Stem ====================
        # Use GroupNorm instead of BatchNorm for better stability with small batches
        stem_num_groups = _get_num_groups(hidden_channels[0])
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(num_groups=stem_num_groups, num_channels=hidden_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # ==================== Stages ====================
        self.stages = nn.ModuleList()

        in_ch = hidden_channels[0]
        for out_ch in hidden_channels[1:]:
            stage = nn.Sequential(
                ConvBlock(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                ResidualBlock(out_ch),
            )
            self.stages.append(stage)
            in_ch = out_ch

        # ==================== Pooling ====================
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ==================== Projection ====================
        # Added Dropout for regularization to prevent overfitting
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels[-1], out_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode observation image to conditioning vector.
        
        Args:
            x: (B, C, H, W) observation image, normalized to [0, 1] or [-1, 1]
            
        Returns:
            (B, out_dim) conditioning vector
        """
        # Stem
        h = self.stem(x)
        
        # Stages
        for stage in self.stages:
            h = stage(h)
        
        # Global pooling
        h = self.pool(h)  # (B, C, 1, 1)
        h = h.flatten(1)  # (B, C)
        
        # Projection
        return self.projection(h)
    
    def forward_multiscale(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode observation image, returning both global vector AND multi-scale feature maps.
        
        Used for spatial feature injection into UNet skip connections.
        
        Args:
            x: (B, C, H, W) observation image
            
        Returns:
            global_cond: (B, out_dim) global conditioning vector (same as forward())
            spatial_features: List of feature maps at each scale:
                [0] after stem:   (B, hidden_channels[0], H/4, W/4)    e.g. (B, 32, 56, 56)
                [1] after stage0: (B, hidden_channels[1], H/8, W/8)    e.g. (B, 64, 28, 28)
                [2] after stage1: (B, hidden_channels[2], H/16, W/16)  e.g. (B, 128, 14, 14)
                [3] after stage2: (B, hidden_channels[3], H/32, W/32)  e.g. (B, 256, 7, 7)
        """
        spatial_features = []
        
        # Stem
        h = self.stem(x)
        spatial_features.append(h)  # (B, 32, 56, 56) for 224x224 input
        
        # Stages
        for stage in self.stages:
            h = stage(h)
            spatial_features.append(h)
        
        # Global pooling + projection (same as forward)
        pooled = self.pool(h)  # (B, C, 1, 1)
        pooled = pooled.flatten(1)  # (B, C)
        global_cond = self.projection(pooled)
        
        return global_cond, spatial_features


class LLMConditionProjector(nn.Module):
    """
    Projects LLM token features to conditioning dimension.
    
    Handles variable-length sequences by pooling.
    
    Args:
        input_dim: LLM hidden dimension
        output_dim: Output conditioning dimension
        hidden_dim: Intermediate dimension
        pool_method: How to pool sequence ('mean', 'first', 'last', 'max', 'attention')
        num_attention_heads: Number of heads for attention pooling (only used when pool_method='attention')
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        output_dim: int = 512,
        hidden_dim: int = 1024,
        pool_method: str = 'attention',  # 默认改为 attention
        dropout: float = 0.1,
        num_attention_heads: int = 4,
    ):
        super().__init__()

        self.pool_method = pool_method

        # Attention pooling (推荐，更好地保留空间信息)
        if pool_method == 'attention':
            self.attention_pool = AttentionPooling(input_dim, num_heads=num_attention_heads)
        else:
            self.attention_pool = None

        # Added Dropout for regularization to prevent overfitting
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, input_dim) or (B, input_dim) LLM features
            
        Returns:
            (B, output_dim) conditioning vector
        """
        # Handle 2D input (already pooled)
        if x.dim() == 2:
            return self.projector(x)
        
        # Pool sequence dimension
        if self.pool_method == 'attention':
            x = self.attention_pool(x)
        elif self.pool_method == 'mean':
            x = x.mean(dim=1)
        elif self.pool_method == 'first':
            x = x[:, 0]
        elif self.pool_method == 'last':
            x = x[:, -1]
        elif self.pool_method == 'max':
            x = x.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pool method: {self.pool_method}")
        
        return self.projector(x)


class MultiModalConditionEncoder(nn.Module):
    """
    Fuses LLM tokens and observation image into conditioning signals.
    
    Supports two conditioning paths:
        1. Global path (FiLM): LLM pool + image -> (B, cond_dim) single vector
        2. Sequence path (Cross-Attention): LLM project -> (B, seq_len, cond_dim) token sequence
    
    Args:
        llm_dim: LLM hidden dimension
        image_channels: Observation image channels
        cond_dim: Output conditioning dimension
        image_encoder_channels: Channel dims for image encoder
        llm_hidden_dim: Intermediate dim for LLM projector
        pool_method: How to pool LLM sequence ('attention', 'mean', 'first', 'last', 'max')
        pool_num_heads: Number of attention heads for attention pooling
        use_sequence_conditioning: Enable dual-path conditioning (global + sequence)
    """
    
    def __init__(
        self,
        llm_dim: int = 2048,
        image_channels: int = 3,
        cond_dim: int = 512,
        image_encoder_channels: List[int] = None,
        llm_hidden_dim: int = 1024,
        pool_method: str = 'attention',
        pool_num_heads: int = 4,
        image_size: Tuple[int, int] = (224, 224),
        dropout: float = 0.1,
        use_image_encoder: bool = True,
        use_sequence_conditioning: bool = False,
    ):
        super().__init__()

        if image_encoder_channels is None:
            image_encoder_channels = [32, 64, 128, 256]

        self.use_image_encoder = use_image_encoder
        self.use_sequence_conditioning = use_sequence_conditioning

        # LLM projector with attention pooling support (global path)
        self.llm_projector = LLMConditionProjector(
            input_dim=llm_dim,
            output_dim=cond_dim,
            hidden_dim=llm_hidden_dim,
            pool_method=pool_method,
            dropout=dropout,
            num_attention_heads=pool_num_heads,
        )

        # Image encoder (only created if use_image_encoder=True)
        if use_image_encoder:
            self.image_encoder = ImageConditionEncoder(
                in_channels=image_channels,
                out_dim=cond_dim,
                hidden_channels=image_encoder_channels,
                image_size=image_size,
                dropout=dropout,
            )
            # Fusion MLP for LLM + Image features
            self.fusion = nn.Sequential(
                nn.Linear(cond_dim * 2, cond_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, cond_dim),
            )
        else:
            # LLM-only mode: simple projection for LLM features
            self.image_encoder = None
            self.fusion = nn.Sequential(
                nn.Linear(cond_dim, cond_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, cond_dim),
            )
        
        # Sequence projector (sequence path - no pooling, keeps full sequence)
        if use_sequence_conditioning:
            self.seq_projector = nn.Sequential(
                nn.LayerNorm(llm_dim),
                nn.Linear(llm_dim, cond_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(cond_dim, cond_dim),
                nn.LayerNorm(cond_dim),
            )
        else:
            self.seq_projector = None
    
    def forward(
        self,
        llm_tokens: torch.Tensor,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse LLM tokens and observation into global conditioning vector.

        Args:
            llm_tokens: (B, seq_len, llm_dim) or (B, llm_dim) LLM features
            observation: (B, C, H, W) observation image (ignored if use_image_encoder=False)

        Returns:
            (B, cond_dim) fused conditioning vector
        """
        # Encode LLM tokens
        llm_cond = self.llm_projector(llm_tokens)  # (B, cond_dim)

        if self.use_image_encoder and self.image_encoder is not None:
            # Encode observation and fuse with LLM features
            img_cond = self.image_encoder(observation)  # (B, cond_dim)
            fused = torch.cat([llm_cond, img_cond], dim=-1)  # (B, cond_dim * 2)
        else:
            # LLM-only mode: use only LLM features
            fused = llm_cond  # (B, cond_dim)

        return self.fusion(fused)  # (B, cond_dim)
    
    def forward_dual(
        self,
        llm_tokens: torch.Tensor,
        observation: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Dual-path conditioning: returns both global vector and sequence.
        
        Args:
            llm_tokens: (B, seq_len, llm_dim) LLM features
            observation: (B, C, H, W) observation image
            
        Returns:
            global_cond: (B, cond_dim) pooled conditioning vector (for FiLM)
            seq_cond: (B, seq_len, cond_dim) projected sequence (for cross-attention)
                      None if use_sequence_conditioning is False
        """
        # Global path (existing)
        global_cond = self.forward(llm_tokens, observation)  # (B, cond_dim)
        
        # Sequence path (new)
        seq_cond = None
        if self.use_sequence_conditioning and self.seq_projector is not None:
            if llm_tokens.dim() == 2:
                llm_tokens = llm_tokens.unsqueeze(1)  # (B, 1, dim)
            seq_cond = self.seq_projector(llm_tokens)  # (B, seq_len, cond_dim)
        
        return global_cond, seq_cond
    
    def forward_with_spatial(
        self,
        llm_tokens: torch.Tensor,
        observation: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Full conditioning: global vector + optional sequence + multi-scale spatial features.
        
        Combines forward_dual() with spatial feature extraction from CNN encoder.
        
        Args:
            llm_tokens: (B, seq_len, llm_dim) LLM features
            observation: (B, C, H, W) observation image
            
        Returns:
            global_cond: (B, cond_dim) pooled conditioning vector (for FiLM)
            seq_cond: (B, seq_len, cond_dim) projected sequence (for cross-attention), or None
            spatial_features: List of CNN feature maps at each scale, or None
        """
        # LLM global path
        llm_cond = self.llm_projector(llm_tokens)  # (B, cond_dim)
        
        spatial_features = None
        if self.use_image_encoder and self.image_encoder is not None and observation is not None:
            # Get both global vector and spatial features from CNN
            img_cond, spatial_features = self.image_encoder.forward_multiscale(observation)
            fused = torch.cat([llm_cond, img_cond], dim=-1)
        else:
            fused = llm_cond
        
        global_cond = self.fusion(fused)
        
        # Sequence path
        seq_cond = None
        if self.use_sequence_conditioning and self.seq_projector is not None:
            if llm_tokens.dim() == 2:
                llm_tokens = llm_tokens.unsqueeze(1)
            seq_cond = self.seq_projector(llm_tokens)
        
        return global_cond, seq_cond, spatial_features
    
    def forward_llm_only(self, llm_tokens: torch.Tensor) -> torch.Tensor:
        """Encode only LLM tokens (for cases without observation)."""
        return self.llm_projector(llm_tokens)
    
    def forward_image_only(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode only observation image."""
        return self.image_encoder(observation)


"""
Image Condition Encoder for Diffusion Heatmap Generation.

This module provides encoders for observation images that produce
conditioning vectors for the diffusion process.

Architecture options:
1. Lightweight CNN encoder (default)
2. ResNet-based encoder (optional)
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU."""
    
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
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with two convolutions."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
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
    ):
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]
        
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.image_size = image_size
        
        # ==================== Stem ====================
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(hidden_channels[0]),
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
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels[-1], out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
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


class LLMConditionProjector(nn.Module):
    """
    Projects LLM token features to conditioning dimension.
    
    Handles variable-length sequences by pooling.
    
    Args:
        input_dim: LLM hidden dimension
        output_dim: Output conditioning dimension
        hidden_dim: Intermediate dimension
        pool_method: How to pool sequence ('mean', 'first', 'last', 'max')
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        output_dim: int = 512,
        hidden_dim: int = 1024,
        pool_method: str = 'mean',
    ):
        super().__init__()
        
        self.pool_method = pool_method
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
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
        if self.pool_method == 'mean':
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
    Fuses LLM tokens and observation image into a single conditioning vector.
    
    Architecture:
        LLM Tokens -> LLMConditionProjector -> (B, cond_dim)
        Observation -> ImageConditionEncoder -> (B, cond_dim)
        Concat -> MLP -> (B, cond_dim)
    
    Args:
        llm_dim: LLM hidden dimension
        image_channels: Observation image channels
        cond_dim: Output conditioning dimension
        image_encoder_channels: Channel dims for image encoder
        llm_hidden_dim: Intermediate dim for LLM projector
        pool_method: How to pool LLM sequence
    """
    
    def __init__(
        self,
        llm_dim: int = 2048,
        image_channels: int = 3,
        cond_dim: int = 512,
        image_encoder_channels: List[int] = None,
        llm_hidden_dim: int = 1024,
        pool_method: str = 'mean',
        image_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        
        if image_encoder_channels is None:
            image_encoder_channels = [32, 64, 128, 256]
        
        # LLM projector
        self.llm_projector = LLMConditionProjector(
            input_dim=llm_dim,
            output_dim=cond_dim,
            hidden_dim=llm_hidden_dim,
            pool_method=pool_method,
        )
        
        # Image encoder
        self.image_encoder = ImageConditionEncoder(
            in_channels=image_channels,
            out_dim=cond_dim,
            hidden_channels=image_encoder_channels,
            image_size=image_size,
        )
        
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
    
    def forward(
        self,
        llm_tokens: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse LLM tokens and observation into conditioning vector.
        
        Args:
            llm_tokens: (B, seq_len, llm_dim) or (B, llm_dim) LLM features
            observation: (B, C, H, W) observation image
            
        Returns:
            (B, cond_dim) fused conditioning vector
        """
        # Encode LLM tokens
        llm_cond = self.llm_projector(llm_tokens)  # (B, cond_dim)
        
        # Encode observation
        img_cond = self.image_encoder(observation)  # (B, cond_dim)
        
        # Fuse
        fused = torch.cat([llm_cond, img_cond], dim=-1)  # (B, cond_dim * 2)
        return self.fusion(fused)  # (B, cond_dim)
    
    def forward_llm_only(self, llm_tokens: torch.Tensor) -> torch.Tensor:
        """Encode only LLM tokens (for cases without observation)."""
        return self.llm_projector(llm_tokens)
    
    def forward_image_only(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode only observation image."""
        return self.image_encoder(observation)


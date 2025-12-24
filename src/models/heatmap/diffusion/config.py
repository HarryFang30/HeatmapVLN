"""
Configuration for Diffusion Heatmap Generation.

This module defines the configuration dataclass for the diffusion-based
heatmap generation head.
"""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class DiffusionHeatmapConfig:
    """
    Configuration for DiffusionHeatmapHead.
    
    Defines all hyperparameters for the diffusion-based heatmap generation
    module, including condition encoding, UNet architecture, and diffusion
    scheduler settings.
    
    Attributes:
        # Condition Encoding
        llm_dim: Dimension of LLM token features (Qwen-2.5VL output)
        image_channels: Number of channels in observation image
        cond_dim: Dimension of fused condition vector
        image_size: Size of input observation image (H, W)
        
        # UNet2D Architecture
        in_channels: Input channels for noisy heatmap
        out_channels: Output channels for predicted noise
        block_out_channels: Channel dimensions for each UNet level
        layers_per_block: Number of residual blocks per level
        attention_levels: Which levels to add attention (0-indexed)
        
        # Diffusion Scheduler
        num_train_timesteps: Total training timesteps
        num_inference_steps: Inference steps (can be less than training)
        beta_schedule: Noise schedule type
        prediction_type: What the model predicts ('epsilon' or 'v_prediction')
        
        # Heatmap Output
        heatmap_size: Output heatmap size (H, W)
    """
    
    # ==================== Condition Encoding ====================
    llm_dim: int = 2048                    # Qwen-2.5VL hidden dimension
    image_channels: int = 3                # RGB observation
    cond_dim: int = 512                    # Fused condition dimension
    image_size: Tuple[int, int] = (224, 224)  # Input observation size
    
    # LLM projection
    llm_hidden_dim: int = 1024             # Intermediate projection dim
    llm_pool_method: str = 'mean'          # How to pool LLM sequence
    
    # Image encoder
    image_encoder_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256]
    )
    image_encoder_use_pretrained: bool = False  # Use pretrained ResNet
    
    # ==================== UNet2D Architecture ====================
    in_channels: int = 1                   # Heatmap is single channel
    out_channels: int = 1                  # Predict single channel noise
    
    # Channel dimensions for each encoder/decoder level
    block_out_channels: Tuple[int, ...] = (64, 128, 256)
    
    # Residual blocks per level
    layers_per_block: int = 2
    
    # Attention at deepest levels (indices into block_out_channels)
    attention_levels: Tuple[int, ...] = (2,)  # Attention at 256-channel level
    
    # Normalization
    norm_num_groups: int = 8               # Groups for GroupNorm
    
    # ==================== Diffusion Scheduler ====================
    num_train_timesteps: int = 100         # Training diffusion steps
    num_inference_steps: int = 10          # Inference steps (faster)
    beta_schedule: str = 'squaredcos_cap_v2'  # Cosine schedule
    prediction_type: str = 'epsilon'       # Predict noise
    clip_sample: bool = True               # Clip during sampling
    
    # ==================== Heatmap Output ====================
    heatmap_size: Tuple[int, int] = (64, 64)  # Output heatmap resolution
    
    # ==================== Training ====================
    dropout: float = 0.0                   # Dropout rate
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.llm_dim > 0, "llm_dim must be positive"
        assert self.cond_dim > 0, "cond_dim must be positive"
        assert len(self.block_out_channels) >= 2, "Need at least 2 UNet levels"
        assert all(lvl < len(self.block_out_channels) for lvl in self.attention_levels), \
            "attention_levels indices must be valid"
        assert self.num_inference_steps <= self.num_train_timesteps, \
            "inference steps cannot exceed training steps"


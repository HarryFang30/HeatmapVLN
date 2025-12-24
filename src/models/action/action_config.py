"""
Configuration for Diffusion Action Head

Provides configuration dataclass for the diffusion-based action generation module.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch


@dataclass
class DiffusionActionConfig:
    """
    Configuration for Diffusion Action Head.
    
    Attributes:
        # Action dimensions
        action_dim: Dimension of action output (2 for dx/dy)
        pred_horizon: Number of action steps to predict
        
        # Conditioning
        cond_dim: Input dimension from LLM features
        encoding_size: Projected condition dimension for U-Net
        
        # U-Net architecture
        down_dims: Channel dimensions for each U-Net level
        diffusion_step_embed_dim: Dimension for timestep embedding
        kernel_size: Convolution kernel size
        n_groups: Number of groups for GroupNorm
        cond_predict_scale: Use scale+bias FiLM vs just bias
        
        # Diffusion settings
        num_diffusion_iters: Number of denoising steps
        beta_schedule: Noise schedule type
        
        # Device settings
        device: Target device for computation
        dtype: Data type for computation
    """
    
    # Action dimensions
    action_dim: int = 2  # (dx, dy) for 2D navigation
    pred_horizon: int = 1  # Number of future steps to predict
    
    # Conditioning from LLM
    cond_dim: int = 2048  # Qwen2.5-VL output dimension
    encoding_size: int = 768  # Projected dimension
    
    # U-Net architecture (matching DifNav defaults)
    down_dims: List[int] = field(default_factory=lambda: [192, 384, 768])
    diffusion_step_embed_dim: int = 256
    kernel_size: int = 3
    n_groups: int = 8
    cond_predict_scale: bool = False
    
    # Diffusion settings
    num_diffusion_iters: int = 10
    beta_schedule: str = 'squaredcos_cap_v2'
    clip_sample: bool = True
    prediction_type: str = 'epsilon'  # predict noise
    
    # Device settings
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    
    # Action statistics for normalization (will be updated from data)
    action_stats_min: List[float] = field(default_factory=lambda: [-1.07, -1.05])
    action_stats_max: List[float] = field(default_factory=lambda: [1.07, 1.03])
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.action_dim > 0, "action_dim must be positive"
        assert self.pred_horizon > 0, "pred_horizon must be positive"
        assert self.num_diffusion_iters > 0, "num_diffusion_iters must be positive"
        assert len(self.down_dims) > 0, "down_dims must not be empty"
        assert len(self.action_stats_min) == self.action_dim, \
            f"action_stats_min length ({len(self.action_stats_min)}) must match action_dim ({self.action_dim})"
        assert len(self.action_stats_max) == self.action_dim, \
            f"action_stats_max length ({len(self.action_stats_max)}) must match action_dim ({self.action_dim})"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'action_dim': self.action_dim,
            'pred_horizon': self.pred_horizon,
            'cond_dim': self.cond_dim,
            'encoding_size': self.encoding_size,
            'down_dims': self.down_dims,
            'diffusion_step_embed_dim': self.diffusion_step_embed_dim,
            'kernel_size': self.kernel_size,
            'n_groups': self.n_groups,
            'cond_predict_scale': self.cond_predict_scale,
            'num_diffusion_iters': self.num_diffusion_iters,
            'beta_schedule': self.beta_schedule,
            'clip_sample': self.clip_sample,
            'prediction_type': self.prediction_type,
            'action_stats_min': self.action_stats_min,
            'action_stats_max': self.action_stats_max,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'DiffusionActionConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


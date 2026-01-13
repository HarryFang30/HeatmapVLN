"""
Action Module - Diffusion Policy for VLN Navigation

This module provides action generation capabilities using Diffusion Policy,
adapted from DifNav for Vision-Language Navigation tasks.

Components:
- DiffusionActionHead: Main action generation head using conditional diffusion (legacy)
- TransformerActionHead: Transformer-based action head with diffusion (InternNav style)
- StopPredictionHead: Binary classifier for STOP action prediction
- ProgressPredictionHead: Progress prediction head (0-1 continuous value)
- DiscreteActionHead: Optional full discrete action classifier
- ConditionalUnet1D: U-Net noise prediction network
- DDPMScheduler: DDPM-based noise scheduling

Usage:
    from src.models.action import TransformerActionHead, ProgressPredictionHead
    
    action_head = TransformerActionHead(vlm_token_dim=1024, predict_size=24)
    progress_head = ProgressPredictionHead(input_dim=1024)
    
    trajectory = action_head.get_trajectory(llm_features)  # (B, 24, 3)
    progress = progress_head.get_progress(llm_features)    # (B,)
"""

from .diffusion_action_head import DiffusionActionHead
from .action_config import DiffusionActionConfig
from .utils import normalize_actions, unnormalize_actions, ActionStats
from .stop_head import StopPredictionHead, DiscreteActionHead
from .transformer_action_head import TransformerActionHead, create_transformer_action_head
from .progress_head import ProgressPredictionHead, create_progress_head

__all__ = [
    # Legacy diffusion action head
    'DiffusionActionHead',
    'DiffusionActionConfig', 
    'normalize_actions',
    'unnormalize_actions',
    'ActionStats',
    # New transformer action head (InternNav style)
    'TransformerActionHead',
    'create_transformer_action_head',
    # Stop prediction
    'StopPredictionHead',
    'DiscreteActionHead',
    # Progress prediction
    'ProgressPredictionHead',
    'create_progress_head',
]


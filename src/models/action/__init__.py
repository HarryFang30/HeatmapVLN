"""
Action Module - Diffusion Policy for VLN Navigation

This module provides action generation capabilities using Diffusion Policy,
adapted from DifNav for Vision-Language Navigation tasks.

Components:
- DiffusionActionHead: Main action generation head using conditional diffusion
- StopPredictionHead: Binary classifier for STOP action prediction
- DiscreteActionHead: Optional full discrete action classifier
- ConditionalUnet1D: U-Net noise prediction network
- DDPMScheduler: DDPM-based noise scheduling

Usage:
    from src.models.action import DiffusionActionHead, DiffusionActionConfig, StopPredictionHead
    
    config = DiffusionActionConfig()
    action_head = DiffusionActionHead(config)
    stop_head = StopPredictionHead(input_dim=2048)
    
    actions = action_head(llm_features)  # (B, pred_horizon, action_dim)
    stop_prob = stop_head(llm_features)  # (B,)
"""

from .diffusion_action_head import DiffusionActionHead
from .action_config import DiffusionActionConfig
from .utils import normalize_actions, unnormalize_actions, ActionStats
from .stop_head import StopPredictionHead, DiscreteActionHead

__all__ = [
    'DiffusionActionHead',
    'DiffusionActionConfig', 
    'normalize_actions',
    'unnormalize_actions',
    'ActionStats',
    'StopPredictionHead',
    'DiscreteActionHead',
]


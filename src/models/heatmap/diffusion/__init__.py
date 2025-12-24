"""
Diffusion components for heatmap generation.

This module provides diffusion-based heatmap generation using a lightweight
ConditionalUnet2D architecture.
"""

from .config import DiffusionHeatmapConfig
from .unet2d import ConditionalUnet2D
from .image_encoder import ImageConditionEncoder

__all__ = [
    'DiffusionHeatmapConfig',
    'ConditionalUnet2D',
    'ImageConditionEncoder',
]


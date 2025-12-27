# Heatmap Module
# 
# This module provides diffusion-based heatmap generation for VLN navigation.
# 
# Core Components:
# - DiffusionHeatmapHead: Diffusion-based heatmap generation from LLM tokens + observation
# - Diffusion components: UNet2D, MultiModalConditionEncoder, Configuration

from .diffusion_heatmap_head import DiffusionHeatmapHead, create_diffusion_heatmap_head
from .diffusion import DiffusionHeatmapConfig

# Version info
__version__ = "2.0.0"

# Main exports
__all__ = [
    # Diffusion heatmap generation (primary)
    "DiffusionHeatmapHead",
    "DiffusionHeatmapConfig",
    "create_diffusion_heatmap_head",
    
    # Module metadata
    "__version__",
]

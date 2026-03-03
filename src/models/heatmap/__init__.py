# Heatmap Module
# 
# This module provides heatmap generation heads for VLN navigation.
# 
# Core Components:
# - DiffusionHeatmapHead: Diffusion-based heatmap generation (iterative denoising)
# - DirectHeatmapHead: Direct prediction heatmap generation (single-pass FPN decoder)
# - Diffusion components: UNet2D, MultiModalConditionEncoder, Configuration

from .diffusion_heatmap_head import DiffusionHeatmapHead, create_diffusion_heatmap_head
from .diffusion import DiffusionHeatmapConfig
from .direct_heatmap_head import DirectHeatmapHead, DirectHeatmapConfig

# Version info
__version__ = "3.0.0"

# Main exports
__all__ = [
    # Diffusion heatmap generation
    "DiffusionHeatmapHead",
    "DiffusionHeatmapConfig",
    "create_diffusion_heatmap_head",
    
    # Direct prediction heatmap generation (Plan C)
    "DirectHeatmapHead",
    "DirectHeatmapConfig",
    
    # Module metadata
    "__version__",
]

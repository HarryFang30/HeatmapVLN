# Heatmap Module
# 
# This module provides heatmap generation heads for VLN navigation.
# 
# Core Components:
# - DiffusionHeatmapHead: Diffusion-based heatmap generation (iterative denoising)
# - DirectHeatmapHead: Direct prediction heatmap generation (single-pass FPN decoder)
# - DPTHeatmapHead: DPT-style dense prediction from LLM visual tokens
# - Diffusion components: UNet2D, MultiModalConditionEncoder, Configuration

from .diffusion_heatmap_head import DiffusionHeatmapHead, create_diffusion_heatmap_head
from .diffusion import DiffusionHeatmapConfig
from .direct_heatmap_head import DirectHeatmapHead, DirectHeatmapConfig
from .dpt_heatmap_head import DPTHeatmapHead, DPTHeatmapConfig

# Version info
__version__ = "4.0.0"

# Main exports
__all__ = [
    # Diffusion heatmap generation
    "DiffusionHeatmapHead",
    "DiffusionHeatmapConfig",
    "create_diffusion_heatmap_head",
    
    # Direct prediction heatmap generation (Plan C)
    "DirectHeatmapHead",
    "DirectHeatmapConfig",
    
    # DPT heatmap generation
    "DPTHeatmapHead",
    "DPTHeatmapConfig",
    
    # Module metadata
    "__version__",
]

# Heatmap Module
# 
# This module provides heatmap generation and visualization utilities.
# 
# Core Components:
# - DiffusionHeatmapHead: Diffusion-based heatmap generation from LLM tokens + observation
# - Heatmap generators: Create target heatmaps from various annotation types
# - Visualization tools: Comprehensive plotting and analysis utilities

from .diffusion_heatmap_head import DiffusionHeatmapHead, create_diffusion_heatmap_head
from .diffusion import DiffusionHeatmapConfig
from .generator import (
    generate_hm_from_pt, 
    generate_target_heatmap_from_annotation,
    masked_mean,
    masked_softmax,
    convert_xyxy_to_cxcywh,
    create_multi_scale_heatmap,
    apply_heatmap_augmentation
)

# Visualization tools (optional import due to matplotlib dependency)
try:
    from .visualizer import (
        visualize_points_and_heatmap,
        visualize_bboxes_and_heatmap, 
        visualize_multi_view_heatmaps,
        visualize_attention_comparison,
        create_heatmap_animation_frames
    )
except ImportError:
    # Visualization functions not available - set to None
    visualize_points_and_heatmap = None
    visualize_bboxes_and_heatmap = None
    visualize_multi_view_heatmaps = None
    visualize_attention_comparison = None
    create_heatmap_animation_frames = None

# Version info
__version__ = "2.0.0"

# Main exports
__all__ = [
    # Diffusion heatmap generation (primary)
    "DiffusionHeatmapHead",
    "DiffusionHeatmapConfig",
    "create_diffusion_heatmap_head",
    
    # Generation utilities
    "generate_hm_from_pt",
    "generate_target_heatmap_from_annotation",
    "masked_mean",
    "masked_softmax", 
    "convert_xyxy_to_cxcywh",
    "create_multi_scale_heatmap",
    "apply_heatmap_augmentation",
    
    # Visualization utilities
    "visualize_points_and_heatmap",
    "visualize_bboxes_and_heatmap",
    "visualize_multi_view_heatmaps", 
    "visualize_attention_comparison",
    "create_heatmap_animation_frames",
    
    # Module metadata
    "__version__",
]

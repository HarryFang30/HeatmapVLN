# BridgeVLA-style Heatmap Module
# 
# This module provides comprehensive heatmap generation and visualization utilities
# adapted from BridgeVLA's approach to unified vision-language-action representation.
# 
# Core Components:
# - ConvexUpSample: Learned upsampling for generating high-resolution heatmaps
# - LLMToHeatmapConverter: Convert LLM token outputs to 2D spatial heatmaps  
# - Heatmap generators: Create target heatmaps from various annotation types
# - Visualization tools: Comprehensive plotting and analysis utilities

from .upsampling import ConvexUpSample
from .converter import LLMToHeatmapConverter
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
__version__ = "1.0.0"
__author__ = "Adapted from BridgeVLA"

# Main exports
__all__ = [
    # Core modules
    "ConvexUpSample",
    "LLMToHeatmapConverter", 
    
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
    "__author__"
]

# Convenience function for quick heatmap generation
def quick_heatmap_from_llm(hidden_states, attention_mask, num_views=1, **kwargs):
    """
    Convenience function for quick heatmap generation from LLM outputs.
    
    Args:
        hidden_states: LLM hidden states tensor
        attention_mask: Attention mask tensor  
        num_views: Number of camera views
        **kwargs: Additional arguments for LLMToHeatmapConverter
        
    Returns:
        Generated heatmaps tensor
    """
    converter = LLMToHeatmapConverter(**kwargs)
    return converter(hidden_states, attention_mask, num_views)

# Add convenience function to exports
__all__.append("quick_heatmap_from_llm")
"""
VLN Models Module - Phase 3 Complete Implementation
==================================================

This module provides the complete dual-encoder spatial processing architecture
for Vision-Language Navigation tasks. It integrates VGGT (3D encoder) and
DINOv3 (2D encoder) with space-aware frame sampling and performance optimizations.

Key Components:
- VGGT Integration: 3D geometry processing for all N_m frames
- DINOv3 Compatibility: 2D semantic processing for selected N_k keyframes
- Spatial-MLLM Integration: Complete pipeline orchestration
- Performance Optimizations: Memory-efficient, high-speed processing
- Heatmap Generation: First-person inter-frame spatial visualization

Architecture Overview:
```
N_m Video Frames → VGGT (ALL frames) → Geometry Extraction
                                    ↓
                              Space-aware Sampling → N_k indices
                                    ↓
            ┌─ Index Selection ─ VGGT Features (3D Path)
            │                        ↓
            └─ Index Selection → Original Frames → DINOv3 (2D Path)
                                    ↓
                            Feature Fusion (3D + 2D)
                                    ↓
                            LLM Token Projection
                                    ↓
                        First-Person Inter-Frame Heatmaps
```
"""

# === PHASE 3: SPATIAL ENCODER INTEGRATION ===

# Core spatial encoders
from .vggt.models.vggt import VGGT
from .dinov3.vision_transformer import (
    DinoVisionTransformer,
    vit_base,
    vit_large,
    vit_giant
)
from .dinov3.hub import (
    load_dinov3_model,
    load_local_dinov3,
    verify_model_integrity
)

# Integration layers
from .vggt_integration import (
    VGGTProcessor,
    VGGTIntegrationConfig,
    create_vggt_processor
)
from .dinov3_compatibility import (
    DINOv3CompatibilityLayer,
    DINOv3CompatConfig,
    create_dinov3_compatibility_layer
)

# Complete pipeline integration
from .spatial_mllm_compat import (
    SpatialMLLMPipeline,
    SpatialMLLMIntegrationConfig,
    create_spatial_mllm_pipeline
)

# Performance optimizations
from .performance_optimizer import (
    DualEncoderPerformanceOptimizer,
    PerformanceConfig,
    create_performance_optimizer
)

# === EXISTING COMPONENTS ===

# Import Spatial-MLLM components (existing)
try:
    from .llm import (
        # Main Spatial-MLLM model with VGGT integration
        Qwen2_5_VL_VGGTForConditionalGeneration,
        Qwen2_5_VLProcessor,
        
        # Base Qwen2.5-VL models
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLModel,
        Qwen2_5_VLPreTrainedModel,
        
        # Configuration
        Qwen2_5_VLConfig,
        Qwen2_5_VLVisionConfig,
        
        # Processing
        Qwen2_5_VLProcessorKwargs,
    )
except ImportError as e:
    print(f"Warning: Spatial-MLLM components not available: {e}")
    # Set to None for graceful degradation
    (Qwen2_5_VL_VGGTForConditionalGeneration, Qwen2_5_VLProcessor, 
     Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel, 
     Qwen2_5_VLPreTrainedModel, Qwen2_5_VLConfig, 
     Qwen2_5_VLVisionConfig, Qwen2_5_VLProcessorKwargs) = [None] * 8

# Import Heatmap components (with fallback for missing dependencies)
try:
    from .heatmap import (
        # Core heatmap modules
        ConvexUpSample,
        LLMToHeatmapConverter,
        
        # Generation utilities
        generate_hm_from_pt,
        generate_target_heatmap_from_annotation,
        masked_mean,
        masked_softmax,
        convert_xyxy_to_cxcywh,
        create_multi_scale_heatmap,
        apply_heatmap_augmentation,
        
        # Visualization utilities
        visualize_points_and_heatmap,
        visualize_bboxes_and_heatmap,
        visualize_multi_view_heatmaps,
        visualize_attention_comparison,
        create_heatmap_animation_frames,
        
        # Convenience function
        quick_heatmap_from_llm,
    )
except ImportError as e:
    print(f"Warning: Heatmap visualization components not available due to missing dependencies: {e}")
    # Core components should still work
    from .heatmap.upsampling import ConvexUpSample
    from .heatmap.converter import LLMToHeatmapConverter
    from .heatmap.generator import (
        generate_hm_from_pt,
        generate_target_heatmap_from_annotation,
        masked_mean,
        masked_softmax,
        convert_xyxy_to_cxcywh,
        create_multi_scale_heatmap,
        apply_heatmap_augmentation,
    )
    from .heatmap import quick_heatmap_from_llm
    
    # Set visualization components to None
    visualize_points_and_heatmap = None
    visualize_bboxes_and_heatmap = None  
    visualize_multi_view_heatmaps = None
    visualize_attention_comparison = None
    create_heatmap_animation_frames = None

# Import other model components
try:
    from .mlp import MLP
except ImportError:
    MLP = None

try:
    from .dinov3 import DinoV3Aggregator  
except ImportError:
    DinoV3Aggregator = None

__all__ = [
    # === PHASE 3: SPATIAL ENCODER INTEGRATION ===
    # Core Models
    'VGGT',
    'DinoVisionTransformer',
    'vit_base',
    'vit_large', 
    'vit_giant',
    
    # Model Loading & Utilities
    'load_dinov3_model',
    'load_pretrained_model',
    'verify_model_integrity',
    
    # Integration Components
    'VGGTProcessor',
    'VGGTIntegrationConfig',
    'create_vggt_processor',
    'DINOv3CompatibilityLayer',
    'DINOv3CompatConfig',
    'create_dinov3_compatibility_layer',
    
    # Complete Pipeline
    'SpatialMLLMPipeline',
    'SpatialMLLMIntegrationConfig',
    'create_spatial_mllm_pipeline',
    
    # Performance Optimization
    'DualEncoderPerformanceOptimizer',
    'PerformanceConfig',
    'create_performance_optimizer',
    
    # === EXISTING COMPONENTS ===
    # Spatial-MLLM Components
    "Qwen2_5_VL_VGGTForConditionalGeneration",  # Main Spatial-MLLM with spatial reasoning
    "Qwen2_5_VLProcessor",                      # Spatial-aware processor
    "Qwen2_5_VLForConditionalGeneration",       # Base Qwen2.5-VL
    "Qwen2_5_VLModel",
    "Qwen2_5_VLPreTrainedModel", 
    "Qwen2_5_VLConfig",
    "Qwen2_5_VLVisionConfig",
    "Qwen2_5_VLProcessorKwargs",
    
    # Heatmap Components
    "ConvexUpSample",                           # BridgeVLA-style learned upsampling
    "LLMToHeatmapConverter",                    # LLM token to heatmap pipeline
    "generate_hm_from_pt",                      # Point to heatmap generation
    "generate_target_heatmap_from_annotation",  # Training target generation
    "quick_heatmap_from_llm",                   # Convenience function
    
    # Heatmap utilities
    "masked_mean", "masked_softmax", "convert_xyxy_to_cxcywh",
    "create_multi_scale_heatmap", "apply_heatmap_augmentation",
    
    # Visualization
    "visualize_points_and_heatmap", "visualize_bboxes_and_heatmap", 
    "visualize_multi_view_heatmaps", "visualize_attention_comparison",
    "create_heatmap_animation_frames",
    
    # Other Components
    "MLP",                                      # Multi-layer perceptron
    "DinoV3Aggregator",                         # DINOv3 feature aggregator
]


# === PHASE 3 CONVENIENCE FUNCTIONS ===

def create_optimized_vln_pipeline(
    target_keyframes: int = 16,
    total_frames: int = 128,
    dinov3_model_size: str = "large",
    enable_optimizations: bool = True,
    device: str = "cuda"
):
    """
    Create a complete, optimized VLN pipeline with performance enhancements.
    
    This is the recommended way to create a production-ready VLN pipeline
    that includes all optimizations and compatibility layers.
    
    Args:
        target_keyframes: Number of keyframes to select (N_k)
        total_frames: Total input frames (N_m)
        dinov3_model_size: Size of DINOv3 model ("base", "large", "giant")
        enable_optimizations: Enable performance optimizations
        device: Computing device
        
    Returns:
        Tuple of (pipeline, optimizer) for complete VLN processing
    """
    
    # Create main pipeline
    pipeline = create_spatial_mllm_pipeline(
        target_keyframes=target_keyframes,
        total_frames=total_frames,
        dinov3_model_size=dinov3_model_size,
        device=device,
        verbose=True
    )
    
    # Create performance optimizer if enabled
    optimizer = None
    if enable_optimizations:
        optimizer = create_performance_optimizer(
            enable_all_optimizations=True,
            max_memory_mb=12000,
            target_fps=25.0
        )
    
    return pipeline, optimizer


def get_model_info():
    """Get information about available models and configurations."""
    
    info = {
        'spatial_encoders': {
            'vggt': {
                'description': '3D geometry encoder for all N_m frames',
                'default_config': {
                    'img_size': 518,
                    'patch_size': 14,
                    'embed_dim': 1024
                }
            },
            'dinov3': {
                'description': '2D semantic encoder for selected N_k keyframes', 
                'available_sizes': ['base', 'large', 'giant'],
                'default_config': {
                    'patch_size': 14,
                    'img_size': 518
                }
            }
        },
        'pipeline_features': [
            'Space-aware frame sampling with greedy maximum coverage',
            'Dual-encoder architecture (3D + 2D)',
            'Feature fusion with multiple strategies',
            'First-person inter-frame heatmap generation',
            'Performance optimizations and memory management',
            'Multi-GPU support and async processing'
        ],
        'recommended_config': {
            'target_keyframes': 16,
            'total_frames': 128,
            'dinov3_model_size': 'large',
            'sampling_method': 'hybrid',
            'device': 'cuda'
        }
    }
    
    return info


# Add convenience functions to __all__
__all__.extend([
    'create_optimized_vln_pipeline',
    'get_model_info'
])

# === Usage Examples ===
"""
# 1. Spatial-MLLM Usage (Vision-Language Navigation)
# from src.models import Qwen2_5_VL_VGGTForConditionalGeneration, Qwen2_5_VLProcessor  # Circular import - commented out

model = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained("path/to/model")
processor = Qwen2_5_VLProcessor.from_pretrained("path/to/model")

# Process video input with spatial understanding
inputs = processor(text=[text], videos=video_inputs)
inputs.update({"videos_input": torch.stack(video_inputs) / 255.0})
outputs = model.generate(**inputs)

# 2. Heatmap Generation from LLM outputs
# from src.models import LLMToHeatmapConverter, quick_heatmap_from_llm  # Circular import - commented out

converter = LLMToHeatmapConverter(vlm_dim=2048, patch_size=16, target_size=224)
heatmaps = converter(hidden_states, attention_mask, num_views=3)

# Or use convenience function
heatmaps = quick_heatmap_from_llm(hidden_states, attention_mask, num_views=3)

# 3. Integration: Spatial-MLLM + Heatmap Pipeline
model = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained(model_path)
outputs = model.generate(**inputs, output_hidden_states=True)

# Extract hidden states and convert to heatmaps
hidden_states = outputs.hidden_states[-1]  # Last layer
heatmaps = quick_heatmap_from_llm(hidden_states, inputs.attention_mask)
"""
"""
Data loading and preprocessing utilities for VLN datasets

Active components used by the pipeline:
- vln_heatmap_adapter: Training dataset for heatmap generation
- keyframe_selector: Keyframe selection (used by spatial_mllm_compat)
- frame_sampler: Space-aware sampling (dependency of keyframe_selector)
- spatial_analysis: Spatial novelty detection (dependency of keyframe_selector)
- algorithm_registry: Algorithm registration and base classes
"""

# Core frame sampling components (used by keyframe_selector and algorithm_registry)
from .frame_sampler import (
    SpaceAwareFrameSampler,
    SamplingConfig,
    create_frame_sampler
)

from .spatial_analysis import (
    SpatialNoveltyDetector,
    SpatialAnalysisConfig,
    create_spatial_analyzer
)

from .keyframe_selector import (
    KeyframeSelector,
    KeyframeSelectionConfig,
    create_keyframe_selector
)

__all__ = [
    # Frame Sampling Components (active dependencies)
    'SpaceAwareFrameSampler',
    'SamplingConfig',
    'create_frame_sampler',

    # Spatial Analysis (active dependency)
    'SpatialNoveltyDetector',
    'SpatialAnalysisConfig',
    'create_spatial_analyzer',

    # Keyframe Selector (used by spatial_mllm_compat)
    'KeyframeSelector',
    'KeyframeSelectionConfig',
    'create_keyframe_selector',
]

# Note: The following modules are imported directly by scripts and don't need __init__ exports:
# - vln_heatmap_adapter (VLNHeatmapDataset) - used by train_full_model.py, evaluate.py
# - algorithm_registry (AlgorithmRegistry, BaseFrameSampler) - used internally
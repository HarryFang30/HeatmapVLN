"""
Data loading and preprocessing utilities for VLN datasets

Active components used by the pipeline:
- vln_sliding_window_dataset: Core training dataset (滑动窗口)
- keyframe_selector: Keyframe selection (used by spatial_mllm_compat)
- frame_sampler: Space-aware sampling (dependency of keyframe_selector)

Removed modules:
- vln_heatmap_adapter: Deprecated, functionality merged into sliding_window
- spatial_analysis: Removed, novelty strategies fallback to greedy_coverage
- algorithm_registry: Removed, over-engineered for single algorithm use
"""

# Core frame sampling components
from .frame_sampler import (
    SpaceAwareFrameSampler,
    SamplingConfig,
    create_frame_sampler
)

from .keyframe_selector import (
    KeyframeSelector,
    KeyframeSelectionConfig,
    create_keyframe_selector
)

__all__ = [
    # Frame Sampling Components
    'SpaceAwareFrameSampler',
    'SamplingConfig',
    'create_frame_sampler',

    # Keyframe Selector (used by spatial_mllm_compat)
    'KeyframeSelector',
    'KeyframeSelectionConfig',
    'create_keyframe_selector',
]

# Training dataset classes
from .vln_sliding_window_dataset import VLNSlidingWindowDataset, create_sliding_window_dataloader

__all__ += [
    # Training Datasets
    'VLNSlidingWindowDataset',
    'create_sliding_window_dataloader',
]

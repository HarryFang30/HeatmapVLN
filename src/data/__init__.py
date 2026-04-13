"""
Data loading and preprocessing utilities for VLN datasets.

Modules:
- augmentation: Image augmentation transforms
- heatmap_geometry: Pinhole projection and Gaussian heatmap rendering
- trajectory_utils: Trajectory coordinate transforms and resampling
- sliding_window_dataset: Core sliding-window dataset
- trajectory_dataset: Multi-step trajectory prediction dataset
- factory: Centralized dataset construction from config
"""

from .sliding_window_dataset import (
    VLNSlidingWindowDataset,
    create_sliding_window_dataloader,
)
from .trajectory_dataset import (
    VLNTrajectoryDataset,
    create_trajectory_dataloader,
)
from .factory import (
    build_dataset,
    build_sliding_window_dataset,
    build_trajectory_dataset,
)
from .packing_collator import PackingCollatorForVLN

__all__ = [
    'VLNSlidingWindowDataset',
    'VLNTrajectoryDataset',
    'create_sliding_window_dataloader',
    'create_trajectory_dataloader',
    'build_dataset',
    'build_sliding_window_dataset',
    'build_trajectory_dataset',
    'PackingCollatorForVLN',
]

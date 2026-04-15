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

from __future__ import annotations

import importlib

from .factory import (
    build_dataset,
    build_sliding_window_dataset,
    build_trajectory_dataset,
)
from .packing_collator import PackingCollatorForVLN

__all__ = [
    'PackingCollatorForVLN',
    'VLNSlidingWindowDataset',
    'VLNTrajectoryDataset',
    'build_dataset',
    'build_sliding_window_dataset',
    'build_trajectory_dataset',
    'create_sliding_window_dataloader',
    'create_trajectory_dataloader',
]

_LAZY_ATTRS = {
    "VLNSlidingWindowDataset": ("sliding_window_dataset", "VLNSlidingWindowDataset"),
    "create_sliding_window_dataloader": ("sliding_window_dataset", "create_sliding_window_dataloader"),
    "VLNTrajectoryDataset": ("trajectory_dataset", "VLNTrajectoryDataset"),
    "create_trajectory_dataloader": ("trajectory_dataset", "create_trajectory_dataloader"),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

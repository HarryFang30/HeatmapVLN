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
    build_trajectory_dagger_dataset,
    build_trajectory_dataset,
)
from .packing_collator import PackingCollatorForVLN

__all__ = [
    'PackingCollatorForVLN',
    'VLNSlidingWindowDataset',
    'VLNTrajectoryDataset',
    'TrajectoryDaggerDataset',
    'SourceMixtureDataset',
    'DeterministicMixtureSampler',
    'build_dataset',
    'build_sliding_window_dataset',
    'build_trajectory_dagger_dataset',
    'build_trajectory_dataset',
    'build_expert_dagger_mixture',
    'create_sliding_window_dataloader',
    'create_trajectory_dataloader',
    'trajectory_dagger_collate_fn',
]

_LAZY_ATTRS = {
    "VLNSlidingWindowDataset": ("sliding_window_dataset", "VLNSlidingWindowDataset"),
    "create_sliding_window_dataloader": ("sliding_window_dataset", "create_sliding_window_dataloader"),
    "VLNTrajectoryDataset": ("trajectory_dataset", "VLNTrajectoryDataset"),
    "create_trajectory_dataloader": ("trajectory_dataset", "create_trajectory_dataloader"),
    "TrajectoryDaggerDataset": ("trajectory_dagger_dataset", "TrajectoryDaggerDataset"),
    "SourceMixtureDataset": ("trajectory_dagger_dataset", "SourceMixtureDataset"),
    "DeterministicMixtureSampler": (
        "trajectory_dagger_dataset",
        "DeterministicMixtureSampler",
    ),
    "build_expert_dagger_mixture": (
        "trajectory_dagger_dataset",
        "build_expert_dagger_mixture",
    ),
    "trajectory_dagger_collate_fn": (
        "trajectory_dagger_dataset",
        "trajectory_dagger_collate_fn",
    ),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

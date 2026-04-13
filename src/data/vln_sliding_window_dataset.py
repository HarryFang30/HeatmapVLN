"""Backward-compatible re-exports.

All symbols that were previously defined in this monolithic module are
now split across focused sub-modules.  This file re-exports everything
so that existing ``from src.data.vln_sliding_window_dataset import X``
statements continue to work without modification.
"""

# Dataset classes
# Augmentation
from .augmentation import (
    ColorJitterAugmentation,
    GaussianNoiseAugmentation,
    InternNavStyleAugmentation,
)

# Heatmap geometry
from .heatmap_geometry import (
    compute_adaptive_sigma_pinhole,
    compute_history_heatmap,
    draw_gaussian_point,
    project_point_pinhole,
)
from .sliding_window_dataset import (
    VLNSlidingWindowDataset,
    _evict_from_page_cache,
    create_sliding_window_dataloader,
)
from .trajectory_dataset import (
    VLNTrajectoryDataset,
    create_trajectory_dataloader,
)

# Trajectory utilities
from .trajectory_utils import (
    apply_trajectory_augmentation,
    compute_history_rel_poses,
    get_trajectory_relative_to_frame,
    interpolate_and_resample_trajectory,
    smooth_and_resample_trajectory,
    xy_to_delta_xyt,
)

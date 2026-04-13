"""Backward-compatible re-exports.

All symbols that were previously defined in this monolithic module are
now split across focused sub-modules.  This file re-exports everything
so that existing ``from src.data.vln_sliding_window_dataset import X``
statements continue to work without modification.
"""

# Dataset classes
from .sliding_window_dataset import (  # noqa: F401
    VLNSlidingWindowDataset,
    create_sliding_window_dataloader,
    _evict_from_page_cache,
)
from .trajectory_dataset import (  # noqa: F401
    VLNTrajectoryDataset,
    create_trajectory_dataloader,
)

# Augmentation
from .augmentation import (  # noqa: F401
    ColorJitterAugmentation,
    GaussianNoiseAugmentation,
    InternNavStyleAugmentation,
)

# Heatmap geometry
from .heatmap_geometry import (  # noqa: F401
    project_point_pinhole,
    compute_adaptive_sigma_pinhole,
    draw_gaussian_point,
    compute_history_heatmap,
)

# Trajectory utilities
from .trajectory_utils import (  # noqa: F401
    compute_history_rel_poses,
    get_trajectory_relative_to_frame,
    smooth_and_resample_trajectory,
    xy_to_delta_xyt,
    interpolate_and_resample_trajectory,
    apply_trajectory_augmentation,
)

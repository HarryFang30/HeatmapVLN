"""
Dataset factory functions — centralized dataset construction from config.

Eliminates the 15-25 line parameter lists duplicated across train.py,
evaluation scripts, and visualization scripts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from .sliding_window_dataset import VLNSlidingWindowDataset
    from .trajectory_dataset import VLNTrajectoryDataset

logger = logging.getLogger(__name__)


def build_sliding_window_dataset(
    cfg: dict[str, Any],
    split: str = "train",
    **overrides: Any,
) -> VLNSlidingWindowDataset:
    """Build a VLNSlidingWindowDataset from a full config dict.

    All constructor parameters are extracted from ``cfg['data']`` and
    ``cfg['data']['sliding_window']``.  Any keyword argument in
    *overrides* takes precedence, allowing callers to set eval-specific
    values (e.g. ``enable_augmentation=False``).
    """
    from .sliding_window_dataset import VLNSlidingWindowDataset

    data_cfg = cfg["data"]
    sw_cfg = data_cfg.get("sliding_window", {})

    kwargs = dict(
        root=data_cfg["root"],
        split=split,
        min_history=sw_cfg["min_history"],
        num_history_sample=sw_cfg["num_history_sample"],
        image_size=tuple(data_cfg["image_size"]),
        hm_size=tuple(data_cfg["init_hm_size"]),
        load_depth=sw_cfg.get("load_depth", True),
        cache_poses=sw_cfg.get("cache_poses", True),
        sample_stride=sw_cfg.get("sample_stride", 1),
        clip_level_sampling=sw_cfg.get("clip_level_sampling", True),
        samples_per_clip=sw_cfg.get("samples_per_clip", 2),
        defer_heatmap_to_gpu=sw_cfg.get("defer_heatmap_to_gpu", False),
        load_history_frames=sw_cfg.get("load_history_frames", True),
        max_clips=sw_cfg.get("max_clips", 0),
    )
    kwargs.update(overrides)
    return VLNSlidingWindowDataset(**kwargs)


def build_trajectory_dataset(
    cfg: dict[str, Any],
    split: str = "train",
    **overrides: Any,
) -> VLNTrajectoryDataset:
    """Build a VLNTrajectoryDataset from a full config dict.

    Parameters are extracted from ``cfg['data']``, falling back through
    ``cfg['data']['trajectory']`` then ``cfg['data']['sliding_window']``.
    """
    from .trajectory_dataset import VLNTrajectoryDataset

    data_cfg = cfg["data"]
    traj_cfg = data_cfg.get("trajectory", data_cfg.get("sliding_window", {}))
    future_cfg = traj_cfg.get("future_heatmap", {})
    if not isinstance(future_cfg, dict):
        raise TypeError("data.trajectory.future_heatmap must be a mapping")

    kwargs = dict(
        root=data_cfg["root"],
        split=split,
        min_history=traj_cfg.get("min_history", 5),
        num_history_sample=traj_cfg.get("num_history_sample", 8),
        image_size=tuple(data_cfg["image_size"]),
        hm_size=tuple(data_cfg["init_hm_size"]),
        load_depth=traj_cfg.get("load_depth", True),
        cache_poses=traj_cfg.get("cache_poses", True),
        sample_stride=traj_cfg.get("sample_stride", 1),
        clip_level_sampling=traj_cfg.get("clip_level_sampling", True),
        samples_per_clip=traj_cfg.get("samples_per_clip", 8),
        random_subsequence=traj_cfg.get("random_subsequence", False),
        min_subsequence_length=traj_cfg.get("min_subsequence_length", 30),
        subsequence_samples_per_clip=traj_cfg.get("subsequence_samples_per_clip", 3),
        predict_horizon=traj_cfg.get("predict_horizon", 24),
        action_scale=traj_cfg.get("action_scale", 4.0),
        enable_augmentation=traj_cfg.get("enable_augmentation", True),
        enable_trajectory_augmentation=traj_cfg.get("enable_trajectory_augmentation", True),
        load_traj_images=traj_cfg.get("load_traj_images", False),
        load_history_frames=traj_cfg.get("load_history_frames", True),
        traj_image_size=tuple(traj_cfg.get("traj_image_size", [224, 224])),
        compute_pixel_goal=traj_cfg.get("compute_pixel_goal", False),
        load_lookdown_for_system2=traj_cfg.get(
            "load_lookdown_for_system2",
            traj_cfg.get("load_lookdown_for_sft", False),
        ),
        pixel_goal_direction=traj_cfg.get("pixel_goal_direction", "front"),
        load_history_heatmap=traj_cfg.get("load_history_heatmap", True),
        require_sft_target=traj_cfg.get("require_sft_target", False),
        sft_include_turns=traj_cfg.get("sft_include_turns", True),
        sft_include_forward=traj_cfg.get("sft_include_forward", False),
        sft_num_future_steps=traj_cfg.get("sft_num_future_steps", 4),
        system2_sample_step=traj_cfg.get("system2_sample_step", 4),
        system2_min_pixel_goal_len=traj_cfg.get("system2_min_pixel_goal_len", 3),
        system2_stop_oversample=traj_cfg.get("system2_stop_oversample", 5),
        system2_stop_path_radius_m=traj_cfg.get("system2_stop_path_radius_m", 0.0),
        system2_near_stop_hard_negative_oversample=traj_cfg.get(
            "system2_near_stop_hard_negative_oversample", 0,
        ),
        system2_near_stop_hard_negative_min_path_m=traj_cfg.get(
            "system2_near_stop_hard_negative_min_path_m", 0.0,
        ),
        system2_near_stop_hard_negative_max_path_m=traj_cfg.get(
            "system2_near_stop_hard_negative_max_path_m", 0.0,
        ),
        system2_near_stop_hard_negative_min_goal_distance_m=traj_cfg.get(
            "system2_near_stop_hard_negative_min_goal_distance_m", 0.0,
        ),
        system2_near_stop_hard_negative_max_goal_distance_m=traj_cfg.get(
            "system2_near_stop_hard_negative_max_goal_distance_m", 0.0,
        ),
        include_stop_samples_random_subsequence=traj_cfg.get(
            "include_stop_samples_random_subsequence", False,
        ),
        panoramic_vlm_input=traj_cfg.get("panoramic_vlm_input", True),
        compute_pano_view_pixel_goal=traj_cfg.get("compute_pano_view_pixel_goal"),
        pano_max_side_dist_m=traj_cfg.get("pano_max_side_dist_m", 6.0),
        trajectory_target_convention=traj_cfg.get(
            "trajectory_target_convention", "legacy_pitched_camera"
        ),
        load_future_trajectory_heatmap=future_cfg.get("enabled", False),
        future_heatmap_size=tuple(future_cfg.get("heatmap_size", [64, 64])),
        future_agent_camera_height_m=future_cfg.get(
            "agent_camera_height_m", 1.25
        ),
        max_clips=traj_cfg.get("max_clips", 0),
    )
    kwargs.update(overrides)
    return VLNTrajectoryDataset(**kwargs)


def build_dataset(
    cfg: dict[str, Any],
    split: str = "train",
    **overrides: Any,
) -> Union[VLNSlidingWindowDataset, VLNTrajectoryDataset]:
    """Build the appropriate dataset based on ``cfg['data']['dataset_type']``.

    Dispatches to :func:`build_sliding_window_dataset` or
    :func:`build_trajectory_dataset`.
    """
    dataset_type = cfg["data"].get("dataset_type", "sliding_window")

    if dataset_type == "trajectory":
        return build_trajectory_dataset(cfg, split, **overrides)
    else:
        return build_sliding_window_dataset(cfg, split, **overrides)

"""
Dataset factory functions — centralized dataset construction from config.

Eliminates the 15-25 line parameter lists duplicated across train.py,
evaluation scripts, and visualization scripts.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Union

from ..config_schema import (
    DEPRECATED_HISTORY_CONFIG_KEY,
    SINGLE_VIEW_HISTORY_CONFIG_KEY,
    migrate_single_view_history_config,
)

if TYPE_CHECKING:
    from .sliding_window_dataset import VLNSlidingWindowDataset
    from .trajectory_dagger_dataset import (
        SourceMixtureDataset,
        TrajectoryDaggerDataset,
    )
    from .trajectory_dataset import VLNTrajectoryDataset

logger = logging.getLogger(__name__)


def _resolve_single_view_history_setting(
    section: dict[str, Any],
    *,
    section_path: str,
) -> bool:
    normalized = migrate_single_view_history_config(
        section,
        section_path=section_path,
    )
    value = normalized.get(SINGLE_VIEW_HISTORY_CONFIG_KEY, True)
    if type(value) is not bool:
        raise TypeError(
            f"{section_path}.{SINGLE_VIEW_HISTORY_CONFIG_KEY} must be a boolean, "
            f"got {value!r}"
        )
    return value


def _pop_single_view_history_override(
    overrides: dict[str, Any],
    *,
    context: str,
) -> tuple[bool, bool]:
    present = any(
        key in overrides
        for key in (SINGLE_VIEW_HISTORY_CONFIG_KEY, DEPRECATED_HISTORY_CONFIG_KEY)
    )
    if not present:
        return False, True

    values = {
        key: overrides.pop(key)
        for key in (SINGLE_VIEW_HISTORY_CONFIG_KEY, DEPRECATED_HISTORY_CONFIG_KEY)
        if key in overrides
    }
    return True, _resolve_single_view_history_setting(
        values,
        section_path=f"{context}.overrides",
    )


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
    overrides = dict(overrides)
    has_history_override, history_override = _pop_single_view_history_override(
        overrides,
        context="build_sliding_window_dataset",
    )
    single_view_history = (
        history_override
        if has_history_override
        else _resolve_single_view_history_setting(
            sw_cfg,
            section_path="data.sliding_window",
        )
    )

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
        enable_augmentation=sw_cfg.get("enable_augmentation", False),
        clip_level_sampling=sw_cfg.get("clip_level_sampling", True),
        samples_per_clip=sw_cfg.get("samples_per_clip", 2),
        defer_heatmap_to_gpu=sw_cfg.get("defer_heatmap_to_gpu", False),
        load_single_view_history_frames=single_view_history,
        single_view_rgb_input=sw_cfg.get("single_view_rgb_input", False),
        amb3r_pose_cache_root=sw_cfg.get("amb3r_pose_cache_root"),
        require_amb3r_pose_cache=sw_cfg.get("require_amb3r_pose_cache", False),
        amb3r_pose_cache_max_clips=sw_cfg.get("amb3r_pose_cache_max_clips", 16),
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
    overrides = dict(overrides)
    has_history_override, history_override = _pop_single_view_history_override(
        overrides,
        context="build_trajectory_dataset",
    )
    single_view_history = (
        history_override
        if has_history_override
        else _resolve_single_view_history_setting(
            traj_cfg,
            section_path="data.trajectory",
        )
    )

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
        load_single_view_history_frames=single_view_history,
        single_view_rgb_input=traj_cfg.get("single_view_rgb_input", False),
        amb3r_pose_cache_root=traj_cfg.get("amb3r_pose_cache_root"),
        require_amb3r_pose_cache=traj_cfg.get(
            "require_amb3r_pose_cache", False
        ),
        amb3r_pose_cache_max_clips=traj_cfg.get(
            "amb3r_pose_cache_max_clips", 16
        ),
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
        include_stop_samples_random_subsequence=traj_cfg.get(
            "include_stop_samples_random_subsequence", False,
        ),
        panoramic_vlm_input=traj_cfg.get("panoramic_vlm_input", True),
        compute_pano_view_pixel_goal=traj_cfg.get("compute_pano_view_pixel_goal"),
        compute_aligned_native_pixel_goal=traj_cfg.get(
            "compute_aligned_native_pixel_goal", False
        ),
        pano_max_side_dist_m=traj_cfg.get("pano_max_side_dist_m", 6.0),
        trajectory_target_convention=traj_cfg.get(
            "trajectory_target_convention", "legacy_pitched_camera"
        ),
        load_future_trajectory_heatmap=future_cfg.get("enabled", False),
        future_heatmap_size=tuple(future_cfg.get("heatmap_size", [64, 64])),
        max_clips=traj_cfg.get("max_clips", 0),
    )
    kwargs.update(overrides)
    return VLNTrajectoryDataset(**kwargs)


_SEALED_DAGGER_READERS: dict[str, Any] = {}


def _sealed_dagger_reader(dataset_cls: Any, kwargs: dict[str, Any]) -> Any:
    """Build a sealed-collection reader once per process, then reuse it.

    Constructing one costs a pass over every ``episode.tar`` in the collection
    to verify the ledger -- around fifteen minutes for the 10804-episode round
    on shared storage.  Train and validation read the *same* sealed pool and
    are separated afterwards (by scene, by source type, or by an index view),
    so building it twice doubles the startup of every run for nothing, and
    multiplies that by the rank count on a multi-GPU job.

    The reader is immutable after construction, so sharing one between two
    views is safe.  The key is the full constructor argument set, so any
    difference at all -- different roots, source types, history length, image
    size -- builds its own reader rather than silently returning the wrong one.
    """
    key = json.dumps(
        {name: _hashable(value) for name, value in sorted(kwargs.items())},
        sort_keys=True,
        default=str,
    )
    cached = _SEALED_DAGGER_READERS.get(key)
    if cached is not None:
        logger.info(
            "Reusing the sealed DAgger reader already built in this process "
            "(%d states); the ledger pass is not repeated",
            len(cached),
        )
        return cached
    dataset = dataset_cls(**kwargs)
    _SEALED_DAGGER_READERS[key] = dataset
    return dataset


def _hashable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return str(value)


def build_trajectory_dagger_dataset(
    cfg: dict[str, Any],
    split: str = "train",
    **overrides: Any,
) -> "TrajectoryDaggerDataset":
    """Build the lazy sealed trajectory-DAgger tar dataset."""
    from .trajectory_dagger_dataset import TrajectoryDaggerDataset

    data_cfg = cfg["data"]
    dagger_cfg = data_cfg.get("trajectory_dagger", {})
    use_validation_roots = split != data_cfg.get("train_split", "train") and (
        dagger_cfg.get("val_collection_roots") is not None
        or dagger_cfg.get("val_collection_root") is not None
    )
    if use_validation_roots:
        roots = dagger_cfg.get("val_collection_roots")
        if roots is None:
            roots = dagger_cfg.get("val_collection_root")
    else:
        roots = dagger_cfg.get("collection_roots")
    if roots is None:
        roots = dagger_cfg.get("collection_root")
    if roots is None:
        raise ValueError(
            "data.trajectory_dagger.collection_roots is required"
        )
    kwargs = {
        "collection_roots": roots,
        "allow_unsealed_debug": dagger_cfg.get(
            "allow_unsealed_debug", False
        ),
        "source_types": dagger_cfg.get("source_types"),
        "num_history": dagger_cfg.get("num_history", 8),
        "image_size": tuple(data_cfg["image_size"]),
        "verify_tar_sha256": dagger_cfg.get(
            "verify_tar_sha256", False
        ),
        "require_lookdown": dagger_cfg.get(
            "require_lookdown", False
        ),
        "expected_policy_mode": dagger_cfg.get(
            "expected_policy_mode"
        ),
        "expected_policy_fingerprint": dagger_cfg.get(
            "expected_policy_fingerprint"
        ),
    }
    scene_split = overrides.pop("scene_split", None)
    kwargs.update(overrides)
    dataset = _sealed_dagger_reader(TrajectoryDaggerDataset, kwargs)

    sft_cfg = data_cfg.get("dagger_system2_sft") or {}
    if not sft_cfg.get("enabled", False):
        return dataset

    # System2 supervision is a property of the *labels*, not of the reader, so
    # it is attached here rather than inside the sealed-collection dataset.
    from .dagger_system2_sft import DaggerSystem2SFTDataset

    oracle_views = sft_cfg.get("oracle_views_jsonl")
    if use_validation_roots and sft_cfg.get("val_oracle_views_jsonl"):
        oracle_views = sft_cfg["val_oracle_views_jsonl"]
    if not oracle_views:
        raise ValueError(
            "data.dagger_system2_sft.oracle_views_jsonl is required when the "
            "System2 SFT relabelling is enabled"
        )
    if scene_split is None:
        scene_split = (
            "val" if split != data_cfg.get("train_split", "train") else "train"
        )
    wrapped = DaggerSystem2SFTDataset(
        dataset,
        oracle_views=oracle_views,
        max_turns=int(sft_cfg.get("max_turns", 4)),
        scene_split=str(scene_split),
        val_scene_pct=int(sft_cfg.get("val_scene_pct", 25)),
        stop_supervision=bool(sft_cfg.get("stop_supervision", False)),
        stop_horizon_m=float(sft_cfg.get("stop_horizon_m", 1.0)),
        stop_oversample=int(sft_cfg.get("stop_oversample", 1)),
        cognition_prefix=bool(sft_cfg.get("cognition_prefix", False)),
        # The val slice never sees a placeholder: the placeholder pass at
        # evaluation is applied explicitly by the evaluator.
        prefix_placeholder_fraction=(
            float(sft_cfg.get("prefix_placeholder_fraction", 0.0))
            if scene_split != "val"
            else 0.0
        ),
        reference_path_json=sft_cfg.get("reference_path_json"),
        prefix_distance_bins_m=list(sft_cfg.get("prefix_distance_bins_m", [2.0, 5.0])),
        prefix_progress_bins=int(sft_cfg.get("prefix_progress_bins", 4)),
    )
    logger.info(
        "DAgger System2 SFT relabelling (%s): %s",
        scene_split,
        json.dumps(wrapped.summary(), ensure_ascii=False, sort_keys=True),
    )
    return wrapped


def build_expert_dagger_mixture_dataset(
    cfg: dict[str, Any],
    split: str = "train",
    **overrides: Any,
) -> "SourceMixtureDataset":
    """Build zero-copy expert/normal-DAgger/hard-DAgger source views.

    Generic overrides target the expert trajectory dataset.  Callers can pass
    ``expert_overrides`` and ``dagger_overrides`` dictionaries when the two
    source constructors need different evaluation settings.
    """
    from .trajectory_dagger_dataset import build_expert_dagger_mixture

    forwarded = dict(overrides)
    expert_overrides = dict(forwarded.pop("expert_overrides", {}))
    dagger_overrides = dict(forwarded.pop("dagger_overrides", {}))
    expert_overrides.update(forwarded)

    expert_dataset = build_trajectory_dataset(
        cfg,
        split,
        **expert_overrides,
    )
    dagger_dataset = build_trajectory_dagger_dataset(
        cfg,
        split,
        **dagger_overrides,
    )
    return build_expert_dagger_mixture(expert_dataset, dagger_dataset)


def build_dataset(
    cfg: dict[str, Any],
    split: str = "train",
    **overrides: Any,
) -> Union[
    VLNSlidingWindowDataset,
    VLNTrajectoryDataset,
    TrajectoryDaggerDataset,
    SourceMixtureDataset,
]:
    """Build the dataset selected by ``cfg['data']['dataset_type']``."""
    dataset_type = cfg["data"].get("dataset_type", "sliding_window")

    if dataset_type == "sliding_window":
        return build_sliding_window_dataset(cfg, split, **overrides)
    if dataset_type == "trajectory":
        return build_trajectory_dataset(cfg, split, **overrides)
    if dataset_type == "trajectory_dagger":
        return build_trajectory_dagger_dataset(
            cfg, split, **overrides
        )
    if dataset_type == "expert_dagger_mixture":
        return build_expert_dagger_mixture_dataset(
            cfg, split, **overrides
        )
    raise ValueError(
        f"Unsupported data.dataset_type: {dataset_type!r}"
    )

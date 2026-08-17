from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scripts.training.collate import collate_fn

from src.data.future_trajectory_batch import (
    assert_no_future_teacher_inputs,
    future_target_to_tensors,
    stack_future_trajectory_targets,
)
from src.data.future_trajectory_heatmap import (
    FUTURE_TIME_RANGES,
    build_future_target_from_action_and_poses,
    relative_future_centers_from_world,
    render_future_trajectory_heatmaps,
)
from src.data.trajectory_dataset import VLNTrajectoryDataset


def _intrinsics(size: int = 384) -> np.ndarray:
    return np.asarray(
        (
            (size / 2, 0, (size - 1) / 2),
            (0, size / 2, (size - 1) / 2),
            (0, 0, 1),
        ),
        dtype=np.float32,
    )


def _base_collate_sample() -> dict:
    return {
        "history_frames": torch.zeros(1, 3, 2, 2),
        "current_frame": torch.zeros(3, 2, 2),
        "heatmap": torch.zeros(1, 2, 2),
        "action": torch.zeros(2),
        "action_valid": 1.0,
        "discrete_action": 1,
        "is_stop": 0.0,
        "text": "go",
    }


def test_four_temporal_tubes_and_cardinal_endpoint_anchors() -> None:
    points = np.zeros((32, 3), dtype=np.float32)
    points[0:8] = np.asarray([0.0, 0.0, -2.0], dtype=np.float32)
    points[8:16] = np.asarray([2.0, 0.0, 0.0], dtype=np.float32)
    points[16:24] = np.asarray([0.0, 0.0, 2.0], dtype=np.float32)
    points[24:32] = np.asarray([-2.0, 0.0, 0.0], dtype=np.float32)

    target = render_future_trajectory_heatmaps(
        points, intrinsics=_intrinsics()
    )

    assert tuple(FUTURE_TIME_RANGES) == ((1, 8), (9, 16), (17, 24), (25, 32))
    assert target.heatmap.shape == (4, 4, 64, 64)
    assert target.anchor_heatmap.shape == (4, 4, 64, 64)
    assert target.view5.tolist() == [1, 2, 3, 4]
    np.testing.assert_array_equal(target.visibility, np.eye(4, dtype=np.float32))


def test_flat_expert_camera_height_cancels_but_stairs_are_retained() -> None:
    trajectory = np.zeros((32, 3), dtype=np.float32)
    trajectory[:, 0] = 0.5
    current = np.eye(4, dtype=np.float32)
    current[1, 3] = 1.25
    future = np.repeat(np.eye(4, dtype=np.float32)[None], 5, axis=0)
    future[:, 1, 3] = np.linspace(1.25, 2.25, 5)
    future[:, 2, 3] = -np.linspace(0.0, 4.0, 5)

    relative = relative_future_centers_from_world(current, future)
    np.testing.assert_allclose(relative[:, 1], np.linspace(0.0, 1.0, 5))
    target = build_future_target_from_action_and_poses(
        trajectory,
        action_scale=4.0,
        current_camera_c2w=current,
        raw_future_poses=future,
        intrinsics=_intrinsics(),
    )
    last_peak = np.unravel_index(
        target.anchor_heatmap[3, 0].argmax(), (64, 64)
    )
    assert last_peak[0] < 32


def test_valid_stop_has_four_none_bins_and_invalid_target_is_masked() -> None:
    trajectory = np.zeros((32, 3), dtype=np.float32)
    current = np.eye(4, dtype=np.float32)
    future = current[None]

    stop = build_future_target_from_action_and_poses(
        trajectory,
        action_scale=4.0,
        current_camera_c2w=current,
        raw_future_poses=future,
        intrinsics=_intrinsics(),
    )
    invalid = build_future_target_from_action_and_poses(
        trajectory,
        action_scale=4.0,
        current_camera_c2w=current,
        raw_future_poses=future,
        intrinsics=_intrinsics(),
        trajectory_valid=False,
    )

    assert stop.time_mask.tolist() == [True, True, True, True]
    assert stop.view5.tolist() == [0, 0, 0, 0]
    assert invalid.time_mask.tolist() == [False, False, False, False]


def test_mixed_future_batch_uses_false_mask_and_no_teacher_geometry() -> None:
    points = np.zeros((32, 3), dtype=np.float32)
    points[:, 2] = -2.0
    future = future_target_to_tensors(
        render_future_trajectory_heatmaps(points, intrinsics=_intrinsics())
    )

    stacked = stack_future_trajectory_targets([future, {"text": "history"}])

    assert stacked["future_trajectory_heatmap"].shape == (2, 4, 4, 64, 64)
    assert stacked["future_trajectory_target_present"].tolist() == [True, False]
    assert not stacked["future_trajectory_time_mask"][1].any()
    assert_no_future_teacher_inputs(stacked)


def test_all_unsupervised_future_batch_keeps_fixed_false_contract() -> None:
    stacked = stack_future_trajectory_targets(
        [{"text": "dagger-a"}, {"text": "dagger-b"}]
    )

    assert stacked["future_trajectory_heatmap"].shape == (2, 4, 4, 64, 64)
    assert stacked["future_trajectory_target_present"].tolist() == [False, False]
    assert not stacked["future_trajectory_time_mask"].any()
    assert not stacked["future_trajectory_visibility"].any()
    assert torch.isnan(stacked["future_trajectory_anchor_uv"]).all()


def test_generic_collator_is_unchanged_when_disabled_and_stacks_when_enabled() -> None:
    plain = _base_collate_sample()
    plain_result = collate_fn([plain])
    assert not any(key.startswith("future_trajectory_") for key in plain_result)

    points = np.zeros((32, 3), dtype=np.float32)
    points[:, 2] = -2.0
    enabled = _base_collate_sample()
    enabled.update(
        future_target_to_tensors(
            render_future_trajectory_heatmaps(points, intrinsics=_intrinsics())
        )
    )
    enabled_result = collate_fn([enabled])
    assert enabled_result["future_trajectory_heatmap"].shape == (
        1,
        4,
        4,
        64,
        64,
    )


def test_expert_dataset_builds_future_labels_on_the_fly_without_depth() -> None:
    dataset = object.__new__(VLNTrajectoryDataset)
    dataset.sample_index = [(0, 1)]
    dataset.clips = [Path("/synthetic/clip")]
    dataset.random_subsequence = False
    dataset.num_history_sample = 1
    dataset.load_history_frames = False
    dataset.image_size = (384, 384)
    dataset._is_panoramic = True
    dataset.panoramic_vlm_input = True
    dataset.trajectory_target_convention = "internnav_habitat"
    dataset.load_traj_images = False
    dataset.load_history_heatmap = False
    dataset.defer_heatmap_to_gpu = False
    dataset.hm_size = (64, 64)
    dataset.predict_horizon = 32
    dataset.action_scale = 4.0
    dataset.enable_trajectory_augmentation = False
    dataset.compute_pano_view_pixel_goal = False
    dataset.load_lookdown_for_system2 = False
    dataset.load_future_trajectory_heatmap = True
    dataset.future_heatmap_size = (64, 64)
    dataset.future_agent_camera_height_m = 1.25
    dataset._system2_sft_kind_override = {}

    poses = []
    for idx in range(5):
        pose = np.eye(4, dtype=np.float32)
        pose[1, 3] = 1.25 + 0.1 * idx
        pose[2, 3] = -0.25 * idx
        poses.append(pose)
    trajectory = np.zeros((32, 3), dtype=np.float32)
    trajectory[:, 0] = 0.5

    dataset._load_meta = lambda _clip_idx: {
        "num_frames": 5,
        "instruction": "upstairs",
    }
    dataset._sample_history_indices = lambda _start, _current, _count: [0]
    dataset._load_all_views = lambda _clip, _t: torch.zeros(4, 3, 2, 2)
    dataset._load_history_panoramas = (
        lambda _clip, indices: torch.zeros(len(indices), 4, 3, 2, 2)
    )
    dataset._load_poses = lambda _clip_idx: poses
    dataset._load_intrinsics = lambda _clip_idx, _clip_dir: (
        (384, 384),
        _intrinsics(),
    )
    dataset._compute_trajectory = lambda *args, **kwargs: (
        trajectory.copy(),
        1.0,
        0.5,
    )
    dataset._load_actions = lambda _clip: None
    dataset._load_discrete_actions = lambda _clip: None
    dataset._collect_turn_actions = lambda _actions, _t: []
    dataset._resolve_farthest_pixel_goal = lambda **kwargs: (
        3,
        np.asarray([192, 192], dtype=np.int32),
    )

    result = dataset._build_sample(0)

    assert result["future_trajectory_heatmap"].shape == (4, 4, 64, 64)
    assert result["future_trajectory_time_mask"].all()
    assert not any(
        "future" in key and ("pose" in key or "depth" in key)
        for key in result
    )

    # A turn/fallback row has no native System1 treatment. It retains the
    # fixed tensor contract but cannot consume the privileged remaining route
    # as Future supervision.
    dataset._resolve_farthest_pixel_goal = lambda **kwargs: None
    turn_result = dataset._build_sample(0)
    assert not turn_result["future_trajectory_time_mask"].any()


def test_future_dataset_mode_requires_native_32_steps_and_disables_aug() -> None:
    base_dataset_class = VLNTrajectoryDataset.__mro__[1]
    with patch.object(
        base_dataset_class,
        "__init__",
        return_value=None,
    ), patch.object(
        VLNTrajectoryDataset,
        "random_subsequence",
        False,
        create=True,
    ):
        dataset = VLNTrajectoryDataset(
            root="/unused",
            split="train",
            predict_horizon=32,
            trajectory_target_convention="internnav_habitat",
            enable_trajectory_augmentation=True,
            load_future_trajectory_heatmap=True,
        )
        assert dataset.load_future_trajectory_heatmap is True
        assert dataset.enable_trajectory_augmentation is False

        try:
            VLNTrajectoryDataset(
                root="/unused",
                split="train",
                predict_horizon=24,
                trajectory_target_convention="internnav_habitat",
                load_future_trajectory_heatmap=True,
            )
        except ValueError as exc:
            assert "predict_horizon=32" in str(exc)
        else:
            raise AssertionError("24-step Future target mode was accepted")

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.config_schema import TrajectoryConfig
from src.data.amb3r_pose_cache import (
    AMB3R_ENDPOINT_ROW_POLICY,
    AMB3R_ENDPOINT_SNAPSHOT_TIMING,
    AMB3R_HISTORY_POSE_CONVENTION,
    AMB3R_POSE_CACHE_FILENAME,
    AMB3R_POSE_CACHE_SCHEMA,
    AMB3R_POSE_CONVENTION,
    AMB3R_POSE_PROVIDER,
    AMB3RPoseCache,
    AMB3RPoseCacheError,
)
from src.data.factory import build_trajectory_dataset
from src.data.internnav_heatmap_control_collator import (
    InternNavHeatmapControlCollator,
)
from src.data.trajectory_dataset import VLNTrajectoryDataset


def _write_endpoint_cache(dataset_root: Path, cache_root: Path) -> Path:
    clip = dataset_root / "train" / "scene_a" / "clip_000001"
    clip.mkdir(parents=True)
    output = cache_root / "scene_a" / "clip_000001"
    output.mkdir(parents=True)
    current = np.asarray([1, 3, 5], dtype=np.int64)
    history = np.asarray([[0, -1], [0, 2], [0, 4]], dtype=np.int64)
    counts = np.asarray([1, 2, 2], dtype=np.int64)
    poses = np.zeros((3, 2, 4), dtype=np.float32)
    poses[..., 2] = 1.0
    np.savez(
        output / AMB3R_POSE_CACHE_FILENAME,
        current_frame_ids=current,
        history_frame_ids=history,
        history_counts=counts,
        history_rel_poses=poses,
    )
    manifest = {
        "schema": AMB3R_POSE_CACHE_SCHEMA,
        "clip_key": "scene_a/clip_000001",
        "causal": True,
        "num_history": 2,
        "min_history": 1,
        "pose_convention": AMB3R_POSE_CONVENTION,
        "history_pose_convention": AMB3R_HISTORY_POSE_CONVENTION,
        "pose_provider": "amb3r_vo_da3",
        "per_episode_gt_scale_used": False,
        "gt_pose_read_by_exporter": False,
        "endpoint_only": True,
        "row_policy": AMB3R_ENDPOINT_ROW_POLICY,
        "query_only_at_map_endpoints": True,
        "query_every_frame_from_min_history": False,
        "query_every_frame": False,
        "snapshot_timing": AMB3R_ENDPOINT_SNAPSHOT_TIMING,
        "future_pose_revisions_used": False,
        "translation_scale": 1.0,
        "frame_count": 6,
        "map_init_window": 2,
        "map_every": 2,
        "query_rows": 3,
    }
    (output / f"{AMB3R_POSE_CACHE_FILENAME}.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return clip


def test_endpoint_v2_supports_explicit_split_and_exact_lookup(tmp_path: Path) -> None:
    dataset_root = tmp_path / "expert"
    cache_root = tmp_path / "cache"
    clip = _write_endpoint_cache(dataset_root, cache_root)
    cache = AMB3RPoseCache(
        cache_root,
        dataset_root=dataset_root,
        num_history=2,
        min_history=1,
    )

    assert cache.clip_key(clip) == "scene_a/clip_000001"
    assert cache.current_frame_ids(clip, expected_frame_count=6).tolist() == [1, 3, 5]
    value = cache.lookup(
        clip, current_frame_id=3, history_frame_ids=np.asarray([0, 2])
    )
    assert value.shape == (2, 4)
    with pytest.raises(AMB3RPoseCacheError, match="history identity mismatch"):
        cache.lookup(
            clip, current_frame_id=3, history_frame_ids=np.asarray([1, 2])
        )


def test_internnav_index_uses_endpoint_ids_not_legacy_step_grid() -> None:
    dataset = object.__new__(VLNTrajectoryDataset)
    dataset.sample_index = []
    dataset._sample_subsequence_range = {}
    dataset.system2_sample_step = 4
    dataset.system2_stop_oversample = 0
    dataset.system2_min_pixel_goal_len = 3
    dataset.sft_include_turns = True
    dataset.load_traj_images = True
    dataset.min_history = 5
    dataset.clips = [Path("/synthetic/scene_a/clip_000001")]
    dataset._clip_valid_frames = {0: [19, 27, 35]}
    dataset.amb3r_pose_cache = object()
    dataset._rng = np.random.RandomState(0)
    dataset._load_meta = lambda _: {"num_frames": 36}
    dataset._load_discrete_actions = lambda _: np.ones(36, dtype=np.int64)
    dataset._internnav_sft_frame_kind = lambda *args: "pixel"

    dataset._build_internnav_sample_index()

    assert sorted(frame for _, frame in dataset.sample_index) == [19, 27]


class _FakeCache:
    def __init__(self, expected_history: list[int], value: np.ndarray) -> None:
        self.expected_history = expected_history
        self.value = value
        self.calls = []

    def lookup(self, clip_dir, *, current_frame_id, history_frame_ids):
        self.calls.append((Path(clip_dir), int(current_frame_id), history_frame_ids.tolist()))
        assert int(current_frame_id) == 19
        assert history_frame_ids.tolist() == self.expected_history
        return self.value.copy()


def _one_expert_sample() -> tuple[dict, _FakeCache]:
    dataset = object.__new__(VLNTrajectoryDataset)
    dataset.sample_index = [(0, 19)]
    dataset.clips = [Path("/synthetic/scene_a/clip_000001")]
    dataset.root = Path("/synthetic")
    dataset.random_subsequence = False
    dataset.num_history_sample = 8
    dataset.load_single_view_history_frames = True
    dataset.image_size = (384, 384)
    dataset._is_panoramic = True
    dataset.panoramic_vlm_input = False
    dataset.trajectory_target_convention = "internnav_habitat"
    dataset.load_traj_images = True
    dataset.load_history_heatmap = True
    dataset.defer_heatmap_to_gpu = False
    dataset.hm_size = (64, 64)
    dataset.predict_horizon = 32
    dataset.action_scale = 4.0
    dataset.enable_trajectory_augmentation = False
    dataset.compute_pano_view_pixel_goal = False
    dataset.compute_aligned_native_pixel_goal = False
    dataset.load_lookdown_for_system2 = True
    dataset.load_future_trajectory_heatmap = True
    dataset.future_heatmap_size = (64, 64)
    dataset.traj_sequence_max_len = 12
    dataset.traj_image_size = (2, 2)

    history = np.linspace(0, 18, 8, dtype=np.int64).tolist()
    cached = np.zeros((8, 4), dtype=np.float32)
    cached[:, 0] = np.linspace(-1.0, 0.0, 8)
    cached[:, 2] = 1.0
    cache = _FakeCache(history, cached)
    dataset.amb3r_pose_cache = cache

    poses = []
    for frame in range(30):
        pose = np.eye(4, dtype=np.float32)
        pose[1, 3] = 1.25 + max(frame - 20, 0) * 0.05
        pose[2, 3] = -0.25 * frame
        poses.append(pose)
    system1 = np.zeros((32, 3), dtype=np.float32)
    system1[:, 0] = 1.0
    K = np.asarray(
        ((192.0, 0.0, 191.5), (0.0, 192.0, 191.5), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )

    dataset._load_meta = lambda _: {"num_frames": 30, "instruction": "upstairs"}
    dataset._sample_history_indices = lambda *_: np.asarray(history, dtype=np.int64)
    dataset._load_frames = lambda _clip, ids: torch.zeros(len(ids), 3, 2, 2)
    dataset._load_frame = lambda *_args, **_kwargs: torch.zeros(3, 2, 2)
    dataset._load_poses = lambda _: poses
    dataset._load_intrinsics = lambda *_: ((384, 384), K)
    dataset._compute_per_history_multiview_heatmaps = lambda **_: (
        torch.zeros(8, 4, 64, 64),
        torch.ones(8, 4),
    )
    dataset._compute_trajectory = lambda *args, **kwargs: (system1.copy(), 1.0, 0.5)
    dataset._load_actions = lambda _: None
    dataset._load_discrete_actions = lambda _: None
    dataset._collect_turn_actions = lambda *_: []
    dataset._resolve_farthest_pixel_goal = lambda **_: (5, [192, 192])
    dataset._load_traj_image_raw = lambda *_args, **_kwargs: np.zeros((2, 2, 3), dtype=np.uint8)

    return dataset._build_sample(0), cache


def test_one_expert_sample_is_amb3r_only_and_future_target_is_no_depth() -> None:
    sample, cache = _one_expert_sample()

    assert cache.calls and sample["history_pose_provider"] == AMB3R_POSE_PROVIDER
    assert torch.equal(
        sample["history_rel_poses"], torch.from_numpy(cache.value)
    )
    assert sample["trajectory"].shape == (12, 32, 3)
    assert sample["heatmap"].shape == (8, 4, 64, 64)
    assert sample["gt_visibility"].shape == (8, 4)
    assert sample["future_trajectory_heatmap"].shape == (4, 4, 64, 64)
    assert sample["future_trajectory_time_mask"].all()
    assert sample["future_trajectory_target_source"] == "expert_system1_action_target"
    forbidden = {
        key
        for key in sample
        if "future" in key.lower()
        and any(word in key.lower() for word in ("pose", "depth", "c2w", "camera_point"))
    }
    assert forbidden == set()


class _FakeNativeCollator:
    @staticmethod
    def _stack_padded_history_frames(samples):
        return {"history_frames": torch.stack([s["history_frames"] for s in samples])}

    @staticmethod
    def _stack_padded_first_dim(samples, key):
        return torch.stack([s[key] for s in samples])

    def __call__(self, samples):
        result = {"pano_inputs": {"input_ids": torch.ones(len(samples), 1, dtype=torch.long)}}
        for sample in samples:
            sample.clear()
        return result


def _contract_collator() -> InternNavHeatmapControlCollator:
    collator = object.__new__(InternNavHeatmapControlCollator)
    collator.include_future_trajectory_targets = True
    collator.required_history_pose_provider = AMB3R_POSE_PROVIDER
    collator.native_collator = _FakeNativeCollator()
    collator._encode_heatmap_images = lambda samples: (
        {
            "pixel_values": torch.zeros(9, 3, 2, 2),
            "image_grid_thw": torch.ones(9, 3, dtype=torch.long),
        },
        [8],
    )
    return collator


def test_joint_collator_stacks_future_and_fails_closed_on_provider() -> None:
    sample, _ = _one_expert_sample()
    collator = _contract_collator()
    output = collator([sample])
    assert output["heatmap"].shape == (1, 8, 4, 64, 64)
    assert output["gt_visibility"].shape == (1, 8, 4)
    assert output["future_trajectory_heatmap"].shape == (1, 4, 4, 64, 64)
    assert output["future_trajectory_target_present"].tolist() == [True]
    assert output["history_pose_provider"] == [AMB3R_POSE_PROVIDER]
    assert output["sample_identity"] == [sample["sample_identity"]]
    assert "history_poses" not in output and "current_pose" not in output

    bad, _ = _one_expert_sample()
    bad["history_pose_provider"] = "habitat_gt"
    with pytest.raises(ValueError, match="GT fallback/mixed pose domains"):
        collator([bad])


def test_typed_config_and_factory_forward_strict_inputs(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    trajectory = TrajectoryConfig.model_validate(
        {
            "predict_horizon": 32,
            "load_traj_images": True,
            "require_sft_target": True,
            "enable_trajectory_augmentation": False,
            "trajectory_target_convention": "internnav_habitat",
            "single_view_rgb_input": True,
            "amb3r_pose_cache_root": str(cache_root),
            "require_amb3r_pose_cache": True,
            "future_heatmap": {"enabled": True, "heatmap_size": [64, 64]},
        }
    )
    assert trajectory.future_heatmap.enabled

    cfg = {
        "data": {
            "root": "/expert",
            "image_size": [384, 384],
            "init_hm_size": [64, 64],
            "trajectory": trajectory.model_dump(),
        }
    }
    with patch("src.data.trajectory_dataset.VLNTrajectoryDataset") as dataset_cls:
        build_trajectory_dataset(cfg)
        kwargs = dataset_cls.call_args.kwargs
    assert kwargs["single_view_rgb_input"] is True
    assert kwargs["require_amb3r_pose_cache"] is True
    assert kwargs["load_future_trajectory_heatmap"] is True
    assert kwargs["future_heatmap_size"] == (64, 64)

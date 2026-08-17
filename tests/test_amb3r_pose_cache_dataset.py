from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config_schema import SlidingWindowConfig
from src.data.amb3r_pose_cache import (
    AMB3R_ENDPOINT_ROW_POLICY,
    AMB3R_ENDPOINT_SNAPSHOT_TIMING,
    AMB3R_POSE_CACHE_FILENAME,
    AMB3R_POSE_CACHE_SCHEMA,
    AMB3R_HISTORY_POSE_CONVENTION,
    AMB3R_POSE_CONVENTION,
    AMB3RPoseCache,
    AMB3RPoseCacheError,
)
from src.data.sliding_window_dataset import VLNSlidingWindowDataset
from src.data.factory import build_sliding_window_dataset
from unittest.mock import patch


def _write_cache(
    root: Path,
    *,
    history_ids: list[int] | None = None,
    rel_poses: np.ndarray | None = None,
    frame_count: int = 6,
    map_init_window: int = 6,
    map_every: int = 8,
) -> np.ndarray:
    clip_key = "scene_a/clip_000001"
    target = root / clip_key / AMB3R_POSE_CACHE_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    current_ids = list(range(map_init_window - 1, frame_count, map_every))
    if current_ids[-1] != frame_count - 1:
        current_ids.append(frame_count - 1)
    histories = []
    for current in current_ids:
        histories.append(
            np.arange(current, dtype=np.int64)
            if current <= 8
            else np.linspace(0, current - 1, 8, dtype=np.int64)
        )
    if history_ids is not None:
        if len(current_ids) != 1:
            raise ValueError("history_ids override only supports a one-row cache")
        histories[0] = np.asarray(history_ids, dtype=np.int64)
    counts = np.asarray([len(ids) for ids in histories], dtype=np.int64)
    padded_ids = np.full((len(current_ids), 8), -1, dtype=np.int64)
    for row, ids in enumerate(histories):
        padded_ids[row, : len(ids)] = ids
    if rel_poses is None:
        rel_poses = np.asarray(
            [[float(index + 1), 0.1 * (index + 1), 1.0, 0.0]
             for index in range(len(histories[0]))],
            dtype=np.float32,
        )
    padded_poses = np.zeros((len(current_ids), 8, 4), dtype=np.float32)
    for row, ids in enumerate(histories):
        if row == 0:
            padded_poses[row, : len(ids)] = rel_poses
        else:
            padded_poses[row, : len(ids), 0] = np.arange(1, len(ids) + 1)
            padded_poses[row, : len(ids), 2] = 1.0
    np.savez_compressed(
        target,
        current_frame_ids=np.asarray(current_ids, dtype=np.int64),
        history_frame_ids=padded_ids,
        history_counts=counts,
        history_rel_poses=padded_poses,
    )
    target.with_suffix(target.suffix + ".json").write_text(
        json.dumps(
            {
                "schema": AMB3R_POSE_CACHE_SCHEMA,
                "clip_key": clip_key,
                "frame_count": frame_count,
                "query_rows": len(current_ids),
                "causal": True,
                "num_history": 8,
                "min_history": 5,
                "pose_convention": AMB3R_POSE_CONVENTION,
                "history_pose_convention": AMB3R_HISTORY_POSE_CONVENTION,
                "translation_scale": 1.0,
                "per_episode_gt_scale_used": False,
                "gt_pose_read_by_exporter": False,
                "pose_provider": "amb3r_vo_da3",
                "endpoint_only": True,
                "row_policy": AMB3R_ENDPOINT_ROW_POLICY,
                "query_only_at_map_endpoints": True,
                "query_every_frame_from_min_history": False,
                "query_every_frame": False,
                "snapshot_timing": AMB3R_ENDPOINT_SNAPSHOT_TIMING,
                "future_pose_revisions_used": False,
                "map_init_window": map_init_window,
                "map_every": map_every,
            }
        ),
        encoding="utf-8",
    )
    return rel_poses


def _write_panoramic_clip(root: Path, *, total: int = 6) -> Path:
    clip = root / "scene_a" / "clip_000001"
    chunks = clip / "chunks"
    chunks.mkdir(parents=True)
    height = width = 12
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], total, axis=0)
    # Habitat c2w translation; targets are deliberately non-trivial and are
    # identical in the GT-input and AMB3R-input dataset instances.
    poses[:, 2, 3] = -np.arange(total, dtype=np.float32) * 0.2
    payload: dict[str, np.ndarray] = {
        "frame_ids": np.arange(total, dtype=np.int32),
    }
    for direction in ("front", "right", "back", "left"):
        payload[f"rgb_{direction}"] = np.zeros(
            (total, height, width, 3), dtype=np.uint8
        )
        payload[f"depth_{direction}"] = np.full(
            (total, height, width), 8.0, dtype=np.float32
        )
        payload[f"pose_{direction}"] = poses.copy()
    np.savez_compressed(chunks / "chunk_00000.npz", **payload)
    (clip / "meta.json").write_text(
        json.dumps(
            {
                "scene_id": "scene_a",
                "episode_id": "episode_a",
                "num_frames": total,
                "storage_format": "chunks",
                "data_format": {"depth_unit": "meters"},
            }
        ),
        encoding="utf-8",
    )
    (clip / "intrinsics.json").write_text(
        json.dumps(
            {
                "width": width,
                "height": height,
                "K": [
                    [6.0, 0.0, 5.5],
                    [0.0, 6.0, 5.5],
                    [0.0, 0.0, 1.0],
                ],
            }
        ),
        encoding="utf-8",
    )
    return clip


def _dataset(data_root: Path, cache_root: Path | None) -> VLNSlidingWindowDataset:
    return VLNSlidingWindowDataset(
        root=str(data_root),
        split="all",
        min_history=5,
        num_history_sample=8,
        image_size=(12, 12),
        hm_size=(8, 8),
        load_depth=True,
        cache_poses=True,
        sample_stride=1,
        enable_augmentation=False,
        clip_level_sampling=False,
        load_single_view_history_frames=True,
        single_view_rgb_input=True,
        amb3r_pose_cache_root=None if cache_root is None else str(cache_root),
        require_amb3r_pose_cache=cache_root is not None,
    )


def test_lookup_uses_clip_current_and_exact_history_identity(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    clip = data_root / "scene_a" / "clip_000001"
    clip.mkdir(parents=True)
    cache_root = tmp_path / "cache"
    expected = _write_cache(cache_root)
    cache = AMB3RPoseCache(
        cache_root,
        dataset_root=data_root,
        num_history=8,
        min_history=5,
    )

    actual = cache.lookup(
        clip,
        current_frame_id=5,
        history_frame_ids=np.arange(5, dtype=np.int64),
    )
    np.testing.assert_array_equal(actual, expected)

    with pytest.raises(AMB3RPoseCacheError, match="history identity mismatch"):
        cache.lookup(
            clip,
            current_frame_id=5,
            history_frame_ids=np.asarray([0, 1, 2, 3, 3], dtype=np.int64),
        )


def test_manifest_forbids_non_native_translation_scale(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    clip = data_root / "scene_a" / "clip_000001"
    clip.mkdir(parents=True)
    cache_root = tmp_path / "cache"
    _write_cache(cache_root)
    manifest_path = (
        cache_root
        / "scene_a"
        / "clip_000001"
        / f"{AMB3R_POSE_CACHE_FILENAME}.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["translation_scale"] = 1.2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cache = AMB3RPoseCache(
        cache_root,
        dataset_root=data_root,
        num_history=8,
        min_history=5,
    )
    with pytest.raises(AMB3RPoseCacheError, match="translation_scale"):
        cache.lookup(
            clip,
            current_frame_id=5,
            history_frame_ids=np.arange(5, dtype=np.int64),
        )


def test_cache_rejects_current_frames_outside_endpoint_policy(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    clip = data_root / "scene_a" / "clip_000001"
    clip.mkdir(parents=True)
    cache_root = tmp_path / "cache"
    _write_cache(cache_root)
    cache_path = (
        cache_root / "scene_a" / "clip_000001" / AMB3R_POSE_CACHE_FILENAME
    )
    with np.load(cache_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    arrays["current_frame_ids"] = np.asarray([6], dtype=np.int64)
    np.savez_compressed(cache_path, **arrays)
    cache = AMB3RPoseCache(
        cache_root,
        dataset_root=data_root,
        num_history=8,
        min_history=5,
    )
    with pytest.raises(AMB3RPoseCacheError, match="map-update endpoints"):
        cache.lookup(
            clip,
            current_frame_id=6,
            history_frame_ids=np.arange(5, dtype=np.int64),
        )


def test_dataset_replaces_only_model_pose_input_not_gt_targets(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_panoramic_clip(data_root)
    cache_root = tmp_path / "cache"
    expected = _write_cache(cache_root)

    gt_sample = _dataset(data_root, None)[0]
    vo_sample = _dataset(data_root, cache_root)[0]

    np.testing.assert_array_equal(vo_sample["history_rel_poses"].numpy(), expected)
    assert vo_sample["history_pose_provider"] == "amb3r_vo_cache"
    assert gt_sample["history_pose_provider"] == "habitat_gt"
    assert not np.array_equal(
        vo_sample["history_rel_poses"].numpy(),
        gt_sample["history_rel_poses"].numpy(),
    )
    # Heatmap/visibility still come from the exact same GT c2w/depth path.
    np.testing.assert_array_equal(vo_sample["heatmap"].numpy(), gt_sample["heatmap"].numpy())
    np.testing.assert_array_equal(
        vo_sample["gt_visibility"].numpy(), gt_sample["gt_visibility"].numpy()
    )


def test_missing_cache_row_escapes_dummy_fallback(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_panoramic_clip(data_root)
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    with pytest.raises(AMB3RPoseCacheError, match="Missing required"):
        _dataset(data_root, cache_root)


def test_sparse_endpoint_ids_are_the_only_dataset_sample_population(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _write_panoramic_clip(data_root, total=36)
    cache_root = tmp_path / "cache"
    _write_cache(
        cache_root,
        frame_count=36,
        map_init_window=20,
        map_every=8,
    )

    dataset = VLNSlidingWindowDataset(
        root=str(data_root),
        split="all",
        min_history=5,
        num_history_sample=8,
        image_size=(12, 12),
        hm_size=(8, 8),
        load_depth=True,
        cache_poses=True,
        sample_stride=1,
        enable_augmentation=False,
        samples_per_clip=15,
        clip_level_sampling=True,
        load_single_view_history_frames=True,
        single_view_rgb_input=True,
        amb3r_pose_cache_root=str(cache_root),
        require_amb3r_pose_cache=True,
    )

    expected = {19, 27, 35}
    assert len(dataset) == 3
    assert {current for _, current in dataset.sample_index} == expected
    dataset.set_epoch(4)
    assert len(dataset) == 3
    assert {current for _, current in dataset.sample_index} == expected
    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample["history_pose_provider"] == "amb3r_vo_cache"
        assert int(sample["sample_identity"].rsplit("@", 1)[1]) in expected


def test_sparse_endpoint_train_and_val_sample_counts_follow_configuration(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _write_panoramic_clip(data_root, total=180)
    cache_root = tmp_path / "cache"
    _write_cache(
        cache_root,
        frame_count=180,
        map_init_window=20,
        map_every=8,
    )
    eligible = set(range(19, 180, 8)) | {179}

    common = dict(
        root=str(data_root),
        split="all",
        min_history=5,
        num_history_sample=8,
        image_size=(12, 12),
        hm_size=(8, 8),
        load_depth=False,
        cache_poses=True,
        sample_stride=1,
        enable_augmentation=False,
        clip_level_sampling=True,
        load_single_view_history_frames=True,
        single_view_rgb_input=True,
        amb3r_pose_cache_root=str(cache_root),
        require_amb3r_pose_cache=True,
    )
    train_dataset = VLNSlidingWindowDataset(samples_per_clip=15, **common)
    val_dataset = VLNSlidingWindowDataset(samples_per_clip=8, **common)

    assert len(train_dataset) == 15
    assert len(val_dataset) == 8
    assert {current for _, current in train_dataset.sample_index} <= eligible
    assert {current for _, current in val_dataset.sample_index} <= eligible


def test_random_subsequence_sampling_never_reintroduces_dense_currents(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _write_panoramic_clip(data_root, total=180)
    cache_root = tmp_path / "cache"
    _write_cache(
        cache_root,
        frame_count=180,
        map_init_window=20,
        map_every=8,
    )
    eligible = set(range(19, 180, 8)) | {179}
    dataset = VLNSlidingWindowDataset(
        root=str(data_root),
        split="train",
        min_history=5,
        num_history_sample=8,
        image_size=(12, 12),
        hm_size=(8, 8),
        load_depth=False,
        cache_poses=True,
        sample_stride=1,
        enable_augmentation=False,
        samples_per_clip=15,
        clip_level_sampling=True,
        random_subsequence=True,
        min_subsequence_length=30,
        subsequence_samples_per_clip=4,
        load_single_view_history_frames=True,
        single_view_rgb_input=True,
        amb3r_pose_cache_root=str(cache_root),
        require_amb3r_pose_cache=True,
    )

    assert dataset.sample_index
    assert {current for _, current in dataset.sample_index} <= eligible
    dataset.set_epoch(7)
    assert dataset.sample_index
    assert {current for _, current in dataset.sample_index} <= eligible


def test_legacy_every_frame_cache_schema_is_rejected(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    clip = data_root / "scene_a" / "clip_000001"
    clip.mkdir(parents=True)
    cache_root = tmp_path / "cache"
    _write_cache(cache_root)
    manifest_path = (
        cache_root
        / "scene_a"
        / "clip_000001"
        / f"{AMB3R_POSE_CACHE_FILENAME}.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "heatmapvln-amb3r-causal-training-cache-v1"
    manifest["query_every_frame_from_min_history"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    cache = AMB3RPoseCache(
        cache_root,
        dataset_root=data_root,
        num_history=8,
        min_history=5,
    )
    with pytest.raises(AMB3RPoseCacheError, match="manifest mismatch"):
        cache.current_frame_ids(clip, expected_frame_count=6)


def test_amb3r_mode_forbids_dummy_on_non_cache_sample_error(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_panoramic_clip(data_root)
    cache_root = tmp_path / "cache"
    _write_cache(cache_root)
    dataset = _dataset(data_root, cache_root)
    # Make a normal input/label decoding error happen before cache lookup.
    dataset._load_frames = lambda *args, **kwargs: (_ for _ in ()).throw(
        FileNotFoundError("synthetic missing rgb")
    )

    with pytest.raises(RuntimeError, match="forbids dummy-sample fallback"):
        dataset[0]


def test_config_forbids_amb3r_optional_fallback() -> None:
    with pytest.raises(ValueError, match="optional fallback"):
        SlidingWindowConfig(
            single_view_rgb_input=True,
            amb3r_pose_cache_root="/cache",
            require_amb3r_pose_cache=False,
        )

    validated = SlidingWindowConfig(
        single_view_rgb_input=True,
        amb3r_pose_cache_root="/cache",
        require_amb3r_pose_cache=True,
    )
    assert validated.require_amb3r_pose_cache is True


def test_factory_passes_strict_amb3r_cache_configuration() -> None:
    cfg = {
        "data": {
            "root": "/dataset",
            "image_size": [384, 384],
            "init_hm_size": [64, 64],
            "sliding_window": {
                "min_history": 5,
                "num_history_sample": 8,
                "single_view_rgb_input": True,
                "amb3r_pose_cache_root": "/cache",
                "require_amb3r_pose_cache": True,
                "amb3r_pose_cache_max_clips": 7,
            },
        }
    }
    with patch(
        "src.data.sliding_window_dataset.VLNSlidingWindowDataset"
    ) as mock_dataset:
        build_sliding_window_dataset(cfg, split="train")
    kwargs = mock_dataset.call_args.kwargs
    assert kwargs["amb3r_pose_cache_root"] == "/cache"
    assert kwargs["require_amb3r_pose_cache"] is True
    assert kwargs["amb3r_pose_cache_max_clips"] == 7

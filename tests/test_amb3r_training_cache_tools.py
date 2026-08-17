from __future__ import annotations

import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "amb3r_vo"
sys.path.insert(0, str(TOOLS_DIR))

from cache_contract import (
    CACHE_SCHEMA,
    HISTORY_POSE_CONVENTION,
    POSE_CONVENTION,
    ROW_POLICY,
    endpoint_frame_ids,
    history_indices,
    sidecar_path,
    validate_clip_cache,
)
from build_training_cache_plan import PLAN_SCHEMA, _clip_rows
import export_training_cache_shard


def _write_plan_clip(clip_dir: Path, frame_count: int = 40) -> None:
    (clip_dir / "chunks").mkdir(parents=True)
    (clip_dir / "meta.json").write_text(
        json.dumps({"num_frames": frame_count}),
        encoding="utf-8",
    )
    np.savez_compressed(
        clip_dir / "chunks" / "chunk_000000.npz",
        frame_ids=np.arange(frame_count, dtype=np.int64),
    )


def test_clip_plan_limit_stops_before_scanning_later_clips(tmp_path: Path) -> None:
    scene_root = tmp_path / "dataset" / "train" / "scene"
    _write_plan_clip(scene_root / "clip_000001")
    # This later clip is intentionally malformed. A bounded smoke plan must not
    # inspect the rest of a large dataset after selecting its requested clips.
    (scene_root / "clip_000002").mkdir(parents=True)

    rows = _clip_rows(
        tmp_path / "dataset",
        ["train"],
        max_clips_per_split=1,
    )
    assert [row["clip_key"] for row in rows] == ["scene/clip_000001"]

    try:
        _clip_rows(tmp_path / "dataset", ["train"])
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("unbounded discovery must still validate every clip")


def test_mxc_launcher_forces_safe_da3_attention_compatibility() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_amb3r_pose_training_cache_8gpu_mxc500.sh"
    )
    lines = {
        line.strip()
        for line in launcher.read_text(encoding="utf-8").splitlines()
    }
    assert "export DA3_DISABLE_XFORMERS=1" in lines
    assert "export DA3_SDPA_QUERY_CHUNK_SIZE=256" in lines
    assert not any(
        line.startswith("export DA3_DISABLE_XFORMERS=")
        and line != "export DA3_DISABLE_XFORMERS=1"
        for line in lines
    )
    assert not any(
        line.startswith("export DA3_SDPA_QUERY_CHUNK_SIZE=")
        and line != "export DA3_SDPA_QUERY_CHUNK_SIZE=256"
        for line in lines
    )


def test_endpoint_v2_launchers_use_disjoint_defaults_and_forward_map_schedule() -> None:
    root = Path(__file__).resolve().parents[1]
    cache_launcher = (
        root / "scripts" / "run_amb3r_pose_training_cache_8gpu_mxc500.sh"
    ).read_text(encoding="utf-8")
    train_launcher = (
        root / "scripts" / "run_heatmap_amb3r_pose_adapt_8gpu_mxc500.sh"
    ).read_text(encoding="utf-8")
    smoke_launcher = (
        root / "scripts" / "run_heatmap_amb3r_pose_adapt_8gpu_smoke_mxc500.sh"
    ).read_text(encoding="utf-8")
    pipeline = (
        root / "scripts" / "run_amb3r_pose_adapt_pipeline_8gpu_mxc500.sh"
    ).read_text(encoding="utf-8")

    assert "heatmap_randomwalk_amb3r_endpoint_cache_v2" in cache_launcher
    assert '--map-init-window "${MAP_INIT_WINDOW}"' in cache_launcher
    assert '--map-every "${MAP_EVERY}"' in cache_launcher
    assert "heatmap_randomwalk_amb3r_endpoint_cache_v2" in train_launcher
    assert "output_heatmap_amb3r_pose_adapt_endpoint_v2" in train_launcher
    assert "heatmapvln-amb3r-endpoint-pose-cache-ready-v2" in train_launcher
    assert "official_map_update_endpoints_plus_final" in train_launcher
    assert "heatmap_amb3r_pose_adapt_8gpu_smoke_v2" in smoke_launcher
    assert "FORMAL_MAP_INIT_WINDOW" in pipeline
    assert "FORMAL_MAP_EVERY" in pipeline
    assert "heatmap_randomwalk_amb3r_endpoint_cache_v2" in pipeline
    assert "output_heatmap_amb3r_pose_adapt_endpoint_v2" in pipeline


def test_cache_launcher_uses_ready_markers_and_single_import_prewarm() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_amb3r_pose_training_cache_8gpu_mxc500.sh"
    ).read_text(encoding="utf-8")
    root_guard = launcher.index('if [[ -f "${root_ready}" ]]')
    root_validation = launcher.index("validate_training_cache.py", root_guard)
    shard_guard = launcher.index('if [[ -f "${shard_ready}" ]]')
    shard_validation = launcher.index(
        "validate_training_cache_shard.py", shard_guard
    )
    prewarm = launcher.index("prewarming DA3/online backend imports once")
    worker_launch = launcher.index("export_training_cache_shard.py", prewarm)

    assert root_guard < root_validation < shard_guard < shard_validation
    assert shard_validation < prewarm < worker_launch
    assert "AMB3R_PREWARM_IMPORTS=${AMB3R_PREWARM_IMPORTS:-1}" in launcher
    assert "PYTHONDONTWRITEBYTECODE=1" in launcher[prewarm:worker_launch]
    assert "from amb3r.model_zoo import load_model" in launcher[prewarm:worker_launch]
    assert "from src.vo.online_amb3r import" in launcher[prewarm:worker_launch]


def test_validation_disabled_guard_precedes_all_val_dataset_construction() -> None:
    train_source = (
        Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    ).read_text(encoding="utf-8")
    guard = train_source.index("if not validation_enabled:")
    first_val_build = train_source.index("val_dataset = build_dataset(", guard)
    assert guard < first_val_build
    guarded_region = train_source[guard:first_val_build]
    assert "val_dataset = None" in guarded_region


def _write_valid_cache(path: Path, frame_count: int = 40) -> None:
    minimum = 5
    maximum = 8
    current = endpoint_frame_ids(frame_count, 20, 8)
    final_tail_length = (frame_count - 20) % 8
    ids = np.full((len(current), maximum), -1, dtype=np.int64)
    counts = np.zeros(len(current), dtype=np.int64)
    poses = np.zeros((len(current), maximum, 4), dtype=np.float32)
    for row, value in enumerate(current):
        selected = history_indices(int(value), maximum)
        counts[row] = len(selected)
        ids[row, : len(selected)] = selected
        poses[row, : len(selected), 2] = 1.0
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        current_frame_ids=current,
        history_frame_ids=ids,
        history_counts=counts,
        history_rel_poses=poses,
    )
    sidecar_path(path).write_text(
        json.dumps(
            {
                "schema": CACHE_SCHEMA,
                "clip_key": "scene/clip_000001",
                "causal": True,
                "num_history": maximum,
                "min_history": minimum,
                "frame_count": frame_count,
                "query_rows": len(current),
                "current_frame_start": int(current[0]),
                "current_frame_end_inclusive": int(current[-1]),
                "pose_convention": POSE_CONVENTION,
                "history_pose_convention": HISTORY_POSE_CONVENTION,
                "translation_scale": 1.0,
                "per_episode_gt_scale_used": False,
                "gt_pose_read_by_exporter": False,
                "pose_provider": "amb3r_vo_da3",
                "endpoint_only": True,
                "row_policy": ROW_POLICY,
                "query_only_at_map_endpoints": True,
                "query_every_frame": False,
                "query_every_frame_from_min_history": False,
                "map_init_window": 20,
                "map_every": 8,
                "snapshot_timing": (
                    "immediately_after_endpoint_mapping_before_ingesting_later_rgb"
                ),
                "future_pose_revisions_used": False,
                "forced_final_tail": final_tail_length != 0,
                "final_tail_length": final_tail_length,
                "endpoint_kind_counts": {
                    "initialization": 1,
                    "periodic": (frame_count - 20) // 8,
                    "forced_final_tail": int(final_tail_length != 0),
                },
                "provider_phase_counts": {
                    "stateful_backend": len(current),
                },
            }
        ),
        encoding="utf-8",
    )


def test_history_sampling_matches_dataset_rule() -> None:
    np.testing.assert_array_equal(history_indices(5, 8), np.arange(5))
    np.testing.assert_array_equal(history_indices(8, 8), np.arange(8))
    np.testing.assert_array_equal(
        history_indices(9, 8), np.linspace(0, 8, 8, dtype=np.int64)
    )


def test_endpoint_schedule_has_official_updates_and_unique_final() -> None:
    np.testing.assert_array_equal(endpoint_frame_ids(20), [19])
    np.testing.assert_array_equal(endpoint_frame_ids(28), [19, 27])
    np.testing.assert_array_equal(endpoint_frame_ids(29), [19, 27, 28])
    np.testing.assert_array_equal(endpoint_frame_ids(40), [19, 27, 35, 39])


def test_validate_complete_cache(tmp_path: Path) -> None:
    path = tmp_path / "scene" / "clip_000001" / "amb3r_pose_cache.npz"
    _write_valid_cache(path)
    result = validate_clip_cache(
        path,
        expected_clip_key="scene/clip_000001",
        expected_frame_count=40,
    )
    assert result["query_rows"] == 4


def test_reject_future_history(tmp_path: Path) -> None:
    path = tmp_path / "scene" / "clip_000001" / "amb3r_pose_cache.npz"
    _write_valid_cache(path)
    with np.load(path) as value:
        arrays = {name: value[name] for name in value.files}
    arrays["history_frame_ids"][0, 0] = 19
    np.savez_compressed(path, **arrays)
    try:
        validate_clip_cache(path, expected_clip_key="scene/clip_000001")
    except ValueError as exc:
        assert "history IDs" in str(exc) or "non-causal" in str(exc)
    else:
        raise AssertionError("future history was accepted")


def test_export_queries_only_causal_map_endpoints(
    tmp_path: Path, monkeypatch
) -> None:
    clip_dir = tmp_path / "dataset" / "scene" / "clip_000001"
    clip_dir.mkdir(parents=True)
    frames = np.zeros((40, 2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(
        export_training_cache_shard,
        "_load_rgb_only",
        lambda _: (
            np.arange(40, dtype=np.int64),
            frames,
            {"scene_id": "scene", "episode_id": "episode"},
        ),
    )

    class FakeSession:
        def __init__(self) -> None:
            self.events: list[tuple[str, int]] = []
            self.latest = -1

        def reset(self, session_id: str, *, max_frames: int) -> None:
            assert max_frames == 40
            self.latest = -1

        def ingest(self, session_id: str, *, frame_id: int, **_) -> None:
            assert frame_id == self.latest + 1
            self.latest = frame_id
            self.events.append(("ingest", frame_id))

        def query(
            self,
            session_id: str,
            *,
            current_frame_id: int,
            history_frame_ids: list[int],
            translation_scale: float,
        ) -> SimpleNamespace:
            assert current_frame_id == self.latest
            self.events.append(("query", current_frame_id))
            poses = np.zeros((len(history_frame_ids), 4), dtype=np.float32)
            poses[:, 0] = float(current_frame_id)
            poses[:, 2] = 1.0
            return SimpleNamespace(
                history_rel_poses=poses,
                ready=True,
                provider_phase="stateful_backend",
                last_mapped_frame_id=current_frame_id,
            )

    session = FakeSession()
    cache_root = tmp_path / "cache"
    result = export_training_cache_shard._export_clip(
        entry={
            "clip_key": "scene/clip_000001",
            "clip_path": str(clip_dir),
            "split": "train",
            "frame_count": 40,
        },
        cache_root=cache_root,
        session=session,
        num_history=8,
        min_history=5,
        map_init_window=20,
        map_every=8,
        resolution=(518, 392),
        checkpoint_path=tmp_path / "checkpoint",
    )
    assert result["query_rows"] == 4
    assert [value for kind, value in session.events if kind == "query"] == [
        19,
        27,
        35,
        39,
    ]
    for endpoint in (19, 27, 35, 39):
        query_index = session.events.index(("query", endpoint))
        assert session.events[query_index - 1] == ("ingest", endpoint)
    cache_path = cache_root / "scene" / "clip_000001" / "amb3r_pose_cache.npz"
    with np.load(cache_path) as payload:
        np.testing.assert_array_equal(
            payload["current_frame_ids"], [19, 27, 35, 39]
        )
        np.testing.assert_array_equal(
            payload["history_rel_poses"][:, 0, 0], [19, 27, 35, 39]
        )


def test_stale_shard_ready_is_removed_before_model_load(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    amb3r_root = tmp_path / "amb3r"
    checkpoint = amb3r_root / "checkpoints" / "da3"
    cache_root = tmp_path / "cache"
    for directory in (repo, checkpoint, cache_root):
        directory.mkdir(parents=True)

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": PLAN_SCHEMA,
                "cache_root": str(cache_root),
                "num_history": 8,
                "min_history": 5,
                "translation_scale": 1.0,
                "map_init_window": 20,
                "map_every": 8,
                "num_shards": 1,
                "shards": [{"shard_id": 0, "clips": []}],
            }
        ),
        encoding="utf-8",
    )
    control_root = cache_root / "_control"
    control_root.mkdir()
    shard_ready = control_root / "shard_00.ready.json"
    shard_ready.write_text('{"complete": true}\n', encoding="utf-8")
    failures_path = control_root / "shard_00.failures.jsonl"
    historical_failures = '{"error": "historical diagnostic"}\n'
    failures_path.write_text(historical_failures, encoding="utf-8")

    load_calls: list[tuple[str, str]] = []

    def fail_model_load(name: str, *, ckpt_path: str):
        load_calls.append((name, ckpt_path))
        raise RuntimeError("synthetic model-load failure")

    amb3r_package = types.ModuleType("amb3r")
    amb3r_package.__path__ = []
    model_zoo = types.ModuleType("amb3r.model_zoo")
    model_zoo.load_model = fail_model_load
    online_amb3r = types.ModuleType("src.vo.online_amb3r")
    online_amb3r.OnlineAMB3RSession = object
    online_amb3r.StatefulAMB3RBackend = object
    monkeypatch.setitem(sys.modules, "amb3r", amb3r_package)
    monkeypatch.setitem(sys.modules, "amb3r.model_zoo", model_zoo)
    monkeypatch.setitem(sys.modules, "src.vo.online_amb3r", online_amb3r)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_training_cache_shard.py",
            "--plan",
            str(plan_path),
            "--shard-id",
            "0",
            "--repo",
            str(repo),
            "--amb3r-root",
            str(amb3r_root),
            "--da3-checkpoint",
            str(checkpoint),
            "--allowed-root",
            str(tmp_path),
        ],
    )

    try:
        export_training_cache_shard.main()
    except RuntimeError as exc:
        assert str(exc) == "synthetic model-load failure"
    else:
        raise AssertionError("synthetic model-load failure was not propagated")

    assert load_calls == [("da3", str(checkpoint))]
    assert not shard_ready.exists()
    assert failures_path.read_text(encoding="utf-8") == historical_failures

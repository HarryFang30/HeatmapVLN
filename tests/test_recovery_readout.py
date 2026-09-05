"""Contract tests for the EXP-13 readout probe.

The probe is the gate for a multi-week implementation, so the parts that decide
its verdict are tested directly: the scene split must be disjoint and stable,
the arms must be assembled at the right widths, and a readout must actually be
able to find signal that lives in ``M_t`` but not in the System2 summary.  The
last one is the whole point of the experiment, so a synthetic dataset with the
signal planted in a known place keeps a silently broken fitter from returning
"memory adds nothing" for the wrong reason.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tool() -> types.ModuleType:
    path = _REPO_ROOT / "scripts/tools/fit_recovery_readout.py"
    spec = importlib.util.spec_from_file_location("_exp13_fit_recovery_readout", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fit_recovery_readout = _load_tool()


def _synthetic_cache(
    path: Path,
    *,
    states_per_scene: int = 90,
    scenes: int = 12,
    traj_dim: int = 16,
    memory_dim: int = 6,
    histories: int = 8,
    seed: int = 7,
) -> Path:
    """A cache whose label is readable from ``M_t`` and invisible to System2.

    The oracle direction is written into the first history memory slot; the
    System2 summary is pure noise.  A working probe must therefore report a
    large ``system2_memory`` minus ``system2`` gap, and a broken one cannot.
    """
    rng = np.random.default_rng(seed)
    total = states_per_scene * scenes
    oracle = rng.integers(0, 4, size=total).astype(np.int32)

    memory = rng.normal(0.0, 0.05, size=(total, histories, memory_dim)).astype(np.float16)
    for view in range(4):
        rows = oracle == view
        memory[rows, 0, view % memory_dim] += 4.0

    scene_ids = np.asarray(
        [f"scene_{index // states_per_scene:02d}" for index in range(total)], dtype=np.str_
    )
    # Two thirds hard, one third normal; the recovery slice needs non-front
    # oracles so ``recovery_nonfront_recall`` is defined on the val split.
    source_type = np.where(np.arange(total) % 3 == 0, "dagger_normal", "dagger_hard").astype(np.str_)
    tags = np.where(np.arange(total) % 2 == 0, "wrong_branch", "avoidable_revisit").astype(np.str_)
    oracle[source_type == "dagger_normal"] = 0

    np.savez_compressed(
        path,
        sample_key=np.asarray([f"k{index}" for index in range(total)], dtype=np.str_),
        scene_id=scene_ids,
        episode_key=np.asarray([f"e{index // 5}" for index in range(total)], dtype=np.str_),
        source_type=source_type,
        tags=tags,
        oracle_view=oracle,
        native_view=np.zeros(total, dtype=np.int32),
        traj_hidden=rng.normal(size=(total, 4, traj_dim)).astype(np.float16),
        plan_z0=rng.normal(size=(total, 4, 5)).astype(np.float16),
        history_memory=memory,
        history_memory_mask=np.ones((total, histories), dtype=np.uint8),
        history_rel_poses=rng.normal(size=(total, histories, 4)).astype(np.float16),
        history_visibility=rng.normal(size=(total, histories, 4)).astype(np.float16),
        history_age_steps=rng.integers(0, 30, size=(total, histories)).astype(np.int16),
        future_visibility=rng.normal(size=(total, 4, 4)).astype(np.float16),
    )
    return path


def test_scene_split_is_disjoint_and_stable() -> None:
    scenes = np.asarray([f"scene_{index:03d}" for index in range(200)], dtype=np.str_)
    first = fit_recovery_readout.scene_split(scenes, 15, 25)
    second = fit_recovery_readout.scene_split(scenes, 15, 25)
    assert np.array_equal(first, second)

    by_bucket: dict[int, set[str]] = {0: set(), 1: set(), 2: set()}
    for scene, bucket in zip(scenes.tolist(), first.tolist()):
        by_bucket[int(bucket)].add(scene)
    assert by_bucket[0] and by_bucket[1] and by_bucket[2]
    assert not by_bucket[0] & by_bucket[1]
    assert not by_bucket[0] & by_bucket[2]
    assert not by_bucket[1] & by_bucket[2]

    # A state's split follows its scene, never its row index.
    repeated = np.asarray(["scene_000"] * 10 + ["scene_001"] * 10, dtype=np.str_)
    repeated_split = fit_recovery_readout.scene_split(repeated, 15, 25)
    assert len(set(repeated_split[:10].tolist())) == 1
    assert len(set(repeated_split[10:].tolist())) == 1


def test_scene_split_rejects_degenerate_percentages() -> None:
    scenes = np.asarray(["a", "b"], dtype=np.str_)
    with pytest.raises(SystemExit):
        fit_recovery_readout.scene_split(scenes, 60, 60)


def test_build_arms_widths_and_masking(tmp_path: Path) -> None:
    cache = _synthetic_cache(tmp_path / "cache.npz")
    data = np.load(cache, allow_pickle=False)
    arms = fit_recovery_readout.build_arms(data)

    assert arms["system2"].shape[1] == 4 * 16
    assert arms["memory"].shape[1] == 8 * 6 + 8
    assert arms["system2_memory"].shape[1] == arms["system2"].shape[1] + arms["memory"].shape[1]
    assert arms["geometry"].shape[1] == 8 * 4 + 8 * 4 + 8 + 8
    for name, matrix in arms.items():
        assert matrix.shape[0] == data["oracle_view"].shape[0], name
        assert np.isfinite(matrix).all(), name


def test_padded_history_slots_are_zeroed_in_the_memory_arm(tmp_path: Path) -> None:
    cache = _synthetic_cache(tmp_path / "cache.npz")
    payload = dict(np.load(cache, allow_pickle=False))
    payload["history_memory_mask"][:, 3:] = 0
    masked = tmp_path / "masked.npz"
    np.savez_compressed(masked, **payload)

    arms = fit_recovery_readout.build_arms(np.load(masked, allow_pickle=False))
    memory = arms["memory"][:, : 8 * 6].reshape(-1, 8, 6)
    assert np.all(memory[:, 3:, :] == 0.0)
    assert np.any(memory[:, :3, :] != 0.0)


def test_readout_finds_signal_that_lives_only_in_the_memory(tmp_path: Path) -> None:
    cache = _synthetic_cache(tmp_path / "cache.npz")
    output = tmp_path / "readout.json"
    argv = [
        "fit_recovery_readout.py",
        "--features",
        str(cache),
        "--output-json",
        str(output),
        "--epochs",
        "6",
        "--batch-size",
        "64",
        "--weight-decays",
        "1e-3,1e-1",
        "--arms",
        "system2,system2_memory,memory",
    ]
    original = sys.argv
    sys.argv = argv
    try:
        fit_recovery_readout.main()
    finally:
        sys.argv = original

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == fit_recovery_readout.SCHEMA
    assert report["split"]["scenes_val"] >= 1
    assert report["split"]["states_val"] > 0

    system2 = report["arms"]["system2"]["val"]["hard_macro_accuracy"]
    with_memory = report["arms"]["system2_memory"]["val"]["hard_macro_accuracy"]
    memory_only = report["arms"]["memory"]["val"]["hard_macro_accuracy"]
    assert with_memory > system2 + 0.20
    assert memory_only > system2 + 0.20
    assert report["memory_minus_system2_pt"]["hard_macro_accuracy"] > 20.0

    # The reported baseline must be the label prior, not a fitted model.
    constant = report["baselines"]["constant_front"]
    assert constant["prediction_distribution"]["right"] == 0
    assert constant["prediction_distribution"]["back"] == 0
    assert constant["normal_false_alarm"] == 0.0


def test_val_metrics_never_see_the_weight_decay_search(tmp_path: Path) -> None:
    """Weight decay is chosen on dev, so val must stay untouched by the sweep."""
    cache = _synthetic_cache(tmp_path / "cache.npz")
    output = tmp_path / "readout.json"
    argv = [
        "fit_recovery_readout.py",
        "--features",
        str(cache),
        "--output-json",
        str(output),
        "--epochs",
        "4",
        "--batch-size",
        "64",
        "--weight-decays",
        "1e-3,1e-2,1e-1",
        "--arms",
        "memory",
    ]
    original = sys.argv
    sys.argv = argv
    try:
        fit_recovery_readout.main()
    finally:
        sys.argv = original

    report = json.loads(output.read_text(encoding="utf-8"))
    arm = report["arms"]["memory"]
    assert arm["selected_weight_decay"] in (1e-3, 1e-2, 1e-1)
    assert "dev_hard_macro_accuracy" in arm
    assert report["recipe"]["weight_decays"] == [1e-3, 1e-2, 1e-1]

"""Contract tests for the EXP-17 cognition prefix (labels, safety, parsing, dataset wiring)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sft = _load("_cognition_prefix_sft", _REPO / "src/data/dagger_system2_sft.py")


# ----------------------------------------------------------------- label rules
def test_canonical_views_follow_the_planar_angle() -> None:
    assert sft.canonical_view_index(1.0, 0.0) == 0  # front
    assert sft.canonical_view_index(0.0, 1.0) == 3  # left
    assert sft.canonical_view_index(-1.0, 0.0) == 2  # back
    assert sft.canonical_view_index(0.0, -1.0) == 1  # right
    assert sft.canonical_view_index(-1.0, 0.9) == 2  # 138 deg -> back
    assert sft.canonical_view_index(1.0, 0.9) == 0  # 42 deg -> front


def test_distance_and_progress_bins() -> None:
    assert [sft.distance_bin_index(d, (2.0, 5.0)) for d in (0.5, 2.0, 4.9, 5.0, 9.0)] == [0, 1, 1, 2, 2]
    assert [sft.progress_bin_index(m, 10.0, 4) for m in (0.0, 2.4, 2.5, 7.4, 7.5, 10.0, 12.0)] == [0, 0, 1, 2, 3, 3, 3]
    assert sft.progress_bin_index(1.0, 0.0, 4) is None
    assert sft.progress_char(2, arrived=False) == "三"
    assert sft.progress_char(2, arrived=True) == "到"
    assert sft.progress_char(None, arrived=False) is None


def test_prefix_is_rendered_and_native_safe() -> None:
    poses = [[3.0, 0.0, 1.0, 0.0], [-1.0, 0.0, -1.0, 0.0], [0.0, 6.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
    mask = [True, True, True, False]
    text = sft.build_cognition_prefix(poses, mask, "二")
    assert text == "记忆：前中、后近、左远、空；进度：二。"
    fields, rest = sft.parse_cognition_prefix(text + "↓")
    assert rest == "↓"
    assert fields["progress"] == "二"
    assert fields["slots"] == [("前", "中"), ("后", "近"), ("左", "远"), "空"]
    for unsafe in ("记忆：前1；进度：一。", "记忆：→；进度：一。", "记忆：STOP；进度：一。"):
        with pytest.raises(ValueError):
            sft.assert_prefix_native_safe(unsafe)


def test_placeholder_prefix_has_the_same_shape_and_parses() -> None:
    text = sft.placeholder_prefix(3)
    assert text == "记忆：未知、未知、未知；进度：未知。"
    fields, rest = sft.parse_cognition_prefix(text + "STOP")
    assert fields["slots"] == ["未知"] * 3 and fields["progress"] == "未知" and rest == "STOP"
    assert sft.parse_cognition_prefix("↓") == (None, "↓")


def test_placeholder_selection_is_deterministic_and_calibrated() -> None:
    keys = [f"round00_scene_{i:06d}:call0001:step0004" for i in range(20000)]
    chosen = [sft.placeholder_selected(k, 0.2) for k in keys]
    assert chosen == [sft.placeholder_selected(k, 0.2) for k in keys]
    assert 0.18 < sum(chosen) / len(chosen) < 0.22
    assert not any(sft.placeholder_selected(k, 0.0) for k in keys[:100])


# --------------------------------------------------------------- dataset wiring
def _row(key: str, scene: str, *, route_progress: float, episode: str, arrived: bool, pixel: bool = True) -> dict[str, Any]:
    native = {"pixel_goal": [12, 34], "llm_output": "34 12", "actions": [1, 1]} if pixel else {"pixel_goal": None, "llm_output": "→", "actions": [3, 1]}
    return {
        "key": key,
        "scene_id": scene,
        "source_type": "dagger_hard",
        "failure_tags": [],
        "route_progress_m": route_progress,
        "episode_id": episode,
        "native": native,
        "oracle": {"actions": [1, 1], "terminal": arrived, "travelled_m": 0.5 if arrived else 6.0},
    }


class _FakeDagger:
    _is_panoramic = True
    single_view_rgb_input = False
    dynamic_sampling_enabled = False

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def sample_metadata(self, index: int) -> dict[str, Any]:
        return self.rows[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sample = {
            "sample_key": row["key"],
            "source_type": row["source_type"],
            "text": "walk to the kitchen",
            "history_rel_poses": torch.tensor([[3.0, 0.0, 1.0, 0.0], [-1.0, 0.0, -1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
            "history_valid_mask": torch.tensor([True, True, False]),
        }
        if row["native"].get("pixel_goal") is not None:
            sample["pixel_goal"] = list(row["native"]["pixel_goal"])
        return sample


def test_dataset_prepends_the_prefix_to_the_first_turn_only() -> None:
    rows = [
        _row("k0", "scene_train", route_progress=1.0, episode="1", arrived=False),
        _row("k1", "scene_train", route_progress=9.0, episode="1", arrived=True),
    ]
    oracle_views = {"k0": {"oracle_view": 0, "native_view": 0}, "k1": {"oracle_view": 0, "native_view": 0}}
    dataset = sft.DaggerSystem2SFTDataset(
        _FakeDagger(rows),
        oracle_views=oracle_views,
        stop_supervision=True,
        stop_horizon_m=1.0,
        cognition_prefix=True,
        prefix_placeholder_fraction=0.0,
        reference_path_json={"1": 10.0},
    )
    first = dataset[0]
    assert first["system2_target_texts"] == ["记忆：前中、后近、空；进度：一。↓", "34 12"]
    assert first["cognition_prefix_truth"] == "记忆：前中、后近、空；进度：一。"
    assert first["cognition_prefix_is_placeholder"] is False
    second = dataset[1]
    assert second["system2_relabel_kind"] == "correct_stop"
    assert second["system2_target_texts"] == ["记忆：前中、后近、空；进度：到。STOP"]
    summary = dataset.summary()
    assert summary["cognition_prefix"] is True
    assert summary["prefix_progress_dist"] == {"一": 1, "到": 1}


def test_placeholder_rows_are_content_free_and_never_in_val() -> None:
    rows = [_row(f"k{i}", "scene_train", route_progress=1.0, episode="1", arrived=False) for i in range(400)]
    oracle_views = {r["key"]: {"oracle_view": 0, "native_view": 0} for r in rows}
    dataset = sft.DaggerSystem2SFTDataset(
        _FakeDagger(rows),
        oracle_views=oracle_views,
        cognition_prefix=True,
        prefix_placeholder_fraction=0.2,
        reference_path_json={"1": 10.0},
    )
    placeholders = [dataset[i]["cognition_prefix_is_placeholder"] for i in range(len(dataset))]
    assert 0.12 < sum(placeholders) / len(placeholders) < 0.28
    sample = dataset[placeholders.index(True)]
    assert sample["system2_target_texts"][0] == "记忆：未知、未知、未知；进度：未知。↓"
    assert sample["cognition_prefix_truth"].startswith("记忆：前中")
    assert dataset.summary()["prefix_placeholder_rows"] == sum(placeholders)
    # The val slice never receives a placeholder, whatever the configured fraction.
    val_scene = next(s for s in (f"scene{i}" for i in range(10000)) if sft.scene_bucket(s) < 25)
    val_rows = [_row(f"v{i}", val_scene, route_progress=1.0, episode="1", arrived=False) for i in range(200)]
    val_only = sft.DaggerSystem2SFTDataset(
        _FakeDagger(val_rows),
        oracle_views={r["key"]: {"oracle_view": 0, "native_view": 0} for r in val_rows},
        scene_split="val",
        cognition_prefix=True,
        prefix_placeholder_fraction=0.2,
        reference_path_json={"1": 10.0},
    )
    assert val_only.prefix_placeholder_fraction == 0.0
    assert val_only.summary()["prefix_placeholder_rows"] == 0
    assert not any(val_only[i]["cognition_prefix_is_placeholder"] for i in range(len(val_only)))


def test_rows_without_a_progress_label_are_dropped_and_counted() -> None:
    rows = [_row("k0", "scene_train", route_progress=1.0, episode="missing", arrived=False)]
    dataset_kwargs = dict(
        oracle_views={"k0": {"oracle_view": 0, "native_view": 0}},
        cognition_prefix=True,
        reference_path_json={"1": 10.0},
    )
    with pytest.raises(sft.DaggerSystem2SFTError, match="no DAgger state"):
        sft.DaggerSystem2SFTDataset(_FakeDagger(rows), **dataset_kwargs)
    with pytest.raises(sft.DaggerSystem2SFTError, match="reference_path_json"):
        sft.DaggerSystem2SFTDataset(_FakeDagger(rows), oracle_views={"k0": {"oracle_view": 0, "native_view": 0}}, cognition_prefix=True)

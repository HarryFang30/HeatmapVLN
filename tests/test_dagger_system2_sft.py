"""Contract tests for the EXP-13 DAgger relabelling.

The supervision this builds is the whole experiment, and two of its failure
modes are silent: emitting a turn in the wrong direction, and "correcting"
states where native was already right (which would fine-tune the policy away
from behaviour EXP-01 spent a month certifying).  Both are pinned here, along
with the scene split that has to agree with the readout probe's.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module() -> types.ModuleType:
    path = _REPO_ROOT / "src/data/dagger_system2_sft.py"
    spec = importlib.util.spec_from_file_location("_exp13_dagger_system2_sft", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sft = _load_module()


def _sample(
    key: str,
    *,
    scene_id: str = "scene_a",
    source_type: str = "dagger_hard",
    oracle_actions: list[int] | None = None,
    native_actions: list[int] | None = None,
    pixel_goal: list[int] | None = None,
    llm_output: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    native: dict[str, Any] = {"actions": native_actions or [1, 1]}
    if pixel_goal is not None:
        native["pixel_goal"] = pixel_goal
        native["llm_output"] = (
            llm_output if llm_output is not None else f"{pixel_goal[1]} {pixel_goal[0]}"
        )
    elif llm_output is not None:
        native["llm_output"] = llm_output
    return {
        "key": key,
        "scene_id": scene_id,
        "source_type": source_type,
        "failure_tags": tags or [],
        "native": native,
        "oracle": {"actions": oracle_actions if oracle_actions is not None else [1, 1], "terminal": False},
    }


def _row(oracle_view: int | None, native_view: int | None) -> dict[str, Any]:
    return {"oracle_view": oracle_view, "native_view": native_view}


class _FakeDagger:
    """The parts of TrajectoryDaggerDataset the relabeller depends on."""

    _is_panoramic = True
    single_view_rgb_input = False
    dynamic_sampling_enabled = False

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def sample_metadata(self, index: int) -> dict[str, Any]:
        return self.samples[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        row = {
            "sample_key": sample["key"],
            "source_type": sample["source_type"],
            "text": "walk to the kitchen",
        }
        pixel_goal = sample["native"].get("pixel_goal")
        if pixel_goal is not None:
            row["pixel_goal"] = list(pixel_goal)
            row["pano_pixel_goal"] = list(pixel_goal)
            row["pano_sample_kind"] = "pixel"
        return row


def test_leading_turn_run_reads_only_the_opening_same_direction_run() -> None:
    assert sft.leading_turn_run([3, 3, 3, 1, 3]) == (3, 3)
    assert sft.leading_turn_run([2, 2, 3, 3]) == (2, 2)
    assert sft.leading_turn_run([1, 3, 3]) == (0, None)
    assert sft.leading_turn_run([]) == (0, None)


def test_native_answer_is_reproduced_in_the_released_v_u_order() -> None:
    # The collector stores [u, v]; the policy says "v u".
    assert sft.native_pixel_answer([247, 450]) == "450 247"


def test_a_disagreeing_oracle_turn_is_corrected_in_the_arrow_protocol() -> None:
    sample = _sample("k", oracle_actions=[2, 2, 2, 2, 2, 1], pixel_goal=[247, 450])
    plan = sft.plan_for_sample(sample, _row(oracle_view=3, native_view=0), max_turns=4)
    assert plan["kind"] == "correct_turn"
    assert plan["target_texts"] == ["←←←←"]
    assert plan["turn_run"] == 5
    assert plan["emitted_turns"] == 4
    # A turn answer must not keep the pixel goal, or the prompt would still
    # carry the look-down turn the policy no longer needs.
    assert plan["drop_pixel_goal"] is True


def test_turn_direction_follows_the_oracle_not_the_native_policy() -> None:
    right = sft.plan_for_sample(
        _sample("k", oracle_actions=[3, 3], native_actions=[2, 2]),
        _row(oracle_view=1, native_view=3),
    )
    assert right["target_texts"] == ["→→"]
    left = sft.plan_for_sample(
        _sample("k", oracle_actions=[2, 2], native_actions=[3, 3]),
        _row(oracle_view=3, native_view=1),
    )
    assert left["target_texts"] == ["←←"]


def test_agreement_keeps_the_frozen_policys_own_answer() -> None:
    sample = _sample("k", oracle_actions=[2, 2, 1], pixel_goal=[247, 450])
    plan = sft.plan_for_sample(sample, _row(oracle_view=0, native_view=0))
    assert plan["kind"] == "keep_pixel"
    assert plan["target_texts"] == ["↓", "450 247"]
    assert plan["drop_pixel_goal"] is False


def test_a_forward_oracle_is_never_corrected_even_when_views_disagree() -> None:
    # There is no turn to emit, so the arrow protocol has nothing to say.
    plan = sft.plan_for_sample(
        _sample("k", oracle_actions=[1, 1, 2], pixel_goal=[10, 20]),
        _row(oracle_view=2, native_view=0),
    )
    assert plan["kind"] == "keep_pixel"


def test_missing_oracle_directions_never_produce_a_correction() -> None:
    sample = _sample("k", oracle_actions=[2, 2], pixel_goal=[10, 20])
    assert sft.plan_for_sample(sample, None)["kind"] == "keep_pixel"
    assert sft.plan_for_sample(sample, _row(None, 0))["kind"] == "keep_pixel"
    assert sft.plan_for_sample(sample, _row(3, None))["kind"] == "keep_pixel"


def test_a_pixel_goal_that_disagrees_with_the_recorded_answer_is_dropped() -> None:
    sample = _sample("k", pixel_goal=[247, 450], llm_output="1 2")
    plan = sft.plan_for_sample(sample, _row(0, 0))
    assert plan["kind"] is None
    assert "disagrees" in plan["reason"]


def test_native_turn_and_stop_answers_are_reproduced() -> None:
    turn = sft.plan_for_sample(
        _sample("k", oracle_actions=[1], native_actions=[3, 3, 1]), _row(0, 0)
    )
    assert turn["kind"] == "keep_turn"
    assert turn["target_texts"] == ["→→"]

    stop = sft.plan_for_sample(
        _sample("k", oracle_actions=[1], native_actions=[0]), _row(0, 0)
    )
    assert stop["kind"] == "keep_stop"
    assert stop["target_texts"] == ["STOP"]


def test_dataset_attaches_targets_and_reports_what_it_changed() -> None:
    samples = [
        _sample("a", oracle_actions=[2, 2, 2], pixel_goal=[10, 20], tags=["wrong_branch"]),
        _sample("b", oracle_actions=[1, 1], pixel_goal=[30, 40]),
        _sample("c", source_type="dagger_normal", oracle_actions=[1], pixel_goal=[50, 60]),
    ]
    views = {"a": _row(3, 0), "b": _row(0, 0), "c": _row(0, 0)}
    dataset = sft.DaggerSystem2SFTDataset(_FakeDagger(samples), oracle_views=views)

    assert len(dataset) == 3
    summary = dataset.summary()
    assert summary["kinds"] == {"correct_turn": 1, "keep_pixel": 2}
    assert summary["corrected_fraction"] == pytest.approx(1 / 3)
    assert summary["by_source_type"]["dagger_normal"] == {"keep_pixel": 1}

    corrected = dataset[0]
    assert corrected["system2_target_texts"] == ["←←←"]
    assert corrected["system2_relabel_kind"] == "correct_turn"
    assert "pixel_goal" not in corrected
    assert "pano_pixel_goal" not in corrected

    kept = dataset[1]
    assert kept["system2_target_texts"] == ["↓", "40 30"]
    assert kept["pixel_goal"] == [30, 40]


def test_unlabelled_rows_are_dropped_and_counted_never_defaulted() -> None:
    samples = [
        _sample("a", oracle_actions=[1], pixel_goal=[10, 20]),
        _sample("b", oracle_actions=[1], native_actions=[], llm_output=None),
    ]
    dataset = sft.DaggerSystem2SFTDataset(
        _FakeDagger(samples), oracle_views={"a": _row(0, 0), "b": _row(0, 0)}
    )
    assert len(dataset) == 1
    assert sum(dataset.dropped.values()) == 1


def test_rows_without_an_oracle_row_are_dropped_by_default() -> None:
    samples = [_sample("a", pixel_goal=[10, 20]), _sample("b", pixel_goal=[30, 40])]
    dataset = sft.DaggerSystem2SFTDataset(_FakeDagger(samples), oracle_views={"a": _row(0, 0)})
    assert len(dataset) == 1
    assert dataset.dropped["no_oracle_row"] == 1


def test_scene_split_is_disjoint_and_matches_the_readout_probe_hash() -> None:
    import hashlib

    scenes = [f"scene_{index:03d}" for index in range(60)]
    samples = [
        _sample(f"k{index}", scene_id=scene, pixel_goal=[10, 20])
        for index, scene in enumerate(scenes)
    ]
    views = {f"k{index}": _row(0, 0) for index in range(len(scenes))}

    train = sft.DaggerSystem2SFTDataset(
        _FakeDagger(samples), oracle_views=views, scene_split="train", val_scene_pct=25
    )
    val = sft.DaggerSystem2SFTDataset(
        _FakeDagger(samples), oracle_views=views, scene_split="val", val_scene_pct=25
    )
    train_scenes = {plan["scene_id"] for plan in train.plans}
    val_scenes = {plan["scene_id"] for plan in val.plans}
    assert train_scenes and val_scenes
    assert not train_scenes & val_scenes
    assert train_scenes | val_scenes == set(scenes)

    # Identical to fit_recovery_readout.scene_split's bucketing.
    for scene in val_scenes:
        digest = hashlib.md5(scene.encode("utf-8")).hexdigest()
        assert int(digest[:8], 16) % 100 < 25


def test_dataset_refuses_a_reader_without_sealed_metadata() -> None:
    class _NoMetadata:
        def __len__(self) -> int:
            return 0

    with pytest.raises(sft.DaggerSystem2SFTError, match="sample_metadata"):
        sft.DaggerSystem2SFTDataset(_NoMetadata(), oracle_views={})

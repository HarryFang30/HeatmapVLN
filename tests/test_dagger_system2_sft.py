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


# --- EXP-14: the stop decision -------------------------------------------------


def _terminal_sample(
    key: str,
    *,
    travelled_m: Any,
    terminal: bool = True,
    pixel_goal: list[int] | None = None,
    native_actions: list[int] | None = None,
    oracle_actions: list[int] | None = None,
    use_default_pixel: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    if pixel_goal is None and use_default_pixel:
        pixel_goal = [10, 20]
    sample = _sample(
        key,
        pixel_goal=pixel_goal,
        native_actions=native_actions,
        oracle_actions=oracle_actions,
        **kwargs,
    )
    sample["oracle"]["terminal"] = terminal
    sample["oracle"]["travelled_m"] = travelled_m
    sample["oracle"]["kind"] = "route_recovery"
    return sample


def _scene_in(split: str) -> str:
    for index in range(1000):
        name = f"scene_{index:04d}"
        in_val = sft.scene_bucket(name) < 25
        if (split == "val") == in_val:
            return name
    raise AssertionError("no scene name landed in the requested split")


def test_stop_supervision_is_off_unless_requested() -> None:
    # The EXP-13 arms must relabel exactly as their ledger entry registered.
    sample = _terminal_sample("k", travelled_m=0.4)
    assert sft.plan_for_sample(sample, _row(0, 0))["kind"] == "keep_pixel"


def test_an_oracle_at_the_goal_relabels_a_walking_native_to_stop() -> None:
    sample = _terminal_sample("k", travelled_m=0.4)
    plan = sft.plan_for_sample(sample, _row(0, 0), stop_supervision=True, stop_horizon_m=1.0)
    assert plan["kind"] == "correct_stop"
    assert plan["target_texts"] == ["STOP"]
    assert plan["drop_pixel_goal"] is True
    assert plan["oracle_remaining_m"] == pytest.approx(0.4)
    assert plan["oracle_kind"] == "route_recovery"


def test_a_terminal_route_longer_than_the_horizon_is_not_a_stop() -> None:
    sample = _terminal_sample("k", travelled_m=1.6)
    plan = sft.plan_for_sample(sample, _row(0, 0), stop_supervision=True, stop_horizon_m=1.0)
    assert plan["kind"] == "keep_pixel"
    # Exactly on the horizon counts as within it.
    on_edge = _terminal_sample("k", travelled_m=1.0)
    assert sft.plan_for_sample(on_edge, _row(0, 0), stop_supervision=True)["kind"] == "correct_stop"


def test_a_non_terminal_route_is_never_a_stop_even_when_short() -> None:
    sample = _terminal_sample("k", travelled_m=0.2, terminal=False)
    assert sft.plan_for_sample(sample, _row(0, 0), stop_supervision=True)["kind"] == "keep_pixel"


def test_stop_outranks_a_disagreeing_turn_and_only_when_switched_on() -> None:
    sample = _terminal_sample("k", travelled_m=0.5, oracle_actions=[2, 2, 1])
    assert sft.plan_for_sample(sample, _row(3, 0), stop_supervision=True)["kind"] == "correct_stop"
    assert sft.plan_for_sample(sample, _row(3, 0))["kind"] == "correct_turn"


def test_a_native_that_already_stopped_is_kept_not_corrected() -> None:
    sample = _terminal_sample(
        "k", travelled_m=0.1, native_actions=[0], use_default_pixel=False
    )
    plan = sft.plan_for_sample(sample, _row(0, 0), stop_supervision=True)
    assert plan["kind"] == "keep_stop"
    assert plan["target_texts"] == ["STOP"]


def test_malformed_terminal_metadata_never_produces_a_stop() -> None:
    sample = _terminal_sample("k", travelled_m="nan")
    assert sft.plan_for_sample(sample, _row(0, 0), stop_supervision=True)["kind"] == "keep_pixel"
    sample["oracle"]["travelled_m"] = None
    assert sft.plan_for_sample(sample, _row(0, 0), stop_supervision=True)["kind"] == "keep_pixel"
    sample["oracle"]["travelled_m"] = -0.5
    assert sft.plan_for_sample(sample, _row(0, 0), stop_supervision=True)["kind"] == "keep_pixel"
    assert sft.oracle_stops_within({"terminal": True, "travelled_m": "0.2"}, 1.0) == (True, 0.2)
    assert sft.oracle_stops_within("not a dict", 1.0) == (False, None)


def test_stop_oversampling_applies_to_train_only_and_is_reported() -> None:
    train_scene, val_scene = _scene_in("train"), _scene_in("val")
    samples = [
        _terminal_sample("a", travelled_m=0.3, scene_id=train_scene),
        _sample("b", pixel_goal=[30, 40], scene_id=train_scene),
        _terminal_sample("c", travelled_m=0.3, scene_id=val_scene),
    ]
    views = {key: _row(0, 0) for key in "abc"}

    train = sft.DaggerSystem2SFTDataset(
        _FakeDagger(samples),
        oracle_views=views,
        scene_split="train",
        stop_supervision=True,
        stop_oversample=3,
    )
    assert len(train) == 4  # a x3 + b
    summary = train.summary()
    assert summary["states"] == 4
    assert summary["unique_states"] == 2
    assert summary["oversampled_copies"] == 2
    assert summary["kinds"] == {"correct_stop": 1, "keep_pixel": 1}
    assert summary["corrected_fraction"] == pytest.approx(0.5)
    assert summary["corrected_stop_fraction"] == pytest.approx(0.5)
    assert summary["corrected_turn_fraction"] == 0
    targets = [train[index]["system2_target_texts"] for index in range(len(train))]
    assert targets.count(["STOP"]) == 3
    assert "pixel_goal" not in train[0]

    val = sft.DaggerSystem2SFTDataset(
        _FakeDagger(samples),
        oracle_views=views,
        scene_split="val",
        stop_supervision=True,
        stop_oversample=3,
    )
    assert len(val) == 1
    assert val.summary()["stop_oversample"] == 1
    assert val.summary()["oversampled_copies"] == 0


def test_stop_knobs_are_validated() -> None:
    samples = [_sample("a", pixel_goal=[10, 20])]
    views = {"a": _row(0, 0)}
    with pytest.raises(sft.DaggerSystem2SFTError, match="stop_horizon_m"):
        sft.DaggerSystem2SFTDataset(
            _FakeDagger(samples), oracle_views=views, stop_supervision=True, stop_horizon_m=0.0
        )
    with pytest.raises(sft.DaggerSystem2SFTError, match="stop_oversample"):
        sft.DaggerSystem2SFTDataset(_FakeDagger(samples), oracle_views=views, stop_oversample=0)
    with pytest.raises(sft.DaggerSystem2SFTError, match="requires stop_supervision"):
        sft.DaggerSystem2SFTDataset(_FakeDagger(samples), oracle_views=views, stop_oversample=2)


def test_the_exp13_summary_numbers_do_not_move_without_stop_supervision() -> None:
    # The 13-B log line reads corrected_fraction; adding EXP-14 must not change it.
    samples = [
        _terminal_sample("a", travelled_m=0.2, oracle_actions=[2, 2, 2], tags=["wrong_branch"]),
        _sample("b", pixel_goal=[1, 2]),
    ]
    dataset = sft.DaggerSystem2SFTDataset(
        _FakeDagger(samples), oracle_views={"a": _row(3, 0), "b": _row(0, 0)}
    )
    summary = dataset.summary()
    assert summary["kinds"] == {"correct_turn": 1, "keep_pixel": 1}
    assert summary["corrected_fraction"] == pytest.approx(0.5)
    assert summary["corrected_turn_fraction"] == pytest.approx(0.5)
    assert summary["corrected_stop_fraction"] == 0
    assert summary["states"] == summary["unique_states"] == 2
    assert summary["stop_supervision"] is False


def test_a_stop_needs_no_direction_row_but_everything_else_still_does() -> None:
    # The states with no EXP-12 row are the ones already inside the oracle's
    # goal tolerance; under stop supervision they are STOP targets, not drops.
    at_goal = _terminal_sample("a", travelled_m=0.0, oracle_actions=[])
    walking = _sample("b", pixel_goal=[30, 40])
    samples = [at_goal, walking]

    with_stop = sft.DaggerSystem2SFTDataset(
        _FakeDagger(samples), oracle_views={}, stop_supervision=True
    )
    assert len(with_stop) == 1
    assert with_stop.plans[0]["kind"] == "correct_stop"
    assert with_stop.dropped == {"no_oracle_row": 1}

    # The EXP-13 arms (no stop supervision) drop both, exactly as before: with
    # nothing left to label the constructor refuses.
    with pytest.raises(sft.DaggerSystem2SFTError, match="no DAgger state could be relabelled"):
        sft.DaggerSystem2SFTDataset(_FakeDagger(samples), oracle_views={})

    # And a row-less walking state is still dropped even under stop supervision.
    mixed = sft.DaggerSystem2SFTDataset(
        _FakeDagger([walking, _sample("c", pixel_goal=[1, 2])]),
        oracle_views={"c": _row(0, 0)},
        stop_supervision=True,
    )
    assert len(mixed) == 1
    assert mixed.dropped == {"no_oracle_row": 1}

"""Contract tests for the decision-level scorer shared by EXP-13 B and EXP-14.

The scorer turns per-state rows into the numbers the ledger's criteria read.
Two properties are pinned: the stop metrics must be computed from the
``predicted_is_stop`` flag exactly (recall on ``correct_stop`` rows, false alarm
on every row that is not a stop target), and rows written by the v1 schema --
which has no such flag -- must be excluded rather than counted as "did not
stop", which would silently deflate the false-alarm rate that caps the arm.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tool() -> types.ModuleType:
    path = _REPO_ROOT / "scripts/tools/eval_system2_recovery_decisions.py"
    spec = importlib.util.spec_from_file_location("_exp13_eval_system2_decisions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _state(
    kind: str,
    *,
    source: str = "dagger_hard",
    predicted_stop: bool | None = False,
    predicted_direction: str | None = None,
    target_direction: str | None = None,
    first_match: bool = True,
    all_match: bool = True,
) -> dict[str, Any]:
    row = {
        "sample_key": f"{kind}-{source}",
        "scene_id": "scene",
        "source_type": source,
        "relabel_kind": kind,
        "failure_tags": [],
        "target_texts": ["STOP"] if kind in tool.STOP_TARGET_KINDS else ["↓", "1 2"],
        "predicted_first_text": "STOP" if predicted_stop else "↓",
        "target_first_text": "STOP" if kind in tool.STOP_TARGET_KINDS else "↓",
        "predicted_direction": predicted_direction,
        "target_direction": target_direction,
        "target_is_stop": kind in tool.STOP_TARGET_KINDS,
        "first_token_match": first_match,
        "all_supervised_tokens_match": all_match,
        "supervised_tokens": 2,
    }
    if predicted_stop is not None:
        row["predicted_is_stop"] = predicted_stop
    return row


def test_stop_recall_and_false_alarm_are_read_from_the_first_token() -> None:
    states = [
        _state("correct_stop", predicted_stop=True),
        _state("correct_stop", predicted_stop=False),
        _state("keep_pixel", source="dagger_normal", predicted_stop=False),
        _state("keep_pixel", source="dagger_normal", predicted_stop=True),
        _state("correct_turn", predicted_direction="left", target_direction="left"),
    ]
    report = tool.summarise(states)
    assert report["stop_recall"] == pytest.approx(0.5)
    assert report["stop_recall_states"] == 2
    # Every row that is not a stop target counts: two keep_pixel and one correct_turn.
    assert report["stop_false_alarm"] == pytest.approx(1 / 3)
    assert report["stop_false_alarm_states"] == 3
    assert report["stop_false_alarm_by_source"] == {
        "dagger_hard": pytest.approx(0.0),
        "dagger_normal": pytest.approx(0.5),
    }
    assert report["by_kind"]["keep_pixel"]["predicted_stop"] == pytest.approx(0.5)
    # The EXP-13 numbers are untouched by the additions.
    assert report["recovery_turn_accuracy"] == 1.0
    assert report["normal_preservation"] == 1.0


def test_v1_rows_without_the_stop_flag_are_excluded_not_defaulted() -> None:
    states = [
        _state("correct_stop", predicted_stop=None),
        _state("keep_pixel", predicted_stop=None),
        _state("keep_pixel", predicted_stop=True),
    ]
    report = tool.summarise(states)
    assert report["stop_recall"] is None
    assert report["stop_recall_states"] == 0
    assert report["stop_false_alarm"] == pytest.approx(1.0)
    assert report["stop_false_alarm_states"] == 1
    assert report["by_kind"]["correct_stop"]["predicted_stop"] is None


def test_a_native_stop_that_is_kept_is_not_a_false_alarm_candidate() -> None:
    states = [
        _state("keep_stop", predicted_stop=True),
        _state("keep_pixel", predicted_stop=False),
    ]
    report = tool.summarise(states)
    assert report["stop_false_alarm"] == pytest.approx(0.0)
    assert report["stop_false_alarm_states"] == 1


def test_an_empty_report_carries_none_not_zero() -> None:
    report = tool.summarise([])
    assert report["stop_recall"] is None
    assert report["stop_false_alarm"] is None
    assert report["recovery_turn_accuracy"] is None
    assert report["stop_false_alarm_by_source"] == {}


def test_stop_recall_is_split_on_the_backtrack_tag() -> None:
    # 84.9% of the stop rows carry necessary_backtrack, so a model that learned
    # the tag rather than the goal would look good overall and bad off-tag.
    def tagged(kind: str, *, stop: bool, tag: bool) -> dict[str, Any]:
        row = _state(kind, predicted_stop=stop)
        row["failure_tags"] = ["necessary_backtrack"] if tag else ["wrong_branch"]
        return row

    states = [
        tagged("correct_stop", stop=True, tag=True),
        tagged("correct_stop", stop=True, tag=True),
        tagged("correct_stop", stop=False, tag=False),
        tagged("correct_stop", stop=False, tag=False),
        tagged("keep_pixel", stop=True, tag=True),
        tagged("keep_pixel", stop=False, tag=False),
    ]
    report = tool.summarise(states)
    assert report["stop_recall"] == pytest.approx(0.5)
    split = report["stop_recall_by_backtrack"]
    assert split["with_necessary_backtrack"] == pytest.approx(1.0)
    assert split["with_necessary_backtrack_states"] == 2
    assert split["without_necessary_backtrack"] == pytest.approx(0.0)
    assert split["without_necessary_backtrack_states"] == 2
    fa = report["stop_false_alarm_by_backtrack"]
    assert fa["with_necessary_backtrack"] == pytest.approx(1.0)
    assert fa["without_necessary_backtrack"] == pytest.approx(0.0)


def test_the_backtrack_split_is_none_when_a_side_is_empty() -> None:
    row = _state("correct_stop", predicted_stop=True)
    row["failure_tags"] = ["necessary_backtrack"]
    report = tool.summarise([row])
    split = report["stop_recall_by_backtrack"]
    assert split["with_necessary_backtrack"] == pytest.approx(1.0)
    assert split["without_necessary_backtrack"] is None
    assert split["without_necessary_backtrack_states"] == 0

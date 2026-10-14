"""EXP-25 G-A and round 1b's S0, stratified by the previous call and the distance."""

from __future__ import annotations

import json

import pytest

from scripts.tools import stratified_gates as sg
from src.data.s2_sft_dataset import decision_class
from tests.test_spinbreak_stream import _answer_calls
from tests.test_s2_sft_dataset import _write_record


@pytest.mark.parametrize("text", ["STOP", "", "no arrows", "←←", "→", "↑↑", "↓", "245 320", "12", "→←", "x STOP ←"])
def test_decision_class_matches_the_trainers(text):
    assert sg.decision_class(text) == decision_class(text)


def _row(path, index, d, on, off):
    return {"uid": f"{path}#{index}", "distance_to_goal": d, "on": {"class": on}, "off": {"class": off}}


def test_bins_reversal_and_criteria(tmp_path):
    records = tmp_path / "records"
    # previous calls: 0 none, 1 after pixel, 2 after left turn, 3 after right turn, 4 after pixel
    path = _write_record(records, "sceneA", 1, _answer_calls(["pixel", "left", "right", "pixel", "pixel"]))
    rows = [
        _row(path, 0, 7.0, "pixel", "pixel"),            # none, > 6 m
        _row(path, 1, 2.0, "stop", "pixel"),             # after pixel, 1-3 m: adapter stops
        _row(path, 2, 0.8, "stop", "turn_left"),         # after a left turn, <= 1 m: spin state, adapter stops
        _row(path, 3, 4.0, "turn_left", "turn_right"),   # after a right turn: reverses, direction differs from base
        _row(path, 4, 5.0, "pixel", "pixel"),            # after pixel, 3-6 m
    ]
    joined, stats = sg.attach_context(rows, None)
    assert stats["rows"] == 5 and [r["prev_class"] for r in joined] == [None, "pixel", "turn_left", "turn_right", "pixel"]
    result = sg.stratify(joined, {"gates": {"G0_adapter_off_replay": {"value": 0.995}}})
    bins = result["bins"]
    assert bins["pixel/1-3"] == {"n": 1, "on_stops": 1, "off_stops": 0, "on_rate": 1.0, "off_rate": 0.0, "delta": 1.0}
    assert bins["turn/le1"]["on_rate"] == 1.0 and bins["none/gt6"]["n"] == 1 and bins["pixel/3-6"]["n"] == 1
    ga = result["G_A"]["criteria"]
    assert ga["a_spin_stop_rate"]["value"] == 1.0 and ga["a_spin_stop_rate"]["pass"]
    assert ga["c_pixel_1_3_stop_rate"]["value"] == 1.0 and not ga["c_pixel_1_3_stop_rate"]["pass"]
    # two calls follow a turn (2 and 3); call 3 answers left after a right turn
    assert ga["d_post_turn_reversal"]["value"] == 0.5 and ga["d_post_turn_reversal"]["n"] == 2
    assert ga["d_turn_direction_differs_from_base"]["value"] == 1.0
    assert ga["e_class_agree_ge3m"]["value"] == round(2 / 3, 6)
    assert ga["e_G0"]["pass"] and not result["G_A"]["all_pass"]
    assert result["S0_round1b"]["closed_loop"] == "no"
    # the spin state used for training also needs an earlier call within 1 m: none here
    assert result["report_only"]["spin_strict_ge1_earlier_within_1m"]["n"] == 0
    assert sg.summary_line(result).startswith("FAIL G-A ")


def test_cli_reroots_records_and_reads_g0(tmp_path):
    records = tmp_path / "records"
    path = _write_record(records, "sceneA", 1, _answer_calls(["pixel", "left", "left"]))
    calls = tmp_path / "gates_x.calls.jsonl"
    elsewhere = f"/elsewhere/records/sceneA/{path.name}"
    rows = [_row(elsewhere, 1, 5.0, "pixel", "pixel"), _row(elsewhere, 2, 0.5, "stop", "turn_left")]
    calls.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "gates_x.json").write_text(json.dumps({"gates": {"G0_adapter_off_replay": {"value": 1.0}}}))
    # empty bins (no call beyond 6 m here) cannot pass: an unmeasured criterion is a FAIL
    assert sg.main(["--calls", str(calls), "--records", str(records)]) == 1
    result = json.loads((tmp_path / "gates_x.stratified.json").read_text())
    assert result["inputs"]["join"]["rows"] == 2 and result["G_A"]["criteria"]["e_G0"]["value"] == 1.0
    assert result["G_A"]["criteria"]["b_gt6_stop_increase"] == {"value": None, "threshold": 0.003, "direction": "<=",
                                                               "pass": False, "n": 0, "on": None, "base": None}
    assert result["S0_round1b"]["reversal_le_0.02_recheck_diagnosis"] is True

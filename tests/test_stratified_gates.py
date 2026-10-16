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


# --------------------------------------------------------------------------- #
# EXP-28 O3: (b)-(d) as G-A, (e) at 0.92 without G0, and (f)
# --------------------------------------------------------------------------- #
def _joined(prev, d, on, off):
    return {"uid": "x#0", "prev_class": prev, "earlier_within_1m": 0, "distance_to_goal": d,
            "on": {"class": on}, "off": {"class": off}}


def _o3_rows(*, le1_on, le1_off, pixel_on, pixel_off, n=8, far_agree=100, far=100):
    """``n`` calls at d <= 1 m (after a turn) and ``n`` after a pixel at 1-3 m; the first ``*_on`` /
    ``*_off`` of each answer STOP.  Then ``far`` calls at 4 m (no previous call), ``far_agree`` of them
    with the base's class."""
    rows = [_joined("turn_left", 0.5, "stop" if i < le1_on else "turn_left", "stop" if i < le1_off else "turn_left")
            for i in range(n)]
    rows += [_joined("pixel", 2.0, "stop" if i < pixel_on else "pixel", "stop" if i < pixel_off else "pixel")
             for i in range(n)]
    rows += [_joined(None, 4.0, "pixel" if i < far_agree else "turn_left", "pixel") for i in range(far)]
    return rows


def test_exp28_o3_keeps_b_to_d_lowers_e_and_adds_f():
    result = sg.stratify(_o3_rows(le1_on=3, le1_off=4, pixel_on=3, pixel_off=4, far_agree=93), None)
    o3 = result["O3_exp28"]["criteria"]
    assert set(o3) == {"b_pixel_3_6_stop_increase", "b_turn_3_6_stop_increase", "b_gt6_stop_increase",
                       "c_pixel_1_3_stop_rate", "d_post_turn_reversal", "d_turn_direction_differs_from_base",
                       "e_class_agree_ge3m", "f_le1_stop_keep", "f_pixel_1_3_stop_keep"}
    for key in ("b_pixel_3_6_stop_increase", "b_turn_3_6_stop_increase", "b_gt6_stop_increase",
                "c_pixel_1_3_stop_rate", "d_post_turn_reversal", "d_turn_direction_differs_from_base"):
        assert o3[key] == result["G_A"]["criteria"][key]
    # (e): 0.93 passes EXP-28's 0.92 and fails G-A's 0.95; no G0 in O3
    assert o3["e_class_agree_ge3m"]["value"] == 0.93 and o3["e_class_agree_ge3m"]["pass"]
    assert not result["G_A"]["criteria"]["e_class_agree_ge3m"]["pass"]
    # (f): 3 of 8 against the base's 4 of 8 is exactly 0.75 x base -- on the line, a pass
    f = o3["f_le1_stop_keep"]
    assert (f["value"], f["base"], f["threshold"], f["n"], f["pass"]) == (0.375, 0.5, 0.375, 8, True)
    assert o3["f_pixel_1_3_stop_keep"]["pass"] and o3["f_pixel_1_3_stop_keep"]["n"] == 8
    assert result["report_only"]["le1_any_previous"]["off_stops"] == 4


def test_exp28_o3_f_fails_when_the_adapter_stops_less():
    o3 = sg.stratify(_o3_rows(le1_on=2, le1_off=4, pixel_on=4, pixel_off=4), None)["O3_exp28"]
    assert not o3["criteria"]["f_le1_stop_keep"]["pass"] and o3["criteria"]["f_pixel_1_3_stop_keep"]["pass"]
    assert not o3["all_pass"]
    o3 = sg.stratify(_o3_rows(le1_on=8, le1_off=4, pixel_on=2, pixel_off=3), None)["O3_exp28"]["criteria"]
    assert o3["f_le1_stop_keep"]["pass"] and not o3["f_pixel_1_3_stop_keep"]["pass"]
    # the base never stops there: nothing to keep, a pass
    assert sg.stratify(_o3_rows(le1_on=0, le1_off=0, pixel_on=0, pixel_off=0), None)[
        "O3_exp28"]["criteria"]["f_le1_stop_keep"]["pass"]
    # no call within 1 m: unmeasured, a FAIL
    empty = sg.stratify(_o3_rows(le1_on=0, le1_off=0, pixel_on=0, pixel_off=0, n=0), None)["O3_exp28"]["criteria"]
    assert empty["f_le1_stop_keep"]["n"] == 0 and not empty["f_le1_stop_keep"]["pass"]


def test_cli_gate_exp28_o3_decides_the_exit_code(tmp_path, capsys):
    records = tmp_path / "records"
    # previous calls: 0 none, 1 after pixel, 2 after a left turn, 3 after pixel, 4 after pixel
    path = _write_record(records, "sceneA", 1, _answer_calls(["pixel", "left", "pixel", "pixel", "pixel"]))
    rows = [
        _row(path, 0, 7.0, "pixel", "pixel"),          # > 6 m
        _row(path, 1, 4.0, "pixel", "pixel"),          # after pixel, 3-6 m
        _row(path, 2, 5.0, "turn_left", "turn_left"),  # after a left turn, 3-6 m: no reversal, same direction
        _row(path, 3, 2.0, "pixel", "pixel"),          # after pixel, 1-3 m: nobody stops
        _row(path, 4, 0.5, "stop", "stop"),            # <= 1 m: both stop
    ]
    calls = tmp_path / "gates_x.calls.jsonl"
    calls.write_text("".join(json.dumps(r) + "\n" for r in rows))
    # G-A fails here (no spin state for (a), no G0 report); O3 passes
    assert sg.main(["--calls", str(calls)]) == 1
    assert capsys.readouterr().out.startswith("FAIL G-A ")
    assert sg.main(["--calls", str(calls), "--gate", "exp28-o3"]) == 0
    assert capsys.readouterr().out.startswith("PASS EXP-28 O3 ")
    result = json.loads((tmp_path / "gates_x.stratified.json").read_text())
    assert result["inputs"]["gate"] == "exp28-o3" and result["O3_exp28"]["all_pass"]
    # the adapter no longer stops within 1 m: (f) fails
    rows[-1] = _row(path, 4, 0.5, "pixel", "stop")
    calls.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert sg.main(["--calls", str(calls), "--gate", "exp28-o3"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("FAIL EXP-28 O3 ") and "f_le1_stop_keep=0.0(FAIL)" in out
    with pytest.raises(ValueError):
        sg.gate_block(result, "o4")

"""EXP-19 [F] build_records.py: verdict thresholds, joins and gates, plus the synthetic pipeline end to end.

numpy / pandas / PIL / scipy; no torch (the accumulator self-check is switched off here and runs in the
real build instead).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.exp18 import geometry as geo
from scripts.exp19 import build_records as br
from scripts.exp19 import synthetic_traces

F, L, R, S = 1, 2, 3, 0


# --------------------------------------------------------------------------- #
# Pre-registered thresholds (README EXP-19 判据), boundaries inclusive as written
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pck8,gain,low,expected", [
    (0.80, 0.20, 0.10, "support"),
    (0.85, 0.85 - 0.65, 0.10, "support"),  # 0.19999999999999996 in floats is the 20pt it is
    (0.7999, 0.30, 0.20, "partial"),
    (0.90, 0.1999, 0.15, "partial"),
    (0.90, 0.30, 0.0999, "partial"),
    (0.60, 0.05, 0.01, "partial"),
    (0.60, 0.0499, 0.01, "refute"),
    (0.4999, 0.30, 0.20, "refute"),
    (None, 0.3, 0.2, "missing"),
])
def test_h1_decision(pck8, gain, low, expected):
    assert br.h1_decision(pck8, gain, low) == expected


@pytest.mark.parametrize("n,agreement,gain,expected", [
    (49, 1.0, 1.0, "not_measured"),
    (50, 0.60, 0.20, "support"),
    (50, 0.5999, 0.5999, "partial"),
    (80, 0.90, 0.1999, "partial"),
    (80, 0.30, 0.05, "partial"),
    (80, 0.30, 0.0499, "refute"),
])
def test_h2_decision(n, agreement, gain, expected):
    assert br.h2_decision(n, agreement, gain) == expected


@pytest.mark.parametrize("rate,expected", [(0.10, "support"), (20 / 200, "support"), (0.0999, "report_only"),
                                           (0.02, "report_only"), (0.0199, "refute"), (0.0, "refute"),
                                           (None, "missing")])
def test_h3_decision(rate, expected):
    assert br.h3_decision(rate) == expected


def test_h3_decision_needs_every_ready_call():
    assert br.h3_decision(0.5, n_missing=0) == "support"
    assert br.h3_decision(0.5, n_missing=1) == "missing"


def _h3_rows(n=10):
    return [{"ready": True, "ep_key": f"ep{i % 2}", "call_index": i, "category": "T1", "cf_changed": i % 2 == 0,
             "endpoint_shift_m": 0.1 * i, "replay_actions_equal": True} for i in range(n)]


def test_h3_block_does_not_shrink_its_denominator():
    rows = _h3_rows()
    vd = br.h3_block(rows, 50, 0)["verdict"]
    assert vd["verdict"] == "support" and vd["change_rate"] == 0.5 and vd["n"] == 10 and vd["n_missing_counterfactual"] == 0
    rows[3]["cf_changed"] = None  # a diagnostic that failed or was skipped on one ready call
    out = br.h3_block(rows, 50, 0)
    vd = out["verdict"]
    assert vd["verdict"] == "missing" and vd["change_rate"] is None and "1 of 10 ready calls" in vd["reason"]
    assert vd["change_rate_on_subset"] == 5 / 9 and vd["verdict_on_subset"] == "support"
    assert vd["n"] == 9 and vd["n_ready_calls"] == 10 and vd["missing_counterfactual_calls"] == [["ep1", 3]]
    assert out["n_missing_counterfactual"] == 1 and out["n_with_counterfactual"] == 9


def test_final_verdicts_void_on_gates_and_h1_on_a_failed_self_check():
    blocks = {"H1": {"verdict": br._verdict("support", joint_pck8=0.9, floor_pck8=0.3)},
              "H2": {"verdict": br._verdict("not_measured")}, "H3": {"verdict": br._verdict("report_only")}}
    assert {h: v["verdict"] for h, v in br.final_verdicts(blocks, [], {"ok": True}).items()} == {
        "H1": "support", "H2": "not_measured", "H3": "report_only"}
    v = br.final_verdicts(blocks, [], {"ok": False})
    assert v["H1"]["verdict"] == "void" and v["H1"]["verdict_if_valid"] == "support" and "self-check" in v["H1"]["void_reasons"][0]
    assert v["H2"]["verdict"] == "not_measured" and v["H3"]["verdict"] == "report_only"
    v = br.final_verdicts(blocks, ["trace-neutrality gate failed"], None)
    assert all(x["verdict"] == "void" and x["void_reasons"] == ["trace-neutrality gate failed"] for x in v.values())
    assert all("criterion" in x for x in v.values())


def test_compare_outcome_includes_how_the_episode_ended():
    rerun = {"success": 0.0, "oracle_success": 1.0, "ended_by": "stop", "steps": 80}
    ref = {"success": 0.0, "os": 1.0, "ended_by": "stop", "steps": 78}
    cmp = br.compare_outcome(rerun, ref)
    assert cmp["same"] and cmp["ended_by_equal"] and cmp["steps_equal"] is False  # steps are descriptive only
    assert not br.compare_outcome(dict(rerun, ended_by="step_cap", steps=500), ref)["same"]  # F1 -> F2 is a new outcome
    assert not br.compare_outcome(dict(rerun, oracle_success=0.0), ref)["same"]
    old = {"success": 0.0, "os": 1.0}  # a reference without ended_by / steps: compared on what it has
    assert br.compare_outcome(rerun, old)["same"] and br.compare_outcome(rerun, old)["ended_by_equal"] is None
    assert br.compare_outcome(None, ref) is None and br.compare_outcome(rerun, None) is None


def test_criteria_text_is_the_ledger_text():
    assert "PCK@8 ≥ 0.80、比平凡基线高 ≥ 20pt、且该差的 95% CI 下界 ≥ 10pt" in br.CRITERIA["H1"]
    assert "参照非前视的时段 n < 50 → 没测出来" in br.CRITERIA["H2"]
    assert "改变率 ≥ 10% → 支持依赖" in br.CRITERIA["H3"] and "改变率 < 2% → 否定" in br.CRITERIA["H3"]


# --------------------------------------------------------------------------- #
# Joins
# --------------------------------------------------------------------------- #
def test_executed_actions_follow_the_action_records():
    calls = [{"current_capture_step": 0, "system2_call_index": 0}, {"current_capture_step": 3, "system2_call_index": 1}]
    actions = [
        {"step_before": 0, "action": L, "phase": "rpc_first", "system2_call_index": 0},
        {"step_before": 1, "action": L, "phase": "local_action", "system2_call_index": 0},
        {"step_before": 2, "action": F, "phase": "local_action"},  # no field: step range -> call 0
        {"step_before": 3, "action": F, "phase": "rpc_first", "system2_call_index": 1},
        {"step_before": 4, "action": S, "phase": "auto_stop", "system2_call_index": None},  # no call made it
    ]
    out, check = br.executed_by_call(calls, actions)
    assert out == {0: [L, L, F], 1: [F]}
    assert check == {"actions_without_call_index_field": 1, "actions_of_no_call": 1,
                     "call_index_vs_step_range_mismatch": 0}
    actions[3]["system2_call_index"] = 0  # contradicts the step range
    assert br.executed_by_call(calls, actions)[1]["call_index_vs_step_range_mismatch"] == 1


def test_compare_calls_matches_by_index_and_strips_text():
    ref = [{"call_index": i, "step": 4 * i, "kind": "trajectory", "vlm_output": f"{100 + i} 50", "actions": [1, 1, 1, 1]}
           for i in range(4)]
    rerun = [dict(c, vlm_output=c["vlm_output"] + "\n") for c in ref]
    assert br.compare_calls(rerun, ref)["all_identical"]
    del rerun[1]  # a missing trace is a difference, not a shift of every later call
    res = br.compare_calls(rerun, ref)
    assert res["first_divergent_call"] == 1 and res["identical_calls"] == 3
    rerun[-1] = dict(rerun[-1], actions=[1, 1, 2, 1])
    assert br.compare_calls(rerun, ref)["per_call_identical"] == [True, False, True, False]


@pytest.mark.parametrize("category,outcome,holds", [
    ("T1", {"success": 1.0, "oracle_success": 1.0, "ended_by": "stop"}, True),
    ("T3", {"success": 0.0, "oracle_success": 1.0, "ended_by": "stop"}, False),
    ("F1", {"success": 0.0, "oracle_success": 1.0, "ended_by": "stop"}, True),
    ("F1", {"success": 0.0, "oracle_success": 1.0, "ended_by": "step_cap"}, False),
    ("F1", {"success": 0.0, "oracle_success": 0.0, "ended_by": "stop"}, False),
    ("F2", {"success": 0.0, "oracle_success": 1.0, "ended_by": "step_cap"}, True),
    ("F2", {"success": 0.0, "oracle_success": 0.0, "ended_by": "stop"}, False),
])
def test_rerun_predicate(category, outcome, holds):
    assert br.rerun_predicate(category, outcome) is holds
    assert br.rerun_predicate(category, None) is None


def test_score_slots_uses_the_none_logit_zero_and_the_gt_view_peak():
    logits = np.full((8, 4), -1.0, np.float32)
    logits[0, geo.LEFT] = 2.0   # slot 0: left
    logits[1, geo.RIGHT] = -0.5  # slot 1: every logit < 0 -> none
    peaks = np.zeros((8, 4, 2), np.int64)
    peaks[0, geo.LEFT] = (30, 40)
    peaks[0, geo.BACK] = (32, 32)
    cls = np.array([1 + geo.LEFT, 1 + geo.BACK] + [0] * 6)
    row, col = np.array([32, 32] + [-1] * 6), np.array([36, 32] + [-1] * 6)
    out = br.score_slots(logits, peaks, cls, row, col)
    assert out["view5"][:2].tolist() == [1 + geo.LEFT, 0]
    assert out["sq"][0] == 2 ** 2 + 4 ** 2
    assert (out["row"][1], out["col"][1]) == (0, 0)  # the peak in the GT (back) view, not the predicted one


# --------------------------------------------------------------------------- #
# The whole pipeline on the synthetic artifact tree
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("convention", ["field_vu", "field_uv"])
def test_synthetic_pipeline_end_to_end(tmp_path, convention):
    root = tmp_path / "exp19"
    assert synthetic_traces.main(["--root", str(root), "--pixel-goal-convention", convention]) == 0
    code = br.main(["--exp-root", str(root), "--run", "synth", "--bootstrap-reps", "200", "--no-self-check",
                    "--topdown-root", str(tmp_path / "no_topdown")])
    assert code == 0
    m = json.loads((root / "metrics" / "metrics.json").read_text())
    assert m["validity"]["valid"] and m["gates"]["trace_neutrality"]["pass"] and m["gates"]["code_equivalence"]["pass"]
    assert m["warnings"] == []
    for e in m["episodes"]:  # the synthetic sensor sits exactly at the derived camera pose
        assert e["checks"]["sensor_pose"]["max_abs_position_m"] < 1e-9 and e["checks"]["renders_c2w_max_abs_err"] == 0.0
    assert m["pixel_goal_convention"]["convention"] == convention and not m["pixel_goal_convention"]["forced"]
    assert set(m["verdicts"]) == {"H1", "H2", "H3"} and all("criterion" in v for v in m["verdicts"].values())
    assert m["main_cases"]["T1"] == "SynthSceneA_0007" and m["main_cases"]["F1"] == "SynthSceneB_0042"
    by_key = {e["ep_key"]: e for e in m["episodes"]}
    assert by_key["SynthSceneA_0007"]["all_calls_identical"] and by_key["SynthSceneB_0042"]["first_divergent_call"] == 9
    rows = [json.loads(line) for line in (root / "metrics" / "calls.jsonl").read_text().splitlines()]
    assert len(rows) == sum(e["n_calls"] for e in m["episodes"])
    for e in m["episodes"]:
        bundle = json.loads((root / "records" / f"{e['ep_key']}_bundle.json").read_text())
        assert bundle["schema"] == br.SCHEMA_BUNDLE and [k["label"] for k in bundle["key_steps"]] == ["K1", "K2", "K3", "K4"]
        with np.load(root / "records" / f"{e['ep_key']}_bundle.npz") as z:
            assert z["k0_hist_pred"].shape == (8, 4, 64, 64) and z["k3_path_cam"].shape == (33, 3)
            for i, k in enumerate(bundle["key_steps"]):
                # synthetic System2 aims at System1's endpoint: whichever text order was written, the
                # resolved (column, row) must land on the endpoint's projection in the decision image
                h, w = z[f"k{i}_decision_rgb"].shape[:2]
                assert [w, h] == k["decision_image_wh"]
                u, v = k["pixel_goal_uv"]
                assert 0 <= u < w and 0 <= v < h
                if k["path_uv_index"] and k["path_uv_index"][-1] == 32:
                    eu, ev = k["path_uv"][-1]
                    if 0 <= eu <= w - 1 and 0 <= ev <= h - 1:
                        assert abs(u - eu) <= 1.0 and abs(v - ev) <= 1.0


def _synthetic_tree(tmp_path):
    root = tmp_path / "exp19"
    assert synthetic_traces.main(["--root", str(root)]) == 0
    return root


def _build(root, tmp_path):
    return br.main(["--exp-root", str(root), "--run", "synth", "--bootstrap-reps", "50", "--no-self-check",
                    "--topdown-root", str(tmp_path / "no_topdown")])


def _ready_trace_files(root):
    out = []
    for path in sorted((root / "runs" / "synth").glob("gpu*/trace/*/call_*.json")):
        c = json.loads(path.read_text())
        if (c.get("response") or {}).get("kind") == "trajectory" and c["response"].get("ppa_applied") is True:
            out.append(path)
    return out


def test_ready_call_without_counterfactual_makes_h3_missing_not_a_subset_rate(tmp_path):
    root = _synthetic_tree(tmp_path)
    path = _ready_trace_files(root)[2]
    c = json.loads(path.read_text())
    c["diagnostic"]["counterfactual_no_memory"] = None
    path.write_text(json.dumps(c))
    assert _build(root, tmp_path) == 0  # H3 is its own question: H1 / H2 and validity are untouched
    m = json.loads((root / "metrics" / "metrics.json").read_text())
    assert m["validity"]["valid"] and m["verdicts"]["H3"]["verdict"] == "missing"
    assert m["verdicts"]["H3"]["n_missing_counterfactual"] == 1 and m["verdicts"]["H3"]["change_rate"] is None
    assert m["verdicts"]["H1"]["verdict"] != "missing"
    assert any(w.startswith("H3 missing: 1 ready calls") for w in m["warnings"])


def test_ready_call_without_system1_path_stops_the_build(tmp_path):
    root = _synthetic_tree(tmp_path)
    path = _ready_trace_files(root)[0]  # K1 of its episode, so the bundle stage would need it too
    c = json.loads(path.read_text())
    c["selected_path_xy"] = None
    path.write_text(json.dumps(c))
    npz = path.with_suffix(".npz")
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files if k != "selected_path_xy"}
    np.savez_compressed(npz, **arrays)
    with pytest.raises(ValueError, match="ready call without selected_path_xy"):
        _build(root, tmp_path)


def test_selected_path_falls_back_to_the_json_copy(tmp_path):
    path = np.linspace(0, 1, 66).reshape(33, 2)
    np.savez(tmp_path / "with.npz", selected_path_xy=path[None])  # a leading batch dim is dropped
    np.savez(tmp_path / "without.npz", other=np.zeros(1))
    with np.load(tmp_path / "with.npz") as z:
        assert np.array_equal(br.selected_path({}, z), path)
    with np.load(tmp_path / "without.npz") as z:
        assert np.array_equal(br.selected_path({"selected_path_xy": path.tolist()}, z), path)
        assert br.selected_path({"selected_path_xy": None}, z) is None
        with pytest.raises(ValueError, match="shape"):
            br.selected_path({"selected_path_xy": path[:5].tolist()}, z)

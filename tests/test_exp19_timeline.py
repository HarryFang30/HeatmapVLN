"""EXP-19 figures v2 [T] build_timeline.py: bearing conventions, pure helpers, and the synthetic tree end to end.

numpy / pandas / PIL / scipy; no torch (build_records runs with --no-self-check here).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from scripts.exp18 import geometry as geo
from scripts.exp19 import build_records as br
from scripts.exp19 import build_timeline as tl
from scripts.exp19 import gt
from scripts.exp19 import synthetic_traces

F, L, R, S = 1, 2, 3, 0


# --------------------------------------------------------------------------- #
# Bearing axis and rings (left-positive, +180 at column 0)
# --------------------------------------------------------------------------- #
def test_bearing_axis_runs_from_left_edge_to_right_edge():
    b = tl.bearing_axis()
    assert b.shape == (360,) and b[0] == 179.5 and b[180] == -0.5 and b[359] == -179.5
    assert np.array_equal(b, np.float32(179.5) - np.arange(360, dtype=np.float32))
    assert tl.bearing_to_column(90.0) == 90 and tl.bearing_to_column(-90.0) == 270
    assert tl.bearing_to_column(179.9) == 0 and tl.bearing_to_column(-179.9) == 359


def _spot(view: int, row: float, col: float, sigma: float = 1.5) -> np.ndarray:
    maps = np.zeros((4, 64, 64), dtype=np.float32)
    yy, xx = np.mgrid[0:64, 0:64]
    maps[view] = np.exp(-((yy - row) ** 2 + (xx - col) ** 2) / (2 * sigma ** 2))
    return maps


@pytest.mark.parametrize("view,col,bearing", [
    (geo.FRONT, 32, 0.0), (geo.LEFT, 32, 90.0), (geo.RIGHT, 32, -90.0),
    (geo.FRONT, 0, 45.0),  # left edge of the front view is 45 deg to the left
    (geo.FRONT, 63, -44.1),  # right edge (index 63: coordinate 63 -> -atan(31 / 32))
])
def test_ring_1d_puts_a_spot_at_its_bearing(view, col, bearing):
    ring = tl.ring_1d(_spot(view, 32, col))
    peak = float(tl.bearing_axis()[int(ring.argmax())])
    assert abs(peak - bearing) <= 1.0, (peak, bearing)
    assert 0.8 <= ring.max() <= 1.0


def test_ring_1d_back_view_centre_sits_at_both_edges():
    ring = tl.ring_1d(_spot(geo.BACK, 32, 32))
    assert int(ring.argmax()) in (0, 359) and ring[0] > 0.8 and ring[359] > 0.8 and ring[180] == 0.0


def test_hist_ring_is_the_v1_composite_on_the_bearing_axis():
    pred = np.zeros((8, 4, 64, 64), dtype=np.float32)
    pred[0] = 0.01 * _spot(geo.LEFT, 32, 32)  # tiny gated values: the composite rescales per slot
    pred[1] = 0.02 * _spot(geo.RIGHT, 32, 32)
    pred[2] = 0.05 * _spot(geo.FRONT, 32, 32)  # padded slot: ignored
    none_p = np.array([0.2, 0.6, 0.0, 1, 1, 1, 1, 1], dtype=np.float32)
    mask = np.array([True, True] + [False] * 6)
    ring = tl.hist_ring(pred, none_p, mask)
    # each slot peaks at its confidence; the 1-degree bilinear resampling keeps >= 85 % of a 1.5 px spot's peak
    for bearing, conf in ((90.0, 0.8), (-90.0, 0.4)):
        assert 0.85 * conf <= ring[tl.bearing_to_column(bearing)] <= conf
    assert ring[tl.bearing_to_column(0.0)] < 1e-6


def test_pred_peaks_bearing_conf_and_padding():
    pred = np.zeros((8, 4, 64, 64), dtype=np.float32)
    pred[0] = _spot(geo.LEFT, 32, 32)
    pred[1] = _spot(geo.FRONT, 30, 0)
    none_p = np.array([0.1, 0.3, 0.9, 1, 1, 1, 1, 1], dtype=np.float32)
    mask = np.array([True, True, True] + [False] * 5)
    b, c = tl.pred_peaks(pred, none_p, mask)
    assert b[0] == pytest.approx(90.0) and b[1] == pytest.approx(45.0)
    assert np.isnan(b[2]) and c[2] == pytest.approx(0.1)  # valid slot with an all-zero map: no peak
    assert np.isnan(b[3:]).all() and np.isnan(c[3:]).all()
    assert c[0] == pytest.approx(0.9) and c[1] == pytest.approx(0.7)


# --------------------------------------------------------------------------- #
# GT bearings and System1 path: same convention (left-positive)
# --------------------------------------------------------------------------- #
def _cam(x: float, z: float, yaw_deg: float = 0.0) -> np.ndarray:
    """Front camera c2w of a body at (x, 0, z) facing yaw (left-positive about +y, 0 = world -z)."""
    h = math.radians(yaw_deg) / 2
    return gt.camera_c2w([x, 0.0, z], [math.cos(h), 0.0, math.sin(h), 0.0])


def test_gt_bearings_are_left_positive():
    cur = _cam(0.0, 0.0)
    past = [_cam(-1.0, 0.0), _cam(1.0, 0.0), _cam(0.0, 1.0), _cam(0.0, -1.0)]  # left, right, behind, ahead
    assert np.allclose(tl.gt_bearings(cur, past), [90.0, -90.0, 180.0, 0.0], atol=1e-9)  # behind = +180
    assert tl.gt_bearings(cur, []).shape == (0,)


def test_left_turn_moves_past_bearings_towards_negative():
    past = [_cam(0.0, 2.0), _cam(-1.0, 3.0), _cam(1.5, 2.5)]  # behind, behind-left, behind-right
    before = tl.gt_bearings(_cam(0.0, 0.0, 0.0), past)
    after = tl.gt_bearings(_cam(0.0, 0.0, 30.0), past)  # two LEFT actions in place
    assert np.allclose(tl.wrap(after - before), -30.0, atol=1e-9)


def test_path_polar_left_is_positive_and_the_robot_has_no_bearing():
    path = np.zeros((33, 2))
    path[1:, 0] = np.linspace(0.1, 2.0, 32)
    path[1:, 1] = np.linspace(0.0, 1.0, 32) ** 2  # curving left
    b, d = tl.path_polar(path)
    assert np.isnan(b[0]) and d[0] == 0.0 and (b[1:] >= 0).all() and b[-1] > 20.0
    assert d[-1] == pytest.approx(math.hypot(2.0, 1.0), rel=1e-6)
    # same numbers as the v1 strip (figures.bundle.path_directions on the camera-frame path)
    from scripts.exp19.figures import bundle as bd
    bb, _, keep = bd.path_directions(gt.path_camera_points(path))
    assert np.allclose(b[keep], bb, atol=1e-4)


def test_next_steps_call_state_and_step_actions():
    assert tl.next_steps([0, 4, 4, 9], 12).tolist() == [4, 4, 9, 12]
    assert tl.next_steps([], 5).shape == (0,)
    assert tl.call_state("trajectory", True) == "ready" and tl.call_state("trajectory", False) == "warmup"
    assert tl.call_state("native_actions", False) == "native_actions" and tl.call_state("stop", False) == "stop"
    acts = [{"step_before": 0, "action": F}, {"step_before": 1, "action": L}, {"step_before": 1, "action": R},
            {"step_before": 3, "action": S}]
    sa, dup = tl.step_actions(acts, 4)
    assert sa.tolist() == [F, L, -1, S] and dup == 1 and sa.dtype == np.int8
    assert tl.net_turn([L, L, F, R]) == 15.0 and tl.point_at_distance(np.array([np.nan, 5.0, 7.0]),
                                                                       np.array([0.0, 0.5, 1.2])) == 7.0


# --------------------------------------------------------------------------- #
# Synthetic artifact tree end to end
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("exp19_timeline")
    root = tmp / "exp19"
    assert synthetic_traces.main(["--root", str(root)]) == 0
    assert br.main(["--exp-root", str(root), "--run", "synth", "--bootstrap-reps", "50", "--no-self-check",
                    "--topdown-root", str(tmp / "no_topdown")]) == 0
    code = tl.main(["--exp-root", str(root), "--run", "synth", "--self-check"])
    return {"root": root, "code": code, "out": root / "records_v2"}


def test_synthetic_timeline_schema_and_self_check(synth):
    out = synth["out"]
    report = json.loads((out / "timeline_self_check.json").read_text())
    failed = {k: v for k, v in report["checks"].items() if not v["pass"]}
    assert synth["code"] == 0 and report["ok"], failed
    c2 = report["checks"]["2_key_rows_vs_v1_bundle"]
    assert c2["n_key_rows"] == 8 and c2["tensor_max_abs_diff"] == 0.0 and c2["ring_max_abs_diff"] == 0.0
    assert report["checks"]["3_turn_shifts_gt"]["left"]["n"] > 0 and report["checks"]["3_turn_shifts_gt"]["right"]["n"] > 0
    for ep_key in ("SynthSceneA_0007", "SynthSceneB_0042"):
        t = tl.load_timeline(out / f"{ep_key}_timeline.npz")
        a, meta = t["arrays"], t["meta"]
        rec = json.loads((synth["root"] / "records" / f"{ep_key}.json").read_text())
        ready = [c for c in rec["calls"] if c["ready"]]
        R = len(ready)
        assert meta["schema"] == tl.SCHEMA and meta["counts"]["n_ready"] == R and R > 4
        shapes = {"hist_ring": (R, 360), "hist_pred_peak_bearing": (R, 8), "hist_pred_conf": (R, 8),
                  "hist_gt_bearing": (R, 8), "hist_gt_visible": (R, 8), "fut_ring": (R, 4, 360),
                  "s1_path_bearing": (R, 33), "s1_path_dist": (R, 33), "bearing_deg": (360,),
                  "calls_step": (len(rec["calls"]),)}
        for name, shape in shapes.items():
            assert a[name].shape == shape, name
        assert a["call_index"].tolist() == [c["call_index"] for c in ready]
        assert a["step"].tolist() == [c["step"] for c in ready]
        # next_step = step of the next call of any kind; the last call runs to the episode's end
        steps = [c["step"] for c in rec["calls"]]
        assert a["calls_next_step"].tolist() == steps[1:] + [rec["outcome"]["steps"]]
        pos = {c["call_index"]: j for j, c in enumerate(rec["calls"])}
        assert [a["calls_next_step"][pos[c["call_index"]]] for c in ready] == a["next_step"].tolist()
        assert int(a["first_ready_step"]) == ready[0]["step"] and int(a["episode_steps"]) == rec["outcome"]["steps"]
        assert a["calls_ready"].sum() == R and set(a["calls_state"].tolist()) >= {"ready", "warmup", "stop"}
        assert a["key_labels"].tolist() == [k["label"] for k in rec["key_steps"]]
        assert a["key_step"].tolist() == [k["step"] for k in rec["key_steps"]]
        # padding: NaN outside the valid slots, GT NaN exactly where invisible
        assert np.isnan(a["hist_pred_conf"][~a["hist_mask"]]).all()
        assert (np.isfinite(a["hist_gt_bearing"]) == a["hist_gt_visible"]).all()
        assert not (a["hist_gt_visible"] & ~a["hist_mask"]).any()
        assert ((a["hist_ring"] >= 0) & (a["hist_ring"] <= 1)).all()
        # the step-action strip is the action records
        assert len(a["step_action"]) == rec["outcome"]["steps"]
        assert meta["inputs"]["steps.jsonl"]["sha256"] and meta["npz_sha256"]


def test_refuses_to_write_into_v1_outputs(synth):
    with pytest.raises(SystemExit):
        tl.main(["--exp-root", str(synth["root"]), "--run", "synth", "--out-dir", str(synth["root"] / "records")])
    with pytest.raises(SystemExit):
        tl.main(["--exp-root", str(synth["root"]), "--run", "synth", "--out-dir", str(synth["root"] / "metrics")])

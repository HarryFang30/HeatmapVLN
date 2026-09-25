"""EXP-19 figures v2 (scripts/exp19/figures/{fig_v2,panels_v2,timeline_panel}.py).

Bundles come from ``synthetic_bundle.make_bundle`` with every external source
off (generated route, fake top-down level, procedural images), and timelines
from ``timeline_panel.synthetic_timeline``, so no server data is needed.  The
render tests need matplotlib.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.exp18 import geometry as geo
from scripts.exp19.figures import bundle as bd
from scripts.exp19.figures import synthetic_bundle as sb
from scripts.exp19.figures import timeline_panel as tp


def _calls(b: bd.Bundle, pose_from: int = 16, every: int = 4) -> list:
    """A call list matching the synthetic bundle: a call every 4 steps (call index = step // 4), ready from step 20
    (as ``synthetic_bundle.pick_key_steps``), warm-up until ``pose_from``, the last call STOP."""
    n = int(b.outcome["steps"])
    keyed = {int(k.call_index): k for k in b.keys}
    calls = []
    for s in range(0, max(n - 1, 1), every):
        ci = s // every
        acts = list(keyed[ci].executed_actions) if ci in keyed else [bd.FORWARD, bd.LEFT, bd.FORWARD, bd.RIGHT]
        calls.append({"call_index": ci, "step": s, "kind": "trajectory", "ppa_applied": s >= 20,
                      "pose_ready": s >= pose_from, "executed_actions": acts})
    calls.append({"call_index": calls[-1]["call_index"] + 1, "step": n - 1, "kind": "stop", "ppa_applied": False,
                  "pose_ready": True, "executed_actions": [bd.STOP]})
    free = [c for c in calls if c["ppa_applied"] and c["call_index"] not in keyed]
    if free:  # one System2 arrow answer after the first ready call: a "no map" span
        free[0].update(kind="native_actions", ppa_applied=False)
    return calls


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp("exp19_fig_v2")
    path = sb.make_bundle(out, episodes_path=None, clip_dir=None, topdown_root=out / "no_maps_here", seed=3)
    b = bd.load_bundle(path)
    calls = _calls(b)
    meta, a = tp.synthetic_timeline(b, calls)
    tl_dir = out / "records_v2"
    tp.write_timeline(tl_dir / f"{b.ep_key}_timeline", meta, a)
    record = Path(path).parent / f"{b.ep_key}.json"
    record.write_text(json.dumps({"calls": calls}), encoding="utf-8")
    return {"out": out, "bundle": path, "b": b, "calls": calls, "meta": meta, "a": a, "tl_dir": tl_dir,
            "record": record}


# --------------------------------------------------------------------------- #
# Timeline data
# --------------------------------------------------------------------------- #
def test_synthetic_timeline_is_valid_and_marked(synth):
    meta, a = synth["meta"], synth["a"]
    assert tp.validate_timeline(meta, a) == [] and meta["synthetic"] is True
    tl = tp.load_timeline(synth["tl_dir"] / f"{synth['b'].ep_key}_timeline.json")
    assert tl.synthetic and tl.R == int(np.sum(a["calls_ready"]))
    assert tp.check_against_bundle(tl, synth["b"]) == []
    assert set(tl.key_rows()) == {k.label for k in synth["b"].keys}


def test_timeline_validation_lists_problems(synth):
    meta, a = synth["meta"], dict(synth["a"])
    bad = dict(a)
    bad["bearing_deg"] = a["bearing_deg"][::-1].copy()  # right edge first: wrong convention
    assert any("bearing_deg" in e for e in tp.validate_timeline(meta, bad))
    bad = dict(a)
    bad["hist_ring"] = a["hist_ring"][:, :359]
    assert any("hist_ring" in e for e in tp.validate_timeline(meta, bad))
    bad = dict(a)
    bad["key_step"] = a["key_step"] + 1
    assert any("key K" in e for e in tp.validate_timeline(meta, bad))
    bad = dict(a)
    del bad["fut_ring"]
    assert "missing array 'fut_ring'" in tp.validate_timeline(meta, bad)


def test_bin_bearings_follow_the_ring_convention():
    """Bin i is centred on +179.5 - i: left edge +180 (behind via the left), centre ahead, right edge -180."""
    b = tp.bin_bearings()
    assert b[0] == pytest.approx(179.5) and b[180] == pytest.approx(-0.5) and b[-1] == pytest.approx(-179.5)
    # a Gaussian on the LEFT view's centre (bearing +90) peaks at bin 89/90 of the ring
    maps = np.zeros((4, 64, 64))
    maps[geo.LEFT] = np.exp(-((np.arange(64)[None, :] - 32) ** 2 + (np.arange(64)[:, None] - 32) ** 2) / 8.0)
    ring = tp.ring_of(maps)
    assert abs(b[int(np.argmax(ring))] - 90.0) <= 1.0


def test_wrapped_rows_continue_the_ring_past_plus_minus_180():
    """The timeline's bearing axis runs WRAP_PAD deg past +-180: the rows above +180 repeat the bins just right of
    -180, the rows below -180 the bins just left of +180 (same bearing, one turn away)."""
    pad = tp.WRAP_PAD
    b = tp.bin_bearings()
    img = b[:, None, None] * np.ones((1, 2, 4))
    out = tp.wrap_rows(img, pad)
    assert out.shape[0] == 360 + 2 * pad
    shown = np.linspace(180.0 + pad - 0.5, -180.0 - pad + 0.5, 360 + 2 * pad)  # row centres, top to bottom
    assert np.allclose(geo.wrap_deg(shown), out[:, 0, 0])
    assert tp.bearing_ylim() == (-180.0 - pad, 180.0 + pad)


def test_synthetic_rings_turn_with_the_executed_turns(synth):
    """Stand-in rings copied from a key moment are shifted by the net turn in between: a left turn moves a past
    direction to the right (towards negative bearings)."""
    ring = np.zeros(360)
    ring[90] = 1.0  # bearing +89.5
    turned = tp._roll(ring, -15.0)  # the robot turned left by 15 deg since
    assert tp.bin_bearings()[int(np.argmax(turned))] == pytest.approx(89.5 - 15.0)


def test_spans_warmup_ends_at_the_first_pose_ready_call(synth):
    tl = tp.load_timeline(synth["tl_dir"] / f"{synth['b'].ep_key}_timeline.json", record_path=synth["record"])
    spans = tl.spans()
    assert spans[0] == ("warmup", 0.0, 16.0)  # pose ready from step 16, first ready call at step 20
    kinds = [k for k, *_ in spans]
    assert kinds.count("warmup") == 1 and "nomap" in kinds
    nomap = [(s0, s1) for k, s0, s1 in spans if k == "nomap"]
    assert (16.0, 20.0) in nomap  # the pose-ready call before the first ready call has no map either
    assert nomap[-1][1] == tl.steps  # the STOP call at the end
    # without per-call pose readiness the warm-up runs to the first ready call
    a = {k: v for k, v in synth["a"].items() if k != "calls_pose_ready"}
    bare = tp.Timeline(synth["meta"], a)
    assert bare.spans()[0] == ("warmup", 0.0, 20.0)


def test_pose_ready_from_record_is_aligned_by_call_index(synth, tmp_path):
    calls = list(reversed(synth["calls"]))  # order in the record must not matter
    rec = tmp_path / "rec.json"
    rec.write_text(json.dumps({"calls": calls}), encoding="utf-8")
    by_call = tp.pose_ready_from_record(rec)
    assert by_call[4] is True and by_call[3] is False
    assert tp.pose_ready_from_record(tmp_path / "missing.json") is None


def test_path_endpoint_bearing_skips_points_at_the_robot():
    a = {"s1_path_bearing": np.array([[np.nan, 10.0, 20.0, np.nan]]), "s1_path_dist": np.array([[0.0, 1.0, 2.0,
                                                                                                    0.01]])}
    tl = tp.Timeline({}, a)
    assert tl.path_end_bearing()[0] == pytest.approx(20.0)


# --------------------------------------------------------------------------- #
# Marks and colours
# --------------------------------------------------------------------------- #
def test_marks_are_small_and_the_heat_is_a_faint_field():
    from scripts.exp19.figures import panels_v2 as p2

    assert 2.2 <= p2.PRED_MS + p2.PRED_RING <= 2.8 and p2.PRED_RING == pytest.approx(0.4)
    assert 3.0 <= p2.GT_MS <= 3.5 and p2.GT_MEW == pytest.approx(0.7)
    assert 1.2 <= p2.PATH_MS <= 1.5 and 5.0 <= p2.GOAL_MS <= 6.0
    assert 2 <= p2.PATH_EVERY <= 4 and p2.HAIR <= 0.4
    rgba = p2.heat_rgba(np.array([0.0, 0.5, 1.0, 3.0]), p2.HIST_COLOR, p2.HEAT_ALPHA)
    assert rgba[..., 3].max() == pytest.approx(p2.HEAT_ALPHA) and rgba[0, 3] == 0.0
    assert p2.HEAT_ALPHA <= 0.6 and p2.TL_HEAT_ALPHA <= 0.65
    # later time bins are painted on top: where two bins overlap fully, the colour is the later bin's
    bins = np.zeros((4, 1))
    bins[1] = bins[3] = 1.0
    out = p2.bins_rgba(bins, alpha_max=1.0)
    from matplotlib.colors import to_rgb

    assert np.allclose(out[0, :3], to_rgb(p2.FUT_BIN_COLORS[3]))
    assert p2.MIN_FS >= 6.0 and p2.FS["title"] == pytest.approx(7.5)


def test_history_strip_marks_use_the_visibility_threshold(synth):
    from scripts.exp19.figures import panels_v2 as p2

    ks = synth["b"].keys[0]
    gt, pred = p2.history_strip_marks(ks)
    conf = 1.0 - np.asarray(ks.hist_none)
    assert {k for k, *_ in pred} == set(np.nonzero(ks.hist_mask & (conf >= 0.5))[0].tolist())
    assert {k for k, *_ in gt} == {k for k, *_ in bd.gt_history_peaks(ks.hist_gt_peak, ks.hist_mask)}


def test_subsample_path_keeps_the_last_waypoint():
    from scripts.exp19.figures import panels_v2 as p2

    idx = p2.subsample_path(33, 3)
    assert idx[0] == 0 and idx[-1] == 32 and np.all(np.diff(idx) <= 3)


def test_start_label_stays_on_the_map():
    """A start at the map's left edge gets its label on the map, not across the edge."""
    from scripts.exp19.figures import panels_v2 as p2

    limits = (0.0, 10.0, 0.0, 10.0)
    route = p2._densify(np.array([[0.3, 5.0], [9.0, 5.0]]), 0.1)
    (ox, oy), ha, va = p2.place_label(np.array([0.3, 5.0]), np.array([-1.0, 0.0]), 20.0, 6.0, limits, 0.05, route)
    left = 0.3 + ox * 0.05 - (20.0 * 0.05 if ha == "right" else (0.0 if ha == "left" else 0.5))
    assert left >= 0.0 and va in ("top", "bottom")  # above or below the route, not on it


def test_muted_ring_fills_pixels_outside_every_view_with_the_surface():
    """Above / below the seams no view sees anything: those ring pixels are the page surface, not black."""
    pytest.importorskip("matplotlib")
    from matplotlib.colors import to_rgb

    from scripts.exp18.figures import style
    from scripts.exp19.figures import panels_v2 as p2

    views = np.zeros((4, 32, 32, 3), np.uint8)  # all-black views: every covered pixel stays dark
    ring = p2.muted_ring(views, 144, p2.HIST_ELEV)
    _, valid = geo.stitch_ring_rgb(views, width=144, height=ring.shape[0], elev_top=p2.HIST_ELEV,
                                   elev_bottom=-p2.HIST_ELEV)
    assert (~valid).any() and valid.any()
    assert np.allclose(ring[~valid], to_rgb(style.SURFACE), atol=1e-6)
    assert ring[valid].max() < 0.7  # covered: black, muted towards white at most


def _dense_timeline(n_calls: int, every: int, keys=(3, 50)) -> tp.Timeline:
    steps = np.arange(n_calls) * every
    R = n_calls
    a = {"step": steps, "next_step": steps + every, "episode_steps": np.asarray(n_calls * every),
         "call_index": np.arange(R), "key_labels": np.asarray([f"K{i + 1}" for i in range(len(keys))]),
         "key_call_index": np.asarray(keys), "key_step": steps[list(keys)]}
    return tp.Timeline({}, a)


def test_marker_stride_thins_dense_timelines_but_keeps_every_key_moment():
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_behavior as fb

    fb.setup("en")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.0, 2.0))
    ax = fig.add_axes([0.3, 0.1, 0.67, 0.8])  # 4.69 in wide, like the page's timeline
    sparse, dense = _dense_timeline(16, 4, keys=(3, 9)), _dense_timeline(125, 4, keys=(3, 50))
    assert tp.marker_stride(sparse, ax) == 1 and tp.marked_rows(sparse, 1) == list(range(16))
    k = tp.marker_stride(dense, ax)
    assert k >= 2 and tp.column_pt(dense, ax) * k >= tp.MARK_SPACING_PT
    rows = tp.marked_rows(dense, k)
    assert {3, 50} <= set(rows) and len(rows) < dense.R
    plt.close(fig)


def test_key_moments_are_hairlines_not_outlined_columns(synth):
    """K1-K4 on the timeline: an ink hairline at each key call's step (no outline box around the column)."""
    pytest.importorskip("matplotlib")
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle

    from scripts.exp18.figures import style
    from scripts.exp19.figures import fig_behavior as fb

    fb.setup("en")
    import matplotlib.pyplot as plt

    tl = tp.load_timeline(synth["tl_dir"] / f"{synth['b'].ep_key}_timeline.json", record_path=synth["record"])
    fig = plt.figure(figsize=(7.0, 2.0))
    ax = fig.add_axes([0.3, 0.1, 0.67, 0.8])
    tp.draw_history_panel(ax, tl, ("b", "l", "a", "r", "b"))
    ink = to_rgba(style.INK)
    outlines = [p for p in ax.patches if isinstance(p, Rectangle) and not p.get_fill()
                and np.allclose(p.get_edgecolor(), ink)]
    assert outlines == []
    vlines = {round(float(ln.get_xdata()[0]), 3) for ln in ax.lines
              if len(ln.get_xdata()) == 2 and ln.get_xdata()[0] == ln.get_xdata()[1]
              and np.allclose(to_rgba(ln.get_color()), ink)}
    assert vlines == {round(float(k.step), 3) for k in synth["b"].keys}
    plt.close(fig)


def test_stride_sentence_names_the_thinned_cases():
    from scripts.exp19.figures import fig_v2 as f2

    assert f2.stride_sentence("en", 1) == ""  # a page's stacking is in its slots clause; only the thinning here
    # the overview (past frames 1, 4, 8 elsewhere) names its stacked rows even when every call is marked
    assert "stacked" in f2.stride_sentence("en", 1, ["F2"]) and "one call in" not in f2.stride_sentence("en", 1, ["F2"])
    assert "叠在列中央" in f2.stride_sentence("zh", 1, ["F2"]) and "每" not in f2.stride_sentence("zh", 1, ["F2"])
    assert "one call in 3" in f2.stride_sentence("en", 3) and "every key moment" in f2.stride_sentence("en", 3)
    assert f2.stride_sentence("en", 2, ["F2"]).startswith("On the long rerun of F2,")
    assert f2.stride_sentence("zh", 2, ["F2"]).startswith("F2 较长") and "每 2 次调用" in f2.stride_sentence("zh", 2,
                                                                                                          ["F2"])


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #
def test_outcome_lines(synth):
    from scripts.exp19.figures import fig_v2 as f2

    b = synth["b"]
    L = f2.LABELS["zh"]
    b.meta["outcome"] = {"success": True, "oracle_success": True, "ne_m": 0.3196, "steps": 64, "ended_by": "stop"}
    assert f2.outcome_line(b, L) == "复跑：成功，第 64 步停下，距目标 0.3 m"
    b.meta["outcome"] = {"success": False, "oracle_success": True, "ne_m": 6.1, "steps": 500, "ended_by": "step_cap"}
    assert f2.outcome_line(b, f2.LABELS["en"]).startswith("Rerun: failure, hit the 500-step limit 6.1 m")


def test_system2_line_names_the_pixel_goal_when_it_fits():
    pytest.importorskip("matplotlib")
    from types import SimpleNamespace

    from scripts.exp19.figures import fig_behavior as fb
    from scripts.exp19.figures import fig_v2 as f2

    fb.setup("zh")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(3, 1))
    L = f2.LABELS["zh"]
    ks = SimpleNamespace(system2_texts=["↓", "198 230"])
    # "↓" is System 2's "look down first", not the executed action "back": glossed whenever it fits
    assert f2.s2_line(fig, ks, L, 200.0, 6.0) == "慢系统：“↓”（先俯视） → “198 230”（像素目标）"
    one = f2.s2_line(fig, ks, L, 125.0, 6.0)
    assert one == "慢系统：“↓”（先俯视） → “198 230”"  # the look-down gloss outranks the pixel-goal gloss
    assert f2.s2_line(fig, ks, L, 40.0, 6.0) == "慢系统：“↓” → “198 230”"  # no room: output only
    two = f2.s2_lines(fig, ks, L, 82.0, 6.0, max_lines=2)  # the overview's narrow column: two lines
    assert len(two) == 2 and "（先俯视）" in two[0] and two[1].startswith("→ ")
    assert f2.s2_line(fig, SimpleNamespace(system2_texts=["←"]), L, 200.0, 6.0) == "慢系统：“←”"
    plt.close(fig)


def test_labels_use_the_agreed_terms():
    from scripts.exp19.figures import fig_v2 as f2

    import re

    zh = json.dumps(f2.LABELS["zh"], ensure_ascii=False) + "".join(f2.CAPTION_PAGE["zh"])
    for term in ("预测历史 affordance map", "真实来路方向", "预测未来 affordance map", "高层决策（像素目标）",
                 "快系统路径", "执行的动作", "预热期（尚无 affordance map）",
                 "慢系统直接给出转向或停止（无 affordance map）", "桥接", "历史认知头", "概括向量"):
        assert term in zh, term
    en_all = json.dumps(f2.LABELS["en"]) + f2.CAPTION_PAGE["en"] + f2.CAPTION_OVERVIEW["en"] + "".join(
        f2.CAPTION_WARM[k]["en"] for k in f2.CAPTION_WARM)
    for banned in ("里程计", "位姿", "odometry", "heading", "理解", "记住"):
        assert banned not in zh and banned not in en_all, banned
    assert not re.search(r"\bposes?\b", en_all) and not hasattr(f2, "CAPTION_WARMUP")
    # no "match" key: how close a dot sits to its circle depends on each panel's scale
    assert "hit" not in f2.LABELS["en"]["legend"] and "hit" not in f2.LEGEND_KEYS
    assert "or STOP" in f2.LABELS["en"]["legend"]["nomap"]
    # the header carries no bookkeeping (candidate rank / main case)
    assert "rank" not in f2.LABELS["en"] and "main_case" not in f2.LABELS["zh"]
    # the one-line data-flow note never draws the future map into the actions; the caption says what the bridge does
    assert "不回流到动作" in "".join(f2.LABELS["zh"]["flow_note"])
    en_flow = "".join(f2.LABELS["en"]["flow_note"])
    assert "does not feed the actions" in en_flow and "both decoded from" in en_flow
    assert "bridge(Z, M)" in f2.CAPTION_PAGE["en"] and "correction" in f2.CAPTION_PAGE["en"]
    for lang in ("en", "zh"):  # the only mathtext is Z~
        assert [r for r in f2.LABELS[lang]["flow_note"] if "$" in r] in ([f2.ZT], [f2.ZT, f2.ZT])


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_rich_wrap_keeps_mathtext_whole():
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_behavior as fb
    from scripts.exp19.figures import fig_v2 as f2

    fb.setup("en")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(3, 1))
    lines = f2.rich_wrap(fig, ["decoded from ", "$\\tilde{Z}$", " and some more words to wrap around"], 6.0, 60.0)
    plt.close(fig)
    assert len(lines) > 1 and ["$\\tilde{Z}$"] in [[r for r in ln if r.startswith("$")] for ln in lines]
    assert all(r == "$\\tilde{Z}$" or "$" not in r for ln in lines for r in ln)


def test_smoke_render_page_overview_and_manifest(synth, tmp_path):
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_v2 as f2

    records = Path(synth["bundle"]).parent
    out = tmp_path / "figures_v2"
    assert f2.main(["--records", str(records), "--timelines", str(synth["tl_dir"]), "--no-verdicts",
                    "--out-dir", str(out), "--lang", "en"]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == f2.MANIFEST_SCHEMA and manifest["claims"] == {"en": []}
    (page,) = manifest["pages"]
    assert page["synthetic_timeline"] is True and page["sha256"]["timeline_npz"]
    w, h = page["size_in"]["en"]
    assert w == pytest.approx(7.0) and h <= f2.PAGE_MAX_H_IN
    assert page["min_font_pt"]["en"] >= 6.0
    assert not any(k in x for x in page["warnings"] for k in ("outside", "budget", "overlapping", "leaders")), \
        page["warnings"]
    assert page["category_rank"] == synth["b"].category_rank and "caption_chars" in page
    assert page["checks"]["n_gt"] > 0 and page["checks"]["turns_drawn"] >= 0
    for f in page["files"]:
        assert Path(f).stat().st_size > 0
    caption = Path(page["files"][2]).read_text(encoding="utf-8")
    assert caption.startswith("SYNTHETIC TIMELINE") and "does not feed the actions" in caption
    assert f2.smooth_sentence("en") in caption  # the display-only blur is disclosed
    assert manifest["display"]["caption_sentence"]["en"] == f2.smooth_sentence("en")
    (ov,) = manifest["overview"]  # the synthetic case is T1: the successes figure
    ov_caption = Path(next(f for f in ov["files"] if f.endswith(".txt"))).read_text(encoding="utf-8")
    assert f2.smooth_sentence("en") in ov_caption and ov["legend_stacked_rows"] == []
    assert ov["group"] == "T" and ov["size_in"]["en"][1] <= f2.OVERVIEW_MAX_H_IN
    assert ov["min_font_pt"]["en"] >= 6.0 and not any("overlapping" in x or "outside" in x or "leaders" in x
                                                       for x in ov["warnings"])
    from PIL import Image

    with Image.open(page["files"][1]) as im:
        assert im.size[0] == 2800  # 7.0 in at 400 dpi


def test_v1_figures_dir_is_refused(synth, tmp_path):
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_v2 as f2

    assert f2.main(["--records", str(Path(synth["bundle"]).parent), "--timelines", str(synth["tl_dir"]),
                    "--no-verdicts", "--out-dir", str(tmp_path / "figures"), "--lang", "en"]) == 2
    assert not (tmp_path / "figures").exists()


# --------------------------------------------------------------------------- #
# Round 1: marks spread by slot, warm-up compression, badge rows, strips, route map, captions, audit
# --------------------------------------------------------------------------- #
def _tl_one_call(nan_slot: int = 5) -> tp.Timeline:
    gt_b = np.linspace(170.0, 100.0, 8)
    gt_b[nan_slot] = np.nan
    a = {"step": np.array([20]), "next_step": np.array([24]), "episode_steps": np.asarray(40),
         "call_index": np.array([3]), "key_labels": np.array(["K1"]), "key_call_index": np.array([3]),
         "key_step": np.array([20]), "hist_gt_bearing": gt_b[None], "hist_gt_visible": np.isfinite(gt_b)[None],
         "hist_pred_peak_bearing": (gt_b + 2.0)[None], "hist_pred_conf": np.full((1, 8), 0.9)}
    return tp.Timeline({}, a)


def test_marks_spread_across_the_column_by_slot():
    """Slot k's dot and circle share x = step + (k + 0.5) / 8 of the column, slot 1 (oldest) at the left; narrow
    columns stack them at the column's centre."""
    tl = _tl_one_call()
    xs = tp.slot_xs(tl, 0, True)
    assert np.all(np.diff(xs) > 0) and 20.0 < xs[0] < xs[-1] < 24.0
    assert xs[0] == pytest.approx(20.25) and xs[-1] == pytest.approx(23.75)
    assert np.allclose(tp.slot_xs(tl, 0, False), 22.0)
    gt, pred = tp.history_marks(tl, [0], True)
    gx = sorted({g[0] for g in gt})
    assert len(gx) == 7 and xs[5] not in gx  # the slot without a visible true direction has no circle
    for x, _ in pred:  # every predicted dot sits in the x position of its own slot
        assert np.any(np.isclose(x, xs))
    assert {round(x, 6) for x, _ in pred} >= {round(x, 6) for x in gx}


def test_warmup_is_compressed_into_a_fixed_block():
    a = {"step": np.array([21, 25, 29]), "next_step": np.array([25, 29, 33]), "episode_steps": np.asarray(64),
         "first_ready_step": np.asarray(21), "calls_step": np.array([0, 21, 25, 29]),
         "call_index": np.array([1, 2, 3]), "key_labels": np.array([]), "key_call_index": np.array([], int),
         "key_step": np.array([], int)}
    tl = tp.Timeline({}, a)
    tp.compress_warmup(tl, 338.0, 34.0)
    assert tl.compressed and 0.0 < tl.x0 < 21.0
    assert tl.x_of(21) == pytest.approx(21.0) and tl.x_of(40) == pytest.approx(40.0) and tl.x_of(0) == pytest.approx(
        tl.x0)
    per = 338.0 / tl.span_steps()
    assert (21.0 - tl.x0) * per == pytest.approx(34.0)
    assert np.all(np.diff(tl.x_of(np.arange(0, 64, 0.5))) > 0)  # monotonic: order in time is kept
    tp.compress_warmup(tl, 60.0, 34.0)  # too narrow: linear
    assert not tl.compressed and tl.x_of(5) == 5.0
    b = dict(a)
    b["step"], b["calls_step"] = np.array([10, 25, 29]), np.array([0, 10, 25, 29])
    tl2 = tp.Timeline({}, b)
    tl2.pose_ready = np.array([False, False, True, True])  # a ready call inside the warm-up: keep it to scale
    tp.compress_warmup(tl2, 338.0, 34.0)
    assert not tl2.compressed


def test_close_key_badges_get_a_second_row_and_never_touch():
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_behavior as fb

    fb.setup("en")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.0, 2.0))
    steps = np.arange(0, 120, 4)
    a = {"step": steps, "next_step": steps + 4, "episode_steps": np.asarray(120), "call_index": np.arange(30),
         "key_labels": np.array(["K1", "K2", "K3", "K4"]), "key_call_index": np.array([2, 20, 19, 28]),
         "key_step": steps[[2, 20, 19, 28]]}
    tl = tp.Timeline({}, a)
    items = tp.badge_layout(fig, tl, 150.0)
    lev = {it["label"]: it["level"] for it in items}
    assert lev["K3"] == 0 and lev["K2"] == 1  # K2 comes after K3 in time: it is the one raised
    for level in (0, 1):
        row = sorted((it for it in items if it["level"] == level), key=lambda d: d["x"])
        for u, v in zip(row, row[1:]):
            assert v["x"] - u["x"] >= (u["w"] + v["w"]) / 2
    hi = next(it for it in items if it["level"] == 1)
    for lo in (it for it in items if it["level"] == 0):  # the raised badge's leader passes beside every lower badge
        assert abs(lo["x"] - hi["target"]) >= lo["w"] / 2
    assert tp.badge_levels(fig, tl, 150.0) == 2 and tp.badge_levels(fig, tl, 2000.0) == 1
    plt.close(fig)


def test_strips_run_past_plus_minus_180():
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_behavior as fb
    from scripts.exp19.figures import panels_v2 as p2

    img = np.arange(360)[None, :, None] * np.ones((4, 1, 3))
    wide, pad = p2.wrap_columns(img, 8.0)
    assert pad == pytest.approx(8.0) and wide.shape[1] == 376
    assert np.allclose(wide[0, :8, 0], img[0, -8:, 0]) and np.allclose(wide[0, -8:, 0], img[0, :8, 0])
    fb.setup("en")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(2.0, 0.5))
    ax = fig.add_axes([0, 0, 1, 1])
    p2.setup_strip(ax, 45.0)
    assert ax.get_xlim() == (180.0 + p2.STRIP_PAD, -180.0 - p2.STRIP_PAD)
    # System 1 waypoints a few degrees apart are thinned to separate dots; the endpoint is always kept
    b = np.linspace(-6.0, 6.0, 33)
    idx = p2.thin_marks(ax, b, np.zeros(33), p2.PATH_SEP_PT)
    assert idx[-1] == 32 and 2 <= len(idx) < 33
    kx = abs(p2.cd.pts_to_data(ax, 1.0)[0])
    assert np.all(np.diff(b[idx]) / kx >= p2.PATH_SEP_PT - 1e-9)
    plt.close(fig)


def test_clustered_key_badges_fan_out_without_crossing_leaders():
    from scripts.exp19.figures import panels_v2 as p2

    per_pt = 0.02  # map units per point
    limits = (0.0, 10.0, 0.0, 10.0)
    for centre in (np.array([5.0, 5.0]), np.array([0.3, 5.0])):  # mid-map and at the map's left edge
        keys = centre + np.array([[0.0, 0.0], [0.03, 0.01], [-0.02, 0.03], [0.01, -0.03]])
        spots = p2.badge_spots(keys, keys, [], limits, per_pt)
        hw, hh = p2.BADGE_HALF
        for i in range(4):
            q = spots[i]
            assert limits[0] < q[0] - hw * per_pt and q[0] + hw * per_pt < limits[1]
            for j in range(i + 1, 4):
                r = spots[j]
                apart = abs(q[0] - r[0]) >= 2 * hw * per_pt or abs(q[1] - r[1]) >= 2 * hh * per_pt
                assert apart, (i, j)
                assert not p2._segs_cross(keys[i], q, keys[j], r)


def test_leader_audit_finds_a_leader_through_a_label():
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_behavior as fb
    from scripts.exp19.figures import fig_v2 as f2

    fb.setup("en")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(2.0, 2.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(5.0, 5.0, "start", ha="center", va="center", fontsize=6)
    ax.text(8.0, 8.0, "K2", ha="center", va="center", fontsize=6)
    ax.plot([1.0, 8.0], [5.0, 5.05], gid="leader:K2")  # runs through "start"
    assert f2.leader_crossings(fig) and "start" in f2.leader_crossings(fig)[0]
    ax.lines[0].set_data([8.0, 8.0], [2.0, 8.0])
    assert f2.leader_crossings(fig) == []
    plt.close(fig)


def test_caption_names_key_moments_out_of_time_order_and_merges_fallbacks():
    from types import SimpleNamespace

    from scripts.exp19.figures import fig_v2 as f2

    ks = [SimpleNamespace(label="K1", step=22), SimpleNamespace(label="K2", step=35),
          SimpleNamespace(label="K3", step=31), SimpleNamespace(label="K4", step=87)]
    assert f2.order_pairs(ks, "en") == "K3 (step 31) comes before K2 (step 35) in time"
    assert f2.order_pairs(ks[:2], "en") == ""
    assert "K3（第 31 步）在时间上早于 K2（第 35 步）" in f2.order_pairs(ks, "zh")
    k2 = SimpleNamespace(label="K2", branch="K2_fallback", step=35, index=1)
    k3 = SimpleNamespace(label="K3", branch="K3_two_thirds", step=50, index=2)
    k3f = SimpleNamespace(label="K3", branch="K3_f1_closest", step=31, index=2)
    b1, b2 = SimpleNamespace(keys=[k2, k3f]), SimpleNamespace(keys=[k2, k3])
    text = f2.merged_exceptions([b1, b2], [[k2, k3f], [k2, k3]], ["F1", "F2"], "en")
    assert text.count("K2 is") == 1 and "In F1 and F2, K2 is" in text


def test_gif_preview_of_long_reruns_is_lighter():
    an = pytest.importorskip("scripts.exp19.figures.animate_v2")
    assert an.gif_settings(64) == (2, 256) and an.gif_settings(500) == (4, 64)
    assert "下 = 右转" in an.ALABELS["zh"]["legend_turns"] and "down = right" in an.ALABELS["en"]["legend_turns"]
    assert "hit" not in an.STRIP_LEGEND and "frame" in an.STRIP_LEGEND
    assert "或停止" in an.ALABELS["zh"]["legend_nomap"]
    assert an.INSET["w"] <= an.CAM["w"] / 3 + 1e-6


def test_overview_names_the_rules_by_reference_but_states_a_fallback_outside_the_pre_registration():
    """The overview caption refers to the pre-registered rules (stated in each episode's caption) without their
    formulas; a key moment chosen by a fallback that is not pre-registered is still stated."""
    from types import SimpleNamespace

    from scripts.exp19.figures import fig_v2 as f2

    def k(label, branch):
        return SimpleNamespace(label=label, branch=branch)
    bs = [SimpleNamespace(keys=[k("K1", "K1_first")]), SimpleNamespace(keys=[k("K1", "K1_first")])]
    chosen = [[k("K2", "K2_fallback"), k("K3", "K3_f1_closest")], [k("K2", "K2_turn"), k("K3", "K3_f1_fallback_after")]]
    text = f2.merged_exceptions(bs, chosen, ["F1", "F2"], "en", branches="nonreg")
    assert "In F2, K3 is" in text and "not in the pre-registration" in text and "K2" not in text
    assert f2.merged_exceptions(bs, chosen[:1], ["F1"], "en", branches="nonreg") == ""
    for lang in ("en", "zh"):
        ov = f2.CAPTION_OVERVIEW[lang]
        assert "⌊" not in ov and "{key_rules}" not in ov and ("pre-registered" in ov or "预注册" in ov)
        assert len(ov) < (900 if lang == "en" else 400)


# --------------------------------------------------------------------------- #
# Round 2: unwrapped history axis, slot legend, compressed no-map spans, slot rows on the strips, online cut
# --------------------------------------------------------------------------- #
def test_history_axis_has_up_left_with_behind_in_the_middle():
    """The history panel runs, top to bottom, ahead (y 360) / left (270) / behind (180, middle) / right (90) /
    ahead (y 0): up = left as on the future panel and the turn track, a band of past directions around +-180 is
    one piece, and a left turn (every bearing decreases) moves it UP, like the turn track's bar."""
    assert tp.hist_ylim() == (-tp.WRAP_PAD, 360.0 + tp.WRAP_PAD)
    assert tp.HIST_TICKS == (0.0, 90.0, 180.0, 270.0, 360.0)
    assert tp.hist_y(175.0) == 185.0 and tp.hist_y(-175.0) == 175.0  # both sides of behind: 10 deg apart
    assert tp.hist_y(90.0) == 270.0 and tp.hist_y(-90.0) == 90.0  # left above behind, right below it
    assert tp.hist_y(3.0) == 357.0 and tp.hist_y(-4.0) == 4.0  # ahead: one mark, at the end its sign gives
    turned = [tp.hist_y(b - 15.0) - tp.hist_y(b) for b in (-170.0, 175.0)]  # the robot turned left by 15 deg
    assert turned == [15.0, 15.0]  # both moved up by 15, across 180 without a jump
    b = tp.bin_bearings()
    img = b[:, None, None] * np.ones((1, 2, 4))
    pad = tp.WRAP_PAD
    out = tp.hist_rows(img, pad)
    shown = np.linspace(360.0 + pad - 0.5, -pad + 0.5, 360 + 2 * pad)  # row centres (y), top to bottom
    assert np.allclose(geo.wrap_deg(-shown), out[:, 0, 0])  # y = -bearing (mod 360)
    assert out.shape[0] == 360 + 2 * pad
    en, zh = tp_labels()
    assert en == ("ahead", "right", "back", "left", "ahead") and zh == ("前", "右", "后", "左", "前")


def tp_labels():
    from scripts.exp19.figures import fig_v2 as f2

    return f2.LABELS["en"]["y_hist"], f2.LABELS["zh"]["y_hist"]


def _sparse_timeline() -> tp.Timeline:
    """500 steps, warm-up 0-20, System 2 answering with turns 20-165, ready calls 165-285 every 11 steps, turns to
    the end (the shape of F2_1)."""
    ready = np.arange(165, 286, 11)
    calls = np.concatenate([[0, 5, 10, 15], np.arange(20, 165, 5), ready, np.arange(286, 500, 5)])
    kinds = np.array(["trajectory"] * 4 + ["native_actions"] * len(np.arange(20, 165, 5)) + ["trajectory"] * len(ready)
                     + ["native_actions"] * len(np.arange(286, 500, 5)))
    nxt = np.append(ready[1:], 286)
    a = {"step": ready, "next_step": nxt, "episode_steps": np.asarray(500), "first_ready_step": np.asarray(165),
         "calls_step": calls, "calls_kind": kinds, "call_index": np.nonzero(np.isin(calls, ready))[0],
         "key_labels": np.array(["K1"]), "key_call_index": np.nonzero(calls == 165)[0], "key_step": np.array([165])}
    tl = tp.Timeline({}, a)
    tl.pose_ready = calls >= 20
    return tl


def test_long_nomap_spans_are_compressed_when_ready_calls_are_sparse():
    tl = _sparse_timeline()
    assert tp.ready_cover(tl) < tp.NOMAP_COVER_MAX
    tp.compress_axis(tl, 338.0, 34.0)
    assert [(a, b) for a, b, _ in tl.blocks] == [(20.0, 165.0), (286.0, 500.0)]
    assert not tl.compressed  # a 20-step warm-up at this scale is about one block wide: drawn to scale
    x = tl.x_of(np.arange(0, 500.5, 0.5))
    assert np.all(np.diff(x) > 0)  # order in time is kept
    per = 338.0 / tl.span_steps()
    for a, b, width in tl.blocks:  # every block about the fixed width; widths are whole raster columns
        assert (tl.x_of(b) - tl.x_of(a)) * per == pytest.approx(34.0, abs=0.3 * per)
        assert width * tp.UPSAMPLE == pytest.approx(round(width * tp.UPSAMPLE))
    assert tl.x_of(200.0) - tl.x_of(190.0) == pytest.approx(10.0)  # the ready part keeps one step per x unit
    assert tl.breaks() == [20.0, 165.0, 286.0]
    xs = tl.x_of(np.asarray(tl.a["step"], float))
    assert np.allclose((xs - tp.raster_xlim(tl)[0]) * tp.UPSAMPLE, np.round((xs - tp.raster_xlim(tl)[0]) * tp.UPSAMPLE))
    # a dense rerun keeps its no-map spans to scale
    dense = _dense_timeline(16, 4, keys=(3, 9))
    dense.a.update(calls_step=np.asarray(dense.a["step"]), first_ready_step=np.asarray(0))
    tp.compress_axis(dense, 338.0, 34.0)
    assert dense.blocks == []


def test_step_axis_names_both_ends_of_a_compressed_span():
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_behavior as fb

    fb.setup("en")
    import matplotlib.pyplot as plt

    tl = _sparse_timeline()
    tp.compress_axis(tl, 338.0, 34.0)
    fig = plt.figure(figsize=(7.0, 2.0))
    ax = fig.add_axes([0.3, 0.1, 0.67, 0.8])
    ax.set_xlim(*tl.xlim())
    ticks = tp.axis_ticks(fig, tl, tp.per_step_pt(tl, ax))
    labels = [lab for _, lab in ticks]
    assert {"0", "20", "165", "286", "500"} <= set(labels)
    xs = [x for x, _ in ticks]
    assert xs == sorted(xs) and len(set(labels)) == len(labels)
    plt.close(fig)


def test_overview_marks_past_frames_1_4_8_only():
    tl = _tl_one_call(nan_slot=5)
    gt, pred = tp.history_marks(tl, [0], True, slots=tp.OVERVIEW_SLOTS)
    xs = tp.slot_xs(tl, 0, True)
    assert sorted({round(x, 6) for x, _ in pred}) == sorted(round(float(xs[k]), 6) for k in tp.OVERVIEW_SLOTS)
    assert tp.OVERVIEW_SLOTS == (0, 3, 7) and tp.slot_gap(tp.OVERVIEW_SLOTS) == 3 and tp.slot_gap() == 1


def test_history_strip_marks_sit_at_their_elevation_joined_when_apart_and_whole_inside_the_strip(synth):
    """Every mark at its own (bearing, elevation); a hairline joins a frame's peak to its true direction when the
    dot is not inside its circle, the short way round the ring."""
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_behavior as fb
    from scripts.exp19.figures import panels_v2 as p2

    assert p2.pair_segment(170.0, 5.0, -175.0, -20.0) == (170.0, 5.0, 185.0, -20.0)  # across +-180: 15 deg
    assert p2.pair_segment(10.0, 0.0, 30.0, 12.0) == (10.0, 0.0, 30.0, 12.0)
    fb.setup("en")
    import matplotlib.pyplot as plt

    ks = synth["b"].keys[0]
    fig = plt.figure(figsize=(1.64, 0.41))
    ax = fig.add_axes([0, 0, 1, 1])
    info = p2.draw_history_strip(ax, ks, 400, 320)
    gt, pred = p2.history_strip_marks(ks)
    drawn = [ln for ln in ax.lines if ln.get_marker() == "o"]
    ys = sorted({round(float(y), 6) for ln in drawn for y in ln.get_ydata()})
    want = sorted({round(float(e), 6) for _, _, e in gt + pred if abs(e) <= p2.HIST_ELEV})
    assert ys == want  # marks at their true elevation, nothing else
    assert info["scale"] == 1.0 and info["pairs"] >= 0
    plt.close(fig)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(1.08, 0.27))
    ax = fig.add_axes([0, 0, 1, 1])
    p2.setup_strip(ax, 45.0)
    lim = 180.0 + p2.STRIP_PAD
    for b in (179.0, -179.5, 175.0, 180.0, 90.0):
        pos = p2.inside_copies(ax, b, 1.9)
        assert pos and all(abs(x) <= lim for x in pos)
        assert all(abs(abs(x) - 180.0) <= 180.0 for x in pos)
    assert p2.inside_copies(ax, 90.0, 1.9) == [90.0]
    plt.close(fig)


def test_strip_ticks_keep_or_drop_both_ends_together():
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_behavior as fb
    from scripts.exp19.figures import fig_v2 as f2
    from scripts.exp19.figures import panels_v2 as p2

    fb.setup("en")
    import matplotlib.pyplot as plt

    for w, want in ((0.7, 3), (1.08, 5), (1.67, 5)):  # the overview's 1.08 in strips keep "back" at both ends
        fig = plt.figure(figsize=(w, 0.3))
        ax = fig.add_axes([0, 0.5, 1, 0.5])
        p2.setup_strip(ax, 30.0)
        edges = p2.strip_ticks(ax, f2.LABELS["en"]["axis"])
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert len(labels) == want and edges == (want == 5)
        assert labels[0] != "back" or labels[-1] == "back"
        assert list(ax.get_xticks(minor=True)) == [180.0, 90.0, 0.0, -90.0, -180.0]  # the marks stay at the bearings
        if edges:  # the edge labels stay inside the strip, clear of their neighbours
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            boxes = [t.get_window_extent(r) for t in ax.get_xticklabels()]
            axb = ax.get_window_extent(r)
            assert boxes[0].x0 >= axb.x0 - 0.5 and boxes[-1].x1 <= axb.x1 + 0.5
            gap = min(b.x0 - a.x1 for a, b in zip(boxes, boxes[1:])) * 72.0 / fig.dpi
            assert gap >= p2.STRIP_LABEL_AIR - 0.3
        plt.close(fig)


def test_route_note_for_a_route_that_hides_under_the_start():
    from scripts.exp19.figures import panels_v2 as p2

    ref = np.array([[0.0, 0.0], [6.0, 0.0], [6.0, 4.0]])
    goal = np.array([6.0, 4.0])
    small = np.array([[0.0, 0.0], [0.3, 0.2], [0.5, 0.5], [0.2, 0.6]])
    assert p2.route_note_m(small, ref, goal, 3.0, 1.95, 1.9) == pytest.approx(0.8)
    assert p2.route_note_m(ref, ref, goal, 3.0, 1.95, 1.9) is None


def test_captions_and_legend_say_what_is_drawn():
    from scripts.exp19.figures import fig_v2 as f2

    en, zh = f2.LABELS["en"]["legend"], f2.LABELS["zh"]["legend"]
    assert "visible from here" in en["gt"] and "仅画此处可见" in zh["gt"]
    assert "not given to the model" in en["frame"] and "未输入模型" in zh["frame"]
    for key in ("slots", "slots_some", "slots_ov"):  # a column's spread is by past frame, not by step
        assert "not time" in en[key] and "非时间" in zh[key]
    assert "1, 4, 8" in en["slots_ov"] and "1、4、8" in zh["slots_ov"] and "1 = oldest" in en["frames"]
    assert "stacked" in en["slots_stacked"] and "叠" in zh["slots_stacked"] and "peak" in en["pair"]
    assert set(f2.SLOT_KEYS) <= set(f2.LEGEND_KEYS) and "frames" in f2.LEGEND_KEYS and "pair" in f2.LEGEND_KEYS
    for mode, key in f2.SLOT_KEY.items():  # one slot entry per figure, the one that says what is drawn
        keys = f2.legend_keys_for(True, mode=mode)
        assert [k for k in keys if k in f2.SLOT_KEYS] == [key]
    assert [k for k in f2.legend_keys_for(True, overview=True) if k in f2.SLOT_KEYS] == ["slots_ov"]
    assert "frames" in f2.legend_keys_for(True, overview=True) and "frames" not in f2.legend_keys_for(True)
    assert "stop" not in f2.legend_keys_for(False)
    assert len(f2.LEGEND_GROUPS) == 3 and len(f2.LABELS["en"]["legend_groups"]) == 3
    for lang in ("en", "zh"):
        page, ov = f2.CAPTION_PAGE[lang], f2.CAPTION_OVERVIEW[lang]
        assert ("a call after the warm-up that returned a pixel goal" in page) == (lang == "en")
        assert ("预热期之后、慢系统给出像素目标的调用" in page) == (lang == "zh")
        for t in (page, ov):
            assert ("orange dot without a blue circle" in t) or ("有橙点而无蓝圈" in t)
        assert ("8°" in page) and ("360°" in page) and ("360°" in ov) and ("8°" in ov)
        assert "nearly always" not in page and "几乎总在" not in page  # design wording, not a claim about the data
        assert ("up = left" in page and "up = left" in ov) or ("向上 = 向左" in page and "向上 = 向左" in ov)
        for mode in ("all", "some", "148", "stacked"):
            assert f2.CAPTION_SLOTS[lang][mode]


def test_overview_warmup_clause_follows_each_case():
    from types import SimpleNamespace

    from scripts.exp19.figures import fig_v2 as f2

    a = SimpleNamespace(compressed=True, blocks=[])
    b = SimpleNamespace(compressed=False, blocks=[])
    assert f2.warm_overview([a, a], ["T1", "T2"], "en") == f2.CAPTION_WARM["overview"]["en"]
    some = f2.warm_overview([a, b], ["F1", "F2"], "en")
    assert "in F1 only" in some and "(" not in some
    assert "F1" in f2.warm_overview([a, b], ["F1", "F2"], "zh") and f2.warm_overview([b, b], ["F1", "F2"], "en") == ""


def test_animation_timeline_is_cut_at_the_playhead():
    an = pytest.importorskip("scripts.exp19.figures.animate_v2")
    tl = _tl_one_call()
    tl.a.update({"calls_step": np.array([0, 20, 24]), "calls_index": np.array([2, 3, 4]),
                 "calls_kind": np.array(["trajectory"] * 3),
                 "calls_ready": np.array([False, True, True]), "first_ready_step": np.asarray(20),
                 "s1_path_bearing": np.full((1, 33), 3.0), "s1_path_dist": np.linspace(0.0, 2.0, 33)[None]})
    up = an.TimelineUpTo(tl, 21)  # two steps into a column that runs to step 24 (the call's 4-action chunk)
    assert int(up.a["column_end"][0]) == 22 and up.now == 22  # the field is painted up to the playhead's step
    assert int(up.a["next_step"][0]) == 24  # ... while the column (the call's own chunk) is known at the call
    plan = tp.mark_plan(tl, 7.0 * 72.0)  # all 8 slots spread (a wide column)
    assert plan.slots[0] == tp.ALL_SLOTS
    static_gt, static_pred = tp.history_marks(tl, [0], plan=plan)
    shown = []
    for t in (20, 21, 22, 23, 30):  # marks keep their static positions and appear as the painted field reaches them
        tlt = an.TimelineUpTo(tl, t)
        gt, pred = an.revealed_history_marks(tlt, plan)
        end = float(tlt.a["column_end"][0])
        assert set(gt) <= set(static_gt) and set(pred) <= set(static_pred)
        assert all(x <= end + 1e-9 for x, _ in gt + pred)  # nothing right of the painted field
        assert {m for m in static_gt if m[0] <= end} == set(gt)
        shown.append(len(gt) + len(pred))
        xs, _ = an.revealed_future_marks(tlt, plan)
        assert all(x <= end + 1e-9 for x in xs) and (len(xs) > 0) == (end >= 22.0)  # the endpoint at the centre
    assert shown == sorted(shown) and 0 < shown[0] < shown[-1]  # revealed step by step, never taken back
    assert an.revealed_history_marks(an.TimelineUpTo(tl, 30), plan) == (static_gt, static_pred)  # all, once painted
    assert an.TimelineUpTo(tl, 19).R == 0  # nothing before the call
    img = tp.column_raster(up, np.ones((1, 360, 4)))
    assert img[0, :, 3].sum() == pytest.approx((22 - 20) * tp.UPSAMPLE)  # two steps painted
    # the playhead sits at the painted frontier (t + 1): no field, span, turn bar or mark right of it
    tl.a["step_action"] = np.array([bd.FORWARD] * 21 + [bd.LEFT, bd.FORWARD, bd.RIGHT] + [bd.FORWARD] * 16)
    for t in range(0, 41):
        tlt = an.TimelineUpTo(tl, t)
        head = an.playhead_step(tlt)
        assert head == min(t + 1, 40)
        assert an.painted_max_x(tlt, plan) <= tlt.x_of(head) + 1e-9
        assert all(float(tlt.x_of(k)) <= head for k in tp.key_steps(tlt).values())  # K lines at their steps
    tlt = an.TimelineUpTo(tl, 21)
    assert an.painted_max_x(tlt, plan) == pytest.approx(22.0)  # the turn of step 21 is painted, up to 22
    assert an.HEIGHT_PX % 2 == 0 and an.WIDTH_PX == 1920
    assert "slots" in an.BOTTOM_LEGEND2 and "pair" in an.STRIP_LEGEND
    assert an.NOW_LW > tp.KEY_LW and an.NOW_DASH  # the playhead is not a key-moment hairline


# --------------------------------------------------------------------------- #
# Round 3: mark plan (air between circles), stacked note, axis end, online marks that never move
# --------------------------------------------------------------------------- #
def test_mark_plan_keeps_air_between_neighbouring_circles():
    """All 8 slots where a column leaves GAP_PT between neighbouring circles, else past frames 1, 4, 8 (per
    column), else (long reruns) stacked at the centre on every n-th call; the scale keeps the gap."""
    wide = _dense_timeline(16, 4, keys=(3, 9))  # 64 steps, a call every 4 steps
    for width_pt, mode in ((64 * 7.0, "all"), (64 * 4.5, "148"), (64 * 0.7, "stacked")):
        plan = tp.mark_plan(wide, width_pt)
        assert plan.mode == mode, (width_pt, plan.mode)
        col = 4 * width_pt / 64.0
        if plan.spread:
            slots = plan.slots[0]
            pitch = col * tp.slot_gap(slots) / tp.NUM_SLOTS
            assert tp.ring_outer_pt(plan.scale) + tp.GAP_PT <= pitch + 1e-6
            assert tp.SPREAD_MIN_SCALE <= plan.scale <= tp.TL_MARK and plan.rows == list(range(wide.R))
        else:
            assert all(v is None for v in plan.slots.values()) and set(plan.rows) >= {3, 9}
    mixed = _dense_timeline(16, 4, keys=(3, 9))
    mixed.a["next_step"] = mixed.a["next_step"].copy()
    mixed.a["next_step"][-1] = mixed.a["step"][-1] + 2  # one 2-step chunk (the rerun ended)
    plan = tp.mark_plan(mixed, 64 * 7.0)
    assert plan.mode == "some" and plan.slots[15] == tp.OVERVIEW_SLOTS and plan.slots[0] == tp.ALL_SLOTS
    assert tp.mark_plan(wide, 64 * 7.0, slots=tp.OVERVIEW_SLOTS).mode == "148"


def test_compressed_axis_names_its_last_step_and_the_unit_sits_left():
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_behavior as fb

    fb.setup("en")
    import matplotlib.pyplot as plt

    tl = _sparse_timeline()
    tp.compress_axis(tl, 338.0, 34.0)
    fig = plt.figure(figsize=(7.0, 1.0))
    ax = fig.add_axes([0.33, 0.5, 0.67, 0.3])
    ax.set_xlim(*tl.xlim())
    ticks = tp.step_axis(ax, tl, "step")
    assert ticks[-1] == (pytest.approx(tl.xlim()[1]), "500")
    assert ax.get_xticklabels()[-1].get_ha() == "right"  # the end label stays on the page
    unit = [t for t in ax.texts if t.get_text() == "step"]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    assert unit and unit[0].get_window_extent(r).x1 < ax.get_window_extent(r).x0  # left of the axis
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Round 5: display smoothing, opaque note clear of the key lines, header ink, legend wording, animation glyphs
# --------------------------------------------------------------------------- #
def test_display_smoothing_removes_the_grid_ripple_and_keeps_peaks():
    """The heat fields are drawn blurred along the bearing (circularly): a 5.6 deg ripple (the decoder's 4-pixel
    grid) nearly vanishes, every ring keeps its maximum and its peak stays in place; the stored ring is untouched."""
    from scripts.exp19.figures import panels_v2 as p2

    i = np.arange(360)
    bump = np.exp(-0.5 * (np.minimum(np.abs(i - 100), 360 - np.abs(i - 100)) / 6.0) ** 2)
    ring = np.stack([bump * (1.0 + 0.15 * np.cos(2 * np.pi * i / 5.6)), 0.4 * np.roll(bump, 262)])
    raw = ring.copy()
    out = p2.smooth_rings(ring)
    assert np.array_equal(ring, raw)
    assert np.allclose(out.max(-1), ring.max(-1))  # each ring keeps its own maximum
    assert abs(int(np.argmax(out[0])) - 100) <= 2
    ripple = lambda r: np.abs(np.diff(r[60:140], 2)).max()  # noqa: E731
    assert ripple(out[0]) < 0.25 * ripple(ring[0])
    assert out[1][359] > 0.01 and out[1][0] > 0.01  # circular: a bump at +180 / -180 spreads across the seam
    img = np.zeros((90, 360))
    img[40:44, 200:204] = 1.0
    sm = p2.smooth_ring_image(img)
    assert sm.max() == pytest.approx(1.0) and np.unravel_index(np.argmax(sm), sm.shape)[1] in range(199, 205)
    assert p2.smooth_rings(np.zeros((2, 360))).max() == 0.0


def test_timeline_fields_are_drawn_smoothed_from_the_stored_rings():
    tl = _tl_one_call()
    ring = np.zeros((1, 360))
    ring[0, 170:190:3] = 1.0  # a comb: the thin bands of the maps' grid
    tl.a["hist_ring"] = ring
    tl.a["fut_ring"] = np.repeat(ring[:, None], 4, axis=1)
    h, f = tp.display_hist_rings(tl), tp.display_fut_rings(tl)
    assert h.shape == (1, 360) and f.shape == (1, 4, 360)
    assert h.max() == pytest.approx(1.0) and np.ptp(h[0, 172:188]) < 0.25  # the comb becomes one smooth band
    assert tl.a["hist_ring"][0, 171] == 0.0  # the stored ring is untouched


def test_stacked_stride_is_in_the_legend_not_over_the_data():
    """A long rerun's stacked marks carry no in-panel note (it hid key-moment hairlines): the legend's timeline
    entry names the stride on a page, and an overview gets its own legend row for its stacked rows."""
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_v2 as f2

    f2.setup("en")
    import matplotlib.pyplot as plt

    dense = _dense_timeline(125, 4, keys=(3, 50))
    for name in ("hist_gt_bearing", "hist_pred_peak_bearing"):
        dense.a[name] = np.full((dense.R, 8), 170.0)
    dense.a.update(hist_gt_visible=np.ones((dense.R, 8), bool), hist_pred_conf=np.full((dense.R, 8), 0.9),
                   hist_ring=np.zeros((dense.R, 360)), calls_step=np.asarray(dense.a["step"]),
                   first_ready_step=np.asarray(0))
    fig = plt.figure(figsize=(7.0, 2.0))
    ax = fig.add_axes([0.33, 0.1, 0.67, 0.8])
    info = tp.draw_history_panel(ax, dense, ("a", "r", "b", "l", "a"))
    assert info["mode"] == "stacked" and info["marker_stride"] >= 2
    assert [t for t in ax.texts if t.get_bbox_patch() is not None] == []  # no box over the data
    plt.close(fig)
    for lang, tail in (("en", "one call in 3 (and every key moment)"), ("zh", "每 3 次调用画一次（关键时刻都画）")):
        L = f2.LABELS[lang]
        text = f2.slot_legend_text(L, "stacked", 3)
        assert text.startswith(L["legend"]["slots_stacked"]) and text.endswith(tail)
        assert f2.slot_legend_text(L, "stacked", 1) == L["legend"]["slots_stacked"]
        assert f2.slot_legend_text(L, "148", 3) == L["legend"]["slots_ov"]
        assert f2.split_at_semicolon(text)[1] == tail  # a two-line entry breaks before the stride
    assert f2.stacked_overview_text(f2.LABELS["en"], "en", [("F2", 3)]) == (
        "F2 timeline: a call's 8 past frames stacked mid-column; one call in 3 (and every key moment)")
    assert f2.stacked_overview_text(f2.LABELS["zh"], "zh", [("F2", 3)]) == (
        "F2 的时间线：一次调用的 8 个历史帧叠在列中央；每 3 次调用画一次（关键时刻都画）")
    assert "one call in" not in f2.stacked_overview_text(f2.LABELS["en"], "en", [("F2", 1)])
    for lang in ("en", "zh"):  # the overview's legend: the slot entry, its stacked row right after it
        f2.setup(lang)
        fig = plt.figure(figsize=(7.0, 1.0))
        L = f2.legend_labels(f2.LABELS[lang], {"slots_stacked_ov": f2.stacked_overview_text(f2.LABELS[lang], lang,
                                                                                            [("F2", 3)])})
        lay = f2.legend_columns(fig, L, 504.0, f2.legend_keys_for(True, overview=True) + f2.SLOT_EXTRA)
        keys = [k for _, it in lay["cols"] for k, _ in it]
        assert keys.index("slots_stacked_ov") == keys.index("slots_ov") + 1
        entry = next(lines for _, it in lay["cols"] for k, lines in it if k == "slots_stacked_ov")
        assert L["legend"]["slots_stacked_ov"] in (" ".join(entry), "".join(entry))  # nothing dropped
        plt.close(fig)
    assert "stacked_note" not in f2.LABELS["en"] and not hasattr(tp, "stacked_note")


def test_strip_labels_keep_three_points_of_air_and_their_ticks():
    """The overview's 1.08 in strip (en): "back" at both edges and every neighbour >= 3 pt apart -- "right" moves
    towards the middle just enough, still over its tick; the page's wider strips keep their centred labels."""
    from scripts.exp19.figures import panels_v2 as p2

    widths = [12.69, 8.1, 16.74, 11.79, 12.69]  # back / left / ahead / right / back at 6 pt (Nimbus Sans)
    for w_pt in (77.76, 120.06):
        per = w_pt / 376.0
        pos = [(188.0 - b) * per for b in (180, 90, 0, -90, -180)]
        lefts = p2.strip_label_layout(pos, widths, w_pt)
        assert lefts is not None and lefts[0] >= 0.0 and lefts[4] + widths[4] <= w_pt + 1e-9
        gaps = [lefts[i + 1] - (lefts[i] + widths[i]) for i in range(4)]
        assert min(gaps) >= p2.STRIP_LABEL_AIR - 1e-6
        for x, p, w in zip(lefts[1:4], pos[1:4], widths[1:4]):
            assert x - 1e-6 <= p <= x + w + 1e-6  # each label still over its own tick
        if w_pt < 100:
            assert lefts[3] < pos[3] - widths[3] / 2 - 0.5  # "right" moved towards the middle
    assert p2.strip_label_layout([1.0, 12.0, 25.0, 38.0, 49.0], widths, 50.4) is None  # too narrow: no edge labels


def test_slot_labels_use_measured_widths_in_every_language():
    pytest.importorskip("matplotlib")
    from scripts.exp18.figures import common_draw as cd
    from scripts.exp19.figures import fig_v2 as f2

    import matplotlib.pyplot as plt

    decisions = {}
    for lang in ("en", "zh"):
        f2.setup(lang)
        fig = plt.figure(figsize=(2.0, 1.0))
        need = 2 * tp.SLOT_LABEL_INSET_PT + cd.text_width_pt(fig, "1", 6.0) + cd.text_width_pt(fig, "8", 6.0) + 2.0
        assert 8.0 < need < 13.0
        assert tp.slot_labels_fit(fig, need) and not tp.slot_labels_fit(fig, need - 0.05)
        decisions[lang] = [tp.slot_labels_fit(fig, w) for w in (15.69, 16.1, 11.1, 10.16, 2.7)]
        plt.close(fig)
    assert decisions["en"] == decisions["zh"] and decisions["en"][:3] == [True, True, True]


def test_animation_note_discloses_the_smoothing_and_fits_the_frame():
    pytest.importorskip("matplotlib")
    an = pytest.importorskip("scripts.exp19.figures.animate_v2")
    from scripts.exp19.figures import fig_behavior as fb
    from scripts.exp19.figures import fig_v2 as f2

    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    assert f2.smooth_sentence("en") == ("Heat fields are drawn with a Gaussian blur (σ = 2° on the 360° strips, 3.5° in "
                                        "bearing on the timelines) for display; dots mark the raw peaks.")
    assert f2.smooth_sentence("zh") == "热力场为显示做了高斯平滑（环视条带 σ = 2°，时间线方位向 σ = 3.5°）；圆点为未平滑的峰值。"
    for lang in ("en", "zh"):
        f2.setup(lang)
        fig = plt.figure(figsize=(an.W, an.H))
        page = fb.Page(fig, an.W, an.H)
        n = an.draw_notes(page, an.TLG["notes"], an.ALABELS[lang], lang)
        assert n <= 4 and an.TLG["notes"] + n * f2.LINE <= an.H
        text = "".join(t.get_text() for t in fig.findobj(Text) if t.get_text())
        assert ("Gaussian blur" in text) if lang == "en" else ("高斯平滑" in text)
        assert f2.texts_outside(fig) == []
        plt.close(fig)


def test_header_lines_keep_the_italic_ink_inside_the_page():
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_v2 as f2

    f2.setup("en")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.0, 1.0))
    text = ("“walk through the doorway directly away from the chair, continue straight and turn right, immediately "
            "continue to the left into the pantry area. walk past the sink, fridge, oven, and stop before you get to "
            "the dining room table.”")
    lines = f2.wrap_ink(fig, text, f2.FS["body"], 504.0, fontstyle="italic")
    assert " ".join(lines) == text
    for line in lines:
        left, right = f2.ink_extent_pt(line, f2.FS["body"], fontstyle="italic")
        assert left >= -0.5 and right <= 504.0 - f2.INK_MARGIN_PT
    plt.close(fig)


def test_legend_says_which_frames_a_column_marks_in_plain_words():
    from scripts.exp19.figures import fig_v2 as f2

    en, zh = f2.LABELS["en"]["legend"], f2.LABELS["zh"]["legend"]
    assert en["slots_some"] == ("timeline: a call's past frames 1→8 left→right (not time); narrow columns: frames 1, "
                                "4, 8 only")
    assert zh["slots_some"] == "时间线：同一次调用的历史帧 1→8 从左到右（非时间）；窄列只画 1、4、8"
    assert "narrow: 1, 4, 8" not in en["slots_some"] and "窄列 1、4、8" not in zh["slots_some"]
    assert f2.split_at_semicolon(en["slots_some"]) == ["timeline: a call's past frames 1→8 left→right (not time);",
                                                       "narrow columns: frames 1, 4, 8 only"]
    assert f2.split_at_semicolon(zh["slots_some"])[0].endswith("；")
    assert f2.split_at_semicolon("no semicolon here") is None


def test_legend_slot_entry_moves_beside_the_timeline_context_only_when_that_saves_a_row():
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_v2 as f2

    for lang in ("en", "zh"):
        f2.setup(lang)
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(7.0, 1.0))
        L = f2.LABELS[lang]
        base = f2.legend_columns(fig, L, 504.0, f2.legend_keys_for(False, mode="all"))
        some = f2.legend_columns(fig, L, 504.0, f2.legend_keys_for(False, mode="some"))
        assert some["rows"] <= base["rows"]  # the longer wording costs no extra row
        for lay in (base, some):
            items = [(k, lines) for _, it in lay["cols"] for k, lines in it]
            slot = [(k, lines) for k, lines in items if k in f2.SLOT_KEYS]
            assert len(slot) == 1
            key, lines = slot[0]
            assert L["legend"][key] in (" ".join(lines), "".join(lines))  # nothing dropped by the wrap
            if lay["slot_group"] == 2:  # beside the timeline's context, right after "System 2 gave turns or STOP"
                keys3 = [k for k, _ in lay["cols"][2][1]]
                assert keys3[keys3.index("nomap") + 1] in f2.SLOT_KEYS
        plt.close(fig)


def test_animation_position_is_a_plain_dot_and_the_strip_legend_is_one_line():
    an = pytest.importorskip("scripts.exp19.figures.animate_v2")
    from scripts.exp19.figures import fig_v2 as f2

    assert 4.0 <= an.POS_MS <= 5.0 and 0.0 < an.POS_HALO <= 1.0
    f2.setup("en")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(1.0, 1.0))
    ax = fig.add_axes([0, 0, 1, 1])
    an.position_mark(ax, 0.5, 0.5)
    halo, dot = ax.lines[-2:]
    assert dot.get_markerfacecolor() != "white" and dot.get_markeredgecolor() == "none"
    assert dot.get_markersize() == an.POS_MS and halo.get_markersize() == pytest.approx(an.POS_MS + 2 * an.POS_HALO)
    assert halo.get_markeredgecolor() == "none"  # no ring: unlike the pixel goal's ring glyph
    width = an.STRIP["w"] * 72.0 - 17.0
    for lang in ("en", "zh"):
        f2.setup(lang)
        text = an.ALABELS[lang]["legend_frame"]
        assert an.legend_lines(fig, text, an.FS["small"], width) == [text]
    f2.setup("en")
    lines = an.legend_lines(fig, "alpha beta gamma delta epsilon zeta eta theta", an.FS["small"], 95.0)
    assert len(lines) == 1 or " " in lines[-1].strip()  # never a lone word on the last line
    plt.close(fig)

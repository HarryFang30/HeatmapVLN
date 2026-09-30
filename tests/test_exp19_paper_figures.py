"""EXP-19 paper figures (scripts/exp19/figures/fig_paper.py) and the warm-up crop of the timeline.

Synthetic bundle and timeline as in ``test_exp19_figures_v2.py`` (no server data); the render tests need
matplotlib.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from scripts.exp19.figures import bundle as bd
from scripts.exp19.figures import synthetic_bundle as sb
from scripts.exp19.figures import timeline_panel as tp


def _calls(b: bd.Bundle, pose_from: int = 16, every: int = 4) -> list:
    """A call every 4 steps, ready from step 20, warm-up until ``pose_from``, the last call STOP (as the v2 tests)."""
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
    return calls


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp("exp19_fig_paper")
    path = sb.make_bundle(out, episodes_path=None, clip_dir=None, topdown_root=out / "no_maps_here", seed=3)
    b = bd.load_bundle(path)
    calls = _calls(b)
    meta, a = tp.synthetic_timeline(b, calls)
    tl_dir = out / "records_v2"
    tp.write_timeline(tl_dir / f"{b.ep_key}_timeline", meta, a)
    record = Path(path).parent / f"{b.ep_key}.json"
    record.write_text(json.dumps({"calls": calls}), encoding="utf-8")
    return {"out": out, "bundle": path, "b": b, "tl_dir": tl_dir, "record": record}


def _timeline(synth) -> tp.Timeline:
    b = synth["b"]
    return tp.load_timeline(tp.timeline_path_for(synth["tl_dir"], b.ep_key), record_path=synth["record"])


# --------------------------------------------------------------------------- #
# Warm-up crop
# --------------------------------------------------------------------------- #
def test_crop_starts_the_axis_at_the_first_ready_call_and_keeps_it_linear(synth):
    tl = tp.crop_warmup(_timeline(synth))
    first = float(np.min(tl.a["step"]))
    assert tl.cropped and not tl.compressed and tl.blocks == []
    assert tl.xlim() == (first, float(tl.steps))
    assert tl.breaks() == []  # no axis-break mark: the warm-up is left out, not squeezed
    assert tl.x_of(first) == first and tl.x_of(float(tl.steps)) == float(tl.steps)
    s_k, x_k = tl.knots()
    np.testing.assert_allclose(s_k, x_k)  # slope 1 from the first ready call on


def test_crop_is_undone_by_compression(synth):
    tl = tp.crop_warmup(_timeline(synth))
    tp.compress_axis(tl, 300.0, tp.WARM_BLOCK_PT["page"])
    assert not tl.cropped


def _label_boxes(fig, ticks, per, x_end, fs=6.0):
    from scripts.exp18.figures import common_draw as cd

    boxes = []
    for x, lab in ticks:
        w = cd.text_width_pt(fig, lab, fs)
        lo = (x - x_end) * per - w if abs(x - x_end) < 1e-9 else (x - x_end) * per - w / 2  # the end label: ha right
        boxes.append((lo, lo + w, lab))
    return sorted(boxes)


@pytest.mark.parametrize("width_pt", [260.0, 330.0, 400.0])
def test_cropped_axis_names_its_first_and_last_step_and_labels_never_touch(synth, width_pt):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tl = tp.crop_warmup(_timeline(synth))
    fig = plt.figure(figsize=(4, 1))
    per = width_pt / tl.span_steps()
    ticks = tp.axis_ticks(fig, tl, per)
    assert ticks[0] == (tl.x0, str(int(tl.x0))) and ticks[-1] == (float(tl.steps), str(tl.steps))
    boxes = _label_boxes(fig, ticks, per, float(tl.steps))
    for (a0, a1, la), (b0, b1, lb) in zip(boxes, boxes[1:]):
        assert b0 - a1 >= 2.9, (la, lb, b0 - a1)
    plt.close(fig)


def test_a_round_tick_next_to_the_right_aligned_end_label_is_dropped():
    """The axis ends at step 61, 39 steps after 22, drawn ~390 pt wide: "60" would sit 10 pt left of the end tick,
    under the right-aligned "61"."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = {"step": np.array([22, 30]), "next_step": np.array([30, 61]), "episode_steps": np.asarray(61),
         "call_index": np.array([5, 7]), "key_labels": np.array(["K1"]), "key_call_index": np.array([5]),
         "calls_step": np.array([0, 4, 8, 12, 16, 22, 26, 30]), "first_ready_step": np.asarray(22)}
    tl = tp.crop_warmup(tp.Timeline({}, a))
    fig = plt.figure(figsize=(4, 1))
    labels = [lab for _, lab in tp.axis_ticks(fig, tl, 390.0 / tl.span_steps())]
    plt.close(fig)
    assert labels[0] == "22" and labels[-1] == "61" and "60" not in labels and "50" in labels


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
POLICY_WORDS = ("pose", "odometry", " vo ", "heatmap", "no claim", "does not feed", "example", "accurate")


@pytest.fixture(scope="module")
def rendered(synth, tmp_path_factory):
    """Both figures, three rows each (the synthetic episode three times, as a paper figure has), en and zh."""
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_paper as fp

    out = tmp_path_factory.mktemp("exp19_fig_paper_out") / "figures_paper"
    assert fp.main(["--records", str(Path(synth["bundle"]).parent), "--timelines", str(synth["tl_dir"]),
                    "--out-dir", str(out), "--lang", "en", "zh", "--fig-a", "T1", "T1", "T1",
                    "--fig-b", "T1", "T1", "T1"]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    return {"out": out, "manifest": manifest, "by": {f["figure"]: f for f in manifest["figures"]}}


def test_render_both_figures_editable_clean_and_at_the_aspect(rendered):
    from scripts.exp19.figures import fig_paper as fp

    out, manifest = rendered["out"], rendered["manifest"]
    assert manifest["schema"] == fp.MANIFEST_SCHEMA and manifest["paper_width_in"] == pytest.approx(7.16)
    assert [f["figure"] for f in manifest["figures"]] == ["fig_a", "fig_b"]
    for fig in manifest["figures"]:
        for lang in ("en", "zh"):
            w, h = fig["size_in"][lang]
            assert w == pytest.approx(7.16) and fp.ASPECT_RANGE[0] <= h / w <= fp.ASPECT_RANGE[1]
            c = fig["checks"][lang]
            assert c["outside"] == 0 and c["overlaps"] == 0 and c["leader_crossings"] == 0, c
            assert c["min_font_pt"] >= 6.0 and c["legend_lines"] == 2
            assert fig["warnings"][lang] == [], fig["warnings"][lang]
        files = [out / name for lang in ("en", "zh") for name in fig["files"][lang]]
        assert sorted(f.suffix for f in files) == sorted([".pdf", ".svg", ".png", ".txt"] * 2)
        assert all(f.stat().st_size > 0 for f in files)
        stem = "fig_a_key_moments" if fig["figure"] == "fig_a" else "fig_b_online_timeline"
        svg = (out / f"{stem}_en.svg").read_text(encoding="utf-8")
        assert svg.count("<text") > 10  # live text, editable
        assert b"/Subtype /Type3" not in (out / f"{stem}_en.pdf").read_bytes()  # TrueType text, editable
        assert re.search(r">\d+(\.\d+)? m</text>", svg)  # the scale bar's halo label is text too (``editable_text``)
        assert fig["checks"]["en"]["halo_texts_boxed"] > 0
    fb_checks = rendered["by"]["fig_b"]["checks"]["en"]["checks"][0]
    assert fb_checks["x0_step"] == fb_checks["first_ready_step"] > 0  # the warm-up is left out
    assert fb_checks["compressed_warmup"] is False


def test_key_moments_are_numbered_in_time_order_and_alike_in_both_figures(rendered, synth):
    b = synth["b"]
    want = [[k.label, str(i + 1), int(k.step)] for i, k in enumerate(sorted(b.keys, key=lambda k: int(k.step)))]
    assert [m[0] for m in want] == ["K1", "K3", "K2", "K4"]  # the synthetic K3 (step 32) comes before K2 (step 36)
    for lang in ("en", "zh"):
        rows_a = rendered["by"]["fig_a"]["checks"][lang]["checks"]
        rows_b = rendered["by"]["fig_b"]["checks"][lang]["checks"]
        for ra, rb in zip(rows_a, rows_b):
            assert ra["ep_key"] == rb["ep_key"] == b.ep_key
            assert [m[:3] for m in ra["moments"]] == [m[:3] for m in rb["moments"]] == want
            assert ra["route_badges"] == rb["route_badges"] == 4  # circled on both route maps
            badges = rb["badges"]  # the timeline's badges: 1-4 left to right
            assert [x[0] for x in badges] == ["1", "2", "3", "4"]
            assert [x[1] for x in badges] == sorted(x[1] for x in badges)


def test_bearings_and_strip_rows_are_named_once(rendered):
    out = rendered["out"]
    en = (out / "fig_a_key_moments_en.svg").read_text(encoding="utf-8")
    for word in ("ahead", "left", "right", "History", "Future"):
        assert en.count(f">{word}</text>") == 1, word
    assert en.count(">back</text>") == 2  # both ends of the one labelled strip
    zh = (out / "fig_a_key_moments_zh.svg").read_text(encoding="utf-8")
    for word in ("前", "左", "右", "历", "未"):  # zh row names are stacked upright characters
        assert zh.count(f">{word}</text>") == 1, word


def test_chips_in_the_rendered_figure_leave_every_pixel_goal_free(rendered):
    for lang in ("en", "zh"):
        c = rendered["by"]["fig_a"]["checks"][lang]
        assert c["chips_cover_goal"] == 0 and len(c["chips_corners"]) == 12


def test_captions_describe_without_policy_words_and_disclose_the_smoothing(rendered):
    out = rendered["out"]
    cap_a = (out / "fig_a_key_moments_caption_en.txt").read_text(encoding="utf-8")
    cap_b = (out / "fig_b_online_timeline_caption_en.txt").read_text(encoding="utf-8")
    assert "smoothed for display (σ = 2°)" in cap_a and "smoothed for display (σ = 3.5°)" in cap_b
    for cap in (cap_a, cap_b):
        low = cap.lower()
        assert not any(w in low for w in POLICY_WORDS), cap
        assert "past frames 1 (oldest), 4 and 8 of the eight given to System 2" in cap
        assert "Key moments 1–4 are chosen among the System 2 calls with an affordance map: the first, " in cap
        assert 3 <= cap.count(". ") + 1 <= 5  # 3-5 sentences
        assert max(len(t.split()) for t in cap.split(". ")) <= 60, cap  # short sentences
    assert "executed actions (chips)" in cap_a and "pixel goal (ring)" in cap_a
    assert "System 1 path (dots, also on the future map)" in cap_a  # the dots on the future strips
    assert "history (orange; camera's 79° view boxed)" in cap_a  # the box is on the history strip only
    assert "The x-axis is the step, starting at the first System 2 call" in cap_b  # the ticks are absolute steps
    assert "up = left" in cap_b and "grey columns are calls without" in cap_b
    for name in ("fig_a_key_moments_caption_zh.txt", "fig_b_online_timeline_caption_zh.txt"):
        zh = (out / name).read_text(encoding="utf-8")
        assert "为显示做了平滑（σ = " in zh and "heatmap" not in zh.lower() and "位姿" not in zh and "里程计" not in zh
        assert 3 <= zh.count("。") <= 5 and "第 1（最早）、4、8 帧" in zh


def test_manifest_keeps_the_other_figure_and_language(synth, tmp_path):
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_paper as fp

    base = ["--records", str(Path(synth["bundle"]).parent), "--timelines", str(synth["tl_dir"]),
            "--out-dir", str(tmp_path / "p")]
    assert fp.main(base + ["--lang", "en", "--fig-a", "--fig-b", "T1"]) == 0  # figure B, en
    assert fp.main(base + ["--lang", "zh", "--fig-a", "T1", "--fig-b"]) == 0  # figure A, zh
    assert fp.main(base + ["--lang", "zh", "--fig-a", "--fig-b", "T1"]) == 0  # figure B, zh
    m = json.loads((tmp_path / "p" / "manifest.json").read_text(encoding="utf-8"))
    by = {f["figure"]: f for f in m["figures"]}
    assert set(by["fig_a"]["files"]) == {"zh"} and set(by["fig_b"]["files"]) == {"en", "zh"}
    # one row is far from the page aspect: that is the only warning
    assert all("height / width" in w for f in m["figures"] for lang in f["warnings"] for w in f["warnings"][lang])


def test_output_dir_of_the_other_figure_sets_is_refused(synth, tmp_path):
    from scripts.exp19.figures import fig_paper as fp

    for name in ("figures", "figures_v2"):
        assert fp.main(["--records", str(Path(synth["bundle"]).parent), "--timelines", str(synth["tl_dir"]),
                        "--out-dir", str(tmp_path / name), "--lang", "en"]) == 2
        assert not (tmp_path / name).exists()


def test_chips_leave_the_corner_holding_the_pixel_goal(synth):
    from scripts.exp19.figures import fig_paper as fp

    ks = synth["b"].keys[0]
    h, w = ks.decision_rgb.shape[:2]
    ks_right = _with_marks(ks, goal=(w - 10.0, h - 8.0))
    ks_free = _with_marks(ks, goal=(w / 2, h / 2))
    ks_both = _with_marks(ks, goal=(w - 10.0, h - 8.0), extra=[(10.0, h - 8.0)])
    assert fp.chips_corner(ks_right, 1.5, 40.0) == "left"
    assert fp.chips_corner(ks_free, 1.5, 40.0) == "right"
    assert fp.chips_corner(ks_both, 1.5, 40.0) == "top"  # both bottom corners taken: top right


def test_chips_never_cover_the_pixel_goal(synth):
    """Wherever the pixel goal is (a grid over the image, a path down the middle and along the bottom), the chips as
    drawn (``chips_box`` of the chosen corner) never hold it."""
    from scripts.exp19.figures import fig_paper as fp

    ks = synth["b"].keys[0]
    h, w = ks.decision_rgb.shape[:2]
    path = [(w / 2, v) for v in np.linspace(h - 5, 5, 12)] + [(u, h - 6.0) for u in np.linspace(5, w - 5, 12)]
    for width_pt in (20.0, 34.0, fp.chips_width(ks)):
        for u in np.linspace(1.0, w - 2.0, 9):
            for v in np.linspace(1.0, h - 2.0, 7):
                k = _with_marks(ks, goal=(u, v), extra=path)
                corner = fp.chips_corner(k, 1.27, width_pt)
                box = fp.chips_box(k, 1.27, width_pt, corner, fp.GOAL_R_PT)  # the ring clear, not only its centre
                assert not fp._inside(np.array([u, v]), box)[0], (u, v, corner)


def test_chips_keep_off_the_pixel_goal_ring_not_just_its_centre(synth):
    """A goal whose centre is 3 pt right of the bottom-right chips' left edge, low in the image: under ``CHIP_AIR_PT``
    alone its centre would count as clear, but its ring (``GOAL_R_PT`` = 3.65 pt) reaches the chips."""
    from scripts.exp19.figures import fig_paper as fp

    ks = synth["b"].keys[0]
    h, w = ks.decision_rgb.shape[:2]
    width_pt, w_in = 30.0, 1.27
    u0, v0, _, _ = fp.chips_box(ks, w_in, width_pt, "right")
    per = w / (w_in * 72.0)
    k = _with_marks(ks, goal=(u0 - 3.0 * per, h - 10.0))  # 3 pt left of the chips box, level with the chips
    corner = fp.chips_corner(k, w_in, width_pt)
    assert corner != "right"
    ring = fp.chips_box(k, w_in, width_pt, corner, fp.GOAL_R_PT)
    assert not fp._inside(np.array([u0 - 3.0 * per, h - 10.0]), ring)[0]


def test_a_route_leader_keeps_off_another_key_dot():
    """``leader_clear_pt``: a leader passing 1 pt from another key position costs its configuration; off (0, the
    default of every other caller), the score is as before."""
    from scripts.exp19.figures import panels_v2 as p2

    per_pt = 0.01
    p, other = np.array([0.0, 0.0]), np.array([0.08, 0.01])  # the other dot 8 pt along, 1 pt off the leader
    q = np.array([0.2, 0.0])
    keys = np.array([p, other])
    limits = (-1.0, 1.0, -1.0, 1.0)
    base = p2._config_score([p], [q], np.zeros((0, 2)), [], limits, per_pt, [20.0])
    assert p2._config_score([p], [q], np.zeros((0, 2)), [], limits, per_pt, [20.0], keys=keys) == base
    assert p2._config_score([p], [q], np.zeros((0, 2)), [], limits, per_pt, [20.0], keys=keys, clear_pt=3.5) < base
    away = np.array([-0.2, 0.0])  # leaving p away from the other dot: no cost
    assert p2._config_score([p], [away], np.zeros((0, 2)), [], limits, per_pt, [20.0], keys=keys, clear_pt=3.5) == \
        p2._config_score([p], [away], np.zeros((0, 2)), [], limits, per_pt, [20.0])


def test_stacked_timelines_are_named_in_the_caption():
    """A long rerun's timeline stacks all eight past frames mid-column for one call in k: the caption says which
    rows, since its "past frames 1, 4 and 8" does not hold there."""
    from scripts.exp19.figures import fig_paper as fp

    rows = [{"mode": "148", "marker_stride": 1}, {"mode": "stacked", "marker_stride": 2},
            {"mode": "stacked", "marker_stride": 1}]
    en = fp.stacked_sentence(rows, "en")
    assert en.startswith(" On the long reruns of (b) and (c), all eight past frames' marks are stacked")
    assert "one call in 2" in en
    assert "(b)、(c)" in fp.stacked_sentence(rows, "zh")
    assert fp.stacked_sentence(rows[:1], "en") == "" == fp.stacked_sentence([], "zh")


def _with_marks(ks, goal, extra=()):
    """The key moment with its pixel goal and System 1 path replaced (wherever the bundle keeps them)."""
    path = np.asarray([[ks.decision_rgb.shape[1] / 2, 5.0]] + [list(p) for p in extra], dtype=np.float64)
    meta, arrays = dict(ks.meta), dict(ks.arrays)
    for name, value in (("pixel_goal_uv", list(goal)), ("path_uv", path)):
        (arrays if name in arrays else meta)[name] = value
    return bd.KeyStep(index=ks.index, meta=meta, arrays=arrays)


def _keys(*steps_branches):
    from types import SimpleNamespace as K

    return [K(label=f"K{i + 1}", index=i, step=s, branch=br) for i, (s, br) in enumerate(steps_branches)]


F1_KEYS = ((22, "K1_first"), (35, "K2_fallback"), (31, "K3_f1_closest"), (72, "K4_last"))  # the RTX 4090 F1 case
T1_KEYS = ((22, "K1_first"), (35, "K2_turn"), (39, "K3_two_thirds"), (56, "K4_last"))


def test_f1_is_renumbered_in_time_order():
    """F1's K3 (step 31, the closest approach) comes before its K2 (step 35): they are drawn as 2 and 3."""
    from types import SimpleNamespace

    from scripts.exp19.figures import fig_paper as fp

    f1 = SimpleNamespace(keys=_keys(*F1_KEYS))
    assert fp.time_numbers(f1) == {"K1": "1", "K3": "2", "K2": "3", "K4": "4"}
    assert [k.label for k in fp.time_order(f1.keys)] == ["K1", "K3", "K2", "K4"]


def test_timeline_key_moments_take_their_numbers(synth):
    from scripts.exp19.figures import fig_paper as fp

    tl = tp.crop_warmup(_timeline(synth))
    nums = fp.time_numbers(synth["b"])
    got = fp.number_keys(tl, nums)
    assert [g[:2] for g in got] == [("K1", "1"), ("K3", "2"), ("K2", "3"), ("K4", "4")]
    assert [g[2] for g in got] == sorted(g[2] for g in got)
    assert sorted(tl.key_rows()) == ["1", "2", "3", "4"]  # the badge row draws the numbers


def test_moment_rules_follow_each_rows_branches():
    from scripts.exp19.figures import fig_paper as fp

    t1, f1 = fp.time_order(_keys(*T1_KEYS)), fp.time_order(_keys(*F1_KEYS))
    t2 = fp.time_order(_keys((20, "K1_first"), (28, "K2_fallback"), (36, "K3_two_thirds"), (52, "K4_last")))
    text = fp.moment_rules([("a", t1), ("b", t1), ("c", f1)], "en")
    assert text == ("Key moments 1–4 are chosen among the System 2 calls with an affordance map: the first, the later "
                    "one with the largest executed turn, the one nearest two thirds of the episode and the last; in "
                    "(c), 2 is the last one up to the closest approach to the goal and 3 the one a third of the way "
                    "through (no later one's executed turn reaches 30°).")
    text = fp.moment_rules([("a", t1), ("b", t2), ("c", t1)], "en")
    assert text.endswith("; in (b), 2 is the one a third of the way through "
                         "(no later one's executed turn reaches 30°).")
    zh = fp.moment_rules([("a", t1), ("b", t1), ("c", f1)], "zh")
    assert zh.startswith("关键时刻 1–4 从给出 affordance map 的慢系统调用中选取：第一次、其后执行转角最大的一次、")
    assert zh.endswith("；(c) 中 2 为最接近目标那一步及之前的最后一次，3 为位于三分之一处的一次（其后没有一次执行转角达到 30°）。")
    assert "(a)" not in fp.moment_rules([("a", t1)], "en")  # a row that took the common rules is not named


def test_the_k2_fallback_is_worded_as_keysteps_decides_it():
    """keysteps' K2 fallback looks only at the other ready calls (the calls with an affordance map), not at the
    no-map calls' turns (F1 turns 180° in its grey span) nor at K1's own chunk: the wording says "no later one"
    inside the pool the sentence names, never "no executed turn"."""
    from scripts.exp19 import keysteps as ksel
    from scripts.exp19.figures import fig_paper as fp

    ready = [{"call_index": i, "step": 20 + 4 * i, "executed_actions": [2, 2, 1, 1] if i == 0 else [1, 2, 1, 3]}
             for i in range(7)]
    got = ksel.select_key_steps(ready, category="T1", episode_steps=50)
    assert got[1]["branch"] == "K2_fallback" and ksel.net_turn_deg(ready[0]["executed_actions"]) == 30.0
    keys = fp.time_order(_keys(*[(g["step"], g["branch"]) for g in got]))
    en, zh = fp.moment_rules([("a", keys)], "en"), fp.moment_rules([("a", keys)], "zh")
    assert "no later one's executed turn reaches 30°" in en and "no executed turn" not in en
    assert "其后没有一次执行转角达到 30°" in zh and "（没有执行转角" not in zh


def test_moment_rules_for_short_reruns_and_no_key_moments():
    from scripts.exp19.figures import fig_paper as fp

    t1 = fp.time_order(_keys(*T1_KEYS))
    lt4 = fp.time_order(_keys((21, "all_lt4"), (25, "all_lt4")))
    text = fp.moment_rules([("a", t1), ("b", lt4)], "en")
    assert "{" not in text and text.endswith("; (b) has only 2, all shown.")
    one = fp.time_order(_keys((21, "all_lt4")))
    assert fp.moment_rules([("a", t1), ("b", one)], "en").endswith("; (b) has only one.")
    assert fp.moment_rules([("a", [])], "en") == "" and fp.moment_rules([], "zh") == ""
    for lang in ("en", "zh"):  # every row a short rerun: one plain sentence, no template left over
        for rows in ([("a", lt4)], [("a", lt4), ("b", one)]):
            text = fp.moment_rules(rows, lang)
            assert text == fp.RULES[lang]["all"] + fp.RULES[lang]["end"] and "{" not in text and "(a)" not in text


def test_pick_takes_the_one_main_case_of_a_category(synth, tmp_path):
    from scripts.exp19.figures import fig_paper as fp

    b = synth["b"]
    main = bd.Bundle(path=b.path, meta=dict(b.meta), keys=b.keys)
    other = bd.Bundle(path=b.path, meta={**b.meta, "ep_key": "zzz_0001", "is_main": False, "memberships": [],
                                         "predicate_holds_on_rerun": False}, keys=b.keys)
    cat = main.membership(main=True)["category"]
    for order in ([main, other], [other, main]):
        picked, warnings = fp.pick(order, [cat])
        assert picked == [main] and warnings == []
    picked, warnings = fp.pick([main, other], ["zzz_0001"])
    assert picked == [other] and warnings and "no longer meets" in warnings[0]
    with pytest.raises(SystemExit):
        fp.pick([main, main], [cat])


def test_touching_circled_numbers_are_caught(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.exp19.figures import fig_paper as fp

    fig = plt.figure(figsize=(2, 1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 144)
    ax.set_ylim(0, 72)
    fp.num_badge(ax, 20.0, 36.0, "1")
    fp.num_badge(ax, 60.0, 36.0, "2")
    assert fp.badge_overlaps(fig) == []
    fp.num_badge(ax, 25.0, 36.0, "3")  # 5 pt from "1": the circles touch, the digits' boxes do not
    assert fp.badge_overlaps(fig) == ["'1' x '3'"]
    plt.close(fig)


def test_compact_148_groups_a_calls_marks_and_labels_them(synth):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tl = tp.crop_warmup(_timeline(synth))
    xs = tp.slot_xs(tl, 0, True, tp.OVERVIEW_SLOTS)
    s0, s1 = float(tl.a["step"][0]), float(tl.a["next_step"][0])
    np.testing.assert_allclose(xs[list(tp.OVERVIEW_SLOTS)], s0 + (np.arange(3) + 0.5) / 8 * 0 +
                               np.array([0.5, 3.5, 7.5]) / 8 * (s1 - s0))  # off: spread by slot
    tl.compact148 = True
    xs = tp.slot_xs(tl, 0, True, tp.OVERVIEW_SLOTS)
    np.testing.assert_allclose(xs[list(tp.OVERVIEW_SLOTS)], s0 + np.array(tp.COMPACT_148) * (s1 - s0))
    fig = plt.figure(figsize=(6, 1.2))
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.8])
    tp.draw_history_panel(ax, tl, ("ahead", "right", "back", "left", "ahead"), slots=tp.OVERVIEW_SLOTS)
    plan = tp.mark_plan(tl, ax, tp.OVERVIEW_SLOTS)
    labels = tp.slot_end_labels(ax, tl, plan)
    assert labels in (["1", "4", "8"], [])
    plt.close(fig)


def test_future_labels_sit_on_their_lines_when_there_is_room():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for h_in, centred in ((0.28, False), (0.42, True)):
        fig = plt.figure(figsize=(4, 2))
        ax = fig.add_axes([0.2, 0.2, 0.7, h_in / 2])
        ax.set_ylim(*tp.bearing_ylim())
        tp.fut_ticks(ax, ("left", "ahead", "right"))
        vas = [t.get_va() for t in ax.get_yticklabels()]
        assert (not any(v in ("bottom", "top") for v in vas)) is centred, (h_in, vas)
        plt.close(fig)


def test_route_badges_keep_off_the_route():
    from scripts.exp19.figures import panels_v2 as p2

    per_pt = 0.01
    route = np.column_stack([np.linspace(0, 2, 101), np.zeros(101)])
    on = np.array([1.0, 0.0])
    off = np.array([1.0, (p2.BADGE_HALF[1] + 3.0) * per_pt])
    assert p2.covered_points(on, route, per_pt) > 0 and p2.covered_points(off, route, per_pt) == 0
    limits = (-1.0, 3.0, -1.0, 1.0)
    assert p2._config_score([on], [off], route, [], limits, per_pt, [11.0]) > \
        p2._config_score([on], [on], route, [], limits, per_pt, [11.0])


def test_halo_texts_become_boxed_text(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.exp18.figures import common_draw as cd
    from scripts.exp19.figures import fig_paper as fp

    fig = plt.figure(figsize=(2, 1))
    t = fig.text(0.5, 0.5, "start", path_effects=cd.HALO_THIN)
    assert fp.editable_text(fig) == 1 and t.get_path_effects() == [] and t.get_bbox_patch() is not None
    plt.close(fig)

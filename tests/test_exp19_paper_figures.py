"""EXP-19 paper figures (scripts/exp19/figures/fig_paper.py) and the warm-up crop of the timeline.

Synthetic bundle and timeline as in ``test_exp19_figures_v2.py`` (no server data); the render tests need
matplotlib.
"""
from __future__ import annotations

import json
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
def test_render_both_figures_editable_and_clean(synth, tmp_path):
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_paper as fp

    records = Path(synth["bundle"]).parent
    out = tmp_path / "figures_paper"
    assert fp.main(["--records", str(records), "--timelines", str(synth["tl_dir"]), "--out-dir", str(out),
                    "--lang", "en", "zh", "--fig-a", "T1", "--fig-b", "T1", "--keys", "K1", "K2", "K4"]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == fp.MANIFEST_SCHEMA and manifest["paper_width_in"] == pytest.approx(7.16)
    kinds = [f["figure"] for f in manifest["figures"]]
    assert kinds == ["fig_a", "fig_b"]
    for fig in manifest["figures"]:
        for lang in ("en", "zh"):
            assert fig["size_in"][lang][0] == pytest.approx(7.16)
            c = fig["checks"][lang]
            assert c["outside"] == 0 and c["overlaps"] == 0 and c["leader_crossings"] == 0, c
            assert c["min_font_pt"] >= 6.0
        # one synthetic row is far from 1:1; the only warnings allowed are about the aspect
        assert all("height / width" in w for lang in ("en", "zh") for w in fig["warnings"][lang]), fig["warnings"]
        files = [out / name for lang in ("en", "zh") for name in fig["files"][lang]]
        assert sorted(f.suffix for f in files) == sorted([".pdf", ".svg", ".png", ".txt"] * 2)
        for f in files:
            assert f.stat().st_size > 0 and fig["files"][f.name.rsplit("_", 1)[-1].split(".")[0]][f.name]
        svg = (out / "fig_a_key_moments_en.svg" if fig["figure"] == "fig_a" else
               out / "fig_b_online_timeline_en.svg").read_text(encoding="utf-8")
        assert svg.count("<text") > 10  # live text, editable
        pdf = (out / (Path(svg_name(fig)).stem + ".pdf")).read_bytes()
        assert b"/Subtype /Type3" not in pdf  # TrueType text, editable
        assert '>start<' in svg  # the route map's halo labels are text too (``editable_text``)
    fb_checks = manifest["figures"][1]["checks"]["en"]["checks"][0]
    assert fb_checks["x0_step"] == fb_checks["first_ready_step"] > 0  # the warm-up is left out
    assert fb_checks["compressed_warmup"] is False
    cap = (out / "fig_b_online_timeline_caption_en.txt").read_text(encoding="utf-8")
    assert "the steps before it are not shown" in cap and "no claim" in cap
    assert "does not feed the actions" in cap and "σ = 3.5°" in cap and "σ = 2°" not in cap
    for text in (cap, (out / "fig_a_key_moments_caption_en.txt").read_text(encoding="utf-8")):
        low = text.lower()
        assert not any(w in low for w in ("pose", "odometry", " vo ", "heatmap", "accurate"))


def svg_name(fig) -> str:
    return "fig_a_key_moments_en.svg" if fig["figure"] == "fig_a" else "fig_b_online_timeline_en.svg"


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
    assert fp.chips_corner(ks_both, 1.5, 40.0) == "right"  # both corners taken: the default


def _with_marks(ks, goal, extra=()):
    """The key moment with its pixel goal and System 1 path replaced (wherever the bundle keeps them)."""
    path = np.asarray([[ks.decision_rgb.shape[1] / 2, 5.0]] + [list(p) for p in extra], dtype=np.float64)
    meta, arrays = dict(ks.meta), dict(ks.arrays)
    for name, value in (("pixel_goal_uv", list(goal)), ("path_uv", path)):
        (arrays if name in arrays else meta)[name] = value
    return bd.KeyStep(index=ks.index, meta=meta, arrays=arrays)


def test_key_rules_name_the_cases_that_took_another_branch():
    from types import SimpleNamespace as K

    from scripts.exp19.figures import fig_paper as fp

    cases = [("a", [K(label="K1", branch="K1_first", step=20), K(label="K2", branch="K2_turn", step=30)]),
             ("b", [K(label="K1", branch="K1_first", step=20), K(label="K2", branch="K2_fallback", step=28)]),
             ("c", [K(label="K1", branch="K1_first", step=22), K(label="K2", branch="K2_turn", step=35),
                    K(label="K3", branch="K3_f1_closest", step=31)])]
    text = fp.key_rules(cases, "en")
    assert text.startswith("K1 = the first call with an affordance map; K2 = the call with an affordance map, "
                           "other than K1, whose executed chunk has the largest net turn")
    assert "; in (b), K2 = the call a third of the way through the calls with an affordance map, as no other" in text
    assert "K3 = the last call with an affordance map, other than K1 and K2, at or before the step closest" in text
    assert text.endswith("not in time order.")  # (c): K3 at step 31 before K2 at step 35
    assert "；(b) 中 K2 为" in fp.key_rules(cases, "zh")
    in_order = fp.key_rules(cases[:2], "en")
    assert "time order" not in in_order


def test_key_rules_for_short_reruns_and_no_key_moments():
    from types import SimpleNamespace as K

    from scripts.exp19.figures import fig_paper as fp

    lt4 = [("a", [K(label="K1", branch="K1_first", step=20), K(label="K2", branch="K2_turn", step=30)]),
           ("b", [K(label="K1", branch="all_lt4", step=21), K(label="K2", branch="all_lt4", step=25)])]
    text = fp.key_rules(lt4, "en")
    assert "{" not in text and "In (b), the rerun has only 2 ready calls" in text
    assert fp.key_rules([("a", [])], "en") == "" and fp.key_rules([], "zh") == ""


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
    with pytest.raises(SystemExit):
        fp.check_keys(["K1", "K1"])
    with pytest.raises(SystemExit):
        fp.check_keys(["K5"])


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

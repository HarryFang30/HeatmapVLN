"""EXP-19 figures (scripts/exp19/figures): bundle schema, caption claim rules, strip conventions, smoke render.

The bundles here come from ``synthetic_bundle.make_bundle`` with every external
source switched off (generated route, fake top-down level, procedural images),
so the tests need no server data.  The smoke render needs matplotlib.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.exp18 import geometry as geo
from scripts.exp19.figures import bundle as bd
from scripts.exp19.figures import synthetic_bundle as sb


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp("exp19_fig")
    path = sb.make_bundle(out, episodes_path=None, clip_dir=None, topdown_root=out / "no_maps_here", seed=3)
    return out, path


def _load_raw(path):
    json_path, npz_path = bd.bundle_paths(path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(npz_path) as z:
        arrays = {k: z[k] for k in z.files}
    return meta, arrays


def _write(tmp_path, meta, arrays, name="x_0001_bundle"):
    stem = tmp_path / name
    stem.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    np.savez(stem.with_suffix(".npz"), **arrays)
    return stem.with_suffix(".json")


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def test_synthetic_bundle_is_valid_and_clean(synth):
    _, path = synth
    b = bd.load_bundle(path)
    assert b.synthetic and b.schema == bd.SCHEMA
    assert 1 <= len(b.keys) <= 4 and [k.label for k in b.keys] == [f"K{i + 1}" for i in range(len(b.keys))]
    assert b.warnings == []
    ks = b.keys[0]
    assert ks.hist_pred.shape == (8, 4, 64, 64) and ks.history_rgb.shape[0] == ks.history_count
    assert ks.pano_rgb.shape[0] == 4 and ks.pano_rgb.shape[1] == ks.pano_rgb.shape[2]


def test_validation_lists_every_problem(synth, tmp_path):
    _, path = synth
    meta, arrays = _load_raw(path)
    bad_meta, bad_arrays = copy.deepcopy(meta), dict(arrays)
    del bad_meta["instruction"]
    assert any("missing field 'instruction'" in e for e in bd.validate_bundle(bad_meta, arrays))

    bad_meta = copy.deepcopy(meta)
    bad_meta["ep_key"] = "wrong_0001"
    bad_meta["key_steps"][0]["executed_actions"] = [1, 7]
    bad_meta["key_steps"][-1]["label"] = "K1"
    bad_arrays["k0_hist_pred"] = arrays["k0_hist_pred"][:, :3]
    bad_arrays["k0_hist_mask"] = arrays["k0_hist_mask"].astype(np.uint8)
    errors = " | ".join(bd.validate_bundle(bad_meta, bad_arrays))
    for needle in ("bundle.ep_key", "executed_actions", "duplicate labels", "npz k0_hist_pred: shape",
                   "npz k0_hist_mask: dtype"):
        assert needle in errors, (needle, errors)
    with pytest.raises(bd.BundleError):
        bd.load_bundle(_write(tmp_path, bad_meta, bad_arrays))


def test_one_based_npz_prefix_is_accepted_with_a_warning(synth, tmp_path):
    _, path = synth
    meta, arrays = _load_raw(path)
    shifted = {}
    for name, a in arrays.items():
        i, rest = name[1:].split("_", 1)
        shifted[f"k{int(i) + 1}_{rest}"] = a
    b = bd.load_bundle(_write(tmp_path, meta, shifted))
    assert any("1-based" in w for w in b.warnings)
    np.testing.assert_array_equal(b.keys[0].hist_pred, arrays["k0_hist_pred"])


def test_open_fields_from_build_records_are_drawable(synth, tmp_path):
    """[F] may leave these None: no map, no outcome, no eval-log reference, no first System2 turn."""
    from scripts.exp19.figures import fig_behavior as fb

    _, path = synth
    meta, arrays = _load_raw(path)
    meta.update(outcome=None, eval_log_outcome=None, predicate_holds_on_rerun=None)
    meta["topdown"].update(level_index=None, note="no top-down map: FileNotFoundError")
    meta["fidelity"] = {"first_divergent_call": None, "identical_calls": None, "total_calls": None}
    meta["key_steps"][0].update(system2_first_output=None, response_actions=None, cf_actions=None, cf_changed=None)
    meta["key_steps"][1]["pixel_goal_uv"] = [700.0, 20.0]  # off the image: a (row, col) swap would do this
    b = bd.load_bundle(_write(tmp_path, meta, arrays))
    text = " | ".join(b.warnings)
    for needle in ("no top-down map", "no rerun outcome", "no eval-log reference", "K2: pixel_goal_uv"):
        assert needle in text, (needle, text)
    assert b.keys[0].system2_texts == [meta["key_steps"][0]["system2_output"]]
    assert fb.resolve_level(b) is None
    assert fb.outcome_line(b, fb.LABELS["en"]) == fb.LABELS["en"]["outcome_missing"]
    cap = fb.caption_text(b, "en", [])
    assert "fidelity" not in cap.lower() and cap.endswith(".")


# --------------------------------------------------------------------------- #
# Caption claims follow the pre-registered wording rules
# --------------------------------------------------------------------------- #
def _verdicts(h1="support", h2="not_measured", h3="report_only", rate=0.25):
    return {"H1": {"verdict": h1, "joint_pck8": 0.86, "floor_pck8": 0.21, "gain": 0.65, "gain_ci95_low": 0.5},
            "H2": {"verdict": h2, "n_nonfront_bins": 12}, "H3": {"verdict": h3, "change_rate": rate, "n": 200}}


def test_claims_follow_the_wording_rules():
    from scripts.exp19.figures.fig_behavior import claim_sentences

    assert claim_sentences(None, "en") == []
    en = claim_sentences(_verdicts(), "en")
    assert en == ["In the closed-loop reruns the history affordance map agrees with the ground truth "
                  "(PCK@8 0.860, trivial baseline 0.210)."]
    partial = " ".join(claim_sentences(_verdicts(h1="partial"), "en"))
    assert "better than a trivial baseline" in partial and "agree" not in partial
    assert claim_sentences(_verdicts(h1="refute"), "en") == []
    for h2 in ("not_measured", "partial", "refute"):
        assert not any("future" in c for c in claim_sentences(_verdicts(h1="refute", h2=h2), "en"))
    assert claim_sentences(_verdicts(h1="refute", h2="support"), "zh") == ["未来 affordance map 与随后的动作方向一致。"]
    assert claim_sentences(_verdicts(h1="refute", h3="support", rate=0.125), "zh") == ["撤掉历史记忆时 12% 的决定点动作块改变。"]
    assert claim_sentences(_verdicts(h1="refute", h3="refute"), "en") == [
        "The affordance maps are shown as a display of the model's internal state."]
    for word in ("missing", "void"):
        assert claim_sentences(_verdicts(h1=word, h2=word, h3=word), "en") == []
    for lang in ("en", "zh"):
        every = " ".join(claim_sentences(_verdicts(h2="support", h3="support"), lang)).lower()
        assert not any(w in every for w in ("understand", "remember", "理解", "记住", "better decision", "更好"))


def test_claims_refuse_unknown_verdicts_and_missing_numbers():
    from scripts.exp19.figures.fig_behavior import claim_sentences

    with pytest.raises(ValueError, match="not one of"):
        claim_sentences(_verdicts(h1="supported"), "en")
    with pytest.raises(ValueError, match="not one of"):
        claim_sentences(_verdicts(h2="report_only"), "en")  # H2 has no "report only" branch
    v = _verdicts()
    v["H1"]["joint_pck8"] = None
    with pytest.raises(ValueError, match="joint_pck8"):
        claim_sentences(v, "en")
    with pytest.raises(ValueError, match="lack H3"):
        claim_sentences({k: v[k] for k in ("H1", "H2")}, "en")


def test_claims_read_build_records_verdicts():
    """The verdict dict build_records.py actually writes (H3 block from per-call rows) drives the caption."""
    br = pytest.importorskip("scripts.exp19.build_records")
    from scripts.exp19.figures.fig_behavior import claim_sentences

    rows = [{"ready": True, "cf_changed": i < 3, "ep_key": f"s_{i % 3:04d}", "endpoint_shift_m": 0.1 * i,
             "replay_actions_equal": True, "category": "T1"} for i in range(10)]
    h3 = br.h3_block(rows, reps=50, seed=0)["verdict"]
    assert h3["verdict"] == "support"
    verdicts = {"H1": br._verdict("partial", joint_pck8=0.7, floor_pck8=0.4),
                "H2": br._verdict("not_measured"), "H3": h3}
    assert claim_sentences(verdicts, "en")[-1] == ("When the history memory is removed, 30% of the decision points "
                                                   "change their action chunk.")


# --------------------------------------------------------------------------- #
# What the strips show
# --------------------------------------------------------------------------- #
def test_history_composite_normalises_each_slot_and_fades_by_none():
    pred = np.zeros((8, 4, 64, 64))
    pred[0, 0, 10, 10] = 0.2  # weak but confident slot
    pred[1, 2, 32, 32] = 0.9  # strong slot the head calls "not visible"
    pred[2, 1, 5, 5] = 1.0  # padded slot
    none_p = np.array([0.0, 0.75, 0, 0, 0, 0, 0, 0])
    mask = np.array([True, True] + [False] * 6)
    comp = bd.history_pred_composite(pred, none_p, mask)
    assert comp.shape == (4, 64, 64)
    assert comp[0, 10, 10] == pytest.approx(1.0) and comp[2, 32, 32] == pytest.approx(0.25)
    assert comp[1].max() == 0.0


def test_path_bearing_is_left_positive_at_camera_height():
    # camera coords: x right, y up, -z forward; floor at y = -1.25
    path = np.array([[0.0, -1.25, 0.0], [0.0, -1.25, -2.0], [-2.0, -1.25, 0.0], [2.0, -1.25, 0.0]])
    bearing, elev, idx = bd.path_directions(path)
    np.testing.assert_array_equal(idx, [1, 2, 3])  # the robot's own position has no direction
    np.testing.assert_allclose(bearing, [0.0, 90.0, -90.0], atol=1e-9)
    np.testing.assert_allclose(elev, 0.0, atol=1e-9)  # future labels put waypoints at camera height
    assert bd.net_turn_deg([bd.LEFT, bd.LEFT, bd.FORWARD, bd.RIGHT]) == 15.0


def test_strip_is_centred_ahead_with_left_on_the_left():
    pn = pytest.importorskip("scripts.exp19.figures.panels")
    maps = np.zeros((4, 64, 64))
    maps[geo.LEFT, 32, 32] = 1.0  # optical axis of the left view
    ring = pn.ring_heat(maps, width=720, elev=10.0)
    col = int(ring.max(0).argmax())
    assert geo.ring_column_azimuths(720)[col] == pytest.approx(90.0, abs=0.6)
    assert col < 360  # xlim runs +180 -> -180, so +90 (left) is in the left half
    assert pn.front_columns(720).sum() == 158  # the deployed camera's 79 deg, not the re-render's 90


def test_directions_beyond_the_band_become_edge_marks():
    """The history strip is a fixed +-15 deg band.  On stairs a true past direction (always a mark) or a visible
    slot's predicted peak beyond it is an edge triangle, never squashed onto the edge as if it were there."""
    pn = pytest.importorskip("scripts.exp19.figures.panels")
    from scripts.exp19.figures.fig_behavior import has_edge_marks

    def row(elev_deg):  # label row of a point straight along a view's axis at this elevation (+ = up)
        return 32.0 - 32.0 * np.tan(np.radians(elev_deg))

    gt = [(0, geo.BACK, row(-5.0), 32.0), (1, geo.BACK, row(-20.0), 32.0)]
    pred = [(0, geo.LEFT, row(10.0), 32.0), (1, geo.LEFT, row(29.4), 32.0)]
    marks = pn.history_marks(gt, pred, 15.0)
    assert [(k, inside) for k, _, _, inside in marks] == [("gt", True), ("gt", False), ("pred", False)]
    assert marks[1][2] == pytest.approx(-20.0, abs=0.1) and marks[2][1] == pytest.approx(90.0, abs=0.1)

    base = {"hist_gt_peak": np.full((8, 3), -1.0), "hist_mask": np.zeros(8, bool),
            "hist_pred": np.zeros((8, 4, 64, 64)), "hist_none": np.ones(8)}
    base["hist_mask"][:2] = True
    base["hist_gt_peak"][0] = (geo.BACK, row(-5.0), 32.0)
    base["hist_pred"][0, geo.BACK, int(round(row(-5.0))), 32] = 1.0
    base["hist_none"][0] = 0.1
    assert not has_edge_marks([bd.KeyStep(0, {}, base)], 15.0)  # flat floor
    stairs = copy.deepcopy(base)
    stairs["hist_pred"][1, geo.LEFT, 63, 32] = 1.0  # far down, but the head calls the slot "none": no mark
    assert not has_edge_marks([bd.KeyStep(0, {}, stairs)], 15.0)
    stairs["hist_none"][1] = 0.2
    assert has_edge_marks([bd.KeyStep(0, {}, stairs)], 15.0)


def test_row_labels_stay_inside_the_strip():
    """The quietest sector can be back-right (start -135 deg); the label must then end inside the -180 edge."""
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_behavior as fb
    from scripts.exp19.figures import panels as pn

    fb.setup("en")
    import matplotlib.pyplot as plt

    hist = np.zeros((4, 64, 64))
    for view in (geo.FRONT, geo.LEFT, geo.RIGHT):
        hist[view, 26:38, 20:44] = 1.0
    hist[geo.BACK, 26:38, 44:60] = 1.0  # back-left half (the back view's right columns)
    assert pn.quietest_sector(pn.ring_heat(hist, 720, 15.0)) == -135.0
    fut = np.repeat(hist[None], 4, axis=0)  # every bin: the same quiet back-right sector
    fig = plt.figure(figsize=(7.0, 2.0))
    views = np.full((4, 64, 64, 3), 128, np.uint8)
    for j, y0 in enumerate((0.1, 0.55)):
        ax = fig.add_axes([0.0, y0, 1.0, 0.35])  # the strip spans the page: nothing may run off it
        if j == 0:
            pn.draw_history_strip(ax, views, hist, [], [], 15.0, 720, 720, ("F", "L", "R", "B"), label="history row")
        else:
            pn.draw_future_strip(ax, fut, np.zeros((33, 3)), 9.0, 720, label="future row")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        box = ax.get_window_extent(renderer)
        (text,) = [t for t in ax.texts if t.get_text().endswith("row")]
        ext = text.get_window_extent(renderer)
        assert box.x0 <= ext.x0 and ext.x1 <= box.x1, (text.get_text(), ext, box)
        assert ext.x0 > box.x0 + 0.6 * box.width  # it still sits in the right-hand (quiet) part
    plt.close(fig)


def test_stop_chip_text_is_legible():
    pn = pytest.importorskip("scripts.exp19.figures.panels")
    for size in (6.2, 6.6, 7.6):
        assert pn.stop_fs(size) >= 5.0 and pn.chip_width(bd.STOP, size) >= 3.1 * pn.stop_fs(size)


# --------------------------------------------------------------------------- #
# Caption facts: outcome, fidelity, key-moment rules, floor note
# --------------------------------------------------------------------------- #
def _bundle(meta_updates=None, keys=()):
    meta = {"outcome": {"success": False, "oracle_success": True, "ne_m": 3.6, "steps": 85, "ended_by": "stop"},
            "eval_log_outcome": None, "fidelity": {"first_divergent_call": None, "identical_calls": None,
                                                   "total_calls": None}}
    meta.update(meta_updates or {})
    return bd.Bundle(Path("x_bundle.json"), meta, list(keys))


def test_outcome_line_names_how_the_rerun_ended():
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_behavior as fb

    L = fb.LABELS["en"]
    o = {"success": False, "oracle_success": False, "ne_m": 5.25, "steps": 212, "ended_by": "other"}
    assert fb.outcome_line(_bundle({"outcome": o}), L) == ("Rerun: failure, ended without STOP after 212 steps, "
                                                          "5.2 m from the goal.")
    o.update(ended_by="step_cap", steps=500, ne_m=float("nan"), oracle_success=True)
    assert fb.outcome_line(_bundle({"outcome": o}), L) == (
        "Rerun: failure, hit the 500-step limit at an unrecorded distance from the goal (it had been within the "
        "success radius earlier).")
    o.update(ended_by="stop", success=True, ne_m=0.4, steps=64)
    assert fb.outcome_line(_bundle({"outcome": o}), fb.LABELS["zh"]) == "复跑结果：成功，共 64 步，停下时距目标 0.4 m。"


def test_fidelity_uses_build_records_outcome_definition_and_counts_calls_from_one():
    """F1 in the log (os, STOP) but the rerun hits the step cap: success alone agrees, the outcome does not."""
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_behavior as fb

    ref = {"success": 0.0, "spl": 0.0, "os": 1.0, "ne": 3.4, "steps": 78, "ended_by": "stop"}
    rerun = {"success": False, "oracle_success": True, "ne_m": 4.0, "steps": 500, "ended_by": "step_cap"}
    fid = {"first_divergent_call": 6, "identical_calls": 6, "total_calls": 21}
    b = _bundle({"outcome": rerun, "eval_log_outcome": ref, "fidelity": fid})
    assert fb.fidelity_sentence(b, "en") == (
        "Rerun fidelity: 6 of 21 System2 calls identical to the seed-42 main-table evaluation, the first difference "
        "at the 7th call; different outcome (rerun: failure at the step limit after reaching the success radius; main "
        "table: failure ending with STOP after reaching the success radius).")
    assert "首个分歧在第 7 次调用；结局不同" in fb.fidelity_sentence(b, "zh")
    rerun.update(ended_by="stop", oracle_success=False)  # now only the oracle success differs
    assert "different outcome (rerun: failure ending with STOP; main table: failure ending with STOP after" \
        in fb.fidelity_sentence(b, "en")
    rerun.update(oracle_success=True)
    assert fb.fidelity_sentence(b, "en").endswith("the 7th call; same outcome.")
    assert [fb._ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 101, 111)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "101st", "111th"]

    br = pytest.importorskip("scripts.exp19.build_records")
    for s in (0.0, 1.0):
        for os_ in (0.0, 1.0):
            for end in ("stop", "step_cap", "other", None):
                r = dict(ref, success=s, os=os_, ended_by=end)
                for o in ({"success": bool(s), "oracle_success": True, "ended_by": "stop"},
                          {"success": False, "oracle_success": bool(os_), "ended_by": "step_cap"}):
                    assert fb.same_outcome(o, r) == br.compare_outcome(dict(o, steps=1), r)["same"], (o, r)


def _key(label, branch, index=0):
    return bd.KeyStep(index, {"label": label, "branch": branch}, {})


def test_key_rules_follow_the_recorded_branches():
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_behavior as fb

    full = [_key("K1", "K1_first", 0), _key("K2", "K2_turn", 1), _key("K3", "K3_two_thirds", 2),
            _key("K4", "K4_last", 3)]
    en = fb.key_rules_sentence(full, "en")
    assert en.startswith("K1 is the first ready call; K2 the other ready call whose executed action chunk has the "
                         "largest |net turn| (at least 30°); K3 ") and en.endswith("K4 the last ready call.")
    assert "净转角绝对值最大" in fb.key_rules_sentence(full, "zh")
    two = [_key("K1", "all_lt4", 0), _key("K2", "all_lt4", 1)]
    assert fb.key_rules_sentence(two, "en") == ("The rerun has only 2 ready calls, fewer than four, so all of them are "
                                                "drawn: K1, K2 in call order.")
    assert fb.key_rules_sentence(two[:1], "en") == "The rerun has only one ready call, drawn as K1."
    fallback = [_key("K1", "K1_first"), _key("K2", "K2_fallback"), _key("K3", "K3_f1_fallback_after"),
                _key("K4", "K4_shifted")]
    en = fb.key_rules_sentence(fallback, "en")
    assert "⌊(n−1)/3⌋" in en and "not in the pre-registration" in en and "not already chosen" in en

    # the main figure: the drawn labels, and what differs from the rule's main clause, per case
    b = bd.Bundle(Path("x"), {}, two)
    chosen = fb.main_keys_of(b, ("K2", "K3"))
    assert [k.label for k in chosen] == ["K1", "K2"]
    assert fb.main_exceptions(b, chosen, "F2", "en") == (" F2 has only 2 ready calls, so its key moments are all of "
                                                         "them in call order; K1, K2 shown.")
    b4 = bd.Bundle(Path("x"), {}, [_key("K1", "K1_first", 0), _key("K2", "K2_turn", 1),
                                   _key("K3", "K3_f1_fallback_after", 2), _key("K4", "K4_last", 3)])
    chosen = fb.main_keys_of(b4, ("K2", "K3"))
    assert [k.label for k in chosen] == ["K2", "K3"]
    assert fb.main_exceptions(b4, chosen, "F1", "en").startswith(" In F1, K3 is the first other ready call after")
    assert fb.main_exceptions(bd.Bundle(Path("x"), {}, []), [], "T3", "en") == (" T3 has no ready call, so no key "
                                                                                 "moment is shown.")
    assert fb.main_rules_sentence(("K2", "K3"), ("F1", "F2"), "en").endswith(
        "K3 the other ready call whose step is closest to 2/3 of the episode's steps (for F1: the last other ready "
        "call at or before the step closest to the goal).")


def test_every_branch_of_the_key_moment_rule_has_caption_text():
    """Branches as scripts/exp19/keysteps.py actually tags them, on inputs that reach each one."""
    keysteps = pytest.importorskip("scripts.exp19.keysteps")
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_behavior as fb

    def ready(steps, turns):
        return [{"call_index": i, "step": s, "executed_actions": [bd.LEFT] * t} for i, (s, t) in
                enumerate(zip(steps, turns))]

    seen = set()
    for r, cat, closest in ((ready([20, 30, 40], [0, 0, 0]), "T1", None),
                            (ready([20, 30, 40, 50, 60], [0, 3, 0, 0, 0]), "T1", None),
                            (ready([20, 30, 40, 50, 60], [0, 0, 0, 0, 0]), "F2", None),
                            (ready([20, 30, 40, 50, 60], [0, 0, 0, 0, 0]), "F1", 45),
                            (ready([20, 30, 40, 50], [0, 0, 0, 0]), "F1", 10),
                            (ready([20, 30, 40, 50], [0, 0, 0, 3]), "T1", None)):
        seen |= {k["branch"] for k in keysteps.select_key_steps(r, category=cat, episode_steps=70,
                                                                closest_step=closest)}
    assert seen == set(bd.KEY_BRANCHES)
    for lang in ("en", "zh"):
        assert set(bd.KEY_BRANCHES) <= set(fb.KEY_RULES[lang])


def test_floor_note_only_when_the_dotted_route_is_visible():
    pytest.importorskip("matplotlib")
    from types import SimpleNamespace

    from scripts.exp19.figures import fig_behavior as fb

    L = fb.LABELS["en"]
    scene = SimpleNamespace(level_index=lambda y: (np.asarray(y) > 1.5).astype(int))
    topdown = (scene, SimpleNamespace(index=0))
    turn_in_place = {"route_xz": [[0.0, 0.0]] * 3 + [[0.0, 0.25 * i] for i in range(1, 9)],
                     "route_y": [2.0, 2.0, 2.0] + [0.1] * 8}
    b = bd.Bundle(Path("x"), turn_in_place, [])
    assert fb.floor_note(b, topdown, L) == ""  # the off-floor steps are a turn in place: nothing dotted to see
    stairs = {"route_xz": [[0.0, 0.25 * i] for i in range(12)], "route_y": [0.1] * 8 + [0.6, 1.2, 1.8, 2.4]}
    assert fb.floor_note(bd.Bundle(Path("x"), stairs, []), topdown, L) == L["floor_note"]
    assert fb.floor_note(bd.Bundle(Path("x"), stairs, []), None, L) == L["floor_note_nomap"]
    assert fb.floor_note(b, None, L) == L["floor_note_nomap"]  # 1.9 m height range, no map to show it on
    flat = {"route_xz": stairs["route_xz"], "route_y": [0.1] * 12}
    assert fb.floor_note(bd.Bundle(Path("x"), flat, []), None, L) == ""


# --------------------------------------------------------------------------- #
# Smoke render
# --------------------------------------------------------------------------- #
def test_smoke_render_page_main_figure_and_manifest(synth, tmp_path):
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_behavior as fb

    out, path = synth
    assert fb.main(["--records", str(Path(path).parent), "--no-verdicts", "--out-dir", str(tmp_path),
                    "--lang", "en", "--main"]) == 0
    b = bd.load_bundle(path)
    stem = tmp_path / b.category / f"{b.category_rank}_{b.ep_key}"
    for suffix in ("_en.pdf", "_en.png", "_caption_en.txt"):
        assert (stem.parent / (stem.name + suffix)).stat().st_size > 0
    assert (tmp_path / "main" / "main_T_en.png").stat().st_size > 0  # the synthetic case is T1: successes figure
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == fb.MANIFEST_SCHEMA and manifest["claims"] == {"en": []}
    assert manifest["figures"][0]["synthetic"] is True
    (main_t,) = manifest["main"]["figures"]
    assert main_t["group"] == "T" and main_t["cases"][0]["ep_key"] == b.ep_key
    assert manifest["figures"][0]["size_in"]["en"][1] <= fb.PAGE_MAX_H_IN
    assert main_t["size_in"]["en"][1] <= fb.MAIN_MAX_H_IN and main_t["warnings"] == []
    assert not any("budget" in w for w in manifest["figures"][0]["warnings"])
    from PIL import Image

    with Image.open(stem.parent / (stem.name + "_en.png")) as im:
        assert im.size[0] == 2800  # 7.0 in at 400 dpi

    # --main-only keeps the pages' record in the manifest and replaces only "main"
    assert fb.main(["--records", str(Path(path).parent), "--no-verdicts", "--out-dir", str(tmp_path),
                    "--lang", "en", "--main-only"]) == 0
    again = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert again["figures"] == manifest["figures"] and again["main"]["figures"][0]["group"] == "T"


def test_void_batch_is_refused_or_stamped(synth, tmp_path):
    """A failed validity gate voids the batch (整批作废): no figures unless --allow-void, and then stamped."""
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_behavior as fb

    _, path = synth
    verdicts = {h: {"verdict": "void", "verdict_if_valid": "support", "void_reasons": ["trace-neutrality gate failed"]}
                for h in ("H1", "H2", "H3")}
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"verdicts": verdicts, "gates": {"trace_neutrality": {"pass": False}},
                                   "validity": {"valid": False, "reasons": ["trace-neutrality gate failed (9/10)"]}}))
    assert fb.load_metrics(metrics)[1] == ["trace-neutrality gate failed (9/10)"]
    args = ["--bundle", str(path), "--metrics", str(metrics), "--out-dir", str(tmp_path / "fig"), "--lang", "en"]
    assert fb.main(args) == 3 and not (tmp_path / "fig").exists()
    assert fb.main(args + ["--allow-void"]) == 0
    manifest = json.loads((tmp_path / "fig" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["void_batch"] == ["trace-neutrality gate failed (9/10)"] and manifest["claims"] == {"en": []}
    caption = Path(manifest["figures"][0]["files"][2]).read_text(encoding="utf-8")
    assert caption.startswith("VOID BATCH: a pre-registered validity condition failed (trace-neutrality gate failed")
    metrics.write_text(json.dumps({"verdicts": verdicts, "gates": {"trace_neutrality": {"pass": False}}}))
    assert fb.load_metrics(metrics)[1] == ["trace_neutrality gate failed"]  # no "validity": read the gates


def test_rerun_without_ready_call_gets_a_route_only_page(synth, tmp_path):
    """build_records.py writes key_steps = [] when a rerun has no ready call; every episode still gets its page."""
    pytest.importorskip("matplotlib")
    from scripts.exp19.figures import fig_behavior as fb

    _, path = synth
    meta, _ = _load_raw(path)
    meta["key_steps"] = []
    json_path = _write(tmp_path, meta, {}, name=f"{meta['ep_key']}_bundle")
    b = bd.load_bundle(json_path)
    assert b.keys == [] and "no ready call: page shows the route only" in b.warnings
    res = fb.make_episode_figure(json_path, tmp_path / "fig", lang="en", verdicts=None,
                                 topdown_root=meta["topdown"]["root"])
    assert all(Path(f).stat().st_size > 0 for f in res["files"])
    caption = Path(res["files"][2]).read_text(encoding="utf-8")
    assert "no ready System2 call" in caption and "key moments" not in caption

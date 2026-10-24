"""EXP-28 O4/O5 (scripts/exp28/hm3d_route_gate.py) and the verdict over O1-O5.

The records live in the flat synthetic world of ``test_s2_route_labels`` (geodesics are straight lines,
snapping is the identity, the lookdown depth is the analytic depth of the floor), so every judgement can
be worked out by hand.  No model and no simulator: ``decode`` is exercised by its dry run and by the
instruction swap it feeds to ``inference_messages``; the decoded rows of ``score`` are written here.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from scripts.exp28 import hm3d_route_gate as g
from scripts.tools import label_route_corrections as lr
from src.data.s2_oracle_labels import yaw_to_quat_wxyz
from src.data.s2_records import EpisodeRecord, EpisodeRecordWriter, label_path_for, read_labels
from src.data.s2_route_labels import RouteLabelConfig
from tests.test_s2_oracle_labels import FACE_NEG_Z, _floor_pixel, _jpeg, floor_depth
from tests.test_s2_route_labels import FlatWorld

REPO = Path(__file__).resolve().parents[1]
CONFIG = RouteLabelConfig()
DIGEST = CONFIG.digest()

# test_s2_route_labels._record's episode: the goal 8 m down -z; four calls --
#   0 at the start facing the route, turns right 4x  -> correction (turned away from the visible route)
#   1 yawed 100 deg right, turns right 4x            -> anchor (the hidden route: keep turning right)
#   2 yawed 160 deg right, walks on (pixel)          -> correction (no progress; target: keep turning right)
#   3 1.5 m from the goal                            -> line A's, no label
POSES = [([0.0, 0.0, 0.0], FACE_NEG_Z), ([0.0, 0.0, 0.0], yaw_to_quat_wxyz(math.radians(-100))),
         ([0.0, 0.0, 0.0], yaw_to_quat_wxyz(math.radians(-160))), ([0.0, 0.0, -6.5], FACE_NEG_Z)]
CALLS = [
    {"llm_output": "→→→→", "native_first_output": "→→→→", "native_lookdown_turns": 0, "kind": "native_actions"},
    {"llm_output": "→→→→", "native_first_output": "→→→→", "native_lookdown_turns": 0, "kind": "native_actions"},
    {"llm_output": "320 300", "native_first_output": "↓", "native_lookdown_turns": 1, "kind": "trajectory"},
    {"llm_output": "←", "native_first_output": "←", "native_lookdown_turns": 0, "kind": "native_actions"},
]
KINDS = {0: "correction", 1: "anchor", 2: "correction"}
AHEAD = ["↓", "%d %d" % _floor_pixel(0.0, -2.0)]  # 2 m straight ahead on the floor
FLAT_NAVMESH = b"flat navmesh"  # the bytes of the score tests' navmesh file (``flat_navmesh``)
FLAT_SHA = hashlib.sha256(FLAT_NAVMESH).hexdigest()


def _record(root, scene, episode, instruction="walk ahead", **episode_extra):
    """An EXP-28 client record: it names the sha256 of the navmesh it ran on (None: an older client's)."""
    episode_extra.setdefault("navmesh_sha256", FLAT_SHA)
    episode_extra = {k: v for k, v in episode_extra.items() if v is not None}
    reference = [[0.0, 0.0, 0.0], [0.0, 0.0, -4.0], [0.0, 0.0, -8.0]]
    writer = EpisodeRecordWriter(
        root=root, scene_id=scene, episode_id=episode,
        episode={"reference_path": reference, "goals": [{"position": reference[-1], "radius": 3.0}],
                 "instruction": instruction, "outcome": {"success": 1.0}, **episode_extra},
    )
    for step, (position, rotation) in enumerate(POSES):
        writer.add_step(step, position, rotation)
        writer.add_front(step, _jpeg((64, 48)))
    for index, call in enumerate(CALLS):
        position, rotation = POSES[index]
        writer.add_call({"call_index": index, "step_id": index, "history_step_ids": [], **call},
                        lookdown_jpeg=_jpeg(), lookdown_depth_m=floor_depth(position, rotation))
    return writer.commit()


def _label(path, label_root, config=CONFIG):
    return lr.label_record(path, label_root=label_root, geometry_for_scene=lambda scene: FlatWorld(),
                           dataset_index=None, config=config, force=True)


def _world(tmp_path, *, held=("flatA",), train=("trainScene",), per_scene=3):
    """Records of held-out and training scenes, every one labelled; returns (record root, label root)."""
    records, labels = tmp_path / "records", tmp_path / "labels"
    for scene in held + train:
        for episode in range(1, per_scene + 1):
            _label(_record(records, scene, episode, instruction=f"walk ahead to door {scene} {episode}"), labels)
    return records, labels


def _select(records, labels, **kwargs):
    kwargs.setdefault("scenes", {"flatA"})
    return g.select_states([records], labels, config_digest=DIGEST, **kwargs)


# --------------------------------------------------------------------------- #
# Scenes and the O5 shuffle
# --------------------------------------------------------------------------- #
def test_scene_keys_accept_hm3d_directories_and_ids(tmp_path):
    assert g.scene_key("00481-5jLhtVmWd5F") == g.scene_key(" 5jLhtVmWd5F ") == "5jLhtVmWd5F"
    assert g.scene_key("17DRP5sb8fy") == "17DRP5sb8fy" and g.scene_key("a-b") == "a-b"
    listing = tmp_path / "heldout.txt"
    listing.write_text("# B3\n00481-5jLhtVmWd5F\n\n00393-dD37vuCa4FE  # another\nNBg5UqG3di3\n")
    assert g.holdout_scene_set(g.read_scene_list(listing)) == {"5jLhtVmWd5F", "dD37vuCa4FE", "NBg5UqG3di3"}


def _pool(scene, n, text=lambda i: f"instruction {i}"):
    return [{"key": f"{scene}_{i:06d}", "scene_id": scene, "instruction": text(i)} for i in range(n)]


def test_shuffle_gives_every_episode_another_instruction_of_its_own_scene():
    pool = _pool("A", 7) + _pool("B", 1) + _pool("C", 3, text=lambda i: "Same  words") + _pool("D", 2)
    donors = g.shuffle_donors(pool, seed=42)
    a = {k: v for k, v in donors.items() if k.startswith("A_")}
    assert all(v["donor"].startswith("A_") and v["donor"] != k for k, v in a.items())
    assert sorted(v["donor"] for v in a.values()) == sorted(a)  # a permutation: everyone gives once
    assert all(v["instruction"] == next(p["instruction"] for p in pool if p["key"] == v["donor"]) for v in a.values())
    assert donors["B_000000"] == {"donor": None, "instruction": None, "why": "single_record_in_scene"}
    assert all(v["why"] == "no_other_instruction_in_scene" for k, v in donors.items() if k.startswith("C_"))
    assert donors["D_000000"]["donor"] == "D_000001" and donors["D_000001"]["donor"] == "D_000000"
    assert g.shuffle_donors(list(reversed(pool)), seed=42) == donors
    assert any(g.shuffle_donors(pool, seed=s) != donors for s in (1, 2, 3))


def test_a_donor_with_the_same_words_is_passed_over():
    pool = [{"key": "A_0", "scene_id": "A", "instruction": "Go left."},
            {"key": "A_1", "scene_id": "A", "instruction": "go   LEFT."},
            {"key": "A_2", "scene_id": "A", "instruction": "Go right."}]
    donors = g.shuffle_donors(pool, seed=0)
    assert donors["A_0"]["instruction"] == donors["A_1"]["instruction"] == "Go right."
    assert donors["A_2"]["instruction"].lower().startswith("go")


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def test_selection_keeps_held_out_states_beyond_three_metres_within_the_window(tmp_path):
    records, labels = _world(tmp_path)
    states, meta = _select(records, labels)
    assert {s["scene_id"] for s in states} == {"flatA"}  # trainScene never enters
    assert len(states) == 9 and meta["episodes_selected"] == meta["records_in_pool"] == 3
    assert {(s["call_index"], s["label_kind"]) for s in states} == set(KINDS.items())
    assert meta["stats"]["calls_within_radius_or_skipped"] == 3  # call 3 of each episode
    first = next(s for s in states if s["key"] == "flatA_000001#0")
    assert first["recorded"] == ["→→→→"] and first["label_agree"] is False and first["distance_to_goal"] == 8.0
    assert first["shuffle"]["donor"].startswith("flatA_") and first["shuffle"]["donor"] != "flatA_000001"
    assert "shuffle" not in next(s for s in states if s["label_kind"] == "anchor")
    pixel = next(s for s in states if s["call_index"] == 2)
    assert pixel["recorded"] == ["↓", "320 300"] and pixel["previous_turn_direction"] == -1
    assert pixel["label_target"] == ["→→→→"]

    window, meta = _select(records, labels, max_call_index=2)
    assert {s["call_index"] for s in window} == {0, 1} and meta["stats"]["calls_beyond_window"] == 6
    few, meta = _select(records, labels, max_episodes=1, seed=3)
    assert meta["episodes_selected"] == 1 and len({s["record_key"] for s in few}) == 1
    # a subset does not move anyone's shuffled instruction (the pool is every held-out record)
    full = {s["key"]: s.get("shuffle") for s in states}
    assert all(full[s["key"]] == s.get("shuffle") for s in few)
    with pytest.raises(ValueError, match="no held-out scenes"):
        _select(records, labels, scenes=set())
    # a listed held-out scene without records is named (the verdict carries it as a note)
    _, meta = _select(records, labels, scenes={"flatA", "noRecords"})
    assert meta["scenes_without_records"] == ["noRecords"] and meta["scenes_with_records"] == ["flatA"]


def test_o4_reads_only_the_base_models_records(tmp_path):
    base = {"model_version": "heatmapvln-r2r", "supported_formats": ["heatmapvln-device:npu"]}
    assert g.served_adapter({"server": base}) is None and g.served_adapter({}) is None
    tagged = {"model_version": "heatmapvln-r2r+system2-lora:" + "c" * 64, "supported_formats": []}
    assert g.served_adapter({"server": tagged}) == "c" * 64
    assert g.served_adapter({"server": {"supported_formats": ["system2-lora-v1", "system2-lora-sha256:" + "d" * 64]}}) \
        == "d" * 64
    records, labels = _world(tmp_path, train=())
    _label(_record(records, "flatA", 7, instruction="served by C1", server=tagged), labels)  # e.g. proxy P's arm
    with pytest.raises(ValueError, match="O4 reads the base model's records"):
        _select(records, labels)


def test_a_record_without_labels_stops_the_selection(tmp_path):
    records, labels = _world(tmp_path, train=())
    _record(records, "flatA", 9, instruction="unlabelled")  # committed after the labeller ran
    with pytest.raises(ValueError, match="no line-B labels"):
        _select(records, labels)


def test_selection_refuses_labels_of_other_rules_or_records(tmp_path):
    records, labels = _world(tmp_path, train=())
    path = sorted(records.glob("flatA/*.s2rec.tar"))[0]
    _label(path, labels, config=RouteLabelConfig(pixel_max_ahead_m=2.5))
    with pytest.raises(ValueError, match="labelled by"):
        _select(records, labels)
    _label(path, labels)
    label_file = label_path_for(path, labels)
    rows = read_labels(label_file)
    rows[0]["record_sha256"] = "0" * 64
    label_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="another version"):
        _select(records, labels)


# --------------------------------------------------------------------------- #
# Line B's judgement in a recorded state
# --------------------------------------------------------------------------- #
def test_contexts_reproduce_the_line_c_labeller(tmp_path):
    line_c = lr.RL.RouteLabelConfig(terminal_stop_radius_m=1.0)
    record = EpisodeRecord(_record(tmp_path / "records", "flatA", 1))
    rows = lr.label_record_rows(record, FlatWorld(), dataset_entry=None, config=line_c)
    contexts = g.call_contexts(record, FlatWorld(), dataset_entry=None, config=line_c)
    for row in rows:
        recorded = [row["model_turn1"]] + ([row["model_turn2"]] if row["model_turn2"] is not None else [])
        judged = g.judge(contexts[row["call_index"]], recorded)
        assert (judged["agree"], judged["kind"], judged["reason"]) == (row["agree"], row["kind"], row["reason"])


def test_contexts_reproduce_the_labeller_and_judge_any_answer(tmp_path):
    record = EpisodeRecord(_record(tmp_path / "records", "flatA", 1))
    rows = lr.label_record_rows(record, FlatWorld(), dataset_entry=None, config=CONFIG)
    contexts = g.call_contexts(record, FlatWorld(), dataset_entry=None, config=CONFIG)
    for row in rows:
        recorded = [row["model_turn1"]] + ([row["model_turn2"]] if row["model_turn2"] is not None else [])
        judged = g.judge(contexts[row["call_index"]], recorded)
        assert (judged["agree"], judged["kind"], judged["reason"]) == (row["agree"], row["kind"], row["reason"])
    only = g.call_contexts(record, FlatWorld(), dataset_entry=None, config=CONFIG, only=[2])
    assert set(only) == {2}
    # the previous executed turn still comes from call 1 (right): the hidden route keeps turning right
    assert g.judge(only[2], ["→"])["reason"] == "turn_matches_rule"
    assert g.judge(only[2], ["←"])["agree"] is True  # the route is 160 deg behind: either way
    # call 0 faces the route: a pixel 2 m ahead makes progress, a STOP 8 m out never agrees
    assert g.judge(contexts[0], AHEAD) == {"agree": True, "kind": "anchor", "reason": "model_pixel_progress",
                                           "model_kind": "pixel"}
    assert g.judge(contexts[0], ["STOP"])["reason"] == "model_stop_beyond_radius"
    assert g.judge(contexts[0], ["↑↑"])["agree"] is False


def test_answers_take_the_recorded_call_shape():
    assert g.answer_call(["↓", "12 34"]) == {"llm_output": "12 34", "native_first_output": "↓",
                                             "native_lookdown_turns": 1}
    assert g.answer_call(["←←"])["native_lookdown_turns"] == 0
    with pytest.raises(ValueError):
        g.answer_call([])


# --------------------------------------------------------------------------- #
# O4 and O5 from scored rows
# --------------------------------------------------------------------------- #
def _arm(cls, agree=None):
    turns = {"stop": ["STOP"], "pixel": ["↓", "300 300"], "turn_left": ["←"], "turn_right": ["→"]}[cls]
    return {"turns": turns, "class": cls, "agree": agree}


def _rows(*, corrections, on_agree, off_agree, anchors=0, kept=0, extra=0, on_stops=0, off_stops=0,
          shuffled=None, no_donor=0):
    """``corrections`` rows (the first ``on_agree`` / ``off_agree`` agreeing), ``anchors`` (``kept`` keep their
    class), ``extra`` 'none' rows; the first ``on_stops`` / ``off_stops`` of all rows answer STOP (a STOP never
    agrees, so stops go to rows after the agreeing ones).  ``shuffled``: (on, off) agreeing counts."""
    rows = []
    for i in range(corrections):
        row = {"label_kind": "correction", "record_key": f"s_{i % 4:06d}", "scene_id": "s", "distance_to_goal": 7.0,
               "recorded_model_kind": "turn", "previous_turn_direction": 0, "label_target": ["←"],
               "on": _arm("pixel", i < on_agree), "off": _arm("pixel", i < off_agree), "recorded": ["→"],
               "shuffle": {"donor": None if i < no_donor else "s_x"}, "shuffled": None}
        if shuffled is not None and i >= no_donor:
            j = i - no_donor
            row["shuffled"] = {"donor": "s_x", "on": _arm("pixel", j < shuffled[0]), "off": _arm("pixel", j < shuffled[1])}
        rows.append(row)
    for i in range(anchors):
        rows.append({"label_kind": "anchor", "record_key": "s_000009", "scene_id": "s", "distance_to_goal": 4.0,
                     "on": _arm("pixel" if i < kept else "turn_left", True), "off": _arm("pixel", True),
                     "recorded": ["↓", "300 300"]})
    for _ in range(extra):
        rows.append({"label_kind": "none", "record_key": "s_000010", "scene_id": "s", "distance_to_goal": 9.0,
                     "on": _arm("turn_right"), "off": _arm("turn_right"), "recorded": ["→"]})
    for arm, count in (("on", on_stops), ("off", off_stops)):
        for row in rows[::-1][:count]:
            row[arm] = _arm("stop", False)
    return rows


def test_o4_lines_are_exact_and_inclusive():
    result = g.summarize(_rows(corrections=20, on_agree=4, off_agree=1, anchors=25, kept=23, extra=5,
                               on_stops=3, off_stops=2), shuffle_requested=False)
    c = result["O4"]["criteria"]
    assert c["correction_agree_gain"]["value"] == 0.15 and c["correction_agree_gain"]["pass"]
    assert c["correction_agree_gain"]["on_rate"] == 0.2 and c["correction_agree_gain"]["base_rate"] == 0.05
    assert c["anchor_class_keep"]["value"] == 0.92 and c["anchor_class_keep"]["pass"]
    assert c["far_stop_vs_base"]["pass"] and c["far_stop_vs_base"]["n"] == 50
    assert c["far_stop_vs_base"]["threshold"] == 0.06 and result["O4"]["all_pass"]
    assert not result["O5"]["all_pass"] and "without --shuffle-instructions" in \
        result["O5"]["criteria"]["shuffle_vs_own_instruction"]["why"]

    fail = g.summarize(_rows(corrections=20, on_agree=3, off_agree=1, anchors=25, kept=22, extra=5,
                             on_stops=4, off_stops=2), shuffle_requested=False)["O4"]["criteria"]
    assert not any(v["pass"] for v in fail.values())
    # the base never stops beyond 3 m: the line is zero, and one adapter STOP fails it
    assert g.summarize(_rows(corrections=4, on_agree=4, off_agree=0, extra=3), shuffle_requested=False)[
        "O4"]["criteria"]["far_stop_vs_base"]["pass"]
    assert not g.summarize(_rows(corrections=4, on_agree=3, off_agree=0, extra=3, on_stops=1),
                           shuffle_requested=False)["O4"]["criteria"]["far_stop_vs_base"]["pass"]
    # no anchors or no corrections: unmeasured, a FAIL
    empty = g.summarize([], shuffle_requested=True)
    assert not empty["O4"]["all_pass"] and not empty["O5"]["all_pass"]


def test_o5_compares_the_two_gains_on_the_same_corrections():
    # 20 corrections with a donor: gain with the own words 10/20, so the line is 3/20 under shuffled words
    ok = g.summarize(_rows(corrections=20, on_agree=11, off_agree=1, shuffled=(5, 2)), shuffle_requested=True)
    o5 = ok["O5"]["criteria"]["shuffle_vs_own_instruction"]
    assert (o5["value"], o5["threshold"], o5["delta_corr"], o5["pass"]) == (0.15, 0.15, 0.5, True)
    assert o5["ratio"] == 0.3 and o5["base_rate_shuffled"] == 0.1 and o5["delta_shuf_vs_base_own_instruction"] == 0.2
    bad = g.summarize(_rows(corrections=20, on_agree=11, off_agree=1, shuffled=(6, 2)), shuffle_requested=True)
    assert not bad["O5"]["all_pass"]
    # corrections without a donor are not part of O5's batch
    # (rows 0-3 have none; among rows 4-23 the adapter agrees on 4-14, the base on 4: 10/20 again)
    part = g.summarize(_rows(corrections=24, on_agree=15, off_agree=5, shuffled=(5, 2), no_donor=4),
                       shuffle_requested=True)["O5"]["criteria"]["shuffle_vs_own_instruction"]
    assert part["n"] == 20 and part["corrections_without_donor"] == 4 and part["delta_corr"] == 0.5
    assert part["pass"] and part["value"] == 0.15
    # a correction with a donor but no shuffled answer: O5 was not measured on the whole batch
    rows = _rows(corrections=20, on_agree=11, off_agree=1, shuffled=(5, 2))
    rows[-1]["shuffled"] = None
    partial = g.summarize(rows, shuffle_requested=True)["O5"]["criteria"]["shuffle_vs_own_instruction"]
    assert not partial["pass"] and partial["corrections_not_decoded_shuffled"] == 1
    # nothing learnt: the ratio is degenerate, and the verdict says so
    flat = g.summarize(_rows(corrections=20, on_agree=1, off_agree=1, shuffled=(1, 1)), shuffle_requested=True)
    assert flat["O5"]["criteria"]["shuffle_vs_own_instruction"]["delta_corr_nonpositive"]


def test_report_only_subsets():
    rows = _rows(corrections=8, on_agree=4, off_agree=0)
    rows[0].update(recorded_model_kind="stop", distance_to_goal=4.0)  # R2: no training sample
    rows[1].update(previous_turn_direction=-1, label_target=["←←"])    # R6: target reverses the executed right turn
    rows[2].update(previous_turn_direction=1, label_target=["←←"])     # same direction: kept
    report = g.summarize(rows, shuffle_requested=False)["report_only"]
    assert report["corrections_training_eligible_r2_r6"]["n"] == 6
    assert set(report["corrections_by_recorded_kind"]) == {"stop", "turn"}
    assert report["corrections_by_distance"]["3-6"]["n"] == 1


# --------------------------------------------------------------------------- #
# score, end to end on the flat world
# --------------------------------------------------------------------------- #
def _decoded(states, *, on=None, adapter="a" * 64, shuffled_on=None):
    """Decode rows: the base replays the recorded answer; the adapter answers ``on(state)``."""
    rows = []
    for s in states:
        turns = on(s) if on else s["recorded"]
        row = {"key": s["key"], "adapter_sha256": adapter, "device": "cuda", "record_device": "npu",
               "on": g._arm(turns), "off": g._arm(s["recorded"])}
        if g._needs_shuffle(s) and shuffled_on is not None:
            row["shuffled"] = {"donor": s["shuffle"]["donor"], "on": g._arm(shuffled_on(s)), "off": g._arm(s["recorded"])}
        rows.append(row)
    return rows


def _write_decoded(output, rows, shard="00of01"):
    path = output.with_name(f"{output.stem}.decode.shard{shard}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _score_argv(records, labels, scenes_dir, output):
    return ["score", "--record-root", str(records), "--label-root", str(labels), "--scenes", "flatA",
            "--scenes-dir", str(scenes_dir), "--output", str(output)]


@pytest.fixture()
def flat_navmesh(tmp_path, monkeypatch):
    scenes_dir = tmp_path / "scenes"
    (scenes_dir / "flatA").mkdir(parents=True)
    (scenes_dir / "flatA" / "flatA.navmesh").write_bytes(FLAT_NAVMESH)
    monkeypatch.setattr(g.lr, "RouteNavmeshGeometry", lambda path: FlatWorld())
    return scenes_dir


def test_score_judges_every_answer_and_passes_a_route_follower(tmp_path, flat_navmesh):
    records, labels = _world(tmp_path)
    states, _ = _select(records, labels)
    output = tmp_path / "o4" / "hm3d.json"

    def follower(state):  # walks on where the base turned away; keeps the anchors' answers
        return AHEAD if state["call_index"] == 0 else ["→"] if state["call_index"] == 2 else state["recorded"]

    _write_decoded(output, _decoded(states, on=follower, shuffled_on=lambda s: s["recorded"]))
    assert g.main(_score_argv(records, labels, flat_navmesh, output)) == 0
    report = json.loads(output.read_text())
    c = report["O4"]["criteria"]
    assert report["valid"] and c["correction_agree_gain"]["value"] == 1.0
    assert c["correction_agree_gain"]["base_rate"] == 0.0 and c["anchor_class_keep"]["value"] == 1.0
    o5 = report["O5"]["criteria"]["shuffle_vs_own_instruction"]
    assert o5["n"] == 6 and o5["value"] == 0.0 and o5["pass"]
    rows = [json.loads(line) for line in output.with_suffix(".calls.jsonl").read_text().splitlines()]
    assert len(rows) == 9
    zero = next(r for r in rows if r["key"] == "flatA_000001#0")
    assert zero["on"]["agree"] is True and zero["on"]["judge_reason"] == "model_pixel_progress"
    assert zero["off"]["agree"] is False and zero["shuffled"]["off"]["agree"] is False

    # the same follower under every episode's shuffled words too: a route prior, O5 fails
    _write_decoded(output, _decoded(states, on=follower, shuffled_on=follower))
    assert g.main(_score_argv(records, labels, flat_navmesh, output)) == 1
    assert not json.loads(output.read_text())["O5"]["all_pass"]


def test_score_refuses_what_it_cannot_measure(tmp_path, flat_navmesh):
    records, labels = _world(tmp_path)
    states, _ = _select(records, labels)
    output = tmp_path / "o4" / "hm3d.json"
    shard = _write_decoded(output, _decoded(states[:-1]))
    assert g.main(_score_argv(records, labels, flat_navmesh, output)) == 2
    assert "not decoded" in json.loads(output.read_text())["invalid"]

    rows = _decoded(states)
    rows[0]["adapter_sha256"] = "b" * 64
    shard.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert g.main(_score_argv(records, labels, flat_navmesh, output)) == 2
    assert "2 adapters" in json.loads(output.read_text())["invalid"]

    shard.write_text("".join(json.dumps(r) + "\n" for r in _decoded(states)))
    path = sorted(records.glob("flatA/*.s2rec.tar"))[0]
    label_file = label_path_for(path, labels)
    rows = read_labels(label_file)
    rows[0].update(kind="anchor", agree=True)  # not what line B says about the recorded answer
    label_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert g.main(_score_argv(records, labels, flat_navmesh, output)) == 2
    assert "does not reproduce" in json.loads(output.read_text())["invalid"]


def test_score_without_a_complete_o5_is_invalid_not_a_fail(tmp_path, flat_navmesh):
    records, labels = _world(tmp_path)
    states, _ = _select(records, labels)
    output = tmp_path / "o4" / "hm3d.json"
    # decoded without --shuffle-instructions: O4 is there to read, but the pass is not (a FAIL here would kill)
    shard = _write_decoded(output, _decoded(states))
    assert g.main(_score_argv(records, labels, flat_navmesh, output)) == 2
    report = json.loads(output.read_text())
    assert not report["valid"] and "O5 not measured" in report["invalid"]
    assert report["O4"]["criteria"]["correction_agree_gain"]["n"] == 6
    assert len(output.with_suffix(".calls.jsonl").read_text().splitlines()) == 9  # rows kept for a look
    # one correction's shuffled answer missing
    rows = _decoded(states, shuffled_on=lambda s: s["recorded"])
    victim = next(r for r in rows if "shuffled" in r)
    del victim["shuffled"]
    shard.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert g.main(_score_argv(records, labels, flat_navmesh, output)) == 2
    assert "1 of them have no shuffled answer" in json.loads(output.read_text())["invalid"]
    # a shuffled answer decoded under another donor (the held-out pool changed after decoding)
    rows = _decoded(states, shuffled_on=lambda s: s["recorded"])
    victim = next(r for r in rows if "shuffled" in r)
    victim["shuffled"]["donor"] = "flatA_999999"
    shard.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert g.main(_score_argv(records, labels, flat_navmesh, output)) == 2
    assert "another donor" in json.loads(output.read_text())["invalid"]
    # ... which a resumed decode does not count as done
    state = next(s for s in states if s["key"] == victim["key"])
    assert not g._complete(victim, state, True) and g._complete(victim, state, False)
    assert g._complete(next(r for r in _decoded(states, shuffled_on=lambda s: s["recorded"])
                            if r["key"] == victim["key"]), state, True)


def test_measurement_problems_name_what_is_missing():
    assert g.measurement_problems(g.summarize(_rows(corrections=20, on_agree=11, off_agree=1, anchors=3, kept=3,
                                                    shuffled=(5, 2)), shuffle_requested=True)) == []
    problems = g.measurement_problems(g.summarize([], shuffle_requested=True))
    assert problems[:2] == ["O4: no correction state", "O4: no anchor state"] and "O5 not measured" in problems[2]


def test_score_refuses_a_record_collected_on_another_navmesh(tmp_path, flat_navmesh):
    records, labels = tmp_path / "records", tmp_path / "labels"
    for episode in (1, 2):
        _label(_record(records, "flatA", episode, instruction=f"go {episode}", navmesh_sha256="f" * 64), labels)
    states, _ = _select(records, labels)
    output = tmp_path / "o4" / "hm3d.json"
    _write_decoded(output, _decoded(states))
    assert g.main(_score_argv(records, labels, flat_navmesh, output)) == 2
    assert "navmesh" in json.loads(output.read_text())["invalid"]
    # a record that names no navmesh was not collected by EXP-28's client: nothing ties it to this one
    records, labels = tmp_path / "records2", tmp_path / "labels2"
    for episode in (1, 2):
        _label(_record(records, "flatA", episode, instruction=f"go {episode}", navmesh_sha256=None), labels)
    states, _ = _select(records, labels)
    _write_decoded(output, _decoded(states, shuffled_on=lambda s: s["recorded"]))
    assert g.main(_score_argv(records, labels, flat_navmesh, output)) == 2
    assert "no navmesh_sha256" in json.loads(output.read_text())["invalid"]


# --------------------------------------------------------------------------- #
# decode without a model
# --------------------------------------------------------------------------- #
def test_shards_partition_the_states_and_the_dry_run_counts(tmp_path, capsys):
    keys = [f"s_{i:06d}#{j}" for i in range(30) for j in range(5)]
    parts = [{k for k in keys if g.shard_of(k, 3) == i} for i in range(3)]
    assert set().union(*parts) == set(keys) and sum(map(len, parts)) == len(keys) and all(parts)
    assert g.parse_shard("2/3") == (2, 3)
    with pytest.raises(ValueError):
        g.parse_shard("3/3")
    records, labels = _world(tmp_path)
    argv = ["decode", "--record-root", str(records), "--label-root", str(labels), "--scenes", "00012-flatA",
            "--output", str(tmp_path / "o4" / "hm3d.json"), "--adapter", "none", "--model-path", "none",
            "--shard", "1/2", "--dry-run"]
    assert g.main(argv) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["states"] == 9 and out["corrections_shuffle"] == {"donor": 6}


def test_the_shuffled_prompt_changes_only_the_instruction(tmp_path):
    pytest.importorskip("vla_rpc.core.image")
    from src.data.s2_sft_dataset import inference_messages

    record = EpisodeRecord(_record(tmp_path / "records", "flatA", 1, instruction="Walk past the sofa.",
                                   vlm_image_size=[384, 384]))
    call = record.calls[2]
    messages, images, lookdown = inference_messages(record, call)
    swapped = g.WithInstruction(record, "Turn left at the stairs.")
    messages2, images2, lookdown2 = inference_messages(swapped, call)
    text, text2 = messages[0]["content"][0]["text"], messages2[0]["content"][0]["text"]
    assert "Walk past the sofa." in text and "Turn left at the stairs." in text2 and "sofa" not in text2
    assert text2.replace("Turn left at the stairs.", "Walk past the sofa.") == text
    for a, b in zip(images + [lookdown], images2 + [lookdown2]):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert record.episode["instruction"] == "Walk past the sofa."  # the record itself is untouched


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def _agreement(n, hits, prefix="k"):
    return {f"{prefix}{i}": i < hits for i in range(n)}


def test_o1_o2_compare_the_same_samples_down_to_one_point():
    at_line = g.compare_agreement(_agreement(1000, 849), _agreement(1000, 859), reference=g.O1_REFERENCE)
    assert at_line["pass"] and at_line["delta"] == -0.01 and at_line["base_reproduces_ledger"]
    below = g.compare_agreement(_agreement(1000, 848), _agreement(1000, 859), reference=g.O1_REFERENCE)
    assert not below["pass"] and below["threshold"] == 0.849
    other = g.compare_agreement(_agreement(10, 10, "x"), _agreement(10, 9), reference=g.O2_REFERENCE)
    assert not other["pass"] and not other["same_samples"] and other["only_candidate"] == 10
    off_ledger = g.compare_agreement(_agreement(100, 90), _agreement(100, 90), reference=g.O2_REFERENCE)
    assert off_ledger["pass"] and not off_ledger["base_reproduces_ledger"]


@pytest.mark.parametrize("failed, attempt, step", [
    ((), 1, "val"),
    ((), 2, "val"),
    (("O4_correction",), 1, "retrain_2_epochs"),
    (("O1",), 1, "scale_0.5"),
    (("O4_stop",), 1, "scale_0.5"),
    (("O3", "O4_correction"), 1, "scale_0.5"),
    (("O5",), 1, "kill"),
    (("O5", "O1"), 1, "kill"),
    (("O4_correction", "O5"), 1, "kill"),
    (("O2",), 2, "kill"),
    (("O4_correction",), 2, "kill"),
])
def test_next_step_follows_the_pre_registered_retry(failed, attempt, step):
    groups = {name: name not in failed for name in g.GROUPS}
    assert g.next_step(groups, attempt) == step


def _o3(passing=True, *, rows=2000, joined=None):
    """``stratified_gates.py --gate exp28-o3``'s output: the block and the join of its rows to their records."""
    return {"O3_exp28": {"criteria": {"f_le1_stop_keep": {"pass": passing}}, "all_pass": passing},
            "inputs": {"rows_in_file": rows, "join": {"rows": rows if joined is None else joined}}}


def _o4(**fail):
    criteria = {k: {"pass": not fail.get(k)} for k in ("correction_agree_gain", "anchor_class_keep", "far_stop_vs_base")}
    o5 = {"pass": not fail.get("o5"), "delta_corr_nonpositive": bool(fail.get("degenerate"))}
    return {"valid": True, "O4": {"criteria": criteria}, "O5": {"criteria": {"shuffle_vs_own_instruction": o5}}}


def test_verdict_groups_notes_and_invalid_measurements():
    ok = g.compare_agreement(_agreement(1000, 859), _agreement(1000, 859), reference=0.859)
    result = g.verdict(o1=ok, o2=ok, o3=_o3(), o4=_o4(), attempt=1)
    assert result["all_pass"] and result["next"] == "val" and result["notes"] == []
    result = g.verdict(o1=ok, o2=ok, o3=_o3(), o4=_o4(correction_agree_gain=True, o5=True, degenerate=True), attempt=1)
    assert result["next"] == "kill" and any("delta_corr <= 0" in n for n in result["notes"])
    assert g.verdict(o1=ok, o2=ok, o3=_o3(False), o4=_o4(), attempt=1)["next"] == "scale_0.5"
    invalid = g.verdict(o1=ok, o2=ok, o3=_o3(), o4={"valid": False, "invalid": "GateInvalid: 3 states"}, attempt=1)
    assert invalid["next"] == "invalid" and not invalid["measured"] and not invalid["all_pass"]
    unpaired = g.compare_agreement(_agreement(10, 10, "x"), _agreement(10, 9), reference=0.975)
    assert g.verdict(o1=ok, o2=unpaired, o3=_o3(), o4=_o4(), attempt=1)["next"] == "invalid"
    # nothing to compare is not a FAIL (which would spend the retry): invalid
    empty = g.compare_agreement({}, {}, reference=0.975)
    assert g.verdict(o1=ok, o2=empty, o3=_o3(), o4=_o4(), attempt=1)["next"] == "invalid"
    # O3 rows that did not join their records were dropped by stratified_gates.py
    dropped = g.verdict(o1=ok, o2=ok, o3=_o3(rows=2000, joined=1990), o4=_o4(), attempt=1)
    assert dropped["next"] == "invalid" and any("1990 of 2000" in n for n in dropped["notes"])
    assert g.verdict(o1=ok, o2=ok, o3=_o3(rows=0), o4=_o4(), attempt=1)["next"] == "invalid"
    # a smoke subset of the held-out episodes never reads as a gate pass
    smoke = {**_o4(), "selection": {"settings": {"max_episodes": 5}}}
    result = g.verdict(o1=ok, o2=ok, o3=_o3(), o4=smoke, attempt=1)
    assert result["next"] == "invalid" and not result["all_pass"] and any("smoke" in n for n in result["notes"])
    # held-out scenes without base records: reported, still a gate
    thin = {**_o4(), "selection": {"settings": {"max_episodes": 0}, "scenes_without_records": ["X"]}}
    result = g.verdict(o1=ok, o2=ok, o3=_o3(), o4=thin, attempt=1)
    assert result["next"] == "val" and any("no base records" in n for n in result["notes"])
    with pytest.raises(ValueError, match="O3_exp28"):
        g.verdict(o1=ok, o2=ok, o3={"G_A": {}}, o4=_o4(), attempt=1)


def _teacher_forced(path, hits, n=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": {"overall_rate": hits / n}}))
    rows = [{"key": f"k{i}", "class_match": i < hits} for i in range(n)]
    path.with_suffix(".calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_verdict_cli_reads_the_tools_outputs(tmp_path, capsys):
    paths = {
        "o1": _teacher_forced(tmp_path / "o1" / "scalevln.json", 85),
        "o1-base": _teacher_forced(tmp_path / "base" / "o1_scalevln.json", 86),
        "o2": _teacher_forced(tmp_path / "o2" / "r2r.json", 96),
        "o2-base": _teacher_forced(tmp_path / "base" / "o2_r2r.json", 98),
    }
    (tmp_path / "o3.json").write_text(json.dumps(_o3()))
    (tmp_path / "o4.json").write_text(json.dumps(_o4()))
    argv = ["verdict", "--o3", str(tmp_path / "o3.json"), "--o4", str(tmp_path / "o4.json"),
            "--output", str(tmp_path / "verdict.json")]
    for name, path in paths.items():
        argv += [f"--{name}", str(path)]
    assert g.main(argv) == 1  # O2 dropped 2 pt
    result = json.loads((tmp_path / "verdict.json").read_text())
    assert result["groups"]["O1"] and not result["groups"]["O2"] and result["next"] == "scale_0.5"
    assert capsys.readouterr().out.startswith("FAIL EXP-28 offline gates attempt 1:")
    assert g.main(argv + ["--attempt", "2"]) == 1
    assert json.loads((tmp_path / "verdict.json").read_text())["next"] == "kill"
    capsys.readouterr()
    # an O3 from before (f) existed, or a missing rows file: nothing judged, exit 2 (never the FAIL of 1)
    (tmp_path / "o3.json").write_text(json.dumps({"G_A": {"criteria": {}, "all_pass": True}}))
    assert g.main(argv) == 2 and capsys.readouterr().out.startswith("INVALID EXP-28 offline gates attempt 1:")
    (tmp_path / "o3.json").write_text(json.dumps(_o3()))
    paths["o2-base"].with_suffix(".calls.jsonl").unlink()
    assert g.main(argv) == 2


def test_the_launcher_parses():
    assert subprocess.run(["bash", "-n", str(REPO / "scripts" / "exp28" / "run_offline_gates.sh")]).returncode == 0

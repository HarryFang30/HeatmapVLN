"""EXP-28 line-B stream rules (ledger R1, R2, R6, R7) in build_route_stream.py, and its held-out-scene file."""

from __future__ import annotations

import json

import pytest

from scripts.tools import build_route_stream as rs
from src.data.s2_route_labels import RouteLabelConfig
from src.data.s2_sft_dataset import index_calls
from src.data.s2_stream import MANIFEST_NAME, read_manifest, read_shard
from tests.test_s2_sft_dataset import _write_labels, _write_record

DEFAULT_LABELER = RouteLabelConfig().digest()

ANSWERS = {
    "P": {"llm_output": "245 320", "native_first_output": "↓", "native_lookdown_turns": 1},
    "L": {"llm_output": "←←", "native_first_output": "←←", "native_lookdown_turns": 0},
    "R": {"llm_output": "→→", "native_first_output": "→→", "native_lookdown_turns": 0},
    "S": {"llm_output": "STOP", "native_first_output": "STOP", "native_lookdown_turns": 0},
}
RECORDED = {"P": ("↓", "245 320"), "L": ("←←",), "R": ("→→",), "S": ("STOP",)}
TARGETS = {"P": ("↓", "300 300"), "L": ("←",), "R": ("→",)}


def _episode(records, labels, scene, episode, spec, *, unlabelled=(), labeler=DEFAULT_LABELER, record_sha256=None):
    """spec: per call (answer, kind, target, distance); answer in ANSWERS, kind 'a' (anchor:
    target = the recorded answer), 'c' (correction to TARGETS[target]) or 'n' (no target).
    The rows carry what label_route_corrections.py stamps on them: labeller digest, line,
    the record's sidecar sha256 (``record_sha256`` overrides it: a stale label file)."""
    calls = []
    for i, (answer, _, _, _) in enumerate(spec):
        step = 2 * i + 2
        calls.append({"call_index": i, "step_id": step, "history_step_ids": list(range(0, step, 2))[-3:],
                      **ANSWERS[answer]})
    path = _write_record(records, scene, episode, calls)
    sidecar = path.with_name(path.name + ".sha256").read_text().split()[0]
    stamp = {"labeler": labeler, "line": "B", "record_sha256": record_sha256 or sidecar}
    rows = []
    for i, (answer, kind, target, distance) in enumerate(spec):
        if i in unlabelled:
            continue
        row = {"call_index": i, "distance_to_goal": distance, **stamp}
        if kind == "a":
            recorded = RECORDED[answer]
            row.update(oracle_kind="turn", oracle_turn1="→", oracle_turn2=None, agree=True, kind="anchor",
                       target_turn1=recorded[0], target_turn2=recorded[1] if len(recorded) > 1 else None)
        elif kind == "c":
            turns = TARGETS[target]
            row.update(oracle_kind="pixel" if target == "P" else "turn", oracle_turn1=turns[0],
                       oracle_turn2=turns[1] if len(turns) > 1 else None, agree=False, kind="correction",
                       target_turn1=turns[0], target_turn2=turns[1] if len(turns) > 1 else None)
        else:
            row.update(skip=True, oracle_kind="skip", agree=None, kind="none")
        rows.append(row)
    _write_labels(path, labels, rows)
    return path


def _pools(records, labels, **kwargs):
    refs, _ = index_calls([records], [labels], split="all", holdout=())
    kwargs.setdefault("max_corrections_per_episode", None)
    kwargs.setdefault("max_anchors_per_episode", None)
    return rs.route_pools(refs, **kwargs)


# --------------------------------------------------------------------------- #
# Scenes
# --------------------------------------------------------------------------- #
def test_scene_key_joins_hm3d_dir_names_and_bare_ids():
    assert rs.scene_key("00481-5jLhtVmWd5F") == "5jLhtVmWd5F"
    assert rs.scene_key(" 5jLhtVmWd5F\n") == "5jLhtVmWd5F"
    assert rs.scene_key("5jLhtVmWd5F.basis") == "5jLhtVmWd5F"
    assert rs.scene_key("17DRP5sb8fy") == "17DRP5sb8fy"  # MP3D: unchanged
    assert rs.scene_key("0481-5jLhtVmWd5F") == "0481-5jLhtVmWd5F"  # not the 5-digit prefix
    assert rs.holdout_match_set(["00481-5jLhtVmWd5F"]) == {"00481-5jLhtVmWd5F", "5jLhtVmWd5F"}


def test_scene_list_file_formats(tmp_path):
    text = tmp_path / "held.txt"
    text.write_text("# EXP-28 held-out\n00481-5jLhtVmWd5F\n00393-dD37vuCa4FE, wuKkTq5GJbi  # B3\n\n00481-5jLhtVmWd5F\n")
    assert rs.read_scene_list(text) == ("00393-dD37vuCa4FE", "00481-5jLhtVmWd5F", "wuKkTq5GJbi")
    listed = tmp_path / "held.json"
    listed.write_text(json.dumps(["b", "a"]))
    assert rs.read_scene_list(listed) == ("a", "b")
    wrapped = tmp_path / "held2.json"
    wrapped.write_text(json.dumps({"scenes": ["c"]}))
    assert rs.read_scene_list(wrapped) == ("c",)
    empty = tmp_path / "empty.txt"
    empty.write_text("# nothing\n")
    with pytest.raises(SystemExit, match="names no scene"):
        rs.read_scene_list(empty)


# --------------------------------------------------------------------------- #
# R6: reversals
# --------------------------------------------------------------------------- #
REVERSAL_SPEC = [
    ("L", "a", None, 6.0),  # 0 anchor; no previous call
    ("P", "c", "R", 6.0),   # 1 correction to a right turn right after the executed left turn: reversal
    ("R", "a", None, 6.0),  # 2 anchor; the previous call was a pixel goal: no executed turn
    ("L", "a", None, 6.0),  # 3 anchor turning left after the executed right turn: reversal
    ("P", "c", "L", 6.0),   # 4 correction to the left after a left: kept
    ("P", "c", "P", 6.0),   # 5 correction to a pixel goal: never a reversal
]


def test_executed_direction_reads_the_final_turn():
    assert rs.executed_direction(("←←",)) == 1 and rs.executed_direction(("→",)) == -1
    assert rs.executed_direction(("↓", "245 320")) == 0
    assert rs.executed_direction(("STOP",)) == 0 and rs.executed_direction(("←→",)) == 0
    assert rs.executed_direction(None) == 0


@pytest.mark.parametrize("call", [
    {"llm_output": "←←", "native_first_output": "←←", "native_lookdown_turns": 0},
    {"llm_output": "→", "native_first_output": "→", "native_lookdown_turns": 0},
    {"llm_output": "←←←←→→", "native_first_output": "←←←←→→", "native_lookdown_turns": 0},  # 4 executed
    {"llm_output": "←→", "native_first_output": "←→", "native_lookdown_turns": 0},
    {"llm_output": "←↑", "native_first_output": "←↑", "native_lookdown_turns": 0},
    {"llm_output": "↑↑", "native_first_output": "↑↑", "native_lookdown_turns": 0},
    {"llm_output": "STOP", "native_first_output": "STOP", "native_lookdown_turns": 0},
    {"llm_output": "", "native_first_output": "", "native_lookdown_turns": 0},
    {"llm_output": "245 320", "native_first_output": "↓", "native_lookdown_turns": 1},
    {"llm_output": "→→", "native_first_output": "↓", "native_lookdown_turns": 1},  # looked down, then turned
])
def test_the_stream_reads_the_executed_turn_as_the_labeller_does(call):
    """R6 compares a sample's target with the previous call's executed turn, read from the
    labelled refs (``recorded_turns``) or, for an unlabelled call, from the record
    (``parse_model_answer``, the labeller's own reading): the two must agree."""
    from src.data.s2_oracle_labels import parse_model_answer
    from src.data.s2_route_labels import executed_turn_direction
    from src.data.s2_sft_dataset import recorded_turns

    call = {"call_index": 0, **call}
    assert rs.executed_direction(recorded_turns(call)) == executed_turn_direction(parse_model_answer(call))


def test_reversals_are_counted_and_dropped_for_corrections_and_anchors(tmp_path):
    records, labels = tmp_path / "records", tmp_path / "labels"
    _episode(records, labels, "kfPV7w3FaU5", 1, REVERSAL_SPEC)
    corrections, anchors, stats = _pools(records, labels)
    assert [r.call_index for r in corrections] == [1, 4, 5] and [r.call_index for r in anchors] == [0, 2, 3]
    assert stats["reversed_corrections_kept"] == 1 and stats["reversed_anchors_kept"] == 1

    corrections, anchors, stats = _pools(records, labels, drop_reversals=True)
    assert [r.call_index for r in corrections] == [4, 5] and [r.call_index for r in anchors] == [0, 2]
    assert stats["reversal_dropped_corrections"] == 1 and stats["reversal_dropped_anchors"] == 1
    report = rs.reversal_report(stats)
    assert report == {"dropped_corrections": 1, "dropped_anchors": 1, "corrections_labelled": 3,
                      "share_of_corrections": round(1 / 3, 4), "reversed_targets_kept": 0}
    assert stats["previous_answer_read_from_record"] == 0


def test_the_previous_answer_of_an_unlabelled_call_is_read_from_the_record(tmp_path):
    records, labels = tmp_path / "records", tmp_path / "labels"
    path = _episode(records, labels, "kfPV7w3FaU5", 1, REVERSAL_SPEC[:2], unlabelled={0})
    corrections, anchors, stats = _pools(records, labels, drop_reversals=True)
    assert corrections == [] and anchors == []
    assert stats["reversal_dropped_corrections"] == 1 and stats["previous_answer_read_from_record"] == 1
    assert rs.record_directions(str(path)) == {0: 1, 1: 0}  # the labeller's reading of the record's calls


def test_the_previous_call_is_the_one_just_before_even_across_a_gap(tmp_path):
    records, labels = tmp_path / "records", tmp_path / "labels"
    _episode(records, labels, "kfPV7w3FaU5", 1, REVERSAL_SPEC[:2])
    refs, _ = index_calls([records], [labels], split="all", holdout=())
    refs = [refs[0], rs.dataclasses.replace(refs[1], call_index=3)]  # calls 0 and 3 labelled; 1, 2 not
    reads = []

    def from_record(path):
        reads.append(path)
        return {0: 1, 2: 0}  # call 2 executed no turn

    corrections, _, stats = rs.route_pools(refs, max_corrections_per_episode=None, max_anchors_per_episode=None,
                                           drop_reversals=True, directions_from_record=from_record)
    assert [r.call_index for r in corrections] == [3] and stats["reversal_dropped_corrections"] == 0
    assert len(reads) == 1  # one read per record

    corrections, _, _ = rs.route_pools(refs, max_corrections_per_episode=None, max_anchors_per_episode=None,
                                       drop_reversals=True, directions_from_record=lambda path: {0: 1})
    assert corrections == []  # without call 2 in the record, call 0 is the one before: a reversal


# --------------------------------------------------------------------------- #
# R2: corrections of the model's own STOP
# --------------------------------------------------------------------------- #
def test_stop_corrections_only_beyond_5m_and_one_per_episode(tmp_path):
    records, labels = tmp_path / "records", tmp_path / "labels"
    _episode(records, labels, "kfPV7w3FaU5", 1, [
        ("S", "c", "P", 4.0),   # 0: 3-5 m, no sample
        ("S", "c", "P", 5.0),   # 1: not beyond 5 m
        ("S", "c", "P", 6.0),   # 2: the earliest beyond 5 m: kept
        ("S", "c", "L", 7.0),   # 3: a second one: over the cap
        ("P", "c", "P", 8.0),   # 4: an ordinary correction
    ])
    plain, _, stats = _pools(records, labels)
    assert [r.call_index for r in plain] == [0, 1, 2, 3, 4] and stats["stop_answer_corrections"] == 4
    corrections, _, stats = _pools(records, labels, stop_correction_min_distance_m=5.0,
                                   max_stop_corrections_per_episode=1)
    assert [r.call_index for r in corrections] == [2, 4]
    assert corrections[0].turns == TARGETS["P"]  # the oracle's answer, never STOP
    assert stats["stop_answer_corrections_labelled"] == 4
    assert stats["stop_answer_corrections_dropped_near"] == 2
    assert stats["stop_answer_corrections_dropped_by_cap"] == 1
    assert stats["stop_answer_corrections"] == 1


# --------------------------------------------------------------------------- #
# R1 + R7: first 48 calls, at most 8 corrections (earliest) and 12 anchors (seeded)
# --------------------------------------------------------------------------- #
def test_first_calls_and_per_episode_caps(tmp_path):
    records, labels = tmp_path / "records", tmp_path / "labels"
    spec = [("P", "c", "P", 6.0)] * 12 + [("P", "a", None, 6.0)] * 15 + [("P", "c", "P", 6.0)] * 3
    _episode(records, labels, "kfPV7w3FaU5", 1, spec)
    kwargs = dict(first_calls=27, max_corrections_per_episode=8, max_anchors_per_episode=12, seed=42)
    corrections, anchors, stats = _pools(records, labels, **kwargs)
    assert stats["beyond_first_calls"] == 3
    assert [r.call_index for r in corrections] == list(range(8))
    assert len(anchors) == 12 and {r.call_index for r in anchors} <= set(range(12, 27))
    assert [r.call_index for r in anchors] == sorted(r.call_index for r in anchors)
    _, anchors_again, _ = _pools(records, labels, **kwargs)
    assert [r.call_index for r in anchors_again] == [r.call_index for r in anchors]
    _, other_seed, _ = _pools(records, labels, **{**kwargs, "seed": 7})
    assert len(other_seed) == 12


def test_min_distance_drops_rows_the_default_labeller_would_have_skipped(tmp_path):
    records, labels = tmp_path / "records", tmp_path / "labels"
    _episode(records, labels, "kfPV7w3FaU5", 1, [
        ("P", "c", "P", 2.5),   # 0: a labeller with another label_min_distance_m labelled it
        ("P", "a", None, 3.0),  # 1: not beyond 3 m
        ("S", "c", "P", 3.5),   # 2: beyond: R2 then drops it (3-5 m STOP)
        ("P", "c", "L", 3.01),  # 3: kept
    ])
    corrections, anchors, stats = _pools(records, labels, min_distance_m=3.0, stop_correction_min_distance_m=5.0)
    assert [r.call_index for r in corrections] == [3] and anchors == []
    assert stats["within_min_distance"] == 2 and stats["stop_answer_corrections_dropped_near"] == 1
    plain, _, stats = _pools(records, labels)
    assert len(plain) == 3 and stats["within_min_distance"] == 0  # off by default (EXP-25 rebuilds as it was)


# --------------------------------------------------------------------------- #
# End to end: held-out file, every rule, no padding
# --------------------------------------------------------------------------- #
@pytest.fixture
def exp28_roots(tmp_path):
    records, labels = tmp_path / "records", tmp_path / "labels"
    _episode(records, labels, "kfPV7w3FaU5", 1, REVERSAL_SPEC + [("S", "c", "P", 4.0), ("S", "c", "P", 9.0)]
             + [("P", "a", None, 6.0)] * 4)
    _episode(records, labels, "kfPV7w3FaU5", 2, [("P", "c", "L", 6.0), ("L", "a", None, 6.0), ("L", "a", None, 6.0),
                                                 ("P", "a", None, 7.0), ("P", "a", None, 8.0)])
    _episode(records, labels, "5jLhtVmWd5F", 3, [("P", "c", "P", 6.0)] * 3 + [("P", "a", None, 6.0)] * 6)  # held out
    held = tmp_path / "heldout_scenes.txt"
    held.write_text("# EXP-28 held-out scenes\n00481-5jLhtVmWd5F\n")
    (tmp_path / "train_scenes.txt").write_text("# EXP-28 training scenes\nkfPV7w3FaU5\n")  # T2 writes it next to held
    return records, labels, held


def exp28_argv(records, labels, held, out):
    return ["--record-root", str(records), "--label-root", str(labels), "--out-dir", str(out),
            "--holdout-scenes-file", str(held), "--train-scenes-file", str(held.parent / "train_scenes.txt"),
            "--first-calls", "48", "--min-distance", "3", "--stop-correction-min-distance", "5",
            "--max-stop-corrections-per-episode", "1", "--drop-reversals", "--no-pad",
            "--shard-size", "8", "--global-batch", "8", "--world-size", "1", "--workers", "1"]


def test_end_to_end_unpadded_stream(exp28_roots, tmp_path):
    records, labels, held = exp28_roots
    out = tmp_path / "stream_dagger"
    assert rs.main(exp28_argv(records, labels, held, out)) == 0
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert manifest["schema"] == rs.UNPADDED_SCHEMA
    with pytest.raises(ValueError, match="schema"):
        read_manifest(out)  # the trainer refuses an unpadded stream
    assert manifest["holdout"]["scenes"] == ["00481-5jLhtVmWd5F"]
    assert manifest["holdout"]["file"] == str(held.resolve())
    # episode 1: corrections 4, 5 and the STOP at 9 m; anchors 0, 2 and the four pixel anchors.
    # episode 2: correction 0, anchors 1, 2, 3, 4 (no executed turn before 1; 2 follows a left turn with a left).
    pools = manifest["selection"]["pools"]
    assert pools["corrections"] == 4 and pools["anchors"] == 10
    assert pools["reversal_dropped_corrections"] == 1 and pools["reversal_dropped_anchors"] == 1
    assert pools["stop_answer_corrections_dropped_near"] == 1 and pools["stop_answer_corrections"] == 1
    composition = manifest["selection"]["composition"]
    assert composition["by_category"]["B_anchor"] == 2 * 4  # anchors = 2 x corrections, no padding
    assert manifest["total_samples"] == 12 and manifest["num_shards"] == 2 and composition["padded"] is False
    assert composition["reversals"]["reversed_targets_kept"] == 0
    assert composition["B_stop_answer_corrections"] == 1
    knobs = manifest["selection"]["knobs"]
    assert knobs["drop_reversals"] is True and knobs["first_calls"] == 48 and knobs["no_pad"] is True
    assert knobs["min_distance"] == 3.0 and pools.get("within_min_distance", 0) == 0
    assert manifest["selection"]["train_scenes"]["scenes"] == ["kfPV7w3FaU5"]
    labels_info = manifest["selection"]["labels"]
    assert labels_info["labelers"] == {DEFAULT_LABELER: 2} and labels_info["lines"] == {"B": 2}  # the held-out record is not read
    assert labels_info["stale_label_files"] == 0
    samples = [s for shard in manifest["shards"] for s in read_shard(out / shard["name"])]
    assert [len(read_shard(out / s["name"])) for s in manifest["shards"]] == [8, 4]
    assert len(samples) == 12 and not any(s.key.startswith("5jLhtVmWd5F") for s in samples)
    assert all(s.turns[-1] != "STOP" for s in samples)


def test_dry_run_and_the_flags_no_pad_refuses(exp28_roots, tmp_path):
    records, labels, held = exp28_roots
    out = tmp_path / "dry"
    assert rs.main(exp28_argv(records, labels, held, out) + ["--dry-run"]) == 0
    assert not (out / MANIFEST_NAME).exists()
    with pytest.raises(SystemExit, match="no-pad"):
        rs.main(exp28_argv(records, labels, held, out) + ["--replay-root", str(tmp_path)])


# --------------------------------------------------------------------------- #
# R1: training scenes only; labels of these records, by the pre-registered labeller
# --------------------------------------------------------------------------- #
def test_a_record_of_a_scene_neither_held_out_nor_training_is_refused(exp28_roots, tmp_path):
    records, labels, held = exp28_roots
    _episode(records, labels, "17DRP5sb8fy", 4, [("P", "c", "P", 6.0), ("P", "a", None, 6.0)])  # an MP3D record
    with pytest.raises(SystemExit, match="neither held out nor in"):
        rs.main(exp28_argv(records, labels, held, tmp_path / "out") + ["--dry-run"])
    (tmp_path / "train_scenes.txt").write_text("kfPV7w3FaU5\n17DRP5sb8fy\n")
    assert rs.main(exp28_argv(records, labels, held, tmp_path / "out") + ["--dry-run"]) == 0


def test_a_training_list_naming_a_held_out_scene_is_refused(exp28_roots, tmp_path):
    records, labels, held = exp28_roots
    (tmp_path / "train_scenes.txt").write_text("kfPV7w3FaU5\n5jLhtVmWd5F\n")  # bare ID of the held-out 00481-...
    with pytest.raises(SystemExit, match="lists held-out scenes"):
        rs.main(exp28_argv(records, labels, held, tmp_path / "out") + ["--dry-run"])


def test_stale_label_files_are_refused(exp28_roots, tmp_path):
    records, labels, held = exp28_roots
    _episode(records, labels, "kfPV7w3FaU5", 9, [("P", "c", "P", 6.0)], record_sha256="0" * 64)
    with pytest.raises(SystemExit, match="another version of their record"):
        rs.main(exp28_argv(records, labels, held, tmp_path / "out") + ["--dry-run"])


def test_the_labeller_digest_is_recorded(exp28_roots, tmp_path, capsys):
    records, labels, held = exp28_roots
    _episode(records, labels, "kfPV7w3FaU5", 9, [("P", "c", "P", 6.0)], labeler="0123456789ab")
    out = tmp_path / "stream_dagger"
    assert rs.main(exp28_argv(records, labels, held, out)) == 0
    assert "the default RouteLabelConfig" in capsys.readouterr().err
    info = json.loads((out / MANIFEST_NAME).read_text())["selection"]["labels"]
    assert info["labelers"] == {DEFAULT_LABELER: 2, "0123456789ab": 1} and info["default_labeler"] == DEFAULT_LABELER

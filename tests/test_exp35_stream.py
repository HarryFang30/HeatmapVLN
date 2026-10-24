"""EXP-35's stream builder (scripts/exp35/build_exp35_stream.py): D1 copied byte for byte, D2 from
line-C labels with its kl flags, the gate episodes held out, and what it refuses."""

from __future__ import annotations

import json
import tarfile
from collections import Counter
from pathlib import Path

import pytest

from scripts.exp34 import build_exp34_stream as X34
from scripts.exp35 import build_exp35_stream as X
from scripts.tools import build_s2_train_stream as bst
from src.data.interndata_replay import ReplaySample
from src.data.s2_route_labels import RouteLabelConfig
from src.data.s2_stream import MANIFEST_NAME, PackedSample, read_shard
from tests.test_exp34_stream import ANSWERS, EPISODE_A, EPISODE_B, RECORDED, TARGETS, TRAIN, _episode, _fields
from tests.test_s2_sft_dataset import _write_labels, _write_record

LINE_C = RouteLabelConfig(terminal_stop_radius_m=1.0).digest()
C_TARGETS = {**TARGETS, "S": ("STOP",)}
ANSWERS = {**ANSWERS, "SX": {"llm_output": "STOP.", "native_first_output": "STOP.", "native_lookdown_turns": 0}}
RECORDED = {**RECORDED, "SX": ("STOP.",)}


def _episode_c(records, labels, scene, episode, spec, *, labeler=LINE_C, line="C"):
    """spec per call: (answer, kind, d, target): 'c' correction to C_TARGETS[target], 'a' anchor, 'n' no
    target.  Rows as label_route_corrections.py --terminal_stop_radius_m 1.0 writes them."""
    calls = []
    for i, (answer, *_rest) in enumerate(spec):
        step = 2 * i + 2
        calls.append({"call_index": i, "step_id": step, "history_step_ids": list(range(0, step, 2))[-3:],
                      **ANSWERS[answer]})
    path = _write_record(records, scene, episode, calls)
    stamp = {"labeler": labeler, "line": line,
             "record_sha256": path.with_name(path.name + ".sha256").read_text().split()[0]}
    rows = []
    for i, (answer, kind, d, target) in enumerate(spec):
        row = {"call_index": i, "distance_to_goal": d, **stamp}
        if kind == "a":
            rec = RECORDED[answer]
            row.update(kind="anchor", agree=True, oracle_kind="stop" if d <= 1.0 else "turn", reason="model_stop",
                       oracle_turn1="STOP" if d <= 1.0 else "→", oracle_turn2=None,
                       target_turn1=rec[0], target_turn2=rec[1] if len(rec) > 1 else None)
        elif kind == "c":
            turns = C_TARGETS[target]
            row.update(kind="correction", agree=False, oracle_kind={"S": "stop", "P": "pixel"}.get(target, "turn"),
                       reason="model_turn_within_stop_radius" if target == "S" else "model_pixel_no_progress",
                       oracle_turn1=turns[0], oracle_turn2=turns[1] if len(turns) > 1 else None,
                       target_turn1=turns[0], target_turn2=turns[1] if len(turns) > 1 else None)
        else:
            row.update(kind="none", skip=False, agree=None, oracle_kind="skip", reason="turn_unjudgeable")
        rows.append(row)
    _write_labels(path, labels, rows)
    return path


EPISODE_C1 = [  # scene TRAIN[0], episode 10
    ("P", "c", 8.0, "L"), ("P", "a", 7.0, None), ("L", "c", 2.0, "P"), ("R", "c", 0.6, "S"), ("S", "a", 2.0, None),
    ("P", "n", 1.5, None),
]
EPISODE_C2 = [  # scene TRAIN[1], episode 11: the gate's in test_gate_episodes_stay_out
    ("P", "a", 9.0, None), ("P", "c", 6.0, "L"), ("L", "c", 0.8, "S"), ("SX", "a", 0.5, None),
]
GATE = [("P", "c", 8.0, "L"), ("R", "c", 0.5, "S")]  # scene TRAIN[0], episode 12: the default gate episode


@pytest.fixture
def world(tmp_path, monkeypatch):
    records, labels, replay = tmp_path / "records", tmp_path / "labels", tmp_path / "replay"
    _episode(records, labels, TRAIN[0], 1, EPISODE_A)
    _episode(records, labels, TRAIN[1], 2, EPISODE_B)
    (replay / TRAIN[0] / "meta").mkdir(parents=True)

    def fake_scene_samples(scene_dir):
        for i in range(400):
            kind = "stop" if i % 40 == 0 else "turn"
            yield ReplaySample(Path(scene_dir).name, i, "walk", 4, (0, 1), kind, "STOP" if kind == "stop" else "←", None)

    monkeypatch.setattr(X34.brs, "scene_samples", fake_scene_samples)
    real = bst._pack

    def pack(item, size, quality):  # replay frames are not on disk here
        if item["source"] == "dagger":
            return real(item, size, quality)
        return PackedSample(item["key"], "replay", "walk", tuple(item["turns"]), tuple(size), (b"h",), b"c", b"l")

    monkeypatch.setattr(bst, "_pack", pack)
    d1 = tmp_path / "d1"
    assert X34.main(["--source", "r2r", str(records), str(labels), "--replay-root", str(replay), "--out-dir", str(d1),
                     "--shard-size", "64", "--global-batch", "64", "--workers", "1", "--stop-band", "0", "1"]) == 0
    records2, labels2 = tmp_path / "records2", tmp_path / "labels2"
    _episode_c(records2, labels2, TRAIN[0], 10, EPISODE_C1)
    _episode_c(records2, labels2, TRAIN[1], 11, EPISODE_C2)
    _episode_c(records2, labels2, TRAIN[0], 12, GATE)
    gate = tmp_path / "gate_cohort"
    gate.mkdir()
    (gate / "shard_000.json").write_text(json.dumps({"episodes": [{"scene_id": TRAIN[0], "episode_id": 12}]}))
    return d1, records2, labels2, replay


def _argv(d1, records2, labels2, replay, out, *, name="envdrop3", adapter="base"):
    return ["--d1-stream", str(d1), "--source", name, str(records2), str(labels2), adapter, "--replay-root", str(replay),
            "--out-dir", str(out), "--shard-size", "64", "--global-batch", "64", "--workers", "1",
            "--holdout-cohort-shards", str(Path(d1).parent / "gate_cohort"), "0"]


def _samples(stream):
    manifest = json.loads((stream / MANIFEST_NAME).read_text())
    metas, samples = [], []
    for shard in manifest["shards"]:
        with tarfile.open(stream / shard["name"]) as tar:
            metas += [json.loads(tar.extractfile(m).read()) for m in tar.getmembers() if m.name.endswith(".json")]
        samples += read_shard(stream / shard["name"])
    return manifest, metas, samples


def test_d1_byte_for_byte_and_d2_parts_with_their_kl(world, tmp_path):
    d1, records2, labels2, replay = world
    out = tmp_path / "stream"
    assert X.main(_argv(d1, records2, labels2, replay, out)) == 0
    manifest, metas, samples = _samples(out)
    _, d1_metas, d1_samples = _samples(d1)
    assert manifest["total_samples"] == len(metas) == len(d1_metas) + manifest["selection"]["stats"]["total"]
    assert manifest["total_samples"] % 64 == 0
    # every D1 sample, content and kl / part, exactly once as many times as in D1
    d1_counter = Counter(_fields(s) for s in d1_samples)
    out_counter = Counter(_fields(s) for s in samples)
    assert all(out_counter[k] >= n for k, n in d1_counter.items())
    assert Counter((m["key"], m["part"], m["kl"]) for m in d1_metas) <= Counter((m["key"], m["part"], m["kl"]) for m in metas)
    d2 = [m for m in metas if m["part"] in X.PARTS]
    assert all(m["kl"] is X.KL[m["part"]] for m in d2)
    a, b = f"{TRAIN[0]}_000010#envdrop3", f"{TRAIN[1]}_000011#envdrop3"
    cor = sorted((m["key"], tuple(m["turns"])) for m in d2 if m["part"] == "COR2")
    assert cor == [(f"{a}.0", ("←",)), (f"{a}.2", ("↓", "300 300")), (f"{a}.3", ("STOP",)),
                   (f"{b}.1", ("←",)), (f"{b}.2", ("STOP",))]
    agr = {m["key"]: tuple(m["turns"]) for m in d2 if m["part"] == "AGR2"}
    assert set(agr) <= {f"{a}.1", f"{a}.4", f"{b}.0"} and len(agr) == 3 and agr.get(f"{a}.4") == ("STOP",)
    # REP2 is InternData not already in D1
    d1_rep = {m["key"] for m in d1_metas if m["part"] == "REP"}
    rep2 = [m["key"] for m in d2 if m["part"] == "REP2"]
    rep2_stops = [m["key"] for m in d2 if m["part"] == "REP2" and m["turns"] == ["STOP"]]
    assert rep2 and rep2_stops and not set(rep2) & d1_rep and all(rep2_stops.count(k) == 5 for k in rep2_stops)
    # one order over both: sha256('exp35-order:' + key + '#' + copy), copies of a key interchangeable
    keys = [m["key"] for m in metas]
    ranks = sorted((X.hashlib.sha256(f"{X.ORDER_RANK}{k}#{c}".encode()).hexdigest(), k)
                   for k, n in Counter(keys).items() for c in range(n))
    assert keys == [k for _, k in ranks]
    tsv = [line.split("\t") for line in (out / X34.KEYS_TSV).read_text().splitlines()]
    assert [(r[0], r[1], r[2]) for r in tsv] == [(m["key"], m["part"], str(m["kl"]).lower()) for m in metas]
    assert manifest["d1"]["samples"] == len(d1_metas) and manifest["holdout"]["count"] == 26
    # every key the trainer reads from EXP-34's manifest is here too ("drawn" among them)
    d1_manifest = json.loads((d1 / MANIFEST_NAME).read_text())
    assert set(d1_manifest) <= set(manifest) and manifest["drawn"]["category_COR2"] == 5
    assert sum(v for k, v in manifest["drawn"].items() if k.startswith("category_")) == manifest["total_samples"]
    # the gate episode is out; a STOP anchor with more than a plain STOP is dropped
    assert not any("_000012#" in m["key"] for m in metas)
    assert manifest["selection"]["counts"]["AGR_dropped_unreadable"] == 1


def test_order_spreads_copies_of_a_key_and_is_deterministic():
    keys = ["replay:s_000001@0:x"] * 5 + [f"k{i}" for i in range(200)]
    first = X.order(keys)
    assert first == X.order(keys) and sorted(first) == list(range(len(keys)))
    positions = sorted(first.index(i) for i in range(5))
    assert positions[-1] - positions[0] > 5  # not five in a row


def test_gate_episodes_stay_out(world, tmp_path):
    d1, records2, labels2, replay = world
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    (cohort / "shard_002.json").write_text(json.dumps({"episodes": [{"scene_id": TRAIN[1], "episode_id": 11}]}))
    out = tmp_path / "stream"
    assert X.main(_argv(d1, records2, labels2, replay, out) + ["--holdout-cohort-shards", str(cohort), "2"]) == 0
    manifest, metas, _ = _samples(out)
    assert not any("_000011#" in m["key"] for m in metas)
    assert manifest["selection"]["sources"]["envdrop3"]["calls_held_out_for_the_gate"] == 4
    assert manifest["holdout"]["gate_episodes"] == 1


def test_refusals(world, tmp_path):
    d1, records2, labels2, replay = world
    with pytest.raises(SystemExit, match="are D1's"):
        X.main(_argv(d1, records2, labels2, replay, tmp_path / "s1", name="r2r") + ["--dry-run"])
    with pytest.raises(SystemExit, match="served by"):
        X.main(_argv(d1, records2, labels2, replay, tmp_path / "s2", adapter="9f" * 32) + ["--dry-run"])
    # line-B labels, or line C of another radius, are not this labeller's
    rb, lb = tmp_path / "rb", tmp_path / "lb"
    _episode_c(rb, lb, TRAIN[0], 20, EPISODE_C1, labeler=RouteLabelConfig().digest(), line="B")
    with pytest.raises(SystemExit, match="not line C's"):
        X.main(_argv(d1, rb, lb, replay, tmp_path / "s3") + ["--dry-run"])
    rc, lc = tmp_path / "rc", tmp_path / "lc"
    _episode_c(rc, lc, TRAIN[0], 21, EPISODE_C1, labeler=RouteLabelConfig(terminal_stop_radius_m=1.5).digest())
    with pytest.raises(SystemExit, match="not line C's"):
        X.main(_argv(d1, rc, lc, replay, tmp_path / "s4") + ["--dry-run"])
    # a STOP target beyond the STOP radius, a STOP anchor beyond 3 m
    rd, ld = tmp_path / "rd", tmp_path / "ld"
    _episode_c(rd, ld, TRAIN[0], 22, [("P", "c", 8.0, "L"), ("L", "c", 2.0, "S")])
    with pytest.raises(SystemExit, match="STOP target"):
        X.main(_argv(d1, rd, ld, replay, tmp_path / "s5") + ["--dry-run"])
    re_, le = tmp_path / "re", tmp_path / "le"
    _episode_c(re_, le, TRAIN[0], 23, [("P", "c", 8.0, "L"), ("S", "a", 4.0, None)])
    with pytest.raises(SystemExit, match="STOP anchor"):
        X.main(_argv(d1, re_, le, replay, tmp_path / "s6") + ["--dry-run"])
    # the gate's shards must match a record
    nowhere = tmp_path / "nowhere"
    nowhere.mkdir()
    (nowhere / "shard_000.json").write_text(json.dumps({"episodes": [{"scene_id": TRAIN[0], "episode_id": 999}]}))
    with pytest.raises(SystemExit, match="matched no record"):
        X.main(_argv(d1, records2, labels2, replay, tmp_path / "s8") + ["--holdout-cohort-shards", str(nowhere), "0",
                                                                        "--dry-run"])
    # D1 must be EXP-34's stream, intact
    shard = sorted(d1.glob("stream-*.tar"))[0]
    shard.write_bytes(shard.read_bytes()[:-10] + b"0" * 10)
    with pytest.raises(SystemExit, match="sha256"):
        X.main(_argv(d1, records2, labels2, replay, tmp_path / "s7") + ["--dry-run"])

"""EXP-34's stream builder (scripts/exp34/build_exp34_stream.py): D1's parts on line-B labels, kl flags,
held-out scenes, and the packing path reused from build_s2_train_stream."""

from __future__ import annotations

import json
import math
import tarfile
from pathlib import Path

import pytest

from scripts.exp34 import build_exp34_stream as X
from scripts.tools import build_s2_train_stream as bst
from src.data.interndata_replay import ReplaySample
from src.data.s2_records import EpisodeRecord
from src.data.s2_route_labels import RouteLabelConfig
from src.data.s2_sft_dataset import R2R_TRAIN_SCENES, CallRef, holdout_scenes
from src.data.s2_stream import MANIFEST_NAME, STREAM_SCHEMA, PackedSample, read_shard, write_shard
from tests.test_s2_sft_dataset import _write_labels, _write_record

TRAIN = [s for s in R2R_TRAIN_SCENES if s not in X.R2R_HOLDOUT][:2]
ANSWERS = {
    "P": {"llm_output": "245 320", "native_first_output": "↓", "native_lookdown_turns": 1},
    "L": {"llm_output": "←←", "native_first_output": "←←", "native_lookdown_turns": 0},
    "R": {"llm_output": "→→", "native_first_output": "→→", "native_lookdown_turns": 0},
    "S": {"llm_output": "STOP", "native_first_output": "STOP", "native_lookdown_turns": 0},
    "LS": {"llm_output": "←←STOP", "native_first_output": "←←STOP", "native_lookdown_turns": 0},
}
RECORDED = {"P": ("↓", "245 320"), "L": ("←←",), "R": ("→→",), "S": ("STOP",), "LS": ("←←STOP",)}
TARGETS = {"P": ("↓", "300 300"), "L": ("←",), "R": ("→",)}


def _episode(records, labels, scene, episode, spec):
    """spec per call: (answer, kind, d, target): kind 'c' correction to TARGETS[target], 'a' anchor,
    't' within 3 m (line B's skip), 'n' no target beyond 3 m.  Rows as label_route_corrections.py writes them."""
    calls = []
    for i, (answer, *_rest) in enumerate(spec):
        step = 2 * i + 2
        calls.append({"call_index": i, "step_id": step, "history_step_ids": list(range(0, step, 2))[-3:],
                      **ANSWERS[answer]})
    path = _write_record(records, scene, episode, calls)
    stamp = {"labeler": RouteLabelConfig().digest(), "line": "B",
             "record_sha256": path.with_name(path.name + ".sha256").read_text().split()[0]}
    rows = []
    for i, (answer, kind, d, target) in enumerate(spec):
        row = {"call_index": i, "distance_to_goal": d, **stamp}
        if kind == "a":
            rec = RECORDED[answer]
            row.update(kind="anchor", agree=True, oracle_kind="turn", oracle_turn1="→", oracle_turn2=None,
                       reason="model_pixel_progress", target_turn1=rec[0], target_turn2=rec[1] if len(rec) > 1 else None)
        elif kind == "c":
            turns = TARGETS[target]
            row.update(kind="correction", agree=False, oracle_kind="pixel" if target == "P" else "turn",
                       oracle_turn1=turns[0], oracle_turn2=turns[1] if len(turns) > 1 else None,
                       reason="model_pixel_no_progress", target_turn1=turns[0],
                       target_turn2=turns[1] if len(turns) > 1 else None)
        elif kind == "t":
            row.update(kind="none", skip=True, agree=None, oracle_kind="skip", reason="within_line_a_radius")
        else:
            row.update(kind="none", skip=False, agree=None, oracle_kind="skip", reason="turn_unjudgeable")
        rows.append(row)
    _write_labels(path, labels, rows)
    return path


EPISODE_A = [  # scene TRAIN[0], episode 1
    ("P", "a", 9.0, None), ("P", "c", 8.0, "L"), *[("P", "a", 7.0 - 0.1 * k, None) for k in range(8)],
    ("P", "c", 5.0, "P"), ("L", "t", 2.5, None), ("S", "t", 0.5, None),
]
EPISODE_B = [  # scene TRAIN[1], episode 2
    ("L", "a", 9.0, None),
    ("P", "c", 8.0, "R"),    # right after the executed left: reversal, dropped
    ("P", "c", 7.0, "L"),
    ("LS", "t", 2.0, None),  # a turn string with STOP: dropped
    ("R", "t", 1.5, None),
    ("S", "t", 0.4, None),
    ("P", "n", 6.0, None),   # no target beyond 3 m: not taken
]


def _cand(part, episode, call, *, d=5.0, turns=("←",)):
    ref = CallRef(record_path=f"/r/{episode}.s2rec.tar", scene_id=TRAIN[0], episode_id=episode, call_index=call,
                  recorded=turns, distance_to_goal=d, turns=turns)
    return X.Candidate(part, "s", ref, X.sample_key(ref, "s"), "r", 0)


def test_held_out_lists_are_the_r2r_six_and_the_hm3d_twenty():
    assert X.R2R_HOLDOUT == holdout_scenes()
    held = X.held_out_scenes()
    assert len(X.HM3D_HOLDOUT) == 20 and len(held) == 26 and "5jLhtVmWd5F" in held
    on_disk = Path("/workspace/exp28/episodes/splits/heldout_scenes.txt")
    if on_disk.is_file():
        from scripts.tools.build_route_stream import read_scene_list
        assert set(read_scene_list(on_disk)) == set(X.HM3D_HOLDOUT)


def test_d1_draw_one_to_one_episode_caps_and_term_take():
    cands = [_cand("COR", 1, i) for i in range(2)] + [_cand("AGR", 1, 10 + i) for i in range(10)]
    cands += [_cand("TERM", 1, 30 + i, d=1.0) for i in range(5)]
    cands += [_cand("COR", 2, 0)] + [_cand("AGR", 2, 10 + i) for i in range(2)] + [_cand("TERM", 2, 30, d=2.0)]
    cands += [_cand("COR", 3, i) for i in range(30)]  # more than 24
    parts, info = X.draw_dagger(cands)
    assert len(parts["COR"]) == 2 + 1 + 24 and info["cor_dropped_by_episode_cap"] == 6
    assert [c.ref.call_index for c in parts["COR"] if c.ref.episode_id == 3] == list(range(24))  # the earliest
    # AGR: 6 of episode 1 (lowest rank) + 2 of episode 2 = 8 < 27 COR, so all 8, flagged
    ep1 = sorted((c for c in cands if c.part == "AGR" and c.ref.episode_id == 1), key=lambda c: X.rank(X.AGR_RANK, c.key))
    assert {c.key for c in parts["AGR"]} == {c.key for c in ep1[:6]} | {c.key for c in cands if c.part == "AGR" and c.ref.episode_id == 2}
    assert info["agr_capped_by_pool"] is True
    assert len(parts["TERM"]) == 6  # min(6 candidates, ceil(27 / 2))
    # 1:1 when the pool is larger, by global rank; TERM = ceil(COR / 2) of the lowest ranks
    small = [c for c in cands if not (c.part == "COR" and c.ref.episode_id == 3)]
    parts, info = X.draw_dagger(small)
    pool = ep1[:6] + [c for c in cands if c.part == "AGR" and c.ref.episode_id == 2]
    assert [c.key for c in parts["AGR"]] == [c.key for c in sorted(pool, key=lambda c: X.rank(X.AGR_RANK, c.key))[:3]]
    terms = sorted((c for c in small if c.part == "TERM"), key=lambda c: X.rank(X.TERM_RANK, c.key))
    assert [c.key for c in parts["TERM"]] == [c.key for c in terms[: math.ceil(3 / 2)]]
    with pytest.raises(SystemExit, match="keys repeat"):
        X.draw_dagger(small + small[:1])


def test_stop_band_drops_term_stops_from_the_end_of_their_rank():
    cor = [_cand("COR", 1, i) for i in range(40)]
    term = [_cand("TERM", 2, i, d=0.5, turns=("STOP",) if i % 2 else ("←",)) for i in range(40)]
    term = sorted(term, key=lambda c: X.rank(X.TERM_RANK, c.key))
    units = X.replay_units([{"turns": ["←"], "key": f"r{i}"} for i in range(200)])
    parts, band = X.compose({"COR": cor, "AGR": [], "TERM": term}, units, shard_size=64, band=(0.03, 0.10))
    total = sum(len(v) for v in parts.values())
    stops = sum(X.is_stop(c.ref.turns) for c in parts["TERM"])
    assert total == band["total"] and total % 64 == 0 and stops / total == pytest.approx(band["stop_share"], abs=1e-6)
    assert 0.08 < band["stop_share"] <= 0.10 and band["term_dropped_stop"] == band["term_dropped_for_band"] > 0
    kept = {c.key for c in parts["TERM"]}
    dropped = [c for c in term if c.key not in kept]
    last_stops = [c.key for c in term if X.is_stop(c.ref.turns)][-len(dropped):]
    assert {c.key for c in dropped} == set(last_stops) and stops == 20 - len(dropped)
    assert len(parts["REP"]) >= 0.2 * total


def _fields(s):
    """A packed sample's content without the trainer's ``kl`` field (PackedSample grows one with the EXP-34 trainer)."""
    return (s.key, s.source, s.instruction, tuple(s.turns), tuple(s.vlm_image_size), tuple(s.history_jpegs),
            s.current_jpeg, s.lookdown_jpeg)


def test_kl_shard_is_write_shards_layout_plus_kl_and_part(tmp_path):
    samples = [PackedSample(f"k{i}", "dagger", "go", ("↓", "1 2"), (384, 384), (b"h0", b"h1"), b"cur", b"look")
               for i in range(2)]
    items = [{"kl": False, "part": "COR"}, {"kl": True, "part": "REP"}]
    write_shard(tmp_path / "a.tar", samples)
    X.write_kl_shard(tmp_path / "b.tar", samples, items)
    back = read_shard(tmp_path / "b.tar")
    assert [_fields(s) for s in back] == [_fields(s) for s in samples]
    # a read_shard that knows kl (the EXP-34 trainer's) reads exactly the tag written here
    assert [getattr(s, "kl", item["kl"]) for s, item in zip(back, items)] == [False, True]

    def members(path):
        with tarfile.open(path) as tar:
            return {m.name: tar.extractfile(m).read() for m in tar.getmembers()}

    a, b = members(tmp_path / "a.tar"), members(tmp_path / "b.tar")
    assert list(a) == list(b)
    for name in a:
        if name.endswith(".json"):
            meta = json.loads(b[name])
            assert {k: v for k, v in meta.items() if k not in ("kl", "part")} == json.loads(a[name])
            assert (meta["kl"], meta["part"]) == ((False, "COR") if name.startswith("000000") else (True, "REP"))
        else:
            assert a[name] == b[name]


@pytest.fixture
def roots(tmp_path, monkeypatch):
    records, labels, replay = tmp_path / "records", tmp_path / "labels", tmp_path / "replay"
    _episode(records, labels, TRAIN[0], 1, EPISODE_A)
    _episode(records, labels, TRAIN[1], 2, EPISODE_B)
    _episode(records, labels, "17DRP5sb8fy", 3, [("P", "c", 8.0, "L"), ("S", "t", 0.5, None)])  # R2R held out
    _episode(records, labels, "5jLhtVmWd5F", 4, [("P", "c", 8.0, "L"), ("S", "t", 0.5, None)])  # HM3D held out
    for scene in (TRAIN[0], "17DRP5sb8fy"):
        (replay / scene / "meta").mkdir(parents=True)

    def fake_scene_samples(scene_dir):
        for i in range(120):
            kind = "stop" if i % 40 == 0 else "turn"
            yield ReplaySample(Path(scene_dir).name, i, "walk", 4, (0, 1), kind, "STOP" if kind == "stop" else "←", None)

    monkeypatch.setattr(X.brs, "scene_samples", fake_scene_samples)
    packed = []
    real = bst._pack

    def pack(item, size, quality):  # the real packer for records; replay frames are not on disk here
        packed.append(item["key"])
        if item["source"] == "dagger":
            return real(item, size, quality)
        return PackedSample(item["key"], "replay", "walk", tuple(item["turns"]), tuple(size), (b"h",), b"c", b"l")

    monkeypatch.setattr(bst, "_pack", pack)
    return records, labels, replay, packed


def _argv(records, labels, replay, out):
    return ["--source", "r2r", str(records), str(labels), "--replay-root", str(replay), "--out-dir", str(out),
            "--shard-size", "64", "--global-batch", "64", "--workers", "1", "--stop-band", "0", "1"]


def test_end_to_end_parts_kl_holdout_and_the_reused_packer(roots, tmp_path):
    records, labels, replay, packed = roots
    out = tmp_path / "stream"
    # --first-calls 10: a.10 (COR) is cut, a.11 / a.12 (TERM) are not: TERM takes calls of any index
    assert X.main(_argv(records, labels, replay, out) + ["--first-calls", "10"]) == 0
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert manifest["schema"] == STREAM_SCHEMA and manifest["total_samples"] == 64
    metas, samples = [], []
    for shard in manifest["shards"]:
        with tarfile.open(out / shard["name"]) as tar:
            metas += [json.loads(tar.extractfile(m).read()) for m in tar.getmembers() if m.name.endswith(".json")]
        samples += read_shard(out / shard["name"])
    assert sorted(packed) == sorted(m["key"] for m in metas)  # every sample went through build_s2_train_stream._pack
    assert all(m["kl"] is X.KL[m["part"]] for m in metas)
    by_part = {p: [m for m in metas if m["part"] == p] for p in X.PARTS}
    a, b = f"{TRAIN[0]}_000001#r2r", f"{TRAIN[1]}_000002#r2r"
    assert sorted((m["key"], tuple(m["turns"])) for m in by_part["COR"]) == [
        (f"{a}.1", ("←",)), (f"{b}.2", ("←",))]  # b.1 is a reversal, a.10 past --first-calls
    assert len(by_part["AGR"]) == 2 and all(m["key"].startswith(a) or m["key"] == f"{b}.0" for m in by_part["AGR"])
    term_pool = [f"{a}.11", f"{a}.12", f"{b}.4", f"{b}.5"]  # b.3 holds a STOP in its turn string
    want = sorted(term_pool, key=lambda k: X.rank(X.TERM_RANK, k))[:1]
    assert [m["key"] for m in by_part["TERM"]] == want
    assert manifest["selection"]["dagger"]["term_pool"] == 4
    assert len(by_part["REP"]) == 64 - 5 and len(by_part["REP"]) >= 0.2 * 64
    rep_stops = [m["key"] for m in by_part["REP"] if m["turns"] == ["STOP"]]
    assert all(rep_stops.count(k) == 5 for k in rep_stops)  # whole STOP units
    assert not any(s in m["key"] for m in metas for s in ("17DRP5sb8fy", "5jLhtVmWd5F"))
    # DAgger samples carry the record's bytes
    for sample in samples:
        if sample.source == "dagger":
            scene_episode, call = sample.key.split("#")
            record = EpisodeRecord(next(Path(records).glob(f"*/{scene_episode}.s2rec.tar")))
            index = int(call.split(".")[1])
            c = {int(c["call_index"]): c for c in record.calls}[index]
            assert sample.current_jpeg == record.front_jpeg(c["step_id"])
            assert sample.lookdown_jpeg == record.lookdown_jpeg(index)
    tsv = [line.split("\t") for line in (out / X.KEYS_TSV).read_text().splitlines()]
    assert [(r[0], r[1], r[2]) for r in tsv] == [(m["key"], m["part"], str(m["kl"]).lower()) for m in metas]
    assert manifest["holdout"]["count"] == 26 and manifest["selection"]["parts"]["COR"] == 2


def test_a_record_of_a_non_training_scene_is_refused(roots, tmp_path):
    records, labels, replay, _ = roots
    _episode(records, labels, "2azQ1b91cZZ", 5, [("P", "c", 8.0, "L")])  # val_unseen
    with pytest.raises(SystemExit, match="neither held out nor training"):
        X.main(_argv(records, labels, replay, tmp_path / "s") + ["--dry-run"])

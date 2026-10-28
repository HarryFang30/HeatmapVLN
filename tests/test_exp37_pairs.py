"""EXP-37's pair data: the sidecar of EXP-35's stream (build_pair_sidecar.py), the DPO pair stream packed
from it (pack_pair_stream.py), and G1's held-out pairs (build_heldout_pairs.py)."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from scripts.exp34 import build_exp34_stream as X34
from scripts.exp35 import build_exp35_stream as X35
from scripts.exp37 import build_heldout_pairs as H
from scripts.exp37 import build_pair_sidecar as SC
from scripts.exp37 import pack_pair_stream as PK
from src.data.s2_records import EpisodeRecord, iter_record_paths, label_path_for, read_labels
from src.data.s2_sft_dataset import recorded_turns
from src.data.s2_stream import MANIFEST_NAME, read_shard
from tests import test_exp34_stream as T34
from tests import test_exp35_stream as T35
from tests.test_exp35_stream import world  # noqa: F401  (the fixture: EXP-34's D1 and the round-2 records)

ADAPTER = "9f" * 32


def _add_model_turns(record_root, label_root):
    """Label rows carry the learner's answer as label_route_corrections.py writes it (model_turn1/2)."""
    for path in iter_record_paths(record_root):
        calls = {int(c["call_index"]): c for c in EpisodeRecord(path).calls}
        label = label_path_for(path, label_root)
        rows = read_labels(label)
        for row in rows:
            turns = recorded_turns(calls[int(row["call_index"])])
            row["model_turn1"], row["model_turn2"] = turns[0], turns[1] if len(turns) > 1 else None
        label.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _shard_samples(stream):
    manifest = json.loads((Path(stream) / MANIFEST_NAME).read_text())
    metas, members = [], []
    for shard in manifest["shards"]:
        with tarfile.open(Path(stream) / shard["name"]) as tar:
            blobs = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
        i = 0
        while f"{i:06d}.json" in blobs:
            metas.append(json.loads(blobs[f"{i:06d}.json"]))
            members.append({k.split(".", 1)[1]: v for k, v in blobs.items() if k.startswith(f"{i:06d}.")})
            i += 1
    return manifest, metas, members


@pytest.fixture
def streams(world, tmp_path, monkeypatch):  # noqa: F811
    """EXP-35's stream over D1 (source r2r, base) plus round-2 sources envdrop2 (base) and envdrop3 (an adapter)."""
    d1, records2, labels2, replay = world
    monkeypatch.setitem(T35.ANSWERS, "PC", {"llm_output": "290 295", "native_first_output": "↓",
                                            "native_lookdown_turns": 1})
    T35._episode_c(records2, labels2, T35.TRAIN[1], 13, [("PC", "c", 7.0, "P"), ("S", "c", 6.0, "L"),
                                                        ("P", "c", 4.0, "L")])
    records3, labels3 = tmp_path / "records3", tmp_path / "labels3"
    T35._episode_c(records3, labels3, T35.TRAIN[0], 20, [("P", "c", 8.0, "L")])
    for records, labels in ((tmp_path / "records", tmp_path / "labels"), (records2, labels2), (records3, labels3)):
        _add_model_turns(records, labels)
    monkeypatch.setattr(X34, "_record_adapter", lambda p: ADAPTER if "records3" in str(p) else None)
    out = tmp_path / "x35"
    assert X35.main(["--d1-stream", str(d1), "--source", "envdrop2", str(records2), str(labels2), "base",
                     "--source", "envdrop3", str(records3), str(labels3), ADAPTER, "--replay-root", str(replay),
                     "--out-dir", str(out), "--shard-size", "64", "--global-batch", "64", "--workers", "1",
                     "--holdout-cohort-shards", str(tmp_path / "gate_cohort"), "0"]) == 0
    monkeypatch.undo()  # the sidecar reads the records' own server tags
    return d1, out


def _sidecar(d1, x35, out):
    assert SC.main(["--stream", str(x35), "--d1-stream", str(d1), "--out-dir", str(out), "--workers", "1"]) == 0
    pairs = [json.loads(line) for line in (out / SC.PAIRS_NAME).read_text().splitlines()]
    return pairs, json.loads((out / SC.SUMMARY_NAME).read_text())


# --------------------------------------------------------------------------- #
# One pair
# --------------------------------------------------------------------------- #
def test_judge_pair_rules():
    px, turn = ("↓", "300 300"), ("←",)
    assert SC.judge_pair(("↓", "245 320"), turn, d=5.0, call_index=0) == (
        None, {"chosen_class": "turn_left", "rejected_class": "pixel", "pixel_dist": None})
    assert SC.judge_pair(turn, px, d=3.0, call_index=0)[0] == "within_far_radius"  # d > 3 m, strictly
    assert SC.judge_pair(turn, px, d=None, call_index=0)[0] == "within_far_radius"
    assert SC.judge_pair(turn, px, d=9.0, call_index=48)[0] == "beyond_first_calls"
    assert SC.judge_pair(("STOP",), px, d=9.0, call_index=1)[0] == "rejected_stop"
    assert SC.judge_pair(("",), px, d=9.0, call_index=1)[0] == "rejected_stop"  # the server's fallback_stop
    assert SC.judge_pair(turn, ("STOP",), d=9.0, call_index=1)[0] == "chosen_stop"
    assert SC.judge_pair(("←←STOP",), px, d=9.0, call_index=1)[0] == "rejected_turn_with_stop"
    assert SC.judge_pair(turn, turn, d=9.0, call_index=1)[0] == "rejected_equals_chosen"
    # pixel -> pixel: under 25 px dropped, 25 px kept
    why, meta = SC.judge_pair(("↓", "300 324"), px, d=9.0, call_index=1)
    assert why == "pixel_pair_within_min_px" and meta["pixel_dist"] == 24.0
    why, meta = SC.judge_pair(("↓", "315 320"), px, d=9.0, call_index=1)
    assert why is None and meta["pixel_dist"] == 25.0
    pair = SC.pair_meta(source="r2r", d=9.12345, call_index=1, judged=meta, episode="s_000001")
    assert pair == {"source": "r2r", "learner": "base", "d": 9.1235, "call_index": 1, "chosen_class": "pixel",
                    "rejected_class": "pixel", "pixel_dist": 25.0, "episode": "s_000001"}
    assert SC.pair_type(pair) == "pixel->pixel" and SC.key_source("a_000001#envdrop2.17") == ("envdrop2", 17)


# --------------------------------------------------------------------------- #
# The sidecar
# --------------------------------------------------------------------------- #
def test_sidecar_pairs_of_the_base_learner_only(streams, tmp_path):
    d1, x35 = streams
    pairs, summary = _sidecar(d1, x35, tmp_path / "pairs")
    a, b = f"{T34.TRAIN[0]}_000001#r2r", f"{T34.TRAIN[1]}_000002#r2r"
    c1, c2, c3 = (f"{T35.TRAIN[0]}_000010#envdrop2", f"{T35.TRAIN[1]}_000011#envdrop2",
                  f"{T35.TRAIN[1]}_000013#envdrop2")
    got = {p["key"]: (tuple(p["rejected_turns"]), tuple(p["chosen_turns"])) for p in pairs}
    assert got == {
        f"{a}.1": (("↓", "245 320"), ("←",)), f"{a}.10": (("↓", "245 320"), ("↓", "300 300")),
        f"{b}.2": (("↓", "245 320"), ("←",)),
        f"{c1}.0": (("↓", "245 320"), ("←",)), f"{c2}.1": (("↓", "245 320"), ("←",)),
        f"{c3}.2": (("↓", "245 320"), ("←",)),
    }
    counts = summary["counts"]
    assert counts["dropped:learner_not_base"] == 1 and counts["dropped:learner_not_base:envdrop3"] == 1
    assert counts["dropped:rejected_stop"] == 1 and counts["dropped:pixel_pair_within_min_px"] == 1
    assert counts["dropped:within_far_radius"] == 3  # line C's 2 m pixel and two STOP targets within 1 m
    assert counts["rows_COR"] == 3 and counts["rows_COR2"] == 9
    stats = summary["stats"]
    assert stats["by_source"] == {"envdrop2": 3, "r2r": 3} and stats["by_type"] == {"pixel->pixel": 1, "pixel->turn": 5}
    assert stats["by_part"] == {"COR": 3, "COR2": 3} and stats["pixel_pair_distance"] == {"<100px": 1}
    # every pair names its EXP-35 sample, in stream order, with the contract's meta
    _, metas, _ = _shard_samples(x35)
    assert [p["position"] for p in pairs] == sorted(p["position"] for p in pairs)
    for p in pairs:
        meta = metas[p["position"]]
        assert p["shard"] == f"stream-{p['position'] // 64:06d}.tar" and p["index"] == p["position"] % 64
        assert meta["key"] == p["key"] and meta["turns"] == p["chosen_turns"] and meta["kl"] is False
        assert set(p["pair"]) == {"source", "learner", "d", "call_index", "chosen_class", "rejected_class",
                                  "pixel_dist", "episode"}
        assert p["pair"]["learner"] == "base" and p["pair"]["d"] > 3.0
        assert p["pair"]["episode"] == p["key"].split("#")[0] and p["record"].endswith(f"#{p['pair']['call_index']}")
    assert summary["stream"]["manifest_sha256"] == hashlib.sha256((x35 / MANIFEST_NAME).read_bytes()).hexdigest()
    assert summary["pairs_sha256"] == hashlib.sha256((tmp_path / "pairs" / SC.PAIRS_NAME).read_bytes()).hexdigest()


def test_sidecar_refusals(streams, tmp_path, monkeypatch):
    d1, x35 = streams
    with pytest.raises(SystemExit, match="streams built by"):
        SC.main(["--stream", str(x35), "--d1-stream", str(x35), "--out-dir", str(tmp_path / "p0"), "--workers", "1"])
    monkeypatch.setattr(SC, "served_adapter", lambda episode: ADAPTER)
    with pytest.raises(SystemExit, match="served with System2 LoRA"):
        SC.main(["--stream", str(x35), "--d1-stream", str(d1), "--out-dir", str(tmp_path / "p1"), "--workers", "1"])
    monkeypatch.undo()
    # a label's learner answer that is not the record's
    label = next(Path(p) for p in (tmp_path / "labels").rglob("*.s2labels.jsonl") if "_000001" in p.name)
    rows = read_labels(label)
    rows[1]["model_turn1"], rows[1]["model_turn2"] = "→→", None
    label.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(SystemExit, match="label model turns"):
        SC.main(["--stream", str(x35), "--d1-stream", str(d1), "--out-dir", str(tmp_path / "p2"), "--workers", "1"])


# --------------------------------------------------------------------------- #
# The pair stream
# --------------------------------------------------------------------------- #
def test_pack_copies_samples_byte_for_byte_and_adds_the_pair(streams, tmp_path):
    d1, x35 = streams
    pairs, _ = _sidecar(d1, x35, tmp_path / "pairs")
    out = tmp_path / "pair_stream"
    assert PK.main(["--pairs", str(tmp_path / "pairs" / SC.PAIRS_NAME), "--stream-dir", str(x35), "--out-dir", str(out),
                    "--shard-size", "2", "--global-batch", "2", "--world-size", "1", "--workers", "1"]) == 0
    manifest, metas, members = _shard_samples(out)
    x35_manifest, x35_metas, x35_members = _shard_samples(x35)
    assert manifest["schema"] == PK.PAIR_SCHEMA and manifest["pairs"] is True
    assert manifest["total_samples"] == len(metas) == 6 and manifest["num_shards"] == 3
    assert manifest["selection"]["dropped_tail"] == 0 and manifest["total_samples"] % manifest["global_batch"] == 0
    assert set(x35_manifest) - {"d1"} <= set(manifest)  # every key the trainer reads from a stream manifest
    assert manifest["holdout"] == x35_manifest["holdout"] and manifest["drawn"]["category_COR2"] == 3
    # order: sha256('exp37-order:' + key)
    keys = [m["key"] for m in metas]
    assert keys == sorted(keys, key=lambda k: hashlib.sha256(f"exp37-order:{k}".encode()).hexdigest())
    by_key = {p["key"]: p for p in pairs}
    source = {m["key"]: (m, mem) for m, mem in zip(x35_metas, x35_members)}
    for meta, mem in zip(metas, members):
        old_meta, old_mem = source[meta["key"]]
        assert {k: v for k, v in mem.items() if k != "json"} == {k: v for k, v in old_mem.items() if k != "json"}
        assert {k: v for k, v in meta.items() if k not in ("rejected_turns", "pair")} == old_meta
        assert list(meta)[: len(old_meta)] == list(old_meta)  # the old fields first, in their order
        assert meta["kl"] is False and meta["turns"] == by_key[meta["key"]]["chosen_turns"]
        assert meta["rejected_turns"] == by_key[meta["key"]]["rejected_turns"]
        assert meta["pair"] == by_key[meta["key"]]["pair"]
    # the trainer's reader takes the shards (turns = chosen)
    samples = [s for shard in manifest["shards"] for s in read_shard(out / shard["name"])]
    assert [list(s.turns) for s in samples] == [m["turns"] for m in metas]
    assert all(getattr(s, "rejected_turns", None) in (None, tuple(m["rejected_turns"])) for s, m in zip(samples, metas))
    tsv = [line.split("\t") for line in (out / X34.KEYS_TSV).read_text().splitlines()]
    assert [r[0] for r in tsv] == keys and {r[1] for r in tsv} == {"COR", "COR2"}


def test_pack_drops_the_tail_and_refuses_another_stream(streams, tmp_path):
    d1, x35 = streams
    _sidecar(d1, x35, tmp_path / "pairs")
    pairs_file = str(tmp_path / "pairs" / SC.PAIRS_NAME)
    out = tmp_path / "subset"
    assert PK.main(["--pairs", pairs_file, "--stream-dir", str(x35), "--out-dir", str(out), "--limit", "5",
                    "--shard-size", "2", "--global-batch", "2", "--world-size", "1", "--workers", "1"]) == 0
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert manifest["total_samples"] == 4 and manifest["selection"]["dropped_tail"] == 1
    assert manifest["selection"]["limit"] == 5 and manifest["selection"]["dropped_tail_keys_sha256"]
    with pytest.raises(SystemExit, match="another stream"):
        PK.main(["--pairs", pairs_file, "--stream-dir", str(d1), "--out-dir", str(tmp_path / "s1"),
                 "--shard-size", "2", "--global-batch", "2", "--world-size", "1", "--workers", "1"])
    with pytest.raises(SystemExit, match="do not fill one shard"):
        PK.main(["--pairs", pairs_file, "--stream-dir", str(x35), "--out-dir", str(tmp_path / "s2"),
                 "--shard-size", "64", "--global-batch", "64", "--world-size", "1", "--workers", "1"])
    # a sidecar edited after it was written
    (tmp_path / "pairs" / SC.PAIRS_NAME).write_text(Path(pairs_file).read_text().replace("←", "→", 1))
    with pytest.raises(SystemExit, match="pairs_sha256"):
        PK.main(["--pairs", pairs_file, "--stream-dir", str(x35), "--out-dir", str(tmp_path / "s3"),
                 "--shard-size", "2", "--global-batch", "2", "--world-size", "1", "--workers", "1"])


# --------------------------------------------------------------------------- #
# G1's held-out pairs
# --------------------------------------------------------------------------- #
def test_heldout_pairs_packed_from_the_records(tmp_path, monkeypatch):
    monkeypatch.setitem(T34.TARGETS, "Q", ("↓", "250 330"))  # 11.2 px from the recorded 245 320
    records, labels = tmp_path / "records", tmp_path / "labels"
    scene = X34.R2R_HOLDOUT[0]
    path = T34._episode(records, labels, scene, 5, [
        ("P", "c", 8.0, "L"), ("P", "a", 7.5, None), ("S", "c", 7.0, "L"), ("P", "c", 5.0, "P"), ("L", "t", 2.0, None),
    ])
    T34._episode(records, labels, X34.R2R_HOLDOUT[1], 6, [("P", "c", 9.0, "L")])
    T34._episode(records, labels, X34.R2R_HOLDOUT[1], 8, [("P", "c", 9.0, "Q")])  # a near pixel pair: G1 keeps it
    T34._episode(records, labels, T34.TRAIN[0], 7, [("P", "c", 9.0, "L")])  # a training scene: not in the pool
    _add_model_turns(records, labels)
    stream_manifest = tmp_path / "stream_manifest.json"
    stream_manifest.write_text(json.dumps({"holdout": {"scenes": sorted(X34.held_out_scenes())}}))
    out = tmp_path / "heldout"
    argv = ["--source", "r2r-H", str(records), str(labels), "r2r", "--stream-manifest", str(stream_manifest),
            "--out-dir", str(out)]
    assert H.main(argv) == 0
    manifest, metas, members = _shard_samples(out)
    assert manifest["schema"] == PK.PAIR_SCHEMA and manifest["pairs"] is True and manifest["heldout"] is True
    k = f"{scene}_000005#r2r-H"
    got = {m["key"]: (tuple(m["rejected_turns"]), tuple(m["turns"])) for m in metas}
    near = f"{X34.R2R_HOLDOUT[1]}_000008#r2r-H.0"
    assert got == {f"{k}.0": (("↓", "245 320"), ("←",)), f"{k}.3": (("↓", "245 320"), ("↓", "300 300")),
                   f"{X34.R2R_HOLDOUT[1]}_000006#r2r-H.0": (("↓", "245 320"), ("←",)),
                   near: (("↓", "245 320"), ("↓", "250 330"))}
    counts = manifest["selection"]["counts"]
    assert counts["dropped:rejected_stop:r2r-H"] == 1 and counts["states_anchor:r2r-H"] == 1
    assert not any(c.startswith("dropped:pixel_pair_within_min_px") for c in counts)
    assert manifest["selection"]["settings"]["min_pixel_px"] == 0.0
    assert manifest["total_samples"] == 4 and manifest["shards"][0]["samples"] == 4
    # the training pairs' 25 px rule only on request
    assert H.main(argv[:-1] + [str(tmp_path / "h25"), "--min-pixel-px", "25"]) == 0
    m25, metas25, _ = _shard_samples(tmp_path / "h25")
    assert {m["key"] for m in metas25} == set(got) - {near}
    assert m25["selection"]["counts"]["dropped:pixel_pair_within_min_px:r2r-H"] == 1
    # images byte for byte from the record, as build_exp34_stream packs a call
    record = EpisodeRecord(path)
    for meta, mem in zip(metas, members):
        if not meta["key"].startswith(k):
            continue
        call = record.calls[meta["pair"]["call_index"]]
        assert mem["cur.jpg"] == record.front_jpeg(call["step_id"])
        assert mem["look.jpg"] == record.lookdown_jpeg(call["call_index"])
        assert [mem[f"h{i:02d}.jpg"] for i in range(meta["num_history"])] == \
            [record.front_jpeg(s) for s in call["history_step_ids"]]
        assert meta["kl"] is False and meta["part"] == "HELD" and meta["source"] == "dagger"
        assert meta["pair"]["learner"] == "base" and meta["pair"]["source"] == "r2r-H"
        assert meta["instruction"] == record.episode["instruction"]
    # a scene the training stream did not hold out is refused
    stream_manifest.write_text(json.dumps({"holdout": {"scenes": sorted(X34.held_out_scenes() - {scene})}}))
    with pytest.raises(SystemExit, match="did not hold out"):
        H.main(argv[:-1] + [str(tmp_path / "h2")])

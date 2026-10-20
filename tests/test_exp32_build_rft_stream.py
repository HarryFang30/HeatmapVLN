"""EXP-32 RFT stream (ledger 第二部分 C, RF1-RF7) in scripts/exp32/build_rft_stream.py, and its merge (T1)."""

from __future__ import annotations

import io
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.exp28 import merge_streams as ms
from scripts.exp32 import build_rft_stream as brs
from scripts.tools.build_route_stream import UNPADDED_SCHEMA
from src.data.s2_records import EpisodeRecord, EpisodeRecordWriter
from src.data.s2_sft_dataset import holdout_scenes, index_calls, recorded_turns
from src.data.s2_stream import MANIFEST_NAME, read_manifest, read_shard
from tests.test_exp28_merge_streams import R2R_SCENES, _replay_samples, _replay_stream

ANSWERS = {
    "P": {"llm_output": "245 320", "native_first_output": "↓", "native_lookdown_turns": 1},  # pixel
    "L": {"llm_output": "←←", "native_first_output": "←←", "native_lookdown_turns": 0},
    "R": {"llm_output": "→→", "native_first_output": "→→", "native_lookdown_turns": 0},
    "S": {"llm_output": "STOP", "native_first_output": "STOP", "native_lookdown_turns": 0},
    "F": {"llm_output": "↑↑", "native_first_output": "↑↑", "native_lookdown_turns": 0},  # forward
    "I": {"llm_output": "12", "native_first_output": "12", "native_lookdown_turns": 0},  # invalid: one integer
    "D": {"llm_output": "↓", "native_first_output": "↓", "native_lookdown_turns": 1},  # looked down twice
    "Q": {"llm_output": "300 200", "native_first_output": "↓", "native_lookdown_turns": 1},  # another pixel
}
TRAIN = ("kfPV7w3FaU5", "00100-AAAAAAAAAAA", "BBBBBBBBBBB")
HELD = "00481-5jLhtVmWd5F"


def _jpeg(seed: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), ((37 * seed) % 256, (11 * seed) % 256, (seed // 7) % 256)).save(buf, format="JPEG")
    return buf.getvalue()


def _record(root, scene, episode, spec, *, success=True, seed=1337, call_seeds=None, adapter=None, shard=30,
            device="npu"):
    """spec: per call (answer in ANSWERS, distance_to_goal or None); call i at step 2i + 2."""
    formats = ["json+jpeg"] + ([f"heatmapvln-device:{device}"] if device != "cuda" else [])
    version = "ppa-stage2-online-amb3r:zhr_1"
    if adapter:
        formats.append(f"system2-lora-sha256:{adapter}")
        version += f"+system2-lora:{adapter}"
    writer = EpisodeRecordWriter(root=Path(root), scene_id=scene, episode_id=episode, episode={
        "instruction": f"walk to the door of {scene} {episode}", "split": "train", "vlm_image_size": [384, 384],
        "outcome": {"success": 1.0 if success else 0.0, "steps": 2 * len(spec) + 2}, "protocol_seed": seed,
        "episode_list": f"/workspace/x/cohorts/shard_{shard:03d}.json",
        "server": {"model_version": version, "supported_formats": formats}})
    for step in range(0, 2 * len(spec) + 3, 2):
        writer.add_step(step, [0.0, 0.0, -0.25 * step], [1.0, 0.0, 0.0, 0.0])
        writer.add_front(step, _jpeg(1000 * episode + step))
    for i, (answer, distance) in enumerate(spec):
        step = 2 * i + 2
        call_seed = (call_seeds or {}).get(i, seed)
        writer.add_call({"call_index": i, "step_id": step, "history_step_ids": list(range(0, step, 2))[-3:],
                         "kind": "native_actions", "distance_to_goal": distance,
                         "sampling": {"protocol_seed": call_seed, "system2_call_index": i}, **ANSWERS[answer]},
                        lookdown_jpeg=_jpeg(77 + 5 * episode + i), lookdown_depth_m=None)
    return writer.commit()


# p1's episode 1: one call per RF2/RF3 outcome
EVERY_REASON = [
    ("L", 6.0),   # 0 kept (no call before it)
    ("R", 6.0),   # 1 reversal: right after the executed left
    ("P", 6.0),   # 2 kept
    ("S", 6.0),   # 3 class_stop
    ("F", 6.0),   # 4 class_forward
    ("I", 6.0),   # 5 class_invalid
    ("D", 6.0),   # 6 class_lookdown
    ("P", 2.5),   # 7 within_min_distance
    ("P", 3.0),   # 8 within_min_distance (not beyond 3 m)
    ("P", None),  # 9 no_distance
    ("L", 3.01),  # 10 kept (the call before was a pixel goal)
    ("L", 7.0),   # 11 kept (left after left)
    ("R", 7.0),   # 12 reversal
]


@pytest.fixture
def rft_roots(tmp_path):
    p1, r0s, p2, anchors = (tmp_path / n for n in ("p1", "r0s", "p2", "anchors"))
    _record(p1, "kfPV7w3FaU5", 1, EVERY_REASON)                                    # 4 samples
    _record(p1, "kfPV7w3FaU5", 2, [("P", 6.0)] * 3, success=False)                 # not_success
    _record(p1, "AAAAAAAAAAA", 3, [("P", 6.0)] * 3, adapter="ab12cd34")            # served_adapter
    _record(p1, "AAAAAAAAAAA", 4, [("P", 6.0), ("L", 6.0), ("P", 5.0)])            # 3
    _record(p1, "BBBBBBBBBBB", 5, [("L", 6.0), ("L", 6.0)], device="cuda")         # 2
    _record(r0s, "AAAAAAAAAAA", 4, [("P", 6.0), ("L", 6.0), ("P", 5.0)])           # duplicate of p1's
    _record(r0s, "BBBBBBBBBBB", 5, [("P", 6.0), ("Q", 6.0), ("R", 6.0)])           # 3, second of episode 5
    _record(r0s, "kfPV7w3FaU5", 6, [("P", 8.0), ("R", 8.0)])                       # 2
    _record(p2, "BBBBBBBBBBB", 5, [("R", 6.0)], seed=42)                           # over_episode_cap
    _record(p2, "BBBBBBBBBBB", 7, [("P", 6.0), ("Q", 6.0), ("S", 2.0)], seed=42)   # 2
    for episode, scene in ((101, "kfPV7w3FaU5"), (102, "AAAAAAAAAAA"), (103, "BBBBBBBBBBB"), (104, "kfPV7w3FaU5")):
        _record(anchors, scene, episode, [("P", 6.0), ("Q", 6.5)] * 4, seed=42, shard=20 + episode % 7)  # 8 each
    _record(anchors, "5jLhtVmWd5F", 105, [("P", 6.0)] * 3, seed=42, shard=3)              # held out: skipped
    _record(anchors, "kfPV7w3FaU5", 106, [("P", 6.0)] * 3, seed=42, shard=5)              # shard_below_20
    _record(anchors, "AAAAAAAAAAA", 107, [("P", 6.0)] * 3, seed=42, shard=25, success=False)
    _record(anchors, "BBBBBBBBBBB", 108, [("L", 6.0), ("R", 6.0), ("S", 6.0), ("P", 6.0)], seed=42, shard=40)
    held = tmp_path / "heldout_scenes.txt"
    held.write_text(f"# EXP-28 held-out scenes\n{HELD}\n")
    train = tmp_path / "train_scenes.txt"
    train.write_text("# EXP-28 training scenes\n" + "\n".join(TRAIN) + "\n")
    return {"p1": p1, "r0s": r0s, "p2": p2, "anchors": anchors, "held": held, "train": train}


def _argv(roots, out, *extra, sources=None):
    sources = sources or [f"p1={roots['p1']}:1337", f"r0s={roots['r0s']}:1337", f"p2={roots['p2']}:42"]
    argv = []
    for source in sources:
        argv += ["--rft-source", source]
    return argv + ["--anchor-root", str(roots["anchors"]), "--train-scenes-file", str(roots["train"]),
                   "--holdout-scenes-file", str(roots["held"]), "--first-calls", "48", "--min-distance", "3",
                   "--max-anchors-per-episode", "6", "--anchor-ratio", "1.0", "--anchor-seed", "32",
                   "--max-trajectories-per-episode", "2", "--no-pad", "--seed", "42", "--shard-size", "8",
                   "--global-batch", "8", "--world-size", "1", "--workers", "1", "--out-dir", str(out), *extra]


# --------------------------------------------------------------------------- #
# RF2 / RF3 pieces
# --------------------------------------------------------------------------- #
def test_call_verdict_at_its_lines():
    pixel, left, right = ("↓", "245 320"), ("←←",), ("→",)
    assert brs.call_verdict(47, pixel, 6.0, 0) is None
    assert brs.call_verdict(48, pixel, 6.0, 0) == "beyond_first_calls"
    assert brs.call_verdict(10, pixel, None, 0) == "no_distance"
    assert brs.call_verdict(10, pixel, 3.0, 0) == "within_min_distance"
    assert brs.call_verdict(10, pixel, 3.0001, 0) is None
    assert brs.call_verdict(10, ("STOP",), 9.0, 0) == "class_stop"
    assert brs.call_verdict(10, ("",), 9.0, 0) == "class_stop"  # the server's fallback STOP
    assert brs.call_verdict(10, ("↑↑",), 9.0, 0) == "class_forward"
    assert brs.call_verdict(10, ("7",), 9.0, 0) == "class_invalid"
    assert brs.call_verdict(10, ("↓", "↓"), 9.0, 0) == "class_lookdown"
    assert brs.call_verdict(10, right, 9.0, 1) == "reversal" and brs.call_verdict(10, left, 9.0, -1) == "reversal"
    assert brs.call_verdict(10, left, 9.0, 1) is None and brs.call_verdict(10, right, 9.0, 0) is None
    assert brs.call_verdict(10, pixel, 9.0, 1) is None  # a pixel goal executes no turn
    assert brs.call_verdict(5, pixel, 2.0, 0, first_calls=4) == "beyond_first_calls"  # first failing rule counts
    assert brs.previous_direction({0: 1, 1: 0, 2: -1}, 2) == 0
    assert brs.previous_direction({0: 1, 3: -1}, 3) == 1  # a gap: the record's call just before
    assert brs.previous_direction({0: 1}, 0) == 0


def test_rft_calls_on_one_record(tmp_path):
    path = _record(tmp_path / "r", "kfPV7w3FaU5", 1, EVERY_REASON)
    kept, counts = brs.rft_calls(path)
    assert [c.call_index for c in kept] == [0, 2, 10, 11]
    assert [c.decision for c in kept] == ["turn_left", "pixel", "turn_left", "turn_left"]
    assert kept[1].turns == ("↓", "245 320") and kept[1].distance_to_goal == 6.0  # both turns, verbatim
    assert counts == Counter({"kept": 4, "reversal": 2, "within_min_distance": 2, "class_stop": 1, "class_forward": 1,
                              "class_invalid": 1, "class_lookdown": 1, "no_distance": 1})
    assert brs.rft_calls(EpisodeRecord(path))[0] == kept
    assert [c.call_index for c in brs.rft_calls(path, first_calls=3)[0]] == [0, 2]


def test_sources_and_seeds_are_parsed_against_a4():
    assert brs.parse_source("p1=/w/exp32/collect_a/records:1337") == ("p1", "/w/exp32/collect_a/records", 1337)
    assert brs.parse_source("p2=/a:b/records:42") == ("p2", "/a:b/records", 42)
    assert brs.parse_source("p3=/r:2024")[2] == 2024 and brs.parse_source("r0s=/r:1337")[0] == "r0s"
    for bad, match in (("p1=/r:42", "System1 seed 1337"), ("r0s=/r:42", "System1 seed 1337"),
                       ("p2=/r:1337", "System1 seed 42"), ("g=/r:42", "TAG=ROOT:SEED"), ("p1=/r", "TAG=ROOT:SEED"),
                       ("p1=/r:x", "not an integer")):
        with pytest.raises(SystemExit, match=match):
            brs.parse_source(bad)
    assert brs.seed_share_ok(1, 100) and brs.seed_share_ok(0, 0) and brs.seed_share_ok(1, 101)
    assert not brs.seed_share_ok(2, 100) and not brs.seed_share_ok(1, 99)
    assert brs.episode_shard({"episode_list": "/workspace/exp28/episodes/cohorts/shard_016.json"}) == 16
    assert brs.episode_shard({}) is None
    assert brs.episode_success({"outcome": {"success": 1.0}}) and not brs.episode_success({"outcome": {"success": 0.0}})
    assert not brs.episode_success({})


def test_anchor_draw_ranks_caps_per_episode_and_reports_a_small_pool(tmp_path):
    root = tmp_path / "a"
    for episode in (1, 2):
        _record(root, "kfPV7w3FaU5", episode, [("P", 6.0)] * 9, seed=42)
    refs, _ = index_calls([root], None, split="all", holdout=(), require_labels=False)
    pool = {}
    for ref in refs:
        pool.setdefault((ref.scene_id, ref.episode_id), []).append(ref)
    chosen, info = brs.draw_anchors(pool, 10, per_episode=6, anchor_seed=32)
    assert len(chosen) == 10 and max(Counter(r.episode_id for r in chosen).values()) <= 6
    ranks = sorted(brs.anchor_rank(32, r.uid) for r in refs)
    per_episode = {e: sorted(brs.anchor_rank(32, r.uid) for r in v)[:6] for e, v in pool.items()}
    assert sorted(brs.anchor_rank(32, r.uid) for r in chosen) == sorted(x for v in per_episode.values() for x in v)[:10]
    assert brs.anchor_rank(32, refs[0].uid) in ranks and info["anchors_capped_by_pool"] is False
    assert info["episodes_capped"] == 2 and info["pool_after_episode_cap"] == 12
    again, _ = brs.draw_anchors(pool, 10, per_episode=6, anchor_seed=32)
    assert again == chosen
    other, _ = brs.draw_anchors(pool, 10, per_episode=6, anchor_seed=7)
    assert {r.uid for r in other} != {r.uid for r in chosen}
    everything, info = brs.draw_anchors(pool, 20, per_episode=6, anchor_seed=32)
    assert len(everything) == 12 and info["anchors_capped_by_pool"] is True and info["wanted"] == 20


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
def _stream_samples(out):
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    return manifest, [s for shard in manifest["shards"] for s in read_shard(out / shard["name"])]


def test_end_to_end_rf1_to_rf7(rft_roots, tmp_path):
    out = tmp_path / "stream_rft"
    assert brs.main(_argv(rft_roots, out)) == 0
    manifest, samples = _stream_samples(out)
    assert manifest["schema"] == UNPADDED_SCHEMA and manifest["builder"] == brs.BUILDER
    assert manifest["builder"].startswith(ms.RFT_BUILDER)
    with pytest.raises(ValueError, match="schema"):
        read_manifest(out)  # the trainer refuses an unpadded stream

    # RF1: records and trajectories, with counts
    sources = manifest["selection"]["sources"]
    assert sources["p1"]["records"] == {"records": 5, "not_success": 1, "served_adapter": 1, "success": 3}
    assert sources["r0s"]["records"] == {"records": 3, "success": 3, "duplicate_of_p1": 1}
    assert sources["p2"]["records"] == {"records": 2, "success": 2, "over_episode_cap": 1}
    assert [sources[t]["trajectories"] for t in ("p1", "r0s", "p2")] == [3, 2, 1]
    assert sources["p1"]["trajectory_devices"] == {"npu": 2, "cuda": 1}
    # RF2 / RF3: calls, with counts
    assert sources["p1"]["calls"] == {"kept": 9, "reversal": 2, "within_min_distance": 2, "class_stop": 1,
                                      "class_forward": 1, "class_invalid": 1, "class_lookdown": 1, "no_distance": 1}
    assert sources["r0s"]["calls"] == {"kept": 5} and sources["p2"]["calls"] == {"kept": 2, "within_min_distance": 1}
    drawn = manifest["drawn"]
    assert {k: v for k, v in drawn.items() if k.startswith("category_")} == {
        "category_rft_p1": 9, "category_rft_r0s": 5, "category_rft_p2": 2, "category_rft_p3": 0, "category_anchor": 16}
    assert drawn["dagger"] == manifest["total_samples"] == len(samples) == 32
    assert drawn.get("dagger_target_stop", 0) == 0 and not any(s.turns[-1] == "STOP" for s in samples)

    # RF5: anchors 1:1, at most 6 per episode, from shard >= 20 successes only
    anchors = manifest["selection"]["anchors"]
    assert anchors["records"] == {"held_out_scene_skipped": 1, "records": 7, "shard_below_20": 1, "not_success": 1,
                                  "success": 5}
    assert anchors["calls"] == {"kept": 34, "reversal": 1, "class_stop": 1}
    assert anchors["draw"]["wanted"] == anchors["draw"]["taken"] == 16 and not anchors["draw"]["anchors_capped_by_pool"]
    anchor_keys = [s.key for s in samples if "#g." in s.key]
    assert len(anchor_keys) == 16 and max(Counter(k.split("#")[0] for k in anchor_keys).values()) <= 6
    assert not any(k.startswith(("kfPV7w3FaU5_000106", "AAAAAAAAAAA_000107", "5jLhtVmWd5F")) for k in anchor_keys)
    assert manifest["selection"]["composition"]["anchors_capped_by_pool"] is False

    # RF7: keys; RF4: targets and bytes are the record's
    assert all(ms.sample_scene(s.key) in {"kfPV7w3FaU5", "AAAAAAAAAAA", "BBBBBBBBBBB"} for s in samples)
    assert {s.key.split("#")[1].split(".")[0] for s in samples} == {"p1", "r0s", "p2", "g"}
    assert "BBBBBBBBBBB_000005#p1.0" in {s.key for s in samples} and "BBBBBBBBBBB_000005#r0s.1" in {s.key for s in samples}
    assert not any(s.key.startswith("AAAAAAAAAAA_000004#r0s") for s in samples)  # the duplicate
    rows = {line.split("\t")[0]: line.split("\t") for line in (out / brs.KEYS_TSV).read_text().splitlines()}
    assert list(rows) == [s.key for s in samples]  # stream order
    for sample in samples:
        _, uid, category, device = rows[sample.key]
        path, call_index = uid.rsplit("#", 1)
        record = EpisodeRecord(path)
        call = {int(c["call_index"]): c for c in record.calls}[int(call_index)]
        assert sample.turns == recorded_turns(call)
        assert sample.current_jpeg == record.front_jpeg(call["step_id"])
        assert sample.lookdown_jpeg == record.lookdown_jpeg(int(call_index))
        assert sample.history_jpegs == tuple(record.front_jpeg(s) for s in call["history_step_ids"])
        assert category == ("anchor" if "#g." in sample.key else "rft_" + sample.key.split("#")[1].split(".")[0])
        assert device == ("cuda" if path.endswith("BBBBBBBBBBB_000005.s2rec.tar") and "/p1/" in path else "npu")
    records = [line.split("\t") for line in (out / brs.RECORDS_LIST).read_text().splitlines()]
    assert sorted((Path(p).name, d, t) for p, d, t in records) == sorted([
        ("kfPV7w3FaU5_000001.s2rec.tar", "npu", "p1"), ("AAAAAAAAAAA_000004.s2rec.tar", "npu", "p1"),
        ("BBBBBBBBBBB_000005.s2rec.tar", "cuda", "p1"), ("BBBBBBBBBBB_000005.s2rec.tar", "npu", "r0s"),
        ("kfPV7w3FaU5_000006.s2rec.tar", "npu", "r0s"), ("BBBBBBBBBBB_000007.s2rec.tar", "npu", "p2")])

    # what merge_streams reads back: the keys hash, the shards, the scenes
    loaded, entries, _ = ms.load_stream(out, schemas=(UNPADDED_SCHEMA,), verify=True)
    assert ms.dagger_keys_sha256(entries) == manifest["selection"]["selected_keys_sha256"]
    assert [len(read_shard(out / s["name"])) for s in manifest["shards"]] == [8, 8, 8, 8]
    assert manifest["selection"]["train_scenes"]["scenes"] == ["AAAAAAAAAAA", "BBBBBBBBBBB", "kfPV7w3FaU5"]
    assert manifest["label_roots"] == [] and len(manifest["record_roots"]) == 4
    assert manifest["holdout"]["scenes"] == [HELD]
    assert manifest["selection"]["composition"]["reversals"] == {"dropped_rft": 2, "dropped_anchors": 1,
                                                                 "reversed_targets_kept": 0}

    # the same inputs give the same stream
    again = tmp_path / "again"
    assert brs.main(_argv(rft_roots, again)) == 0
    assert json.loads((again / MANIFEST_NAME).read_text())["selection"]["selected_keys_sha256"] == \
        manifest["selection"]["selected_keys_sha256"]
    with pytest.raises(SystemExit, match="already holds"):
        brs.main(_argv(rft_roots, out))


def test_rft_calls_keeps_exactly_what_the_builder_packs(rft_roots, tmp_path):
    out = tmp_path / "stream_rft"
    assert brs.main(_argv(rft_roots, out)) == 0
    packed = Counter()
    for line in (out / brs.KEYS_TSV).read_text().splitlines():
        key, uid, category, _ = line.split("\t")
        if category != "anchor":
            path, call_index = uid.rsplit("#", 1)
            packed[path] += 1
            assert int(call_index) in {c.call_index for c in brs.rft_calls(path)[0]}
    for line in (out / brs.RECORDS_LIST).read_text().splitlines():
        path = line.split("\t")[0]
        assert packed[path] == len(brs.rft_calls(path)[0])


def test_dry_run_prints_the_dose_and_writes_nothing(rft_roots, tmp_path, capsys):
    out = tmp_path / "dry"
    assert brs.main(_argv(rft_roots, out, "--dry-run")) == 0
    assert not out.exists()
    printed = capsys.readouterr().out
    first = json.loads(printed[: printed.index("\n}\n") + 2])
    assert sum(v for k, v in first["drawn"].items() if k.startswith("category_rft_")) == 16 == \
        first["composition"]["n_rft"]


def test_a_seed_mismatch_is_refused(rft_roots, tmp_path, monkeypatch):
    _record(rft_roots["p1"], "kfPV7w3FaU5", 9, [("P", 6.0)] * 3, call_seeds={1: 42})  # one call reseeded
    with pytest.raises(SystemExit, match="another System1 seed than 1337"):
        brs.main(_argv(rft_roots, tmp_path / "out", "--dry-run"))
    # at most 1% is left out and counted (here the line is moved to 1/2 to show it)
    monkeypatch.setattr(brs, "MAX_SEED_MISMATCH_SHARE", Fraction(1, 2))
    out = tmp_path / "out2"
    assert brs.main(_argv(rft_roots, out)) == 0
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert manifest["selection"]["sources"]["p1"]["records"]["seed_mismatch"] == 1
    assert not any(s.key.startswith("kfPV7w3FaU5_000009") for s in _stream_samples(out)[1])


def test_an_anchor_of_another_seed_is_refused(rft_roots, tmp_path):
    _record(rft_roots["anchors"], "kfPV7w3FaU5", 120, [("P", 6.0)] * 3, seed=1337, shard=30)
    with pytest.raises(SystemExit, match="another System1 seed than 42"):
        brs.main(_argv(rft_roots, tmp_path / "out", "--dry-run"))


def test_a_declared_seed_off_a4_is_refused(rft_roots, tmp_path):
    with pytest.raises(SystemExit, match="System1 seed 1337"):
        brs.main(_argv(rft_roots, tmp_path / "out", "--dry-run", sources=[f"p1={rft_roots['p1']}:42"]))
    with pytest.raises(SystemExit, match="tag twice"):
        brs.main(_argv(rft_roots, tmp_path / "out", "--dry-run",
                       sources=[f"p1={rft_roots['p1']}:1337", f"p1={rft_roots['r0s']}:1337"]))


@pytest.mark.parametrize("where, scene, match", [
    ("p1", "5jLhtVmWd5F", "held-out scenes"),
    ("r0s", "17DRP5sb8fy", "held-out scenes"),  # EXP-23's six R2R scenes are held out too
    ("p2", "ZZZZZZZZZZZ", "outside the training list"),
    ("anchors", "ZZZZZZZZZZZ", "outside the training list"),
])
def test_held_out_and_other_scenes_are_refused(rft_roots, tmp_path, where, scene, match):
    if scene == "17DRP5sb8fy":
        scene = holdout_scenes()[0]
    seed = 42 if where in ("p2", "anchors") else 1337
    _record(rft_roots[where], scene, 77, [("P", 6.0)] * 2, seed=seed, shard=30)
    with pytest.raises(SystemExit, match=match):
        brs.main(_argv(rft_roots, tmp_path / "out", "--dry-run"))


def test_a_training_list_naming_a_held_out_scene_is_refused(rft_roots, tmp_path):
    rft_roots["train"].write_text("kfPV7w3FaU5\nAAAAAAAAAAA\nBBBBBBBBBBB\n5jLhtVmWd5F\n")
    with pytest.raises(SystemExit, match="lists held-out scenes"):
        brs.main(_argv(rft_roots, tmp_path / "out", "--dry-run"))


def test_a_small_anchor_pool_is_taken_whole_and_reported(rft_roots, tmp_path, capsys):
    small = tmp_path / "small_anchors"
    _record(small, "BBBBBBBBBBB", 108, [("L", 6.0), ("R", 6.0), ("S", 6.0), ("P", 6.0)], seed=42, shard=40)
    out = tmp_path / "out"
    assert brs.main(_argv({**rft_roots, "anchors": small}, out)) == 0
    assert "16 anchors wanted, the capped pool holds 2" in capsys.readouterr().err
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert manifest["drawn"]["category_anchor"] == 2 and manifest["total_samples"] == 18
    assert manifest["selection"]["composition"]["anchors_capped_by_pool"] is True


def test_an_empty_or_missing_source_is_refused(rft_roots, tmp_path):
    (tmp_path / "empty").mkdir()
    for root, match in ((tmp_path / "empty", "no committed record"), (tmp_path / "missing", "not a directory")):
        with pytest.raises(SystemExit, match=match):
            brs.main(_argv(rft_roots, tmp_path / "out", "--dry-run",
                           sources=[f"p1={rft_roots['p1']}:1337", f"p3={root}:2024"]))


def test_no_pad_is_required(rft_roots, tmp_path):
    argv = [a for a in _argv(rft_roots, tmp_path / "out") if a != "--no-pad"]
    with pytest.raises(SystemExit, match="--no-pad is required"):
        brs.main(argv)


# --------------------------------------------------------------------------- #
# Through T1: merge_streams.py with one r2r replay root at 20%
# --------------------------------------------------------------------------- #
def test_the_rft_stream_merges_end_to_end(rft_roots, tmp_path):
    rft = tmp_path / "stream_rft"
    assert brs.main(_argv(rft_roots, rft)) == 0
    replay = _replay_stream(tmp_path, _replay_samples(R2R_SCENES, 12, 1, np.random.default_rng(0)),
                            root_scenes={"r2r": R2R_SCENES})
    out = tmp_path / "stream"
    argv = ["--dagger-stream", str(rft), "--replay-stream", str(replay), "--holdout-scenes-file", str(rft_roots["held"]),
            "--replay-fraction", "0.2", "--replay-shares", "equal", "--out-dir", str(out), "--shard-size", "8",
            "--global-batch", "8", "--world-size", "1", "--workers", "1"]
    with pytest.raises(SystemExit, match="EXP-28's rules"):
        ms.main(argv)
    assert ms.main(argv + ["--allow-other-rules"]) == 0
    manifest = read_manifest(out)  # the trainer's reader accepts the merged stream
    total, n_replay = ms.stream_size(32, 0.2, 8)
    assert (manifest["total_samples"], manifest["drawn"]["replay"]) == (total, n_replay) == (40, 8)
    assert manifest["drawn"]["category_rft_p1"] == 9 and manifest["drawn"]["category_anchor"] == 16
    assert manifest["inputs"]["dagger"]["builder"] == brs.BUILDER
    assert "5jLhtVmWd5F" in manifest["holdout"]["scenes"]
    samples = [s for shard in manifest["shards"] for s in read_shard(out / shard["name"])]
    assert Counter(s.source for s in samples) == {"dagger": 32, "replay": 8}
    assert not any(s.turns[-1] == "STOP" for s in samples if s.source == "dagger")

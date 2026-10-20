"""EXP-28 merge of the line-B DAgger stream with the expert-replay stream (ledger R7, R8)."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

import numpy as np
import pytest

from scripts.exp28 import merge_streams as ms
from scripts.tools import build_route_stream as rs
from src.data.s2_route_labels import RouteLabelConfig
from src.data.s2_sft_dataset import R2R_TRAIN_SCENES, holdout_scenes
from src.data.s2_stream import MANIFEST_NAME, STREAM_SCHEMA, PackedSample, read_manifest, read_shard, shard_name, write_shard
from tests.test_exp28_route_stream import exp28_argv, exp28_roots  # noqa: F401  (fixture)

R2R_SCENES = [s for s in R2R_TRAIN_SCENES if s not in holdout_scenes()][:2]
HM3D_SCENES = ["00100-AAAAAAAAAAA", "00101-BBBBBBBBBBB"]


def _sample(key, source, turns, n_history=2):
    return PackedSample(key=key, source=source, instruction=f"walk to {key}", turns=tuple(turns),
                        vlm_image_size=(384, 384), history_jpegs=tuple(f"h{k}|{key}".encode() for k in range(n_history)),
                        current_jpeg=f"cur|{key}".encode(), lookdown_jpeg=f"look|{key}".encode())


def _write_stream(out, samples, shard_size, manifest):
    out.mkdir(parents=True)
    shards = []
    for s in range(-(-len(samples) // shard_size)):
        digest = write_shard(out / shard_name(s), samples[s * shard_size:(s + 1) * shard_size])
        shards.append({"name": shard_name(s), "sha256": digest})
    manifest = {"total_samples": len(samples), "shards": shards, "shard_size": shard_size, **manifest}
    (out / MANIFEST_NAME).write_text(json.dumps(manifest))
    return out


DEFAULT_LABELER = RouteLabelConfig().digest()


def _dagger_stream(out, samples, shard_size=8, holdout=("00481-5jLhtVmWd5F",), knobs=None,
                   train=("kfPV7w3FaU5",), labelers=None, anchors_capped=False, builder=rs.BUILDER, labels=True,
                   drawn=None):
    """A DAgger stream manifest as build_route_stream.py --no-pad writes it (the fields merge reads).
    ``builder`` / ``labels=False`` / ``drawn`` give the RFT stream's (build_rft_stream.py: no labels)."""
    keys = hashlib.sha256("\n".join(s.key + "|" + "\x1f".join(s.turns) for s in samples).encode()).hexdigest()
    knobs = dict({**ms.EXP28_DAGGER_KNOBS, "train_scenes_file": "/x/train_scenes.txt"} if knobs is None else knobs)
    selection = {"selected_keys_sha256": keys, "knobs": knobs, "train_scenes": {"scenes": list(train)},
                 "composition": {"anchors_capped_by_pool": anchors_capped}}
    if labels:
        selection["labels"] = {"labelers": labelers or {DEFAULT_LABELER: 1}, "lines": {"B": 1}}
    return _write_stream(out, samples, shard_size, {
        "schema": rs.UNPADDED_SCHEMA, "builder": builder, "seed": 42, "replay_fraction": 0.0,
        "holdout": {"scenes": list(holdout)}, "record_roots": ["/r"], "label_roots": ["/l"] if labels else [],
        "replay_root": "", "selection": selection,
        "drawn": drawn or {"dagger": len(samples), "category_B_anchor": len(samples)}})


def _replay_samples(scenes, per_scene, stops_per_scene, rng):
    """per scene: ``per_scene`` pixel/turn samples and ``stops_per_scene`` STOPs x5."""
    out = []
    for scene in scenes:
        for i in range(per_scene):
            turns = ("↓", f"{200 + i} 300") if i % 2 else ("←←",)
            out.append(_sample(f"replay:{scene}_{i:06d}@{4 * i}:abcd{i:04d}", "replay", turns))
        for j in range(stops_per_scene):
            out += [_sample(f"replay:{scene}_{100 + j:06d}@99:stop{j:04d}", "replay", ("STOP",))] * 5
    return [out[int(i)] for i in rng.permutation(len(out))]


def _replay_stream(tmp_path, samples, shard_size=8, stop_upsample=5, holdout=(), root_scenes=None):
    """``root_scenes``: replay root name -> its scene dirs (default: r2r and ScaleVLN, EXP-28's R8)."""
    roots = {}
    root_scenes = root_scenes or {"r2r": R2R_SCENES, "scalevln": HM3D_SCENES + ["00481-5jLhtVmWd5F"]}
    for name, scenes in sorted(root_scenes.items()):
        roots[name] = tmp_path / f"root_{name}"
        for scene in scenes:
            (roots[name] / scene).mkdir(parents=True)
    keys = hashlib.sha256("\n".join(s.key for s in samples).encode()).hexdigest()
    roots = {k: str(v) for k, v in roots.items()}
    return _write_stream(tmp_path / "stream_replay", samples, shard_size, {
        "schema": STREAM_SCHEMA, "builder": "scripts/tools/build_replay_stream.py (EXP-26)", "seed": 42,
        "replay_fraction": 1.0, "roots": roots, "record_roots": [], "label_roots": [],
        "replay_root": json.dumps(roots, sort_keys=True), "holdout": {"scenes": sorted(holdout)},
        "selection": {"stop_upsample": stop_upsample, "keys_sha256": keys}, "drawn": {}})


@pytest.fixture
def replay_stream(tmp_path):
    samples = _replay_samples(R2R_SCENES + HM3D_SCENES, per_scene=12, stops_per_scene=2, rng=np.random.default_rng(0))
    samples += [_sample("replay:00100-AAAAAAAAAAA_000777@50:cut00000", "replay", ("STOP",))] * 3  # trimmed copies
    return _replay_stream(tmp_path, samples, holdout=("00481-5jLhtVmWd5F", *holdout_scenes()))


def _merge_argv(dagger, replay, held, out, *extra):
    return ["--dagger-stream", str(dagger), "--replay-stream", str(replay), "--holdout-scenes-file", str(held),
            "--out-dir", str(out), "--shard-size", "8", "--global-batch", "8", "--world-size", "1",
            "--workers", "1", *extra]


# --------------------------------------------------------------------------- #
# Pieces
# --------------------------------------------------------------------------- #
def test_stream_size_puts_replay_at_least_at_its_fraction_in_whole_shards():
    assert ms.stream_size(14, 0.3, 8) == (24, 10)  # 6 needed, padded to whole shards
    assert ms.stream_size(7, 0.3, 10) == (10, 3)  # 7 / 0.7 is 10 up to float noise
    total, replay = ms.stream_size(18000, 0.3, 256)
    assert total % 256 == 0 and replay >= 0.3 * total and replay - 256 < 0.3 * total


def test_apportion_and_shares():
    assert ms.apportion(11, {"r2r": 1.0, "scalevln": 1.0}) == {"r2r": 6, "scalevln": 5}
    assert ms.apportion(10, {"a": 0.25, "b": 0.75}) == {"a": 3, "b": 7}
    assert ms.parse_shares("equal", ["r2r", "scalevln"]) == {"r2r": 1.0, "scalevln": 1.0}
    assert ms.parse_shares("pooled", ["r2r", "scalevln"]) == {"all": 1.0}
    assert ms.parse_shares("r2r=0.5,scalevln=0.5", ["r2r", "scalevln"]) == {"r2r": 0.5, "scalevln": 0.5}
    with pytest.raises(SystemExit, match="roots are"):
        ms.parse_shares("r2r=1", ["r2r", "scalevln"])


def test_sample_scene_from_keys():
    assert ms.sample_scene("kfPV7w3FaU5_000012#3") == "kfPV7w3FaU5"
    assert ms.sample_scene("replay:00481-5jLhtVmWd5F_000004@16:1a2b3c4d") == "00481-5jLhtVmWd5F"
    assert ms.sample_scene("replay:17DRP5sb8fy_000004@16:1a2b3c4d") == "17DRP5sb8fy"


def test_read_entry_is_read_shard_byte_for_byte(tmp_path):
    samples = [_sample(f"s_{i:06d}#0", "dagger", ("←",), n_history=i % 3) for i in range(5)]
    write_shard(tmp_path / "x.tar", samples)
    entries = ms.index_shard(tmp_path / "x.tar")
    assert [ms.read_entry(e) for e in entries] == read_shard(tmp_path / "x.tar") == samples


def test_replay_draw_keeps_stop_copies_whole_and_splits_by_root():
    def entry(key, turns, i):
        return ms.Entry("x.tar", i, key, "replay", turns, ())

    units = {"a": [[entry(f"a{i}", ("←",), i)] for i in range(10)] + [[entry("aS", ("STOP",), 99)] * 5],
             "b": [[entry(f"b{i}", ("←",), i)] for i in range(10)]}
    chosen, counts = ms.draw_replay(units, 12, {"a": 1.0, "b": 1.0}, np.random.default_rng(0))
    assert counts["a"] == 6 and counts["b"] == 6 and len(chosen) == 12
    assert Counter(e.key for e in chosen)["aS"] in (0, 5)
    with pytest.raises(SystemExit, match="larger replay stream"):
        ms.draw_replay(units, 40, {"a": 1.0, "b": 1.0}, np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
def test_merge_end_to_end(exp28_roots, replay_stream, tmp_path):  # noqa: F811
    records, labels, held = exp28_roots
    dagger = tmp_path / "stream_dagger"
    assert rs.main(exp28_argv(records, labels, held, dagger)) == 0
    out = tmp_path / "stream"
    assert ms.main(_merge_argv(dagger, replay_stream, held, out)) == 0

    manifest = read_manifest(out)  # the trainer's reader accepts it
    dagger_manifest = json.loads((dagger / MANIFEST_NAME).read_text())
    replay_manifest = json.loads((replay_stream / MANIFEST_NAME).read_text())
    assert manifest["record_roots"] == dagger_manifest["record_roots"] and manifest["record_roots"]
    assert manifest["label_roots"] == dagger_manifest["label_roots"]
    assert manifest["replay_root"] == replay_manifest["replay_root"]
    assert manifest["holdout"]["scenes"] == sorted({"5jLhtVmWd5F", *holdout_scenes()})
    assert manifest["selection"]["dagger_rules"]["mismatches"] == {}  # build_route_stream's EXP-28 flags
    assert manifest["selection"]["merge_rules"]["mismatches"] == {}
    assert manifest["selection"]["composition"]["replay_by_dataset"] == {"r2r": 6, "hm3d": 6}

    shards = [read_shard(out / s["name"]) for s in manifest["shards"]]
    assert all(len(s) == 8 for s in shards) and manifest["total_samples"] == 8 * len(shards)
    samples = [x for shard in shards for x in shard]
    n_dagger, n_replay = sum(s.source == "dagger" for s in samples), sum(s.source == "replay" for s in samples)
    assert n_dagger == dagger_manifest["total_samples"] == 12
    assert (manifest["total_samples"], n_replay) == ms.stream_size(12, 0.3, 8) == (24, 12)
    assert manifest["replay_fraction"] == 0.5 and manifest["drawn"]["replay"] == 12
    assert manifest["drawn"]["replay_root_r2r"] == manifest["drawn"]["replay_root_scalevln"] == 6

    # every sample is its input's bytes; STOP copies whole; the cut-short STOP never drawn
    sources = {s.key: s for d in (dagger, replay_stream)
               for shard in json.loads((d / MANIFEST_NAME).read_text())["shards"] for s in read_shard(d / shard["name"])}
    assert all(sources[s.key] == s for s in samples)
    copies = Counter(s.key for s in samples if s.turns == ("STOP",))
    assert all(n == 5 for n in copies.values()) and "replay:00100-AAAAAAAAAAA_000777@50:cut00000" not in copies
    assert not any(s.turns[-1] == "STOP" for s in samples if s.source == "dagger")
    assert manifest["selection"]["composition"]["replay_units"]["stop_units_cut_short"] == 1

    with pytest.raises(SystemExit, match="already holds"):
        ms.main(_merge_argv(dagger, replay_stream, held, out))


def test_the_draw_is_seeded(exp28_roots, replay_stream, tmp_path):  # noqa: F811
    records, labels, held = exp28_roots
    dagger = tmp_path / "stream_dagger"
    rs.main(exp28_argv(records, labels, held, dagger))
    keys = []
    for name in ("a", "b"):
        ms.main(_merge_argv(dagger, replay_stream, held, tmp_path / name))
        keys.append(json.loads((tmp_path / name / MANIFEST_NAME).read_text())["selection"]["keys_sha256"])
    assert keys[0] == keys[1]


def test_held_out_scenes_in_replay_are_refused(tmp_path):
    dagger = _dagger_stream(tmp_path / "d", [_sample(f"kfPV7w3FaU5_000001#{i}", "dagger", ("←",)) for i in range(8)])
    held = tmp_path / "held.txt"
    held.write_text("5jLhtVmWd5F\n")  # the bare ID matches the replay's 00481-... directory
    leak = _replay_samples(["00481-5jLhtVmWd5F"], per_scene=4, stops_per_scene=0, rng=np.random.default_rng(0))
    ok = _replay_samples(HM3D_SCENES, per_scene=12, stops_per_scene=0, rng=np.random.default_rng(1))
    replay = _replay_stream(tmp_path, ok + leak)
    with pytest.raises(SystemExit, match="held-out scenes"):
        ms.main(_merge_argv(dagger, replay, held, tmp_path / "out"))


def test_exp23_holdout_in_replay_is_refused(tmp_path):
    dagger = _dagger_stream(tmp_path / "d", [_sample(f"kfPV7w3FaU5_000001#{i}", "dagger", ("←",)) for i in range(8)])
    held = tmp_path / "held.txt"
    held.write_text("00481-5jLhtVmWd5F\n")
    leak = _replay_samples([holdout_scenes()[0]], per_scene=4, stops_per_scene=0, rng=np.random.default_rng(0))
    replay = _replay_stream(tmp_path, leak + _replay_samples(HM3D_SCENES, 12, 0, np.random.default_rng(1)))
    with pytest.raises(SystemExit, match="held-out scenes"):
        ms.main(_merge_argv(dagger, replay, held, tmp_path / "out"))


@pytest.mark.parametrize("case, match", [
    ("dagger_stop", "target STOP"),
    ("dagger_replay", "non-DAgger"),
    ("upsample", "repeats each STOP"),
    ("small", "larger replay stream"),
    ("corrupt", "sha256"),
])
def test_bad_inputs_are_refused(tmp_path, case, match):
    dagger_samples = [_sample(f"kfPV7w3FaU5_000001#{i}", "dagger", ("←",)) for i in range(16)]
    if case == "dagger_stop":
        dagger_samples[3] = _sample("kfPV7w3FaU5_000001#3", "dagger", ("STOP",))
    if case == "dagger_replay":
        dagger_samples[3] = _sample("replay:00100-AAAAAAAAAAA_000001@4:abcd0001", "replay", ("←",))
    dagger = _dagger_stream(tmp_path / "d", dagger_samples)
    held = tmp_path / "held.txt"
    held.write_text("00481-5jLhtVmWd5F\n")
    per_scene = 1 if case == "small" else 12
    replay = _replay_stream(tmp_path, _replay_samples(R2R_SCENES + HM3D_SCENES, per_scene, 1, np.random.default_rng(0)),
                            stop_upsample=3 if case == "upsample" else 5)
    if case == "corrupt":
        shard = dagger / shard_name(0)
        shard.write_bytes(shard.read_bytes()[:-1] + b"\x01")
    with pytest.raises(SystemExit, match=match):
        ms.main(_merge_argv(dagger, replay, held, tmp_path / "out"))


def test_a_dagger_stream_without_exp28_rules_is_refused_unless_allowed(tmp_path):
    knobs = {**ms.EXP28_DAGGER_KNOBS, "drop_reversals": False, "train_scenes_file": "/x/train_scenes.txt"}
    knobs.pop("first_calls")
    dagger = _dagger_stream(tmp_path / "d", [_sample(f"kfPV7w3FaU5_000001#{i}", "dagger", ("←",)) for i in range(8)],
                            knobs=knobs)
    held = tmp_path / "held.txt"
    held.write_text("00481-5jLhtVmWd5F\n")
    replay = _replay_stream(tmp_path, _replay_samples(R2R_SCENES + HM3D_SCENES, 12, 1, np.random.default_rng(0)))
    assert ms.rule_mismatches(knobs) == {"first_calls": None, "drop_reversals": False}
    with pytest.raises(SystemExit, match="EXP-28's rules"):
        ms.main(_merge_argv(dagger, replay, held, tmp_path / "out"))
    assert ms.main(_merge_argv(dagger, replay, held, tmp_path / "out", "--allow-other-rules")) == 0
    rules = read_manifest(tmp_path / "out")["selection"]["dagger_rules"]
    assert rules["mismatches"] == {"first_calls": None, "drop_reversals": False}


def test_pooled_shares_need_no_root_dirs(tmp_path):
    dagger = _dagger_stream(tmp_path / "d", [_sample(f"kfPV7w3FaU5_000001#{i}", "dagger", ("←",)) for i in range(8)])
    held = tmp_path / "held.txt"
    held.write_text("00481-5jLhtVmWd5F\n")
    replay = _replay_stream(tmp_path, _replay_samples(HM3D_SCENES, 12, 1, np.random.default_rng(0)))
    for name in ("r2r", "scalevln"):
        for scene_dir in (tmp_path / f"root_{name}").iterdir():
            scene_dir.rmdir()
        (tmp_path / f"root_{name}").rmdir()
    with pytest.raises(SystemExit, match="not a directory"):
        ms.main(_merge_argv(dagger, replay, held, tmp_path / "out", "--dry-run"))
    with pytest.raises(SystemExit, match="not EXP-28's R8"):  # one pool: no r2r / ScaleVLN halves
        ms.main(_merge_argv(dagger, replay, held, tmp_path / "out", "--replay-shares", "pooled"))
    assert ms.main(_merge_argv(dagger, replay, held, tmp_path / "out", "--replay-shares", "pooled",
                               "--allow-other-rules")) == 0
    manifest = read_manifest(tmp_path / "out")
    assert manifest["total_samples"] == 16 and manifest["drawn"]["replay_root_all"] == 8
    assert manifest["selection"]["merge_rules"]["mismatches"] == {"replay_by_dataset": {"hm3d": 8}}


# --------------------------------------------------------------------------- #
# The rules the merge holds the inputs to
# --------------------------------------------------------------------------- #
def _small_inputs(tmp_path, **dagger_kwargs):
    dagger = _dagger_stream(tmp_path / "d", [_sample(f"kfPV7w3FaU5_000001#{i}", "dagger", ("←",)) for i in range(16)],
                            **dagger_kwargs)
    held = tmp_path / "held.txt"
    held.write_text("00481-5jLhtVmWd5F\n")
    replay = _replay_stream(tmp_path, _replay_samples(R2R_SCENES + HM3D_SCENES, 12, 1, np.random.default_rng(0)))
    return dagger, replay, held


def test_replay_dataset_reads_the_scene_of_the_key():
    assert ms.replay_dataset("replay:00481-5jLhtVmWd5F_000004@16:1a2b3c4d") == "hm3d"
    assert ms.replay_dataset(f"replay:{R2R_SCENES[0]}_000004@16:1a2b3c4d") == "r2r"
    assert ms.replay_dataset("replay:2azQ1b91cZZ_000004@16:1a2b3c4d") == "other"  # MP3D val_unseen
    assert ms.merge_rule_mismatches(0.3, 5, {"r2r": 7, "hm3d": 6}) == {}
    assert ms.merge_rule_mismatches(0.3, 5, {"r2r": 8, "hm3d": 6}) == {"replay_by_dataset": {"r2r": 8, "hm3d": 6}}
    assert ms.merge_rule_mismatches(0.2, 1, {"r2r": 1, "hm3d": 1}) == {"replay_fraction": 0.2, "stop_upsample": 1}


@pytest.mark.parametrize("dagger_kwargs, key", [
    ({"labelers": {"0123456789ab": 1}}, "labelers"),
    ({"labelers": {DEFAULT_LABELER: 3, "0123456789ab": 1}}, "labelers"),
    ({"anchors_capped": True}, "anchors_capped_by_pool"),
    ({"knobs": dict(ms.EXP28_DAGGER_KNOBS)}, "train_scenes_file"),
    ({"knobs": {**ms.EXP28_DAGGER_KNOBS, "train_scenes_file": "/t", "min_distance": None}}, "min_distance"),
    ({"knobs": {**ms.EXP28_DAGGER_KNOBS, "train_scenes_file": "/t", "no_pad": False}}, "no_pad"),
])
def test_a_dagger_stream_off_the_rules_is_refused(tmp_path, dagger_kwargs, key):
    dagger, replay, held = _small_inputs(tmp_path, **dagger_kwargs)
    with pytest.raises(SystemExit, match="EXP-28's rules"):
        ms.main(_merge_argv(dagger, replay, held, tmp_path / "out"))
    assert ms.main(_merge_argv(dagger, replay, held, tmp_path / "out", "--allow-other-rules")) == 0
    assert key in read_manifest(tmp_path / "out")["selection"]["dagger_rules"]["mismatches"]


@pytest.mark.parametrize("extra", [
    ("--replay-shares", "r2r=1,scalevln=0"),  # all r2r
    ("--replay-fraction", "0.2"),
])
def test_a_draw_off_r8_is_refused(tmp_path, extra):
    dagger, replay, held = _small_inputs(tmp_path)
    with pytest.raises(SystemExit, match="not EXP-28's R8"):
        ms.main(_merge_argv(dagger, replay, held, tmp_path / "out", *extra, "--dry-run"))


def test_a_dagger_sample_outside_the_training_scenes_is_refused_always(tmp_path):
    dagger, replay, held = _small_inputs(tmp_path, train=("someOtherID",))
    for extra in ((), ("--allow-other-rules",)):
        with pytest.raises(SystemExit, match="outside the stream's training scenes"):
            ms.main(_merge_argv(dagger, replay, held, tmp_path / "out", *extra))


# --------------------------------------------------------------------------- #
# EXP-32 (T1): the RFT stream as the DAgger part, one r2r replay root at 20%
# --------------------------------------------------------------------------- #
RFT_BUILDER_STRING = "scripts/exp32/build_rft_stream.py (EXP-32)"
# the knobs build_rft_stream.py writes: not EXP-28's (no labels, 6 anchors per episode, no R2 cap)
RFT_KNOBS = {"first_calls": 48, "min_distance": 3.0, "drop_reversals": True, "no_pad": True,
             "max_anchors_per_episode": 6, "anchor_ratio": 1.0, "anchor_seed": 32, "max_trajectories_per_episode": 2,
             "train_scenes_file": "/x/train_scenes.txt", "holdout_scenes_file": "/x/heldout_scenes.txt"}


def _rft_samples(n, stop_at=None):
    out = []
    for i in range(n):
        tag, turns = ("g", ("↓", f"{200 + i % 50} 300")) if i % 2 else ("p1", ("←←",))
        if i == stop_at:
            turns = ("STOP",)
        out.append(_sample(f"kfPV7w3FaU5_{i // 10:06d}#{tag}.{i % 10}", "dagger", turns))
    return out


def _rft_stream(out, samples, shard_size=8):
    drawn = {"dagger": len(samples), "category_rft_p1": sum("#p1." in s.key for s in samples),
             "category_rft_r0s": 0, "category_rft_p2": 0, "category_rft_p3": 0,
             "category_anchor": sum("#g." in s.key for s in samples)}
    return _dagger_stream(out, samples, shard_size=shard_size, knobs=RFT_KNOBS, builder=RFT_BUILDER_STRING,
                          labels=False, drawn=drawn)


def _r2r_only_replay(tmp_path, per_scene=12, stops_per_scene=1, shard_size=8):
    samples = _replay_samples(R2R_SCENES, per_scene, stops_per_scene, np.random.default_rng(0))
    return _replay_stream(tmp_path, samples, shard_size=shard_size, root_scenes={"r2r": R2R_SCENES})


def test_rft_keys_parse_to_their_scene():
    assert ms.RFT_BUILDER == "scripts/exp32/build_rft_stream.py" and RFT_BUILDER_STRING.startswith(ms.RFT_BUILDER)
    for tag in ("p1", "r0s", "p2", "p3", "g"):
        assert ms.sample_scene(f"kfPV7w3FaU5_000012#{tag}.37") == "kfPV7w3FaU5"
        assert ms.sample_scene(f"00100-AAAAAAAAAAA_306510#{tag}.0") == "00100-AAAAAAAAAAA"


def test_an_rft_stream_merges_only_with_allow_other_rules(tmp_path):
    dagger = _rft_stream(tmp_path / "d", _rft_samples(16))
    held = tmp_path / "held.txt"
    held.write_text("00481-5jLhtVmWd5F\n")
    replay = _r2r_only_replay(tmp_path)
    argv = _merge_argv(dagger, replay, held, tmp_path / "out", "--replay-fraction", "0.2")
    with pytest.raises(SystemExit, match="EXP-28's rules"):
        ms.main(argv)
    assert ms.main(argv + ["--allow-other-rules"]) == 0
    manifest = read_manifest(tmp_path / "out")
    assert manifest["inputs"]["dagger"]["builder"] == RFT_BUILDER_STRING
    mismatches = manifest["selection"]["dagger_rules"]["mismatches"]
    assert mismatches["max_anchors_per_episode"] == 6 and mismatches["labelers"] is None
    assert manifest["selection"]["merge_rules"]["mismatches"]["replay_fraction"] == 0.2
    assert manifest["label_roots"] == [] and manifest["record_roots"] == ["/r"]
    assert {k: v for k, v in manifest["drawn"].items() if k.startswith("category_rft")} == {
        "category_rft_p1": 8, "category_rft_r0s": 0, "category_rft_p2": 0, "category_rft_p3": 0}
    assert manifest["drawn"]["category_anchor"] == 8
    samples = [s for shard in manifest["shards"] for s in read_shard(tmp_path / "out" / shard["name"])]
    assert sum(s.source == "dagger" for s in samples) == 16 and not any(
        s.turns[-1] == "STOP" for s in samples if s.source == "dagger")


def test_a_stop_target_in_an_rft_stream_is_refused_always(tmp_path):
    dagger = _rft_stream(tmp_path / "d", _rft_samples(16, stop_at=5))
    held = tmp_path / "held.txt"
    held.write_text("00481-5jLhtVmWd5F\n")
    replay = _r2r_only_replay(tmp_path)
    with pytest.raises(SystemExit, match="target STOP"):
        ms.main(_merge_argv(dagger, replay, held, tmp_path / "out", "--replay-fraction", "0.2", "--allow-other-rules"))


@pytest.mark.parametrize("builder", ["scripts/tools/build_s2_train_stream.py", "scripts/exp32/other.py (EXP-32)", ""])
def test_an_unknown_builder_is_still_refused(tmp_path, builder):
    dagger = _dagger_stream(tmp_path / "d", _rft_samples(16), knobs=RFT_KNOBS, builder=builder, labels=False)
    held = tmp_path / "held.txt"
    held.write_text("00481-5jLhtVmWd5F\n")
    replay = _r2r_only_replay(tmp_path)
    with pytest.raises(SystemExit, match="was built by"):
        ms.main(_merge_argv(dagger, replay, held, tmp_path / "out", "--replay-fraction", "0.2", "--allow-other-rules"))


@pytest.mark.parametrize("n", [1, 3, 4, 255, 256, 300, 1023, 7497, 15000])
def test_a_fifth_replay_gives_whole_shards_of_n_plus_a_quarter(n):
    total, replay = ms.stream_size(n, 0.2, 256)
    assert total == math.ceil((n + math.ceil(n / 4)) / 256) * 256
    assert replay == total - n and replay >= math.ceil(n / 4)


def test_one_r2r_root_at_a_fifth_end_to_end(tmp_path):
    n = 300
    dagger = _rft_stream(tmp_path / "d", _rft_samples(n), shard_size=256)
    held = tmp_path / "held.txt"
    held.write_text("00481-5jLhtVmWd5F\n")
    replay = _r2r_only_replay(tmp_path, per_scene=110, stops_per_scene=2, shard_size=256)
    out = tmp_path / "out"
    assert ms.main(["--dagger-stream", str(dagger), "--replay-stream", str(replay), "--holdout-scenes-file", str(held),
                    "--out-dir", str(out), "--replay-fraction", "0.2", "--replay-shares", "equal",
                    "--allow-other-rules", "--workers", "1"]) == 0  # shard 256, global batch 64, world size 8
    manifest = read_manifest(out)
    expected = math.ceil((n + math.ceil(n / 4)) / 256) * 256
    assert manifest["total_samples"] == expected == 512 and manifest["num_shards"] == 2
    assert manifest["drawn"]["replay"] == expected - n == manifest["drawn"]["replay_root_r2r"]
    assert manifest["selection"]["composition"]["replay_by_dataset"] == {"r2r": expected - n}
    assert manifest["selection"]["merge_rules"]["mismatches"] == {
        "replay_fraction": 0.2, "replay_by_dataset": {"r2r": expected - n}}
    samples = [s for shard in manifest["shards"] for s in read_shard(out / shard["name"])]
    assert len(samples) == expected and Counter(s.source for s in samples) == {"dagger": n, "replay": expected - n}
    stops = Counter(s.key for s in samples if s.turns == ("STOP",))
    assert all(v == 5 for v in stops.values())  # STOP copies whole; STOPs only in replay

"""EXP-32 checks before training (ledger 第二部分 A1-A4, D1, D3) in scripts/exp32/aud.py."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from scripts.exp28 import merge_streams as ms
from scripts.exp32 import aud
from scripts.exp32 import build_rft_stream as brs
from scripts.exp30.probe_judge import same_answer
from src.data.s2_records import EpisodeRecord
from src.data.s2_sft_dataset import decision_class, recorded_turns
from src.data.s2_stream import MANIFEST_NAME, read_shard, write_shard
from tests.test_exp28_merge_streams import R2R_SCENES, _replay_stream, _sample
from tests.test_exp32_build_rft_stream import EVERY_REASON, HELD, TRAIN, _record

SAMPLED_4090 = ("2026-10-20 12:21:57,429 WARNING heatmapvln-rpc-server: System2 SAMPLES ({'temperature': 0.7, "
                "'top_p': 0.95}): rollouts for self-training only, never an evaluation\n")
SAMPLED_910B = "2026-10-20T04:20:00Z System2 SAMPLES (T=0.7 top_p=0.95)\n"
GREEDY = "2026-10-16 13:40:00,000 INFO heatmapvln-rpc-server: NextDiT action head available: True\n"
OTHER_T = ("2026-10-20 12:21:57,429 WARNING heatmapvln-rpc-server: System2 SAMPLES ({'temperature': 1.0, "
           "'top_p': 0.95})\n")


def _git(repo, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "safe.directory=*", "-C", str(repo),
                           *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def code(tmp_path):
    """A checkout whose first commit samples without the top_k fix and whose second has it."""
    repo = tmp_path / "code"
    server = repo / aud.SERVER_FILE
    server.parent.mkdir(parents=True)
    _git(repo, "init", "-q")
    server.write_text("kwargs.update(do_sample=True, **sampling)\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "sampling without top_k")
    old = _git(repo, "rev-parse", "HEAD")
    server.write_text("kwargs.update(do_sample=True, top_k=0, **sampling)\n")
    _git(repo, "commit", "-qam", "top_k fix")
    return {"repo": repo, "old": old, "fixed": _git(repo, "rev-parse", "HEAD")}


def _servers(directory, tag, commit, logs, dirty=0):
    """A 910B server set as collect_4090.sh pulls it (npu_servers/npu_servers_exp28_<tag>)."""
    server_set = directory / "npu_servers" / f"npu_servers_exp28_{tag}"
    (server_set / "logs").mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(logs):
        (server_set / "logs" / f"model_{i}.log").write_text(text)
        (server_set / "logs" / f"vo_{i}.log").write_text("VO server device: npu:0\n")
    (server_set / "servers.json").write_text(json.dumps({"repo_commit": commit, "repo_dirty": dirty, "slots": len(logs),
                                                         "system2_lora_sha256": ""}))


def _local(directory, tag, commit, logs):
    """A 4090 local run: runtime/<stamp>/logs/model_*.log, and the collect log naming the commit before the stamp."""
    stamp = f"x28loc{tag}_20261020_120000"
    logs_dir = directory / "runtime" / stamp / "logs"
    logs_dir.mkdir(parents=True)
    for i, text in enumerate(logs):
        (logs_dir / f"model_{i}.log").write_text(text)
    (logs_dir / "client_shard_020.log").write_text("no server here\n")
    (directory / "runtime" / f"x28ext{tag}_20261020_115900" / "logs").mkdir(parents=True)  # external: no model logs
    (directory / "logs").mkdir(exist_ok=True)
    with open(directory / "logs" / "collect_local.log", "a") as handle:
        handle.write(f"2026-10-20T12:00:00Z mode=local tag={tag} wt=/workspace/exp23/wt_x31 commit={commit} "
                     f"shards=20-39 adapter=none\n2026-10-20T12:00:01Z STATUS collecting ({stamp})\n"
                     f"2026-10-20T12:05:01Z STATUS collecting: 3 records, 0 shards done\n")


@pytest.fixture
def colls(tmp_path, code):
    """Collection directories as collect_4090.sh leaves them: pass 1, R0-S, pass 2 and collect_base."""
    c = tmp_path / "coll"
    p1, r0s, p2, base = (c / n for n in ("p1", "r0s", "p2", "base"))
    # pass 1 (seed 1337): 3 successes of 8 records (A3 0.375)
    _record(p1 / "records", "kfPV7w3FaU5", 1, EVERY_REASON)                           # 4 RFT samples
    _record(p1 / "records", "AAAAAAAAAAA", 4, [("P", 6.0), ("L", 6.0), ("P", 5.0)])   # 3
    _record(p1 / "records", "BBBBBBBBBBB", 5, [("L", 6.0), ("L", 6.0)], device="cuda")  # 2
    for episode, scene, first in ((2, "kfPV7w3FaU5", "P"), (8, "AAAAAAAAAAA", "Q"), (9, "BBBBBBBBBBB", "P"),
                                  (10, "kfPV7w3FaU5", "L"), (11, "AAAAAAAAAAA", "R")):
        _record(p1 / "records", scene, episode, [(first, 6.0), ("P", 6.0)], success=False)
    _record(r0s / "records", "BBBBBBBBBBB", 5, [("P", 6.0), ("Q", 6.0), ("R", 6.0)])  # 3, second of episode 5
    _record(r0s / "records", "kfPV7w3FaU5", 6, [("P", 8.0), ("R", 8.0)])               # 2
    _record(r0s / "records", "AAAAAAAAAAA", 8, [("Q", 6.0)] * 2, success=False)
    _record(p2 / "records", "BBBBBBBBBBB", 7, [("P", 6.0), ("Q", 6.0), ("S", 2.0)], seed=42)  # 2
    _record(p2 / "records", "kfPV7w3FaU5", 10, [("L", 6.0)], seed=42, success=False)
    # collect_base (seed 42): the anchors' successes (shards >= 20), its failures on the sampled episodes, held out
    for episode, scene in ((101, "kfPV7w3FaU5"), (102, "AAAAAAAAAAA"), (103, "BBBBBBBBBBB"), (104, "kfPV7w3FaU5")):
        _record(base / "records", scene, episode, [("P", 6.0), ("Q", 6.5)] * 4, seed=42, shard=20 + episode % 7)
    # the base's call 0 on pass 1's eight episodes: the same on 1, 5, 2 and 9 (A2 4/8)
    for episode, scene, first in ((1, "kfPV7w3FaU5", "L"), (4, "AAAAAAAAAAA", "Q"), (5, "BBBBBBBBBBB", "L"),
                                  (2, "kfPV7w3FaU5", "P"), (8, "AAAAAAAAAAA", "P"), (9, "BBBBBBBBBBB", "P"),
                                  (10, "kfPV7w3FaU5", "R"), (11, "AAAAAAAAAAA", "L"), (6, "kfPV7w3FaU5", "P"),
                                  (7, "BBBBBBBBBBB", "L")):
        _record(base / "records", scene, episode, [(first, 6.0), ("P", 6.0)], seed=42, shard=30, success=False)
    _record(base / "records", "5jLhtVmWd5F", 105, [("P", 6.0)] * 3, seed=42, shard=3)  # a held-out episode
    # the servers: both log formats on the sampled sources, greedy logs on collect_base
    _servers(p1, "p1", code["fixed"], [SAMPLED_4090, SAMPLED_4090])
    _local(p1, "p1", code["fixed"], [SAMPLED_4090])
    _servers(r0s, "r0s", code["fixed"], [SAMPLED_4090, SAMPLED_910B])
    _servers(p2, "p2", code["fixed"], [SAMPLED_4090])
    _local(p2, "p2", code["fixed"], [SAMPLED_4090, SAMPLED_4090])
    _servers(base, "base", code["old"], [GREEDY, GREEDY])
    _local(base, "base", code["old"], [GREEDY])
    held = tmp_path / "heldout_scenes.txt"
    held.write_text(f"# EXP-28 held-out scenes\n{HELD}\n")
    train = tmp_path / "train_scenes.txt"
    train.write_text("# EXP-28 training scenes\n" + "\n".join(TRAIN) + "\n")
    return {"p1": p1, "r0s": r0s, "p2": p2, "base": base, "held": held, "train": train, "code": code, "tmp": tmp_path}


def _validity_argv(colls, out, sources=None):
    sources = sources or [f"p1={colls['p1']}:1337", f"r0s={colls['r0s']}/records:1337", f"p2={colls['p2']}:42"]
    argv = ["validity"]
    for source in sources:
        argv += ["--source", source]
    return argv + ["--anchor-dir", str(colls["base"]), "--git-repo", str(colls["code"]["repo"]), "--workers", "1",
                   "--out", str(out)]


# --------------------------------------------------------------------------- #
# A: validity
# --------------------------------------------------------------------------- #
def test_both_sampling_log_forms_match_and_a_greedy_log_does_not(tmp_path):
    assert aud.SAMPLES_LOG.search(SAMPLED_4090) and aud.SAMPLES_LOG.search(SAMPLED_910B)
    for text in (GREEDY, OTHER_T, "System2 SAMPLES (T=0.7 top_p=1.0)\n", "System2 SAMPLES ({'temperature': 0.7})\n"):
        assert not aud.SAMPLES_LOG.search(text)
    log = tmp_path / "model_0.log"
    log.write_text(GREEDY + SAMPLED_4090 + OTHER_T)
    assert aud.log_lines(log) == {"matches": 1, "any_samples_line": 2}


def test_launch_commits_name_each_local_runtime(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "collect_local.log").write_text(
        "T mode=local tag=p2 wt=/w commit=aaaaaaa1 shards=20-39 adapter=none\nT STATUS collecting (x28locp2_1)\n"
        "T STATUS collecting: 4 records, 1 shards done\nT STATUS STOPPED: x\n"
        "T mode=local tag=p2 wt=/w commit=bbbbbbb2 shards=20-39 adapter=none\nT STATUS collecting (x28locp2_2)\n")
    (logs / "collect_external.log").write_text("T mode=external tag=p2 wt=/w commit=ccccccc3 shards=0-19 adapter=none\n"
                                               "T STATUS collecting (x28extp2_1)\n")
    assert aud.launch_commits(logs) == {"x28locp2_1": "aaaaaaa1", "x28locp2_2": "bbbbbbb2", "x28extp2_1": "ccccccc3"}


def test_collections_are_parsed_against_a4(tmp_path):
    assert aud.parse_collection("p1=/w/exp32/collect_a:1337") == ("p1", Path("/w/exp32/collect_a"), 1337)
    assert aud.parse_collection("r0s=/w/exp31/r0s/records:1337")[1] == Path("/w/exp31/r0s")
    assert aud.parse_collection("heldout_s42=/w/exp32/heldout_s42:42")[2] == 42
    for bad, match in (("p2=/r:1337", "System1 seed 42"), ("p3=/r:42", "System1 seed 2024"),
                       ("heldout_s42=/r:1337", "names System1 seed 42"), ("g=/r:42", "TAG is one of"),
                       ("p1=/r", "TAG=DIR:SEED"), ("p1=/r:x", "not an integer")):
        with pytest.raises(SystemExit, match=match):
            aud.parse_collection(bad)


def _summary(i, *, success=True, seeds=(1337,), call0=("←←",), device="npu", scene="kfPV7w3FaU5"):
    return aud.RecordSummary(f"/r/{scene}/{scene}_{i:06d}.s2rec.tar", scene, i, success, frozenset(seeds), call0, device)


def test_a2_call0_identity_must_be_below_090():
    base = {("kfPV7w3FaU5", i): _summary(i, seeds=(42,)) for i in range(10)}
    nine = [_summary(i, call0=("←←",) if i < 9 else ("→",)) for i in range(10)]
    eight = [_summary(i, call0=("←←",) if i < 8 else ("↓", "245 320")) for i in range(10)]
    assert aud.call0_identity(nine, base)["rate"] == 0.9 and not aud.call0_identity(nine, base)["pass"]
    result = aud.call0_identity(eight + [_summary(77)], base)
    assert result["pass"] and (result["pairs"], result["identical"], result["unpaired"]) == (10, 8, 1)
    assert not aud.call0_identity([_summary(77)], base)["pass"]  # nothing paired: not measured, not met
    cross = [_summary(i, device="cuda") for i in range(3)]
    assert aud.call0_identity(cross, base)["same_device"] == {"rate": None}


@pytest.mark.parametrize("successes, n, in_band", [(20, 100, True), (19, 100, False), (50, 100, True), (51, 100, False),
                                                   (1, 3, True), (0, 0, False)])
def test_a3_band_is_closed(successes, n, in_band):
    source = [_summary(i, success=i < successes) for i in range(n)]
    assert aud.success_band(source)["in_band"] is in_band


def test_a4_one_percent_of_records_may_differ():
    ok = [_summary(i, seeds=(1337,) if i else (1337, 42)) for i in range(100)]
    result = aud.seed_share(ok, 1337)
    assert result["pass"] and result["mismatched"] == 1
    two = [_summary(i, seeds=(1337,) if i > 1 else (None,)) for i in range(100)]  # a call without a seed differs
    assert not aud.seed_share(two, 1337)["pass"]
    assert aud.seed_share([_summary(0, seeds=())], 1337)["pass"]  # no calls: nothing to disagree (the builder's reading)
    assert not aud.seed_share([], 1337)["pass"]


def test_validity_passes_on_well_formed_collections(colls, capsys):
    out = colls["tmp"] / "validity.json"
    assert aud.main(_validity_argv(colls, out)) == aud.PASS
    assert capsys.readouterr().out.startswith("VALIDITY PASS: p1 valid, r0s valid, p2 valid, anchor valid")
    report = json.loads(out.read_text())
    p1 = report["sources"]["p1"]
    assert p1["records_root"] == str((colls["p1"] / "records").resolve()) and p1["valid"]
    assert p1["a1"]["n_logs"] == p1["a1"]["n_matching"] == 3
    assert p1["a1"]["serving_commits"] == {colls["code"]["fixed"]: {
        "top_k_fix": True, "server_sets": ["npu_servers/npu_servers_exp28_p1", "runtime/x28locp1_20261020_120000"]}}
    assert (p1["a2"]["pairs"], p1["a2"]["identical"], p1["a2"]["pass"], p1["a2"]["gated"]) == (8, 4, True, True)
    assert (p1["a3"]["records"], p1["a3"]["successes"], p1["a3"]["in_band"]) == (8, 3, True) and "note" not in p1["a3"]
    assert p1["a4"]["mismatched"] == 0 and p1["a4"]["pass"]
    r0s = report["sources"]["r0s"]
    assert r0s["a1"]["n_matching"] == 2 and not r0s["a2"]["gated"] and r0s["a2"]["identical"] == 1
    assert report["anchor"]["valid"]
    assert report["anchor"]["a1"]["n_logs"] == 3 and report["anchor"]["a1"]["n_matching"] == 0
    assert report["anchor"]["a4"]["records"] == 15


@pytest.mark.parametrize("damage, who, reason", [
    ("greedy_log", "p1", "lack the sampling line"),
    ("other_temperature", "p2", "lack the sampling line"),
    ("old_commit", "p1", "has no top_k=0"),
    ("unknown_commit", "r0s", "git show"),
    ("dirty", "r0s", "uncommitted changes"),
    ("no_launch_commit", "p2", "names the commit"),
    ("anchor_samples", "anchor", "show the sampling line"),
])
def test_a1_voids_a_source(colls, damage, who, reason, capsys):
    code = colls["code"]
    if damage == "greedy_log":
        (colls["p1"] / "runtime" / "x28locp1_20261020_120000" / "logs" / "model_1.log").write_text(GREEDY)
    elif damage == "other_temperature":
        (colls["p2"] / "npu_servers" / "npu_servers_exp28_p2" / "logs" / "model_0.log").write_text(OTHER_T)
    elif damage == "old_commit":
        _servers(colls["p1"], "p1", code["old"], [SAMPLED_4090, SAMPLED_4090])
    elif damage == "unknown_commit":
        _servers(colls["r0s"], "r0s", "d" * 40, [SAMPLED_4090, SAMPLED_910B])
    elif damage == "dirty":
        _servers(colls["r0s"], "r0s", code["fixed"], [SAMPLED_4090, SAMPLED_910B], dirty=1)
    elif damage == "no_launch_commit":
        (colls["p2"] / "logs" / "collect_local.log").unlink()
    elif damage == "anchor_samples":
        (colls["base"] / "npu_servers" / "npu_servers_exp28_base" / "logs" / "model_1.log").write_text(SAMPLED_910B)
    out = colls["tmp"] / "validity.json"
    assert aud.main(_validity_argv(colls, out)) == aud.FAIL
    assert capsys.readouterr().out.startswith("VALIDITY FAIL")
    report = json.loads(out.read_text())
    entry = report["anchor"] if who == "anchor" else report["sources"][who]
    assert not entry["valid"] and any(reason in r for r in entry["void_reasons"]), entry["void_reasons"]
    others = [e for name, e in [*report["sources"].items(), ("anchor", report["anchor"])] if name != who]
    assert all(e["valid"] for e in others)  # a void source does not void the others


def test_a2_a3_a4_void_pass_1_or_note_it(colls, monkeypatch):
    out = colls["tmp"] / "validity.json"
    monkeypatch.setattr(aud, "A2_BELOW", aud.Fraction(1, 2))  # 4/8 identical is not below 1/2
    assert aud.main(_validity_argv(colls, out)) == aud.FAIL
    assert any(r.startswith("A2:") for r in json.loads(out.read_text())["sources"]["p1"]["void_reasons"])
    monkeypatch.undo()
    monkeypatch.setattr(aud, "A3_BAND", (aud.Fraction(2, 5), aud.Fraction(1, 2)))  # 3/8 below the band, A1 met
    assert aud.main(_validity_argv(colls, out)) == aud.PASS
    assert "write it in the run record" in json.loads(out.read_text())["sources"]["p1"]["a3"]["note"]
    monkeypatch.undo()
    _record(colls["p2"] / "records", "AAAAAAAAAAA", 12, [("P", 6.0)] * 2, seed=42, call_seeds={1: 1337})
    assert aud.main(_validity_argv(colls, out)) == aud.FAIL
    p2 = json.loads(out.read_text())["sources"]["p2"]
    assert p2["a4"]["mismatched"] == 1 and p2["a4"]["records"] == 3 and not p2["valid"]


# --------------------------------------------------------------------------- #
# D1: the stream
# --------------------------------------------------------------------------- #
def _build_streams(colls):
    """The RFT stream from the collections, merged with an r2r replay stream at 20% (48 samples, 5 STOPs)."""
    tmp = colls["tmp"]
    rft = tmp / "stream_rft"
    assert brs.main([
        "--rft-source", f"p1={colls['p1']}/records:1337", "--rft-source", f"r0s={colls['r0s']}/records:1337",
        "--rft-source", f"p2={colls['p2']}/records:42", "--anchor-root", f"{colls['base']}/records",
        "--train-scenes-file", str(colls["train"]), "--holdout-scenes-file", str(colls["held"]),
        "--first-calls", "48", "--min-distance", "3", "--max-anchors-per-episode", "6", "--anchor-ratio", "1.0",
        "--anchor-seed", "32", "--max-trajectories-per-episode", "2", "--no-pad", "--seed", "42", "--shard-size", "8",
        "--global-batch", "8", "--world-size", "1", "--workers", "1", "--out-dir", str(rft)]) == 0
    samples = [_sample(f"replay:{R2R_SCENES[i % 2]}_{i:06d}@{4 * i}:abcd{i:04d}", "replay", ("←←",))
               for i in range(11)]
    samples += [_sample(f"replay:{R2R_SCENES[0]}_000100@99:stop0000", "replay", ("STOP",))] * 5
    replay = _replay_stream(tmp, samples, root_scenes={"r2r": R2R_SCENES})
    merged = tmp / "stream"
    assert ms.main(["--dagger-stream", str(rft), "--replay-stream", str(replay), "--holdout-scenes-file",
                    str(colls["held"]), "--replay-fraction", "0.2", "--replay-shares", "equal", "--allow-other-rules",
                    "--out-dir", str(merged), "--shard-size", "16", "--global-batch", "8", "--world-size", "1",
                    "--workers", "1"]) == 0
    validity = tmp / "validity.json"
    assert aud.main(_validity_argv(colls, validity)) == aud.PASS
    return rft, merged, validity


def _stream_argv(colls, merged, validity, out, *extra):
    return ["stream", "--stream", str(merged), "--validity", str(validity), "--anchor-root", f"{colls['base']}/records",
            "--train-scenes-file", str(colls["train"]), "--holdout-scenes-file", str(colls["held"]),
            "--git-repo", str(colls["code"]["repo"]), "--base-commit", colls["code"]["fixed"], "--workers", "1",
            "--out", str(out), *extra]


@pytest.fixture
def small_dose(monkeypatch):
    """B2's lines scaled to the fixture's 16 RFT samples: no pass 3 is due."""
    monkeypatch.setattr(aud, "DOSE_PASS3_BELOW", 10)
    monkeypatch.setattr(aud, "DOSE_TRAIN_MIN", 5)


def test_stream_audit_passes_and_reads_every_line(colls, small_dose, capsys, monkeypatch):
    rft, merged, validity = _build_streams(colls)
    capsys.readouterr()
    out = colls["tmp"] / "aud_stream.json"
    assert aud.main(_stream_argv(colls, merged, validity, out)) == aud.PASS
    assert capsys.readouterr().out.startswith("AUD PASS ")
    report = json.loads(out.read_text())
    items = report["items"]
    assert report["verdict"] == "PASS" and report["failed"] == [] and report["total_samples"] == 48
    assert items["1_leaks"]["pass"] and items["1_leaks"]["merged_holdout_complete"]
    rules = items["2_rules"]
    assert rules["pass"] and rules["reasons"] == [] and rules["knobs_off"] == {}
    assert rules["samples"] == {"rft": 16, "anchor": 16, "problems": {}, "examples": {}}
    assert rules["records"]["rft"] == 6 and rules["records"]["anchor"] >= 3 and rules["records"]["problems"] == {}
    assert rules["stop_forward_invalid_targets"] == rules["reversals_left"] == rules["within_3m_or_beyond_call_47"] == 0
    assert rules["trajectories"] == {"listed": 6, "episodes": 5, "max_per_episode": 2}
    assert rules["anchors"]["max_per_episode"] <= 6 and rules["anchors"]["lowest_ranks_off"] == 0
    rft_keys = sorted(k for k, row in aud.read_keys_tsv((rft / brs.KEYS_TSV).read_text())[0].items()
                      if row.category != "anchor")
    assert items["3_bytes"] == {"seed": 32, "wanted": 16, "checked": 16, "mismatched": 0, "examples": [],
                                "keys": rft_keys, "pass": True}
    assert items["4_stop_share"]["stops"] == 5 and items["4_stop_share"]["share"] == round(5 / 48, 6)
    assert items["5_dose"]["n_rft"] == 16 and items["5_dose"]["by_source"] == {"p1": 9, "r0s": 5, "p2": 2, "p3": 0}
    assert items["6_code_unchanged"]["pass"] and items["6_code_unchanged"]["changed"] == []
    # the pre-registered B2 lines: 16 RFT samples after pass 2 ask for pass 3 first
    monkeypatch.setattr(aud, "DOSE_PASS3_BELOW", 4000)
    monkeypatch.setattr(aud, "DOSE_TRAIN_MIN", 2000)
    assert aud.main(_stream_argv(colls, merged, validity, out)) == aud.FAIL
    assert json.loads(out.read_text())["failed"] == ["5_dose"]


def test_byte_draw_is_seeded_and_capped(colls, small_dose):
    _, merged, validity = _build_streams(colls)
    out = colls["tmp"] / "aud_stream.json"
    picked = []
    for seed in ("32", "32", "7"):
        argv = _stream_argv(colls, merged, validity, out, "--byte-samples", "5", "--byte-seed", seed)
        assert aud.main(argv) == aud.PASS
        picked.append(json.loads(out.read_text())["items"]["3_bytes"])
    assert picked[0] == picked[1] and picked[0]["wanted"] == picked[0]["checked"] == len(picked[0]["keys"]) == 5
    assert picked[2]["keys"] != picked[0]["keys"] and all("#g." not in k for k in picked[2]["keys"])


def _tamper(merged, key, field):
    manifest = json.loads((merged / MANIFEST_NAME).read_text())
    for shard in manifest["shards"]:
        samples = read_shard(merged / shard["name"])
        if any(s.key == key for s in samples):
            changed = []
            for s in samples:
                if s.key == key:
                    value = getattr(s, field)
                    value = (value[:-1] + (value[-1][:-2] + b"\x00\x00",)) if field == "history_jpegs" else \
                        value[:-2] + b"\x00\x00"
                    s = dataclasses.replace(s, **{field: value})
                changed.append(s)
            shard["sha256"] = write_shard(merged / shard["name"], changed)  # the stream still verifies
    (merged / MANIFEST_NAME).write_text(json.dumps(manifest))


@pytest.mark.parametrize("field", ["current_jpeg", "lookdown_jpeg", "history_jpegs"])
def test_one_tampered_jpeg_fails_the_audit(colls, small_dose, field, capsys):
    rft, merged, validity = _build_streams(colls)
    capsys.readouterr()
    rows, _ = aud.read_keys_tsv((rft / brs.KEYS_TSV).read_text())
    key = next(k for k, row in sorted(rows.items()) if row.category == "rft_p1" and row.call_index == 2)
    _tamper(merged, key, field)
    out = colls["tmp"] / "aud_stream.json"
    assert aud.main(_stream_argv(colls, merged, validity, out)) == aud.FAIL
    assert capsys.readouterr().out.startswith("AUD FAIL: 3_bytes")
    report = json.loads(out.read_text())
    assert report["failed"] == ["3_bytes"]
    assert report["items"]["3_bytes"]["examples"] == [{"key": key, "fields": [field]}]
    assert report["items"]["3_bytes"]["checked"] == 16 and report["items"]["3_bytes"]["mismatched"] == 1


def test_a_void_source_or_another_anchor_root_fails_the_rules(colls, small_dose):
    _, merged, validity = _build_streams(colls)
    report = json.loads(validity.read_text())
    report["sources"]["r0s"]["valid"] = False
    report["sources"]["r0s"]["a1"]["pass"] = False
    report["sources"]["r0s"]["void_reasons"] = ["A1: greedy"]
    void = colls["tmp"] / "validity_void.json"
    void.write_text(json.dumps(report))
    out = colls["tmp"] / "aud_stream.json"
    assert aud.main(_stream_argv(colls, merged, void, out)) == aud.FAIL
    rules = json.loads(out.read_text())["items"]["2_rules"]
    assert not rules["pass"] and any("source r0s: A1 not met" in r for r in rules["reasons"])
    del report["sources"]["p2"]
    void.write_text(json.dumps({**report, "sources": {k: v for k, v in report["sources"].items() if k != "r0s"}}))
    assert aud.main(_stream_argv(colls, merged, void, out)) == aud.FAIL
    reasons = json.loads(out.read_text())["items"]["2_rules"]["reasons"]
    assert any("source p2: no validity report" in r for r in reasons)
    other = colls["tmp"] / "other_base"
    (other / "records").mkdir(parents=True)
    argv = _stream_argv(colls, merged, validity, out)
    argv[argv.index("--anchor-root") + 1] = str(other / "records")
    assert aud.main(argv) == aud.FAIL
    reasons = json.loads(out.read_text())["items"]["2_rules"]["reasons"]
    assert any("the anchors were drawn from" in r for r in reasons) and any("anchor root" in r for r in reasons)


def test_changed_code_fails_item_6(colls, small_dose):
    _, merged, validity = _build_streams(colls)
    repo = colls["code"]["repo"]
    (repo / "scripts" / "exp32").mkdir(parents=True)
    (repo / "scripts" / "exp32" / "aud.py").write_text("# new\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "exp32 only")
    out = colls["tmp"] / "aud_stream.json"
    assert aud.main(_stream_argv(colls, merged, validity, out)) == aud.PASS
    (repo / "scripts" / "run_ppa_x.sh").write_text("echo\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "a launcher")
    assert aud.main(_stream_argv(colls, merged, validity, out)) == aud.FAIL
    report = json.loads(out.read_text())
    assert report["failed"] == ["6_code_unchanged"]
    assert report["items"]["6_code_unchanged"]["changed"] == ["scripts/run_ppa_x.sh"]


def test_git_unchanged_reads_the_ledgers_paths(code):
    repo, fixed = code["repo"], code["fixed"]
    assert aud.git_unchanged(repo, fixed, "HEAD")["pass"]
    assert aud.git_unchanged(repo, code["old"], fixed)["changed"] == [aud.SERVER_FILE]
    for path in ("src/data/x.py", "scripts/train_system2_lora.py", "scripts/ascend/run.sh",
                 "scripts/exp28/merge_streams.py"):
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(path)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "changes")
    assert aud.git_unchanged(repo, fixed, "HEAD")["changed"] == ["scripts/ascend/run.sh", "scripts/train_system2_lora.py",
                                                                 "src/data/x.py"]
    bad = aud.git_unchanged(repo, "0" * 40, "HEAD")
    assert not bad["pass"] and "cannot resolve" in bad["reasons"][0]


def _entry(key, source="dagger", turns=("←←",)):
    return ms.Entry("x.tar", 0, key, source, turns, ())


def test_leak_check_by_key_scene():
    train, held = {"kfPV7w3FaU5", "AAAAAAAAAAA"}, {"5jLhtVmWd5F", "17DRP5sb8fy"}
    clean = [_entry("kfPV7w3FaU5_000001#p1.3"), _entry("AAAAAAAAAAA_000004#g.0"),
             _entry(f"replay:{R2R_SCENES[0]}_000004@16:1a2b3c4d", "replay")]
    assert aud.leak_check(clean, train=train, held=held)["pass"]
    for bad, field in ((_entry("5jLhtVmWd5F_000001#p1.3"), "held_out"),
                       (_entry("ZZZZZZZZZZZ_000001#r0s.3"), "dagger_outside_training_scenes"),
                       (_entry("replay:17DRP5sb8fy_000004@16:1a2b3c4d", "replay"), "held_out"),
                       (_entry("replay:00100-AAAAAAAAAAA_000004@16:1a2b3c4d", "replay"), "replay_not_r2r"),
                       (_entry("kfPV7w3FaU5_000009#3", "other"), "other_sources")):
        result = aud.leak_check([*clean, bad], train=train, held=held)
        assert not result["pass"] and result[field], (bad.key, result)


@pytest.mark.parametrize("stops, total, ok", [(3, 100, True), (299, 10000, False), (12, 100, True), (1201, 10000, False),
                                              (5, 48, True), (0, 0, False)])
def test_stop_share_band_is_closed(stops, total, ok):
    assert aud.stop_share(stops, total)["pass"] is ok


@pytest.mark.parametrize("counts, tags, ok, why", [
    ({"p1": 3000, "r0s": 500, "p2": 500}, ("p1", "r0s", "p2"), True, ""),
    ({"p1": 3000, "r0s": 500, "p2": 499}, ("p1", "r0s", "p2"), False, "pass 3 runs first"),
    ({"p1": 3000, "r0s": 500, "p2": 500, "p3": 10}, ("p1", "r0s", "p2", "p3"), False, "was not to run"),
    ({"p1": 1500, "r0s": 300, "p2": 100, "p3": 100}, ("p1", "r0s", "p2", "p3"), True, ""),
    ({"p1": 1500, "r0s": 300, "p2": 100, "p3": 99}, ("p1", "r0s", "p2", "p3"), False, "no training"),
    ({"p1": 1500, "r0s": 300, "p2": 300}, ("p1", "r0s", "p2", "p3"), True, ""),  # pass 3 ran, found nothing
    ({"p1": 4000, "r0s": 500}, ("p1", "r0s"), False, "B1 runs pass 2 always"),
])
def test_b2_dose_rule(counts, tags, ok, why):
    result = aud.dose_rule({f"category_rft_{t}": n for t, n in counts.items()}, tags)
    assert result["pass"] is ok and (ok or any(why in r for r in result["reasons"])), result


def test_audit_record_names_every_broken_sample(tmp_path):
    path = str(_record(tmp_path / "p1" / "records", "kfPV7w3FaU5", 1, EVERY_REASON))
    calls = {int(c["call_index"]): c for c in EpisodeRecord(path).calls}

    def sample(call, key_tag="p1", turns=None):
        return (brs.sample_key("kfPV7w3FaU5", 1, key_tag, call), call, turns or recorded_turns(calls[call]), "npu", None)

    info = aud.audit_record((path, "p1", [sample(0), sample(2), sample(1), sample(3), sample(4), sample(5), sample(7),
                                          sample(9), sample(12), sample(10, turns=("→→",)),
                                          sample(11, key_tag="r0s")],
                             32))
    assert info["problems"] == {"reversal": 2, "class_stop": 1, "class_forward": 1, "class_invalid": 1,
                                "within_min_distance": 1, "no_distance": 1, "target_not_recorded": 1, "key": 1}
    assert info["success"] and info["seeds"] == {1337} and info["adapter"] is None and info["shard"] == 30
    assert info["episode"] == ("kfPV7w3FaU5", 1) and len(info["answers_sha256"]) == 64
    long = str(_record(tmp_path / "p1" / "records", "kfPV7w3FaU5", 2, [("P", 6.0)] * 50))
    assert aud.audit_record((long, "p1", [(brs.sample_key("kfPV7w3FaU5", 2, "p1", 48), 48, ("↓", "245 320"), "npu",
                                           None)], 32))["problems"] == {"beyond_first_calls": 1}


def test_anchor_samples_must_be_their_episodes_lowest_ranks(tmp_path):
    path = str(_record(tmp_path / "base" / "records", "kfPV7w3FaU5", 101, [("P", 6.0)] * 9, seed=42))
    ranked = sorted(range(9), key=lambda i: brs.anchor_rank(32, f"{path}#{i}"))

    def job(chosen):
        return (path, "g", [(brs.sample_key("kfPV7w3FaU5", 101, "g", i), i, ("↓", "245 320"), "npu", None)
                            for i in chosen], 32)

    assert aud.audit_record(job(ranked[:4]))["anchor_lowest_ranks"]
    assert not aud.audit_record(job(ranked[1:5]))["anchor_lowest_ranks"]
    info = {"path": path, "tag": "g", "success": True, "seeds": frozenset({42}), "adapter": None,
            "scene": "kfPV7w3FaU5", "dir_scene": "kfPV7w3FaU5", "shard": 19}
    root = str(tmp_path / "base" / "records")
    assert aud.record_problems(info, root=root, seed=42, train={"kfPV7w3FaU5"}, held=set()) == ["shard_below_20"]
    assert aud.record_problems({**info, "shard": 20}, root=root, seed=42, train={"kfPV7w3FaU5"}, held=set()) == []
    assert aud.record_problems({**info, "shard": 20, "seeds": frozenset({42, None}), "success": False}, root="/elsewhere",
                               seed=42, train=set(), held={"kfPV7w3FaU5"}) == \
        ["outside_its_root", "not_success", "seed_mismatch", "held_out_scene"]


# --------------------------------------------------------------------------- #
# D3: the sampled dose
# --------------------------------------------------------------------------- #
def _other_answer(recorded, *, near):
    """Not the recorded answer: ``near``, a pixel goal a few px off (``same_answer``); else STOP (a left turn
    for a STOP)."""
    if near and decision_class(recorded[-1]) == "pixel":
        return ["↓", "250 325"] if recorded[-1] != "250 325" else ["↓", "255 325"]
    return ["STOP"] if decision_class(recorded[-1]) != "stop" else ["←"]


def _parity(colls, rft, n_inexact, *, staged_by_tag=True, n=20, report=None):
    """D3's output over the stream's npu RFT records, staged on the 910B as <stage>/<tag>/<scene dir>/<file>;
    the first ``n_inexact`` rows replay another answer (alternately a diverging and a near one)."""
    trajectories = aud.read_rft_records((rft / brs.RECORDS_LIST).read_text())
    rows = []
    for original, (device, tag) in sorted(trajectories.items(), key=lambda x: (x[1][1], x[0])):
        if device != "npu":
            continue
        staged = "/home/x/zhr_1/exp32/d3_records/" + (tag if staged_by_tag else "records") + "/" + \
            "/".join(Path(original).parts[-2:])
        for call in sorted(EpisodeRecord(original).calls, key=lambda c: int(c["call_index"])):
            recorded = list(recorded_turns(call))
            rows.append({"uid": f"{staged}#{call['call_index']}", "key": f"x#{call['call_index']}", "recorded": recorded})
    rows = rows[:n]
    for i, row in enumerate(rows):
        row["replay"] = _other_answer(row["recorded"], near=i % 2 == 1) if i < n_inexact else list(row["recorded"])
        row["exact"] = row["replay"] == row["recorded"]
    parity_dir = colls["tmp"] / "d3"
    parity_dir.mkdir(exist_ok=True)
    (parity_dir / "parity_g0.json").write_text(json.dumps({
        "schema": aud.PARITY_SCHEMA, "n": len(rows), "exact_rate": sum(r["exact"] for r in rows) / len(rows),
        "split": "all", "record_device": "npu", "adapter": None, "device": "npu", **(report or {})}))
    (parity_dir / "parity_g0.calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return parity_dir, rows


def _dose_argv(colls, parity_dir, rft, *extra):
    return ["dose", "--parity-dir", str(parity_dir), "--rft-stream", str(rft), "--m-states",
            str(colls["tmp"] / "m_states.jsonl"), "--out", str(colls["tmp"] / "dose.json"), *extra]


@pytest.mark.parametrize("n_inexact, status", [(2, aud.PASS), (3, aud.PASS), (1, aud.FAIL), (0, aud.FAIL)])
def test_d3_g_samp_at_the_010_line(colls, n_inexact, status, capsys):
    rft, _, _ = _build_streams(colls)
    parity_dir, rows = _parity(colls, rft, n_inexact)
    assert len(rows) == 20
    capsys.readouterr()
    assert aud.main(_dose_argv(colls, parity_dir, rft)) == status
    assert capsys.readouterr().out.startswith(f"D3 {'PASS' if status == aud.PASS else 'FAIL'}: G-samp")
    report = json.loads((colls["tmp"] / "dose.json").read_text())
    assert report["exact"] == 20 - n_inexact and report["g_samp"] == n_inexact / 20


def test_d3_divergence_effective_dose_and_m_states(colls):
    rft, _, _ = _build_streams(colls)
    parity_dir, rows = _parity(colls, rft, 6)
    assert aud.main(_dose_argv(colls, parity_dir, rft)) == aud.PASS
    report = json.loads((colls["tmp"] / "dose.json").read_text())
    keys, _ = aud.read_keys_tsv((rft / brs.KEYS_TSV).read_text())
    in_stream = {(r.record_path, r.call_index): r for r in keys.values() if r.category != "anchor"}
    trajectories = aud.read_rft_records((rft / brs.RECORDS_LIST).read_text())
    by_tail = {}
    for original, (_, tag) in trajectories.items():
        by_tail[(tag, *Path(original).parts[-2:])] = original
    expected, hits = [], 0
    for row in rows:
        staged, _, call = row["uid"].rpartition("#")
        parts = Path(staged).parts
        original = by_tail[(parts[-3], *parts[-2:])]
        if (original, int(call)) in in_stream:
            hits += 1
            if not same_answer(row["replay"], row["recorded"]):
                expected.append(in_stream[(original, int(call))].key)
    diverged_all = sum(not same_answer(r["replay"], r["recorded"]) for r in rows)
    assert report["divergence_same_answer"]["all"]["diverged"] == diverged_all
    assert report["divergence_same_answer"]["stream_rft_rows"] == {"diverged": len(expected), "n": hits,
                                                                    "rate": round(len(expected) / hits, 6)}
    assert 0 < len(expected) < hits and report["n_rft"] == 16
    assert report["effective_dose"] == round(16 * len(expected) / hits, 1)
    states = [json.loads(line) for line in (colls["tmp"] / "m_states.jsonl").read_text().splitlines()]
    assert [s["key"] for s in states] == sorted(expected) and report["m_states"]["n"] == len(expected)
    assert all(s["record_device"] == "npu" and s["tag"] in brs.RFT_TAGS and s["base_replay_npu"] != s["recorded"]
               for s in states)
    assert report["records"]["by_device"] == {"npu": report["records"]["rft_trajectories"]}


def test_d3_copies_must_map_to_one_trajectory(colls):
    rft, _, _ = _build_streams(colls)
    parity_dir, rows = _parity(colls, rft, 4, staged_by_tag=False, n=40)
    # p1's (cuda) and R0-S's (npu) episode 5 share <scene dir>/<file>: without the tag directory it is ambiguous
    assert any("BBBBBBBBBBB_000005" in r["uid"] for r in rows)
    with pytest.raises(SystemExit, match="cannot tell which RFT trajectory"):
        aud.main(_dose_argv(colls, parity_dir, rft))
    trajectories = aud.read_rft_records((rft / brs.RECORDS_LIST).read_text())
    mapping = colls["tmp"] / "map.tsv"
    lines = []
    for staged in sorted({r["uid"].rpartition("#")[0] for r in rows}):
        tail = Path(staged).parts[-2:]
        original = next(o for o, (device, _) in trajectories.items() if Path(o).parts[-2:] == tail and device == "npu")
        lines.append(f"{staged}\t{original}\n")
    mapping.write_text("".join(lines))
    assert aud.main(_dose_argv(colls, parity_dir, rft, "--record-map", str(mapping))) == aud.PASS
    # a copy whose recorded answer is not its original's is refused
    rows[0]["recorded"] = ["↓", "1 2"]
    (parity_dir / "parity_g0.calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="do not carry their RFT record's recorded answer"):
        aud.main(_dose_argv(colls, parity_dir, rft, "--record-map", str(mapping)))


@pytest.mark.parametrize("report, match", [({"split": "holdout"}, "--parity-split all"),
                                           ({"record_device": "cuda"}, "--parity-device npu"),
                                           ({"schema": "other"}, "schema"), ({"n": 19}, "the report says 19")])
def test_d3_refuses_another_parity_run(colls, report, match):
    rft, _, _ = _build_streams(colls)
    parity_dir, _ = _parity(colls, rft, 2, report=report)
    with pytest.raises(SystemExit, match=match):
        aud.main(_dose_argv(colls, parity_dir, rft))


def test_counts_of_the_fixture_stream(colls):
    """The fixture's RFT stream, as the D1 and D3 tests above assume it."""
    rft, merged, _ = _build_streams(colls)
    rows, twice = aud.read_keys_tsv((rft / brs.KEYS_TSV).read_text())
    assert not twice and Counter(r.category for r in rows.values()) == {
        "rft_p1": 9, "rft_r0s": 5, "rft_p2": 2, "anchor": 16}
    assert {(tag, device) for device, tag in aud.read_rft_records((rft / brs.RECORDS_LIST).read_text()).values()} == {
        ("p1", "npu"), ("p1", "cuda"), ("r0s", "npu"), ("p2", "npu")}
    assert json.loads((merged / MANIFEST_NAME).read_text())["total_samples"] == 48

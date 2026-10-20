"""EXP-32 offline gates R and M (ledger 第二部分 F) in scripts/exp32/rft_gate.py."""

from __future__ import annotations

import contextlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.exp32 import aud
from scripts.exp32 import build_rft_stream as brs
from scripts.exp32 import rft_gate as rg
from tests.test_exp32_build_rft_stream import EVERY_REASON, _argv, _record, rft_roots  # noqa: F401

HELD_FILE_LINES = ("00481-5jLhtVmWd5F", "00800-TEEsavR23oF")
D3_DIR = "/home/zhr/work/zhr/zhr_1/exp32/d3_records"  # where the 910B would hold D3's records


# --------------------------------------------------------------------------- #
# select-r
# --------------------------------------------------------------------------- #
@pytest.fixture
def heldout(tmp_path):
    s42, s1337 = tmp_path / "heldout_s42" / "records", tmp_path / "heldout_s1337" / "records"
    _record(s42, "5jLhtVmWd5F", 1, EVERY_REASON, seed=42, device="cuda")                    # R: calls 0, 2, 10, 11
    _record(s42, "5jLhtVmWd5F", 2, [("P", 6.0)] * 10, success=False, seed=42, device="cuda")  # R_fail pool: 10
    _record(s42, "TEEsavR23oF", 3, [("L", 6.0), ("P", 4.0), ("S", 2.0)], seed=42, device="cuda")  # R: 0, 1
    _record(s1337, "5jLhtVmWd5F", 1, [("P", 6.0), ("Q", 5.0)], seed=1337, device="cuda")  # R: 0, 1 (same episode)
    _record(s1337, "TEEsavR23oF", 4, [("L", 6.0), ("L", 6.0), ("R", 6.0)], success=False, seed=1337,
            device="cuda")                                                                  # R_fail pool: 0, 1
    held = tmp_path / "heldout_scenes.txt"
    held.write_text("# EXP-28 held-out scenes\n" + "\n".join(HELD_FILE_LINES) + "\n")
    return {"s42": s42, "s1337": s1337, "held": held}


def _select_r(roots, out, *extra, sources=None):
    sources = sources or [f"s42={roots['s42']}:42", f"s1337={roots['s1337']}:1337"]
    argv = ["select-r"]
    for source in sources:
        argv += ["--source", source]
    return rg.main(argv + ["--holdout-scenes-file", str(roots["held"]), "--output", str(out), *extra])


def _states(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_r_takes_the_rf2_rf3_calls_of_successful_held_out_trajectories(heldout, tmp_path):
    out = tmp_path / "rgate" / "r_states.jsonl"
    assert _select_r(heldout, out) == 0
    states = _states(out)
    assert {s["group"] for s in states} == {"R"}
    got = sorted((s["source"], s["scene_id"], s["episode_id"], s["call_index"]) for s in states)
    want = [("s42", "5jLhtVmWd5F", 1, c) for c in (0, 2, 10, 11)] + [("s42", "TEEsavR23oF", 3, c) for c in (0, 1)]
    assert got == sorted(want + [("s1337", "5jLhtVmWd5F", 1, c) for c in (0, 1)])
    # the RFT target rule (build_rft_stream.rft_calls) exactly, with the recorded answer verbatim
    for state in states:
        kept = {c.call_index: c for c in brs.rft_calls(state["record_path"])[0]}
        assert list(kept[state["call_index"]].turns) == state["recorded"]
        assert state["uid"] == f"{state['record_path']}#{state['call_index']}"
        assert state["distance_to_goal"] > 3.0 and state["call_index"] < 48
    # both seeds' trajectories of one episode are one bootstrap cluster
    assert {s["record_key"] for s in states if s["episode_id"] == 1} == {"5jLhtVmWd5F_000001"}
    meta = json.loads(out.with_suffix(".selection.json").read_text())
    assert meta["states"] == {"R": 8, "R_fail": 0, "R_fail_pool": 12}
    assert meta["sources"]["s42"]["records"] == {"records": 3, "success": 2, "failed": 1}
    assert meta["sources"]["s42"]["calls_success"]["reversal"] == 2
    assert meta["sources"]["s1337"]["calls_failed"] == {"kept": 2, "reversal": 1}


def test_r_fail_is_seeded_down_to_rs_size(heldout, tmp_path):
    out = tmp_path / "r_states.jsonl"
    assert _select_r(heldout, out, "--failed") == 0
    states = _states(out)
    fail = [s for s in states if s["group"] == "R_fail"]
    assert Counter(s["group"] for s in states) == {"R": 8, "R_fail": 8}
    failed = [str(x) for root, pattern in ((heldout["s42"], "*/*_000002*.tar"), (heldout["s1337"], "*/*_000004*.tar"))
              for x in Path(root).resolve().glob(pattern)]
    pool = [f"{p}#{c.call_index}" for p in failed for c in brs.rft_calls(p)[0]]
    assert len(pool) == 12
    assert {s["uid"] for s in fail} == set(sorted(pool, key=lambda u: rg.fail_rank(32, u))[:8])
    again = tmp_path / "again.jsonl"
    assert _select_r(heldout, again, "--failed") == 0 and _states(again) == states
    other = tmp_path / "other.jsonl"
    assert _select_r(heldout, other, "--failed", "--fail-seed", "7") == 0
    assert {s["uid"] for s in _states(other) if s["group"] == "R_fail"} != {s["uid"] for s in fail}


def test_r_refuses_what_is_not_bases_held_out_sampling(heldout, tmp_path):
    out = tmp_path / "r.jsonl"
    with pytest.raises(SystemExit, match="B3's held-out sampling runs"):
        _select_r(heldout, out, sources=[f"s42={heldout['s42']}:7"])
    with pytest.raises(SystemExit, match="twice"):
        _select_r(heldout, out, sources=[f"s42={heldout['s42']}:42", f"s42={heldout['s1337']}:1337"])
    with pytest.raises(SystemExit, match="same record"):
        _select_r(heldout, out, sources=[f"a={heldout['s42']}:42", f"b={heldout['s42']}:42"])
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no committed record"):
        _select_r(heldout, out, sources=[f"s42={tmp_path / 'empty'}:42"])
    _record(heldout["s42"], "kfPV7w3FaU5", 9, [("P", 6.0)] * 2, seed=42)  # a training scene
    with pytest.raises(SystemExit, match="not held out"):
        _select_r(heldout, out)


def test_r_refuses_a_served_adapter_and_counts_seed_mismatches(heldout, tmp_path, monkeypatch):
    out = tmp_path / "r.jsonl"
    lora = tmp_path / "lora"
    _record(lora, "5jLhtVmWd5F", 5, [("P", 6.0)] * 2, seed=42, adapter="ab12cd34")
    with pytest.raises(SystemExit, match="served with System2 LoRA ab12cd34"):
        _select_r(heldout, out, sources=[f"s42={lora}:42"])
    _record(heldout["s42"], "5jLhtVmWd5F", 6, [("P", 6.0)] * 3, seed=42, call_seeds={2: 1337})
    with pytest.raises(SystemExit, match="another System1 seed than 42"):
        _select_r(heldout, out)
    monkeypatch.setattr(brs, "MAX_SEED_MISMATCH_SHARE", Fraction(1, 2))  # show the left-out record is counted
    assert _select_r(heldout, out) == 0
    meta = json.loads(out.with_suffix(".selection.json").read_text())
    assert meta["sources"]["s42"]["records"]["seed_mismatch"] == 1
    assert not any(s["episode_id"] == 6 for s in _states(out))


# --------------------------------------------------------------------------- #
# select-m
# --------------------------------------------------------------------------- #
@pytest.fixture
def stream(rft_roots, tmp_path):  # noqa: F811
    out = tmp_path / "stream_rft"
    assert brs.main(_argv(rft_roots, out)) == 0
    paths = {}
    for line in (out / brs.RECORDS_LIST).read_text().splitlines():
        path, device, tag = line.split("\t")
        paths[(tag, Path(path).name)] = path
    return {"dir": out, "paths": paths}


def _d3_row(path, call, recorded, replay):
    head = Path(path).name[: -len(".s2rec.tar")]
    return {"key": f"{head}#{call}", "uid": f"{path}#{call}", "scene_id": head.rsplit("_", 1)[0], "kind": "native",
            "recorded": list(recorded), "replay": list(replay), "exact": list(recorded) == list(replay),
            "turn1_match": recorded[0] == replay[0], "class_match": True, "seconds": 1.0}


def _d3(tmp_path, rows, *, record_device="npu", device="npu"):
    d3 = tmp_path / "d3"
    d3.mkdir(exist_ok=True)
    (d3 / "parity_g0.calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d3 / "parity_g0.json").write_text(json.dumps({"n": len(rows), "exact_rate": 0.5, "record_device": record_device,
                                                   "device": device}))
    return d3 / "parity_g0.calls.jsonl"


PIX = ("↓", "245 320")


KF1, BB5, BB7 = "kfPV7w3FaU5_000001.s2rec.tar", "BBBBBBBBBBB_000005.s2rec.tar", "BBBBBBBBBBB_000007.s2rec.tar"


def _m_rows(stream):
    return [
        _d3_row(f"{D3_DIR}/p1/kfPV7w3FaU5/{KF1}", 0, ("←←",), ("→→",)),                 # M (one trajectory)
        _d3_row(f"{D3_DIR}/p1/kfPV7w3FaU5/{KF1}", 2, PIX, ("↓", "260 330")),             # replay agrees (15/10 px)
        _d3_row(f"{D3_DIR}/p1/kfPV7w3FaU5/{KF1}", 1, ("→→",), ("→→",)),                  # RF3 left it out
        _d3_row(f"{D3_DIR}/r0s/BBBBBBBBBBB/{BB5}", 1, ("↓", "300 200"), PIX),             # M (55 px off; p1 or r0s:
        #                                                                                   the tag directory tells)
        _d3_row(f"{D3_DIR}/records/BBBBBBBBBBB/{BB7}", 0, PIX, ("STOP",)),                # M (the episode's only one)
        _d3_row(stream["paths"][("r0s", "kfPV7w3FaU5_000006.s2rec.tar")], 0, PIX, ("←",)),  # M (the same path)
    ]


def test_m_takes_d3s_diverged_rft_rows_mapped_as_aud_maps_them(stream, tmp_path):
    calls = _d3(tmp_path, _m_rows(stream))
    out = tmp_path / "rgate" / "m_states.jsonl"
    argv = ["select-m", "--d3-calls", str(calls), "--rft-stream", str(stream["dir"]), "--output", str(out)]
    assert rg.main(argv) == 0
    states = _states(out)
    paths = stream["paths"]
    assert sorted((s["source"], s["uid"]) for s in states) == sorted([
        ("p1", paths[("p1", KF1)] + "#0"), ("r0s", paths[("r0s", BB5)] + "#1"), ("p2", paths[("p2", BB7)] + "#0"),
        ("r0s", paths[("r0s", "kfPV7w3FaU5_000006.s2rec.tar")] + "#0")])
    assert {s["group"] for s in states} == {"M"} and all(s["record_device"] == "npu" for s in states)
    bb5 = {s["uid"]: s for s in states}[paths[("r0s", BB5)] + "#1"]
    assert bb5["recorded"] == ["↓", "300 200"] and bb5["npu_replay"] == list(PIX)
    assert bb5["rft_key"] == "BBBBBBBBBBB_000005#r0s.1" and bb5["d3_uid"].startswith(D3_DIR + "/r0s/")
    keys = {line.split("\t")[0] for line in (stream["dir"] / brs.KEYS_TSV).read_text().splitlines()}
    assert {s["rft_key"] for s in states} <= keys
    meta = json.loads(out.with_suffix(".selection.json").read_text())
    assert meta["counts"] == {"d3_rows": 6, "rft_samples": 5, "replay_agrees": 1, "divergent": 4,
                              "call_not_an_rft_sample": 1}
    assert meta["records"] == {"d3": 4, "rft_trajectories": 4, "by_tag": {"p1": 1, "r0s": 2, "p2": 1}}
    assert meta["divergence_rate_of_rft_rows"] == 0.8 and meta["d3_exact_rate"] == 0.5

    # the same rows through aud.py dose's own reading: the same M states
    dose = tmp_path / "dose"
    dose.mkdir()
    rows = _m_rows(stream)
    (dose / "parity_g0.calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (dose / "parity_g0.json").write_text(json.dumps({"schema": "heatmapvln-s2-parity-v1", "n": len(rows),
                                                     "split": "all", "record_device": "npu", "device": "npu",
                                                     "exact_rate": 0.5}))
    aud.main(["dose", "--parity-dir", str(dose), "--rft-stream", str(stream["dir"]), "--m-states",
              str(dose / "m_states.jsonl"), "--out", str(dose / "dose.json")])
    assert sorted(json.loads(line)["uid"] for line in (dose / "m_states.jsonl").read_text().splitlines()) == \
        sorted(s["uid"] for s in states)

    # an episode with two trajectories needs the tag directory (or --record-map) to tell them
    untagged = f"{D3_DIR}/records/BBBBBBBBBBB/{BB5}"
    rows = [_d3_row(untagged, 1, ("↓", "300 200"), PIX)]
    with pytest.raises(SystemExit, match="cannot tell which RFT trajectory"):
        rg.main(argv[:2] + [str(_d3(tmp_path, rows))] + argv[3:])
    record_map = tmp_path / "record_map.tsv"
    record_map.write_text(f"{untagged}\t{paths[('r0s', BB5)]}\n")
    assert rg.main(argv[:2] + [str(_d3(tmp_path, rows))] + argv[3:] + ["--record-map", str(record_map)]) == 0
    assert [s["uid"] for s in _states(out)] == [paths[("r0s", BB5)] + "#1"]


def test_m_refuses_d3_records_that_are_not_the_streams(stream, tmp_path):
    out = tmp_path / "m.jsonl"
    argv = ["select-m", "--rft-stream", str(stream["dir"]), "--output", str(out), "--d3-calls"]
    for rows, match in (
        ([_d3_row(f"{D3_DIR}/p1/kfPV7w3FaU5/{KF1}", 0, ("→→",), ("→→",))], "not the stream's"),
        ([_d3_row(f"{D3_DIR}/p1/BBBBBBBBBBB/{BB5}", 0, ("←←",), ("→→",))], "answered by cuda"),  # p1's is cuda
        ([_d3_row(f"{D3_DIR}/p1/kfPV7w3FaU5/kfPV7w3FaU5_000099.s2rec.tar", 0, PIX, ("←",))], "cannot tell"),
        ([{**_d3_row(f"{D3_DIR}/p1/kfPV7w3FaU5/{KF1}", 0, ("←←",), ("→→",)), "key": "kfPV7w3FaU5_000002#0"}],
         "does not name its record"),
    ):
        with pytest.raises(SystemExit, match=match):
            rg.main(argv + [str(_d3(tmp_path, rows))])
    with pytest.raises(SystemExit, match="D3 replays npu calls"):
        rg.main(argv + [str(_d3(tmp_path, _m_rows(stream), record_device="cuda"))])
    keys = stream["dir"] / brs.KEYS_TSV
    keys.write_text(keys.read_text().replace("#p1.0\t", "#p1.1\t", 1))
    with pytest.raises(SystemExit, match="manifest names"):
        rg.main(argv + [str(_d3(tmp_path, _m_rows(stream)))])


# --------------------------------------------------------------------------- #
# score: the lines
# --------------------------------------------------------------------------- #
def _row(group, episode, on_agree, off_agree, source="s42", decision="pixel"):
    return {"group": group, "source": source, "record_key": f"S_{episode:06d}", "decision": decision,
            "on": {"turns": ["x"], "class": "pixel"}, "off": {"turns": ["y"], "class": "pixel"},
            "on_agree": on_agree, "off_agree": off_agree}


def _divergent(n_episodes, per_episode, agreeing_episodes, group="R", agreeing_per_episode=1, start=0):
    """Divergent rows (the base disagrees): the first ``agreeing_per_episode`` of each agreeing episode agree."""
    rows = []
    for e in range(n_episodes):
        for i in range(per_episode):
            rows.append(_row(group, start + e, e in agreeing_episodes and i < agreeing_per_episode, False))
    return rows


TWO_SEEDS, THREE_SEEDS = {42, 1337}, {42, 1337, 2024}


def test_r_line_is_inclusive_and_needs_a_positive_lower_bound():
    on_line = rg.r_summary(_divergent(80, 2, set(range(8))) + [_row("R", 99, True, True)] * 5, seeds=TWO_SEEDS)
    assert on_line["divergent"] == 160 and on_line["candidate_agree"] == 8 and on_line["value"] == 0.05
    assert on_line["states"] == 165 and on_line["ci95_episode_bootstrap"][0] > 0
    assert on_line["pass"] is True and on_line["report_only"] is False
    below = rg.r_summary(_divergent(80, 2, set(range(7))), seeds=TWO_SEEDS)
    assert below["value"] == 0.04375 and below["pass"] is False
    # D_R above the line, but all of it in one episode: the cluster bootstrap's lower bound is 0
    lumped = rg.r_summary(_divergent(1, 10, {0}, agreeing_per_episode=10) + _divergent(75, 2, set(), start=1),
                          seeds=TWO_SEEDS)
    assert lumped["value"] == 0.0625 and lumped["ci95_episode_bootstrap"][0] <= 0
    assert lumped["ci_lower_gt_0"] is False and lumped["pass"] is False
    assert lumped["bootstrap"]["resamples"] == 20_000


def test_r_is_report_only_under_150_divergent_states_once_seed_2024_is_in():
    at = rg.r_summary(_divergent(75, 2, set(range(40))), seeds=TWO_SEEDS)
    assert at["divergent"] == 150 and at["report_only"] is False and at["pass"] is True
    assert at["needs_seed_2024"] is False
    rows = _divergent(149, 1, set(range(149))) + [_row("R", 500 + i, True, True) for i in range(50)]
    under = rg.r_summary(rows, seeds=THREE_SEEDS)
    assert under["divergent"] == 149 and under["report_only"] is True and under["pass"] is None
    assert under["value"] == 1.0 and under["needs_seed_2024"] is False  # report-only however high
    before = rg.r_summary(rows, seeds=TWO_SEEDS)  # B3: under 150, seed 2024 is added once before R stands down
    assert before["report_only"] is False and before["needs_seed_2024"] is True and before["pass"] is None
    assert rg.r_summary([], seeds=THREE_SEEDS)["report_only"] is True


def test_m_line_and_which_gate_reads():
    def m_rows(n, on, off):
        return [_row("M", i, i < on, i < off, source="p1") for i in range(n)]

    assert rg.m_summary(m_rows(20, 5, 2))["value"] == 0.15 and rg.m_summary(m_rows(20, 5, 2))["learnt"] is True
    assert rg.m_summary(m_rows(20, 4, 2))["learnt"] is False
    assert rg.m_summary([])["learnt"] is None and rg.m_summary([])["measured"] is False

    gated = rg.summarize(_divergent(75, 2, set(range(40))) + m_rows(20, 4, 2), r_seeds=TWO_SEEDS)
    assert gated["gate"] == {"uses": "R", "pass": True, "why": "150 divergent states (>= 150)"}
    assert gated["M"]["learnt"] is False  # still read: it picks the retry after an R failure
    few = _divergent(10, 1, {0})
    via_m = rg.summarize(few + m_rows(20, 5, 2), r_seeds=THREE_SEEDS)
    assert via_m["gate"]["uses"] == "M" and via_m["gate"]["pass"] is True and via_m["R"]["report_only"]
    assert rg.summarize(few + m_rows(20, 4, 2), r_seeds=THREE_SEEDS)["gate"]["pass"] is False
    alone = rg.summarize(few, r_seeds=THREE_SEEDS)
    assert rg.measurement_problems(alone) == ["R is report-only and M has no state: neither can gate"]
    assert rg.measurement_problems(gated) == []
    waiting = rg.summarize(few + m_rows(20, 5, 2), r_seeds=TWO_SEEDS)  # M is not read before the seed-2024 pass
    assert waiting["gate"]["uses"] is None and waiting["gate"]["pass"] is None
    assert waiting["gate"]["needs"] == "heldout_s2024" and "seed-2024" in rg.measurement_problems(waiting)[0]
    with_fail = rg.summarize(_divergent(75, 2, set(range(40))) + _divergent(10, 2, {0, 1}, group="R_fail"),
                             r_seeds=TWO_SEEDS)
    assert with_fail["R_fail"] == {"states": 20, "divergent": 20, "candidate_agree": 2, "value": 0.1, "measured": True}
    assert with_fail["delta_R_minus_R_fail"] == round(40 / 150 - 0.1, 6)


# --------------------------------------------------------------------------- #
# decode (no GPU: the model is faked) and score end to end
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_model(monkeypatch):
    import torch

    import scripts.train_system2_lora as T
    import src.data.s2_sft_dataset as D

    model = SimpleNamespace(off=False, eval=lambda: None, decoded=[])

    @contextlib.contextmanager
    def adapter_off(m):
        m.off = True
        try:
            yield
        finally:
            m.off = False

    def greedy(m, processor, messages, images, lookdown, device):
        m.decoded.append((messages, m.off))
        uid = messages
        if m.off:
            return ("→→",)  # the base disagrees with every recorded pixel or left turn
        return ("↓", "250 330") if int(uid.rsplit("#", 1)[1]) % 2 == 0 else ("↑↑",)

    monkeypatch.setattr(T, "init_runtime", lambda kind: SimpleNamespace(device=torch.device("cpu")))
    monkeypatch.setattr(T, "seed_everything", lambda seed, device: None)
    monkeypatch.setattr(T, "load_system2", lambda path, device: (model, None))
    monkeypatch.setattr(T, "load_adapter", lambda m, path: m)
    monkeypatch.setattr(T, "adapter_off", adapter_off)
    monkeypatch.setattr(T, "greedy_native_decision", greedy)
    monkeypatch.setattr(D, "inference_messages",
                        lambda record, call, default_vlm_image_size: (f"{record.path}#{call['call_index']}", [], None))
    return model


def _adapter(tmp_path, data=b"lora-1"):
    adapter = tmp_path / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_model.safetensors").write_bytes(data)
    return adapter


def _decode(states, adapter, out, *extra):
    return rg.main(["decode", "--r-states", str(states), "--adapter", str(adapter), "--model-path", "/m",
                    "--output", str(out), "--device", "cuda", *extra])


def test_decode_shards_resumes_and_refuses_another_adapter(heldout, tmp_path, fake_model, capsys):
    states = tmp_path / "rgate" / "r_states.jsonl"
    assert _select_r(heldout, states, "--failed") == 0
    adapter = _adapter(tmp_path)
    out = tmp_path / "rgate" / "rft_gate.json"
    assert _decode(states, adapter, out, "--dry-run") == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == {
        "states": {"R": 8, "R_fail": 8}, "calls": 16, "shard": 16}
    assert fake_model.decoded == []
    assert _decode(states, adapter, out, "--shard", "0/2") == 0 and _decode(states, adapter, out, "--shard", "1/2") == 0
    shards = [out.with_name(f"rft_gate.decode.shard{i:02d}of02.jsonl") for i in (0, 1)]
    rows = [json.loads(line) for p in shards for line in p.read_text().splitlines()]
    assert sorted(r["uid"] for r in rows) == sorted(s["uid"] for s in _states(states))
    assert all(rg.shard_of(json.loads(line)["uid"], 2) == i for i, p in enumerate(shards)
               for line in p.read_text().splitlines())
    assert {r["adapter_sha256"] for r in rows} == {rg._sha256(adapter / "adapter_model.safetensors")}
    assert all(r["off"] == {"turns": ["→→"], "class": "turn_right"} for r in rows)
    assert Counter(off for _, off in fake_model.decoded) == {False: 16, True: 16}  # on, then off inside adapter_off
    assert _decode(states, adapter, out, "--shard", "0/2") == 0
    assert len(fake_model.decoded) == 32  # resumed: nothing new
    _adapter(tmp_path, b"lora-2")
    with pytest.raises(SystemExit, match="another adapter"):
        _decode(states, adapter, out, "--shard", "0/2")
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text("".join(json.dumps({**s, "recorded": ["STOP"]}) + "\n" for s in _states(states)))
    with pytest.raises(SystemExit, match="select again"):
        _decode(tampered, adapter, tmp_path / "other" / "rft_gate.json")


def _score(r_states, out, *extra):
    return rg.main(["score", "--r-states", str(r_states), "--output", str(out), *extra])


def test_score_end_to_end(heldout, tmp_path, fake_model, monkeypatch, capsys):
    states = tmp_path / "rgate" / "r_states.jsonl"
    assert _select_r(heldout, states, "--failed") == 0
    adapter = _adapter(tmp_path)
    out = tmp_path / "rgate" / "rft_gate.json"
    assert _decode(states, adapter, out) == 0
    # 8 R states, all divergent (the fake base turns right): report-only and no M -> invalid
    assert _score(states, out, "--adapter", str(adapter)) == 2
    report = json.loads(out.read_text())
    assert report["valid"] is False and "without B3's seed-2024 pass" in report["invalid"]
    assert report["R"]["divergent"] == 8 and report["R"]["needs_seed_2024"] is True
    assert report["gate"]["needs"] == "heldout_s2024"

    # R gated (the line moved down to these 8 states): the candidate answers its fake pixel on even calls
    monkeypatch.setattr(rg, "R_MIN_DIVERGENT", 8)
    assert _score(states, out, "--adapter", str(adapter)) in (0, 1)
    report = json.loads(out.read_text())
    r_states = [s for s in _states(states) if s["group"] == "R"]
    want = sum(1 for s in r_states if s["call_index"] % 2 == 0 and s["decision"] == "pixel"
               and all(abs(a - b) <= 50 for a, b in zip(map(int, s["recorded"][-1].split()), (250, 330))))
    assert report["valid"] and report["R"]["candidate_agree"] == want and report["gate"]["uses"] == "R"
    assert report["R_fail"]["measured"] and report["R"]["by_source"]["s1337"]["states"] == 2
    assert report["selection"]["R"]["states"]["R"] == 8
    rows = [json.loads(line) for line in out.with_suffix(".calls.jsonl").read_text().splitlines()]
    assert len(rows) == 16 and all(not r["off_agree"] for r in rows)

    # B3's seed-2024 pass added (resume decodes only its states): R is report-only, and M states (here: the same
    # calls read as M) make the gate readable
    monkeypatch.setattr(rg, "R_MIN_DIVERGENT", 150)
    s2024 = tmp_path / "heldout_s2024" / "records"
    _record(s2024, "TEEsavR23oF", 3, [("L", 6.0), ("P", 7.0)], seed=2024, device="cuda")
    three = [f"s42={heldout['s42']}:42", f"s1337={heldout['s1337']}:1337", f"s2024={s2024}:2024"]
    assert _select_r(heldout, states, "--failed", sources=three) == 0
    before = len(fake_model.decoded)
    assert _decode(states, adapter, out) == 0
    assert len(fake_model.decoded) == before + 2 * 4  # 2 new R states, and R_fail grows to 10
    m_states = tmp_path / "rgate" / "m_states.jsonl"
    m_states.write_text("".join(json.dumps({**s, "group": "M"}) + "\n" for s in r_states))
    assert _score(states, out, "--adapter", str(adapter)) == 2
    assert "neither can gate" in json.loads(out.read_text())["invalid"]
    code = _score(states, out, "--m-states", str(m_states), "--adapter", str(adapter))
    report = json.loads(out.read_text())
    assert report["valid"] and report["gate"]["uses"] == "M" and report["M"]["states"] == 8
    assert report["R"]["report_only"] and report["R"]["seeds"] == [42, 1337, 2024] and report["R"]["states"] == 10
    assert code == (0 if report["M"]["learnt"] else 1) and report["M"]["value"] == round(want / 8, 6)

    # what cannot be measured: another adapter, a state not decoded, a file of the wrong kind
    assert _score(states, out, "--m-states", str(m_states), "--adapter", str(_adapter(tmp_path / "x", b"b"))) == 2
    assert "--adapter is" in json.loads(out.read_text())["invalid"]
    assert _score(m_states, out) == 2 and "not a select-r state" in json.loads(out.read_text())["invalid"]
    extra = tmp_path / "rgate" / "extra.jsonl"
    extra.write_text(states.read_text() + json.dumps({**r_states[0], "uid": r_states[0]["uid"] + "0"}) + "\n")
    assert _score(extra, out) == 2 and "not decoded" in json.loads(out.read_text())["invalid"]
    shard = out.with_name("rft_gate.decode.shard00of01.jsonl")
    lines = shard.read_text().splitlines()
    shard.write_text("\n".join(lines[:-1] + [json.dumps({**json.loads(lines[-1]), "adapter_sha256": "f00"})]) + "\n")
    assert _score(states, out) == 2 and "2 adapters" in json.loads(out.read_text())["invalid"]
    capsys.readouterr()


def test_the_cli_parses():
    with pytest.raises(SystemExit):
        rg.parse_args(["decode", "--adapter", "a", "--model-path", "m", "--output", "o"])  # no states
    args = rg.parse_args(["score", "--r-states", "r", "--output", "o"])
    assert args.m_states is None and args.adapter == ""
    assert rg.R_MIN_DELTA == Fraction(5, 100) and rg.M_MIN_DELTA == Fraction(15, 100) and rg.R_MIN_DIVERGENT == 150

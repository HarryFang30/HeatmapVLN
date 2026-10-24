"""EXP-35's offline gate (scripts/exp35/stop_rate_gate.py): the gate episodes' states, STOP rates by
distance with the adapter on and off, and the far-STOP line in whole counts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.exp35 import stop_rate_gate as G
from src.data.s2_records import EpisodeRecordWriter
from tests.test_s2_oracle_labels import _jpeg

X34M = "9f" * 32
ANSWER = {
    "S": {"llm_output": "STOP", "native_first_output": "STOP", "native_lookdown_turns": 0},
    "L": {"llm_output": "←←", "native_first_output": "←←", "native_lookdown_turns": 0},
    "P": {"llm_output": "245 320", "native_first_output": "↓", "native_lookdown_turns": 1},
}


def _record(root, scene, episode, calls, *, adapter=X34M):
    """calls: (answer, distance to goal) per call."""
    version = "ppa-stage2-online-amb3r:zhr_1" + (f"+system2-lora:{adapter}" if adapter else "")
    writer = EpisodeRecordWriter(root=root, scene_id=scene, episode_id=episode,
                                 episode={"instruction": "go", "split": "train", "server": {"model_version": version}})
    for i, (answer, d) in enumerate(calls):
        writer.add_step(i, [0.0, 0.0, -0.25 * i], [1.0, 0.0, 0.0, 0.0])
        writer.add_front(i, _jpeg((64, 48)))
        writer.add_call({"call_index": i, "step_id": i, "history_step_ids": [], "kind": "trajectory",
                         "distance_to_goal": d, **ANSWER[answer]},
                        lookdown_jpeg=_jpeg(), lookdown_depth_m=np.full((480, 640), 1.5, dtype=np.float32))
    return writer.commit()


@pytest.fixture
def gate(tmp_path):
    records, cohort = tmp_path / "records", tmp_path / "cohorts"
    cohort.mkdir()
    (cohort / "shard_000.json").write_text(json.dumps({"episodes": [{"scene_id": "sceneA", "episode_id": 1}]}))
    (cohort / "shard_001.json").write_text(json.dumps({"episodes": [{"scene_id": "sceneB", "episode_id": 2}]}))
    _record(records, "sceneA", 1, [("P", 8.0), ("L", 5.0), ("L", 2.0), ("S", 0.6)])
    _record(records, "sceneB", 2, [("L", 9.0), ("S", 2.5), ("L", float("inf"))])
    _record(records, "sceneC", 3, [("S", 7.0)])  # not a gate episode
    return records, cohort


def _sel(records, cohort, shards="000,001", adapter=X34M):
    return G.select_states([str(records)], str(cohort), shards.split(","), learner_adapter=adapter)


def test_states_are_every_call_of_the_gate_episodes(gate):
    records, cohort = gate
    states, meta = _sel(*gate)
    assert [s["key"] for s in states] == ["sceneA_000001#0", "sceneA_000001#1", "sceneA_000001#2", "sceneA_000001#3",
                                          "sceneB_000002#0", "sceneB_000002#1"]
    assert [s["band"] for s in states] == [">3m", ">3m", "1-3m", "<=1m", ">3m", "1-3m"]
    assert states[0]["recorded"] == ["↓", "245 320"] and states[3]["recorded"] == ["STOP"]
    assert meta["episodes_found"] == 2 and meta["episodes_missing"] == 0 and meta["stats"]["no_distance"] == 1
    # a gate episode with no record yet is counted, not invented
    (cohort / "shard_002.json").write_text(json.dumps({"episodes": [{"scene_id": "sceneD", "episode_id": 4}]}))
    _, meta = _sel(records, cohort, "000,001,002")
    assert meta["episodes_missing"] == 1
    # the records must be the learner's own
    with pytest.raises(SystemExit, match="served by"):
        _sel(records, cohort, adapter="ab" * 32)


def _rows(states, on, off, adapter="cd" * 32):
    return {s["key"]: {"key": s["key"], "adapter_sha256": adapter, "on": {"turns": [a]}, "off": {"turns": [b]}}
            for s, a, b in zip(states, on, off)}


def test_rates_by_band_and_the_far_stop_line_in_whole_counts():
    far = [{"key": f"e_{i:06d}#{i}", "band": ">3m", "recorded": ["←"]} for i in range(200)]
    near = [{"key": f"n_{i:06d}#{i}", "band": "<=1m", "recorded": ["STOP"]} for i in range(10)]
    states = far + near
    # off: 2 far STOPs; on: 4 -> +1.0 pt on 200 calls, exactly on the line: passes
    on = ["STOP"] * 4 + ["←"] * 196 + ["STOP"] * 9 + ["←"]
    off = ["STOP"] * 2 + ["←"] * 198 + ["STOP"] * 5 + ["←"] * 5
    out = G.summarize(states, _rows(states, on, off))
    assert out["bands"][">3m"] == {"calls": 200, "episodes": 200, "stop_on": 4, "stop_off": 2, "stop_recorded": 0,
                                   "rate_on": 0.02, "rate_off": 0.01, "rate_recorded": 0.0, "on_only": 2, "off_only": 0}
    assert out["bands"]["<=1m"]["stop_on"] == 9 and out["bands"]["<=1m"]["rate_recorded"] == 1.0
    assert out["gate"]["pass"] is True and out["gate"]["delta_pt"] == 1.0
    on[4] = "STOP"  # one more far STOP: +1.5 pt
    assert G.summarize(states, _rows(states, on, off))["gate"]["pass"] is False
    # an empty answer is the server's fallback STOP
    assert G.is_stop([""]) and not G.is_stop(["↓", "245 320"])


def test_score_end_to_end_and_invalid_inputs(gate, tmp_path):
    import hashlib

    records, cohort = gate
    states, _ = _sel(*gate)
    output = tmp_path / "gate" / "stop_gate.json"
    output.parent.mkdir()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    sha = hashlib.sha256(b"weights").hexdigest()
    manifest = tmp_path / "stream_manifest.json"
    manifest.write_text(json.dumps({"holdout": {"gate_cohort_shards": [str(cohort), "000,001"]}}))
    argv = ["score", "--record-root", str(records), "--cohort-dir", str(cohort), "--shards", "000,001",
            "--learner-adapter", X34M, "--output", str(output), "--adapter", str(adapter), "--stream-manifest", str(manifest)]
    assert G.main(argv) == 2  # nothing decoded yet
    assert json.loads(output.read_text())["status"] == "invalid"
    rows = _rows(states, ["←", "←", "←", "STOP", "←", "STOP"], ["←", "←", "←", "STOP", "←", "←"], adapter=sha)
    half = sorted(rows)[:3]
    for i, keys in enumerate((half, sorted(set(rows) - set(half)))):
        path = output.with_name(f"stop_gate.decode.shard{i:02d}of02.jsonl")
        path.write_text("".join(json.dumps(rows[k]) + "\n" for k in keys))
    assert G.main(argv) == 0
    result = json.loads(output.read_text())
    assert result["status"] == "pass" and result["bands"]["1-3m"]["stop_on"] == 1
    assert result["bands"][">3m"]["calls"] == 3 and result["gate"]["delta_pt"] == 0.0
    # an invalid input is exit 2 and overwrites the old pass, never the 1 of a failed gate
    bad_learner = argv[:argv.index("--learner-adapter") + 1] + ["ab" * 32] + argv[argv.index("--learner-adapter") + 2:]
    assert G.main(bad_learner) == 2 and json.loads(output.read_text())["status"] == "invalid"
    other_stream = tmp_path / "other_manifest.json"
    other_stream.write_text(json.dumps({"holdout": {"gate_cohort_shards": [str(cohort), "002,003"]}}))
    assert G.main(argv[:-1] + [str(other_stream)]) == 2
    assert "held out shards" in json.loads(output.read_text())["problems"][0]
    assert G.main(argv) == 0
    # rows of another adapter than the candidate's are not its measurement
    other = output.with_name("stop_gate.decode.shard01of02.jsonl")
    other.write_text(other.read_text().replace(sha, "ef" * 32))
    assert G.main(argv) == 2


def test_dry_run_counts_the_states(gate, tmp_path, capsys):
    records, cohort = gate
    argv = ["decode", "--record-root", str(records), "--cohort-dir", str(cohort), "--shards", "000,001",
            "--learner-adapter", X34M, "--adapter", str(tmp_path), "--model-path", str(tmp_path),
            "--output", str(tmp_path / "g.json"), "--shard", "0/1", "--dry-run"]
    assert G.main(argv) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["states"] == 6

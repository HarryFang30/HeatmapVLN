"""EXP-25 line B stream: capped corrections, anchors from the model's own answers, no STOP."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from scripts.tools import build_route_stream as rs
from src.data.s2_sft_dataset import S2SFTError, index_calls
from src.data.s2_stream import MANIFEST_NAME, read_shard
from tests.test_s2_sft_dataset import _write_labels, _write_record

PIXEL = {"llm_output": "245 320", "native_first_output": "↓", "native_lookdown_turns": 1}
LEFT = {"llm_output": "←←", "native_first_output": "←←", "native_lookdown_turns": 0}


def _episode(records, labels, episode, kinds):
    """kinds: per call 'c' (correction to a right turn), 'a' (anchor), 'n' (no target)."""
    calls = []
    for i, _ in enumerate(kinds):
        step = 2 * i + 2
        calls.append({"call_index": i, "step_id": step, "history_step_ids": list(range(0, step, 2))[-3:],
                      **(PIXEL if i % 2 == 0 else LEFT)})
    path = _write_record(records, "sceneA", episode, calls)
    rows = []
    for call, kind in zip(calls, kinds):
        recorded = (call["native_first_output"], call["llm_output"]) if call["native_lookdown_turns"] else (call["llm_output"],)
        row = {"call_index": call["call_index"], "distance_to_goal": 5.0, "oracle_kind": "turn",
               "oracle_turn1": "→→", "oracle_turn2": None}
        if kind == "c":
            row.update(agree=False, target_turn1="→→", target_turn2=None, kind="correction")
        elif kind == "a":
            row.update(agree=True, target_turn1=recorded[0], target_turn2=recorded[1] if len(recorded) > 1 else None,
                       kind="anchor")
        else:
            row.update(agree=None, skip=True, oracle_kind="skip", oracle_turn1=None, kind="none")
        rows.append(row)
    _write_labels(path, labels, rows)
    return path


@pytest.fixture
def roots(tmp_path):
    records, labels = tmp_path / "records", tmp_path / "labels"
    _episode(records, labels, 1, "ccccacaa")  # 5 corrections, 3 anchors
    _episode(records, labels, 2, "aaaanaaa")  # 7 anchors
    return records, labels


def test_pools_cap_corrections_earliest_and_anchors_uniformly(roots):
    refs, _ = index_calls([roots[0]], [roots[1]], split="all", holdout=())
    corrections, anchors, stats = rs.route_pools(refs, max_corrections_per_episode=3, max_anchors_per_episode=4)
    assert [r.call_index for r in corrections] == [0, 1, 2]  # the earliest three of 0,1,2,3,5
    assert all(r.turns == ("→→",) for r in corrections)
    assert stats["corrections_before_cap"] == 5 and stats["anchors_before_cap"] == 10 and len(anchors) == 3 + 4
    assert all(r.turns == r.recorded for r in anchors)
    uniform, _, _ = rs.route_pools(refs, max_corrections_per_episode=3, max_anchors_per_episode=None,
                                   correction_pick="uniform")
    assert len(uniform) == 3 and {r.call_index for r in uniform} <= {0, 1, 2, 3, 5}


def test_a_stop_target_in_line_b_is_refused(tmp_path):
    records, labels = tmp_path / "records", tmp_path / "labels"
    path = _episode(records, labels, 1, "c")
    _write_labels(path, labels, [{"call_index": 0, "distance_to_goal": 5.0, "oracle_kind": "stop",
                                  "oracle_turn1": "STOP", "oracle_turn2": None, "agree": False}])
    refs, _ = index_calls([records], [labels], split="all", holdout=())
    with pytest.raises(S2SFTError, match="never targets STOP"):
        rs.route_pools(refs, max_corrections_per_episode=None, max_anchors_per_episode=None)


def test_take_subsamples_and_repeats():
    rng = np.random.default_rng(0)
    assert len(rs.take(list(range(10)), 0.5, rng)) == 5
    out = rs.take(list(range(10)), 1.5, rng)
    assert len(out) == 15 and Counter(out).most_common(1)[0][1] == 2


def test_end_to_end_stream(roots, tmp_path):
    records, labels = roots
    out = tmp_path / "stream"
    argv = ["--record-root", str(records), "--label-root", str(labels), "--out-dir", str(out),
            "--max-corrections-per-episode", "3", "--anchors-per-correction", "1", "--shard-size", "8",
            "--global-batch", "8", "--world-size", "1", "--holdout-count", "0", "--workers", "1"]
    assert rs.main(argv) == 0
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert manifest["total_samples"] == 8 and manifest["num_shards"] == 1
    drawn = manifest["drawn"]
    # 3 corrections, 3 anchors, padded with spare anchors (no replay) to a whole shard
    assert drawn["category_B_correction_turn"] == 3 and drawn["category_B_anchor"] == 5
    assert "dagger_target_stop" not in drawn
    assert manifest["selection"]["knobs"]["max_corrections_per_episode"] == 3
    samples = [s for shard in manifest["shards"] for s in read_shard(out / shard["name"])]
    assert len(samples) == 8 and len({s.key for s in samples}) == 8
    with pytest.raises(SystemExit, match="already holds"):
        rs.main(argv)

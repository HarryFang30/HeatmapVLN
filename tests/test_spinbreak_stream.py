"""EXP-25 SpinBreak stream: STOP only in the spin trap, every other target the model's own answer."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from scripts.tools import build_spinbreak_stream as sb
from src.data.s2_records import EpisodeRecordWriter
from src.data.s2_sft_dataset import index_calls
from src.data.s2_stream import MANIFEST_NAME, read_shard
from tests.test_s2_sft_dataset import INSTRUCTION, _jpeg, _write_labels, _write_record

ANSWERS = {
    "pixel": {"llm_output": "245 320", "native_first_output": "↓", "native_lookdown_turns": 1},
    "left": {"llm_output": "←←", "native_first_output": "←←", "native_lookdown_turns": 0},
    "right": {"llm_output": "→", "native_first_output": "→", "native_lookdown_turns": 0},
    "stop": {"llm_output": "STOP", "native_first_output": "STOP", "native_lookdown_turns": 0},
}


def _answer_calls(answers):
    calls = []
    for i, answer in enumerate(answers):
        step = 2 * i + 2
        calls.append({"call_index": i, "step_id": step, "history_step_ids": list(range(0, step, 2))[-3:], **ANSWERS[answer]})
    return calls


def _episode(records, labels, episode, plan):
    """plan: [(answer, distance)]; returns the record path."""
    path = _write_record(records, "sceneA", episode, _answer_calls([a for a, _ in plan]))
    _write_labels(path, labels, [{"call_index": i, "agree": None, "distance_to_goal": d} for i, (_, d) in enumerate(plan)])
    return path


FAILED = [  # spins inside 1 m and never stops
    ("pixel", 5.0), ("left", 0.9), ("left", 0.8), ("left", 0.7), ("right", 0.7), ("left", 0.6), ("left", 0.5),
    ("pixel", 4.0),
]
SUCCESS = [
    ("left", 7.0),    # 0 first call beyond 3 m: no label
    ("pixel", 7.0),   # 1 after a turn, > 3 m: anchor_post_turn
    ("pixel", 5.0),   # 2 after a pixel, 3-6 m: anchor_post_pixel
    ("pixel", 7.5),   # 3 after a pixel, > 6 m: no label
    ("right", 2.0),   # 4 after a pixel, 1.25-3 m, kept going: anchor_keep_going
    ("pixel", 2.5),   # 5 after a turn, 1.25-3 m: no label
    ("pixel", 0.9),   # 6 after a pixel, <= 1 m: no label (first call within 1 m)
    ("left", 0.5),    # 7 after a pixel: not a spin state
    ("left", 0.5),    # 8 after a turn, an earlier call within 1 m, did not stop: stop_spin
    ("stop", 0.4),    # 9 the model's own STOP within 1 m: anchor_own_stop
]
FAR_STOP = [("pixel", 4.0), ("stop", 2.0)]  # a successful STOP at 2 m is never a target


@pytest.fixture
def episodes(tmp_path):
    records, labels = tmp_path / "records", tmp_path / "labels"
    paths = {
        "failed": _episode(records, labels, 1, FAILED),
        "success": _episode(records, labels, 2, SUCCESS),
        "far_stop": _episode(records, labels, 3, FAR_STOP),
    }
    refs, _ = index_calls([records], [labels], split="train", holdout=())
    success = {str(paths["failed"]): False, str(paths["success"]): True, str(paths["far_stop"]): True}
    return paths, refs, success


def test_categories_follow_the_exp25_setup(episodes):
    paths, refs, success = episodes
    selected, stats = sb.categorize(refs, success)
    got = {(ref.record_path, ref.call_index): (category, ref.turns) for ref, category in selected}
    failed, ok = str(paths["failed"]), str(paths["success"])
    assert got == {
        # failed episode: the first three spin states only (call 1 has no earlier call within 1 m)
        (failed, 2): ("stop_spin", ("STOP",)),
        (failed, 3): ("stop_spin", ("STOP",)),
        (failed, 4): ("stop_spin", ("STOP",)),
        (ok, 1): ("anchor_post_turn", ("↓", "245 320")),
        (ok, 2): ("anchor_post_pixel", ("↓", "245 320")),
        (ok, 4): ("anchor_keep_going", ("→",)),
        (ok, 8): ("stop_spin", ("STOP",)),
        (ok, 9): ("anchor_own_stop", ("STOP",)),
    }
    assert stats["skip_spin_over_episode_cap"] == 2
    assert stats["skip_post_turn_1.25_3m_unlabelled"] == 1
    assert stats["skip_own_stop_beyond_own_stop_m"] == 1
    assert stats["stop_spin_failed_episode"] == 3 and stats["stop_spin_successful_episode"] == 1
    assert stats["episodes_successful"] == 2 and stats["episodes_failed"] == 1


def test_no_oracle_target_reaches_the_stream(episodes):
    _, refs, success = episodes
    # Even with oracle answers in the labels, a target is STOP or the recorded answer.
    for ref, category in sb.categorize(refs, success)[0]:
        assert ref.turns == ("STOP",) if category == "stop_spin" else ref.turns == ref.recorded


def test_draw_takes_every_stop_and_fills_with_anchors_without_replacement():
    pools = {"stop_spin": list(range(5)), "anchor_own_stop": list(range(100, 110)),
             "anchor_post_pixel": list(range(200, 230)), "anchor_post_turn": list(range(300, 320)),
             "anchor_keep_going": list(range(400, 410))}
    replay = list(range(1000, 1050))
    chosen, counts = sb.draw(pools, replay, total=64, replay_samples=20, own_stop_max=4, seed=0)
    assert len(chosen) == 64 and len({x for x, _ in chosen}) == 64
    assert counts["stop_spin"] == 5 and counts["anchor_own_stop"] == 4 and counts["replay"] == 20
    assert sum(counts[c] for c in sb.ANCHOR_CATEGORIES) == 64 - 29
    _, capped = sb.draw(pools, replay, total=64, replay_samples=20, own_stop_max=None, spin_stop_max=2, seed=0)
    assert capped["stop_spin"] == 2 and capped["anchor_own_stop"] == 10
    with pytest.raises(SystemExit, match="anchors needed"):
        sb.draw(pools, replay, total=200, replay_samples=20, own_stop_max=None, seed=0)
    with pytest.raises(SystemExit, match="more than the total"):
        sb.draw(pools, replay, total=16, replay_samples=20, own_stop_max=None, seed=0)


def _record_with_outcome(root, episode, plan, success):
    writer = EpisodeRecordWriter(root=root, scene_id="sceneB", episode_id=episode,
                                 episode={"instruction": INSTRUCTION, "split": "train", "vlm_image_size": [384, 384],
                                          "outcome": {"success": float(success)}})
    calls = _answer_calls([a for a, _ in plan])
    for step in sorted({s for c in calls for s in c["history_step_ids"] + [c["step_id"]]}):
        writer.add_step(step, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        writer.add_front(step, _jpeg(1000 * episode + step))
    for call, (_, d) in zip(calls, plan):
        writer.add_call({"kind": "x", "distance_to_goal": d, **call}, lookdown_jpeg=_jpeg(77 + call["call_index"]),
                        lookdown_depth_m=np.full((480, 640), 1.5, dtype=np.float32))
    return writer.commit()


def test_end_to_end_stream_is_what_the_trainer_reads(tmp_path):
    records, labels = tmp_path / "records", tmp_path / "labels"
    for episode, (plan, success) in enumerate([(FAILED, False), (SUCCESS, True)], start=1):
        path = _record_with_outcome(records, episode, plan, success)
        _write_labels(path, labels, [{"call_index": i, "agree": None, "distance_to_goal": d} for i, (_, d) in enumerate(plan)])
    out = tmp_path / "stream"
    argv = ["--record-root", str(records), "--label-root", str(labels), "--out-dir", str(out),
            "--replay-samples", "0", "--total-samples", "8", "--shard-size", "8", "--global-batch", "8",
            "--world-size", "1", "--holdout-count", "0", "--workers", "1"]
    assert sb.main(argv) == 0
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    for key in ("schema", "shard_size", "global_batch", "holdout", "total_samples", "shards", "seed", "num_shards",
                "replay_fraction", "epochs", "selection", "drawn"):
        assert key in manifest
    assert manifest["num_shards"] == 1 and manifest["total_samples"] == 8
    samples = read_shard(out / manifest["shards"][0]["name"])
    assert len(samples) == 8
    targets = Counter(s.turns for s in samples)
    assert targets[("STOP",)] == 5  # 4 spin STOPs + the model's own STOP
    assert manifest["drawn"]["category_stop_spin"] == 4 and manifest["drawn"]["category_anchor_own_stop"] == 1
    assert manifest["selection"]["stop_share"] == round(5 / 8, 4)
    assert all(s.source == "dagger" and s.instruction == INSTRUCTION for s in samples)
    with pytest.raises(SystemExit, match="already holds"):
        sb.main(argv)

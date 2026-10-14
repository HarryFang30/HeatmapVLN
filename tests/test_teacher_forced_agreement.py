from scripts.tools.teacher_forced_agreement import (
    b3_reading,
    cluster_bootstrap_ci,
    pick_samples,
    prefetched,
    sample_key,
    score,
    summarize,
)
from src.data.interndata_replay import ReplaySample


def _sample(scene, ep, frame, kind="turn", turn1="←", turn2=None, instruction="Go."):
    return ReplaySample(scene_id=scene, episode_index=ep, instruction=instruction, frame_id=frame,
                        history_frame_ids=(), kind=kind, turn1=turn1, turn2=turn2)


def test_score_classes_and_pixel_distance():
    row = score(("↓", "100 200"), ("↓", "103 204"))
    assert row["expert_class"] == row["model_class"] == "pixel"
    assert row["class_match"] and not row["exact"]
    assert row["pixel_distance_px"] == 5.0
    row = score(("←←",), ("←",))
    assert row["class_match"] and not row["exact"] and "pixel_distance_px" not in row
    row = score(("STOP",), ("↓", "10 20"))
    assert (row["expert_class"], row["model_class"], row["class_match"]) == ("stop", "pixel", False)
    assert score(("→",), ("↑↑",))["model_class"] == "forward"


def test_pick_dedupes_and_subsets_per_scene_deterministically():
    samples = [_sample("a", e, f) for e in range(5) for f in range(0, 40, 4)] + [_sample("b", 0, 0)]
    samples.append(_sample("a", 0, 0))  # same instruction twice in one task string
    picks, stats = pick_samples(samples, max_per_scene=7, seed=1)
    assert stats["duplicate_keys"] == 1
    assert stats["scenes"]["a"] == {"samples": 50, "picked": 7, "episodes": 5, "expert_classes": {"turn_left": 50}}
    assert stats["scenes"]["b"]["picked"] == 1
    again, _ = pick_samples(list(reversed(samples)), max_per_scene=7, seed=1)
    assert [sample_key(s) for s in picks] == [sample_key(s) for s in again]
    other, _ = pick_samples(samples, max_per_scene=7, seed=2)
    assert {sample_key(s) for s in picks} != {sample_key(s) for s in other}
    everything, _ = pick_samples(samples, max_per_scene=0, seed=1)
    assert len(everything) == 51


def test_summary_and_reading():
    rows = []
    for ep in range(10):
        rows.append({"scene_id": "s", "episode_index": ep, **score(("↓", "100 100"), ("↓", "100 110"))})
        rows.append({"scene_id": "s", "episode_index": ep, **score(("STOP",), ("STOP",) if ep < 8 else ("←",))})
    summary = summarize(rows)
    assert summary["n"] == 20 and summary["episodes"] == 10
    assert summary["overall_rate"] == 0.9
    assert summary["by_expert_class"]["stop"] == {
        "n": 10, "agree": 8, "rate": 0.8, "model_classes": {"stop": 8, "turn_left": 2}, "exact": 0.8}
    assert summary["by_expert_class"]["pixel"]["rate"] == 1.0
    assert summary["macro_rate_over_expert_classes"] == 0.9
    assert summary["pixel_vs_pixel"]["median_px"] == 10.0
    assert summary["pixel_vs_pixel"]["within_10px"] == 1.0
    assert summary["b3"]["reading"] == "GO"
    low, high = summary["overall_ci95_episode_bootstrap"]
    assert low <= 0.9 <= high
    assert b3_reading(0.93) == "GO" and b3_reading(0.95) == "undecided" and b3_reading(0.97) == "NO-GO"
    assert cluster_bootstrap_ci([]) is None


def test_prefetch_keeps_order():
    items = list(range(23))
    assert [(i, r) for i, r in prefetched(lambda x: x * x, items, workers=3, window=4)] == [(i, i * i) for i in items]
    assert list(prefetched(lambda x: x, [], workers=2, window=4)) == []
    assert [r for _, r in prefetched(lambda x: -x, items, workers=0, window=4)] == [-i for i in items]

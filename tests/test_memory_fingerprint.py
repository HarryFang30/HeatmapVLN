import json

from scripts.tools.memory_fingerprint import advantage, load, verdict


def _rows(n_eps, agree, near, *, start=4):
    rows = []
    for e in range(n_eps):
        for k in range(10):
            ok = k < agree
            rows.append({"scene_id": "s", "episode_index": e, "frame_id": start + k, "expert_class": "pixel",
                         "class_match": ok, "pixel_distance_px": 3.0 if k < near else 40.0})
    return rows


def test_load_drops_stop_targets_and_the_first_frames(tmp_path):
    path = tmp_path / "c.jsonl"
    rows = [{"scene_id": "s", "episode_index": 0, "frame_id": f, "expert_class": c, "class_match": True}
            for f, c in ((0, "pixel"), (4, "stop"), (6, "turn_left"), (8, "pixel"))]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert [r["frame_id"] for r in load(path)] == [6, 8]


def test_advantage_and_verdicts():
    control = advantage(_rows(40, 10, 9), _rows(40, 10, 6), resamples=500)
    assert control["px10"]["delta"] == 0.3 and control["px10"]["ci95"][0] > 0
    assert control["agree"]["delta"] == 0.0
    seen = advantage(_rows(40, 10, 8), _rows(40, 10, 6), resamples=500)
    unseen = advantage(_rows(40, 10, 6), _rows(40, 10, 6), resamples=500)
    assert verdict(control, seen)["reading"] == "seen"
    assert verdict(control, unseen)["reading"] == "unseen-compatible"
    flat = advantage(_rows(40, 10, 6), _rows(40, 10, 6), resamples=500)
    assert verdict(flat, seen)["reading"].startswith("undetermined")

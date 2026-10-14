"""EXP-25 B1/B2 probes: the scene-stratified sample (scripts/exp25/make_probe_lists.py) and the pure parts of
the metrics (scripts/exp25/probe_metrics.py): instruction-start type, first-pixel class, E7, log parsing."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from scripts.exp25 import make_probe_lists as mpl
from scripts.exp25 import probe_metrics as pm


def _split(path: Path, sizes: dict) -> Path:
    eps, eid = [], 0
    for scene, n in sizes.items():
        for _ in range(n):
            eid += 1
            eps.append({"episode_id": eid, "scene_id": f"mp3d/{scene}/{scene}.glb", "info": {"geodesic_distance": 1.0},
                        "instruction": {"instruction_text": f"walk {eid}"}})
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"episodes": eps, "instruction_vocab": {"word_list": ["walk"]}}, fh)
    return path


def test_allocate_is_proportional_by_largest_remainder():
    alloc = mpl.allocate({"a": 50, "b": 30, "c": 15, "d": 5}, 10)
    assert alloc == {"a": 5, "b": 3, "c": 2, "d": 0}  # c's remainder 0.5 beats d's 0.5 on the name tie-break
    assert sum(mpl.allocate({"a": 1, "b": 1, "c": 1}, 3).values()) == 3
    with pytest.raises(ValueError):
        mpl.allocate({"a": 2}, 3)


def test_draw_is_seeded_and_stratified():
    eps = [{"scene_id": f"mp3d/s{i % 4}/s{i % 4}.glb", "episode_id": i} for i in range(400)]
    first, per_scene = mpl.draw(eps, 40, seed=25)
    again, _ = mpl.draw(list(reversed(eps)), 40, seed=25)
    other, _ = mpl.draw(eps, 40, seed=26)
    assert first == again and first != other
    assert len(first) == len(set(first)) == 40 and first == sorted(first)
    assert all(v == {"episodes": 100, "chosen": 10} for v in per_scene.values())


def test_plan_directory_matches_the_launcher_contract(tmp_path):
    split = _split(tmp_path / "val_seen.json.gz", {"A": 12, "B": 6, "C": 2})
    tools = tmp_path / "locked_tools"
    tools.mkdir()
    out = tmp_path / "plan"
    assert mpl.main(["--split-file", str(split), "--out", str(out), "--n", "10", "--seed", "25", "--shards", "3",
                     "--tools", str(tools)]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    chosen = {(e["scene_id"], e["episode_id"]) for e in manifest["episodes"]}
    assert len(chosen) == 10 and manifest["per_scene"]["A"]["chosen"] == 6
    seen = set()
    for shard in range(3):
        listing = json.loads((out / "cohorts" / f"shard_0{shard}.json").read_text())
        assert (out / "lists" / f"shard_0{shard}.json").read_text() == (out / "cohorts" / f"shard_0{shard}.json").read_text()
        keys = [(e["scene_id"], e["episode_id"]) for e in listing["episodes"]]
        with gzip.open(out / "cohorts" / f"dataset_shard_0{shard}.json.gz", "rt") as fh:
            data = json.load(fh)
        assert data["instruction_vocab"] == {"word_list": ["walk"]}
        assert [(mpl.scene_name(e["scene_id"]), e["episode_id"]) for e in data["episodes"]] == keys
        seen |= set(keys)
    assert seen == chosen
    assert [s["episodes"] for s in manifest["shards"]] == [4, 3, 3]
    assert (out / "tools").resolve() == tools.resolve()


@pytest.mark.parametrize("text,kind", [
    ("Turn left and walk past the bed.", "turn"),
    ("  turn around . walk down the hall .", "turn"),
    ("Make a right and go up the stairs.", "turn"),
    ("Take a sharp left into the kitchen.", "turn"),
    ("Go left at the end of the hallway.", "turn"),
    ("Face the window, then walk forward.", "turn"),
    ("Walk forward and turn left.", "other"),
    ("Go straight past the couch.", "other"),
    ("Exit the bedroom and turn right.", "other"),
    ("Facing the stairs, walk up.", "other"),
    ("Turnstile ahead, walk through.", "other"),
])
def test_instruction_type(text, kind):
    assert pm.instruction_type(text) == kind


def test_first_pixel_class_and_category():
    def call(kind, g=5.0, gT=None, tdist=2.0):
        return {"kind": kind, "g": g, "gT": gT, "tdist": tdist}

    ep = {"calls": [call("native_actions"), call("trajectory", gT=None),
                    call("trajectory", g=5.0, gT=5.75, tdist=2.5), call("trajectory", g=5.0, gT=3.0), call("stop")],
          "success": 0.0, "os": 0.0, "ne": 7.0}
    first = pm.first_pixel(ep)
    assert first is ep["calls"][2] and pm.pixel_class(first) == "bad"  # ratio exactly -0.3
    assert pm.pixel_class(call("trajectory", g=5.0, gT=4.0)) == "good"
    assert pm.pixel_class(call("trajectory", g=5.0, gT=4.9, tdist=0.1)) == "mid"  # 0.1 / max(0.1, 0.25) = 0.4
    assert pm.category(ep) == "E7"
    assert pm.category(dict(ep, ne=6.99)) == "E<7"
    assert pm.category(dict(ep, os=1.0)) == "O"
    assert pm.category(dict(ep, calls=ep["calls"][:-1])) == "L0"
    assert pm.category(dict(ep, success=1.0, os=1.0)) == "S"


def test_summary_counts_b_over_episodes_with_a_pixel_call():
    def ep(cls_gT, success=0.0, os_=0.0, ne=8.0, text="walk"):
        calls = [] if cls_gT is None else [{"kind": "trajectory", "g": 5.0, "gT": cls_gT, "tdist": 2.0}]
        return {"calls": calls + [{"kind": "stop", "g": 1.0}], "success": success, "os": os_, "ne": ne,
                "spl": success, "instruction": text}

    rows = [ep(6.0), ep(3.0, 1.0, 1.0, 1.0), ep(None), ep(6.0, text="Turn left")]
    s = pm.summarize(rows)
    assert s["B"]["k"] == 2 and s["B"]["n"] == 3 and s["episodes_without_first_pixel"] == 1
    assert s["E7"]["k"] == 3 and s["sr"]["k"] == 1
    assert s["by_instruction_type"]["turn"]["B"] == pm.share(1, 1)
    assert pm.wilson(0, 10)[0] == 0.0


def test_parse_client_log_keys_by_scene_and_episode(tmp_path):
    log = tmp_path / "client_shard_00.log"
    log.write_text(
        "[1/2] Episode s8pcmisQ38h_0007: Walk ...\n"
        "  step_id: 0, RPC kind=native_actions, VLM output: ←←\n"
        "  [debug] actions=[2, 2, 0, 0]\n"
        "  step_id: 2, RPC kind=trajectory, VLM output: 301 225\n"
        "  [debug] trajectory traj_goal=(2.44,0.03), direct=2.44, path_len=2.52, actions=[1, 1, 1, 3]\n"
        "  step_id: 6, RPC kind=stop, VLM output: STOP\n"
        "  => success: 1.0, spl: 0.9, os: 1.0, ne: 0.5, vlm_calls: 3, trajectory_calls: 1\n"
        "[2/2] Episode s8pcmisQ38h_0008: Walk ...\n"
        "  step_id: 0, RPC kind=trajectory, VLM output: 300 200\n"
    )
    blocks = pm.parse_client_log(log)
    assert list(blocks) == [("s8pcmisQ38h", 7)]  # the unfinished episode is not a block
    b = blocks[("s8pcmisQ38h", 7)]
    assert [c["kind"] for c in b["calls"]] == ["native_actions", "trajectory", "stop"]
    assert b["calls"][1]["actions"] == [1, 1, 1, 3] and b["final"]["ne"] == 0.5

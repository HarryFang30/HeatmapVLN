"""EXP-33 analysis on synthetic runs: the run checks refuse what the criteria call unusable, and the
pre-registered readings come out of known effects."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from scripts.exp33 import analyze

N_PER_SHARD = 40  # 320 episodes: enough for the planted effects to clear the bootstrap


def _keys():
    return [(f"scene{shard}", shard * 1000 + i) for shard in range(8) for i in range(N_PER_SHARD)]


def _write_inputs(root: Path):
    subset, cohorts = root / "subset", root / "cohorts"
    subset.mkdir()
    cohorts.mkdir()
    for shard in range(8):
        eps = [{"scene_id": s, "episode_id": e} for s, e in _keys() if s == f"scene{shard}"]
        (subset / f"shard_0{shard}.json").write_text(json.dumps({"episodes": eps}))
        dataset = {"episodes": [{"scene_id": f"mp3d/{s}/{s}.glb", "episode_id": e,
                                 "info": {"geodesic_distance": 2.0 + (e % 40) / 2.0}} for s, e in _keys()
                                if s == f"scene{shard}"]}
        with gzip.open(cohorts / f"dataset_shard_0{shard}.json.gz", "wt") as handle:
            json.dump(dataset, handle)
    (subset / "manifest.json").write_text(json.dumps({"quartile_edges_m": [2.0, 7.0, 12.0, 17.0, 21.5]}))
    return subset, cohorts


def _success(view: str, arm: str, index: int) -> float:
    """V0 60%; V1 and V2 the same episodes; V3 loses a third of them; A1 equals A0."""
    base = index % 10 < 6
    if view == "v3":
        return float(base and index % 3 != 0)
    return float(base)


def _write_run(runs: Path, arm: str, view: str, *, mutate=None):
    out = analyze.run_dir(runs, arm, view)
    for shard in range(8):
        rows = []
        for index, (scene, episode) in enumerate(_keys()):
            if scene != f"scene{shard}":
                continue
            success = _success(view, arm, index)
            row = {"scene_id": scene, "episode_id": episode, "success": success, "spl": 0.8 * success,
                   "os": max(success, 0.5), "ne": 1.0 if success else 6.0, "steps": 120 if success else 500,
                   "history_pose_source": "amb3r_vo_da3", "rpc_protocol_seed": 42, "ppa_applied_calls": 10}
            if analyze.MOUNTS[view] is not None:
                row["camera_mount"] = analyze.MOUNTS[view]
            if mutate is not None:
                row = mutate(row)
            rows.append(json.dumps(row))
        path = out / "workers" / f"shard_0{shard}" / "progress.json"
        path.parent.mkdir(parents=True)
        path.write_text("\n".join(rows) + "\n")


def test_planted_effects_give_the_pre_registered_readings(tmp_path):
    subset, cohorts = _write_inputs(tmp_path)
    runs = tmp_path / "runs"
    for arm in analyze.ARMS:
        for view in analyze.VIEWS:
            _write_run(runs, arm, view)
    out = tmp_path / "out"
    analyze.main(["--runs", str(runs), "--subset", str(subset), "--cohorts", str(cohorts), "--out", str(out)])
    result = json.loads((out / "exp33_results.json").read_text())
    cost = result["viewpoint_cost"]
    assert cost["a0_v1"]["deployment_reading"] == "usable as is" and cost["a0_v1"]["direction"] == "not measurable"
    assert cost["a0_v3"]["success"]["diff"] == pytest.approx(-0.2, abs=0.01)
    assert cost["a0_v3"]["deployment_reading"] == "large loss" and cost["a0_v3"]["direction"] == "drops"
    assert all(entry["reading"] == "not measurable" for entry in result["head"].values())
    assert result["interaction"]["v3"]["success"]["diff"] == 0.0
    assert result["v1_vs_v2"]["reading"] == "indistinguishable"
    assert result["verdicts"] == {"H1": "supported", "H2": "supported", "H3": "supported"}
    assert result["runs"]["a0_v3"]["step_cap_hits"] > result["runs"]["a0_v0"]["step_cap_hits"]
    assert sum(q["n"] for q in result["runs"]["a0_v0"]["sr_by_geodesic_quartile"]) == 8 * N_PER_SHARD


@pytest.mark.parametrize(
    "mutate, phrase",
    [
        (lambda row: {k: v for k, v in row.items() if k != "camera_mount"}, "camera_mount"),
        (lambda row: dict(row, rpc_protocol_seed=1337), "another protocol seed"),
        (lambda row: dict(row, history_pose_source="habitat_gt"), "AMB3R"),
        (lambda row: dict(row, ppa_applied_calls=0), "never applied"),
    ],
)
def test_a_run_the_criteria_call_unusable_is_refused(tmp_path, mutate, phrase):
    subset, cohorts = _write_inputs(tmp_path)
    runs = tmp_path / "runs"
    _write_run(runs, "a0", "v1", mutate=mutate)
    with pytest.raises(SystemExit, match=phrase):
        analyze.load_run(analyze.run_dir(runs, "a0", "v1"), "v1", analyze.subset_keys(subset))


def test_a_short_run_is_refused(tmp_path):
    subset, _ = _write_inputs(tmp_path)
    runs = tmp_path / "runs"
    _write_run(runs, "a0", "v0")
    progress = analyze.run_dir(runs, "a0", "v0") / "workers" / "shard_03" / "progress.json"
    progress.write_text("\n".join(progress.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(SystemExit, match="1 subset episodes missing"):
        analyze.load_run(analyze.run_dir(runs, "a0", "v0"), "v0", analyze.subset_keys(subset))

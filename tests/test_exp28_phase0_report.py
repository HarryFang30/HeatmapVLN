"""EXP-28 Phase-0: probe_metrics' --scenes navmesh root (scripts/exp25/probe_metrics.py) and the GO/NO-GO report
(scripts/exp28/phase0_report.py): the ledger's tiers, exact comparison, the scene-cluster bootstrap, the input checks,
and probe_metrics -> report end to end on HM3D-style scene ids with a flat-floor stand-in for habitat_sim."""
from __future__ import annotations

import glob
import gzip
import importlib.util
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scripts.exp25 import probe_metrics as pm
from scripts.exp28 import phase0_report as p0

FAKE_HABITAT_SIM = '''
"""Flat-floor stand-in for habitat_sim's path finding: geodesic = straight-line distance on the xz plane."""
import math

import numpy as np


class PathFinder:
    def __init__(self):
        self.is_loaded = False

    def load_nav_mesh(self, path):
        with open(path) as fh:
            self.is_loaded = fh.read().startswith("FAKE-NAVMESH")
        return self.is_loaded

    def find_path(self, path):
        a = np.asarray(path.requested_start, dtype=np.float64)
        ends = path.requested_ends if hasattr(path, "requested_ends") else [path.requested_end]
        ends = np.asarray(ends, dtype=np.float64).reshape(-1, 3)
        d = [math.hypot(e[0] - a[0], e[2] - a[2]) for e in ends]
        path.geodesic_distance = min(d)
        path.points = [a.astype(np.float32), ends[int(np.argmin(d))].astype(np.float32)]
        return True

    def snap_point(self, p):
        q = np.array(p, dtype=np.float32)
        q[1] = 0.0
        return q

    def try_step(self, a, b):
        return np.array(b, dtype=np.float32)


class ShortestPath:
    def __init__(self):
        self.requested_start = None
        self.requested_end = None
        self.points = []
        self.geodesic_distance = float("inf")


class MultiGoalShortestPath:
    def __init__(self):
        self.requested_start = None
        self.requested_ends = None
        self.points = []
        self.geodesic_distance = float("inf")
'''


@pytest.fixture()
def fake_habitat(tmp_path, monkeypatch):
    """The stand-in, both imported here (forked Pool workers inherit it) and first on sys.path (spawned ones)."""
    root = tmp_path / "fake_habitat"
    root.mkdir()
    (root / "habitat_sim.py").write_text(FAKE_HABITAT_SIM)
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("habitat_sim", root / "habitat_sim.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "habitat_sim", module)
    monkeypatch.setattr(pm, "_PF", {})
    monkeypatch.setattr(pm, "SCENES", pm.DEFAULT_SCENES)
    return module


@pytest.fixture()
def restore_scenes(monkeypatch):
    monkeypatch.setattr(pm, "_PF", {})
    monkeypatch.setattr(pm, "SCENES", pm.DEFAULT_SCENES)


# ---------------------------------------------------------------- probe_metrics --scenes

def test_probe_metrics_default_navmesh_root_is_still_mp3d(restore_scenes):
    assert pm.DEFAULT_SCENES == pm.SCENES == "/dataset/mp3d"
    assert pm.navmesh_path("17DRP5sb8fy") == "/dataset/mp3d/17DRP5sb8fy/17DRP5sb8fy.navmesh"
    pm.set_scenes("/workspace/exp28/scene_datasets/hm3d")
    assert pm.navmesh_path("5jLhtVmWd5F") == "/workspace/exp28/scene_datasets/hm3d/5jLhtVmWd5F/5jLhtVmWd5F.navmesh"


def test_pathfinder_refuses_a_missing_or_unloaded_navmesh(tmp_path, fake_habitat):
    pm.set_scenes(tmp_path)
    with pytest.raises(FileNotFoundError):
        pm.pathfinder("5jLhtVmWd5F")
    (tmp_path / "5jLhtVmWd5F").mkdir()
    (tmp_path / "5jLhtVmWd5F" / "5jLhtVmWd5F.navmesh").write_text("v2 navmesh")
    with pytest.raises(RuntimeError, match="did not load"):
        pm.pathfinder("5jLhtVmWd5F")
    (tmp_path / "5jLhtVmWd5F" / "5jLhtVmWd5F.navmesh").write_text("FAKE-NAVMESH")
    pf = pm.pathfinder("5jLhtVmWd5F")
    assert pf.is_loaded and pm.pathfinder("5jLhtVmWd5F") is pf


SCENE_BAD, SCENE_GOOD = "5jLhtVmWd5F", "wuKkTq5GJbi"
FAKE_NAVMESH_SHA = p0.hashlib.sha256(b"FAKE-NAVMESH").hexdigest()


def _hm3d_run(root: Path, *, ne_bad: float = 8.0, navmesh_sha: str = FAKE_NAVMESH_SHA) -> list[str]:
    """Two HM3D-style episodes: the first pixel goal (straight ahead, about 0.96 m) leads away from the goal 7 m
    behind (bad; four steps on, a STOP 8 m out: E7) or towards the goal 1.5 m ahead (good; four steps on, 0.5 m:
    success).  The client's ne is the replay's end distance unless ``ne_bad`` says otherwise.  Returns
    probe_metrics' arguments."""
    eps, log, prog = [], [], []
    for i, (scene, goal_z, success, ne) in enumerate([(SCENE_BAD, 7.0, 0.0, ne_bad), (SCENE_GOOD, -1.5, 1.0, 0.5)]):
        eid = i + 1
        eps.append({"episode_id": eid, "scene_id": f"hm3d/{scene}/{scene}.glb", "start_position": [0.0, 0.0, 0.0],
                    "start_rotation": [0.0, 0.0, 0.0, 1.0], "goals": [{"position": [0.0, 0.0, goal_z]}],
                    "info": {"geodesic_distance": abs(goal_z), "navmesh_sha256": navmesh_sha},
                    "instruction": {"instruction_text": "walk to the door"}})
        log += [f"[{eid}/2] Episode {scene}_{eid}: walk to the door",
                "  step_id: 0, RPC kind=trajectory, VLM output: 320 400",
                "  [debug] trajectory traj_goal=(0.96,0.00), direct=0.96, path_len=0.96, actions=[1, 1, 1, 1]",
                "  step_id: 4, RPC kind=stop, VLM output: STOP",
                f"  => success: {success}, spl: {success}, os: {success}, ne: {ne}, vlm_calls: 2, trajectory_calls: 1"]
        prog.append({"scene_id": scene, "episode_id": eid, "success": success, "spl": success, "os": success,
                     "ne": ne, "steps": 5, "rpc_protocol_seed": 42})
        (root / "scenes" / scene).mkdir(parents=True)
        (root / "scenes" / scene / f"{scene}.navmesh").write_text("FAKE-NAVMESH")
    with gzip.open(root / "dataset_shard_000.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({"episodes": eps}, fh)
    (root / "client_shard_000.log").write_text("\n".join(log) + "\n")
    (root / "progress.json").write_text("".join(json.dumps(r) + "\n" for r in prog))
    return ["--datasets", str(root / "dataset_shard_000.json.gz"), "--progress", str(root / "progress.json"),
            "--logs", str(root / "client_shard_000.log"), "--workers", "1"]


def test_probe_metrics_reads_hm3d_navmeshes_from_scenes_then_phase0_report(tmp_path, fake_habitat, capsys):
    run = _hm3d_run(tmp_path)
    with pytest.raises(FileNotFoundError):  # default root: /dataset/mp3d has no HM3D scene
        pm.main(["--name", "nope", "--out", str(tmp_path / "out"), *run])
    pm.set_scenes(pm.DEFAULT_SCENES)
    assert pm.main(["--name", "hm3d", "--out", str(tmp_path / "out"), "--scenes", str(tmp_path / "scenes"), *run]) == 0
    rows = json.loads((tmp_path / "out" / "hm3d.decomp.json").read_text())
    by_scene = {r["scene"]: r for r in rows}
    assert set(by_scene) == {SCENE_BAD, SCENE_GOOD}
    for scene, cls in ((SCENE_BAD, "bad"), (SCENE_GOOD, "good")):
        first = pm.first_pixel(by_scene[scene])
        assert first["tdist"] == pytest.approx(0.96, abs=0.01) and pm.pixel_class(first) == cls
    assert pm.category(by_scene[SCENE_BAD]) == "E7" and pm.category(by_scene[SCENE_GOOD]) == "S"
    capsys.readouterr()

    code = p0.main(["--decomp", f"x={tmp_path / 'out' / 'hm3d.decomp.json'}", "--expect", "x=2",
                    "--progress", f"x={tmp_path / 'progress.json'}", "--bootstrap-draws", "200",
                    "--out", str(tmp_path / "report")])
    report = json.loads((tmp_path / "report" / "phase0.phase0.json").read_text())
    assert report["decision"]["counts"] == {"sr": [1, 2], "B": [1, 2], "E7": [1, 2]}
    assert report["decision"]["verdict"] == "GO" and code == 0  # 50% / 50% / 50%: B, E7 high and SR <= 85%
    assert report["protocol"]["x"]["rows_checked"] == 2
    replay = report["pooled"]["replay"]
    assert replay["off"]["k"] == 0 and replay["end_replayed_vs_ne_m"]["max"] == 0.0
    assert replay["start_g_vs_info_geodesic_m"]["max"] == 0.0 and report["replay_gate"]["gate"] == "passed"
    assert capsys.readouterr().out.startswith("# EXP-28 Phase-0: GO")


def test_probe_metrics_refuses_a_navmesh_other_than_the_episodes(tmp_path, fake_habitat):
    run = _hm3d_run(tmp_path, navmesh_sha="0" * 64)  # the episode file was built on another navmesh
    with pytest.raises(RuntimeError, match="was built on"):
        pm.main(["--name", "x", "--out", str(tmp_path / "out"), "--scenes", str(tmp_path / "scenes"), *run])
    pm.set_scenes(tmp_path / "nowhere")
    pm.check_navmesh("17DRP5sb8fy", {"episode_id": 1, "info": {"geodesic_distance": 3.0}})  # MP3D: nothing read


def test_replay_gate_refuses_a_replay_that_is_not_the_clients_walk(tmp_path, fake_habitat, capsys):
    run = _hm3d_run(tmp_path, ne_bad=9.0)  # the client ended 9 m out, the replay 8 m: not the same walk
    assert pm.main(["--name", "hm3d", "--out", str(tmp_path / "out"), "--scenes", str(tmp_path / "scenes"), *run]) == 0
    decomp = f"x={tmp_path / 'out' / 'hm3d.decomp.json'}"
    capsys.readouterr()
    assert p0.main(["--decomp", decomp, "--bootstrap-draws", "50", "--out", str(tmp_path / "rep")]) == 2
    assert "replay gate: 1 of 2" in capsys.readouterr().err and not (tmp_path / "rep").exists()
    code = p0.main(["--decomp", decomp, "--no-replay-gate", "--bootstrap-draws", "50", "--out", str(tmp_path / "rep")])
    assert code == 0 and capsys.readouterr().out.splitlines()[4].startswith("Replay gate skipped")
    report = json.loads((tmp_path / "rep" / "phase0.phase0.json").read_text())
    assert report["pooled"]["replay"]["off_examples"] == [[SCENE_BAD, 1]]


def test_replay_check_counts_offsets_and_nonfinite_geodesics():
    ok = _row("A", 1, success=True)
    moved = dict(_row("A", 2), ne=2.3)                                  # replayed end 2.0 m, client 2.3 m
    island = _row("A", 3)
    island["calls"][0] = dict(island["calls"][0], g=math.inf)          # a replayed position off the goal's island
    near = dict(_row("A", 4), ne=2.2)                                   # 0.2 m: within the tolerance
    rc = p0.replay_check([ok, moved, island, near, dict(_row("A", 5), calls=[])])
    assert rc["off"] == pm.share(2, 4) and rc["off_examples"] == [["A", 2], ["A", 3]]
    assert rc["nonfinite_g_episodes"] == 1 and rc["episodes_without_calls"] == 1
    assert rc["end_replayed_vs_ne_m"]["max"] == pytest.approx(0.3)


def test_real_navmesh_loads_through_the_scenes_root(tmp_path, restore_scenes):
    pytest.importorskip("habitat_sim")
    found = (sorted(glob.glob("/workspace/exp28/scene_datasets/hm3d/*/*.navmesh"))[:1]
             or sorted(glob.glob("/dataset/mp3d/*/*.navmesh"))[:1])
    if not found:
        pytest.skip("no navmesh on this machine")
    path = Path(found[0])
    pm.set_scenes(path.parent.parent)
    pf = pm.pathfinder(path.parent.name)
    p = np.asarray(pf.get_random_navigable_point(), dtype=np.float32)
    assert pf.is_loaded and math.isfinite(pm.geo(pf, p, [p]))
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "junk.navmesh").write_bytes(b"\0" * 4096)  # bad magic: load_nav_mesh fails, is_loaded False
    pm.set_scenes(tmp_path)
    with pytest.raises(RuntimeError, match="did not load"):
        pm.pathfinder("junk")


# ---------------------------------------------------------------- the ledger's tiers

F = Fraction


@pytest.mark.parametrize("sr,b,e7,tier,verdict", [
    (60, 9, 8, "GO", "GO"),
    (60, 8, 7, "GO", "GO"),                        # GO's lines are inclusive
    (85, 9, 8, "NO-GO", "NO-GO"),                  # SR = 85% meets both; NO-GO is checked first
    (F(8499, 100), 9, 8, "GO", "GO"),
    (60, F(11, 2), 9, "NO-GO", "NO-GO"),           # B <= 5.5%
    (60, 9, F(4999, 1000), "NO-GO", "NO-GO"),      # E7 < 5.0%
    (60, F(5501, 1000), 7, "middle", "GO"),        # B in (5.5, 8): E7 >= 7 decides
    (60, 7, 6, "middle", "NO-GO"),
    (60, 9, 6, "middle", "NO-GO"),                 # B >= 8 but E7 in [5, 7)
    (60, 9, 5, "middle", "NO-GO"),                 # E7 = 5.0% is not E7 < 5.0%
])
def test_decide_follows_the_ledger(sr, b, e7, tier, verdict):
    d = p0.decide(F(sr), F(b), F(e7))
    assert (d["tier"], d["verdict"]) == (tier, verdict)


def test_decide_is_exact_not_on_rounded_percentages():
    b = p0.exact_pct(pm.share(1999, 25000))  # 7.996%: share() prints it as 8.0
    assert pm.share(1999, 25000)["pct"] == 8.0
    assert p0.decide(F(60), b, F(6))["verdict"] == "NO-GO"  # rounded it would be GO's B, and E7 6 < 7 still fails
    assert p0.decide(F(60), b, F(8))["tier"] == "middle"
    assert p0.exact_pct(pm.share(64, 800)) == 8 and p0.decide(F(60), F(8), F(7))["tier"] == "GO"


def test_decide_without_b_is_undecidable():
    assert p0.decide(F(60), None, F(8))["verdict"] is None
    assert p0.decide(None, None, None)["tier"] == "undecidable"


@pytest.mark.parametrize("raw,key", [
    ("hm3d/5jLhtVmWd5F/5jLhtVmWd5F.glb", "5jLhtVmWd5F"),
    ("train/00481-5jLhtVmWd5F/5jLhtVmWd5F.basis.glb", "5jLhtVmWd5F"),
    ("00481-5jLhtVmWd5F", "5jLhtVmWd5F"),
    ("5jLhtVmWd5F.basis.glb", "5jLhtVmWd5F"),
    ("5jLhtVmWd5F", "5jLhtVmWd5F"),
    ("mp3d/17DRP5sb8fy/17DRP5sb8fy.glb", "17DRP5sb8fy"),
])
def test_scene_key(raw, key):
    assert p0.scene_key(raw) == key


# ---------------------------------------------------------------- the scene-cluster bootstrap

def _row(scene, eid, *, success=False, pixel="good", e7=False, text="walk"):
    """A decomposed episode: first pixel goal good/bad/mid/None; E7 = failed, never in 3 m, STOP at 8 m.  The
    replay's end distance (the last call's next_g) is the client's ne, as on MP3D."""
    ne = 0.5 if success else (8.0 if e7 else 2.0)
    calls = [] if pixel is None else [{"kind": "trajectory", "g": 5.0, "tdist": 2.0,
                                       "gT": {"good": 3.0, "bad": 6.0, "mid": 4.5}[pixel]}]
    calls.append({"kind": "stop" if (e7 or success) else "trajectory", "g": 1.0, "next_g": ne})
    return {"scene": scene, "episode_id": eid, "success": 1.0 if success else 0.0, "os": 1.0 if success else 0.0,
            "ne": ne, "spl": 1.0 if success else 0.0, "instruction": text, "calls": calls}


def test_episode_counts_follow_probe_metrics():
    rows = [_row("A", 1, success=True), _row("A", 2, pixel="bad", e7=True), _row("A", 3, pixel=None),
            _row("A", 4, pixel="mid")]
    totals = np.sum([p0.episode_counts(r) for r in rows], axis=0)
    s = pm.summarize(rows)
    got = {name: v.item() for name, v in p0.rates(totals).items()}
    assert got == {"sr": s["sr"]["pct"], "os": s["os"]["pct"], "B": pytest.approx(100 / 3), "E7": s["E7"]["pct"]}
    assert s["B"]["k"] == 1 and s["B"]["n"] == 3


def test_bootstrap_resamples_scenes_not_episodes():
    clustered = [p0.scene_matrix([_row("A", i, success=True) for i in range(10)]
                                 + [_row("B", i) for i in range(10, 20)])[1]]
    mixed = [p0.scene_matrix([_row(s, 10 * k + i, success=i % 2 == 0) for k, s in enumerate("AB")
                              for i in range(10)])[1]]
    wide = p0.cluster_bootstrap(clustered, 2000, 0)["sr"]
    flat = p0.cluster_bootstrap(mixed, 2000, 0)["sr"]
    assert wide["ci95"] == [0.0, 100.0] and wide["se"] > 30
    assert flat["ci95"] == [50.0, 50.0] and flat["se"] == 0.0


def test_bootstrap_keeps_each_half_its_scenes():
    half_a = p0.scene_matrix([_row("A", i, success=True) for i in range(10)])[1]
    half_b = p0.scene_matrix([_row("B", i) for i in range(10, 20)])[1]
    stratified = p0.cluster_bootstrap([half_a, half_b], 2000, 0)["sr"]
    assert stratified["ci95"] == [50.0, 50.0]  # one scene per half: every draw is the sample itself
    assert p0.cluster_bootstrap([half_a, half_b], 500, 3) == p0.cluster_bootstrap([half_a, half_b], 500, 3)


# ---------------------------------------------------------------- main

def _half(prefix: str, n_scenes: int, per_scene: int, *, bad: int, e7: int, success: int, start: int = 0):
    """Episodes over ``n_scenes`` scenes; the first ``bad`` have a bad first pixel, the next ``e7`` end E7,
    the last ``success`` succeed."""
    rows, n = [], n_scenes * per_scene
    for i in range(n):
        scene = f"{prefix}{i // per_scene:09d}"
        rows.append(_row(scene, start + i, pixel="bad" if i < bad else "good", e7=bad <= i < bad + e7,
                         success=i >= n - success))
    return rows


def _write(path: Path, rows) -> str:
    path.write_text(json.dumps(rows))
    return str(path)


def _progress(path: Path, rows, **extra) -> str:
    path.write_text("".join(json.dumps({"scene_id": r["scene"], "episode_id": r["episode_id"],
                                        "success": r["success"], "ne": r["ne"], "rpc_protocol_seed": 42, **extra})
                            + "\n" for r in rows))
    return str(path)


@pytest.fixture()
def halves(tmp_path):
    a = _half("HB", 4, 5, bad=2, e7=2, success=10)                    # 20 episodes, 4 scenes
    b = _half("TR", 4, 5, bad=2, e7=2, success=10, start=100)         # 20 episodes, 4 other scenes
    return {"a": a, "b": b, "a_path": _write(tmp_path / "a.decomp.json", a),
            "b_path": _write(tmp_path / "b.decomp.json", b)}


def _run(tmp_path, *argv):
    code = p0.main([*argv, "--bootstrap-draws", "300", "--out", str(tmp_path / "rep")])
    path = tmp_path / "rep" / "phase0.phase0.json"
    return code, (json.loads(path.read_text()) if code != 2 and path.exists() else None)


def test_main_go_with_halves_overlap_and_protocol(tmp_path, halves, capsys):
    a, b = halves["a"], halves["b"]
    overlap = tmp_path / "overlap.json"
    overlap.write_text(json.dumps([[f"00481-{a[0]['scene']}", a[0]["episode_id"]],
                                   {"scene_id": f"hm3d/{a[1]['scene']}/{a[1]['scene']}.glb",
                                    "episode_id": a[1]["episode_id"]}]))
    code, rep = _run(tmp_path, "--decomp", f"910b={halves['a_path']}", f"4090={halves['b_path']}",
                     "--expect", "910b=20", "4090=20", "--overlap-keys", str(overlap),
                     "--progress", f"910b={_progress(tmp_path / 'pa.json', a)}",
                     f"4090={_progress(tmp_path / 'pb.json', b)}")
    assert code == 0
    d = rep["decision"]
    assert (d["tier"], d["verdict"]) == ("GO", "GO")
    assert d["counts"] == {"sr": [20, 40], "B": [4, 40], "E7": [4, 40]} and d["on"] == {"sr": 50.0, "B": 10.0,
                                                                                        "E7": 10.0}
    assert rep["pooled"]["scenes"] == 8 and rep["halves"]["910b"]["scenes"] == 4
    assert rep["halves"]["4090"]["point"]["os"]["k"] == 10
    lo, hi = rep["pooled"]["cluster_bootstrap"]["B"]["ci95"]
    assert lo <= 10.0 <= hi
    assert rep["overlap"]["by_set"]["910b"]["share"]["k"] == 2
    assert rep["overlap"]["by_set"]["pooled"]["share"] == pm.share(2, 40)
    assert rep["overlap"]["by_set"]["4090"]["share"]["k"] == 0
    assert set(rep["protocol"]) == {"910b", "4090"}
    assert capsys.readouterr().out.startswith("# EXP-28 Phase-0: GO (tier GO)")


def test_main_no_go_when_sr_reaches_85(tmp_path):
    rows = _half("HB", 4, 5, bad=2, e7=2, success=16) + _half("TR", 4, 5, bad=2, e7=0, success=18, start=100)
    code, rep = _run(tmp_path, "--decomp", f"x={_write(tmp_path / 'x.json', rows)}")
    assert rep["decision"]["counts"]["sr"] == [34, 40]  # exactly 85%
    assert code == 1 and rep["decision"]["reasons"] == ["SR >= 85%"]


def _cohort(root: Path, parts) -> str:
    """A ``heatmapvln-r2r-train-cohorts-v1`` cohort; ``parts`` is [(part, [rows of one shard, ...])], in shard order."""
    root.mkdir()
    records, index = [], 0
    for part, shards in parts:
        for rows in shards:
            path = root / f"shard_{index:03d}.json"
            path.write_text(json.dumps({"part": part, "episodes": [
                {"scene_id": r["scene"], "episode_id": r["episode_id"], "trajectory_id": r["episode_id"]}
                for r in rows]}))
            records.append({"index": index, "path": path.name, "sha256": p0.sha256(str(path)), "part": part,
                            "episodes": len(rows)})
            index += 1
    (root / "manifest.json").write_text(json.dumps({"schema": "heatmapvln-r2r-train-cohorts-v1", "split": "train",
                                                    "num_shards": index, "shards": records}))
    return str(root)


@pytest.mark.parametrize("spec,shards", [("0-19", list(range(20))), ("20-22,5", [20, 21, 22, 5]), ("7", [7]),
                                         (" 3 , 1-2", [3, 1, 2])])
def test_parse_shards_is_ppa_collect_shards(spec, shards):
    assert p0.parse_shards(spec) == shards


@pytest.mark.parametrize("spec", ["3-1", "a", "1,1", "0-2,2", "", "1;2"])
def test_parse_shards_refuses(spec):
    with pytest.raises(p0.InputError):
        p0.parse_shards(spec)


@pytest.fixture()
def cohort_run(tmp_path):
    """Held-out shards 0-1 (one scene each, 5 episodes), train shards 2-5 (one scene each, 4 episodes); one decomp
    file holding every held-out episode and the train episodes of shards 2, 3, 4 and two of shard 5's."""
    held = _half("HB", 2, 5, bad=1, e7=1, success=5)
    train = _half("TR", 4, 4, bad=2, e7=2, success=8, start=100)
    cohort = _cohort(tmp_path / "cohort", [("heldout", [held[:5], held[5:]]),
                                           ("train_pass1", [train[i:i + 4] for i in range(0, 16, 4)])])
    run = held + train[:14]
    return {"held": held, "train": train, "cohort": cohort, "decomp": _write(tmp_path / "all.decomp.json", run)}


def test_main_shards_split_one_decomp_into_halves_and_limit_takes_the_launcher_order(tmp_path, cohort_run):
    path, held, train = cohort_run["decomp"], cohort_run["held"], cohort_run["train"]
    code, rep = _run(tmp_path, "--decomp", f"910b={path}", f"4090={path}", "--cohort", cohort_run["cohort"],
                     "--shards", "910b=0-1", "4090=3,2,4", "--limit", "4090=10", "--expect", "910b=10", "4090=10")
    assert code in (0, 1)
    assert rep["inputs"]["910b"]["kept"] == 10 and rep["inputs"]["910b"]["rows_left_out"] == 14
    assert rep["inputs"]["910b"]["cohort_parts"] == ["heldout"]
    assert rep["inputs"]["4090"]["cohort_parts"] == ["train_pass1"]
    # shard 3, then shard 2, then two of shard 4: train[4:8] + train[0:4] + train[8:10]
    assert rep["halves"]["4090"]["scenes"] == 3 and rep["halves"]["4090"]["episodes_per_scene"]["min"] == 2
    kept = {(r["scene"], r["episode_id"]) for r in train[4:8] + train[0:4] + train[8:10]}
    assert rep["halves"]["4090"]["point"]["sr"]["k"] == sum(r["success"] > 0 for r in train if
                                                            (r["scene"], r["episode_id"]) in kept)
    assert rep["cohort"]["dir"] == cohort_run["cohort"]
    md = (tmp_path / "rep" / "phase0.phase0.md").read_text()
    assert "Halves from the cohort: 910b shards 0-1 (heldout); 4090 shards 3,2,4 first 10 (train_pass1)." in md


def test_main_shards_never_fill_a_gap_with_a_later_episode(tmp_path, cohort_run, capsys):
    path = cohort_run["decomp"]
    # shard 5 has only two of its four episodes decomposed: the first 14 of 2-5 are there, the first 16 are not
    code, rep = _run(tmp_path, "--decomp", f"4090={path}", "--cohort", cohort_run["cohort"], "--shards", "4090=2-5",
                     "--limit", "4090=14")
    assert code in (0, 1) and rep["inputs"]["4090"]["kept"] == 14
    assert _run(tmp_path, "--decomp", f"4090={path}", "--cohort", cohort_run["cohort"], "--shards", "4090=2-5")[0] == 2
    assert "2 of the 16 episodes" in capsys.readouterr().err
    assert _run(tmp_path, "--decomp", f"4090={path}", "--cohort", cohort_run["cohort"], "--shards", "4090=2-5",
                "--limit", "4090=17")[0] == 2  # more than the shards hold


def test_main_shards_input_errors(tmp_path, cohort_run, capsys):
    path, cohort = cohort_run["decomp"], cohort_run["cohort"]
    assert _run(tmp_path, "--decomp", f"x={path}", "--limit", "x=5")[0] == 2                     # no --shards
    assert _run(tmp_path, "--decomp", f"x={path}", "--shards", "x=0")[0] == 2                    # no --cohort
    assert _run(tmp_path, "--decomp", f"x={path}", "--cohort", cohort, "--shards", "x=9")[0] == 2  # no such shard
    assert _run(tmp_path, "--decomp", f"x={path}", "--cohort", cohort, "--shards", "x=0", "x=1")[0] == 2
    err = capsys.readouterr().err
    assert "--limit needs --shards" in err and "--shards needs --cohort" in err and "shard 9" in err
    Path(cohort, "shard_001.json").write_text("{}")
    assert _run(tmp_path, "--decomp", f"x={path}", "--cohort", cohort, "--shards", "x=0")[0] == 2
    assert "sha256" in capsys.readouterr().err


def test_main_refuses_inputs_that_break_the_design(tmp_path, halves, capsys):
    a_path = halves["a_path"]
    assert _run(tmp_path, "--decomp", f"910b={a_path}", f"4090={a_path}")[0] == 2  # same episodes twice
    leak = [dict(r, episode_id=r["episode_id"] + 1000) for r in halves["a"][:5]]
    assert _run(tmp_path, "--decomp", f"910b={a_path}", f"4090={_write(tmp_path / 'leak.json', leak)}")[0] == 2
    assert _run(tmp_path, "--decomp", f"910b={a_path}", "--expect", "910b=400")[0] == 2
    assert _run(tmp_path, "--decomp", f"910b={a_path}", "--expect", "4090=20")[0] == 2
    budget = _progress(tmp_path / "pb.json", halves["a"], stop_at_step_budget=True, step_budget_stop=False)
    assert _run(tmp_path, "--decomp", f"910b={a_path}", "--progress", f"910b={budget}")[0] == 2
    seed = _progress(tmp_path / "ps.json", halves["a"], rpc_protocol_seed=1337)
    assert _run(tmp_path, "--decomp", f"910b={a_path}", "--progress", f"910b={seed}")[0] == 2
    short = _progress(tmp_path / "pshort.json", halves["a"][:-1])
    assert _run(tmp_path, "--decomp", f"910b={a_path}", "--progress", f"910b={short}")[0] == 2
    lora = _progress(tmp_path / "plora.json", halves["a"], rpc_model_version="heatmapvln-r2r+system2-lora:" + "a" * 64)
    assert _run(tmp_path, "--decomp", f"910b={a_path}", "--progress", f"910b={lora}")[0] == 2
    base = _progress(tmp_path / "pbase.json", halves["a"], rpc_model_version="heatmapvln-r2r")
    assert _run(tmp_path, "--decomp", f"910b={a_path}", "--progress", f"910b={base}")[0] == 0
    assert _run(tmp_path, "--decomp", f"910b={tmp_path / 'missing.json'}")[0] == 2  # never 1, which reads as NO-GO
    err = capsys.readouterr().err
    assert "both" in err and "more than one half" in err and "step-budget" in err and "seed" in err
    assert "System2 LoRA" in err


def test_main_never_exits_1_on_a_failure_after_the_inputs(tmp_path, halves, monkeypatch, capsys):
    monkeypatch.setattr(p0, "markdown", lambda report: 1 / 0)
    assert _run(tmp_path, "--decomp", f"910b={halves['a_path']}")[0] == 2
    assert "ZeroDivisionError" in capsys.readouterr().err


def test_markdown_names_unchecked_halves(tmp_path, halves, capsys):
    a = halves["a"]
    assert _run(tmp_path, "--decomp", f"910b={halves['a_path']}", f"4090={halves['b_path']}",
                "--progress", f"910b={_progress(tmp_path / 'pa.json', a)}")[0] == 0
    out = capsys.readouterr().out
    assert "Protocol NOT checked (no --progress): 4090." in out and "Replay gate passed: 0.0% (0/40)" in out

"""EXP-28 T1: scripts/exp28/prepare_hm3d_scenes.py.

The scene list, the tree, rerun rules, the client's AGENT_0 lookup and the manifest run on small
synthetic inputs with a fake per-scene worker; nothing here needs habitat.  The last tests import
habitat_sim / habitat-lab and skip without them; a real one-scene bake runs only when
``EXP28_TEST_HM3D_SCENE`` names an HM3D scene (``00xxx-<ID>`` under ``EXP28_TEST_HM3D_ROOT``).
"""

import gzip
import hashlib
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path

import pytest

from scripts.exp28 import prepare_hm3d_scenes as prep

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "scripts" / "evaluation" / "r2r_val_unseen.py"
SCENES = {"5jLhtVmWd5F": "00481-5jLhtVmWd5F", "dD37vuCa4FE": "00393-dD37vuCa4FE", "wuKkTq5GJbi": "00396-wuKkTq5GJbi"}
PARAMS = {
    "habitat_sim_version": "0.1.7",
    "client": {"agent_0": {"HEIGHT": 1.5, "RADIUS": 0.1}},
    "agent_height": 1.5,
    "agent_radius": 0.1,
    "navmesh_settings": {"cell_size": 0.05, "agent_height": 1.5, "agent_radius": 0.1},
}
PARAMS["navmesh_settings_sha256"] = prep.settings_digest(PARAMS["navmesh_settings"], "0.1.7")


def _navmesh_bytes(version=1, payload=b"\0" * 64):
    return struct.pack("<iii", prep.RECAST_SET_MAGIC, version, 1) + payload


def _hm3d(tmp_path, scenes=SCENES, navmesh_versions=(2, None, 1)):
    root = tmp_path / "hm3d" / "train"
    for (scene, scene_dir), version in zip(sorted(scenes.items()), navmesh_versions):
        (root / scene_dir).mkdir(parents=True)
        (root / scene_dir / f"{scene}.basis.glb").write_bytes(b"glTF" + scene.encode())
        if version is not None:
            (root / scene_dir / f"{scene}.basis.navmesh").write_bytes(_navmesh_bytes(version))
    (root / "not-a-scene").mkdir()
    return root


def _snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------------------
# scene list
# ----------------------------------------------------------------------------
def test_read_scene_entries_in_every_format(tmp_path):
    text = tmp_path / "s.txt"
    text.write_text("# held out\n00481-5jLhtVmWd5F\n\n dD37vuCa4FE  # bare id\n")
    assert prep.read_scene_entries(text) == ["00481-5jLhtVmWd5F", "dD37vuCa4FE"]

    listing = tmp_path / "s.json"
    listing.write_text(json.dumps(["00481-5jLhtVmWd5F", {"scene_dir": "00393-dD37vuCa4FE"}]))
    assert prep.read_scene_entries(listing) == ["00481-5jLhtVmWd5F", "00393-dD37vuCa4FE"]

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_scenes": ["a"], "heldout_scenes": ["b"], "seed": 1}))
    assert prep.read_scene_entries(split) == ["b", "a"]  # keys in sorted order

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"scenes": {"5jLhtVmWd5F": {}, "dD37vuCa4FE": {}}}))
    assert prep.read_scene_entries(manifest) == ["5jLhtVmWd5F", "dD37vuCa4FE"]

    episodes = tmp_path / "e.json.gz"
    with gzip.open(episodes, "wt", encoding="utf-8") as handle:
        json.dump({"episodes": [{"scene_id": "hm3d/00481-5jLhtVmWd5F/5jLhtVmWd5F.basis.glb"}]}, handle)
    assert prep.read_scene_entries(episodes) == ["hm3d/00481-5jLhtVmWd5F/5jLhtVmWd5F.basis.glb"]

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"seed": 1}))
    with pytest.raises(SystemExit, match="no 'episodes'"):
        prep.read_scene_entries(bad)


def test_resolve_scene_entries(tmp_path):
    index = prep.index_hm3d_root(_hm3d(tmp_path))
    assert index == {scene: [scene_dir] for scene, scene_dir in SCENES.items()}  # "not-a-scene" ignored
    entries = [
        "hm3d/00481-5jLhtVmWd5F/5jLhtVmWd5F.basis.glb",  # the ScaleVLN json's scene_id (no train/)
        "dD37vuCa4FE",
        "00396-wuKkTq5GJbi",
        "5jLhtVmWd5F",  # duplicate
        "00999-5jLhtVmWd5F",  # right ID, wrong directory
        "zzzzzzzzzzz",  # unknown ID
        "not a scene",
    ]
    resolved, problems, duplicates = prep.resolve_scene_entries(entries, index)
    assert resolved == sorted(SCENES.items())
    assert duplicates == 1
    assert [p["entry"] for p in problems] == ["00999-5jLhtVmWd5F", "zzzzzzzzzzz", "not a scene"]
    assert "it has ['00481-5jLhtVmWd5F']" in problems[0]["error"]


def _interndata_scene(root, name, with_meta=True):
    (root / name / "meta").mkdir(parents=True)
    if with_meta:
        (root / name / "meta" / "episodes.jsonl").write_text("{}\n")


def test_scene_entries_from_dir_takes_what_build_hm3d_episodes_takes(tmp_path):
    # build_hm3d_episodes.interndata_scene_dirs: a child directory with meta/episodes.jsonl is a scene.
    _interndata_scene(tmp_path, "00481-5jLhtVmWd5F")
    _interndata_scene(tmp_path, "00393-dD37vuCa4FE", with_meta=False)  # a partial download
    (tmp_path / "annotations").mkdir()  # 11 characters: a bare-ID pattern would take it
    (tmp_path / ".cache").mkdir()
    (tmp_path / "00396-wuKkTq5GJbi.tar").write_bytes(b"")
    assert prep.scene_entries_from_dir(tmp_path) == (["00481-5jLhtVmWd5F"], ["00393-dD37vuCa4FE", "annotations"])


def test_scenes_from_dir_counts_against_expect_count(tmp_path, capsys):
    hm3d = _hm3d(tmp_path)
    (tmp_path / "mp3d").mkdir()
    train, b3 = tmp_path / "interndata" / "scalevln_train", tmp_path / "interndata" / "scalevln"
    _interndata_scene(train, "00393-dD37vuCa4FE")
    _interndata_scene(train, "00396-wuKkTq5GJbi", with_meta=False)
    _interndata_scene(b3, "00481-5jLhtVmWd5F")
    args = _args(tmp_path, hm3d, "--scenes-from-dir", str(train), "--scenes-from-dir", str(b3), "--dry-run")
    assert prep.main([*args, "--expect-count", "2"]) == 0
    out = capsys.readouterr().out
    report = json.loads(out[out.index("{\n"):])
    assert report["requested"]["resolved"] == 2
    assert report["requested"]["sources"][0]["skipped_without_meta_episodes"] == ["00396-wuKkTq5GJbi"]
    assert prep.main([*args, "--expect-count", "3"]) == 2


def test_recast_set_header(tmp_path):
    for version in (1, 2):
        path = tmp_path / f"v{version}.navmesh"
        path.write_bytes(_navmesh_bytes(version))
        assert prep.recast_set_header(path) == {"magic_ok": True, "version": version, "tiles": 1}
        assert prep.header_label(prep.recast_set_header(path)) == str(version)
    assert (tmp_path / "v1.navmesh").read_bytes()[:4] == b"TESM"  # 'MSET' little-endian, as Recast writes it
    junk = tmp_path / "junk.navmesh"
    junk.write_bytes(b"x" * 12)
    assert prep.header_label(prep.recast_set_header(junk)) == "bad-header"
    short = tmp_path / "short.navmesh"
    short.write_bytes(b"TESM")
    assert prep.recast_set_header(short)["magic_ok"] is False
    assert prep.recast_set_header(tmp_path / "absent.navmesh") is None
    assert prep.header_label(None) == "missing"


# ----------------------------------------------------------------------------
# the client's AGENT_0
# ----------------------------------------------------------------------------
class _Node:
    """A YACS-like config node: attribute access creates children; defrost/freeze are no-ops."""

    def __init__(self):
        object.__setattr__(self, "_values", {})

    def __getattr__(self, name):
        values = object.__getattribute__(self, "_values")
        if name not in values:
            values[name] = _Node()
        return values[name]

    def __setattr__(self, name, value):
        self._values[name] = value

    def defrost(self):
        pass

    def freeze(self):
        pass


def _fake_get_config():
    cfg = _Node()
    cfg.SIMULATOR.AGENT_0.HEIGHT = 1.5
    cfg.SIMULATOR.AGENT_0.RADIUS = 0.1
    return cfg


def test_the_real_client_builder_sets_no_agent_size():
    static = prep.inspect_client_builder(CLIENT.read_text(encoding="utf-8"))
    assert static["base"] == "habitat.config.default.get_config()"
    assert set(static["agent_0_assignments"]) == {"SENSORS"}
    assert not static["sets_height"] and not static["sets_radius"]
    assert static["merges"] == []


def test_build_client_config_runs_the_real_builder():
    cfg = prep.build_client_config(CLIENT.read_text(encoding="utf-8"), {
        "habitat.config.default.get_config": _fake_get_config,
        "habitat.config.default.Config": _Node,
    }, prep.client_builder_args(Path("/tmp/tree")))
    assert prep.agent_dims(cfg) == (1.5, 0.1)
    assert cfg.SIMULATOR.TURN_ANGLE == 15  # it really ran the client's assignments
    assert cfg.DATASET.SCENES_DIR == "/tmp/tree"


def test_a_client_that_sets_the_agent_size_is_seen_and_used():
    source = (
        "from habitat.config.default import get_config as g\n"
        "def build_habitat_config(args):\n"
        "    cfg = g()\n"
        "    cfg.SIMULATOR.AGENT_0.RADIUS = 0.18\n"
        "    cfg.SIMULATOR.AGENT_0.HEIGHT = 0.88\n"
        "    cfg.merge_from_list(['A', 1])\n"
        "    return cfg\n"
    )
    static = prep.inspect_client_builder(source)
    assert static["agent_0_assignments"] == {"RADIUS": 0.18, "HEIGHT": 0.88}
    assert static["sets_height"] and static["sets_radius"]
    assert static["merges"] == ["cfg.merge_from_list"]
    cfg = prep.build_client_config(source.replace("    cfg.merge_from_list(['A', 1])\n", ""),
                                   {"habitat.config.default.get_config": _fake_get_config}, None)
    assert prep.agent_dims(cfg) == (0.88, 0.18)


def test_agent_size_equal_up_to_float32():
    radius32 = struct.unpack("<f", struct.pack("<f", 0.1))[0]  # what habitat_sim hands back
    assert prep.agent_size_equal((1.5, 0.1), (1.5, radius32))
    assert not prep.agent_size_equal((1.5, 0.1), (0.88, 0.18))


def test_summarize_vs_original_reports_without_gating():
    def row(diff, ratio, points):
        return {"vs_original": {"compared": True, "bounds_max_abs_diff": diff, "area_ratio": ratio,
                                "rebaked_points_navigable_on_original": points, "points": 32}}

    scenes = {
        "aaaaaaaaaaa": row(0.05, 0.98, 31),
        "bbbbbbbbbbb": row(4.0, 0.2, 3),  # a mesh in another frame would look like this
        "ccccccccccc": row(0.10, None, 32),
        "ddddddddddd": {"vs_original": {"compared": False, "reason": "original navmesh 2"}},
        "eeeeeeeeeee": {},
    }
    summary = prep.summarize_vs_original(scenes)
    assert summary["report_only"] and summary["compared"] == 3
    assert summary["not_compared"] == {"original navmesh 2": 1, "not recorded": 1}
    assert summary["bounds_max_abs_diff_m"] == {"min": 0.05, "median": 0.10, "max": 4.0}
    assert summary["area_ratio"] == {"min": 0.2, "median": 0.98, "max": 0.98}
    assert summary["scenes_bounds_diff_over_1m"] == ["bbbbbbbbbbb"]
    assert summary["scenes_points_fraction_under_0.5"] == ["bbbbbbbbbbb"]
    assert "3/32" in prep.describe_vs_original(scenes["bbbbbbbbbbb"]["vs_original"])
    assert "not compared" in prep.describe_vs_original(scenes["ddddddddddd"]["vs_original"])
    assert prep.summarize_vs_original({})["bounds_max_abs_diff_m"] is None


# ----------------------------------------------------------------------------
# the tree and rerun rules
# ----------------------------------------------------------------------------
def test_check_roots_refuses_to_write_near_the_originals(tmp_path):
    hm3d = _hm3d(tmp_path)
    with pytest.raises(SystemExit, match="overlaps"):
        prep.check_roots(hm3d / "tree", hm3d)
    with pytest.raises(SystemExit, match="overlaps"):
        prep.check_roots(tmp_path / "x", tmp_path / "x" / "hm3d" / "sub")
    (tmp_path / "linked").symlink_to(hm3d)
    with pytest.raises(SystemExit, match="symlink"):
        prep.check_roots(tmp_path / "linked", tmp_path / "elsewhere")
    prep.check_roots(tmp_path / "tree", hm3d)  # fine


def test_links_are_created_once_and_conflicts_reported(tmp_path):
    target = tmp_path / "t.glb"
    target.write_bytes(b"x")
    link = tmp_path / "tree" / "a" / "a.glb"
    assert prep.link_state(link, target) == "missing"
    assert prep.ensure_link(link, target) is None
    assert os.readlink(link) == str(target) and prep.link_state(link, target) == "ok"
    assert prep.ensure_link(link, target) is None  # idempotent
    other = tmp_path / "other.glb"
    assert "conflict" in prep.ensure_link(link, other)


def test_classify_navmesh(tmp_path):
    plan = prep.scene_plan(tmp_path / "tree", tmp_path / "hm3d", "5jLhtVmWd5F", "00481-5jLhtVmWd5F")
    sha = PARAMS["navmesh_settings_sha256"]
    assert prep.classify_navmesh(plan, sha, False)[0] == "bake"
    plan["navmesh"].parent.mkdir(parents=True)
    plan["navmesh"].write_bytes(_navmesh_bytes())
    assert prep.classify_navmesh(plan, sha, False) == ("bake", "navmesh without its meta file (interrupted bake)")
    meta = {"schema": prep.META_SCHEMA, "navmesh_settings_sha256": sha, "glb_target": str(plan["glb_target"]),
            "navmesh_sha256": _sha(_navmesh_bytes())}
    plan["meta"].write_text(json.dumps(meta))
    assert prep.classify_navmesh(plan, sha, False)[0] == "keep"
    assert prep.classify_navmesh(plan, None, False)[0] == "keep"
    assert prep.classify_navmesh(plan, sha, True) == ("bake", "--rebake")
    assert prep.classify_navmesh(plan, "other", False)[0] == "fail"
    plan["navmesh"].write_bytes(_navmesh_bytes(payload=b"changed"))
    assert "changed after its bake" in prep.classify_navmesh(plan, sha, False)[1]
    plan["navmesh"].unlink()
    plan["navmesh"].symlink_to(tmp_path / "elsewhere.navmesh")
    assert prep.classify_navmesh(plan, sha, False)[0] == "fail"


# ----------------------------------------------------------------------------
# main, with a fake per-scene worker
# ----------------------------------------------------------------------------
def _args(tmp_path, hm3d, *extra):
    return ["--hm3d-root", str(hm3d), "--out-root", str(tmp_path / "tree"), "--mp3d-target",
            str(tmp_path / "mp3d"), "--no-xvfb", *extra]


def _fake_worker(fail=()):
    calls = []

    def run(task, display, log_path, timeout):
        calls.append((task["scene"], task["action"], display))
        if task["scene"] in fail:
            return {"scene": task["scene"], "action": task["action"], "ok": False, "error": "boom",
                    "log": str(log_path)}
        if task["action"] == "bake":
            data = _navmesh_bytes(payload=task["scene"].encode())
            navmesh = Path(task["navmesh"])
            navmesh.write_bytes(data)
            meta = prep.make_meta(task, navmesh_sha256=_sha(data), navmesh_bytes=len(data),
                                  recast_set={"magic_ok": True, "version": 1, "tiles": 1},
                                  glb_sha256=_sha(Path(task["glb_target"]).read_bytes()),
                                  pathfinder={"is_loaded": True, "bounds": [[0, 0, 0], [1, 1, 1]],
                                              "navigable_area": 1.0, "random_points_navigable": 8},
                                  simulator_check={"is_loaded": True, "matches_file": True})
            prep.atomic_write_json(Path(task["meta"]), meta)
        else:
            meta = prep.read_meta(Path(task["meta"]))
        return {"scene": task["scene"], "action": task["action"], "ok": True, "meta": meta, "log": str(log_path)}

    run.calls = calls
    return run


@pytest.fixture()
def fake_habitat(monkeypatch):
    monkeypatch.setattr(prep, "resolve_bake_parameters", lambda *a, **k: dict(PARAMS))
    monkeypatch.setenv("DISPLAY", ":99")


def test_dry_run_writes_nothing(tmp_path, capsys):
    hm3d = _hm3d(tmp_path)
    (tmp_path / "mp3d" / "17DRP5sb8fy").mkdir(parents=True)
    before = _snapshot(tmp_path)
    rc = prep.main(_args(tmp_path, hm3d, "--scenes", "5jLhtVmWd5F,00393-dD37vuCa4FE", "--expect-count", "2",
                         "--dry-run"))
    assert rc == 0
    assert _snapshot(tmp_path) == before and not (tmp_path / "tree").exists()
    out = capsys.readouterr().out
    report = json.loads(out[out.index("{"):])
    assert report["navmesh_actions"] == {"bake": 2}
    assert report["original_navmesh_versions"] == {"2": 1, "missing": 1}  # 5jLh: v2, dD37: none (_hm3d)
    assert report["client_builder"]["sets_radius"] is False
    rc = prep.main(_args(tmp_path, hm3d, "--scenes", "5jLhtVmWd5F", "--expect-count", "2", "--dry-run"))
    assert rc == 2


def test_run_bakes_keeps_and_reports_failures(tmp_path, monkeypatch, fake_habitat, capsys):
    hm3d = _hm3d(tmp_path)
    (tmp_path / "mp3d").mkdir()
    originals = _snapshot(hm3d)
    worker = _fake_worker()
    monkeypatch.setattr(prep, "run_scene_child", worker)
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text("\n".join(SCENES.values()) + "\n")

    assert prep.main(_args(tmp_path, hm3d, "--scenes-file", str(scenes_file), "--workers", "2")) == 0
    tree = tmp_path / "tree"
    assert os.readlink(tree / "mp3d") == str(tmp_path / "mp3d")
    for scene, scene_dir in SCENES.items():
        assert os.readlink(tree / "hm3d" / scene / f"{scene}.glb") == str(hm3d / scene_dir / f"{scene}.basis.glb")
    manifest = json.loads((tree / prep.MANIFEST_NAME).read_text())
    assert manifest["scene_count"] == 3 and manifest["all_requested_ready"] and manifest["failures"] == []
    for scene, meta in manifest["scenes"].items():
        assert meta["navmesh_sha256"] == _sha((tree / "hm3d" / scene / f"{scene}.navmesh").read_bytes())
        assert meta["glb"] == f"hm3d/{scene}/{scene}.glb" and meta["status_this_run"] == "baked"
    assert manifest["requested"]["statuses"] == {"baked": 3}
    assert manifest["requested"]["scenes"] == dict(sorted(SCENES.items()))
    assert manifest["mp3d"]["ids_checked"] == 0
    assert manifest["original_navmesh_versions"] == {"2": 1, "missing": 1, "1": 1}
    assert sorted(c[1] for c in worker.calls) == ["bake"] * 3
    assert {c[2] for c in worker.calls} == {":99"}
    assert _snapshot(hm3d) == originals  # the originals are never written

    # Rerun: everything is kept; one scene that fails its re-check is not listed as ready.
    worker = _fake_worker(fail={"dD37vuCa4FE"})
    monkeypatch.setattr(prep, "run_scene_child", worker)
    assert prep.main(_args(tmp_path, hm3d, "--scenes-file", str(scenes_file))) == 2
    manifest = json.loads((tree / prep.MANIFEST_NAME).read_text())
    assert sorted(c[1] for c in worker.calls) == ["keep"] * 3
    assert sorted(manifest["scenes"]) == ["5jLhtVmWd5F", "wuKkTq5GJbi"]
    assert [f["scene"] for f in manifest["failures"]] == ["dD37vuCa4FE"]
    assert manifest["requested"]["statuses"] == {"kept": 2, "failed": 1}

    # A navmesh changed behind the manifest's back is refused, not rebaked; --rebake replaces it.
    navmesh = tree / "hm3d" / "wuKkTq5GJbi" / "wuKkTq5GJbi.navmesh"
    navmesh.write_bytes(_navmesh_bytes(payload=b"tampered"))
    monkeypatch.setattr(prep, "run_scene_child", _fake_worker())
    assert prep.main(_args(tmp_path, hm3d, "--scenes", "wuKkTq5GJbi")) == 2
    assert prep.main(_args(tmp_path, hm3d, "--scenes", "wuKkTq5GJbi", "--rebake")) == 0
    assert _snapshot(hm3d) == originals


def test_one_writing_run_at_a_time(tmp_path, monkeypatch, fake_habitat):
    hm3d = _hm3d(tmp_path)
    (tmp_path / "mp3d").mkdir()
    monkeypatch.setattr(prep, "run_scene_child", _fake_worker())
    held = prep.lock_tree(tmp_path / "tree")
    with pytest.raises(SystemExit, match="another prepare_hm3d_scenes.py run"):
        prep.main(_args(tmp_path, hm3d, "--scenes", "5jLhtVmWd5F"))
    assert prep.main(_args(tmp_path, hm3d, "--scenes", "5jLhtVmWd5F", "--dry-run")) == 0  # a dry run never locks
    held.close()
    assert prep.main(_args(tmp_path, hm3d, "--scenes", "5jLhtVmWd5F")) == 0
    assert prep.main(_args(tmp_path, hm3d, "--scenes", "5jLhtVmWd5F")) == 0  # released after each run


def test_global_problems_fail_the_run(tmp_path, monkeypatch, fake_habitat):
    hm3d = _hm3d(tmp_path)
    monkeypatch.setattr(prep, "run_scene_child", _fake_worker())
    # no mp3d target, and one HM3D ID that is also an MP3D scene name
    assert prep.main(_args(tmp_path, hm3d, "--scenes", "5jLhtVmWd5F")) == 2
    manifest = json.loads((tmp_path / "tree" / prep.MANIFEST_NAME).read_text())
    assert "is not a directory" in manifest["failures"][0]["error"]
    assert manifest["scene_count"] == 1  # the scene itself was baked
    (tmp_path / "mp3d" / "5jLhtVmWd5F").mkdir(parents=True)
    assert prep.main(_args(tmp_path, hm3d, "--scenes", "5jLhtVmWd5F")) == 2
    manifest = json.loads((tmp_path / "tree" / prep.MANIFEST_NAME).read_text())
    assert "also MP3D scene IDs" in manifest["failures"][0]["error"]


FAKE_HABITAT_SIM = '''
"""A stand-in for habitat_sim 0.1.7: a navmesh is a Recast set header plus JSON derived from the glb."""
import json, os, signal, struct
from . import agent
__version__ = "0.1.7-fake"  # a worker that imported the real one fails the version check
MAGIC = 0x4D534554

class NavMeshSettings:
    def set_defaults(self):
        self.cell_size, self.cell_height, self.agent_height, self.agent_radius = 0.05, 0.2, 1.5, 0.1
        self.agent_max_climb, self.agent_max_slope, self.verts_per_poly = 0.2, 45.0, 6
        self.filter_ledge_spans = True

class PathFinder:
    def __init__(self):
        self.is_loaded, self._mesh, self._i = False, None, 0
    def load_nav_mesh(self, path):
        with open(path, "rb") as handle:
            head, body = handle.read(12), handle.read()
        magic, version, _ = struct.unpack("<iii", head)
        if magic == MAGIC and version == 1:  # 0.1.7 refuses anything else without an error
            try:
                self._mesh, self.is_loaded = json.loads(body), True
            except ValueError:  # a corrupt body: not loaded, no exception
                pass
        return self.is_loaded
    def save_nav_mesh(self, path):
        with open(path, "wb") as handle:
            handle.write(struct.pack("<iii", MAGIC, 1, 1) + json.dumps(self._mesh).encode())
        return True
    def get_bounds(self):
        return tuple(self._mesh["bounds"])
    @property
    def navigable_area(self):
        return self._mesh["area"]
    def seed(self, value):
        self._i = value
    def get_random_navigable_point(self):
        self._i += 1
        return [0.1 * self._i, 0.0, 0.0]
    def is_navigable(self, point):
        return True

class SensorType:
    COLOR = "color"

class SensorSpec:
    pass

class SimulatorConfiguration:
    def __init__(self):
        self.scene_id, self.gpu_device_id = "", 0

class Configuration:
    def __init__(self, sim_cfg, agents):
        self.sim_cfg, self.agents = sim_cfg, agents

class Simulator:
    def __init__(self, config):
        scene = config.sim_cfg.scene_id
        with open(scene, "rb") as handle:
            self._glb = handle.read()
        if b"crash" in self._glb:
            os.kill(os.getpid(), signal.SIGSEGV)
        self.pathfinder = PathFinder()
        navmesh = os.path.splitext(scene)[0] + ".navmesh"  # 0.1.7's lookup next to the scene
        if os.path.exists(navmesh):
            self.pathfinder.load_nav_mesh(navmesh)
    def recompute_navmesh(self, pathfinder, settings):
        size = float(len(self._glb))
        pathfinder._mesh = {"bounds": [[0.0, 0.0, 0.0], [size, 1.0, settings.agent_radius]], "area": size}
        pathfinder.is_loaded = True
        return True
    def close(self):
        pass
'''
FAKE_AGENT = '''
class AgentConfiguration:
    def __init__(self):
        self.height, self.radius, self.sensor_specifications = 1.5, 0.1, []
'''


def test_the_real_worker_subprocess_with_a_fake_habitat_sim(tmp_path, monkeypatch):
    package = tmp_path / "fake_site" / "habitat_sim"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(FAKE_HABITAT_SIM)
    (package / "agent.py").write_text(FAKE_AGENT)
    # Loaded here under a private name, so this process's habitat_sim (if any) is untouched.
    spec = importlib.util.spec_from_file_location("_exp28_fake_habitat_sim", package / "__init__.py",
                                                  submodule_search_locations=[str(package)])
    fake = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, fake)
    spec.loader.exec_module(fake)
    settings = prep.navmesh_settings_dict(prep.make_navmesh_settings(fake, 0.1, 1.5))
    params = dict(PARAMS, habitat_sim_version=fake.__version__, navmesh_settings=settings,
                  navmesh_settings_sha256=prep.settings_digest(settings, fake.__version__))
    monkeypatch.setenv("EXP28_PREP_IMPORT_TORCH", "0")
    monkeypatch.setattr(prep, "resolve_bake_parameters", lambda *a, **k: params)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "fake_site"))  # the worker interpreter imports the fake
    monkeypatch.setenv("DISPLAY", ":99")
    # sorted: 5jLhtVmWd5F v2, crashcrash1 none, dD37vuCa4FE v1 with a body no loader reads, wuKkTq5GJbi below
    hm3d = _hm3d(tmp_path, dict(SCENES, crashcrash1="00999-crashcrash1"), navmesh_versions=(2, None, 1, None))
    (hm3d / "00999-crashcrash1" / "crashcrash1.basis.glb").write_bytes(b"crash")
    # wuKkTq5GJbi's own navmesh: v1, which the fake reads; the rebake (bounds [[0,0,0],[15,1,0.1]], area 15
    # from its 15-byte glb) is compared with it, for the record only.
    (hm3d / "00396-wuKkTq5GJbi" / "wuKkTq5GJbi.basis.navmesh").write_bytes(
        _navmesh_bytes(1, json.dumps({"bounds": [[0.0, 0.0, 0.0], [15.5, 1.0, 0.1]], "area": 12.0}).encode()))
    (tmp_path / "mp3d").mkdir()
    originals = _snapshot(hm3d)
    tree = tmp_path / "tree"
    stale = tree / "hm3d" / "dD37vuCa4FE" / ".dD37vuCa4FE.navmesh.tmp4242"  # a worker killed at its timeout
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"partial")

    assert prep.main(_args(tmp_path, hm3d, "--all-scenes", "--workers", "2", "--timeout", "120")) == 2
    manifest = json.loads((tree / prep.MANIFEST_NAME).read_text())
    assert sorted(manifest["scenes"]) == sorted(SCENES)
    (failure,) = manifest["failures"]
    assert failure["scene"] == "crashcrash1" and "without a result" in failure["error"]
    assert not stale.exists()
    vs = {scene: meta["vs_original"] for scene, meta in manifest["scenes"].items()}
    assert vs["5jLhtVmWd5F"] == {"compared": False, "reason": "original navmesh 2"}
    # an unreadable original is reported; the scene itself is ready (the comparison never gates)
    assert vs["dD37vuCa4FE"] == {"compared": False, "reason": "original v1 navmesh does not load"}
    assert vs["wuKkTq5GJbi"]["compared"] and vs["wuKkTq5GJbi"]["rebaked_points_navigable_on_original"] == 32
    assert vs["wuKkTq5GJbi"]["bounds_max_abs_diff"] == pytest.approx(0.5)
    assert vs["wuKkTq5GJbi"]["area_ratio"] == pytest.approx(15 / 12)
    summary = manifest["vs_original_navmesh"]
    assert summary["compared"] == 1
    assert summary["not_compared"] == {"original navmesh 2": 1, "original v1 navmesh does not load": 1}
    for scene in SCENES:
        navmesh = tree / "hm3d" / scene / f"{scene}.navmesh"
        meta = manifest["scenes"][scene]
        assert prep.recast_set_header(navmesh)["version"] == 1
        assert meta["navmesh_sha256"] == _sha(navmesh.read_bytes())
        assert navmesh.stat().st_mode & 0o777 == 0o444
        assert meta["simulator_check"] == {"is_loaded": True, "matches_file": True}
        assert meta["pathfinder"]["random_points_navigable"] == prep.RANDOM_POINTS
        assert meta["glb_sha256"] == _sha((hm3d / SCENES[scene] / f"{scene}.basis.glb").read_bytes())
        assert not list(navmesh.parent.glob(".*.tmp*"))
    assert (tree / "prepare_logs").is_dir()
    assert _snapshot(hm3d) == originals

    # Rerun the three good scenes: the worker re-checks them (keep) and the manifest is unchanged.
    assert prep.main(_args(tmp_path, hm3d, "--scenes", ",".join(SCENES), "--workers", "3")) == 0
    again = json.loads((tree / prep.MANIFEST_NAME).read_text())
    assert again["requested"]["statuses"] == {"kept": 3}
    assert {s: m["navmesh_sha256"] for s, m in again["scenes"].items()} == \
        {s: m["navmesh_sha256"] for s, m in manifest["scenes"].items()}
    # The crashed scene's directory is reported, not listed.
    assert [p["scene"] for p in again["tree_problems_not_requested"]] == ["crashcrash1"]


def test_parse_args_wants_exactly_one_kind_of_source():
    with pytest.raises(SystemExit):
        prep.parse_args([])
    with pytest.raises(SystemExit):
        prep.parse_args(["--all-scenes", "--scenes", "x"])
    assert prep.parse_args(["--scenes", "x", "--scenes-file", "f"]).scenes == ["x"]


# ----------------------------------------------------------------------------
# with habitat
# ----------------------------------------------------------------------------
def _habitat_sim():
    """habitat_sim imported as the script imports it (NumPy aliases first), or skip."""
    try:
        return prep.import_habitat_sim()
    except ImportError as exc:
        pytest.skip(f"no habitat_sim: {exc}")


def test_navmesh_settings_with_habitat_sim():
    hs = _habitat_sim()
    settings = prep.navmesh_settings_dict(prep.make_navmesh_settings(hs, 0.1, 1.5))
    assert settings["agent_radius"] == pytest.approx(0.1) and settings["agent_height"] == pytest.approx(1.5)
    assert settings["cell_size"] == pytest.approx(0.05)
    assert json.loads(json.dumps(settings)) == settings  # survives the task file unchanged


def test_client_agent_size_with_habitat_lab(tmp_path):
    _habitat_sim()
    habitat_default = pytest.importorskip("habitat.config.default")
    params = prep.resolve_bake_parameters(CLIENT, tmp_path, "")
    default = habitat_default.get_config().SIMULATOR.AGENT_0
    assert (params["agent_height"], params["agent_radius"]) == (float(default.HEIGHT), float(default.RADIUS))
    assert params["navmesh_settings"]["agent_radius"] == pytest.approx(params["agent_radius"])


@pytest.mark.skipif(not os.environ.get("EXP28_TEST_HM3D_SCENE"), reason="set EXP28_TEST_HM3D_SCENE=00xxx-<ID>")
def test_real_bake_of_one_scene(tmp_path):
    _habitat_sim()
    pytest.importorskip("habitat")
    hm3d = Path(os.environ.get("EXP28_TEST_HM3D_ROOT", str(prep.DEFAULT_HM3D_ROOT)))
    xvfb_base = int(os.environ.get("EXP28_TEST_XVFB_BASE", "795"))
    (tmp_path / "mp3d").mkdir()

    def bake(tree, *extra):
        rc = prep.main(["--hm3d-root", str(hm3d), "--out-root", str(tree), "--mp3d-target", str(tmp_path / "mp3d"),
                        "--scenes", os.environ["EXP28_TEST_HM3D_SCENE"], "--workers", "1", *extra])
        manifest = json.loads((tree / prep.MANIFEST_NAME).read_text())
        assert rc == 0, manifest["failures"]
        (meta,) = manifest["scenes"].values()
        return manifest, meta

    manifest, meta = bake(tmp_path / "tree", "--xvfb-base", str(xvfb_base))
    assert meta["recast_set"]["version"] == 1 and meta["pathfinder"]["is_loaded"]
    assert meta["simulator_check"] == {"is_loaded": True, "matches_file": True}
    assert meta["agent"] == {"height": manifest["agent"]["height"], "radius": manifest["agent"]["radius"]}
    print("client AGENT_0:", json.dumps(manifest["client"]["agent_0"]),
          "equals AgentConfiguration():", manifest["client"]["agent_0_equals_habitat_sim_default"])
    print(prep.describe_vs_original(meta["vs_original"]), f"(load {meta['scene_load_seconds']} s, "
          f"bake {meta['bake_seconds']} s)")
    # The same scene without textures: the navmesh comes from the collision mesh, so it must be the same file,
    # or --skip-textures (which the settings digest does not cover) must not be mixed with textured bakes.
    _, untextured = bake(tmp_path / "tree_untextured", "--skip-textures", "--no-verify-sim",
                         "--xvfb-base", str(xvfb_base + 1))
    assert untextured["navmesh_sha256"] == meta["navmesh_sha256"]
    print(f"--skip-textures: same navmesh; load {untextured['scene_load_seconds']} s")

"""EXP-28 T4: the client (scripts/evaluation/r2r_val_unseen.py) on HM3D scenes.

Every episode the client runs is checked right after env.reset(): its scene has a navmesh
(habitat-sim 0.1.7 cannot read HM3D v0.2's version-2 navmeshes and says nothing) and its goal
is reachable from its start.  An HM3D scene (EXP-28's tree hm3d/<ID>/<ID>.glb) has the sha256
of the navmesh Habitat loads in the s2 record's episode.json, checked against the episode
file's ``info.navmesh_sha256`` when it names one.  A scene has one key everywhere: its
directory, which must be its file stem and one \\w+ token (EXP-28's tree hm3d/<ID>/<ID>.glb),
so the fixed-cohort filter, the progress rows and the log parsers agree.

MP3D must not notice: the same keys as before (on the R2R-CE files themselves where they are
present, EnvDrop's included: the 4090 collects EnvDrop with this client), and the same actions, requests, output files, log and record bytes as the client at
its last commit before EXP-28 (BASELINE_COMMIT, read with git; skipped without it).  The pure
parts run on the client's own definitions, lifted out of the file (no torch, no habitat); the
runs use the client end to end on the fakes of test_s2_record_collection; the tests that use
habitat_sim on real MP3D navmeshes run where the 4090 container has them.
"""

from __future__ import annotations

import __future__
import ast
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.exp19 import select_cases as sc
from scripts.exp25 import probe_metrics as pm
from tests import test_s2_record_collection as harness
from tests.test_s2_record_collection import client  # noqa: F401  (fixture)

CLIENT = Path(__file__).resolve().parents[1] / "scripts" / "evaluation" / "r2r_val_unseen.py"
# The last commit that changed the client before EXP-28: MP3D runs must match it byte for byte.
BASELINE_COMMIT = "61cd69ce80cdd389df7b571e4f83c6c6d7507a5c"
LOOPS = ("_run_eval_panoramic_vlm", "run_eval_rpc_panoramic", "run_eval")
# The 4090 container (docs/ops/deploy_rtx4090.md); the tests that read them skip elsewhere.
R2R_ROOT = Path("/workspace/R2R_VLNCE_v1-3_preprocessed")
MP3D_ROOT = Path("/dataset/mp3d")
# The R2R-CE files this client runs on MP3D (EnvDrop: EXP-25's collection, resumable on this code).
MP3D_SPLITS = ("val_unseen", "val_seen", "train", "envdrop")
# val_unseen's 11 scenes and a few train ones.
MP3D_SCENES = (
    "2azQ1b91cZZ", "8194nk5LbLH", "EU6Fwq7SyZv", "QUCTc6BB5sX", "TbHJrupSAjP", "X7HyMhZNoso",
    "Z6MFQCViBuw", "oLBMNvg9in8", "pLe4wQe7qrG", "x8F5xyUWy9e", "zsNo4HB9uLZ",
    "17DRP5sb8fy", "5LpN3gDmAk7", "1LXtFkjw3qL", "s8pcmisQ38h",
)
# B3's held-out HM3D scenes, as <scene directory>, <ID>.
HM3D_SCENES = (("00481-5jLhtVmWd5F", "5jLhtVmWd5F"), ("00393-dD37vuCa4FE", "dD37vuCa4FE"))

# What an MP3D record's episode.json held before EXP-28: run_meta, the collector's episode
# fields, the outcome and what EpisodeRecordWriter.commit adds.
MP3D_EPISODE_JSON_KEYS = {
    "split", "data_path", "data_sha256", "episode_list", "episode_list_sha256", "config_path",
    "config_sha256", "client_code_sha256", "protocol_seed", "rpc_sampling_protocol", "rpc_protocol",
    "rpc_policy_mode", "rpc_policy_fingerprint", "history_pose_source", "vlm_image_size",
    "traj_image_size", "lookdown_image_size", "num_history", "rpc_jpeg_quality",
    "max_steps_per_episode", "server",
    "collector", "trajectory_id", "instruction", "instruction_raw", "goals", "reference_path",
    "start_position", "start_rotation_xyzw", "geodesic_distance", "pose_frame",
    "outcome",
    "schema", "scene_id", "episode_id", "num_steps", "num_calls", "camera", "committed_unix",
}


# ── the client's own definitions, without its imports ──────────────────────


_HELPERS = (
    "_SCENE_KEY_RE",
    "_episode_scene_key",
    "_habitat_episode_key",
    "_episode_navmesh_sha256",
    "_check_reset_episode",
    "_sha256_file",
)


def _defined_names(node: ast.stmt) -> set[str]:
    if isinstance(node, ast.FunctionDef):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {target.id for target in node.targets if isinstance(target, ast.Name)}
    return set()


@pytest.fixture(scope="module")
def h() -> SimpleNamespace:
    """The helpers as the client file defines them, executed on their own."""
    tree = ast.parse(CLIENT.read_text(encoding="utf-8"))
    body = [node for node in tree.body if _defined_names(node) & set(_HELPERS)]
    assert set().union(*map(_defined_names, body)) == set(_HELPERS)
    namespace: dict[str, Any] = {
        "re": re, "os": os, "math": math, "hashlib": hashlib, "Path": Path, "Any": Any,
    }
    code = compile(
        ast.Module(body=body, type_ignores=[]), str(CLIENT), "exec",
        flags=__future__.annotations.compiler_flag, dont_inherit=True,
    )
    exec(code, namespace)
    return SimpleNamespace(**{name: namespace[name] for name in _HELPERS})


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _old_loop_key(scene_id: str) -> str:
    return scene_id.split("/")[-2]  # the three episode loops, before EXP-28


def _old_filter_key(scene_id: str) -> str:
    return Path(scene_id).stem  # _habitat_episode_key, before EXP-28


# ── one scene key ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("root", ["mp3d", "/dataset/mp3d", "/workspace/exp28/scene_datasets/mp3d"])
def test_mp3d_scene_keys_are_the_old_ones(h, root):
    for scene in MP3D_SCENES:
        scene_id = f"{root}/{scene}/{scene}.glb"
        assert h._episode_scene_key(scene_id) == _old_loop_key(scene_id) == _old_filter_key(scene_id) == scene
        episode = SimpleNamespace(scene_id=scene_id, episode_id="12")
        assert h._habitat_episode_key(episode) == (scene, 12)


@pytest.mark.parametrize("split", MP3D_SPLITS)
def test_r2r_ce_files_keep_every_scene_key(h, split):
    path = R2R_ROOT / split / f"{split}.json.gz"
    if not path.is_file():
        pytest.skip(f"no {path} here")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        episodes = json.load(handle)["episodes"]
    assert episodes
    for episode in episodes:
        # As the dataset file has it, and as Habitat hands it over (joined to the scenes dir).
        for scene_id in (episode["scene_id"], os.path.join("/workspace/exp28/scene_datasets", episode["scene_id"])):
            key = h._episode_scene_key(scene_id)
            assert key == _old_loop_key(scene_id) == _old_filter_key(scene_id)


def test_the_exp28_hm3d_tree_gives_the_bare_id_everywhere(h):
    for _, scene in HM3D_SCENES:
        scene_id = f"/workspace/exp28/scene_datasets/hm3d/{scene}/{scene}.glb"
        assert h._episode_scene_key(scene_id) == _old_loop_key(scene_id) == _old_filter_key(scene_id) == scene
        assert h._episode_scene_key(f"hm3d/{scene}/{scene}.glb") == scene
        # The two log parsers read the client's header line back to the same key.
        line = f"[1/1] Episode {scene}_0007: Walk past the sofa"
        for regex in (pm.HEADER_RE, sc.HEADER_RE):
            assert regex.match(line).group(3, 4) == (scene, "0007")
        assert pm.scene_name(scene_id) == scene  # probe_metrics keys the episode file by file stem


def test_the_episodes_and_cohorts_build_hm3d_ids_the_client_keys(h):
    # T2 writes the episodes' scene_id and the cohorts' scene key; the client must read both back
    # as the same bare ID, also for StreamVLN's episode ids, which run past four digits.
    from scripts.exp28 import build_hm3d_episodes as bhe
    from scripts.tools.build_r2r_train_cohorts import scene_name as cohort_scene_name

    for scene_dir, scene in HM3D_SCENES:
        assert bhe.parse_source_scene(f"hm3d/{scene_dir}/{scene}.basis.glb") == (scene_dir, scene)
        assert bhe.dir_id(scene_dir) == scene
        scene_id = bhe.hm3d_scene_id(scene)
        for loaded in (scene_id, os.path.join("/workspace/exp28/scene_datasets", scene_id)):
            assert h._episode_scene_key(loaded) == cohort_scene_name(scene_id) == pm.scene_name(loaded) == scene
        episode = SimpleNamespace(scene_id=os.path.join("/workspace/exp28/scene_datasets", scene_id), episode_id=154321)
        assert h._habitat_episode_key(episode) == (scene, 154321)
        line = f"[3/400] Episode {scene}_{154321:04d}: Walk past the sofa"
        for regex in (pm.HEADER_RE, sc.HEADER_RE):
            assert regex.match(line).group(3, 4) == (scene, "154321")


@pytest.mark.parametrize(
    "layout",
    [
        "hm3d/train/{dir}/{id}.basis.glb",  # HM3D's own layout: 00xxx-<ID> vs <ID>.basis
        "hm3d/{dir}/{id}.basis.glb",  # the annotation's scene_id (no train/)
        "hm3d/{id}/{id}.basis.glb",  # the ID directory, the original file name
        "hm3d/{dir}/{dir}.glb",  # stem and directory agree, but the log drops the hyphen
    ],
)
def test_layouts_that_key_a_scene_two_ways_are_refused(h, layout):
    scene_dir, scene = HM3D_SCENES[0]
    scene_id = "/workspace/exp23/data/" + layout.format(dir=scene_dir, id=scene)
    with pytest.raises(ValueError, match=r"is not <scene>/<scene>\.<ext>"):
        h._episode_scene_key(scene_id)
    with pytest.raises(ValueError, match=r"is not <scene>/<scene>\.<ext>"):
        h._habitat_episode_key(SimpleNamespace(scene_id=scene_id, episode_id=1))


def test_the_log_parsers_drop_a_hyphenated_scene_without_a_word():
    # Why the client refuses 00xxx-<ID> directories: probe_metrics would silently skip them.
    for regex in (pm.HEADER_RE, sc.HEADER_RE):
        assert regex.match("[1/1] Episode 00481-5jLhtVmWd5F_0007: Walk") is None


def test_the_fixed_cohort_filter_keys_hm3d_as_the_loop_does(client, monkeypatch, tmp_path):
    root = "/workspace/exp28/scene_datasets"
    dataset_episodes = [
        SimpleNamespace(scene_id=f"{root}/hm3d/{scene}/{scene}.glb", episode_id=str(episode_id))
        for (_, scene) in HM3D_SCENES
        for episode_id in (3, 1)
    ] + [SimpleNamespace(scene_id=f"{root}/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb", episode_id="3")]
    built = []

    class Env:
        def __init__(self, config, dataset=None):
            self.episodes = list(dataset.episodes)
            built.append(self)

        def close(self):
            pass

    habitat = SimpleNamespace(
        make_dataset=lambda id_dataset, config: SimpleNamespace(episodes=list(dataset_episodes)), Env=Env
    )
    monkeypatch.setattr(client, "habitat", habitat)
    wanted = [("zsNo4HB9uLZ", 3), (HM3D_SCENES[1][1], 1), (HM3D_SCENES[0][1], 3)]
    episode_list = tmp_path / "shard_000.json"
    episode_list.write_text(json.dumps({"episodes": [{"scene_id": s, "episode_id": e} for s, e in wanted]}))
    hab_cfg = SimpleNamespace(DATASET=SimpleNamespace(TYPE="R2RVLN-v1"))
    env = client._create_habitat_env(hab_cfg, SimpleNamespace(episode_list=str(episode_list)))
    assert [client._habitat_episode_key(episode) for episode in env.episodes] == wanted
    assert [(client._episode_scene_key(e.scene_id), int(e.episode_id)) for e in env.episodes] == wanted

    # HM3D's own layout fails on the layout, not as "keys absent from the configured dataset".
    scene_dir, scene = HM3D_SCENES[0]
    dataset_episodes[0].scene_id = f"/workspace/exp23/data/hm3d/train/{scene_dir}/{scene}.basis.glb"
    with pytest.raises(ValueError, match=r"is not <scene>/<scene>\.<ext>"):
        client._create_habitat_env(hab_cfg, SimpleNamespace(episode_list=str(episode_list)))


# ── the check after env.reset() ────────────────────────────────────────────


class _Sim:
    def __init__(self, *, loaded=True, geodesic=4.0):
        self.pathfinder = SimpleNamespace(is_loaded=loaded)
        self.geodesic = geodesic
        self.queries: list[tuple] = []

    def geodesic_distance(self, position_a, position_b, episode=None):
        self.queries.append((position_a, position_b, episode))
        return self.geodesic


def _reset_episode(scene_id="mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb", info=None):
    return SimpleNamespace(
        episode_id=7,
        scene_id=scene_id,
        start_position=[1.0, 0.1, 2.0],
        goals=[SimpleNamespace(position=[4.0, 0.1, 2.0]), SimpleNamespace(position=[1.0, 0.1, 9.0])],
        info={"geodesic_distance": 4.0} if info is None else info,
        _shortest_path_cache="untouched",
    )


def test_a_navigable_mp3d_episode_passes_and_only_reads(h):
    sim, episode = _Sim(), _reset_episode()
    assert h._check_reset_episode(SimpleNamespace(_sim=sim), episode) is None
    # DistanceToGoal's query (start to every goal), without the episode: no path cache.
    assert sim.queries == [([1.0, 0.1, 2.0], [[4.0, 0.1, 2.0], [1.0, 0.1, 9.0]], None)]
    assert episode._shortest_path_cache == "untouched"
    # MP3D names no navmesh and is not under hm3d/: nothing is read (this one is not on disk).
    assert h._episode_navmesh_sha256(episode) is None


def test_a_scene_without_a_loaded_navmesh_stops_the_run(h):
    sim = _Sim(loaded=False)
    with pytest.raises(RuntimeError, match="no navmesh loaded for mp3d/zsNo4HB9uLZ"):
        h._check_reset_episode(SimpleNamespace(_sim=sim), _reset_episode())
    assert sim.queries == []


@pytest.mark.parametrize("geodesic", [math.inf, math.nan])
def test_an_unreachable_goal_stops_the_run(h, geodesic):
    with pytest.raises(RuntimeError, match="start-to-goal geodesic is"):
        h._check_reset_episode(SimpleNamespace(_sim=_Sim(geodesic=geodesic)), _reset_episode())


def _hm3d_scene(root: Path, scene: str, content: bytes = b"rebaked v1 navmesh") -> tuple[str, str]:
    navmesh = root / "hm3d" / scene / f"{scene}.navmesh"
    navmesh.parent.mkdir(parents=True, exist_ok=True)
    navmesh.write_bytes(content + scene.encode())
    return str(root / "hm3d" / scene / f"{scene}.glb"), _sha(navmesh.read_bytes())


def test_an_hm3d_episode_names_the_navmesh_habitat_loads(h, tmp_path):
    scene = HM3D_SCENES[0][1]
    scene_id, sha = _hm3d_scene(tmp_path, scene)
    episode = _reset_episode(scene_id, info={"geodesic_distance": 4.0, "navmesh_sha256": sha})
    assert h._episode_navmesh_sha256(episode) == sha
    h._check_reset_episode(SimpleNamespace(_sim=_Sim()), episode)

    (tmp_path / "hm3d" / scene / f"{scene}.navmesh").write_bytes(b"baked again with another agent radius")
    with pytest.raises(RuntimeError, match=f"but the episode file was built on {sha}"):
        h._check_reset_episode(SimpleNamespace(_sim=_Sim()), episode)
    (tmp_path / "hm3d" / scene / f"{scene}.navmesh").unlink()
    with pytest.raises(RuntimeError, match="no navmesh file"):
        h._episode_navmesh_sha256(episode)


def test_an_hm3d_episode_file_that_names_no_navmesh_still_records_it(h, tmp_path):
    # build_hm3d_episodes writes only info.geodesic_distance; the tree alone says HM3D.
    scene = HM3D_SCENES[1][1]
    scene_id, sha = _hm3d_scene(tmp_path, scene)
    episode = _reset_episode(scene_id)
    assert "navmesh_sha256" not in episode.info
    assert h._episode_navmesh_sha256(episode) == sha
    h._check_reset_episode(SimpleNamespace(_sim=_Sim()), episode)
    (tmp_path / "hm3d" / scene / f"{scene}.navmesh").unlink()
    with pytest.raises(RuntimeError, match=f"no navmesh file {re.escape(str(tmp_path))}"):
        h._check_reset_episode(SimpleNamespace(_sim=_Sim()), _reset_episode(scene_id))
    # An MP3D scene with a navmesh on disk is still not read.
    navmesh = tmp_path / "mp3d" / "zsNo4HB9uLZ" / "zsNo4HB9uLZ.navmesh"
    navmesh.parent.mkdir(parents=True)
    navmesh.write_bytes(b"mp3d navmesh")
    assert h._episode_navmesh_sha256(_reset_episode(str(navmesh.with_suffix(".glb")))) is None


def _real_mp3d_split(split: str) -> tuple[Any, Any, dict[str, list[dict]]]:
    """habitat_sim, HabitatSim's module and the split's episodes by scene; skips off the 4090."""
    habitat_sim = pytest.importorskip("habitat_sim")
    simulator = pytest.importorskip("habitat.sims.habitat_simulator.habitat_simulator")
    path = R2R_ROOT / split / f"{split}.json.gz"
    if not path.is_file() or not MP3D_ROOT.is_dir():
        pytest.skip(f"no {path} or {MP3D_ROOT} here")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        episodes = json.load(handle)["episodes"]
    by_scene: dict[str, list[dict]] = {}
    for raw in episodes:
        by_scene.setdefault(raw["scene_id"].split("/")[-2], []).append(raw)
    return habitat_sim, simulator, by_scene


def _real_env(habitat_sim, simulator, scene: str) -> SimpleNamespace:
    """An env whose sim is a real navmesh behind HabitatSim's own geodesic_distance.

    That method reads nothing of the simulator but its pathfinder, so no renderer is needed.
    """
    pathfinder = habitat_sim.PathFinder()
    pathfinder.load_nav_mesh(str(MP3D_ROOT / scene / f"{scene}.navmesh"))
    sim = SimpleNamespace(pathfinder=pathfinder)
    sim.geodesic_distance = types.MethodType(simulator.HabitatSim.geodesic_distance, sim)
    return SimpleNamespace(_sim=sim)


def _real_episode(scene: str, raw: dict) -> SimpleNamespace:
    """As the R2R-CE loader hands it over (scene_id joined to the scenes dir; ``info`` may be absent)."""
    return SimpleNamespace(
        episode_id=raw["episode_id"],
        scene_id=str(MP3D_ROOT / scene / f"{scene}.glb"),
        start_position=raw["start_position"],
        goals=[SimpleNamespace(position=goal["position"]) for goal in raw["goals"]],
        info=raw.get("info"),
        _shortest_path_cache="untouched",
    )


def test_the_check_on_a_real_mp3d_navmesh(h):
    habitat_sim, simulator, by_scene = _real_mp3d_split("val_unseen")
    unloaded = SimpleNamespace(_sim=SimpleNamespace(pathfinder=habitat_sim.PathFinder()))
    with pytest.raises(RuntimeError, match="no navmesh loaded"):
        h._check_reset_episode(unloaded, _reset_episode())

    scene, episodes = sorted(by_scene.items())[0]
    env = _real_env(habitat_sim, simulator, scene)
    assert env._sim.pathfinder.is_loaded
    for raw in episodes[:5]:
        episode = _real_episode(scene, raw)
        h._check_reset_episode(env, episode)
        assert episode._shortest_path_cache == "untouched"
        geodesic = env._sim.geodesic_distance(episode.start_position, [goal.position for goal in episode.goals])
        expected = float(raw["info"]["geodesic_distance"])
        assert abs(geodesic - expected) <= max(1.0, 0.15 * expected)  # the dataset's own start-goal geodesic


@pytest.mark.parametrize("split", MP3D_SPLITS)
def test_every_r2r_ce_episode_passes_the_check(h, split):
    # The check stops a run; on MP3D it must never have anything to stop.
    habitat_sim, simulator, by_scene = _real_mp3d_split(split)
    failures, checked = [], 0
    for scene, episodes in sorted(by_scene.items()):
        env = _real_env(habitat_sim, simulator, scene)
        for raw in episodes:
            checked += 1
            try:
                h._check_reset_episode(env, _real_episode(scene, raw))
            except RuntimeError as error:
                failures.append(str(error))
    assert checked == sum(map(len, by_scene.values())) > 0
    assert not failures, f"{len(failures)}/{checked} {split} episodes would stop a run, e.g. {failures[:5]}"


# ── the loops, on the source ───────────────────────────────────────────────


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def test_every_loop_keys_and_checks_each_episode_it_runs():
    source = CLIENT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert 'split("/")[-2]' not in source  # one derivation of the scene key
    for name in LOOPS:
        fn = _function(tree, name)
        loop = next(
            node for node in ast.walk(fn)
            if isinstance(node, ast.While)
            and any(ast.unparse(statement) == "observations = env.reset()" for statement in node.body)
        )
        body = [ast.unparse(statement) for statement in loop.body]
        reset = body.index("observations = env.reset()")
        key = body.index("scene_id = _episode_scene_key(episode.scene_id)")
        resumed = body.index("if ep_key in done_set:\n    continue")
        check = body.index("_check_reset_episode(env, episode)")
        runs = body.index("eval_count += 1")
        # Checked once the episode is known to run, before anything of it is run or logged.
        assert reset < key < resumed < check < runs, name
    checks = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_check_reset_episode"
    ]
    resets = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "reset"
        and ast.unparse(node.func.value) == "env"
    ]
    assert len(checks) == len(resets) == len(LOOPS)


# ── end to end ─────────────────────────────────────────────────────────────


def _record_members(path: str) -> dict[str, Any]:
    """Every member of a record tar; episode.json parsed, without its commit time."""
    with tarfile.open(path) as tar:
        members = {member.name: tar.extractfile(member).read() for member in tar.getmembers()}
    episode = json.loads(members.pop("episode.json"))
    episode.pop("committed_unix")
    return {"episode.json": episode, **members}


def _rows(run) -> list[dict]:
    rows = []
    for row in run.progress:
        row = dict(row)
        record = dict(row["s2_record"])
        record["path"] = str(Path(record["path"]).resolve().relative_to(Path(run.records).resolve()))
        rows.append({**row, "s2_record": record})
    return rows


@pytest.mark.parametrize("mode", ["vo", "native"])
def test_mp3d_runs_are_unchanged(client, monkeypatch, tmp_path, mode):
    unchecked = []
    real_check = client._check_reset_episode
    monkeypatch.setattr(client, "_check_reset_episode", lambda env, episode: unchecked.append(episode))
    before = harness._run(client, monkeypatch, tmp_path, mode=mode, record=True, tag="before")
    monkeypatch.setattr(client, "_check_reset_episode", real_check)
    after = harness._run(client, monkeypatch, tmp_path, mode=mode, record=True, tag="after")

    assert len(unchecked) == len(harness.EPISODES)  # the loop calls it once per episode it runs
    assert before.env.geodesic_queries == []
    assert [(query[2], len(query[1])) for query in after.env.geodesic_queries] == [(None, 1)] * len(harness.EPISODES)
    assert after.env.trace == before.env.trace
    assert len(after.env.primitive) == len(before.env.primitive)
    for (key, action, position, rotation), (key0, action0, position0, rotation0) in zip(
        after.env.primitive, before.env.primitive
    ):
        assert (key, action, position.tolist(), rotation.tolist()) == (
            key0, action0, position0.tolist(), rotation0.tolist()
        )
    for name in ("method", "payload", "blobs"):
        assert [r[name] for r in after.server.requests] == [r[name] for r in before.server.requests]
    assert _rows(after) == _rows(before)
    assert [(row["scene_id"], row["episode_id"]) for row in after.progress] == [spec[:2] for spec in harness.EPISODES]
    for row_after, row_before in zip(after.progress, before.progress):
        members = _record_members(row_after["s2_record"]["path"])
        assert members == _record_members(row_before["s2_record"]["path"])
        assert set(members["episode.json"]) | {"committed_unix"} == MP3D_EPISODE_JSON_KEYS


def _baseline_client(monkeypatch) -> types.ModuleType:
    """The client as BASELINE_COMMIT has it, run as if it were the file on disk.

    Its ``__file__`` is the current client's, so it finds the repository the same way and
    stamps the same client_code_sha256: only the code it runs differs.  Call it inside a test
    that uses the ``client`` fixture, whose stubs it imports.
    """
    try:
        shown = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(CLIENT.parents[2]), "show",
             f"{BASELINE_COMMIT}:scripts/evaluation/r2r_val_unseen.py"],
            capture_output=True, check=False,
        )
    except FileNotFoundError:
        pytest.skip("no git here")
    if shown.returncode != 0:
        pytest.skip(f"{BASELINE_COMMIT[:7]} is not in this checkout: {shown.stderr.decode(errors='replace').strip()}")
    name = "_exp28_t4_baseline_client"
    module = types.ModuleType(name)
    module.__file__ = str(CLIENT)
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(shown.stdout, str(CLIENT), "exec", dont_inherit=True), module.__dict__)
    return module


def _output_files(run_root: Path, tag: str) -> dict[str, bytes]:
    """Every file the run wrote under out_<tag>, its own directory names made tag-free."""
    output = run_root / f"out_{tag}"
    return {
        str(path.relative_to(output)): path.read_bytes().replace(f"_{tag}".encode(), b"_TAG")
        for path in sorted(output.rglob("*")) if path.is_file()
    }


@pytest.mark.parametrize("mode", ["vo", "native"])
def test_mp3d_runs_match_the_client_before_exp28(client, monkeypatch, tmp_path, capfd, mode):
    baseline = _baseline_client(monkeypatch)
    capfd.readouterr()
    before = harness._run(baseline, monkeypatch, tmp_path, mode=mode, record=True, tag="t4before")
    log_before = capfd.readouterr().out.replace("_t4before", "_TAG")
    after = harness._run(client, monkeypatch, tmp_path, mode=mode, record=True, tag="t4after")
    log_after = capfd.readouterr().out.replace("_t4after", "_TAG")

    assert after.env.trace and after.env.trace == before.env.trace
    assert [(k, a, p.tolist(), r.tolist()) for k, a, p, r in after.env.primitive] == [
        (k, a, p.tolist(), r.tolist()) for k, a, p, r in before.env.primitive
    ]
    for name in ("method", "payload", "blobs"):
        assert [r[name] for r in after.server.requests] == [r[name] for r in before.server.requests]
    assert _output_files(tmp_path, "t4after") == _output_files(tmp_path, "t4before")  # progress rows and all
    assert log_after == log_before  # what probe_metrics and select_cases parse
    for row_after, row_before in zip(after.progress, before.progress, strict=True):
        # Record tars hold their commit time; everything else, bytes included, is the same.
        assert _record_members(row_after["s2_record"]["path"]) == _record_members(row_before["s2_record"]["path"])


def _hm3d_env_class(
    root: Path,
    *,
    named: bool = True,
    loaded: bool = True,
    geodesic: float | None = None,
    rebake: bool = False,
    missing: bool = False,
):
    """test_s2_record_collection's FakeEnv with its two episodes moved into EXP-28's HM3D tree.

    ``named``: the episode file names the navmesh's sha256 (``info.navmesh_sha256``).
    """

    class HM3DEnv(harness.FakeEnv):
        instances: list = []

        def __init__(self, quaternion):
            super().__init__(quaternion)
            for episode, (_, scene) in zip(self.episodes, HM3D_SCENES):
                episode.scene_id, sha = _hm3d_scene(root, scene)
                if named:
                    episode.info = {**episode.info, "navmesh_sha256": sha}
                if rebake:
                    (root / "hm3d" / scene / f"{scene}.navmesh").write_bytes(b"baked again")
                if missing:
                    (root / "hm3d" / scene / f"{scene}.navmesh").unlink()
            self._sim.pathfinder.is_loaded = loaded
            HM3DEnv.instances.append(self)

        def _geodesic(self, position_a, position_b, episode=None):
            distance = super()._geodesic(position_a, position_b, episode)
            return distance if geodesic is None else geodesic

    return HM3DEnv


@pytest.mark.parametrize("named", [True, False])
def test_hm3d_records_carry_the_navmesh_and_the_log_parses(client, monkeypatch, tmp_path, capfd, named):
    root = tmp_path / "scene_datasets"
    monkeypatch.setattr(harness, "FakeEnv", _hm3d_env_class(root, named=named))
    run = harness._run(client, monkeypatch, tmp_path, mode="native", record=True, tag="hm3d")
    expected = [(scene, spec[1]) for (_, scene), spec in zip(HM3D_SCENES, harness.EPISODES)]
    assert [(row["scene_id"], row["episode_id"]) for row in run.progress] == expected
    records = client._load_s2_records_module()
    for row, (scene, episode_id) in zip(run.progress, expected):
        path = records.record_path(run.records, scene, episode_id)
        assert Path(row["s2_record"]["path"]).resolve() == path.resolve()
        record = records.EpisodeRecord(row["s2_record"]["path"], verify=True)
        navmesh = root / "hm3d" / scene / f"{scene}.navmesh"
        assert record.episode["navmesh_sha256"] == _sha(navmesh.read_bytes())
        assert set(record.episode) == MP3D_EPISODE_JSON_KEYS | {"navmesh_sha256"}
    headers = [m.group(3, 4) for m in map(pm.HEADER_RE.match, capfd.readouterr().out.splitlines()) if m]
    assert [(scene, int(episode_id)) for scene, episode_id in headers] == expected


@pytest.mark.parametrize(
    "broken, phrase",
    [
        ({"loaded": False}, "no navmesh loaded"),
        ({"geodesic": math.inf}, "start-to-goal geodesic is inf"),
        ({"rebake": True}, "but the episode file was built on"),
        ({"missing": True, "named": False}, "no navmesh file"),
    ],
)
def test_a_broken_hm3d_scene_stops_before_its_first_action(client, monkeypatch, tmp_path, broken, phrase):
    env_class = _hm3d_env_class(tmp_path / "scene_datasets", **broken)
    monkeypatch.setattr(harness, "FakeEnv", env_class)
    with pytest.raises(RuntimeError, match=phrase):
        harness._run(client, monkeypatch, tmp_path, mode="native", record=True, tag="broken")
    env = env_class.instances[-1]
    assert env.trace == [] and env.primitive == []
    assert not list((tmp_path / "records_broken").rglob("*.s2rec.tar"))

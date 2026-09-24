"""Select the EXP-18 render sets for tiers C, D and E and write collector configs.

Implements the pre-registered selection rules (docs/experiments/README.md, EXP-18):

  C  R2R_VLNCE_v1-3 val_unseen, 11 scenes.  Per scene, episodes sorted by
     sha1("<scene>:<episode_id>"); the first 30 are rendered (finalize keeps
     the first 25 with >= 20 frames).
  D  ScaleVLN subset_150k on HM3D train.  Candidate scenes appear in the
     episode file and have a .basis.navmesh that the pinned habitat-sim 0.1.7
     can load; sorted by sha1(<bare HM3D id>), first 30.  (178 of the 740
     hm3d/train navmeshes are Recast set version 2, written by a newer
     habitat-sim: PathFinder refuses them and the collector core-dumps at task
     init, so such a scene has no usable navmesh; the skipped ones are listed
     in the manifest.)  Per scene, first 6 episodes by
     sha1("<scene>:<episode_id>") are rendered (finalize keeps the first 4
     with >= 20 frames).
  E  Designed routes in the 11 C scenes, written as R2R-VLNCE episodes:
     * 2 out-and-back: the scene's first two C episodes (sha1 order), path =
       reference_path + reversed(reference_path)[1:], goal = start.
     * 2 loops from the start pose of the scene's first C episode (falling back
       to the next C episode in sha1 order when that start admits no loop, e.g.
       it lies on a staircase; recorded as loop_start_rank): 3 waypoints
       on the start floor (|dy| < 0.5 m); for EVERY pair of {start, w1, w2,
       w3} (the ledger's "两两"; 6 pairs, not only the route legs) geodesic in
       [3, 8] m and geodesic/euclidean <= 1.5; path = [start, w1, w2, w3,
       start].  numpy default_rng seeded with int(sha1(scene), 16) % 2**32;
       the two loops are drawn one after the other from the same generator.
       (LOOP_WAYPOINTS = 2 would give the triangle reading of "3 个导航点".)

The panoramic collector (VLN-CE ``collect/panoramic/collector.py``) walks
``episode.reference_path`` point by point with a ShortestPathFollower
(goal radius 0.2 m), so a designed route is followed in order with no
collector patch.  Its meta.json only carries scene_id, episode_id,
trajectory_id, instruction and reference_path, so the route pattern is encoded
in trajectory_id / instruction and the full route metadata goes to the
``E_routes.json`` sidecar (keyed by episode_id) and the selection manifest.

Outputs (``--out-dir``, default ``<RENDER_ROOT>/configs``):
  C.yaml, D.yaml, E.yaml      copies of vlnce_collect.yaml with DATASET set to
                              absolute paths (E also: more steps, SPL/SUCCESS
                              dropped because goal == start divides by zero)
  D_episodes.json.gz          the selected ScaleVLN episodes, scene_id rewritten
                              to hm3d/train/<dir>/<id>.basis.glb (the file's own
                              hm3d/<dir>/... does not resolve on disk)
  E_episodes.json.gz          the designed episodes
  E_routes.json               per-episode route metadata
  selection_manifest.json     every selected episode, sha1 key and rank, seeds

Runs under envs/vlnce (Python 3.8): stdlib + numpy + habitat_sim (PathFinder
on the navmesh only, no GL context).  Example::

    cd <repo> && /mnt/afs/liwenhao/agent/370910109/envs/vlnce/bin/python \\
        -m scripts.exp18.render.select_episodes
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import math
import re
import struct
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.exp18 import common  # noqa: E402
from scripts.exp18.render import layout  # noqa: E402

SCHEMA = "exp18-render-selection-v2"  # v2: loop rules on all pairs; D needs a loadable navmesh
BASE_CONFIG = common.VLNCE_PROJECT / "habitat_extensions" / "config" / "vlnce_collect.yaml"
SCENES_DIR = common.VLNCE_PROJECT / "data" / "scene_datasets"

C_SCENES = (
    "2azQ1b91cZZ", "8194nk5LbLH", "EU6Fwq7SyZv", "QUCTc6BB5sX", "TbHJrupSAjP", "X7HyMhZNoso",
    "Z6MFQCViBuw", "oLBMNvg9in8", "pLe4wQe7qrG", "x8F5xyUWy9e", "zsNo4HB9uLZ",
)
D_EPISODES_RENDERED_PER_SCENE = 6

LOOP_WAYPOINTS = 3
LOOP_GEODESIC_RANGE = (3.0, 8.0)  # every pair of {start, waypoints}
LOOP_MAX_DETOUR = 1.5             # geodesic/euclidean, every pair of {start, waypoints}
LOOP_MAX_DY = 0.5
LOOP_SNAP_TOLERANCE = 0.3  # max horizontal snap offset of a uniform (x, z) draw
LOOP_TRIES_PER_POINT = 1000
LOOP_MAX_RESTARTS = 200


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_gz(path: Path) -> dict:
    with gzip.open(str(path), "rt") as handle:
        return json.load(handle)


def _write_json_gz(path: Path, payload: dict) -> None:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
        handle.write(json.dumps(payload).encode("utf-8"))
    path.write_bytes(buffer.getvalue())


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _scene_of(scene_path: str) -> str:
    """'mp3d/X/X.glb' -> 'X'; 'hm3d/00796-m49/m49.basis.glb' -> 'm49' (bare id)."""
    name = scene_path.split("/")[-1]
    for suffix in (".glb", ".basis"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _ranked(scene: str, episodes: list) -> list:
    keyed = [(common.sha1_key(f"{scene}:{ep['episode_id']}"), ep) for ep in episodes]
    keyed.sort(key=lambda item: item[0])
    return keyed


def _render_config(base_text: str, *, split: str, data_path: Path, scenes, episode_ids,
                   title: str, max_episode_steps: int | None = None,
                   measurements: list | None = None) -> str:
    """Copy vlnce_collect.yaml, replacing its trailing DATASET block (and optionally
    ENVIRONMENT.MAX_EPISODE_STEPS / TASK.MEASUREMENTS).  Lists are one-line JSON."""
    lines = base_text.splitlines()
    dataset_at = [i for i, line in enumerate(lines) if line.rstrip() == "DATASET:"]
    if len(dataset_at) != 1:
        raise ValueError(f"{BASE_CONFIG}: expected one top-level DATASET block")
    tail = [line for line in lines[dataset_at[0] + 1:] if line.strip() and not line.lstrip().startswith("#")]
    if any(not line.startswith(" ") for line in tail):
        raise ValueError(f"{BASE_CONFIG}: DATASET is no longer the last top-level block")
    text = "\n".join(lines[: dataset_at[0]]) + "\n"
    if max_episode_steps is not None:
        text, count = re.subn(r"(?m)^(\s+)MAX_EPISODE_STEPS:\s*\d+\s*$",
                              r"\g<1>MAX_EPISODE_STEPS: %d" % max_episode_steps, text)
        if count != 1:
            raise ValueError("could not rewrite ENVIRONMENT.MAX_EPISODE_STEPS")
    if measurements is not None:
        text, count = re.subn(r"(?m)^(\s+)MEASUREMENTS:\s*\[[^\]]*\]",
                              r"\g<1>MEASUREMENTS: " + json.dumps(measurements).replace('"', ""), text)
        if count != 1:
            raise ValueError("could not rewrite TASK.MEASUREMENTS")
    header = [
        f"# EXP-18 {title}",
        f"# Generated by scripts/exp18/render/select_episodes.py from {BASE_CONFIG};",
        "# only DATASET"
        + (", ENVIRONMENT.MAX_EPISODE_STEPS" if max_episode_steps is not None else "")
        + (", TASK.MEASUREMENTS" if measurements is not None else "")
        + " differ.  Paths are absolute; run the collector with cwd=VLN-CE anyway.",
    ]
    dataset = [
        "DATASET:",
        "  TYPE: VLN-CE-v1",
        f"  SPLIT: {split}",
        f"  DATA_PATH: {json.dumps(str(data_path))}",
        f"  SCENES_DIR: {json.dumps(str(SCENES_DIR) + '/')}",
        f"  CONTENT_SCENES: {json.dumps(sorted(scenes))}",
        f"  EPISODES_ALLOWED: {json.dumps([str(e) for e in episode_ids])}",
    ]
    return "\n".join(header) + "\n" + text + "\n".join(dataset) + "\n"


# ----------------------------------------------------------------------------
# tier C
# ----------------------------------------------------------------------------
def select_c(val_unseen: dict, smoke: bool) -> "OrderedDict[str, dict]":
    by_scene = {}
    for ep in val_unseen["episodes"]:
        by_scene.setdefault(_scene_of(ep["scene_id"]), []).append(ep)
    if sorted(by_scene) != sorted(C_SCENES):
        raise ValueError(f"val_unseen scenes changed: {sorted(by_scene)}")
    per_scene = common.TIERS["C"]["episodes_rendered_per_scene"]
    selection = OrderedDict()
    for scene in sorted(by_scene):
        ranked = _ranked(scene, by_scene[scene])[:per_scene]
        rows = [{
            "episode_id": str(ep["episode_id"]),
            "sha1": key,
            "rank": rank,
            "trajectory_id": str(ep["trajectory_id"]),
            "num_reference_points": len(ep["reference_path"]),
            "_episode": ep,
        } for rank, (key, ep) in enumerate(ranked)]
        selection[scene] = {"available": len(by_scene[scene]), "episodes": rows}
    if smoke:
        first = next(iter(selection))
        selection = OrderedDict([(first, dict(selection[first], episodes=selection[first]["episodes"][:2]))])
    return selection


# ----------------------------------------------------------------------------
# tier D
# ----------------------------------------------------------------------------
def select_d(scalevln: dict, smoke: bool) -> tuple:
    by_scene, scene_dirs = {}, {}
    for ep in scalevln["episodes"]:
        parts = ep["scene_id"].split("/")  # hm3d/<NNNNN-id>/<id>.basis.glb
        bare = _scene_of(ep["scene_id"])
        by_scene.setdefault(bare, []).append(ep)
        scene_dirs.setdefault(bare, parts[-2])
    candidates = []
    for bare, hm3d_dir in scene_dirs.items():
        root = common.HM3D_SCENES / hm3d_dir
        if (root / f"{bare}.basis.navmesh").is_file() and (root / f"{bare}.basis.glb").is_file():
            candidates.append(bare)
    candidates.sort(key=common.sha1_key)
    num_scenes = common.TIERS["D"]["num_scenes"]
    chosen, skipped = [], []
    for candidate_rank, bare in enumerate(candidates):
        if len(chosen) == num_scenes:
            break
        navmesh = common.HM3D_SCENES / scene_dirs[bare] / f"{bare}.basis.navmesh"
        if _navmesh_loadable(navmesh):
            chosen.append((candidate_rank, bare))
        else:
            skipped.append({"scene": bare, "candidate_rank": candidate_rank, "navmesh": str(navmesh),
                            "recast_set_version": _recast_set_version(navmesh)})
    selection = OrderedDict()
    for scene_rank, (candidate_rank, bare) in enumerate(chosen):
        ranked = _ranked(bare, by_scene[bare])[:D_EPISODES_RENDERED_PER_SCENE]
        rows = [{
            "episode_id": str(ep["episode_id"]),
            "sha1": key,
            "rank": rank,
            "trajectory_id": str(ep["trajectory_id"]),
            "num_reference_points": len(ep["reference_path"]),
            "_episode": ep,
        } for rank, (key, ep) in enumerate(ranked)]
        selection[bare] = {
            "scene_sha1": common.sha1_key(bare),
            "scene_rank": scene_rank,
            "candidate_rank": candidate_rank,  # rank among scenes with a navmesh file
            "hm3d_dir": scene_dirs[bare],
            "render_scene_dir": f"{bare}.basis",  # collector names folders after the glb stem
            "glb": str(common.HM3D_SCENES / scene_dirs[bare] / f"{bare}.basis.glb"),
            "available": len(by_scene[bare]),
            "episodes": rows,
        }
    if smoke:
        first = next(iter(selection))
        selection = OrderedDict([(first, dict(selection[first], episodes=selection[first]["episodes"][:2]))])
    stats = {
        "scenes_in_episode_file": len(by_scene),
        "scenes_with_navmesh": len(candidates),
        "scenes_skipped_unloadable_navmesh": skipped,
    }
    return selection, stats


def _recast_set_version(navmesh: Path) -> int:
    with open(navmesh, "rb") as handle:
        return struct.unpack("<ii", handle.read(8))[1]  # (magic 'MSET', version)


def _navmesh_loadable(navmesh: Path) -> bool:
    """True if the pinned habitat-sim loads it (every Recast set v1 file does, no v2 file does)."""
    import habitat_sim

    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh(str(navmesh))
    return bool(pf.is_loaded)


# ----------------------------------------------------------------------------
# tier E
# ----------------------------------------------------------------------------
class _NavMesh:
    def __init__(self, navmesh: Path):
        import habitat_sim  # deferred: only E needs it

        self._habitat_sim = habitat_sim
        self.pf = habitat_sim.PathFinder()
        self.pf.load_nav_mesh(str(navmesh))
        if not self.pf.is_loaded:
            raise RuntimeError(f"failed to load navmesh {navmesh}")
        lower, upper = self.pf.get_bounds()
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)

    def geodesic(self, a, b) -> float:
        path = self._habitat_sim.ShortestPath()
        path.requested_start = np.asarray(a, dtype=np.float32)
        path.requested_end = np.asarray(b, dtype=np.float32)
        if not self.pf.find_path(path):
            return math.inf
        return float(path.geodesic_distance)

    def path_length(self, points) -> float:
        return float(sum(self.geodesic(a, b) for a, b in zip(points[:-1], points[1:])))

    def draw_floor_point(self, rng, floor_y: float):
        """Uniform (x, z) over the navmesh bounds, snapped at the start floor height."""
        x = rng.uniform(self.lower[0], self.upper[0])
        z = rng.uniform(self.lower[2], self.upper[2])
        snapped = np.asarray(self.pf.snap_point(np.array([x, floor_y, z], dtype=np.float32)), dtype=np.float64)
        if not np.all(np.isfinite(snapped)):
            return None
        if abs(snapped[1] - floor_y) >= LOOP_MAX_DY:
            return None
        if math.hypot(snapped[0] - x, snapped[2] - z) > LOOP_SNAP_TOLERANCE:
            return None
        if not self.pf.is_navigable(snapped.astype(np.float32)):
            return None
        return snapped


def _euclid(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _pair_ok(nav: _NavMesh, a, b) -> bool:
    lo, hi = LOOP_GEODESIC_RANGE
    geo = nav.geodesic(a, b)
    return lo <= geo <= hi and geo <= LOOP_MAX_DETOUR * _euclid(a, b)


def _sample_loop(nav: _NavMesh, rng, start) -> dict:
    """Sequential rejection sampling of the waypoints; restart from w1 if a later one cannot be placed.

    A candidate must satisfy the geodesic range and the detour ratio against the
    start AND every waypoint placed so far, so all pairs hold once the loop is complete.
    """
    start = np.asarray(start, dtype=np.float64)
    draws = 0
    for restart in range(LOOP_MAX_RESTARTS):
        points = [start]
        for _slot in range(LOOP_WAYPOINTS):
            placed = None
            for _ in range(LOOP_TRIES_PER_POINT):
                draws += 1
                cand = nav.draw_floor_point(rng, float(start[1]))
                if cand is not None and all(_pair_ok(nav, p, cand) for p in points):
                    placed = cand
                    break
            if placed is None:
                break
            points.append(placed)
        if len(points) == LOOP_WAYPOINTS + 1:
            path = points + [start]
            legs = [nav.geodesic(a, b) for a, b in zip(path[:-1], path[1:])]
            names = ["start"] + [f"w{i + 1}" for i in range(LOOP_WAYPOINTS)]
            pairs = [(i, j) for i in range(len(points)) for j in range(i + 1, len(points))]
            pair_geo = {(i, j): nav.geodesic(points[i], points[j]) for i, j in pairs}
            pair_ratio = {(i, j): pair_geo[i, j] / _euclid(points[i], points[j]) for i, j in pairs}
            return {
                "waypoints": [[float(v) for v in p] for p in points[1:]],
                "reference_path": [[float(v) for v in p] for p in path],
                "leg_geodesic_m": [round(v, 4) for v in legs],
                "leg_detour_ratio": [round(g / _euclid(a, b), 4) for g, a, b in zip(legs, path[:-1], path[1:])],
                "pairwise_geodesic_m": {f"{names[i]}-{names[j]}": round(pair_geo[i, j], 4) for i, j in pairs},
                "pairwise_detour_ratio": {f"{names[i]}-{names[j]}": round(pair_ratio[i, j], 4) for i, j in pairs},
                "route_length_m": round(float(sum(legs)), 4),
                "restarts": restart,
                "draws": draws,
            }
    raise RuntimeError(f"no loop found after {LOOP_MAX_RESTARTS} restarts ({draws} draws)")


def _designed_episode(*, episode_id, scene, pattern, source, reference_path, route_length, trajectory_id):
    return {
        "episode_id": episode_id,
        "trajectory_id": trajectory_id,
        "scene_id": f"mp3d/{scene}/{scene}.glb",
        "start_position": [float(v) for v in source["start_position"]],
        "start_rotation": [float(v) for v in source["start_rotation"]],
        # SPL reads DistanceToGoal, not info; kept for readers of the episode file.
        "info": {"geodesic_distance": 0.0, "exp18_pattern": pattern,
                 "exp18_route_length_m": route_length, "exp18_source_episode_id": str(source["episode_id"])},
        "goals": [{"position": [float(v) for v in source["start_position"]], "radius": 3.0}],
        "instruction": {"instruction_text": f"EXP-18 designed route: {pattern.replace('_', '-')}",
                        "instruction_tokens": []},
        "reference_path": [[float(v) for v in p] for p in reference_path],
    }


def build_e(c_full: "OrderedDict[str, dict]", smoke: bool):
    """c_full: the un-truncated C selection (E always uses each scene's first C episodes)."""
    n_oab = common.TIERS["E"]["out_and_back_per_scene"]
    n_loop = common.TIERS["E"]["loop_per_scene"]
    scenes = OrderedDict()
    episodes = []
    for scene, entry in c_full.items():
        nav = _NavMesh(common.MP3D_SCENES / scene / f"{scene}.navmesh")
        seed = int(common.sha1_key(scene), 16) % 2 ** 32
        rows = []
        for k in range(n_oab):
            src_row = entry["episodes"][k]
            src = src_row["_episode"]
            ref = [list(p) for p in src["reference_path"]]
            path = ref + ref[::-1][1:]
            length = round(nav.path_length(path), 4)
            eid = f"exp18E_{scene}_oab{k}"
            rows.append({
                "episode_id": eid, "pattern": "out_and_back", "index": k,
                "source_episode_id": src_row["episode_id"], "source_sha1": src_row["sha1"],
                "source_trajectory_id": src_row["trajectory_id"],
                "turnaround_index": len(ref) - 1, "reference_path": path, "route_length_m": length,
                "start_position": src["start_position"], "start_rotation": src["start_rotation"],
            })
            episodes.append(_designed_episode(
                episode_id=eid, scene=scene, pattern="out_and_back", source=src, reference_path=path,
                route_length=length, trajectory_id=f"exp18E|out_and_back|src={src_row['episode_id']}"))
        # Loop start: the first C episode (sha1 order) whose start admits both
        # loops; the generator is re-seeded per candidate, so a scene whose rank-0
        # start works gets exactly the rank-0 draw.  Rank-0 fails in scenes whose
        # first episode starts on a staircase or in a tiny floor pocket.
        loop_failures = []
        for src_row in entry["episodes"]:
            src = src_row["_episode"]
            rng = np.random.default_rng(seed)
            try:
                loops = [_sample_loop(nav, rng, src["start_position"]) for _ in range(n_loop)]
                break
            except RuntimeError as exc:
                loop_failures.append({"rank": src_row["rank"], "episode_id": src_row["episode_id"],
                                      "start_position": src["start_position"], "error": str(exc)})
        else:
            raise RuntimeError(f"{scene}: no C episode start admits {n_loop} loops")
        for k, loop in enumerate(loops):
            eid = f"exp18E_{scene}_loop{k}"
            rows.append(dict({
                "episode_id": eid, "pattern": "loop", "index": k,
                "source_episode_id": src_row["episode_id"], "source_sha1": src_row["sha1"],
                "source_rank": src_row["rank"],
                "seed": seed, "start_position": src["start_position"], "start_rotation": src["start_rotation"],
            }, **loop))
            episodes.append(_designed_episode(
                episode_id=eid, scene=scene, pattern="loop", source=src, reference_path=loop["reference_path"],
                route_length=loop["route_length_m"], trajectory_id=f"exp18E|loop|seed={seed}|k={k}"))
        oab_sources = [r["source_trajectory_id"] for r in rows if r["pattern"] == "out_and_back"]
        scenes[scene] = {
            "seed": seed,
            "loop_start_rank": src_row["rank"],
            "loop_start_failures": loop_failures,
            "out_and_back_same_trajectory": len(set(oab_sources)) < len(oab_sources),
            "episodes": rows,
        }
    if smoke:
        first = next(iter(scenes))
        keep = {f"exp18E_{first}_oab0", f"exp18E_{first}_loop0"}
        scenes = OrderedDict([(first, dict(scenes[first], episodes=[
            r for r in scenes[first]["episodes"] if r["episode_id"] in keep]))])
        episodes = [e for e in episodes if e["episode_id"] in keep]
    return scenes, episodes


# ----------------------------------------------------------------------------
def _strip_private(selection: "OrderedDict[str, dict]") -> "OrderedDict[str, dict]":
    out = OrderedDict()
    for scene, entry in selection.items():
        out[scene] = dict(entry, episodes=[{k: v for k, v in row.items() if not k.startswith("_")}
                                           for row in entry["episodes"]])
    return out


def _episode_ids(selection) -> list:
    return [row["episode_id"] for entry in selection.values() for row in entry["episodes"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=layout.CONFIG_DIR,
                        help="where configs + manifest go (default: %(default)s; env EXP18_RENDER_CONFIG_DIR)")
    parser.add_argument("--smoke", action="store_true",
                        help="dev smoke subset: C 2 eps of the first scene, D 2 eps of the first scene, "
                             "E out-and-back #0 + loop #0 of the first scene")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing manifest/config set (never do this after rendering started)")
    args = parser.parse_args()

    out_dir = args.out_dir
    manifest_file = layout.manifest_path(out_dir)
    if manifest_file.exists() and not args.overwrite:
        raise SystemExit(f"{manifest_file} exists; pass --overwrite to regenerate (only before rendering)")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    base_text = BASE_CONFIG.read_text()

    print(f"[C] loading {common.R2R_VAL_UNSEEN_EPISODES}", flush=True)
    val_unseen = _load_json_gz(common.R2R_VAL_UNSEEN_EPISODES)
    c_full = select_c(val_unseen, smoke=False)
    c_sel = select_c(val_unseen, smoke=args.smoke)

    print(f"[D] loading {common.SCALEVLN_EPISODES}", flush=True)
    scalevln = _load_json_gz(common.SCALEVLN_EPISODES)
    d_sel, d_stats = select_d(scalevln, smoke=args.smoke)

    print("[E] building designed routes", flush=True)
    e_scenes, e_episodes = build_e(c_full, smoke=args.smoke)

    # --- episode files ------------------------------------------------------
    d_episodes = []
    for bare, entry in d_sel.items():
        for row in entry["episodes"]:
            ep = copy.deepcopy(row["_episode"])
            ep["scene_id"] = f"hm3d/train/{entry['hm3d_dir']}/{bare}.basis.glb"
            d_episodes.append(ep)
    d_json = out_dir / "D_episodes.json.gz"
    e_json = out_dir / "E_episodes.json.gz"
    _write_json_gz(d_json, {"episodes": d_episodes, "instruction_vocab": scalevln["instruction_vocab"]})
    _write_json_gz(e_json, {"episodes": e_episodes, "instruction_vocab": val_unseen["instruction_vocab"]})
    routes = OrderedDict((row["episode_id"], dict(row, scene=scene))
                         for scene, entry in e_scenes.items() for row in entry["episodes"])
    _write_json(out_dir / "E_routes.json", routes)

    # --- configs --------------------------------------------------------------
    configs = {
        "C": _render_config(base_text, split="val_unseen", data_path=common.R2R_VAL_UNSEEN_EPISODES,
                            scenes=list(c_sel), episode_ids=_episode_ids(c_sel),
                            title="tier C (R2R val_unseen) render config"),
        "D": _render_config(base_text, split="train", data_path=d_json,
                            scenes=[entry["render_scene_dir"] for entry in d_sel.values()],
                            episode_ids=_episode_ids(d_sel),
                            title="tier D (HM3D via ScaleVLN subset_150k) render config"),
        "E": _render_config(base_text, split="val_unseen", data_path=e_json,
                            scenes=list(e_scenes), episode_ids=_episode_ids(e_scenes),
                            title="tier E (designed routes) render config",
                            max_episode_steps=layout.E_MAX_EPISODE_STEPS,
                            measurements=["DISTANCE_TO_GOAL"]),
    }
    for tier, text in configs.items():
        layout.config_path(tier, out_dir).write_text(text)

    manifest = OrderedDict([
        ("schema", SCHEMA),
        ("created_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        ("smoke", bool(args.smoke)),
        ("base_config", {"path": str(BASE_CONFIG), "sha256": hashlib.sha256(base_text.encode()).hexdigest()}),
        ("sources", {
            "C": {"path": str(common.R2R_VAL_UNSEEN_EPISODES), "sha256": _sha256(common.R2R_VAL_UNSEEN_EPISODES)},
            "D": {"path": str(common.SCALEVLN_EPISODES), "sha256": _sha256(common.SCALEVLN_EPISODES)},
        }),
        ("rules", {
            "sort_key": "sha1 hex of '<scene>:<episode_id>' (scene = bare id), ascending",
            "C": "per val_unseen scene, render first 30; keep first 25 with num_frames >= 20",
            "D": "scenes in ScaleVLN subset_150k whose .basis.navmesh loads in the pinned habitat-sim 0.1.7 "
                 "(Recast set v2 files do not: listed in tiers.D.scenes_skipped_unloadable_navmesh), sorted by "
                 "sha1(bare id), first 30; per scene render first 6, keep first 4 with num_frames >= 20",
            "E": "per C scene: 2 out-and-back (first two C episodes, ref + reversed(ref)[1:], goal = start) "
                 "+ 2 loops (start pose of the first C episode whose start admits both loops, sha1 order; "
                 "numpy default_rng(int(sha1(scene),16) % 2**32), re-seeded per candidate start; "
                 f"{LOOP_WAYPOINTS} waypoints |dy| < 0.5 m; for every pair of {{start, waypoints}}: geodesic "
                 f"in [3, 8] m and geodesic/euclidean <= 1.5; path = [start, w1..w{LOOP_WAYPOINTS}, start]; "
                 "sequential rejection, restart from w1 if a later waypoint fails); keep all with num_frames >= 20",
            "loop_sampler": {"waypoints": LOOP_WAYPOINTS, "pairs": "all pairs of {start, waypoints}",
                             "geodesic_range_m": LOOP_GEODESIC_RANGE, "max_detour": LOOP_MAX_DETOUR,
                             "max_dy_m": LOOP_MAX_DY, "snap_tolerance_m": LOOP_SNAP_TOLERANCE,
                             "tries_per_point": LOOP_TRIES_PER_POINT, "max_restarts": LOOP_MAX_RESTARTS,
                             "draw": "x,z ~ U(navmesh bounds), snap_point at start y"},
            "min_frames": layout.MIN_FRAMES,
        }),
        ("tiers", OrderedDict([
            ("C", {"config": "C.yaml", "data_path": str(common.R2R_VAL_UNSEEN_EPISODES),
                   "keep_per_scene": common.TIERS["C"]["clips_per_scene"],
                   "num_episodes": len(_episode_ids(c_sel)), "scenes": _strip_private(c_sel)}),
            ("D", dict({"config": "D.yaml", "data_path": str(d_json),
                        "keep_per_scene": common.TIERS["D"]["clips_per_scene"],
                        "num_episodes": len(_episode_ids(d_sel)), "scenes": _strip_private(d_sel)}, **d_stats)),
            ("E", {"config": "E.yaml", "data_path": str(e_json), "routes": "E_routes.json",
                   "keep_per_scene": None, "num_episodes": len(e_episodes), "scenes": e_scenes}),
        ])),
    ])
    _write_json(manifest_file, manifest)

    # --- summary --------------------------------------------------------------
    print(f"[C] {len(c_sel)} scenes, {manifest['tiers']['C']['num_episodes']} episodes: "
          + ", ".join(f"{s}={len(e['episodes'])}/{e['available']}" for s, e in c_sel.items()))
    print(f"[D] {d_stats['scenes_with_navmesh']}/{d_stats['scenes_in_episode_file']} scenes have a navmesh; "
          f"{len(d_sel)} scenes, {len(d_episodes)} episodes: " + ", ".join(d_sel))
    skipped = d_stats["scenes_skipped_unloadable_navmesh"]
    print(f"[D] skipped {len(skipped)} scenes whose navmesh habitat-sim cannot load: "
          + ", ".join(f"{r['scene']}(rank {r['candidate_rank']}, v{r['recast_set_version']})" for r in skipped))
    for scene, entry in e_scenes.items():
        parts = []
        for row in entry["episodes"]:
            extra = ""
            if row["pattern"] == "loop":
                worst = max(row["pairwise_detour_ratio"].values())
                extra = f" legs={row['leg_geodesic_m']} max_pair_ratio={worst:.3f} draws={row['draws']}"
            parts.append(f"{row['episode_id'].split('_')[-1]}={row['route_length_m']:.1f}m{extra}")
        flag = " SAME-TRAJECTORY out-and-backs" if entry["out_and_back_same_trajectory"] else ""
        if entry["loop_start_rank"]:
            flag += f" LOOP-START-FALLBACK rank={entry['loop_start_rank']}"
        print(f"[E] {scene} seed={entry['seed']}: " + "; ".join(parts) + flag)
    print(f"[done] wrote {out_dir} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()

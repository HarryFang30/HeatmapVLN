#!/usr/bin/env python3
"""EXP-28 V0: do the HM3D episodes, rendered the way the client renders them, show what InternData shows?

V0 (ledger EXP-28 A, fixed before any HM3D closed-loop result) on >= 10 training scenes, 40 episodes.  Each
episode of the VLN-CE file (StreamVLN's ``scalevln_subset_150k.json.gz``, or ``build_hm3d_episodes.py``'s
conversion of it) is paired with the InternData-N1 ScaleVLN episode of the same scene whose instruction is
the same after the client's normalisation (:func:`match_key`); then

(a) the front RGB at the json start pose, rendered with the client's own Habitat config
    (``scripts/evaluation/r2r_val_unseen.py`` ``build_habitat_config``: 640x480, HFOV 79, sensor at
    (0, 1.25, 0)), is compared with InternData's frame 0 (``interndata_replay._frame_path``,
    ``125cm_0deg``) by NCC on 160x120 grey; the same start position turned +90 deg (left) is the negative
    control.  Pass: median NCC >= 0.80, >= 90% of episodes >= 0.60, and NCC(correct) - NCC(+90 deg) >= 0.30
    on >= 90% of episodes;
(b) InternData's per-frame relative pose is mapped into Habitat,
    ``p_hab = p_start + R_yaw0 . (-y, z - 1.25, -x)``, and its heading is the start heading turned by the
    frame's yaw relative to frame 0.  Pass: >= 95% of all frames snap to the navmesh within 0.3 m, the last
    frame's median distance to the goal is <= 1.0 m, and the median NCC of the middle frames
    (``--mid-frames`` per episode, rendered at the mapped pose) is >= 0.70.
Fewer than 10 scenes or 40 episodes evaluated without error is a fail (``n_scenes`` / ``n_episodes``).

Coordinates.  Habitat: y up, the agent looks along -z, x right; a yaw about +y is a left turn when positive
(``s2_oracle_labels.yaw_to_quat_wxyz``); dataset quaternions are (x, y, z, w).  InternData-N1: z up, x
forward, y left, origin on the floor under the start camera, so frame 0's camera sits at (0, 0, 1.25).  That
is the convention ``tests/test_s2_oracle_labels.py::test_projection_reproduces_an_interndata_n1_goal_label``
pins down on a real r2r goal label: ``(x, y, z)_hab = (-y, z, -x)`` for a start at the origin facing -z, the
camera 1.25 m above its floor point.  The formula above is that convention with the start pose applied, and
``test_exp28_verify_hm3d_render`` reproduces the same label through :func:`interndata_to_habitat`.
InternData poses are read as camera-to-world; the yaw of ``R_t R_0^T`` about InternData's z is the frame's
heading relative to frame 0 whatever the camera axis convention, as long as the camera only yaws.  Reported
alongside, not judged: the yaw integrated from the actions (15 deg per turn) against the pose yaw, how far
``R_t R_0^T`` tilts off a pure yaw, frame 0's pose of the front column and of ``pose.125cm_30deg`` (the
ledger's T0 column: translation expected near (0, 0, 1.25)), the mapped trajectory's distance to the
reference path, and the last frame's distance to the goal under the other seven signed axis assignments
(``mapping_variants``) -- the first places to look when (b) fails.

Scenes.  A scene is keyed by its 11-character HM3D ID (``00xxx-<ID>`` dirs, ``<ID>.basis.glb`` and
``hm3d/<ID>/<ID>.glb`` all give ``<ID>``).  The mesh is EXP-28's tree ``<--scenes-dir>/hm3d/<ID>/<ID>.glb``
with its rebaked v1 navmesh ``<ID>.navmesh`` next to it -- what the collection loads; the navmesh is loaded
explicitly, its return value checked, and one that does not load fails the scene (never rebaked here, since
the collection would not rebake it either); its sha256 is compared with ``prepare_hm3d_scenes``' meta file.
A scene without a tree mesh is not drawn.  For diagnosis only, ``--glb-source raw`` (or ``auto`` for scenes
the tree lacks) loads the original ``<--hm3d-root>/train/00xxx-<ID>/<ID>.basis.glb`` with the tree's navmesh
when there is one, else its own; a navmesh that does not load there (HM3D v0.2's version-2 navmesh under
Habitat-Sim 0.1.7) is rebaked in process like the ScaleVLN fallback (``NavMeshSettings.set_defaults()``, the
client's AGENT_0 radius and height) unless ``--require-navmesh-file``.  The summary counts mesh and navmesh
sources and records every scene's navmesh with its sha256.  Training scenes are the scenes of both the json
and ``--interndata-root`` minus the 20 held out: ``--holdout-file`` (a JSON list, a JSON object with a
``heldout``/``holdout``/``heldout_scenes``/``holdout_scenes``/``scenes`` list, or one name per line, such as
``build_hm3d_episodes.py``'s ``splits/heldout_scenes.txt``; it must name 20 scenes, B3's five among them),
else the ledger's rule as that script applies it (B3's five plus the 15 smallest
``sha256("exp28-hm3d-heldout:" + dir name)`` among the InternData scene dirs the json annotates; the summary
says whether a given file agrees with it).  Scenes are taken in ``sha256(seed:ID)`` order, each with
``--episodes-per-scene`` matched episodes whose InternData frame 0 is on disk, until ``--num-scenes`` scenes.
Pairs are distinct InternData episodes; an instruction that two InternData episodes of a scene share is
ambiguous and skipped, and so is an InternData episode that json episodes of two different routes (start,
start rotation, goal) match.  A dry run resolves meshes too, so it selects what the rendered run will.

Rendering needs an X display with llvmpipe GLX, as the client's: ``--xvfb-display N`` starts a private Xvfb
on ``:N`` exactly as ``scripts/run_s2_dagger_collect_cuda.sh`` does (and stops it on exit); otherwise
``DISPLAY`` must be set.  ``--dry-run`` only matches, selects and computes the pose-only numbers (no Habitat).

Writes to ``--out-dir``: ``v0_summary.json`` (schema ``heatmapvln-exp28-v0-v1``: verdict per criterion,
quantiles, selection, match statistics over every json x InternData scene, per-scene navmesh provenance,
sensor spec, versions), ``v0_episodes.jsonl`` (one row per episode) and ``png/<ID>_<episode_id>.png``
(row 1: InternData frame 0 | render at the json start | render turned +90 deg; then one row per middle frame:
InternData frame t | render at the mapped pose).  Exit 0 when V0 passes (or after a dry run), 1 when it
does not, 2 when no verdict was reached (bad inputs, no display, any error before the summary).  An episode
whose RGB sensor is not at the requested pose (beyond 1e-3) is an error, not a score.

4090 container (fjl-habitat), CPU rendering::

    cd /workspace/HeatmapVLN && /opt/conda/bin/python scripts/exp28/verify_hm3d_render.py \\
        --episodes /workspace/exp28/data/scalevln_subset_150k.json.gz \\
        --holdout-file /workspace/exp28/episodes/splits/heldout_scenes.txt \\
        --interndata-root /workspace/exp23/data/interndata_n1/scalevln_train \\
        --scenes-dir /workspace/exp28/scene_datasets --hm3d-root /workspace/exp23/data/hm3d \\
        --out-dir /workspace/exp28/v0 --xvfb-display 395

(``--episodes`` may also be ``build_hm3d_episodes.py``'s ``pool.json.gz``: one instruction per route, already
filtered on the navmesh, so fewer matches and a test of the survivors only.)
"""

from __future__ import annotations

import argparse
import atexit
import collections
import dataclasses
import datetime
import gzip
import hashlib
import importlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools import label_s2_records as base  # noqa: E402  (numpy-only loader of src/data)

REC, ORL = base.REC, base.ORL
IR = importlib.import_module(ORL.__name__.rsplit(".", 1)[0] + ".interndata_replay")

SCHEMA = "heatmapvln-exp28-v0-v1"
SUMMARY_NAME = "v0_summary.json"
ROWS_NAME = "v0_episodes.jsonl"

# Ledger EXP-28 A / V0, written before any HM3D closed-loop result.  Not options.
NCC_SIZE = (160, 120)  # (width, height) of the grey images NCC compares
START_NCC_FLOOR = 0.60
YAW90_MARGIN = 0.30
SNAP_MAX_M = 0.3
NEGATIVE_YAW_DEG = 90.0
SENSOR_POSE_TOL = 1e-3  # m / rotation-matrix entry: the sensor must sit where the render was asked for
HELDOUT_COUNT = 20
V0_CRITERIA: dict[str, tuple[str, float]] = {
    "a1_start_ncc_median": (">=", 0.80),
    "a2_start_ncc_ge_0.60_share": (">=", 0.90),
    "a3_yaw90_margin_ge_0.30_share": (">=", 0.90),
    "b1_snap_le_0.3m_frame_share": (">=", 0.95),
    "b2_end_to_goal_median_m": ("<=", 1.0),
    "b3_mid_ncc_median": (">=", 0.70),
    "n_scenes": (">=", 10),
    "n_episodes": (">=", 40),
}
RENDER_CRITERIA = ("a1_start_ncc_median", "a2_start_ncc_ge_0.60_share", "a3_yaw90_margin_ge_0.30_share",
                   "b1_snap_le_0.3m_frame_share", "b3_mid_ncc_median")

# Ledger EXP-28 设置 / 场景划分.
B3_SCENE_DIRS = (
    "00481-5jLhtVmWd5F", "00393-dD37vuCa4FE", "00396-wuKkTq5GJbi", "00560-gjhYih4upQ9", "00770-NBg5UqG3di3",
)
HELDOUT_SALT = "exp28-hm3d-heldout:"
HASHED_HELDOUT = 15

CAMERA_HEIGHT_M = float(REC.CAMERA["height_m"])  # 1.25, the client's RGB_SENSOR.POSITION[1]
TURN_DEG = 15.0  # the client's SIMULATOR.TURN_ANGLE; InternData's turn actions
TURN_LEFT, TURN_RIGHT = 2, 3  # InternData action ids (interndata_replay.IDX2ACTION)
TILE = (320, 240)
QUANTILES = (5, 25, 50, 75, 95)

DEFAULT_INTERNDATA_ROOT = "/workspace/exp23/data/interndata_n1/scalevln_train"
DEFAULT_SCENES_DIR = "/workspace/exp28/scene_datasets"
DEFAULT_HM3D_ROOT = "/workspace/exp23/data/hm3d"
DEFAULT_MP3D_ROOT = "/dataset/mp3d"  # the client's GL pre-initialisation scene (as the 4090 launcher)

CONVENTIONS = (
    "Habitat: y up, agent looks along -z, x right, yaw about +y left-positive, dataset quaternions (x, y, z, w). "
    "InternData-N1: z up, x forward, y left, origin on the floor under the start camera (frame 0 camera at "
    "(0, 0, 1.25)); poses camera-to-world. Floor point of frame t: p_hab = p_start + R(q_start) . "
    "(-y_t, z_t - 1.25, -x_t). Heading of frame t: q_start * q_yaw(theta_t), theta_t = yaw of R_t R_0^T about "
    "InternData z (left positive). Renders put the agent body at the floor point (sensor 1.25 m above), "
    "sensors reset to level."
)


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #
def hm3d_id(name: str) -> str:
    """The 11-character HM3D ID of a scene path, glb name or ``00xxx-<ID>`` directory name."""
    text = str(name).replace("\\", "/").rstrip("/")
    last = text.split("/")[-1]
    if last.endswith(".glb") or last.endswith(".navmesh"):
        stem = last.split(".")[0]
    else:
        stem = last
    prefix, sep, rest = stem.partition("-")
    if sep and prefix.isdigit() and rest:
        return rest
    return stem


def match_key(text: str) -> str:
    """Scene-local instruction key: whitespace collapsed, the client's final ``.``/``!``/``?`` dropped, lower case.

    ``interndata_replay.client_instruction`` is the client's own normalisation; collapsing whitespace first
    keeps a trailing space (R2R-style ``". "``) from hiding the final period from it.
    """
    return IR.client_instruction(" ".join(str(text).split())).lower()


def _sha_key(*parts: Any) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | os.PathLike) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _drop_instruction_tokens(obj: dict) -> dict:
    # 155k episodes x ~200 token ids are never read here (as build_hm3d_episodes.load_json_gz drops them).
    if "instruction_text" in obj:
        obj.pop("instruction_tokens", None)
    return obj


def load_vlnce_episodes(path: str | os.PathLike) -> list[dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle, object_hook=_drop_instruction_tokens)
    episodes = data["episodes"] if isinstance(data, Mapping) else data
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{path}: no episodes")
    return episodes


def episode_instruction(episode: Mapping[str, Any]) -> str:
    instruction = episode.get("instruction")
    if isinstance(instruction, Mapping):
        return str(instruction.get("instruction_text", ""))
    return str(instruction or "")


def group_by_scene(episodes: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for episode in episodes:
        out[hm3d_id(episode["scene_id"])].append(episode)
    return dict(out)


@dataclasses.dataclass(frozen=True)
class InternEpisode:
    scene_dir: Path
    episode_index: int
    length: int
    instructions: tuple[str, ...]


def interndata_scene_dirs(roots: Sequence[str | os.PathLike]) -> dict[str, Path]:
    """HM3D ID -> InternData scene dir (``<dir>/meta/episodes.jsonl`` present); the first root wins."""
    out: dict[str, Path] = {}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"InternData root not found: {root}")
        for scene_dir in sorted(root.iterdir()):
            if (scene_dir / "meta" / "episodes.jsonl").is_file():
                out.setdefault(hm3d_id(scene_dir.name), scene_dir)
    return out


def read_interndata_scene(scene_dir: str | os.PathLike) -> list[InternEpisode]:
    scene_dir = Path(scene_dir)
    out = []
    with open(scene_dir / "meta" / "episodes.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            ep = json.loads(line)
            out.append(
                InternEpisode(
                    scene_dir=scene_dir,
                    episode_index=int(ep["episode_index"]),
                    length=int(ep["length"]),
                    instructions=tuple(str(ep["tasks"][0]).split(IR.INSTRUCTION_SEP)),
                )
            )
    return out


def load_interndata_track(scene_dir: str | os.PathLike, episode_index: int) -> dict[str, Any]:
    """Parquet actions and 4x4 front-camera poses of one InternData episode."""
    import pyarrow.parquet as pq

    path = Path(scene_dir) / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
    names = set(pq.read_schema(path).names)
    column = next((f"pose.{s}" for s in (IR.FRONT_SETTING, IR.LOOKDOWN_SETTING) if f"pose.{s}" in names), None)
    if column is None:
        raise KeyError(f"{path}: no pose.{IR.FRONT_SETTING} / pose.{IR.LOOKDOWN_SETTING} column")
    # The ledger's T0 reads frame 0 of the lookdown column; it is read along for the report only.
    lookdown = f"pose.{IR.LOOKDOWN_SETTING}"
    extra = [lookdown] if lookdown in names and lookdown != column else []
    table = pq.read_table(path, columns=["action", column, *extra]).to_pydict()
    poses = np.asarray([np.asarray(m, dtype=np.float64).reshape(4, 4) for m in table[column]])
    actions = [int(a) for a in table["action"]]
    if len(actions) != len(poses):
        raise ValueError(f"{path}: {len(actions)} actions for {len(poses)} poses")
    out = {"actions": actions, "poses": poses, "pose_column": column, "parquet": str(path)}
    if extra and table[lookdown]:
        out["lookdown_pose0"] = np.asarray(table[lookdown][0], dtype=np.float64).reshape(4, 4)
    return out


def interndata_frame(scene_dir: str | os.PathLike, episode_index: int, frame_id: int) -> Image.Image:
    path = IR._frame_path(Path(scene_dir), IR.FRONT_SETTING, int(episode_index), int(frame_id))
    if not path.is_file():
        raise FileNotFoundError(f"InternData frame not found: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


# --------------------------------------------------------------------------- #
# Held-out scenes and selection
# --------------------------------------------------------------------------- #
def heldout_by_rule(scene_dir_names: Iterable[str]) -> list[str]:
    """Ledger rule: B3's five plus the 15 smallest sha256("exp28-hm3d-heldout:" + dir name) of the others."""
    b3 = {hm3d_id(d) for d in B3_SCENE_DIRS}
    pool = sorted({str(d) for d in scene_dir_names if hm3d_id(d) not in b3})
    hashed = sorted(pool, key=lambda d: hashlib.sha256((HELDOUT_SALT + d).encode("utf-8")).hexdigest())
    return sorted(B3_SCENE_DIRS) + hashed[:HASHED_HELDOUT]


def read_holdout_file(path: str | os.PathLike) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if isinstance(data, Mapping):
        for key in ("heldout", "holdout", "heldout_scenes", "holdout_scenes", "scenes"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            raise ValueError(f"{path}: no held-out scene list")
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of scenes")
    return [str(item) for item in data]


def check_heldout(names: Sequence[str], source: Any) -> None:
    """The ledger holds out exactly 20 scenes, B3's five among them (e.g. not train_scenes.txt by mistake)."""
    ids = [hm3d_id(n) for n in names]
    missing_b3 = sorted({hm3d_id(d) for d in B3_SCENE_DIRS} - set(ids))
    if len(set(ids)) != HELDOUT_COUNT or len(ids) != len(set(ids)) or missing_b3:
        raise ValueError(f"{source}: {len(ids)} entries ({len(set(ids))} distinct), B3 missing {missing_b3}; "
                         f"expected the {HELDOUT_COUNT} held-out scenes of build_hm3d_episodes.py")


@dataclasses.dataclass(frozen=True)
class Pair:
    scene: str  # HM3D ID
    episode: Mapping[str, Any]
    intern: InternEpisode
    exact_client_text: bool  # client_instruction(json) == client_instruction(InternData), byte for byte


def route_signature(episode: Mapping[str, Any]) -> tuple:
    """Start position, start rotation and goal at 1 cm / 1e-3: the json episodes of one route share it."""
    return (
        tuple(round(float(v), 2) for v in episode["start_position"]),
        tuple(round(float(v), 3) for v in episode["start_rotation"]),
        tuple(round(float(v), 2) for v in episode["goals"][0]["position"]),
    )


def match_scene(
    json_episodes: Sequence[Mapping[str, Any]], intern_episodes: Sequence[InternEpisode], *, seed: str, scene: str
) -> tuple[list[Pair], dict[str, int]]:
    """Unique (json episode, InternData episode) pairs of one scene, in ``sha256(seed:scene:episode_id)`` order.

    A key two InternData episodes share is ambiguous; a json episode matching it is dropped.  An InternData
    episode that json episodes of two different routes (:func:`route_signature`) match -- through one shared
    instruction or through its two different ones -- is a route conflict: nothing says which route InternData
    walked, so none of them is paired.  An InternData episode matched by several json episodes of one route is
    paired once (the first in hash order).
    """
    owners: dict[str, set[int]] = collections.defaultdict(set)
    texts: dict[tuple[str, int], str] = {}
    by_index = {ep.episode_index: ep for ep in intern_episodes}
    for ep in intern_episodes:
        for text in ep.instructions:
            key = match_key(text)
            owners[key].add(ep.episode_index)
            texts.setdefault((key, ep.episode_index), text)
    stats = collections.Counter(json_episodes=len(json_episodes), interndata_episodes=len(intern_episodes),
                                interndata_instructions=sum(len(ep.instructions) for ep in intern_episodes))
    ordered = sorted(json_episodes, key=lambda e: _sha_key(seed, scene, e["episode_id"]))
    candidates: list[tuple[Mapping[str, Any], str, int]] = []
    routes_of: dict[int, set[tuple]] = collections.defaultdict(set)
    for episode in ordered:
        key = match_key(episode_instruction(episode))
        found = owners.get(key, set())
        if not found:
            stats["unmatched"] += 1
            continue
        if len(found) > 1:
            stats["ambiguous"] += 1
            continue
        index = next(iter(found))
        candidates.append((episode, key, index))
        routes_of[index].add(route_signature(episode))
    pairs, used = [], set()
    for episode, key, index in candidates:
        stats["matched"] += 1
        if len(routes_of[index]) > 1:
            stats["route_conflict"] += 1
            continue
        if index in used:
            stats["interndata_episode_reused"] += 1
            continue
        used.add(index)
        exact = IR.client_instruction(episode_instruction(episode)) == IR.client_instruction(texts[(key, index)])
        stats["exact_client_text"] += int(exact)
        pairs.append(Pair(scene=scene, episode=episode, intern=by_index[index], exact_client_text=exact))
    stats["pairs"] = len(pairs)
    return pairs, dict(stats)


def resolve_scene(
    scene_id: str, scene: str, *, scenes_dir: str | os.PathLike, hm3d_root: str | os.PathLike, glb_source: str
) -> tuple[Path, Path | None, str]:
    """(mesh to load, navmesh to load or None for the mesh's own ``<stem>.navmesh``, "tree" | "raw").

    "tree" is what the collection loads: ``hm3d/<ID>/<ID>.glb`` with the rebaked ``hm3d/<ID>/<ID>.navmesh``.
    """
    tree_glb = Path(scenes_dir) / "hm3d" / scene / f"{scene}.glb"
    tree_navmesh = tree_glb.with_suffix(".navmesh")
    if glb_source in ("auto", "tree") and tree_glb.is_file():
        return tree_glb, tree_navmesh, "tree"
    if glb_source == "tree":
        raise FileNotFoundError(f"scene tree mesh not found: {tree_glb}")
    relative = str(scene_id)
    for prefix in ("data/scene_datasets/", "scene_datasets/"):
        if relative.startswith(prefix):
            relative = relative[len(prefix):]
    # The original first; the tree's own symlink (a pool.json.gz scene_id) is not "raw".
    candidates: list[Path] = []
    root = Path(hm3d_root)
    patterns = (f"train/*-{scene}/{scene}.basis.glb", f"*-{scene}/{scene}.basis.glb", f"train/*-{scene}/{scene}.glb")
    for pattern in patterns:
        candidates += sorted(root.glob(pattern)) if root.is_dir() else []
    candidates.append(Path(scenes_dir) / relative)
    if relative.startswith("hm3d/") and not relative.startswith("hm3d/train/"):
        candidates.append(Path(scenes_dir) / ("hm3d/train/" + relative[len("hm3d/"):]))
    for glb in candidates:
        if glb.is_file() and hm3d_id(glb.name) == scene and os.path.abspath(glb) != os.path.abspath(tree_glb):
            return glb, (tree_navmesh if tree_navmesh.is_file() else None), "raw"
    raise FileNotFoundError(f"no mesh for scene {scene} ({scene_id}) under {scenes_dir} or {hm3d_root}")


# --------------------------------------------------------------------------- #
# Geometry (numpy only)
# --------------------------------------------------------------------------- #
def xyzw_to_wxyz(q: Sequence[float]) -> list[float]:
    x, y, z, w = [float(v) for v in q]
    return [w, x, y, z]


def quat_mul_wxyz(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    aw, ax, ay, az = [float(v) for v in a]
    bw, bx, by, bz = [float(v) for v in b]
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def heading_rotation_wxyz(start_rotation_xyzw: Sequence[float], yaw_rad: float) -> np.ndarray:
    """The start body rotation turned by ``yaw_rad`` about its up axis (left positive)."""
    q = quat_mul_wxyz(xyzw_to_wxyz(start_rotation_xyzw), ORL.yaw_to_quat_wxyz(float(yaw_rad)))
    return q / np.linalg.norm(q)


def yaw_of_wxyz(q: Sequence[float]) -> float:
    """Heading about +y of a body rotation (0 faces -z, left positive)."""
    forward = ORL.quat_wxyz_to_rot(q) @ np.array([0.0, 0.0, -1.0])
    return math.atan2(-forward[0], -forward[2])


def wrap_rad(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + math.pi) % (2.0 * math.pi) - math.pi


# Signed assignments of InternData's horizontal axes to Habitat's local (right, back) = (X, Z).  "design" is the
# ledger's (-y, -x); the other seven are reported as diagnostics only.
MAPPING_VARIANTS: dict[str, tuple[tuple[int, float], tuple[int, float]]] = {
    "design(-y,-x)": ((1, -1.0), (0, -1.0)),
    "(+y,-x)": ((1, 1.0), (0, -1.0)),
    "(-y,+x)": ((1, -1.0), (0, 1.0)),
    "(+y,+x)": ((1, 1.0), (0, 1.0)),
    "(-x,-y)": ((0, -1.0), (1, -1.0)),
    "(+x,-y)": ((0, 1.0), (1, -1.0)),
    "(-x,+y)": ((0, -1.0), (1, 1.0)),
    "(+x,+y)": ((0, 1.0), (1, 1.0)),
}


def interndata_to_habitat(
    translations: np.ndarray,
    start_position: Sequence[float],
    start_rotation_xyzw: Sequence[float],
    camera_height: float = CAMERA_HEIGHT_M,
    variant: str = "design(-y,-x)",
) -> np.ndarray:
    """InternData camera translations [T, 3] -> Habitat floor points [T, 3]: p_start + R . (-y, z - h, -x)."""
    t = np.asarray(translations, dtype=np.float64).reshape(-1, 3)
    (ax, sx), (az, sz) = MAPPING_VARIANTS[variant]
    local = np.stack([sx * t[:, ax], t[:, 2] - camera_height, sz * t[:, az]], axis=-1)
    rotation = ORL.quat_wxyz_to_rot(xyzw_to_wxyz(start_rotation_xyzw))
    return np.asarray(start_position, dtype=np.float64) + local @ rotation.T


def habitat_to_interndata(
    points: np.ndarray,
    start_position: Sequence[float],
    start_rotation_xyzw: Sequence[float],
    camera_height: float = CAMERA_HEIGHT_M,
) -> np.ndarray:
    """Inverse of :func:`interndata_to_habitat` (design mapping): Habitat floor points -> InternData cameras."""
    rotation = ORL.quat_wxyz_to_rot(xyzw_to_wxyz(start_rotation_xyzw))
    offsets = np.asarray(points, dtype=np.float64).reshape(-1, 3) - np.asarray(start_position, dtype=np.float64)
    local = offsets @ rotation
    return np.stack([-local[:, 2], -local[:, 0], local[:, 1] + camera_height], axis=-1)


def relative_yaws(rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(yaw of R_t R_0^T about z [rad, left positive], its tilt off a pure z-rotation [deg]) per frame."""
    r = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    m = r @ r[0].T
    yaw = np.arctan2(m[:, 1, 0], m[:, 0, 0])
    tilt = np.degrees(np.arccos(np.clip(m[:, 2, 2], -1.0, 1.0)))
    return yaw, tilt


def action_yaws(parquet_actions: Sequence[int], turn_deg: float = TURN_DEG) -> np.ndarray:
    """Heading of every frame relative to frame 0 [rad] from the parquet actions (action t led to frame t)."""
    actions = np.asarray([int(a) for a in parquet_actions][1:], dtype=np.int64)
    steps = (actions == TURN_LEFT).astype(np.float64) - (actions == TURN_RIGHT).astype(np.float64)
    return np.radians(turn_deg) * np.concatenate([[0.0], np.cumsum(steps)])


def point_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance of every point [N, 3] to the polyline [M, 3]."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    line = np.asarray(polyline, dtype=np.float64).reshape(-1, 3)
    if len(line) == 1:
        return np.linalg.norm(p - line[0], axis=-1)
    a, b = line[:-1], line[1:]
    ab = b - a
    denom = np.maximum((ab * ab).sum(-1), 1e-12)
    t = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(-1) / denom[None], 0.0, 1.0)
    nearest = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(p[:, None, :] - nearest, axis=-1).min(axis=1)


def mid_frame_ids(length: int, k: int) -> list[int]:
    """``k`` frames evenly spaced strictly between the first and the last."""
    if length < 3 or k <= 0:
        return []
    ids = np.unique(np.round(np.linspace(0, length - 1, k + 2)).astype(int))
    return [int(i) for i in ids if 0 < i < length - 1]


def camera_c2w(
    position: Sequence[float], rotation_wxyz: Sequence[float], height: float = CAMERA_HEIGHT_M
) -> np.ndarray:
    """The level RGB sensor's camera-to-world: [R(q) | p + R(q) (0, h, 0)]."""
    rotation = ORL.quat_wxyz_to_rot(rotation_wxyz)
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = np.asarray(position, dtype=np.float64) + rotation @ np.array([0.0, height, 0.0])
    return out


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
_BOX = getattr(Image, "Resampling", Image).BOX


def _pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., :3]
    return Image.fromarray(np.ascontiguousarray(array.astype(np.uint8)))


def grey_small(image: Any, size: tuple[int, int] = NCC_SIZE) -> np.ndarray:
    """Grey (ITU-R 601 luma), area-resampled to ``size`` (width, height), as float64."""
    return np.asarray(_pil(image).convert("L").resize(size, _BOX), dtype=np.float64)


def ncc(a: np.ndarray, b: np.ndarray) -> float | None:
    """Zero-mean normalised cross-correlation of two equal-shape images; None when either is flat."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else None


def image_ncc(a: Any, b: Any) -> float | None:
    return ncc(grey_small(a), grey_small(b))


def write_side_by_side(path: Path, rows: Sequence[Sequence[tuple[str, Any]]]) -> None:
    """Tiles of TILE size, one list of (label, image or None) per row."""
    width, height = TILE
    cols = max(len(row) for row in rows)
    canvas = Image.new("RGB", (cols * width, len(rows) * height), (32, 32, 32))
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        for j, (label, image) in enumerate(row):
            x, y = j * width, i * height
            if image is not None:
                canvas.paste(_pil(image).convert("RGB").resize(TILE, _BOX), (x, y))
            draw.rectangle([x, y, x + 7 * len(label) + 6, y + 14], fill=(0, 0, 0))
            draw.text((x + 3, y + 2), label, fill=(255, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    canvas.save(tmp, format="PNG")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# One episode
# --------------------------------------------------------------------------- #
def _r(value: Any, digits: int = 4) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_r(v, digits) for v in np.asarray(value, dtype=np.float64).tolist()]
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def episode_geometry(pair: Pair, track: Mapping[str, Any], *, heading_source: str = "pose") -> dict[str, Any]:
    """Pose-only numbers of one pair (no renderer): mapping, headings, T0 view of frame 0, diagnostics."""
    ep = pair.episode
    poses = np.asarray(track["poses"], dtype=np.float64)
    start, start_rot = ep["start_position"], ep["start_rotation"]
    goal = np.asarray(ep["goals"][0]["position"], dtype=np.float64)
    reference = np.asarray(ep.get("reference_path") or [start, goal], dtype=np.float64)
    mapped = interndata_to_habitat(poses[:, :3, 3], start, start_rot)
    yaw_pose, tilt = relative_yaws(poses[:, :3, :3])
    yaw_act = action_yaws(track["actions"])
    if len(yaw_act) != len(yaw_pose):
        raise ValueError(f"{len(yaw_act)} action headings for {len(yaw_pose)} poses")
    yaws = yaw_pose if heading_source == "pose" else yaw_act
    heading_diff = np.degrees(np.abs(wrap_rad(yaw_pose - yaw_act)))
    ref_dev = point_to_polyline(mapped, reference)
    variants = {}
    for name in MAPPING_VARIANTS:
        end = interndata_to_habitat(poses[-1:, :3, 3], start, start_rot, variant=name)[0]
        variants[name] = _r(np.linalg.norm(end - goal), 3)
    t0 = poses[0, :3, 3]
    lookdown0 = track.get("lookdown_pose0")
    return {
        "mapped": mapped,
        "yaws": yaws,
        "geometry": {
            "num_frames": int(len(poses)),
            "pose_column": track.get("pose_column"),
            "frame0_translation": _r(t0),
            "frame0_translation_dev_m": _r(np.linalg.norm(t0 - np.array([0.0, 0.0, CAMERA_HEIGHT_M])), 4),
            "frame0_rotation_identity_err": _r(np.abs(poses[0, :3, :3] - np.eye(3)).max(), 4),
            # The ledger's T0 column (pose.125cm_30deg[0]): translation and rotation, as read.
            "lookdown_frame0_translation": None if lookdown0 is None else _r(lookdown0[:3, 3]),
            "lookdown_frame0_rotation": None if lookdown0 is None else [_r(r) for r in lookdown0[:3, :3]],
            "start_rotation_yaw_only_err": _r(np.abs(ORL.quat_wxyz_to_rot(xyzw_to_wxyz(start_rot))[:, 1]
                                                     - np.array([0.0, 1.0, 0.0])).max(), 5),
            "end_to_goal_m": _r(np.linalg.norm(mapped[-1] - goal), 6),
            "end_to_goal_xz_m": _r(np.linalg.norm((mapped[-1] - goal)[[0, 2]]), 4),
            "end_dy_m": _r((mapped[-1] - goal)[1], 4),
            "ref_path_dev_m": {"p50": _r(np.median(ref_dev), 3), "max": _r(ref_dev.max(), 3)},
            "heading_pose_vs_actions_deg": {"p50": _r(np.median(heading_diff), 3), "max": _r(heading_diff.max(), 3)},
            "pose_tilt_deg_max": _r(tilt.max(), 3),
            "final_yaw_pose_deg": _r(math.degrees(yaw_pose[-1]), 2),
            "final_yaw_actions_deg": _r(math.degrees(yaw_act[-1]), 2),
            "mapping_variants_end_to_goal_m": variants,
            "info_geodesic_distance": (ep.get("info") or {}).get("geodesic_distance"),
        },
    }


def verify_episode(
    pair: Pair,
    renderer: Any | None,
    *,
    png_dir: Path | None,
    mid_frames: int,
    heading_source: str = "pose",
) -> dict[str, Any]:
    """One row: pose-only geometry, plus snap / NCC / PNG when a renderer is given."""
    t_start = time.time()
    ep, intern = pair.episode, pair.intern
    row: dict[str, Any] = {
        "scene": pair.scene,
        "interndata_scene_dir": intern.scene_dir.name,
        "episode_id": ep["episode_id"],
        "trajectory_id": ep.get("trajectory_id"),
        "interndata_episode_index": intern.episode_index,
        "instruction": episode_instruction(ep)[:160],
        "exact_client_text": pair.exact_client_text,
    }
    track = load_interndata_track(intern.scene_dir, intern.episode_index)
    if len(track["poses"]) != intern.length:
        raise ValueError(f"parquet has {len(track['poses'])} frames, episodes.jsonl says {intern.length}")
    geo = episode_geometry(pair, track, heading_source=heading_source)
    row.update(geo["geometry"])
    if renderer is None:
        row["seconds"] = round(time.time() - t_start, 2)
        return row

    mapped, yaws = geo["mapped"], geo["yaws"]
    start, start_rot = ep["start_position"], ep["start_rotation"]
    goal = np.asarray(ep["goals"][0]["position"], dtype=np.float64)
    snaps = []
    for point in mapped:
        snapped = renderer.snap(point)
        snaps.append(math.inf if snapped is None else float(np.linalg.norm(np.asarray(snapped) - point)))
    row["snap_m"] = [_r(s, 5) for s in snaps]  # None = no navmesh point found (counts as > 0.3 m)
    row["snap_le_0.3m_share"] = _r(np.mean([s <= SNAP_MAX_M for s in snaps]), 4)
    row["end_geodesic_to_goal_m"] = _r(renderer.geodesic(mapped[-1], goal), 4)

    pose_errs = []

    def render(position, rotation_wxyz):
        image = renderer.render(position, rotation_wxyz)
        err = getattr(renderer, "last_pose_err", None)
        if err is not None:
            pose_errs.append(float(err))
        return image

    frame0 = interndata_frame(intern.scene_dir, intern.episode_index, 0)
    row["interndata_frame_size"] = list(frame0.size)
    start_wxyz = heading_rotation_wxyz(start_rot, 0.0)
    ours = render(start, start_wxyz)
    turned = render(start, heading_rotation_wxyz(start_rot, math.radians(NEGATIVE_YAW_DEG)))
    row["render_size"] = [int(np.asarray(ours).shape[1]), int(np.asarray(ours).shape[0])]
    start_ncc, yaw90_ncc = image_ncc(ours, frame0), image_ncc(turned, frame0)
    row["start_ncc"] = _r(start_ncc, 6)
    row["yaw90_ncc"] = _r(yaw90_ncc, 6)
    # A flat render has no NCC: a flat start counts as -1 (evaluate_v0), a flat +90 deg render leaves no
    # margin to show (None, counted as failing a3).
    row["start_margin"] = None if start_ncc is None or yaw90_ncc is None else _r(start_ncc - yaw90_ncc, 6)
    tiles = [[(f"InternData f0  ep{intern.episode_index}", frame0),
              (f"habitat start  ncc {row['start_ncc']}", ours),
              (f"habitat +90deg  ncc {row['yaw90_ncc']}", turned)]]
    mids = []
    for frame_id in mid_frame_ids(intern.length, mid_frames):
        theirs = interndata_frame(intern.scene_dir, intern.episode_index, frame_id)
        image = render(mapped[frame_id], heading_rotation_wxyz(start_rot, yaws[frame_id]))
        value = _r(image_ncc(image, theirs), 6)
        mids.append({"frame": frame_id, "ncc": value, "snap_m": _r(snaps[frame_id], 5),
                     "yaw_deg": _r(math.degrees(yaws[frame_id]), 2)})
        tiles.append([(f"InternData f{frame_id}", theirs), (f"habitat mapped f{frame_id}  ncc {value}", image)])
    row["mid"] = mids
    row["sensor_pose_err_max"] = _r(max(pose_errs), 6) if pose_errs else None
    if pose_errs and max(pose_errs) > SENSOR_POSE_TOL:
        # Every NCC above would compare a view from somewhere else: an error, not a low score.
        raise RuntimeError(f"the RGB sensor was {max(pose_errs):.4g} off the requested pose")
    if png_dir is not None:
        name = f"{pair.scene}_{ep['episode_id']}.png"
        write_side_by_side(png_dir / name, tiles)
        row["png"] = str(Path("png") / name)
    row["seconds"] = round(time.time() - t_start, 2)
    return row


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def quantiles(values: Iterable[float | None]) -> dict[str, Any]:
    v = np.asarray([x for x in values if x is not None and math.isfinite(float(x))], dtype=np.float64)
    out: dict[str, Any] = {"n": int(v.size)}
    for q in QUANTILES:
        out[f"p{q}"] = round(float(np.percentile(v, q)), 4) if v.size else None
    return out


def _ncc_or_floor(value: Any) -> float:
    return -1.0 if value is None else float(value)  # a flat image (None) is the worst NCC


def _compare(op: str, value: float | None, threshold: float) -> bool | None:
    if value is None:
        return None
    return value >= threshold if op == ">=" else value <= threshold


def evaluate_v0(rows: Sequence[Mapping[str, Any]], *, rendered: bool) -> dict[str, Any]:
    """V0 criteria on the rows; errored rows only count against the sample size."""
    ok = [r for r in rows if not r.get("error")]
    values: dict[str, float | None] = {
        "n_scenes": float(len({r["scene"] for r in ok})),
        "n_episodes": float(len(ok)),
        "b2_end_to_goal_median_m": (float(np.median([math.inf if r.get("end_to_goal_m") is None else r["end_to_goal_m"]
                                                     for r in ok])) if ok else None),
    }
    if rendered:
        start = [_ncc_or_floor(r.get("start_ncc")) for r in ok]
        margins = [(-math.inf if r.get("start_margin") is None else float(r["start_margin"])) for r in ok]
        snaps = [math.inf if s is None else float(s) for r in ok for s in r.get("snap_m", [])]
        mids = [_ncc_or_floor(m.get("ncc")) for r in ok for m in r.get("mid", [])]
        values.update({
            "a1_start_ncc_median": float(np.median(start)) if start else None,
            "a2_start_ncc_ge_0.60_share": float(np.mean([v >= START_NCC_FLOOR for v in start])) if start else None,
            "a3_yaw90_margin_ge_0.30_share": float(np.mean([m >= YAW90_MARGIN for m in margins])) if margins else None,
            "b1_snap_le_0.3m_frame_share": float(np.mean([s <= SNAP_MAX_M for s in snaps])) if snaps else None,
            "b3_mid_ncc_median": float(np.median(mids)) if mids else None,
        })
    criteria = {}
    for name, (op, threshold) in V0_CRITERIA.items():
        value = values.get(name)
        if name in RENDER_CRITERIA and not rendered:
            criteria[name] = {"op": op, "threshold": threshold, "value": None, "pass": None, "note": "not rendered"}
            continue
        # An infinite median (no navmesh point, unreadable pose) is reported as null and fails.
        criteria[name] = {"op": op, "threshold": threshold, "value": _r(value, 4),
                          "pass": _compare(op, value, threshold)}
    if not rendered:
        verdict = "dry_run"
    else:
        verdict = "pass" if all(c["pass"] is True for c in criteria.values()) else "fail"
    return {"verdict": verdict, "criteria": criteria, "errors": len(rows) - len(ok)}


def report_quantiles(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]
    by_scene: dict[str, list] = collections.defaultdict(list)
    for r in ok:
        by_scene[r["scene"]].append(r)
    return {
        "start_ncc": quantiles(r.get("start_ncc") for r in ok),
        "yaw90_ncc": quantiles(r.get("yaw90_ncc") for r in ok),
        "start_margin": quantiles(r.get("start_margin") for r in ok),
        "mid_ncc": quantiles(m.get("ncc") for r in ok for m in r.get("mid", [])),
        "snap_m": quantiles(s for r in ok for s in r.get("snap_m", [])),
        "snap_none_frames": sum(1 for r in ok for s in r.get("snap_m", []) if s is None),
        "end_to_goal_m": quantiles(r.get("end_to_goal_m") for r in ok),
        "end_to_goal_xz_m": quantiles(r.get("end_to_goal_xz_m") for r in ok),
        "end_geodesic_to_goal_m": quantiles(r.get("end_geodesic_to_goal_m") for r in ok),
        "ref_path_dev_p50_m": quantiles(r["ref_path_dev_m"]["p50"] for r in ok if r.get("ref_path_dev_m")),
        "heading_pose_vs_actions_p50_deg": quantiles(r["heading_pose_vs_actions_deg"]["p50"] for r in ok
                                                     if r.get("heading_pose_vs_actions_deg")),
        "pose_tilt_deg_max": quantiles(r.get("pose_tilt_deg_max") for r in ok),
        "sensor_pose_err_max": quantiles(r.get("sensor_pose_err_max") for r in ok),
        "t0_frame0_translation_dev_m": quantiles(r.get("frame0_translation_dev_m") for r in ok),
        "t0_frame0_rotation_identity_err": quantiles(r.get("frame0_rotation_identity_err") for r in ok),
        "mapping_variants_end_to_goal_p50_m": {
            name: quantiles(r["mapping_variants_end_to_goal_m"][name] for r in ok
                            if r.get("mapping_variants_end_to_goal_m"))["p50"]
            for name in MAPPING_VARIANTS
        },
        "exact_client_text_share": _r(np.mean([bool(r.get("exact_client_text")) for r in ok]), 4) if ok else None,
        "per_scene": {
            scene: {"episodes": len(rs),
                    "start_ncc_p50": quantiles(r.get("start_ncc") for r in rs)["p50"],
                    "mid_ncc_p50": quantiles(m.get("ncc") for r in rs for m in r.get("mid", []))["p50"],
                    "end_to_goal_p50_m": quantiles(r.get("end_to_goal_m") for r in rs)["p50"]}
            for scene, rs in sorted(by_scene.items())
        },
    }


# --------------------------------------------------------------------------- #
# Habitat (the client's own config)
# --------------------------------------------------------------------------- #
def _tcp_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


_LLVMPIPE_ENV = {"LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe", "MESA_LOADER_DRIVER_OVERRIDE": "swrast"}


def start_xvfb(display_num: int, xvfb_bin: str | None, log_path: Path) -> subprocess.Popen:
    """A private Xvfb with llvmpipe GLX, as ``run_s2_dagger_collect_cuda.sh`` starts one per slot."""
    binary = xvfb_bin or shutil.which("Xvfb")
    if not binary:
        raise SystemExit("Xvfb not found: pass --xvfb-bin, or run under an existing DISPLAY")
    port = 6000 + int(display_num)
    if _tcp_open(port):
        raise SystemExit(f"display :{display_num} is already active; choose another --xvfb-display")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")
    proc = subprocess.Popen(
        [binary, f":{display_num}", "-screen", "0", "1024x768x24", "-nolock", "-nolisten", "unix",
         "-listen", "tcp", "+iglx", "-ac"],
        stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **_LLVMPIPE_ENV},
    )

    def stop() -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.close()

    atexit.register(stop)
    for _ in range(60):
        if _tcp_open(port):
            break
        if proc.poll() is not None:
            raise SystemExit(f"Xvfb :{display_num} exited; see {log_path}")
        time.sleep(1)
    else:
        raise SystemExit(f"Xvfb :{display_num} not ready after 60 s; see {log_path}")
    os.environ.update({"DISPLAY": f"127.0.0.1:{display_num}.0", **_LLVMPIPE_ENV})
    return proc


def import_client(scenes_dir: str | os.PathLike | None = None):
    """The evaluation client module, imported the way the 4090 collection launcher runs it.

    As ``run_s2_dagger_collect_cuda.sh``: GL pre-initialised on an MP3D scene of ``<scenes-dir>/mp3d``
    (``HEATMAPVLN_MP3D_ROOT``; ``/dataset/mp3d`` when the tree has none), which must precede habitat-lab.
    """
    tree_mp3d = Path(scenes_dir) / "mp3d" if scenes_dir is not None else None
    os.environ.setdefault("HEATMAPVLN_PREINIT_GL", "0")
    os.environ.setdefault("HEATMAPVLN_PREINIT_EMPTY_GL", "1")
    os.environ.setdefault("HEATMAPVLN_MP3D_ROOT",
                          str(tree_mp3d) if tree_mp3d is not None and tree_mp3d.is_dir() else DEFAULT_MP3D_ROOT)
    os.environ.setdefault("HABITAT_GL_GPU_ID", "0")
    return importlib.import_module("scripts.evaluation.r2r_val_unseen")


def client_habitat_config(client: Any, scene: str | os.PathLike, sim_gpu_id: int = 0):
    """``build_habitat_config`` exactly as the client builds it, with ``SIMULATOR.SCENE`` set to ``scene``."""
    args = argparse.Namespace(scenes_dir=str(Path(scene).parent), data_path="unused-by-the-simulator",
                              sim_gpu_id=int(sim_gpu_id), dataset_split="train")
    cfg = client.build_habitat_config(args)
    cfg.defrost()
    cfg.SIMULATOR.SCENE = str(scene)
    cfg.freeze()
    return cfg


def sensor_spec(cfg: Any) -> dict[str, Any]:
    sim = cfg.SIMULATOR
    return {
        "RGB_SENSOR": {"WIDTH": int(sim.RGB_SENSOR.WIDTH), "HEIGHT": int(sim.RGB_SENSOR.HEIGHT),
                       "HFOV": float(sim.RGB_SENSOR.HFOV), "POSITION": [float(v) for v in sim.RGB_SENSOR.POSITION]},
        "AGENT_0": {"HEIGHT": float(sim.AGENT_0.HEIGHT), "RADIUS": float(sim.AGENT_0.RADIUS),
                    "SENSORS": list(sim.AGENT_0.SENSORS)},
        "TURN_ANGLE": float(sim.TURN_ANGLE),
    }


def check_sensor_spec(spec: Mapping[str, Any]) -> None:
    """The client's RGB sensor must be the recorded deployment camera (s2_records.CAMERA)."""
    rgb = spec["RGB_SENSOR"]
    expected = (int(REC.CAMERA["width"]), int(REC.CAMERA["height"]), float(REC.CAMERA["hfov_deg"]),
                [0.0, CAMERA_HEIGHT_M, 0.0], TURN_DEG)
    got = (rgb["WIDTH"], rgb["HEIGHT"], rgb["HFOV"], rgb["POSITION"], spec["TURN_ANGLE"])
    if got != expected:
        raise RuntimeError(f"client RGB sensor {got} differs from the deployment camera {expected}")


class ClientSimRenderer:
    """habitat-lab ``Sim-v0`` from the client's config; one per process, reconfigured per scene."""

    def __init__(self, client: Any, *, sim_gpu_id: int = 0, require_navmesh_file: bool = False):
        self.client = client
        self.sim_gpu_id = sim_gpu_id
        self.require_navmesh_file = require_navmesh_file
        self.sim = None
        self.cfg = None
        self.last_pose_err: float | None = None

    def load(self, glb: Path, navmesh: Path | None = None, *, require_file: bool = False) -> dict[str, Any]:
        """Open ``glb`` with the client's config and load ``navmesh`` (default: the mesh's ``<stem>.navmesh``).

        The navmesh is loaded explicitly and its return value checked: Habitat-Sim 0.1.7 does not check it
        itself, and a failed load (a version-2 navmesh) can leave the pathfinder as it was, i.e. still
        ``is_loaded`` with the previous scene's navmesh.  A file that does not load is rebaked in process only
        when neither ``require_file`` nor ``--require-navmesh-file`` is set.
        """
        import habitat
        import habitat_sim

        t0 = time.time()
        cfg = client_habitat_config(self.client, glb, self.sim_gpu_id)
        if self.sim is None:
            self.sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
        else:
            self.sim.reconfigure(cfg.SIMULATOR)
        self.cfg = cfg
        info: dict[str, Any] = {"glb": str(glb), "glb_realpath": os.path.realpath(glb)}
        source_path = Path(navmesh) if navmesh is not None else Path(os.path.splitext(str(glb))[0] + ".navmesh")
        returned = self.sim.pathfinder.load_nav_mesh(str(source_path)) if source_path.is_file() else None
        loaded = (returned if isinstance(returned, bool) else source_path.is_file()) and self.sim.pathfinder.is_loaded
        info["navmesh_load_returned"] = returned if isinstance(returned, bool) else None
        if loaded:
            info.update({"navmesh_source": "file", "navmesh": str(source_path),
                         "navmesh_sha256": sha256_file(source_path)})
            meta = _read_json(source_path.with_name(source_path.name + ".meta.json"))
            if meta is not None:  # prepare_hm3d_scenes.py's record of the bake
                info["navmesh_meta_sha256"] = meta.get("navmesh_sha256")
                info["navmesh_matches_meta"] = meta.get("navmesh_sha256") == info["navmesh_sha256"]
        else:
            if require_file or self.require_navmesh_file:
                raise RuntimeError(f"navmesh {source_path} did not load for {glb} (load_nav_mesh -> {returned})")
            agent = cfg.SIMULATOR.AGENT_0
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            settings.agent_radius = float(agent.RADIUS)
            settings.agent_height = float(agent.HEIGHT)
            built = self.sim.recompute_navmesh(self.sim.pathfinder, settings)
            if not built or not self.sim.pathfinder.is_loaded:
                raise RuntimeError(f"could not rebake a navmesh for {glb}")
            info.update({"navmesh_source": "recomputed_in_process", "navmesh_not_loaded": str(source_path),
                         "navmesh_settings": {"set_defaults": True, "agent_radius": float(agent.RADIUS),
                                              "agent_height": float(agent.HEIGHT)}})
        info["load_s"] = round(time.time() - t0, 2)
        return info

    def render(self, position: Sequence[float], rotation_wxyz: Sequence[float]) -> np.ndarray:
        import quaternion  # noqa: F401  (registers np.quaternion)

        pos = np.asarray(position, dtype=np.float32)
        q = np.quaternion(*[float(v) for v in rotation_wxyz])
        self.sim.set_agent_state(pos, q, reset_sensors=True)
        obs = self.sim.get_sensor_observations()
        rgb = np.ascontiguousarray(np.asarray(obs["rgb"])[..., :3])
        sensor = self.sim.get_agent(0).get_state().sensor_states["rgb"]
        expected = camera_c2w(position, rotation_wxyz, float(self.cfg.SIMULATOR.RGB_SENSOR.POSITION[1]))
        actual_r = ORL.quat_wxyz_to_rot([sensor.rotation.w, sensor.rotation.x, sensor.rotation.y, sensor.rotation.z])
        self.last_pose_err = max(float(np.abs(np.asarray(sensor.position, dtype=np.float64) - expected[:3, 3]).max()),
                                 float(np.abs(actual_r - expected[:3, :3]).max()))
        return rgb

    def snap(self, point: Sequence[float]) -> np.ndarray | None:
        snapped = np.asarray(self.sim.pathfinder.snap_point(np.asarray(point, dtype=np.float32)), dtype=np.float64)
        return snapped if snapped.shape == (3,) and np.all(np.isfinite(snapped)) else None

    def geodesic(self, a: Sequence[float], b: Sequence[float]) -> float:
        import habitat_sim

        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(a, dtype=np.float32)
        path.requested_end = np.asarray(b, dtype=np.float32)
        found = self.sim.pathfinder.find_path(path)
        distance = float(path.geodesic_distance)
        return distance if found and math.isfinite(distance) else math.inf

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()
            self.sim = None


def versions() -> dict[str, Any]:
    """Versions of what is already imported; importing habitat-lab here could precede the GL context."""
    out = {"python": sys.version.split()[0], "numpy": np.__version__}
    for name in ("habitat_sim", "habitat"):
        module = sys.modules.get(name)
        out[name] = getattr(module, "__version__", "unknown") if module is not None else None
    return out


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "-c", "safe.directory=*", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def select(
    by_scene: Mapping[str, Sequence[Mapping[str, Any]]],
    intern_dirs: Mapping[str, Path],
    heldout: set[str],
    *,
    seed: str,
    num_scenes: int,
    per_scene: int,
    scene_ok: Any = None,
) -> tuple[list[tuple[str, list[Pair]]], dict[str, Any]]:
    """Training scenes in ``sha256(seed:ID)`` order, ``per_scene`` pairs each, until ``num_scenes``.

    Every json x InternData scene is matched for the statistics; ``scene_ok(scene)`` (mesh found) gates
    selection only, as does InternData's frame 0 of a pair being on disk (a pair without it is passed over
    and counted, so a scene with metadata but no extracted views is not drawn).
    """
    common = sorted(set(by_scene) & set(intern_dirs))
    stats: dict[str, collections.Counter] = {"train": collections.Counter(), "heldout": collections.Counter()}
    chosen: list[tuple[str, list[Pair]]] = []
    skipped: dict[str, str] = {}
    for scene in sorted(common, key=lambda s: _sha_key(seed, s)):
        pairs, scene_stats = match_scene(by_scene[scene], read_interndata_scene(intern_dirs[scene]),
                                         seed=seed, scene=scene)
        part = "heldout" if scene in heldout else "train"
        stats[part].update(scene_stats)
        stats[part]["scenes"] += 1
        if part == "heldout" or len(chosen) >= num_scenes:
            continue
        with_frames = []
        for pair in pairs:
            if len(with_frames) == per_scene:
                break
            if IR._frame_path(pair.intern.scene_dir, IR.FRONT_SETTING, pair.intern.episode_index, 0).is_file():
                with_frames.append(pair)
            else:
                stats[part]["pairs_without_frame0"] += 1
        if len(with_frames) < per_scene:
            skipped[scene] = f"{len(with_frames)} pairs with InternData frame 0 < {per_scene} ({len(pairs)} pairs)"
            continue
        if scene_ok is not None:
            reason = scene_ok(scene)
            if reason:
                skipped[scene] = reason
                continue
        chosen.append((scene, with_frames))
    summary = {
        "json_scenes": len(by_scene),
        "interndata_scenes": len(intern_dirs),
        "common_scenes": len(common),
        "heldout_in_common": sorted(set(common) & heldout),
        "match": {part: dict(counter) for part, counter in stats.items()},
        "skipped_scenes": skipped,
    }
    for part in ("train", "heldout"):
        c = stats[part]
        n = c["json_episodes"]
        summary["match"][part]["matched_share_of_json"] = _r(c["matched"] / n, 4) if n else None
        # InternData has the (scene, key) at all -- build_hm3d_episodes' overlap, ambiguous keys included.
        summary["match"][part]["overlap_share_of_json"] = _r((c["matched"] + c["ambiguous"]) / n, 4) if n else None
    return chosen, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", required=True,
                        help="VLN-CE episodes (.json.gz), StreamVLN's or build_hm3d_episodes'")
    parser.add_argument("--interndata-root", action="append", default=None,
                        help=f"InternData ScaleVLN scene dirs (repeatable; default {DEFAULT_INTERNDATA_ROOT})")
    parser.add_argument("--scenes-dir", default=DEFAULT_SCENES_DIR, help="EXP-28 scene tree (hm3d/<ID>/<ID>.glb)")
    parser.add_argument("--hm3d-root", default=DEFAULT_HM3D_ROOT,
                        help="unpacked HM3D (train/00xxx-<ID>/<ID>.basis.glb)")
    parser.add_argument("--glb-source", choices=("tree", "auto", "raw"), default="tree",
                        help="tree (default): the collection's hm3d/<ID>/<ID>.glb and its navmesh file, a scene "
                             "without them is not drawn; auto: the tree, else the original basis.glb "
                             "(diagnosis only); raw: the original basis.glb")
    parser.add_argument("--require-navmesh-file", action="store_true",
                        help="fail a scene whose navmesh does not load instead of rebaking it in process")
    parser.add_argument("--holdout-file", default=None, help="held-out scenes (else the ledger's rule)")
    parser.add_argument("--num-scenes", type=int, default=10)
    parser.add_argument("--episodes-per-scene", type=int, default=4)
    parser.add_argument("--mid-frames", type=int, default=3, help="middle frames rendered per episode")
    parser.add_argument("--heading-source", choices=("pose", "actions"), default="pose",
                        help="middle-frame heading: InternData pose yaw (default) or integrated turn actions")
    parser.add_argument("--seed", default="exp28-v0")
    parser.add_argument("--sim-gpu-id", type=int, default=0)
    parser.add_argument("--xvfb-display", type=int, default=None, help="start a private Xvfb on :N (llvmpipe)")
    parser.add_argument("--xvfb-bin", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dry-run", action="store_true", help="match, select and the pose-only numbers; no Habitat")
    parser.add_argument("--force", action="store_true", help="replace an existing summary")
    args = parser.parse_args(argv)
    if not args.interndata_root:
        args.interndata_root = [DEFAULT_INTERNDATA_ROOT]
    return args


def run_v0(args: argparse.Namespace, renderer: Any | None = None) -> dict[str, Any]:
    """Select, verify and judge; ``renderer`` None means pose-only (``--dry-run``)."""
    out_dir = Path(args.out_dir)
    if (out_dir / SUMMARY_NAME).exists() and not args.force:
        raise FileExistsError(f"{out_dir / SUMMARY_NAME} exists; pass --force to replace it")
    importlib.import_module("pyarrow.parquet")  # InternData's poses; missing it would error every episode
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    episodes = load_vlnce_episodes(args.episodes)
    by_scene = group_by_scene(episodes)
    intern_dirs = interndata_scene_dirs(args.interndata_root)
    # As build_hm3d_episodes.split_scenes: only scenes the json annotates can be drawn.
    by_rule = heldout_by_rule(d.name for scene, d in intern_dirs.items() if scene in by_scene)
    if args.holdout_file:
        heldout_names, heldout_source = read_holdout_file(args.holdout_file), str(args.holdout_file)
        check_heldout(heldout_names, args.holdout_file)
    else:
        heldout_names = by_rule
        heldout_source = (f"rule: B3 + {HASHED_HELDOUT} smallest sha256('{HELDOUT_SALT}' + dir) of the "
                          "--interndata-root scenes the json annotates")
    heldout = {hm3d_id(name) for name in heldout_names}
    scene_ids = {hm3d_id(e["scene_id"]): str(e["scene_id"]) for e in episodes}

    resolved: dict[str, tuple[Path, Path | None, str]] = {}

    # Resolved in a dry run too, so that it selects exactly the scenes the rendered run will.
    def scene_ok(scene: str) -> str | None:
        try:
            resolved[scene] = resolve_scene(scene_ids[scene], scene, scenes_dir=args.scenes_dir,
                                            hm3d_root=args.hm3d_root, glb_source=args.glb_source)
        except FileNotFoundError as exc:
            return str(exc)
        return None

    chosen, selection = select(by_scene, intern_dirs, heldout, seed=args.seed, num_scenes=args.num_scenes,
                               per_scene=args.episodes_per_scene, scene_ok=scene_ok)
    print(f"[v0] {len(episodes)} json episodes, {selection['common_scenes']} json x InternData scenes, "
          f"{len(heldout)} held out; selected {len(chosen)} scenes x {args.episodes_per_scene}", flush=True)
    if len(chosen) < args.num_scenes:  # V0 cannot pass; say why before anything renders
        for scene, reason in list(selection["skipped_scenes"].items())[:5]:
            print(f"[v0] WARNING: scene {scene} not drawn: {reason}", flush=True)

    rows: list[dict[str, Any]] = []
    scenes_info: dict[str, Any] = {}
    png_dir = out_dir / "png" if renderer is not None else None
    if (out_dir / SUMMARY_NAME).exists() and (out_dir / "png").is_dir():
        shutil.rmtree(out_dir / "png")  # --force: no picture of the replaced run may sit beside this one
    for scene, pairs in chosen:
        glb, navmesh, how = resolved[scene]
        scenes_info[scene] = {"mesh_source": how, "glb": str(glb), "navmesh": None if navmesh is None else str(navmesh)}
        if renderer is not None:
            try:
                # The tree's navmesh is the collection's: it must load as it is, never be rebaked here.
                scenes_info[scene] = {"mesh_source": how,
                                      **renderer.load(glb, navmesh, require_file=(how == "tree"))}
            except Exception as exc:  # the scene's episodes fail; the rest still run
                scenes_info[scene] = {"mesh_source": how, "glb": str(glb), "error": f"{type(exc).__name__}: {exc}"}
                for pair in pairs:
                    rows.append({"scene": scene, "episode_id": pair.episode["episode_id"],
                                 "interndata_episode_index": pair.intern.episode_index,
                                 "error": f"scene load: {type(exc).__name__}: {exc}"})
                print(f"[v0] scene {scene}: load failed: {exc}", flush=True)
                continue
        for pair in pairs:
            try:
                row = verify_episode(pair, renderer, png_dir=png_dir, mid_frames=args.mid_frames,
                                     heading_source=args.heading_source)
            except Exception as exc:
                row = {"scene": scene, "episode_id": pair.episode["episode_id"],
                       "interndata_episode_index": pair.intern.episode_index,
                       "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=4)}
            rows.append(row)
            shown = ("start_ncc", "yaw90_ncc", "snap_le_0.3m_share", "end_to_goal_m", "frame0_translation_dev_m",
                     "error")
            print(f"[v0] {scene} ep {row['episode_id']}: "
                  + " ".join(f"{k}={row[k]}" for k in shown if row.get(k) is not None), flush=True)

    judged = evaluate_v0(rows, rendered=renderer is not None)
    spec = None
    if renderer is not None and getattr(renderer, "cfg", None) is not None:
        spec = sensor_spec(renderer.cfg)
    summary = {
        "schema": SCHEMA,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "verdict": judged["verdict"],
        "criteria": judged["criteria"],
        "errors": judged["errors"],
        "report": report_quantiles(rows),
        "conventions": CONVENTIONS,
        "inputs": {
            "episodes": str(args.episodes),
            "episodes_sha256": sha256_file(args.episodes),
            "interndata_roots": [str(r) for r in args.interndata_root],
            "scenes_dir": str(args.scenes_dir),
            "hm3d_root": str(args.hm3d_root),
            "glb_source": args.glb_source,
            "heading_source": args.heading_source,
            "seed": args.seed,
            "num_scenes": args.num_scenes,
            "episodes_per_scene": args.episodes_per_scene,
            "mid_frames": args.mid_frames,
            "ncc_size": list(NCC_SIZE),
            "negative_yaw_deg": NEGATIVE_YAW_DEG,
        },
        "heldout": {"source": heldout_source, "scenes": sorted(heldout_names),
                    "matches_rule": {hm3d_id(n) for n in heldout_names} == {hm3d_id(n) for n in by_rule}},
        "selection": {**selection, "chosen": [
            {"scene": scene, "interndata_scene_dir": pairs[0].intern.scene_dir.name,
             "episodes": [[p.episode["episode_id"], p.intern.episode_index] for p in pairs]}
            for scene, pairs in chosen]},
        "scenes": scenes_info,
        "mesh_sources": dict(collections.Counter(i.get("mesh_source") for i in scenes_info.values())),
        "navmesh_sources": dict(collections.Counter(i.get("navmesh_source") for i in scenes_info.values()
                                                    if "navmesh_source" in i)),
        "pose_columns": dict(collections.Counter(r.get("pose_column") for r in rows if not r.get("error"))),
        # The tree's navmesh differs from the sha256 prepare_hm3d_scenes.py recorded when it baked it.
        "navmesh_not_matching_meta": sorted(name for name, i in scenes_info.items()
                                            if i.get("navmesh_matches_meta") is False),
        "sensor_spec": spec,
        "versions": versions() if renderer is not None else {"python": sys.version.split()[0]},
        "git_head": git_head(),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "seconds": round(time.time() - t0, 1),
    }
    _write_text(out_dir / ROWS_NAME, "".join(json.dumps(_jsonable(r), ensure_ascii=False) + "\n" for r in rows))
    _write_text(out_dir / SUMMARY_NAME, json.dumps(_jsonable(summary), ensure_ascii=False, indent=1) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (Path(args.out_dir) / SUMMARY_NAME).exists() and not args.force:
        print(f"[v0] ERROR: {Path(args.out_dir) / SUMMARY_NAME} exists; pass --force to replace it", file=sys.stderr)
        return 2
    renderer = None
    try:
        if not args.dry_run:
            if args.xvfb_display is not None:
                start_xvfb(args.xvfb_display, args.xvfb_bin, Path(args.out_dir) / "xvfb.log")
            elif not os.environ.get("DISPLAY"):
                raise SystemExit("rendering needs an X display: pass --xvfb-display N or set DISPLAY (or --dry-run)")
            client = import_client(args.scenes_dir)
            renderer = ClientSimRenderer(client, sim_gpu_id=args.sim_gpu_id,
                                         require_navmesh_file=args.require_navmesh_file)
            check_sensor_spec(sensor_spec(client_habitat_config(client, "unused.glb", args.sim_gpu_id)))
        summary = run_v0(args, renderer)
    except SystemExit as exc:  # start_xvfb's and the display check's messages: no verdict, so not exit 1
        if exc.code in (None, 0):
            raise
        print(f"[v0] ERROR: {exc.code}", file=sys.stderr)
        return 2
    except Exception as exc:  # anything before a verdict is exit 2, never 1 (= "V0 fails")
        print(f"[v0] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        if not isinstance(exc, (FileNotFoundError, FileExistsError, ValueError, KeyError, ImportError)):
            traceback.print_exc()
        return 2
    finally:
        if renderer is not None:
            renderer.close()
    print(f"[v0] verdict: {summary['verdict']}", flush=True)
    for name, c in summary["criteria"].items():
        print(f"[v0]   {name}: {c['value']} {c['op']} {c['threshold']} -> {c['pass']}", flush=True)
    print(f"[v0] meshes {summary['mesh_sources']}, navmeshes {summary['navmesh_sources']}, "
          f"pose columns {summary['pose_columns']}, errors {summary['errors']}", flush=True)
    if set(summary["mesh_sources"]) - {"tree"} or set(summary["navmesh_sources"]) - {"file"}:
        print("[v0] WARNING: not every scene was the collection's tree mesh with its navmesh file", flush=True)
    if summary["navmesh_not_matching_meta"]:
        print(f"[v0] WARNING: navmesh changed since prepare_hm3d_scenes.py baked it: "
              f"{summary['navmesh_not_matching_meta']}", flush=True)
    print(f"[v0] summary: {Path(args.out_dir) / SUMMARY_NAME}", flush=True)
    return 0 if summary["verdict"] in ("pass", "dry_run") else 1


if __name__ == "__main__":
    sys.exit(main())

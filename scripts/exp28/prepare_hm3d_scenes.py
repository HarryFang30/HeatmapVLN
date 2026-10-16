#!/usr/bin/env python3
"""EXP-28 T1: the HM3D scene tree, every navmesh rebaked to Recast set v1 for habitat-sim 0.1.7.

HM3D v0.2 ships ``<ID>.basis.navmesh`` as Recast set version 2 (or not at all, 60 of the 800 train
scenes); habitat-sim 0.1.7 loads only version 1 and otherwise leaves the pathfinder unloaded without an
error (README EXP-18: 178/740 are v2).  This script bakes each scene's navmesh once, from the scene's own
collision mesh, into a separate tree the client and the labellers read (ledger EXP-28 设置, 场景树):

  <out-root>/mp3d                           -> /dataset/mp3d
  <out-root>/hm3d/<ID>/<ID>.glb             -> <hm3d-root>/<00xxx-ID>/<ID>.basis.glb (symlink; never written)
  <out-root>/hm3d/<ID>/<ID>.navmesh         the rebaked Recast set v1 navmesh (read-only)
  <out-root>/hm3d/<ID>/<ID>.navmesh.meta.json   how it was baked: settings, sha256s, pathfinder checks
  <out-root>/hm3d_navmesh_manifest.json     every scene of the tree: navmesh sha256, scene count, failures
  <out-root>/prepare_logs/<stamp>/          one log, task and result file per scene
  <out-root>/.prepare_hm3d_scenes.lock      held by a run that writes (one at a time)

The 11-character directory keeps one scene key everywhere: the client keys a scene by its directory and
requires ``<scene>/<scene>.<ext>`` with ``<scene>`` one ``\\w+`` token (r2r_val_unseen.py
``_episode_scene_key``), which HM3D's own ``00xxx-<ID>/<ID>.basis.glb`` is not; and habitat-sim 0.1.7
looks for ``splitext(scene_id)[0] + ".navmesh"``, i.e. ``hm3d/<ID>/<ID>.navmesh``, the path
``label_route_corrections.py --scenes_dir <out-root>/hm3d`` and ``probe_metrics.py --scenes`` read.

Bake rule (one per scene, in its own subprocess so a habitat-sim crash or hang fails only that scene):
- settings: ``NavMeshSettings.set_defaults()``, then ``agent_radius`` / ``agent_height`` from the client's
  ``hab_cfg.SIMULATOR.AGENT_0``.  The client's ``build_habitat_config`` (scripts/evaluation/r2r_val_unseen.py)
  is run as written on habitat-lab's ``get_config()``; it assigns no AGENT_0 size, so the values are
  habitat-lab's defaults.  Both, and habitat-sim's ``AgentConfiguration()`` defaults, are printed.  The
  repo's configs/ hold no habitat simulator keys; the client builds its habitat config in code.
- the scene loads through ``habitat_sim.Simulator`` on ``hm3d/<ID>/<ID>.glb`` (GLX, so an X server:
  one Xvfb per worker, llvmpipe, as the launchers do), then ``sim.recompute_navmesh(sim.pathfinder,
  settings)``, the call of the ScaleVLN in-process fallback, and ``save_nav_mesh`` to a temporary file;
- before it takes the final name: Recast set magic and version 1; a fresh PathFinder loads it
  (``is_loaded``) with the in-memory bounds and navigable area, and 8 seeded random points are navigable;
- after: a second Simulator opened on ``hm3d/<ID>/<ID>.glb``, as the client opens a scene, must come up
  with ``pathfinder.is_loaded`` and the same navmesh (``--no-verify-sim`` skips this second load);
- report only (never a failure; D0 is ``is_loaded``): where HM3D's own ``<ID>.basis.navmesh`` is Recast set v1,
  which 0.1.7 reads, the rebaked navmesh against it (bounds, area, rebaked random points navigable on it), per
  scene in the meta file and summarised in the manifest's ``vs_original_navmesh``: a mesh loaded in another
  frame, or another agent size than HM3D's own bake, shows there before T2 filters an episode on it.

Reruns: a navmesh whose meta file records the same settings and sha256 is kept (reloaded and checked
again); a navmesh without a meta file (an interrupted bake) is rebaked; one baked under other settings,
or changed since, is a failure unless ``--rebake``.  Nothing is ever written under ``--hm3d-root``.
Exit 0 when every requested scene is ready, 2 when any failed (the manifest is written regardless).

Scenes come from ``--scenes-file`` (text, one per line; JSON list; JSON object with ``scenes`` or
``*scenes`` lists; VLN-CE ``.json.gz`` episodes; this script's own manifest), ``--scenes-from-dir``
(InternData-N1 scene dirs: the child directories with ``meta/episodes.jsonl``, the set
build_hm3d_episodes.py uses, so ``--expect-count 305`` checks this tree against it), ``--scenes`` (comma
list) or ``--all-scenes``; an entry may be ``00xxx-<ID>``, the bare ``<ID>`` or a scene_id path.  Inside
fjl-habitat::

  /opt/conda/bin/python scripts/exp28/prepare_hm3d_scenes.py \\
      --scenes-from-dir /workspace/exp23/data/interndata_n1/scalevln_train \\
      --scenes-from-dir /workspace/exp23/data/interndata_n1/scalevln --expect-count 305 --dry-run
  /opt/conda/bin/python scripts/exp28/prepare_hm3d_scenes.py --scenes-file scenes.txt --workers 8
"""

from __future__ import annotations

import argparse
import ast
import builtins
import fcntl
import gzip
import hashlib
import json
import math
import os
import queue
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

SCHEMA = "exp28-hm3d-scene-tree-v1"
META_SCHEMA = "exp28-hm3d-navmesh-v1"
SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]

DEFAULT_OUT_ROOT = Path("/workspace/exp28/scene_datasets")
DEFAULT_HM3D_ROOT = Path("/workspace/exp23/data/hm3d/train")
DEFAULT_MP3D_TARGET = Path("/dataset/mp3d")
DEFAULT_CLIENT = REPO / "scripts" / "evaluation" / "r2r_val_unseen.py"
CLIENT_BUILDER = "build_habitat_config"
MANIFEST_NAME = "hm3d_navmesh_manifest.json"
EXPECTED_HABITAT_SIM = "0.1.7"

RECAST_SET_MAGIC = 0x4D534554  # 'MSET', little-endian on disk (b"TESM")
RECAST_SET_VERSION = 1  # the only Recast set version habitat-sim 0.1.7 loads

SCENE_DIR_RE = re.compile(r"(?<![A-Za-z0-9])(\d{5})-([A-Za-z0-9]{11})(?![A-Za-z0-9])")
SCENE_ID_RE = re.compile(r"[A-Za-z0-9]{11}")

NAVMESH_SETTING_FIELDS = (
    "cell_size", "cell_height", "agent_height", "agent_radius", "agent_max_climb", "agent_max_slope",
    "region_min_size", "region_merge_size", "edge_max_len", "edge_max_error", "verts_per_poly",
    "detail_sample_dist", "detail_sample_max_error", "filter_low_hanging_obstacles", "filter_ledge_spans",
    "filter_walkable_low_height_spans",
)
INCLUDE_STATIC_OBJECTS = False
RANDOM_POINTS = 8
BOUNDS_TOL = 1e-4
AREA_RTOL = 1e-4
# The comparison with HM3D's own v1 navmeshes is reported, never gated (D0 is is_loaded alone); these only
# pick the scenes the manifest lists for a look.
ORIGINAL_POINTS = 32
LOOK_BOUNDS_DIFF_M = 1.0
LOOK_POINTS_FRACTION = 0.5

# The launchers' software GL (run_ppa_r2r_val_unseen_cuda.sh): Xvfb over TCP, Mesa llvmpipe.
GL_ENV = {"LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe", "MESA_LOADER_DRIVER_OVERRIDE": "swrast"}
XVFB_ARGS = ("-screen", "0", "1024x768x24", "-nolock", "-nolisten", "unix", "-listen", "tcp", "+iglx", "-ac")

_PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    with _PRINT_LOCK:
        print(f"[prepare-hm3d] {message}", flush=True)


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def recast_set_header(path: Path) -> dict | None:
    """``{"magic_ok", "version", "tiles"}`` of a Recast navmesh set file; None when the file is absent."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(12)
    except FileNotFoundError:
        return None
    if len(head) < 12:
        return {"magic_ok": False, "version": None, "tiles": None}
    magic, version, tiles = struct.unpack("<iii", head)
    return {"magic_ok": magic == RECAST_SET_MAGIC, "version": version, "tiles": tiles}


def header_label(header: dict | None) -> str:
    """Histogram key: "1", "2", ..., "missing" or "bad-header"."""
    if header is None:
        return "missing"
    if not header["magic_ok"]:
        return "bad-header"
    return str(header["version"])


# ----------------------------------------------------------------------------
# which scenes
# ----------------------------------------------------------------------------
def read_scene_entries(path: Path) -> list[str]:
    """Scene entries of a text, JSON or VLN-CE ``.json.gz`` file (see the module docstring)."""
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return _entries_from_json(json.load(handle), path)
    if path.suffix == ".json":
        return _entries_from_json(json.loads(path.read_text(encoding="utf-8")), path)
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    return entries


def _entries_from_json(data: Any, path: Path) -> list[str]:
    if isinstance(data, dict):
        if isinstance(data.get("episodes"), list):
            items: Any = [episode["scene_id"] for episode in data["episodes"]]
        elif "scenes" in data:
            items = data["scenes"]
        else:
            keys = sorted(k for k, v in data.items() if k.endswith("scenes") and isinstance(v, list))
            if not keys:
                raise SystemExit(f"{path}: no 'episodes', 'scenes' or '*scenes' list (keys: {sorted(data)[:20]})")
            items = [item for key in keys for item in data[key]]
    else:
        items = data
    if isinstance(items, dict):  # e.g. this script's manifest: {"scenes": {ID: {...}}}
        items = list(items)
    if not isinstance(items, list):
        raise SystemExit(f"{path}: the scene list is a {type(items).__name__}, not a list")
    entries = []
    for item in items:
        if isinstance(item, dict):
            item = next((item[k] for k in ("scene_dir", "scene_id", "scene") if isinstance(item.get(k), str)), None)
        if not isinstance(item, str):
            raise SystemExit(f"{path}: cannot read a scene from {item!r}")
        entries.append(item)
    return entries


def scene_entries_from_dir(directory: Path) -> tuple[list[str], list[str]]:
    """InternData-N1 scene dirs of ``directory``: ``(taken, skipped)`` child directory names.

    Taken are the children with ``meta/episodes.jsonl``, the definition build_hm3d_episodes.py
    (``interndata_scene_dirs``) uses, so ``--expect-count`` compares this tree with the scenes T2 can use;
    a child without it (a partial download, a non-scene directory) is skipped and reported.
    """
    if not directory.is_dir():
        raise SystemExit(f"--scenes-from-dir {directory} is not a directory")
    taken, skipped = [], []
    for child in sorted(directory.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        (taken if (child / "meta" / "episodes.jsonl").is_file() else skipped).append(child.name)
    return taken, skipped


def index_hm3d_root(root: Path) -> dict[str, list[str]]:
    """``{ID: [directory names]}`` of the ``00xxx-<ID>`` directories directly under ``root``."""
    if not root.is_dir():
        raise SystemExit(f"--hm3d-root {root} is not a directory")
    index: dict[str, list[str]] = {}
    for child in sorted(root.iterdir()):
        match = SCENE_DIR_RE.fullmatch(child.name)
        if match and child.is_dir():
            index.setdefault(match.group(2), []).append(child.name)
    return index


def resolve_scene_entries(
    entries: Sequence[str], index: Mapping[str, Sequence[str]]
) -> tuple[list[tuple[str, str]], list[dict], int]:
    """Entries -> sorted unique ``(ID, 00xxx-ID)``, unresolvable entries, number of duplicates."""
    resolved: dict[str, str] = {}
    problems: list[dict] = []
    duplicates = 0
    for raw in entries:
        text = str(raw).strip()
        match = SCENE_DIR_RE.search(text)
        if match:
            scene, scene_dir = match.group(2), match.group(0)
            if scene_dir not in index.get(scene, ()):
                found = list(index.get(scene, ()))
                problems.append({"entry": text, "error": f"no directory {scene_dir} under the HM3D root"
                                 + (f" (it has {found})" if found else "")})
                continue
        else:
            token = Path(text).name.split(".", 1)[0]
            if not SCENE_ID_RE.fullmatch(token):
                problems.append({"entry": text, "error": "not an HM3D scene: want 00xxx-<ID>, <ID> or a scene_id path"})
                continue
            scene = token
            dirs = list(index.get(scene, ()))
            if len(dirs) != 1:
                problems.append({"entry": text, "error": f"{len(dirs)} 00xxx-{scene} directories under the HM3D root"
                                 + (f": {dirs}" if dirs else "")})
                continue
            scene_dir = dirs[0]
        if scene in resolved:
            duplicates += 1
            if resolved[scene] != scene_dir:
                problems.append({"entry": text, "error": f"{scene} also requested as {resolved[scene]}"})
            continue
        resolved[scene] = scene_dir
    return sorted(resolved.items()), problems, duplicates


# ----------------------------------------------------------------------------
# the client's AGENT_0
# ----------------------------------------------------------------------------
def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _import_table(tree: ast.Module) -> dict[str, str]:
    """Local name -> ``module.name`` for every ``from module import name [as alias]`` in the module."""
    table = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                table[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return table


def _builder(tree: ast.Module) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == CLIENT_BUILDER:
            return node
    raise SystemExit(f"the client source has no top-level {CLIENT_BUILDER}()")


def inspect_client_builder(source: str) -> dict:
    """What ``build_habitat_config`` starts from and what it assigns under ``SIMULATOR.AGENT_0``."""
    tree = ast.parse(source)
    imports = _import_table(tree)
    function = _builder(tree)
    cfg_var, base = None, None
    for statement in function.body:
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name) and isinstance(statement.value, ast.Call)):
            cfg_var = statement.targets[0].id
            called = _dotted(statement.value.func) or "?"
            base = imports.get(called, called)
            break
    prefix = f"{cfg_var}.SIMULATOR.AGENT_0"
    assignments: dict[str, Any] = {}
    merges = []
    for node in ast.walk(function):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            name = _dotted(target)
            if name == prefix or (name and name.startswith(prefix + ".")):
                key = name[len(prefix) + 1:] or "<whole node>"
                try:
                    assignments[key] = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError):
                    assignments[key] = ast.unparse(node.value)
        if isinstance(node, ast.Call):
            called = _dotted(node.func) or ""
            if called.rsplit(".", 1)[-1] in ("merge_from_file", "merge_from_list", "merge_from_other_cfg"):
                merges.append(called)
    return {
        "function": CLIENT_BUILDER,
        "base": f"{base}()",
        "agent_0_assignments": assignments,
        "sets_height": "HEIGHT" in assignments or "<whole node>" in assignments,
        "sets_radius": "RADIUS" in assignments or "<whole node>" in assignments,
        "merges": merges,
    }


def build_client_config(source: str, resolve_import: Mapping[str, Any], args: Any) -> Any:
    """Run the client's ``build_habitat_config`` as written, its imports bound from ``resolve_import``.

    ``resolve_import`` maps ``module.name`` (e.g. ``habitat.config.default.get_config``) to the object.
    """
    tree = ast.parse(source)
    namespace: dict[str, Any] = {"__builtins__": builtins}
    for local, full in _import_table(tree).items():
        if full in resolve_import:
            namespace[local] = resolve_import[full]
    module = ast.Module(body=[_builder(tree)], type_ignores=[])
    exec(compile(module, f"<client {CLIENT_BUILDER}>", "exec"), namespace)
    try:
        return namespace[CLIENT_BUILDER](args)
    except (NameError, AttributeError) as exc:
        raise SystemExit(f"cannot run the client's {CLIENT_BUILDER} outside the client ({exc}); "
                         "teach build_client_config the new name") from exc


def client_builder_args(out_root: Path) -> SimpleNamespace:
    """The client arguments ``build_habitat_config`` reads; none of them touches AGENT_0."""
    return SimpleNamespace(scenes_dir=str(out_root), data_path="", sim_gpu_id=0, dataset_split="train")


def agent_dims(cfg: Any) -> tuple[float, float]:
    """``(height, radius)`` of ``cfg.SIMULATOR.AGENT_0``."""
    agent = cfg.SIMULATOR.AGENT_0
    return float(agent.HEIGHT), float(agent.RADIUS)


def agent_size_equal(a: Sequence[float], b: Sequence[float]) -> bool:
    """Same ``(height, radius)`` up to float32 rounding (habitat_sim stores them as float32)."""
    return all(math.isclose(float(x), float(y), rel_tol=1e-6, abs_tol=1e-6) for x, y in zip(a, b))


# ----------------------------------------------------------------------------
# habitat (imported only where needed: the dry run and the tests need neither)
# ----------------------------------------------------------------------------
def import_habitat_sim():
    """habitat_sim, imported as the client imports it (NumPy aliases patched, torch first)."""
    try:
        import numpy as np
    except ImportError:  # habitat_sim's own import then says what is missing
        np = None
    if np is not None:
        for name, alias in (("float", np.float64), ("int", np.int64), ("bool", np.bool_)):
            if not hasattr(np, name):
                setattr(np, name, alias)
    if os.environ.get("EXP28_PREP_IMPORT_TORCH", "1") != "0":
        try:  # r2r_val_unseen.py imports torch before habitat_sim
            import torch  # noqa: F401
        except ImportError:
            pass
    import habitat_sim

    return habitat_sim


def make_navmesh_settings(hs: Any, radius: float, height: float) -> Any:
    settings = hs.NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius = radius
    settings.agent_height = height
    return settings


def navmesh_settings_dict(settings: Any) -> dict:
    out = {}
    for name in NAVMESH_SETTING_FIELDS:
        if hasattr(settings, name):
            value = getattr(settings, name)
            out[name] = value if isinstance(value, (bool, int)) else float(value)
    return out


def settings_digest(settings: Mapping[str, Any], habitat_sim_version: str) -> str:
    return canonical_sha256({"navmesh_settings": dict(settings), "include_static_objects": INCLUDE_STATIC_OBJECTS,
                             "habitat_sim": habitat_sim_version, "recast_set_version": RECAST_SET_VERSION})


def resolve_bake_parameters(client_script: Path, out_root: Path, expect_habitat_sim: str) -> dict:
    """Client AGENT_0 size, the NavMeshSettings and the habitat-sim version; prints them for the record."""
    hs = import_habitat_sim()
    version = str(getattr(hs, "__version__", "unknown"))
    if expect_habitat_sim and version != expect_habitat_sim:
        raise SystemExit(f"habitat_sim {version} here, expected {expect_habitat_sim} (--habitat-sim-version)")
    import habitat.config.default as habitat_default

    source = client_script.read_text(encoding="utf-8")
    static = inspect_client_builder(source)
    if static["merges"]:
        raise SystemExit(f"the client's {CLIENT_BUILDER} merges other configs ({static['merges']}); "
                         "its AGENT_0 size then depends on those files")
    cfg = build_client_config(source, {
        "habitat.config.default.get_config": habitat_default.get_config,
        "habitat.config.default.Config": habitat_default.Config,
    }, client_builder_args(out_root))
    height, radius = agent_dims(cfg)
    default_height, default_radius = agent_dims(habitat_default.get_config())
    legacy = hs.agent.AgentConfiguration()
    settings = navmesh_settings_dict(make_navmesh_settings(hs, radius, height))
    sim_default = {"height": float(legacy.height), "radius": float(legacy.radius)}
    client = {
        "script": str(client_script),
        "script_sha256": sha256_file(client_script),
        "builder": static,
        "agent_0": {"HEIGHT": height, "RADIUS": radius},
        "habitat_lab_default": {"file": str(Path(habitat_default.__file__).resolve()),
                                "HEIGHT": default_height, "RADIUS": default_radius},
        "habitat_sim_agent_configuration_default": sim_default,
        "agent_0_equals_habitat_sim_default": agent_size_equal((height, radius),
                                                               (sim_default["height"], sim_default["radius"])),
    }
    assigned = ", ".join(sorted(static["agent_0_assignments"])) or "nothing"
    log(f"client hab_cfg: {client_script.name}::{CLIENT_BUILDER} (sha256 {client['script_sha256'][:12]}) starts "
        f"from {static['base']} and assigns SIMULATOR.AGENT_0.{{{assigned}}}")
    log(f"client hab_cfg SIMULATOR.AGENT_0: HEIGHT={height} RADIUS={radius}; habitat-lab default "
        f"({client['habitat_lab_default']['file']}): HEIGHT={default_height} RADIUS={default_radius}; "
        f"habitat_sim AgentConfiguration(): height={legacy.height} radius={legacy.radius}")
    if not client["agent_0_equals_habitat_sim_default"]:
        log("WARNING: the client's AGENT_0 size is not habitat_sim's AgentConfiguration() size. A Simulator that "
            "recomputes its navmesh in memory when the agent size differs from that default would pass the "
            "client-style check without reading the baked file; the PathFinder checks still read it.")
    log(f"habitat_sim {version}; NavMeshSettings.set_defaults() + client agent size: {json.dumps(settings)}")
    return {
        "habitat_sim_version": version,
        "client": client,
        "agent_height": height,
        "agent_radius": radius,
        "navmesh_settings": settings,
        "navmesh_settings_sha256": settings_digest(settings, version),
    }


# ----------------------------------------------------------------------------
# the tree
# ----------------------------------------------------------------------------
def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


LOCK_NAME = ".prepare_hm3d_scenes.lock"


def lock_tree(out_root: Path) -> Any:
    """An exclusive, non-blocking lock on the tree: two runs at once would interleave navmesh and meta
    writes.  The open file holds it; closing it (or the process ending) releases it."""
    out_root.mkdir(parents=True, exist_ok=True)
    handle = open(out_root / LOCK_NAME, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit(f"another prepare_hm3d_scenes.py run holds {out_root / LOCK_NAME}") from None
    return handle


def check_roots(out_root: Path, hm3d_root: Path) -> None:
    """Refuse any layout in which a write into the tree could land in the HM3D originals."""
    hm3d_tree = out_root / "hm3d"
    for path in (out_root, hm3d_tree):
        if path.is_symlink():
            raise SystemExit(f"{path} is a symlink; the tree must be real directories")
    if _inside(hm3d_tree, hm3d_root) or _inside(hm3d_root, hm3d_tree):
        raise SystemExit(f"--out-root {out_root} overlaps --hm3d-root {hm3d_root}; the originals are never written")


def scene_plan(out_root: Path, hm3d_root: Path, scene: str, scene_dir: str) -> dict:
    tree_dir = out_root / "hm3d" / scene
    return {
        "scene": scene,
        "scene_dir": scene_dir,
        "tree_dir": tree_dir,
        "glb_link": tree_dir / f"{scene}.glb",
        "glb_target": hm3d_root / scene_dir / f"{scene}.basis.glb",
        "navmesh": tree_dir / f"{scene}.navmesh",
        "meta": tree_dir / f"{scene}.navmesh.meta.json",
        "original_navmesh": hm3d_root / scene_dir / f"{scene}.basis.navmesh",
    }


def link_state(link: Path, target: Path) -> str:
    """"missing", "ok" or "conflict: ..." for a symlink that must point at ``target``."""
    if link.is_symlink():
        current = os.readlink(link)
        return "ok" if current == str(target) else f"conflict: links to {current}"
    if link.exists():
        return "conflict: exists and is not a symlink"
    return "missing"


def ensure_link(link: Path, target: Path) -> str | None:
    """Create ``link -> target`` unless it already is; the problem as text, or None."""
    state = link_state(link, target)
    if state.startswith("conflict"):
        return f"{link}: {state} (want {target})"
    if state == "missing":
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(str(target), str(link))
    return None


def tree_dir_problem(plan: Mapping[str, Any]) -> str | None:
    tree_dir = Path(plan["tree_dir"])
    if tree_dir.is_symlink():
        return f"{tree_dir} is a symlink; scene directories must be real"
    if tree_dir.exists() and not tree_dir.is_dir():
        return f"{tree_dir} exists and is not a directory"
    if Path(plan["navmesh"]).is_symlink():
        return f"{plan['navmesh']} is a symlink; the navmesh must be a file of the tree"
    return None


def read_meta(path: Path) -> dict | None:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return meta if isinstance(meta, dict) and meta.get("schema") == META_SCHEMA else None


def classify_navmesh(plan: Mapping[str, Any], settings_sha256: str | None, rebake: bool) -> tuple[str, str]:
    """``("bake" | "keep" | "fail", reason)`` for a scene's tree navmesh; ``settings_sha256`` None skips
    the settings check (dry run)."""
    navmesh, meta_path = Path(plan["navmesh"]), Path(plan["meta"])
    if Path(navmesh).is_symlink():
        return "fail", "the navmesh is a symlink"
    if not navmesh.exists():
        return "bake", "no navmesh yet"
    if rebake:
        return "bake", "--rebake"
    meta = read_meta(meta_path)
    if meta is None:
        return "bake", "navmesh without its meta file (interrupted bake)"
    if settings_sha256 is not None and meta.get("navmesh_settings_sha256") != settings_sha256:
        return "fail", "baked under other settings; --rebake to replace it"
    if meta.get("glb_target") != str(plan["glb_target"]):
        return "fail", f"baked from {meta.get('glb_target')}; --rebake to replace it"
    if sha256_file(navmesh) != meta.get("navmesh_sha256"):
        return "fail", "the navmesh changed after its bake; --rebake to replace it"
    return "keep", "baked earlier under the same settings"


def mp3d_ids(target: Path) -> set[str]:
    if not target.is_dir():
        return set()
    return {child.name for child in target.iterdir() if SCENE_ID_RE.fullmatch(child.name)}


# ----------------------------------------------------------------------------
# one scene (runs in a subprocess: --worker-task)
# ----------------------------------------------------------------------------
def pathfinder_summary(pf: Any, points: int = RANDOM_POINTS, seed: int = 0) -> dict:
    out: dict[str, Any] = {"is_loaded": bool(pf.is_loaded)}
    if not out["is_loaded"]:
        return out
    lower, upper = pf.get_bounds()
    out["bounds"] = [[float(v) for v in lower], [float(v) for v in upper]]
    area = getattr(pf, "navigable_area", None)
    if callable(area):
        area = area()
    out["navigable_area"] = None if area is None else float(area)
    pf.seed(seed)
    navigable = 0
    for _ in range(points):
        point = pf.get_random_navigable_point()
        if all(math.isfinite(float(v)) for v in point) and pf.is_navigable(point):
            navigable += 1
    out["random_points_navigable"] = navigable
    return out


def summaries_match(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if not (a.get("is_loaded") and b.get("is_loaded")):
        return False
    # The upper y bound is left out: on 0.1.7 the in-memory mesh's top reaches agent height plus a cell above
    # the saved one's (00481-5jLhtVmWd5F: 8.066 m in memory, 6.466 m read back), with identical polygons.
    (lo_a, hi_a), (lo_b, hi_b) = a["bounds"], b["bounds"]
    pairs = list(zip(lo_a, lo_b)) + [(hi_a[0], hi_b[0]), (hi_a[2], hi_b[2])]
    if any(abs(x - y) > BOUNDS_TOL for x, y in pairs):
        return False
    if a.get("random_points_navigable") != b.get("random_points_navigable"):
        return False
    area_a, area_b = a.get("navigable_area"), b.get("navigable_area")
    if area_a is None or area_b is None:
        return area_a is None and area_b is None
    return abs(area_a - area_b) <= AREA_RTOL * max(1.0, abs(area_a))


def compare_with_original(hs: Any, original: Mapping[str, Any], navmesh: Path, baked: Mapping[str, Any]) -> dict:
    """Report only, never a failure: the scene's own HM3D navmesh, when this habitat-sim reads it (Recast set
    v1; EXP-18 found 562 of the 740), against the rebaked one -- bounds, navigable area, and how many seeded
    random points of the rebaked navmesh the original calls navigable.  A mesh loaded in another frame, or a
    bake with another agent size than HM3D's own, shows here before any episode is filtered on it."""
    header = original.get("header")
    if not (header and header.get("magic_ok") and header.get("version") == RECAST_SET_VERSION):
        return {"compared": False, "reason": f"original navmesh {header_label(header)}"}
    try:
        original_pf = hs.PathFinder()
        original_pf.load_nav_mesh(str(original["path"]))
        summary = pathfinder_summary(original_pf)
        if not summary["is_loaded"]:
            return {"compared": False, "reason": "original v1 navmesh does not load"}
        rebaked_pf = hs.PathFinder()
        rebaked_pf.load_nav_mesh(str(navmesh))
        rebaked_pf.seed(1)
        on_original = 0
        for _ in range(ORIGINAL_POINTS):
            point = rebaked_pf.get_random_navigable_point()
            if all(math.isfinite(float(v)) for v in point) and original_pf.is_navigable(point):
                on_original += 1
        bounds_diff = max(abs(x - y) for row_a, row_b in zip(baked["bounds"], summary["bounds"])
                          for x, y in zip(row_a, row_b))
        area_ratio = None
        if baked.get("navigable_area") is not None and summary.get("navigable_area"):
            area_ratio = baked["navigable_area"] / summary["navigable_area"]
        return {"compared": True, "original_pathfinder": summary, "bounds_max_abs_diff": bounds_diff,
                "area_ratio": area_ratio, "rebaked_points_navigable_on_original": on_original,
                "points": ORIGINAL_POINTS}
    except Exception as exc:  # noqa: BLE001 - a report, not a check
        return {"compared": False, "reason": f"{type(exc).__name__}: {exc}"}


def make_meta(task: Mapping[str, Any], **fields: Any) -> dict:
    out_root = Path(task["out_root"])
    meta = {
        "schema": META_SCHEMA,
        "scene": task["scene"],
        "scene_dir": task["scene_dir"],
        "glb": os.path.relpath(task["glb_link"], out_root),
        "glb_target": task["glb_target"],
        "glb_target_realpath": os.path.realpath(task["glb_target"]),
        "navmesh": os.path.relpath(task["navmesh"], out_root),
        "navmesh_settings": task["navmesh_settings"],
        "navmesh_settings_sha256": task["navmesh_settings_sha256"],
        "include_static_objects": INCLUDE_STATIC_OBJECTS,
        "habitat_sim_version": task["habitat_sim_version"],
        "agent": {"height": task["agent_height"], "radius": task["agent_radius"]},
        "original_navmesh": task["original_navmesh"],
        "baked_utc": utc_now(),
    }
    meta.update(fields)
    return meta


def _camera_spec(hs: Any) -> Any:
    """The 64x64 colour camera the client's GL pre-initialisation uses (r2r_val_unseen.py)."""
    spec_cls = getattr(hs, "CameraSensorSpec", None) or hs.SensorSpec
    sensor = spec_cls()
    sensor.uuid = "rgb"
    sensor.sensor_type = hs.SensorType.COLOR
    if hasattr(hs, "SensorSubType"):
        sensor.sensor_subtype = hs.SensorSubType.PINHOLE
    sensor.resolution = [64, 64]
    return sensor


def open_simulator(hs: Any, task: Mapping[str, Any]) -> tuple[Any, bool]:
    """A Simulator on the tree's ``<ID>.glb`` (not the resolved file, so the navmesh lookup is the
    client's); returns it and whether textures were loaded."""
    sim_cfg = hs.SimulatorConfiguration()
    sim_cfg.scene_id = str(task["glb_link"])
    sim_cfg.gpu_device_id = 0
    if hasattr(sim_cfg, "enable_physics"):
        sim_cfg.enable_physics = False
    textures = True
    if task["skip_textures"] and hasattr(sim_cfg, "requires_textures"):
        sim_cfg.requires_textures = False
        textures = False
    agent = hs.agent.AgentConfiguration()
    agent.height = task["agent_height"]
    agent.radius = task["agent_radius"]
    agent.sensor_specifications = [_camera_spec(hs)]
    return hs.Simulator(hs.Configuration(sim_cfg, [agent])), textures


def bake_scene(task: Mapping[str, Any], hs: Any) -> dict:
    navmesh = Path(task["navmesh"])
    for stale in navmesh.parent.glob(f".{navmesh.name}.tmp*"):  # left by a worker killed at its timeout
        stale.unlink()
    temporary = navmesh.with_name(f".{navmesh.name}.tmp{os.getpid()}")
    started = time.time()
    sim, textures = open_simulator(hs, task)
    load_seconds = time.time() - started
    try:
        settings = make_navmesh_settings(hs, task["agent_radius"], task["agent_height"])
        if navmesh_settings_dict(settings) != task["navmesh_settings"]:
            raise RuntimeError(f"NavMeshSettings differ from the parent's: {navmesh_settings_dict(settings)}")
        bake_started = time.time()
        if not sim.recompute_navmesh(sim.pathfinder, settings) or not sim.pathfinder.is_loaded:
            raise RuntimeError("recompute_navmesh left no loaded pathfinder")
        bake_seconds = time.time() - bake_started
        if sim.pathfinder.save_nav_mesh(str(temporary)) is False or not temporary.is_file():
            raise RuntimeError(f"save_nav_mesh did not write {temporary}")
        in_memory = pathfinder_summary(sim.pathfinder)
    finally:
        sim.close()
    try:
        header = recast_set_header(temporary)
        if not (header and header["magic_ok"] and header["version"] == RECAST_SET_VERSION):
            raise RuntimeError(f"saved navmesh header {header}, want Recast set version {RECAST_SET_VERSION}")
        pf = hs.PathFinder()
        pf.load_nav_mesh(str(temporary))
        on_disk = pathfinder_summary(pf)
        if not on_disk["is_loaded"]:
            raise RuntimeError("a fresh PathFinder does not load the saved navmesh")
        if not summaries_match(in_memory, on_disk):
            raise RuntimeError(f"saved navmesh differs from the baked one: {in_memory} vs {on_disk}")
        if on_disk["random_points_navigable"] != RANDOM_POINTS:
            raise RuntimeError(f"only {on_disk['random_points_navigable']}/{RANDOM_POINTS} random points navigable")
        navmesh_sha256, navmesh_bytes = sha256_file(temporary), temporary.stat().st_size
        Path(task["meta"]).unlink(missing_ok=True)  # a navmesh without meta is rebaked by the next run
        os.replace(temporary, navmesh)
        os.chmod(navmesh, 0o444)
    finally:
        if temporary.exists():
            temporary.unlink()
    simulator_check = None
    if task["verify_sim"]:
        sim, _ = open_simulator(hs, task)
        try:
            client_view = pathfinder_summary(sim.pathfinder)
        finally:
            sim.close()
        simulator_check = {"is_loaded": client_view["is_loaded"], "matches_file": summaries_match(on_disk, client_view)}
        if not (simulator_check["is_loaded"] and simulator_check["matches_file"]):
            raise RuntimeError(f"a Simulator on {task['glb_link']} does not load this navmesh: {client_view}")
    vs_original = compare_with_original(hs, task["original_navmesh"], navmesh, on_disk)
    glb = Path(task["glb_target"])
    meta = make_meta(
        task,
        navmesh_sha256=navmesh_sha256,
        navmesh_bytes=navmesh_bytes,
        recast_set=header,
        glb_sha256=sha256_file(glb),
        glb_bytes=glb.stat().st_size,
        textures_loaded=textures,
        pathfinder=on_disk,
        simulator_check=simulator_check,
        vs_original=vs_original,
        scene_load_seconds=round(load_seconds, 3),
        bake_seconds=round(bake_seconds, 3),
    )
    atomic_write_json(Path(task["meta"]), meta)
    return meta


def verify_kept(task: Mapping[str, Any], hs: Any) -> dict:
    navmesh = Path(task["navmesh"])
    meta = read_meta(Path(task["meta"]))
    if meta is None:
        raise RuntimeError("meta file vanished")
    header = recast_set_header(navmesh)
    if not (header and header["magic_ok"] and header["version"] == RECAST_SET_VERSION):
        raise RuntimeError(f"navmesh header {header}, want Recast set version {RECAST_SET_VERSION}")
    if sha256_file(navmesh) != meta["navmesh_sha256"]:
        raise RuntimeError("navmesh sha256 differs from its meta file")
    pf = hs.PathFinder()
    pf.load_nav_mesh(str(navmesh))
    on_disk = pathfinder_summary(pf)
    if not on_disk["is_loaded"] or on_disk["random_points_navigable"] != RANDOM_POINTS:
        raise RuntimeError(f"kept navmesh fails the PathFinder check: {on_disk}")
    if not summaries_match(meta["pathfinder"], on_disk):
        raise RuntimeError(f"kept navmesh loads differently than at its bake: {on_disk}")
    if sha256_file(Path(task["glb_target"])) != meta["glb_sha256"]:
        raise RuntimeError("the source glb changed since the bake; --rebake")
    return meta


def worker_main(task_path: Path) -> int:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {"scene": task["scene"], "action": task["action"], "ok": False}
    started = time.time()
    try:
        hs = import_habitat_sim()
        version = str(getattr(hs, "__version__", "unknown"))
        if version != task["habitat_sim_version"]:
            raise RuntimeError(f"habitat_sim {version} in the worker, {task['habitat_sim_version']} in the parent")
        result["meta"] = bake_scene(task, hs) if task["action"] == "bake" else verify_kept(task, hs)
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001 - every failure is reported per scene
        traceback.print_exc()
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["seconds"] = round(time.time() - started, 3)
    atomic_write_json(Path(task["result"]), result)
    return 0 if result["ok"] else 1


# ----------------------------------------------------------------------------
# running the scenes
# ----------------------------------------------------------------------------
def tcp_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def start_xvfb(number: int, xvfb_bin: str, log_path: Path) -> subprocess.Popen:
    if tcp_open(6000 + number):
        raise SystemExit(f"display :{number} is already active; choose another --xvfb-base")
    with open(log_path, "ab") as handle:
        proc = subprocess.Popen([xvfb_bin, f":{number}", *XVFB_ARGS], env={**os.environ, **GL_ENV},
                                stdout=handle, stderr=subprocess.STDOUT)
    for _ in range(60):
        if tcp_open(6000 + number):
            return proc
        if proc.poll() is not None:
            break
        time.sleep(1)
    proc.kill()
    raise SystemExit(f"Xvfb :{number} did not come up; see {log_path}")


def child_env(display: str) -> dict:
    env = {**os.environ, **GL_ENV, "DISPLAY": display}
    if env.get("__GLX_VENDOR_LIBRARY_NAME") == "nvidia":  # as r2r_val_unseen.py: NVIDIA GLX is unstable here
        env.pop("__GLX_VENDOR_LIBRARY_NAME")
    return env


def _tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except FileNotFoundError:
        return ""


def run_scene_child(task: Mapping[str, Any], display: str, log_path: Path, timeout: float) -> dict:
    """Run one scene's task in a fresh interpreter; its result, or why there is none."""
    task_path = Path(task["result"]).with_name(f"{task['scene']}.task.json")
    atomic_write_json(task_path, dict(task))
    result_path = Path(task["result"])
    if result_path.exists():
        result_path.unlink()
    command = [sys.executable, "-u", str(SCRIPT), "--worker-task", str(task_path)]
    with open(log_path, "ab") as handle:
        try:
            returncode: Any = subprocess.run(command, env=child_env(display), stdout=handle,
                                             stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            returncode = f"timeout after {timeout:.0f} s"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        result = {"scene": task["scene"], "action": task["action"], "ok": False,
                  "error": f"worker ended ({returncode}) without a result; log tail:\n{_tail(log_path)}"}
    result["returncode"] = returncode
    result["log"] = str(log_path)
    return result


def run_tasks(tasks: Sequence[dict], displays: Sequence[str], log_dir: Path, timeout: float) -> list[dict]:
    slots: queue.Queue = queue.Queue()
    for display in displays:
        slots.put(display)
    done = [0]

    def one(task: dict) -> dict:
        display = slots.get()
        try:
            result = run_scene_child(task, display, log_dir / f"{task['scene']}.log", timeout)
        finally:
            slots.put(display)
        with _PRINT_LOCK:
            done[0] += 1
            count = done[0]
        if result.get("ok"):
            meta = result.get("meta", {})
            log(f"[{count}/{len(tasks)}] {task['scene']} {task['action']} ok {result.get('seconds')} s "
                f"navmesh sha256 {str(meta.get('navmesh_sha256'))[:16]}; "
                f"{describe_vs_original(meta.get('vs_original'))}")
        else:
            first = str(result.get("error", "")).splitlines()[0] if result.get("error") else ""
            log(f"[{count}/{len(tasks)}] {task['scene']} {task['action']} FAILED: {first} (log {result.get('log')})")
        return result

    with ThreadPoolExecutor(max_workers=max(1, len(displays))) as pool:
        return list(pool.map(one, tasks))


# ----------------------------------------------------------------------------
# the manifest
# ----------------------------------------------------------------------------
def describe_vs_original(row: Mapping[str, Any] | None) -> str:
    if not row:
        return "vs original: not recorded"
    if not row.get("compared"):
        return f"vs original: not compared ({row.get('reason')})"
    area = row.get("area_ratio")
    return (f"vs original v1: bounds diff {row['bounds_max_abs_diff']:.3f} m, area ratio "
            f"{'n/a' if area is None else f'{area:.3f}'}, {row['rebaked_points_navigable_on_original']}/"
            f"{row['points']} rebaked points navigable on it")


def _spread(values: Sequence[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"min": ordered[0], "median": ordered[len(ordered) // 2], "max": ordered[-1]}


def summarize_vs_original(scenes: Mapping[str, Mapping[str, Any]]) -> dict:
    """Report only: the rebaked navmeshes against HM3D's own where it is Recast set v1."""
    compared = {s: m["vs_original"] for s, m in scenes.items() if (m.get("vs_original") or {}).get("compared")}
    reasons = Counter(str((m.get("vs_original") or {}).get("reason", "not recorded"))
                      for s, m in scenes.items() if s not in compared)
    fractions = {s: r["rebaked_points_navigable_on_original"] / r["points"] for s, r in compared.items()}
    return {
        "report_only": True,
        "compared": len(compared),
        "not_compared": dict(reasons),
        "bounds_max_abs_diff_m": _spread([r["bounds_max_abs_diff"] for r in compared.values()]),
        "area_ratio": _spread([r["area_ratio"] for r in compared.values() if r.get("area_ratio") is not None]),
        "rebaked_points_navigable_on_original_fraction": _spread(list(fractions.values())),
        f"scenes_bounds_diff_over_{LOOK_BOUNDS_DIFF_M:g}m": sorted(
            s for s, r in compared.items() if r["bounds_max_abs_diff"] > LOOK_BOUNDS_DIFF_M),
        f"scenes_points_fraction_under_{LOOK_POINTS_FRACTION:g}": sorted(
            s for s, f in fractions.items() if f < LOOK_POINTS_FRACTION),
    }


def scan_tree(out_root: Path, settings_sha256: str) -> tuple[dict[str, dict], list[dict]]:
    """Every ready scene of ``<out-root>/hm3d`` (meta file plus cheap re-checks), and the not-ready ones."""
    scenes: dict[str, dict] = {}
    problems: list[dict] = []
    hm3d_tree = out_root / "hm3d"
    if not hm3d_tree.is_dir():
        return scenes, problems
    for tree_dir in sorted(hm3d_tree.iterdir()):
        if not (tree_dir.is_dir() and SCENE_ID_RE.fullmatch(tree_dir.name)):
            continue
        scene = tree_dir.name
        navmesh, glb_link = tree_dir / f"{scene}.navmesh", tree_dir / f"{scene}.glb"
        meta = read_meta(tree_dir / f"{scene}.navmesh.meta.json")
        error = None
        if not navmesh.is_file() or meta is None:
            error = "no navmesh with its meta file"
        elif meta.get("navmesh_settings_sha256") != settings_sha256:
            error = "baked under other settings"
        elif sha256_file(navmesh) != meta.get("navmesh_sha256"):
            error = "navmesh sha256 differs from its meta file"
        elif not glb_link.is_symlink() or os.readlink(glb_link) != meta.get("glb_target"):
            error = f"{glb_link.name} does not link to {meta.get('glb_target')}"
        elif not glb_link.is_file():
            error = f"{glb_link.name} is a dangling link"
        if error:
            problems.append({"scene": scene, "error": error})
        else:
            scenes[scene] = meta
    return scenes, problems


def build_manifest(
    *, out_root: Path, hm3d_root: Path, mp3d: dict, params: dict, options: dict, requested: dict,
    requested_scenes: Mapping[str, str], results: Mapping[str, dict], failures: list[dict], log_dir: Path,
) -> dict:
    """``requested_scenes``: ``{ID: 00xxx-ID}`` of this run."""
    scenes, tree_problems = scan_tree(out_root, params["navmesh_settings_sha256"])
    requested_set = set(requested_scenes)
    failed = {f.get("scene") for f in failures}
    for problem in tree_problems:
        if problem["scene"] in requested_set and problem["scene"] not in failed:
            failures.append({"scene": problem["scene"], "stage": "tree", "error": problem["error"]})
    for scene, result in results.items():  # a scene that failed this run is not ready, whatever is on disk
        if not result.get("ok"):
            scenes.pop(scene, None)
    for scene, meta in scenes.items():
        result = results.get(scene)
        meta["requested_this_run"] = scene in requested_set
        meta["status_this_run"] = None if result is None else ("baked" if result["action"] == "bake" else "kept")
    ready = sorted(s for s in requested_scenes if s in scenes)
    return {
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "out_root": str(out_root),
        "hm3d_root": str(hm3d_root),
        "mp3d": mp3d,
        "habitat_sim_version": params["habitat_sim_version"],
        "client": params["client"],
        "agent": {"height": params["agent_height"], "radius": params["agent_radius"]},
        "navmesh_settings": params["navmesh_settings"],
        "navmesh_settings_sha256": params["navmesh_settings_sha256"],
        "bake": dict(options, include_static_objects=INCLUDE_STATIC_OBJECTS, recast_set_version=RECAST_SET_VERSION,
                     settings_rule="NavMeshSettings.set_defaults(); agent_radius/agent_height = client AGENT_0"),
        "requested": dict(requested, ready=len(ready), statuses=dict(Counter(
            ("baked" if r["action"] == "bake" else "kept") if r.get("ok") else "failed" for r in results.values())),
            scenes=dict(sorted(requested_scenes.items()))),
        "scene_count": len(scenes),
        "all_requested_ready": not failures and len(ready) == len(requested_scenes),
        "original_navmesh_versions": dict(Counter(
            header_label((m.get("original_navmesh") or {}).get("header")) for m in scenes.values())),
        "vs_original_navmesh": summarize_vs_original({s: scenes[s] for s in ready}),
        "failures": failures,
        "tree_problems_not_requested": [p for p in tree_problems if p["scene"] not in requested_set],
        "log_dir": str(log_dir),
        "scenes": scenes,
    }


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--worker-task", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--scenes-file", type=Path, action="append", default=[])
    parser.add_argument("--scenes-from-dir", type=Path, action="append", default=[],
                        help="take every child directory with meta/episodes.jsonl (InternData-N1 scene dirs, "
                             "00xxx-<ID>), as build_hm3d_episodes.py does")
    parser.add_argument("--scenes", action="append", default=[], help="comma-separated scene entries")
    parser.add_argument("--all-scenes", action="store_true", help="every 00xxx-<ID> directory of --hm3d-root")
    parser.add_argument("--expect-count", type=int, help="fail unless exactly this many scenes resolve")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--hm3d-root", type=Path, default=DEFAULT_HM3D_ROOT)
    parser.add_argument("--mp3d-target", type=Path, default=DEFAULT_MP3D_TARGET)
    parser.add_argument("--client-script", type=Path, default=DEFAULT_CLIENT)
    parser.add_argument("--manifest", type=Path, help=f"default <out-root>/{MANIFEST_NAME}")
    parser.add_argument("--log-dir", type=Path, help="default <out-root>/prepare_logs/<UTC stamp>")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds per scene")
    parser.add_argument("--xvfb-base", type=int, default=770, help="worker i gets Xvfb :<base+i>")
    parser.add_argument("--xvfb-bin", default=shutil.which("Xvfb") or "Xvfb")
    parser.add_argument("--no-xvfb", action="store_true", help="use the inherited DISPLAY for every worker")
    parser.add_argument("--habitat-sim-version", default=EXPECTED_HABITAT_SIM, help="'' accepts any")
    parser.add_argument("--rebake", action="store_true", help="rebake requested scenes even if a navmesh exists")
    parser.add_argument("--skip-textures", action="store_true",
                        help="load scenes without textures when habitat-sim offers requires_textures")
    parser.add_argument("--no-verify-sim", action="store_true", help="skip the second, client-style Simulator load")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; write nothing, import no habitat")
    args = parser.parse_args(argv)
    if args.worker_task is None:
        sources = bool(args.scenes_file or args.scenes_from_dir or args.scenes)
        if args.all_scenes == sources:
            parser.error("give --all-scenes, or one or more of --scenes-file / --scenes-from-dir / --scenes")
        if args.workers < 1:
            parser.error("--workers must be >= 1")
    return args


def collect_entries(args: argparse.Namespace, index: Mapping[str, Sequence[str]]) -> tuple[list[str], list[dict]]:
    entries: list[str] = []
    sources: list[dict] = []
    if args.all_scenes:
        entries = [dirs[0] for dirs in index.values() if len(dirs) == 1]
        entries += [scene for scene, dirs in index.items() if len(dirs) != 1]  # resolves to an error
        sources.append({"kind": "all-scenes", "root": str(args.hm3d_root), "entries": len(entries)})
    for path in args.scenes_file:
        found = read_scene_entries(path)
        entries += found
        sources.append({"kind": "scenes-file", "path": str(path), "sha256": sha256_file(path), "entries": len(found)})
    for directory in args.scenes_from_dir:
        found, skipped = scene_entries_from_dir(directory)
        entries += found
        sources.append({"kind": "scenes-from-dir", "path": str(directory), "entries": len(found),
                        "skipped_without_meta_episodes": skipped})
        if skipped:
            log(f"--scenes-from-dir {directory}: skipped {len(skipped)} dirs without meta/episodes.jsonl: "
                f"{skipped[:10]}{' ...' if len(skipped) > 10 else ''}")
    for text in args.scenes:
        found = [item.strip() for item in text.split(",") if item.strip()]
        entries += found
        sources.append({"kind": "scenes", "entries": len(found)})
    return entries, sources


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_task is not None:
        return worker_main(args.worker_task)

    out_root, hm3d_root = args.out_root.absolute(), args.hm3d_root.absolute()
    mp3d_target = args.mp3d_target.absolute()
    check_roots(out_root, hm3d_root)
    index = index_hm3d_root(hm3d_root)
    entries, sources = collect_entries(args, index)
    resolved, problems, duplicates = resolve_scene_entries(entries, index)
    failures: list[dict] = [{"scene": None, "stage": "resolve", **p} for p in problems]
    if args.expect_count is not None and len(resolved) != args.expect_count:
        failures.append({"scene": None, "stage": "resolve",
                         "error": f"{len(resolved)} scenes resolved, --expect-count {args.expect_count}"})
    requested = {"sources": sources, "entries": len(entries), "resolved": len(resolved), "duplicates": duplicates}

    mp3d_link = out_root / "mp3d"
    mp3d = {"link": str(mp3d_link), "target": str(mp3d_target), "target_exists": mp3d_target.is_dir(),
            "state": link_state(mp3d_link, mp3d_target)}
    if not mp3d["target_exists"]:
        failures.append({"scene": None, "stage": "tree", "error": f"--mp3d-target {mp3d_target} is not a directory"})
    if mp3d["state"].startswith("conflict"):
        failures.append({"scene": None, "stage": "tree", "error": f"{mp3d_link}: {mp3d['state']}"})
    # Only the MP3D scenes under --mp3d-target; build_hm3d_episodes.py adds R2R train's for the full check.
    known_mp3d = mp3d_ids(mp3d_target)
    mp3d["ids_checked"] = len(known_mp3d)
    collisions = sorted({scene for scene, _ in resolved} & known_mp3d)
    if collisions:
        failures.append({"scene": None, "stage": "tree", "error": f"HM3D IDs also MP3D scene IDs: {collisions}"})

    plans = [scene_plan(out_root, hm3d_root, scene, scene_dir) for scene, scene_dir in resolved]
    for plan in plans:
        header = recast_set_header(plan["original_navmesh"])
        plan["original"] = {"path": str(plan["original_navmesh"]), "exists": header is not None, "header": header}
        if not plan["glb_target"].is_file() or plan["glb_target"].stat().st_size == 0:
            plan["problem"] = ("source", f"no {plan['glb_target'].name} in {plan['scene_dir']}")
        else:
            problem = tree_dir_problem(plan)
            state = link_state(plan["glb_link"], plan["glb_target"])
            if problem:
                plan["problem"] = ("tree", problem)
            elif state.startswith("conflict"):
                plan["problem"] = ("tree", f"{plan['glb_link']}: {state} (want {plan['glb_target']})")
            plan["link_state"] = state

    if args.dry_run:
        actions: Counter = Counter()
        for plan in plans:
            if "problem" in plan:
                verdict, reason = "fail", plan["problem"][1]
            else:
                verdict, reason = classify_navmesh(plan, None, args.rebake)
            actions[verdict] += 1
            if verdict == "fail":
                failures.append({"scene": plan["scene"], "stage": plan.get("problem", ("navmesh",))[0],
                                 "error": reason})
            print(f"[dry-run] {plan['scene']} {plan['scene_dir']} link={plan.get('link_state', '-')} "
                  f"navmesh={verdict} ({reason}) original={header_label(plan['original']['header'])}", flush=True)
        static = inspect_client_builder(args.client_script.read_text(encoding="utf-8"))
        print(json.dumps({
            "dry_run": True, "out_root": str(out_root), "hm3d_root": str(hm3d_root), "requested": requested,
            "mp3d": mp3d, "mp3d_id_collisions": collisions, "client_builder": static,
            "navmesh_actions": dict(actions), "rebake": args.rebake,
            "original_navmesh_versions": dict(Counter(header_label(p["original"]["header"]) for p in plans)),
            "failures": failures,
        }, indent=1, sort_keys=True), flush=True)
        return 0 if not failures else 2

    lock = lock_tree(out_root)  # held until the manifest is written (or the process ends)
    params = resolve_bake_parameters(args.client_script, out_root, args.habitat_sim_version)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    log_dir = (args.log_dir or out_root / "prepare_logs" / stamp).absolute()
    log_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "hm3d").mkdir(parents=True, exist_ok=True)
    if mp3d["target_exists"] and not mp3d["state"].startswith("conflict"):
        ensure_link(mp3d_link, mp3d_target)
        mp3d["state"] = link_state(mp3d_link, mp3d_target)

    tasks = []
    for plan in plans:
        if "problem" in plan:
            stage, error = plan["problem"]
            failures.append({"scene": plan["scene"], "stage": stage, "error": error})
            continue
        problem = ensure_link(plan["glb_link"], plan["glb_target"])
        if problem:
            failures.append({"scene": plan["scene"], "stage": "tree", "error": problem})
            continue
        action, reason = classify_navmesh(plan, params["navmesh_settings_sha256"], args.rebake)
        if action == "fail":
            failures.append({"scene": plan["scene"], "stage": "navmesh", "error": reason})
            continue
        tasks.append({
            "scene": plan["scene"], "scene_dir": plan["scene_dir"], "action": action, "reason": reason,
            "out_root": str(out_root), "glb_link": str(plan["glb_link"]), "glb_target": str(plan["glb_target"]),
            "navmesh": str(plan["navmesh"]), "meta": str(plan["meta"]), "original_navmesh": plan["original"],
            "result": str(log_dir / f"{plan['scene']}.result.json"),
            "agent_height": params["agent_height"], "agent_radius": params["agent_radius"],
            "navmesh_settings": params["navmesh_settings"],
            "navmesh_settings_sha256": params["navmesh_settings_sha256"],
            "habitat_sim_version": params["habitat_sim_version"],
            "skip_textures": args.skip_textures, "verify_sim": not args.no_verify_sim,
        })
    log(f"{len(resolved)} scenes: {sum(t['action'] == 'bake' for t in tasks)} to bake, "
        f"{sum(t['action'] == 'keep' for t in tasks)} to re-check, {len(failures)} failures so far; logs {log_dir}")

    results: dict[str, dict] = {}
    xvfbs: list[subprocess.Popen] = []
    try:
        if tasks:
            workers = min(args.workers, len(tasks))
            if args.no_xvfb:
                if not os.environ.get("DISPLAY"):
                    raise SystemExit("--no-xvfb needs DISPLAY")
                displays = [os.environ["DISPLAY"]] * workers
            else:
                displays = []
                for slot in range(workers):
                    number = args.xvfb_base + slot
                    xvfbs.append(start_xvfb(number, args.xvfb_bin, log_dir / f"xvfb_{number}.log"))
                    displays.append(f"127.0.0.1:{number}.0")
            for result in run_tasks(tasks, displays, log_dir, args.timeout):
                results[result["scene"]] = result
                if not result.get("ok"):
                    failures.append({"scene": result["scene"], "stage": result["action"],
                                     "error": result.get("error"), "returncode": result.get("returncode"),
                                     "log": result.get("log")})
    finally:
        for proc in xvfbs:
            proc.terminate()
        for proc in xvfbs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    options = {"skip_textures": args.skip_textures, "verify_sim": not args.no_verify_sim, "rebake": args.rebake,
               "workers": args.workers, "timeout_s": args.timeout}
    manifest = build_manifest(
        out_root=out_root, hm3d_root=hm3d_root, mp3d=dict(mp3d, mp3d_id_collisions=collisions), params=params,
        options=options, requested=requested, requested_scenes=dict(resolved),
        results=results, failures=failures, log_dir=log_dir)
    manifest_path = (args.manifest or out_root / MANIFEST_NAME).absolute()
    atomic_write_json(manifest_path, manifest)
    lock.close()
    print(json.dumps({
        "manifest": str(manifest_path), "scene_count": manifest["scene_count"],
        "requested": manifest["requested"]["resolved"], "ready": manifest["requested"]["ready"],
        "statuses": manifest["requested"]["statuses"], "all_requested_ready": manifest["all_requested_ready"],
        "original_navmesh_versions": manifest["original_navmesh_versions"],
        "vs_original_navmesh": manifest["vs_original_navmesh"],
        "failures": len(failures), "first_failures": failures[:10],
    }, indent=1, sort_keys=True, default=str), flush=True)
    return 0 if manifest["all_requested_ready"] else 2


if __name__ == "__main__":
    sys.exit(main())

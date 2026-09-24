"""Paper-quality orthographic top-down maps for EXP-18 (habitat-sim 0.1.7, ``envs/vlnce``).

For every scene: detect floor levels on the navmesh (area-weighted height
histogram; floors chained by height gaps <= ``--level-merge`` become sub-levels
of the same map: split-level floors, sunken garages; different levels'
navmask slices do not overlap).  For each level render an orthographic RGB
view from ``--cam-height`` above its main navmesh height, and one more from
``--cam-height`` above each sub-level below the main floor, each clipping
geometry more than ``--below`` under its own height; the map composites them
per pixel (``composite``).  Each map is cropped to the level's navigable
bounds plus ``--margin``, at ``--mpp`` m/px with the longest side capped at
``--max-px``.  Next to it go the level's
navmesh mask on the *same* pixel grid, an RGBA copy in which everything far
from the level's floor footprint fades to transparent white (hides other
floors, exterior scan junk, skylights), and ``topdown.json`` with the exact
transform.  MP3D scenes also get room polygons + labels per level from the
``.house`` file (and those polygons join the fade footprint).

Output (resumable: a scene whose ``topdown.json`` exists is skipped)::

    <out>/<scene>/level_<i>_rgb.png      RGB, white where nothing was hit
    <out>/<scene>/level_<i>_rgba.png     faded copy, alpha = footprint weight
    <out>/<scene>/level_<i>_navmask.png  255 = navigable within +-0.5 m of a sub-level height
    <out>/<scene>/topdown.json           transform, levels, rooms, QA, timings

Pixel grid of level i (see ``topdown_io.py`` for loaders and heading helpers)::

    col = (x - x0)/mpp - 0.5,  row = (z - z0)/mpp - 0.5     (integer = pixel centre)
    image right = +x, image down = +z; imshow extent = [x0, x0 + W*mpp, z0 + H*mpp, z0]

The grid is aligned to habitat's ``get_topdown_view`` lattice (``lo + k*mpp``),
so the navmask is sliced from it exactly, without resampling.  Stair points
more than 0.5 m from every sub-level are off every navmask by construction.

Needs an X display with llvmpipe GLX: run through ``run_topdown.sh`` (many
scenes, one Xvfb per process) or ``with_xvfb.sh`` (one process).  Example::

    DISPLAY_NUM=371 bash scripts/exp18/topdown/with_xvfb.sh \\
        -m scripts.exp18.topdown.render_topdown --scenes 2azQ1b91cZZ kfPV7w3FaU5
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import json
import math
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.exp18 import common  # noqa: E402
from scripts.exp18.topdown.mp3d_house import parse_house, rooms_by_level  # noqa: E402
from scripts.exp18.topdown.topdown_io import SCHEMA, Level, canonical_scene_id  # noqa: E402

CONVENTIONS = (
    "habitat world, y up. Map = orthographic view from +y, image right=+x, image down=+z (not mirrored). "
    "col=(x-x0)/mpp-0.5, row=(z-z0)/mpp-0.5, integer=pixel centre; imshow extent=[x0,x0+W*mpp,z0+H*mpp,z0]. "
    "Heading: yaw about +y, forward=(-sin yaw,-cos yaw) in (x,z) = (col,row) direction; from camera c2w forward=-R[:,2]."
)


# ------------------------------------------------------------------ scenes ---

def _scene_from_key(key: str) -> str:
    key = key.strip()
    if key.endswith(".glb") or key.endswith(".navmesh"):
        return canonical_scene_id(key)
    return canonical_scene_id(key.split("/")[0])


def scenes_from_manifest(path: Path) -> list:
    """Scene ids from a text list (``scene`` or ``scene/clip`` per line) or a JSON(.gz) manifest.

    JSON is searched for string lists and ``scene`` / ``scene_id`` / ``scene_name``
    fields under ``scenes`` / ``episodes`` / ``clips`` / ``selected`` / ``items``.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        text = handle.read()
    found = []
    if text.lstrip().startswith(("{", "[")):
        def visit(obj):
            if isinstance(obj, dict):
                for key in ("scene", "scene_id", "scene_name"):
                    if isinstance(obj.get(key), str):
                        found.append(obj[key])
                for key in ("scenes", "episodes", "clips", "selected", "items"):
                    if key in obj:
                        visit(obj[key])
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, str):
                        found.append(item)
                    else:
                        visit(item)
        visit(json.loads(text))
    else:
        found = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    scenes = []
    for key in found:
        scene = _scene_from_key(key)
        if scene and scene not in scenes:
            scenes.append(scene)
    return scenes


def scenes_for_tier(tier: str, manifest=None) -> list:
    """Scenes of a pre-registered tier (A-E, see ledger EXP-18)."""
    tier = tier.upper()
    if tier in ("A", "B"):
        path = Path(manifest) if manifest else common.clip_list_path(tier)
        if path.exists():
            return sorted(scenes_from_manifest(path))
        val = set(common.R2R_V2_VAL_SCENES)
        names = sorted(p.name for p in Path(common.R2R_V2_ROOT).iterdir() if p.is_dir())
        return [s for s in names if (s in val) == (tier == "B")]
    if tier in ("C", "E"):
        with gzip.open(str(common.R2R_VAL_UNSEEN_EPISODES), "rt") as handle:
            episodes = json.load(handle)["episodes"]
        return sorted({canonical_scene_id(ep["scene_id"]) for ep in episodes})
    if tier == "D":
        path = Path(manifest) if manifest else common.clip_list_path("D")
        if not path.exists():
            raise SystemExit("tier D: no HM3D selection found at %s; pass --manifest or --scenes" % path)
        return sorted(scenes_from_manifest(path))
    raise SystemExit("unknown tier %r" % tier)


class SceneAssets:
    def __init__(self, mp3d_root: Path, hm3d_root: Path):
        self.mp3d_root = Path(mp3d_root)
        self.hm3d_root = Path(hm3d_root)
        self._hm3d_dirs = None

    def resolve(self, scene: str) -> dict:
        cid = canonical_scene_id(scene)
        glb = self.mp3d_root / cid / (cid + ".glb")
        if glb.exists():
            house = glb.with_suffix(".house")
            return {"scene": cid, "dataset": "mp3d", "glb": glb, "navmesh": glb.with_suffix(".navmesh"),
                    "house": house if house.exists() else None}
        if self._hm3d_dirs is None:
            self._hm3d_dirs = {}
            if self.hm3d_root.is_dir():
                for name in os.listdir(self.hm3d_root):
                    self._hm3d_dirs[canonical_scene_id(name)] = self.hm3d_root / name
        directory = self._hm3d_dirs.get(cid)
        if directory is None:
            raise FileNotFoundError("scene %s: not under %s or %s" % (cid, self.mp3d_root, self.hm3d_root))
        return {"scene": cid, "dataset": "hm3d", "glb": directory / (cid + ".basis.glb"),
                "navmesh": directory / (cid + ".basis.navmesh"), "house": None}


# ----------------------------------------------------------------- navmesh ---

NAV_SLICE_M = 0.5  # get_topdown_view(mpp, y): navigable within +-0.5 m of y (habitat-sim 0.1.7)


def floor_levels(ps, n: int = 20000, bin_m: float = 0.1, min_frac: float = 0.03, peak_win_m: float = 0.15,
                 merge_m: float = 1.1, max_up_m: float = 1.2, storey_m: float = 2.0, stack_m2: float = 4.0,
                 stack_mpp: float = 0.05, seed: int = 0) -> list:
    """Floor levels from an area-weighted histogram of random navigable points.

    ``get_random_navigable_point`` samples uniformly by area, so mass is a
    share of walkable area.  A local maximum of the ``bin_m`` histogram is a
    floor if the mass within ``peak_win_m`` of it reaches ``min_frac`` (a
    window, not the single bin: a floor whose heights straddle a bin edge
    counts in full); smaller platforms, stairs and most furniture tops are
    dropped, and peaks within 0.3 m of a larger one are plateau duplicates.

    Floors are chained single-linkage along height: consecutive floors at most
    ``merge_m`` apart are sub-levels of one level (split-level houses, sunken
    garages, a garden below the deck), one map per level, navmask = union of
    the sub-level slices.  So sub-levels of different levels are more than
    ``merge_m`` >= 2 * NAV_SLICE_M apart and no floor lands in two levels'
    navmasks.  A chain whose sub-levels >= ``storey_m`` apart have navmesh
    stacked over each other (>= ``stack_m2`` of shared x-z: storeys bridged by
    stair/landing peaks) is cut at its largest gap between them; such a cut
    may leave adjacent levels' slices overlapping, which ``slice_gap_below_m``
    records.  The main floor (``floor_y``, camera reference) is the largest
    sub-level within ``max_up_m`` of the top one.  Returns levels sorted by
    height: ``{"floor_y", "sub_levels", "y_min", "y_max", "area_frac",
    "slice_gap_below_m"}``.
    """
    if merge_m < 2 * NAV_SLICE_M:
        raise ValueError("merge_m=%.3f < %.1f m: levels' +-%.1f m navmask slices would overlap"
                         % (merge_m, 2 * NAV_SLICE_M, NAV_SLICE_M))
    ps.seed(seed)
    ys = np.array([ps.get_random_navigable_point()[1] for _ in range(n)], dtype=np.float64)
    ys = ys[np.isfinite(ys)]
    edges = np.arange(ys.min() - bin_m / 2, ys.max() + bin_m, bin_m)
    hist, edges = np.histogram(ys, bins=edges)
    peaks = []
    for i in range(len(hist)):
        left = hist[i - 1] if i > 0 else -1
        right = hist[i + 1] if i + 1 < len(hist) else -1
        if hist[i] >= left and hist[i] >= right:
            sel = np.abs(ys - 0.5 * (edges[i] + edges[i + 1])) < peak_win_m
            if sel.mean() >= min_frac:
                peaks.append((float(np.median(ys[sel])), float(sel.mean())))
    if not peaks:  # nothing reaches min_frac: fall back to the mode
        i = int(hist.argmax())
        sel = np.abs(ys - 0.5 * (edges[i] + edges[i + 1])) < peak_win_m
        peaks.append((float(np.median(ys[sel])), float(sel.mean())))
    unique = []  # plateau bins give near-duplicate peaks; keep the larger
    for y, frac in sorted(peaks, key=lambda c: -c[1]):
        if all(abs(y - other[0]) >= 0.3 for other in unique):
            unique.append((y, frac))
    chains = []  # single linkage along height
    for y, frac in sorted(unique):
        if chains and y - chains[-1][-1][0] <= merge_m:
            chains[-1].append((y, frac))
        else:
            chains.append([(y, frac)])

    slices = {}

    def coarse(y):
        if y not in slices:
            slices[y] = np.asarray(ps.get_topdown_view(stack_mpp, float(y)), dtype=bool)
        return slices[y]

    done, todo = [], chains
    while todo:
        chain = todo.pop()
        stacked = [((coarse(a) & coarse(b)).sum() * stack_mpp ** 2, i, j)
                   for i, (a, _) in enumerate(chain) for j, (b, _) in enumerate(chain[i + 1:], i + 1)
                   if b - a >= storey_m]
        stacked = [t for t in stacked if t[0] >= stack_m2]
        if not stacked:
            done.append(chain)
            continue
        _, i, j = max(stacked)
        k = max(range(i, j), key=lambda k: chain[k + 1][0] - chain[k][0])
        todo += [chain[:k + 1], chain[k + 1:]]
    levels = []
    for chain in sorted(done):
        subs = [y for y, _ in chain]
        main = max((c for c in chain if c[0] >= subs[-1] - max_up_m), key=lambda c: c[1])[0]
        near = np.min(np.abs(ys[:, None] - np.asarray(subs)[None]), axis=1) < 0.25
        levels.append({"floor_y": main, "sub_levels": subs, "y_min": subs[0], "y_max": subs[-1],
                       "area_frac": round(float(near.mean()), 4),
                       "slice_gap_below_m": None if not levels else round(subs[0] - levels[-1]["y_max"] - 2 * NAV_SLICE_M, 4)})
    return levels


def camera_plan(level: dict, cam_height: float, below: float) -> tuple:
    """Cameras of one level and the camera of each sub-level.

    Every sub-level at or below the main floor gets its own camera
    ``cam_height`` above it, clipping ``below`` under it (a sunken floor is not
    hidden under its own ceiling, and nothing is clipped by a far plane set for
    another height); sub-levels above the main floor (within ``max_up_m``)
    share the main floor's camera, which keeps furniture tops from pulling a
    camera up into the ceiling.  Camera heights never decrease with height.
    """
    cams, sub_cam = [], []
    for y in level["sub_levels"]:
        if y <= level["floor_y"] + 1e-6:
            cams.append({"sub_y": float(y), "cam_y": float(y + cam_height), "far_m": float(cam_height + below)})
        sub_cam.append(len(cams) - 1)
    return cams, sub_cam


def _slice_union(ps, mpp: float, heights) -> np.ndarray:
    """``get_topdown_view`` (navigable within +-0.5 m of the height) OR-ed over heights."""
    out = None
    for y in heights:
        view = np.asarray(ps.get_topdown_view(mpp, float(y)), dtype=bool)
        out = view if out is None else (out | view)
    return out


def level_grid(ps, level: dict, margin: float, target_mpp: float, max_px: int,
               coarse_mpp: float = 0.05) -> dict:
    """Pixel grid of one level: navigable bounds + margin, aligned to the navmesh lattice."""
    lo, hi = [np.asarray(b, dtype=np.float64) for b in ps.get_bounds()]
    coarse = _slice_union(ps, coarse_mpp, level["sub_levels"])
    rows, cols = np.nonzero(coarse)
    if len(rows):
        xmin, xmax = lo[0] + (cols.min() - 1) * coarse_mpp, lo[0] + (cols.max() + 1) * coarse_mpp
        zmin, zmax = lo[2] + (rows.min() - 1) * coarse_mpp, lo[2] + (rows.max() + 1) * coarse_mpp
    else:
        xmin, xmax, zmin, zmax = lo[0], hi[0], lo[2], hi[2]
    span = max(xmax - xmin, zmax - zmin) + 2 * margin
    mpp = max(float(target_mpp), span / (max_px - 3))
    c0 = int(math.floor((xmin - margin - lo[0]) / mpp))
    c1 = int(math.ceil((xmax + margin - lo[0]) / mpp))
    r0 = int(math.floor((zmin - margin - lo[2]) / mpp))
    r1 = int(math.ceil((zmax + margin - lo[2]) / mpp))
    width, height = c1 - c0 + 1, r1 - r0 + 1
    x0 = lo[0] + (c0 - 0.5) * mpp
    z0 = lo[2] + (r0 - 0.5) * mpp
    return {
        "floor_y": float(level["floor_y"]), "sub_levels": [float(y) for y in level["sub_levels"]],
        "y_min": float(level["y_min"]), "y_max": float(level["y_max"]),
        "mpp": mpp, "width": width, "height": height,
        "x0": float(x0), "z0": float(z0), "lattice_origin_rc": [r0, c0],
        "center_xz": [float(x0 + width * mpp / 2), float(z0 + height * mpp / 2)],
        "nav_bounds_xz": [float(xmin), float(xmax), float(zmin), float(zmax)],
    }


def level_navmask(ps, grid: dict) -> tuple:
    """Navmesh mask on the level grid, sliced from ``get_topdown_view(mpp, y)``.

    Returns ``(union, per_sub)``: the level navmask and one mask per sub-level
    (the composite uses the latter to pick each pixel's camera).
    """
    r0, c0 = grid["lattice_origin_rc"]
    h, w = grid["height"], grid["width"]
    per_sub = []
    for y in grid["sub_levels"]:
        full = np.asarray(ps.get_topdown_view(grid["mpp"], float(y)), dtype=bool)
        out = np.zeros((h, w), dtype=bool)
        rs, re_ = max(r0, 0), min(r0 + h, full.shape[0])
        cs, ce = max(c0, 0), min(c0 + w, full.shape[1])
        if rs < re_ and cs < ce:
            out[rs - r0:re_ - r0, cs - c0:ce - c0] = full[rs:re_, cs:ce]
        per_sub.append(out)
    return np.logical_or.reduce(per_sub), per_sub


def composite(images: list, sub_masks: list, sub_cam: list) -> tuple:
    """Per-pixel camera choice for a level rendered by several cameras.

    A navigable pixel takes the camera of the lowest sub-level whose slice
    covers it (the lowest camera, so no ceiling); every other pixel (walls,
    furniture, margin) takes the camera of the nearest navigable pixel.
    Returns ``(rgb, label)``; label = camera index per pixel.
    """
    import cv2
    label = np.full(sub_masks[0].shape, -1, dtype=np.int16)
    for k in reversed(range(len(sub_masks))):
        label[sub_masks[k]] = sub_cam[k]
    covered = label >= 0
    if not covered.any():
        label[:] = sub_cam[0]
    elif not covered.all():
        _, idx = cv2.distanceTransformWithLabels(np.where(covered, 0, 255).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
        lut = np.zeros(int(idx.max()) + 1, dtype=np.int16)
        lut[idx[covered]] = label[covered]
        label = lut[idx]
    rgb = images[0].copy()
    for c in range(1, len(images)):
        pick = label == c
        rgb[pick] = images[c][pick]
    return rgb, label


# ------------------------------------------------------------------ render ---

class OrthoRenderer:
    """One habitat Simulator per process, reconfigured per scene (scene load ~5-20 s).

    Each camera is a sensor at its absolute world position (the agent stays at
    the origin, unrotated), so one ``get_sensor_observations`` renders them all.
    """

    def __init__(self, near: float):
        self.near = near
        self.sim = None

    def _spec(self, uuid: str, grid: dict, cam: dict):
        import habitat_sim
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = habitat_sim.SensorType.COLOR
        spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        spec.resolution = [grid["height"], grid["width"]]
        spec.position = [grid["center_xz"][0], cam["cam_y"], grid["center_xz"][1]]
        spec.orientation = [-math.pi / 2, 0.0, 0.0]          # straight down: image right=+x, down=+z
        spec.ortho_scale = 1.0 / (grid["width"] * grid["mpp"])  # visible width = 1/ortho_scale metres
        spec.near = self.near
        spec.far = cam["far_m"]                              # clips geometry `below` under its sub-level
        spec.clear_color = [1.0, 1.0, 1.0, 1.0]
        return spec

    def render(self, glb: Path, grids: list, cams: list) -> tuple:
        """``cams[i]`` = camera list of level i; returns (images[i][c], load_s, render_s)."""
        import habitat_sim
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(glb)
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [self._spec("level_%d_cam_%d" % (i, c), g, cam)
                                           for i, g in enumerate(grids) for c, cam in enumerate(cams[i])]
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        t0 = time.time()
        if self.sim is None:
            self.sim = habitat_sim.Simulator(cfg)
        else:
            try:
                self.sim.reconfigure(cfg)
            except Exception:  # fall back to a fresh simulator
                traceback.print_exc()
                self.close()
                self.sim = habitat_sim.Simulator(cfg)
        load_s = time.time() - t0
        state = habitat_sim.AgentState()
        state.position = np.zeros(3, dtype=np.float32)
        state.rotation = np.quaternion(1, 0, 0, 0)
        self.sim.get_agent(0).set_state(state)
        t1 = time.time()
        obs = self.sim.get_sensor_observations()
        render_s = time.time() - t1
        images = [[np.ascontiguousarray(obs["level_%d_cam_%d" % (i, c)][..., :3]) for c in range(len(cams[i]))]
                  for i in range(len(grids))]
        return images, load_s, render_s

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None


# -------------------------------------------------------------------- fade ---

def _dist_px(targets: np.ndarray) -> np.ndarray:
    """Euclidean distance (px) from every pixel to the nearest ``True`` pixel of ``targets``."""
    import cv2
    src = np.where(targets, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def rooms_mask(rooms: list, grid: dict) -> np.ndarray:
    """Rasterise MP3D room floor polygons onto the level grid (1/16 px vertex precision)."""
    import cv2
    mask = np.zeros((grid["height"], grid["width"]), dtype=np.uint8)
    for room in rooms:
        if not room.get("polygon_xz"):
            continue
        poly = np.asarray(room["polygon_xz"], dtype=np.float64)
        col = (poly[:, 0] - grid["x0"]) / grid["mpp"] - 0.5
        row = (poly[:, 1] - grid["z0"]) / grid["mpp"] - 0.5
        pts = np.rint(np.stack([col, row], axis=1) * 16).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1, lineType=cv2.LINE_8, shift=4)
    return mask.astype(bool)


def footprint_weight(navmask: np.ndarray, mpp: float, radius_m: float, feather_m: float,
                     close_m: float, fill_holes_m2: float, keep=None) -> tuple:
    """Per-pixel keep weight in [0, 1] and the floor footprint it is measured from.

    footprint = navmask (plus ``keep``, e.g. MP3D room polygons, which cover
    floor the navmesh skips), morphologically closed with a ``close_m`` disk
    (folds in beds, sofas and other furniture the navmesh flows around), plus
    enclosed non-footprint holes up to ``fill_holes_m2``.  weight = 1 within
    ``radius_m`` of the footprint, ramping to 0 over ``feather_m``.
    ``close_m = fill_holes_m2 = 0`` and no ``keep`` gives a plain
    distance-to-navmesh fade.
    """
    import cv2
    pad = int(math.ceil((close_m + radius_m + feather_m) / mpp)) + 2
    nav = np.pad(navmask if keep is None else (navmask | keep), pad)
    foot = nav
    if close_m > 0:
        r_px = close_m / mpp
        dilated = _dist_px(nav) <= r_px
        foot = (_dist_px(~dilated) > r_px) | nav
    if fill_holes_m2 > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats((~foot).astype(np.uint8), connectivity=4)
        border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        fill = stats[:, cv2.CC_STAT_AREA] * mpp * mpp <= fill_holes_m2
        fill[0] = False
        fill[border] = False
        if fill.any():
            foot = foot | fill[labels]
    dist_m = _dist_px(foot) * mpp
    if feather_m > 0:
        weight = np.clip(1.0 - (dist_m - radius_m) / feather_m, 0.0, 1.0)
    else:
        weight = (dist_m <= radius_m).astype(np.float32)
    sl = (slice(pad, -pad), slice(pad, -pad))
    return weight[sl].astype(np.float32), foot[sl]


def faded_rgba(rgb: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """RGB blended to white by ``1 - weight`` and alpha = weight (looks right with or without alpha)."""
    w = weight[..., None]
    out = rgb.astype(np.float32) * w + 255.0 * (1.0 - w)
    alpha = np.rint(weight * 255.0).astype(np.uint8)
    return np.dstack([np.rint(out).astype(np.uint8), alpha])


# --------------------------------------------------------------- per scene ---

def _save_png(array: np.ndarray, path: Path) -> None:
    from PIL import Image
    tmp = path.with_name(path.name + ".tmp")
    Image.fromarray(array).save(str(tmp), format="PNG")
    os.replace(str(tmp), str(path))


def _qa_points(ps, level_meta: list, scene_dir: Path, n: int, seed: int = 1) -> None:
    """Project random navigable points with the saved transform and check the saved navmask."""
    if n <= 0:
        return
    ps.seed(seed)
    pts = np.array([ps.get_random_navigable_point() for _ in range(n)], dtype=np.float64)
    for i, meta in enumerate(level_meta):
        subs = np.asarray(meta["sub_levels"])
        sel = pts[np.min(np.abs(pts[:, 1:2] - subs[None]), axis=1) < 0.3]
        level = Level(meta, scene_dir)  # reads the navmask PNG back from disk
        if len(sel) == 0:
            meta["qa"]["nav_points"] = {"n": 0}
            continue
        on0 = level.on_navmask(sel[:, 0], sel[:, 2], tol_px=0)
        on1 = level.on_navmask(sel[:, 0], sel[:, 2], tol_px=1)
        meta["qa"]["nav_points"] = {"n": int(len(sel)), "on_mask": round(float(on0.mean()), 5),
                                    "on_mask_tol1px": round(float(on1.mean()), 5)}


def process_scene(assets: dict, out_root: Path, renderer: OrthoRenderer, args) -> dict:
    import habitat_sim
    scene = assets["scene"]
    scene_dir = out_root / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    timing = {}
    t_scene = time.time()

    t0 = time.time()
    ps = habitat_sim.PathFinder()
    ps.load_nav_mesh(str(assets["navmesh"]))
    if not ps.is_loaded:
        raise RuntimeError("navmesh failed to load: %s" % assets["navmesh"])
    levels = floor_levels(ps, n=args.level_samples, bin_m=args.level_bin, min_frac=args.level_min_frac,
                          peak_win_m=args.level_peak_win, merge_m=args.level_merge, max_up_m=args.level_max_up,
                          storey_m=args.level_storey, stack_m2=args.level_stack_m2)
    grids = [level_grid(ps, lv, args.margin, args.mpp, args.max_px) for lv in levels]
    plans = [camera_plan(lv, args.cam_height, args.below) for lv in levels]
    timing["levels_s"] = round(time.time() - t0, 3)
    for lv in levels:
        if lv["slice_gap_below_m"] is not None and lv["slice_gap_below_m"] <= 0:
            print("  [%s] WARNING level subs=%s: navmask slice overlaps the level below by %.2f m (stacked chain cut)"
                  % (scene, [round(y, 2) for y in lv["sub_levels"]], -lv["slice_gap_below_m"]), flush=True)

    t0 = time.time()
    navmasks = [level_navmask(ps, g) for g in grids]
    timing["navmask_s"] = round(time.time() - t0, 3)

    shots, load_s, render_s = renderer.render(assets["glb"], grids, [cams for cams, _ in plans])
    timing["sim_load_s"] = round(load_s, 3)
    timing["render_s"] = round(render_s, 3)
    t0 = time.time()
    rgbs, cam_frac = [], []
    for (nav, sub_masks), (cams, sub_cam), images in zip(navmasks, plans, shots):
        if len(cams) == 1:
            rgbs.append(images[0])
            cam_frac.append([1.0])
        else:
            rgb, label = composite(images, sub_masks, sub_cam)
            rgbs.append(rgb)
            cam_frac.append([round(float((label == c).mean()), 5) for c in range(len(cams))])
    navmasks = [nav for nav, _ in navmasks]
    timing["composite_s"] = round(time.time() - t0, 3)

    per_level_rooms, unassigned = [[] for _ in levels], []
    if assets["house"] is not None:
        per_level_rooms, unassigned = rooms_by_level(parse_house(assets["house"]), [lv["sub_levels"] for lv in levels])

    level_meta = []
    for i, (lv, grid, nav, rgb) in enumerate(zip(levels, grids, navmasks, rgbs)):
        t0 = time.time()
        keep = rooms_mask(per_level_rooms[i], grid) if (args.fade_rooms and per_level_rooms[i]) else None
        radius = args.fade_radius if keep is not None else args.fade_radius_navonly
        weight, foot = footprint_weight(nav, grid["mpp"], radius, args.fade_feather,
                                        args.fade_close, args.fade_fill_holes, keep=keep)
        rgba = faded_rgba(rgb, weight)
        fade_s = time.time() - t0
        files = {"rgb": "level_%d_rgb.png" % i, "rgba": "level_%d_rgba.png" % i,
                 "navmask": "level_%d_navmask.png" % i}
        t0 = time.time()
        _save_png(rgb, scene_dir / files["rgb"])
        _save_png(rgba, scene_dir / files["rgba"])
        _save_png(nav.astype(np.uint8) * 255, scene_dir / files["navmask"])
        save_s = time.time() - t0
        background = np.all(rgb == 255, axis=-1)
        mpp = grid["mpp"]
        meta = dict(grid)
        meta.update({
            "index": i,
            "area_frac": lv["area_frac"],
            "extent": [grid["x0"], grid["x0"] + grid["width"] * mpp, grid["z0"] + grid["height"] * mpp, grid["z0"]],
            "nav_area_m2": round(float(nav.sum()) * mpp * mpp, 3),
            "footprint_area_m2": round(float(foot.sum()) * mpp * mpp, 3),
            "fade_radius_m": radius,
            "footprint_has_rooms": keep is not None,
            "slice_gap_below_m": lv["slice_gap_below_m"],
            "cameras": [dict(cam, pixel_frac=f) for cam, f in zip(plans[i][0], cam_frac[i])],
            "files": files,
            "qa": {
                "nav_pixels_rendered": round(float(1.0 - background[nav].mean()) if nav.any() else 0.0, 5),
                "background_frac": round(float(background.mean()), 5),
                "faded_out_frac": round(float((weight < 0.5).mean()), 5),
            },
            "timing_s": {"fade": round(fade_s, 3), "save": round(save_s, 3)},
            "rooms": per_level_rooms[i],
        })
        level_meta.append(meta)
        print("  [%s] level %d y=%.3f subs=%s cams=%d area=%.2f %dx%d px %.2f cm/px nav=%.0f m2 rendered-on-nav=%.4f rooms=%d" % (
            scene, i, grid["floor_y"], [round(y, 2) for y in grid["sub_levels"]], len(plans[i][0]), lv["area_frac"],
            grid["width"], grid["height"], mpp * 100, meta["nav_area_m2"], meta["qa"]["nav_pixels_rendered"],
            len(per_level_rooms[i])), flush=True)

    t0 = time.time()
    _qa_points(ps, level_meta, scene_dir, args.qa_points)
    timing["qa_s"] = round(time.time() - t0, 3)
    timing["total_s"] = round(time.time() - t_scene, 3)

    lo, hi = ps.get_bounds()
    record = {
        "schema": SCHEMA,
        "scene": scene,
        "dataset": assets["dataset"],
        "assets": {k: (None if assets[k] is None else str(assets[k])) for k in ("glb", "navmesh", "house")},
        "navmesh_bounds": {"lo": [float(v) for v in lo], "hi": [float(v) for v in hi]},
        "conventions": CONVENTIONS,
        "params": {
            "cam_height_m": args.cam_height, "below_m": args.below, "near_m": args.near,
            "margin_m": args.margin, "target_mpp": args.mpp, "max_px": args.max_px,
            "levels": {"samples": args.level_samples, "bin_m": args.level_bin, "min_frac": args.level_min_frac,
                       "peak_win_m": args.level_peak_win, "merge_m": args.level_merge, "linkage": "single",
                       "max_up_m": args.level_max_up, "storey_m": args.level_storey,
                       "stack_m2": args.level_stack_m2, "seed": 0},
            "fade": {"radius_m": args.fade_radius, "radius_navonly_m": args.fade_radius_navonly,
                     "feather_m": args.fade_feather,
                     "close_m": args.fade_close, "fill_holes_m2": args.fade_fill_holes,
                     "rooms_in_footprint": bool(args.fade_rooms and assets["house"] is not None)},
            "navmask_max_y_delta_m": NAV_SLICE_M,
        },
        "levels": level_meta,
        "rooms_unassigned": unassigned,
        "timing_s": timing,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "habitat_sim": getattr(habitat_sim, "__version__", "unknown"),
        "tool": "scripts/exp18/topdown/render_topdown.py",
    }
    tmp = scene_dir / "topdown.json.tmp"
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(str(tmp), str(scene_dir / "topdown.json"))
    return record


# -------------------------------------------------------------------- main ---

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_argument_group("scenes")
    src.add_argument("--scenes", nargs="*", default=None,
                     help="scene ids (MP3D or HM3D; 'm49MsVC7BwA' and 'm49MsVC7BwA.basis' both work); "
                          "comma/space separated, @file reads a list")
    src.add_argument("--tier", choices=["A", "B", "C", "D", "E"], default=None,
                     help="derive scenes from an EXP-18 tier (A/B: clip list or R2R v2 dirs; C/E: val_unseen; "
                          "D: --manifest or the D clip list)")
    src.add_argument("--manifest", default=None, help="scene/clip list or JSON(.gz) selection manifest for --tier")
    src.add_argument("--shard", default="0/1", help="i/n: process scenes[i::n] of the sorted list")
    src.add_argument("--list", action="store_true", help="print the resolved scenes and exit")
    src.add_argument("--mp3d-root", default=str(common.MP3D_SCENES))
    src.add_argument("--hm3d-root", default=str(common.HM3D_SCENES))
    p.add_argument("--out", default=str(common.EXP_ROOT / "topdown"))
    p.add_argument("--overwrite", action="store_true", help="re-render scenes that already have topdown.json")
    cam = p.add_argument_group("camera / grid")
    cam.add_argument("--cam-height", type=float, default=1.9,
                     help="camera height above the main floor's / a lower sub-level's navmesh y (m)")
    cam.add_argument("--below", type=float, default=0.65,
                     help="far plane: each camera clips geometry this far below its own floor height (m); "
                          "0.65 covers the navmask's +-0.5 m slice (sunken floors, first stair steps)")
    cam.add_argument("--near", type=float, default=0.01)
    cam.add_argument("--margin", type=float, default=1.5, help="crop margin around the level's navigable bounds (m)")
    cam.add_argument("--mpp", type=float, default=0.0125, help="target metres per pixel")
    cam.add_argument("--max-px", type=int, default=4096, help="cap on the longest image side (raises mpp)")
    fade = p.add_argument_group("fade (rgba)")
    fade.add_argument("--fade-radius", type=float, default=0.6,
                      help="keep pixels within this distance of the footprint (m) when MP3D rooms are in it")
    fade.add_argument("--fade-radius-navonly", type=float, default=1.2,
                      help="same, for navmesh-only footprints (HM3D: its navmesh stops 0.4-1 m short of walls)")
    fade.add_argument("--fade-feather", type=float, default=0.3, help="linear fade width beyond the radius (m)")
    fade.add_argument("--fade-close", type=float, default=1.25,
                      help="closing radius that folds furniture into the footprint (m); 0 = navmesh only")
    fade.add_argument("--fade-fill-holes", type=float, default=16.0,
                      help="fill enclosed non-footprint holes up to this area (m^2); 0 = off")
    fade.add_argument("--no-fade-rooms", dest="fade_rooms", action="store_false",
                      help="MP3D: do not add the level's .house room polygons to the footprint")
    lvl = p.add_argument_group("floor levels")
    lvl.add_argument("--level-samples", type=int, default=20000)
    lvl.add_argument("--level-bin", type=float, default=0.1)
    lvl.add_argument("--level-min-frac", type=float, default=0.03,
                     help="a floor needs this share of navigable area within --level-peak-win of its peak")
    lvl.add_argument("--level-peak-win", type=float, default=0.15,
                     help="half-width of a floor peak's mass window (m); 0.15 = the peak bin and its two neighbours")
    lvl.add_argument("--level-merge", type=float, default=1.1,
                     help="floors chained by height gaps <= this form one level (sub-levels) (m); must be >= 1.0 so "
                          "different levels' +-0.5 m navmask slices cannot overlap; 1.1 sits between the 0.2 m "
                          "navmesh height steps")
    lvl.add_argument("--level-max-up", type=float, default=1.2,
                     help="main floor (camera reference) = largest sub-level within this of the level's top (m)")
    lvl.add_argument("--level-storey", type=float, default=2.0,
                     help="sub-levels this far apart with stacked navmesh are separate storeys: cut the chain (m)")
    lvl.add_argument("--level-stack-m2", type=float, default=4.0, help="stacked x-z area that counts as storeys (m^2)")
    p.add_argument("--qa-points", type=int, default=2000, help="random navigable points projected for QA")
    args = p.parse_args(argv)
    if args.level_merge < 2 * NAV_SLICE_M:
        p.error("--level-merge must be >= %.1f m (levels' +-%.1f m navmask slices would overlap)"
                % (2 * NAV_SLICE_M, NAV_SLICE_M))
    return args


def resolve_scene_list(args) -> list:
    scenes = []
    for value in args.scenes or []:
        if value.startswith("@"):
            scenes.extend(scenes_from_manifest(Path(value[1:])))
        else:
            scenes.extend(v for v in value.replace(",", " ").split() if v)
    if args.tier:
        scenes.extend(scenes_for_tier(args.tier, args.manifest))
    if not scenes:
        raise SystemExit("no scenes: give --scenes or --tier")
    unique = sorted({canonical_scene_id(s) for s in scenes})
    index, count = [int(v) for v in args.shard.split("/")]
    return unique[index::count]


def main(argv=None) -> int:
    args = parse_args(argv)
    scenes = resolve_scene_list(args)
    assets_db = SceneAssets(Path(args.mp3d_root), Path(args.hm3d_root))
    out_root = Path(args.out)
    print("[topdown] shard %s: %d scenes -> %s" % (args.shard, len(scenes), out_root), flush=True)
    if args.list:
        for scene in scenes:
            try:
                a = assets_db.resolve(scene)
                print("  %s %s %s" % (scene, a["dataset"], a["glb"]))
            except FileNotFoundError as exc:
                print("  %s MISSING (%s)" % (scene, exc))
        return 0
    if not os.environ.get("DISPLAY"):
        raise SystemExit("DISPLAY is not set: run through run_topdown.sh or with_xvfb.sh (Xvfb + llvmpipe)")

    renderer = OrthoRenderer(args.near)
    failed, done, skipped = [], 0, 0
    try:
        for k, scene in enumerate(scenes):
            if (out_root / scene / "topdown.json").exists() and not args.overwrite:
                skipped += 1
                print("[topdown] (%d/%d) %s: exists, skip" % (k + 1, len(scenes), scene), flush=True)
                continue
            try:
                assets = assets_db.resolve(scene)
                for key in ("glb", "navmesh"):
                    if not Path(assets[key]).exists():
                        raise FileNotFoundError("%s missing: %s" % (key, assets[key]))
                print("[topdown] (%d/%d) %s [%s]" % (k + 1, len(scenes), scene, assets["dataset"]), flush=True)
                record = process_scene(assets, out_root, renderer, args)
                done += 1
                print("[topdown] %s done: %d levels, %s" % (scene, len(record["levels"]), record["timing_s"]), flush=True)
            except Exception:
                failed.append(scene)
                print("[topdown] %s FAILED" % scene, flush=True)
                traceback.print_exc()
                sys.stdout.flush()
    finally:
        renderer.close()
    print("[topdown] shard %s summary: done=%d skipped=%d failed=%d %s" % (
        args.shard, done, skipped, len(failed), " ".join(failed)), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

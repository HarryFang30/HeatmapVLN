#!/usr/bin/env python3
"""Write a SYNTHETIC EXP-19 figure bundle, so the figure code can be built and tested before any rerun.

The bundle follows ``exp19-figure-bundle-v1`` and is marked
``"synthetic": true`` (the page then prints a red "synthetic" banner).  Its
content is plausible, not real:

* Route, reference path, start, goal and instruction come from a real R2R
  val_unseen episode.  The per-step states are rebuilt from the GT
  ``locations`` + ``actions`` in ``val_unseen_gt.json.gz``, and the map is that
  scene's EXP-18 top-down map.  All of these are read-only.
* Images are real renders from an ``r2r_panoramic_data_v2`` training clip.
  They come from another scene, so they do not match the map.
* The affordance maps are made from this geometry.  The ground truth puts a
  Gaussian at the true direction of each past camera centre, like the labels
  but without occlusion.  The "prediction" is the same Gaussian at a jittered
  bearing, and the future bins follow the next 3.3 m of the route.  So the
  figure's bearing conventions can be checked by eye against the map: a past
  frame behind-left on the map must be a blue circle left of the back edge.

* F1 / F2 routes are the GT route plus a synthetic walk back: F1 turns
  around at the goal and stops once it is more than 3.5 m away again (it was
  inside the success radius earlier), F2 walks all the way back to the start
  and is labelled as hitting the step cap.
* Stand-in "ready calls" come every 4 steps from step 20, and the key moments
  among them are chosen by the real rule (``scripts/exp19/keysteps.py``), so
  the bundle carries real ``branch`` tags for the captions.

The JSON carries the same fields as ``scripts/exp19/build_records.py`` writes
(``memberships``, ``conventions``, per-key ``branch`` / ``npz_prefix`` / ...),
so the figure code is exercised on the real layout.

When a source is missing (unit tests, a laptop), each part falls back to a
generated stand-in: an L-shaped route, a fake top-down level written next to
the bundle, and procedural images.

Usage::

  python -m scripts.exp19.figures.synthetic_bundle --out-dir /tmp/exp19_dev_G/synth \\
      [--scene zsNo4HB9uLZ] [--episode-id N] [--category T1] [--clip-dir <r2r v2 clip>] [--topdown-root DIR]
  python -m scripts.exp19.figures.synthetic_bundle --out-dir /tmp/exp19_dev_G/synth --main-set
      # five bundles, one per category, on the rank-0 candidates' scenes / episodes (is_main)
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import zlib
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from scripts.exp18 import geometry as geo
from scripts.exp18.topdown import topdown_io
from scripts.exp19 import keysteps
from scripts.exp19.figures import bundle as bd

WORKSPACE = "/mnt/afs/liwenhao/agent/370910109"
DEFAULT_EPISODES = WORKSPACE + "/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
DEFAULT_CLIP = WORKSPACE + "/r2r_panoramic_data_v2/train/1LXtFkjw3qL/clip_000452"
LOOKDOWN_PITCH_DEG = 30.0
EVAL_HFOV_DEG = 79.0
EVAL_W, EVAL_H = 640, 480
MODEL_SIZE = 384
PATH_SPACING_M = 0.1  # 33 points over 3.3 m, like the resampled System1 target
# rank-0 candidates of cases/candidates.json (select_cases.py dev run): scene, episode per category
MAIN_SET = {"T1": ("X7HyMhZNoso", 601), "T2": ("EU6Fwq7SyZv", 346), "T3": ("zsNo4HB9uLZ", 713),
            "F1": ("zsNo4HB9uLZ", 163), "F2": ("x8F5xyUWy9e", 950)}


# --------------------------------------------------------------------------- #
# Route: per-step states (position, yaw) of an episode
# --------------------------------------------------------------------------- #
def _states_from_gt(ep: dict, gt: dict):
    """Per-step (floor position, yaw deg) replaying the GT actions; locations advance on FORWARD."""
    x, y, z, w = ep["start_rotation"]  # dataset order xyzw
    yaw = math.degrees(float(topdown_io.yaw_from_quaternion([w, x, y, z])))
    locs = np.asarray(gt["locations"], dtype=np.float64)
    pos, li = locs[0].copy(), 0
    states, actions = [(pos.copy(), yaw)], []
    for a in gt["actions"]:
        a = int(a)
        if a == bd.STOP:
            break
        if a == bd.LEFT:
            yaw += 15.0
        elif a == bd.RIGHT:
            yaw -= 15.0
        elif a == bd.FORWARD:
            li = min(li + 1, len(locs) - 1)
            pos = locs[li].copy()
        actions.append(a)
        states.append((pos.copy(), yaw))
    return states, actions


def _synthetic_states():
    """L-shaped route: 4 m ahead, left 90 deg, 3 m, right 90 deg, 3 m (yaw 0 looks along -z)."""
    plan = [bd.FORWARD] * 16 + [bd.LEFT] * 6 + [bd.FORWARD] * 12 + [bd.RIGHT] * 6 + [bd.FORWARD] * 12
    pos, yaw = np.zeros(3), 0.0
    states, actions = [(pos.copy(), yaw)], []
    for a in plan:
        if a == bd.LEFT:
            yaw += 15.0
        elif a == bd.RIGHT:
            yaw -= 15.0
        else:
            f = topdown_io.yaw_to_forward(math.radians(yaw))
            pos = pos + 0.25 * np.array([f[0], 0.0, f[1]])
        actions.append(a)
        states.append((pos.copy(), yaw))
    return states, actions


def walk_back(states, actions, stop_when):
    """Turn around in place (12 x LEFT) and walk the route backwards until ``stop_when(position)``.

    Heading follows the direction of travel, so the synthetic history geometry
    stays consistent with the map.  Used for the synthetic F1 / F2 routes.
    """
    states, actions = list(states), list(actions)
    pos, yaw = states[-1]
    for _ in range(12):
        yaw += 15.0
        states.append((pos.copy(), yaw))
        actions.append(bd.LEFT)
    for p, _ in reversed(states[:-13]):
        step = (p - pos)[[0, 2]]
        if np.linalg.norm(step) < 1e-6:
            continue
        yaw = math.degrees(float(topdown_io.forward_to_yaw(step)))
        pos = p.copy()
        states.append((pos.copy(), yaw))
        actions.append(bd.FORWARD)
        if stop_when(pos):
            break
    return states, actions


def camera_c2w(pos, yaw_deg: float) -> np.ndarray:
    T = geo.yaw_rotation(yaw_deg)
    T[:3, 3] = np.asarray(pos, dtype=np.float64) + np.array([0.0, bd.CAMERA_HEIGHT_M, 0.0])
    return T


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
class ClipImages:
    """Real 256 px renders from an R2R v2 clip, or procedural stand-ins when the clip is absent."""

    def __init__(self, clip_dir: Optional[str], seed: int):
        self.rng = np.random.default_rng(seed)
        self.chunk = None
        path = Path(clip_dir) / "chunks" / "chunk_00000.npz" if clip_dir else None
        if path is not None and path.is_file():
            with np.load(path, allow_pickle=True) as z:  # the clip's own JPEG object arrays
                self.chunk = {k: z[k] for k in ("rgb_front", "rgb_right", "rgb_back", "rgb_left", "rgb_front_down")}
            self.n = len(self.chunk["rgb_front"])

    @property
    def real(self) -> bool:
        return self.chunk is not None

    def view(self, name: str, frame: int) -> np.ndarray:
        if self.chunk is not None:
            data = self.chunk[f"rgb_{name}"][frame % self.n]
            return np.asarray(Image.open(io.BytesIO(bytes(data))).convert("RGB"))
        rng = np.random.default_rng(zlib.crc32(f"{name}:{frame}".encode()))
        yy, xx = np.mgrid[0:256, 0:256] / 255.0
        base = np.stack([0.55 + 0.3 * yy, 0.5 + 0.2 * xx, 0.45 + 0.1 * (1 - yy)], -1)
        for _ in range(6):
            x0, y0 = rng.integers(0, 200, 2)
            base[y0:y0 + rng.integers(20, 80), x0:x0 + rng.integers(20, 80)] = rng.uniform(0.2, 0.9, 3)
        return (np.clip(base, 0, 1) * 255).astype(np.uint8)


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))


# --------------------------------------------------------------------------- #
# Geometry of the synthetic predictions
# --------------------------------------------------------------------------- #
def _gauss(center_uv, sigma: float, peak: float) -> np.ndarray:
    ys, xs = np.mgrid[0:bd.HM_SIZE, 0:bd.HM_SIZE]
    return peak * np.exp(-((xs - center_uv[0]) ** 2 + (ys - center_uv[1]) ** 2) / (2 * sigma ** 2))


def _direction_pixel(bearing: float, elev: float):
    view, u, v = geo.bearing_elev_to_pixel(bearing, elev)
    if not (np.isfinite(u) and np.isfinite(v) and 0 <= u < bd.HM_SIZE and 0 <= v < bd.HM_SIZE):
        return None
    return int(view), float(u), float(v)


def history_maps(c2w, hist_c2w, rng):
    """GT and 'predicted' history maps for the slots (no occlusion test; FOV + distance like the labels)."""
    S, V, H = bd.NUM_SLOTS, bd.NUM_VIEWS, bd.HM_SIZE
    gt = np.zeros((S, V, H, H), np.float32)
    vis = np.zeros((S, V), np.float32)
    peak = -np.ones((S, 3), np.float32)
    pred = np.zeros((S, V, H, H), np.float32)
    none = np.ones(S, np.float32)
    mask = np.zeros(S, bool)
    for k, T in enumerate(hist_c2w):
        mask[k] = True
        fwd, left, up = geo.world_to_rel(c2w, T[:3, 3])
        dist = math.sqrt(fwd ** 2 + left ** 2 + up ** 2)
        px = None
        if 1e-4 < dist <= 15.0:
            bearing = math.degrees(math.atan2(left, fwd))
            elev = math.degrees(math.atan2(up, math.hypot(fwd, left)))
            px = _direction_pixel(bearing, elev)
        if px is None:
            none[k] = 0.85  # e.g. the current position: nothing to see
            pred[k, 2] = _gauss((32, 32), 6.0, 1.0)
            pred[k] = pred[k] / pred[k].sum() * (1 - none[k])
            continue
        view, u, v = px
        sigma = float(np.clip(16.0 / max(dist, 1e-3), 4.0, 8.0))
        gt[k, view] = _gauss((u, v), sigma, 0.7 + 0.3 / (1 + dist / 5.0))
        vis[k, view] = 1.0
        r, c = divmod(int(gt[k, view].argmax()), H)
        peak[k] = (view, r, c)
        jitter = _direction_pixel(bearing + rng.normal(0, 4.0), elev + rng.normal(0, 1.0)) or px
        none[k] = float(rng.uniform(0.02, 0.15))
        g = _gauss((jitter[1], jitter[2]), sigma * 0.7, 1.0)
        pred[k, jitter[0]] = g / g.sum() * (1 - none[k])
    return gt, vis, peak, pred, none, mask


def future_path(states, s: int):
    """33 floor points every 0.1 m along the route ahead of step s (clamped at its end), world xz."""
    pts = np.array([st[0] for st in states[s:]], dtype=np.float64)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(pts[:, [0, 2]], axis=0), axis=1) > 1e-6])
    pts = pts[keep]
    if len(pts) < 2:
        return np.repeat(pts[:1], bd.PATH_POINTS, axis=0)
    seg = np.linalg.norm(np.diff(pts[:, [0, 2]], axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    want = np.minimum(np.arange(bd.PATH_POINTS) * PATH_SPACING_M, arc[-1])
    return np.stack([np.interp(want, arc, pts[:, i]) for i in range(3)], axis=1)


def future_maps(path_cam, rng):
    """Four bins x four views: a Gaussian per waypoint at camera height (the future-label placement)."""
    fut = np.zeros((bd.NUM_BINS, bd.NUM_VIEWS, bd.HM_SIZE, bd.HM_SIZE), np.float32)
    bearing, elev, idx = bd.path_directions(path_cam)
    for b in range(bd.NUM_BINS):
        conf = float(rng.uniform(0.55, 0.95))
        for j in range(8 * b + 1, 8 * b + 9):
            hit = np.nonzero(idx == j)[0]
            if not hit.size:
                continue
            px = _direction_pixel(bearing[hit[0]] + rng.normal(0, 2.0), elev[hit[0]])
            if px is None:
                continue
            view, u, v = px
            dist = float(np.hypot(path_cam[j, 0], path_cam[j, 2]))
            sigma = float(np.clip(16.0 / max(dist, 0.3), 4.0, 8.0))
            fut[b, view] = np.maximum(fut[b, view], _gauss((u, v), sigma, conf))
    vis = fut.reshape(bd.NUM_BINS, bd.NUM_VIEWS, -1).max(-1)
    return fut, vis


def project_eval_camera(path_cam, lookdown: bool):
    """(u, v) of floor points in the 640x480 HFOV-79 eval camera, pitched down 30 deg for the look-down."""
    f = (EVAL_W / 2) / math.tan(math.radians(EVAL_HFOV_DEG / 2))
    p = np.asarray(path_cam, dtype=np.float64)
    if lookdown:
        a = math.radians(LOOKDOWN_PITCH_DEG)  # rotate the world by +pitch about x = camera pitched down
        R = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        p = p @ R.T
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    ok = z < -0.1
    u = np.where(ok, f * x / np.where(ok, -z, 1) + EVAL_W / 2, np.nan)
    v = np.where(ok, f * (-y) / np.where(ok, -z, 1) + EVAL_H / 2, np.nan)
    inside = ok & (u >= 0) & (u < EVAL_W) & (v >= 0) & (v < EVAL_H)
    return np.stack([u, v], 1)[inside]


# --------------------------------------------------------------------------- #
# Fake top-down level (fallback)
# --------------------------------------------------------------------------- #
def write_fake_topdown(root: Path, scene: str, xz: np.ndarray, floor_y: float, mpp: float = 0.025) -> None:
    lo, hi = xz.min(0) - 2.5, xz.max(0) + 2.5
    w, h = int(math.ceil((hi[0] - lo[0]) / mpp)), int(math.ceil((hi[1] - lo[1]) / mpp))
    rgb = np.full((h, w, 3), 236, np.uint8)
    rgb[::80] = 200
    rgb[:, ::80] = 200
    nav = np.zeros((h, w), np.uint8)
    rows = ((xz[:, 1] - lo[1]) / mpp).astype(int)
    cols = ((xz[:, 0] - lo[0]) / mpp).astype(int)
    for r, c in zip(rows, cols):
        nav[max(r - 40, 0):r + 40, max(c - 40, 0):c + 40] = 255
    rgb[nav > 0] = (214, 206, 190)
    d = root / scene
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(d / "level_0_rgb.png")
    Image.fromarray(np.dstack([rgb, np.full((h, w), 255, np.uint8)])).save(d / "level_0_rgba.png")
    Image.fromarray(nav).save(d / "level_0_navmask.png")
    meta = {"schema": topdown_io.SCHEMA, "scene": scene, "fake": True,
            "levels": [{"index": 0, "floor_y": floor_y, "x0": float(lo[0]), "z0": float(lo[1]), "mpp": mpp,
                        "width": w, "height": h, "rooms": [],
                        "files": {"rgb": "level_0_rgb.png", "rgba": "level_0_rgba.png",
                                  "navmask": "level_0_navmask.png"}}]}
    (d / "topdown.json").write_text(json.dumps(meta, indent=1))


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #
def pick_key_steps(actions: List[int], category: str, dist_to_goal: np.ndarray, first_ready: int = 20,
                   every: int = 4) -> List[dict]:
    """Stand-in ready calls every ``every`` steps from ``first_ready`` on (the chunk after each is executed), and the
    key moments among them by the pre-registered rule (``keysteps.select_key_steps``), in label order."""
    n_steps = len(actions)
    calls = list(range(first_ready, max(n_steps - 1, first_ready + 1), every)) or [min(first_ready, n_steps - 1)]
    ready = [{"call_index": c // every, "step": c, "executed_actions": actions[c:c + every]} for c in calls]
    closest = keysteps.closest_approach_step(list(range(len(dist_to_goal))), list(dist_to_goal))
    return keysteps.select_key_steps(ready, category=category, episode_steps=n_steps + 1, closest_step=closest)


def make_bundle(out_dir, scene: str = "zsNo4HB9uLZ", episode_id: Optional[int] = None, category: str = "T1",
                episodes_path: Optional[str] = DEFAULT_EPISODES, clip_dir: Optional[str] = DEFAULT_CLIP,
                topdown_root=None, seed: int = 0, front_key: int = 2, rank: int = 0, is_main: bool = True) -> Path:
    """Write ``<out_dir>/records/<ep_key>_bundle.{json,npz}``; returns the json path."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    sources = {}
    ep = None
    if episodes_path and Path(episodes_path).is_file():
        with gzip.open(episodes_path, "rt") as f:
            eps = [e for e in json.load(f)["episodes"] if topdown_io.canonical_scene_id(e["scene_id"]) == scene]
        with gzip.open(str(episodes_path).replace("val_unseen.json.gz", "val_unseen_gt.json.gz"), "rt") as f:
            gts = json.load(f)
        if episode_id is None:  # a mid-length episode with turns: long enough for four distinct key moments
            eps = sorted(eps, key=lambda e: (abs(len(gts[str(e["episode_id"])]["actions"]) - 70), e["episode_id"]))
        ep = next(e for e in eps if episode_id is None or int(e["episode_id"]) == int(episode_id))
        states, actions = _states_from_gt(ep, gts[str(ep["episode_id"])])
        sources["episode"] = episodes_path
    else:
        states, actions = _synthetic_states()
        sources["episode"] = "generated L-shaped route"
    episode_id = int(ep["episode_id"]) if ep else int(episode_id or 7)
    ep_key = f"{scene}_{episode_id:04d}"
    ref = np.asarray(ep["reference_path"], dtype=np.float64) if ep else np.array([s[0] for s in states])[::8]
    goal = np.asarray(ep["goals"][0]["position"], dtype=np.float64) if ep else states[-1][0] + np.array([0.6, 0, 0.4])
    ended_by = "stop"
    if category == "F1":  # overshoot: back out of the success radius, then stop
        states, actions = walk_back(states, actions, lambda p: np.linalg.norm((p - goal)[[0, 2]]) > 3.5)
    elif category == "F2":  # wander back to the start; labelled as the step cap
        states, actions = walk_back(states, actions, lambda p: False)
        ended_by = "step_cap"
    route = np.array([s[0] for s in states])
    instruction = (ep["instruction"]["instruction_text"] if ep
                   else "Walk forward, turn left at the table, then go right into the kitchen and stop.")

    root = Path(topdown_root) if topdown_root else topdown_io.default_root()
    try:
        td = topdown_io.load_topdown(scene, root)
        counts = Counter(int(v) for v in td.level_index(route[:, 1]))  # the level of most steps, as [F] picks it
        level_index = max(counts, key=lambda k: (counts[k], -k))
        levels_visited = sorted(counts)
        sources["topdown"] = str(root)
    except (OSError, ValueError, KeyError):
        root = out_dir / "fake_topdown"
        write_fake_topdown(root, scene, np.concatenate([route[:, [0, 2]], ref[:, [0, 2]], goal[None, [0, 2]]]),
                           float(route[0, 1]))
        level_index, levels_visited = 0, [0]
        sources["topdown"] = "fake level written next to the bundle"
    images = ClipImages(clip_dir, seed)
    sources["images"] = str(clip_dir) if images.real else "procedural"

    n_steps = len(actions)
    dist = np.linalg.norm((route - goal)[:, [0, 2]], axis=1)
    keys = pick_key_steps(actions, category, dist)
    meta_keys, arrays = [], {}
    for i, key in enumerate(keys):
        s = key["step"]
        pos, yaw = states[s]
        c2w = camera_c2w(pos, yaw)
        hist_steps = sorted(set(int(round(t)) for t in np.linspace(0, s - 1, bd.NUM_SLOTS)))
        hist_c2w = [camera_c2w(*states[h]) for h in hist_steps]
        gt, vis, peak, pred, none, mask = history_maps(c2w, hist_c2w, rng)
        path_world = future_path(states, s)
        fwd, left, _ = geo.world_to_rel(c2w, path_world + np.array([0.0, bd.CAMERA_HEIGHT_M, 0.0]))
        wobble = np.linspace(0, 1, bd.PATH_POINTS) ** 2 * rng.normal(0, 0.12)
        path_cam = np.stack([-(left + wobble), np.full_like(fwd, -bd.CAMERA_HEIGHT_M), -fwd], 1)
        fut, fut_vis = future_maps(path_cam, rng)
        lookdown = i != front_key
        uv = project_eval_camera(path_cam, lookdown)
        frame = 7 + 11 * i
        if lookdown:
            dec = _resize(images.view("front_down", frame), EVAL_W, EVAL_H)
        else:
            dec = _resize(images.view("front", frame), MODEL_SIZE, MODEL_SIZE)
            uv = uv * np.array([MODEL_SIZE / EVAL_W, MODEL_SIZE / EVAL_H])
        if len(uv):
            goal_uv = [float(np.round(uv[-1, 0])), float(np.round(uv[-1, 1]))]
        else:  # path not in view (turning around): a ready call still has a pixel goal; put it just ahead
            goal_uv = [float(dec.shape[1] // 2), float(round(dec.shape[0] * 0.85))]
        text = f"{int(goal_uv[1])} {int(goal_uv[0])}"  # System2 writes 'row col'
        chunk = [a for a in actions[s:s + 4]] or [bd.STOP]
        cf = list(chunk) if i != 1 else [bd.FORWARD] * len(chunk)  # one changed counterfactual, for the glyphs
        meta_keys.append({
            "label": key["label"], "rule": key["rule"], "branch": key["branch"], "ready_index": key["ready_index"],
            "call_index": key["call_index"], "step": s, "npz_prefix": f"k{i}_",
            "position_xz": [float(pos[0]), float(pos[2])],
            "dist_to_goal_m": float(np.linalg.norm((pos - goal)[[0, 2]])),
            "decision_image": "lookdown" if lookdown else "front",
            "decision_image_wh": [EVAL_W, EVAL_H] if lookdown else [MODEL_SIZE, MODEL_SIZE],
            "system2_first_output": "↓" if lookdown else text, "system2_output": text,
            "pixel_goal_raw": goal_uv, "pixel_goal_uv": goal_uv, "path_uv": uv.round(2).tolist(),
            "path_uv_index": list(range(len(uv))), "executed_actions": chunk,
            "response_actions": chunk, "cf_actions": cf, "cf_changed": cf != chunk,
            "history_steps": hist_steps, "history_count": len(hist_steps),
            "h1_call": {"n_valid": len(hist_steps), "n_visible": int((vis.max(1) > 0).sum()), "pck8": None,
                        "floor_joint8": None},
            "h2_call": {"pred_view5": None, "ref_view5_system1": None, "ref_view5_executed": None,
                        "pred_max_prob": None},
        })
        p = f"k{i}_"
        arrays[p + "decision_rgb"] = dec
        arrays[p + "front_native"] = _resize(images.view("front", frame), EVAL_W, EVAL_H)
        arrays[p + "history_rgb"] = np.stack([_resize(images.view("front", 3 * j), MODEL_SIZE, MODEL_SIZE)
                                              for j in range(len(hist_steps))])
        arrays[p + "pano_rgb"] = np.stack([images.view(v, frame) for v in geo.VIEW_NAMES])
        arrays[p + "hist_pred"] = pred
        arrays[p + "hist_none"] = none
        arrays[p + "hist_mask"] = mask
        arrays[p + "hist_gt"] = gt
        arrays[p + "hist_gt_vis"] = vis
        arrays[p + "hist_gt_peak"] = peak
        arrays[p + "fut_pred"] = fut
        arrays[p + "fut_vis"] = fut_vis.astype(np.float32)
        arrays[p + "path_cam"] = path_cam.astype(np.float32)
        arrays[p + "path_xz_world"] = path_world[:, [0, 2]].astype(np.float32)

    ne = float(dist[-1])
    success = ne <= 3.0 and ended_by == "stop"
    outcome = {"success": bool(success), "oracle_success": bool((dist <= 3.0).any()), "ne_m": round(ne, 3),
               "steps": 500 if ended_by == "step_cap" else n_steps + 1, "ended_by": ended_by}
    predicate = {"T1": success, "T2": success, "T3": success, "F1": outcome["oracle_success"] and not success,
                 "F2": ended_by == "step_cap" and not success}[category]
    member = {"category": category, "rank": rank, "is_main": is_main, "predicate_holds_on_rerun": bool(predicate)}
    meta = {
        "schema": bd.SCHEMA, "synthetic": True, "synthetic_sources": sources,
        "scene_id": scene, "episode_id": episode_id, "ep_key": ep_key, "category": category, "category_rank": rank,
        "is_main": is_main, "predicate_holds_on_rerun": bool(predicate), "memberships": [member],
        "instruction": instruction, "outcome": outcome,
        "eval_log_outcome": {"success": float(success), "spl": None, "os": float(outcome["oracle_success"]),
                             "ne": round(ne, 3), "steps": outcome["steps"], "ended_by": ended_by},
        "fidelity": {"first_divergent_call": 6, "identical_calls": 6, "total_calls": max(n_steps // 4, 7),
                     "reference_calls": max(n_steps // 4, 7)},
        "topdown": {"root": str(root), "scene": scene, "level_index": level_index, "levels_visited": levels_visited},
        "route_steps": list(range(len(route))),
        "route_xz": route[:, [0, 2]].round(4).tolist(), "route_y": route[:, 1].round(4).tolist(),
        "reference_path_xz": ref[:, [0, 2]].round(4).tolist(), "start_xz": route[0, [0, 2]].round(4).tolist(),
        "goal_xz": goal[[0, 2]].round(4).tolist(), "goal_radius_m": 3.0, "key_steps": meta_keys,
        "conventions": {"view_order": list(geo.VIEW_NAMES),
                        "image_uv": "pixel index coordinates: pixel (col i, row j) is centred at (i, j); u = column",
                        "pixel_goal": {"convention": "synthetic", "forced": False, "meaning": "synthetic"}},
    }
    errors = bd.validate_bundle(meta, arrays)
    if errors:
        raise bd.BundleError("synthetic bundle invalid: " + "; ".join(errors))
    rec = out_dir / "records"
    rec.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rec / f"{ep_key}_bundle.npz", **arrays)
    json_path = rec / f"{ep_key}_bundle.json"
    json_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return json_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scene", default="zsNo4HB9uLZ")
    ap.add_argument("--episode-id", type=int, default=None)
    ap.add_argument("--category", default="T1", choices=bd.CATEGORIES)
    ap.add_argument("--episodes", default=DEFAULT_EPISODES, help="val_unseen.json.gz (its _gt file next to it)")
    ap.add_argument("--clip-dir", default=DEFAULT_CLIP, help="r2r_panoramic_data_v2 clip for real RGB")
    ap.add_argument("--topdown-root", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--main-set", action="store_true", help="one is_main bundle per category (MAIN_SET episodes)")
    args = ap.parse_args(argv)
    jobs = ([(c, sc, e) for c, (sc, e) in MAIN_SET.items()] if args.main_set
            else [(args.category, args.scene, args.episode_id)])
    for j, (category, scene, episode_id) in enumerate(jobs):
        path = make_bundle(args.out_dir, scene=scene, episode_id=episode_id, category=category,
                           episodes_path=args.episodes, clip_dir=args.clip_dir, topdown_root=args.topdown_root,
                           seed=args.seed + j)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

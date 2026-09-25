#!/usr/bin/env python3
"""EXP-19 figures v2 [T]: the online timeline of one rerun episode (presentation data only).

For every ready call of a rerun episode (kind = trajectory and ppa_applied, the calls the
key moments are drawn from), in call order, this module stores what the history head,
the future head and System1 put out, folded onto one bearing axis, plus the true
directions of the past frames.  The v2 figures draw it as a step x bearing timeline;
the animation replays it.  Nothing here is a metric: records/ and metrics/ ([F],
``build_records.py``) stay the only source of numbers, and this module never writes there.

Reads (the same joins as [F], through ``build_records.Episode``)
  runs/<run>/gpu*/trace/<ep_key>/call_XXX.{json,npz}   the tensors each call returned
  runs/<run>/gpu*/steps/<ep_key>/steps.jsonl           GT body states, executed actions
  renders/<ep_key>.npz                                 front depth at the call steps (label visibility)
  records/<ep_key>.json                                key moments, category ([F])
  records/<ep_key>_bundle.{json,npz}                   is_main; the self-check compares against it
  cases/candidates.json                                category / rank membership
Writes
  <out-dir>/<ep_key>_timeline.npz + .json   (schema exp19-timeline-v1), default <exp-root>/records_v2
  <out-dir>/timeline_self_check.json        with --self-check

Bearing convention (one for every array, left-positive, exp18 ring convention):
  bearing = atan2(left, forward) in degrees in the current front camera frame at the call, wrapped to
  (-180, 180]: 0 = straight ahead, +90 = left, -90 = right, 180 = behind.  ``bearing_deg[i] = 179.5 - i``:
  column 0 is the left edge (+180, behind via the left), column 180 straight ahead,
  column 359 the right edge (-180, behind via the right), as ``geo.ring_column_azimuths(360)``.

Per ready call r (arrays with a leading R):
  call_index, step, next_step (step of the next call of any kind, or the episode's step count)
  hist_ring [R, 360]         predicted history composite: per slot heatmaps_gated / its own peak x (1 - P(none)),
                             max over valid slots (``figures.bundle.history_pred_composite``, the v1 strip's
                             tensor), stitched with ``geo.stitch_ring`` (width 360, elevation +-45 deg, 0 outside
                             every view), then max over the elevation rows
  hist_pred_peak_bearing [R, 8], hist_pred_conf [R, 8]
                             bearing of each valid slot's joint argmax of heatmaps_gated over (view, row, col)
                             (first occurrence, the v1 orange peak), and 1 - P(none); NaN for padded slots;
                             the bearing is also NaN when the slot's gated map is all zero (no peak)
  hist_gt_bearing [R, 8], hist_gt_visible [R, 8]
                             atan2(left, forward) of the past camera centre (recorded GT state) in the current
                             front camera frame; NaN when the slot is padded or invisible in every view under
                             the label code (the H1 visibility: class > 0)
  fut_ring [R, 4, 360]       future_heatmaps_gated of each time bin (waypoints 1-8, 9-16, 17-24, 25-32) stitched
                             the same way (max over elevation); values are the gated maps as returned (peak =
                             that view's visibility probability, in [0, 1])
  s1_path_bearing [R, 33], s1_path_dist [R, 33]
                             System1's selected mean path (robot frame at the call, x forward, y left, row 0 =
                             the robot): atan2(y, x) in degrees and the distance in metres; the bearing is NaN
                             closer than 0.05 m (row 0), where a point has no direction
  extras: hist_steps [R, 8] (capture step of each slot, -1 padded; slot 0 = oldest), hist_mask [R, 8],
          exec_actions [R, A] (executed actions of the call's chunk, -1 padded)
Episode level:
  bearing_deg [360]; calls_index / calls_step / calls_next_step int [C]; calls_kind str [C] (response kind of
  every call: trajectory / native_actions / stop); calls_ready bool [C]; calls_state str [C] (ready / warmup =
  a trajectory call without the heads / native_actions = System2 answered with arrows, no affordance map / stop);
  calls_system2_first / calls_system2_output str [C]; step_action int8 [S] (action executed at each step,
  -1 none; 0 STOP 1 forward 2 left 3 right); first_ready_step (-1 if none); episode_steps;
  key_labels str [<=4], key_call_index int [<=4], key_step int [<=4] (records/<ep_key>.json key_steps).

--self-check (run on the written files) checks the conventions on the real data and writes
timeline_self_check.json; exit code 5 if any check fails:
  (0) ring vs peaks: the history ring at each confident slot's peak bearing carries that slot's confidence
  (1) predicted peak bearings agree with the GT bearings on T1/T3 episodes (median |diff|), better than the
      mirrored (left <-> right) and the front <-> back swapped readings; GT atan2 bearings agree with the
      bearing of the label code's GT peak pixel on every ready call
  (2) the key-moment rows reproduce the v1 bundle's tensors exactly (hist_pred, hist_none, hist_mask, fut_pred,
      the ring recomputed from the bundle, GT visibility, peak and path bearings)
  (3) an executed left turn between two ready calls shifts the GT bearings of the shared past frames towards
      negative at the next ready call (right turn: positive)
  (4) the System1 path bearing of all-forward calls is about 0; its sign follows the chunk's net turn

Usage (dev machine, envs/qwen25, CPU only):
  cd <src> && PYTHONDONTWRITEBYTECODE=1 <qwen25 python> -m scripts.exp19.build_timeline --run main --self-check
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.exp18 import geometry as geo  # noqa: E402
from scripts.exp19 import build_records as br  # noqa: E402
from scripts.exp19 import gt  # noqa: E402
from scripts.exp19.figures import bundle as bd  # noqa: E402

SCHEMA = "exp19-timeline-v1"
RING_WIDTH = 360
RING_ELEV_DEG = 45.0
NUM_SLOTS = 8
NUM_BINS = 4
PATH_POINTS = 33
MIN_PATH_DIST_M = 0.05
CONF_MIN = 0.5  # a slot's predicted peak is drawn (and checked) when 1 - P(none) >= 0.5
STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
TURN_DEG = 15.0

CONVENTIONS = {
    "bearing": "atan2(left, forward) in degrees in the current front camera frame at the call, left-positive: "
               "0 ahead, +90 left, -90 right, +-180 behind (exp18 ring convention)",
    "bearing_deg": "ring column centres: bearing_deg[i] = 179.5 - i (column 0 = left edge +180, 180 = ahead, "
                   "359 = right edge -180), geo.ring_column_azimuths(360)",
    "hist_ring": "max over elevation of geo.stitch_ring(figures.bundle.history_pred_composite(heatmaps_gated, "
                 "P(none), mask), width 360, elevation +-45 deg, fill 0); composite = max over valid slots of "
                 "(slot map / its own peak) x (1 - P(none)), in [0, 1]",
    "hist_pred_peak_bearing": "bearing of the first-occurrence joint argmax of each valid slot's heatmaps_gated over "
                              "(view, row, col) (label pixel convention, geo.pixel_to_bearing_elev); NaN for padded "
                              "slots and all-zero maps",
    "hist_pred_conf": "1 - P(none) of each valid slot; NaN for padded slots; peaks are drawn when >= 0.5",
    "hist_gt_bearing": "atan2(left, forward) of the past camera centre (recorded GT state) in the current front "
                       "camera frame (geo.history_bearing_deg); NaN when padded or invisible in every view under the "
                       "label code (gt.history_labels + gt.gt_view_class_and_peak class > 0, the H1 visibility)",
    "fut_ring": "max over elevation of geo.stitch_ring(future_heatmaps_gated[bin]) per time bin (waypoints 1-8, "
                "9-16, 17-24, 25-32), same geometry as hist_ring; values as returned (peak = view probability)",
    "s1_path": "System1 selected mean path [33, 2] (robot frame at the call: x forward, y left, row 0 = robot): "
               "s1_path_bearing = atan2(y, x) deg (NaN closer than 0.05 m), s1_path_dist = hypot(x, y) m",
    "next_step": "step of the next call of any kind, or the episode's step count after the last call; the span "
                 "[step, next_step) is the steps the call's chunk (and any replan at the same step) covers",
    "calls_state": "ready (kind trajectory and ppa_applied) / warmup (trajectory without the heads) / "
                   "native_actions (System2 answered with arrows: no affordance map) / stop",
    "step_action": "action executed at each step (action record step_before), -1 none; 0 STOP 1 forward 0.25 m "
                   "2 left 15 deg 3 right 15 deg",
    "hist_steps": "capture step of each history slot, -1 padded; slot 0 = the oldest frame",
    "exec_actions": "executed actions of the call's chunk (build_records.executed_by_call), -1 padded",
    "key": "key moments as records/<ep_key>.json key_steps (pre-registered rule, scripts/exp19/keysteps.py)",
}

# Self-check pass thresholds (fixed before the checks were run on real data).
CHECK = {
    "ring_at_peak_min_ratio": 0.5,     # (0) ring value at a confident peak >= 0.5 x its confidence ...
    "ring_at_peak_min_share": 0.99,    # ... for >= 99% of the confident slots
    "peak_vs_gt_median_max_deg": 15.0,  # (1) median |pred peak - GT| on T1/T3 confident visible slots
    "gt_vs_label_pixel_max_deg": 2.0,  # (1b) atan2 GT bearing vs the label's GT peak pixel (quantisation)
    "key_gt_vs_bundle_max_deg": 2.0,   # (2) GT bearing vs the bundle's GT peak pixel
    "key_peak_max_deg": 1e-4,          # (2) peak bearing vs the v1 peaks (float32 storage)
    "key_path_max_deg": 1e-3,          # (2) path bearing vs the bundle's path_cam
    "turn_min_deg": 30.0,              # (3) pairs of ready calls with |executed net turn| >= 30 deg ...
    "turn_sign_min_share": 0.9,        # ... shift the shared GT bearings the opposite way in >= 90% of pairs
    "forward_1m_median_max_deg": 10.0,  # (4) all-forward calls: median |path bearing| at ~1 m
    "path_turn_sign_min_share": 0.8,   # (4) sign(path bearing at ~1 m) = sign(net turn), |turn| >= 30 deg
}


# --------------------------------------------------------------------------- #
# Pure functions (unit tested)
# --------------------------------------------------------------------------- #
def bearing_axis(width: int = RING_WIDTH) -> np.ndarray:
    """Bearing of each ring column centre: 179.5 - i for width 360."""
    return geo.ring_column_azimuths(width).astype(np.float32)


def bearing_to_column(bearing, width: int = RING_WIDTH):
    """Nearest ring column of a bearing (wraps: +180 and -180 both reach the edges)."""
    x = np.asarray(geo.azimuth_to_ring_x(bearing, width), dtype=np.float64)
    return np.mod(np.rint(x), width).astype(np.int64)


def ring_1d(maps: np.ndarray, width: int = RING_WIDTH, elev: float = RING_ELEV_DEG) -> np.ndarray:
    """Four 64x64 label-convention maps [4, H, W] -> [width] float32: stitched ring, max over elevation."""
    ring, _ = geo.stitch_ring(np.asarray(maps, dtype=np.float32), width=width, elev_top=elev, elev_bottom=-elev,
                              fill=0.0)
    return ring.max(axis=0).astype(np.float32)


def hist_ring(pred: np.ndarray, none_p: np.ndarray, mask: np.ndarray, width: int = RING_WIDTH) -> np.ndarray:
    """Predicted history composite on the bearing axis (the v1 strip's composite, max over elevation)."""
    return ring_1d(bd.history_pred_composite(pred, none_p, mask), width)


def fut_rings(fut: np.ndarray, width: int = RING_WIDTH) -> np.ndarray:
    """future_heatmaps_gated [4 bins, 4 views, 64, 64] -> [4, width]."""
    return np.stack([ring_1d(fut[b], width) for b in range(fut.shape[0])]).astype(np.float32)


def pred_peaks(pred: np.ndarray, none_p: np.ndarray, mask: np.ndarray):
    """(bearing [8], conf [8]) of each slot's joint argmax of heatmaps_gated; NaN for padded / all-zero slots."""
    pred = np.asarray(pred, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    n = pred.shape[0]
    bearing = np.full(n, np.nan, dtype=np.float32)
    conf = np.where(mask, 1.0 - np.asarray(none_p, dtype=np.float64), np.nan).astype(np.float32)
    for k in np.nonzero(mask)[0]:
        if not np.isfinite(pred[k]).any() or float(np.nanmax(pred[k])) <= 0.0:
            continue
        view, row, col = np.unravel_index(int(np.argmax(pred[k])), pred[k].shape)
        b, _ = geo.pixel_to_bearing_elev(int(view), float(col), float(row))
        bearing[k] = float(b)
    return bearing, conf


def gt_bearings(cur_c2w: np.ndarray, hist_c2w: Sequence[np.ndarray]) -> np.ndarray:
    """atan2(left, forward) of each past camera centre in the current front camera frame [K]."""
    if len(hist_c2w) == 0:
        return np.zeros(0, dtype=np.float64)
    b = np.atleast_1d(np.asarray(geo.history_bearing_deg(cur_c2w, np.stack(hist_c2w)), dtype=np.float64))
    return np.atleast_1d(geo.wrap_deg(b))  # (-180, 180]: straight behind is +180 (column 0), like the ring


def path_polar(path_xy: np.ndarray, min_dist: float = MIN_PATH_DIST_M):
    """System1 path [33, 2] (x forward, y left) -> (bearing deg, NaN closer than min_dist; distance m)."""
    p = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    dist = np.hypot(p[:, 0], p[:, 1])
    bearing = np.atleast_1d(geo.wrap_deg(np.degrees(np.arctan2(p[:, 1], p[:, 0]))))
    bearing[dist < min_dist] = np.nan
    return bearing.astype(np.float32), dist.astype(np.float32)


def next_steps(call_steps: Sequence[int], episode_steps: int) -> np.ndarray:
    """Step of the next call (any kind) for each call in call order; the episode's step count for the last."""
    s = [int(x) for x in call_steps]
    return np.asarray(s[1:] + [int(episode_steps)], dtype=np.int32) if s else np.zeros(0, np.int32)


def call_state(kind: Optional[str], ready: bool) -> str:
    if ready:
        return "ready"
    if kind == "trajectory":
        return "warmup"
    return str(kind) if kind else "unknown"


def net_turn(actions: Sequence[int]) -> float:
    a = [int(x) for x in actions]
    return TURN_DEG * (a.count(LEFT) - a.count(RIGHT))


def wrap(d):
    return np.mod(np.asarray(d, dtype=np.float64) + 180.0, 360.0) - 180.0


def point_at_distance(bearing: np.ndarray, dist: np.ndarray, d: float = 1.0) -> float:
    """Bearing of the first path point at least ``d`` m away (the last point if none is)."""
    far = np.nonzero(np.asarray(dist) >= d)[0]
    i = int(far[0]) if far.size else len(dist) - 1
    return float(bearing[i])


# --------------------------------------------------------------------------- #
# Episode -> timeline
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pad(rows: List[Sequence[int]], width: int, fill: int = -1, dtype=np.int32) -> np.ndarray:
    out = np.full((len(rows), width), fill, dtype=dtype)
    for i, r in enumerate(rows):
        out[i, :len(r)] = list(r)[:width]
    return out


def step_actions(actions: Sequence[dict], n_steps: int):
    """step_action int8 [S]: the action executed at each step (first record per step), -1 none; + duplicates."""
    top = max([n_steps] + [int(a["step_before"]) + 1 for a in actions])
    out = np.full(top, -1, dtype=np.int8)
    dup = 0
    for a in actions:
        s = int(a["step_before"])
        if out[s] >= 0:
            dup += 1
            continue
        out[s] = int(a["action"])
    return out, dup


def build_episode(ep: "br.Episode", record: dict, bundle_meta: Optional[dict], memberships: list):
    """(arrays, sidecar json) of one episode."""
    calls = ep.trace["calls"]
    outcome = ep.outcome()
    states = sorted(ep.steps["states"])
    episode_steps = int(outcome["steps"]) if outcome else (states[-1] if states else 0)
    executed, _ = br.executed_by_call(calls, ep.steps["actions"])
    warnings = []

    c_index, c_step, c_kind, c_ready, c_first, c_out = [], [], [], [], [], []
    for c in calls:
        resp = c.get("response") or {}
        kind = resp.get("kind")
        c_index.append(int(c["system2_call_index"]))
        c_step.append(int(c["current_capture_step"]))
        c_kind.append(str(kind))
        c_ready.append(kind == "trajectory" and resp.get("ppa_applied") is True)
        c_first.append(str(resp.get("native_first_output") or ""))
        c_out.append(str(resp.get("llm_output") or ""))
    if any(b < a for a, b in zip(c_step, c_step[1:])):
        raise ValueError(f"{ep.ep_key}: call steps are not non-decreasing in call order: {c_step}")
    c_next = next_steps(c_step, episode_steps)

    # the records' view of the same calls must agree (same trace, joined by [F])
    rec_calls = {int(r["call_index"]): r for r in record.get("calls") or []}
    for i, idx in enumerate(c_index):
        r = rec_calls.get(idx)
        if r is None or int(r["step"]) != c_step[i] or bool(r["ready"]) != c_ready[i] or str(r["kind"]) != c_kind[i]:
            raise ValueError(f"{ep.ep_key} call {idx}: trace (step {c_step[i]}, {c_kind[i]}, ready {c_ready[i]}) "
                             f"differs from records ({r and (r['step'], r['kind'], r['ready'])})")

    R = int(sum(c_ready))
    A = max([4] + [len(executed.get(i, [])) for i in c_index])
    out = {
        "bearing_deg": bearing_axis(),
        "call_index": np.zeros(R, np.int32), "step": np.zeros(R, np.int32), "next_step": np.zeros(R, np.int32),
        "hist_ring": np.zeros((R, RING_WIDTH), np.float32),
        "hist_pred_peak_bearing": np.full((R, NUM_SLOTS), np.nan, np.float32),
        "hist_pred_conf": np.full((R, NUM_SLOTS), np.nan, np.float32),
        "hist_gt_bearing": np.full((R, NUM_SLOTS), np.nan, np.float32),
        "hist_gt_visible": np.zeros((R, NUM_SLOTS), bool),
        "hist_steps": np.full((R, NUM_SLOTS), -1, np.int32), "hist_mask": np.zeros((R, NUM_SLOTS), bool),
        "fut_ring": np.zeros((R, NUM_BINS, RING_WIDTH), np.float32),
        "s1_path_bearing": np.full((R, PATH_POINTS), np.nan, np.float32),
        "s1_path_dist": np.full((R, PATH_POINTS), np.nan, np.float32),
        "exec_actions": np.full((R, A), -1, np.int8),
    }
    r = 0
    for i, c in enumerate(calls):
        if not c_ready[i]:
            continue
        idx, step = c_index[i], c_step[i]
        hist_steps = [int(s) for s in (c.get("history_capture_steps") or [])]
        if c["_npz"] is None:
            raise ValueError(f"{ep.ep_key} call {idx}: ready call without an npz")
        with np.load(c["_npz"], allow_pickle=False) as z:
            gated = br.npz_array(z, "hist_heatmaps_gated").astype(np.float32)
            none_p = br.npz_array(z, "hist_none_probability").astype(np.float32)
            mask = (br.npz_array(z, "hist_mask").astype(bool) if "hist_mask" in z.files
                    else np.arange(NUM_SLOTS) < len(hist_steps))
            fut = br.npz_array(z, "fut_heatmaps_gated").astype(np.float32)
            path = br.selected_path(c, z)
        if path is None:
            raise ValueError(f"{ep.ep_key} call {idx}: ready call without selected_path_xy")
        if int(mask.sum()) != len(hist_steps) or not mask[:len(hist_steps)].all():
            raise ValueError(f"{ep.ep_key} call {idx}: hist_mask {mask.astype(int).tolist()} vs {len(hist_steps)} "
                             "history steps")
        # GT: label visibility exactly as H1 (build_records._ready_call), bearing of the camera centre
        _, depth, _ = ep.render(step)
        hist_c2w = [ep.cam(s) for s in hist_steps]
        gmap, gvis = gt.history_labels(hist_c2w, ep.cam(step), depth)
        gmap, gvis, _ = gt.pad_slots(gmap, gvis)
        cls, _, _ = gt.gt_view_class_and_peak(gmap, gvis)
        visible = (cls > 0) & mask
        gb = np.full(NUM_SLOTS, np.nan)
        gb[:len(hist_steps)] = gt_bearings(ep.cam(step), hist_c2w)
        pb, conf = pred_peaks(gated, none_p, mask)
        sb, sd = path_polar(path)

        out["call_index"][r], out["step"][r], out["next_step"][r] = idx, step, c_next[i]
        out["hist_ring"][r] = hist_ring(gated, none_p, mask)
        out["hist_pred_peak_bearing"][r], out["hist_pred_conf"][r] = pb, conf
        out["hist_gt_bearing"][r] = np.where(visible, gb, np.nan)
        out["hist_gt_visible"][r] = visible
        out["hist_steps"][r, :len(hist_steps)] = hist_steps
        out["hist_mask"][r] = mask
        out["fut_ring"][r] = fut_rings(fut)
        out["s1_path_bearing"][r], out["s1_path_dist"][r] = sb, sd
        ex = executed.get(idx, [])
        out["exec_actions"][r, :len(ex)] = ex
        r += 1

    sa, dup = step_actions(ep.steps["actions"], episode_steps)
    if dup:
        warnings.append(f"{dup} steps with more than one action record (first kept)")
    ready_steps = [s for s, ok in zip(c_step, c_ready) if ok]
    keys = record.get("key_steps") or []
    ready_idx = set(out["call_index"].tolist())
    for k in keys:
        if int(k["call_index"]) not in ready_idx:
            raise ValueError(f"{ep.ep_key}: key moment {k['label']} call {k['call_index']} is not a ready call")
        j = int(np.nonzero(out["call_index"] == int(k["call_index"]))[0][0])
        if int(out["step"][j]) != int(k["step"]):
            raise ValueError(f"{ep.ep_key}: key moment {k['label']} step {k['step']} vs call step {out['step'][j]}")
    out.update({
        "calls_index": np.asarray(c_index, np.int32), "calls_step": np.asarray(c_step, np.int32),
        "calls_next_step": c_next, "calls_kind": np.asarray(c_kind, dtype=str),
        "calls_ready": np.asarray(c_ready, bool),
        "calls_state": np.asarray([call_state(k, rd) for k, rd in zip(c_kind, c_ready)], dtype=str),
        "calls_system2_first": np.asarray(c_first, dtype=str), "calls_system2_output": np.asarray(c_out, dtype=str),
        "step_action": sa,
        "first_ready_step": np.int32(ready_steps[0] if ready_steps else -1),
        "episode_steps": np.int32(episode_steps),
        "key_labels": np.asarray([str(k["label"]) for k in keys], dtype=str).reshape(-1),
        "key_call_index": np.asarray([int(k["call_index"]) for k in keys], np.int32),
        "key_step": np.asarray([int(k["step"]) for k in keys], np.int32),
    })
    for name in ("calls_kind", "calls_state", "calls_system2_first", "calls_system2_output", "key_labels"):
        if out[name].size == 0:
            out[name] = np.zeros(0, dtype="<U1")

    primary = memberships[0] if memberships else {"category": None, "rank": None}
    states_count: Dict[str, int] = {}
    for s in out["calls_state"].tolist():
        states_count[s] = states_count.get(s, 0) + 1
    meta = {
        "schema": SCHEMA, "ep_key": ep.ep_key, "scene_id": ep.scene_id, "episode_id": ep.episode_id,
        "category": primary["category"], "rank": primary["rank"],
        "is_main": bool((bundle_meta or {}).get("is_main", False)),
        "is_main_any": bool(bd.is_main_case(bundle_meta)) if bundle_meta else False,
        "memberships": [{"category": m["category"], "rank": m["rank"]} for m in memberships],
        "outcome": outcome,
        "key_steps": [{"label": k["label"], "call_index": int(k["call_index"]), "step": int(k["step"]),
                       "branch": k.get("branch")} for k in keys],
        "counts": {"n_calls": len(calls), "n_ready": R, "n_key": len(keys), "episode_steps": episode_steps,
                   "first_ready_step": int(out["first_ready_step"]), "calls_by_state": states_count,
                   "n_confident_slots": int(np.nansum(out["hist_pred_conf"] >= CONF_MIN)),
                   "n_gt_visible_slots": int(out["hist_gt_visible"].sum()),
                   "n_valid_slots": int(out["hist_mask"].sum())},
        "conventions": CONVENTIONS,
        "arrays": {k: {"shape": list(np.shape(v)), "dtype": str(np.asarray(v).dtype)} for k, v in out.items()},
        "warnings": warnings,
    }
    return out, meta


def episode_inputs(ep: "br.Episode", paths: dict) -> Dict[str, str]:
    """sha256 of every input file of one episode (trace files by name)."""
    files = dict(paths)
    files["steps.jsonl"] = ep.steps_dir / "steps.jsonl"
    for c in ep.trace["calls"]:
        idx = int(c["system2_call_index"])
        files[f"trace/call_{idx:03d}.json"] = ep.trace_dir / f"call_{idx:03d}.json"
        if c["_npz"] is not None:
            files[f"trace/call_{idx:03d}.npz"] = Path(c["_npz"])
    return {name: {"path": str(p), "sha256": sha256_file(Path(p))} for name, p in files.items() if Path(p).is_file()}


def write_timeline(out_dir: Path, ep_key: str, arrays: dict, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{ep_key}_timeline.npz", **arrays)
    meta = dict(meta)
    meta["npz_sha256"] = sha256_file(out_dir / f"{ep_key}_timeline.npz")
    (out_dir / f"{ep_key}_timeline.json").write_text(
        json.dumps(br._clean(meta), indent=1, ensure_ascii=False, default=br._json_default) + "\n", encoding="utf-8")


def load_timeline(path) -> dict:
    """{"arrays", "meta"} of ``<ep_key>_timeline`` given its .npz, its .json or the common stem."""
    p = Path(path)
    stem = p.with_suffix("") if p.suffix in (".npz", ".json") else p
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA:
        raise ValueError(f"{stem}.json: schema {meta.get('schema')!r}, expected {SCHEMA}")
    return {"arrays": arrays, "meta": meta}


# --------------------------------------------------------------------------- #
# Self-check on the written files
# --------------------------------------------------------------------------- #
def _stats(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0, "median": None, "p90": None, "max": None}
    return {"n": int(x.size), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(x.max())}


def check_ring_vs_peaks(a: dict) -> dict:
    """(0) ring value at each confident slot's peak bearing (nearest column +-1) / the slot's confidence."""
    ratios, lateral = [], []
    for r in range(a["hist_ring"].shape[0]):
        for k in range(NUM_SLOTS):
            b, c = a["hist_pred_peak_bearing"][r, k], a["hist_pred_conf"][r, k]
            if not (np.isfinite(b) and np.isfinite(c) and c >= CONF_MIN):
                continue
            col = int(bearing_to_column(b))
            cols = [(col + d) % RING_WIDTH for d in (-1, 0, 1)]
            ratio = float(a["hist_ring"][r, cols].max()) / float(c)
            ratios.append(ratio)
            if 30.0 <= abs(float(b)) <= 150.0:
                lateral.append(ratio)
    return {"ratios": ratios, "lateral": lateral}


def check_peak_vs_gt(a: dict) -> dict:
    """(1) circular |pred peak - GT| on confident, GT-visible slots, plus the mirrored / swapped readings."""
    pb, gb, cf = a["hist_pred_peak_bearing"], a["hist_gt_bearing"], a["hist_pred_conf"]
    sel = np.isfinite(pb) & np.isfinite(gb) & (np.nan_to_num(cf) >= CONF_MIN) & a["hist_gt_visible"]
    p, g = pb[sel].astype(np.float64), gb[sel].astype(np.float64)
    lat = (np.abs(g) >= 30.0) & (np.abs(g) <= 150.0)
    return {"identity": geo.circular_abs_diff(p, g) if p.size else np.zeros(0),
            "mirror": geo.circular_abs_diff(p, -g) if p.size else np.zeros(0),
            "swap": geo.circular_abs_diff(p, g + 180.0) if p.size else np.zeros(0),
            "lateral_identity": geo.circular_abs_diff(p[lat], g[lat]) if lat.any() else np.zeros(0),
            "lateral_mirror": geo.circular_abs_diff(p[lat], -g[lat]) if lat.any() else np.zeros(0)}


def check_gt_vs_label_pixel(ep: "br.Episode", a: dict) -> List[float]:
    """(1b) atan2 GT bearing vs the bearing of the label code's GT peak pixel, every ready call."""
    out = []
    for r in range(a["step"].shape[0]):
        step = int(a["step"][r])
        hs = [int(s) for s in a["hist_steps"][r] if s >= 0]
        _, depth, _ = ep.render(step)
        gmap, gvis = gt.history_labels([ep.cam(s) for s in hs], ep.cam(step), depth)
        gmap, gvis, _ = gt.pad_slots(gmap, gvis)
        cls, row, col = gt.gt_view_class_and_peak(gmap, gvis)
        for k in np.nonzero(cls > 0)[0]:
            b, _ = geo.pixel_to_bearing_elev(int(cls[k] - 1), float(col[k]), float(row[k]))
            out.append(float(geo.circular_abs_diff(b, a["hist_gt_bearing"][r, k])))
    return out


def _trace_tensors(ep: "br.Episode", call_index: int) -> dict:
    c = next(c for c in ep.trace["calls"] if int(c["system2_call_index"]) == int(call_index))
    with np.load(c["_npz"], allow_pickle=False) as z:
        return {"hist_pred": br.npz_array(z, "hist_heatmaps_gated").astype(np.float32),
                "hist_none": br.npz_array(z, "hist_none_probability").astype(np.float32),
                "hist_mask": (br.npz_array(z, "hist_mask").astype(bool) if "hist_mask" in z.files else None),
                "fut_pred": br.npz_array(z, "fut_heatmaps_gated").astype(np.float32)}


def _max_circ(a, b) -> tuple:
    """(max circular |a - b| over entries finite in both, number of entries finite in only one)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    both = np.isfinite(a) & np.isfinite(b)
    d = geo.circular_abs_diff(a[both], b[both]) if both.any() else np.zeros(0)
    return (float(np.max(d)) if np.size(d) else 0.0), int(np.sum(np.isfinite(a) != np.isfinite(b)))


def check_key_rows(ep: "br.Episode", a: dict, bundle_json: Path) -> dict:
    """(2) key-moment rows vs the v1 bundle: the bundle's tensors are the trace's, and the stored ring, GT
    visibility, GT / peak / path bearings are what the bundle's tensors give."""
    b = bd.load_bundle(bundle_json, validate=False)
    res = {"n_keys": len(b.keys), "tensor_max_abs_diff": 0.0, "ring_max_abs_diff": 0.0, "fut_ring_max_abs_diff": 0.0,
           "gt_visible_mismatch": 0, "gt_bearing_max_deg": 0.0, "pred_peak_max_deg": 0.0, "pred_peak_nan_mismatch": 0,
           "path_max_deg": 0.0, "path_nan_mismatch": 0, "missing_rows": 0,
           "labels_match": [str(x) for x in a["key_labels"]] == [k.meta["label"] for k in b.keys]}
    for ks in b.keys:
        rows = np.nonzero(a["call_index"] == int(ks.meta["call_index"]))[0]
        if not rows.size or int(a["step"][rows[0]]) != int(ks.meta["step"]):
            res["missing_rows"] += 1
            continue
        r = int(rows[0])
        # the bundle's tensors are the trace tensors this row was built from
        tr = _trace_tensors(ep, int(ks.meta["call_index"]))
        mask_b = np.asarray(ks.hist_mask, bool)
        diffs = [np.abs(np.asarray(ks.hist_pred, np.float32) - tr["hist_pred"]).max(),
                 np.abs(np.asarray(ks.hist_none, np.float32) - tr["hist_none"]).max(),
                 np.abs(np.asarray(ks.fut_pred, np.float32) - tr["fut_pred"]).max(),
                 float(np.any(mask_b != a["hist_mask"][r]))]
        if tr["hist_mask"] is not None:
            diffs.append(float(np.any(mask_b != tr["hist_mask"])))
        res["tensor_max_abs_diff"] = max(res["tensor_max_abs_diff"], float(max(diffs)))
        # the stored rings are the bundle's tensors on the bearing axis
        res["ring_max_abs_diff"] = max(res["ring_max_abs_diff"], float(np.abs(
            hist_ring(ks.hist_pred, ks.hist_none, ks.hist_mask) - a["hist_ring"][r]).max()))
        res["fut_ring_max_abs_diff"] = max(res["fut_ring_max_abs_diff"], float(np.abs(
            fut_rings(np.asarray(ks.fut_pred, np.float32)) - a["fut_ring"][r]).max()))
        # GT: the bundle's GT peak pixel (validate.py semantics) vs the stored visibility and atan2 bearing
        peak = np.asarray(ks.hist_gt_peak, dtype=np.float64)
        vis_b = (peak[:, 0] >= 0) & mask_b
        res["gt_visible_mismatch"] += int(np.sum(vis_b != a["hist_gt_visible"][r]))
        for k in np.nonzero(vis_b & a["hist_gt_visible"][r])[0]:
            gbear, _ = geo.pixel_to_bearing_elev(int(peak[k, 0]), float(peak[k, 2]), float(peak[k, 1]))
            res["gt_bearing_max_deg"] = max(res["gt_bearing_max_deg"],
                                            float(geo.circular_abs_diff(gbear, a["hist_gt_bearing"][r, k])))
        # predicted peaks: the v1 orange peaks (figures.bundle.pred_history_peaks) of the confident slots
        pb, _ = pred_peaks(ks.hist_pred, ks.hist_none, ks.hist_mask)
        d, n = _max_circ(pb, a["hist_pred_peak_bearing"][r])
        res["pred_peak_max_deg"], res["pred_peak_nan_mismatch"] = max(res["pred_peak_max_deg"], d), \
            res["pred_peak_nan_mismatch"] + n
        for k, view, row, col in bd.pred_history_peaks(ks.hist_pred, ks.hist_none, ks.hist_mask):
            v1b, _ = geo.pixel_to_bearing_elev(view, col, row)
            res["pred_peak_max_deg"] = max(res["pred_peak_max_deg"], float(geo.circular_abs_diff(
                v1b, a["hist_pred_peak_bearing"][r, k])) if np.isfinite(a["hist_pred_peak_bearing"][r, k]) else 999.0)
        # System1 path: the v1 strip's directions (figures.bundle.path_directions of path_cam)
        bb, _, keep = bd.path_directions(ks.path_cam)
        full = np.full(PATH_POINTS, np.nan)
        full[keep] = bb
        d, n = _max_circ(full, a["s1_path_bearing"][r])
        res["path_max_deg"], res["path_nan_mismatch"] = max(res["path_max_deg"], d), res["path_nan_mismatch"] + n
    return res


def check_turn_shift(a: dict) -> List[dict]:
    """(3) consecutive ready calls: executed net turn in [step_r, step_r+1) vs the shift of shared GT bearings."""
    out = []
    sa = a["step_action"]
    for r in range(a["step"].shape[0] - 1):
        s0, s1 = int(a["step"][r]), int(a["step"][r + 1])
        acts = [int(x) for x in sa[s0:s1] if x >= 0]
        turn = net_turn(acts)
        shared = []
        for k0 in range(NUM_SLOTS):
            hs = int(a["hist_steps"][r, k0])
            if hs < 0 or not a["hist_gt_visible"][r, k0]:
                continue
            k1 = np.nonzero(a["hist_steps"][r + 1] == hs)[0]
            if k1.size and a["hist_gt_visible"][r + 1, int(k1[0])]:
                shared.append(float(wrap(a["hist_gt_bearing"][r + 1, int(k1[0])] - a["hist_gt_bearing"][r, k0])))
        out.append({"step": s0, "next_ready_step": s1, "net_turn_deg": turn, "n_forward": acts.count(FORWARD),
                    "n_shared": len(shared), "median_shift_deg": float(np.median(shared)) if shared else None})
    return out


def check_path(a: dict) -> dict:
    """(4) System1 path bearing at ~1 m and at the endpoint vs the executed chunk."""
    fwd_1m, fwd_end, turn_rows = [], [], []
    for r in range(a["step"].shape[0]):
        acts = [int(x) for x in a["exec_actions"][r] if x >= 0]
        b1 = point_at_distance(a["s1_path_bearing"][r], a["s1_path_dist"][r], 1.0)
        be = float(a["s1_path_bearing"][r][-1])
        if acts and all(x == FORWARD for x in acts) and len(acts) >= 3:
            fwd_1m.append(abs(b1))
            fwd_end.append(abs(be))
        t = net_turn(acts)
        if abs(t) >= CHECK["turn_min_deg"]:
            turn_rows.append((t, b1, be))
    return {"forward_abs_1m": fwd_1m, "forward_abs_end": fwd_end, "turn_rows": turn_rows}


def self_check(out_dir: Path, ep_keys: Sequence[str], episodes: Dict[str, "br.Episode"], records_dir: Path) -> dict:
    per_ep, ring_all, ring_lat = {}, [], []
    peak = {k: [] for k in ("identity", "mirror", "swap", "lateral_identity", "lateral_mirror")}
    peak_t13 = {k: [] for k in peak}
    peak_by_cat: Dict[str, list] = {}
    gt_pix, turns, fwd1, fwde, trows, keys = [], [], [], [], [], {}
    for ep_key in ep_keys:
        t = load_timeline(out_dir / f"{ep_key}_timeline.npz")
        a, meta = t["arrays"], t["meta"]
        cat = meta["category"]
        rv = check_ring_vs_peaks(a)
        ring_all += rv["ratios"]
        ring_lat += rv["lateral"]
        pv = check_peak_vs_gt(a)
        for k in peak:
            peak[k] += list(np.asarray(pv[k], dtype=np.float64))
            if cat in ("T1", "T3"):
                peak_t13[k] += list(np.asarray(pv[k], dtype=np.float64))
        peak_by_cat.setdefault(cat, []).extend(list(np.asarray(pv["identity"], dtype=np.float64)))
        gp = check_gt_vs_label_pixel(episodes[ep_key], a)
        gt_pix += gp
        tr = check_turn_shift(a)
        turns += [dict(x, ep_key=ep_key) for x in tr]
        pc = check_path(a)
        fwd1 += pc["forward_abs_1m"]
        fwde += pc["forward_abs_end"]
        trows += pc["turn_rows"]
        keys[ep_key] = check_key_rows(episodes[ep_key], a, records_dir / f"{ep_key}_bundle.json")
        per_ep[ep_key] = {"category": cat, "n_ready": int(a["step"].shape[0]),
                          "peak_vs_gt_median_deg": _stats(pv["identity"])["median"],
                          "n_peak_vs_gt": int(np.size(pv["identity"])),
                          "ring_at_peak_min_ratio": float(min(rv["ratios"])) if rv["ratios"] else None,
                          "gt_vs_label_pixel_max_deg": float(max(gp)) if gp else None,
                          "key_rows": keys[ep_key]}

    ring_all, ring_lat = np.asarray(ring_all), np.asarray(ring_lat)
    c0 = {"n_confident_slots": int(ring_all.size), "n_lateral": int(ring_lat.size),
          "share_ratio_ge_min": float(np.mean(ring_all >= CHECK["ring_at_peak_min_ratio"])) if ring_all.size else None,
          "share_lateral_ratio_ge_min": (float(np.mean(ring_lat >= CHECK["ring_at_peak_min_ratio"]))
                                         if ring_lat.size else None),
          "ratio": _stats(ring_all)}
    c0["ratio"]["min"] = float(ring_all.min()) if ring_all.size else None
    c0["pass"] = bool(ring_all.size and c0["share_ratio_ge_min"] >= CHECK["ring_at_peak_min_share"]
                      and (not ring_lat.size or c0["share_lateral_ratio_ge_min"] >= CHECK["ring_at_peak_min_share"]))

    def block(d):
        return {k: _stats(v) for k, v in d.items()}

    c1 = {"T1_T3": block(peak_t13), "all": block(peak),
          "by_category_identity": {c: _stats(v) for c, v in sorted(peak_by_cat.items())},
          "share_within_15deg_T1_T3": (float(np.mean(np.asarray(peak_t13["identity"]) <= 15.0))
                                       if peak_t13["identity"] else None),
          "gt_atan2_vs_label_peak_pixel_deg": _stats(gt_pix)}
    s = c1["T1_T3"]
    c1["pass"] = bool(s["identity"]["n"] and s["identity"]["median"] <= CHECK["peak_vs_gt_median_max_deg"]
                      and s["identity"]["median"] < s["mirror"]["median"] and s["identity"]["median"] < s["swap"]["median"]
                      and (not s["lateral_identity"]["n"]
                           or s["lateral_identity"]["median"] < s["lateral_mirror"]["median"])
                      and c1["gt_atan2_vs_label_peak_pixel_deg"]["n"]
                      and c1["gt_atan2_vs_label_peak_pixel_deg"]["max"] <= CHECK["gt_vs_label_pixel_max_deg"])

    def worst(name):
        return float(max([v[name] for v in keys.values()] + [0.0]))

    def total(name):
        return int(sum(v[name] for v in keys.values()))

    c2 = {"episodes": keys, "n_key_rows": total("n_keys"), "missing_rows": total("missing_rows"),
          "tensor_max_abs_diff": worst("tensor_max_abs_diff"), "ring_max_abs_diff": worst("ring_max_abs_diff"),
          "fut_ring_max_abs_diff": worst("fut_ring_max_abs_diff"), "gt_visible_mismatch": total("gt_visible_mismatch"),
          "gt_bearing_max_deg": worst("gt_bearing_max_deg"), "pred_peak_max_deg": worst("pred_peak_max_deg"),
          "pred_peak_nan_mismatch": total("pred_peak_nan_mismatch"), "path_max_deg": worst("path_max_deg"),
          "path_nan_mismatch": total("path_nan_mismatch"),
          "labels_match": all(v["labels_match"] for v in keys.values())}
    c2["pass"] = bool(c2["n_key_rows"] and c2["missing_rows"] == 0 and c2["tensor_max_abs_diff"] == 0.0
                      and c2["ring_max_abs_diff"] == 0.0 and c2["fut_ring_max_abs_diff"] == 0.0
                      and c2["gt_visible_mismatch"] == 0 and c2["gt_bearing_max_deg"] <= CHECK["key_gt_vs_bundle_max_deg"]
                      and c2["pred_peak_max_deg"] <= CHECK["key_peak_max_deg"] and c2["pred_peak_nan_mismatch"] == 0
                      and c2["path_max_deg"] <= CHECK["key_path_max_deg"] and c2["path_nan_mismatch"] == 0
                      and c2["labels_match"])

    left = [t for t in turns if t["net_turn_deg"] >= CHECK["turn_min_deg"] and t["median_shift_deg"] is not None]
    right = [t for t in turns if t["net_turn_deg"] <= -CHECK["turn_min_deg"] and t["median_shift_deg"] is not None]
    straight = [t for t in turns if t["net_turn_deg"] == 0 and t["median_shift_deg"] is not None]
    with_shift = [t for t in turns if t["median_shift_deg"] is not None]
    x = np.asarray([t["net_turn_deg"] for t in with_shift])
    y = np.asarray([t["median_shift_deg"] for t in with_shift])
    c3 = {"n_pairs": len(turns), "n_pairs_with_shared_frames": len(with_shift),
          "left": {"n": len(left), "share_shift_negative": float(np.mean([t["median_shift_deg"] < 0 for t in left]))
                   if left else None, "median_shift_deg": _stats([t["median_shift_deg"] for t in left])["median"],
                   "median_net_turn_deg": _stats([t["net_turn_deg"] for t in left])["median"],
                   "median_residual_deg": _stats([t["median_shift_deg"] + t["net_turn_deg"] for t in left])["median"]},
          "right": {"n": len(right), "share_shift_positive": float(np.mean([t["median_shift_deg"] > 0 for t in right]))
                    if right else None, "median_shift_deg": _stats([t["median_shift_deg"] for t in right])["median"],
                    "median_net_turn_deg": _stats([t["net_turn_deg"] for t in right])["median"],
                    "median_residual_deg": _stats([t["median_shift_deg"] + t["net_turn_deg"] for t in right])["median"]},
          "no_turn": {"n": len(straight), "median_abs_shift_deg": _stats([abs(t["median_shift_deg"])
                                                                          for t in straight])["median"]},
          "pearson_shift_vs_minus_turn": (float(np.corrcoef(y, -x)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0
                                          else None),
          "median_shared_frames_per_pair": _stats([t["n_shared"] for t in turns])["median"]}
    c3["pass"] = bool(left and right and c3["left"]["share_shift_negative"] >= CHECK["turn_sign_min_share"]
                      and c3["right"]["share_shift_positive"] >= CHECK["turn_sign_min_share"])

    agree = [np.sign(b1) == np.sign(t) for t, b1, _ in trows if np.isfinite(b1)]
    agree_end = [np.sign(be) == np.sign(t) for t, _, be in trows if np.isfinite(be)]
    c4 = {"n_forward_calls": len(fwd1), "forward_abs_bearing_1m_deg": _stats(fwd1),
          "forward_abs_bearing_end_deg": _stats(fwde),
          "n_turn_calls": len(trows), "share_sign_1m_equals_turn": float(np.mean(agree)) if agree else None,
          "share_sign_end_equals_turn": float(np.mean(agree_end)) if agree_end else None}
    c4["pass"] = bool(fwd1 and c4["forward_abs_bearing_1m_deg"]["median"] <= CHECK["forward_1m_median_max_deg"]
                      and agree and c4["share_sign_1m_equals_turn"] >= CHECK["path_turn_sign_min_share"])

    checks = {"0_ring_vs_peaks": c0, "1_peak_vs_gt": c1, "2_key_rows_vs_v1_bundle": c2, "3_turn_shifts_gt": c3,
              "4_system1_path": c4}
    return {"schema": SCHEMA + "-self-check", "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "thresholds": CHECK, "ok": all(c["pass"] for c in checks.values()), "checks": checks,
            "episodes": per_ep, "turn_pairs": turns}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--exp-root", type=Path, default=br.EXP_ROOT)
    p.add_argument("--run", default=env("EXP19_RUN"), help="runs/<run> to read (required)")
    p.add_argument("--candidates", type=Path, default=None, help="default <exp-root>/cases/candidates.json")
    p.add_argument("--renders-dir", type=Path, default=None, help="default <exp-root>/renders")
    p.add_argument("--records-dir", type=Path, default=None, help="[F] records, read only; default <exp-root>/records")
    p.add_argument("--out-dir", type=Path, default=None, help="default <exp-root>/records_v2")
    p.add_argument("--episodes", nargs="*", default=None, help="ep_keys (default: every candidate in the run)")
    p.add_argument("--self-check", action="store_true", help="check the conventions on the written files")
    p.add_argument("--no-build", action="store_true", help="with --self-check: check existing files only")
    args = p.parse_args(argv)
    if not args.run:
        p.error("--run (or EXP19_RUN) is required")
    root = args.exp_root
    args.candidates = args.candidates or root / "cases" / "candidates.json"
    args.renders_dir = args.renders_dir or root / "renders"
    args.records_dir = args.records_dir or root / "records"
    args.out_dir = args.out_dir or root / "records_v2"
    forbidden = {Path(d).resolve() for d in (args.records_dir, root / "records", root / "metrics", root / "figures")}
    if args.out_dir.resolve() in forbidden:
        p.error(f"--out-dir {args.out_dir} would write into records/, metrics/ or figures/ (v1 outputs, read only)")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    cands = br.load_candidates(args.candidates)
    found = br.discover_run(args.exp_root / "runs" / args.run)
    keys = [k for k in (args.episodes or sorted(cands["members"])) if k in found]
    missing = [k for k in (args.episodes or sorted(cands["members"])) if k not in found]
    if missing:
        log(f"not in the run, skipped: {missing}")
    episodes = {}
    for ep_key in keys:
        ep = br.Episode(ep_key, found[ep_key], args.renders_dir)
        episodes[ep_key] = ep
        if args.no_build:
            continue
        record_path = args.records_dir / f"{ep_key}.json"
        bundle_path = args.records_dir / f"{ep_key}_bundle.json"
        record = br.read_json(record_path)
        bundle_meta = br.read_json(bundle_path) if bundle_path.is_file() else None
        members = cands["members"].get(ep_key, [])
        if record.get("memberships") and [m["category"] for m in record["memberships"]] != \
                [m["category"] for m in members]:
            raise ValueError(f"{ep_key}: records memberships differ from candidates.json")
        arrays, meta = build_episode(ep, record, bundle_meta, members)
        meta.update(run=args.run, created_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(), git_sha=br.git_sha(),
                    source="scripts/exp19/build_timeline.py",
                    inputs=episode_inputs(ep, {"record": record_path, "bundle_json": bundle_path,
                                               "bundle_npz": bundle_path.with_suffix(".npz"),
                                               "renders": args.renders_dir / f"{ep_key}.npz",
                                               "candidates": args.candidates}))
        write_timeline(args.out_dir, ep_key, arrays, meta)
        log(f"{ep_key} ({meta['category']}): {meta['counts']['n_calls']} calls, {meta['counts']['n_ready']} ready, "
            f"first ready step {meta['counts']['first_ready_step']}, {meta['counts']['episode_steps']} steps, "
            f"keys {[(k['label'], k['step']) for k in meta['key_steps']]}")
    if not args.self_check:
        return 0
    report = self_check(args.out_dir, keys, episodes, args.records_dir)
    report.update(run=args.run, episodes_checked=keys, out_dir=str(args.out_dir))
    (args.out_dir / "timeline_self_check.json").write_text(
        json.dumps(br._clean(report), indent=1, ensure_ascii=False, default=br._json_default) + "\n", encoding="utf-8")
    for name, c in report["checks"].items():
        log(f"self-check {name}: {'PASS' if c['pass'] else 'FAIL'}")
    log(f"wrote {args.out_dir / 'timeline_self_check.json'}; ok = {report['ok']}")
    return 0 if report["ok"] else 5


if __name__ == "__main__":
    raise SystemExit(main())

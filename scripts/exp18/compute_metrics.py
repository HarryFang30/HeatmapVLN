#!/usr/bin/env python3
"""EXP-18: pre-registered metrics, CIs and H1/H2/H3 verdicts from the History Head dumps.

Reads ``EXP_ROOT/dumps/<tier>/**/*.npz`` for tiers A-E (missing tiers are
reported, not fatal; dump schema v1 draft or v2, see dump_history_predictions.py)
and computes exactly what docs/experiments/README.md EXP-18 writes down:

* population: cache endpoint rows (``cache_endpoint_frame_ids``; t = 19, 27, ...
  plus the final frame), restricted to rows with VO poses so both arms score the
  same slots; if the tier has a clip list (``common.clip_list_path``), only its
  clips count.
* joint PCK@8 / @4: ``scripts/training/validate.py::_HeatmapJointMetricAccumulator``
  (5-way view class correct AND argmax distance in the GT view <= r px; the
  denominator is the GT-visible slots).  v2 dumps carry the exact f32 argmaxes
  (``*_view_peak_yx``, ``gt_view_class``); v1 dumps are re-scored from the
  stored f16 maps.  ``--self-check`` re-scores every tier/arm with the real
  accumulator and asserts equality.
* bearing error (deg): GT = atan2(left, forward) of the GT rel pose; prediction
  = joint argmax of ``heatmaps_gated`` over 4x64x64 converted by
  scripts/exp18/geometry.py; circular |diff|; median, P90, share <= 15 deg.
* visibility: AUROC (tie-aware) and F1 at 0.5 of 1 - none_probability vs
  "visible in any view".
* constant straight-behind floor: class back, peak (32, 32) in every view,
  never none; bearing error |180 - GT bearing|.
* 95% CIs: scene-cluster bootstrap, 10000 reps, root seed 0.  Each tier draws
  from its own stream ``SeedSequence(0, spawn_key=(i,))`` (i = index in
  "ABCDE"), so tier differences are independent resamples of the two tiers
  while arms/floor inside a tier are paired (same draws).
* descriptive strata (GT view, distance 0-2/2-5/5-10/>10 m, age, path-length
  tertile, |dy| > 1 m, E pattern); a cell with n < 100 is reported, never concluded.
* H1/H2/H3 verdicts with the ledger's thresholds, n < 100 rule and attribution rule
  (H2 归因 through the full (VO, GT) verdict table H2_ATTRIBUTION: written only
  where both arm verdicts assert it, else None with attribution_note).

Outputs (``--out-dir``, default EXP_ROOT/metrics): metrics.json, summary.md,
slots.parquet (one row per valid history slot of a scored row; ``gt_*`` columns
are ground truth, ``pred_<arm>_*`` the arm's prediction for arm vo, gt (the
GT-pose arm) and floor), rows.parquet
(per scored row: position, distance from start, GT-visible bearing span),
episodes.parquet (per clip: VO/GT PCK, median bearing error, path length,
pattern, E out-and-back turnaround frame via geometry.out_and_back_turnaround)
- the last two feed select_cases.py.  Falls back to .csv.gz without
pyarrow.

Usage (dev machine, envs/qwen25):
  cd <repo> && PYTHONDONTWRITEBYTECODE=1 <qwen25 python> -m scripts.exp18.compute_metrics [--self-check]
Env fallbacks: EXP18_ROOT (via common), EXP18_DUMPS_ROOT, EXP18_METRICS_DIR, EXP18_TIERS,
EXP18_BOOTSTRAP_REPS, EXP18_METRICS_WORKERS, EXP18_SELF_CHECK=1.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.exp18 import common  # noqa: E402
from scripts.exp18 import geometry as geo  # noqa: E402

SCHEMA = "heatmapvln-exp18-metrics-v1"
TIER_ORDER = "ABCDE"
ARMS = ("vo", "gt")
SCORED = ("vo", "gt", "floor")
FLOOR_CLASS = 1 + geo.BACK
N_MIN = 100  # strata / H3: below this, report the number, write no conclusion
CI_PCT = (2.5, 97.5)
DIST_BINS = ((0.0, 2.0, "0-2m"), (2.0, 5.0, "2-5m"), (5.0, 10.0, "5-10m"), (10.0, math.inf, ">10m"))
AGE_BINS = ((1, 10, "1-10f"), (11, 20, "11-20f"), (21, 40, "21-40f"), (41, 80, "41-80f"), (81, 10 ** 9, ">80f"))
E_PATTERNS = ("out_and_back", "loop")
VERDICT_ZH = {"support": "支持", "partial": "部分", "refute": "否定", "not_measured": "没测出来", "missing": "缺数据"}
# Ledger thresholds (EXP-18 判据), in fractions / degrees.
H1 = {"pck8_min": 0.80, "bearing_median_max": 5.0, "gain_min": 0.20, "gain_ci_low_min": 0.10,
      "refute_gain_below": 0.05, "refute_pck8_below": 0.50}
H2 = {"C": {"delta_min": -0.05, "ci_low_min": -0.10, "refute_delta_max": -0.15},
      "D": {"delta_min": -0.10, "ci_low_min": -0.15, "refute_delta_max": -0.20}}
H3 = {"support_min": 0.70, "refute_below": 0.40}
# H2 归因规则 as a full (VO verdict, GT verdict) table (ledger §5 #13).  An attribution explains a VO-arm
# FAILURE, so it is written only where both arm verdicts assert what the sentence claims: "瓶颈在里程计"
# needs VO refute + GT support, "头本身不泛化" needs VO refute + GT refute.  In every other cell the
# attribution is None and the reason is recorded (ledger §5 #28: a threshold must not speak for a claim
# its own verdict does not make).  The literal reading ("不过" = below the support line, so 没测出来
# counts as 不过) goes to metrics.json as attribution_literal_reading for the record, never to summary.md.
H2_ATTRIBUTION = {("refute", "support"): "泛化瓶颈在里程计，不在热力头",
                  ("refute", "refute"): "头本身不泛化"}
H2_NO_ATTRIBUTION_VO = {"support": "不适用：VO 臂过支持线，无失败可归因",
                        "not_measured": "不适用：VO 臂没测出来，无失败可归因",
                        "missing": "不适用：VO 臂缺数据"}
H2_NO_ATTRIBUTION_GT = {"not_measured": "不归因：VO 臂否定，但 GT 臂没测出来，分不开头与里程计",
                        "missing": "不归因：VO 臂否定，但 GT 臂缺数据"}


# --------------------------------------------------------------------------- #
# Per-clip extraction
# --------------------------------------------------------------------------- #
def endpoint_rule(frame_count: int) -> np.ndarray:
    """AMB3R endpoint rows when a dump has no cache ids: 19, 27, ... plus the final frame."""
    if frame_count < common.MIN_FRAMES_FOR_AMB3R:
        return np.zeros(0, dtype=np.int64)
    ids = set(range(common.MIN_FRAMES_FOR_AMB3R - 1, frame_count, 8)) | {frame_count - 1}
    return np.asarray(sorted(ids), dtype=np.int64)


def resolve_pattern(meta: dict, episode_id: str, clip: str) -> str:
    """E route pattern from the designed episode's ids (render/select_episodes.py naming)."""
    info = meta.get("info") if isinstance(meta.get("info"), dict) else {}
    for value in (meta.get("exp18_pattern"), info.get("exp18_pattern"), meta.get("pattern"),
                  meta.get("trajectory_id"), episode_id, meta.get("episode_id"), clip):
        text = str(value or "").lower().replace("-", "_")
        if "out_and_back" in text or "_oab" in text:
            return "out_and_back"
        if "loop" in text:
            return "loop"
    return ""


def _take_view(a: np.ndarray, view: np.ndarray) -> np.ndarray:
    """a [..., 4, *rest] -> a[..., view, *rest]."""
    idx = view.reshape(view.shape + (1,) * (a.ndim - view.ndim))
    return np.take_along_axis(a, idx, axis=view.ndim)[(slice(None),) * view.ndim + (0,)]


def _view_peaks(maps: np.ndarray) -> np.ndarray:
    """Per-view first-occurrence argmax [..., 4, H, W] -> [..., 4, 2] (y, x)."""
    y, x = geo.argmax_pixel(maps)
    return np.stack([y, x], axis=-1).astype(np.int64)


def load_clip(path: str, tier: str, peaks_from: str = "auto", keep_arrays: bool = False) -> dict:
    """One dump npz -> per-slot / per-row / per-episode records of its query rows.

    peaks_from: "auto" uses the dump's exact argmax fields when present (v2),
    "arrays" always re-derives them from the stored f16 maps (v1 behaviour).
    keep_arrays: also return the raw tensors of the query rows (self-check).
    """
    with np.load(path, allow_pickle=False) as z:
        files = set(z.files)

        def s(key, default=""):
            return str(z[key]) if key in files else default

        scene, clip, schema = s("scene"), s("clip"), s("schema")
        meta = json.loads(s("meta_json", "{}") or "{}")
        episode_id = s("episode_id") or str(meta.get("episode_id", ""))
        arms = [str(a) for a in z["arms"]] if "arms" in files else []
        frame_count = int(z["frame_count"])
        clip_c2w = z["clip_c2w"].astype(np.float64)
        endpoints = z["cache_endpoint_frame_ids"].astype(np.int64) if "cache_endpoint_frame_ids" in files else np.zeros(0, np.int64)
        t_all = z["current_frame_ids"].astype(np.int64)
        vo_avail_all = z["vo_available"].astype(bool) if "vo_available" in files else np.zeros(len(t_all), bool)
        if "vo" not in arms:
            vo_avail_all = np.zeros(len(t_all), bool)
        query_ids = endpoints if endpoints.size else endpoint_rule(frame_count)
        q = np.nonzero(np.isin(t_all, query_ids))[0]
        Q = len(q)
        exact = peaks_from == "auto" and {"gt_view_class", "gt_view_peak_yx"} <= files and all(
            {f"pred_{a}_view_peak_yx", f"pred_{a}_gated_argmax"} <= files for a in arms)

        hist_mask = z["history_mask"][q].astype(bool)
        gt_vis = z["gt_visibility"][q].astype(np.float32) > 0.5
        raw = {}
        if exact:
            gt_class = z["gt_view_class"][q].astype(np.int64)
            gt_peaks = z["gt_view_peak_yx"][q].astype(np.int64)
        else:
            gt_hm = z["gt_heatmap"][q].astype(np.float32)
            peak = gt_hm.reshape(gt_hm.shape[:3] + (-1,)).max(-1)
            eligible = gt_vis & (peak > 0)
            gt_view = np.where(eligible, peak, -np.inf).argmax(-1)
            gt_class = np.where(eligible.any(-1), gt_view + 1, 0).astype(np.int64)
            gt_peaks = _view_peaks(gt_hm)
            if keep_arrays:
                raw["gt_heatmap"] = gt_hm
        if keep_arrays:
            raw.update(gt_visibility=z["gt_visibility"][q].astype(np.float32), history_mask=hist_mask,
                       gt_view_class=gt_class, gt_view_peak_yx=gt_peaks)
            if "gt_heatmap" not in raw:
                raw["gt_heatmap"] = z["gt_heatmap"][q].astype(np.float32)
        gt_rel = z["gt_rel_poses"][q].astype(np.float64)
        cur_c2w = z["current_c2w"][q].astype(np.float64)
        hist_c2w = z["history_c2w"][q].astype(np.float64)
        hist_ids = z["history_frame_ids"][q].astype(np.int64)

        arm_data = {}
        for arm in ARMS:
            if arm not in arms:
                continue
            logits = z[f"pred_{arm}_visibility_logits"][q].astype(np.float32)
            none_p = z[f"pred_{arm}_none_probability"][q].astype(np.float64)
            if exact:
                view_peaks = z[f"pred_{arm}_view_peak_yx"][q].astype(np.int64)
                ga = z[f"pred_{arm}_gated_argmax"][q].astype(np.int64)
                gview, grow, gcol = ga[..., 0], ga[..., 1], ga[..., 2]
            else:
                gated = z[f"pred_{arm}_heatmaps_gated"][q].astype(np.float32)
                view_peaks = _view_peaks(gated)
                gview, grow, gcol = geo.argmax_view_pixel(gated)
                if keep_arrays:
                    raw[f"pred_{arm}_heatmaps_gated"] = gated
            if keep_arrays:
                raw[f"pred_{arm}_visibility_logits"] = logits
                raw[f"pred_{arm}_view_peak_yx"] = view_peaks
                if f"pred_{arm}_heatmaps_gated" not in raw:
                    raw[f"pred_{arm}_heatmaps_gated"] = z[f"pred_{arm}_heatmaps_gated"][q].astype(np.float32)
            available = vo_avail_all[q] if arm == "vo" else np.ones(Q, bool)
            arm_data[arm] = (logits, none_p, view_peaks, gview, grow, gcol, available)

    # ---- per slot (valid history slots of query rows) -------------------------
    qi, k = np.nonzero(hist_mask)
    visible = gt_class[qi, k] > 0
    cls = gt_class[qi, k]
    tv = np.clip(cls - 1, 0, 3)
    gt_yx = _take_view(gt_peaks[qi, k], tv)  # [S, 2]
    gt_bearing = np.asarray(geo.bearing_from_rel_pose(gt_rel[qi, k, 0], gt_rel[qi, k, 1]), dtype=np.float64)
    cur_pos = geo.c2w_position(cur_c2w)
    hist_pos = geo.c2w_position(hist_c2w)
    delta = hist_pos[qi, k] - cur_pos[qi]
    label_bearing = np.asarray(geo.pixel_to_bearing_elev(tv, gt_yx[:, 1], gt_yx[:, 0])[0], dtype=np.float64)
    positions = geo.c2w_position(clip_c2w)
    plen = geo.path_length(positions)
    slots = {
        "row": q[qi], "query_idx": qi, "t": t_all[q][qi], "k": k, "hist_frame": hist_ids[qi, k],
        "age": t_all[q][qi] - hist_ids[qi, k], "is_final_row": t_all[q][qi] == frame_count - 1,
        "vo_row": vo_avail_all[q][qi],
        "gt_vis_front": gt_vis[qi, k, 0], "gt_vis_right": gt_vis[qi, k, 1],
        "gt_vis_back": gt_vis[qi, k, 2], "gt_vis_left": gt_vis[qi, k, 3],
        "gt_visible": visible, "gt_class": cls.astype(np.int8),
        "gt_bearing": gt_bearing, "gt_forward_m": gt_rel[qi, k, 0], "gt_left_m": gt_rel[qi, k, 1],
        "gt_dist_m": np.linalg.norm(delta, axis=-1), "gt_dy_m": delta[:, 1],
        "multi_floor": np.abs(delta[:, 1]) > 1.0,
        "gt_peak_row": np.where(visible, gt_yx[:, 0], -1), "gt_peak_col": np.where(visible, gt_yx[:, 1], -1),
        "gt_label_bearing": np.where(visible, label_bearing, np.nan),
        "path_length_m": np.full(len(qi), plen),
    }
    for arm, (logits, none_p, view_peaks, gview, grow, gcol, available) in arm_data.items():
        pred_cls = np.concatenate([np.zeros_like(logits[..., :1]), logits], axis=-1).argmax(-1)[qi, k]
        p_yx = _take_view(view_peaks[qi, k], tv)
        sq = ((p_yx - gt_yx) ** 2).sum(-1)
        ok = available[qi]
        pb = np.asarray(geo.pixel_to_bearing_elev(np.maximum(gview[qi, k], 0), gcol[qi, k], grow[qi, k])[0])
        pb = np.where(ok & (gview[qi, k] >= 0), pb, np.nan)
        slots.update({
            f"pred_{arm}_available": ok,
            f"pred_{arm}_view5": np.where(ok, pred_cls, -1).astype(np.int8),
            f"pred_{arm}_sq_err": np.where(visible & ok, sq, -1),
            f"pred_{arm}_px_err": np.where(visible & ok, np.sqrt(sq), np.nan),
            f"pred_{arm}_joint4": visible & ok & (pred_cls == cls) & (sq <= 16),
            f"pred_{arm}_joint8": visible & ok & (pred_cls == cls) & (sq <= 64),
            f"pred_{arm}_argmax_view": np.where(ok, gview[qi, k], -1).astype(np.int8),
            f"pred_{arm}_argmax_row": np.where(ok, grow[qi, k], -1).astype(np.int16),
            f"pred_{arm}_argmax_col": np.where(ok, gcol[qi, k], -1).astype(np.int16),
            f"pred_{arm}_bearing": pb,
            f"pred_{arm}_bearing_err": np.where(visible, np.asarray(geo.circular_abs_diff(pb, gt_bearing)), np.nan),
            f"pred_{arm}_none_prob": np.where(ok, none_p[qi, k], np.nan),
        })
    floor_sq = (gt_yx[:, 0] - geo.FLOOR_PEAK_YX[0]) ** 2 + (gt_yx[:, 1] - geo.FLOOR_PEAK_YX[1]) ** 2
    slots.update({
        "pred_floor_available": np.ones(len(qi), bool),
        "pred_floor_view5": np.full(len(qi), FLOOR_CLASS, np.int8),
        "pred_floor_sq_err": np.where(visible, floor_sq, -1),
        "pred_floor_px_err": np.where(visible, np.sqrt(floor_sq), np.nan),
        "pred_floor_joint4": visible & (cls == FLOOR_CLASS) & (floor_sq <= 16),
        "pred_floor_joint8": visible & (cls == FLOOR_CLASS) & (floor_sq <= 64),
        "pred_floor_bearing_err": np.where(visible, np.asarray(geo.circular_abs_diff(180.0, gt_bearing)), np.nan),
        "pred_floor_none_prob": np.zeros(len(qi)),
    })

    # ---- per query row -----------------------------------------------------------
    start = positions[0]
    spans, n_vis_row = np.zeros(Q), np.zeros(Q, np.int64)
    for r in range(Q):
        sel = (qi == r) & visible
        n_vis_row[r] = int(sel.sum())
        spans[r] = geo.circular_range_deg(gt_bearing[sel])
    rows = {
        "row": q, "query_idx": np.arange(Q), "t": t_all[q], "is_final": t_all[q] == frame_count - 1,
        "vo_row": vo_avail_all[q], "cur_x": cur_pos[:, 0], "cur_y": cur_pos[:, 1], "cur_z": cur_pos[:, 2],
        "yaw_deg": np.asarray(geo.c2w_yaw_deg(cur_c2w), dtype=np.float64).reshape(Q),
        "dist_from_start_m": np.linalg.norm(cur_pos - start, axis=-1),
        "n_valid": hist_mask.sum(-1), "n_visible": n_vis_row, "gt_bearing_span_deg": spans,
    }
    pattern = resolve_pattern(meta, episode_id, clip) if tier == "E" else ""
    # E out-and-back: turnaround frame from the route itself (select_cases.py); -1 otherwise
    turn = (geo.out_and_back_turnaround(positions, meta.get("reference_path")) if pattern == "out_and_back"
            else {"frame": -1, "miss_m": np.nan, "waypoints_missed": 0, "note": ""})
    episode = {
        "tier": tier, "scene": scene, "clip": clip, "clip_key": f"{scene}/{clip}", "episode_id": episode_id,
        "npz_path": str(path), "schema": schema, "peaks_source": "exact_fields" if exact else "f16_arrays",
        "arms": ",".join(arms), "n_frames": frame_count, "n_rows_npz": len(t_all), "n_query_rows": Q,
        "n_query_rows_vo": int(vo_avail_all[q].sum()), "n_nonquery_rows": len(t_all) - Q,
        "query_source": "cache_endpoints" if endpoints.size else "rule",
        "path_length_m": plen, "y_range_m": float(np.ptp(positions[:, 1])) if len(positions) else 0.0,
        "start_x": float(start[0]), "start_y": float(start[1]), "start_z": float(start[2]),
        "pattern": pattern, "turnaround_frame": turn["frame"], "turnaround_miss_m": turn["miss_m"],
        "turnaround_waypoints_missed": turn["waypoints_missed"], "turnaround_note": turn["note"],
    }
    return {"slots": slots, "rows": rows, "episode": episode, "raw": raw if keep_arrays else None}


def _load_one(args):
    return load_clip(*args)


def _frame(records: list, key: str, keys: dict) -> pd.DataFrame:
    parts = []
    for rec in records:
        df = pd.DataFrame(rec[key])
        for col, val in keys(rec).items():
            df.insert(0, col, val)
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _ep_keys(rec: dict) -> dict:
    e = rec["episode"]
    # inserted at column 0 in reverse: final order tier, scene, clip, clip_key, episode_id
    return {"episode_id": e["episode_id"], "clip_key": e["clip_key"], "clip": e["clip"], "scene": e["scene"],
            "tier": e["tier"]}


def discover_npz(tier_dir: Path) -> list:
    found = []
    for dirpath, dirnames, filenames in os.walk(tier_dir, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if d != "manifests")
        found.extend(str(Path(dirpath) / f) for f in sorted(filenames) if f.endswith(".npz"))
    return found


def load_tier(tier: str, dumps_root: Path, workers: int, clip_list_mode: str, log) -> dict:
    tier_dir = dumps_root / tier
    info = {"tier": tier, "dump_dir": str(tier_dir), "present": tier_dir.is_dir(), "warnings": []}
    if not info["present"]:
        return info
    paths = discover_npz(tier_dir)
    info["npz_found"] = len(paths)
    if not paths:
        info["present"] = False
        return info
    t0 = time.time()
    jobs = [(p, tier) for p in paths]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(_load_one, jobs, chunksize=4))
    else:
        records = [_load_one(j) for j in jobs]
    log(f"[{tier}] loaded {len(records)} npz in {time.time() - t0:.1f}s")
    keys = [r["episode"]["clip_key"] for r in records]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise RuntimeError(f"[{tier}] duplicate clips in dumps: {dup[:5]}")
    # Pre-registered clip list.
    list_path = common.clip_list_path(tier)
    if clip_list_mode != "ignore" and list_path.is_file():
        wanted = set(common.read_clip_list(tier))
        missing = sorted(wanted - set(keys))
        extra = sorted(set(keys) - wanted)
        records = [r for r in records if r["episode"]["clip_key"] in wanted]
        info["clip_list"] = {"path": str(list_path), "listed": len(wanted), "dumped": len(set(keys) & wanted),
                             "missing": missing, "extra_ignored": extra}
        if missing:
            info["warnings"].append(f"{len(missing)} listed clips have no dump; metrics cover the rest")
    else:
        if clip_list_mode == "require":
            raise FileNotFoundError(f"[{tier}] no clip list at {list_path}")
        info["clip_list"] = {"path": str(list_path), "listed": None, "note": "no clip list; every dumped clip counts"}
        info["warnings"].append("no pre-registered clip list found; all dumped clips were scored")
    episodes = pd.DataFrame([r["episode"] for r in records])
    if tier == "E":
        unresolved = int((episodes["pattern"] == "").sum())
        info["pattern_counts"] = {str(k): int(v) for k, v in episodes["pattern"].value_counts().items()}
        if unresolved:
            info["warnings"].append(f"{unresolved} E episodes have no recognisable route pattern (see resolve_pattern)")
        no_turn = int(((episodes["pattern"] == "out_and_back") & (episodes["turnaround_frame"] < 0)).sum())
        if no_turn:
            info["warnings"].append(f"{no_turn} out-and-back episodes have no palindromic reference_path in meta_json; "
                                    "select_cases falls back to the farthest-from-start row for them")
        n_miss = int((episodes["turnaround_waypoints_missed"] > 0).sum())
        if n_miss:
            info["warnings"].append(f"{n_miss} out-and-back episodes never came within 0.5 m of some route point "
                                    "before the turnaround (turnaround frame = closest approach; check the render)")
    slots = _frame(records, "slots", _ep_keys)
    rows = _frame(records, "rows", _ep_keys)
    has_vo = bool((episodes["n_query_rows_vo"] > 0).any())
    info.update(has_vo=has_vo, schemas=sorted(set(episodes["schema"])), peaks_source=sorted(set(episodes["peaks_source"])),
                query_source=sorted(set(episodes["query_source"])),
                nonquery_rows_dropped=int(episodes["n_nonquery_rows"].sum()))
    if has_vo:
        n_drop = int((episodes["n_query_rows"] - episodes["n_query_rows_vo"]).sum())
        info["query_rows_without_vo_dropped"] = n_drop
        if n_drop:
            info["warnings"].append(f"{n_drop} endpoint rows without VO poses dropped from both arms")
        slots = slots[slots["vo_row"]].reset_index(drop=True)
        rows = rows[rows["vo_row"]].reset_index(drop=True)
    else:
        info["warnings"].append("no VO arm in this tier: GT arm only, H1/H2 on this tier cannot be judged")
    info["paths"] = [r["episode"]["npz_path"] for r in records]
    return {**info, "episodes": episodes, "slots": slots, "rows": rows}


# --------------------------------------------------------------------------- #
# Scene-cluster bootstrap
# --------------------------------------------------------------------------- #
class SceneBootstrap:
    """Scene-cluster resampling: rep r weights every slot of scene s by M[r, s]."""

    def __init__(self, scenes: np.ndarray, reps: int, rng: np.random.Generator, chunk: int = 200):
        self.names, self.codes = np.unique(np.asarray(scenes, dtype=str), return_inverse=True)
        S = len(self.names)
        draws = rng.integers(0, S, size=(reps, S)) if S else np.zeros((reps, 0), np.int64)
        self.M = np.zeros((reps, S), dtype=np.float64)
        np.add.at(self.M, (np.repeat(np.arange(reps), S), draws.ravel()), 1.0)
        self.S, self.reps, self.chunk = S, reps, chunk

    def _scene_sum(self, w: np.ndarray) -> np.ndarray:
        return np.bincount(self.codes, weights=np.asarray(w, dtype=np.float64), minlength=self.S)

    def ratio(self, num, den):
        n_s, d_s = self._scene_sum(num), self._scene_sum(den)
        point = n_s.sum() / d_s.sum() if d_s.sum() > 0 else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            reps = (self.M @ n_s) / (self.M @ d_s)
        return float(point), reps

    def quantiles(self, values, mask, qs):
        mask = np.asarray(mask, bool) & np.isfinite(values)
        x = np.asarray(values, dtype=np.float64)[mask]
        if x.size == 0:
            return [np.nan] * len(qs), np.full((len(qs), self.reps), np.nan)
        order = np.argsort(x, kind="stable")
        xs, cs = x[order], self.codes[mask][order]
        points = [float(np.percentile(x, 100 * q)) for q in qs]
        out = np.full((len(qs), self.reps), np.nan)
        for s0 in range(0, self.reps, self.chunk):
            cw = np.cumsum(self.M[s0:s0 + self.chunk][:, cs], axis=1)
            tot = cw[:, -1]
            for i, q in enumerate(qs):
                pos = q * (tot - 1)
                lo, hi = np.floor(pos), np.ceil(pos)
                ilo = (cw > lo[:, None]).argmax(1)
                ihi = (cw > hi[:, None]).argmax(1)
                val = xs[ilo] + (pos - lo) * (xs[ihi] - xs[ilo])
                out[i, s0:s0 + self.chunk] = np.where(tot > 0, val, np.nan)
        return points, out

    def auroc(self, score, label, mask):
        mask = np.asarray(mask, bool)
        sc, lab, codes = np.asarray(score, np.float64)[mask], np.asarray(label, bool)[mask], self.codes[mask]
        uniq, inv = np.unique(sc, return_inverse=True)
        P = np.zeros((self.S, len(uniq)))
        N = np.zeros((self.S, len(uniq)))
        np.add.at(P, (codes[lab], inv[lab]), 1.0)
        np.add.at(N, (codes[~lab], inv[~lab]), 1.0)

        def auc(Mc):
            p, n = Mc @ P, Mc @ N
            below = np.cumsum(n, axis=1) - n
            with np.errstate(invalid="ignore", divide="ignore"):
                return (p * (below + 0.5 * n)).sum(1) / (p.sum(1) * n.sum(1))

        point = float(auc(np.ones((1, self.S)))[0])
        reps = np.concatenate([auc(self.M[s0:s0 + self.chunk]) for s0 in range(0, self.reps, self.chunk)])
        return point, reps

    def f1(self, pred, label, mask):
        mask = np.asarray(mask, bool)
        pred, label = np.asarray(pred, bool) & mask, np.asarray(label, bool) & mask
        tp, fp, fn = (self._scene_sum(a) for a in (pred & label, pred & ~label, ~pred & label))
        point = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()) if (tp.sum() + fp.sum() + fn.sum()) else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            reps = 2 * (self.M @ tp) / (2 * (self.M @ tp) + self.M @ fp + self.M @ fn)
        return float(point), reps


def ci(reps) -> list:
    reps = np.asarray(reps, dtype=np.float64)
    if not np.isfinite(reps).any():
        return [None, None]
    lo, hi = np.nanpercentile(reps, CI_PCT)
    return [float(lo), float(hi)]


def stat(point, reps) -> dict:
    point = None if point is None or not np.isfinite(point) else float(point)
    return {"value": point, "ci95": ci(reps), "nan_reps": int(np.sum(~np.isfinite(reps)))}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def hist_quantile(sq_errors: np.ndarray, q: float) -> float:
    """validate.py _histogram_quantile on integer squared errors, same arithmetic."""
    s = np.sort(np.asarray(sq_errors, dtype=np.int64))
    if s.size == 0:
        return 0.0
    position = (s.size - 1) * q
    lower_rank, upper_rank = math.floor(position), math.ceil(position)
    lower, upper = math.sqrt(int(s[lower_rank])), math.sqrt(int(s[upper_rank]))
    return lower + (position - lower_rank) * (upper - lower)


def validator_metrics(df: pd.DataFrame, arm: str) -> dict:
    """Exactly the dict _HeatmapJointMetricAccumulator.compute() returns, from slot records."""
    valid = len(df)
    vis = df["gt_visible"].to_numpy()
    cls = df["gt_class"].to_numpy().astype(np.int64)
    visible = int(vis.sum())
    view5 = int((df[f"pred_{arm}_view5"].to_numpy().astype(np.int64) == cls).sum())
    j4, j8 = df[f"pred_{arm}_joint4"].to_numpy(), df[f"pred_{arm}_joint8"].to_numpy()
    sq = df[f"pred_{arm}_sq_err"].to_numpy()[vis]
    m = {
        "val_heatmap_joint_pck4": int(j4.sum()) / visible if visible > 0 else 0.0,
        "val_heatmap_joint_pck8": int(j8.sum()) / visible if visible > 0 else 0.0,
        "val_heatmap_pixel_error_median": hist_quantile(sq, 0.5),
        "val_heatmap_pixel_error_p90": hist_quantile(sq, 0.9),
        "val_heatmap_view5_accuracy": view5 / valid if valid > 0 else 0.0,
        "val_heatmap_valid_count": float(valid),
        "val_heatmap_visible_count": float(visible),
        "val_heatmap_none_count": float(valid - visible),
    }
    sup4, sup8 = [], []
    for v, name in enumerate(geo.VIEW_NAMES):
        rows = vis & (cls == v + 1)
        count = int(rows.sum())
        p4 = int((j4 & rows).sum()) / count if count > 0 else 0.0
        p8 = int((j8 & rows).sum()) / count if count > 0 else 0.0
        m[f"val_heatmap_{name}_pck4"], m[f"val_heatmap_{name}_pck8"] = p4, p8
        m[f"val_heatmap_{name}_count"] = float(count)
        if count > 0:
            sup4.append(p4)
            sup8.append(p8)
    m["val_heatmap_macro_joint_pck4"] = sum(sup4) / len(sup4) if sup4 else 0.0
    m["val_heatmap_macro_joint_pck8"] = sum(sup8) / len(sup8) if sup8 else 0.0
    m["val_heatmap_supported_direction_count"] = float(len(sup8))
    return m


def arm_block(df: pd.DataFrame, arm: str, boot: SceneBootstrap):
    """Headline metrics of one arm on one tier: (json block, bootstrap reps by metric)."""
    vis = df["gt_visible"].to_numpy()
    ones = np.ones(len(df), bool)
    reps = {}
    out = {"n_valid_slots": int(len(df)), "n_visible_slots": int(vis.sum())}
    for key, col in (("joint_pck8", "joint8"), ("joint_pck4", "joint4")):
        p, r = boot.ratio(df[f"pred_{arm}_{col}"].to_numpy() & vis, vis)
        out[key], reps[key] = stat(p, r), r
    p, r = boot.ratio(df[f"pred_{arm}_view5"].to_numpy().astype(np.int64) == df["gt_class"].to_numpy(), ones)
    out["view5_accuracy"], reps["view5_accuracy"] = stat(p, r), r
    berr = df[f"pred_{arm}_bearing_err"].to_numpy()
    (med, p90), (rmed, rp90) = boot.quantiles(berr, vis, (0.5, 0.9))
    out["bearing_err_median_deg"], reps["bearing_err_median_deg"] = stat(med, rmed), rmed
    out["bearing_err_p90_deg"], reps["bearing_err_p90_deg"] = stat(p90, rp90), rp90
    p, r = boot.ratio(vis & (np.nan_to_num(berr, nan=np.inf) <= 15.0), vis)
    out["bearing_err_le15_share"], reps["bearing_err_le15_share"] = stat(p, r), r
    score = 1.0 - df[f"pred_{arm}_none_prob"].to_numpy()
    p, r = boot.auroc(score, vis, ones)
    out["visibility_auroc"], reps["visibility_auroc"] = stat(p, r), r
    pred = score >= 0.5
    p, r = boot.f1(pred, vis, ones)
    out["visibility_f1"], reps["visibility_f1"] = stat(p, r), r
    tp = int((pred & vis).sum())
    out["visibility_precision"] = tp / int(pred.sum()) if pred.sum() else None
    out["visibility_recall"] = tp / int(vis.sum()) if vis.sum() else None
    out["validator"] = validator_metrics(df, arm)
    return out, reps


def strata_block(df: pd.DataFrame, tier: str, path_cuts, arms) -> dict:
    vis = df["gt_visible"].to_numpy()
    fam = {}

    def cells(labels: np.ndarray, order):
        out = {}
        for name in order:
            sel = vis & (labels == name)
            n = int(sel.sum())
            cell = {"n": n, "conclusion_allowed": n >= N_MIN}
            for arm in arms:
                be = df[f"pred_{arm}_bearing_err"].to_numpy()[sel]
                cell[arm] = {
                    "joint_pck8": float(df[f"pred_{arm}_joint8"].to_numpy()[sel].mean()) if n else None,
                    "joint_pck4": float(df[f"pred_{arm}_joint4"].to_numpy()[sel].mean()) if n else None,
                    "bearing_err_median_deg": float(np.median(be)) if n else None,
                    "bearing_err_le15_share": float((be <= 15).mean()) if n else None,
                }
            out[str(name)] = cell
        return out

    view = np.asarray(geo.VIEW_CLASS_NAMES, dtype=object)[df["gt_class"].to_numpy().astype(np.int64)]
    fam["gt_view"] = cells(view, geo.VIEW_NAMES)
    dist = df["gt_dist_m"].to_numpy()
    dlab = np.full(len(df), "", dtype=object)
    for lo, hi, name in DIST_BINS:
        dlab[(dist >= lo) & (dist < hi)] = name
    fam["distance"] = cells(dlab, [b[2] for b in DIST_BINS])
    age = df["age"].to_numpy()
    alab = np.full(len(df), "", dtype=object)
    for lo, hi, name in AGE_BINS:
        alab[(age >= lo) & (age <= hi)] = name
    fam["age"] = cells(alab, [b[2] for b in AGE_BINS])
    pl = df["path_length_m"].to_numpy()
    plab = np.where(pl <= path_cuts[0], "T1", np.where(pl <= path_cuts[1], "T2", "T3")).astype(object)
    fam["path_length_tertile"] = cells(plab, ["T1", "T2", "T3"])
    fam["path_length_tertile"]["cutpoints_m"] = [float(c) for c in path_cuts]
    fam["multi_floor"] = cells(np.where(df["multi_floor"].to_numpy(), "dy>1m", "same_floor").astype(object),
                               ["same_floor", "dy>1m"])
    if tier == "E":
        fam["pattern"] = cells(df["pattern"].to_numpy().astype(object), E_PATTERNS)
    fam["_bins"] = {"distance_m": [[b[0], None if math.isinf(b[1]) else b[1]] for b in DIST_BINS],
                    "age_frames": [[b[0], None if b[1] >= 10 ** 9 else b[1]] for b in AGE_BINS],
                    "path_length": "tertiles of per-episode path length within the tier",
                    "multi_floor": "|y_history - y_current| > 1 m (slot level)",
                    "denominator": "GT-visible slots; n < 100 -> report only"}
    return fam


def consistency_block(df: pd.DataFrame) -> dict:
    vis = df["gt_visible"].to_numpy()
    err = np.asarray(geo.circular_abs_diff(df["gt_label_bearing"].to_numpy()[vis], df["gt_bearing"].to_numpy()[vis]))
    sector = geo.view_for_bearing(df["gt_bearing"].to_numpy()[vis])
    agree = (df["gt_class"].to_numpy()[vis] - 1) == sector
    return {"label_peak_vs_pose_bearing_deg": {"max": float(err.max()) if err.size else None,
                                               "p99": float(np.percentile(err, 99)) if err.size else None,
                                               "share_le_2deg": float((err <= 2).mean()) if err.size else None},
            "gt_view_class_equals_bearing_sector_share": float(agree.mean()) if agree.size else None}


def episode_block(tier_data: dict) -> pd.DataFrame:
    eps = tier_data["episodes"].copy()
    slots, rows = tier_data["slots"], tier_data["rows"]
    vis = slots[slots["gt_visible"]]
    g = vis.groupby("clip_key") if len(vis) else None
    eps["n_scored_rows"] = eps["clip_key"].map(rows.groupby("clip_key").size()).fillna(0).astype(int)
    eps["n_visible_slots"] = eps["clip_key"].map(g.size() if g is not None else {}).fillna(0).astype(int)
    for arm in SCORED:
        if f"pred_{arm}_joint8" not in slots:
            continue
        for key, col, fn in (("pck8", "joint8", "mean"), ("pck4", "joint4", "mean"),
                             ("bearing_err_median", "bearing_err", "median")):
            eps[f"{arm}_{key}"] = eps["clip_key"].map(g[f"pred_{arm}_{col}"].agg(fn) if g is not None else {})
    eps["max_row_span_deg"] = eps["clip_key"].map(rows.groupby("clip_key")["gt_bearing_span_deg"].max())
    # per scored row: PCK@8 of its GT-visible slots (NaN if none) for the case figures
    for arm in SCORED:
        if f"pred_{arm}_joint8" in slots and g is not None:
            per_row = vis.groupby(["clip_key", "row"])[f"pred_{arm}_joint8"].mean().rename(f"{arm}_pck8_row")
            rows = rows.merge(per_row.reset_index(), on=["clip_key", "row"], how="left")
    tier_data["rows"] = rows
    eps["order_key"] = [common.sha1_key(f"{s}:{e}") if tier_data["tier"] in "CD" else common.sha1_key(f"{s}/{c}")
                        for s, c, e in zip(eps["scene"], eps["clip"], eps["episode_id"])]
    eps["order_key_rule"] = "sha1(scene:episode_id)" if tier_data["tier"] in "CD" else "sha1(scene/clip)"
    return eps


def verdict(name: str, reasons: dict) -> dict:
    return {"verdict": name, "label_zh": VERDICT_ZH[name], **reasons}


def h1_verdict(arm_blk: dict, gain: dict) -> dict:
    p = arm_blk["joint_pck8"]["value"]
    med = arm_blk["bearing_err_median_deg"]["value"]
    g, g_lo = gain["value"], gain["ci95"][0]
    if p is None or g is None or med is None or g_lo is None:
        return verdict("missing", {"reason": "metric undefined"})
    reasons = {"joint_pck8": p, "bearing_err_median_deg": med, "gain_over_floor": g, "gain_ci95_low": g_lo}
    if p >= H1["pck8_min"] and med <= H1["bearing_median_max"] and g >= H1["gain_min"] and g_lo >= H1["gain_ci_low_min"]:
        return verdict("support", reasons)
    if g < H1["refute_gain_below"] or p < H1["refute_pck8_below"]:
        return verdict("refute", reasons)
    return verdict("partial", reasons)


def h2_verdict(tier: str, delta: dict) -> dict:
    d, lo = delta["value"], delta["ci95"][0]
    if d is None or lo is None:
        return verdict("missing", {"reason": "delta undefined"})
    th = H2[tier]
    reasons = {"delta_pck8": d, "delta_ci95": delta["ci95"]}
    if d >= th["delta_min"] and lo >= th["ci_low_min"]:
        return verdict("support", reasons)
    if d <= th["refute_delta_max"]:
        return verdict("refute", reasons)
    return verdict("not_measured", reasons)


def h2_attribution(vo_verdict: str, gt_verdict: str) -> dict:
    """Ledger H2 归因规则 through the full decision table H2_ATTRIBUTION (see there)."""
    key = (vo_verdict, gt_verdict)
    literal = None
    if vo_verdict not in ("support", "missing"):
        literal = "泛化瓶颈在里程计，不在热力头" if gt_verdict == "support" else "头本身不泛化"
    if key in H2_ATTRIBUTION:
        return {"attribution": H2_ATTRIBUTION[key], "attribution_note": None, "attribution_literal_reading": literal}
    note = H2_NO_ATTRIBUTION_VO.get(vo_verdict) or H2_NO_ATTRIBUTION_GT.get(gt_verdict, "不归因")
    return {"attribution": None, "attribution_note": note, "attribution_literal_reading": literal}


def h3_verdict(n: int, pck8) -> dict:
    reasons = {"n_front_visible_slots": n, "joint_pck8": pck8}
    if n < N_MIN:
        return verdict("not_measured", dict(reasons, reason=f"n < {N_MIN}"))
    if pck8 >= H3["support_min"]:
        return verdict("support", reasons)
    if pck8 < H3["refute_below"]:
        return verdict("refute", reasons)
    return verdict("partial", reasons)


def compute_all(tiers: dict, reps_n: int, seed: int, log) -> dict:
    result = {"tiers": {}, "tier_differences": {}, "verdicts": {}}
    reps_store = {}
    for tier in TIER_ORDER:
        td = tiers.get(tier)
        if td is None or not td.get("present") or "slots" not in td or not len(td["slots"]):
            result["tiers"][tier] = {"present": False, "label": common.TIERS[tier]["label"],
                                     **({k: v for k, v in (td or {}).items() if k in ("dump_dir", "warnings")})}
            continue
        t0 = time.time()
        df = td["slots"]
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(TIER_ORDER.index(tier),)))
        boot = SceneBootstrap(df["scene"].to_numpy(), reps_n, rng)
        block = {"present": True, "label": common.TIERS[tier]["label"], "n_clips": int(len(td["episodes"])),
                 "n_scenes": int(boot.S), "scenes": [str(s) for s in boot.names], "n_rows": int(len(td["rows"])),
                 "n_valid_slots": int(len(df)), "n_visible_slots": int(df["gt_visible"].sum()),
                 **{k: td.get(k) for k in ("dump_dir", "npz_found", "clip_list", "has_vo", "schemas",
                                           "peaks_source", "query_source", "nonquery_rows_dropped",
                                           "query_rows_without_vo_dropped", "pattern_counts", "warnings")},
                 "arms": {}, "gain_over_floor": {}}
        reps_store[tier] = {}
        for arm in SCORED:
            if arm == "vo" and not td["has_vo"]:
                continue
            blk, reps = arm_block(df, arm, boot)
            block["arms"][arm], reps_store[tier][arm] = blk, reps
        for arm in ARMS:
            if arm in reps_store[tier]:
                point = block["arms"][arm]["joint_pck8"]["value"] - block["arms"]["floor"]["joint_pck8"]["value"]
                block["gain_over_floor"][arm] = stat(point, reps_store[tier][arm]["joint_pck8"]
                                                     - reps_store[tier]["floor"]["joint_pck8"])
        eps = td["episodes_table"]
        cuts = np.quantile(eps["path_length_m"].to_numpy(), [1 / 3, 2 / 3]) if len(eps) else [np.nan, np.nan]
        block["strata"] = strata_block(df, tier, cuts, list(block["arms"]))
        block["consistency"] = consistency_block(df)
        result["tiers"][tier] = block
        log(f"[{tier}] metrics + {reps_n} bootstrap reps over {boot.S} scenes in {time.time() - t0:.1f}s")

    # Tier differences vs B (independent resamples per tier).
    diff_metrics = ("joint_pck8", "joint_pck4", "bearing_err_median_deg", "bearing_err_le15_share")
    for tier in TIER_ORDER:
        if tier == "B" or tier not in reps_store or "B" not in reps_store:
            continue
        entry = {}
        for arm in SCORED:
            if arm in reps_store[tier] and arm in reps_store["B"]:
                entry[arm] = {m: stat(result["tiers"][tier]["arms"][arm][m]["value"]
                                      - result["tiers"]["B"]["arms"][arm][m]["value"],
                                      reps_store[tier][arm][m] - reps_store["B"][arm][m]) for m in diff_metrics}
        result["tier_differences"][f"{tier}-B"] = entry

    # H1: every tier A-E, VO arm (GT arm reported alongside, not judged).
    h1 = {}
    for tier in TIER_ORDER:
        blk = result["tiers"][tier]
        if not blk.get("present") or "vo" not in blk.get("arms", {}):
            h1[tier] = verdict("missing", {"reason": "tier not dumped" if not blk.get("present") else "no VO arm"})
            continue
        h1[tier] = h1_verdict(blk["arms"]["vo"], blk["gain_over_floor"]["vo"])
        h1[tier]["gt_arm_same_rule"] = h1_verdict(blk["arms"]["gt"], blk["gain_over_floor"]["gt"])["verdict"]
    result["verdicts"]["H1"] = h1

    # H2: C vs B and D vs B on the VO arm; attribution from the GT arm on the same line.
    h2 = {}
    for tier in ("C", "D"):
        diff = result["tier_differences"].get(f"{tier}-B", {})
        if "vo" not in diff:
            h2[tier] = verdict("missing", {"reason": f"{tier} or B not dumped with a VO arm"})
            continue
        vo = h2_verdict(tier, diff["vo"]["joint_pck8"])
        gt = h2_verdict(tier, diff["gt"]["joint_pck8"]) if "gt" in diff else verdict("missing", {})
        vo["gt_arm_same_rule"] = gt["verdict"]
        vo.update(h2_attribution(vo["verdict"], gt["verdict"]))
        vo["attribution_rule"] = ("written only when both arm verdicts assert it: VO refute + GT support -> 瓶颈在里程计; "
                                  "VO refute + GT refute -> 头本身不泛化; otherwise None with attribution_note. "
                                  "attribution_literal_reading = ledger text with 不过 = below the support line "
                                  "(not adopted, kept for the record)")
        h2[tier] = vo
    result["verdicts"]["H2"] = h2

    # H3: tier E, GT-view front visible slots, VO and GT arms judged separately.
    h3 = {}
    blk = result["tiers"]["E"]
    if blk.get("present"):
        df = tiers["E"]["slots"]
        front = df[df["gt_visible"] & (df["gt_class"] == 1 + geo.FRONT)]
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(TIER_ORDER.index("E"),)))
        boot = SceneBootstrap(front["scene"].to_numpy(), reps_n, rng)
        for arm in ARMS:
            if arm == "vo" and not tiers["E"]["has_vo"]:
                h3[arm] = verdict("missing", {"reason": "no VO arm"})
                continue
            n = int(len(front))
            p, r = boot.ratio(front[f"pred_{arm}_joint8"].to_numpy(), np.ones(n, bool)) if n else (np.nan, np.full(reps_n, np.nan))
            h3[arm] = h3_verdict(n, p if n else None)
            h3[arm]["joint_pck8_ci95"] = ci(r)
            h3[arm]["note"] = "CI: same scene-cluster bootstrap, descriptive only"
    else:
        h3 = {arm: verdict("missing", {"reason": "tier E not dumped"}) for arm in ARMS}
    result["verdicts"]["H3"] = h3
    return result


# --------------------------------------------------------------------------- #
# Self-check against validate.py
# --------------------------------------------------------------------------- #
def self_check(tiers: dict, log) -> dict:
    import torch

    from scripts.training.validate import _HeatmapJointMetricAccumulator as Acc

    def onehot(peaks_yx: np.ndarray, values: np.ndarray, hw=(64, 64)) -> torch.Tensor:
        """[..., 4, 2] peaks with per-view values -> [..., 4, H, W] maps whose argmax is the peak."""
        out = np.zeros(peaks_yx.shape[:-1] + hw, dtype=np.float32)
        y, x = np.clip(peaks_yx[..., 0], 0, hw[0] - 1), np.clip(peaks_yx[..., 1], 0, hw[1] - 1)
        idx = np.indices(peaks_yx.shape[:-1])
        out[tuple(idx) + (y, x)] = values
        return torch.from_numpy(out)

    report = {"ok": True, "checks": []}
    for tier, td in tiers.items():
        if not td.get("present") or "paths" not in td:
            continue
        exact_available = "exact_fields" in td.get("peaks_source", [])
        modes = ["auto", "arrays"] if exact_available else ["arrays"]
        for mode in modes:
            accs = {arm: Acc(heatmap_size=(64, 64), device=torch.device("cpu")) for arm in SCORED}
            recs = []
            for path in td["paths"]:
                rec = load_clip(path, tier, peaks_from=mode, keep_arrays=True)
                raw = rec["raw"]
                keep = rec["rows"]["vo_row"] if td["has_vo"] else np.ones(len(rec["rows"]["row"]), bool)
                rec["slots"] = {k: v[keep[rec["slots"]["query_idx"]]] for k, v in rec["slots"].items()}
                recs.append(rec)
                if not keep.any():
                    continue
                hm = torch.from_numpy(raw["history_mask"][keep])
                gt_vis = torch.from_numpy(raw["gt_visibility"][keep])
                if mode == "auto":  # exact argmax fields -> maps with the same argmaxes
                    cls = raw["gt_view_class"][keep]
                    vals = (raw["gt_visibility"][keep] > 0.5).astype(np.float32)
                    vals += (np.arange(4) == (cls[..., None] - 1)).astype(np.float32)
                    gt_maps = onehot(raw["gt_view_peak_yx"][keep], vals)
                else:
                    gt_maps = torch.from_numpy(raw["gt_heatmap"][keep])
                for arm in SCORED:
                    if arm == "floor":
                        n, k = raw["history_mask"][keep].shape
                        logits = torch.full((n, k, 4), -10.0)
                        logits[..., geo.BACK] = 10.0
                        peaks = np.broadcast_to(np.asarray(geo.FLOOR_PEAK_YX), (n, k, 4, 2))
                        maps = onehot(peaks, np.ones((n, k, 4), np.float32))
                    elif f"pred_{arm}_visibility_logits" in raw and not (arm == "vo" and not td["has_vo"]):
                        logits = torch.from_numpy(raw[f"pred_{arm}_visibility_logits"][keep])
                        maps = (onehot(raw[f"pred_{arm}_view_peak_yx"][keep], np.ones(raw[f"pred_{arm}_view_peak_yx"][keep].shape[:-1], np.float32))
                                if mode == "auto" else torch.from_numpy(raw[f"pred_{arm}_heatmaps_gated"][keep]))
                    else:
                        continue
                    accs[arm].update(pred_visibility_logits=logits, pred_heatmaps=maps, gt_visibility=gt_vis,
                                     gt_heatmaps=gt_maps, history_mask=hm)
            slots = _frame(recs, "slots", _ep_keys)
            for arm in SCORED:
                if arm == "vo" and not td["has_vo"]:
                    continue
                ref = accs[arm].compute()
                mine = validator_metrics(slots, arm)
                mism = {k: [mine[k], ref[k]] for k in ref if mine.get(k) != ref[k]}
                main = validator_metrics(td["slots"], arm) if mode == "auto" or not exact_available else None
                main_mism = {k: [main[k], ref[k]] for k in ref if main is not None and main.get(k) != ref[k]}
                entry = {"tier": tier, "arm": arm, "mode": "exact_fields" if mode == "auto" else "f16_arrays",
                         "validator_joint_pck8": ref["val_heatmap_joint_pck8"],
                         "validator_joint_pck4": ref["val_heatmap_joint_pck4"],
                         "numpy_joint_pck8": mine["val_heatmap_joint_pck8"],
                         "bit_exact": not mism, "mismatches": mism,
                         "reported_metrics_bit_exact": (not main_mism) if main is not None else None,
                         "reported_mismatches": main_mism}
                report["ok"] &= entry["bit_exact"] and entry["reported_metrics_bit_exact"] in (True, None)
                report["checks"].append(entry)
                log(f"[self-check] {tier}/{arm}/{entry['mode']}: validator pck8={ref['val_heatmap_joint_pck8']!r} "
                    f"numpy={mine['val_heatmap_joint_pck8']!r} bit_exact={entry['bit_exact']}")
    return report


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def write_table(df: pd.DataFrame, stem: Path) -> Path:
    try:
        path = stem.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    except Exception:  # noqa: BLE001 - no pyarrow: fall back to csv
        path = stem.with_suffix(".csv.gz")
        df.to_csv(path, index=False)
    return path


def read_table(stem: Path) -> pd.DataFrame:
    """Read a table written by write_table (parquet preferred)."""
    if stem.with_suffix(".parquet").is_file():
        return pd.read_parquet(stem.with_suffix(".parquet"))
    return pd.read_csv(stem.with_suffix(".csv.gz"))


def _fmt(x, pct=False, nd=1):
    if x is None:
        return "—"
    return f"{100 * x:.{nd}f}" if pct else f"{x:.{nd}f}"


def _fmt_stat(s, pct=True, nd=1):
    if not s or s.get("value") is None:
        return "—"
    lo, hi = s["ci95"]
    ci_txt = f" [{_fmt(lo, pct, nd)}, {_fmt(hi, pct, nd)}]" if lo is not None else ""
    return f"{_fmt(s['value'], pct, nd)}{ci_txt}"


def summary_md(res: dict) -> str:
    L = [f"# EXP-18 metrics summary", "", f"Generated {res['created_utc']} by `scripts/exp18/compute_metrics.py` "
         f"(bootstrap {res['bootstrap']['reps']} reps, root seed {res['bootstrap']['seed']}, scene clusters). "
         "Percent values are ×100; brackets are 95% CIs.", ""]
    L += ["## Headline (VO arm = criterion arm)", "",
          "| tier | clips | scenes | visible slots | VO PCK@8 | GT PCK@8 | floor PCK@8 | VO − floor (pt) | VO bearing med (°) | VO P90 (°) | VO ≤15° | VO vis AUROC | VO vis F1 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tier in TIER_ORDER:
        b = res["tiers"][tier]
        if not b.get("present"):
            L.append(f"| {tier} | — | — | — | not dumped | | | | | | | | |")
            continue
        a = b["arms"]
        vo = a.get("vo", {})
        L.append(" | ".join([
            f"| {tier}", str(b["n_clips"]), str(b["n_scenes"]), str(b["n_visible_slots"]),
            _fmt_stat(vo.get("joint_pck8")), _fmt_stat(a["gt"]["joint_pck8"]), _fmt_stat(a["floor"]["joint_pck8"]),
            _fmt_stat(b["gain_over_floor"].get("vo")), _fmt_stat(vo.get("bearing_err_median_deg"), pct=False, nd=2),
            _fmt_stat(vo.get("bearing_err_p90_deg"), pct=False, nd=1), _fmt_stat(vo.get("bearing_err_le15_share")),
            _fmt_stat(vo.get("visibility_auroc"), pct=False, nd=3), _fmt_stat(vo.get("visibility_f1"), pct=False, nd=3),
        ]) + " |")
    L += ["", "## Verdicts (ledger thresholds)", ""]
    for tier, v in res["verdicts"]["H1"].items():
        extra = f" (GT arm, same rule: {v.get('gt_arm_same_rule')})" if v.get("gt_arm_same_rule") else ""
        L.append(f"- **H1 {tier}**: {v['label_zh']} / {v['verdict']}{extra}")
    for tier, v in res["verdicts"]["H2"].items():
        L.append(f"- **H2 {tier} vs B**: {v['label_zh']} / {v['verdict']}"
                 + (f"; Δ PCK@8 = {_fmt(v.get('delta_pck8'), True)} pt, CI {[_fmt(x, True) for x in v.get('delta_ci95', [None, None])]}"
                    if v.get("delta_pck8") is not None else "")
                 + (f"; GT arm: {v.get('gt_arm_same_rule')}; 归因: {v.get('attribution') or v.get('attribution_note')}"
                    if "attribution" in v else ""))
    for arm, v in res["verdicts"]["H3"].items():
        L.append(f"- **H3 ({arm} arm)**: {v['label_zh']} / {v['verdict']}"
                 + (f"; n = {v.get('n_front_visible_slots')}, PCK@8 = {_fmt(v.get('joint_pck8'), True)}"
                    if v.get("n_front_visible_slots") is not None else ""))
    if res["tier_differences"]:
        L += ["", "## Tier differences vs B (independent scene resamples)", "",
              "| diff | arm | Δ PCK@8 (pt) | Δ PCK@4 (pt) | Δ bearing median (°) |", "|---|---|---|---|---|"]
        for name, entry in res["tier_differences"].items():
            for arm, m in entry.items():
                L.append(f"| {name} | {arm} | {_fmt_stat(m['joint_pck8'])} | {_fmt_stat(m['joint_pck4'])} | "
                         f"{_fmt_stat(m['bearing_err_median_deg'], pct=False, nd=2)} |")
    L += ["", "## Descriptive strata (GT-visible slots; *n < 100: number only, no conclusion*)", ""]
    for tier in TIER_ORDER:
        b = res["tiers"][tier]
        if not b.get("present"):
            continue
        L += [f"### Tier {tier}", "", "| family | cell | n | VO PCK@8 | GT PCK@8 | floor PCK@8 | VO bearing med (°) |",
              "|---|---|---|---|---|---|---|"]
        for fam, cells in b["strata"].items():
            if fam.startswith("_"):
                continue
            for cell, c in cells.items():
                if not isinstance(c, dict) or "n" not in c:
                    continue
                star = "" if c["conclusion_allowed"] else " *"
                L.append(f"| {fam} | {cell} | {c['n']}{star} | {_fmt(c.get('vo', {}).get('joint_pck8'), True)} | "
                         f"{_fmt(c.get('gt', {}).get('joint_pck8'), True)} | {_fmt(c.get('floor', {}).get('joint_pck8'), True)} | "
                         f"{_fmt(c.get('vo', {}).get('bearing_err_median_deg'), nd=2)} |")
        cons = b["consistency"]["label_peak_vs_pose_bearing_deg"]
        L += ["", f"Label/pose consistency: GT label peak vs GT pose bearing max {_fmt(cons['max'], nd=2)}°, "
              f"P99 {_fmt(cons['p99'], nd=2)}°; GT view class = bearing sector for "
              f"{_fmt(b['consistency']['gt_view_class_equals_bearing_sector_share'], True, 2)}% of visible slots.", ""]
        for w in b.get("warnings") or []:
            L.append(f"- warning: {w}")
        L.append("")
    if res.get("self_check"):
        sc = res["self_check"]
        L += ["## Self-check vs validate.py", "", f"overall bit-exact: **{sc['ok']}**", ""]
        for c in sc["checks"]:
            L.append(f"- {c['tier']}/{c['arm']}/{c['mode']}: validator joint PCK@8 {c['validator_joint_pck8']!r}, "
                     f"numpy {c['numpy_joint_pck8']!r}, bit-exact {c['bit_exact']}")
    return "\n".join(L) + "\n"


def parse_args(argv=None) -> argparse.Namespace:
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dumps-root", type=Path, default=Path(env("EXP18_DUMPS_ROOT", str(common.EXP_ROOT / "dumps"))))
    p.add_argument("--out-dir", type=Path, default=Path(env("EXP18_METRICS_DIR", str(common.EXP_ROOT / "metrics"))))
    p.add_argument("--tiers", default=env("EXP18_TIERS", TIER_ORDER), help="subset of ABCDE (default all)")
    p.add_argument("--bootstrap-reps", type=int, default=int(env("EXP18_BOOTSTRAP_REPS", "10000")))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=int(env("EXP18_METRICS_WORKERS", "8")))
    p.add_argument("--clip-list", choices=("auto", "require", "ignore"), default=env("EXP18_CLIP_LIST_MODE", "auto"),
                   help="auto: filter by common.clip_list_path(tier) when it exists")
    p.add_argument("--self-check", action="store_true", default=env("EXP18_SELF_CHECK", "0") == "1",
                   help="re-score with validate.py's _HeatmapJointMetricAccumulator (imports torch)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.time()

    def log(msg):
        print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tiers = {}
    for tier in args.tiers:
        if tier not in TIER_ORDER:
            raise SystemExit(f"unknown tier {tier!r}")
        td = load_tier(tier, args.dumps_root, args.workers, args.clip_list, log)
        if td.get("present"):
            td["episodes_table"] = episode_block(td)
            td["slots"] = td["slots"].merge(td["episodes"][["clip_key", "pattern"]], on="clip_key", how="left")
        tiers[tier] = td
        log(f"[{tier}] present={td.get('present')} npz={td.get('npz_found', 0)} "
            f"slots={len(td['slots']) if 'slots' in td else 0}")
    res = compute_all(tiers, args.bootstrap_reps, args.seed, log)
    res = {"schema": SCHEMA, "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
           "dumps_root": str(args.dumps_root), "out_dir": str(args.out_dir), "exp_root": str(common.EXP_ROOT),
           "bootstrap": {"reps": args.bootstrap_reps, "seed": args.seed, "cluster": "scene", "ci_percentiles": CI_PCT,
                         "streams": "np.random.default_rng(SeedSequence(seed, spawn_key=(index in 'ABCDE',)))",
                         "tier_difference": "independent per-tier resamples; arms and floor paired within a tier"},
           "definitions": {
               "population": "cache endpoint rows (t = 19, 27, ... + final) with VO poses; clip list if present",
               "joint_pck": "validate.py _HeatmapJointMetricAccumulator: pred class == GT class and argmax dist <= r "
                            "in the GT view; denominator GT-visible slots",
               "bearing": "GT atan2(left, forward) of GT rel pose; pred = argmax of heatmaps_gated over 4x64x64 "
                          "(scripts/exp18/geometry.py, label pixel convention index i <-> coordinate i)",
               "visibility": "score 1 - none_probability, label = GT visible in any view; F1 at score >= 0.5",
               "floor": "class back, peak (32, 32) in every view, never none; bearing err |180 - GT|",
               "thresholds": {"H1": H1, "H2": H2, "H3": H3, "n_min": N_MIN}},
           **res}
    if args.self_check:
        res["self_check"] = self_check(tiers, log)
    frames = {"slots": [], "rows": [], "episodes": []}
    for td in tiers.values():
        if td.get("present"):
            frames["slots"].append(td["slots"])
            frames["rows"].append(td["rows"])
            frames["episodes"].append(td["episodes_table"])
    res["tables"] = {}
    for name, parts in frames.items():
        if parts:
            res["tables"][name] = str(write_table(pd.concat(parts, ignore_index=True), args.out_dir / name))
    (args.out_dir / "metrics.json").write_text(json.dumps(_clean(res), indent=1, ensure_ascii=False,
                                                          default=_json_default) + "\n", encoding="utf-8")
    (args.out_dir / "summary.md").write_text(summary_md(res), encoding="utf-8")
    log(f"wrote {args.out_dir / 'metrics.json'}, summary.md, {', '.join(res['tables'].values())}")
    if args.self_check and not res["self_check"]["ok"]:
        log("SELF-CHECK FAILED: numpy metrics differ from validate.py")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

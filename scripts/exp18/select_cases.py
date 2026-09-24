#!/usr/bin/env python3
"""EXP-18: pre-registered case selection for the main figure, the gallery and the route-pattern figure.

Implements docs/experiments/README.md EXP-18 "案例挑选规则" on the per-episode
and per-row tables written by compute_metrics.py (episodes/rows .parquet),
so the choice is fixed before anyone looks at a picture:

* main figure (tier C): candidates = >= 4 scored endpoint rows, path length
  >= 8 m, and at least one row whose GT-visible history bearings span >= 90
  deg (circular range).  Ranked by |episode VO PCK@8 - median of the tier-C
  per-episode VO PCK@8| ascending, top 5 (typical, not best; all 5 go to the
  supplement).  Key rows: first endpoint row, the row with the largest span
  (ties: earliest), the final row.
* gallery (each tier A-E): the episodes at the 10th / 50th / 90th percentile
  of per-episode VO median bearing error: the episode whose value is closest
  to ``np.percentile`` (linear) of that tier's values.  The 90th is the
  failure case and must be shown.
* route-pattern figure (tier E): per pattern (out_and_back, loop) the episode
  whose VO PCK@8 is closest to that pattern's median.  Turn-around row:
  - out_and_back: the scored row nearest in time (ties: earlier) to the frame
    where the clip reaches the route's own turnaround, reference_path[(len-1)//2]
    (= E_routes.json turnaround_index; compute_metrics stores the frame as
    ``turnaround_frame`` via geometry.out_and_back_turnaround).  The point
    farthest from the start is not always the turnaround: an outbound waypoint
    can lie farther out.  Falls back to the rule below (and says so) when the
    dump carries no palindromic reference_path.
  - loop: the scored row whose current position is farthest (3D euclidean)
    from the clip's first frame (ties: earliest).
  Key rows = the scored rows just before and after it (they straddle the
  turnaround frame by construction), the turn-around row itself, and the final
  row (back at the start).  cases.json records the rule used per pattern.

Ties (distance equal to 1e-12) go to the smaller order key: sha1("<scene>:<episode_id>")
for C/D (their selection key), sha1("<scene>/<clip>") for A/B/E.  Episodes
with no GT-visible slot have no PCK/bearing value and are skipped.

Usage: cd <repo> && <qwen25 python> -m scripts.exp18.select_cases [--metrics-dir DIR] [--out FILE]
Env fallbacks: EXP18_ROOT (via common), EXP18_METRICS_DIR.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.exp18 import common  # noqa: E402
from scripts.exp18.compute_metrics import E_PATTERNS, TIER_ORDER, read_table  # noqa: E402

SCHEMA = "heatmapvln-exp18-cases-v1"
MAIN = {"tier": "C", "min_rows": 4, "min_path_m": 8.0, "min_span_deg": 90.0, "top": 5}
GALLERY_PERCENTILES = (10, 50, 90)
EPISODE_FIELDS = ("tier", "scene", "clip", "clip_key", "episode_id", "npz_path", "order_key", "pattern",
                  "n_scored_rows", "n_visible_slots", "path_length_m", "max_row_span_deg", "vo_pck8", "gt_pck8",
                  "floor_pck8", "vo_bearing_err_median", "gt_bearing_err_median")


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def episode_record(ep: pd.Series) -> dict:
    return {k: _py(ep[k]) for k in EPISODE_FIELDS if k in ep.index}


def row_record(r: pd.Series, role: str) -> dict:
    out = {"role": role}
    for k in ("row", "query_idx", "t", "is_final", "dist_from_start_m", "gt_bearing_span_deg", "n_visible",
              "vo_pck8_row", "gt_pck8_row"):
        if k in r.index:
            out[k] = _py(r[k])
    return out


def nearest(df: pd.DataFrame, col: str, target: float) -> pd.Series:
    """Episode whose ``col`` is closest to ``target``; ties -> smaller order_key."""
    d = (df[col] - target).abs().round(12)
    return df.assign(_d=d).sort_values(["_d", "order_key"], kind="stable").iloc[0]


def scored_rows(rows: pd.DataFrame, ep: pd.Series) -> pd.DataFrame:
    # clip keys repeat across tiers (C and E renders both start at clip_000001), so match the tier too
    sel = (rows["tier"] == ep["tier"]) & (rows["clip_key"] == ep["clip_key"])
    return rows[sel].sort_values("t", kind="stable").reset_index(drop=True)


def main_figure(eps: pd.DataFrame, rows: pd.DataFrame) -> dict:
    c = eps[(eps["tier"] == MAIN["tier"]) & eps["vo_pck8"].notna()]
    if c.empty:
        return {"status": "missing", "reason": "tier C has no scored episode with a VO arm"}
    median = float(np.median(c["vo_pck8"]))
    cand = c[(c["n_scored_rows"] >= MAIN["min_rows"]) & (c["path_length_m"] >= MAIN["min_path_m"])
             & (c["max_row_span_deg"] >= MAIN["min_span_deg"])]
    ranked = cand.assign(abs_diff=(cand["vo_pck8"] - median).abs().round(12)).sort_values(
        ["abs_diff", "order_key"], kind="stable")
    selected = []
    for rank, (_, ep) in enumerate(ranked.head(MAIN["top"]).iterrows(), start=1):
        r = scored_rows(rows, ep)
        span_idx = int(r["gt_bearing_span_deg"].to_numpy().argmax())  # first max = earliest
        final = r[r["is_final"]]
        key_rows = [row_record(r.iloc[0], "first_endpoint"), row_record(r.iloc[span_idx], "max_span"),
                    row_record(final.iloc[-1] if len(final) else r.iloc[-1], "final")]
        selected.append(dict(episode_record(ep), rank=rank, abs_diff_from_tier_median=_py(ep["abs_diff"]),
                             key_rows=key_rows))
    return {"status": "ok", "tier": MAIN["tier"], "tier_median_episode_vo_pck8": median,
            "n_episodes": int(len(c)), "n_candidates": int(len(cand)), "criteria": MAIN, "selected": selected}


def gallery(eps: pd.DataFrame) -> dict:
    out = {}
    for tier in TIER_ORDER:
        t = eps[(eps["tier"] == tier) & eps["vo_bearing_err_median"].notna()] if "vo_bearing_err_median" in eps else eps[:0]
        if t.empty:
            out[tier] = {"status": "missing"}
            continue
        values = t["vo_bearing_err_median"].to_numpy(dtype=np.float64)
        picks = []
        for p in GALLERY_PERCENTILES:
            target = float(np.percentile(values, p))
            ep = nearest(t, "vo_bearing_err_median", target)
            picks.append(dict(episode_record(ep), percentile=p, target_vo_bearing_err_median=target))
        keys = [pk["clip_key"] for pk in picks]
        out[tier] = {"status": "ok", "n_episodes": int(len(t)), "picks": picks,
                     "duplicate_picks": len(set(keys)) < len(keys)}
    return out


def turnaround_row(r: pd.DataFrame, ep: pd.Series, pattern: str):
    """Index into the episode's scored rows ``r`` of its turnaround row, plus how it was chosen."""
    farthest = int(r["dist_from_start_m"].to_numpy().argmax())  # first max = earliest
    frame = ep.get("turnaround_frame", -1)
    frame = int(frame) if frame is not None and np.isfinite(frame) else -1
    info = {"turnaround_frame": frame if frame >= 0 else None,
            "farthest_from_start_row_t": int(r["t"].iloc[farthest])}
    if pattern == "out_and_back" and frame >= 0:
        turn = int(np.abs(r["t"].to_numpy() - frame).argmin())  # first min = earlier row
        info["turnaround_rule"] = "route turnaround frame (reference_path midpoint) -> nearest scored row"
        info["turnaround_miss_m"] = _py(ep.get("turnaround_miss_m"))
    else:
        turn = farthest
        info["turnaround_rule"] = "scored row farthest (3D euclidean) from the first frame" + (
            " (FALLBACK: no palindromic reference_path in the dump)" if pattern == "out_and_back" else "")
    info["turnaround_row_t"] = int(r["t"].iloc[turn])
    return turn, info


def pattern_figure(eps: pd.DataFrame, rows: pd.DataFrame) -> dict:
    out = {"_unresolved_pattern_episodes": int(((eps["tier"] == "E") & (eps["pattern"] == "")).sum())}
    e = eps[(eps["tier"] == "E") & eps["vo_pck8"].notna()] if "vo_pck8" in eps else eps[:0]
    for pattern in E_PATTERNS:
        p = e[e["pattern"] == pattern]
        if p.empty:
            out[pattern] = {"status": "missing", "n_episodes": 0}
            continue
        median = float(np.median(p["vo_pck8"]))
        ep = nearest(p, "vo_pck8", median)
        r = scored_rows(rows, ep)
        turn, turn_info = turnaround_row(r, ep, pattern)
        key_rows = []
        if turn > 0:
            key_rows.append(row_record(r.iloc[turn - 1], "before_turnaround"))
        key_rows.append(row_record(r.iloc[turn], "turnaround"))
        if turn + 1 < len(r):
            key_rows.append(row_record(r.iloc[turn + 1], "after_turnaround"))
        final = r[r["is_final"]]
        last = final.iloc[-1] if len(final) else r.iloc[-1]
        if int(last["row"]) not in [k["row"] for k in key_rows]:
            key_rows.append(row_record(last, "return_to_start"))
        out[pattern] = dict(episode_record(ep), status="ok", n_episodes=int(len(p)), pattern_median_vo_pck8=median,
                            **turn_info, key_rows=key_rows)
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--metrics-dir", type=Path,
                   default=Path(os.environ.get("EXP18_METRICS_DIR", str(common.EXP_ROOT / "metrics"))))
    p.add_argument("--out", type=Path, default=None, help="default: <metrics-dir>/cases.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    eps = read_table(args.metrics_dir / "episodes")
    rows = read_table(args.metrics_dir / "rows")
    eps["pattern"] = eps["pattern"].fillna("").astype(str) if "pattern" in eps else ""
    res = {
        "schema": SCHEMA, "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "metrics_dir": str(args.metrics_dir),
        "tiers_present": sorted(set(eps["tier"])),
        "rules": {"source": "docs/experiments/README.md EXP-18 案例挑选规则",
                  "ties": "equal distance (1e-12) -> smaller order_key; order_key = sha1(scene:episode_id) for C/D, "
                          "sha1(scene/clip) for A/B/E",
                  "percentile": "np.percentile linear over the tier's per-episode VO median bearing errors; "
                                "pick the episode nearest to it",
                  "turnaround": {"out_and_back": "scored row nearest (ties: earlier) to the frame reaching "
                                                  "reference_path[(len-1)//2] (renderer turnaround_index), matched "
                                                  "in order; falls back to the loop rule without a palindromic path",
                                 "loop": "scored row farthest (3D euclidean) from the clip's first frame; "
                                         "ties -> earliest"}},
        "main_figure": main_figure(eps, rows),
        "gallery": gallery(eps),
        "pattern_figure": pattern_figure(eps, rows),
    }
    out = args.out or args.metrics_dir / "cases.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    mf = res["main_figure"]
    print(f"main figure: {mf.get('status')} candidates={mf.get('n_candidates')} "
          f"selected={[s['clip_key'] for s in mf.get('selected', [])]}")
    for tier, g in res["gallery"].items():
        print(f"gallery {tier}: {g.get('status')} {[(p['percentile'], p['clip_key']) for p in g.get('picks', [])]}")
    for pattern, f in res["pattern_figure"].items():
        if isinstance(f, dict):
            print(f"pattern {pattern}: {f.get('status')} {f.get('clip_key', '')} "
                  f"turnaround t={f.get('turnaround_row_t')} ({f.get('turnaround_rule')}; "
                  f"farthest-from-start t={f.get('farthest_from_start_row_t')})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

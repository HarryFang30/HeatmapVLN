#!/usr/bin/env python3
"""EXP-25 B1/B2 probe metrics from a closed-loop run's client logs (CPU only; Habitat PathFinder, no renderer).

Definitions (docs/experiments/README.md EXP-25, fixed before the probes ran):
- B: share of episodes whose FIRST pixel goal is "bad".  The first pixel goal is the first ``trajectory``
  call whose "u v" answer projects onto the floor and snaps to the navmesh; with g the geodesic distance to
  the goal at the agent's position, gT the geodesic distance at the pixel target's point and tdist the
  target's straight-line distance, the call is bad when (g - gT) / max(tdist, 0.25) <= -0.3, good when the
  ratio is >= 0.5, mid otherwise.  Denominator: episodes that have such a call.
- E7: failed, never within 3 m (os = 0), ended with STOP, and stopped >= 7 m from the goal.
- instruction-start type: "turn" when the instruction (lower-cased, leading punctuation dropped) starts with
  a turn phrase (TURN_START_RE), else "other".

The pose replay and the pixel projection are the 2026-10-15 research decomposition (``decompose.py``) without
change: the logged actions of each call are replayed on the navmesh from the episode start (try_step, 0.25 m,
15 deg), the 30 deg look-down 640x480 HFOV-79 camera at 1.25 m projects the pixel onto the floor plane.

  python scripts/exp25/probe_metrics.py --name val_seen --datasets <plan>/cohorts/dataset_shard_0*.json.gz \
      --progress '<run>/workers/shard_0*/progress.json' --logs '<run>/runtime/*/logs/client_shard_0*.log' \
      --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import re
from multiprocessing import Pool

import numpy as np

SCENES = "/dataset/mp3d"

# ---- parser: scripts/exp19/select_cases.py parse_client_log, verbatim regexes ----
HEADER_RE = re.compile(r"^\[(\d+)/(\d+)\] Episode (\w+?)_(\d+): ")
STEP_RE = re.compile(r"step_id: (\d+), RPC kind=(\w+), VLM output:(.*)$")
TRAJ_RE = re.compile(r"\[debug\] trajectory (?:native_)?traj_goal=\(([-\d.]+),([-\d.]+)\), direct=([-\d.]+), "
                     r"path_len=([-\d.]+), actions=\[([\d, ]*)\]")
ACTIONS_RE = re.compile(r"\[debug\] actions=\[([\d, ]*)\]")
END_RE = re.compile(r"=> success: ([\d.]+), spl: ([\d.]+), os: ([\d.]+), ne: ([\d.]+), vlm_calls: (\d+), "
                    r"trajectory_calls: (\d+)")

_MOD = r"(?:immediate|quick|sharp|hard|slight|full|complete)\s+"
_TURN = r"(?:left|right|u-?\s?turn|180|360)"
TURN_START_RE = re.compile(
    r"^(?:then\s+|now\s+|first\s+|first,\s+)?(?:"
    r"turn|rotate|spin|pivot|veer|face|u-?\s?turn|about\s+face"
    rf"|(?:make|take|do)\s+an?\s+(?:{_MOD})?{_TURN}"
    r"|(?:go|head|bear|hang)\s+(?:left|right)"
    r")\b"
)

BAD, GOOD = -0.3, 0.5


def instruction_type(text: str) -> str:
    s = re.sub(r"^[\s\W_]+", "", str(text or "").lower())
    return "turn" if TURN_START_RE.match(s) else "other"


def scene_name(scene_id: str) -> str:
    return os.path.splitext(os.path.basename(str(scene_id)))[0]


def _ints(text):
    return [int(x) for x in text.split(",") if x.strip()]


def parse_client_log(path):
    with open(path, errors="replace", newline="\n") as handle:
        lines = handle.readlines()
    blocks, cur = {}, None
    for physical in lines:
        for line in physical.rstrip("\n").split("\r"):
            m = HEADER_RE.match(line)
            if m:
                cur = {"scene_id": m.group(3), "episode_id": int(m.group(4)), "calls": [], "final": None}
                continue
            if cur is None:
                continue
            s = line.strip()
            m = STEP_RE.match(s)
            if m:
                cur["calls"].append({"step": int(m.group(1)), "kind": m.group(2), "out": m.group(3).strip(),
                                     "actions": None, "traj_goal": None})
                continue
            m = TRAJ_RE.match(s)
            if m and cur["calls"]:
                cur["calls"][-1]["traj_goal"] = [float(m.group(1)), float(m.group(2))]
                cur["calls"][-1]["actions"] = _ints(m.group(5))
                continue
            m = ACTIONS_RE.match(s)
            if m and cur["calls"]:
                cur["calls"][-1]["actions"] = _ints(m.group(1))
                continue
            m = END_RE.search(s)
            if m:
                g = m.groups()
                cur["final"] = {"success": float(g[0]), "spl": float(g[1]), "os": float(g[2]), "ne": float(g[3]),
                                "vlm_calls": int(g[4]), "trajectory_calls": int(g[5])}
                blocks[(cur["scene_id"], cur["episode_id"])] = cur
                cur = None
    return blocks


# ---- decomposition (research decompose.py, unchanged) ----
_PF = {}
W, H, HFOV, CAMH, PITCH = 640, 480, 79.0, 1.25, 30.0
F = (W / 2.0) / math.tan(math.radians(HFOV) / 2.0)
CX, CY = (W - 1) / 2.0, (H - 1) / 2.0


def pathfinder(scene):
    import habitat_sim
    if scene not in _PF:
        pf = habitat_sim.PathFinder()
        pf.load_nav_mesh(f"{SCENES}/{scene}/{scene}.navmesh")
        _PF[scene] = pf
    return _PF[scene]


def geo(pf, a, ends):
    import habitat_sim
    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = np.asarray(a, dtype=np.float32)
    path.requested_ends = np.asarray(ends, dtype=np.float32).reshape(-1, 3)
    pf.find_path(path)
    return float(path.geodesic_distance)


def geo_path(pf, a, b):
    import habitat_sim
    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(a, dtype=np.float32)
    path.requested_end = np.asarray(b, dtype=np.float32)
    pf.find_path(path)
    return [np.asarray(p, dtype=np.float64) for p in path.points]


def yaw_to(theta, pos, tgt):
    fx, fz = -math.sin(theta), -math.cos(theta)
    tx, tz = tgt[0] - pos[0], tgt[2] - pos[2]
    cross = fz * tx - fx * tz
    dot = fx * tx + fz * tz
    return math.degrees(math.atan2(cross, dot))  # left positive


def pixel_floor(pos, theta, u, v):
    """Ray through pixel (u, v) of the 30deg look-down camera intersected with the floor plane."""
    d = np.array([(u - CX) / F, -(v - CY) / F, -1.0])
    a = -math.radians(PITCH)
    pitch = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    c, s = math.cos(theta), math.sin(theta)
    yaw = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    dw = yaw @ pitch @ d
    if dw[1] >= -1e-6:
        return None, None
    t = CAMH / -dw[1]
    p = np.asarray(pos, dtype=np.float64) + np.array([0.0, CAMH, 0.0]) + t * dw
    p[1] = pos[1]
    return p, float(math.hypot(p[0] - pos[0], p[2] - pos[2]))


def first_ahead(points, pos, min_d=0.5):
    for p in points[1:]:
        if math.hypot(p[0] - pos[0], p[2] - pos[2]) >= min_d:
            return p
    return points[-1] if points else None


def decompose(args):
    ep, block, prog_row = args
    scene = ep["scene_id"].split("/")[1]
    pf = pathfinder(scene)
    goal = [g["position"] for g in ep["goals"]]
    gpos = np.asarray(goal[0], dtype=np.float64)
    pos = np.asarray(ep["start_position"], dtype=np.float32)
    q = ep["start_rotation"]
    theta = 2.0 * math.atan2(q[1], q[3])
    calls = block["calls"]
    steps_total = int(prog_row["steps"])
    out = []
    n_fwd = n_blocked = 0
    for i, c in enumerate(calls):
        s = c["step"]
        g_now = geo(pf, pos, goal)
        row = {"i": i, "step": s, "kind": c["kind"], "g": round(g_now, 3)}
        gp = geo_path(pf, pos, gpos)
        nxt = first_ahead(gp, pos) if gp else None
        if nxt is not None:
            row["yaw_route"] = round(yaw_to(theta, pos, nxt), 1)
        if c["kind"] == "trajectory":
            try:
                u, v = [float(x) for x in c["out"].split()[:2]]
            except Exception:
                u = v = None
            if u is not None:
                T, dist = pixel_floor(pos, theta, u, v)
                if T is not None:
                    Ts = np.asarray(pf.snap_point(T.astype(np.float32)), dtype=np.float64)
                    snap_err = float(np.linalg.norm(Ts - T)) if np.all(np.isfinite(Ts)) else None
                    g_T = geo(pf, Ts, goal) if snap_err is not None else None
                    row.update({"u": u, "v": v, "tdist": round(dist, 2),
                                "snap": None if snap_err is None else round(snap_err, 2),
                                "gT": None if g_T is None or not math.isfinite(g_T) else round(g_T, 3),
                                "yaw_T": round(yaw_to(theta, pos, T), 1)})
        # execute this call's actions
        if c["kind"] == "stop":
            row["next_g"] = round(g_now, 3)
            out.append(row)
            break
        if i + 1 < len(calls):
            n = calls[i + 1]["step"] - s
        else:
            n = steps_total - s - (1 if prog_row.get("step_budget_stop") else 0)
        acts = [a for a in (c["actions"] or [])[:n] if a != 0]
        p0 = pos.copy()
        theta0 = theta
        for a in acts:
            if a == 1:
                fwd = np.array([-math.sin(theta), 0.0, -math.cos(theta)], dtype=np.float32)
                end = pos + 0.25 * fwd
                newp = np.asarray(pf.try_step(pos, end), dtype=np.float32)
                n_fwd += 1
                if float(np.linalg.norm(newp - pos)) < 0.20:
                    n_blocked += 1
                pos = newp
            elif a == 2:
                theta += math.radians(15)
            elif a == 3:
                theta -= math.radians(15)
        disp = float(np.linalg.norm((pos - p0)[[0, 2]]))
        row["disp"] = round(disp, 2)
        row["n_fwd"] = sum(1 for a in acts if a == 1)
        row["n_turn"] = sum(1 for a in acts if a in (2, 3))
        if "yaw_T" in row and disp >= 0.3:
            row["yaw_disp"] = round(yaw_to(theta0, p0, pos), 1)
        out.append(row)
    for k in range(len(out) - 1):
        out[k]["next_g"] = out[k + 1]["g"]
    if out and "next_g" not in out[-1]:
        out[-1]["next_g"] = round(geo(pf, pos, goal), 3)
    return {"scene": scene, "episode_id": int(ep["episode_id"]), "success": prog_row["success"],
            "os": prog_row["os"], "ne": prog_row["ne"], "spl": prog_row["spl"], "steps": steps_total,
            "geo0": float(ep["info"]["geodesic_distance"]), "n_fwd": n_fwd, "n_blocked": n_blocked,
            "instruction": ep["instruction"]["instruction_text"], "calls": out}


# ---- per-episode classes and the summary ----
def category(e) -> str:
    """S / O (reached 3 m, failed) / L0 (never reached, no STOP) / E<7 / E7 (never reached, stopped >= 7 m)."""
    if e["success"] > 0:
        return "S"
    if e["os"] > 0:
        return "O"
    last = e["calls"][-1]["kind"] if e["calls"] else ""
    if last != "stop":
        return "L0"
    return "E7" if e["ne"] >= 7 else "E<7"


def first_pixel(e):
    return next((c for c in e["calls"] if c["kind"] == "trajectory" and c.get("gT") is not None), None)


def pixel_ratio(c) -> float:
    return (c["g"] - c["gT"]) / max(c["tdist"], 0.25)


def pixel_class(c) -> str | None:
    if c is None:
        return None
    r = pixel_ratio(c)
    return "bad" if r <= BAD else ("good" if r >= GOOD else "mid")


def wilson(k: int, n: int, z: float = 1.959964) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (mid - half), 2), round(100 * (mid + half), 2)]


def share(k: int, n: int) -> dict:
    return {"k": k, "n": n, "pct": None if n == 0 else round(100 * k / n, 2), "ci95": wilson(k, n)}


def mean_ci(xs, scale=1.0) -> dict:
    xs = np.asarray(xs, dtype=np.float64) * scale
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    m = float(xs.mean())
    se = float(xs.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return {"n": n, "mean": round(m, 3), "ci95": [round(m - 1.959964 * se, 3), round(m + 1.959964 * se, 3)]}


def summarize(rows) -> dict:
    n = len(rows)
    cats = [category(e) for e in rows]
    fp = [pixel_class(first_pixel(e)) for e in rows]
    with_pix = [i for i in range(n) if fp[i] is not None]
    out = {
        "episodes": n,
        "sr": share(sum(e["success"] > 0 for e in rows), n),
        "os": share(sum(e["os"] > 0 for e in rows), n),
        "spl": mean_ci([e["spl"] for e in rows], 100.0),
        "ne": mean_ci([e["ne"] for e in rows]),
        "categories": {k: share(cats.count(k), n) for k in ("S", "O", "E<7", "E7", "L0")},
        "E7": share(cats.count("E7"), n),
        "B": share(sum(fp[i] == "bad" for i in with_pix), len(with_pix)),
        "B_over_all_episodes": share(sum(f == "bad" for f in fp), n),
        "episodes_without_first_pixel": n - len(with_pix),
        "first_pixel": {},
        "by_instruction_type": {},
    }
    for q in ("good", "mid", "bad"):
        idx = [i for i in with_pix if fp[i] == q]
        out["first_pixel"][q] = {"share": share(len(idx), len(with_pix)),
                                 "sr": share(sum(rows[i]["success"] > 0 for i in idx), len(idx))}
    types = [instruction_type(e["instruction"]) for e in rows]
    for t in ("turn", "other"):
        idx = [i for i in range(n) if types[i] == t]
        pix = [i for i in idx if fp[i] is not None]
        out["by_instruction_type"][t] = {
            "episodes": len(idx),
            "B": share(sum(fp[i] == "bad" for i in pix), len(pix)),
            "E7": share(sum(cats[i] == "E7" for i in idx), len(idx)),
            "sr": share(sum(rows[i]["success"] > 0 for i in idx), len(idx)),
        }
    return out


def load_episodes(paths):
    eps = {}
    for p in paths:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for e in json.load(fh)["episodes"]:
                eps[(scene_name(e["scene_id"]), int(e["episode_id"]))] = e
    return eps


def load_progress(patterns):
    prog = {}
    for pattern in patterns:
        for p in sorted(glob.glob(pattern)):
            with open(p) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        prog[(scene_name(r["scene_id"]), int(r["episode_id"]))] = r
    return prog


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--datasets", nargs="+", required=True, help="R2R-CE json.gz files holding the run's episodes")
    ap.add_argument("--progress", nargs="+", required=True, help="glob(s) of progress.json / progress.jsonl")
    ap.add_argument("--logs", nargs="+", required=True, help="glob(s) of client_shard_*.log, later files win")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)

    eps = load_episodes(args.datasets)
    prog = load_progress(args.progress)
    blocks = {}
    for pattern in args.logs:
        for p in sorted(glob.glob(pattern)):
            blocks.update(parse_client_log(p))
    keys = sorted(set(prog) & set(blocks))
    missing = sorted(set(prog) - set(blocks))
    for k in keys:  # the log's own end line must agree with progress.json
        f, r = blocks[k]["final"], prog[k]
        if abs(f["success"] - float(r["success"])) > 1e-6 or abs(f["ne"] - float(r["ne"])) > 1e-3:
            raise SystemExit(f"log/progress disagree for {k}: {f} vs success={r['success']} ne={r['ne']}")
    jobs = [(eps[k], blocks[k], prog[k]) for k in keys]
    jobs.sort(key=lambda j: j[0]["scene_id"])
    with Pool(args.workers) as pool:
        rows = pool.map(decompose, jobs, chunksize=4)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.name}.decomp.json"), "w") as fh:
        json.dump(rows, fh)
    summary = {"name": args.name, "progress_rows": len(prog), "episodes_with_log": len(keys),
               "progress_rows_without_log": [list(k) for k in missing], **summarize(rows)}
    with open(os.path.join(args.out, f"{args.name}.summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

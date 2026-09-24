#!/usr/bin/env python3
"""EXP-19 [A]: pre-registered case selection, per-GPU episode lists and eval-log references.

Implements docs/experiments/README.md EXP-19 "候选集" on the two main-table
evaluations of the deployed arm (seed 42 and seed 1337, both 1839 episodes of
R2R val_unseen), so the 15 episodes are fixed before any rerun:

* pool: the category predicate holds in BOTH seeds, ``ppa_applied_calls >= 5``
  in both seeds, and the category's seed-42 extra condition holds.
* categories (predicates on both seeds | seed-42 extra):
    T1 multi-room, multi-turn  success; >= 2 reference turns >= 45 deg; >= 3 room
                               runs; reference dy <= 1 m           | steps <= 150
    T2 cross-floor             success; dy > 1 m                   | steps <= 150
    T3 long range              success; geodesic >= 12 m           | steps <= 200
    F1 reached, stopped wrong  os = 1, success = 0, ended by STOP (not the cap)
    F2 wandering               success = 0, ended at the 500-step cap
* order: |steps_42 - median steps_42 of the category pool| ascending (typical,
  not best); ties -> smaller sha1("<scene>:<episode_id>").
* pick: walk that order, skip an episode whose scene the category already
  holds, take the first 3 -> 5 x 3 = 15 episodes.

The reference-path features are the selection scout's definitions verbatim
(ref_turns45, n_room_runs, dy, geodesic, ended_by; see DEFINITIONS).  The
client-log parser is the scout's too, extended to keep every call field the
rerun is compared against.

Outputs (``--out-dir``, default EXP19_ROOT/cases):
  candidates.json                   schema exp19-candidates-v1: inputs (sha256),
                                    definitions, per-category pool size, median,
                                    ordered pool head, the 3 candidates, GPU lists
                                    and the self-check table
  episode_lists/gpu<j>.json         client ``--episode_list`` files (cohort format
                                    of evaluation_plans/.../cohorts/shard_XX.json)
  eval_log_reference/<ep_key>.json  schema exp19-eval-log-ref-v1: every System2 call
                                    of the seed-42 client log for that episode

GPU lists: longest-processing-time greedy on steps_42 (longest first, onto the
least-loaded GPU, ties -> lower index); inside a list the order is category
order, then rank.  Everything is deterministic; no timestamps are written.

``--self-check`` recomputes the scout's categories whose definitions coincide
with the ones here and compares the counts with the scout's published numbers
(plus log/progress consistency), prints the table and exits 1 on a mismatch
without writing anything.  A normal run stores the same table in
candidates.json and refuses to write if it fails.

Usage (dev machine, envs/qwen25 or envs/vlnce; stdlib only):
  cd <repo> && PYTHONDONTWRITEBYTECODE=1 <python> -m scripts.exp19.select_cases [--out-dir DIR] [--num-gpus N]
Env fallbacks: EXP19_WORKSPACE, EXP19_ROOT, EXP19_GIT_SHA (else <repo>/.exp19_git_sha).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]

WORKSPACE = Path(os.environ.get("EXP19_WORKSPACE", "/mnt/afs/liwenhao/agent/370910109"))
EXP_ROOT = Path(os.environ.get("EXP19_ROOT", str(WORKSPACE / "model" / "exp19_behavior_viz")))
EVAL_42 = WORKSPACE / "model" / "eval_ppa_refine_v2_nativefix_r2r_val_unseen_8gpu"
EVAL_1337 = WORKSPACE / "model" / "eval_ppa_refine_v2_nativefix_r2r_val_unseen_8gpu_seed1337"
DEFAULT_PROGRESS = {"42": EVAL_42 / "merged" / "progress.jsonl", "1337": EVAL_1337 / "merged" / "progress.jsonl"}
DEFAULT_LOG_GLOBS = {
    "42": str(EVAL_42 / "runtime" / "20260831_075400_job1" / "logs" / "client_[0-9].log"),
    "1337": str(EVAL_1337 / "runtime" / "20260904_042313_job1" / "logs" / "client_[0-9].log"),
}
VAL_UNSEEN = (WORKSPACE / "habitat" / "VLN-CE" / "data" / "datasets" / "R2R_VLNCE_v1-3_preprocessed"
              / "val_unseen" / "val_unseen.json.gz")
MP3D_SCENES = WORKSPACE / "habitat" / "VLN-CE" / "data" / "scene_datasets" / "mp3d"

CANDIDATES_SCHEMA = "exp19-candidates-v1"
EVAL_LOG_REF_SCHEMA = "exp19-eval-log-ref-v1"
SEEDS = ("42", "1337")
NUM_EPISODES = 1839
STEP_CAP = 500  # --max_steps_per_episode of the main-table launcher
MIN_PPA_APPLIED = 5
TOP_PER_CATEGORY = 3
POOL_HEAD = 20
MAX_GPUS = 3  # dev-machine rule (ledger §0.7)

# reference-path feature constants (scout values)
MIN_SEGMENT_M = 0.3
TURN_DEG = 45.0
PANO_HEIGHT_M = 1.4
PANO_MAX_DY_M = 1.0
PANO_MATCH_M = 0.5


def _t1(s, d):
    return d["success"] and s["ref_turns45"] >= 2 and s["n_room_runs"] >= 3 and s["dy_m"] <= 1.0


def _t2(s, d):
    return d["success"] and s["dy_m"] > 1.0


def _t3(s, d):
    return d["success"] and s["geodesic_m"] >= 12.0


def _f1(s, d):
    return d["os"] and not d["success"] and d["ended_by"] == "stop"


def _f2(s, d):
    return not d["success"] and d["ended_by"] == "step_cap"


# name, predicate on (static features, one seed's run) that must hold in both seeds,
# seed-42 extra on the seed-42 run (None = no extra), and the literal text.
CATEGORIES = {
    "T1": {"name": "multi-room, multi-turn", "both": _t1, "seed42": lambda d: d["steps"] <= 150,
           "text": "both seeds: success; ref_turns45 >= 2; n_room_runs >= 3; dy_m <= 1.0 | seed 42: steps <= 150"},
    "T2": {"name": "cross-floor", "both": _t2, "seed42": lambda d: d["steps"] <= 150,
           "text": "both seeds: success; dy_m > 1.0 | seed 42: steps <= 150"},
    "T3": {"name": "long range", "both": _t3, "seed42": lambda d: d["steps"] <= 200,
           "text": "both seeds: success; geodesic_m >= 12.0 | seed 42: steps <= 200"},
    "F1": {"name": "reached the goal area, stopped wrong", "both": _f1, "seed42": None,
           "text": "both seeds: os = 1; success = 0; ended_by = stop"},
    "F2": {"name": "wandering", "both": _f2, "seed42": None,
           "text": "both seeds: success = 0; ended_by = step_cap"},
}
CATEGORY_ORDER = tuple(CATEGORIES)

DEFINITIONS = {
    "source": "docs/experiments/README.md EXP-19 候选集 (pre-registered); feature code = selection scout, verbatim",
    "pool": f"category predicate holds in both seeds (42, 1337) AND ppa_applied_calls >= {MIN_PPA_APPLIED} in both "
            "seeds AND the category's seed-42 extra condition holds",
    "categories": {c: v["text"] for c, v in CATEGORIES.items()},
    "success, os, ne, spl, steps, ppa_applied": "merged/progress.jsonl fields success, os, ne, spl, steps "
                                                "(primitive actions incl. the final STOP), ppa_applied_calls",
    "ended_by": "'stop' if the episode's last System2 call in the client log has RPC kind=stop; else 'step_cap' if "
                f"steps >= {STEP_CAP}; else 'other'",
    "ref_turns45": f"reference_path consecutive points -> xz segments; drop segments with xz length <= {MIN_SEGMENT_M} m; "
                   "heading = atan2(dx, dz) in degrees; count |wrap(heading[i+1] - heading[i])| >= "
                   f"{TURN_DEG:g} deg between consecutive remaining segments",
    "n_room_runs": "each reference_path point (habitat x, y, z) -> nearest MP3D .house panorama ('P' line, 32-char "
                   "name) in the horizontal plane (house x = x, house y = -z) among panoramas with "
                   f"|pano_z - (y + {PANO_HEIGHT_M})| <= {PANO_MAX_DY_M} m; its region ('R' line) if the horizontal "
                   f"distance < {PANO_MATCH_M} m, else unmatched; unmatched points and region < 0 dropped; consecutive "
                   "equal regions merged; n_room_runs = number of runs",
    "dy_m": "max(y) - min(y) over reference_path",
    "geodesic_m": "episode info.geodesic_distance in val_unseen.json.gz",
    "order": "|steps_42 - median(steps_42 of the category pool)| ascending; ties -> smaller "
             "sha1('<scene_id>:<episode_id>') hex digest (episode_id as a plain integer)",
    "pick": f"walk the order, skip an episode whose scene is already picked in the same category, "
            f"take the first {TOP_PER_CATEGORY}",
    "gpu_lists": "longest-processing-time greedy on steps_42 (descending; ties category order, rank) onto the GPU "
                 "with the smallest summed steps_42 (ties lower index); each list ordered by category, then rank",
}

# The scout's published counts (exp19_scout_pool.py §4 and §1 output) for the
# categories whose definitions coincide with this module's features:
# (seed 42, seed 1337, both seeds, both seeds & ppa_applied >= 5 in both).
SCOUT_COUNTS = {
    "S1 success & ref_turns45 >= 2": ((lambda s, d: d["success"] and s["ref_turns45"] >= 2), (410, 390, 324, 308)),
    "S5 success & n_room_runs >= 4": ((lambda s, d: d["success"] and s["n_room_runs"] >= 4), (481, 459, 393, 372)),
    "S6 dy > 1 m": ((lambda s, d: s["dy_m"] > 1.0), (261, 261, 261, 194)),
    "S6s dy > 1 m & success (= T2 before its step limit)": (_t2, (134, 127, 108, 88)),
    "S8 geodesic >= 10 m & success": ((lambda s, d: d["success"] and s["geodesic_m"] >= 10), (276, 270, 227, 224)),
    "F1a os & !success & STOP (= F1)": (_f1, (125, 122, 53, 53)),
    "F2 !success & 500-step cap (= F2)": (_f2, (129, 146, 72, 55)),
}
SCOUT_TERMINATION = {"42": {"stop": 1710, "step_cap": 129, "ppa>=5": 1672},
                     "1337": {"stop": 1693, "step_cap": 146, "ppa>=5": 1674}}


# ---------------------------------------------------------------------------
# client logs
# ---------------------------------------------------------------------------
HEADER_RE = re.compile(r"^\[(\d+)/(\d+)\] Episode (\w+?)_(\d+): ")
STEP_RE = re.compile(r"step_id: (\d+), RPC kind=(\w+), VLM output:(.*)$")
TRAJ_RE = re.compile(r"\[debug\] trajectory (?:native_)?traj_goal=\(([-\d.]+),([-\d.]+)\), direct=([-\d.]+), "
                     r"path_len=([-\d.]+), actions=\[([\d, ]*)\]")
ACTIONS_RE = re.compile(r"\[debug\] actions=\[([\d, ]*)\]")
VO_RE = re.compile(r"\[amb3r-vo\] frame=(\d+) history=\[([\d, ]*)\] ready=(\w+) phase=(\w+)(?: revision=(\d+))?")
END_RE = re.compile(r"=> success: ([\d.]+), spl: ([\d.]+), os: ([\d.]+), ne: ([\d.]+), vlm_calls: (\d+), "
                    r"trajectory_calls: (\d+)")


def _ints(text: str) -> list:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_client_log(path, lines=None) -> tuple:
    """Episode blocks of one client log.

    Returns ``(blocks, n_duplicate, n_incomplete)``; ``blocks`` maps episode_id to
    ``{"scene_id", "episode_id", "client_log", "line_start", "line_end", "calls", "final"}``
    (1-based line numbers of the header and the ``=> success`` line).  A block
    without its ``=> success`` line (interrupted episode) is dropped; a repeated
    complete block replaces the earlier one (the later run wrote the progress row).
    ``call_index`` is the 0-based order of ``step_id: ..., RPC kind=`` lines in the
    block, i.e. the client's ``system2_call_index``.  The ``[amb3r-vo]`` line printed
    just before a call belongs to that call.

    Line numbers count ``\\n`` only (as ``grep -n`` / ``sed -n`` do); the tqdm bar
    shares the file and writes ``\\r``, so each physical line is parsed as its
    ``\\r``-separated fragments.
    """
    if lines is None:
        with open(path, errors="replace", newline="\n") as handle:
            lines = handle.readlines()
    blocks, dup, incomplete = {}, 0, 0
    cur, pending_vo = None, None
    fragments = ((n, frag) for n, physical in enumerate(lines, start=1)
                 for frag in physical.rstrip("\n").split("\r"))
    for lineno, line in fragments:
        m = HEADER_RE.match(line)
        if m:
            if cur is not None:
                incomplete += 1
            cur = {"scene_id": m.group(3), "episode_id": int(m.group(4)), "client_log": str(path),
                   "line_start": lineno, "line_end": None, "calls": [], "final": None}
            pending_vo = None
            continue
        if cur is None:
            continue
        s = line.strip()
        m = STEP_RE.match(s)
        if m:
            vo = pending_vo or {}
            cur["calls"].append({
                "call_index": len(cur["calls"]), "step": int(m.group(1)), "kind": m.group(2),
                "vlm_output": m.group(3).strip(), "actions": None, "traj_goal": None,
                "vo_frame": vo.get("frame"), "vo_history": vo.get("history"), "vo_ready": vo.get("ready"),
                "vo_phase": vo.get("phase"), "vo_revision": vo.get("revision"),
            })
            pending_vo = None
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
        m = VO_RE.match(s)
        if m:
            pending_vo = {"frame": int(m.group(1)), "history": _ints(m.group(2)), "ready": m.group(3) == "True",
                          "phase": m.group(4), "revision": None if m.group(5) is None else int(m.group(5))}
            continue
        m = END_RE.search(s)
        if m:
            g = m.groups()
            cur["final"] = {"success": float(g[0]), "spl": float(g[1]), "os": float(g[2]), "ne": float(g[3]),
                            "vlm_calls": int(g[4]), "trajectory_calls": int(g[5])}
            cur["line_end"] = lineno
            if cur["episode_id"] in blocks:
                dup += 1
            blocks[cur["episode_id"]] = cur
            cur, pending_vo = None, None
    if cur is not None:
        incomplete += 1
    return blocks, dup, incomplete


def parse_client_logs(pattern: str) -> tuple:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no client log matches {pattern}")
    blocks, dup, incomplete = {}, 0, 0
    for path in paths:
        b, d, i = parse_client_log(path)
        dup += d + len(set(b) & set(blocks))
        incomplete += i
        blocks.update(b)
    return blocks, dup, incomplete, paths


def ended_by(block: dict, steps: int) -> str:
    calls = block["calls"]
    if calls and calls[-1]["kind"] == "stop":
        return "stop"
    if steps >= STEP_CAP:
        return "step_cap"
    return "other"


# ---------------------------------------------------------------------------
# reference path features (scout definitions)
# ---------------------------------------------------------------------------
def wrap_deg(a: float) -> float:
    return (a + 180) % 360 - 180


def heading_deg(p, q) -> float:
    return math.degrees(math.atan2(q[0] - p[0], q[2] - p[2]))


def reference_turns(path, min_segment_m: float = MIN_SEGMENT_M, thr_deg: float = TURN_DEG) -> int:
    segs = [(path[i], path[i + 1]) for i in range(len(path) - 1)
            if math.hypot(path[i + 1][0] - path[i][0], path[i + 1][2] - path[i][2]) > min_segment_m]
    dh = [wrap_deg(heading_deg(*segs[i + 1]) - heading_deg(*segs[i])) for i in range(len(segs) - 1)]
    return sum(abs(a) >= thr_deg for a in dh)


def reference_dy(path) -> float:
    ys = [p[1] for p in path]
    return max(ys) - min(ys)


def parse_house(path) -> tuple:
    """``(panoramas, regions)``: panoramas = [(x, y, z, region)] in house coordinates, regions = {idx: (level, code)}."""
    panos, regions = [], {}
    with open(path) as handle:
        for line in handle:
            t = line.split()
            if not t:
                continue
            if t[0] == "R":
                regions[int(t[1])] = (int(t[2]), t[5])
            if t[0] == "P" and len(t) >= 8 and len(t[1]) == 32:
                panos.append((float(t[5]), float(t[6]), float(t[7]), int(t[3])))
    return panos, regions


def reference_rooms(panos, path) -> list:
    """Region index per reference point (None when no panorama within PANO_MATCH_M)."""
    seq = []
    for (x, y, z) in path:
        mx, my = x, -z  # habitat (x, y, z) -> house (x, -z, y)
        best = None
        for (px, py, pz, ri) in panos:
            if abs(pz - (y + PANO_HEIGHT_M)) > PANO_MAX_DY_M:
                continue
            d = math.hypot(px - mx, py - my)
            if best is None or d < best[0]:
                best = (d, ri)
        seq.append(best[1] if best is not None and best[0] < PANO_MATCH_M else None)
    return seq


def room_runs(rooms) -> list:
    rr = [r for r in rooms if r is not None and r >= 0]
    return [r for i, r in enumerate(rr) if i == 0 or r != rr[i - 1]]


# ---------------------------------------------------------------------------
# selection (pure)
# ---------------------------------------------------------------------------
def order_key(scene_id: str, episode_id: int) -> str:
    return hashlib.sha1(f"{scene_id}:{int(episode_id)}".encode("utf-8")).hexdigest()


def order_pool(pool: list) -> tuple:
    """``(median steps_42, pool sorted by (|steps_42 - median|, sha1 key))``; adds ``sort_key`` to each entry."""
    if not pool:
        return None, []
    median = float(statistics.median(e["steps_42"] for e in pool))
    ranked = [dict(e, sort_key=[abs(e["steps_42"] - median), order_key(e["scene_id"], e["episode_id"])])
              for e in pool]
    ranked.sort(key=lambda e: (e["sort_key"][0], e["sort_key"][1]))
    return median, ranked


def pick_distinct_scenes(ordered: list, top: int = TOP_PER_CATEGORY) -> tuple:
    """First ``top`` entries of ``ordered`` with pairwise distinct scenes; also the entries skipped on the way."""
    picked, skipped, scenes = [], [], set()
    for e in ordered:
        if len(picked) == top:
            break
        if e["scene_id"] in scenes:
            skipped.append(e)
            continue
        scenes.add(e["scene_id"])
        picked.append(e)
    return picked, skipped


def shard_lpt(candidates: list, num_gpus: int) -> list:
    """Longest-processing-time greedy on ``steps_42``; returns ``num_gpus`` lists in (category, rank) order."""
    cat_idx = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    lists, load = [[] for _ in range(num_gpus)], [0] * num_gpus
    for c in sorted(candidates, key=lambda c: (-c["steps_42"], cat_idx[c["category"]], c["rank"])):
        j = min(range(num_gpus), key=lambda k: (load[k], k))
        lists[j].append(c)
        load[j] += c["steps_42"]
    return [sorted(lst, key=lambda c: (cat_idx[c["category"]], c["rank"])) for lst in lists]


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_progress(path) -> dict:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {int(r["episode_id"]): r for r in rows}
    if not (len(by_id) == len(rows) == NUM_EPISODES):
        raise ValueError(f"{path}: {len(rows)} rows, {len(by_id)} unique episode ids (expected {NUM_EPISODES})")
    return by_id


def run_record(row: dict, block: dict) -> dict:
    steps = int(row["steps"])
    return {"success": int(row["success"]), "os": int(row["os"]), "ne": float(row["ne"]), "spl": float(row["spl"]),
            "steps": steps, "vlm_calls": int(row["vlm_calls"]), "trajectory_calls": int(row["trajectory_calls"]),
            "ppa_applied": int(row["ppa_applied_calls"]), "ended_by": ended_by(block, steps)}


def load_inputs(args) -> dict:
    progress = {seed: load_progress(getattr(args, f"progress_{seed}")) for seed in SEEDS}
    eids = sorted(progress["42"])
    for seed in SEEDS:
        if sorted(progress[seed]) != eids:
            raise ValueError(f"seed {seed} progress covers different episodes")
        bad = [e for e in eids if progress[seed][e]["scene_id"] != progress["42"][e]["scene_id"]]
        if bad:
            raise ValueError(f"seed {seed} progress scene ids differ, e.g. episode {bad[0]}")

    logs, log_stats, log_paths = {}, {}, {}
    for seed in SEEDS:
        blocks, dup, incomplete, paths = parse_client_logs(getattr(args, f"logs_{seed}"))
        logs[seed], log_paths[seed] = blocks, paths
        missing = [e for e in eids if e not in blocks]
        mismatch = [e for e in eids if e in blocks and (
            blocks[e]["scene_id"] != progress[seed][e]["scene_id"]
            or blocks[e]["final"]["vlm_calls"] != int(progress[seed][e]["vlm_calls"])
            or len(blocks[e]["calls"]) != int(progress[seed][e]["vlm_calls"])
            or abs(blocks[e]["final"]["success"] - float(progress[seed][e]["success"])) > 1e-6)]
        if missing or mismatch:
            raise ValueError(f"seed {seed} client logs: {len(missing)} episodes missing, {len(mismatch)} disagree "
                             f"with progress.jsonl (e.g. {(missing + mismatch)[:5]})")
        ready_mismatch = [e for e in eids if sum(c["kind"] == "trajectory" and c["vo_ready"] is True
                                                 for c in blocks[e]["calls"]) != int(progress[seed][e]["ppa_applied_calls"])]
        log_stats[seed] = {"blocks": len(blocks), "duplicate_blocks": dup, "incomplete_blocks": incomplete,
                           "ready_trajectory_calls_vs_ppa_applied_mismatch": len(ready_mismatch)}

    dataset = {int(e["episode_id"]): e for e in json.load(gzip.open(args.dataset, "rt"))["episodes"]}
    if sorted(dataset) != eids:
        raise ValueError(f"{args.dataset} covers different episodes than progress.jsonl")
    scenes = sorted({progress["42"][e]["scene_id"] for e in eids})
    house_paths = {sc: Path(args.mp3d_dir) / sc / f"{sc}.house" for sc in scenes}
    houses = {sc: parse_house(p) for sc, p in house_paths.items()}

    static = {}
    for e in eids:
        ep, sc = dataset[e], progress["42"][e]["scene_id"]
        if Path(ep["scene_id"]).stem != sc:
            raise ValueError(f"episode {e}: dataset scene {ep['scene_id']} != progress scene {sc}")
        path = ep["reference_path"]
        panos, regions = houses[sc]
        runs = room_runs(reference_rooms(panos, path))
        static[e] = {"scene_id": sc, "episode_id": e, "geodesic_m": float(ep["info"]["geodesic_distance"]),
                     "dy_m": reference_dy(path), "ref_turns45": reference_turns(path), "n_room_runs": len(runs),
                     "room_codes": [regions[r][1] if r in regions else "?" for r in runs],
                     "instruction": ep["instruction"]["instruction_text"].strip()}
    runs = {seed: {e: run_record(progress[seed][e], logs[seed][e]) for e in eids} for seed in SEEDS}

    input_files = [*[getattr(args, f"progress_{s}") for s in SEEDS], *log_paths["42"], *log_paths["1337"],
                   args.dataset, *[house_paths[sc] for sc in scenes]]
    return {"eids": eids, "static": static, "runs": runs, "logs": logs, "progress": progress,
            "log_stats": log_stats, "inputs": {str(p): sha256_file(p) for p in input_files}}


# ---------------------------------------------------------------------------
# selection over the loaded tables
# ---------------------------------------------------------------------------
def category_pool(data: dict, category: str) -> list:
    spec = CATEGORIES[category]
    pool = []
    for e in data["eids"]:
        s, d42, d1337 = data["static"][e], data["runs"]["42"][e], data["runs"]["1337"][e]
        if not (spec["both"](s, d42) and spec["both"](s, d1337)):
            continue
        if min(d42["ppa_applied"], d1337["ppa_applied"]) < MIN_PPA_APPLIED:
            continue
        if spec["seed42"] is not None and not spec["seed42"](d42):
            continue
        pool.append({"scene_id": s["scene_id"], "episode_id": e, "steps_42": d42["steps"]})
    return pool


def ep_key(scene_id: str, episode_id: int) -> str:
    return f"{scene_id}_{int(episode_id):04d}"


def candidate_record(data: dict, category: str, rank: int, entry: dict) -> dict:
    e = entry["episode_id"]
    s, d42, d1337 = data["static"][e], data["runs"]["42"][e], data["runs"]["1337"][e]
    block = data["logs"]["42"][e]
    return {
        "rank": rank, "category": category, "ep_key": ep_key(s["scene_id"], e), "scene_id": s["scene_id"],
        "episode_id": e, "steps_42": d42["steps"], "steps_1337": d1337["steps"],
        "success_42": d42["success"], "os_42": d42["os"], "ne_42": d42["ne"], "ended_by_42": d42["ended_by"],
        "success_1337": d1337["success"], "os_1337": d1337["os"], "ne_1337": d1337["ne"],
        "ended_by_1337": d1337["ended_by"], "ppa_applied_42": d42["ppa_applied"],
        "ppa_applied_1337": d1337["ppa_applied"], "geodesic_m": s["geodesic_m"], "dy_m": s["dy_m"],
        "ref_turns45": s["ref_turns45"], "n_room_runs": s["n_room_runs"], "room_codes": s["room_codes"],
        "instruction": s["instruction"], "sort_key": entry["sort_key"],
        "eval_log": {"client_log": block["client_log"], "line_start": block["line_start"],
                     "line_end": block["line_end"]},
    }


def select(data: dict) -> dict:
    categories = {}
    for c in CATEGORY_ORDER:
        pool = category_pool(data, c)
        median, ordered = order_pool(pool)
        picked, skipped = pick_distinct_scenes(ordered)
        categories[c] = {
            "name": CATEGORIES[c]["name"], "predicate": CATEGORIES[c]["text"], "pool_size": len(pool),
            "pool_scenes": len({e["scene_id"] for e in pool}), "median_steps_42": median,
            "ordered_pool_head": [{"ep_key": ep_key(e["scene_id"], e["episode_id"]), "steps_42": e["steps_42"],
                                   "sort_key": e["sort_key"]} for e in ordered[:POOL_HEAD]],
            "skipped_same_scene": [ep_key(e["scene_id"], e["episode_id"]) for e in skipped],
            "candidates": [candidate_record(data, c, rank, e) for rank, e in enumerate(picked)],
        }
    return categories


def eval_log_reference(data: dict, cand: dict) -> dict:
    e = cand["episode_id"]
    block, row = data["logs"]["42"][e], data["progress"]["42"][e]
    return {"schema": EVAL_LOG_REF_SCHEMA, "ep_key": cand["ep_key"], "scene_id": cand["scene_id"],
            "episode_id": e, "seed": 42, "client_log": block["client_log"], "line_start": block["line_start"],
            "line_end": block["line_end"],
            "notes": "call_index = 0-based order of 'step_id: ..., RPC kind=' lines in the block (= system2_call_index); "
                     "vo_* from the [amb3r-vo] line printed before the call; vlm_output stripped of surrounding "
                     "whitespace; traj_goal = System1 mean endpoint (x forward, y left, m, 2 decimals); "
                     "final from the '=> success' line, final.steps and final.ended_by from progress.jsonl / this module",
            "calls": block["calls"],
            "final": dict(block["final"], steps=int(row["steps"]), ended_by=data["runs"]["42"][e]["ended_by"])}


def self_check(data: dict) -> dict:
    rows = []

    def add(check, expected, got):
        rows.append({"check": check, "expected": expected, "got": got, "ok": expected == got})

    for seed in SEEDS:
        st = data["log_stats"][seed]
        add(f"seed {seed}: client-log blocks", NUM_EPISODES, st["blocks"])
        add(f"seed {seed}: duplicate blocks", 0, st["duplicate_blocks"])
        add(f"seed {seed}: ready trajectory calls != ppa_applied_calls", 0,
            st["ready_trajectory_calls_vs_ppa_applied_mismatch"])
        runs = data["runs"][seed]
        for kind in ("stop", "step_cap"):
            add(f"seed {seed}: ended_by={kind}", SCOUT_TERMINATION[seed][kind],
                sum(runs[e]["ended_by"] == kind for e in data["eids"]))
        add(f"seed {seed}: ended_by=stop at steps >= {STEP_CAP}", 0,
            sum(runs[e]["ended_by"] == "stop" and runs[e]["steps"] >= STEP_CAP for e in data["eids"]))
        add(f"seed {seed}: ppa_applied >= {MIN_PPA_APPLIED}", SCOUT_TERMINATION[seed]["ppa>=5"],
            sum(runs[e]["ppa_applied"] >= MIN_PPA_APPLIED for e in data["eids"]))
    for name, (pred, expected) in SCOUT_COUNTS.items():
        s, r42, r1337 = data["static"], data["runs"]["42"], data["runs"]["1337"]
        a = {e for e in data["eids"] if pred(s[e], r42[e])}
        b = {e for e in data["eids"] if pred(s[e], r1337[e])}
        both = a & b
        act = {e for e in both if min(r42[e]["ppa_applied"], r1337[e]["ppa_applied"]) >= MIN_PPA_APPLIED}
        add(f"scout {name} (42 | 1337 | both | both & ppa>=5)", list(expected), [len(a), len(b), len(both), len(act)])
    return {"ok": all(r["ok"] for r in rows), "rows": rows}


def resolve_git_sha(repo: Path) -> str:
    """EXP19_GIT_SHA, else <repo>/.exp19_git_sha (staged archives have no .git)."""
    sha = os.environ.get("EXP19_GIT_SHA", "").strip()
    if sha:
        return sha
    marker = repo / ".exp19_git_sha"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    return "unknown"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def print_self_check(check: dict) -> None:
    for r in check["rows"]:
        print(f"  [{'ok' if r['ok'] else 'MISMATCH'}] {r['check']}: expected {r['expected']} got {r['got']}")
    print(f"self-check: {'PASSED' if check['ok'] else 'FAILED'}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=EXP_ROOT / "cases")
    p.add_argument("--num-gpus", type=int, default=MAX_GPUS)
    p.add_argument("--self-check", action="store_true",
                   help="compare counts with the scout and exit (1 on mismatch) without writing")
    p.add_argument("--progress-42", type=Path, default=DEFAULT_PROGRESS["42"])
    p.add_argument("--progress-1337", type=Path, default=DEFAULT_PROGRESS["1337"])
    p.add_argument("--logs-42", default=DEFAULT_LOG_GLOBS["42"], help="glob of the seed-42 client logs")
    p.add_argument("--logs-1337", default=DEFAULT_LOG_GLOBS["1337"], help="glob of the seed-1337 client logs")
    p.add_argument("--dataset", type=Path, default=VAL_UNSEEN)
    p.add_argument("--mp3d-dir", type=Path, default=MP3D_SCENES)
    args = p.parse_args(argv)
    if not 1 <= args.num_gpus <= MAX_GPUS:
        p.error(f"--num-gpus must be in [1, {MAX_GPUS}] (dev-machine limit), got {args.num_gpus}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    data = load_inputs(args)
    check = self_check(data)
    print_self_check(check)
    if args.self_check:
        return 0 if check["ok"] else 1
    if not check["ok"]:
        print("refusing to write: self-check failed", file=sys.stderr)
        return 1

    categories = select(data)
    cands = [c for cat in CATEGORY_ORDER for c in categories[cat]["candidates"]]
    keys = [c["ep_key"] for c in cands]
    if len(set(keys)) != len(keys):
        dups = sorted({k for k in keys if keys.count(k) > 1})
        print(f"refusing to write: an episode is a candidate in two categories: {dups}", file=sys.stderr)
        return 1
    shards = shard_lpt(cands, args.num_gpus)

    out = args.out_dir
    dataset_sha = data["inputs"][str(args.dataset)]
    lists_dir = out / "episode_lists"
    for j, shard in enumerate(shards):
        write_json(lists_dir / f"gpu{j}.json", {
            "cohort_name": f"exp19_behavior_viz_gpu{j}_of_{args.num_gpus}", "dataset_sha256": dataset_sha,
            "num_shards": args.num_gpus, "shard_index": j,
            "episodes": [{"scene_id": c["scene_id"], "episode_id": c["episode_id"]} for c in shard]})
    for stale in sorted(lists_dir.glob("gpu*.json")):
        m = re.fullmatch(r"gpu(\d+)\.json", stale.name)
        if m and int(m.group(1)) >= args.num_gpus:
            stale.unlink()
            print(f"removed stale {stale}")
    for c in cands:
        write_json(out / "eval_log_reference" / f"{c['ep_key']}.json", eval_log_reference(data, c))

    result = {
        "schema": CANDIDATES_SCHEMA, "git_sha": resolve_git_sha(SOURCE_ROOT), "inputs": data["inputs"],
        "definitions": DEFINITIONS,
        "constants": {"min_ppa_applied": MIN_PPA_APPLIED, "top_per_category": TOP_PER_CATEGORY,
                      "step_cap": STEP_CAP, "category_order": list(CATEGORY_ORDER)},
        "log_stats": data["log_stats"],
        "categories": categories,
        "episode_lists": {f"gpu{j}": [c["ep_key"] for c in shard] for j, shard in enumerate(shards)},
        "episode_list_steps_42": {f"gpu{j}": sum(c["steps_42"] for c in shard) for j, shard in enumerate(shards)},
        "self_check": check,
    }
    write_json(out / "candidates.json", result)

    for cat in CATEGORY_ORDER:
        cd = categories[cat]
        print(f"{cat} ({cd['name']}): pool {cd['pool_size']} in {cd['pool_scenes']} scenes, "
              f"median steps_42 {cd['median_steps_42']}")
        for c in cd["candidates"]:
            print(f"   #{c['rank']} {c['ep_key']} steps42={c['steps_42']} steps1337={c['steps_1337']} "
                  f"ne42={c['ne_42']:.2f} ppa={c['ppa_applied_42']}/{c['ppa_applied_1337']} "
                  f"|d|={c['sort_key'][0]:g}")
    for j, shard in enumerate(shards):
        print(f"gpu{j}: {[c['ep_key'] for c in shard]} (steps_42 {sum(c['steps_42'] for c in shard)})")
    print(f"wrote {out / 'candidates.json'}, {len(shards)} episode lists, {len(cands)} eval-log references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

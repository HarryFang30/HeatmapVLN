"""EXP-33 analysis: the pre-registered comparisons over the 8 runs (docs/experiments/README.md, EXP-33).

Reads each run's eight ``workers/shard_0N/progress.json``, refuses a run unless it holds exactly the
500 subset episodes, once each, every row with the run's own ``camera_mount`` (none for V0), AMB3R
poses, seed 42 and the bridge applied somewhere; then pairs episodes and bootstraps (2000 draws,
``paired_closed_loop_bootstrap.compare``):

1. viewpoint cost: each arm's Vk against its V0, with the deployment reading for A0;
2. the cognition head: A0 against A1 at every viewpoint, plus the interaction
   (A0_Vk - A0_V0) - (A1_Vk - A1_V0), reported only;
3. A0 V1 against A0 V2;
and the report-only readouts (steps, step-cap hits, bridge-applied calls, SR per geodesic quartile,
A0 V0 against the EXP-21 default point).  Writes ``exp33_results.json`` and ``exp33_results.md``.

    python scripts/exp33/analyze.py --runs /workspace/eval_runs --subset /workspace/exp21/subset500 \
        --cohorts <locked plan>/cohorts --reference <A0 seed 42 merged/progress.jsonl> --out <dir>
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.tools.paired_closed_loop_bootstrap import METRICS, bootstrap_ci, compare, mean, scene_name  # noqa: E402

ARMS = ("a0", "a1")
VIEWS = ("v0", "v1", "v2", "v3")
MOUNTS = {
    "v0": None,
    "v1": {"height_m": 0.6, "pitch_deg": -15.0, "lookdown": "same_frame"},
    "v2": {"height_m": 0.7, "pitch_deg": -15.0, "lookdown": "same_frame"},
    "v3": {"height_m": 0.4, "pitch_deg": 0.0, "lookdown": "same_frame"},
}
NUM_SHARDS = 8
STEP_CAP = 500
DRAWS = 2000

Key = tuple  # (scene, episode id as int)


def run_dir(runs: Path, arm: str, view: str) -> Path:
    return runs / f"exp33_{arm}_{view}_seed42_4090"


def subset_keys(subset: Path) -> set[Key]:
    keys: set[Key] = set()
    for shard in range(NUM_SHARDS):
        listed = json.loads((subset / f"shard_0{shard}.json").read_text(encoding="utf-8"))
        keys |= {(str(e["scene_id"]), int(e["episode_id"])) for e in listed["episodes"]}
    return keys


def load_run(path: Path, view: str, want: set[Key]) -> dict[Key, dict]:
    """The run's rows by episode, or SystemExit naming what makes it unusable (criteria 4)."""
    rows: dict[Key, dict] = {}
    for shard in range(NUM_SHARDS):
        progress = path / "workers" / f"shard_0{shard}" / "progress.json"
        for line in progress.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["scene_id"]), int(row["episode_id"]))
            if key in rows:
                raise SystemExit(f"{path}: {key} recorded twice")
            rows[key] = row
    problems = []
    if set(rows) != want:
        problems.append(f"{len(want - set(rows))} subset episodes missing, {len(set(rows) - want)} extra")
    if any(row.get("camera_mount") != MOUNTS[view] for row in rows.values()):
        problems.append(f"a row's camera_mount is not {MOUNTS[view]}")
    if any(row.get("history_pose_source") != "amb3r_vo_da3" for row in rows.values()):
        problems.append("a row without AMB3R poses")
    if any(int(row.get("rpc_protocol_seed", -1)) != 42 for row in rows.values()):
        problems.append("a row with another protocol seed")
    if sum(int(row.get("ppa_applied_calls", 0)) for row in rows.values()) <= 0:
        problems.append("the bridge was never applied")
    if problems:
        raise SystemExit(f"{path}: " + "; ".join(problems))
    return rows


def paired(treatment: dict[Key, dict], control: dict[Key, dict], seed: int) -> dict:
    keys = sorted(set(treatment) & set(control), key=lambda k: (k[1], k[0]))
    ids = [(0, *key) for key in keys]
    t = {(0, *key): treatment[key] for key in keys}
    c = {(0, *key): control[key] for key in keys}
    return compare(t, c, ids, random.Random(seed), DRAWS)


def interaction(a0k: dict, a0v0: dict, a1k: dict, a1v0: dict, seed: int) -> dict:
    keys = sorted(set(a0k) & set(a0v0) & set(a1k) & set(a1v0), key=lambda k: (k[1], k[0]))
    rng = random.Random(seed)
    out: dict = {"n": len(keys)}
    for metric in METRICS:
        diffs = [
            (float(a0k[k][metric]) - float(a0v0[k][metric])) - (float(a1k[k][metric]) - float(a1v0[k][metric]))
            for k in keys
        ]
        out[metric] = {"diff": mean(diffs), "ci95": list(bootstrap_ci(diffs, rng, DRAWS))}
    return out


def direction(entry: dict) -> str:
    lo, hi = entry["success"]["ci95"]
    return "drops" if hi < 0 else ("gains" if lo > 0 else "not measurable")


def deployment_reading(entry: dict) -> str:
    """A0's ΔSR point estimate against the pre-registered lines (criteria 1)."""
    d = 100 * entry["success"]["diff"]
    if d >= -5:
        return "usable as is"
    if d <= -10 and entry["success"]["ci95"][1] < 0:
        return "large loss"
    return "moderate loss"


def head_reading(entry: dict) -> str:
    lo, hi = entry["success"]["ci95"]
    return "drags" if hi < 0 else ("helps" if lo > 0 else "not measurable")


def descriptives(rows: dict[Key, dict], geodesic: dict[Key, float], edges: list[float]) -> dict:
    values = list(rows.values())
    quartiles = []
    for q in range(4):
        lo, hi = edges[q], edges[q + 1]
        members = [r for k, r in rows.items() if lo <= geodesic[k] < hi or (q == 3 and geodesic[k] == hi)]
        quartiles.append({"geodesic_m": [lo, hi], "n": len(members),
                          "sr": 100 * mean([float(r["success"]) for r in members]) if members else None})
    return {
        "n": len(values),
        "sr": 100 * mean([float(r["success"]) for r in values]),
        "spl": 100 * mean([float(r["spl"]) for r in values]),
        "os": 100 * mean([float(r["os"]) for r in values]),
        "ne": mean([float(r["ne"]) for r in values]),
        "mean_steps": mean([float(r["steps"]) for r in values]),
        "step_cap_hits": sum(int(r["steps"]) >= STEP_CAP for r in values),
        "mean_ppa_applied_calls": mean([float(r.get("ppa_applied_calls", 0)) for r in values]),
        "sr_by_geodesic_quartile": quartiles,
    }


def load_geodesic(cohorts: Path) -> dict[Key, float]:
    out: dict[Key, float] = {}
    for shard in range(NUM_SHARDS):
        with gzip.open(cohorts / f"dataset_shard_0{shard}.json.gz", "rt", encoding="utf-8") as handle:
            for episode in json.load(handle)["episodes"]:
                out[(scene_name(episode["scene_id"]), int(episode["episode_id"]))] = float(episode["info"]["geodesic_distance"])
    return out


def fmt(entry: dict, metric: str) -> str:
    scale, unit = (1.0, " m") if metric == "ne" else (100.0, "")
    lo, hi = entry["ci95"]
    head = f"{entry['treatment'] * scale:.2f}{unit} vs {entry['control'] * scale:.2f}{unit} → " if "treatment" in entry else ""
    return f"{head}{entry['diff'] * scale:+.2f} [{lo * scale:+.2f}, {hi * scale:+.2f}]"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None, help="EXP-21 default point: A0 seed 42 merged/progress.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=None, help="arm_view runs to load (partial readout; verdicts need all 8)")
    args = parser.parse_args(argv)

    want = subset_keys(args.subset)
    edges = json.loads((args.subset / "manifest.json").read_text(encoding="utf-8"))["quartile_edges_m"]
    geodesic = load_geodesic(args.cohorts)
    runs = {}
    for arm in ARMS:
        for view in VIEWS:
            name = f"{arm}_{view}"
            if args.only is None or name in args.only:
                runs[name] = load_run(run_dir(args.runs, arm, view), view, want)

    result: dict = {"runs": {name: descriptives(rows, geodesic, edges) for name, rows in runs.items()},
                    "viewpoint_cost": {}, "head": {}, "interaction": {}}
    seed = 0
    for arm in ARMS:
        for view in VIEWS[1:]:
            if f"{arm}_{view}" in runs and f"{arm}_v0" in runs:
                seed += 1
                entry = paired(runs[f"{arm}_{view}"], runs[f"{arm}_v0"], seed)
                entry["direction"] = direction(entry)
                if arm == "a0":
                    entry["deployment_reading"] = deployment_reading(entry)
                result["viewpoint_cost"][f"{arm}_{view}"] = entry
    for view in VIEWS:
        if f"a0_{view}" in runs and f"a1_{view}" in runs:
            seed += 1
            entry = paired(runs[f"a0_{view}"], runs[f"a1_{view}"], seed)
            entry["reading"] = head_reading(entry)
            result["head"][view] = entry
    for view in VIEWS[1:]:
        names = [f"a0_{view}", "a0_v0", f"a1_{view}", "a1_v0"]
        if all(name in runs for name in names):
            seed += 1
            result["interaction"][view] = interaction(*(runs[name] for name in names), seed)
    if "a0_v1" in runs and "a0_v2" in runs:
        seed += 1
        entry = paired(runs["a0_v1"], runs["a0_v2"], seed)
        entry["reading"] = "indistinguishable" if direction(entry) == "not measurable" else "differ"
        result["v1_vs_v2"] = entry
    if args.reference is not None and "a0_v0" in runs:
        rows = {}
        for line in args.reference.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                key = (str(row["scene_id"]), int(row["episode_id"]))
                if key in want:
                    rows[key] = row
        if set(rows) != want:
            raise SystemExit(f"{args.reference}: holds {len(rows)} of the {len(want)} subset episodes")
        seed += 1
        result["a0_v0_vs_exp21_default"] = paired(runs["a0_v0"], rows, seed)

    if len(runs) == 8:
        cost = result["viewpoint_cost"]
        readings = {view: cost[f"a0_{view}"]["deployment_reading"] for view in VIEWS[1:]}
        if readings["v1"] == readings["v2"] == "usable as is" and readings["v3"] == "large loss":
            h1 = "supported"
        elif "large loss" in (readings["v1"], readings["v2"]) or direction(cost["a0_v3"]) == "not measurable":
            h1 = "rejected"
        else:
            h1 = "partly supported"
        h2 = "rejected" if any(entry["reading"] == "drags" for entry in result["head"].values()) else "supported"
        h3 = "supported" if result["v1_vs_v2"]["reading"] == "indistinguishable" else "rejected"
        result["verdicts"] = {"H1": h1, "H2": h2, "H3": h3}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "exp33_results.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["| run | SR | SPL | NE (m) | mean steps | at 500 steps | bridge calls/ep |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, d in result["runs"].items():
        lines.append(f"| {name} | {d['sr']:.2f} | {d['spl']:.2f} | {d['ne']:.3f} | {d['mean_steps']:.1f} | "
                     f"{d['step_cap_hits']} | {d['mean_ppa_applied_calls']:.1f} |")
    for title, table in (("viewpoint cost (Vk vs V0, same arm)", result["viewpoint_cost"]),
                         ("A0 vs A1", result["head"])):
        lines += ["", f"**{title}**", "", "| | ΔSR | ΔSPL | ΔNE (m) | reading |", "|---|---|---|---|---|"]
        for name, entry in table.items():
            reading = entry.get("deployment_reading", entry.get("reading", entry.get("direction")))
            lines.append(f"| {name} | {fmt(entry['success'], 'success')} | {fmt(entry['spl'], 'spl')} | "
                         f"{fmt(entry['ne'], 'ne')} | {reading} ({entry.get('direction', '')}) |")
    if result["interaction"]:
        lines += ["", "**interaction (A0_Vk − A0_V0) − (A1_Vk − A1_V0), reported only**", ""]
        lines += [f"- {view}: ΔSR {fmt(entry['success'], 'success')}" for view, entry in result["interaction"].items()]
    if "v1_vs_v2" in result:
        lines += ["", f"**A0 V1 vs V2**: ΔSR {fmt(result['v1_vs_v2']['success'], 'success')} → {result['v1_vs_v2']['reading']}"]
    if "a0_v0_vs_exp21_default" in result:
        lines += ["", f"**A0 V0 (FA2) vs EXP-21 default point (sdpa)**: ΔSR {fmt(result['a0_v0_vs_exp21_default']['success'], 'success')}"]
    if "verdicts" in result:
        lines += ["", f"**verdicts**: {json.dumps(result['verdicts'])}"]
    (args.out / "exp33_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

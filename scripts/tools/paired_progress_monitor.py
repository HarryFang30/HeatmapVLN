#!/usr/bin/env python3
"""Paired SR/SPL of a running evaluation against a finished reference, so far.

EXP-23 replaced the subset500 canary with a stop rule inside the full run (ledger,
2026-10-13): once at least ``--min-episodes`` episodes are paired, a candidate whose
SR is ``--abort-delta`` points or more below the reference on those same episodes is
stopped.  This reads the running evaluation's per-shard progress files (JSONL rows
with ``scene_id``, ``episode_id``, ``success``, ``spl``) and the reference's merged
``progress.jsonl``, pairs them by episode, and prints the paired differences with a
normal-approximation 95% interval.

A candidate run with ``--stop_at_step_budget`` (EXP-24) is compared twice:

* ``sr`` / ``spl`` -- the standard protocol, which the stop rule and EXP-23's H1 use.
  Exactly derivable from the switched-on run: an episode that ended with the budget's
  STOP (``step_budget_stop``) would have run out of steps without one, and Habitat only
  counts an episode whose STOP was called, so it scores 0.
* ``sr_step_budget`` -- the switched-on numbers against the reference's estimate under
  the same rule (a reference episode that used the whole budget counts as a success if
  it ended within the success distance; ±1 action at the boundary, EXP-24).  Report only.

Exit status: 0 = keep going, 3 = the stop rule fired, 2 = nothing to pair yet.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path


def read_rows(paths) -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows[(str(row["scene_id"]), int(row["episode_id"]))] = row
    return rows


def standard(rows: dict) -> dict:
    """Rows as the standard protocol scores them: a step-budget STOP is no STOP."""
    out = {}
    for key, row in rows.items():
        if row.get("step_budget_stop"):
            row = dict(row, success=0.0, spl=0.0)
        out[key] = row
    return out


def with_step_budget(rows: dict, *, max_steps: int, success_distance: float) -> dict:
    """A reference run's rows as if the budget's last action had been STOP (estimate)."""
    out = {}
    for key, row in rows.items():
        if int(row.get("steps", 0)) >= max_steps and float(row.get("ne", float("inf"))) < success_distance:
            row = dict(row, success=1.0)
        out[key] = row
    return out


def paired(candidate: dict, reference: dict, field: str) -> dict:
    keys = sorted(set(candidate) & set(reference))
    diffs = [float(candidate[k][field]) - float(reference[k][field]) for k in keys]
    n = len(diffs)
    if n == 0:
        return {"n": 0}
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    half = 1.96 * math.sqrt(var / n) if n > 1 else float("nan")
    return {
        "n": n,
        "candidate": round(100 * sum(float(candidate[k][field]) for k in keys) / n, 2),
        "reference": round(100 * sum(float(reference[k][field]) for k in keys) / n, 2),
        "delta_pt": round(100 * mean, 2),
        "ci95_pt": [round(100 * (mean - half), 2), round(100 * (mean + half), 2)],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True, help="glob of the running eval's progress files (JSONL)")
    parser.add_argument("--reference", required=True, help="the reference run's merged progress.jsonl")
    parser.add_argument("--min-episodes", type=int, default=600)
    parser.add_argument("--abort-delta", type=float, default=-3.0, help="SR points; stop at or below this")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--success-distance", type=float, default=3.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    raw = read_rows(sorted(glob.glob(args.candidate)))
    reference = read_rows([args.reference])
    candidate = standard(raw)
    report = {
        "candidate_rows": len(raw),
        "sr": paired(candidate, reference, "success"),
        "spl": paired(candidate, reference, "spl"),
        "rule": {"min_episodes": args.min_episodes, "abort_delta_pt": args.abort_delta, "on": "standard protocol"},
    }
    if any(row.get("stop_at_step_budget") for row in raw.values()):
        budget_ref = with_step_budget(reference, max_steps=args.max_steps, success_distance=args.success_distance)
        report["step_budget_stops"] = sum(bool(row.get("step_budget_stop")) for row in raw.values())
        report["sr_step_budget"] = paired(raw, budget_ref, "success")
        report["spl_step_budget"] = paired(raw, reference, "spl")
    sr = report["sr"]
    report["abort"] = bool(sr.get("n", 0) >= args.min_episodes and sr["delta_pt"] <= args.abort_delta)
    print(json.dumps(report, sort_keys=True))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if sr.get("n", 0) == 0:
        return 2
    return 3 if report["abort"] else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""EXP-12 D3b: how much of the closed-loop failure mass is excess-travel wandering.

The completed evaluations kept only ``progress.jsonl`` and ``result.json`` --
``TrajectoryStepRecorder`` runs solely in the DAgger collection path -- so no
per-step positions exist for val_unseen and revisits cannot be counted directly.
This tool therefore computes the pre-registered proxy: a failed episode counts as
"wandering" when its primitive-step count exceeds ``--excess-ratio`` times the
steps a straight geodesic run would need (``geodesic / --step-size``).

The proxy is an **upper bound** on revisit-type failure and must never be read
back as a revisit rate: an episode can burn steps on turns, recovery probes, or
a wrong corridor without ever revisiting anything.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_geodesic(path: Path) -> dict[str, float]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        str(episode["episode_id"]): float(episode["info"]["geodesic_distance"])
        for episode in payload["episodes"]
    }


def summarise(
    progress_path: Path,
    geodesic: dict[str, float],
    *,
    step_size: float,
    excess_ratio: float,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines() if line]
    missing = 0
    failures: list[dict[str, Any]] = []
    successes = 0
    max_steps = max(int(row["steps"]) for row in rows)
    for row in rows:
        key = str(row["episode_id"])
        if key not in geodesic:
            missing += 1
            continue
        if float(row["success"]) > 0.5:
            successes += 1
            continue
        minimum_steps = geodesic[key] / step_size
        failures.append(
            {
                "steps": int(row["steps"]),
                "minimum_steps": minimum_steps,
                "ratio": int(row["steps"]) / minimum_steps if minimum_steps > 0 else float("inf"),
                "timeout": int(row["steps"]) >= max_steps,
                "os": float(row["os"]),
                "ne": float(row["ne"]),
            }
        )
    ratios = np.asarray([f["ratio"] for f in failures], dtype=np.float64)
    wandering = ratios >= excess_ratio
    return {
        "episodes": len(rows),
        "episodes_missing_geodesic": missing,
        "successes": successes,
        "failures": len(failures),
        "max_steps_observed": max_steps,
        "wandering_failures": int(wandering.sum()),
        "wandering_fail_frac": float(wandering.mean()) if len(ratios) else None,
        "wandering_frac_of_all_episodes": float(wandering.sum() / len(rows)) if rows else None,
        "failure_step_ratio_median": float(np.median(ratios)) if len(ratios) else None,
        "failure_timeout_frac": float(np.mean([f["timeout"] for f in failures])) if failures else None,
        "wandering_and_timeout": int(sum(1 for f, w in zip(failures, wandering) if w and f["timeout"])),
        "wandering_and_oracle_success": int(
            sum(1 for f, w in zip(failures, wandering) if w and f["os"] > 0.5)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, action="append", required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="val_unseen.json.gz for geodesic distances")
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--excess-ratio", type=float, default=3.0)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    geodesic = load_geodesic(args.dataset)
    report = {
        "schema": "heatmapvln-exp12-wandering-failures-v1",
        "inputs": {
            "dataset": str(args.dataset),
            "step_size_m": args.step_size,
            "excess_ratio": args.excess_ratio,
            "episodes_in_dataset": len(geodesic),
        },
        "arms": {
            str(path.parent.parent.name): summarise(
                path, geodesic, step_size=args.step_size, excess_ratio=args.excess_ratio
            )
            for path in args.progress
        },
    }
    fractions = [
        arm["wandering_fail_frac"]
        for arm in report["arms"].values()
        if arm["wandering_fail_frac"] is not None
    ]
    report["wandering_fail_frac_min"] = min(fractions) if fractions else None
    report["wandering_fail_frac_max"] = max(fractions) if fractions else None

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()

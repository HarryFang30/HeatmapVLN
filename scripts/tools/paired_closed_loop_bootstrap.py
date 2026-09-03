#!/usr/bin/env python3
"""Paired bootstrap of two closed-loop R2R evaluations on the same episodes.

Both inputs are ``merged/progress.jsonl`` files (one row per episode with
``episode_id``, ``success``, ``spl``, ``os``, ``ne``).  Episodes are paired by
id, so the per-episode difference cancels episode difficulty; the bootstrap
resamples episodes with replacement.  Optional strata come from the episode
geodesic distance in the Habitat dataset json.gz (``--dataset``) and are
declared in advance via ``--geodesic-min``.

The ledger cites these numbers, so the output is written as JSON next to a
markdown table and the script is deterministic (``--bootstrap-seed``).
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

METRICS = ("success", "spl", "os", "ne")


def load_progress(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows[str(row["episode_id"])] = row
    return rows


def load_geodesic(path: Path) -> dict[str, float]:
    with gzip.open(path) as handle:
        dataset = json.load(handle)
    return {
        str(episode["episode_id"]): float(episode["info"]["geodesic_distance"])
        for episode in dataset["episodes"]
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def bootstrap_ci(diffs: list[float], rng: random.Random, draws: int) -> tuple[float, float]:
    n = len(diffs)
    means = sorted(mean([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def compare(
    treatment: dict[str, dict],
    control: dict[str, dict],
    ids: list[str],
    rng: random.Random,
    draws: int,
) -> dict[str, dict]:
    out: dict[str, dict] = {"n": len(ids)}
    for metric in METRICS:
        a = [float(treatment[i][metric]) for i in ids]
        b = [float(control[i][metric]) for i in ids]
        diffs = [x - y for x, y in zip(a, b, strict=True)]
        lo, hi = bootstrap_ci(diffs, rng, draws)
        out[metric] = {
            "treatment": mean(a),
            "control": mean(b),
            "diff": mean(diffs),
            "ci95": [lo, hi],
        }
    return out


def fmt(entry: dict, metric: str) -> str:
    scale = 1.0 if metric == "ne" else 100.0
    unit = " m" if metric == "ne" else ""
    lo, hi = entry["ci95"]
    return (
        f"{entry['treatment'] * scale:.2f}{unit} vs {entry['control'] * scale:.2f}{unit} "
        f"→ {entry['diff'] * scale:+.2f} [{lo * scale:+.2f}, {hi * scale:+.2f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", type=Path, required=True, help="merged/progress.jsonl of the treatment arm")
    parser.add_argument("--control", type=Path, required=True, help="merged/progress.jsonl of the control arm")
    parser.add_argument("--dataset", type=Path, default=None, help="val_unseen.json.gz for geodesic strata")
    parser.add_argument("--geodesic-min", type=float, action="append", default=[], help="pre-declared stratum: geodesic distance >= this many metres")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--label", default="treatment_vs_control")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    treatment = load_progress(args.treatment)
    control = load_progress(args.control)
    ids = sorted(set(treatment) & set(control), key=int)
    if not ids:
        raise SystemExit("no paired episodes")
    missing = (set(treatment) | set(control)) - set(ids)
    rng = random.Random(args.bootstrap_seed)

    report = {
        "label": args.label,
        "treatment": str(args.treatment),
        "control": str(args.control),
        "paired_episodes": len(ids),
        "unpaired_episodes": len(missing),
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
        "strata": {"all": compare(treatment, control, ids, rng, args.bootstrap_draws)},
    }
    if args.geodesic_min:
        if args.dataset is None:
            raise SystemExit("--geodesic-min requires --dataset")
        geodesic = load_geodesic(args.dataset)
        for threshold in args.geodesic_min:
            subset = [i for i in ids if geodesic[i] >= threshold]
            report["strata"][f"geodesic_ge_{threshold:g}m"] = compare(
                treatment, control, subset, rng, args.bootstrap_draws
            )

    both = sum(1 for i in ids if treatment[i]["success"] and control[i]["success"])
    only_t = sum(1 for i in ids if treatment[i]["success"] and not control[i]["success"])
    only_c = sum(1 for i in ids if control[i]["success"] and not treatment[i]["success"])
    report["success_overlap"] = {"both": both, "treatment_only": only_t, "control_only": only_c}

    print(f"# {args.label}: {len(ids)} paired episodes ({len(missing)} unpaired dropped)")
    print("| stratum | n | SR | SPL | OS | NE |")
    print("|---|---|---|---|---|---|")
    for name, entry in report["strata"].items():
        cells = " | ".join(fmt(entry[m], m) for m in METRICS)
        print(f"| {name} | {entry['n']} | {cells} |")
    print(f"success overlap: both={both} treatment_only={only_t} control_only={only_c}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()

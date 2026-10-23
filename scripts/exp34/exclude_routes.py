#!/usr/bin/env python3
"""Drop already-collected routes from an R2R-CE-format split before building a new cohort.

EXP-34's second EnvDrop collection must not revisit the 2,500 routes of EXP-25's cohort
(1,771 of them are already in the training stream), and must not touch the held-out
scenes.  ``build_r2r_train_cohorts.py`` can exclude scenes but not routes, so this writes
a filtered copy of the split (every other top-level key unchanged) for it to read:

    /opt/conda/bin/python scripts/exp34/exclude_routes.py \\
        --dataset /workspace/R2R_VLNCE_v1-3_preprocessed/envdrop/envdrop.json.gz \\
        --exclude-cohort /workspace/exp23/envdrop1/cohorts \\
        --exclude-scenes 17DRP5sb8fy,5LpN3gDmAk7,gZ6f7yhEvPG,kEZ7cmS4wCh,r47D5H71a5s,VzqfbhrpDEA \\
        --output /workspace/exp34/collect_envdrop2/envdrop_minus_exp25.json.gz

A route is ``(scene, trajectory_id)``, as in ``build_r2r_train_cohorts.py``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath


def scene_name(raw_scene_id: str) -> str:
    return PurePosixPath(str(raw_scene_id)).parent.name


def routes_in_cohort(cohort_dir: Path) -> set[tuple[str, int]]:
    routes: set[tuple[str, int]] = set()
    shards = sorted(cohort_dir.glob("dataset_shard_*.json.gz"))
    if not shards:
        raise SystemExit(f"no dataset_shard_*.json.gz under {cohort_dir}")
    for path in shards:
        with gzip.open(path, "rt") as handle:
            for episode in json.load(handle)["episodes"]:
                routes.add((scene_name(episode["scene_id"]), int(episode["trajectory_id"])))
    return routes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exclude-cohort", action="append", default=[], help="cohort directory; repeatable")
    parser.add_argument("--exclude-scenes", default="", help="comma-separated scene names")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    with gzip.open(args.dataset, "rt") as handle:
        data = json.load(handle)
    excluded_routes: set[tuple[str, int]] = set()
    for cohort in args.exclude_cohort:
        excluded_routes |= routes_in_cohort(Path(cohort))
    excluded_scenes = {s for s in args.exclude_scenes.split(",") if s}
    known_scenes = {scene_name(e["scene_id"]) for e in data["episodes"]}
    unknown = excluded_scenes - known_scenes
    if unknown:
        raise SystemExit(f"unknown scenes: {sorted(unknown)}")

    kept = [
        e
        for e in data["episodes"]
        if scene_name(e["scene_id"]) not in excluded_scenes
        and (scene_name(e["scene_id"]), int(e["trajectory_id"])) not in excluded_routes
    ]
    out = dict(data)
    out["episodes"] = kept
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".part")
    with gzip.open(tmp, "wt") as handle:
        json.dump(out, handle)
    tmp.replace(output)
    print(json.dumps({
        "dataset": args.dataset,
        "episodes_in": len(data["episodes"]),
        "episodes_kept": len(kept),
        "routes_excluded": len(excluded_routes),
        "scenes_excluded": sorted(excluded_scenes),
        "output": str(output),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

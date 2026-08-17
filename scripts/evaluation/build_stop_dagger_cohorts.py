#!/usr/bin/env python3
"""Build disjoint, scene-balanced R2R train cohorts for STOP DAgger collection."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SELECTIONS = ("longest", "q75", "median", "q25", "shortest")
_SELECTION_QUANTILES = {"q75": 0.75, "median": 0.5, "q25": 0.25}


def _scene_id(raw_scene_id: str) -> str:
    path = Path(str(raw_scene_id))
    return path.parent.name if path.suffix else path.name


def _stable_tie_break(scene_id: str, episode_id: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{scene_id}:{episode_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _read_dataset(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, list) or not episodes:
        raise RuntimeError(f"Dataset has no non-empty episodes array: {path}")
    return episodes


def _read_excluded_keys(paths: Iterable[Path]) -> set[tuple[str, int]]:
    excluded: set[tuple[str, int]] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if "scene_id" not in row or "episode_id" not in row:
                    raise RuntimeError(
                        f"Missing scene_id/episode_id at {path}:{line_number}"
                    )
                excluded.add((str(row["scene_id"]), int(row["episode_id"])))
    return excluded


def _rollout_report_label_paths(paths: Iterable[Path]) -> list[Path]:
    labels: list[Path] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "passed":
            raise RuntimeError(f"Rollout report did not pass validation: {path}")
        roots = payload.get("roots")
        if not isinstance(roots, list) or len(roots) != int(
            payload.get("root_count", -1)
        ):
            raise RuntimeError(f"Rollout report root contract is invalid: {path}")
        for entry in roots:
            root = entry.get("root") if isinstance(entry, dict) else None
            if not isinstance(root, str) or not root:
                raise RuntimeError(f"Rollout report contains an invalid root: {path}")
            labels_path = Path(root) / "system2_stop_multimodal_examples.jsonl"
            if not labels_path.is_file():
                raise FileNotFoundError(
                    f"Rollout report labels are missing: {labels_path}"
                )
            labels.append(labels_path)
    return labels


def shard_cohort(
    cohort: list[dict[str, Any]],
    *,
    num_shards: int,
) -> list[list[dict[str, Any]]]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if num_shards > len(cohort):
        raise ValueError("num_shards cannot exceed the cohort size")
    shards = [cohort[index::num_shards] for index in range(num_shards)]
    if any(not shard for shard in shards):
        raise RuntimeError("Cohort sharding produced an empty shard")
    keys = [
        (str(row["scene_id"]), int(row["episode_id"]))
        for shard in shards
        for row in shard
    ]
    if len(keys) != len(cohort) or len(keys) != len(set(keys)):
        raise RuntimeError("Cohort shards are incomplete or overlap")
    return shards


def build_cohorts(
    episodes: list[dict[str, Any]],
    *,
    excluded: set[tuple[str, int]],
    num_cohorts: int,
    per_scene_per_cohort: int,
    seed: int,
    allow_incomplete_scenes: bool = False,
    selection: str = "longest",
) -> list[list[dict[str, Any]]]:
    if num_cohorts < 1 or per_scene_per_cohort < 1:
        raise ValueError("num_cohorts and per_scene_per_cohort must be >= 1")
    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {SELECTIONS}, got {selection!r}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for episode in episodes:
        scene_id = _scene_id(str(episode["scene_id"]))
        episode_id = int(episode["episode_id"])
        key = (scene_id, episode_id)
        if key in seen:
            raise RuntimeError(f"Duplicate dataset episode key: {key}")
        seen.add(key)
        if key in excluded:
            continue
        distance = float((episode.get("info") or {}).get("geodesic_distance", 0.0))
        grouped[scene_id].append(
            {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "geodesic_distance": distance,
                "baseline": "train_split_closed_loop_dagger",
            }
        )

    required_per_scene = num_cohorts * per_scene_per_cohort
    insufficient = {
        scene_id: len(candidates)
        for scene_id, candidates in grouped.items()
        if len(candidates) < required_per_scene
    }
    if insufficient:
        if not allow_incomplete_scenes:
            raise RuntimeError(
                f"Scenes need at least {required_per_scene} unused episodes: {insufficient}"
            )
        for scene_id in insufficient:
            del grouped[scene_id]
    if not grouped:
        raise RuntimeError("No scenes have enough unused episodes for the requested cohorts")

    cohorts: list[list[dict[str, Any]]] = [[] for _ in range(num_cohorts)]
    for scene_id in sorted(grouped):
        if selection in {"longest", "shortest"}:
            direction = -1.0 if selection == "longest" else 1.0
            candidates = sorted(
                grouped[scene_id],
                key=lambda row: (
                    direction * float(row["geodesic_distance"]),
                    _stable_tie_break(scene_id, int(row["episode_id"]), seed),
                ),
            )[:required_per_scene]
        else:
            distance_ordered = sorted(
                grouped[scene_id],
                key=lambda row: (
                    float(row["geodesic_distance"]),
                    _stable_tie_break(scene_id, int(row["episode_id"]), seed),
                ),
            )
            quantile = _SELECTION_QUANTILES[selection]
            target_index = int((len(distance_ordered) - 1) * quantile + 0.5)
            candidates = [
                row
                for _, row in sorted(
                    enumerate(distance_ordered),
                    key=lambda item: (
                        abs(item[0] - target_index),
                        item[0],
                    ),
                )[:required_per_scene]
            ]
        for cohort_index in range(num_cohorts):
            start = cohort_index * per_scene_per_cohort
            end = start + per_scene_per_cohort
            cohorts[cohort_index].extend(candidates[start:end])

    all_keys = [
        (str(row["scene_id"]), int(row["episode_id"]))
        for cohort in cohorts
        for row in cohort
    ]
    if len(all_keys) != len(set(all_keys)):
        raise RuntimeError("Generated cohorts are not episode-disjoint")
    return cohorts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--exclude-labels-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--exclude-rollout-report", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--num-cohorts", type=int, default=1)
    parser.add_argument("--shards-per-cohort", type=int, default=1)
    parser.add_argument("--per-scene-per-cohort", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection", choices=SELECTIONS, default="longest")
    parser.add_argument("--allow-incomplete-scenes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    episodes = _read_dataset(args.dataset)
    report_labels = _rollout_report_label_paths(args.exclude_rollout_report)
    excluded_label_paths = [*args.exclude_labels_jsonl, *report_labels]
    excluded = _read_excluded_keys(excluded_label_paths)
    cohorts = build_cohorts(
        episodes,
        excluded=excluded,
        num_cohorts=args.num_cohorts,
        per_scene_per_cohort=args.per_scene_per_cohort,
        seed=args.seed,
        allow_incomplete_scenes=args.allow_incomplete_scenes,
        selection=args.selection,
    )
    dataset_scenes = {_scene_id(str(episode["scene_id"])) for episode in episodes}
    selected_scenes = {str(row["scene_id"]) for row in cohorts[0]}
    skipped_scenes = sorted(dataset_scenes - selected_scenes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, cohort in enumerate(cohorts):
        shards = shard_cohort(cohort, num_shards=args.shards_per_cohort)
        for shard_index, shard in enumerate(shards):
            shard_suffix = (
                f"_shard_{shard_index:02d}"
                if args.shards_per_cohort > 1
                else ""
            )
            output = args.output_dir / f"{args.prefix}_{index:02d}{shard_suffix}.json"
            if output.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite cohort: {output}")
            distances = [float(row["geodesic_distance"]) for row in shard]
            payload = {
                "description": (
                    f"Scene-balanced {args.selection} unused R2R train episodes for "
                    "closed-loop System2 STOP DAgger collection."
                ),
                "dataset": str(args.dataset),
                "seed": int(args.seed),
                "selection": args.selection,
                "excluded_episode_count": len(excluded),
                "excluded_label_files": [
                    str(path) for path in excluded_label_paths
                ],
                "excluded_rollout_reports": [
                    str(path) for path in args.exclude_rollout_report
                ],
                "skipped_scene_ids": skipped_scenes,
                "cohort_index": index,
                "num_cohorts": len(cohorts),
                "parent_cohort_episode_count": len(cohort),
                "shard_index": shard_index,
                "num_shards": len(shards),
                "episodes": shard,
            }
            output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"{output}: episodes={len(shard)} scenes="
                f"{len({row['scene_id'] for row in shard})} "
                f"skipped_scenes={skipped_scenes} "
                f"distance_m=[{min(distances):.2f},{max(distances):.2f}] "
                f"mean={sum(distances) / len(distances):.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

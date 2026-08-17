#!/usr/bin/env python3
"""Validate and merge train-split System2 on-policy rollout collections."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

SCHEMA = "heatmapvln-system2-stop-multimodal-example-v1"
VIEWS = frozenset(("front", "right", "back", "left"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_bucket(target: int | None, terminal: bool) -> str:
    if target is None:
        return "ambiguous"
    if target == 1:
        return "original_correct_stop" if terminal else "add_positive"
    return "false_stop_negative" if terminal else "regular_negative"


def _expected_target(distance_m: float, positive_m: float, negative_m: float) -> int | None:
    if not math.isfinite(distance_m) or distance_m < 0.0:
        raise ValueError(f"Invalid distance_to_goal_m: {distance_m}")
    if distance_m <= positive_m:
        return 1
    if distance_m >= negative_m:
        return 0
    return None


def _resolve_image(root: Path, raw_path: Any, *, key: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"Invalid image path for {key}: {raw_path!r}")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Image path escapes rollout root for {key}: {raw_path}")
    # Avoid Path.resolve() here: a validated base report can reference hundreds
    # of thousands of AFS images, and lexical validation is sufficient for the
    # recorder-owned relative paths. New files are still stat'ed and decoded.
    return root / relative


def _record_images(root: Path, record: dict[str, Any]) -> list[Path]:
    key = str(record.get("key") or "<missing>")
    groups = [record.get("current_views")]
    history = record.get("history_views")
    if not isinstance(history, list):
        raise ValueError(f"history_views must be a list for {key}")
    groups.extend(history)
    paths: list[Path] = []
    for group in groups:
        if not isinstance(group, dict) or set(group) != VIEWS:
            raise ValueError(f"Expected exactly four panoramic views for {key}")
        paths.extend(_resolve_image(root, group[view], key=key) for view in sorted(VIEWS))
    if len(paths) != len(set(paths)):
        raise ValueError(f"Duplicate image references within rollout row {key}")
    return paths


def _validate_collection_manifest(root: Path, episode_count: int) -> None:
    manifest_path = root / "eval_manifest.json"
    result_path = root / "result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(
            f"Incomplete rollout root (manifest/result missing): {root}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    required_true = (
        "rpc_require_deterministic_sampling",
        "system2_stop_feature_collection",
        "system2_stop_multimodal_example_collection",
        "system2_stop_collect_force_continue_negatives",
        "system2_stop_collect_oracle_recovery_after_negative",
    )
    for field in required_true:
        if manifest.get(field) is not True:
            raise RuntimeError(f"Collection manifest requires {field}=true: {root}")
    if manifest.get("system2_stop_policy_mode") != "original_system2":
        raise RuntimeError(f"Collection did not use original System2: {root}")
    for field in (
        "system2_stop_head_checkpoint",
        "system2_stop_decision_adapter_checkpoint",
        "system2_temporal_stop_verifier_checkpoint",
    ):
        if manifest.get(field):
            raise RuntimeError(f"Collection unexpectedly enabled {field}: {root}")
    if float(manifest.get("system2_stop_positive_radius_m", -1.0)) != 3.0:
        raise RuntimeError(f"Collection positive radius is not 3.0m: {root}")
    if float(manifest.get("system2_stop_negative_radius_m", -1.0)) != 3.01:
        raise RuntimeError(f"Collection negative radius is not 3.01m: {root}")
    if Path(str(manifest.get("data_path", ""))).parent.name != "train":
        raise RuntimeError(f"Collection did not use the R2R train split: {root}")
    if int(result.get("total_episodes", -1)) != episode_count:
        raise RuntimeError(
            f"Collection episode count mismatch for {root}: "
            f"result={result.get('total_episodes')} labels={episode_count}"
        )


def _decode_image(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty rollout image: {path}")
    with Image.open(path) as image:
        image.load()
        if image.width <= 0 or image.height <= 0:
            raise ValueError(f"Invalid rollout image dimensions: {path}")


def _scene_split(
    records: list[dict[str, Any]],
    *,
    seed: int,
    holdout_fraction: float,
) -> tuple[set[str], set[str]]:
    scenes = sorted({str(record["scene_id"]) for record in records})
    if len(scenes) < 2:
        raise RuntimeError("Rollout validation requires at least two scenes")
    ordered = sorted(
        scenes,
        key=lambda scene_id: hashlib.sha256(f"{seed}:{scene_id}".encode()).digest(),
    )
    holdout_count = min(
        len(scenes) - 1,
        max(1, round(len(scenes) * float(holdout_fraction))),
    )
    validation_scenes = set(ordered[:holdout_count])
    return set(ordered[holdout_count:]), validation_scenes


def _load_base_report(path: Path | None) -> tuple[list[Path], dict[Path, dict[str, Any]]]:
    if path is None:
        return [], {}
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise RuntimeError(f"Base rollout report did not pass: {path}")
    entries = report.get("roots")
    if not isinstance(entries, list) or len(entries) != int(report.get("root_count", -1)):
        raise RuntimeError(f"Invalid base rollout report root contract: {path}")
    roots: list[Path] = []
    summaries: dict[Path, dict[str, Any]] = {}
    for entry in entries:
        raw_root = entry.get("root") if isinstance(entry, dict) else None
        if not isinstance(raw_root, str) or not raw_root:
            raise RuntimeError(f"Invalid base rollout root in {path}")
        root = Path(raw_root).expanduser().resolve()
        if root in summaries:
            raise RuntimeError(f"Duplicate base rollout root: {root}")
        roots.append(root)
        summaries[root] = entry
    return roots, summaries


def _cohort_episode_keys(path: Path) -> set[tuple[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, list) or not episodes:
        raise RuntimeError(f"Cohort has no non-empty episodes list: {path}")
    keys = {
        (str(row.get("scene_id") or ""), int(row.get("episode_id", -1)))
        for row in episodes
        if isinstance(row, dict)
    }
    if len(keys) != len(episodes) or any(not scene or episode_id < 0 for scene, episode_id in keys):
        raise RuntimeError(f"Cohort has invalid or duplicate episode keys: {path}")
    return keys


def validate_rollouts(
    *,
    base_report: Path | None,
    new_roots: list[Path],
    new_cohorts: list[Path] | None = None,
    split_seed: int,
    holdout_fraction: float,
    decode_workers: int,
) -> dict[str, Any]:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    if decode_workers < 1:
        raise ValueError("decode_workers must be >= 1")
    base_roots, base_summaries = _load_base_report(base_report)
    resolved_new = [root.expanduser().resolve() for root in new_roots]
    if new_cohorts is not None and len(new_cohorts) != len(resolved_new):
        raise ValueError("new_cohorts must align one-to-one with new_roots")
    resolved_cohorts = (
        [path.expanduser().resolve() for path in new_cohorts]
        if new_cohorts is not None
        else []
    )
    cohort_by_root = dict(zip(resolved_new, resolved_cohorts))
    roots = [*base_roots, *resolved_new]
    if not roots:
        raise ValueError("At least one rollout root is required")
    if len(roots) != len(set(roots)):
        raise RuntimeError("Rollout roots must be unique")

    all_records: list[dict[str, Any]] = []
    all_keys: set[str] = set()
    episode_owners: dict[tuple[str, int], set[Path]] = {}
    root_reports: list[dict[str, Any]] = []
    new_images: list[Path] = []
    for root in roots:
        labels_path = root / "system2_stop_multimodal_examples.jsonl"
        if not labels_path.is_file():
            raise FileNotFoundError(f"Missing rollout labels: {labels_path}")
        labels_sha256 = _file_sha256(labels_path)
        base_summary = base_summaries.get(root)
        if base_summary is not None and labels_sha256 != base_summary.get("labels_sha256"):
            raise RuntimeError(f"Base rollout labels changed after validation: {labels_path}")

        root_records: list[dict[str, Any]] = []
        root_images: list[Path] = []
        with labels_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                location = f"{labels_path}:{line_number}"
                if record.get("schema") != SCHEMA:
                    raise RuntimeError(f"Unexpected rollout schema at {location}")
                if record.get("dataset_split") != "train":
                    raise RuntimeError(f"Non-train rollout row at {location}")
                key = str(record.get("key") or "")
                if not key or key in all_keys:
                    raise RuntimeError(f"Missing or duplicate rollout key at {location}: {key!r}")
                all_keys.add(key)
                target = record.get("stop_target")
                if target not in (0, 1, None):
                    raise RuntimeError(f"Invalid stop_target at {location}: {target!r}")
                terminal = record.get("original_terminal")
                if not isinstance(terminal, bool):
                    raise RuntimeError(f"Missing original_terminal at {location}")
                distance = float(record.get("distance_to_goal_m", float("nan")))
                expected = _expected_target(distance, 3.0, 3.01)
                if target != expected:
                    raise RuntimeError(
                        f"STOP target/radius mismatch at {location}: "
                        f"distance={distance} target={target} expected={expected}"
                    )
                scene_id = str(record.get("scene_id") or "")
                episode_id = int(record.get("episode_id", -1))
                if not scene_id or episode_id < 0:
                    raise RuntimeError(f"Invalid episode identity at {location}")
                episode_key = (scene_id, episode_id)
                owners = episode_owners.setdefault(episode_key, set())
                if owners and root not in owners and (
                    root in resolved_new
                    or any(owner in resolved_new for owner in owners)
                ):
                    raise RuntimeError(
                        "A new rollout episode overlaps another root: "
                        f"{episode_key} roots={sorted(str(owner) for owner in owners | {root})}"
                    )
                owners.add(root)
                images = _record_images(root, record)
                root_images.extend(images)
                normalized = dict(record)
                normalized["_policy_bucket"] = _policy_bucket(target, terminal)
                root_records.append(normalized)

        if not root_records:
            raise RuntimeError(f"Rollout root contains no rows: {root}")
        root_episode_count = len(
            {(str(row["scene_id"]), int(row["episode_id"])) for row in root_records}
        )
        observed_episode_keys = {
            (str(row["scene_id"]), int(row["episode_id"])) for row in root_records
        }
        cohort_path = cohort_by_root.get(root)
        if cohort_path is not None:
            expected_episode_keys = _cohort_episode_keys(cohort_path)
            if observed_episode_keys != expected_episode_keys:
                raise RuntimeError(
                    f"Rollout/cohort episode mismatch for {root}: "
                    f"missing={sorted(expected_episode_keys - observed_episode_keys)[:5]} "
                    f"unexpected={sorted(observed_episode_keys - expected_episode_keys)[:5]}"
                )
        if root in resolved_new:
            _validate_collection_manifest(root, root_episode_count)
            new_images.extend(root_images)
        elif int(base_summary.get("rows", -1)) != len(root_records):
            raise RuntimeError(f"Base rollout row count changed after validation: {root}")
        elif int(base_summary.get("image_references", -1)) != len(root_images):
            raise RuntimeError(
                f"Base rollout image-reference count changed after validation: {root}"
            )

        combos = Counter(
            f"target={row.get('stop_target')},original_terminal={row['original_terminal']}"
            for row in root_records
        )
        root_reports.append(
            {
                "root": str(root),
                "rows": len(root_records),
                "episodes": root_episode_count,
                "scenes": len({str(row["scene_id"]) for row in root_records}),
                "image_references": len(root_images),
                "labels_sha256": labels_sha256,
                "combos": dict(sorted(combos.items())),
                "images_decoded_in_this_validation": root in resolved_new,
                "cohort": (
                    {
                        "path": str(cohort_path),
                        "sha256": _file_sha256(cohort_path),
                        "episodes": len(observed_episode_keys),
                    }
                    if cohort_path is not None
                    else None
                ),
            }
        )
        all_records.extend(root_records)

    with ThreadPoolExecutor(max_workers=decode_workers) as executor:
        for _ in executor.map(_decode_image, new_images):
            pass

    train_scenes, validation_scenes = _scene_split(
        all_records,
        seed=split_seed,
        holdout_fraction=holdout_fraction,
    )
    split_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
    }
    for record in all_records:
        split = "train" if str(record["scene_id"]) in train_scenes else "validation"
        split_counts[split][str(record["_policy_bucket"])] += 1
    required_buckets = {"add_positive", "regular_negative", "false_stop_negative"}
    for split, counts in split_counts.items():
        missing = sorted(required_buckets - set(counts))
        if missing:
            raise RuntimeError(f"{split} rollout split lacks policy buckets: {missing}")

    labelled_rows = sum(record.get("stop_target") in (0, 1) for record in all_records)
    report = {
        "status": "passed",
        "root_count": len(roots),
        "roots": root_reports,
        "rows": len(all_records),
        "labelled_rows": labelled_rows,
        "ambiguous_rows": len(all_records) - labelled_rows,
        "unique_keys": len(all_keys),
        "unique_episodes": len(episode_owners),
        "repeated_base_episode_count": sum(
            len(owners) > 1 for owners in episode_owners.values()
        ),
        "unique_scenes": len(train_scenes | validation_scenes),
        "image_references": sum(entry["image_references"] for entry in root_reports),
        "decoded_images": len(new_images),
        "base_images_trusted_from_prior_report": sum(
            entry["image_references"]
            for entry in root_reports
            if not entry["images_decoded_in_this_validation"]
        ),
        "base_report": (
            {
                "path": str(base_report.resolve()),
                "sha256": _file_sha256(base_report),
            }
            if base_report is not None
            else None
        ),
        "split_seed": int(split_seed),
        "holdout_scene_fraction": float(holdout_fraction),
        "train_scenes": len(train_scenes),
        "validation_scenes": len(validation_scenes),
        "train_policy_counts": dict(sorted(split_counts["train"].items())),
        "validation_policy_counts": dict(
            sorted(split_counts["validation"].items())
        ),
        "train_false_stop_scenes": len(
            {
                str(row["scene_id"])
                for row in all_records
                if row["_policy_bucket"] == "false_stop_negative"
                and str(row["scene_id"]) in train_scenes
            }
        ),
        "validation_false_stop_scenes": len(
            {
                str(row["scene_id"])
                for row in all_records
                if row["_policy_bucket"] == "false_stop_negative"
                and str(row["scene_id"]) in validation_scenes
            }
        ),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report", type=Path)
    parser.add_argument("--rollout-root", type=Path, action="append", default=[])
    parser.add_argument("--rollout-cohort", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260720)
    parser.add_argument("--holdout-scene-fraction", type=float, default=0.2)
    parser.add_argument("--decode-workers", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite rollout report: {output}")
    if len(args.rollout_root) != len(args.rollout_cohort):
        raise ValueError(
            "Each --rollout-root requires one aligned --rollout-cohort"
        )
    report = validate_rollouts(
        base_report=(args.base_report.expanduser().resolve() if args.base_report else None),
        new_roots=args.rollout_root,
        new_cohorts=args.rollout_cohort,
        split_seed=args.split_seed,
        holdout_fraction=args.holdout_scene_fraction,
        decode_workers=args.decode_workers,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

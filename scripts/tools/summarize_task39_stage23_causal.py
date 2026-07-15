#!/usr/bin/env python3
"""Validate and summarize the preregistered Task39 Stage2/Stage3 causal comparison.

The only estimand in this report is ``pano_control - warmup_original``.  A
failed scientific gate is still a valid report; malformed or incomparable
artifacts raise :class:`ContractError` and produce no summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "heatmapvln-task39-stage23-causal-summary-v1"
EXPECTED_EPISODE_COUNT = 1839
EXPECTED_CODE_COMMIT = "2c5c94daa642690feb28d270c9457c36971bb154"
RPC_PROTOCOL = "heatmapvln-r2r-json-v2"
RPC_SAMPLING_PROTOCOL = "heatmapvln-nextdit-sha256-v1"
RPC_PROTOCOL_SEED = 42
BOOTSTRAP_REPLICATES = 50_000
BOOTSTRAP_SEED = 42

SPL_POINT_THRESHOLD = 0.02
SPL_CI_LOWER_THRESHOLD = 0.0
SR_ONE_SIDED_LOWER_THRESHOLD = -0.02

METRIC_FIELDS = {
    "SPL": "spl",
    "SR": "success",
    "OS": "os",
    "NE": "ne",
    "steps": "steps",
    "vlm_calls": "vlm_calls",
    "trajectory_calls": "trajectory_calls",
}

PROTOCOL_CONTRACT = {
    "rpc_protocol": RPC_PROTOCOL,
    "rpc_sampling_protocol": RPC_SAMPLING_PROTOCOL,
    "rpc_deterministic_sampling_enabled": True,
    "rpc_protocol_seed": RPC_PROTOCOL_SEED,
    "rpc_require_deterministic_sampling": True,
}

# These fields identify the arm or the output artifact and are the only
# manifest values allowed to differ.  Every other manifest key is treated as
# a common-run field and must match exactly, including future fields.
ARM_SPECIFIC_MANIFEST_FIELDS = frozenset(
    {
        "created_at",
        "base_checkpoint",
        "stage3_checkpoint",
    }
)

REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "code_commit",
        "config",
        "base_checkpoint",
        "stage3_checkpoint",
        "expected_epoch",
        "scenes_dir",
        "data_path",
        "rpc_root",
        *PROTOCOL_CONTRACT,
        "auto_stop_distance",
        "oracle_system2",
        "oracle_system2_strategy",
        "oracle_system2_lookahead_m",
        "oracle_system2_min_ahead_m",
        "oracle_system2_max_side_dist_m",
        "trajectory_selection",
        "trajectory_x_sign",
        "trajectory_heading_alignment",
        "system1_coord_order",
    }
)

ROW_RUN_FIELDS = (
    *PROTOCOL_CONTRACT,
    "auto_stop_distance",
    "oracle_system2",
    "trajectory_selection",
    "trajectory_x_sign",
    "trajectory_heading_alignment",
    "system1_coord_order",
)


class ContractError(ValueError):
    """An input artifact violates the preregistered comparison contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read {label} {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object: {path}")
    return value


def _episode_key(item: dict[str, Any], *, label: str) -> tuple[str, int]:
    _require(isinstance(item, dict), f"{label} episode entry must be an object")
    scene_id = item.get("scene_id")
    episode_id = item.get("episode_id")
    _require(isinstance(scene_id, str) and bool(scene_id), f"{label} has an invalid scene_id")
    _require(
        isinstance(episode_id, int) and not isinstance(episode_id, bool),
        f"{label} has a non-integer episode_id for scene {scene_id!r}",
    )
    return scene_id, episode_id


def _require_unique(keys: list[tuple[str, int]], *, label: str) -> None:
    seen: set[tuple[str, int]] = set()
    duplicates: list[tuple[str, int]] = []
    for key in keys:
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen.add(key)
    _require(not duplicates, f"{label} contains duplicate episodes: {duplicates[:10]}")


def load_ordered_cohort(path: str | Path) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    cohort_path = Path(path)
    raw = _load_json(cohort_path, label="ordered cohort")
    episodes = raw.get("episodes")
    _require(isinstance(episodes, list), "ordered cohort must contain an 'episodes' array")
    _require(
        len(episodes) == EXPECTED_EPISODE_COUNT,
        f"ordered cohort must contain exactly {EXPECTED_EPISODE_COUNT} episodes; found {len(episodes)}",
    )
    keys = [_episode_key(item, label="ordered cohort") for item in episodes]
    _require_unique(keys, label="ordered cohort")
    for count_field in ("count", "episode_count"):
        if count_field not in raw:
            continue
        _require(
            raw[count_field] == EXPECTED_EPISODE_COUNT,
            f"ordered cohort {count_field} metadata must equal {EXPECTED_EPISODE_COUNT}",
        )
    identity_sha256 = _sha256_json(keys)
    if "ordered_episode_identity_sha256" in raw:
        _require(
            raw["ordered_episode_identity_sha256"] == identity_sha256,
            "ordered cohort ordered_episode_identity_sha256 does not match its episodes",
        )
    return keys, {
        "path": str(cohort_path.resolve()),
        "file_sha256": _sha256_file(cohort_path),
        "ordered_episode_identity_sha256": identity_sha256,
        "identity_hash_encoding": "canonical_json_array_of_[scene_id,episode_id]_pairs",
        "episodes": len(keys),
        "scenes": len({scene_id for scene_id, _ in keys}),
    }


def _load_progress(path: Path, *, arm: str) -> list[dict[str, Any]]:
    _require(path.is_file(), f"{arm} progress is missing: {path}")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError(f"Cannot read {arm} progress {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{arm} progress line {line_number} is invalid JSON: {exc}") from exc
        _require(
            isinstance(row, dict),
            f"{arm} progress line {line_number} must be a JSON object",
        )
        rows.append(row)
    _require(
        len(rows) == EXPECTED_EPISODE_COUNT,
        f"{arm} progress must contain exactly {EXPECTED_EPISODE_COUNT} rows; found {len(rows)}",
    )
    return rows


def _require_protocol(record: dict[str, Any], *, label: str) -> None:
    for field, expected in PROTOCOL_CONTRACT.items():
        _require(field in record, f"{label} is missing protocol field {field!r}")
        actual = record[field]
        if isinstance(expected, bool):
            valid = actual is expected
        elif isinstance(expected, int):
            valid = isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
        else:
            valid = type(actual) is type(expected) and actual == expected
        _require(
            valid,
            f"{label} protocol mismatch for {field}: expected {expected!r}, found {actual!r}",
        )


def _finite_number(value: Any, *, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    numeric = float(value)
    _require(math.isfinite(numeric), f"{label} must be finite")
    return numeric


def _nonnegative_integer(value: Any, *, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a non-negative integer",
    )
    return int(value)


def _require_nonprivileged(record: dict[str, Any], *, label: str) -> None:
    auto_stop = _finite_number(record.get("auto_stop_distance"), label=f"{label}.auto_stop_distance")
    _require(auto_stop == 0.0, f"{label} enables privileged auto-stop ({auto_stop})")
    _require(record.get("oracle_system2") is False, f"{label} must set oracle_system2=false")
    for optional_field in ("allow_privileged", "rpc_allow_privileged"):
        if optional_field in record:
            _require(
                record[optional_field] is False,
                f"{label} must set {optional_field}=false",
            )


def _validate_manifest(manifest: dict[str, Any], *, arm: str) -> None:
    missing = sorted(REQUIRED_MANIFEST_FIELDS - manifest.keys())
    _require(not missing, f"{arm} eval_manifest is missing fields: {missing}")
    _require(
        manifest["code_commit"] == EXPECTED_CODE_COMMIT,
        f"{arm} eval_manifest code_commit must be {EXPECTED_CODE_COMMIT}; found {manifest['code_commit']!r}",
    )
    _require_protocol(manifest, label=f"{arm} eval_manifest")
    _require_nonprivileged(manifest, label=f"{arm} eval_manifest")


def _common_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key not in ARM_SPECIFIC_MANIFEST_FIELDS}


def _require_same_common_manifest(
    warmup: dict[str, Any],
    pano: dict[str, Any],
) -> dict[str, Any]:
    warmup_common = _common_manifest(warmup)
    pano_common = _common_manifest(pano)
    warmup_keys = set(warmup_common)
    pano_keys = set(pano_common)
    _require(
        warmup_keys == pano_keys,
        "eval_manifest common-field sets differ: "
        f"warmup_only={sorted(warmup_keys - pano_keys)}, "
        f"pano_only={sorted(pano_keys - warmup_keys)}",
    )
    mismatches = [key for key in sorted(warmup_common) if warmup_common[key] != pano_common[key]]
    _require(
        not mismatches,
        f"eval_manifest common run fields differ: {mismatches}",
    )
    _require(
        warmup["base_checkpoint"] != pano["base_checkpoint"],
        "the two arms use the same base_checkpoint; the intended Task38 contrast is absent",
    )
    _require(
        warmup["stage3_checkpoint"] != pano["stage3_checkpoint"],
        "the two arms use the same stage3_checkpoint; the intended causal contrast is absent",
    )
    return warmup_common


def _validate_row(
    row: dict[str, Any],
    *,
    arm: str,
    row_index: int,
    manifest: dict[str, Any],
) -> tuple[str, int]:
    label = f"{arm} progress row {row_index}"
    key = _episode_key(row, label=label)
    _require_protocol(row, label=label)
    _require_nonprivileged(row, label=label)
    for field in ROW_RUN_FIELDS:
        _require(field in row, f"{label} is missing run field {field!r}")
        _require(
            row[field] == manifest[field] and type(row[field]) is type(manifest[field]),
            f"{label} does not match eval_manifest field {field!r}",
        )

    success = _finite_number(row.get("success"), label=f"{label}.success")
    spl = _finite_number(row.get("spl"), label=f"{label}.spl")
    oracle_success = _finite_number(row.get("os"), label=f"{label}.os")
    ne = _finite_number(row.get("ne"), label=f"{label}.ne")
    _require(success in (0.0, 1.0), f"{label}.success must be binary")
    _require(0.0 <= spl <= 1.0, f"{label}.spl must be in [0, 1]")
    _require(0.0 <= oracle_success <= 1.0, f"{label}.os must be in [0, 1]")
    _require(ne >= 0.0, f"{label}.ne must be non-negative")
    _nonnegative_integer(row.get("steps"), label=f"{label}.steps")
    _nonnegative_integer(row.get("vlm_calls"), label=f"{label}.vlm_calls")
    _nonnegative_integer(
        row.get("trajectory_calls"),
        label=f"{label}.trajectory_calls",
    )
    return key


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    aggregate: dict[str, float | int] = {"episodes": len(rows)}
    for metric, field in METRIC_FIELDS.items():
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        aggregate[metric] = float(values.mean())
    return aggregate


def _validate_result(
    result: dict[str, Any],
    *,
    arm: str,
    manifest: dict[str, Any],
    aggregate: dict[str, float | int],
) -> None:
    label = f"{arm} result.json"
    _require_protocol(result, label=label)
    _require_nonprivileged(result, label=label)
    for field in ROW_RUN_FIELDS:
        _require(field in result, f"{label} is missing run field {field!r}")
        _require(
            result[field] == manifest[field] and type(result[field]) is type(manifest[field]),
            f"{label} does not match eval_manifest field {field!r}",
        )
    _require(
        result.get("total_episodes") == EXPECTED_EPISODE_COUNT,
        f"{label}.total_episodes must equal {EXPECTED_EPISODE_COUNT}",
    )
    for metric in ("SPL", "SR", "OS", "NE"):
        actual = _finite_number(result.get(metric), label=f"{label}.{metric}")
        expected = float(aggregate[metric])
        _require(
            math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10),
            f"{label}.{metric} disagrees with progress: result={actual}, progress={expected}",
        )


def _load_arm(
    eval_dir: str | Path,
    *,
    arm: str,
    cohort_keys: list[tuple[str, int]],
) -> dict[str, Any]:
    root = Path(eval_dir)
    _require(root.is_dir(), f"{arm} eval directory is missing: {root}")
    manifest_path = root / "eval_manifest.json"
    progress_path = root / "progress.json"
    result_path = root / "result.json"
    manifest = _load_json(manifest_path, label=f"{arm} eval_manifest")
    result = _load_json(result_path, label=f"{arm} result.json")
    rows = _load_progress(progress_path, arm=arm)
    _validate_manifest(manifest, arm=arm)

    keys = [
        _validate_row(
            row,
            arm=arm,
            row_index=index,
            manifest=manifest,
        )
        for index, row in enumerate(rows, 1)
    ]
    _require_unique(keys, label=f"{arm} progress")
    if keys != cohort_keys:
        mismatch_index = next(
            (index for index, (actual, expected) in enumerate(zip(keys, cohort_keys)) if actual != expected),
            None,
        )
        detail = (
            f"first mismatch at zero-based index {mismatch_index}: "
            f"progress={keys[mismatch_index]!r}, cohort={cohort_keys[mismatch_index]!r}"
            if mismatch_index is not None
            else "episode identities differ"
        )
        raise ContractError(f"{arm} progress does not exactly match ordered cohort; {detail}")

    aggregate = _aggregate(rows)
    _validate_result(
        result,
        arm=arm,
        manifest=manifest,
        aggregate=aggregate,
    )
    return {
        "root": root,
        "manifest": manifest,
        "rows": rows,
        "aggregate": aggregate,
        "artifacts": {
            "eval_dir": str(root.resolve()),
            "eval_manifest_sha256": _sha256_file(manifest_path),
            "progress_sha256": _sha256_file(progress_path),
            "result_sha256": _sha256_file(result_path),
        },
    }


def _paired_effects(
    warmup_rows: list[dict[str, Any]],
    pano_rows: list[dict[str, Any]],
) -> dict[str, float]:
    effects: dict[str, float] = {}
    for metric, field in METRIC_FIELDS.items():
        warmup = np.asarray([float(row[field]) for row in warmup_rows], dtype=np.float64)
        pano = np.asarray([float(row[field]) for row in pano_rows], dtype=np.float64)
        effects[metric] = float((pano - warmup).mean())
    return effects


def paired_scene_cluster_bootstrap(
    cohort_keys: list[tuple[str, int]],
    warmup_rows: list[dict[str, Any]],
    pano_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Resample paired scenes and retain all episodes from each sampled scene."""

    _require(
        len(cohort_keys) == len(warmup_rows) == len(pano_rows),
        "paired bootstrap inputs have different lengths",
    )
    scene_indices: dict[str, list[int]] = defaultdict(list)
    for index, (scene_id, _episode_id) in enumerate(cohort_keys):
        scene_indices[scene_id].append(index)
    scenes = sorted(scene_indices)
    _require(len(scenes) >= 2, "scene-cluster bootstrap requires at least two scenes")

    cluster_counts = np.asarray([len(scene_indices[scene]) for scene in scenes], dtype=np.float64)
    cluster_sums: dict[str, np.ndarray] = {}
    for metric in ("SPL", "SR"):
        field = METRIC_FIELDS[metric]
        differences = np.asarray(
            [float(pano[field]) - float(warmup[field]) for warmup, pano in zip(warmup_rows, pano_rows)],
            dtype=np.float64,
        )
        cluster_sums[metric] = np.asarray(
            [differences[scene_indices[scene]].sum() for scene in scenes],
            dtype=np.float64,
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    distributions = {metric: np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64) for metric in cluster_sums}
    probabilities = np.full(len(scenes), 1.0 / len(scenes), dtype=np.float64)
    chunk_size = 4096
    for start in range(0, BOOTSTRAP_REPLICATES, chunk_size):
        stop = min(start + chunk_size, BOOTSTRAP_REPLICATES)
        weights = rng.multinomial(
            len(scenes),
            probabilities,
            size=stop - start,
        ).astype(np.float64, copy=False)
        denominators = weights @ cluster_counts
        _require(
            bool(np.all(denominators > 0.0)),
            "scene-cluster bootstrap produced an empty replicate",
        )
        for metric, sums in cluster_sums.items():
            distributions[metric][start:stop] = (weights @ sums) / denominators

    spl_distribution = distributions["SPL"]
    sr_distribution = distributions["SR"]
    return {
        "method": "paired_scene_cluster_percentile_bootstrap",
        "resampling_unit": "scene",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "scene_count": len(scenes),
        "scene_episode_counts": {scene: len(scene_indices[scene]) for scene in scenes},
        "SPL": {
            "interval": "two_sided_95_percentile",
            "ci95": [
                float(np.quantile(spl_distribution, 0.025)),
                float(np.quantile(spl_distribution, 0.975)),
            ],
            "bootstrap_mean": float(spl_distribution.mean()),
        },
        "SR": {
            "interval": "one_sided_95_percentile_lower",
            "ci95_lower": float(np.quantile(sr_distribution, 0.05)),
            "bootstrap_mean": float(sr_distribution.mean()),
        },
    }


def _build_gate(effects: dict[str, float], bootstrap: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "delta_SPL_at_least_0.02": effects["SPL"] >= SPL_POINT_THRESHOLD,
        "SPL_two_sided_95_CI_lower_above_0": (float(bootstrap["SPL"]["ci95"][0]) > SPL_CI_LOWER_THRESHOLD),
        "SR_one_sided_95_lower_above_minus_0.02": (float(bootstrap["SR"]["ci95_lower"]) > SR_ONE_SIDED_LOWER_THRESHOLD),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "locked_thresholds": {
            "delta_SPL_minimum": SPL_POINT_THRESHOLD,
            "SPL_two_sided_95_CI_lower_strictly_above": SPL_CI_LOWER_THRESHOLD,
            "SR_one_sided_95_lower_strictly_above": SR_ONE_SIDED_LOWER_THRESHOLD,
        },
    }


def summarize(
    warmup_eval_dir: str | Path,
    pano_control_eval_dir: str | Path,
    ordered_cohort: str | Path,
) -> dict[str, Any]:
    cohort_keys, cohort_summary = load_ordered_cohort(ordered_cohort)
    warmup = _load_arm(
        warmup_eval_dir,
        arm="warmup_original",
        cohort_keys=cohort_keys,
    )
    pano = _load_arm(
        pano_control_eval_dir,
        arm="pano_control",
        cohort_keys=cohort_keys,
    )
    common_manifest = _require_same_common_manifest(
        warmup["manifest"],
        pano["manifest"],
    )
    effects = _paired_effects(warmup["rows"], pano["rows"])
    bootstrap = paired_scene_cluster_bootstrap(
        cohort_keys,
        warmup["rows"],
        pano["rows"],
    )
    gate = _build_gate(effects, bootstrap)
    return {
        "schema": SCHEMA,
        "estimand": "pano_control_minus_warmup_original",
        "preregistration": {
            "code_commit": EXPECTED_CODE_COMMIT,
            "expected_episodes": EXPECTED_EPISODE_COUNT,
            "protocol": PROTOCOL_CONTRACT,
            "bootstrap": {
                "method": "paired_scene_cluster_percentile_bootstrap",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "SPL_interval": "two_sided_95_percentile",
                "SR_interval": "one_sided_95_percentile_lower",
            },
            "thresholds_locked_before_results": True,
            "thresholds": gate["locked_thresholds"],
        },
        "contract": {
            "passed": True,
            "ordered_cohort": cohort_summary,
            "common_manifest": common_manifest,
            "common_manifest_sha256": _sha256_json(common_manifest),
            "only_allowed_manifest_differences": sorted(ARM_SPECIFIC_MANIFEST_FIELDS),
        },
        "arms": {
            "warmup_original": {
                "metrics": warmup["aggregate"],
                "artifacts": warmup["artifacts"],
            },
            "pano_control": {
                "metrics": pano["aggregate"],
                "artifacts": pano["artifacts"],
            },
        },
        "point_effects": effects,
        "paired_scene_cluster_bootstrap": bootstrap,
        "gate": gate,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize the fixed Task39 Stage2/Stage3 causal protocol."
    )
    parser.add_argument(
        "--warmup-eval-dir",
        "--warmup-dir",
        dest="warmup_eval_dir",
        required=True,
    )
    parser.add_argument(
        "--pano-control-eval-dir",
        "--pano-control-dir",
        dest="pano_control_eval_dir",
        required=True,
    )
    parser.add_argument(
        "--ordered-cohort",
        "--cohort",
        dest="ordered_cohort",
        required=True,
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists():
        print(f"Task39 output already exists; overwrite refused: {output}", file=sys.stderr)
        return 2
    try:
        report = summarize(
            args.warmup_eval_dir,
            args.pano_control_eval_dir,
            args.ordered_cohort,
        )
    except ContractError as exc:
        print(f"Task39 contract validation failed: {exc}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError:
        print(f"Task39 output already exists; overwrite refused: {output}", file=sys.stderr)
        return 2
    print(f"Wrote Task39 causal summary to {output}")
    print(f"Task39 gate passed: {report['gate']['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

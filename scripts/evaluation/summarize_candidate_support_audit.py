#!/usr/bin/env python3
"""Summarize local candidate support without inventing a scalar reward.

The ordering is explicitly diagnostic and lexicographic: radius safety first,
then route progress/off-path/collision/revisit/Euclidean-distance auxiliaries.
It is not an authoritative VLN success label and therefore cannot by itself
authorize critic training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


SCHEMA = "candidate-support-summary-v1"
PRIORITY_FIELDS = (
    "entered_without_leaving_radius",
    "entered_radius",
    "did_not_leave_radius",
    "route_progress_delta_m",
    "negative_endpoint_offpath_m",
    "negative_collision_or_stuck_count",
    "not_revisit",
    "negative_min_euclidean_goal_distance_m",
    "negative_endpoint_euclidean_goal_distance_m",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _priority(outcome: dict[str, Any]) -> tuple[float, ...]:
    entered = bool(outcome["entered_euclidean_success_radius"])
    left = bool(outcome["left_euclidean_success_radius"])
    return (
        float(entered and not left),
        float(entered),
        float(not left),
        float(outcome["route_progress_delta_m"]),
        -float(outcome["endpoint_offpath_m"]),
        -float(outcome["collision_or_stuck_count"]),
        float(not bool(outcome["revisit"])),
        -float(outcome["min_euclidean_goal_distance_m"]),
        -float(outcome["endpoint_euclidean_goal_distance_m"]),
    )


def _first_margin(
    best: tuple[float, ...], second: tuple[float, ...]
) -> tuple[str, float]:
    for name, left, right in zip(PRIORITY_FIELDS, best, second):
        if left != right:
            return name, float(left - right)
    return "tie", 0.0


def _strictly_better(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _priority(left) > _priority(right)


def _treatment_arms(candidate_set: dict[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for treatment in candidate_set["treatments"]:
        treatment_id = treatment["treatment_id"]
        for provenance in treatment["provenances"]:
            result[treatment_id].add(str(provenance["arm"]))
    return result


def _source_entries(
    candidate_set: dict[str, Any], arm: str, aggregation: str | None = None
) -> list[dict[str, Any]]:
    entries = [
        entry
        for entry in candidate_set["source_entries"]
        if entry["arm"] == arm
        and (aggregation is None or entry["aggregation"] == aggregation)
    ]
    entries.sort(
        key=lambda entry: (
            entry["sample_index"] is None,
            -1 if entry["sample_index"] is None else int(entry["sample_index"]),
        )
    )
    return entries


def _best_id(ids: Iterable[str], outcomes: dict[str, dict[str, Any]]) -> str:
    unique = sorted(set(ids))
    if not unique:
        raise RuntimeError("candidate group is empty")
    return max(unique, key=lambda treatment_id: (_priority(outcomes[treatment_id]), treatment_id))


def _read_records(root: Path, expected_shards: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    state_keys: set[str] = set()
    for shard_id in range(expected_shards):
        shard = root / f"shard_{shard_id:02d}"
        index_path = shard / "records.jsonl"
        manifest_path = shard / "manifest.json"
        if not index_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"unsealed/missing audit shard: {shard}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("shard_id", -1)) != shard_id:
            raise RuntimeError(f"shard manifest id mismatch: {manifest_path}")
        if manifest.get("records_jsonl_sha256") != _sha256(index_path):
            raise RuntimeError(f"records JSONL SHA256 mismatch: {index_path}")
        rows = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if int(manifest.get("record_count", -1)) != len(rows):
            raise RuntimeError(f"record count mismatch: {shard}")
        shard_array_bytes = 0
        for row in rows:
            key = str(row.get("state_key") or "")
            if not key or key in state_keys:
                raise RuntimeError(f"empty/duplicate state key: {key!r}")
            state_keys.add(key)
            array_path = (shard / str(row.get("array_file") or "")).resolve()
            try:
                array_path.relative_to(shard.resolve())
            except ValueError as exc:
                raise RuntimeError(f"array path escapes shard: {array_path}") from exc
            if not array_path.is_file():
                raise FileNotFoundError(array_path)
            expected_bytes = int(row.get("array_file_bytes", -1))
            if array_path.stat().st_size != expected_bytes:
                raise RuntimeError(f"array byte count mismatch: {array_path}")
            if _sha256(array_path) != row.get("array_file_sha256"):
                raise RuntimeError(f"array SHA256 mismatch: {array_path}")
            shard_array_bytes += expected_bytes
        if shard_array_bytes != int(manifest.get("array_bytes", -1)):
            raise RuntimeError(f"manifest array byte total mismatch: {shard}")
        records.extend(rows)
        manifests.append(manifest)
    if not records:
        raise RuntimeError("candidate audit contains no trajectory states")
    return records, manifests


def _empty_counts() -> Counter[str]:
    return Counter(
        states=0,
        local_positive_union=0,
        local_positive_native=0,
        heatmap_adds_positive_support=0,
        native_mean_local_best=0,
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def summarize(records: list[dict[str, Any]], manifests: list[dict[str, Any]]) -> dict[str, Any]:
    counts = _empty_counts()
    strata: dict[str, Counter[str]] = defaultdict(_empty_counts)
    unique_counts: list[int] = []
    native_mean_ranks: list[int] = []
    margin_values: dict[str, list[float]] = defaultdict(list)
    baseline_better = Counter()
    k_stats: dict[str, dict[int, Counter[str]]] = {
        "native": defaultdict(Counter),
        "heatmap_control": defaultdict(Counter),
        "paired_union": defaultdict(Counter),
    }

    for record in records:
        candidate_set = record["candidate_set"]
        outcomes = {
            outcome["treatment_id"]: outcome
            for outcome in record["local_outcomes"]
        }
        treatment_ids = {
            treatment["treatment_id"] for treatment in candidate_set["treatments"]
        }
        if treatment_ids != set(outcomes):
            raise RuntimeError(
                f"treatment/outcome closure mismatch at {record['state_key']}"
            )
        baseline_id = candidate_set["baselines"]["native_trajectory_mean"]
        baseline = outcomes[baseline_id]
        arms = _treatment_arms(candidate_set)
        native_ids = [
            treatment_id
            for treatment_id, source_arms in arms.items()
            if "native" in source_arms
        ]
        native_best_id = _best_id(native_ids, outcomes)
        union_best_id = _best_id(treatment_ids, outcomes)
        native_positive = _strictly_better(outcomes[native_best_id], baseline)
        union_positive = _strictly_better(outcomes[union_best_id], baseline)
        heatmap_adds = _strictly_better(
            outcomes[union_best_id], outcomes[native_best_id]
        ) and "heatmap_control" in arms[union_best_id]

        counts["states"] += 1
        counts["local_positive_native"] += int(native_positive)
        counts["local_positive_union"] += int(union_positive)
        counts["heatmap_adds_positive_support"] += int(heatmap_adds)
        counts["native_mean_local_best"] += int(not union_positive)
        unique_counts.append(int(candidate_set["unique_treatment_count"]))

        ordered = sorted(
            treatment_ids,
            key=lambda treatment_id: (_priority(outcomes[treatment_id]), treatment_id),
            reverse=True,
        )
        if len(ordered) >= 2:
            margin_name, margin = _first_margin(
                _priority(outcomes[ordered[0]]), _priority(outcomes[ordered[1]])
            )
            margin_values[margin_name].append(margin)

        native_entries = _source_entries(candidate_set, "native")
        mean_entries = [
            entry for entry in native_entries if entry["aggregation"] == "trajectory_mean"
        ]
        sample_entries = [
            entry for entry in native_entries if entry["aggregation"] == "sample"
        ]
        if len(mean_entries) != 1:
            raise RuntimeError("native trajectory mean provenance is not unique")
        mean_priority = _priority(outcomes[mean_entries[0]["base_treatment_id"]])
        native_mean_ranks.append(
            1
            + sum(
                _priority(outcomes[entry["base_treatment_id"]]) > mean_priority
                for entry in sample_entries
            )
        )

        for name, treatment_id in candidate_set["baselines"].items():
            if name == "native_trajectory_mean":
                continue
            baseline_better[name] += int(
                _strictly_better(outcomes[treatment_id], baseline)
            )

        heatmap_entries = _source_entries(candidate_set, "heatmap_control")
        heatmap_mean_entries = [
            entry
            for entry in heatmap_entries
            if entry["aggregation"] == "trajectory_mean"
        ]
        heatmap_sample_entries = [
            entry
            for entry in heatmap_entries
            if entry["aggregation"] == "sample"
        ]
        if heatmap_entries and len(heatmap_mean_entries) != 1:
            raise RuntimeError("heatmap trajectory mean provenance is not unique")

        def treatment_ids_for(entries: Iterable[dict[str, Any]]) -> set[str]:
            return {
                treatment_id
                for entry in entries
                for treatment_id in entry["treatment_ids"]
            }

        for requested_k in (1, 4, 8, 16, 32):
            native_selected = sample_entries[: min(requested_k, len(sample_entries))]
            heatmap_selected = heatmap_sample_entries[
                : min(requested_k, len(heatmap_sample_entries))
            ]
            native_at_k = {baseline_id} | treatment_ids_for(
                [*mean_entries, *native_selected]
            )
            groups: dict[str, tuple[set[str], int]] = {
                "native": (native_at_k, len(native_selected)),
            }
            if heatmap_entries:
                heatmap_at_k = {baseline_id} | treatment_ids_for(
                    [*heatmap_mean_entries, *heatmap_selected]
                )
                groups["heatmap_control"] = (
                    heatmap_at_k,
                    len(heatmap_selected),
                )
                groups["paired_union"] = (
                    native_at_k | heatmap_at_k,
                    min(len(native_selected), len(heatmap_selected)),
                )
            for group_name, (ids, effective_k) in groups.items():
                best_at_k = _best_id(ids, outcomes)
                bucket = k_stats[group_name][requested_k]
                bucket["states"] += 1
                bucket["positive"] += int(
                    _strictly_better(outcomes[best_at_k], baseline)
                )
                bucket["unique_treatments"] += len(ids)
                bucket["effective_samples"] += effective_k

        active_strata = ["all"] + [
            name
            for name, active in record.get("state_strata", {}).items()
            if bool(active)
        ]
        for stratum in active_strata:
            bucket = strata[stratum]
            bucket["states"] += 1
            bucket["local_positive_native"] += int(native_positive)
            bucket["local_positive_union"] += int(union_positive)
            bucket["heatmap_adds_positive_support"] += int(heatmap_adds)
            bucket["native_mean_local_best"] += int(not union_positive)

    def summarize_counts(bucket: Counter[str]) -> dict[str, Any]:
        states = int(bucket["states"])
        return {
            "states": states,
            "local_positive_support_native_count": int(bucket["local_positive_native"]),
            "local_positive_support_native_rate": _ratio(
                int(bucket["local_positive_native"]), states
            ),
            "local_positive_support_union_count": int(bucket["local_positive_union"]),
            "local_positive_support_union_rate": _ratio(
                int(bucket["local_positive_union"]), states
            ),
            "heatmap_adds_positive_support_count": int(
                bucket["heatmap_adds_positive_support"]
            ),
            "heatmap_adds_positive_support_rate": _ratio(
                int(bucket["heatmap_adds_positive_support"]), states
            ),
            "native_mean_local_best_rate": _ratio(
                int(bucket["native_mean_local_best"]), states
            ),
        }

    state_count = int(counts["states"])
    return {
        "schema": SCHEMA,
        "audit_schema": records[0].get("schema"),
        "decision_status": "insufficient_local_only",
        "decision_reason": (
            "Local open-loop support is diagnostic. Go/no-go additionally requires "
            "one-deviation continuation labels and deployable-feature identifiability."
        ),
        "local_priority": {
            "type": "lexicographic_diagnostic",
            "fields": list(PRIORITY_FIELDS),
            "authoritative_navigation_success": False,
        },
        "overall": summarize_counts(counts),
        "strata": {
            name: summarize_counts(bucket)
            for name, bucket in sorted(strata.items())
        },
        "unique_treatments": {
            "mean": statistics.fmean(unique_counts),
            "median": statistics.median(unique_counts),
            "min": min(unique_counts),
            "max": max(unique_counts),
        },
        "native_mean_rank_in_mean_plus_samples": {
            "mean": statistics.fmean(native_mean_ranks),
            "median": statistics.median(native_mean_ranks),
            "best": min(native_mean_ranks),
            "worst": max(native_mean_ranks),
        },
        "nonlearning_baselines_better_than_native_mean_rate": {
            name: _ratio(int(value), state_count)
            for name, value in sorted(baseline_better.items())
        },
        "candidate_count_sensitivity": {
            group_name: {
                str(k): {
                    "states": int(bucket["states"]),
                    "positive_support_rate": _ratio(
                        int(bucket["positive"]), int(bucket["states"])
                    ),
                    "mean_unique_treatments": _ratio(
                        int(bucket["unique_treatments"]), int(bucket["states"])
                    ),
                    "mean_effective_stochastic_samples": _ratio(
                        int(bucket["effective_samples"]), int(bucket["states"])
                    ),
                }
                for k, bucket in sorted(group_buckets.items())
                if int(bucket["states"])
            }
            for group_name, group_buckets in k_stats.items()
            if any(int(bucket["states"]) for bucket in group_buckets.values())
        },
        "best_second_lexicographic_margin": {
            name: {
                "count": len(values),
                "median": statistics.median(values),
                "mean": statistics.fmean(values),
            }
            for name, values in sorted(margin_values.items())
        },
        "storage": {
            "shards": len(manifests),
            "records": state_count,
            "compressed_array_bytes": sum(
                int(manifest["array_bytes"]) for manifest in manifests
            ),
        },
        "required_next": [
            "one_deviation_native_continuation_at_horizons_1_3_5_end",
            "short_horizon_sign_stability_against_end",
            "candidate_only_vs_system2_vs_heatmap_predictability_probe",
            "matched_vs_shuffled_heatmap_probe",
            "false_positive_intervention_and_realizable_gain",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-shards", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.audit_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not 1 <= int(args.expected_shards) <= 64:
        raise ValueError("expected_shards must be in [1,64]")
    records, manifests = _read_records(root, int(args.expected_shards))
    result = summarize(records, manifests)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

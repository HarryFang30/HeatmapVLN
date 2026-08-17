#!/usr/bin/env python3
"""Strictly validate and summarize the corrected 100-step Task-4 pilot.

The corrected pilot has a shared isolated step-0 evaluation, isolated step-25
and step-50 evaluations for branches B/C, and the two final step-100 reports.
This utility verifies that those reports describe the same experiment before
building heatmap and S1-S2 retention trajectories.

Engineering feasibility and scientific evidence are deliberately separate.
The optional Task-3.5b data report can satisfy the scientific data gate; when
it is absent the scientific status is ``pending`` rather than implicitly
passing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


BRANCH_B = "heatmap-lora"
BRANCH_C = "joint-rehearsal"
STEPS = (0, 25, 50, 100)
EXPECTED_INTERVENTIONS = {
    "standard",
    "blank-images",
    "history-shuffle",
    "current-shuffle",
}
HEATMAP_TRAJECTORY_METRICS = (
    "loss",
    "visibility_auroc",
    "visibility_auprc",
    "visibility_f1",
    "visibility_precision",
    "visibility_recall",
    "visible_view_accuracy",
    "median_pixel_error",
    "median_u_error",
    "pck4",
    "pck8",
    "joint_median_pixel_error",
    "joint_pck4",
    "joint_pck8",
    "samples",
    "visible_history_count",
    "visible_view_count",
)
GENERATION_METRICS = (
    "format_valid",
    "action_valid",
    "category_match",
    "coord_hit",
    "view_hit",
    "stop_hit",
    "turn_hit",
)
CRITICAL_GENERATION_METRICS = (
    "format_valid",
    "category_match",
    "coord_hit",
    "view_hit",
)
SELECTION_KEYS = (
    "sample_count",
    "unique_physical_sample_count",
    "duplicate_physical_sample_count",
    "sample_identity_sha256",
    "sample_identities",
    "scenes",
    "category_counts",
)
DATASET_KEYS = (
    "clip_count",
    "scene_count",
    "scenes",
    "per_scene_clip_counts",
    "clip_identities",
    "clip_identity_sha256",
    "balanced_view_manifest",
)
STREAM_ALGORITHM = "sha256_epoch_rank_no_replacement_v1"
POOL_MODE = "full_source_index_including_stop_oversampling"
LOSS_REDUCTION = "mean_over_all_nonignored_shifted_batch_tokens"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmap-lora-final-report", required=True)
    parser.add_argument("--joint-final-report", required=True)
    parser.add_argument("--shared-step0-report", required=True)
    parser.add_argument("--heatmap-lora-step25-report", required=True)
    parser.add_argument("--heatmap-lora-step50-report", required=True)
    parser.add_argument("--joint-step25-report", required=True)
    parser.add_argument("--joint-step50-report", required=True)
    parser.add_argument("--task35b-report", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _hash_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _same_projection(
    reports: dict[str, dict[str, Any]],
    projection,
) -> bool:
    values = [projection(report) for report in reports.values()]
    return all(value == values[0] for value in values[1:])


def _contract_projection(report: dict[str, Any], key: str, fields: tuple[str, ...]) -> dict[str, Any]:
    value = report.get("contract", {}).get(key, {})
    return {field: value.get(field) for field in fields}


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _sft_selection_self_consistent(selection: dict[str, Any]) -> bool:
    identities = selection.get("sample_identities", [])
    counts = selection.get("category_counts", {})
    sample_count = int(selection.get("sample_count", -1))
    return bool(
        sample_count >= 0
        and len(identities) == sample_count
        and sum(int(counts.get(key, -1)) for key in ("pixel", "stop")) == sample_count
        and selection.get("sample_identity_sha256") == _hash_strings(identities)
        and int(selection.get("unique_physical_sample_count", -1)) == len(set(identities))
        and int(selection.get("duplicate_physical_sample_count", -1))
        == sample_count - len(set(identities))
    )


def _heatmap_selection_self_consistent(selection: dict[str, Any]) -> bool:
    identities = selection.get("sample_identities", [])
    return bool(
        len(identities) == int(selection.get("sample_count", -1))
        and selection.get("sample_identity_sha256")
        == hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest()
    )


def _dataset_self_consistent(dataset: dict[str, Any]) -> bool:
    identities = dataset.get("clip_identities", [])
    scenes = dataset.get("scenes", [])
    per_scene = dataset.get("per_scene_clip_counts", {})
    return bool(
        len(identities) == int(dataset.get("clip_count", -1))
        and len(scenes) == int(dataset.get("scene_count", -1))
        and sum(int(value) for value in per_scene.values()) == len(identities)
        and dataset.get("clip_identity_sha256") == _hash_strings(identities)
    )


def generation_coverage(report: dict[str, Any]) -> dict[str, Any]:
    phases: dict[str, Any] = {}
    for phase in ("generation_before", "generation_after"):
        metrics = report.get("sft_retention", {}).get(phase)
        if not isinstance(metrics, dict):
            phases[phase] = {"complete": False, "reason": "missing"}
            continue
        requested = int(metrics.get("requested_samples", -1))
        attempted = int(metrics.get("attempted_samples", -1))
        evaluated = int(metrics.get("samples", -1))
        errors = int(metrics.get("errors", -1))
        skipped = int(metrics.get("skipped_no_target", -1))
        complete = bool(
            metrics.get("complete_coverage") is True
            and requested > 0
            and requested == attempted == evaluated
            and errors == 0
            and skipped == 0
        )
        phases[phase] = {
            "complete": complete,
            "requested": requested,
            "attempted": attempted,
            "evaluated": evaluated,
            "errors": errors,
            "skipped_no_target": skipped,
        }
    return {
        "complete": all(phase.get("complete", False) for phase in phases.values()),
        "phases": phases,
    }


def _reported_checkpoint_step(report: dict[str, Any]) -> int | None:
    match = re.search(r"checkpoint_step_(\d{6})\.pth$", str(report.get("checkpoint", "")))
    return int(match.group(1)) if match else None


def _late_lora_is_frozen(report: dict[str, Any]) -> bool:
    if report.get("contract", {}).get("frozen_late_layers_unchanged") is not True:
        return False
    layers = report.get("lora_drift", {}).get("layers", {})
    for layer in range(21, 28):
        values = layers.get(str(layer), {})
        if int(values.get("changed_tensors", -1)) != 0:
            return False
        if float(values.get("parameter_delta_norm", float("nan"))) != 0.0:
            return False
    return True


def validate_planned_stream(stream: dict[str, Any]) -> dict[str, Any]:
    batches = stream.get("planned_batches", [])
    flattened_indices: list[int] = []
    flattened_identities: list[str] = []
    categories: Counter = Counter()
    batch_shapes_valid = True
    for batch_number, batch in enumerate(batches):
        indices = [int(index) for index in batch.get("dataset_indices", [])]
        identities = [str(value) for value in batch.get("sample_identities", [])]
        batch_categories = batch.get("category_counts", {})
        expected_start = batch_number * 4
        batch_shapes_valid &= bool(
            int(batch.get("epoch", -1)) == 0
            and int(batch.get("start_position", -1)) == expected_start
            and len(indices) == 4
            and len(identities) == 4
            and sum(int(batch_categories.get(key, 0)) for key in ("pixel", "stop")) == 4
        )
        flattened_indices.extend(indices)
        flattened_identities.extend(identities)
        categories.update({
            key: int(batch_categories.get(key, 0))
            for key in ("pixel", "stop")
        })

    recomputed_index_hash = _hash_strings(str(index) for index in flattened_indices)
    recomputed_identity_hash = _hash_strings(flattened_identities)
    checks = {
        "algorithm": stream.get("algorithm") == STREAM_ALGORITHM,
        "batch_size_four": int(stream.get("batch_size", -1)) == 4,
        "no_replacement_declared": stream.get("no_replacement_within_epoch") is True,
        "candidate_count_7995": int(stream.get("candidate_count", -1)) == 7995,
        "candidate_hash_present": _valid_sha256(stream.get("candidate_dataset_index_sha256")),
        "planned_steps_100": int(stream.get("planned_steps", -1)) == 100,
        "planned_sample_count_400": int(stream.get("planned_sample_count", -1)) == 400,
        "one_partial_epoch": int(stream.get("planned_epoch_count", -1)) == 1,
        "one_hundred_batches": len(batches) == 100,
        "batch_shapes_and_positions": batch_shapes_valid,
        "four_hundred_listed_samples": (
            len(flattened_indices) == len(flattened_identities) == 400
        ),
        "planned_indices_without_replacement": len(set(flattened_indices)) == 400,
        "planned_dataset_index_hash_recomputed": (
            stream.get("planned_dataset_index_sha256") == recomputed_index_hash
        ),
        "planned_identity_hash_recomputed": (
            stream.get("planned_sample_identity_sha256") == recomputed_identity_hash
        ),
        "planned_category_counts_recomputed": stream.get("planned_category_counts") == {
            "pixel": int(categories["pixel"]),
            "stop": int(categories["stop"]),
        },
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "recomputed": {
            "planned_dataset_index_sha256": recomputed_index_hash,
            "planned_sample_identity_sha256": recomputed_identity_hash,
            "planned_category_counts": {
                "pixel": int(categories["pixel"]),
                "stop": int(categories["stop"]),
            },
            "unique_planned_dataset_indices": len(set(flattened_indices)),
        },
    }


def validate_training_log_against_plan(
    report: dict[str, Any],
    stream: dict[str, Any],
    *,
    sft_executed: bool,
) -> bool:
    train_log = report.get("train_log", [])
    planned = stream.get("planned_batches", [])
    if len(train_log) != 100 or len(planned) != 100:
        return False
    for position, (record, expected) in enumerate(zip(train_log, planned), start=1):
        batch = record.get("sft_rehearsal_batch", {})
        if int(record.get("step", -1)) != position:
            return False
        if batch.get("executed") is not sft_executed:
            return False
        for key in (
            "epoch",
            "start_position",
            "dataset_indices",
            "sample_identities",
            "category_counts",
        ):
            if batch.get(key) != expected.get(key):
                return False
        if sft_executed:
            tokens = record.get("lm_sample_label_tokens", [])
            if len(tokens) != 4 or sum(int(value) for value in tokens) != int(
                record.get("lm_label_tokens", -1)
            ):
                return False
        elif int(record.get("lm_label_tokens", -1)) != 0:
            return False
    return True


def _telemetry_valid(report: dict[str, Any], *, expected_sft_steps: int) -> bool:
    telemetry = report.get("contract", {}).get("training_telemetry", {})
    return bool(
        int(telemetry.get("record_count", -1)) == 100
        and int(telemetry.get("expected_record_count", -1)) == 100
        and telemetry.get("every_optimizer_step_recorded") is True
        and int(telemetry.get("executed_sft_steps", -1)) == expected_sft_steps
        and int(telemetry.get("expected_executed_sft_steps", -1)) == expected_sft_steps
        and (
            int(telemetry.get("total_sft_label_tokens", -1)) > 0
            if expected_sft_steps
            else int(telemetry.get("total_sft_label_tokens", -1)) == 0
        )
    )


def validate_contract(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    b_final = reports["b100"]
    c_final = reports["c100"]
    step0 = reports["step0"]
    evaluation_reports = {
        key: reports[key]
        for key in ("step0", "b25", "b50", "c25", "c50")
    }
    b_stream = b_final.get("contract", {}).get("sft_rehearsal_stream", {})
    c_stream = c_final.get("contract", {}).get("sft_rehearsal_stream", {})
    stream_validation = validate_planned_stream(b_stream)
    coverages = {name: generation_coverage(report) for name, report in reports.items()}
    requested_counts = {
        phase.get("requested")
        for coverage in coverages.values()
        for phase in coverage["phases"].values()
        if phase.get("requested", -1) > 0
    }

    b_contract = b_final.get("contract", {})
    c_contract = c_final.get("contract", {})
    step0_contract = step0.get("contract", {})
    b_rehearsal = b_contract.get("sft_rehearsal", {})
    c_rehearsal = c_contract.get("sft_rehearsal", {})
    b_milestones = b_contract.get("milestones", {})
    c_milestones = c_contract.get("milestones", {})

    checks = {
        "correct_final_modes": (
            b_final.get("mode") == BRANCH_B
            and c_final.get("mode") == BRANCH_C
        ),
        "isolated_reports_are_eval_only": all(
            report.get("mode") == "head-only" and int(report.get("train_steps", -1)) == 0
            for report in evaluation_reports.values()
        ),
        "isolated_checkpoint_steps_match": (
            _reported_checkpoint_step(reports["step0"]) == 0
            and _reported_checkpoint_step(reports["b25"]) == 25
            and _reported_checkpoint_step(reports["b50"]) == 50
            and _reported_checkpoint_step(reports["c25"]) == 25
            and _reported_checkpoint_step(reports["c50"]) == 50
        ),
        "midpoint_branch_provenance": (
            "/heatmap-lora/" in str(reports["b25"].get("checkpoint", ""))
            and "/heatmap-lora/" in str(reports["b50"].get("checkpoint", ""))
            and "/joint-rehearsal/" in str(reports["c25"].get("checkpoint", ""))
            and "/joint-rehearsal/" in str(reports["c50"].get("checkpoint", ""))
        ),
        "same_initial_lora": (
            b_contract.get("initial_lora_hash")
            == c_contract.get("initial_lora_hash")
            == step0_contract.get("final_lora_hash")
        ),
        "same_initial_head": (
            b_contract.get("fresh_initial_head_hash")
            == b_contract.get("initial_head_hash")
            == b_contract.get("starting_head_hash")
            == c_contract.get("fresh_initial_head_hash")
            == c_contract.get("initial_head_hash")
            == c_contract.get("starting_head_hash")
            == step0_contract.get("final_head_hash")
        ),
        "all_lora_loads_complete": all(
            int(report.get("load", {}).get("matched_lora_tensors", -1)) == 224
            and int(report.get("contract", {}).get("all_lora_tensors", -1)) == 224
            for report in reports.values()
        ),
        "same_heatmap_train_selection": _same_projection(
            reports,
            lambda report: _contract_projection(report, "heatmap_train", SELECTION_KEYS),
        ),
        "same_heatmap_val_selection": _same_projection(
            reports,
            lambda report: _contract_projection(report, "heatmap_val", SELECTION_KEYS),
        ),
        "same_sft_dataset": _same_projection(
            reports,
            lambda report: _contract_projection(report, "sft_dataset", DATASET_KEYS),
        ),
        "same_sft_scene_partition": _same_projection(
            reports,
            lambda report: report.get("contract", {}).get("sft_scene_partition"),
        ),
        "sft_holdout_partition_valid": all(
            len(partition.get("holdout_scenes", []))
            == int(partition.get("requested_holdout_scene_count", -1))
            == 7
            and not (
                set(partition.get("holdout_scenes", []))
                & set(partition.get("rehearsal_scenes", []))
            )
            and (
                set(partition.get("holdout_scenes", []))
                | set(partition.get("rehearsal_scenes", []))
            )
            == set(report.get("contract", {}).get("sft_dataset", {}).get("scenes", []))
            for report in reports.values()
            for partition in [
                report.get("contract", {}).get("sft_scene_partition", {})
            ]
        ),
        "same_sft_holdout_selection": _same_projection(
            reports,
            lambda report: _contract_projection(report, "sft_retention", SELECTION_KEYS),
        ),
        "same_full_sft_rehearsal_selection": _same_projection(
            reports,
            lambda report: _contract_projection(report, "sft_rehearsal", SELECTION_KEYS),
        ),
        "heatmap_selection_hashes_self_consistent": all(
            _heatmap_selection_self_consistent(
                report.get("contract", {}).get(stream, {})
            )
            for report in reports.values()
            for stream in ("heatmap_train", "heatmap_val")
        ),
        "sft_dataset_manifests_self_consistent": all(
            _dataset_self_consistent(report.get("contract", {}).get("sft_dataset", {}))
            for report in reports.values()
        ),
        "sft_selection_hashes_self_consistent": all(
            _sft_selection_self_consistent(
                report.get("contract", {}).get(stream, {})
            )
            for report in reports.values()
            for stream in ("sft_rehearsal", "sft_retention")
        ),
        "full_7995_candidate_pool": all(
            int(selection.get("sample_count", -1)) == 7995
            and int(selection.get("full_candidate_count_before_optional_cap", -1)) == 7995
            and selection.get("pool_mode") == POOL_MODE
            for selection in (b_rehearsal, c_rehearsal)
        ),
        "same_rehearsal_stream": b_stream == c_stream,
        "planned_stream_self_consistent": stream_validation["passed"],
        "hundred_training_steps": (
            int(b_final.get("train_steps", -1)) == 100
            and int(c_final.get("train_steps", -1)) == 100
        ),
        "same_optimization_hyperparameters": b_final.get("optimization") == c_final.get("optimization"),
        "corrected_sft_hyperparameters": all(
            report.get("optimization", {}).get("sft_batch_size") == 4
            and report.get("optimization", {}).get("sft_pool_mode") == POOL_MODE
            and report.get("optimization", {}).get("sft_stream_algorithm") == STREAM_ALGORITHM
            and report.get("optimization", {}).get("sft_loss_reduction") == LOSS_REDUCTION
            for report in (b_final, c_final)
        ),
        "milestone_plan_is_0_25_50_100": all(
            milestone.get("requested_steps") == [0, 25, 50, 100]
            and milestone.get("effective_steps") == [0, 25, 50, 100]
            and milestone.get("midpoint_evaluation_in_training_process") is False
            for milestone in (b_milestones, c_milestones)
        ),
        "trainable_lora_scope_0_to_20": (
            b_contract.get("trainable_lora_layers")
            == c_contract.get("trainable_lora_layers")
            == list(range(21))
            and int(b_contract.get("max_trainable_lora_layer", -1)) == 20
            and int(c_contract.get("max_trainable_lora_layer", -1)) == 20
        ),
        "lora_layers_21_to_27_frozen": (
            _late_lora_is_frozen(b_final) and _late_lora_is_frozen(c_final)
        ),
        "b_train_log_matches_planned_stream": validate_training_log_against_plan(
            b_final, b_stream, sft_executed=False
        ),
        "c_train_log_matches_planned_stream": validate_training_log_against_plan(
            c_final, c_stream, sft_executed=True
        ),
        "b_training_telemetry_complete": _telemetry_valid(
            b_final, expected_sft_steps=0
        ),
        "c_training_telemetry_complete": _telemetry_valid(
            c_final, expected_sft_steps=100
        ),
        "all_generation_coverage_complete": all(
            coverage["complete"] for coverage in coverages.values()
        ),
        "same_generation_sample_count": len(requested_counts) == 1,
        "same_heatmap_interventions": (
            _same_projection(reports, lambda report: sorted(report.get("heatmap_evaluations", {})))
            and EXPECTED_INTERVENTIONS
            <= set(step0.get("heatmap_evaluations", {}))
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "planned_stream": stream_validation,
        "generation_coverage": coverages,
    }


def _compact_heatmap(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        metric: metrics.get(metric)
        for metric in HEATMAP_TRAJECTORY_METRICS
        if metric in metrics
    }


def _compact_ce(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics.get(key)
        for key in ("loss", "perplexity", "samples", "label_tokens")
    }


def _compact_generation(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "samples",
        "requested_samples",
        "attempted_samples",
        "errors",
        "skipped_no_target",
        "complete_coverage",
        *GENERATION_METRICS,
        "counts",
    )
    return {key: metrics.get(key) for key in keys}


def trajectory_point(report: dict[str, Any]) -> dict[str, Any]:
    evaluations = report["heatmap_evaluations"]
    retention = report["sft_retention"]
    return {
        "heatmap": {
            condition: _compact_heatmap(metrics)
            for condition, metrics in evaluations.items()
        },
        "sft": {
            "teacher_forced": _compact_ce(retention["teacher_forced_after"]),
            "generation": _compact_generation(retention["generation_after"]),
        },
    }


def extract_trajectories(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        BRANCH_B: {
            "0": trajectory_point(reports["step0"]),
            "25": trajectory_point(reports["b25"]),
            "50": trajectory_point(reports["b50"]),
            "100": trajectory_point(reports["b100"]),
        },
        BRANCH_C: {
            "0": trajectory_point(reports["step0"]),
            "25": trajectory_point(reports["c25"]),
            "50": trajectory_point(reports["c50"]),
            "100": trajectory_point(reports["c100"]),
        },
    }


def engineering_gate(
    reports: dict[str, dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    b_standard = reports["b100"]["heatmap_evaluations"]["standard"]
    c_standard = reports["c100"]["heatmap_evaluations"]["standard"]
    baseline_ce = float(
        reports["step0"]["sft_retention"]["teacher_forced_after"]["loss"]
    )
    final_ce = float(
        reports["c100"]["sft_retention"]["teacher_forced_after"]["loss"]
    )
    ce_relative_increase = (
        (final_ce - baseline_ce) / baseline_ce
        if baseline_ce > 0.0
        else float("inf")
    )
    baseline_generation = reports["step0"]["sft_retention"]["generation_after"]
    final_generation = reports["c100"]["sft_retention"]["generation_after"]
    generation_deltas = {
        metric: float(final_generation[metric]) - float(baseline_generation[metric])
        for metric in GENERATION_METRICS
    }
    critical_drop = min(
        generation_deltas[metric]
        for metric in CRITICAL_GENERATION_METRICS
    )
    b_median = float(b_standard["median_pixel_error"])
    c_median = float(c_standard["median_pixel_error"])
    median_ratio = (
        c_median / b_median
        if b_median > 0.0
        else (1.0 if c_median == 0.0 else float("inf"))
    )
    heatmap = {
        "median_ratio_c_over_b": median_ratio,
        "pck8_delta_c_minus_b": float(c_standard["pck8"]) - float(b_standard["pck8"]),
        "joint_pck8_delta_c_minus_b": (
            float(c_standard["joint_pck8"]) - float(b_standard["joint_pck8"])
        ),
    }
    thresholds = {
        "median_ratio_c_over_b_max": 1.10,
        "pck8_delta_c_minus_b_min": -0.05,
        "joint_pck8_delta_c_minus_b_min": -0.05,
        "c_sft_ce_relative_increase_max": 0.10,
        "c_critical_generation_drop_min": -0.02,
    }
    checks = {
        "contract": contract["passed"],
        "heatmap_median_not_worse": median_ratio <= thresholds["median_ratio_c_over_b_max"],
        "heatmap_pck8_not_worse": heatmap["pck8_delta_c_minus_b"] >= thresholds["pck8_delta_c_minus_b_min"],
        "heatmap_joint_pck8_not_worse": heatmap["joint_pck8_delta_c_minus_b"] >= thresholds["joint_pck8_delta_c_minus_b_min"],
        "c_retention_ce_within_ten_percent": ce_relative_increase <= thresholds["c_sft_ce_relative_increase_max"],
        "generation_coverage": all(
            value["complete"] for value in contract["generation_coverage"].values()
        ),
        "c_critical_generation_drop": critical_drop >= thresholds["c_critical_generation_drop_min"],
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "thresholds": thresholds,
        "checks": checks,
        "metrics": {
            "heatmap_c_vs_b": heatmap,
            "c_retention_ce": {
                "step0_loss": baseline_ce,
                "step100_loss": final_ce,
                "relative_increase": ce_relative_increase,
            },
            "c_generation": {
                "delta_step100_minus_step0": generation_deltas,
                "critical_metrics": list(CRITICAL_GENERATION_METRICS),
                "critical_drop": critical_drop,
            },
        },
    }


def scientific_gate(task35b: dict[str, Any] | None) -> dict[str, Any]:
    if task35b is None:
        return {
            "status": "pending",
            "passed": None,
            "checks": {
                "task35b_report_present": False,
                "selection_ready_for_diagnostic": None,
                "empirical_prior_weakened": None,
            },
            "note": "Task-3.5b report was not supplied; scientific evidence is pending.",
        }
    if task35b.get("task") != "task35b_debiased_data_diagnostic":
        raise RuntimeError(f"Unexpected Task-3.5b report task={task35b.get('task')!r}")
    selection_ready = task35b.get("selection_ready_for_diagnostic") is True
    comparison = (
        task35b.get("empirical_prior_strength", {})
        .get("comparison", {})
    )
    prior_weakened = bool(
        comparison.get("available") is True
        and comparison.get("shortcut_reduction", {}).get(
            "empirical_prior_weaker_on_all_localization_checks"
        ) is True
    )
    passed = bool(selection_ready and prior_weakened)
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checks": {
            "task35b_report_present": True,
            "selection_ready_for_diagnostic": selection_ready,
            "empirical_prior_weakened": prior_weakened,
        },
        "prior_comparison": comparison,
    }


def _trajectory_rows(trajectories: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for branch in (BRANCH_B, BRANCH_C):
        for step in STEPS:
            point = trajectories[branch][str(step)]
            standard = point["heatmap"]["standard"]
            ce = point["sft"]["teacher_forced"]
            generation = point["sft"]["generation"]
            row = {
                "branch": branch,
                "step": step,
                **{f"heatmap_{key}": standard.get(key) for key in HEATMAP_TRAJECTORY_METRICS},
                "sft_ce_loss": ce.get("loss"),
                "sft_ce_perplexity": ce.get("perplexity"),
                **{f"generation_{key}": generation.get(key) for key in GENERATION_METRICS},
            }
            for intervention in sorted(EXPECTED_INTERVENTIONS - {"standard"}):
                metrics = point["heatmap"][intervention]
                row[f"{intervention}_pck8"] = metrics.get("pck8")
                row[f"{intervention}_joint_pck8"] = metrics.get("joint_pck8")
            rows.append(row)
    return rows


def build_summary(
    reports: dict[str, dict[str, Any]],
    *,
    task35b: dict[str, Any] | None,
) -> dict[str, Any]:
    contract = validate_contract(reports)
    if not contract["passed"]:
        failed = [name for name, passed in contract["checks"].items() if not passed]
        raise RuntimeError(f"Corrected Task-4 contract failed: {failed}")
    trajectories = extract_trajectories(reports)
    engineering = engineering_gate(reports, contract)
    scientific = scientific_gate(task35b)
    return {
        "task": "task4_corrected_pilot_summary",
        "contract": contract,
        "trajectories": trajectories,
        "engineering_gate": engineering,
        "scientific_gate": scientific,
        "verdict": {
            "engineering_status": engineering["status"],
            "scientific_status": scientific["status"],
            "advance_to_task5": bool(
                engineering["passed"] and scientific.get("passed") is True
            ),
        },
    }


def main() -> int:
    args = parse_args()
    paths = {
        "b100": args.heatmap_lora_final_report,
        "c100": args.joint_final_report,
        "step0": args.shared_step0_report,
        "b25": args.heatmap_lora_step25_report,
        "b50": args.heatmap_lora_step50_report,
        "c25": args.joint_step25_report,
        "c50": args.joint_step50_report,
    }
    reports = {name: load_json(path) for name, path in paths.items()}
    task35b = load_json(args.task35b_report) if args.task35b_report else None
    summary = build_summary(reports, task35b=task35b)
    summary["reports"] = {
        name: str(Path(path).resolve())
        for name, path in paths.items()
    }
    summary["reports"]["task35b"] = (
        str(Path(args.task35b_report).resolve()) if args.task35b_report else None
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "task4_corrected_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=True)
    temporary.replace(summary_path)

    rows = _trajectory_rows(summary["trajectories"])
    with (output_dir / "task4_corrected_trajectory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

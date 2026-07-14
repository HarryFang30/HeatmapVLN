#!/usr/bin/env python3
"""Validate and summarize the matched three-branch Task-4 joint pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


HEATMAP_METRICS = (
    "loss",
    "visibility_auroc",
    "visibility_auprc",
    "visible_view_accuracy",
    "median_pixel_error",
    "pck4",
    "pck8",
    "joint_pck4",
    "joint_pck8",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-only-report", required=True)
    parser.add_argument("--heatmap-lora-report", required=True)
    parser.add_argument("--joint-report", required=True)
    parser.add_argument("--task35-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _contract_hash(report: dict[str, Any], stream: str) -> str:
    return str(report["contract"][stream]["sample_identity_sha256"])


def generation_coverage(report: dict[str, Any]) -> dict[str, Any]:
    phases: dict[str, Any] = {}
    for phase in ("generation_before", "generation_after"):
        generation = report.get("sft_retention", {}).get(phase)
        if not isinstance(generation, dict):
            phases[phase] = {"complete": False, "reason": "missing"}
            continue
        requested = int(generation.get("requested_samples", -1))
        attempted = int(generation.get("attempted_samples", -1))
        evaluated = int(generation.get("samples", -1))
        errors = int(generation.get("errors", -1))
        skipped = int(generation.get("skipped_no_target", -1))
        complete = bool(
            generation.get("complete_coverage", False)
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
        "complete": all(value["complete"] for value in phases.values()),
        "phases": phases,
    }


def sft_category_stratified(report: dict[str, Any], stream: str) -> bool:
    selection = report.get("contract", {}).get(stream, {})
    sample_count = int(selection.get("sample_count", -1))
    counts = selection.get("category_counts", {})
    pixel = int(counts.get("pixel", -1))
    stop = int(counts.get("stop", -1))
    expected_stop = min(max(1, round(sample_count * 0.25)), sample_count - 1)
    return bool(
        sample_count >= 2
        and pixel > 0
        and stop > 0
        and pixel + stop == sample_count
        and stop == expected_stop
    )


def validate_contract(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    head = reports["head-only"]
    heatmap_lora = reports["heatmap-lora"]
    joint = reports["joint-rehearsal"]
    generation = {
        mode: generation_coverage(report)
        for mode, report in reports.items()
    }
    generation_requested = {
        phase["requested"]
        for coverage in generation.values()
        for phase in coverage["phases"].values()
        if "requested" in phase
    }
    checks = {
        "correct_modes": all(report.get("mode") == mode for mode, report in reports.items()),
        "same_fresh_head_initialization": len(
            {report["contract"]["fresh_initial_head_hash"] for report in reports.values()}
        ) == 1,
        "same_recorded_initial_head": len(
            {report["contract"]["initial_head_hash"] for report in reports.values()}
        ) == 1,
        "same_initial_lora": len(
            {report["contract"]["initial_lora_hash"] for report in reports.values()}
        ) == 1,
        "all_224_lora_loaded": all(
            int(report.get("load", {}).get("matched_lora_tensors", -1)) == 224
            and int(report["contract"].get("all_lora_tensors", -1)) == 224
            for report in reports.values()
        ),
        "same_heatmap_train_samples": len(
            {_contract_hash(report, "heatmap_train") for report in reports.values()}
        ) == 1,
        "same_heatmap_val_samples": len(
            {_contract_hash(report, "heatmap_val") for report in reports.values()}
        ) == 1,
        "same_sft_rehearsal_samples": len(
            {_contract_hash(report, "sft_rehearsal") for report in reports.values()}
        ) == 1,
        "same_sft_retention_samples": len(
            {_contract_hash(report, "sft_retention") for report in reports.values()}
        ) == 1,
        "same_sft_dataset_clip_set": len(
            {
                report["contract"]["sft_dataset"]["clip_identity_sha256"]
                for report in reports.values()
            }
        ) == 1,
        "same_sft_dataset_scene_set": len(
            {
                tuple(report["contract"]["sft_dataset"]["scenes"])
                for report in reports.values()
            }
        ) == 1,
        "requested_sft_holdout_satisfied": all(
            len(report["contract"]["sft_scene_partition"]["holdout_scenes"])
            == int(
                report["contract"]["sft_scene_partition"][
                    "requested_holdout_scene_count"
                ]
            )
            for report in reports.values()
        ),
        "sft_pixel_stop_stratification": all(
            sft_category_stratified(report, stream)
            for report in reports.values()
            for stream in ("sft_rehearsal", "sft_retention")
        ),
        "head_only_reuses_trained_head": head["contract"]["starting_head_hash"] != head["contract"]["fresh_initial_head_hash"],
        "adaptation_branches_start_fresh": all(
            report["contract"]["starting_head_hash"] == report["contract"]["fresh_initial_head_hash"]
            for report in (heatmap_lora, joint)
        ),
        "adaptation_steps_match": heatmap_lora.get("train_steps") == joint.get("train_steps") == 500,
        "same_adaptation_hyperparameters": all(
            heatmap_lora["optimization"].get(key) == joint["optimization"].get(key)
            for key in ("head_learning_rate", "lora_learning_rate", "weight_decay", "grad_clip")
        ),
        "reachable_lora_scope_matches": (
            heatmap_lora["contract"]["trainable_lora_layers"]
            == joint["contract"]["trainable_lora_layers"]
            == list(range(21))
        ),
        "late_lora_frozen": all(
            bool(report["contract"]["frozen_late_layers_unchanged"])
            for report in reports.values()
        ),
        "generation_complete_without_errors": all(
            coverage["complete"] for coverage in generation.values()
        ),
        "same_generation_sample_count": len(generation_requested) == 1,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "generation_coverage": generation,
    }


def retention_deltas(report: dict[str, Any]) -> dict[str, Any]:
    retention = report["sft_retention"]
    before_ce = retention["teacher_forced_before"]
    after_ce = retention["teacher_forced_after"]
    before_generation = retention.get("generation_before")
    after_generation = retention.get("generation_after")
    output = {
        "teacher_forced_loss_absolute": after_ce["loss"] - before_ce["loss"],
        "teacher_forced_loss_relative": (
            (after_ce["loss"] - before_ce["loss"]) / before_ce["loss"]
            if before_ce["loss"] > 0.0
            else float("nan")
        ),
    }
    if before_generation is not None and after_generation is not None:
        output["generation"] = {
            metric: after_generation[metric] - before_generation[metric]
            for metric in GENERATION_METRICS
        }
    else:
        output["generation"] = None
    return output


def main() -> int:
    args = parse_args()
    reports = {
        "head-only": load_json(args.head_only_report),
        "heatmap-lora": load_json(args.heatmap_lora_report),
        "joint-rehearsal": load_json(args.joint_report),
    }
    task35 = load_json(args.task35_summary)
    contract = validate_contract(reports)
    if not contract["passed"]:
        raise RuntimeError(f"Task-4 matched contract failed: {contract['checks']}")

    rows = []
    for mode, report in reports.items():
        row = {"mode": mode}
        standard = report["heatmap_evaluations"]["standard"]
        row.update({f"heatmap_{metric}": standard.get(metric) for metric in HEATMAP_METRICS})
        retention = retention_deltas(report)
        row["sft_ce_relative_delta"] = retention["teacher_forced_loss_relative"]
        generation = retention["generation"] or {}
        row.update({f"sft_generation_{metric}_delta": generation.get(metric) for metric in GENERATION_METRICS})
        rows.append(row)

    heatmap_lora = reports["heatmap-lora"]["heatmap_evaluations"]["standard"]
    joint = reports["joint-rehearsal"]["heatmap_evaluations"]["standard"]
    no_signal_effect = task35["effect"]
    task35_passed = bool(task35["verdict"]["sample_specific_localization_passed"])
    joint_retention = retention_deltas(reports["joint-rehearsal"])
    generation_delta = joint_retention["generation"]
    joint_generation_complete = generation_coverage(reports["joint-rehearsal"])["complete"]
    generation_critical_drop = None
    if generation_delta is not None:
        generation_critical_drop = min(
            generation_delta[metric]
            for metric in ("format_valid", "category_match", "coord_hit", "view_hit")
        )

    heatmap_preserved_vs_lora = {
        "median_ratio_joint_over_heatmap_lora": (
            joint["median_pixel_error"] / heatmap_lora["median_pixel_error"]
            if heatmap_lora["median_pixel_error"] > 0.0
            else float("inf")
        ),
        "pck8_delta_joint_minus_heatmap_lora": joint["pck8"] - heatmap_lora["pck8"],
        "joint_pck8_delta_joint_minus_heatmap_lora": (
            joint["joint_pck8"] - heatmap_lora["joint_pck8"]
        ),
    }
    conflict_values = [
        record["cosine"]
        for record in reports["joint-rehearsal"].get("gradient_conflict", [])
        if np.isfinite(record.get("cosine", float("nan")))
    ]
    gradient_conflict = {
        "measurements": len(conflict_values),
        "mean_cosine": float(np.mean(conflict_values)) if conflict_values else float("nan"),
        "negative_fraction": (
            float(np.mean(np.asarray(conflict_values) < 0.0)) if conflict_values else float("nan")
        ),
    }

    thresholds = {
        "task35_sample_specific_localization": True,
        "joint_median_no_worse_than_heatmap_lora_ratio": 1.10,
        "joint_pck8_no_worse_than_heatmap_lora_absolute": -0.05,
        "sft_teacher_forced_relative_loss_increase": 0.10,
        "sft_generation_critical_absolute_drop": -0.02,
    }
    pilot_passed = bool(
        task35_passed
        and heatmap_preserved_vs_lora["median_ratio_joint_over_heatmap_lora"] <= 1.10
        and heatmap_preserved_vs_lora["pck8_delta_joint_minus_heatmap_lora"] >= -0.05
        and joint_retention["teacher_forced_loss_relative"] <= 0.10
        and joint_generation_complete
        and generation_critical_drop is not None
        and generation_critical_drop >= -0.02
    )
    verdict = {
        "thresholds": thresholds,
        "pilot_passed": pilot_passed,
        "advance_to_task5": pilot_passed,
        "note": (
            "Task4 only establishes a feasible joint update. Fresh-probe/head-swap evidence is still required in Task5."
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "task4_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "contract": contract,
        "rows": rows,
        "task35_passed": task35_passed,
        "task35_effect": no_signal_effect,
        "heatmap_preserved_vs_lora": heatmap_preserved_vs_lora,
        "joint_retention": joint_retention,
        "joint_generation_complete": joint_generation_complete,
        "joint_generation_critical_drop": generation_critical_drop,
        "gradient_conflict": gradient_conflict,
        "verdict": verdict,
        "reports": {
            mode: str(Path(path).resolve())
            for mode, path in {
                "head-only": args.head_only_report,
                "heatmap-lora": args.heatmap_lora_report,
                "joint-rehearsal": args.joint_report,
            }.items()
        },
    }
    with (output_dir / "task4_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

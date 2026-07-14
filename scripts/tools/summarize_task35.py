#!/usr/bin/env python3
"""Validate Task-3.5 contracts and compare Full against two no-signal nulls."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

METRICS = (
    "loss",
    "visibility_auroc",
    "visibility_auprc",
    "visibility_f1",
    "visible_view_accuracy",
    "median_pixel_error",
    "pck4",
    "pck8",
    "joint_median_pixel_error",
    "joint_pck4",
    "joint_pck8",
)

MEDIAN_RELATIVE_IMPROVEMENT_THRESHOLD = 0.20
PCK8_ABSOLUTE_DELTA_THRESHOLD = 0.10
JOINT_PCK8_ABSOLUTE_DELTA_THRESHOLD = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-report", required=True)
    parser.add_argument("--no-input-report", required=True)
    parser.add_argument("--empirical-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prediction_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    embedded = report.get("evaluations", {}).get("standard", {}).get("prediction_records")
    if embedded is not None:
        return embedded
    artifact = report.get("artifacts", {}).get("compact_predictions")
    if not artifact:
        raise RuntimeError(f"Report {report.get('mode')} has no compact predictions")
    return load_json(artifact)


def canonical_selection_hash(report: dict[str, Any], split: str) -> str:
    contract = report.get("selection_contract", {}).get(split, {})
    if contract.get("sample_identity_sha256"):
        return str(contract["sample_identity_sha256"])
    selection = report.get("selection", {})
    key = f"{split}_sample_identity_hash"
    if selection.get(key):
        return str(selection[key])
    raise RuntimeError(f"Report {report.get('mode')} lacks {split} selection hash")


def validate_contract(
    full: dict[str, Any],
    no_input: dict[str, Any],
    empirical: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "full_mode": full.get("mode") == "full",
        "no_input_mode": no_input.get("mode") == "no-input",
        "empirical_mode": empirical.get("mode") == "empirical-prior",
        "same_seed": len({full.get("seed"), no_input.get("seed"), empirical.get("seed")}) == 1,
        "same_train_count": len(
            {full.get("train_samples"), no_input.get("train_samples"), empirical.get("train_samples")}
        ) == 1,
        "same_val_count": len(
            {full.get("val_samples"), no_input.get("val_samples"), empirical.get("val_samples")}
        ) == 1,
        "same_history": len(
            {full.get("num_history"), no_input.get("num_history"), empirical.get("num_history")}
        ) == 1,
        "same_clip_snapshot": len(
            {full.get("max_clip_id"), no_input.get("max_clip_id"), empirical.get("max_clip_id")}
        ) == 1,
        "same_train_selection": len(
            {
                canonical_selection_hash(full, "train"),
                canonical_selection_hash(no_input, "train"),
                canonical_selection_hash(empirical, "train"),
            }
        ) == 1,
        "same_val_selection": len(
            {
                canonical_selection_hash(full, "val"),
                canonical_selection_hash(no_input, "val"),
                canonical_selection_hash(empirical, "val"),
            }
        ) == 1,
        "same_initial_head": full.get("initial_head_hash") == no_input.get("initial_head_hash"),
        "same_checkpoint_for_learned_heads": full.get("checkpoint") == no_input.get("checkpoint"),
        "learned_heads_frozen_qwen": (
            int(full.get("trainable_qwen_tensors", -1)) == 0
            and int(no_input.get("trainable_qwen_tensors", -1)) == 0
        ),
        "all_lora_loaded": (
            int(full.get("load", {}).get("matched_lora_tensors", -1)) == 224
            and int(no_input.get("load", {}).get("matched_lora_tensors", -1)) == 224
        ),
        "scene_disjoint": (
            bool(full.get("selection_contract", {}).get("scene_disjoint"))
            and bool(no_input.get("selection_contract", {}).get("scene_disjoint"))
            and bool(empirical.get("scene_disjoint"))
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def metrics_from_compact(
    records: list[dict[str, Any]],
    sample_indices: np.ndarray,
    *,
    history_slot: int | None = None,
) -> dict[str, float]:
    view_correct = 0
    histories = 0
    oracle_errors = []
    joint_errors = []
    for sample_index in sample_indices.tolist():
        record = records[sample_index]
        visibility = np.asarray(record["visibility_logits"], dtype=np.float64)
        gt_visibility = np.asarray(record["gt_visibility"], dtype=np.float64) > 0.5
        pred_xy = np.asarray(record["pred_xy"], dtype=np.float64)
        gt_xy = np.asarray(record["gt_xy"], dtype=np.float64)
        if history_slot is None:
            history_indices = range(gt_visibility.shape[0])
        else:
            if history_slot < 0 or history_slot >= gt_visibility.shape[0]:
                raise RuntimeError(
                    f"history slot {history_slot} is outside sample "
                    f"{record.get('sample_id')!r} with {gt_visibility.shape[0]} slots"
                )
            history_indices = (history_slot,)
        for history_index in history_indices:
            positive_views = np.flatnonzero(gt_visibility[history_index])
            if positive_views.size == 0:
                continue
            histories += 1
            selected_view = int(visibility[history_index].argmax())
            if selected_view in positive_views:
                view_correct += 1
                joint_errors.append(
                    float(np.linalg.norm(pred_xy[history_index, selected_view] - gt_xy[history_index, selected_view]))
                )
            else:
                joint_errors.append(float("inf"))
            for view_index in positive_views:
                oracle_errors.append(
                    float(np.linalg.norm(pred_xy[history_index, view_index] - gt_xy[history_index, view_index]))
                )
    oracle = np.asarray(oracle_errors, dtype=np.float64)
    joint = np.asarray(joint_errors, dtype=np.float64)
    return {
        "visible_history_count": int(histories),
        "visible_view_accuracy": view_correct / max(histories, 1),
        "median_pixel_error": float(np.median(oracle)) if oracle.size else float("nan"),
        "pck8": float((oracle <= 8.0).mean()) if oracle.size else float("nan"),
        "joint_pck8": float((joint <= 8.0).mean()) if joint.size else float("nan"),
    }


def history_slot_count(records: list[dict[str, Any]]) -> int:
    """Validate compact-record slot shapes and return their common count."""
    if not records:
        raise RuntimeError("Cannot compute history-slot diagnostics from zero records")
    expected: int | None = None
    for record in records:
        leading_sizes = {}
        for field in ("visibility_logits", "gt_visibility", "pred_xy", "gt_xy"):
            value = np.asarray(record[field])
            if value.ndim < 2:
                raise RuntimeError(
                    f"Compact field {field} has invalid shape {value.shape} "
                    f"for sample {record.get('sample_id')!r}"
                )
            leading_sizes[field] = int(value.shape[0])
        if len(set(leading_sizes.values())) != 1:
            raise RuntimeError(
                f"Compact history-slot shape mismatch for {record.get('sample_id')!r}: "
                f"{leading_sizes}"
            )
        sample_slots = next(iter(leading_sizes.values()))
        if expected is None:
            expected = sample_slots
        elif sample_slots != expected:
            raise RuntimeError(
                "Compact records have inconsistent history-slot counts: "
                f"expected {expected}, got {sample_slots} for {record.get('sample_id')!r}"
            )
    assert expected is not None
    return expected


def _stronger_null_for_metric(
    slot_metrics: dict[str, dict[str, float]],
    metric: str,
    *,
    higher_is_stronger: bool,
) -> dict[str, Any]:
    candidates = [
        (mode, float(slot_metrics[mode][metric]))
        for mode in ("no-input", "empirical-prior")
        if math.isfinite(float(slot_metrics[mode][metric]))
    ]
    if not candidates:
        return {"value": float("nan"), "source_modes": []}
    select = max if higher_is_stronger else min
    best_value = select(value for _mode, value in candidates)
    source_modes = [
        mode
        for mode, value in candidates
        if math.isclose(value, best_value, rel_tol=0.0, abs_tol=1e-12)
    ]
    return {"value": float(best_value), "source_modes": source_modes}


def build_per_history_slot_diagnostic(
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a descriptive slot-wise breakdown from saved predictions only."""
    mode_records = {
        mode: prediction_records(report)
        for mode, report in reports.items()
    }
    full_ids = [record["sample_id"] for record in mode_records["full"]]
    for mode, records in mode_records.items():
        if [record["sample_id"] for record in records] != full_ids:
            raise RuntimeError(
                f"History-slot diagnostic sample order differs for mode {mode}"
            )
    slot_counts = {
        mode: history_slot_count(records)
        for mode, records in mode_records.items()
    }
    if len(set(slot_counts.values())) != 1:
        raise RuntimeError(f"History-slot count differs across modes: {slot_counts}")
    num_slots = next(iter(slot_counts.values()))
    sample_indices = np.arange(len(full_ids), dtype=np.int64)

    modes: dict[str, dict[str, Any]] = {}
    for mode, records in mode_records.items():
        modes[mode] = {}
        for slot in range(num_slots):
            slot_key = f"slot_{slot}"
            modes[mode][slot_key] = {
                "history_slot": slot,
                **metrics_from_compact(
                    records,
                    sample_indices,
                    history_slot=slot,
                ),
            }

    effects: dict[str, dict[str, Any]] = {}
    for slot in range(num_slots):
        slot_key = f"slot_{slot}"
        slot_metrics = {
            mode: mode_slots[slot_key]
            for mode, mode_slots in modes.items()
        }
        stronger_median = _stronger_null_for_metric(
            slot_metrics,
            "median_pixel_error",
            higher_is_stronger=False,
        )
        stronger_pck8 = _stronger_null_for_metric(
            slot_metrics,
            "pck8",
            higher_is_stronger=True,
        )
        stronger_joint_pck8 = _stronger_null_for_metric(
            slot_metrics,
            "joint_pck8",
            higher_is_stronger=True,
        )
        full_slot = slot_metrics["full"]
        median_value = float(stronger_median["value"])
        effects[slot_key] = {
            "history_slot": slot,
            "stronger_null": {
                "median_pixel_error": stronger_median,
                "pck8": stronger_pck8,
                "joint_pck8": stronger_joint_pck8,
            },
            "effect": {
                "median_relative_improvement_over_stronger_null": (
                    (median_value - full_slot["median_pixel_error"]) / median_value
                    if median_value > 0.0
                    else float("nan")
                ),
                "pck8_delta_over_stronger_null": (
                    full_slot["pck8"] - float(stronger_pck8["value"])
                ),
                "joint_pck8_delta_over_stronger_null": (
                    full_slot["joint_pck8"]
                    - float(stronger_joint_pck8["value"])
                ),
            },
        }

    return {
        "post_hoc": True,
        "affects_aggregate_verdict": False,
        "num_history_slots": num_slots,
        "slot_definition": (
            "Zero-based chronological history order emitted by the dataset; "
            "slot_0 is the oldest sampled history."
        ),
        "metric_notes": {
            "visible_history_count": (
                "Number of validation sample-slot pairs with at least one GT-visible view."
            ),
            "median_pixel_error_and_pck8": (
                "Oracle-conditioned on every GT-visible view."
            ),
            "joint_pck8": (
                "Uses the visibility head's selected view; a wrong view is a failure."
            ),
            "inference": (
                "Descriptive post-hoc localization only; no per-slot threshold, "
                "bootstrap, or verdict gating is applied."
            ),
        },
        "modes": modes,
        "full_vs_stronger_null": effects,
    }


def scene_from_sample_id(sample_id: str) -> str:
    """Extract the scene containing a ``.../scene/clip:frame=...`` sample."""
    clip_identity, separator, _frame = str(sample_id).rpartition(":frame=")
    if not separator:
        raise RuntimeError(f"Sample identity has no ':frame=' suffix: {sample_id!r}")
    scene = PurePosixPath(clip_identity).parent.name
    if not scene:
        raise RuntimeError(f"Cannot extract scene from sample identity: {sample_id!r}")
    return scene


def scene_cluster_indices(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Group ordered compact-prediction indices by their held-out scene."""
    clusters: dict[str, list[int]] = defaultdict(list)
    for sample_index, record in enumerate(records):
        sample_id = record.get("sample_id")
        if not sample_id:
            raise RuntimeError("Compact prediction record lacks sample_id")
        clusters[scene_from_sample_id(str(sample_id))].append(sample_index)
    if len(clusters) < 2:
        raise RuntimeError(
            "Scene-cluster bootstrap requires at least two held-out scenes; "
            f"found {len(clusters)}"
        )
    return {
        scene: np.asarray(indices, dtype=np.int64)
        for scene, indices in sorted(clusters.items())
    }


def paired_scene_cluster_bootstrap(
    full_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Paired bootstrap that resamples scenes and keeps each scene intact.

    Multiple diagnostic samples from one Matterport scene share geometry and
    appearance, so treating them as independent bootstrap units would produce
    anti-conservative confidence intervals.  Each replicate samples the same
    number of scenes with replacement and includes every selected sample from
    each drawn scene.  Full and baseline predictions always use identical
    cluster draws.
    """
    full_ids = [record["sample_id"] for record in full_records]
    baseline_ids = [record["sample_id"] for record in baseline_records]
    if full_ids != baseline_ids:
        raise RuntimeError("Paired bootstrap records are not in the same sample order")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    clusters = scene_cluster_indices(full_records)
    scenes = list(clusters)
    rng = np.random.default_rng(seed)
    distributions: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        drawn_scene_indices = rng.integers(0, len(scenes), size=len(scenes))
        indices = np.concatenate(
            [clusters[scenes[scene_index]] for scene_index in drawn_scene_indices]
        )
        full_metrics = metrics_from_compact(full_records, indices)
        baseline_metrics = metrics_from_compact(baseline_records, indices)
        baseline_median = baseline_metrics["median_pixel_error"]
        relative_median = (
            (baseline_median - full_metrics["median_pixel_error"]) / baseline_median
            if baseline_median > 0.0
            else float("nan")
        )
        distributions["median_relative_improvement"].append(relative_median)
        for metric in ("pck8", "joint_pck8", "visible_view_accuracy"):
            distributions[f"{metric}_delta"].append(full_metrics[metric] - baseline_metrics[metric])
    summary = {}
    for metric, values in distributions.items():
        finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
        if finite.size == 0:
            raise RuntimeError(
                f"No finite scene-bootstrap replicates for metric {metric}"
            )
        summary[metric] = {
            "mean": float(finite.mean()),
            "ci95": [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))],
            "finite_replicates": int(finite.size),
        }
    summary["bootstrap_contract"] = {
        "method": "paired_scene_cluster_percentile_bootstrap",
        "resampling_unit": "scene",
        "scene_count": len(scenes),
        "scene_sample_counts": {
            scene: int(clusters[scene].size)
            for scene in scenes
        },
        "replicates": int(samples),
    }
    return summary


# Backwards-compatible import name; the implementation now clusters by scene.
paired_bootstrap = paired_scene_cluster_bootstrap


def build_verdict(
    effect: dict[str, float],
    bootstrap: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ci_metrics = (
        "median_relative_improvement",
        "pck8_delta",
        "joint_pck8_delta",
    )
    ci_positive = all(
        float(bootstrap[baseline][metric]["ci95"][0]) > 0.0
        for baseline in bootstrap
        for metric in ci_metrics
    )
    checks = {
        "oracle_median_effect": bool(
            effect["median_relative_improvement_over_stronger_null"]
            >= MEDIAN_RELATIVE_IMPROVEMENT_THRESHOLD
        ),
        "oracle_pck8_effect": bool(
            effect["pck8_delta_over_stronger_null"]
            >= PCK8_ABSOLUTE_DELTA_THRESHOLD
        ),
        "joint_pck8_effect": bool(
            effect["joint_pck8_delta_over_stronger_null"]
            >= JOINT_PCK8_ABSOLUTE_DELTA_THRESHOLD
        ),
        "scene_cluster_ci_positive_against_both_nulls": bool(ci_positive),
    }
    return {
        "thresholds": {
            "median_relative_improvement": MEDIAN_RELATIVE_IMPROVEMENT_THRESHOLD,
            "pck8_absolute_delta": PCK8_ABSOLUTE_DELTA_THRESHOLD,
            "joint_pck8_absolute_delta": JOINT_PCK8_ABSOLUTE_DELTA_THRESHOLD,
            # Retain the old field for downstream readers while making the
            # corrected resampling unit explicit.
            "paired_ci_must_exclude_zero_against_both_nulls": True,
            "paired_ci_resampling_unit": "scene",
            "paired_ci_metrics": list(ci_metrics),
        },
        "checks": checks,
        "sample_specific_localization_passed": bool(all(checks.values())),
    }


def main() -> int:
    args = parse_args()
    full = load_json(args.full_report)
    no_input = load_json(args.no_input_report)
    empirical = load_json(args.empirical_report)
    contract = validate_contract(full, no_input, empirical)
    if not contract["passed"]:
        raise RuntimeError(f"Task-3.5 matched contract failed: {contract['checks']}")

    reports = {"full": full, "no-input": no_input, "empirical-prior": empirical}
    rows = []
    for mode, report in reports.items():
        metrics = report["evaluations"]["standard"]
        row = {"mode": mode}
        row.update({metric: metrics.get(metric) for metric in METRICS})
        rows.append(row)

    full_metrics = full["evaluations"]["standard"]
    no_input_metrics = no_input["evaluations"]["standard"]
    empirical_metrics = empirical["evaluations"]["standard"]
    stronger_null_median = min(
        no_input_metrics["median_pixel_error"],
        empirical_metrics["median_pixel_error"],
    )
    stronger_null_pck8 = max(no_input_metrics["pck8"], empirical_metrics["pck8"])
    stronger_null_joint_pck8 = max(
        no_input_metrics["joint_pck8"],
        empirical_metrics["joint_pck8"],
    )
    effect = {
        "median_relative_improvement_over_stronger_null": (
            (stronger_null_median - full_metrics["median_pixel_error"]) / stronger_null_median
            if stronger_null_median > 0.0
            else float("nan")
        ),
        "pck8_delta_over_stronger_null": full_metrics["pck8"] - stronger_null_pck8,
        "joint_pck8_delta_over_stronger_null": (
            full_metrics["joint_pck8"] - stronger_null_joint_pck8
        ),
    }

    full_records = prediction_records(full)
    bootstrap = {
        baseline: paired_scene_cluster_bootstrap(
            full_records,
            prediction_records(reports[baseline]),
            samples=args.bootstrap_samples,
            seed=args.seed + offset,
        )
        for offset, baseline in enumerate(("no-input", "empirical-prior"))
    }
    verdict = build_verdict(effect, bootstrap)
    per_history_slot = build_per_history_slot_diagnostic(reports)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "task35_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "contract": contract,
        "rows": rows,
        "effect": effect,
        "paired_bootstrap": bootstrap,
        "verdict": verdict,
        "per_history_slot_diagnostic": per_history_slot,
        "reports": {
            "full": str(Path(args.full_report).resolve()),
            "no-input": str(Path(args.no_input_report).resolve()),
            "empirical-prior": str(Path(args.empirical_report).resolve()),
        },
    }
    with (output_dir / "task35_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

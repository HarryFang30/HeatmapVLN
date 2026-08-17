#!/usr/bin/env python3
"""Aggregate causal AMB3R-vs-GT frozen-head paired reports."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


_VIEWS = ("front", "right", "back", "left")


def _exact_count(rate: float, denominator: float, *, label: str) -> int:
    value = float(rate) * float(denominator)
    rounded = int(round(value))
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{label} does not recover an integer count: {value}")
    return rounded


def _visibility_metrics(counts: dict[str, int]) -> dict[str, float | int]:
    tp, tn, fp, fn = (int(counts[key]) for key in ("tp", "tn", "fp", "fn"))
    total = tp + tn + fp + fn
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def _aggregate_arm(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    valid = visible = none = 0
    pck4_correct = pck8_correct = view5_correct = 0
    per_view_count = {view: 0 for view in _VIEWS}
    per_view_correct = {view: 0 for view in _VIEWS}
    visibility = {key: 0 for key in ("tp", "tn", "fp", "fn")}

    for report in rows:
        metrics = report["metrics"][arm]
        row_valid = int(metrics["val_heatmap_valid_count"])
        row_visible = int(metrics["val_heatmap_visible_count"])
        valid += row_valid
        visible += row_visible
        none += int(metrics["val_heatmap_none_count"])
        pck4_correct += _exact_count(
            metrics["val_heatmap_joint_pck4"], row_visible, label=f"{arm}.pck4"
        )
        pck8_correct += _exact_count(
            metrics["val_heatmap_joint_pck8"], row_visible, label=f"{arm}.pck8"
        )
        view5_correct += _exact_count(
            metrics["val_heatmap_view5_accuracy"], row_valid, label=f"{arm}.view5"
        )
        for view in _VIEWS:
            count = int(metrics[f"val_heatmap_{view}_count"])
            per_view_count[view] += count
            per_view_correct[view] += _exact_count(
                metrics[f"val_heatmap_{view}_pck8"],
                count,
                label=f"{arm}.{view}.pck8",
            )
        for key in visibility:
            visibility[key] += int(metrics["visibility"][key])

    per_view_rates = {
        view: (
            per_view_correct[view] / per_view_count[view]
            if per_view_count[view]
            else 0.0
        )
        for view in _VIEWS
    }
    supported = [
        per_view_rates[view] for view in _VIEWS if per_view_count[view] > 0
    ]
    return {
        "valid_count": valid,
        "visible_count": visible,
        "none_count": none,
        "joint_pck4_correct": pck4_correct,
        "joint_pck4": pck4_correct / visible if visible else 0.0,
        "joint_pck8_correct": pck8_correct,
        "joint_pck8": pck8_correct / visible if visible else 0.0,
        "view5_correct": view5_correct,
        "view5_accuracy": view5_correct / valid if valid else 0.0,
        "macro_joint_pck8": sum(supported) / len(supported) if supported else 0.0,
        "per_view_pck8": per_view_rates,
        "per_view_count": per_view_count,
        "visibility": _visibility_metrics(visibility),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report_paths = [Path(value).expanduser().resolve(strict=True) for value in args.reports]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    if not reports:
        raise ValueError("At least one paired report is required")
    for path, report in zip(report_paths, reports):
        if report.get("schema") != "heatmapvln-amb3r-paired-frozen-head-audit-v1":
            raise ValueError(f"Unexpected paired report schema: {path}")
        if not report.get("causal_evaluation") or report.get(
            "retrospective_future_pose_updates_allowed"
        ):
            raise ValueError(f"Refusing to aggregate non-causal report: {path}")
        if report.get("per_episode_gt_scale_used"):
            raise ValueError(f"Refusing report with per-episode GT scale: {path}")

    pose_rows = []
    for report_path in report_paths:
        manifest_path = report_path.parents[1] / "amb3r_vo_poses.npz.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metric = manifest["pose_metrics"]["causal_final_current"]
        pose_rows.append(
            {
                "scene_id": manifest.get("scene_id"),
                "episode_id": manifest.get("episode_id"),
                "translation_mae_m_raw": float(metric["translation_mae_m_raw"]),
                "yaw_mae_deg_raw": float(metric["yaw_mae_deg_raw"]),
                "native_scale_ratio_median": float(metric["native_scale_ratio_median"]),
                "frames_per_second": float(manifest["frames_per_second"]),
            }
        )

    arms = {
        arm: _aggregate_arm(reports, arm)
        for arm in ("gt_pose", "amb3r_pose")
    }
    total_maps = sum(int(report["agreement"]["heatmap_map_count"]) for report in reports)

    def weighted_agreement(key: str) -> float:
        if not total_maps:
            return 0.0
        return sum(
            float(report["agreement"][key])
            * int(report["agreement"]["heatmap_map_count"])
            for report in reports
        ) / total_maps

    output = {
        "schema": "heatmapvln-amb3r-causal-paired-summary-v1",
        "report_count": len(reports),
        "sample_count": sum(int(report["samples"]) for report in reports),
        "reports": [str(path) for path in report_paths],
        "per_episode_gt_scale_used": False,
        "metrics": arms,
        "delta_amb3r_minus_gt": {
            key: arms["amb3r_pose"][key] - arms["gt_pose"][key]
            for key in ("joint_pck4", "joint_pck8", "view5_accuracy", "macro_joint_pck8")
        },
        "agreement": {
            "heatmap_map_count": total_maps,
            "mean_peak_shift_px": weighted_agreement("mean_peak_shift_px"),
            "peak_shift_le4_rate": weighted_agreement("peak_shift_le4_rate"),
            "peak_shift_le8_rate": weighted_agreement("peak_shift_le8_rate"),
            "visibility_binary_agreement": weighted_agreement(
                "visibility_binary_agreement"
            ),
            "mean_abs_logit_difference": weighted_agreement(
                "mean_abs_logit_difference"
            ),
        },
        "pose": {
            "translation_mae_m_raw_mean_over_clips": statistics.fmean(
                row["translation_mae_m_raw"] for row in pose_rows
            ),
            "yaw_mae_deg_raw_mean_over_clips": statistics.fmean(
                row["yaw_mae_deg_raw"] for row in pose_rows
            ),
            "native_scale_ratio_median_over_clips": statistics.median(
                row["native_scale_ratio_median"] for row in pose_rows
            ),
            "frames_per_second_mean": statistics.fmean(
                row["frames_per_second"] for row in pose_rows
            ),
            "per_clip": pose_rows,
        },
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), **output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate matched shortcut-probe reports and build a compact comparison table.

The four probes (``full`` / ``vision-only`` / ``pose-only`` / ``no-input``) are
comparable only when they shared a seed, a frozen backbone, a train budget and
a byte-identical fresh head.  Those preconditions are checked fail-closed here
before any number is tabulated, so a table can never quietly mix runs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_MODES = ("full", "vision-only", "pose-only")
OPTIONAL_MODES = ("no-input",)
MODES = REQUIRED_MODES + OPTIONAL_MODES
# The panoramic stack carries rank-32 LoRA on 28 layers x {q,k,v,o} x {A,B}.
LEGACY_LORA_TENSORS = 224
METRICS = (
    "loss",
    "visibility_auroc",
    "visibility_auprc",
    "visibility_f1",
    "visible_view_accuracy",
    "median_pixel_error",
    "median_u_error",
    "pck4",
    "pck8",
    "joint_median_pixel_error",
    "joint_pck4",
    "joint_pck8",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    return parser.parse_args()


def load_report(root: Path, mode: str) -> dict[str, Any]:
    path = root / mode / "report.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing shortcut-probe report: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_reports(root: Path) -> dict[str, dict[str, Any]]:
    reports = {mode: load_report(root, mode) for mode in REQUIRED_MODES}
    for mode in OPTIONAL_MODES:
        if (root / mode / "report.json").is_file():
            reports[mode] = load_report(root, mode)
    return reports


def expected_lora_tensors(architecture: str) -> int:
    # ``internnav_single_view`` runs on the released ViT, which has no adapter
    # at all; requiring 224 there would be a contradiction, not a safeguard.
    return 0 if architecture == "internnav_single_view" else LEGACY_LORA_TENSORS


def matched_contract(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hashes = {mode: report["initial_head_hash"] for mode, report in reports.items()}
    checkpoints = {mode: report.get("checkpoint") for mode, report in reports.items()}
    architectures = {
        mode: report.get("architecture", "legacy_panoramic")
        for mode, report in reports.items()
    }
    train_steps = {mode: int(report["train_steps"]) for mode, report in reports.items()}
    head_numel = {mode: int(report["trainable_head_numel"]) for mode, report in reports.items()}
    qwen_trainable = {mode: int(report["trainable_qwen_tensors"]) for mode, report in reports.items()}
    seeds = {mode: report.get("seed") for mode, report in reports.items()}
    val_hashes = {
        mode: report["selection_contract"]["val"]["sample_identity_sha256"]
        for mode, report in reports.items()
    }
    train_hashes = {
        mode: report["selection_contract"]["train"]["sample_identity_sha256"]
        for mode, report in reports.items()
    }
    # Ground-truth and AMB3R-VO poses are different experiments: the cache
    # restricts usable frames, so mixing the two domains in one table would
    # compare probes that never saw the same samples.
    pose_sources = {
        mode: report.get("history_pose_source", "simulator_ground_truth")
        for mode, report in reports.items()
    }
    pose_cache_roots = {
        mode: report.get("amb3r_pose_cache_root")
        for mode, report in reports.items()
    }
    checks = {
        "same_initial_head_hash": len(set(hashes.values())) == 1,
        "same_checkpoint": len(set(checkpoints.values())) == 1,
        "same_architecture": len(set(architectures.values())) == 1,
        "same_seed": len(set(seeds.values())) == 1,
        "same_train_steps": len(set(train_steps.values())) == 1,
        "same_head_numel": len(set(head_numel.values())) == 1,
        "same_train_selection": len(set(train_hashes.values())) == 1,
        "same_val_selection": len(set(val_hashes.values())) == 1,
        "same_history_pose_source": len(set(pose_sources.values())) == 1,
        "same_pose_cache_root": len(set(pose_cache_roots.values())) == 1,
        "qwen_fully_frozen": all(value == 0 for value in qwen_trainable.values()),
        "all_lora_tensors_matched": all(
            int(report["load"]["matched_lora_tensors"])
            == expected_lora_tensors(architectures[mode])
            for mode, report in reports.items()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "architecture": sorted(set(architectures.values())),
        "history_pose_source": sorted(set(pose_sources.values())),
        "modes": sorted(reports),
        "initial_head_hashes": hashes,
        "val_selection_sha256": val_hashes,
        "trainable_head_numel": head_numel,
        "trainable_qwen_tensors": qwen_trainable,
    }


def metric_row(mode: str, condition: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row = {"mode": mode, "condition": condition}
    row.update({metric: metrics.get(metric) for metric in METRICS})
    row["samples"] = metrics.get("samples")
    return row


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    reports = discover_reports(root)
    contract = matched_contract(reports)
    if not contract["passed"]:
        failed = sorted(name for name, ok in contract["checks"].items() if not ok)
        raise RuntimeError(
            f"Shortcut-probe matched contract failed: {failed}; {contract['checks']}"
        )

    present_modes = [mode for mode in MODES if mode in reports]
    rows = [
        metric_row(mode, "standard", reports[mode]["evaluations"]["standard"])
        for mode in present_modes
    ]
    for condition, metrics in reports["full"]["evaluations"].items():
        if condition != "standard":
            rows.append(metric_row("full", condition, metrics))

    standard = {
        mode: reports[mode]["evaluations"]["standard"]
        for mode in present_modes
    }
    full_standard = standard["full"]
    deltas = {
        f"{mode}_minus_full": {
            metric: standard[mode].get(metric, float("nan")) - full_standard.get(metric, float("nan"))
            for metric in METRICS
        }
        for mode in present_modes
        if mode != "full"
    }
    deltas["full_interventions_minus_standard"] = {
        condition: {
            metric: metrics.get(metric, float("nan")) - full_standard.get(metric, float("nan"))
            for metric in METRICS
        }
        for condition, metrics in reports["full"]["evaluations"].items()
        if condition != "standard"
    }

    csv_path = root / "task3_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "contract": contract,
        "rows": rows,
        "deltas": deltas,
        "reports": {
            mode: str(root / mode / "report.json")
            for mode in present_modes
        },
    }
    json_path = root / "task3_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

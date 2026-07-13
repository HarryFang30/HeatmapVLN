#!/usr/bin/env python3
"""Validate matched Task-3 reports and build a compact comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MODES = ("full", "vision-only", "pose-only")
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    return parser.parse_args()


def load_report(root: Path, mode: str) -> dict[str, Any]:
    path = root / mode / "report.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Task-3 report: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def matched_contract(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hashes = {mode: report["initial_head_hash"] for mode, report in reports.items()}
    checkpoints = {mode: report["checkpoint"] for mode, report in reports.items()}
    train_steps = {mode: int(report["train_steps"]) for mode, report in reports.items()}
    head_numel = {mode: int(report["trainable_head_numel"]) for mode, report in reports.items()}
    qwen_trainable = {mode: int(report["trainable_qwen_tensors"]) for mode, report in reports.items()}
    checks = {
        "same_initial_head_hash": len(set(hashes.values())) == 1,
        "same_checkpoint": len(set(checkpoints.values())) == 1,
        "same_train_steps": len(set(train_steps.values())) == 1,
        "same_head_numel": len(set(head_numel.values())) == 1,
        "qwen_fully_frozen": all(value == 0 for value in qwen_trainable.values()),
        "all_lora_tensors_matched": all(
            int(report["load"]["matched_lora_tensors"]) == 224
            for report in reports.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "initial_head_hashes": hashes,
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
    reports = {mode: load_report(root, mode) for mode in MODES}
    contract = matched_contract(reports)
    if not contract["passed"]:
        raise RuntimeError(f"Task-3 matched contract failed: {contract['checks']}")

    rows = [
        metric_row(mode, "standard", reports[mode]["evaluations"]["standard"])
        for mode in MODES
    ]
    for condition, metrics in reports["full"]["evaluations"].items():
        if condition != "standard":
            rows.append(metric_row("full", condition, metrics))

    standard = {
        mode: reports[mode]["evaluations"]["standard"]
        for mode in MODES
    }
    full_standard = standard["full"]
    deltas = {
        f"{mode}_minus_full": {
            metric: standard[mode].get(metric, float("nan")) - full_standard.get(metric, float("nan"))
            for metric in METRICS
        }
        for mode in ("vision-only", "pose-only")
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
            for mode in MODES
        },
    }
    json_path = root / "task3_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

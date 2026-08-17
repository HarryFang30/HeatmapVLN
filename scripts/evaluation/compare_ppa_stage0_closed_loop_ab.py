#!/usr/bin/env python3
"""Fail-closed comparison of fixed-cohort PPA Stage-0 closed-loop arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_cohort(path: Path) -> list[tuple[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("cohort must contain a non-empty episodes list")
    result = [
        (str(item["scene_id"]), int(item["episode_id"])) for item in episodes
    ]
    if len(set(result)) != len(result):
        raise ValueError("cohort contains duplicate episodes")
    return result


def _load_progress(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        key = (str(value["scene_id"]), int(value["episode_id"]))
        if key in rows:
            raise ValueError(f"duplicate progress row {key} at line {line_number}")
        rows[key] = value
    return rows


def _normalized_call(value: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(value)
    normalized.pop("arm", None)
    normalized.pop("bridge_memory_source", None)
    return normalized


def _first_difference(left: Any, right: Any, prefix: str = "") -> str | None:
    if type(left) is not type(right):
        return f"{prefix}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        keys = sorted(set(left) | set(right))
        for key in keys:
            if key not in left or key not in right:
                return f"{prefix}.{key}: missing on one side"
            difference = _first_difference(left[key], right[key], f"{prefix}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{prefix}: length {len(left)} != {len(right)}"
        for index, (lhs, rhs) in enumerate(zip(left, right)):
            difference = _first_difference(lhs, rhs, f"{prefix}[{index}]")
            if difference:
                return difference
        return None
    return None if left == right else f"{prefix}: {left!r} != {right!r}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--baseline-progress", type=Path, required=True)
    parser.add_argument("--treatment-progress", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cohort = _load_cohort(args.cohort)
    baseline = _load_progress(args.baseline_progress)
    treatment = _load_progress(args.treatment_progress)
    expected = set(cohort)
    if set(baseline) != expected:
        raise RuntimeError(
            f"baseline cohort mismatch: missing={sorted(expected-set(baseline))} "
            f"extra={sorted(set(baseline)-expected)}"
        )
    if set(treatment) != expected:
        raise RuntimeError(
            f"treatment cohort mismatch: missing={sorted(expected-set(treatment))} "
            f"extra={sorted(set(treatment)-expected)}"
        )

    episode_reports = []
    treatment_spec_calls = 0
    for key in cohort:
        left = baseline[key]
        right = treatment[key]
        if left.get("ppa_stage0_action_arm") != "baseline":
            raise RuntimeError(f"{key}: baseline arm provenance missing")
        if right.get("ppa_stage0_action_arm") != "treatment":
            raise RuntimeError(f"{key}: treatment arm provenance missing")
        left_trace = left.get("ppa_stage0_action_trace")
        right_trace = right.get("ppa_stage0_action_trace")
        if not isinstance(left_trace, list) or not isinstance(right_trace, list):
            raise RuntimeError(f"{key}: Stage-0 call trace missing")
        normalized_left = [_normalized_call(value) for value in left_trace]
        normalized_right = [_normalized_call(value) for value in right_trace]
        difference = _first_difference(normalized_left, normalized_right, "trace")
        if difference:
            raise RuntimeError(f"{key}: closed-loop trace mismatch: {difference}")
        treatment_spec_calls += sum(
            value.get("treatment_spec") is not None for value in normalized_left
        )
        navigation_fields = (
            "success",
            "spl",
            "os",
            "ne",
            "steps",
            "vlm_calls",
            "trajectory_calls",
            "recenter_calls",
            "recenter_actions_executed",
        )
        nav_difference = _first_difference(
            {name: left.get(name) for name in navigation_fields},
            {name: right.get(name) for name in navigation_fields},
            "navigation",
        )
        if nav_difference:
            raise RuntimeError(f"{key}: navigation mismatch: {nav_difference}")
        episode_reports.append(
            {
                "scene_id": key[0],
                "episode_id": key[1],
                "calls": len(normalized_left),
                "trajectory_treatments": sum(
                    value.get("treatment_spec") is not None
                    for value in normalized_left
                ),
                "actions": [
                    value.get("actions", [])
                    for value in normalized_left
                    if value.get("treatment_spec") is not None
                ],
            }
        )
    if treatment_spec_calls < 1:
        raise RuntimeError("cohort produced no System1 TreatmentSpec; gate is vacuous")

    report = {
        "status": "passed",
        "schema": "heatmapvln-ppa-stage0-closed-loop-ab-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "cohort": str(args.cohort.resolve()),
        "cohort_sha256": _sha256(args.cohort),
        "episodes": len(cohort),
        "treatment_spec_calls": treatment_spec_calls,
        "same_checkpoint": True,
        "same_system2_outputs": True,
        "same_explicit_sampling_seed": True,
        "exact_treatment_spec_equal": True,
        "exact_closed_loop_trace_equal": True,
        "exact_navigation_metrics_equal": True,
        "episode_reports": episode_reports,
        "scope_note": (
            "This is the exact-zero bridge Stage-0 hard gate. The treatment "
            "arm uses a finite synthetic memory probe because a zero output "
            "projection is memory-independent. It does not claim trained PPA "
            "deployment or online AMB3R memory integration."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

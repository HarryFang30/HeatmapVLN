#!/usr/bin/env python3
"""Select a balanced, prior-preserving System2 on-policy checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class Candidate:
    path: Path
    step: int
    metrics: dict[str, Any]
    harmonic_gain: float
    margin_gap: float


def _load_candidate(path: Path, *, final: bool) -> Candidate | None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("trainable_state_dict")
    if not isinstance(state, dict) or len(state) != 224:
        raise RuntimeError(f"Checkpoint is not a complete 224-LoRA state: {path}")
    if not all(
        "lora_" in str(name) and torch.is_tensor(value)
        for name, value in state.items()
    ):
        raise RuntimeError(f"Checkpoint contains a non-LoRA tensor: {path}")

    step = int(payload["training"]["optimizer_steps"])
    validation = payload["validation"]
    metrics = validation["final" if final else "at_checkpoint"]
    if not metrics.get("quality_passed", False):
        return None
    recall_gain = float(metrics["stop_recall_improvement"])
    false_gain = float(metrics["false_stop_fpr_improvement"])
    harmonic_gain = (
        2.0 * recall_gain * false_gain / max(recall_gain + false_gain, 1e-12)
    )
    margin_gap = float(metrics["positive_false_stop_margin_gap"])
    if not all(math.isfinite(value) for value in (harmonic_gain, margin_gap)):
        raise RuntimeError(f"Non-finite checkpoint selection metric: {path}")
    return Candidate(path, step, metrics, harmonic_gain, margin_gap)


def select_checkpoint(output_dir: Path) -> dict[str, Any]:
    root = output_dir.expanduser().resolve()
    latest = root / "latest.pth"
    if not latest.is_file():
        raise FileNotFoundError(f"Missing final System2 checkpoint: {latest}")

    paths = sorted((root / "validation_checkpoints").glob("step_*.pth"))
    evaluated = len(paths) + 1
    eligible = [
        candidate
        for path in paths
        if (candidate := _load_candidate(path, final=False)) is not None
    ]
    final_candidate = _load_candidate(latest, final=True)
    if final_candidate is not None:
        eligible.append(final_candidate)

    selection_path = root / "selection.json"
    if not eligible:
        result = {
            "status": "failed",
            "reason": "no checkpoint passed recall/FPR/prior gates",
            "evaluated_checkpoints": evaluated,
        }
        temporary = selection_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, selection_path)
        raise RuntimeError(
            "No System2 continuation checkpoint passed the quality gates; "
            f"see {selection_path}"
        )

    best = max(
        eligible,
        key=lambda candidate: (
            candidate.harmonic_gain,
            candidate.margin_gap,
            -candidate.step,
        ),
    )
    selected = root / "selected.pth"
    temporary_link = root / ".selected.pth.tmp"
    if temporary_link.exists() or temporary_link.is_symlink():
        temporary_link.unlink()
    temporary_link.symlink_to(os.path.relpath(best.path, root))
    os.replace(temporary_link, selected)

    result = {
        "status": "passed",
        "selected_checkpoint": str(best.path),
        "selected_step": best.step,
        "selected_metrics": best.metrics,
        "eligible_checkpoints": len(eligible),
        "evaluated_checkpoints": evaluated,
    }
    temporary = selection_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, selection_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = select_checkpoint(args.output_dir)
    print(
        "Full System2 continuation READY: "
        f"selected={args.output_dir / 'selected.pth'} "
        f"step={result['selected_step']} "
        f"eligible={result['eligible_checkpoints']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate the final train.py dry-run report and publish smoke.ready.json."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--allowed-root", default="/mnt/afs/liwenhao/agent/370910109")
    args = parser.parse_args()
    if args.world_size < 1:
        raise ValueError("--world-size must be positive")
    allowed = Path(args.allowed_root).expanduser().resolve(strict=True)
    report_path = Path(args.preflight_report).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().resolve()
    for path in (report_path, output):
        try:
            path.relative_to(allowed)
        except ValueError as exc:
            raise ValueError(f"Path outside {allowed}: {path}") from exc
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise RuntimeError("Training preflight did not pass")
    if report.get("checkpoint_files") != []:
        raise RuntimeError(f"Dry-run wrote checkpoint files: {report.get('checkpoint_files')}")
    audit = (report.get("metrics") or {}).get("pose_adaptation_8gpu_smoke") or {}
    required = {
        "status": "passed",
        "world_size": args.world_size,
        "batch_per_rank": 2,
        "global_identity_count": args.world_size * 2,
        "global_unique_identity_count": args.world_size * 2,
        "providers": ["amb3r_vo_cache"],
        "post_parameter_digest_unique_count": 1,
        "ema_digest_unique_count": 1,
        "gradient_family_tensor_counts": {
            "proj_traj": 2,
            "transformer": 24,
            "visibility": 4,
            "coarse_heatmap": 4,
        },
        "checkpoint_hash_locking": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": audit.get(key)}
        for key, expected in required.items()
        if audit.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Distributed smoke audit mismatch: {mismatches}")
    if audit.get("optimizer_steps_by_rank") != [1] * args.world_size:
        raise RuntimeError("Each rank must complete exactly one optimizer step")
    if audit.get("gradient_hook_tensors_by_rank") != [34] * args.world_size:
        raise RuntimeError("Each rank must see all 34 gradient hooks")
    if audit.get("ema_steps_by_rank") != [1] * args.world_size:
        raise RuntimeError("Each rank must update EMA exactly once")
    family_ranks = audit.get("gradient_families_nonzero_on_ranks") or {}
    for family in ("proj_traj", "transformer", "visibility", "coarse_heatmap"):
        if family_ranks.get(family) != list(range(args.world_size)):
            raise RuntimeError(f"Gradient family {family} is not non-zero on every rank")

    run_dir = report_path.parent.parent
    artifacts = sorted(
        str(path.relative_to(run_dir))
        for suffix in ("*.pth", "*.pt", "*.ckpt", "*.safetensors")
        for path in run_dir.rglob(suffix)
    )
    if artifacts:
        raise RuntimeError(f"Smoke run contains model artifacts: {artifacts}")
    payload = {
        "schema": (
            f"heatmapvln-amb3r-pose-adapt-{args.world_size}gpu-smoke-ready-v1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "preflight_report": str(report_path),
        "run_dir": str(run_dir),
        "audit": audit,
        "checkpoint_files": [],
        "model_artifacts": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Derive the no-checkpoint 8-GPU smoke config from the production config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = Path(args.base)
    output = Path(args.output)
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
    cfg["data"]["num_workers"] = 0
    cfg["data"]["pin_memory"] = False
    cfg["data"]["sliding_window"]["samples_per_clip"] = 16
    cfg["data"]["sliding_window"]["val_samples_per_clip"] = 1
    cfg["optim"]["batch_size"] = 2
    cfg["optim"]["grad_accum_steps"] = 1
    cfg["training"]["stages"][0]["epochs"] = 1
    cfg["validation"]["enabled"] = False
    cfg["validation"]["evaluate_before_training"] = False
    cfg["validation"]["baseline_as_best_threshold"] = False
    cfg["log"]["use_tensorboard"] = False
    cfg["log"]["val_vis_batches"] = 0
    cfg["log"]["mid_epoch_save_every"] = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Extract a deployed History Head into the probe's head-only checkpoint format.

``diagnose_heatmap_shortcuts.py --head-checkpoint`` evaluates a *given* head
under the six interventions, but it needs the head as
``{head_state_dict, initial_head_hash}`` whose key set matches the head the
probe itself builds.  Training checkpoints instead carry the head inside
``trainable_state_dict`` under a ``heatmap_vln.`` prefix, alongside Future and
Bridge tensors.

This tool builds the head from the probe's own config (CPU is fine), overwrites
exactly the tensors the training checkpoint provides, and fails closed if the
checkpoint carries a head tensor the built head does not have, or if any
provided tensor is left unconsumed.  Neither input file is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools.diagnose_heatmap_shortcuts import (  # noqa: E402
    heatmap_head_state_dict,
    load_config,
    state_hash,
)
from scripts.training import build_model  # noqa: E402

HEAD_PREFIX = "heatmap_vln."


def _checkpoint_head_tensors(payload: dict) -> dict[str, torch.Tensor]:
    for key in ("trainable_state_dict", "model_state_dict", "state_dict"):
        state = payload.get(key)
        if isinstance(state, dict) and state:
            tensors = {}
            for name, value in state.items():
                normalized = name[len("module.") :] if name.startswith("module.") else name
                if normalized.startswith(HEAD_PREFIX) and isinstance(value, torch.Tensor):
                    tensors[normalized[len(HEAD_PREFIX) :]] = value
            if tensors:
                return tensors
    raise RuntimeError("checkpoint carries no heatmap_vln.* tensors")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="the config the probe will run with")
    parser.add_argument("--architecture", default="internnav_single_view")
    parser.add_argument("--data-root", required=True, help="only used to satisfy config loading")
    parser.add_argument("--checkpoint", required=True, help="training checkpoint holding heatmap_vln.*")
    parser.add_argument("--internnav-model-path", default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(
        argparse.Namespace(
            config=args.config,
            data_root=args.data_root,
            architecture=args.architecture,
            device=args.device,
            num_history=8,
            internnav_model_path=args.internnav_model_path,
            amb3r_pose_cache_root=None,
            amb3r_pose_cache_max_clips=16,
        )
    )
    model = build_model(cfg, verbose=False, device=args.device, enable_action_head=False)
    # The head is built lazily after the backbone loads, exactly as the probe
    # does it; without this ``model.heatmap_vln`` is still None.
    model.qwen2_5_vl._load_model()
    model._ensure_heatmap_vln()
    built = heatmap_head_state_dict(model.heatmap_vln)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    provided = _checkpoint_head_tensors(payload)

    unexpected = sorted(set(provided) - set(built))
    if unexpected:
        raise RuntimeError(
            f"checkpoint head tensors absent from the built head: {unexpected[:5]}"
        )
    mismatched = sorted(
        name for name in provided if tuple(provided[name].shape) != tuple(built[name].shape)
    )
    if mismatched:
        raise RuntimeError(f"shape mismatch on: {mismatched[:5]}")

    merged = dict(built)
    for name, value in provided.items():
        merged[name] = value.detach().clone()
    untouched = sorted(set(built) - set(provided))

    out = {
        "head_state_dict": merged,
        "initial_head_hash": state_hash(merged),
        "provenance": {
            "source_checkpoint": str(Path(args.checkpoint).resolve()),
            "config": str(Path(args.config).resolve()),
            "architecture": args.architecture,
            "tensors_from_checkpoint": len(provided),
            "tensors_from_fresh_build": len(untouched),
            "untouched_examples": untouched[:10],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(json.dumps(out["provenance"], indent=2, sort_keys=True))
    print(f"initial_head_hash={out['initial_head_hash']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

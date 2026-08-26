#!/usr/bin/env python3
"""Build a self-contained PPA action-refine deployment checkpoint.

Action refinement freezes the complete Heatmap and Future heads and trains only
the Bridge.  Older action-refine checkpoints omitted the frozen Future tensors
from their deployment entry.  This tool copies those tensors bit-for-bit from
the declared parent Stage-2 checkpoint after strict family and equality checks.
It never overwrites either source checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch


HEATMAP_PREFIX = "heatmap_vln."
FUTURE_PREFIX = "past_plan_action.future_head."
BRIDGE_PREFIX = "past_plan_action.bridge."


def _normalize(name: str) -> str:
    if name.startswith("module."):
        name = name[len("module.") :]
    return name.replace(".module.", ".")


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    value = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise RuntimeError(f"checkpoint is not a dict: {path}")
    return value


def _state(payload: Mapping[str, Any], label: str) -> dict[str, torch.Tensor]:
    raw = payload.get("trainable_state_dict")
    if not isinstance(raw, Mapping) or not raw:
        raise RuntimeError(f"{label} lacks trainable_state_dict")
    result: dict[str, torch.Tensor] = {}
    for raw_name, tensor in raw.items():
        name = _normalize(str(raw_name))
        if name in result or not torch.is_tensor(tensor):
            raise RuntimeError(f"invalid or duplicate {label} tensor: {name}")
        result[name] = tensor.detach().cpu()
    return result


def _family(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {name: tensor for name, tensor in state.items() if name.startswith(prefix)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-checkpoint", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    action_path = Path(args.action_checkpoint).resolve(strict=True)
    parent_path = Path(args.parent_checkpoint).resolve(strict=True)
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output_path}")
    if output_path.parent != action_path.parent:
        raise RuntimeError("output must be in the action checkpoint directory")

    action_payload = _load(action_path)
    parent_payload = _load(parent_path)
    action_state = _state(action_payload, "action")
    parent_state = _state(parent_payload, "parent")

    action_heatmap = _family(action_state, HEATMAP_PREFIX)
    action_future = _family(action_state, FUTURE_PREFIX)
    action_bridge = _family(action_state, BRIDGE_PREFIX)
    parent_heatmap = _family(parent_state, HEATMAP_PREFIX)
    parent_future = _family(parent_state, FUTURE_PREFIX)
    if (len(action_heatmap), len(action_future), len(action_bridge)) != (79, 0, 10):
        raise RuntimeError(
            "unexpected action families: "
            f"heatmap={len(action_heatmap)} future={len(action_future)} "
            f"bridge={len(action_bridge)}"
        )
    if len(parent_heatmap) != 79 or len(parent_future) != 11:
        raise RuntimeError(
            "unexpected parent families: "
            f"heatmap={len(parent_heatmap)} future={len(parent_future)}"
        )
    if set(action_heatmap) != set(parent_heatmap):
        raise RuntimeError("action and parent Heatmap key sets differ")
    changed_frozen = [
        name
        for name in action_heatmap
        if not torch.equal(action_heatmap[name], parent_heatmap[name])
    ]
    if changed_frozen:
        raise RuntimeError(
            f"action refinement changed frozen Heatmap tensors: {changed_frozen[:8]}"
        )

    contract = action_payload.get("past_plan_action_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("action checkpoint lacks PPA contract")
    if contract.get("complete_future_head_tensors") != 11:
        raise RuntimeError("action checkpoint Future contract is not 11 tensors")
    if contract.get("bridge_in_deployment_state") is not True:
        raise RuntimeError("action checkpoint contract omits Bridge")

    merged = {**action_state, **parent_future}
    if len(merged) != 100:
        raise RuntimeError(f"merged deployment must contain 100 tensors, got {len(merged)}")
    action_payload["trainable_state_dict"] = merged
    action_payload["weight_semantics"] = {
        **dict(action_payload.get("weight_semantics") or {}),
        "trainable_state_dict": (
            "ema_bridge_plus_frozen_heatmap_and_parent_stage2_future"
        ),
    }
    manifest = dict(action_payload.get("deployment_state_manifest") or {})
    manifest["self_contained_future_head"] = True
    manifest["future_head_tensor_count"] = 11
    manifest["deployment_tensor_count"] = 100
    action_payload["deployment_state_manifest"] = manifest
    action_payload["action_refine_repair"] = {
        "schema": "heatmapvln-ppa-action-refine-repair-v1",
        "action_checkpoint": str(action_path),
        "parent_stage2_checkpoint": str(parent_path),
        "copied_frozen_future_tensors": 11,
        "heatmap_bitwise_equal": True,
    }

    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        torch.save(action_payload, temporary)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(output_path),
                "tensor_count": len(merged),
                "heatmap": 79,
                "future": 11,
                "bridge": 10,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

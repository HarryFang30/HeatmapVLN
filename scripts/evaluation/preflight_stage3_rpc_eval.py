#!/usr/bin/env python3
"""Validate the exact checkpoints and config used by Stage3 RPC evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
import yaml

from scripts.training.utils import _normalize_state_key


_LORA_KEY_RE = re.compile(
    r"\.layers\.(?P<layer>\d+)\.self_attn\."
    r"(?P<module>q_proj|k_proj|v_proj|o_proj)\."
    r"lora_(?P<side>A|B)(?:\.[^.]+)?\.weight$"
)
_REQUIRED_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
_REQUIRED_STAGE3_TRAINABLE = ("pano_latent_adapter",)


def _torch_load(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must contain a dict: {path}")
    return checkpoint


def _checkpoint_state(checkpoint: dict[str, Any], path: str | Path) -> dict[str, torch.Tensor]:
    for key in ("model_state_dict", "trainable_state_dict", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise KeyError(f"No model/trainable/state dict in checkpoint: {path}")


def _adapter_state(checkpoint: dict[str, Any], path: str | Path) -> dict[str, torch.Tensor]:
    direct = checkpoint.get("adapter_state_dict")
    if isinstance(direct, dict):
        return direct

    for key in ("trainable_state_dict", "model_state_dict", "state_dict"):
        candidate = checkpoint.get(key)
        if not isinstance(candidate, dict):
            continue
        state = {
            name.removeprefix("module.").removeprefix("pano_latent_adapter."): value
            for name, value in candidate.items()
            if name.removeprefix("module.").startswith("pano_latent_adapter.")
        }
        if state:
            return state
    raise KeyError(f"No pano_latent_adapter state in checkpoint: {path}")


def _first_stage(config: dict[str, Any]) -> dict[str, Any]:
    stages = config.get("training", {}).get("stages", [])
    if not stages or not isinstance(stages[0], dict):
        raise ValueError("training.stages[0] is required")
    return stages[0]


def _assert_finite(state: dict[str, torch.Tensor], label: str) -> None:
    non_tensors = sorted(name for name, value in state.items() if not torch.is_tensor(value))
    if non_tensors:
        raise TypeError(f"{label} contains non-tensor values: {non_tensors[:5]}")
    nonfinite = [
        name
        for name, value in state.items()
        if not bool(torch.isfinite(value.float()).all())
    ]
    if nonfinite:
        raise ValueError(f"{label} contains non-finite tensors: {nonfinite[:5]}")


def validate_stage3_config(
    config: dict[str, Any],
    *,
    expected_adapter_hidden_dim: int | None = None,
) -> dict[str, Any]:
    stage = _first_stage(config)
    errors: list[str] = []

    if stage.get("name") != "stage3":
        errors.append(f"stage name={stage.get('name')!r}, expected 'stage3'")
    trainable = tuple(stage.get("trainable_modules") or ())
    if trainable != _REQUIRED_STAGE3_TRAINABLE:
        errors.append(
            f"trainable_modules={list(trainable)!r}, expected {list(_REQUIRED_STAGE3_TRAINABLE)!r}"
        )
    for key in (
        "strict_trainable_modules",
        "requires_base_checkpoint",
        "require_complete_internnav_system1",
        "base_checkpoint_lora_only",
    ):
        if stage.get(key) is not True:
            errors.append(f"training.stages[0].{key} must be true")

    trajectory = config.get("data", {}).get("trajectory", {})
    if trajectory.get("panoramic_vlm_input") is not True:
        errors.append("data.trajectory.panoramic_vlm_input must be true")
    if trajectory.get("structured_pano_output") is not True:
        errors.append("data.trajectory.structured_pano_output must be true")

    llm = config.get("model", {}).get("llm", {})
    layers = [int(value) for value in (llm.get("lora_layer_indices") or [])]
    if layers != list(range(28)):
        errors.append(f"lora_layer_indices={layers!r}, expected all layers 0..27")
    targets = tuple(llm.get("lora_target_modules") or ())
    if set(targets) != set(_REQUIRED_LORA_TARGETS) or len(targets) != 4:
        errors.append(
            f"lora_target_modules={list(targets)!r}, expected {list(_REQUIRED_LORA_TARGETS)!r}"
        )
    lora_rank = int(llm.get("lora_rank", 0) or 0)
    if lora_rank != 32:
        errors.append(f"lora_rank={lora_rank}, expected 32")
    if llm.get("use_lora") is not True:
        errors.append("model.llm.use_lora must be true")

    nextdit = config.get("model", {}).get("action_head", {}).get("nextdit", {})
    if nextdit.get("enabled") is not True:
        errors.append("model.action_head.nextdit.enabled must be true")
    adapter = nextdit.get("pano_latent_adapter", {})
    if adapter.get("enabled") is not True:
        errors.append("model.action_head.nextdit.pano_latent_adapter.enabled must be true")
    adapter_hidden_dim = int(adapter.get("hidden_dim", 0) or 0)
    if expected_adapter_hidden_dim is not None and adapter_hidden_dim != expected_adapter_hidden_dim:
        errors.append(
            f"pano adapter hidden_dim={adapter_hidden_dim}, expected {expected_adapter_hidden_dim}"
        )

    if errors:
        raise ValueError("Stage3 evaluation config validation failed:\n  - " + "\n  - ".join(errors))

    return {
        "lora_layers": layers,
        "lora_targets": list(_REQUIRED_LORA_TARGETS),
        "lora_rank": lora_rank,
        "llm_hidden_dim": int(llm.get("hidden_dim", 3584)),
        "adapter_hidden_dim": adapter_hidden_dim,
    }


def validate_base_checkpoint(
    checkpoint_path: str | Path,
    config_summary: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = _torch_load(checkpoint_path)
    state = _checkpoint_state(checkpoint, checkpoint_path)
    normalized = {_normalize_state_key(name): value for name, value in state.items()}
    lora_state = {name: value for name, value in normalized.items() if "lora_" in name}
    _assert_finite(lora_state, "base LoRA checkpoint")

    actual: set[tuple[int, str, str]] = set()
    malformed: list[str] = []
    rank_errors: list[str] = []
    expected_rank = int(config_summary["lora_rank"])
    for name, value in lora_state.items():
        match = _LORA_KEY_RE.search(name)
        if match is None:
            malformed.append(name)
            continue
        layer = int(match.group("layer"))
        module = match.group("module")
        side = match.group("side")
        actual.add((layer, module, side))
        rank_axis = 0 if side == "A" else 1
        if value.ndim != 2 or int(value.shape[rank_axis]) != expected_rank:
            rank_errors.append(f"{name}: shape={tuple(value.shape)}")

    expected = {
        (layer, module, side)
        for layer in config_summary["lora_layers"]
        for module in config_summary["lora_targets"]
        for side in ("A", "B")
    }
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if malformed or rank_errors or missing or unexpected or len(lora_state) != len(expected):
        raise ValueError(
            "Base LoRA checkpoint validation failed: "
            f"tensors={len(lora_state)} expected={len(expected)} "
            f"malformed={malformed[:3]} rank_errors={rank_errors[:3]} "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    if checkpoint.get("stage_name") not in (None, "stage1_s2_panoramic_sft"):
        raise ValueError(
            f"Unexpected base checkpoint stage_name={checkpoint.get('stage_name')!r}"
        )
    return {
        "path": str(Path(checkpoint_path).resolve()),
        "stage_name": checkpoint.get("stage_name"),
        "epoch": checkpoint.get("epoch"),
        "state_tensors": len(state),
        "lora_tensors": len(lora_state),
    }


def validate_stage3_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_epoch: int,
    expected_base_checkpoint: str | Path,
    config_summary: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = _torch_load(checkpoint_path)
    errors: list[str] = []
    if checkpoint.get("stage_name") != "stage3":
        errors.append(f"stage_name={checkpoint.get('stage_name')!r}, expected 'stage3'")
    if int(checkpoint.get("epoch", -1)) != int(expected_epoch):
        errors.append(f"epoch={checkpoint.get('epoch')!r}, expected {expected_epoch}")
    if checkpoint.get("batch") is not None:
        errors.append(f"batch={checkpoint.get('batch')!r}; final epoch checkpoint required")

    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        errors.append("checkpoint config is missing")
    else:
        try:
            validate_stage3_config(
                checkpoint_config,
                expected_adapter_hidden_dim=int(config_summary["adapter_hidden_dim"]),
            )
        except ValueError as exc:
            errors.append(str(exc))
        recorded_base = checkpoint_config.get("runtime", {}).get("base_checkpoint", "")
        if os.path.realpath(str(recorded_base)) != os.path.realpath(str(expected_base_checkpoint)):
            errors.append(
                f"runtime.base_checkpoint={recorded_base!r}, expected {str(expected_base_checkpoint)!r}"
            )

    state = _adapter_state(checkpoint, checkpoint_path)
    _assert_finite(state, "Stage3 pano adapter")
    dim = int(config_summary["llm_hidden_dim"])
    hidden_dim = int(config_summary["adapter_hidden_dim"])
    expected_shapes = {
        "mlp.0.weight": (hidden_dim, dim),
        "mlp.0.bias": (hidden_dim,),
        "mlp.3.weight": (dim, hidden_dim),
        "mlp.3.bias": (dim,),
    }
    actual_shapes = {name: tuple(value.shape) for name, value in state.items()}
    if actual_shapes != expected_shapes:
        errors.append(f"adapter shapes={actual_shapes!r}, expected={expected_shapes!r}")

    trainable_state = checkpoint.get("trainable_state_dict")
    if isinstance(trainable_state, dict):
        unexpected_trainable = sorted(
            name
            for name in trainable_state
            if not name.removeprefix("module.").startswith("pano_latent_adapter.")
        )
        if unexpected_trainable:
            errors.append(f"unexpected trainable tensors={unexpected_trainable[:5]}")

    if errors:
        raise ValueError("Stage3 checkpoint validation failed:\n  - " + "\n  - ".join(errors))

    return {
        "path": str(Path(checkpoint_path).resolve()),
        "stage_name": checkpoint.get("stage_name"),
        "epoch": int(checkpoint["epoch"]),
        "adapter_tensors": len(state),
        "adapter_parameters": sum(value.numel() for value in state.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--stage3-checkpoint", required=True)
    parser.add_argument("--expected-epoch", type=int, default=2)
    parser.add_argument("--expected-adapter-hidden-dim", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Config must contain a mapping: {args.config}")

    config_summary = validate_stage3_config(
        config,
        expected_adapter_hidden_dim=args.expected_adapter_hidden_dim,
    )
    result = {
        "status": "passed",
        "config": config_summary,
        "base_checkpoint": validate_base_checkpoint(args.base_checkpoint, config_summary),
        "stage3_checkpoint": validate_stage3_checkpoint(
            args.stage3_checkpoint,
            expected_epoch=args.expected_epoch,
            expected_base_checkpoint=args.base_checkpoint,
            config_summary=config_summary,
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

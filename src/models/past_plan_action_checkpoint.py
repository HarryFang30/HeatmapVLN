"""Strict, hash-free deployment delta contract for Past→Plan→Action."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from .past_plan_action import PastPlanActionChain, PastPlanActionContractError


PPA_CHECKPOINT_SCHEMA = "heatmapvln-past-plan-action-delta-v1"

_PAST_DELTA_PREFIXES = (
    "coarse.proj_history.",
    "coarse.proj_traj.",
    "coarse.pos_embed",
    "coarse.self_attn.",
    "coarse.heatmap_head.",
    "coarse.vis_head.",
    "fine.",
)

_PAST_FULL_HEAD_PREFIXES = (
    "heatmap_vln.vit_dpt_fusion.",
    "heatmap_vln.llm_dpt_fusion.",
    "heatmap_vln.coarse.",
    "heatmap_vln.fine.",
)


def validate_exact_past_head_warmstart(
    *,
    model: nn.Module,
    checkpoint_state: Mapping[str, torch.Tensor],
    expected_tensor_count: int = 79,
) -> dict[str, int]:
    """Require the complete existing Past Head without pinning file contents."""

    def normalize(name: str) -> str:
        while name.startswith("module."):
            name = name[len("module.") :]
        return name.replace(".module.", ".")

    expected = {
        normalize(name): value
        for name, value in model.state_dict().items()
        if normalize(name).startswith(_PAST_FULL_HEAD_PREFIXES)
    }
    actual = {
        normalize(name): value
        for name, value in checkpoint_state.items()
        if normalize(name).startswith(_PAST_FULL_HEAD_PREFIXES)
    }
    if len(expected) != expected_tensor_count:
        raise PastPlanActionContractError(
            "Current Past Head does not match the audited tensor contract: "
            f"model={len(expected)} required={expected_tensor_count}"
        )
    if set(actual) != set(expected):
        raise PastPlanActionContractError(
            "Past Head checkpoint is incomplete: "
            f"missing={sorted(set(expected) - set(actual))[:8]} "
            f"extra={sorted(set(actual) - set(expected))[:8]}"
        )
    mismatched = [
        name
        for name in expected
        if tuple(expected[name].shape) != tuple(actual[name].shape)
    ]
    if mismatched:
        raise PastPlanActionContractError(
            f"Past Head checkpoint shape mismatch: {mismatched[:8]}"
        )
    return {
        "validated_tensors": len(expected),
        "checkpoint_digest_enforced": 0,
        "file_lock_used": 0,
    }


def _selected_past_parameter_names(past_head: nn.Module) -> tuple[str, ...]:
    names = tuple(
        name
        for name, _ in past_head.named_parameters()
        if any(
            name == prefix or (prefix.endswith(".") and name.startswith(prefix))
            for prefix in _PAST_DELTA_PREFIXES
        )
    )
    if not names:
        raise PastPlanActionContractError("Past delta parameter set is empty")
    return names


def build_past_plan_action_delta_state(
    *, chain: PastPlanActionChain, past_head: nn.Module
) -> OrderedDict[str, torch.Tensor]:
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, parameter in chain.named_parameters():
        state[f"past_plan_action.{name}"] = parameter.detach().cpu().clone()
    past_parameters = dict(past_head.named_parameters())
    for name in _selected_past_parameter_names(past_head):
        state[f"heatmap_vln.{name}"] = past_parameters[name].detach().cpu().clone()
    if not state:
        raise PastPlanActionContractError("Past→Plan→Action delta is empty")
    return state


def load_past_plan_action_delta_state(
    *,
    chain: PastPlanActionChain,
    past_head: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> dict[str, int]:
    destinations: dict[str, nn.Parameter] = {
        f"past_plan_action.{name}": parameter for name, parameter in chain.named_parameters()
    }
    past_parameters = dict(past_head.named_parameters())
    destinations.update(
        {
            f"heatmap_vln.{name}": past_parameters[name]
            for name in _selected_past_parameter_names(past_head)
        }
    )
    actual, expected = set(state), set(destinations)
    if actual != expected:
        raise PastPlanActionContractError(
            "Past→Plan→Action delta key mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    with torch.no_grad():
        for name, destination in destinations.items():
            value = state[name]
            if not torch.is_tensor(value) or tuple(value.shape) != tuple(destination.shape):
                raise PastPlanActionContractError(
                    f"delta shape mismatch for {name}: "
                    f"{getattr(value, 'shape', None)} != {tuple(destination.shape)}"
                )
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise PastPlanActionContractError(
                    f"delta tensor {name} must be finite floating point"
                )
            destination.copy_(value.to(device=destination.device, dtype=destination.dtype))
    return {
        "loaded_tensors": len(destinations),
        "loaded_parameters": int(sum(value.numel() for value in destinations.values())),
    }


def build_past_plan_action_manifest(
    *,
    native_checkpoint_path: str | Path,
    past_head_checkpoint_path: str | Path,
    delta_tensor_count: int,
) -> dict[str, object]:
    if delta_tensor_count <= 0:
        raise ValueError("delta_tensor_count must be positive")
    return {
        "schema": PPA_CHECKPOINT_SCHEMA,
        "native_checkpoint_path": str(native_checkpoint_path),
        "past_head_checkpoint_path": str(past_head_checkpoint_path),
        "load_order": ["native_internnav", "past_head_79", "past_plan_action_delta"],
        "delta_tensor_count": int(delta_tensor_count),
        "checkpoint_digest_enforced": False,
        "file_lock_used": False,
        "plan_token_shape": [4, 768],
        "history_memory_dim": 256,
        "future_heatmap_shape": [4, 4, 64, 64],
        "future_time_ranges": [[1, 8], [9, 16], [17, 24], [25, 32]],
        "direction_order": ["front", "right", "back", "left"],
    }

"""Fail-closed trainability and optimizer contracts for Past→Plan→Action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch
from torch import nn

from .past_plan_action import PastPlanActionChain, PastPlanActionContractError


PastPlanActionStage = Literal["stage1_map_pretrain", "stage2_joint"]

_PAST_TRAINABLE_PREFIXES = (
    "coarse.proj_history.",
    "coarse.proj_traj.",
    "coarse.pos_embed",
    "coarse.self_attn.",
    "coarse.heatmap_head.",
    "coarse.vis_head.",
    "fine.",
)


@dataclass(frozen=True)
class TrainableScopeAudit:
    stage: str
    trainable_tensors: int
    trainable_parameters: int
    future_tensors: int
    bridge_tensors: int
    shared_past_tensors: int
    names: tuple[str, ...]


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        name == prefix or (prefix.endswith(".") and name.startswith(prefix))
        for prefix in prefixes
    )


def _freeze_and_eval(module: nn.Module) -> None:
    module.requires_grad_(False)
    module.eval()


def configure_past_plan_action_stage(
    *,
    stage: PastPlanActionStage,
    chain: PastPlanActionChain,
    past_head: nn.Module,
    native_action_head: nn.Module,
    native_cond_projector: nn.Module,
    other_frozen_modules: Iterable[nn.Module] = (),
) -> TrainableScopeAudit:
    """Set exactly the approved Stage-1 or Stage-2 parameter scope."""

    if stage not in {"stage1_map_pretrain", "stage2_joint"}:
        raise ValueError(f"unsupported Past→Plan→Action stage: {stage!r}")
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float32
        for parameter in chain.parameters()
    ):
        raise PastPlanActionContractError("Past→Plan→Action modules must remain FP32")

    _freeze_and_eval(chain)
    _freeze_and_eval(past_head)
    _freeze_and_eval(native_action_head)
    _freeze_and_eval(native_cond_projector)
    for module in other_frozen_modules:
        _freeze_and_eval(module)

    chain.future_head.requires_grad_(True)
    chain.future_head.train()
    if stage == "stage2_joint":
        chain.bridge.requires_grad_(True)
        chain.bridge.train()

    selected_past_names: list[str] = []
    for name, parameter in past_head.named_parameters():
        if _matches_prefix(name, _PAST_TRAINABLE_PREFIXES):
            parameter.requires_grad_(True)
            selected_past_names.append(name)
    if not selected_past_names:
        raise PastPlanActionContractError(
            "Past Head exposes none of the expected memory/shared-decoder parameters"
        )

    past_head.eval()
    coarse = getattr(past_head, "coarse", None)
    fine = getattr(past_head, "fine", None)
    required_coarse = (
        "proj_history",
        "proj_traj",
        "self_attn",
        "heatmap_head",
        "vis_head",
    )
    if coarse is None or fine is None or any(
        getattr(coarse, name, None) is None for name in required_coarse
    ):
        raise PastPlanActionContractError(
            "Past Head must expose trajectory-guided coarse and fine modules"
        )
    for name in required_coarse:
        getattr(coarse, name).train()
    fine.train()

    if native_action_head.training or native_cond_projector.training:
        raise AssertionError("native action modules must remain eval")
    if any(parameter.requires_grad for parameter in native_action_head.parameters()):
        raise AssertionError("native action head was not fully frozen")
    if any(parameter.requires_grad for parameter in native_cond_projector.parameters()):
        raise AssertionError("native cond_projector was not fully frozen")

    names: list[str] = []
    counts = {"future": 0, "bridge": 0, "past": 0}
    total_parameters = 0
    for prefix, module, family in (
        ("past_plan_action.future_head.", chain.future_head, "future"),
        ("past_plan_action.bridge.", chain.bridge, "bridge"),
        ("heatmap_vln.", past_head, "past"),
    ):
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                names.append(prefix + name)
                counts[family] += 1
                total_parameters += parameter.numel()
    if stage == "stage1_map_pretrain" and counts["bridge"] != 0:
        raise AssertionError("Stage 1 bridge must be frozen")
    if stage == "stage2_joint" and counts["bridge"] == 0:
        raise AssertionError("Stage 2 bridge must be trainable")

    return TrainableScopeAudit(
        stage=stage,
        trainable_tensors=len(names),
        trainable_parameters=int(total_parameters),
        future_tensors=counts["future"],
        bridge_tensors=counts["bridge"],
        shared_past_tensors=counts["past"],
        names=tuple(sorted(names)),
    )


def build_past_plan_action_optimizer(
    *,
    chain: PastPlanActionChain,
    past_head: nn.Module,
    future_lr: float = 1.0e-4,
    bridge_lr: float = 2.0e-5,
    shared_map_lr: float = 2.0e-5,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """Build non-overlapping FP32 groups covering every selected tensor once."""

    if min(future_lr, bridge_lr, shared_map_lr) <= 0:
        raise ValueError("all Past→Plan→Action learning rates must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    groups: list[dict[str, object]] = []
    seen: set[int] = set()
    expected = {
        id(parameter)
        for module in (chain, past_head)
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    for family, module, lr in (
        ("future", chain.future_head, future_lr),
        ("bridge", chain.bridge, bridge_lr),
        ("shared_map", past_head, shared_map_lr),
    ):
        params = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if any(
            not parameter.is_floating_point() or parameter.dtype != torch.float32
            for parameter in params
        ):
            raise PastPlanActionContractError(
                f"optimizer family {family} contains a non-FP32 trainable tensor"
            )
        if any(id(parameter) in seen for parameter in params):
            raise PastPlanActionContractError(
                f"optimizer family {family} contains duplicate shared parameters"
            )
        seen.update(id(parameter) for parameter in params)
        if params:
            groups.append(
                {
                    "params": params,
                    "lr": float(lr),
                    "weight_decay": float(weight_decay),
                    "family": family,
                    "name": f"past_plan_action_{family}",
                }
            )
    if seen != expected:
        raise PastPlanActionContractError(
            "optimizer coverage differs from the exact trainable parameter set"
        )
    return torch.optim.AdamW(groups)


def assert_native_frozen_and_gradient_free(*modules: nn.Module) -> None:
    for module in modules:
        if module.training:
            raise PastPlanActionContractError("native frozen module entered train mode")
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                raise PastPlanActionContractError(
                    f"native parameter unexpectedly trainable: {name}"
                )
            if parameter.grad is not None:
                raise PastPlanActionContractError(
                    f"native frozen parameter received a gradient: {name}"
                )

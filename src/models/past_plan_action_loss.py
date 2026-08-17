"""Explicit staged loss composition for Past→Plan→Action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import torch

from .past_plan_action import PastPlanActionContractError


LossStage = Literal["stage1_map_pretrain", "stage2_joint"]


@dataclass(frozen=True)
class PastPlanActionLossWeights:
    action: float = 1.0
    history: float = 0.3
    future: float = 0.3
    preserve: float = 0.5
    delta_z: float = 0.01

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not torch.isfinite(torch.tensor(value)) or value < 0:
                raise ValueError(f"loss weight {name} must be finite and non-negative")


def _total_from_map_loss(
    value: Mapping[str, torch.Tensor] | None,
    name: str,
) -> torch.Tensor | None:
    if value is None:
        return None
    total = value.get("total")
    if not torch.is_tensor(total) or total.ndim != 0 or not torch.isfinite(total):
        raise PastPlanActionContractError(f"{name} loss lacks a finite scalar total")
    return total


def compose_past_plan_action_loss(
    *,
    stage: LossStage,
    history_loss: Mapping[str, torch.Tensor] | None,
    future_loss: Mapping[str, torch.Tensor] | None,
    action_plan_losses: Mapping[str, torch.Tensor] | None,
    weights: PastPlanActionLossWeights = PastPlanActionLossWeights(),
) -> dict[str, torch.Tensor]:
    """Compose all PPA objectives exactly once.

    Missing History or Future supervision in an alternating Stage-1 batch is
    represented by a differentiable zero.  At least one map target must exist.
    Stage 2 always requires native preservation from its first update.
    """

    weights.validate()
    history = _total_from_map_loss(history_loss, "history")
    future = _total_from_map_loss(future_loss, "future")
    if history is None and future is None:
        raise PastPlanActionContractError("at least one map supervision is required")
    reference = history if history is not None else future
    assert reference is not None
    zero = reference * 0.0
    history = zero if history is None else history
    future = zero if future is None else future

    if stage == "stage1_map_pretrain":
        if action_plan_losses is not None:
            raise PastPlanActionContractError(
                "Stage 1 must not execute the action/preservation path"
            )
        action = preserve = delta = zero
        total = weights.history * history + weights.future * future
    elif stage == "stage2_joint":
        if action_plan_losses is None:
            raise PastPlanActionContractError(
                "Stage 2 requires action, preserve, and delta losses from day one"
            )
        required = ("action", "preserve", "delta_z_l2")
        missing = [key for key in required if key not in action_plan_losses]
        if missing:
            raise PastPlanActionContractError(
                f"Stage 2 action-plan loss is incomplete: missing={missing}"
            )
        action, preserve, delta = (action_plan_losses[key] for key in required)
        for name, value in zip(required, (action, preserve, delta), strict=True):
            if not torch.is_tensor(value) or value.ndim != 0 or not torch.isfinite(value):
                raise PastPlanActionContractError(
                    f"Stage 2 {name} must be a finite scalar"
                )
        total = (
            weights.action * action
            + weights.history * history
            + weights.future * future
            + weights.preserve * preserve
            + weights.delta_z * delta
        )
    else:
        raise ValueError(f"unsupported loss stage: {stage!r}")

    return {
        "total": total,
        "action": action,
        "history": history,
        "future": future,
        "preserve": preserve,
        "delta_z_l2": delta,
    }

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.action.treatment_spec import TrajectoryPostprocessConfig
from src.models.past_plan_action import (
    PastPlanActionContractError,
    verify_stage0_treatment_equivalence,
)


class _FakeFrozenActionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()), requires_grad=False)

    def _fuse_projected_conditions(
        self, value: torch.Tensor, images: torch.Tensor | None
    ) -> torch.Tensor:
        if images is None:
            return value
        marker = images.float().mean().to(dtype=value.dtype).reshape(1, 1, 1)
        return value + marker * 0.0

    def get_trajectory_from_projected(
        self,
        value: torch.Tensor,
        *,
        traj_images: torch.Tensor | None,
        initial_noise: torch.Tensor,
    ) -> torch.Tensor:
        del traj_images
        offset = value.float().mean() * 0.01
        return initial_noise.float() * 0.0 + offset


def test_full_stage0_gate_includes_exact_treatment_spec() -> None:
    action_head = _FakeFrozenActionHead().eval()
    plan = torch.ones((1, 4, 8), dtype=torch.float32)
    noise = torch.zeros((2, 8, 3), dtype=torch.float32)
    report = verify_stage0_treatment_equivalence(
        action_head=action_head,
        plan_z0=plan,
        plan_z=plan.clone(),
        traj_images=None,
        initial_noise=noise,
        postprocess_config=TrajectoryPostprocessConfig(num_sample_trajs=2),
    )
    assert report.plan_equal is True
    assert report.raw_trajectory_equal is True
    assert report.treatment_spec_equal is True
    assert report.treatment_spec["end_reason"] == "anti_deadlock_replan"


def test_full_stage0_gate_refuses_nonidentical_plan() -> None:
    action_head = _FakeFrozenActionHead().eval()
    plan = torch.ones((1, 4, 8), dtype=torch.float32)
    with pytest.raises(PastPlanActionContractError, match="zero bridge"):
        verify_stage0_treatment_equivalence(
            action_head=action_head,
            plan_z0=plan,
            plan_z=plan + 1,
            traj_images=None,
            initial_noise=torch.zeros((2, 8, 3)),
            postprocess_config=TrajectoryPostprocessConfig(num_sample_trajs=2),
        )

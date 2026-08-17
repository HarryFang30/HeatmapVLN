"""Runtime configuration contract for the first Past→Plan→Action version."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


PPA_ARCHITECTURE_ID = "past-plan-action-single-bridge-v1"


@dataclass(frozen=True)
class PastPlanActionConfig:
    """Small, strict runtime view of the nested YAML configuration.

    The feature is disabled by default.  Keeping this object independent from
    Pydantic also lets evaluation/RPC entrypoints validate the same contract
    without importing the training schema.
    """

    enabled: bool = False
    stage: str = "stage0_equivalence"
    plan_dim: int = 768
    memory_dim: int = 256
    bridge_heads: int = 8
    plan_tokens: int = 4
    predict_steps: int = 32
    future_time_bins: int = 4
    future_views: int = 4
    future_heatmap_size: int = 64
    old_heatmap_control_enabled: bool = False
    pano_latent_adapter_enabled: bool = False
    architecture_id: str = PPA_ARCHITECTURE_ID

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "PastPlanActionConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("past_plan_action config must be a mapping")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown past_plan_action config fields: {sorted(unknown)}")
        return cls(**dict(value))

    def validate(self) -> "PastPlanActionConfig":
        if self.architecture_id != PPA_ARCHITECTURE_ID:
            raise ValueError(f"unsupported architecture_id: {self.architecture_id!r}")
        if self.stage not in {
            "stage0_equivalence",
            "stage1_map_pretrain",
            "stage2_joint",
        }:
            raise ValueError(f"unsupported Past→Plan→Action stage: {self.stage!r}")
        expected = {
            "plan_dim": (self.plan_dim, 768),
            "memory_dim": (self.memory_dim, 256),
            "bridge_heads": (self.bridge_heads, 8),
            "plan_tokens": (self.plan_tokens, 4),
            "predict_steps": (self.predict_steps, 32),
            "future_time_bins": (self.future_time_bins, 4),
            "future_views": (self.future_views, 4),
            "future_heatmap_size": (self.future_heatmap_size, 64),
        }
        mismatched = {
            name: {"actual": actual, "required": required}
            for name, (actual, required) in expected.items()
            if actual != required
        }
        if mismatched:
            raise ValueError(f"v1 dimension contract mismatch: {mismatched}")
        if self.enabled and (
            self.old_heatmap_control_enabled or self.pano_latent_adapter_enabled
        ):
            raise ValueError(
                "Past→Plan→Action v1 is mutually exclusive with legacy "
                "per-layer heatmap control and the pano latent adapter"
            )
        return self

    def runtime_manifest(self) -> dict[str, Any]:
        self.validate()
        return {
            "architecture_id": self.architecture_id,
            "enabled": self.enabled,
            "stage": self.stage,
            "plan_shape": [self.plan_tokens, self.plan_dim],
            "history_memory_dim": self.memory_dim,
            "future_shape": [
                self.future_time_bins,
                self.future_views,
                self.future_heatmap_size,
                self.future_heatmap_size,
            ],
            "future_time_ranges": [[1, 8], [9, 16], [17, 24], [25, 32]],
            "direction_order": ["front", "right", "back", "left"],
            "nextdit_modified": False,
            "decoded_heatmap_injected": False,
            "trajectory_heatmap_consistency_loss": False,
            "checkpoint_digest_enforced": False,
            "file_lock_used": False,
        }

"""Fixed-shape collation contract for Future trajectory supervision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .future_trajectory_heatmap import (
    FUTURE_HEATMAP_SCHEMA,
    FUTURE_HEATMAP_TARGET_SOURCE,
    FutureTrajectoryHeatmapTarget,
)


FUTURE_TARGET_KEYS = (
    "future_trajectory_heatmap",
    "future_trajectory_visibility",
    "future_trajectory_time_mask",
    "future_trajectory_anchor_heatmap",
    "future_trajectory_anchor_uv",
    "future_trajectory_view5",
)


class FutureTrajectoryBatchError(ValueError):
    """Raised when a sample exposes a partial or privileged Future target."""


def future_target_to_tensors(
    target: FutureTrajectoryHeatmapTarget,
) -> dict[str, torch.Tensor | str]:
    """Convert rendered labels to the only Future fields allowed in a batch."""

    return {
        "future_trajectory_heatmap": torch.from_numpy(target.heatmap).float(),
        "future_trajectory_visibility": torch.from_numpy(
            target.visibility
        ).float(),
        "future_trajectory_time_mask": torch.from_numpy(target.time_mask).bool(),
        "future_trajectory_anchor_heatmap": torch.from_numpy(
            target.anchor_heatmap
        ).float(),
        "future_trajectory_anchor_uv": torch.from_numpy(target.anchor_uv).float(),
        "future_trajectory_view5": torch.from_numpy(target.view5).long(),
        "future_trajectory_schema": target.schema,
        "future_trajectory_target_source": target.target_source,
    }


def _validate_sample_target(sample: Mapping[str, Any]) -> bool:
    present = [key in sample for key in FUTURE_TARGET_KEYS]
    metadata_present = any(
        key in sample
        for key in (
            "future_trajectory_schema",
            "future_trajectory_target_source",
        )
    )
    if any(present) and not all(present):
        missing = [
            key
            for key, exists in zip(FUTURE_TARGET_KEYS, present, strict=True)
            if not exists
        ]
        raise FutureTrajectoryBatchError(
            f"sample contains a partial Future target; missing={missing}"
        )
    if not all(present):
        if metadata_present:
            raise FutureTrajectoryBatchError(
                "Future target metadata exists without the fixed tensor target"
            )
        return False

    expected = {
        "future_trajectory_heatmap": (4, 4, 64, 64),
        "future_trajectory_visibility": (4, 4),
        "future_trajectory_time_mask": (4,),
        "future_trajectory_anchor_heatmap": (4, 4, 64, 64),
        "future_trajectory_anchor_uv": (4, 2),
        "future_trajectory_view5": (4,),
    }
    for key, shape in expected.items():
        value = sample[key]
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise FutureTrajectoryBatchError(
                f"{key} must be a tensor with shape {shape}, got "
                f"{type(value).__name__}/{getattr(value, 'shape', None)}"
            )
    if sample["future_trajectory_time_mask"].dtype != torch.bool:
        raise FutureTrajectoryBatchError(
            "future_trajectory_time_mask must be bool"
        )
    if sample["future_trajectory_view5"].dtype != torch.long:
        raise FutureTrajectoryBatchError(
            "future_trajectory_view5 must be int64"
        )

    bounded = (
        "future_trajectory_heatmap",
        "future_trajectory_visibility",
        "future_trajectory_anchor_heatmap",
    )
    for key in bounded:
        value = sample[key]
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise FutureTrajectoryBatchError(
                f"{key} must be a finite floating tensor"
            )
        if bool(((value < 0) | (value > 1)).any()):
            raise FutureTrajectoryBatchError(f"{key} must lie in [0,1]")

    anchor_uv = sample["future_trajectory_anchor_uv"]
    if not anchor_uv.is_floating_point():
        raise FutureTrajectoryBatchError(
            "future_trajectory_anchor_uv must be floating"
        )
    view5 = sample["future_trajectory_view5"]
    if bool(((view5 < 0) | (view5 > 4)).any()):
        raise FutureTrajectoryBatchError(
            "future_trajectory_view5 must lie in [0,4]"
        )
    anchor_present = view5 > 0
    if bool((~torch.isfinite(anchor_uv[anchor_present])).any()):
        raise FutureTrajectoryBatchError(
            "visible Future anchors require finite UV"
        )
    if bool(torch.isfinite(anchor_uv[~anchor_present]).any()):
        raise FutureTrajectoryBatchError(
            "NONE Future anchors must use NaN UV"
        )
    if sample.get("future_trajectory_schema") != FUTURE_HEATMAP_SCHEMA:
        raise FutureTrajectoryBatchError(
            "Future target schema is missing or incompatible"
        )
    if (
        sample.get("future_trajectory_target_source")
        != FUTURE_HEATMAP_TARGET_SOURCE
    ):
        raise FutureTrajectoryBatchError(
            "Future target must come from the exact expert System-1 action target"
        )
    return True


def stack_future_trajectory_targets(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, torch.Tensor | list[str]]:
    """Stack expert/mixed rows; rows without System-1 get false masks."""

    if not samples:
        raise FutureTrajectoryBatchError("cannot collate an empty sample list")
    has_target = [_validate_sample_target(sample) for sample in samples]
    defaults = {
        "future_trajectory_heatmap": torch.zeros(4, 4, 64, 64),
        "future_trajectory_visibility": torch.zeros(4, 4),
        "future_trajectory_time_mask": torch.zeros(4, dtype=torch.bool),
        "future_trajectory_anchor_heatmap": torch.zeros(4, 4, 64, 64),
        "future_trajectory_anchor_uv": torch.full((4, 2), torch.nan),
        "future_trajectory_view5": torch.zeros(4, dtype=torch.long),
    }

    result: dict[str, torch.Tensor | list[str]] = {}
    for key in FUTURE_TARGET_KEYS:
        result[key] = torch.stack(
            [
                sample[key] if present else defaults[key].clone()
                for sample, present in zip(samples, has_target, strict=True)
            ],
            dim=0,
        )
    result["future_trajectory_schema"] = [
        str(sample.get("future_trajectory_schema", "none"))
        for sample in samples
    ]
    result["future_trajectory_target_source"] = [
        str(sample.get("future_trajectory_target_source", "none"))
        for sample in samples
    ]
    result["future_trajectory_target_present"] = torch.tensor(
        has_target, dtype=torch.bool
    )
    return result


def assert_no_future_teacher_inputs(batch: Mapping[str, Any]) -> None:
    """Fail closed if raw future pose/depth geometry reaches model input."""

    leaked: set[str] = set()
    for key in batch:
        normalized = str(key).lower()
        if "future" in normalized and any(
            teacher_word in normalized
            for teacher_word in (
                "pose",
                "depth",
                "c2w",
                "camera_point",
            )
        ):
            leaked.add(str(key))
    if leaked:
        raise FutureTrajectoryBatchError(
            "future teacher geometry must never be forwarded to the model: "
            f"{sorted(leaked)}"
        )

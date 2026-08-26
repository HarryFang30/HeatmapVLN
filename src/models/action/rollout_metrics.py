"""Sampled-rollout validation metrics for the Past->Plan->Action bridge.

Teacher-forced velocity MSE cannot see damage along the sampler's own ODE
path: the unconstrained v1 bridge kept a flat action loss while closed-loop SR
collapsed from 63% to 18%.  These helpers score actual sampled trajectory
banks through the exact deployment post-processing from
:mod:`src.models.action.treatment_spec`, so checkpoint selection optimizes a
faithful proxy of what the evaluator executes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .treatment_spec import (
    TrajectoryPostprocessConfig,
    build_treatment_spec,
    reconstruct_xy_from_delta,
    select_trajectory_xy,
)


def _as_delta_array(value: Any) -> np.ndarray:
    array = (
        value.detach().float().cpu().numpy()
        if torch.is_tensor(value)
        else np.asarray(value, dtype=np.float32)
    )
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] < 2:
        raise ValueError(f"expected [N,T,>=2] delta samples, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("delta samples contain non-finite values")
    return array


def selected_endpoint_xy(
    delta_bank: Any,
    config: TrajectoryPostprocessConfig,
) -> np.ndarray:
    """Endpoint of the deployment-selected path, in meters.

    Applies exactly the deployed scaling (``action_scale``, ``x_sign``,
    cumulative reconstruction) and the configured trajectory selection to a
    bank of raw diffusion deltas.
    """

    config.validate()
    deltas = _as_delta_array(delta_bank)[: config.num_sample_trajs].copy()
    if deltas.shape[0] < config.num_sample_trajs:
        raise ValueError(
            f"need {config.num_sample_trajs} samples, got {deltas.shape[0]}"
        )
    deltas[:, :, :2] /= float(config.action_scale)
    deltas[:, :, 0] *= float(config.trajectory_x_sign)
    paths = reconstruct_xy_from_delta(deltas)
    selected, _index = select_trajectory_xy(
        paths,
        config.trajectory_selection,
        discretizer=config,
    )
    return np.asarray(selected[-1, :2], dtype=np.float64)


def gt_endpoint_xy(
    gt_trajectory: Any,
    config: TrajectoryPostprocessConfig,
) -> np.ndarray:
    """Endpoint of the ground-truth scaled-delta trajectory, in meters."""

    array = (
        gt_trajectory.detach().float().cpu().numpy()
        if torch.is_tensor(gt_trajectory)
        else np.asarray(gt_trajectory, dtype=np.float32)
    )
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or array.shape[-1] < 2:
        raise ValueError(f"expected [T,>=2] GT deltas, got {array.shape}")
    deltas = array[None, :, :2].copy()
    deltas /= float(config.action_scale)
    deltas[:, :, 0] *= float(config.trajectory_x_sign)
    path = reconstruct_xy_from_delta(deltas)[0]
    return np.asarray(path[-1, :2], dtype=np.float64)


def compute_rollout_pair_metrics(
    *,
    bank_bridged: Any,
    bank_native: Any,
    gt_trajectory: Any,
    config: TrajectoryPostprocessConfig,
) -> dict[str, float]:
    """Score one bridged/native sampled pair against the GT trajectory.

    Both banks must come from the same shared initial noise so the native and
    bridged rollouts differ only through the Plan delta.  ``action_agreement``
    is 1.0 exactly when the full deployment treatment (selection, scaling,
    discretization, padding, anti-deadlock) yields identical response queues.
    """

    endpoint_bridged = selected_endpoint_xy(bank_bridged, config)
    endpoint_native = selected_endpoint_xy(bank_native, config)
    endpoint_gt = gt_endpoint_xy(gt_trajectory, config)
    spec_bridged = build_treatment_spec(bank_bridged, config)
    spec_native = build_treatment_spec(bank_native, config)
    return {
        "endpoint_error": float(
            np.linalg.norm(endpoint_bridged - endpoint_gt)
        ),
        "endpoint_error_native": float(
            np.linalg.norm(endpoint_native - endpoint_gt)
        ),
        "endpoint_gap_to_native": float(
            np.linalg.norm(endpoint_bridged - endpoint_native)
        ),
        "action_agreement": float(
            spec_bridged.response_actions == spec_native.response_actions
        ),
    }


__all__ = [
    "compute_rollout_pair_metrics",
    "gt_endpoint_xy",
    "selected_endpoint_xy",
]

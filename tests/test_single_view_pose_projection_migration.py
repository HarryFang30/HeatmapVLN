"""Regression tests for the audited legacy-to-Habitat pose projection."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from scripts.training.single_view_pose_projection_migration import (
    SUPPORTED_SPEC,
    PoseProjectionMigrationError,
    legacy_pose_from_physical_pose,
    migrate_proj_traj_tensors,
)


def _sinusoidal_pe_reference(
    value: torch.Tensor,
    *,
    num_freqs: int,
    max_spatial_range: float,
) -> torch.Tensor:
    """Independent implementation of the pinned trajectory PE layout."""
    normalized = value / max_spatial_range
    frequencies = torch.arange(
        num_freqs,
        dtype=value.dtype,
        device=value.device,
    )
    frequencies = 2.0 * math.pi * (2.0**frequencies)
    angles = normalized.unsqueeze(-1) * frequencies
    return torch.cat(
        (
            normalized,
            angles.sin().flatten(-2),
            angles.cos().flatten(-2),
        ),
        dim=-1,
    )


def test_migrated_proj_traj_is_equivalent_for_corrected_pose_fields() -> None:
    generator = torch.Generator().manual_seed(20260802)
    yaw = (
        torch.rand(512, generator=generator, dtype=torch.float64) - 0.5
    ) * (8.0 * math.pi)
    physical_pose = torch.column_stack(
        (
            torch.randn(512, generator=generator, dtype=torch.float64) * 20.0,
            torch.randn(512, generator=generator, dtype=torch.float64) * 20.0,
            yaw.cos(),
            yaw.sin(),
        )
    )
    legacy_pose = legacy_pose_from_physical_pose(physical_pose)

    physical_pe = _sinusoidal_pe_reference(
        physical_pose,
        num_freqs=SUPPORTED_SPEC.num_freqs,
        max_spatial_range=SUPPORTED_SPEC.max_spatial_range,
    )
    legacy_pe = _sinusoidal_pe_reference(
        legacy_pose,
        num_freqs=SUPPORTED_SPEC.num_freqs,
        max_spatial_range=SUPPORTED_SPEC.max_spatial_range,
    )

    weight = torch.randn(
        SUPPORTED_SPEC.d_attn,
        SUPPORTED_SPEC.pe_dim,
        generator=generator,
        dtype=torch.float64,
    )
    bias = torch.randn(
        SUPPORTED_SPEC.d_attn,
        generator=generator,
        dtype=torch.float64,
    )
    source_weight = weight.clone()
    source_bias = bias.clone()

    migrated_weight, migrated_bias = migrate_proj_traj_tensors(weight, bias)

    old_output = F.linear(legacy_pe, weight, bias)
    new_output = F.linear(physical_pe, migrated_weight, migrated_bias)
    assert torch.allclose(old_output, new_output, rtol=2e-15, atol=2e-13)
    assert torch.equal(weight, source_weight)
    assert torch.equal(bias, source_bias)
    assert torch.equal(migrated_bias, bias)
    assert migrated_weight.data_ptr() != weight.data_ptr()
    assert migrated_bias.data_ptr() != bias.data_ptr()


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("source_pose_convention", "unknown_source_convention"),
        ("target_pose_convention", "unknown_target_convention"),
    ],
)
def test_rejects_unapproved_pose_conventions(
    field: str,
    invalid_value: str,
) -> None:
    invalid_spec = replace(SUPPORTED_SPEC, **{field: invalid_value})
    weight = torch.zeros(
        SUPPORTED_SPEC.d_attn,
        SUPPORTED_SPEC.pe_dim,
    )
    bias = torch.zeros(SUPPORTED_SPEC.d_attn)

    with pytest.raises(PoseProjectionMigrationError, match="Unsupported pose/PE"):
        migrate_proj_traj_tensors(weight, bias, invalid_spec)


@pytest.mark.parametrize(
    ("weight_shape", "bias_shape"),
    [
        ((SUPPORTED_SPEC.d_attn, SUPPORTED_SPEC.pe_dim - 1), (SUPPORTED_SPEC.d_attn,)),
        ((SUPPORTED_SPEC.d_attn, SUPPORTED_SPEC.pe_dim), (SUPPORTED_SPEC.d_attn - 1,)),
    ],
)
def test_rejects_unapproved_projection_shapes(
    weight_shape: tuple[int, ...],
    bias_shape: tuple[int, ...],
) -> None:
    with pytest.raises(PoseProjectionMigrationError, match="Unsupported proj_traj shape"):
        migrate_proj_traj_tensors(
            torch.zeros(weight_shape),
            torch.zeros(bias_shape),
        )

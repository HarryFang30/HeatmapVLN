"""Fail-closed migration of the legacy trajectory projection.

The legacy heatmap head was trained with a ``+Z`` pose wrapper while the new
single-view data path exposes the physical Habitat ``-Z`` convention.  For the
four model inputs the relation is

``old_pose = diag(-1, +1, +1, -1) @ new_pose``

where the fields are ``[forward, left, cos(yaw), sin(yaw)]``.  The existing
positional encoder concatenates

``[raw, dim-major sin(raw * frequencies), dim-major cos(raw * frequencies)]``.

Raw values and sine features are odd, whereas cosine features are even.
Consequently the old positional encoding is a diagonal sign transform of the
new positional encoding.  Right-multiplying ``proj_traj.weight`` by that
diagonal transform preserves the complete downstream trajectory token exactly
without training or changing the bias.

This module intentionally supports only the audited current configuration.  A
shape, convention, PE layout, frequency count, range, or attention-width
change requires a new migration version rather than an inferred transform.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

TRANSFORM_SCHEMA_VERSION = 1
TRANSFORM_ALGORITHM_ID = (
    "proj_traj_pe_column_signs_legacy_heatmapvln_wrapper_plus_z_"
    "to_habitat_c2w_minus_z_v1"
)

SOURCE_POSE_CONVENTION = (
    "legacy_heatmapvln_wrapper_c2w_plus_z_fields__"
    "forward_left_cos_yaw_sin_yaw__v1"
)
TARGET_POSE_CONVENTION = (
    "habitat_c2w_minus_z__forward_left_cos_yaw_sin_yaw__v1"
)
POSE_FIELDS = (
    "forward_m",
    "left_m",
    "cos_yaw_left_positive",
    "sin_yaw_left_positive",
)
OLD_FROM_NEW_DIAGONAL = (-1, 1, 1, -1)
PE_LAYOUT = "raw__dim_major_sin__dim_major_cos"

PROJ_TRAJ_WEIGHT_KEY = "heatmap_vln.coarse.proj_traj.weight"
PROJ_TRAJ_BIAS_KEY = "heatmap_vln.coarse.proj_traj.bias"

# Read-only audited on the pinned legacy checkpoint.  These hashes use the
# same name/shape/dtype/byte content algorithm as checkpoint_contract.py.
AUDITED_SOURCE_SELECTED_STATE_CONTENT_SHA256 = (
    "981bbe19260a834bd87cf71766f194d9fb2bc4ac1d7472d8957bfb680a48138e"
)
AUDITED_OUTPUT_SELECTED_STATE_CONTENT_SHA256 = (
    "1e68459757d943184376cd66946e07dd07d5f05b5d322ebbd4d719024d585861"
)
AUDITED_SOURCE_PROJ_TRAJ_WEIGHT_SHA256 = (
    "f13b5c169206003e1b03da0fb2aa1785964c78bd9cdccd2020e2647a09db848b"
)
AUDITED_OUTPUT_PROJ_TRAJ_WEIGHT_SHA256 = (
    "a4ad64bdbd007c9761e0b36582fa8868663d177414ba4c4e7142efb263e038f3"
)
AUDITED_PROJ_TRAJ_BIAS_SHA256 = (
    "19e542e004da6ffe48770e06431c1017c84c768d0464ee9379bedb526594e786"
)


class PoseProjectionMigrationError(RuntimeError):
    """Raised when equivalence cannot be proven from the exact contract."""


@dataclass(frozen=True)
class PoseProjectionMigrationSpec:
    """Complete versioned contract for the one approved migration."""

    schema_version: int = TRANSFORM_SCHEMA_VERSION
    algorithm_id: str = TRANSFORM_ALGORITHM_ID
    source_pose_convention: str = SOURCE_POSE_CONVENTION
    target_pose_convention: str = TARGET_POSE_CONVENTION
    source_pose_fields: tuple[str, ...] = POSE_FIELDS
    target_pose_fields: tuple[str, ...] = POSE_FIELDS
    old_from_new_diagonal: tuple[int, ...] = OLD_FROM_NEW_DIAGONAL
    pe_layout: str = PE_LAYOUT
    pose_dim: int = 4
    num_freqs: int = 16
    max_spatial_range: float = 10.0
    d_attn: int = 256
    weight_key: str = PROJ_TRAJ_WEIGHT_KEY
    bias_key: str = PROJ_TRAJ_BIAS_KEY

    @property
    def pe_dim(self) -> int:
        return self.pose_dim * (1 + 2 * self.num_freqs)


SUPPORTED_SPEC = PoseProjectionMigrationSpec()


def _require_supported_spec(spec: PoseProjectionMigrationSpec) -> None:
    if not isinstance(spec, PoseProjectionMigrationSpec):
        raise PoseProjectionMigrationError(
            "spec must be PoseProjectionMigrationSpec; refusing an unversioned config"
        )
    expected = asdict(SUPPORTED_SPEC)
    actual = asdict(spec)
    mismatches = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in expected
        if actual[name] != expected[name]
    }
    if mismatches:
        raise PoseProjectionMigrationError(
            "Unsupported pose/PE/projection migration config; create a new "
            f"versioned policy instead of inferring it: {mismatches}"
        )


def sinusoidal_pe_column_signs(
    *,
    pose_signs: Sequence[int],
    num_freqs: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return ``PE(old_pose) = signs * PE(new_pose)`` column signs.

    This helper implements the current encoder layout and is useful in tests or
    a future explicitly-versioned policy.  The production migration still
    calls :func:`_require_supported_spec` and therefore cannot silently accept
    a different configuration.
    """

    if isinstance(num_freqs, bool) or not isinstance(num_freqs, int) or num_freqs <= 0:
        raise PoseProjectionMigrationError(
            f"num_freqs must be a positive integer, got {num_freqs!r}"
        )
    signs = tuple(pose_signs)
    if not signs or any(value not in (-1, 1) for value in signs):
        raise PoseProjectionMigrationError(
            f"pose_signs must be a non-empty sequence of +/-1, got {signs!r}"
        )

    raw = torch.tensor(signs, dtype=dtype, device=device)
    # ``angles.sin().flatten(-2)`` is dimension-major, so each raw field's
    # sign is repeated for all frequencies before moving to the next field.
    sine = raw.repeat_interleave(num_freqs)
    cosine = torch.ones(len(signs) * num_freqs, dtype=dtype, device=device)
    return torch.cat((raw, sine, cosine), dim=0)


def supported_column_signs(
    spec: PoseProjectionMigrationSpec = SUPPORTED_SPEC,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the audited PE sign diagonal after validating the whole spec."""

    _require_supported_spec(spec)
    signs = sinusoidal_pe_column_signs(
        pose_signs=spec.old_from_new_diagonal,
        num_freqs=spec.num_freqs,
        dtype=dtype,
        device=device,
    )
    if signs.numel() != spec.pe_dim:
        raise AssertionError((signs.numel(), spec.pe_dim))
    return signs


def legacy_pose_from_physical_pose(
    physical_pose: torch.Tensor,
    spec: PoseProjectionMigrationSpec = SUPPORTED_SPEC,
) -> torch.Tensor:
    """Convert new ``-Z`` model fields to their legacy ``+Z`` values."""

    _require_supported_spec(spec)
    if not torch.is_tensor(physical_pose) or physical_pose.shape[-1:] != (spec.pose_dim,):
        shape = getattr(physical_pose, "shape", None)
        raise PoseProjectionMigrationError(
            f"physical_pose must be a tensor [...,{spec.pose_dim}], got {shape}"
        )
    if not physical_pose.is_floating_point():
        raise PoseProjectionMigrationError("physical_pose must be floating point")
    signs = physical_pose.new_tensor(spec.old_from_new_diagonal)
    return physical_pose * signs


def _validate_projection_tensors(
    weight: torch.Tensor,
    bias: torch.Tensor,
    spec: PoseProjectionMigrationSpec,
) -> None:
    _require_supported_spec(spec)
    if not torch.is_tensor(weight) or not torch.is_tensor(bias):
        raise PoseProjectionMigrationError("proj_traj weight and bias must be tensors")
    if weight.layout != torch.strided or bias.layout != torch.strided:
        raise PoseProjectionMigrationError("proj_traj tensors must be dense strided tensors")
    expected_weight = (spec.d_attn, spec.pe_dim)
    expected_bias = (spec.d_attn,)
    if tuple(weight.shape) != expected_weight or tuple(bias.shape) != expected_bias:
        raise PoseProjectionMigrationError(
            "Unsupported proj_traj shape: "
            f"weight={tuple(weight.shape)} bias={tuple(bias.shape)}; "
            f"expected weight={expected_weight} bias={expected_bias}"
        )
    if not weight.is_floating_point() or not bias.is_floating_point():
        raise PoseProjectionMigrationError("proj_traj tensors must be floating point")
    if weight.dtype != bias.dtype:
        raise PoseProjectionMigrationError(
            f"proj_traj dtype mismatch: weight={weight.dtype}, bias={bias.dtype}"
        )
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise PoseProjectionMigrationError("proj_traj tensors contain non-finite values")


def migrate_proj_traj_tensors(
    weight: torch.Tensor,
    bias: torch.Tensor,
    spec: PoseProjectionMigrationSpec = SUPPORTED_SPEC,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weights preserving ``old_proj(old_pe) == new_proj(new_pe)``.

    ``nn.Linear`` stores weights as ``[out_features, in_features]``.  Therefore
    right multiplication by the diagonal PE transform is implemented by
    column-wise multiplication.  Bias is invariant and is copied unchanged.
    Input tensors are never mutated.
    """

    _validate_projection_tensors(weight, bias, spec)
    signs = supported_column_signs(spec, dtype=weight.dtype, device=weight.device)
    migrated_weight = weight.detach().clone() * signs.unsqueeze(0)
    migrated_bias = bias.detach().clone()
    return migrated_weight, migrated_bias


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    dense = tensor.detach().to(device="cpu").contiguous()
    if dense.layout != torch.strided:
        raise PoseProjectionMigrationError(
            f"Only dense strided tensors can be hashed, got {dense.layout}"
        )
    return dense.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()


def _tensor_record(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "sha256": tensor_sha256(tensor),
    }


def state_content_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, metadata, and bytes independent of ``torch.save``."""

    if not isinstance(state, Mapping) or not state:
        raise PoseProjectionMigrationError("state must be a non-empty tensor mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise PoseProjectionMigrationError("state must map string names to tensors")
        header = json.dumps(
            {
                "name": name,
                "shape": [int(value) for value in tensor.shape],
                "dtype": str(tensor.dtype),
                "numel": int(tensor.numel()),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = _tensor_bytes(tensor)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _validate_state_keys(
    state: Mapping[str, torch.Tensor],
    spec: PoseProjectionMigrationSpec,
) -> None:
    _require_supported_spec(spec)
    if not isinstance(state, Mapping) or not state:
        raise PoseProjectionMigrationError("state must be a non-empty tensor mapping")
    if any(not isinstance(name, str) or not torch.is_tensor(value) for name, value in state.items()):
        raise PoseProjectionMigrationError("state must map string names to tensors")
    required = {spec.weight_key, spec.bias_key}
    missing = sorted(required - set(state))
    aliases = sorted(
        name
        for name in state
        if name not in required
        and (
            name.endswith("coarse.proj_traj.weight")
            or name.endswith("coarse.proj_traj.bias")
        )
    )
    if missing or aliases:
        raise PoseProjectionMigrationError(
            "Canonical proj_traj keys are required exactly once: "
            f"missing={missing}, aliases={aliases}"
        )


def build_pose_projection_transform_manifest(
    source_state: Mapping[str, torch.Tensor],
    migrated_state: Mapping[str, torch.Tensor],
    spec: PoseProjectionMigrationSpec = SUPPORTED_SPEC,
) -> dict[str, Any]:
    """Build and verify provenance for the transformed selected state.

    The resulting mapping is intended for
    ``metadata['pose_projection_migration']``.  The enclosing artifact's
    selected-state content hash must refer to the *output* state, while the
    previously audited hash remains recorded here as the source-before-transform
    hash.  Per-tensor hashes prove exactly which weight changed and that the
    bias and every unrelated tensor stayed byte-identical.
    """

    _validate_state_keys(source_state, spec)
    _validate_state_keys(migrated_state, spec)
    if set(source_state) != set(migrated_state):
        raise PoseProjectionMigrationError("Migration changed the selected-state key set")

    expected_weight, expected_bias = migrate_proj_traj_tensors(
        source_state[spec.weight_key], source_state[spec.bias_key], spec
    )
    if not torch.equal(migrated_state[spec.weight_key], expected_weight):
        raise PoseProjectionMigrationError("Migrated proj_traj weight is not the exact transform")
    if not torch.equal(migrated_state[spec.bias_key], expected_bias):
        raise PoseProjectionMigrationError("Migrated proj_traj bias is not an exact copy")

    changed_unrelated = [
        name
        for name in source_state
        if name not in {spec.weight_key, spec.bias_key}
        and (
            source_state[name].dtype != migrated_state[name].dtype
            or tuple(source_state[name].shape) != tuple(migrated_state[name].shape)
            or not torch.equal(source_state[name], migrated_state[name])
        )
    ]
    if changed_unrelated:
        raise PoseProjectionMigrationError(
            f"Migration changed unrelated tensors: {sorted(changed_unrelated)[:8]}"
        )

    signs = supported_column_signs(spec, dtype=torch.int8, device="cpu")
    source_weight = source_state[spec.weight_key]
    output_weight = migrated_state[spec.weight_key]
    source_bias = source_state[spec.bias_key]
    output_bias = migrated_state[spec.bias_key]
    return {
        "schema_version": spec.schema_version,
        "algorithm_id": spec.algorithm_id,
        "source_pose_convention": spec.source_pose_convention,
        "target_pose_convention": spec.target_pose_convention,
        "source_pose_fields": list(spec.source_pose_fields),
        "target_pose_fields": list(spec.target_pose_fields),
        "old_pose_from_new_pose_diagonal": list(spec.old_from_new_diagonal),
        "sinusoidal_pe_contract": {
            "layout": spec.pe_layout,
            "pose_dim": spec.pose_dim,
            "num_freqs": spec.num_freqs,
            "max_spatial_range": spec.max_spatial_range,
            "pe_dim": spec.pe_dim,
            "column_signs_sha256": tensor_sha256(signs),
            "negative_column_indices": torch.nonzero(signs < 0, as_tuple=False)
            .flatten()
            .tolist(),
        },
        "transformed_tensors": [
            {
                "name": spec.weight_key,
                "operation": "weight_columns_times_pe_old_from_new_sign_diagonal",
                "source": _tensor_record(spec.weight_key, source_weight),
                "output": _tensor_record(spec.weight_key, output_weight),
            },
            {
                "name": spec.bias_key,
                "operation": "identity_copy_bias_invariant",
                "source": _tensor_record(spec.bias_key, source_bias),
                "output": _tensor_record(spec.bias_key, output_bias),
            },
        ],
        "unchanged_tensor_count": len(source_state) - 2,
        "source_selected_state_content_sha256": state_content_sha256(source_state),
        "output_selected_state_content_sha256": state_content_sha256(migrated_state),
    }


def migrate_selected_state_proj_traj(
    source_state: Mapping[str, torch.Tensor],
    spec: PoseProjectionMigrationSpec = SUPPORTED_SPEC,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Clone a selected warm-start state and migrate only ``proj_traj``."""

    _validate_state_keys(source_state, spec)
    migrated = {
        name: tensor.detach().clone()
        for name, tensor in source_state.items()
    }
    migrated_weight, migrated_bias = migrate_proj_traj_tensors(
        source_state[spec.weight_key], source_state[spec.bias_key], spec
    )
    migrated[spec.weight_key] = migrated_weight
    migrated[spec.bias_key] = migrated_bias
    manifest = build_pose_projection_transform_manifest(source_state, migrated, spec)
    return migrated, manifest


def migrate_audited_selected_state_proj_traj(
    source_state: Mapping[str, torch.Tensor],
    spec: PoseProjectionMigrationSpec = SUPPORTED_SPEC,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Migrate only the pinned audited 53-tensor warm-start selection.

    This is the production-facing entry point.  It refuses a same-shaped but
    byte-different source and pins the deterministic transformed output, not
    merely the legacy pre-transform identity.
    """

    _validate_state_keys(source_state, spec)
    source_state_hash = state_content_sha256(source_state)
    source_weight_hash = tensor_sha256(source_state[spec.weight_key])
    source_bias_hash = tensor_sha256(source_state[spec.bias_key])
    source_mismatches = {}
    for name, expected, actual in (
        (
            "source_selected_state_content_sha256",
            AUDITED_SOURCE_SELECTED_STATE_CONTENT_SHA256,
            source_state_hash,
        ),
        (
            "source_proj_traj_weight_sha256",
            AUDITED_SOURCE_PROJ_TRAJ_WEIGHT_SHA256,
            source_weight_hash,
        ),
        (
            "source_proj_traj_bias_sha256",
            AUDITED_PROJ_TRAJ_BIAS_SHA256,
            source_bias_hash,
        ),
    ):
        if actual != expected:
            source_mismatches[name] = {"expected": expected, "actual": actual}
    if source_mismatches:
        raise PoseProjectionMigrationError(
            "Selected state is not the pinned audited pre-transform source: "
            f"{source_mismatches}"
        )

    migrated, manifest = migrate_selected_state_proj_traj(source_state, spec)
    output_state_hash = state_content_sha256(migrated)
    output_weight_hash = tensor_sha256(migrated[spec.weight_key])
    output_bias_hash = tensor_sha256(migrated[spec.bias_key])
    output_mismatches = {}
    for name, expected, actual in (
        (
            "output_selected_state_content_sha256",
            AUDITED_OUTPUT_SELECTED_STATE_CONTENT_SHA256,
            output_state_hash,
        ),
        (
            "output_proj_traj_weight_sha256",
            AUDITED_OUTPUT_PROJ_TRAJ_WEIGHT_SHA256,
            output_weight_hash,
        ),
        (
            "output_proj_traj_bias_sha256",
            AUDITED_PROJ_TRAJ_BIAS_SHA256,
            output_bias_hash,
        ),
    ):
        if actual != expected:
            output_mismatches[name] = {"expected": expected, "actual": actual}
    if output_mismatches:
        raise PoseProjectionMigrationError(
            "Deterministic audited pose-projection output identity mismatch: "
            f"{output_mismatches}"
        )

    manifest = dict(manifest)
    manifest["audited_checkpoint_contract"] = {
        "verified": True,
        "source_selected_state_content_sha256": (
            AUDITED_SOURCE_SELECTED_STATE_CONTENT_SHA256
        ),
        "output_selected_state_content_sha256": (
            AUDITED_OUTPUT_SELECTED_STATE_CONTENT_SHA256
        ),
        "source_proj_traj_weight_sha256": (
            AUDITED_SOURCE_PROJ_TRAJ_WEIGHT_SHA256
        ),
        "output_proj_traj_weight_sha256": (
            AUDITED_OUTPUT_PROJ_TRAJ_WEIGHT_SHA256
        ),
        "proj_traj_bias_sha256": AUDITED_PROJ_TRAJ_BIAS_SHA256,
    }
    return migrated, manifest

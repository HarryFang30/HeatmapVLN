"""Fail-closed loader for the frozen single-view heatmap dependency.

This loader deliberately does not call a generic model ``load_state_dict``.
The checkpoint is an external dependency of heatmap-control training, so it
must identify exactly one complete ``model.heatmap_vln`` parameter set and
must not contain any trainable navigation or language-model tensors.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


STATE_KEY = "trainable_state_dict"
TARGET_PREFIX = "heatmap_vln."
DEPENDENCY_SCHEMA = "frozen-heatmap-checkpoint-v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_MARKERS = (
    "lora",
    "qwen",
    "system1",
    "system2",
    "nextdit",
    "adapter",
    "tokenizer",
    "control",
)


class FrozenHeatmapCheckpointError(RuntimeError):
    """The frozen heatmap dependency failed its integrity contract."""


def compute_file_sha256(path: str | Path) -> str:
    """Compute a checkpoint file digest without deserializing its payload."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_expected_sha256(expected_sha256: str) -> str:
    if (
        not isinstance(expected_sha256, str)
        or _SHA256_PATTERN.fullmatch(expected_sha256) is None
    ):
        raise FrozenHeatmapCheckpointError(
            "expected_sha256 must be exactly 64 lowercase hexadecimal characters"
        )
    return expected_sha256


def _load_weights_only(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # never fall back to pickle-enabled deserialization
        raise FrozenHeatmapCheckpointError(
            "Frozen heatmap loading requires torch.load(..., weights_only=True)"
        ) from exc
    except Exception as exc:
        raise FrozenHeatmapCheckpointError(
            f"Unable to deserialize frozen heatmap checkpoint: {path}"
        ) from exc


def _canonical_key(raw_name: str) -> str:
    if not isinstance(raw_name, str) or not raw_name:
        raise FrozenHeatmapCheckpointError(
            "trainable_state_dict keys must be non-empty strings"
        )
    name = raw_name
    while name.startswith("module."):
        name = name[len("module.") :]
    name = name.replace(".module.", ".")
    if not name:
        raise FrozenHeatmapCheckpointError(
            f"Checkpoint key normalizes to an empty name: {raw_name!r}"
        )
    return name


def _contains_forbidden_marker(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _FORBIDDEN_MARKERS)


def _extract_heatmap_state(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise FrozenHeatmapCheckpointError("Checkpoint payload must be a mapping")
    if STATE_KEY not in payload:
        raise FrozenHeatmapCheckpointError(
            f"Checkpoint must contain payload.{STATE_KEY}; no fallback state key is allowed"
        )
    raw_state = payload[STATE_KEY]
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise FrozenHeatmapCheckpointError(
            f"payload.{STATE_KEY} must be a non-empty tensor mapping"
        )

    normalized: dict[str, torch.Tensor] = {}
    raw_for_canonical: dict[str, str] = {}
    for raw_name, value in raw_state.items():
        name = _canonical_key(raw_name)
        if name in normalized:
            raise FrozenHeatmapCheckpointError(
                "Duplicate checkpoint keys after module-prefix normalization: "
                f"{raw_for_canonical[name]!r} and {raw_name!r} -> {name!r}"
            )
        if _contains_forbidden_marker(name):
            raise FrozenHeatmapCheckpointError(
                f"Forbidden LoRA/Qwen/System1/NextDiT/adapter/tokenizer/control key: {name}"
            )
        if not name.startswith(TARGET_PREFIX):
            raise FrozenHeatmapCheckpointError(
                f"Unexpected non-heatmap checkpoint key: {name}"
            )
        local_name = name[len(TARGET_PREFIX) :]
        if not local_name:
            raise FrozenHeatmapCheckpointError(
                f"Checkpoint key does not name a heatmap parameter: {name}"
            )
        if not torch.is_tensor(value):
            raise FrozenHeatmapCheckpointError(
                f"Checkpoint value is not a tensor: {name}"
            )
        if value.layout != torch.strided:
            raise FrozenHeatmapCheckpointError(
                f"Checkpoint tensor must use dense strided layout: {name}"
            )
        normalized[name] = value
        raw_for_canonical[name] = raw_name
    return normalized


def _resolve_heatmap_module(model: nn.Module) -> nn.Module:
    if not isinstance(model, nn.Module):
        raise FrozenHeatmapCheckpointError("model must be a torch.nn.Module")
    module = model
    visited: set[int] = set()
    while getattr(module, "heatmap_vln", None) is None:
        module_id = id(module)
        if module_id in visited:
            raise FrozenHeatmapCheckpointError("Cyclic model.module wrapper")
        visited.add(module_id)
        wrapped = getattr(module, "module", None)
        if not isinstance(wrapped, nn.Module):
            raise FrozenHeatmapCheckpointError(
                "model.heatmap_vln must exist before loading its frozen checkpoint"
            )
        module = wrapped
    heatmap = getattr(module, "heatmap_vln")
    if not isinstance(heatmap, nn.Module):
        raise FrozenHeatmapCheckpointError("model.heatmap_vln must be a torch.nn.Module")
    return heatmap


def _target_parameters(heatmap: nn.Module) -> dict[str, nn.Parameter]:
    parameters = dict(heatmap.named_parameters())
    if not parameters:
        raise FrozenHeatmapCheckpointError(
            "model.heatmap_vln has no named parameters to load"
        )
    return parameters


def _validate_exact_coverage(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, nn.Parameter],
) -> dict[str, torch.Tensor]:
    source_by_local = {
        name[len(TARGET_PREFIX) :]: tensor for name, tensor in source.items()
    }
    source_names = set(source_by_local)
    target_names = set(target)
    missing = sorted(target_names - source_names)
    unexpected = sorted(source_names - target_names)
    if missing or unexpected:
        raise FrozenHeatmapCheckpointError(
            "Frozen heatmap parameter coverage is not exact: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    shape_mismatches = sorted(
        name
        for name in target_names
        if tuple(source_by_local[name].shape) != tuple(target[name].shape)
    )
    if shape_mismatches:
        details = {
            name: {
                "checkpoint": tuple(source_by_local[name].shape),
                "model": tuple(target[name].shape),
            }
            for name in shape_mismatches[:8]
        }
        raise FrozenHeatmapCheckpointError(
            f"Frozen heatmap parameter shape mismatch: {details}"
        )
    return source_by_local


def _prepare_copies(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, nn.Parameter],
) -> dict[str, torch.Tensor]:
    """Convert every tensor before mutating any model parameter."""
    converted: dict[str, torch.Tensor] = {}
    try:
        for name in sorted(target):
            value = source[name].detach().to(
                device=target[name].device,
                dtype=target[name].dtype,
            ).contiguous()
            if not torch.isfinite(value).all().item():
                raise FrozenHeatmapCheckpointError(
                    f"Frozen heatmap checkpoint contains non-finite values: {name}"
                )
            converted[name] = value
    except FrozenHeatmapCheckpointError:
        raise
    except Exception as exc:
        raise FrozenHeatmapCheckpointError(
            "Unable to convert frozen heatmap tensors to target device/dtype"
        ) from exc
    return converted


def _copy_and_verify(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, nn.Parameter],
) -> None:
    originals: dict[str, torch.Tensor] = {}
    try:
        originals = {
            name: parameter.detach().clone()
            for name, parameter in target.items()
        }
        with torch.no_grad():
            for name in sorted(target):
                target[name].copy_(source[name])
            for name in sorted(target):
                if not torch.equal(target[name].detach(), source[name]):
                    raise FrozenHeatmapCheckpointError(
                        f"Post-copy tensor verification failed: heatmap_vln.{name}"
                    )
    except Exception as exc:
        if originals:
            with torch.no_grad():
                for name, original in originals.items():
                    target[name].copy_(original)
        if isinstance(exc, FrozenHeatmapCheckpointError):
            raise
        raise FrozenHeatmapCheckpointError(
            "Failed while copying the frozen heatmap checkpoint"
        ) from exc


def load_frozen_heatmap_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    expected_sha256: str,
) -> dict[str, Any]:
    """Load and freeze exactly ``model.heatmap_vln`` from an audited file.

    File integrity is verified before deserialization or model inspection.
    Every checkpoint tensor must live below ``heatmap_vln.`` and the canonical
    key set must exactly equal ``heatmap_vln.named_parameters()``.
    """
    expected_digest = _require_expected_sha256(expected_sha256)
    try:
        path = Path(checkpoint_path).expanduser().resolve(strict=True)
        actual_digest = compute_file_sha256(path)
    except Exception as exc:
        if isinstance(exc, FrozenHeatmapCheckpointError):
            raise
        raise FrozenHeatmapCheckpointError(
            f"Unable to hash frozen heatmap checkpoint: {checkpoint_path}"
        ) from exc
    if actual_digest != expected_digest:
        raise FrozenHeatmapCheckpointError(
            "Frozen heatmap checkpoint SHA-256 mismatch: "
            f"expected={expected_digest}, actual={actual_digest}"
        )

    payload = _load_weights_only(path)
    source = _extract_heatmap_state(payload)
    heatmap = _resolve_heatmap_module(model)
    target = _target_parameters(heatmap)
    source_by_local = _validate_exact_coverage(source, target)
    converted = _prepare_copies(source_by_local, target)
    _copy_and_verify(converted, target)

    heatmap.requires_grad_(False)
    heatmap.eval()
    return {
        "schema_version": DEPENDENCY_SCHEMA,
        "dependency_type": "frozen_heatmap",
        "checkpoint_path": str(path),
        "checkpoint_sha256": actual_digest,
        "state_key": STATE_KEY,
        "target_module": "heatmap_vln",
        "tensor_count": len(target),
        "parameter_names": sorted(target),
        "parameter_shapes": {
            name: [int(dimension) for dimension in target[name].shape]
            for name in sorted(target)
        },
        "frozen": True,
    }


__all__ = [
    "FrozenHeatmapCheckpointError",
    "compute_file_sha256",
    "load_frozen_heatmap_checkpoint",
]

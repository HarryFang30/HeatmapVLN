"""Fail-closed panoramic-head -> single-view-head warm-start contract.

The legacy checkpoint contains Qwen LoRA tensors and four heatmap decoder
families.  The native single-view design must never import the Qwen LoRA or
the LLM feature decoder.  It may reuse only:

* ``heatmap_vln.vit_dpt_fusion.*``;
* ``heatmap_vln.fine.*``;
* name- and shape-compatible ``heatmap_vln.coarse.*`` tensors, except for the
  legacy LLM history projection ``coarse.proj_history.*``.  The one
  ``coarse.proj_traj.weight`` tensor receives a pinned column-sign migration
  so the corrected Habitat ``-Z`` fields reproduce the legacy HeatmapVLN
  wrapper's ``+Z`` trajectory token exactly; every other approved tensor
  remains byte-identical.

The source checkpoint used for this migration has a known 289-tensor layout.
This module intentionally locks that layout instead of guessing when the
source or destination changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .single_view_pose_projection_migration import (
    AUDITED_OUTPUT_SELECTED_STATE_CONTENT_SHA256,
    AUDITED_PROJ_TRAJ_BIAS_SHA256,
    AUDITED_SOURCE_PROJ_TRAJ_WEIGHT_SHA256,
    AUDITED_SOURCE_SELECTED_STATE_CONTENT_SHA256,
    AUDITED_OUTPUT_PROJ_TRAJ_WEIGHT_SHA256,
    TARGET_POSE_CONVENTION,
    TRANSFORM_ALGORITHM_ID,
    migrate_audited_selected_state_proj_traj,
)


ARTIFACT_TYPE = "heatmapvln_single_view_vision_only_warmstart"
SCHEMA_VERSION = 2
ARCHITECTURE_ID = "internnav_single_view_vision_only_four_direction_v2"
DIRECTION_ORDER = ("front", "right", "back", "left")
STATE_KEY = "trainable_state_dict"
WARMSTART_POLICY = "internnav_single_view_head_v2"

VIT_PREFIX = "heatmap_vln.vit_dpt_fusion."
LLM_PREFIX = "heatmap_vln.llm_dpt_fusion."
COARSE_PREFIX = "heatmap_vln.coarse."
FINE_PREFIX = "heatmap_vln.fine."
EXCLUDED_COARSE_PREFIX = "heatmap_vln.coarse.proj_history."

# Audited on
# /mnt/afs/lixiaoou/intern/fjl/model/output/run_20260519_232017/
# checkpoints/latest.pth.  A different source must receive a new policy/version.
EXPECTED_SOURCE_COUNTS = {
    "lora": 224,
    "vit_dpt_fusion": 12,
    "llm_dpt_fusion": 10,
    "coarse": 37,
    "fine": 6,
}
EXPECTED_SELECTED_COUNTS = {
    "vit_dpt_fusion": 12,
    "coarse": 35,
    "fine": 6,
}
EXPECTED_EXCLUDED_PROJ_HISTORY = 2
AUDITED_SOURCE_FILE_SHA256 = (
    "fd607a16aaa2e997e77a4c6ed1263fe1c629c2f0d223ec8805bd79cd170c3593"
)
AUDITED_SOURCE_STATE_CONTENT_SHA256 = (
    "64e63579d378b4b9bf871e35fe43c1161cc8b74228058f13e0694d1a93eea308"
)
# The old selected-state hash is pinned *before* the deterministic pose-input
# migration.  The published artifact hash is the migrated state.  Keeping two
# names prevents a provenance report from accidentally claiming that an
# unmodified legacy ``proj_traj`` consumes the corrected Habitat convention.
AUDITED_PRETRANSFORM_SELECTED_STATE_CONTENT_SHA256 = (
    AUDITED_SOURCE_SELECTED_STATE_CONTENT_SHA256
)
AUDITED_SELECTED_STATE_CONTENT_SHA256 = (
    AUDITED_OUTPUT_SELECTED_STATE_CONTENT_SHA256
)

_FORBIDDEN_ARTIFACT_MARKERS = (
    "lora_",
    "qwen",
    "vlm_backbone",
    "system1",
    "nextdit",
    "adapter",
    "llm_dpt_fusion",
)

SEMANTIC_COMPATIBILITY = {
    "source_input": "four_view_panorama_256px",
    "target_input": "single_front_view_384px",
    "source_restore_vit_spatial_layout": False,
    "target_restore_vit_spatial_layout": True,
    "function_equivalent_after_migration": [
        "heatmap_vln.coarse.proj_traj.weight",
        "heatmap_vln.coarse.proj_traj.bias",
    ],
    "shape_compatible_low_lr_initialization": [
        "heatmap_vln.vit_dpt_fusion.*",
        "heatmap_vln.coarse.* (except proj_history; proj_traj migrated)",
        "heatmap_vln.fine.*",
    ],
    "reset_or_new": [
        "heatmap_vln.coarse.proj_history.*",
        "heatmap_vln.llm_dpt_fusion.*",
        "heatmap_vln.vit_panorama_conditioner.*",
        "heatmap_vln.coarse_panorama_conditioner.*",
    ],
    "warning": (
        "Only proj_traj is function-equivalent. Other selected tensors are "
        "initialization priors across changed visual/layout semantics."
    ),
}


class WarmstartContractError(RuntimeError):
    """Raised whenever provenance or tensor scope is not exactly provable."""


def normalize_state_key(raw_name: str) -> str:
    """Canonicalize only wrapper aliases; never rewrite module semantics."""
    name = str(raw_name)
    while name.startswith("module."):
        name = name[len("module.") :]
    while ".module." in name:
        name = name.replace(".module.", ".")
    return name


def _require_tensor_mapping(value: Any, *, label: str) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise WarmstartContractError(f"{label} must be a non-empty tensor mapping")
    bad = [name for name, tensor in value.items() if not torch.is_tensor(tensor)]
    if bad:
        raise WarmstartContractError(
            f"{label} contains non-tensor values: {sorted(map(str, bad))[:5]}"
        )
    return value


def _normalized_unique_state(
    state: Mapping[str, torch.Tensor],
    *,
    label: str,
    allowed_only: bool = False,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    duplicates: list[str] = []
    for raw_name, tensor in state.items():
        name = normalize_state_key(raw_name)
        if allowed_only and not _is_allowed_selected_key(name):
            continue
        if name in result:
            duplicates.append(name)
        result[name] = tensor
    if duplicates:
        raise WarmstartContractError(
            f"{label} has duplicate canonical keys: {sorted(set(duplicates))[:8]}"
        )
    return result


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    dense = tensor.detach().to(device="cpu").contiguous()
    if dense.layout != torch.strided:
        raise WarmstartContractError(
            f"Only dense strided tensors are supported, got {dense.layout}"
        )
    return dense.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()


def _shape(tensor: torch.Tensor) -> list[int]:
    return [int(dim) for dim in tensor.shape]


def tensor_manifest(state: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": _shape(state[name]),
            "dtype": str(state[name].dtype),
            "numel": int(state[name].numel()),
            "sha256": tensor_sha256(state[name]),
        }
        for name in sorted(state)
    ]


def state_content_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash names, metadata, and bytes independently of ``torch.save``."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        header = json.dumps(
            {
                "name": name,
                "shape": _shape(tensor),
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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_family(name: str) -> str:
    lowered = name.lower()
    if "lora_" in lowered:
        return "lora"
    if name.startswith(VIT_PREFIX):
        return "vit_dpt_fusion"
    if name.startswith(LLM_PREFIX):
        return "llm_dpt_fusion"
    if name.startswith(COARSE_PREFIX):
        return "coarse"
    if name.startswith(FINE_PREFIX):
        return "fine"
    return "forbidden_other"


def _selected_family(name: str) -> str | None:
    if name.startswith(VIT_PREFIX):
        return "vit_dpt_fusion"
    if name.startswith(COARSE_PREFIX) and not name.startswith(EXCLUDED_COARSE_PREFIX):
        return "coarse"
    if name.startswith(FINE_PREFIX):
        return "fine"
    return None


def _is_allowed_selected_key(name: str) -> bool:
    return _selected_family(name) is not None


def _assert_no_forbidden_artifact_tensor_names(names: set[str]) -> None:
    forbidden = sorted(
        name
        for name in names
        if (
            not _is_allowed_selected_key(name)
            or name.startswith(EXCLUDED_COARSE_PREFIX)
            or any(marker in name.lower() for marker in _FORBIDDEN_ARTIFACT_MARKERS)
        )
    )
    if forbidden:
        raise WarmstartContractError(
            "Warm-start artifact contains forbidden LoRA/Qwen/System1/adapter/"
            f"LLM tensors: {forbidden[:8]}"
        )


def audit_source_state(
    source_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    source = _normalized_unique_state(
        _require_tensor_mapping(source_state, label="source state"),
        label="source state",
    )
    counts = {name: 0 for name in (*EXPECTED_SOURCE_COUNTS, "forbidden_other")}
    for name in source:
        counts[_source_family(name)] += 1

    actual_known = {name: counts[name] for name in EXPECTED_SOURCE_COUNTS}
    if actual_known != EXPECTED_SOURCE_COUNTS or counts["forbidden_other"]:
        raise WarmstartContractError(
            "Legacy source tensor-count contract failed; refusing to infer a "
            f"new policy. expected={EXPECTED_SOURCE_COUNTS}, actual={counts}"
        )
    excluded = [name for name in source if name.startswith(EXCLUDED_COARSE_PREFIX)]
    if len(excluded) != EXPECTED_EXCLUDED_PROJ_HISTORY:
        raise WarmstartContractError(
            "Expected exactly two legacy coarse.proj_history tensors, got "
            f"{len(excluded)}: {sorted(excluded)}"
        )
    return source, actual_known


def _target_allowed_state(
    target_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return _normalized_unique_state(
        _require_tensor_mapping(target_state, label="target model state"),
        label="target model state",
        allowed_only=True,
    )


def derive_selected_state(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Select only semantically allowed tensors with exact target shapes."""
    source, _ = audit_source_state(source_state)
    target = _target_allowed_state(target_state)
    selected: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    shape_mismatches: list[str] = []

    for name, tensor in source.items():
        if not _is_allowed_selected_key(name):
            continue
        destination = target.get(name)
        if destination is None:
            missing.append(name)
            continue
        if tuple(destination.shape) != tuple(tensor.shape):
            shape_mismatches.append(
                f"{name}: source={tuple(tensor.shape)} target={tuple(destination.shape)}"
            )
            continue
        selected[name] = tensor.detach().to(device="cpu", copy=True)

    if missing or shape_mismatches:
        raise WarmstartContractError(
            "Target architecture is not exactly compatible with the approved "
            "vision-only warm-start set: "
            f"missing={missing[:8]}, shape_mismatches={shape_mismatches[:8]}"
        )

    counts = {name: 0 for name in EXPECTED_SELECTED_COUNTS}
    for name in selected:
        family = _selected_family(name)
        if family is None:
            raise AssertionError(name)
        counts[family] += 1
    if counts != EXPECTED_SELECTED_COUNTS:
        raise WarmstartContractError(
            "Selected tensor-count contract failed: "
            f"expected={EXPECTED_SELECTED_COUNTS}, actual={counts}"
        )
    _assert_no_forbidden_artifact_tensor_names(set(selected))
    return selected, counts


def build_artifact(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    *,
    source_checkpoint: str,
    source_checkpoint_sha256: str,
    enforce_audited_source: bool = True,
) -> dict[str, Any]:
    source, source_counts = audit_source_state(source_state)
    selected_before_pose_migration, selected_counts = derive_selected_state(
        source,
        target_state,
    )
    source_content_hash = state_content_sha256(source)
    pretransform_selected_content_hash = state_content_sha256(
        selected_before_pose_migration
    )
    if enforce_audited_source:
        mismatches = {}
        if source_checkpoint_sha256 != AUDITED_SOURCE_FILE_SHA256:
            mismatches["source_checkpoint_sha256"] = {
                "expected": AUDITED_SOURCE_FILE_SHA256,
                "actual": source_checkpoint_sha256,
            }
        if source_content_hash != AUDITED_SOURCE_STATE_CONTENT_SHA256:
            mismatches["source_state_content_sha256"] = {
                "expected": AUDITED_SOURCE_STATE_CONTENT_SHA256,
                "actual": source_content_hash,
            }
        if (
            pretransform_selected_content_hash
            != AUDITED_PRETRANSFORM_SELECTED_STATE_CONTENT_SHA256
        ):
            mismatches["pretransform_selected_state_content_sha256"] = {
                "expected": AUDITED_PRETRANSFORM_SELECTED_STATE_CONTENT_SHA256,
                "actual": pretransform_selected_content_hash,
            }
        if mismatches:
            raise WarmstartContractError(
                f"Source is not the audited legacy checkpoint: {mismatches}"
            )

    if not enforce_audited_source:
        raise WarmstartContractError(
            "Schema v2 artifacts require the pinned audited source so the "
            "legacy HeatmapVLN wrapper +Z -> Habitat c2w -Z "
            "proj_traj migration is provable"
        )
    try:
        selected, pose_projection_migration = (
            migrate_audited_selected_state_proj_traj(
                selected_before_pose_migration
            )
        )
    except RuntimeError as exc:
        raise WarmstartContractError(
            "Failed the audited pose-projection migration contract"
        ) from exc
    selected_content_hash = state_content_sha256(selected)
    if selected_content_hash != AUDITED_SELECTED_STATE_CONTENT_SHA256:
        raise WarmstartContractError(
            "Migrated selected-state content hash differs from the pinned "
            f"artifact identity: {selected_content_hash}"
        )
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "architecture_id": ARCHITECTURE_ID,
            "direction_order": list(DIRECTION_ORDER),
            "history_pose_convention": TARGET_POSE_CONVENTION,
            "pose_projection_migration": pose_projection_migration,
            "semantic_compatibility": SEMANTIC_COMPATIBILITY,
            "provenance": {
                "source_checkpoint": str(source_checkpoint),
                "source_checkpoint_sha256": str(source_checkpoint_sha256),
                "source_state_key": STATE_KEY,
                "source_state_tensor_count": len(source),
                "source_state_counts": source_counts,
                "source_state_content_sha256": source_content_hash,
            },
            "selection_contract": {
                "allowed_prefixes": [VIT_PREFIX, COARSE_PREFIX, FINE_PREFIX],
                "excluded_prefixes": [EXCLUDED_COARSE_PREFIX, LLM_PREFIX],
                "forbidden_tensor_classes": [
                    "LoRA",
                    "Qwen/System2",
                    "System1/NextDiT",
                    "panoramic adapters",
                    "legacy LLM DPT fusion",
                ],
                "selected_counts": selected_counts,
                "selected_tensor_count": len(selected),
                "pretransform_selected_state_content_sha256": (
                    pretransform_selected_content_hash
                ),
                "selected_state_content_sha256": selected_content_hash,
                "selected_tensor_manifest": tensor_manifest(selected),
            },
        },
        STATE_KEY: selected,
    }


def _torch_load_weights_only(path: str | Path) -> Any:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError as exc:  # never fall back to pickle-enabled loading
        raise WarmstartContractError(
            "This safety contract requires torch.load(..., weights_only=True)"
        ) from exc


def _state_from_payload(payload: Any, *, key: str, label: str) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise WarmstartContractError(f"{label} payload must be a mapping")
    return _require_tensor_mapping(payload.get(key), label=f"{label}.{key}")


def build_artifact_from_files(
    source_checkpoint: str | Path,
    target_state_file: str | Path,
    *,
    target_state_key: str = "model_state_dict",
) -> dict[str, Any]:
    source_path = Path(source_checkpoint).resolve(strict=True)
    target_path = Path(target_state_file).resolve(strict=True)
    source_payload = _torch_load_weights_only(source_path)
    target_payload = _torch_load_weights_only(target_path)
    source_state = _state_from_payload(
        source_payload,
        key=STATE_KEY,
        label="source checkpoint",
    )
    target_state = _state_from_payload(
        target_payload,
        key=target_state_key,
        label="target state file",
    )
    return build_artifact(
        source_state,
        target_state,
        source_checkpoint=str(source_path),
        source_checkpoint_sha256=file_sha256(source_path),
    )


def _require_exact_mapping_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise WarmstartContractError(
            f"{label} keys differ: missing={sorted(expected-actual)}, "
            f"unexpected={sorted(actual-expected)}"
        )


def validate_artifact(
    payload: Any,
    target_state: Mapping[str, torch.Tensor],
    *,
    enforce_audited_source: bool = True,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise WarmstartContractError("Warm-start artifact must be a mapping")
    _require_exact_mapping_keys(
        payload,
        {"artifact_type", "schema_version", "metadata", STATE_KEY},
        label="artifact",
    )
    if payload["artifact_type"] != ARTIFACT_TYPE:
        raise WarmstartContractError(
            f"Wrong artifact_type: {payload['artifact_type']!r}"
        )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise WarmstartContractError(
            f"Wrong schema_version: {payload['schema_version']!r}"
        )

    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise WarmstartContractError("artifact.metadata must be a mapping")
    _require_exact_mapping_keys(
        metadata,
        {
            "architecture_id",
            "direction_order",
            "history_pose_convention",
            "pose_projection_migration",
            "semantic_compatibility",
            "provenance",
            "selection_contract",
        },
        label="artifact.metadata",
    )
    if metadata["architecture_id"] != ARCHITECTURE_ID:
        raise WarmstartContractError("Warm-start architecture_id mismatch")
    if tuple(metadata["direction_order"]) != DIRECTION_ORDER:
        raise WarmstartContractError("Warm-start direction order mismatch")
    if metadata["history_pose_convention"] != TARGET_POSE_CONVENTION:
        raise WarmstartContractError("Warm-start history pose convention mismatch")
    if metadata["semantic_compatibility"] != SEMANTIC_COMPATIBILITY:
        raise WarmstartContractError(
            "Warm-start semantic-compatibility disclosure mismatch"
        )

    provenance = metadata["provenance"]
    selection = metadata["selection_contract"]
    if not isinstance(provenance, Mapping) or not isinstance(selection, Mapping):
        raise WarmstartContractError("Artifact provenance/selection must be mappings")
    _require_exact_mapping_keys(
        provenance,
        {
            "source_checkpoint",
            "source_checkpoint_sha256",
            "source_state_key",
            "source_state_tensor_count",
            "source_state_counts",
            "source_state_content_sha256",
        },
        label="artifact.metadata.provenance",
    )
    _require_exact_mapping_keys(
        selection,
        {
            "allowed_prefixes",
            "excluded_prefixes",
            "forbidden_tensor_classes",
            "selected_counts",
            "selected_tensor_count",
            "pretransform_selected_state_content_sha256",
            "selected_state_content_sha256",
            "selected_tensor_manifest",
        },
        label="artifact.metadata.selection_contract",
    )
    if provenance.get("source_state_key") != STATE_KEY:
        raise WarmstartContractError("Artifact source state key mismatch")
    if provenance.get("source_state_tensor_count") != sum(EXPECTED_SOURCE_COUNTS.values()):
        raise WarmstartContractError("Artifact source tensor count mismatch")
    if provenance.get("source_state_counts") != EXPECTED_SOURCE_COUNTS:
        raise WarmstartContractError("Artifact source family counts mismatch")
    for field in ("source_checkpoint_sha256", "source_state_content_sha256"):
        digest = provenance.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise WarmstartContractError(f"Artifact has invalid {field}")
    if enforce_audited_source:
        if provenance["source_checkpoint_sha256"] != AUDITED_SOURCE_FILE_SHA256:
            raise WarmstartContractError("Artifact source checkpoint SHA-256 mismatch")
        if (
            provenance["source_state_content_sha256"]
            != AUDITED_SOURCE_STATE_CONTENT_SHA256
        ):
            raise WarmstartContractError("Artifact source-state content hash mismatch")

    if selection.get("allowed_prefixes") != [VIT_PREFIX, COARSE_PREFIX, FINE_PREFIX]:
        raise WarmstartContractError("Artifact allowed-prefix policy mismatch")
    if selection.get("excluded_prefixes") != [EXCLUDED_COARSE_PREFIX, LLM_PREFIX]:
        raise WarmstartContractError("Artifact excluded-prefix policy mismatch")
    if selection.get("forbidden_tensor_classes") != [
        "LoRA",
        "Qwen/System2",
        "System1/NextDiT",
        "panoramic adapters",
        "legacy LLM DPT fusion",
    ]:
        raise WarmstartContractError("Artifact forbidden-class policy mismatch")

    state = _normalized_unique_state(
        _state_from_payload(payload, key=STATE_KEY, label="artifact"),
        label="artifact state",
    )
    if any(raw_name != normalize_state_key(raw_name) for raw_name in payload[STATE_KEY]):
        raise WarmstartContractError("Artifact state keys must already be canonical")
    _assert_no_forbidden_artifact_tensor_names(set(state))
    counts = {name: 0 for name in EXPECTED_SELECTED_COUNTS}
    for name in state:
        family = _selected_family(name)
        if family is None:
            raise WarmstartContractError(f"Forbidden artifact tensor: {name}")
        counts[family] += 1
    if counts != EXPECTED_SELECTED_COUNTS:
        raise WarmstartContractError(
            f"Artifact selected counts mismatch: {counts}"
        )
    if selection.get("selected_counts") != counts:
        raise WarmstartContractError("Artifact metadata selected counts mismatch")
    if selection.get("selected_tensor_count") != len(state):
        raise WarmstartContractError("Artifact metadata tensor count mismatch")
    if (
        selection.get("pretransform_selected_state_content_sha256")
        != AUDITED_PRETRANSFORM_SELECTED_STATE_CONTENT_SHA256
    ):
        raise WarmstartContractError(
            "Artifact pre-transform selected-state hash mismatch"
        )
    computed_hash = state_content_sha256(state)
    if selection.get("selected_state_content_sha256") != computed_hash:
        raise WarmstartContractError("Artifact selected-state content hash mismatch")
    if enforce_audited_source and computed_hash != AUDITED_SELECTED_STATE_CONTENT_SHA256:
        raise WarmstartContractError("Artifact audited selected-state hash mismatch")
    if selection.get("selected_tensor_manifest") != tensor_manifest(state):
        raise WarmstartContractError("Artifact selected tensor manifest mismatch")

    migration = metadata["pose_projection_migration"]
    if not isinstance(migration, Mapping):
        raise WarmstartContractError("Artifact pose migration must be a mapping")
    audited_migration = migration.get("audited_checkpoint_contract")
    if not isinstance(audited_migration, Mapping):
        raise WarmstartContractError(
            "Artifact lacks the audited pose-projection migration contract"
        )
    expected_migration = {
        "verified": True,
        "source_selected_state_content_sha256": (
            AUDITED_PRETRANSFORM_SELECTED_STATE_CONTENT_SHA256
        ),
        "output_selected_state_content_sha256": (
            AUDITED_SELECTED_STATE_CONTENT_SHA256
        ),
        "source_proj_traj_weight_sha256": (
            AUDITED_SOURCE_PROJ_TRAJ_WEIGHT_SHA256
        ),
        "output_proj_traj_weight_sha256": (
            AUDITED_OUTPUT_PROJ_TRAJ_WEIGHT_SHA256
        ),
        "proj_traj_bias_sha256": AUDITED_PROJ_TRAJ_BIAS_SHA256,
    }
    if dict(audited_migration) != expected_migration:
        raise WarmstartContractError(
            "Artifact audited pose-projection identities mismatch"
        )
    if migration.get("algorithm_id") != TRANSFORM_ALGORITHM_ID:
        raise WarmstartContractError("Artifact pose migration algorithm mismatch")
    if migration.get("target_pose_convention") != TARGET_POSE_CONVENTION:
        raise WarmstartContractError("Artifact pose migration target mismatch")
    if migration.get("source_selected_state_content_sha256") != (
        AUDITED_PRETRANSFORM_SELECTED_STATE_CONTENT_SHA256
    ):
        raise WarmstartContractError("Artifact pose migration source hash mismatch")
    if migration.get("output_selected_state_content_sha256") != computed_hash:
        raise WarmstartContractError("Artifact pose migration output hash mismatch")
    if tensor_sha256(state["heatmap_vln.coarse.proj_traj.weight"]) != (
        AUDITED_OUTPUT_PROJ_TRAJ_WEIGHT_SHA256
    ):
        raise WarmstartContractError("Artifact migrated proj_traj weight mismatch")
    if tensor_sha256(state["heatmap_vln.coarse.proj_traj.bias"]) != (
        AUDITED_PROJ_TRAJ_BIAS_SHA256
    ):
        raise WarmstartContractError("Artifact proj_traj bias mismatch")

    target = _target_allowed_state(target_state)
    missing = sorted(set(state) - set(target))
    mismatched = sorted(
        name
        for name in state.keys() & target.keys()
        if tuple(state[name].shape) != tuple(target[name].shape)
    )
    if missing or mismatched:
        raise WarmstartContractError(
            "Artifact is incompatible with target model: "
            f"missing={missing[:8]}, shape_mismatch={mismatched[:8]}"
        )
    return {
        "loaded_tensor_count": len(state),
        "selected_counts": counts,
        "selected_state_content_sha256": computed_hash,
        "source_checkpoint_sha256": provenance["source_checkpoint_sha256"],
        "semantic_compatibility": SEMANTIC_COMPATIBILITY,
    }


def assert_model_has_no_lora_or_pano_adapter(model: nn.Module) -> None:
    module = getattr(model, "module", model)
    lora_parameters = [
        name for name, _ in module.named_parameters() if "lora_" in name.lower()
    ]
    lora_state = [name for name in module.state_dict() if "lora_" in name.lower()]
    lora_modules = [
        name
        for name, child in module.named_modules()
        if "lora" in name.lower() or "lora" in type(child).__name__.lower()
    ]
    adapter_modules = [
        name
        for name, child in module.named_modules()
        if (
            "pano_latent_adapter" in name.lower()
            or "panotonextditadapter" in type(child).__name__.lower()
            or "panoramicadapter" in type(child).__name__.lower()
        )
    ]
    adapter_attr = getattr(module, "pano_latent_adapter", None)
    if lora_parameters or lora_state or lora_modules:
        raise WarmstartContractError(
            "Model contains LoRA; native InternNav must be constructed without "
            f"PEFT. params={lora_parameters[:5]} state={lora_state[:5]} "
            f"modules={lora_modules[:5]}"
        )
    if adapter_attr is not None or adapter_modules:
        raise WarmstartContractError(
            "Model contains a forbidden panoramic adapter: "
            f"attribute={type(adapter_attr).__name__ if adapter_attr is not None else None}, "
            f"modules={adapter_modules[:5]}"
        )


def load_artifact_into_model(
    model: nn.Module,
    artifact_path: str | Path,
) -> dict[str, Any]:
    """Load exactly the approved 53 tensors; never call a generic loader."""
    assert_model_has_no_lora_or_pano_adapter(model)
    payload = _torch_load_weights_only(artifact_path)
    module = getattr(model, "module", model)
    target_state_raw = module.state_dict()
    report = validate_artifact(payload, target_state_raw)
    source_state = _normalized_unique_state(
        payload[STATE_KEY],
        label="artifact state",
    )

    actual_by_canonical: dict[str, torch.Tensor] = {}
    for raw_name, destination in target_state_raw.items():
        name = normalize_state_key(raw_name)
        if name not in source_state:
            continue
        if name in actual_by_canonical:
            raise WarmstartContractError(
                f"Target model aliases approved tensor more than once: {name}"
            )
        actual_by_canonical[name] = destination

    if set(actual_by_canonical) != set(source_state):
        raise WarmstartContractError("Approved destination map changed after validation")
    with torch.no_grad():
        for name in sorted(source_state):
            destination = actual_by_canonical[name]
            source = source_state[name].to(
                device=destination.device,
                dtype=destination.dtype,
            )
            destination.copy_(source)
            if not torch.equal(destination.detach(), source):
                raise WarmstartContractError(f"Post-load verification failed for {name}")
    return report


def save_artifact_exclusive(payload: Mapping[str, Any], output: str | Path) -> Path:
    """Atomically publish without ever overwriting an existing artifact."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output_path}")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
            # Cluster jobs may run under a different container UID than the
            # development shell. Match the repository's existing checkpoint
            # convention while keeping publication atomic and non-overwriting.
            os.fchmod(handle.fileno(), 0o644)
        # hard-link publication has O_EXCL semantics and cannot clobber a race.
        os.link(temporary_name, output_path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return output_path

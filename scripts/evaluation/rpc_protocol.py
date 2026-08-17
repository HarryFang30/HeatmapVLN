"""Shared protocol identifiers and deterministic RPC sampling helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

HEATMAPVLN_RPC_PROTOCOL_VERSION = "heatmapvln-r2r-json-v3"
HEATMAPVLN_RPC_CAPABILITY_PANO_TWO_PHASE_FRONT_SYSTEM1 = (
    "pano-two-phase-front-system1-v1"
)
HEATMAPVLN_RPC_SAMPLING_PROTOCOL = "heatmapvln-nextdit-sha256-v1"
HEATMAPVLN_RPC_DEFAULT_PROTOCOL_SEED = 42
HEATMAPVLN_RPC_SAMPLING_FIELD = "deterministic_sampling"

_MAX_TORCH_SEED = (1 << 63) - 1
_SAMPLING_FIELDS = (
    "sampling_protocol",
    "protocol_seed",
    "scene_id",
    "episode_id",
    "system2_call_index",
    "per_call_seed",
    "seed_sha256",
)


def build_rpc_progress_sampling_contract(
    *,
    protocol_seed: int,
    require_deterministic_sampling: bool,
) -> dict[str, Any]:
    """Return fields that every resumable RPC episode result must match."""
    _sampling_key(
        protocol_seed=protocol_seed,
        scene_id="contract-validation",
        episode_id=0,
        system2_call_index=0,
    )
    if not isinstance(require_deterministic_sampling, bool):
        raise ValueError("require_deterministic_sampling must be a boolean")
    return {
        "rpc_protocol": HEATMAPVLN_RPC_PROTOCOL_VERSION,
        "rpc_sampling_protocol": HEATMAPVLN_RPC_SAMPLING_PROTOCOL,
        "rpc_deterministic_sampling_enabled": True,
        "rpc_protocol_seed": protocol_seed,
        "rpc_require_deterministic_sampling": require_deterministic_sampling,
    }


def validate_rpc_progress_sampling_contract(
    result: Any,
    *,
    expected: Mapping[str, Any],
) -> None:
    """Refuse progress rows produced by another RPC sampling contract."""
    if not isinstance(result, Mapping):
        raise ValueError("RPC progress row must be an object")
    mismatches = {
        field: {"expected": value, "actual": result.get(field)}
        for field, value in expected.items()
        if result.get(field) != value
    }
    if mismatches:
        raise ValueError(
            "RPC progress sampling contract mismatch: " + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def _sampling_key(
    *,
    protocol_seed: int,
    scene_id: str,
    episode_id: int,
    system2_call_index: int,
) -> dict[str, Any]:
    if isinstance(protocol_seed, bool) or not isinstance(protocol_seed, int):
        raise ValueError("protocol_seed must be an integer")
    if not 0 <= protocol_seed <= _MAX_TORCH_SEED:
        raise ValueError(f"protocol_seed must be in [0, {_MAX_TORCH_SEED}]")
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ValueError("scene_id must be a non-empty string")
    if isinstance(episode_id, bool) or not isinstance(episode_id, int):
        raise ValueError("episode_id must be an integer")
    if isinstance(system2_call_index, bool) or not isinstance(system2_call_index, int):
        raise ValueError("system2_call_index must be an integer")
    if system2_call_index < 0:
        raise ValueError("system2_call_index must be >= 0")
    return {
        "sampling_protocol": HEATMAPVLN_RPC_SAMPLING_PROTOCOL,
        "protocol_seed": protocol_seed,
        "scene_id": scene_id.strip(),
        "episode_id": episode_id,
        "system2_call_index": system2_call_index,
    }


def derive_rpc_per_call_seed(
    *,
    protocol_seed: int,
    scene_id: str,
    episode_id: int,
    system2_call_index: int,
) -> tuple[int, str]:
    """Derive a stable torch seed from an arm-independent RPC call key."""
    key = _sampling_key(
        protocol_seed=protocol_seed,
        scene_id=scene_id,
        episode_id=episode_id,
        system2_call_index=system2_call_index,
    )
    canonical = json.dumps(
        key,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    # Keep the seed in torch's portable non-negative signed int64 range.
    per_call_seed = int(digest[:16], 16) & _MAX_TORCH_SEED
    return per_call_seed, digest


def build_rpc_sampling_metadata(
    *,
    protocol_seed: int,
    scene_id: str,
    episode_id: int,
    system2_call_index: int,
) -> dict[str, Any]:
    """Build the complete request/response audit record for one RPC call."""
    key = _sampling_key(
        protocol_seed=protocol_seed,
        scene_id=scene_id,
        episode_id=episode_id,
        system2_call_index=system2_call_index,
    )
    per_call_seed, digest = derive_rpc_per_call_seed(
        protocol_seed=protocol_seed,
        scene_id=scene_id,
        episode_id=episode_id,
        system2_call_index=system2_call_index,
    )
    return {
        **key,
        "per_call_seed": per_call_seed,
        "seed_sha256": digest,
    }


def validate_rpc_sampling_metadata(
    metadata: Any,
    *,
    require_deterministic: bool,
) -> dict[str, Any] | None:
    """Validate client metadata and rederive its seed on the server.

    Missing metadata remains accepted for legacy clients unless deterministic
    sampling is explicitly required. Once a client supplies any metadata, it
    is always validated fail-closed rather than silently falling back.
    """
    if metadata is None:
        if require_deterministic:
            raise ValueError("deterministic sampling metadata is required")
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError("deterministic sampling metadata must be an object")
    missing = [field for field in _SAMPLING_FIELDS if field not in metadata]
    if missing:
        raise ValueError("deterministic sampling metadata is incomplete; missing " + ", ".join(missing))
    unexpected = sorted(set(metadata) - set(_SAMPLING_FIELDS))
    if unexpected:
        raise ValueError("deterministic sampling metadata has unexpected fields: " + ", ".join(unexpected))
    if metadata["sampling_protocol"] != HEATMAPVLN_RPC_SAMPLING_PROTOCOL:
        raise ValueError(
            "deterministic sampling protocol mismatch: "
            f"got {metadata['sampling_protocol']!r}, "
            f"expected {HEATMAPVLN_RPC_SAMPLING_PROTOCOL!r}"
        )
    expected = build_rpc_sampling_metadata(
        protocol_seed=metadata["protocol_seed"],
        scene_id=metadata["scene_id"],
        episode_id=metadata["episode_id"],
        system2_call_index=metadata["system2_call_index"],
    )
    if dict(metadata) != expected:
        mismatches = [field for field in _SAMPLING_FIELDS if metadata.get(field) != expected[field]]
        raise ValueError(
            "deterministic sampling metadata failed SHA256 rederivation; mismatched " + ", ".join(mismatches)
        )
    return expected

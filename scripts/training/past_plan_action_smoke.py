"""Strict runtime audit for the four-rank Past->Plan->Action smoke.

The SHA-256 values below are ephemeral equality checks across ranks.  They
are not checkpoint pins and are never persisted as an input requirement.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn


SMOKE_AUDIT_ENV = "PPA_4GPU_SMOKE_AUDIT"
SMOKE_WORLD_SIZE_ENV = "PPA_SMOKE_WORLD_SIZE"
EXPECTED_WORLD_SIZE = 4
EXPECTED_BATCH_PER_RANK = 1
EXPECTED_PROVIDER = "amb3r_vo_cache"
EXPECTED_TRAINABLE_TENSORS = 64
GRADIENT_FAMILIES = {
    "future": "past_plan_action.future_head.",
    "bridge": "past_plan_action.bridge.",
    "shared_past": "heatmap_vln.",
}
EXPECTED_GRADIENT_FAMILY_TENSORS = {
    "future": 11,
    "bridge": 10,
    "shared_past": 43,
}


def expected_smoke_world_size() -> int:
    raw = str(os.environ.get(SMOKE_WORLD_SIZE_ENV, EXPECTED_WORLD_SIZE)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"{SMOKE_WORLD_SIZE_ENV} must be an integer, got {raw!r}"
        ) from exc
    if value < 1:
        raise RuntimeError(f"{SMOKE_WORLD_SIZE_ENV} must be positive, got {value}")
    return value


def smoke_audit_enabled(stage_cfg: Mapping[str, Any]) -> bool:
    raw = str(os.environ.get(SMOKE_AUDIT_ENV, "0")).strip().lower()
    enabled = raw in {"1", "true", "yes", "on"}
    if not enabled:
        return False
    if str(stage_cfg.get("past_plan_action_stage", "")) != "stage2_joint":
        raise RuntimeError(
            f"{SMOKE_AUDIT_ENV}=1 is valid only for PPA stage2_joint"
        )
    if list(stage_cfg.get("trainable_modules", ())) != [
        "past_plan_action",
        "heatmap_vln",
    ]:
        raise RuntimeError(
            f"{SMOKE_AUDIT_ENV}=1 requires the exact PPA trainable scope"
        )
    return True


def _selected_trainable_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    selected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if len(selected) != EXPECTED_TRAINABLE_TENSORS:
        raise RuntimeError(
            "PPA distributed smoke expected exactly "
            f"{EXPECTED_TRAINABLE_TENSORS} trainable tensors, found {len(selected)}"
        )
    family_counts = {
        family: sum(name.startswith(prefix) for name in selected)
        for family, prefix in GRADIENT_FAMILIES.items()
    }
    if family_counts != EXPECTED_GRADIENT_FAMILY_TENSORS:
        raise RuntimeError(
            "PPA distributed smoke trainable-family contract mismatch: "
            f"expected={EXPECTED_GRADIENT_FAMILY_TENSORS} actual={family_counts}"
        )
    covered = {
        name
        for name in selected
        if any(name.startswith(prefix) for prefix in GRADIENT_FAMILIES.values())
    }
    if covered != set(selected):
        raise RuntimeError(
            "PPA distributed smoke found trainable tensors outside the three "
            f"approved families: {sorted(set(selected) - covered)[:8]}"
        )
    return selected


def install_gradient_hooks(
    model: nn.Module,
) -> tuple[dict[str, dict[str, Any]], list[Any]]:
    """Observe every local pre-all-reduce gradient without changing it."""

    records: dict[str, dict[str, Any]] = {}
    handles = []
    for name, parameter in _selected_trainable_parameters(model).items():
        def record(gradient: torch.Tensor, *, parameter_name: str = name):
            value = gradient.detach().float()
            records[parameter_name] = {
                "seen": True,
                "finite": bool(torch.isfinite(value).all().item()),
                "nonzero": bool(torch.count_nonzero(value).item()),
                "l2": float(value.norm().item()),
                "max_abs": float(value.abs().max().item()),
            }
            return gradient

        handles.append(parameter.register_hook(record))
    return records, handles


def tensor_mapping_digest(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        # Reshape first so scalar parameters are represented by their storage
        # bytes too; ``view(uint8)`` rejects a zero-dimensional tensor.
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def build_local_rank_audit(
    *,
    model: nn.Module,
    ema: Any,
    gradient_records: Mapping[str, Mapping[str, Any]],
    identities: Sequence[str],
    providers: Sequence[str],
    optimizer_steps: int,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    selected = _selected_trainable_parameters(model)
    if ema is None:
        raise RuntimeError("PPA distributed smoke requires EMA")
    ema_state = ema.state_dict()
    shadow = ema_state.get("shadow") or {}
    if set(shadow) != set(selected):
        raise RuntimeError(
            "PPA EMA does not exactly track the 64 trainable tensors: "
            f"missing={sorted(set(selected) - set(shadow))[:8]} "
            f"extra={sorted(set(shadow) - set(selected))[:8]}"
        )
    return {
        "rank": int(rank),
        "world_size": int(world_size),
        "identities": [str(value) for value in identities],
        "providers": [str(value) for value in providers],
        "optimizer_steps": int(optimizer_steps),
        "gradient_records": {
            str(name): dict(record)
            for name, record in gradient_records.items()
        },
        "post_parameter_digest": tensor_mapping_digest(selected),
        "ema_digest": tensor_mapping_digest(shadow),
        "ema_step_count": int(ema_state.get("step_count", -1)),
    }


def validate_rank_audits(
    rank_audits: Sequence[Mapping[str, Any]],
    *,
    expected_world_size: int = EXPECTED_WORLD_SIZE,
    expected_batch_per_rank: int = EXPECTED_BATCH_PER_RANK,
) -> dict[str, Any]:
    if len(rank_audits) != expected_world_size:
        raise RuntimeError(
            f"Expected {expected_world_size} PPA rank audits, got {len(rank_audits)}"
        )
    ranks = [int(report.get("rank", -1)) for report in rank_audits]
    if sorted(ranks) != list(range(expected_world_size)):
        raise RuntimeError(f"PPA rank audit coverage mismatch: {ranks}")
    if any(
        int(report.get("world_size", -1)) != expected_world_size
        for report in rank_audits
    ):
        raise RuntimeError("PPA rank audit world_size mismatch")

    identities: list[str] = []
    providers: list[str] = []
    model_digests: list[str] = []
    ema_digests: list[str] = []
    family_nonzero_by_rank: dict[str, list[int]] = {
        family: [] for family in GRADIENT_FAMILIES
    }
    expected_names: set[str] | None = None
    for report in rank_audits:
        rank = int(report["rank"])
        local_identities = [str(value) for value in report.get("identities", [])]
        local_providers = [str(value) for value in report.get("providers", [])]
        if len(local_identities) != expected_batch_per_rank:
            raise RuntimeError(
                f"Rank {rank} expected {expected_batch_per_rank} identities, "
                f"got {local_identities}"
            )
        if len(local_providers) != expected_batch_per_rank:
            raise RuntimeError(f"Rank {rank} provider count mismatch")
        if any(provider != EXPECTED_PROVIDER for provider in local_providers):
            raise RuntimeError(
                f"Rank {rank} used non-AMB3R provider: {local_providers}"
            )
        identities.extend(local_identities)
        providers.extend(local_providers)
        if int(report.get("optimizer_steps", -1)) != 1:
            raise RuntimeError(f"Rank {rank} did not complete one optimizer step")
        if int(report.get("ema_step_count", -1)) != 1:
            raise RuntimeError(f"Rank {rank} EMA step_count is not one")

        records = dict(report.get("gradient_records") or {})
        if len(records) != EXPECTED_TRAINABLE_TENSORS:
            raise RuntimeError(
                f"Rank {rank} saw {len(records)}/{EXPECTED_TRAINABLE_TENSORS} "
                "PPA gradient hooks"
            )
        names = set(records)
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise RuntimeError("PPA gradient-hook names differ across ranks")
        invalid = [
            name
            for name, record in records.items()
            if record.get("seen") is not True or record.get("finite") is not True
        ]
        if invalid:
            raise RuntimeError(
                f"Rank {rank} has missing/non-finite PPA gradients: {invalid[:8]}"
            )
        for family, prefix in GRADIENT_FAMILIES.items():
            family_records = [
                record for name, record in records.items() if name.startswith(prefix)
            ]
            expected_count = EXPECTED_GRADIENT_FAMILY_TENSORS[family]
            if len(family_records) != expected_count:
                raise RuntimeError(
                    f"Rank {rank} PPA gradient family {family} has "
                    f"{len(family_records)}/{expected_count} tensors"
                )
            if not any(record.get("nonzero") is True for record in family_records):
                raise RuntimeError(
                    f"Rank {rank} has no non-zero gradient in PPA family {family}"
                )
            family_nonzero_by_rank[family].append(rank)

        model_digests.append(str(report.get("post_parameter_digest", "")))
        ema_digests.append(str(report.get("ema_digest", "")))

    expected_global = expected_world_size * expected_batch_per_rank
    if len(identities) != expected_global or len(set(identities)) != expected_global:
        duplicates = sorted(
            identity
            for identity in set(identities)
            if identities.count(identity) > 1
        )
        raise RuntimeError(
            f"Expected {expected_global} unique PPA identities; "
            f"duplicates={duplicates[:8]}"
        )
    if len(set(model_digests)) != 1 or not model_digests[0]:
        raise RuntimeError(
            f"Post-step PPA parameters diverged across ranks: {model_digests}"
        )
    if len(set(ema_digests)) != 1 or not ema_digests[0]:
        raise RuntimeError(f"Post-step PPA EMA diverged across ranks: {ema_digests}")

    return {
        "status": "passed",
        "world_size": expected_world_size,
        "batch_per_rank": expected_batch_per_rank,
        "global_identity_count": len(identities),
        "global_unique_identity_count": len(set(identities)),
        "providers": sorted(set(providers)),
        "optimizer_steps_by_rank": [
            int(report["optimizer_steps"]) for report in rank_audits
        ],
        "gradient_hook_tensors_by_rank": [
            len(report["gradient_records"]) for report in rank_audits
        ],
        "gradient_families_nonzero_on_ranks": family_nonzero_by_rank,
        "gradient_family_tensor_counts": dict(EXPECTED_GRADIENT_FAMILY_TENSORS),
        "post_parameter_digest": model_digests[0],
        "post_parameter_digest_unique_count": len(set(model_digests)),
        "ema_digest": ema_digests[0],
        "ema_digest_unique_count": len(set(ema_digests)),
        "ema_steps_by_rank": [
            int(report["ema_step_count"]) for report in rank_audits
        ],
        "checkpoint_hash_locking": False,
    }


def gather_and_validate_local_audit(
    local_audit: Mapping[str, Any] | None,
    *,
    local_error: str | None = None,
) -> dict[str, Any]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("PPA distributed smoke requires initialized torch.distributed")
    world_size = dist.get_world_size()
    expected_world_size = expected_smoke_world_size()
    if world_size != expected_world_size:
        raise RuntimeError(
            "PPA distributed smoke world-size mismatch before collective: "
            f"configured={expected_world_size} actual={world_size}"
        )
    payload = {
        "audit": dict(local_audit) if local_audit is not None else None,
        "error": str(local_error) if local_error else None,
    }
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, payload)
    errors: dict[int, str] = {}
    for rank, item in enumerate(gathered):
        if not isinstance(item, Mapping):
            errors[rank] = f"malformed payload: {type(item).__name__}"
        elif item.get("error"):
            errors[rank] = str(item["error"])
    if errors:
        raise RuntimeError(f"PPA distributed smoke local audit failed: {errors}")
    audits = [item.get("audit") for item in gathered]
    if any(not isinstance(audit, Mapping) for audit in audits):
        raise RuntimeError("PPA distributed smoke gathered malformed rank audit")
    return validate_rank_audits(
        audits,
        expected_world_size=expected_world_size,
    )

"""Strict runtime audit for the distributed pose-adaptation smoke.

The SHA-256 values produced here are ephemeral cross-rank equality checks.
They are never used to pin an input checkpoint or reject a later best.pth.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from .pose_adaptation import (
    EXPECTED_POSE_ADAPTATION_TENSORS,
    POSE_ADAPTATION_PREFIXES,
    is_pose_adaptation_stage,
)


SMOKE_AUDIT_ENV = "POSE_ADAPT_8GPU_SMOKE_AUDIT"
SMOKE_WORLD_SIZE_ENV = "POSE_ADAPT_SMOKE_WORLD_SIZE"
EXPECTED_WORLD_SIZE = 8
EXPECTED_BATCH_PER_RANK = 2
EXPECTED_GLOBAL_IDENTITIES = EXPECTED_WORLD_SIZE * EXPECTED_BATCH_PER_RANK
EXPECTED_PROVIDER = "amb3r_vo_cache"
GRADIENT_FAMILIES = {
    "proj_traj": "heatmap_vln.coarse.proj_traj.",
    "transformer": "heatmap_vln.coarse.self_attn.",
    "visibility": "heatmap_vln.coarse.vis_head.",
    "coarse_heatmap": "heatmap_vln.coarse.heatmap_head.",
}
EXPECTED_GRADIENT_FAMILY_TENSORS = {
    "proj_traj": 2,
    "transformer": 24,
    "visibility": 4,
    "coarse_heatmap": 4,
}


def expected_smoke_world_size() -> int:
    """Return the launcher-declared smoke world size (eight by default)."""

    raw = str(os.environ.get(SMOKE_WORLD_SIZE_ENV, EXPECTED_WORLD_SIZE)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{SMOKE_WORLD_SIZE_ENV} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise RuntimeError(f"{SMOKE_WORLD_SIZE_ENV} must be positive, got {value}")
    return value


def smoke_audit_enabled(stage_cfg: Mapping[str, Any]) -> bool:
    raw = str(os.environ.get(SMOKE_AUDIT_ENV, "0")).strip().lower()
    enabled = raw in {"1", "true", "yes", "on"}
    if enabled and not is_pose_adaptation_stage(stage_cfg):
        raise RuntimeError(
            f"{SMOKE_AUDIT_ENV}=1 is valid only for the exact pose-adaptation stage"
        )
    return enabled


def _selected_trainable_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    selected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if len(selected) != EXPECTED_POSE_ADAPTATION_TENSORS:
        raise RuntimeError(
            "Distributed smoke expected exactly 34 trainable tensors, "
            f"found {len(selected)}"
        )
    invalid = sorted(
        name for name in selected if not name.startswith(POSE_ADAPTATION_PREFIXES)
    )
    if invalid:
        raise RuntimeError(
            "Distributed smoke found trainable tensors outside four prefixes: "
            f"{invalid[:8]}"
        )
    family_counts = {
        family: sum(name.startswith(prefix) for name in selected)
        for family, prefix in GRADIENT_FAMILIES.items()
    }
    if family_counts != EXPECTED_GRADIENT_FAMILY_TENSORS:
        raise RuntimeError(
            "Distributed smoke trainable-family tensor contract mismatch: "
            f"expected={EXPECTED_GRADIENT_FAMILY_TENSORS} actual={family_counts}"
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
    """Content digest used only to prove post-step equality across ranks."""

    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        # NumPy cannot materialize every torch dtype (notably bfloat16).
        # Hash the exact contiguous storage bytes after recording dtype/shape.
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
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
    model_digest = tensor_mapping_digest(selected)
    if ema is None:
        raise RuntimeError("Distributed pose-adaptation smoke requires EMA")
    ema_state = ema.state_dict()
    shadow = ema_state.get("shadow") or {}
    if set(shadow) != set(selected):
        raise RuntimeError(
            "EMA does not exactly track the 34 trainable tensors: "
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
        "post_parameter_digest": model_digest,
        "ema_digest": tensor_mapping_digest(shadow),
        "ema_step_count": int(ema_state.get("step_count", -1)),
    }


def validate_rank_audits(
    rank_audits: Sequence[Mapping[str, Any]],
    *,
    expected_world_size: int = EXPECTED_WORLD_SIZE,
    expected_batch_per_rank: int = EXPECTED_BATCH_PER_RANK,
) -> dict[str, Any]:
    """Fail closed on the complete distributed causal training contract."""

    if len(rank_audits) != expected_world_size:
        raise RuntimeError(
            f"Expected {expected_world_size} rank audits, got {len(rank_audits)}"
        )
    ranks = [int(report.get("rank", -1)) for report in rank_audits]
    if sorted(ranks) != list(range(expected_world_size)):
        raise RuntimeError(f"Rank audit coverage mismatch: {ranks}")
    if any(int(report.get("world_size", -1)) != expected_world_size for report in rank_audits):
        raise RuntimeError("Rank audit world_size mismatch")

    identities: list[str] = []
    providers: list[str] = []
    model_digests: list[str] = []
    ema_digests: list[str] = []
    family_nonzero_by_rank: dict[str, list[int]] = {
        family: [] for family in GRADIENT_FAMILIES
    }
    expected_names: set[str] | None = None
    for report in rank_audits:
        local_identities = [str(value) for value in report.get("identities", [])]
        local_providers = [str(value) for value in report.get("providers", [])]
        if len(local_identities) != expected_batch_per_rank:
            raise RuntimeError(
                f"Rank {report['rank']} expected {expected_batch_per_rank} identities, "
                f"got {local_identities}"
            )
        if len(local_providers) != expected_batch_per_rank:
            raise RuntimeError(f"Rank {report['rank']} provider count mismatch")
        if any(provider != EXPECTED_PROVIDER for provider in local_providers):
            raise RuntimeError(
                f"Rank {report['rank']} used non-AMB3R provider: {local_providers}"
            )
        identities.extend(local_identities)
        providers.extend(local_providers)
        if int(report.get("optimizer_steps", -1)) != 1:
            raise RuntimeError(
                f"Rank {report['rank']} did not complete exactly one optimizer step"
            )
        if int(report.get("ema_step_count", -1)) != 1:
            raise RuntimeError(f"Rank {report['rank']} EMA step_count is not one")

        records = dict(report.get("gradient_records") or {})
        if len(records) != EXPECTED_POSE_ADAPTATION_TENSORS:
            raise RuntimeError(
                f"Rank {report['rank']} saw {len(records)}/34 gradient hooks"
            )
        names = set(records)
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise RuntimeError("Gradient-hook parameter names differ across ranks")
        invalid = [
            name
            for name, record in records.items()
            if record.get("seen") is not True or record.get("finite") is not True
        ]
        if invalid:
            raise RuntimeError(
                f"Rank {report['rank']} has missing/non-finite gradients: {invalid[:8]}"
            )
        for family, prefix in GRADIENT_FAMILIES.items():
            family_records = [record for name, record in records.items() if name.startswith(prefix)]
            expected_family_count = EXPECTED_GRADIENT_FAMILY_TENSORS[family]
            if len(family_records) != expected_family_count:
                raise RuntimeError(
                    f"Rank {report['rank']} gradient family {family} has "
                    f"{len(family_records)}/{expected_family_count} tensors"
                )
            if not any(record.get("nonzero") is True for record in family_records):
                raise RuntimeError(
                    f"Rank {report['rank']} has no non-zero gradient in family {family}"
                )
            family_nonzero_by_rank[family].append(int(report["rank"]))

        model_digests.append(str(report.get("post_parameter_digest", "")))
        ema_digests.append(str(report.get("ema_digest", "")))

    expected_global = expected_world_size * expected_batch_per_rank
    if len(identities) != expected_global or len(set(identities)) != expected_global:
        duplicates = sorted(
            identity for identity in set(identities) if identities.count(identity) > 1
        )
        raise RuntimeError(
            f"Expected {expected_global} globally unique identities; duplicates={duplicates[:8]}"
        )
    if len(set(model_digests)) != 1 or not model_digests[0]:
        raise RuntimeError(f"Post-step parameters diverged across ranks: {model_digests}")
    if len(set(ema_digests)) != 1 or not ema_digests[0]:
        raise RuntimeError(f"Post-step EMA diverged across ranks: {ema_digests}")

    return {
        "status": "passed",
        "world_size": expected_world_size,
        "batch_per_rank": expected_batch_per_rank,
        "global_identity_count": len(identities),
        "global_unique_identity_count": len(set(identities)),
        "providers": sorted(set(providers)),
        "optimizer_steps_by_rank": [int(report["optimizer_steps"]) for report in rank_audits],
        "gradient_hook_tensors_by_rank": [
            len(report["gradient_records"]) for report in rank_audits
        ],
        "gradient_families_nonzero_on_ranks": family_nonzero_by_rank,
        "gradient_family_tensor_counts": dict(EXPECTED_GRADIENT_FAMILY_TENSORS),
        "post_parameter_digest": model_digests[0],
        "post_parameter_digest_unique_count": len(set(model_digests)),
        "ema_digest": ema_digests[0],
        "ema_digest_unique_count": len(set(ema_digests)),
        "ema_steps_by_rank": [int(report["ema_step_count"]) for report in rank_audits],
        "checkpoint_hash_locking": False,
    }


def gather_and_validate_local_audit(
    local_audit: Mapping[str, Any] | None,
    *,
    local_error: str | None = None,
) -> dict[str, Any]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Distributed smoke audit requires initialized torch.distributed")
    world_size = dist.get_world_size()
    expected_world_size = expected_smoke_world_size()
    if world_size != expected_world_size:
        raise RuntimeError(
            "Distributed smoke world-size mismatch before collective: "
            f"configured={expected_world_size} actual={world_size}"
        )
    # Every rank always enters the same collective, including a rank whose
    # local digest/EMA construction failed. This avoids stranding peers in an
    # all_gather when one process raises immediately before the collective.
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
        raise RuntimeError(f"Distributed smoke local audit failed uniformly: {errors}")
    audits = [item.get("audit") for item in gathered]
    if any(not isinstance(audit, Mapping) for audit in audits):
        raise RuntimeError("Distributed smoke gathered a missing/malformed local audit")
    return validate_rank_audits(audits, expected_world_size=expected_world_size)

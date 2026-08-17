"""Strict contracts for AMB3R-pose-only Heatmap Head adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .utils import _load_normalized_state_dict, _normalize_state_key, safe_torch_load


POSE_ADAPTATION_PREFIXES = (
    "heatmap_vln.coarse.proj_traj.",
    "heatmap_vln.coarse.self_attn.",
    "heatmap_vln.coarse.vis_head.",
    "heatmap_vln.coarse.heatmap_head.",
)
HEATMAP_HEAD_PREFIXES = (
    "heatmap_vln.vit_dpt_fusion.",
    "heatmap_vln.vit_panorama_conditioner.",
    "heatmap_vln.coarse_panorama_conditioner.",
    "heatmap_vln.coarse.",
    "heatmap_vln.fine.",
)
EXPECTED_POSE_ADAPTATION_TENSORS = 34
EXPECTED_COMPLETE_HEATMAP_HEAD_TENSORS = 79
PPA_FUTURE_PREFIX = "past_plan_action.future_head."


def configured_pose_adaptation_prefixes(stage_cfg: Mapping[str, Any]) -> tuple[str, ...]:
    raw = stage_cfg.get("heatmap_trainable_parameter_prefixes") or ()
    prefixes = tuple(str(value) for value in raw)
    if not prefixes:
        return ()
    if len(prefixes) != len(set(prefixes)) or set(prefixes) != set(POSE_ADAPTATION_PREFIXES):
        raise ValueError(
            "AMB3R pose adaptation requires exactly the audited four heatmap "
            f"parameter prefixes: {POSE_ADAPTATION_PREFIXES}; got={prefixes}"
        )
    return POSE_ADAPTATION_PREFIXES


def is_pose_adaptation_stage(stage_cfg: Mapping[str, Any]) -> bool:
    return bool(configured_pose_adaptation_prefixes(stage_cfg))


def assert_required_history_pose_provider(
    batch: Mapping[str, Any],
    stage_cfg: Mapping[str, Any],
) -> None:
    """Fail closed before forward if adaptation data could contain GT poses."""
    required = stage_cfg.get("required_history_pose_provider")
    if not required:
        return
    actual = batch.get("history_pose_provider")
    if actual is None:
        raise RuntimeError(
            "Pose-domain adaptation batch is missing history_pose_provider; "
            "refusing to silently train/evaluate on GT pose input"
        )
    values = [actual] if isinstance(actual, str) else list(actual)
    if not values or any(str(value) != str(required) for value in values):
        raise RuntimeError(
            "Pose-domain adaptation requires 100% provider "
            f"{required!r}, got={values[:8]}"
        )


def complete_heatmap_head_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return every learned Heatmap Head tensor, excluding fixed buffers.

    ``SingleViewPanoramaConditioner`` deliberately keeps two persistent
    ``direction_angles_degrees`` buffers so the direction convention remains
    visible in a normal module state dict.  They are deterministic architecture
    constants, not learned checkpoint weights.  The self-contained adaptation
    contract therefore follows ``named_parameters()`` exactly: 79 learned Head
    tensors, while the full module ``state_dict()`` contains 81 entries.
    """
    result: dict[str, torch.Tensor] = {}
    for raw_name, tensor in model.named_parameters():
        name = _normalize_state_key(raw_name)
        if name.startswith(HEATMAP_HEAD_PREFIXES):
            result[name] = tensor.detach()
    if len(result) != EXPECTED_COMPLETE_HEATMAP_HEAD_TENSORS:
        raise RuntimeError(
            "The self-contained AMB3R adaptation checkpoint contract expects "
            f"{EXPECTED_COMPLETE_HEATMAP_HEAD_TENSORS} Heatmap Head tensors, "
            f"found {len(result)}"
        )
    return result


def load_pose_adaptation_initialization(
    model: nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Fresh-init the complete Head from deployment/EMA weights, without hashes.

    Optimizer, scheduler, epoch and online weights are deliberately ignored.
    Exact-resume continues to use the normal ``--resume`` path.
    """
    path = Path(checkpoint_path)
    payload = safe_torch_load(str(path))
    if not isinstance(payload, Mapping):
        raise RuntimeError("Pose-adaptation initializer must be a checkpoint mapping")
    raw_state = payload.get("trainable_state_dict")
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise RuntimeError(
            "Pose-adaptation fresh init requires non-empty trainable_state_dict "
            "(EMA/deployment weights)"
        )
    source = {}
    for raw_name, tensor in raw_state.items():
        name = _normalize_state_key(raw_name)
        if name.startswith(HEATMAP_HEAD_PREFIXES):
            source[name] = tensor
    expected = complete_heatmap_head_state(model)
    if set(source) != set(expected):
        raise RuntimeError(
            "Pose-adaptation initializer is not a complete exact Heatmap Head: "
            f"missing={sorted(set(expected) - set(source))[:8]} "
            f"extra={sorted(set(source) - set(expected))[:8]}"
        )
    shape_mismatch = sorted(
        name for name in expected if tuple(source[name].shape) != tuple(expected[name].shape)
    )
    if shape_mismatch:
        raise RuntimeError(
            "Pose-adaptation initializer has incompatible tensor shapes: "
            f"{shape_mismatch[:8]}"
        )
    missing, unexpected, loaded = _load_normalized_state_dict(model, source)
    if loaded != EXPECTED_COMPLETE_HEATMAP_HEAD_TENSORS or unexpected:
        raise RuntimeError(
            "Pose-adaptation initializer did not load the exact complete Head: "
            f"loaded={loaded} unexpected={unexpected[:8]}"
        )
    return {
        "checkpoint_path": str(path.resolve()),
        "source_state_key": "trainable_state_dict",
        "source_weight_semantics": dict(payload.get("weight_semantics") or {}),
        "loaded_tensor_count": loaded,
        "model_missing_key_count": len(missing),
        "fresh_optimizer_scheduler": True,
        "hash_locking": False,
    }


def load_past_plan_action_initialization(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    stage: str,
) -> dict[str, Any]:
    """Load exact learned inputs for PPA with a fresh optimizer.

    Stage 1 starts from the complete 79-parameter AMB3R-adapted Past Head.
    Stage 2 starts from a Stage-1 deployment entry containing those same 79
    parameters plus every Future Head parameter.  The bridge is intentionally
    never loaded during the stage transition: its output projection must start
    at exact zero so the native action path is initially bitwise identical.
    """
    if stage not in {"stage1_map_pretrain", "stage2_joint"}:
        raise ValueError(f"unsupported Past->Plan->Action stage: {stage!r}")
    path = Path(checkpoint_path)
    payload = safe_torch_load(str(path))
    if not isinstance(payload, Mapping):
        raise RuntimeError("PPA initializer must be a checkpoint mapping")
    raw_state = payload.get("trainable_state_dict")
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise RuntimeError(
            "PPA fresh init requires non-empty trainable_state_dict "
            "(EMA/deployment weights)"
        )
    normalized_source = {
        _normalize_state_key(raw_name): tensor
        for raw_name, tensor in raw_state.items()
    }
    expected_head = complete_heatmap_head_state(model)
    source_head = {
        name: tensor
        for name, tensor in normalized_source.items()
        if name.startswith(HEATMAP_HEAD_PREFIXES)
    }
    if set(source_head) != set(expected_head):
        raise RuntimeError(
            "PPA initializer is not a complete exact 79-parameter Past Head: "
            f"missing={sorted(set(expected_head) - set(source_head))[:8]} "
            f"extra={sorted(set(source_head) - set(expected_head))[:8]}"
        )

    selected = dict(source_head)
    expected_future = {
        _normalize_state_key(name): parameter.detach()
        for name, parameter in model.named_parameters()
        if _normalize_state_key(name).startswith(PPA_FUTURE_PREFIX)
    }
    if stage == "stage2_joint":
        source_future = {
            name: tensor
            for name, tensor in normalized_source.items()
            if name.startswith(PPA_FUTURE_PREFIX)
        }
        if not expected_future or set(source_future) != set(expected_future):
            raise RuntimeError(
                "PPA Stage 2 initializer lacks the exact trained Future Head: "
                f"missing={sorted(set(expected_future) - set(source_future))[:8]} "
                f"extra={sorted(set(source_future) - set(expected_future))[:8]}"
            )
        selected.update(source_future)

    expected_selected = {**expected_head}
    if stage == "stage2_joint":
        expected_selected.update(expected_future)
    shape_mismatch = sorted(
        name
        for name in expected_selected
        if tuple(selected[name].shape) != tuple(expected_selected[name].shape)
    )
    if shape_mismatch:
        raise RuntimeError(
            f"PPA initializer has incompatible tensor shapes: {shape_mismatch[:8]}"
        )
    missing, unexpected, loaded = _load_normalized_state_dict(model, selected)
    if loaded != len(expected_selected) or unexpected:
        raise RuntimeError(
            "PPA initializer did not load its exact learned state: "
            f"loaded={loaded}/{len(expected_selected)} unexpected={unexpected[:8]}"
        )

    bridge = getattr(getattr(model, "past_plan_action", None), "bridge", None)
    out_proj = getattr(getattr(bridge, "cross_attention", None), "out_proj", None)
    if out_proj is None or not torch.equal(
        out_proj.weight.detach(), torch.zeros_like(out_proj.weight)
    ) or not torch.equal(
        out_proj.bias.detach(), torch.zeros_like(out_proj.bias)
    ):
        raise RuntimeError(
            "PPA bridge output projection must remain exact zero at fresh init"
        )
    return {
        "checkpoint_path": str(path.resolve()),
        "source_state_key": "trainable_state_dict",
        "stage": stage,
        "loaded_heatmap_head_tensors": len(expected_head),
        "loaded_future_head_tensors": (
            len(expected_future) if stage == "stage2_joint" else 0
        ),
        "loaded_tensor_count": loaded,
        "model_missing_key_count": len(missing),
        "bridge_zero_initialized": True,
        "fresh_optimizer_scheduler": True,
        "hash_locking": False,
    }

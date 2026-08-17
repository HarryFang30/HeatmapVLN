"""Fail-closed contracts for HeatmapVLN warm-start checkpoints."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .utils import _normalize_state_key


SPATIAL_RESET_POLICY = "spatial_reset_v1"
FULL_HEAD_POLICY = "full_head_v1"
_VIT_DPT_PREFIX = "heatmap_vln.vit_dpt_fusion."
_LLM_DPT_PREFIX = "heatmap_vln.llm_dpt_fusion."
_COARSE_PREFIX = "heatmap_vln.coarse."
_FINE_PREFIX = "heatmap_vln.fine."
_FINE_OUTPUT_KEYS = (
    "heatmap_vln.fine.refine.4.weight",
    "heatmap_vln.fine.refine.4.bias",
)
_RESET_MODULES = ("vit_dpt_fusion", "fine")
_KEPT_MODULES = ("llm_dpt_fusion", "coarse")


def _normalized_unique_state(
    state_dict: dict[str, torch.Tensor],
    *,
    source: str,
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    duplicates: list[str] = []
    for raw_name, value in state_dict.items():
        name = _normalize_state_key(raw_name)
        if name in normalized:
            duplicates.append(name)
        normalized[name] = value
    if duplicates:
        raise RuntimeError(
            f"{source} contains duplicate normalized keys: {sorted(set(duplicates))[:5]}"
        )
    return normalized


def _contract_from_stage(stage_cfg: dict) -> dict[str, Any] | None:
    contract = stage_cfg.get("heatmap_warmstart_contract")
    if contract is None:
        return None
    if not isinstance(contract, dict):
        raise TypeError("heatmap_warmstart_contract must be a mapping")
    if contract.get("policy") not in {
        SPATIAL_RESET_POLICY,
        FULL_HEAD_POLICY,
    }:
        raise ValueError(
            "Unsupported heatmap warm-start policy: "
            f"{contract.get('policy')!r}"
        )
    if "heatmap_vln" not in set(stage_cfg.get("trainable_modules", ())):
        raise RuntimeError(
            "heatmap_warmstart_contract requires heatmap_vln to be trainable"
        )
    return contract


def _validate_full_head_contract(
    state: dict[str, torch.Tensor],
    model_params: dict[str, torch.Tensor],
    contract: dict[str, Any],
    *,
    checkpoint_metadata: dict[str, Any] | None,
    source: str,
) -> dict[str, Any]:
    """Require the complete pretrained HeatmapVLN head plus frozen LoRA.

    The final Fine decoder, rather than ``coarse_heatmap`` alone, was the
    supervised locator in the legacy checkpoint.  A compatibility warm start
    must therefore load every ViT-DPT, LLM-DPT, Coarse and Fine parameter.
    """

    prefix_to_name = {
        _VIT_DPT_PREFIX: "vit_dpt_fusion",
        _LLM_DPT_PREFIX: "llm_dpt_fusion",
        _COARSE_PREFIX: "coarse",
        _FINE_PREFIX: "fine",
    }
    lora_keys = {name for name in state if "lora_" in name}
    module_keys = {
        label: {
            name
            for name in state
            if name.startswith(prefix)
        }
        for prefix, label in prefix_to_name.items()
    }
    allowed_keys = set(lora_keys)
    for keys in module_keys.values():
        allowed_keys.update(keys)
    unexpected = sorted(set(state) - allowed_keys)
    if unexpected:
        raise RuntimeError(
            "Heatmap full-head checkpoint contains forbidden tensors: "
            f"unexpected={unexpected[:8]}"
        )

    expected_counts = {
        "lora": int(contract.get("expected_lora_tensors", 224)),
        "vit_dpt_fusion": int(contract.get("expected_vit_dpt_tensors", 12)),
        "llm_dpt_fusion": int(contract.get("expected_llm_dpt_tensors", 10)),
        "coarse": int(contract.get("expected_coarse_tensors", 37)),
        "fine": int(contract.get("expected_fine_tensors", 6)),
    }
    actual_counts = {
        "lora": len(lora_keys),
        **{
            label: len(keys)
            for label, keys in module_keys.items()
        },
    }
    if actual_counts != expected_counts:
        raise RuntimeError(
            "Heatmap full-head checkpoint tensor-count contract failed: "
            f"expected={expected_counts}, actual={actual_counts}"
        )

    required_model_keys = {
        name
        for name in model_params
        if (
            "lora_" in name
            or any(name.startswith(prefix) for prefix in prefix_to_name)
        )
    }
    missing = sorted(required_model_keys - set(state))
    extra_required = sorted(allowed_keys - required_model_keys)
    if missing or extra_required:
        raise RuntimeError(
            "Heatmap full-head checkpoint does not exactly cover the required "
            "current-model parameters: "
            f"missing={missing[:8]}, unexpected={extra_required[:8]}"
        )

    shape_mismatches = [
        (
            f"{name}: checkpoint={tuple(state[name].shape)} "
            f"model={tuple(model_params[name].shape)}"
        )
        for name in sorted(required_model_keys)
        if tuple(state[name].shape) != tuple(model_params[name].shape)
    ]
    if shape_mismatches:
        raise RuntimeError(
            "Heatmap full-head checkpoint shape contract failed: "
            f"{shape_mismatches[:5]}"
        )

    if bool(contract.get("require_metadata", True)):
        metadata_contract = (checkpoint_metadata or {}).get(
            "heatmap_warmstart_contract"
        )
        expected_metadata = {
            "policy": FULL_HEAD_POLICY,
            "kept_heatmap_modules": [
                "vit_dpt_fusion",
                "llm_dpt_fusion",
                "coarse",
                "fine",
            ],
        }
        if not isinstance(metadata_contract, dict):
            raise RuntimeError(
                "Heatmap full-head checkpoint lacks required contract metadata"
            )
        mismatched_metadata = {
            key: {
                "expected": value,
                "actual": metadata_contract.get(key),
            }
            for key, value in expected_metadata.items()
            if metadata_contract.get(key) != value
        }
        if mismatched_metadata:
            raise RuntimeError(
                "Heatmap full-head checkpoint metadata contract failed: "
                f"{mismatched_metadata}"
            )

    return {
        "policy": FULL_HEAD_POLICY,
        "checkpoint_path": source,
        "expected_loaded_tensors": len(required_model_keys),
        "counts": actual_counts,
        "fine_output_keys": [],
    }


def validate_heatmap_warmstart_contract(
    model: nn.Module,
    checkpoint_state_dict: dict[str, torch.Tensor],
    stage_cfg: dict,
    *,
    checkpoint_metadata: dict[str, Any] | None = None,
    checkpoint_path: str | None = None,
) -> dict[str, Any] | None:
    """Validate the partial spatial-reset initialization before loading it.

    ``spatial_reset_v1`` accepts exactly:

    * every LoRA tensor required by the current model;
    * every ``llm_dpt_fusion`` and ``coarse`` parameter;
    * zero-valued ``fine.refine.4.{weight,bias}``.

    Old ``vit_dpt_fusion`` weights and all other old ``fine`` weights are
    forbidden. Their absence leaves the freshly seeded model initialization in
    place, while the zero final layer makes the initial fine residual exactly
    zero.
    """
    contract = _contract_from_stage(stage_cfg)
    if contract is None:
        return None

    source = checkpoint_path or "warm-start checkpoint"
    state = _normalized_unique_state(checkpoint_state_dict, source=source)
    model_params = _normalized_unique_state(
        dict(model.named_parameters()),
        source="current model",
    )
    if contract.get("policy") == FULL_HEAD_POLICY:
        return _validate_full_head_contract(
            state,
            model_params,
            contract,
            checkpoint_metadata=checkpoint_metadata,
            source=source,
        )

    lora_keys = {name for name in state if "lora_" in name}
    llm_dpt_keys = {name for name in state if name.startswith(_LLM_DPT_PREFIX)}
    coarse_keys = {name for name in state if name.startswith(_COARSE_PREFIX)}
    fine_output_keys = set(_FINE_OUTPUT_KEYS) & set(state)
    allowed_keys = lora_keys | llm_dpt_keys | coarse_keys | fine_output_keys
    unexpected = sorted(set(state) - allowed_keys)
    if unexpected:
        raise RuntimeError(
            "Heatmap spatial-reset checkpoint contains forbidden tensors; "
            "old vit_dpt/fine weights must not be loaded. "
            f"unexpected={unexpected[:8]}"
        )

    expected_counts = {
        "lora": int(contract.get("expected_lora_tensors", 224)),
        "llm_dpt_fusion": int(contract.get("expected_llm_dpt_tensors", 10)),
        "coarse": int(contract.get("expected_coarse_tensors", 37)),
        "fine_zero_output": len(_FINE_OUTPUT_KEYS),
    }
    actual_counts = {
        "lora": len(lora_keys),
        "llm_dpt_fusion": len(llm_dpt_keys),
        "coarse": len(coarse_keys),
        "fine_zero_output": len(fine_output_keys),
    }
    if actual_counts != expected_counts:
        raise RuntimeError(
            "Heatmap spatial-reset checkpoint tensor-count contract failed: "
            f"expected={expected_counts}, actual={actual_counts}"
        )

    required_model_keys = {
        name
        for name in model_params
        if (
            "lora_" in name
            or name.startswith(_LLM_DPT_PREFIX)
            or name.startswith(_COARSE_PREFIX)
        )
    } | set(_FINE_OUTPUT_KEYS)
    missing = sorted(required_model_keys - set(state))
    extra_required = sorted(allowed_keys - required_model_keys)
    if missing or extra_required:
        raise RuntimeError(
            "Heatmap spatial-reset checkpoint does not exactly cover the "
            "required current-model parameters: "
            f"missing={missing[:8]}, unexpected={extra_required[:8]}"
        )

    shape_mismatches = [
        (
            f"{name}: checkpoint={tuple(state[name].shape)} "
            f"model={tuple(model_params[name].shape)}"
        )
        for name in sorted(required_model_keys)
        if tuple(state[name].shape) != tuple(model_params[name].shape)
    ]
    if shape_mismatches:
        raise RuntimeError(
            "Heatmap spatial-reset checkpoint shape contract failed: "
            f"{shape_mismatches[:5]}"
        )

    nonzero_fine = [
        name
        for name in _FINE_OUTPUT_KEYS
        if (
            not bool(torch.isfinite(state[name].float()).all())
            or int(torch.count_nonzero(state[name]).item()) != 0
        )
    ]
    if nonzero_fine:
        raise RuntimeError(
            "Heatmap spatial-reset checkpoint requires an exactly zero fine "
            f"output layer, invalid={nonzero_fine}"
        )

    if bool(contract.get("require_metadata", True)):
        metadata_contract = (checkpoint_metadata or {}).get(
            "heatmap_warmstart_contract"
        )
        expected_metadata = {
            "policy": SPATIAL_RESET_POLICY,
            "kept_heatmap_modules": list(_KEPT_MODULES),
            "reset_heatmap_modules": list(_RESET_MODULES),
            "zero_initialized_parameters": list(_FINE_OUTPUT_KEYS),
        }
        if not isinstance(metadata_contract, dict):
            raise RuntimeError(
                "Heatmap spatial-reset checkpoint lacks required contract metadata"
            )
        mismatched_metadata = {
            key: {
                "expected": value,
                "actual": metadata_contract.get(key),
            }
            for key, value in expected_metadata.items()
            if metadata_contract.get(key) != value
        }
        if mismatched_metadata:
            raise RuntimeError(
                "Heatmap spatial-reset checkpoint metadata contract failed: "
                f"{mismatched_metadata}"
            )

    return {
        "policy": SPATIAL_RESET_POLICY,
        "checkpoint_path": source,
        "expected_loaded_tensors": len(required_model_keys),
        "counts": actual_counts,
        "fine_output_keys": list(_FINE_OUTPUT_KEYS),
    }


def verify_heatmap_warmstart_loaded(
    model: nn.Module,
    report: dict[str, Any] | None,
    *,
    loaded_count: int,
) -> None:
    """Verify loader accounting and the zero-residual invariant after load."""
    if report is None:
        return
    expected_loaded = int(report["expected_loaded_tensors"])
    if loaded_count != expected_loaded:
        raise RuntimeError(
            "Heatmap warm-start loader count mismatch: "
            f"expected={expected_loaded}, loaded={loaded_count}"
        )

    model_params = _normalized_unique_state(
        dict(model.named_parameters()),
        source="loaded model",
    )
    invalid = [
        name
        for name in report["fine_output_keys"]
        if (
            name not in model_params
            or not bool(torch.isfinite(model_params[name].detach().float()).all())
            or int(torch.count_nonzero(model_params[name].detach()).item()) != 0
        )
    ]
    if invalid:
        raise RuntimeError(
            "Heatmap warm-start load did not preserve the exact zero fine "
            f"residual output layer: {invalid}"
        )

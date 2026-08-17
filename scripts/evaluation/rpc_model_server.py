#!/usr/bin/env python3
"""HeatmapVLN model-side RPC server.

Run this in the model environment.  It intentionally does not import Habitat;
the Habitat process sends RGB observations through the vla_rpc bridge and
receives discrete Habitat actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import signal
import sys
from collections.abc import Sequence
from concurrent import futures
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import grpc
import numpy as np
import torch
from PIL import Image
from scripts.evaluation.rpc_protocol import (
    HEATMAPVLN_RPC_PROTOCOL_VERSION,
    HEATMAPVLN_RPC_SAMPLING_FIELD,
    validate_rpc_sampling_metadata,
)
from scripts.training.utils import (
    _normalize_state_key,
    assert_complete_lora_checkpoint_match,
    extract_lora_checkpoint_state,
    load_config,
)
from transformers import LogitsProcessor, LogitsProcessorList
from vla_rpc.core.image import decode_jpeg_to_rgb
from vla_rpc.proto import vla_pb2, vla_pb2_grpc

from src.models.heatmap.input_constructor import (
    construct_input,
    parse_structured_pano_output,
    structured_condition_text,
    vlm_output_requests_stop,
    vlm_output_requests_turn,
)
from src.models.qwen2_5_vl.integration import (
    DEFAULT_LORA_ADAPTER_NAME,
    STOP_DECISION_ADAPTER_NAME,
)
from src.models.runtime_compat import install_flash_attn_stub, install_numpy_legacy_aliases
from src.utils.trajectory_direction import (
    align_trajectory_endpoint_heading,
    view_pixel_target_angle_deg,
)

LOGGER = logging.getLogger("heatmapvln-rpc-server")

STOP_DECISION_CHECKPOINT_SCHEMA = "heatmapvln-system2-stop-decision-adapter-v1"
STOP_DECISION_ADD_AND_VETO_POLICY = "add_and_veto"
STOP_DECISION_VETO_ONLY_POLICY = "veto_only"

MAX_STEPS = 8
MAX_LOCAL_STEPS = 4
PROTO_VERSION = HEATMAPVLN_RPC_PROTOCOL_VERSION
LOCAL_FJL_ROOT = Path(os.environ.get("HEATMAPVLN_FJL_ROOT", "/mnt/afs/lixiaoou/intern/fjl"))
LOCAL_INTERNNAV_MODEL_PATH = Path(
    os.environ.get("HEATMAPVLN_INTERNNAV_MODEL_PATH", str(LOCAL_FJL_ROOT / "InternNav-Model"))
)


def _default_internnav_model_path() -> str:
    for raw in (
        os.environ.get("INTERNNAV_MODEL_PATH"),
        os.environ.get("INTERNNAV_BACKBONE"),
        str(LOCAL_INTERNNAV_MODEL_PATH),
    ):
        if not raw:
            continue
        candidate = Path(os.path.expandvars(os.path.expanduser(str(raw))))
        if candidate.exists():
            return str(candidate.resolve())
    return os.environ.get("INTERNNAV_MODEL_PATH", "")


class ActionCode:
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOKUP = 4
    LOOKDOWN = 5


def _extract_checkpoint_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")
    for key in ("model_state_dict", "trainable_state_dict", "state_dict"):
        state_dict = ckpt.get(key)
        if isinstance(state_dict, dict):
            return state_dict
    if all(torch.is_tensor(value) for value in ckpt.values()):
        return ckpt
    raise KeyError(f"Checkpoint does not contain model_state_dict/trainable_state_dict/state_dict: {checkpoint_path}")


def _extract_checkpoint_config(checkpoint_path: str | None) -> dict:
    if not checkpoint_path:
        return {}
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        return {}
    cfg = ckpt.get("config", {})
    return cfg if isinstance(cfg, dict) else {}


def _state_has_prefix(state_dict: dict[str, torch.Tensor] | None, prefix: str) -> bool:
    if not state_dict:
        return False
    return any(_normalize_state_key(key).startswith(prefix) for key in state_dict)


def _looks_action_only(state_dict: dict[str, torch.Tensor]) -> bool:
    if not state_dict:
        return False
    prefixes = {_normalize_state_key(key).split(".", 1)[0] for key in state_dict}
    return prefixes.issubset({"latent_queries", "nextdit_action_head"})


def _checkpoint_has_base_weights(state_dict: dict[str, torch.Tensor] | None) -> bool:
    return (
        _state_has_prefix(state_dict, "qwen2_5_vl.")
        or _state_has_prefix(state_dict, "qwen3_5.")
        or _state_has_prefix(state_dict, "qwen3_5_vl.")
        or _state_has_prefix(state_dict, "heatmap_vln.")
    )


def _requires_base_checkpoint(cfg: dict, checkpoint_cfg: dict | None = None) -> bool:
    for source in (checkpoint_cfg, cfg):
        if not isinstance(source, dict):
            continue
        stages = source.get("training", {}).get("stages", [])
        if not stages:
            continue
        stage_cfg = stages[0]
        if stage_cfg.get("requires_base_checkpoint") or stage_cfg.get("bridge_only"):
            return True
    return False


def _first_stage_config(*configs: dict | None) -> dict:
    for config in configs:
        if not isinstance(config, dict):
            continue
        stages = config.get("training", {}).get("stages", [])
        if stages and isinstance(stages[0], dict):
            return stages[0]
    return {}


def _assert_finite_state_dict(state_dict: dict[str, torch.Tensor], label: str) -> None:
    non_tensors = [name for name, value in state_dict.items() if not torch.is_tensor(value)]
    if non_tensors:
        raise RuntimeError(f"{label} contains non-tensor values: {non_tensors[:5]}")
    nonfinite = [name for name, value in state_dict.items() if not bool(torch.isfinite(value.float()).all())]
    if nonfinite:
        raise RuntimeError(f"{label} contains non-finite tensors: {nonfinite[:5]}")


def _load_system2_stop_decision_adapter(
    checkpoint_path: str,
    *,
    integration: Any,
    expected_base_checkpoint: str | None = None,
    add_threshold_override: float | None = None,
    veto_threshold_override: float | None = None,
) -> dict[str, Any]:
    """Load a STOP-only LoRA delta while proving the navigation LoRA is unchanged."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Invalid System2 STOP-decision checkpoint: {checkpoint_path}")
    if checkpoint.get("schema") != STOP_DECISION_CHECKPOINT_SCHEMA:
        raise RuntimeError(
            "System2 STOP-decision checkpoint schema mismatch: "
            f"{checkpoint.get('schema')!r}"
        )
    if checkpoint.get("adapter_name") != STOP_DECISION_ADAPTER_NAME:
        raise RuntimeError(
            "System2 STOP-decision checkpoint adapter_name mismatch: "
            f"{checkpoint.get('adapter_name')!r}"
        )
    policy_kind = str(
        checkpoint.get("policy_kind") or STOP_DECISION_ADD_AND_VETO_POLICY
    )
    if policy_kind not in {
        STOP_DECISION_ADD_AND_VETO_POLICY,
        STOP_DECISION_VETO_ONLY_POLICY,
    }:
        raise RuntimeError(
            f"Unsupported System2 STOP-decision policy_kind: {policy_kind!r}"
        )

    adapter_config = checkpoint.get("adapter_config")
    base_contract = checkpoint.get("base_contract")
    thresholds = checkpoint.get("thresholds")
    token_contract = checkpoint.get("token_contract")
    state_dict = checkpoint.get("adapter_state_dict")
    if not all(
        isinstance(value, dict)
        for value in (
            adapter_config,
            base_contract,
            thresholds,
            token_contract,
            state_dict,
        )
    ):
        raise RuntimeError("System2 STOP-decision checkpoint contract is incomplete")
    _assert_finite_state_dict(state_dict, "System2 STOP-decision adapter")
    if thresholds.get("quality_passed") is not True:
        raise RuntimeError(
            "STOP-decision checkpoint failed its validation quality gate: "
            f"{thresholds.get('quality_violations')!r}"
        )
    add_metrics = thresholds.get("add")
    veto_metrics = thresholds.get("veto")
    if not isinstance(add_metrics, dict) or not isinstance(veto_metrics, dict):
        raise RuntimeError("STOP-decision checkpoint has no calibrated add/veto metrics")
    add_enabled = policy_kind == STOP_DECISION_ADD_AND_VETO_POLICY
    if (
        (add_enabled and float(add_metrics.get("recall", 0.0)) < 0.5)
        or (add_enabled and float(add_metrics.get("false_positive_rate", 1.0)) > 0.0)
        or float(veto_metrics.get("recall", 0.0)) < 0.98
        or float(veto_metrics.get("negative_rejection_rate", 0.0)) < 0.2
        or float(thresholds.get("roc_auc", 0.0)) < 0.75
        or int(thresholds.get("veto_reference_positive_count", 0) or 0) <= 0
    ):
        raise RuntimeError("STOP-decision checkpoint calibration metrics are below contract")
    if thresholds.get("policy_kind", policy_kind) != policy_kind:
        raise RuntimeError("STOP-decision checkpoint policy_kind contract is inconsistent")
    if bool(thresholds.get("add_enabled", add_enabled)) != add_enabled:
        raise RuntimeError("STOP-decision checkpoint add-enabled contract is inconsistent")
    if not add_enabled and add_threshold_override is not None:
        raise RuntimeError("A veto-only STOP-decision policy forbids add threshold overrides")
    training = checkpoint.get("training")
    if not isinstance(training, dict) or not (
        0.0 < float(training.get("holdout_scene_fraction", 0.0) or 0.0) < 1.0
        and float(training.get("ranking_loss_weight", 0.0) or 0.0) > 0.0
    ):
        raise RuntimeError(
            "STOP-decision checkpoint lacks scene-held-out ranking-loss training metadata"
        )

    rank = int(adapter_config.get("rank", 0) or 0)
    alpha = int(adapter_config.get("alpha", 0) or 0)
    layers = [int(value) for value in (adapter_config.get("layer_indices") or [])]
    targets = [str(value) for value in (adapter_config.get("target_modules") or [])]
    dropout = float(adapter_config.get("dropout", float("nan")))
    if (
        rank <= 0
        or alpha <= 0
        or not layers
        or layers != sorted(set(layers))
        or min(layers) < 0
        or not targets
        or len(targets) != len(set(targets))
        or dropout != 0.0
    ):
        raise RuntimeError(
            "Invalid System2 STOP-decision adapter config: "
            f"rank={rank} alpha={alpha} layers={layers} targets={targets} "
            f"dropout={dropout}"
        )

    if base_contract.get("default_adapter_name") != DEFAULT_LORA_ADAPTER_NAME:
        raise RuntimeError("STOP-decision checkpoint targets the wrong navigation adapter")
    if int(base_contract.get("default_lora_tensors", 0) or 0) != 224:
        raise RuntimeError("STOP-decision checkpoint does not require all 224 navigation LoRA tensors")
    recorded_base = str(base_contract.get("checkpoint") or "")
    if expected_base_checkpoint and os.path.realpath(recorded_base) != os.path.realpath(
        expected_base_checkpoint
    ):
        raise RuntimeError(
            "STOP-decision checkpoint base path mismatch: "
            f"recorded={recorded_base!r} expected={expected_base_checkpoint!r}"
        )
    expected_default_fingerprint = str(
        base_contract.get("default_lora_fingerprint") or ""
    )
    current_default_fingerprint = integration.lora_adapter_fingerprint(
        DEFAULT_LORA_ADAPTER_NAME
    )
    if current_default_fingerprint != expected_default_fingerprint:
        raise RuntimeError(
            "STOP-decision checkpoint navigation-LoRA fingerprint mismatch: "
            f"current={current_default_fingerprint} "
            f"expected={expected_default_fingerprint}"
        )
    current_token_contract = integration.structured_view_token_contract()
    if token_contract != current_token_contract:
        raise RuntimeError(
            "STOP-decision structured-view token contract mismatch: "
            f"checkpoint={token_contract} runtime={current_token_contract}"
        )

    add_threshold = float(
        thresholds.get("add_stop_threshold")
        if add_threshold_override is None
        else add_threshold_override
    )
    veto_threshold = float(
        thresholds.get("veto_stop_threshold")
        if veto_threshold_override is None
        else veto_threshold_override
    )
    if not (
        np.isfinite(add_threshold)
        and np.isfinite(veto_threshold)
        and 0.0 <= veto_threshold < add_threshold <= 1.0
    ):
        raise RuntimeError(
            "Invalid STOP-decision hysteresis thresholds: "
            f"veto={veto_threshold} add={add_threshold}"
        )
    if not add_enabled and add_threshold != 1.0:
        raise RuntimeError(
            "Veto-only STOP-decision checkpoint must record add_stop_threshold=1.0"
        )

    integration.add_stop_decision_adapter(
        adapter_name=STOP_DECISION_ADAPTER_NAME,
        rank=rank,
        alpha=alpha,
        layer_indices=layers,
        target_modules=targets,
    )
    loaded_tensors = integration.load_lora_adapter_state_dict(
        STOP_DECISION_ADAPTER_NAME,
        state_dict,
    )
    expected_adapter_fingerprint = str(checkpoint.get("adapter_fingerprint") or "")
    current_adapter_fingerprint = integration.lora_adapter_fingerprint(
        STOP_DECISION_ADAPTER_NAME
    )
    if not expected_adapter_fingerprint or (
        current_adapter_fingerprint != expected_adapter_fingerprint
    ):
        raise RuntimeError(
            "STOP-decision adapter fingerprint mismatch after load: "
            f"current={current_adapter_fingerprint} "
            f"expected={expected_adapter_fingerprint}"
        )
    integration.activate_lora_adapters(
        (DEFAULT_LORA_ADAPTER_NAME,),
        trainable_adapters=(),
    )
    integration.model.eval()
    if integration.lora_adapter_fingerprint(
        DEFAULT_LORA_ADAPTER_NAME
    ) != expected_default_fingerprint:
        raise RuntimeError("Navigation LoRA changed while loading STOP-decision adapter")
    return {
        "policy_kind": policy_kind,
        "add_enabled": add_enabled,
        "adapter_name": STOP_DECISION_ADAPTER_NAME,
        "adapter_tensors": loaded_tensors,
        "adapter_parameters": sum(value.numel() for value in state_dict.values()),
        "adapter_fingerprint": current_adapter_fingerprint,
        "default_lora_fingerprint": current_default_fingerprint,
        "add_stop_threshold": add_threshold,
        "veto_stop_threshold": veto_threshold,
        "token_contract": current_token_contract,
    }


def _assert_navigation_only_lora(qwen_integration: Any, *, context: str) -> None:
    active = qwen_integration.active_lora_adapters()
    if active != (DEFAULT_LORA_ADAPTER_NAME,):
        raise RuntimeError(
            f"{context} requires navigation-only LoRA, active adapters={active}"
        )


def _system2_stop_decision_adapter_probe(
    *,
    qwen_integration: Any,
    inputs: dict[str, torch.Tensor],
    adapter_name: str = STOP_DECISION_ADAPTER_NAME,
) -> dict[str, Any]:
    """Score the six structured actions with default+STOP LoRA on one extra forward."""
    _assert_navigation_only_lora(
        qwen_integration,
        context="STOP-decision adapter probe entry",
    )
    contract = qwen_integration.structured_view_token_contract()
    input_ids = inputs.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise RuntimeError("STOP-decision probe requires rank-2 input_ids")
    if input_ids.shape[0] != 1:
        raise RuntimeError(
            f"RPC STOP-decision probe requires batch size 1, got {input_ids.shape[0]}"
        )
    prefix_ids = torch.tensor(
        [contract["prefix_token_ids"]],
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    probe_inputs = dict(inputs)
    probe_inputs["input_ids"] = torch.cat([input_ids, prefix_ids], dim=1)
    for mask_name, fill_value in (("attention_mask", 1), ("mm_token_type_ids", 0)):
        mask = inputs.get(mask_name)
        if mask is None:
            continue
        suffix = torch.full(
            (mask.shape[0], prefix_ids.shape[1]),
            fill_value,
            device=mask.device,
            dtype=mask.dtype,
        )
        probe_inputs[mask_name] = torch.cat([mask, suffix], dim=1)
    probe_inputs.pop("position_ids", None)
    probe_inputs.pop("labels", None)

    try:
        qwen_integration.activate_lora_adapters(
            (DEFAULT_LORA_ADAPTER_NAME, adapter_name),
            trainable_adapters=(),
        )
        with torch.inference_mode():
            hidden, _vision, _num_images, _traj, _lm = (
                qwen_integration._forward_model_inputs(
                    probe_inputs,
                    return_hidden_states=True,
                    skip_lm_head=True,
                    return_last_hidden_state_only=True,
                    extract_vision_hidden_states=False,
                )
            )
        if hidden is None:
            raise RuntimeError("STOP-decision adapter probe returned no hidden state")
        positions = torch.full(
            (hidden.shape[0],),
            hidden.shape[1] - 1,
            device=hidden.device,
            dtype=torch.long,
        )
        class_logits = qwen_integration.structured_view_class_logits(
            hidden,
            positions,
        )
        class_probabilities = torch.softmax(class_logits.float(), dim=-1)
        stop_log_odds = class_logits[:, 0] - torch.logsumexp(
            class_logits[:, 1:], dim=-1
        )
    finally:
        qwen_integration.activate_lora_adapters(
            (DEFAULT_LORA_ADAPTER_NAME,),
            trainable_adapters=(),
        )
    _assert_navigation_only_lora(
        qwen_integration,
        context="STOP-decision adapter probe exit",
    )
    probabilities = class_probabilities[0].detach().float().cpu().tolist()
    if len(probabilities) != len(contract["classes"]) or not all(
        np.isfinite(value) for value in probabilities
    ):
        raise RuntimeError("STOP-decision adapter returned invalid class probabilities")
    return {
        "stop_probability": float(probabilities[0]),
        "stop_log_odds": float(stop_log_odds[0].detach().float().cpu().item()),
        "selected": contract["classes"][int(np.argmax(probabilities))],
        "class_probabilities": dict(zip(contract["classes"], probabilities)),
    }


def _load_system2_stop_head(
    checkpoint_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    add_threshold_override: float | None = None,
    veto_threshold_override: float | None = None,
) -> tuple[torch.nn.Module, float, float]:
    """Load an isolated STOP classifier checkpoint with an exact state contract."""
    from src.models.action import StopPredictionHead

    checkpoint_config = _extract_checkpoint_config(checkpoint_path)
    model_config = checkpoint_config.get("model", {})
    llm_config = model_config.get("llm", {})
    head_config = model_config.get("stop_head", {})
    if not bool(head_config.get("enabled", False)):
        raise RuntimeError("System2 STOP-head checkpoint config does not enable stop_head")
    head = StopPredictionHead(
        input_dim=int(llm_config.get("hidden_dim", 3584)),
        hidden_dim=int(head_config.get("hidden_dim", 512)),
        dropout=float(head_config.get("dropout", 0.1)),
        focal_gamma=float(head_config.get("focal_gamma", 2.0)),
        focal_alpha=float(head_config.get("focal_alpha", 0.5)),
        pos_weight=float(head_config.get("pos_weight", 1.0)),
        bce_mix=float(head_config.get("bce_mix", 0.5)),
    )
    checkpoint_state = _extract_checkpoint_state_dict(checkpoint_path)
    _assert_finite_state_dict(checkpoint_state, "System2 STOP-head checkpoint")
    normalized = {
        _normalize_state_key(name): value
        for name, value in checkpoint_state.items()
    }
    unexpected = sorted(name for name in normalized if not name.startswith("stop_head."))
    if unexpected:
        raise RuntimeError(
            "System2 STOP-head checkpoint contains non-head trainable tensors: "
            f"{unexpected[:5]}"
        )
    head_state = {
        name.removeprefix("stop_head."): value
        for name, value in normalized.items()
    }
    expected = head.state_dict()
    if set(head_state) != set(expected):
        raise RuntimeError(
            "Incomplete System2 STOP-head checkpoint: "
            f"found={len(head_state)} expected={len(expected)} "
            f"missing={sorted(set(expected) - set(head_state))[:5]} "
            f"unexpected={sorted(set(head_state) - set(expected))[:5]}"
        )
    mismatched = sorted(
        name
        for name in expected
        if tuple(head_state[name].shape) != tuple(expected[name].shape)
    )
    if mismatched:
        raise RuntimeError(f"System2 STOP-head shape mismatches: {mismatched[:5]}")
    head.load_state_dict(head_state, strict=True)
    # Match training: retain FP32 classifier parameters and computation even
    # though the frozen Qwen hidden state is produced in BF16.
    head.to(device=device, dtype=torch.float32)
    head.requires_grad_(False)
    head.eval()
    legacy_threshold = float(head_config.get("inference_threshold", 0.5))
    add_threshold = float(
        head_config.get("add_stop_threshold", legacy_threshold)
        if add_threshold_override is None
        else add_threshold_override
    )
    veto_threshold = float(
        head_config.get("veto_stop_threshold", legacy_threshold)
        if veto_threshold_override is None
        else veto_threshold_override
    )
    if not 0.0 <= veto_threshold < add_threshold <= 1.0:
        raise RuntimeError(
            "Invalid System2 STOP-head hysteresis thresholds: "
            f"veto={veto_threshold} add={add_threshold}; expected "
            "0 <= veto < add <= 1"
        )
    return head, add_threshold, veto_threshold


def _load_system2_temporal_stop_verifier(
    checkpoint_path: str,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, float | None]:
    """Load the embedded frozen static prior and veto-only temporal policy."""
    from src.models.action import StopPredictionHead
    from src.models.action.temporal_stop_verifier import (
        TEMPORAL_STOP_FEATURE_NAMES,
        TEMPORAL_STOP_FEATURE_SCHEMA,
        TemporalStopVerifier,
        TemporalStopVerifierEnsemble,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Invalid temporal STOP checkpoint: {checkpoint_path}")
    stage_name = checkpoint.get("stage_name")
    supported_stages = {
        "system2_temporal_stop_verifier",
        "system2_temporal_stop_verifier_ensemble",
    }
    if stage_name not in supported_stages:
        raise RuntimeError(
            "Temporal STOP checkpoint has the wrong stage_name: "
            f"{stage_name!r}"
        )
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("Temporal STOP checkpoint has no config")
    verifier_config = config.get("temporal_stop_verifier")
    static_spec = config.get("source_static_stop_head")
    if not isinstance(verifier_config, dict) or not isinstance(static_spec, dict):
        raise RuntimeError("Temporal STOP checkpoint config is incomplete")
    if verifier_config.get("schema") != TEMPORAL_STOP_FEATURE_SCHEMA:
        raise RuntimeError("Temporal STOP feature schema mismatch")
    if tuple(verifier_config.get("feature_names") or ()) != TEMPORAL_STOP_FEATURE_NAMES:
        raise RuntimeError("Temporal STOP feature names do not match runtime code")
    if verifier_config.get("veto_only") is not True:
        raise RuntimeError("Temporal STOP checkpoint must be veto-only")
    if verifier_config.get("requires_contiguous_zero_based_calls") is not True:
        raise RuntimeError("Temporal STOP checkpoint lacks the required history contract")
    static_head = StopPredictionHead(
        input_dim=int(static_spec["input_dim"]),
        hidden_dim=int(static_spec["hidden_dim"]),
        dropout=float(static_spec["dropout"]),
        focal_gamma=float(static_spec["focal_gamma"]),
        focal_alpha=float(static_spec["focal_alpha"]),
        pos_weight=float(static_spec["pos_weight"]),
        bce_mix=float(static_spec["bce_mix"]),
    )
    raw_static_state = checkpoint.get("source_static_stop_head_state_dict")
    if not isinstance(raw_static_state, dict):
        raise RuntimeError("Temporal STOP checkpoint lacks its frozen static prior")
    static_state = {
        _normalize_state_key(name).removeprefix("stop_head."): value
        for name, value in raw_static_state.items()
        if _normalize_state_key(name).startswith("stop_head.")
    }
    _assert_finite_state_dict(static_state, "Temporal STOP frozen static prior")
    if set(static_state) != set(static_head.state_dict()):
        raise RuntimeError(
            "Temporal STOP frozen static prior is incomplete: "
            f"found={len(static_state)} expected={len(static_head.state_dict())}"
        )
    mismatched_static = [
        name
        for name, value in static_head.state_dict().items()
        if tuple(static_state[name].shape) != tuple(value.shape)
    ]
    if mismatched_static:
        raise RuntimeError(
            f"Temporal STOP static-prior shape mismatch: {mismatched_static[:5]}"
        )
    static_head.load_state_dict(static_state, strict=True)

    raw_verifier_state = checkpoint.get("trainable_state_dict")
    if not isinstance(raw_verifier_state, dict):
        raise RuntimeError("Temporal STOP checkpoint lacks verifier tensors")
    state_prefix = (
        "temporal_stop_ensemble."
        if stage_name == "system2_temporal_stop_verifier_ensemble"
        else "temporal_stop_verifier."
    )
    normalized_verifier_state = {
        _normalize_state_key(name): value for name, value in raw_verifier_state.items()
    }
    unexpected_verifier = sorted(
        name for name in normalized_verifier_state if not name.startswith(state_prefix)
    )
    if unexpected_verifier:
        raise RuntimeError(
            "Temporal STOP checkpoint contains unexpected verifier tensors: "
            f"{unexpected_verifier[:5]}"
        )
    verifier_state = {
        name.removeprefix(state_prefix): value
        for name, value in normalized_verifier_state.items()
        if name.startswith(state_prefix)
    }
    _assert_finite_state_dict(verifier_state, "Temporal STOP verifier")
    threshold: float | None
    if stage_name == "system2_temporal_stop_verifier":
        threshold = float(verifier_config.get("acceptance_threshold", float("nan")))
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise RuntimeError(
                f"Invalid temporal STOP acceptance threshold: {threshold}"
            )
        feature_mean = verifier_state.get("feature_mean")
        feature_scale = verifier_state.get("feature_scale")
        if not torch.is_tensor(feature_mean) or not torch.is_tensor(feature_scale):
            raise RuntimeError("Temporal STOP checkpoint lacks normalization tensors")
        verifier: torch.nn.Module = TemporalStopVerifier(
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            hidden_dim=int(verifier_config["hidden_dim"]),
            dropout=float(verifier_config["dropout"]),
        )
    else:
        threshold = None
        if verifier_config.get("architecture") != "scene_fold_unanimous_ensemble":
            raise RuntimeError("Temporal STOP ensemble architecture is unsupported")
        if verifier_config.get("aggregation") != "unanimous":
            raise RuntimeError("Temporal STOP ensemble must use unanimous aggregation")
        ensemble_size = int(verifier_config.get("ensemble_size", 0) or 0)
        if ensemble_size < 2:
            raise RuntimeError("Temporal STOP ensemble must contain at least two members")
        raw_thresholds = verifier_config.get("acceptance_thresholds")
        if not isinstance(raw_thresholds, list) or len(raw_thresholds) != ensemble_size:
            raise RuntimeError(
                "Temporal STOP ensemble thresholds do not match member count"
            )
        thresholds = torch.tensor(raw_thresholds, dtype=torch.float32)
        if not bool(torch.isfinite(thresholds).all()) or bool(
            ((thresholds < 0.0) | (thresholds > 1.0)).any()
        ):
            raise RuntimeError("Temporal STOP ensemble thresholds must be in [0, 1]")
        members = []
        for member_index in range(ensemble_size):
            member_prefix = f"members.{member_index}."
            feature_mean = verifier_state.get(f"{member_prefix}feature_mean")
            feature_scale = verifier_state.get(f"{member_prefix}feature_scale")
            if not torch.is_tensor(feature_mean) or not torch.is_tensor(feature_scale):
                raise RuntimeError(
                    "Temporal STOP ensemble lacks normalization tensors for "
                    f"member {member_index}"
                )
            members.append(
                TemporalStopVerifier(
                    feature_mean=feature_mean,
                    feature_scale=feature_scale,
                    hidden_dim=int(verifier_config["member_hidden_dim"]),
                    dropout=float(verifier_config["member_dropout"]),
                )
            )
        verifier = TemporalStopVerifierEnsemble(members, thresholds)
    if set(verifier_state) != set(verifier.state_dict()):
        raise RuntimeError(
            "Temporal STOP verifier is incomplete: "
            f"found={len(verifier_state)} expected={len(verifier.state_dict())}"
        )
    mismatched_verifier = [
        name
        for name, value in verifier.state_dict().items()
        if tuple(verifier_state[name].shape) != tuple(value.shape)
    ]
    if mismatched_verifier:
        raise RuntimeError(
            f"Temporal STOP verifier shape mismatch: {mismatched_verifier[:5]}"
        )
    verifier.load_state_dict(verifier_state, strict=True)
    if isinstance(verifier, TemporalStopVerifierEnsemble) and not torch.allclose(
        verifier.acceptance_thresholds.cpu(), thresholds, rtol=0.0, atol=1e-7
    ):
        raise RuntimeError(
            "Temporal STOP ensemble state thresholds do not match checkpoint config"
        )
    for module in (static_head, verifier):
        module.to(device=device, dtype=torch.float32)
        module.requires_grad_(False)
        module.eval()
    return static_head, verifier, threshold


def _load_compatible_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    checkpoint_path: str,
    label: str,
) -> int:
    current_state = model.state_dict()
    normalized_to_actual = {_normalize_state_key(name): name for name in current_state}
    remapped: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    skipped_missing: list[str] = []
    for name, value in state_dict.items():
        normalized_name = _normalize_state_key(name)
        actual_name = normalized_to_actual.get(normalized_name)
        if actual_name is None:
            skipped_missing.append(name)
            continue
        if current_state[actual_name].shape != value.shape:
            skipped_shape.append(
                f"{actual_name}: ckpt {tuple(value.shape)} vs model {tuple(current_state[actual_name].shape)}"
            )
            continue
        remapped[actual_name] = value
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    LOGGER.info(
        "%s loaded: %s (loaded=%d/%d, missing=%d, unexpected=%d)",
        label,
        checkpoint_path,
        len(remapped),
        len(state_dict),
        len(missing),
        len(unexpected),
    )
    if skipped_missing:
        LOGGER.info("Skipped unmatched keys: %d; examples: %s", len(skipped_missing), skipped_missing[:5])
    if skipped_shape:
        LOGGER.info("Skipped shape-mismatched keys: %d; examples: %s", len(skipped_shape), skipped_shape[:3])
    return len(remapped)


def _load_pano_latent_adapter(
    checkpoint_path: str,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
):
    from scripts.evaluation.eval_pano_latent_adapter import _load_adapter_from_checkpoint

    fallback = argparse.Namespace(
        adapter_hidden_dim=2048,
        adapter_dropout=0.0,
        residual=False,
        pre_norm=False,
    )
    adapter, _saved_args = _load_adapter_from_checkpoint(
        Path(checkpoint_path).expanduser(),
        dim=hidden_dim,
        fallback_args=fallback,
        device=device,
        dtype=dtype,
    )
    return adapter


def _maybe_apply_pano_latent_adapter(
    traj_hs: torch.Tensor,
    adapter,
    *,
    view_id: str | None = None,
    pixel_goal: list[int] | None = None,
    image_size: tuple[int, int] | None = None,
    cond_projector: torch.nn.Module | None = None,
) -> torch.Tensor:
    if adapter is None:
        return traj_hs
    orig_dtype = traj_hs.dtype
    adapter_param = next(adapter.parameters(), None)
    adapter_dtype = adapter_param.dtype if adapter_param is not None else orig_dtype

    if hasattr(adapter, "geometry_token"):
        from src.models.adapters import view_ids_to_indices

        if pixel_goal is None:
            raise RuntimeError("Geometry-aware pano adapter requires pixel_goal")
        goal_view = (view_id or "front").lower()
        if goal_view not in {"front", "right", "back", "left"}:
            goal_view = "front"
        if image_size is None:
            image_size = (384, 384)
        width, height = int(image_size[0]), int(image_size[1])
        view_indices = view_ids_to_indices([goal_view], device=traj_hs.device)
        pixel_xy = torch.tensor(
            [[int(pixel_goal[0]), int(pixel_goal[1])]],
            device=traj_hs.device,
            dtype=adapter_dtype,
        )
        image_hw = torch.tensor([[height, width]], device=traj_hs.device, dtype=adapter_dtype)
        out = adapter(traj_hs.to(dtype=adapter_dtype), view_indices, pixel_xy, image_hw)
        return out.to(dtype=orig_dtype)

    if hasattr(adapter, "mlp"):
        adapted = adapter(traj_hs.to(dtype=adapter_dtype))
        if cond_projector is not None:
            proj_dtype = next(cond_projector.parameters()).dtype
            adapted = cond_projector(adapted.to(dtype=proj_dtype))
        return adapted.to(dtype=orig_dtype)

    out = adapter(traj_hs.to(dtype=adapter_dtype))
    return out.to(dtype=orig_dtype)


def _normalize_multimodal_inputs(inputs: dict[str, torch.Tensor]) -> None:
    if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None:
        vgt = inputs["video_grid_thw"]
        if vgt.shape[0] > 0 and vgt[:, 0].max() > 1:
            inputs["video_grid_thw"] = torch.repeat_interleave(vgt, vgt[:, 0], dim=0)
            inputs["video_grid_thw"][:, 0] = 1


def _parse_pano_view_id(llm_output: str) -> str | None:
    parsed = parse_structured_pano_output(llm_output, image_size=None)
    return parsed.view_id


def _parse_pixel_goal(
    llm_output: str,
    image_size: tuple[int, int],
    *,
    allow_legacy_coord: bool = True,
) -> list[int] | None:
    parsed = parse_structured_pano_output(
        llm_output,
        image_size=image_size,
        allow_legacy_coord=allow_legacy_coord,
    )
    if parsed.kind in {"pixel", "legacy_coord"} and parsed.pixel_goal is not None:
        return list(parsed.pixel_goal)
    if not re.search(r"\d", llm_output or ""):
        return None
    return None


def _fallback_replan_action(view_id: str | None) -> int:
    """Turn toward a valid view after malformed waypoint text; never infer STOP."""
    if str(view_id or "").lower() in {"right", "back"}:
        return ActionCode.RIGHT
    return ActionCode.LEFT


def _condition_output_ids_for_pixel_goal(
    output_ids: torch.Tensor,
    prompt_len: int,
    tokenizer,
    pixel_goal: list[int],
    llm_output: str,
    coord_order: str = "generated",
    view_id: str | None = None,
    structured_output: bool = False,
) -> torch.Tensor:
    parsed = parse_structured_pano_output(llm_output, image_size=None)
    use_structured = structured_output or parsed.kind == "pixel"
    if use_structured:
        resolved_view = (view_id or parsed.view_id or "front").lower()
        desired_text = structured_condition_text(resolved_view, pixel_goal)
        generated_text = (llm_output or "").strip()
        if generated_text == desired_text:
            return output_ids
        replacement = tokenizer.encode(desired_text, add_special_tokens=False)
    else:
        coord = [int(c) for c in re.findall(r"\d+", llm_output or "")]
        if coord_order == "generated":
            desired = [int(pixel_goal[0]), int(pixel_goal[1])]
        elif coord_order == "internnav_yx":
            desired = [int(pixel_goal[1]), int(pixel_goal[0])]
        else:
            raise ValueError(f"Unsupported coord_order: {coord_order}")
        if len(coord) >= 2 and [coord[0], coord[1]] == desired:
            return output_ids
        replacement = tokenizer.encode(f"{desired[0]} {desired[1]}", add_special_tokens=False)

    if not replacement:
        return output_ids
    replacement_ids = torch.tensor([replacement], device=output_ids.device, dtype=output_ids.dtype)
    generated_suffix = output_ids[:, prompt_len:]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if (
        eos_token_id is not None
        and generated_suffix.numel() > 0
        and int(generated_suffix[0, -1].item()) == int(eos_token_id)
    ):
        eos = torch.tensor([[eos_token_id]], device=output_ids.device, dtype=output_ids.dtype)
        replacement_ids = torch.cat([replacement_ids, eos], dim=1)
    return torch.cat([output_ids[:, :prompt_len], replacement_ids], dim=1)


def _trajectory_from_condition(
    action_head,
    traj_condition: torch.Tensor,
    *,
    traj_images: torch.Tensor | None,
    generator: torch.Generator | None = None,
):
    if traj_condition.shape[-1] == int(action_head.config.latent_emb_size):
        return action_head.get_trajectory_from_projected(
            traj_condition,
            traj_images=traj_images,
            generator=generator,
        )
    return action_head.get_trajectory(
        traj_condition,
        traj_images=traj_images,
        generator=generator,
    )


def _project_trajectory_condition(action_head, traj_condition: torch.Tensor) -> torch.Tensor:
    """Return the exact 768-D InternNav condition without projecting it twice."""
    if traj_condition.shape[-1] == int(action_head.config.latent_emb_size):
        return traj_condition
    return action_head.cond_projector(traj_condition)


def _lookdown_to_traj_tensor(lookdown_img: Image.Image, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.array(lookdown_img)).to(device=device, dtype=torch.bfloat16) / 255.0


def _finalize_local_actions(action_list: list[int]) -> list[int]:
    if len(action_list) < MAX_STEPS:
        action_list = list(action_list) + [ActionCode.STOP] * (MAX_STEPS - len(action_list))
    if len(action_list) >= MAX_LOCAL_STEPS:
        action_list = action_list[:MAX_LOCAL_STEPS]
    return [int(action) for action in action_list]


def reconstruct_xy_from_delta(delta_xyt: np.ndarray) -> np.ndarray:
    start_xy = np.zeros((len(delta_xyt), 2))
    delta_xy = delta_xyt[:, :, :2]
    cumsum_xy = np.cumsum(delta_xy, axis=1)
    batch_size = delta_xyt.shape[0]
    steps = delta_xyt.shape[1]
    xy = np.zeros((batch_size, steps + 1, 2))
    xy[:, 0] = start_xy
    xy[:, 1:] = start_xy[:, None, :] + cumsum_xy
    return xy


def trajectory_xy_path_len(trajectory: np.ndarray) -> float:
    if trajectory.ndim != 2 or trajectory.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1).sum())


def _trajectory_to_discrete_actions_close_to_goal(
    trajectory: np.ndarray,
    step_size: float = 0.25,
    turn_angle_deg: float = 15,
    lookahead: int = 4,
) -> list[int]:
    actions: list[int] = []
    yaw = 0.0
    pos = trajectory[0]
    turn_angle_rad = np.deg2rad(turn_angle_deg)
    goal = trajectory[-1]

    def normalize_angle(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    while np.linalg.norm(pos - goal) > 0.2:
        dists = np.linalg.norm(trajectory - pos, axis=1)
        nearest_idx = np.argmin(dists)
        target_idx = min(nearest_idx + lookahead, len(trajectory) - 1)
        target = trajectory[target_idx]
        target_dir = target - pos
        if np.linalg.norm(target_dir) < 1e-6:
            break
        target_yaw = np.arctan2(target_dir[1], target_dir[0])
        delta_yaw = normalize_angle(target_yaw - yaw)
        n_turns = round(delta_yaw / turn_angle_rad)
        if n_turns > 0:
            actions += [ActionCode.LEFT] * n_turns
        elif n_turns < 0:
            actions += [ActionCode.RIGHT] * (-n_turns)
        yaw = normalize_angle(yaw + n_turns * turn_angle_rad)
        next_pos = pos + step_size * np.array([np.cos(yaw), np.sin(yaw)])
        if np.linalg.norm(next_pos - goal) > np.linalg.norm(pos - goal):
            break
        actions.append(ActionCode.FORWARD)
        pos = next_pos
    return actions


def _endpoint_medoid_index(all_trajectory: np.ndarray) -> int:
    endpoints = all_trajectory[:, -1, :2]
    dists = np.linalg.norm(endpoints[:, None, :] - endpoints[None, :, :], axis=-1)
    return int(np.argmin(dists.sum(axis=1)))


def _path_medoid_index(all_trajectory: np.ndarray) -> int:
    flat = all_trajectory[:, :, :2].reshape(all_trajectory.shape[0], -1)
    dists = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    return int(np.argmin(dists.sum(axis=1)))


def _median_endpoint_nearest_index(all_trajectory: np.ndarray) -> int:
    endpoints = all_trajectory[:, -1, :2]
    median_endpoint = np.median(endpoints, axis=0)
    return int(np.argmin(np.linalg.norm(endpoints - median_endpoint[None, :], axis=-1)))


def _forward_candidate_stats(all_trajectory: np.ndarray) -> list[tuple[int, int, float, list[int]]]:
    candidates: list[tuple[int, int, float, list[int]]] = []
    for idx, trajectory in enumerate(all_trajectory):
        actions = _trajectory_to_discrete_actions_close_to_goal(trajectory)
        forward_count = sum(1 for action in actions if action == ActionCode.FORWARD)
        if forward_count <= 0:
            continue
        candidates.append((idx, forward_count, trajectory_xy_path_len(trajectory), actions))
    return candidates


def select_trajectory_xy(all_trajectory: np.ndarray, selection: str = "mean") -> tuple[np.ndarray, int | None]:
    if all_trajectory.ndim != 3 or all_trajectory.shape[0] == 0:
        raise ValueError(f"Expected all_trajectory shape (B,T,2), got {all_trajectory.shape}")
    if selection == "mean":
        return np.mean(all_trajectory, axis=0), None
    if selection == "endpoint_medoid":
        idx = _endpoint_medoid_index(all_trajectory)
        return all_trajectory[idx], idx
    if selection == "path_medoid":
        idx = _path_medoid_index(all_trajectory)
        return all_trajectory[idx], idx
    if selection == "median_endpoint_nearest":
        idx = _median_endpoint_nearest_index(all_trajectory)
        return all_trajectory[idx], idx

    forward_candidates = _forward_candidate_stats(all_trajectory)
    if selection == "forward_or_medoid":
        if forward_candidates:
            medoid_idx = _endpoint_medoid_index(all_trajectory)
            medoid_endpoint = all_trajectory[medoid_idx, -1, :2]
            median_path_len = float(np.median([trajectory_xy_path_len(traj) for traj in all_trajectory]))

            def score(item: tuple[int, int, float, list[int]]) -> tuple[float, int, float]:
                idx, forward_count, path_len, _actions = item
                endpoint_dist = float(np.linalg.norm(all_trajectory[idx, -1, :2] - medoid_endpoint))
                return (endpoint_dist, -forward_count, abs(path_len - median_path_len))

            idx = min(forward_candidates, key=score)[0]
            return all_trajectory[idx], idx
        idx = _endpoint_medoid_index(all_trajectory)
        return all_trajectory[idx], idx
    if selection == "longest_forward":
        if forward_candidates:
            idx = max(forward_candidates, key=lambda item: (item[2], item[1]))[0]
            return all_trajectory[idx], idx
        idx = _endpoint_medoid_index(all_trajectory)
        return all_trajectory[idx], idx
    raise ValueError(f"Unsupported trajectory selection: {selection}")


def traj_to_actions(
    dp_actions: torch.Tensor,
    num_sample_trajs: int = 32,
    action_scale: float = 4.0,
    trajectory_selection: str = "mean",
    trajectory_x_sign: float = 1.0,
    target_heading_deg: float | None = None,
) -> list[int]:
    if trajectory_x_sign not in (-1.0, 1.0):
        raise ValueError(f"trajectory_x_sign must be -1 or 1, got {trajectory_x_sign}")
    trajs = dp_actions[:num_sample_trajs].float().detach().cpu().numpy().copy()
    trajs[:, :, :2] /= action_scale
    trajs[:, :, 0] *= trajectory_x_sign
    all_trajectory = reconstruct_xy_from_delta(trajs)
    trajectory, _selected_idx = select_trajectory_xy(all_trajectory, trajectory_selection)
    if target_heading_deg is not None:
        trajectory, _rotation_deg = align_trajectory_endpoint_heading(
            trajectory,
            target_angle_deg=float(target_heading_deg),
        )
    actions = _trajectory_to_discrete_actions_close_to_goal(trajectory)
    return actions if actions else [ActionCode.STOP]


def _trajectory_debug_metrics(
    trajectory: torch.Tensor,
    num_sample_trajs: int,
    action_scale: float,
    trajectory_x_sign: float = 1.0,
) -> dict[str, float] | None:
    if trajectory is None or trajectory.numel() == 0:
        return None
    trajs = trajectory[:num_sample_trajs].float().detach().cpu().numpy().copy()
    if trajs.ndim != 3 or trajs.shape[-1] < 2:
        return None
    trajs[:, :, :2] /= float(action_scale)
    trajs[:, :, 0] *= float(trajectory_x_sign)
    cumsum_xy = np.cumsum(trajs[:, :, :2], axis=1)
    xy = np.concatenate([np.zeros((trajs.shape[0], 1, 2), dtype=cumsum_xy.dtype), cumsum_xy], axis=1)
    mean_xy = xy.mean(axis=0)
    goal_xy = mean_xy[-1]
    direct = float(np.linalg.norm(goal_xy))
    path_len = float(np.linalg.norm(np.diff(mean_xy, axis=0), axis=1).sum())
    return {
        "goal_x_m": float(goal_xy[0]),
        "goal_y_m": float(goal_xy[1]),
        "direct_m": direct,
        "path_len_m": path_len,
    }


def _trajectory_debug_summary(
    trajectory: torch.Tensor,
    num_sample_trajs: int,
    action_scale: float,
    trajectory_x_sign: float = 1.0,
) -> str:
    if trajectory is None or trajectory.numel() == 0:
        return "trajectory=empty"
    metrics = _trajectory_debug_metrics(
        trajectory,
        num_sample_trajs,
        action_scale,
        trajectory_x_sign,
    )
    if metrics is None:
        return f"trajectory_shape={tuple(trajectory.shape)}"
    return (
        f"traj_goal=({metrics['goal_x_m']:.2f},{metrics['goal_y_m']:.2f}), "
        f"direct={metrics['direct_m']:.2f}, path_len={metrics['path_len_m']:.2f}"
    )


class _StructuredViewPrefixLogitsProcessor(LogitsProcessor):
    """Constrain only the three-token ``view: <class>`` protocol prefix."""

    _LABELS = ("stop", "front", "left", "right", "back", "turn")

    def __init__(
        self,
        *,
        tokenizer,
        prompt_len: int,
        excluded_labels: Sequence[str] = (),
    ) -> None:
        excluded = {str(label) for label in excluded_labels}
        unknown = excluded.difference(self._LABELS)
        if unknown:
            raise ValueError(f"Unknown structured view labels to exclude: {sorted(unknown)}")
        labels = tuple(label for label in self._LABELS if label not in excluded)
        if not labels:
            raise ValueError("At least one structured view label must remain enabled")
        patterns = [
            tokenizer.encode(f"view: {label}", add_special_tokens=False)
            for label in labels
        ]
        if any(len(pattern) != 3 for pattern in patterns):
            raise RuntimeError(
                "Structured System2 output requires three-token view decisions"
            )
        prefixes = {tuple(pattern[:2]) for pattern in patterns}
        class_tokens = {int(pattern[2]) for pattern in patterns}
        if len(prefixes) != 1 or len(class_tokens) != len(labels):
            raise RuntimeError(
                "Structured System2 view classes do not share a unique prefix"
            )
        self.prompt_len = int(prompt_len)
        self.prefix = tuple(next(iter(prefixes)))
        self.class_tokens = tuple(sorted(class_tokens))

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        generated_tokens = int(input_ids.shape[-1]) - self.prompt_len
        if generated_tokens == 0:
            allowed = (self.prefix[0],)
        elif generated_tokens == 1:
            allowed = (self.prefix[1],)
        elif generated_tokens == 2:
            allowed = self.class_tokens
        else:
            return scores

        allowed_ids = torch.tensor(allowed, device=scores.device, dtype=torch.long)
        constrained = torch.full_like(scores, -torch.inf)
        constrained.index_copy_(
            1,
            allowed_ids,
            scores.index_select(1, allowed_ids),
        )
        return constrained


def _system2_non_stop_generation_kwargs(
    generation_kwargs: dict[str, Any],
    *,
    tokenizer: Any,
    prompt_len: int,
) -> dict[str, Any]:
    """Build a structured generation request that cannot select the STOP class."""
    constrained_kwargs = dict(generation_kwargs)
    constrained_kwargs["logits_processor"] = LogitsProcessorList(
        [
            _StructuredViewPrefixLogitsProcessor(
                tokenizer=tokenizer,
                prompt_len=prompt_len,
                excluded_labels=("stop",),
            )
        ]
    )
    constrained_kwargs["bad_words_ids"] = _system2_stop_bad_words_ids(tokenizer)
    return constrained_kwargs


def _system2_non_stop_output_or_fallback(
    output: str,
    *,
    system2_call_index: int,
) -> tuple[str, bool]:
    """Return a valid non-STOP decision, falling back without privileged state."""
    parsed = parse_structured_pano_output(output, image_size=None)
    if parsed.kind in {"pixel", "turn"} and not vlm_output_requests_stop(output):
        return output, False
    direction = "left" if int(system2_call_index) % 2 == 0 else "right"
    return f"view: turn_{direction}", True


def _system2_decision_scores(
    *,
    tokenizer,
    sequence: torch.Tensor,
    prompt_len: int,
    generation_scores: Sequence[torch.Tensor],
) -> dict[str, Any]:
    """Score the structured System2 view decision at its first class token.

    Probabilities are normalized over the six valid structured classes, not
    over the full vocabulary. They are diagnostics, not privileged signals.
    """
    labels = ("stop", "front", "left", "right", "back", "turn")
    token_ids: dict[str, int] = {}
    prefix: list[int] | None = None
    for label in labels:
        encoded = tokenizer.encode(f"view: {label}", add_special_tokens=False)
        if len(encoded) != 3:
            return {}
        if prefix is None:
            prefix = encoded[:2]
        elif encoded[:2] != prefix:
            return {}
        token_ids[label] = int(encoded[2])

    generated = sequence[0, prompt_len:].detach().cpu().tolist()
    decision_index = 2
    if (
        prefix is None
        or generated[:2] != prefix
        or len(generated) <= decision_index
        or int(generated[decision_index]) not in token_ids.values()
        or decision_index >= len(generation_scores)
    ):
        return {}

    logits = generation_scores[decision_index][0].float()
    class_logits = torch.stack([logits[token_ids[label]] for label in labels])
    probabilities = torch.softmax(class_logits, dim=0).detach().cpu().tolist()
    selected_token = int(generated[decision_index])
    selected_label = next(
        (label for label, token_id in token_ids.items() if token_id == selected_token),
        "unknown",
    )
    stop_logit = class_logits[0]
    non_stop_logsumexp = torch.logsumexp(class_logits[1:], dim=0)
    return {
        "selected": selected_label,
        "class_probabilities": {
            label: float(probability)
            for label, probability in zip(labels, probabilities)
        },
        "stop_log_odds": float((stop_logit - non_stop_logsumexp).item()),
    }


def _system2_generation_decision_hidden(
    *,
    generation,
    tokenizer,
    prompt_len: int,
) -> torch.Tensor:
    """Return the causal hidden state that predicts the view-class token."""
    labels = ("stop", "front", "left", "right", "back", "turn")
    patterns = [
        tokenizer.encode(f"view: {label}", add_special_tokens=False)
        for label in labels
    ]
    if any(len(pattern) != 3 for pattern in patterns):
        raise RuntimeError("STOP head requires three-token structured view decisions")
    prefixes = {tuple(pattern[:2]) for pattern in patterns}
    class_tokens = {int(pattern[2]) for pattern in patterns}
    if len(prefixes) != 1 or len(class_tokens) != len(labels):
        raise RuntimeError("Structured view classes do not share a unique two-token prefix")
    generated = generation.sequences[0, prompt_len:].detach().cpu().tolist()
    prefix = next(iter(prefixes))
    if generated[:2] != list(prefix) or len(generated) < 3 or int(generated[2]) not in class_tokens:
        raise RuntimeError(
            "System2 generation did not emit the expected structured view prefix: "
            f"generated_ids={generated[:5]}"
        )
    hidden_steps = getattr(generation, "hidden_states", None)
    if hidden_steps is None or len(hidden_steps) <= 2:
        raise RuntimeError("System2 generation did not return decision-step hidden states")
    decision_step = hidden_steps[2]
    if isinstance(decision_step, (tuple, list)):
        if not decision_step:
            raise RuntimeError("System2 decision hidden-state tuple is empty")
        decision_step = decision_step[-1]
    if not torch.is_tensor(decision_step) or decision_step.ndim != 3:
        raise RuntimeError(
            "Unexpected System2 decision hidden-state shape: "
            f"{getattr(decision_step, 'shape', None)}"
        )
    return decision_step[:, -1, :].detach()


def _system2_stop_hidden_alignment(
    generated_hidden: torch.Tensor,
    teacher_forced_hidden: torch.Tensor,
) -> dict[str, float]:
    """Measure whether training and cached generation expose the same state."""
    if generated_hidden.shape != teacher_forced_hidden.shape:
        raise RuntimeError(
            "System2 STOP hidden-state shape mismatch: "
            f"generated={tuple(generated_hidden.shape)} "
            f"teacher_forced={tuple(teacher_forced_hidden.shape)}"
        )
    generated = generated_hidden.detach().float()
    teacher_forced = teacher_forced_hidden.detach().float()
    delta = generated - teacher_forced
    cosine = torch.nn.functional.cosine_similarity(
        generated,
        teacher_forced,
        dim=-1,
    )
    return {
        "cosine_min": float(cosine.min().item()),
        "cosine_mean": float(cosine.mean().item()),
        "max_abs_error": float(delta.abs().max().item()),
        "mean_abs_error": float(delta.abs().mean().item()),
        "generated_norm_mean": float(generated.norm(dim=-1).mean().item()),
        "teacher_forced_norm_mean": float(
            teacher_forced.norm(dim=-1).mean().item()
        ),
    }


def _system2_teacher_forced_decision_hidden(
    *,
    qwen_integration: Any,
    inputs: dict[str, torch.Tensor],
    output_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Recompute the decision state through the rollout-training forward path."""
    structured_prefix = output_ids[:, prompt_len : prompt_len + 2]
    teacher_inputs = dict(inputs)
    teacher_inputs["input_ids"] = torch.cat(
        [inputs["input_ids"], structured_prefix],
        dim=1,
    )
    for mask_name in ("attention_mask", "mm_token_type_ids"):
        mask = inputs.get(mask_name)
        if mask is None:
            continue
        fill_value = 1 if mask_name == "attention_mask" else 0
        suffix = torch.full(
            (mask.shape[0], structured_prefix.shape[1]),
            fill_value,
            device=mask.device,
            dtype=mask.dtype,
        )
        teacher_inputs[mask_name] = torch.cat([mask, suffix], dim=1)
    teacher_inputs.pop("position_ids", None)
    with torch.inference_mode():
        (
            teacher_sequence_hidden,
            _teacher_vision_hidden,
            _teacher_num_image_tokens,
            _teacher_traj_hidden,
            _teacher_lm_output,
        ) = qwen_integration._forward_model_inputs(
            teacher_inputs,
            return_hidden_states=True,
            skip_lm_head=True,
            return_last_hidden_state_only=True,
            extract_vision_hidden_states=False,
        )
    if teacher_sequence_hidden is None:
        raise RuntimeError(
            "System2 STOP alignment check returned no teacher-forced hidden state"
        )
    return teacher_sequence_hidden[:, -1, :]


def _system2_stop_head_decision(
    *,
    stop_probability: float,
    add_stop_threshold: float,
    veto_stop_threshold: float,
    original_output: str,
    original_stop_probability: float | None = None,
    add_min_qwen_stop_probability: float = 0.0,
    constrained_output: str | None = None,
    image_size: tuple[int, int] = (256, 256),
    allow_add_stop: bool = True,
) -> str:
    """Describe the STOP-head decision without changing the waypoint prior."""
    original_requests_stop = vlm_output_requests_stop(original_output)
    if not original_requests_stop:
        if not allow_add_stop:
            return "head_keeps_original_non_stop"
        if stop_probability >= add_stop_threshold:
            if (
                add_min_qwen_stop_probability > 0.0
                and (
                    original_stop_probability is None
                    or original_stop_probability < add_min_qwen_stop_probability
                )
            ):
                return "head_rejects_uncorroborated_stop"
            return "head_adds_stop"
        return "head_keeps_original_non_stop"
    if stop_probability >= veto_stop_threshold:
        return "head_confirms_original_stop"
    if constrained_output is None:
        return "head_requests_stop_veto"
    constrained_turn = vlm_output_requests_turn(constrained_output)
    constrained_pixel = _parse_pixel_goal(
        constrained_output,
        image_size,
        allow_legacy_coord=True,
    )
    if constrained_turn is not None or constrained_pixel is not None:
        return "head_vetoes_stop"
    return "head_veto_fallback_replan"


def _system2_temporal_stop_decision(
    *,
    verifier_probability: float,
    acceptance_threshold: float,
    original_output: str,
) -> str:
    """Return a veto-only decision; original non-STOP output is immutable."""
    if not (
        np.isfinite(verifier_probability)
        and np.isfinite(acceptance_threshold)
        and 0.0 <= verifier_probability <= 1.0
        and 0.0 <= acceptance_threshold <= 1.0
    ):
        raise ValueError("Temporal STOP probabilities and thresholds must be in [0, 1]")
    if not vlm_output_requests_stop(original_output):
        return "temporal_keeps_original_non_stop"
    if verifier_probability >= acceptance_threshold:
        return "temporal_confirms_original_stop"
    return "temporal_requests_stop_veto"


def _validate_system2_stop_threshold_overrides(
    *,
    static_head_enabled: bool,
    temporal_verifier_enabled: bool,
    add_threshold_override: float | None,
    veto_threshold_override: float | None,
) -> None:
    """Validate ownership of STOP thresholds before loading the model."""
    if not temporal_verifier_enabled:
        return
    if veto_threshold_override is not None:
        raise ValueError(
            "system2_stop_veto_threshold cannot be overridden when the temporal "
            "STOP verifier is enabled; the temporal verifier owns veto decisions"
        )
    if add_threshold_override is not None and not static_head_enabled:
        raise ValueError(
            "system2_stop_add_threshold requires a static STOP head when the "
            "temporal STOP verifier is enabled"
        )


def _system2_hybrid_stop_decision(
    *,
    original_output: str,
    temporal_decision: str,
    static_add_decision: str,
) -> str:
    """Keep hybrid policy roles disjoint: temporal vetoes, static only adds."""
    if vlm_output_requests_stop(original_output):
        allowed = {
            "temporal_confirms_original_stop",
            "temporal_requests_stop_veto",
            "temporal_vetoes_original_stop",
        }
        if temporal_decision not in allowed:
            raise ValueError(
                f"Invalid temporal decision for original STOP: {temporal_decision!r}"
            )
        return temporal_decision
    if temporal_decision != "temporal_keeps_original_non_stop":
        raise ValueError(
            f"Temporal policy modified an original non-STOP: {temporal_decision!r}"
        )
    if static_add_decision == "head_adds_stop":
        return "hybrid_static_adds_stop"
    if static_add_decision in {
        "head_keeps_original_non_stop",
        "head_rejects_uncorroborated_stop",
    }:
        return "hybrid_keeps_original_non_stop"
    raise ValueError(
        f"Invalid static add decision for original non-STOP: {static_add_decision!r}"
    )


def _system2_stop_bad_words_ids(tokenizer: Any) -> list[list[int]]:
    """Return the structured STOP class token as a generation constraint."""
    stop_pattern = tokenizer.encode("view: stop", add_special_tokens=False)
    if len(stop_pattern) != 3:
        raise RuntimeError(
            "Cannot constrain System2 STOP token: unexpected tokenization "
            f"{stop_pattern}"
        )
    return [[int(stop_pattern[2])]]


def _validate_system2_force_non_stop_request(
    *,
    force_non_stop: Any,
    feature_dump_enabled: bool,
    stop_head_enabled: bool,
    oracle_system2_enabled: bool,
) -> bool:
    """Fail closed unless forced continuation is an isolated DAgger request."""
    if not isinstance(force_non_stop, bool):
        raise ValueError("system2_force_non_stop must be a boolean")
    if not force_non_stop:
        return False
    if not feature_dump_enabled:
        raise ValueError(
            "system2_force_non_stop is restricted to STOP feature collection"
        )
    if stop_head_enabled:
        raise ValueError(
            "system2_force_non_stop requires the unmodified original System2 policy"
        )
    if oracle_system2_enabled:
        raise ValueError("system2_force_non_stop cannot be combined with oracle System2")
    return True


def _system2_stop_probability(
    stop_head: torch.nn.Module,
    decision_hidden: torch.Tensor,
) -> float:
    """Read the probability returned by StopPredictionHead exactly once."""
    output = stop_head(decision_hidden)
    if not torch.is_tensor(output) or output.numel() != 1:
        raise RuntimeError(
            "System2 STOP head must return exactly one probability, got "
            f"{getattr(output, 'shape', None)}"
        )
    probability = float(output.detach().float().item())
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise RuntimeError(
            f"System2 STOP head returned invalid probability: {probability}"
        )
    return probability


def _pil_from_blob(blob: vla_pb2.BinaryBlob, image_size: tuple[int, int] | None = None) -> Image.Image:
    arr = decode_jpeg_to_rgb(blob.data)
    image = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    if image_size is not None and image.size != image_size:
        image = image.resize(image_size)
    return image


def _blobs_by_name(blobs) -> dict[str, vla_pb2.BinaryBlob]:
    return {blob.name: blob for blob in blobs}


def _write_system2_stop_feature(
    dump_dir: Path,
    *,
    decision_hidden: torch.Tensor,
    sampling_metadata: dict[str, Any],
    original_output: str,
    decision_scores: dict[str, Any],
) -> dict[str, Any]:
    """Atomically persist one frozen-Qwen decision feature for DAgger data."""
    if decision_hidden.shape[0] != 1 or decision_hidden.ndim != 2:
        raise RuntimeError(
            "System2 STOP feature dump requires hidden shape (1, D), got "
            f"{tuple(decision_hidden.shape)}"
        )
    feature = decision_hidden.detach().float().cpu().contiguous().squeeze(0)
    if not torch.isfinite(feature).all():
        raise RuntimeError("System2 STOP feature contains non-finite values")

    scene_id = str(sampling_metadata["scene_id"])
    episode_id = int(sampling_metadata["episode_id"])
    call_index = int(sampling_metadata["system2_call_index"])
    protocol_seed = int(sampling_metadata["protocol_seed"])
    safe_scene = re.sub(r"[^A-Za-z0-9_.-]+", "_", scene_id).strip("._")
    if not safe_scene:
        raise RuntimeError(f"Invalid scene_id for STOP feature dump: {scene_id!r}")
    collection_root = dump_dir.expanduser().resolve().parent
    collection_namespace = hashlib.sha256(
        str(collection_root).encode("utf-8")
    ).hexdigest()[:12]
    key = (
        f"src{collection_namespace}_{safe_scene}_ep{episode_id:06d}_"
        f"call{call_index:05d}_seed{protocol_seed}"
    )
    dump_dir.mkdir(parents=True, exist_ok=True)
    path = dump_dir / f"{key}.pth"
    temporary = dump_dir / f".{key}.{os.getpid()}.tmp"
    payload = {
        "schema": "heatmapvln-system2-stop-feature-v1",
        "key": key,
        "feature": feature,
        "scene_id": scene_id,
        "episode_id": episode_id,
        "system2_call_index": call_index,
        "protocol_seed": protocol_seed,
        "collection_namespace": collection_namespace,
        "collection_root": str(collection_root),
        "original_output": str(original_output),
        "decision_scores": decision_scores,
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {
        "schema": payload["schema"],
        "key": key,
        "path": str(path),
        "hidden_dim": int(feature.numel()),
        "collection_namespace": collection_namespace,
    }


def _augment_system2_stop_feature_with_trajectory(
    feature_record: dict[str, Any],
    *,
    raw_traj_latent: torch.Tensor,
    adapted_traj_latent: torch.Tensor,
    projected_traj_condition: torch.Tensor,
    trajectory: torch.Tensor,
    trajectory_metrics: dict[str, float],
    local_actions: list[int],
    pixel_goal: tuple[int, int],
    pano_goal_view: str,
) -> dict[str, Any]:
    """Atomically add frozen System2/System1 trajectory features to one cache row."""
    path = Path(str(feature_record.get("path", ""))).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"System2 STOP feature payload is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "heatmapvln-system2-stop-feature-v1"
        or payload.get("key") != feature_record.get("key")
    ):
        raise RuntimeError(f"System2 STOP feature payload metadata mismatch: {path}")

    latent_tensors = {
        "raw_traj_latent": raw_traj_latent,
        "adapted_traj_latent": adapted_traj_latent,
        "projected_traj_condition": projected_traj_condition,
    }
    saved_tensors: dict[str, torch.Tensor] = {}
    for name, value in latent_tensors.items():
        if not torch.is_tensor(value) or value.shape[0] != 1:
            raise RuntimeError(
                f"System2 STOP {name} must have a singleton batch dimension, got "
                f"{getattr(value, 'shape', None)}"
            )
        value = value.detach().cpu().contiguous().squeeze(0)
        if not bool(torch.isfinite(value.float()).all()):
            raise RuntimeError(f"System2 STOP {name} contains non-finite values")
        saved_tensors[name] = value
    if not torch.is_tensor(trajectory) or trajectory.ndim != 3 or trajectory.numel() == 0:
        raise RuntimeError(
            "System2 STOP trajectory must have shape (samples, steps, dims), got "
            f"{getattr(trajectory, 'shape', None)}"
        )
    saved_trajectory = trajectory.detach().cpu().contiguous()
    if not bool(torch.isfinite(saved_trajectory.float()).all()):
        raise RuntimeError("System2 STOP trajectory contains non-finite values")

    if not trajectory_metrics or not all(
        np.isfinite(float(value)) for value in trajectory_metrics.values()
    ):
        raise RuntimeError("System2 STOP trajectory metrics must be finite")
    payload.update(
        {
            "trajectory_feature_schema": (
                "heatmapvln-system2-stop-trajectory-feature-v1"
            ),
            **saved_tensors,
            "trajectory": saved_trajectory,
            "trajectory_metrics": {
                str(name): float(value) for name, value in trajectory_metrics.items()
            },
            "local_actions": [int(action) for action in local_actions],
            "pixel_goal": [int(pixel_goal[0]), int(pixel_goal[1])],
            "pano_goal_view": str(pano_goal_view),
        }
    )
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.trajectory.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)

    result = dict(feature_record)
    result.update(
        {
            "trajectory_feature_schema": payload["trajectory_feature_schema"],
            "raw_traj_latent_shape": list(saved_tensors["raw_traj_latent"].shape),
            "adapted_traj_latent_shape": list(
                saved_tensors["adapted_traj_latent"].shape
            ),
            "projected_traj_condition_shape": list(
                saved_tensors["projected_traj_condition"].shape
            ),
        }
    )
    return result


class HeatmapVLNRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        if not hasattr(args, "system2_stop_decision_adapter_checkpoint"):
            args.system2_stop_decision_adapter_checkpoint = None
        _validate_system2_stop_threshold_overrides(
            static_head_enabled=bool(
                args.system2_stop_head_checkpoint
                or args.system2_stop_decision_adapter_checkpoint
            ),
            temporal_verifier_enabled=bool(
                args.system2_temporal_stop_verifier_checkpoint
            ),
            add_threshold_override=args.system2_stop_add_threshold,
            veto_threshold_override=args.system2_stop_veto_threshold,
        )
        install_numpy_legacy_aliases()
        if os.environ.get("HEATMAPVLN_FORCE_FLASH_ATTN_STUB", "0") == "1":
            install_flash_attn_stub(LOGGER)
        self.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.cfg = self._load_runtime_config(args)
        self.model, self.train_cfg = self._load_model(args, self.device)
        self.require_deterministic_sampling = bool(
            getattr(args, "require_deterministic_sampling", False)
        )
        self.system2_stop_head = None
        self.system2_stop_add_head = None
        self.system2_stop_decision_adapter_name = None
        self.system2_stop_decision_adapter_metadata: dict[str, Any] = {}
        self.system2_stop_decision_policy_kind = STOP_DECISION_ADD_AND_VETO_POLICY
        self.system2_stop_decision_add_enabled = True
        self.system2_temporal_static_head = None
        self.system2_temporal_stop_verifier = None
        self.system2_temporal_stop_acceptance_threshold = 0.5
        self.system2_temporal_stop_policy_kind = None
        self.system2_temporal_stop_history = None
        self.system2_stop_add_threshold = 0.9
        self.system2_stop_veto_threshold = 0.5
        self.system2_stop_add_min_qwen_stop_probability = float(
            args.system2_stop_add_min_qwen_stop_probability
        )
        if not (
            np.isfinite(self.system2_stop_add_min_qwen_stop_probability)
            and 0.0 <= self.system2_stop_add_min_qwen_stop_probability <= 1.0
        ):
            raise ValueError(
                "system2_stop_add_min_qwen_stop_probability must be in [0, 1]"
            )
        self._system2_stop_alignment_checked = False
        self.system2_stop_feature_dump_dir = (
            Path(args.system2_stop_feature_dump_dir).expanduser().resolve()
            if args.system2_stop_feature_dump_dir
            else None
        )
        if self.system2_stop_feature_dump_dir is not None:
            self.system2_stop_feature_dump_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.warning(
                "System2 STOP DAgger feature collection is ACTIVE: %s",
                self.system2_stop_feature_dump_dir,
            )
        if args.system2_stop_decision_adapter_checkpoint and (
            args.system2_stop_head_checkpoint
            or args.system2_temporal_stop_verifier_checkpoint
        ):
            raise ValueError(
                "The STOP-decision LoRA is mutually exclusive with static/temporal "
                "STOP policies"
            )
        if args.system2_stop_decision_adapter_checkpoint and int(args.workers) != 1:
            raise ValueError(
                "STOP-decision LoRA switches a process-global PEFT adapter stack and "
                "therefore requires --workers 1"
            )
        if (
            args.system2_stop_decision_adapter_checkpoint
            and self.system2_stop_feature_dump_dir is not None
        ):
            raise ValueError(
                "STOP-decision LoRA inference cannot run during privileged feature collection"
            )
        if (
            args.system2_stop_decision_adapter_checkpoint
            and self.system2_stop_add_min_qwen_stop_probability > 0.0
        ):
            raise ValueError(
                "STOP-decision LoRA does not use the original Qwen STOP probability gate"
            )
        hybrid_stop_policy = bool(
            args.system2_stop_head_checkpoint
            and args.system2_temporal_stop_verifier_checkpoint
        )
        if hybrid_stop_policy and self.system2_stop_add_min_qwen_stop_probability > 0.0:
            raise ValueError(
                "Hybrid STOP inference requires add_min_qwen_stop_probability=0; "
                "the Qwen STOP score is not a valid near-goal add gate"
            )
        if (
            args.system2_temporal_stop_verifier_checkpoint
            and self.system2_stop_feature_dump_dir is not None
        ):
            raise ValueError(
                "Temporal STOP inference cannot be combined with privileged STOP feature collection"
            )
        if args.system2_temporal_stop_verifier_checkpoint:
            if not self.require_deterministic_sampling:
                raise ValueError(
                    "Temporal STOP inference requires --require_deterministic_sampling"
                )
            from src.models.action.temporal_stop_verifier import TemporalStopEpisodeHistory

            (
                self.system2_temporal_static_head,
                self.system2_temporal_stop_verifier,
                self.system2_temporal_stop_acceptance_threshold,
            ) = _load_system2_temporal_stop_verifier(
                args.system2_temporal_stop_verifier_checkpoint,
                device=self.device,
            )
            from src.models.action.temporal_stop_verifier import (
                TemporalStopVerifierEnsemble,
            )

            if isinstance(
                self.system2_temporal_stop_verifier,
                TemporalStopVerifierEnsemble,
            ):
                self.system2_temporal_stop_policy_kind = "scene_fold_unanimous_ensemble"
                threshold_description = (
                    self.system2_temporal_stop_verifier.acceptance_thresholds
                    .detach()
                    .cpu()
                    .tolist()
                )
            else:
                self.system2_temporal_stop_policy_kind = "single"
                threshold_description = [
                    float(self.system2_temporal_stop_acceptance_threshold)
                ]
            self.system2_temporal_stop_history = TemporalStopEpisodeHistory()
            LOGGER.info(
                "Verified veto-only System2 temporal STOP verifier: "
                "kind=%s static_tensors=%d verifier_tensors=%d "
                "acceptance_thresholds=%s; "
                "original non-STOP outputs can never be changed checkpoint=%s",
                self.system2_temporal_stop_policy_kind,
                len(self.system2_temporal_static_head.state_dict()),
                len(self.system2_temporal_stop_verifier.state_dict()),
                threshold_description,
                args.system2_temporal_stop_verifier_checkpoint,
            )
        if args.system2_stop_head_checkpoint:
            (
                loaded_stop_head,
                self.system2_stop_add_threshold,
                self.system2_stop_veto_threshold,
            ) = _load_system2_stop_head(
                args.system2_stop_head_checkpoint,
                device=self.device,
                dtype=self.model.config.dtype,
                add_threshold_override=args.system2_stop_add_threshold,
                veto_threshold_override=args.system2_stop_veto_threshold,
            )
            if hybrid_stop_policy:
                self.system2_stop_add_head = loaded_stop_head
            else:
                self.system2_stop_head = loaded_stop_head
            LOGGER.info(
                "Verified isolated System2 STOP head: tensors=%d "
                "add_threshold=%.4f veto_threshold=%.4f "
                "add_min_qwen_stop_probability=%.6g policy_role=%s; "
                "original Stage1-S2 LoRA remains the only Qwen adapter checkpoint=%s",
                len(loaded_stop_head.state_dict()),
                self.system2_stop_add_threshold,
                self.system2_stop_veto_threshold,
                self.system2_stop_add_min_qwen_stop_probability,
                "add_only_with_temporal_veto" if hybrid_stop_policy else "add_and_veto",
                args.system2_stop_head_checkpoint,
            )
        elif args.system2_stop_decision_adapter_checkpoint:
            self.system2_stop_decision_adapter_metadata = (
                _load_system2_stop_decision_adapter(
                    args.system2_stop_decision_adapter_checkpoint,
                    integration=self.model.qwen2_5_vl,
                    expected_base_checkpoint=args.base_checkpoint,
                    add_threshold_override=args.system2_stop_add_threshold,
                    veto_threshold_override=args.system2_stop_veto_threshold,
                )
            )
            self.system2_stop_decision_adapter_name = str(
                self.system2_stop_decision_adapter_metadata["adapter_name"]
            )
            self.system2_stop_decision_policy_kind = str(
                self.system2_stop_decision_adapter_metadata["policy_kind"]
            )
            self.system2_stop_decision_add_enabled = bool(
                self.system2_stop_decision_adapter_metadata["add_enabled"]
            )
            self.system2_stop_add_threshold = float(
                self.system2_stop_decision_adapter_metadata["add_stop_threshold"]
            )
            self.system2_stop_veto_threshold = float(
                self.system2_stop_decision_adapter_metadata["veto_stop_threshold"]
            )
            LOGGER.info(
                "Verified isolated System2 STOP-decision LoRA: tensors=%d "
                "parameters=%d policy_kind=%s add_enabled=%s "
                "add_threshold=%.4f veto_threshold=%.4f; "
                "navigation generation and System1 latent extraction remain "
                "default-LoRA-only checkpoint=%s",
                self.system2_stop_decision_adapter_metadata["adapter_tensors"],
                self.system2_stop_decision_adapter_metadata["adapter_parameters"],
                self.system2_stop_decision_policy_kind,
                self.system2_stop_decision_add_enabled,
                self.system2_stop_add_threshold,
                self.system2_stop_veto_threshold,
                args.system2_stop_decision_adapter_checkpoint,
            )
        elif self.system2_stop_add_min_qwen_stop_probability > 0.0:
            raise ValueError(
                "system2_stop_add_min_qwen_stop_probability requires a STOP head"
            )
        self.processor = self.model.qwen2_5_vl.processor
        self.processor.tokenizer.padding_side = "left"
        self.action_scale = self.train_cfg.get("data", {}).get("trajectory", {}).get("action_scale", 4.0)
        self.num_sample_trajs = (
            self.train_cfg.get("model", {}).get("action_head", {}).get("nextdit", {}).get("num_sample_trajs", 32)
        )
        self.has_nextdit = self.model.nextdit_action_head is not None and self.model.latent_queries is not None
        self.pano_latent_adapter = self._load_adapter(args)
        if self.pano_latent_adapter is None and getattr(self.model, "pano_latent_adapter", None) is not None:
            self.pano_latent_adapter = self.model.pano_latent_adapter
            self.pano_latent_adapter.eval()
            LOGGER.info("Using model-attached pano latent adapter")
        LOGGER.info("NextDiT action head available: %s", self.has_nextdit)
        LOGGER.info("action_scale=%s num_sample_trajs=%s", self.action_scale, self.num_sample_trajs)
        LOGGER.info(
            "require_deterministic_sampling=%s",
            self.require_deterministic_sampling,
        )

    def _load_runtime_config(self, args: argparse.Namespace) -> dict:
        cfg = load_config(args.config)
        internnav_path = (
            args.internnav_model_path
            or os.environ.get("INTERNNAV_MODEL_PATH")
            or cfg.get("paths", {}).get("internnav_model_path", "")
        )
        if internnav_path:
            internnav_path = os.path.expandvars(os.path.expanduser(str(internnav_path)))
            cfg.setdefault("paths", {})["internnav_model_path"] = internnav_path
            cfg.setdefault("model", {}).setdefault("llm", {})["model_path"] = internnav_path
            nextdit = cfg["model"].setdefault("action_head", {}).setdefault("nextdit", {})
            nextdit["internnav_model_path"] = internnav_path
            nextdit["internnav_system1_path"] = ""
            LOGGER.info("InternNav model path: %s", internnav_path)
        nextdit = cfg.get("model", {}).get("action_head", {}).get("nextdit", {})
        adapter_cfg = nextdit.get("pano_latent_adapter", {})
        if args.pano_latent_adapter_checkpoint and isinstance(adapter_cfg, dict):
            adapter_cfg["pretrained_path"] = ""
        return cfg

    def _load_model(self, args: argparse.Namespace, device: torch.device):
        from scripts.training.model_builder import (
            assert_complete_internnav_system1_load,
            build_model,
        )

        model = build_model(self.cfg, device=str(device), verbose=True)
        model = model.to(device)
        checkpoint_cfg = _extract_checkpoint_config(args.checkpoint)
        stage_cfg = _first_stage_config(checkpoint_cfg, self.cfg)
        if not args.base_checkpoint and checkpoint_cfg:
            recorded_base = checkpoint_cfg.get("runtime", {}).get("base_checkpoint")
            if recorded_base and Path(recorded_base).exists():
                args.base_checkpoint = str(Path(recorded_base).resolve())
                LOGGER.info("Auto-loading base checkpoint from Stage 2 metadata: %s", args.base_checkpoint)

        base_state_dict = _extract_checkpoint_state_dict(args.base_checkpoint) if args.base_checkpoint else None
        checkpoint_state_dict = _extract_checkpoint_state_dict(args.checkpoint) if args.checkpoint else None
        if (
            _requires_base_checkpoint(self.cfg, checkpoint_cfg)
            and not args.base_checkpoint
            and not _checkpoint_has_base_weights(checkpoint_state_dict)
        ):
            raise ValueError("This config/checkpoint requires --base_checkpoint")
        if checkpoint_state_dict and _looks_action_only(checkpoint_state_dict) and not args.base_checkpoint:
            LOGGER.warning("Main checkpoint contains only action-head weights and no base checkpoint was loaded")

        model.qwen2_5_vl._load_model()
        if _state_has_prefix(base_state_dict, "heatmap_vln.") or _state_has_prefix(
            checkpoint_state_dict, "heatmap_vln."
        ):
            model._ensure_heatmap_vln()

        if stage_cfg.get("require_complete_internnav_system1", False):
            required_system1 = assert_complete_internnav_system1_load(model, logger=LOGGER)
            LOGGER.info(
                "Verified complete frozen InternNav System1 for RPC evaluation: %d tensors",
                required_system1,
            )

        if base_state_dict:
            _assert_finite_state_dict(base_state_dict, "Base checkpoint")
            if _requires_base_checkpoint(self.cfg, checkpoint_cfg):
                matched_lora = assert_complete_lora_checkpoint_match(
                    model,
                    base_state_dict,
                    checkpoint_path=args.base_checkpoint,
                )
                LOGGER.info("Verified complete LoRA checkpoint match: %d tensors", matched_lora)
            base_state_to_load = base_state_dict
            if stage_cfg.get("base_checkpoint_lora_only", False):
                base_state_to_load = extract_lora_checkpoint_state(base_state_dict)
                if not base_state_to_load:
                    raise RuntimeError(f"Base checkpoint contains no LoRA tensors: {args.base_checkpoint}")
                LOGGER.info(
                    "Base checkpoint LoRA-only guard: loading %d/%d tensors; "
                    "InternNav System1 and pano adapter cannot be overwritten",
                    len(base_state_to_load),
                    len(base_state_dict),
                )
            loaded_base = _load_compatible_state_dict(
                model,
                base_state_to_load,
                args.base_checkpoint,
                label="Base checkpoint",
            )
            if loaded_base != len(base_state_to_load):
                raise RuntimeError(
                    f"Incomplete base checkpoint load refused: loaded={loaded_base}/{len(base_state_to_load)}"
                )
        if checkpoint_state_dict:
            _assert_finite_state_dict(checkpoint_state_dict, "Main checkpoint")
            loaded_main = _load_compatible_state_dict(
                model,
                checkpoint_state_dict,
                args.checkpoint,
                label="Main checkpoint",
            )
            if loaded_main != len(checkpoint_state_dict):
                raise RuntimeError(
                    f"Incomplete main checkpoint load refused: loaded={loaded_main}/{len(checkpoint_state_dict)}"
                )
        del checkpoint_state_dict
        del base_state_dict
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model.eval()
        return model, self.cfg

    def _load_adapter(self, args: argparse.Namespace):
        if not args.pano_latent_adapter_checkpoint:
            return None
        hidden_dim = int(self.train_cfg.get("model", {}).get("llm", {}).get("hidden_dim", 3584))
        LOGGER.info("Loading pano latent adapter from %s", args.pano_latent_adapter_checkpoint)
        adapter = _load_pano_latent_adapter(
            args.pano_latent_adapter_checkpoint,
            hidden_dim,
            self.device,
            self.model.config.dtype,
        )
        adapter_cfg = (
            self.train_cfg.get("model", {}).get("action_head", {}).get("nextdit", {}).get("pano_latent_adapter", {})
        )
        expected_hidden_dim = int(adapter_cfg.get("hidden_dim", 0) or 0)
        actual_dim = int(getattr(adapter, "dim", hidden_dim))
        actual_hidden_dim = int(getattr(adapter, "hidden_dim", expected_hidden_dim))
        if actual_dim != hidden_dim or (expected_hidden_dim > 0 and actual_hidden_dim != expected_hidden_dim):
            raise RuntimeError(
                "Pano adapter architecture mismatch: "
                f"dim={actual_dim}/{hidden_dim} "
                f"hidden_dim={actual_hidden_dim}/{expected_hidden_dim}"
            )
        _assert_finite_state_dict(adapter.state_dict(), "Pano latent adapter")
        LOGGER.info(
            "Verified pano latent adapter: tensors=%d parameters=%d dim=%d hidden_dim=%d dtype=%s",
            len(adapter.state_dict()),
            sum(parameter.numel() for parameter in adapter.parameters()),
            actual_dim,
            actual_hidden_dim,
            next(adapter.parameters()).dtype,
        )
        return adapter

    def plan_panoramic(self, payload: dict[str, Any], blobs) -> dict[str, Any]:
        require_deterministic = self.require_deterministic_sampling or bool(
            payload.get("require_deterministic_sampling", False)
        )
        sampling_metadata = validate_rpc_sampling_metadata(
            payload.get(HEATMAPVLN_RPC_SAMPLING_FIELD),
            require_deterministic=require_deterministic,
        )
        trajectory_generator = None
        if sampling_metadata is not None:
            trajectory_generator = torch.Generator(device=self.device)
            trajectory_generator.manual_seed(int(sampling_metadata["per_call_seed"]))

        blob_map = _blobs_by_name(blobs)
        vlm_image_size = tuple(payload.get("vlm_image_size") or self.train_cfg["data"]["image_size"])
        traj_image_size = tuple(
            payload.get("traj_image_size")
            or self.train_cfg.get("data", {}).get("trajectory", {}).get("traj_image_size", [224, 224])
        )
        current_views = {
            view: _pil_from_blob(blob_map[f"current/{view}"], vlm_image_size)
            for view in ("front", "right", "back", "left")
        }
        history_panoramas: list[dict[str, Image.Image]] = []
        for hist_idx in range(int(payload.get("num_history", 0))):
            history_panoramas.append(
                {
                    view: _pil_from_blob(blob_map[f"history/{hist_idx}/{view}"], vlm_image_size)
                    for view in ("front", "right", "back", "left")
                }
            )
        lookdown_img = _pil_from_blob(blob_map["lookdown"], traj_image_size)
        instruction = str(payload.get("instruction", ""))
        trajectory_cfg = self.train_cfg.get("data", {}).get("trajectory", {})
        internnav_protocol = trajectory_cfg.get("system2_sft_protocol", "direct").lower() == "internnav"
        structured_pano_output = bool(trajectory_cfg.get("structured_pano_output", True))
        system1_coord_order = str(payload.get("system1_coord_order", "generated"))
        if system1_coord_order == "auto":
            system1_coord_order = "generated"
        trajectory_selection = str(payload.get("trajectory_selection", "mean"))
        trajectory_x_sign = float(payload.get("trajectory_x_sign", 1.0))
        trajectory_heading_alignment = str(payload.get("trajectory_heading_alignment", "none")).lower()
        if trajectory_x_sign not in (-1.0, 1.0):
            raise ValueError(f"trajectory_x_sign must be -1 or 1, got {trajectory_x_sign}")
        if trajectory_heading_alignment not in {"none", "pano_pixel"}:
            raise ValueError(
                f"trajectory_heading_alignment must be none or pano_pixel, got {trajectory_heading_alignment!r}"
            )
        oracle_system2 = payload.get("oracle_system2")
        if not isinstance(oracle_system2, dict):
            oracle_system2 = None
        oracle_system2_text = ""
        if oracle_system2 is not None:
            oracle_system2_text = str(oracle_system2.get("text") or "").strip()
        force_non_stop = _validate_system2_force_non_stop_request(
            force_non_stop=payload.get("system2_force_non_stop", False),
            feature_dump_enabled=self.system2_stop_feature_dump_dir is not None,
            stop_head_enabled=(
                self.system2_stop_head is not None
                or self.system2_stop_add_head is not None
                or self.system2_temporal_stop_verifier is not None
                or self.system2_stop_decision_adapter_name is not None
            ),
            oracle_system2_enabled=bool(oracle_system2_text),
        )

        messages = construct_input(
            current_views=current_views,
            history_panoramas=history_panoramas,
            instruction=instruction,
            pixel_goal=[0, 0],
            internnav_protocol=internnav_protocol,
            structured_pano_output=structured_pano_output,
        )
        messages = [m for m in messages if m["role"] != "assistant"]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        _normalize_multimodal_inputs(inputs)

        prompt_len = inputs["input_ids"].shape[1]
        if oracle_system2_text:
            oracle_ids = self.processor.tokenizer.encode(
                oracle_system2_text,
                add_special_tokens=False,
            )
            if not oracle_ids:
                raise ValueError("oracle_system2.text produced no tokens")
            oracle_suffix = torch.tensor(
                [oracle_ids],
                device=inputs["input_ids"].device,
                dtype=inputs["input_ids"].dtype,
            )
            output_ids = torch.cat([inputs["input_ids"], oracle_suffix], dim=1)
            llm_output = oracle_system2_text
            decision_scores: dict[str, Any] = {}
            stop_head_result: dict[str, Any] = {}
            stop_feature_result: dict[str, Any] = {}
        else:
            generation_kwargs = {
                **inputs,
                "max_new_tokens": 128,
                "do_sample": False,
                "use_cache": True,
                "return_dict_in_generate": True,
            }
            if structured_pano_output:
                generation_kwargs["logits_processor"] = LogitsProcessorList(
                    [
                        _StructuredViewPrefixLogitsProcessor(
                            tokenizer=self.processor.tokenizer,
                            prompt_len=prompt_len,
                        )
                    ]
                )
            if force_non_stop:
                generation_kwargs["bad_words_ids"] = _system2_stop_bad_words_ids(
                    self.processor.tokenizer
                )
            if self.system2_stop_decision_adapter_name is not None:
                _assert_navigation_only_lora(
                    self.model.qwen2_5_vl,
                    context="Original System2 waypoint generation",
                )
            need_stop_decision_hidden = (
                not force_non_stop
                and (
                    self.system2_stop_head is not None
                    or self.system2_stop_add_head is not None
                    or self.system2_temporal_stop_verifier is not None
                    or self.system2_stop_feature_dump_dir is not None
                )
            )
            with torch.inference_mode():
                generation = self.model.qwen2_5_vl.model.generate(
                    **generation_kwargs,
                    output_scores=True,
                    output_hidden_states=need_stop_decision_hidden,
                )
            output_ids = generation.sequences
            original_output = self.processor.tokenizer.decode(
                output_ids[0][prompt_len:],
                skip_special_tokens=True,
            )
            if force_non_stop and vlm_output_requests_stop(original_output):
                raise RuntimeError(
                    "Forced DAgger continuation still generated a STOP response"
                )
            llm_output = original_output
            decision_scores = _system2_decision_scores(
                tokenizer=self.processor.tokenizer,
                sequence=output_ids,
                prompt_len=prompt_len,
                generation_scores=generation.scores,
            )
            stop_head_result = {}
            stop_feature_result = {}
            decision_hidden = None

            if need_stop_decision_hidden:
                decision_hidden = _system2_generation_decision_hidden(
                    generation=generation,
                    tokenizer=self.processor.tokenizer,
                    prompt_len=prompt_len,
                )
            if self.system2_stop_feature_dump_dir is not None and not force_non_stop:
                if sampling_metadata is None:
                    raise RuntimeError(
                        "System2 STOP feature collection requires deterministic sampling metadata"
                    )
                stop_feature_result = _write_system2_stop_feature(
                    self.system2_stop_feature_dump_dir,
                    decision_hidden=decision_hidden,
                    sampling_metadata=sampling_metadata,
                    original_output=original_output,
                    decision_scores=decision_scores,
                )

            alignment_metrics = None
            stop_policy_enabled = (
                self.system2_stop_head is not None
                or self.system2_stop_add_head is not None
                or self.system2_temporal_stop_verifier is not None
            )
            if stop_policy_enabled:
                if decision_hidden is None:
                    raise RuntimeError("System2 STOP policy did not receive decision hidden state")
                if not self._system2_stop_alignment_checked:
                    teacher_decision_hidden = _system2_teacher_forced_decision_hidden(
                        qwen_integration=self.model.qwen2_5_vl,
                        inputs=inputs,
                        output_ids=output_ids,
                        prompt_len=prompt_len,
                    )
                    alignment_metrics = _system2_stop_hidden_alignment(
                        decision_hidden,
                        teacher_decision_hidden,
                    )
                    LOGGER.info(
                        "System2 STOP hidden alignment: %s",
                        json.dumps(alignment_metrics, sort_keys=True),
                    )
                    if alignment_metrics["cosine_min"] < 0.999:
                        raise RuntimeError(
                            "System2 STOP training/generation hidden states are not aligned: "
                            f"{alignment_metrics}"
                        )
                    self._system2_stop_alignment_checked = True

            if self.system2_stop_decision_adapter_name is not None:
                adapter_probe = _system2_stop_decision_adapter_probe(
                    qwen_integration=self.model.qwen2_5_vl,
                    inputs=inputs,
                    adapter_name=self.system2_stop_decision_adapter_name,
                )
                stop_probability = float(adapter_probe["stop_probability"])
                constrained_output = None
                constrained_generation_output = None
                constrained_generation_fallback = False
                decision = _system2_stop_head_decision(
                    stop_probability=stop_probability,
                    add_stop_threshold=self.system2_stop_add_threshold,
                    veto_stop_threshold=self.system2_stop_veto_threshold,
                    original_output=original_output,
                    image_size=vlm_image_size,
                    allow_add_stop=self.system2_stop_decision_add_enabled,
                )
                if decision == "head_adds_stop":
                    llm_output = "view: stop"
                elif decision == "head_requests_stop_veto":
                    _assert_navigation_only_lora(
                        self.model.qwen2_5_vl,
                        context="STOP-veto constrained waypoint generation",
                    )
                    constrained_kwargs = _system2_non_stop_generation_kwargs(
                        generation_kwargs,
                        tokenizer=self.processor.tokenizer,
                        prompt_len=prompt_len,
                    )
                    with torch.inference_mode():
                        constrained_generation = self.model.qwen2_5_vl.model.generate(
                            **constrained_kwargs,
                            output_scores=False,
                        )
                    constrained_ids = constrained_generation.sequences
                    constrained_generation_output = self.processor.tokenizer.decode(
                        constrained_ids[0][prompt_len:],
                        skip_special_tokens=True,
                    )
                    constrained_output, constrained_generation_fallback = (
                        _system2_non_stop_output_or_fallback(
                            constrained_generation_output,
                            system2_call_index=int(
                                (sampling_metadata or {}).get(
                                    "system2_call_index", 0
                                )
                            ),
                        )
                    )
                    decision = _system2_stop_head_decision(
                        stop_probability=stop_probability,
                        add_stop_threshold=self.system2_stop_add_threshold,
                        veto_stop_threshold=self.system2_stop_veto_threshold,
                        original_output=original_output,
                        constrained_output=constrained_output,
                        image_size=vlm_image_size,
                        allow_add_stop=self.system2_stop_decision_add_enabled,
                    )
                    if decision in {
                        "head_vetoes_stop",
                        "head_veto_fallback_replan",
                    }:
                        output_ids = constrained_ids
                        llm_output = constrained_output
                stop_head_result = {
                    "mode": "stop_decision_adapter",
                    "policy_kind": self.system2_stop_decision_policy_kind,
                    "add_enabled": self.system2_stop_decision_add_enabled,
                    "decision": decision,
                    "stop_probability": stop_probability,
                    "stop_log_odds": adapter_probe["stop_log_odds"],
                    "selected": adapter_probe["selected"],
                    "class_probabilities": adapter_probe["class_probabilities"],
                    "threshold": (
                        self.system2_stop_veto_threshold
                        if vlm_output_requests_stop(original_output)
                        else self.system2_stop_add_threshold
                    ),
                    "add_stop_threshold": self.system2_stop_add_threshold,
                    "veto_stop_threshold": self.system2_stop_veto_threshold,
                    "qwen_stop_probability": float(
                        (decision_scores.get("class_probabilities") or {}).get(
                            "stop", 0.0
                        )
                    ),
                    "original_output": original_output,
                    "constrained_output": constrained_output,
                    "constrained_generation_output": constrained_generation_output,
                    "constrained_generation_fallback": constrained_generation_fallback,
                }

            elif self.system2_stop_head is not None:
                head_dtype = next(self.system2_stop_head.parameters()).dtype
                with torch.inference_mode():
                    stop_probability = _system2_stop_probability(
                        self.system2_stop_head,
                        decision_hidden.to(device=self.device, dtype=head_dtype),
                    )

                constrained_output = None
                constrained_generation_output = None
                constrained_generation_fallback = False
                qwen_stop_probability = float(
                    (decision_scores.get("class_probabilities") or {}).get(
                        "stop", 0.0
                    )
                )
                decision = _system2_stop_head_decision(
                    stop_probability=stop_probability,
                    add_stop_threshold=self.system2_stop_add_threshold,
                    veto_stop_threshold=self.system2_stop_veto_threshold,
                    original_output=original_output,
                    original_stop_probability=qwen_stop_probability,
                    add_min_qwen_stop_probability=(
                        self.system2_stop_add_min_qwen_stop_probability
                    ),
                    image_size=vlm_image_size,
                )
                if decision == "head_adds_stop":
                    llm_output = "view: stop"
                elif decision == "head_requests_stop_veto":
                    constrained_kwargs = _system2_non_stop_generation_kwargs(
                        generation_kwargs,
                        tokenizer=self.processor.tokenizer,
                        prompt_len=prompt_len,
                    )
                    with torch.inference_mode():
                        constrained_generation = self.model.qwen2_5_vl.model.generate(
                            **constrained_kwargs,
                            output_scores=False,
                        )
                    constrained_ids = constrained_generation.sequences
                    constrained_generation_output = self.processor.tokenizer.decode(
                        constrained_ids[0][prompt_len:],
                        skip_special_tokens=True,
                    )
                    constrained_output, constrained_generation_fallback = (
                        _system2_non_stop_output_or_fallback(
                            constrained_generation_output,
                            system2_call_index=int(
                                (sampling_metadata or {}).get("system2_call_index", 0)
                            ),
                        )
                    )
                    decision = _system2_stop_head_decision(
                        stop_probability=stop_probability,
                        add_stop_threshold=self.system2_stop_add_threshold,
                        veto_stop_threshold=self.system2_stop_veto_threshold,
                        original_output=original_output,
                        original_stop_probability=qwen_stop_probability,
                        add_min_qwen_stop_probability=(
                            self.system2_stop_add_min_qwen_stop_probability
                        ),
                        constrained_output=constrained_output,
                        image_size=vlm_image_size,
                    )
                    if decision in {
                        "head_vetoes_stop",
                        "head_veto_fallback_replan",
                    }:
                        output_ids = constrained_ids
                        llm_output = constrained_output

                stop_head_result = {
                    "decision": decision,
                    "stop_probability": stop_probability,
                    "threshold": (
                        self.system2_stop_veto_threshold
                        if vlm_output_requests_stop(original_output)
                        else self.system2_stop_add_threshold
                    ),
                    "add_stop_threshold": self.system2_stop_add_threshold,
                    "veto_stop_threshold": self.system2_stop_veto_threshold,
                    "qwen_stop_probability": qwen_stop_probability,
                    "add_min_qwen_stop_probability": (
                        self.system2_stop_add_min_qwen_stop_probability
                    ),
                    "original_output": original_output,
                    "constrained_output": constrained_output,
                    "constrained_generation_output": constrained_generation_output,
                    "constrained_generation_fallback": constrained_generation_fallback,
                }
                if alignment_metrics is not None:
                    stop_head_result["hidden_alignment"] = alignment_metrics

            elif self.system2_temporal_stop_verifier is not None:
                if sampling_metadata is None:
                    raise RuntimeError(
                        "Temporal STOP inference requires deterministic sampling metadata"
                    )
                qwen_stop_log_odds = decision_scores.get("stop_log_odds")
                if qwen_stop_log_odds is None or not np.isfinite(float(qwen_stop_log_odds)):
                    raise RuntimeError(
                        "Temporal STOP inference requires a finite Qwen STOP log-odds"
                    )
                qwen_stop_probability = float(
                    (decision_scores.get("class_probabilities") or {}).get(
                        "stop", 0.0
                    )
                )
                static_add_stop_probability = None
                if self.system2_stop_add_head is not None:
                    add_dtype = next(self.system2_stop_add_head.parameters()).dtype
                    with torch.inference_mode():
                        static_add_stop_probability = _system2_stop_probability(
                            self.system2_stop_add_head,
                            decision_hidden.to(device=self.device, dtype=add_dtype),
                        )
                static_dtype = next(self.system2_temporal_static_head.parameters()).dtype
                with torch.inference_mode():
                    static_stop_probability = _system2_stop_probability(
                        self.system2_temporal_static_head,
                        decision_hidden.to(device=self.device, dtype=static_dtype),
                    )
                from src.models.action.temporal_stop_verifier import TemporalStopObservation

                temporal_features = self.system2_temporal_stop_history.observe(
                    episode_key=(
                        str(sampling_metadata["scene_id"]),
                        int(sampling_metadata["episode_id"]),
                        int(sampling_metadata["protocol_seed"]),
                    ),
                    observation=TemporalStopObservation(
                        call_index=int(sampling_metadata["system2_call_index"]),
                        hidden=decision_hidden.squeeze(0).detach().float().cpu(),
                        static_stop_probability=static_stop_probability,
                        qwen_stop_log_odds=float(qwen_stop_log_odds),
                    ),
                )
                verifier_dtype = next(
                    self.system2_temporal_stop_verifier.parameters()
                ).dtype
                verifier_input = temporal_features.unsqueeze(0).to(
                    device=self.device,
                    dtype=verifier_dtype,
                )
                with torch.inference_mode():
                    if self.system2_temporal_stop_policy_kind == "single":
                        member_probability_tensor = (
                            self.system2_temporal_stop_verifier(verifier_input)
                            .reshape(1)
                        )
                        member_threshold_tensor = torch.tensor(
                            [self.system2_temporal_stop_acceptance_threshold],
                            device=self.device,
                            dtype=torch.float32,
                        )
                    elif (
                        self.system2_temporal_stop_policy_kind
                        == "scene_fold_unanimous_ensemble"
                    ):
                        member_probability_tensor = (
                            self.system2_temporal_stop_verifier
                            .member_probabilities(verifier_input)
                            .reshape(-1)
                        )
                        member_threshold_tensor = (
                            self.system2_temporal_stop_verifier
                            .acceptance_thresholds
                            .detach()
                            .float()
                        )
                    else:
                        raise RuntimeError(
                            "Temporal STOP policy kind was not initialized: "
                            f"{self.system2_temporal_stop_policy_kind!r}"
                        )
                member_probability_tensor = member_probability_tensor.detach().float()
                if (
                    member_probability_tensor.shape != member_threshold_tensor.shape
                    or not bool(torch.isfinite(member_probability_tensor).all())
                    or bool(
                        (
                            (member_probability_tensor < 0.0)
                            | (member_probability_tensor > 1.0)
                        ).any()
                    )
                ):
                    raise RuntimeError(
                        "Temporal STOP verifier returned invalid member probabilities: "
                        f"shape={tuple(member_probability_tensor.shape)} "
                        f"values={member_probability_tensor.cpu().tolist()}"
                    )
                member_margin_tensor = (
                    member_probability_tensor - member_threshold_tensor
                )
                temporal_accepted = bool((member_margin_tensor >= 0.0).all())
                member_probabilities = member_probability_tensor.cpu().tolist()
                member_thresholds = member_threshold_tensor.cpu().tolist()
                member_margins = member_margin_tensor.cpu().tolist()
                temporal_min_margin = float(member_margin_tensor.min().item())
                # This scalar is an acceptance score for legacy logging. The exact
                # unanimous decision and raw calibrated member values are preserved below.
                temporal_acceptance_score = min(
                    max(0.5 + temporal_min_margin, 0.0),
                    1.0,
                )

                constrained_output = None
                constrained_generation_output = None
                constrained_generation_fallback = False
                if not vlm_output_requests_stop(original_output):
                    decision = "temporal_keeps_original_non_stop"
                elif temporal_accepted:
                    decision = "temporal_confirms_original_stop"
                else:
                    decision = "temporal_requests_stop_veto"
                if decision == "temporal_requests_stop_veto":
                    constrained_kwargs = _system2_non_stop_generation_kwargs(
                        generation_kwargs,
                        tokenizer=self.processor.tokenizer,
                        prompt_len=prompt_len,
                    )
                    with torch.inference_mode():
                        constrained_generation = self.model.qwen2_5_vl.model.generate(
                            **constrained_kwargs,
                            output_scores=False,
                        )
                    constrained_ids = constrained_generation.sequences
                    constrained_generation_output = self.processor.tokenizer.decode(
                        constrained_ids[0][prompt_len:],
                        skip_special_tokens=True,
                    )
                    constrained_output, constrained_generation_fallback = (
                        _system2_non_stop_output_or_fallback(
                            constrained_generation_output,
                            system2_call_index=int(
                                sampling_metadata["system2_call_index"]
                            ),
                        )
                    )
                    output_ids = constrained_ids
                    llm_output = constrained_output
                    decision = "temporal_vetoes_original_stop"

                temporal_decision = decision
                static_add_decision = None
                if (
                    self.system2_stop_add_head is not None
                    and not vlm_output_requests_stop(original_output)
                ):
                    static_add_decision = _system2_stop_head_decision(
                        stop_probability=float(static_add_stop_probability),
                        add_stop_threshold=self.system2_stop_add_threshold,
                        veto_stop_threshold=self.system2_stop_veto_threshold,
                        original_output=original_output,
                        original_stop_probability=qwen_stop_probability,
                        add_min_qwen_stop_probability=0.0,
                        image_size=vlm_image_size,
                    )
                    decision = _system2_hybrid_stop_decision(
                        original_output=original_output,
                        temporal_decision=temporal_decision,
                        static_add_decision=static_add_decision,
                    )
                    if decision == "hybrid_static_adds_stop":
                        llm_output = "view: stop"

                hybrid_mode = self.system2_stop_add_head is not None
                effective_stop_probability = (
                    float(static_add_stop_probability)
                    if hybrid_mode and not vlm_output_requests_stop(original_output)
                    else temporal_acceptance_score
                )
                effective_threshold = (
                    self.system2_stop_add_threshold
                    if hybrid_mode and not vlm_output_requests_stop(original_output)
                    else 0.5
                )

                stop_head_result = {
                    "mode": (
                        "hybrid_static_add_temporal_veto"
                        if hybrid_mode
                        else "temporal_veto_only"
                    ),
                    "policy_kind": self.system2_temporal_stop_policy_kind,
                    "decision": decision,
                    "stop_probability": effective_stop_probability,
                    "temporal_decision": temporal_decision,
                    "static_add_decision": static_add_decision,
                    "static_add_stop_probability": static_add_stop_probability,
                    "temporal_acceptance_score": temporal_acceptance_score,
                    "temporal_stop_probability": temporal_acceptance_score,
                    "temporal_accepted": temporal_accepted,
                    "temporal_min_margin": temporal_min_margin,
                    "member_probabilities": member_probabilities,
                    "member_thresholds": member_thresholds,
                    "member_margins": member_margins,
                    "static_stop_probability": static_stop_probability,
                    "threshold": effective_threshold,
                    "add_stop_threshold": (
                        self.system2_stop_add_threshold if hybrid_mode else 1.0
                    ),
                    "veto_stop_threshold": 0.5,
                    "qwen_stop_probability": qwen_stop_probability,
                    "qwen_stop_log_odds": float(qwen_stop_log_odds),
                    "original_output": original_output,
                    "constrained_output": constrained_output,
                    "constrained_generation_output": constrained_generation_output,
                    "constrained_generation_fallback": constrained_generation_fallback,
                    "history_length": self.system2_temporal_stop_history.length,
                }
                if alignment_metrics is not None:
                    stop_head_result["hidden_alignment"] = alignment_metrics
        response: dict[str, Any] = {
            "ok": True,
            "proto_v": PROTO_VERSION,
            "llm_output": llm_output,
            "system2_source": "oracle" if oracle_system2_text else "model",
            "oracle_system2": oracle_system2,
            "actions": [],
            "terminal": False,
            "kind": "unknown",
            "system2_force_non_stop": force_non_stop,
        }
        if decision_scores:
            response["system2_decision_scores"] = decision_scores
        if stop_head_result:
            response["system2_stop_head"] = stop_head_result
            if stop_head_result.get("mode") in {
                "temporal_veto_only",
                "hybrid_static_add_temporal_veto",
            }:
                response["system2_temporal_stop_verifier"] = stop_head_result
        if stop_feature_result:
            response["system2_stop_feature"] = stop_feature_result
        if sampling_metadata is not None:
            response[HEATMAPVLN_RPC_SAMPLING_FIELD] = sampling_metadata
        if vlm_output_requests_stop(llm_output):
            response.update({"kind": "stop", "terminal": True, "actions": [ActionCode.STOP]})
            return response

        turn_dir = vlm_output_requests_turn(llm_output)
        if turn_dir is not None:
            action = ActionCode.LEFT if turn_dir == "left" else ActionCode.RIGHT
            response.update({"kind": "turn", "actions": [int(action)], "turn_direction": turn_dir})
            return response

        pixel_goal = _parse_pixel_goal(
            llm_output,
            vlm_image_size,
            # Main eval compatibility: salvage pure legacy "u v" coordinates
            # even under the structured prompt. Malformed structured `view:`
            # lines remain invalid in parse_structured_pano_output.
            allow_legacy_coord=True,
        )
        pano_goal_view = _parse_pano_view_id(llm_output) or "front"
        response["pixel_goal"] = pixel_goal
        response["pano_goal_view"] = pano_goal_view

        if self.has_nextdit and pixel_goal is not None:
            target_heading_deg = None
            if trajectory_heading_alignment == "pano_pixel":
                target_heading_deg = view_pixel_target_angle_deg(
                    pano_goal_view,
                    pixel_goal,
                    vlm_image_size,
                )
            lookdown_t = _lookdown_to_traj_tensor(lookdown_img, self.device)
            pix_goal_image = lookdown_t.clone()
            traj_images = torch.stack([pix_goal_image, lookdown_t]).unsqueeze(0).to(self.device)
            lq = self.model.latent_queries.expand(1, -1, -1).to(
                device=self.device,
                dtype=self.model.config.dtype,
            )
            condition_output_ids = _condition_output_ids_for_pixel_goal(
                output_ids=output_ids,
                prompt_len=prompt_len,
                tokenizer=self.processor.tokenizer,
                pixel_goal=pixel_goal,
                llm_output=llm_output,
                coord_order=system1_coord_order,
                view_id=pano_goal_view,
                structured_output=structured_pano_output,
            )
            if self.system2_stop_decision_adapter_name is not None:
                _assert_navigation_only_lora(
                    self.model.qwen2_5_vl,
                    context="System1 trajectory-latent extraction",
                )
            with torch.no_grad():
                raw_traj_hs = self.model.qwen2_5_vl.generate_latents(
                    output_ids=condition_output_ids,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    latent_queries=lq,
                    attention_mask=inputs.get("attention_mask"),
                    mm_token_type_ids=inputs.get("mm_token_type_ids"),
                )
                traj_hs = raw_traj_hs
                if self.pano_latent_adapter is not None:
                    traj_hs = _maybe_apply_pano_latent_adapter(
                        traj_hs,
                        self.pano_latent_adapter,
                        view_id=pano_goal_view,
                        pixel_goal=pixel_goal,
                        image_size=vlm_image_size,
                        cond_projector=self.model.nextdit_action_head.cond_projector
                        if self.model.nextdit_action_head is not None
                        else None,
                    )
                trajectory = _trajectory_from_condition(
                    self.model.nextdit_action_head,
                    traj_hs,
                    traj_images=traj_images,
                    generator=trajectory_generator,
                )
            local_actions = _finalize_local_actions(
                traj_to_actions(
                    trajectory,
                    num_sample_trajs=self.num_sample_trajs,
                    action_scale=self.action_scale,
                    trajectory_selection=trajectory_selection,
                    trajectory_x_sign=trajectory_x_sign,
                    target_heading_deg=target_heading_deg,
                )
            )
            if local_actions and local_actions[0] == ActionCode.STOP:
                local_actions = [ActionCode.LEFT]
                response["anti_deadlock"] = True
            trajectory_summary = _trajectory_debug_summary(
                trajectory,
                self.num_sample_trajs,
                self.action_scale,
                trajectory_x_sign,
            )
            trajectory_metrics = _trajectory_debug_metrics(
                trajectory,
                self.num_sample_trajs,
                self.action_scale,
                trajectory_x_sign,
            )
            if stop_feature_result:
                adapted_traj_hs = raw_traj_hs
                if self.pano_latent_adapter is not None:
                    adapted_traj_hs = _maybe_apply_pano_latent_adapter(
                        raw_traj_hs,
                        self.pano_latent_adapter,
                        view_id=pano_goal_view,
                        pixel_goal=pixel_goal,
                        image_size=vlm_image_size,
                        cond_projector=None,
                    )
                projected_traj_condition = _project_trajectory_condition(
                    self.model.nextdit_action_head,
                    traj_hs,
                )
                stop_feature_result = _augment_system2_stop_feature_with_trajectory(
                    stop_feature_result,
                    raw_traj_latent=raw_traj_hs,
                    adapted_traj_latent=adapted_traj_hs,
                    projected_traj_condition=projected_traj_condition,
                    trajectory=trajectory,
                    trajectory_metrics=trajectory_metrics,
                    local_actions=[int(action) for action in local_actions],
                    pixel_goal=pixel_goal,
                    pano_goal_view=pano_goal_view,
                )
                response["system2_stop_feature"] = stop_feature_result
            response.update(
                {
                    "kind": "trajectory",
                    "actions": [int(action) for action in local_actions],
                    "trajectory_summary": trajectory_summary,
                    "trajectory_metrics": trajectory_metrics,
                    "trajectory_x_sign": trajectory_x_sign,
                    "trajectory_heading_alignment": trajectory_heading_alignment,
                    "trajectory_target_heading_deg": target_heading_deg,
                }
            )
            return response

        fallback_action = _fallback_replan_action(pano_goal_view)
        LOGGER.warning(
            "Malformed non-STOP System2 output; replanning with turn action %d: %r",
            fallback_action,
            llm_output,
        )
        response.update(
            {
                "kind": "fallback_replan",
                "terminal": False,
                "actions": [fallback_action],
                "parse_error": "malformed_non_stop_output",
            }
        )
        return response


class HeatmapVLNRPCServicer(vla_pb2_grpc.VLAServicer):
    def __init__(self, runtime: HeatmapVLNRuntime):
        self.runtime = runtime
        self.started = int(torch.cuda.Event(enable_timing=False) is not None)
        self.requests_processed = 0
        self.model_version = "heatmapvln-r2r"

    def InferJSON(self, request: vla_pb2.JSONRequest, context) -> vla_pb2.JSONResponse:
        try:
            payload = json.loads(request.json_payload) if request.json_payload else {}
            if request.method != "plan_panoramic":
                raise ValueError(f"Unsupported method: {request.method}")
            output = self.runtime.plan_panoramic(payload, request.blobs)
            self.requests_processed += 1
            return vla_pb2.JSONResponse(
                ts=request.ts,
                json_payload=json.dumps(output, ensure_ascii=False),
                model_v=self.model_version,
            )
        except Exception as exc:
            LOGGER.exception("InferJSON failed")
            context.set_details(str(exc))
            context.set_code(grpc.StatusCode.INTERNAL)
            return vla_pb2.JSONResponse(ts=request.ts, json_payload=json.dumps({"ok": False, "error": str(exc)}))

    def HealthCheck(self, request: vla_pb2.HealthCheckRequest, context) -> vla_pb2.HealthCheckResponse:
        return vla_pb2.HealthCheckResponse(
            status=vla_pb2.HealthCheckResponse.SERVING,
            message="HeatmapVLN model server is running",
            version=PROTO_VERSION,
            requests_processed=self.requests_processed,
        )

    def GetServerInfo(self, request: vla_pb2.Empty, context) -> vla_pb2.ServerInfo:
        return vla_pb2.ServerInfo(
            version=PROTO_VERSION,
            model_version=self.model_version,
            max_batch_size=1,
            supported_formats=["json+jpeg"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HeatmapVLN model-side RPC server")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base_checkpoint", default=None)
    parser.add_argument("--pano_latent_adapter_checkpoint", default=None)
    parser.add_argument(
        "--system2_stop_head_checkpoint",
        default=None,
        help=(
            "Optional isolated binary STOP-head checkpoint. The original Stage1-S2 "
            "LoRA remains the only Qwen adapter; a vetoed original STOP is regenerated "
            "with its STOP class token constrained."
        ),
    )
    parser.add_argument(
        "--system2_stop_decision_adapter_checkpoint",
        default=None,
        help=(
            "Optional STOP-only LoRA delta. It is added to the frozen default LoRA "
            "for one structured-class scoring forward; waypoint generation and "
            "System1 latent extraction remain default-LoRA-only."
        ),
    )
    parser.add_argument(
        "--system2_temporal_stop_verifier_checkpoint",
        default=None,
        help=(
            "Optional veto-only temporal STOP verifier. It embeds a frozen static "
            "STOP prior, requires deterministic contiguous episode call metadata, "
            "and never changes an original non-STOP output into STOP."
        ),
    )
    parser.add_argument(
        "--system2_stop_add_threshold",
        type=float,
        default=None,
        help="Optional override for adding STOP to an original non-STOP output.",
    )
    parser.add_argument(
        "--system2_stop_veto_threshold",
        type=float,
        default=None,
        help="Optional override for accepting rather than vetoing an original STOP.",
    )
    parser.add_argument(
        "--system2_stop_add_min_qwen_stop_probability",
        type=float,
        default=0.0,
        help=(
            "Minimum original Qwen structured STOP probability required before "
            "the auxiliary head may replace a non-STOP output with STOP."
        ),
    )
    parser.add_argument(
        "--system2_stop_feature_dump_dir",
        default=None,
        help=(
            "Optional DAgger collection directory for frozen-Qwen STOP decision "
            "features. Requires deterministic RPC sampling metadata."
        ),
    )
    parser.add_argument("--internnav_model_path", default=_default_internnav_model_path())
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--require_deterministic_sampling",
        action="store_true",
        help=(
            "Reject legacy/partial RPC requests that do not provide a valid "
            "SHA256-derived deterministic NextDiT sampling key."
        ),
    )
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    runtime = HeatmapVLNRuntime(args)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.workers),
        options=[
            ("grpc.max_send_message_length", 128 * 1024 * 1024),
            ("grpc.max_receive_message_length", 128 * 1024 * 1024),
        ],
    )
    vla_pb2_grpc.add_VLAServicer_to_server(HeatmapVLNRPCServicer(runtime), server)
    address = f"{args.host}:{args.port}"
    bound_port = server.add_insecure_port(address)
    if bound_port == 0:
        raise RuntimeError(f"Could not bind HeatmapVLN RPC server to {address}")
    server.start()
    LOGGER.info("HeatmapVLN RPC server listening on %s", address)

    def _shutdown(_signum, _frame):
        LOGGER.info("Stopping HeatmapVLN RPC server")
        server.stop(grace=5)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    server.wait_for_termination()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

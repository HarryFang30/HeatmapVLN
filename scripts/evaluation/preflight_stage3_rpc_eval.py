#!/usr/bin/env python3
"""Validate the exact checkpoints and config used by Stage3 RPC evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from scripts.training.utils import _normalize_state_key

_LORA_KEY_RE = re.compile(
    r"\.layers\.(?P<layer>\d+)\.self_attn\."
    r"(?P<module>q_proj|k_proj|v_proj|o_proj)\."
    r"lora_(?P<side>A|B)(?:\.[^.]+)?\.weight$"
)
_REQUIRED_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
_REQUIRED_STAGE3_TRAINABLE = ("pano_latent_adapter",)
_STOP_DECISION_SCHEMA = "heatmapvln-system2-stop-decision-adapter-v1"
_STOP_DECISION_ADD_AND_VETO_POLICY = "add_and_veto"
_STOP_DECISION_VETO_ONLY_POLICY = "veto_only"
_STRUCTURED_VIEW_CLASSES = ("stop", "front", "right", "back", "left", "turn")


def _torch_load(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must contain a dict: {path}")
    return checkpoint


def _checkpoint_state(checkpoint: dict[str, Any], path: str | Path) -> dict[str, torch.Tensor]:
    for key in ("model_state_dict", "trainable_state_dict", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise KeyError(f"No model/trainable/state dict in checkpoint: {path}")


def _adapter_state(checkpoint: dict[str, Any], path: str | Path) -> dict[str, torch.Tensor]:
    direct = checkpoint.get("adapter_state_dict")
    if isinstance(direct, dict):
        return direct

    for key in ("trainable_state_dict", "model_state_dict", "state_dict"):
        candidate = checkpoint.get(key)
        if not isinstance(candidate, dict):
            continue
        state = {
            name.removeprefix("module.").removeprefix("pano_latent_adapter."): value
            for name, value in candidate.items()
            if name.removeprefix("module.").startswith("pano_latent_adapter.")
        }
        if state:
            return state
    raise KeyError(f"No pano_latent_adapter state in checkpoint: {path}")


def _first_stage(config: dict[str, Any]) -> dict[str, Any]:
    stages = config.get("training", {}).get("stages", [])
    if not stages or not isinstance(stages[0], dict):
        raise ValueError("training.stages[0] is required")
    return stages[0]


def _assert_finite(state: dict[str, torch.Tensor], label: str) -> None:
    non_tensors = sorted(name for name, value in state.items() if not torch.is_tensor(value))
    if non_tensors:
        raise TypeError(f"{label} contains non-tensor values: {non_tensors[:5]}")
    nonfinite = [
        name
        for name, value in state.items()
        if not bool(torch.isfinite(value.float()).all())
    ]
    if nonfinite:
        raise ValueError(f"{label} contains non-finite tensors: {nonfinite[:5]}")


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(b"heatmapvln-lora-adapter-fp32-v1\0")
    for name, value in sorted(state.items()):
        tensor = value.float().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def validate_stage3_config(
    config: dict[str, Any],
    *,
    expected_adapter_hidden_dim: int | None = None,
) -> dict[str, Any]:
    stage = _first_stage(config)
    errors: list[str] = []

    if stage.get("name") != "stage3":
        errors.append(f"stage name={stage.get('name')!r}, expected 'stage3'")
    trainable = tuple(stage.get("trainable_modules") or ())
    if trainable != _REQUIRED_STAGE3_TRAINABLE:
        errors.append(
            f"trainable_modules={list(trainable)!r}, expected {list(_REQUIRED_STAGE3_TRAINABLE)!r}"
        )
    for key in (
        "strict_trainable_modules",
        "requires_base_checkpoint",
        "require_complete_internnav_system1",
        "base_checkpoint_lora_only",
    ):
        if stage.get(key) is not True:
            errors.append(f"training.stages[0].{key} must be true")

    trajectory = config.get("data", {}).get("trajectory", {})
    if trajectory.get("panoramic_vlm_input") is not True:
        errors.append("data.trajectory.panoramic_vlm_input must be true")
    if trajectory.get("structured_pano_output") is not True:
        errors.append("data.trajectory.structured_pano_output must be true")
    if trajectory.get("trajectory_target_convention") != "internnav_habitat":
        errors.append(
            "data.trajectory.trajectory_target_convention must be "
            "'internnav_habitat'"
        )

    llm = config.get("model", {}).get("llm", {})
    layers = [int(value) for value in (llm.get("lora_layer_indices") or [])]
    if layers != list(range(28)):
        errors.append(f"lora_layer_indices={layers!r}, expected all layers 0..27")
    targets = tuple(llm.get("lora_target_modules") or ())
    if set(targets) != set(_REQUIRED_LORA_TARGETS) or len(targets) != 4:
        errors.append(
            f"lora_target_modules={list(targets)!r}, expected {list(_REQUIRED_LORA_TARGETS)!r}"
        )
    lora_rank = int(llm.get("lora_rank", 0) or 0)
    if lora_rank != 32:
        errors.append(f"lora_rank={lora_rank}, expected 32")
    if llm.get("use_lora") is not True:
        errors.append("model.llm.use_lora must be true")

    nextdit = config.get("model", {}).get("action_head", {}).get("nextdit", {})
    if nextdit.get("enabled") is not True:
        errors.append("model.action_head.nextdit.enabled must be true")
    adapter = nextdit.get("pano_latent_adapter", {})
    if adapter.get("enabled") is not True:
        errors.append("model.action_head.nextdit.pano_latent_adapter.enabled must be true")
    adapter_hidden_dim = int(adapter.get("hidden_dim", 0) or 0)
    if expected_adapter_hidden_dim is not None and adapter_hidden_dim != expected_adapter_hidden_dim:
        errors.append(
            f"pano adapter hidden_dim={adapter_hidden_dim}, expected {expected_adapter_hidden_dim}"
        )

    if errors:
        raise ValueError("Stage3 evaluation config validation failed:\n  - " + "\n  - ".join(errors))

    return {
        "lora_layers": layers,
        "lora_targets": list(_REQUIRED_LORA_TARGETS),
        "lora_rank": lora_rank,
        "llm_hidden_dim": int(llm.get("hidden_dim", 3584)),
        "adapter_hidden_dim": adapter_hidden_dim,
    }


def validate_base_checkpoint(
    checkpoint_path: str | Path,
    config_summary: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = _torch_load(checkpoint_path)
    state = _checkpoint_state(checkpoint, checkpoint_path)
    normalized = {_normalize_state_key(name): value for name, value in state.items()}
    lora_state = {name: value for name, value in normalized.items() if "lora_" in name}
    _assert_finite(lora_state, "base LoRA checkpoint")

    actual: set[tuple[int, str, str]] = set()
    malformed: list[str] = []
    rank_errors: list[str] = []
    expected_rank = int(config_summary["lora_rank"])
    for name, value in lora_state.items():
        match = _LORA_KEY_RE.search(name)
        if match is None:
            malformed.append(name)
            continue
        layer = int(match.group("layer"))
        module = match.group("module")
        side = match.group("side")
        actual.add((layer, module, side))
        rank_axis = 0 if side == "A" else 1
        if value.ndim != 2 or int(value.shape[rank_axis]) != expected_rank:
            rank_errors.append(f"{name}: shape={tuple(value.shape)}")

    expected = {
        (layer, module, side)
        for layer in config_summary["lora_layers"]
        for module in config_summary["lora_targets"]
        for side in ("A", "B")
    }
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if malformed or rank_errors or missing or unexpected or len(lora_state) != len(expected):
        raise ValueError(
            "Base LoRA checkpoint validation failed: "
            f"tensors={len(lora_state)} expected={len(expected)} "
            f"malformed={malformed[:3]} rank_errors={rank_errors[:3]} "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    if checkpoint.get("stage_name") not in (None, "stage1_s2_panoramic_sft"):
        raise ValueError(
            f"Unexpected base checkpoint stage_name={checkpoint.get('stage_name')!r}"
        )
    return {
        "path": str(Path(checkpoint_path).resolve()),
        "stage_name": checkpoint.get("stage_name"),
        "epoch": checkpoint.get("epoch"),
        "state_tensors": len(state),
        "lora_tensors": len(lora_state),
    }


def validate_stage3_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_epoch: int,
    expected_base_checkpoint: str | Path,
    config_summary: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = _torch_load(checkpoint_path)
    errors: list[str] = []
    if checkpoint.get("stage_name") != "stage3":
        errors.append(f"stage_name={checkpoint.get('stage_name')!r}, expected 'stage3'")
    if int(checkpoint.get("epoch", -1)) != int(expected_epoch):
        errors.append(f"epoch={checkpoint.get('epoch')!r}, expected {expected_epoch}")
    if checkpoint.get("batch") is not None:
        errors.append(f"batch={checkpoint.get('batch')!r}; final epoch checkpoint required")

    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        errors.append("checkpoint config is missing")
    else:
        try:
            validate_stage3_config(
                checkpoint_config,
                expected_adapter_hidden_dim=int(config_summary["adapter_hidden_dim"]),
            )
        except ValueError as exc:
            errors.append(str(exc))
        recorded_base = checkpoint_config.get("runtime", {}).get("base_checkpoint", "")
        if os.path.realpath(str(recorded_base)) != os.path.realpath(str(expected_base_checkpoint)):
            errors.append(
                f"runtime.base_checkpoint={recorded_base!r}, expected {str(expected_base_checkpoint)!r}"
            )

    state = _adapter_state(checkpoint, checkpoint_path)
    _assert_finite(state, "Stage3 pano adapter")
    dim = int(config_summary["llm_hidden_dim"])
    hidden_dim = int(config_summary["adapter_hidden_dim"])
    expected_shapes = {
        "mlp.0.weight": (hidden_dim, dim),
        "mlp.0.bias": (hidden_dim,),
        "mlp.3.weight": (dim, hidden_dim),
        "mlp.3.bias": (dim,),
    }
    actual_shapes = {name: tuple(value.shape) for name, value in state.items()}
    if actual_shapes != expected_shapes:
        errors.append(f"adapter shapes={actual_shapes!r}, expected={expected_shapes!r}")

    trainable_state = checkpoint.get("trainable_state_dict")
    if isinstance(trainable_state, dict):
        unexpected_trainable = sorted(
            name
            for name in trainable_state
            if not name.removeprefix("module.").startswith("pano_latent_adapter.")
        )
        if unexpected_trainable:
            errors.append(f"unexpected trainable tensors={unexpected_trainable[:5]}")

    if errors:
        raise ValueError("Stage3 checkpoint validation failed:\n  - " + "\n  - ".join(errors))

    return {
        "path": str(Path(checkpoint_path).resolve()),
        "stage_name": checkpoint.get("stage_name"),
        "epoch": int(checkpoint["epoch"]),
        "adapter_tensors": len(state),
        "adapter_parameters": sum(value.numel() for value in state.values()),
    }


def validate_stop_decision_adapter_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_base_checkpoint: str | Path,
) -> dict[str, Any]:
    checkpoint = _torch_load(checkpoint_path)
    errors: list[str] = []
    if checkpoint.get("schema") != _STOP_DECISION_SCHEMA:
        errors.append(
            f"schema={checkpoint.get('schema')!r}, expected {_STOP_DECISION_SCHEMA!r}"
        )
    if checkpoint.get("adapter_name") != "stop_decision":
        errors.append("adapter_name must be 'stop_decision'")
    policy_kind = str(
        checkpoint.get("policy_kind") or _STOP_DECISION_ADD_AND_VETO_POLICY
    )
    if policy_kind not in {
        _STOP_DECISION_ADD_AND_VETO_POLICY,
        _STOP_DECISION_VETO_ONLY_POLICY,
    }:
        errors.append(f"unsupported policy_kind={policy_kind!r}")
    add_enabled = policy_kind == _STOP_DECISION_ADD_AND_VETO_POLICY

    adapter_config = checkpoint.get("adapter_config")
    if not isinstance(adapter_config, dict):
        errors.append("adapter_config is missing")
        adapter_config = {}
    rank = int(adapter_config.get("rank", 0) or 0)
    alpha = int(adapter_config.get("alpha", 0) or 0)
    layers = [int(value) for value in (adapter_config.get("layer_indices") or [])]
    targets = [str(value) for value in (adapter_config.get("target_modules") or [])]
    dropout = float(adapter_config.get("dropout", float("nan")))
    if rank != 8:
        errors.append(f"adapter rank={rank}, expected 8")
    if alpha != 16:
        errors.append(f"adapter alpha={alpha}, expected 16")
    if layers != list(range(20, 28)):
        errors.append(f"adapter layers={layers}, expected 20..27")
    if targets != list(_REQUIRED_LORA_TARGETS):
        errors.append(
            f"adapter targets={targets}, expected {list(_REQUIRED_LORA_TARGETS)}"
        )
    if dropout != 0.0:
        errors.append(f"adapter dropout={dropout}, expected 0")

    state = checkpoint.get("adapter_state_dict")
    if not isinstance(state, dict):
        errors.append("adapter_state_dict is missing")
        state = {}
    else:
        _assert_finite(state, "STOP-decision adapter")
    actual: set[tuple[int, str, str]] = set()
    malformed: list[str] = []
    rank_errors: list[str] = []
    for name, value in state.items():
        match = _LORA_KEY_RE.search(_normalize_state_key(name))
        if match is None:
            malformed.append(name)
            continue
        layer = int(match.group("layer"))
        module = match.group("module")
        side = match.group("side")
        actual.add((layer, module, side))
        rank_axis = 0 if side == "A" else 1
        if value.ndim != 2 or int(value.shape[rank_axis]) != rank:
            rank_errors.append(f"{name}: shape={tuple(value.shape)}")
    expected = {
        (layer, module, side)
        for layer in range(20, 28)
        for module in _REQUIRED_LORA_TARGETS
        for side in ("A", "B")
    }
    if malformed or rank_errors or actual != expected or len(state) != len(expected):
        errors.append(
            "adapter tensor contract mismatch: "
            f"found={len(state)} expected={len(expected)} "
            f"malformed={malformed[:3]} rank_errors={rank_errors[:3]} "
            f"missing={sorted(expected - actual)[:5]} "
            f"unexpected={sorted(actual - expected)[:5]}"
        )
    recorded_adapter_fingerprint = str(checkpoint.get("adapter_fingerprint") or "")
    actual_adapter_fingerprint = _adapter_fingerprint(state) if state else ""
    if recorded_adapter_fingerprint != actual_adapter_fingerprint:
        errors.append("adapter_fingerprint does not match adapter_state_dict")

    base_contract = checkpoint.get("base_contract")
    if not isinstance(base_contract, dict):
        errors.append("base_contract is missing")
        base_contract = {}
    recorded_base = str(base_contract.get("checkpoint") or "")
    if os.path.realpath(recorded_base) != os.path.realpath(str(expected_base_checkpoint)):
        errors.append(
            f"base checkpoint={recorded_base!r}, expected {str(expected_base_checkpoint)!r}"
        )
    if base_contract.get("default_adapter_name") != "default":
        errors.append("base default_adapter_name must be 'default'")
    if int(base_contract.get("default_lora_tensors", 0) or 0) != 224:
        errors.append("base contract must require 224 navigation LoRA tensors")
    default_fingerprint = str(base_contract.get("default_lora_fingerprint") or "")
    if re.fullmatch(r"[0-9a-f]{64}", default_fingerprint) is None:
        errors.append("base default_lora_fingerprint is invalid")
    recorded_file_sha = str(base_contract.get("checkpoint_file_sha256") or "")
    actual_file_sha = _file_sha256(expected_base_checkpoint)
    if recorded_file_sha != actual_file_sha:
        errors.append("base checkpoint SHA256 does not match STOP-decision contract")

    token_contract = checkpoint.get("token_contract")
    if not isinstance(token_contract, dict):
        errors.append("token_contract is missing")
        token_contract = {}
    classes = tuple(token_contract.get("classes") or ())
    prefix_ids = token_contract.get("prefix_token_ids") or []
    class_ids = token_contract.get("class_token_ids") or []
    patterns = token_contract.get("patterns") or {}
    if token_contract.get("schema") != "heatmapvln-structured-view-token-contract-v1":
        errors.append("structured-view token schema mismatch")
    if classes != _STRUCTURED_VIEW_CLASSES:
        errors.append(f"structured classes={classes}, expected {_STRUCTURED_VIEW_CLASSES}")
    if len(prefix_ids) != 2 or len(class_ids) != 6 or len(set(class_ids)) != 6:
        errors.append("structured-view token ids are not one shared prefix plus six classes")
    if any(
        patterns.get(name) != [*prefix_ids, class_id]
        for name, class_id in zip(_STRUCTURED_VIEW_CLASSES, class_ids)
    ):
        errors.append("structured-view token patterns do not match prefix/class ids")

    thresholds = checkpoint.get("thresholds")
    if not isinstance(thresholds, dict):
        errors.append("thresholds are missing")
        thresholds = {}
    add_threshold = float(thresholds.get("add_stop_threshold", float("nan")))
    veto_threshold = float(thresholds.get("veto_stop_threshold", float("nan")))
    if not (
        math.isfinite(add_threshold)
        and math.isfinite(veto_threshold)
        and 0.0 <= veto_threshold < add_threshold <= 1.0
    ):
        errors.append(
            f"invalid hysteresis thresholds: veto={veto_threshold} add={add_threshold}"
        )
    if thresholds.get("quality_passed") is not True:
        errors.append(
            "STOP-decision validation quality gate did not pass: "
            f"{thresholds.get('quality_violations')!r}"
        )
    add_metrics = thresholds.get("add")
    veto_metrics = thresholds.get("veto")
    if thresholds.get("policy_kind", policy_kind) != policy_kind:
        errors.append("checkpoint and threshold policy_kind values disagree")
    if bool(thresholds.get("add_enabled", add_enabled)) != add_enabled:
        errors.append("checkpoint add-enabled contract disagrees with policy_kind")
    if not add_enabled and add_threshold != 1.0:
        errors.append("veto-only policy must record add_stop_threshold=1.0")
    if not isinstance(add_metrics, dict) or not isinstance(veto_metrics, dict):
        errors.append("STOP-decision add/veto validation metrics are missing")
    else:
        if add_enabled and float(add_metrics.get("recall", 0.0)) < 0.5:
            errors.append("STOP-decision add recall is below 0.5")
        if add_enabled and float(add_metrics.get("false_positive_rate", 1.0)) > 0.0:
            errors.append("STOP-decision add false-positive rate must be zero")
        if float(veto_metrics.get("recall", 0.0)) < 0.98:
            errors.append("STOP-decision veto recall is below 0.98")
        if float(veto_metrics.get("negative_rejection_rate", 0.0)) < 0.2:
            errors.append("STOP-decision veto negative rejection is below 0.2")
    if float(thresholds.get("roc_auc", 0.0)) < 0.75:
        errors.append("STOP-decision validation ROC-AUC is below 0.75")
    if int(thresholds.get("veto_reference_positive_count", 0) or 0) <= 0:
        errors.append("STOP-decision veto calibration has no reference positives")

    training = checkpoint.get("training")
    if not isinstance(training, dict):
        errors.append("STOP-decision training metadata is missing")
    else:
        holdout_fraction = float(training.get("holdout_scene_fraction", 0.0) or 0.0)
        if not 0.0 < holdout_fraction < 1.0:
            errors.append("STOP-decision checkpoint was not validated by held-out scene")
        if float(training.get("ranking_loss_weight", 0.0) or 0.0) <= 0.0:
            errors.append("STOP-decision checkpoint did not use pairwise ranking loss")

    if errors:
        raise ValueError(
            "System2 STOP-decision adapter validation failed:\n  - "
            + "\n  - ".join(errors)
        )
    return {
        "path": str(Path(checkpoint_path).resolve()),
        "policy_kind": policy_kind,
        "add_enabled": add_enabled,
        "adapter_tensors": len(state),
        "adapter_parameters": sum(value.numel() for value in state.values()),
        "adapter_fingerprint": actual_adapter_fingerprint,
        "default_lora_fingerprint": default_fingerprint,
        "add_stop_threshold": add_threshold,
        "veto_stop_threshold": veto_threshold,
        "base_checkpoint_sha256": actual_file_sha,
    }


def validate_stop_head_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_base_checkpoint: str | Path,
) -> dict[str, Any]:
    checkpoint = _torch_load(checkpoint_path)
    errors: list[str] = []
    if checkpoint.get("stage_name") != "system2_stop_head":
        errors.append(
            f"stage_name={checkpoint.get('stage_name')!r}, expected 'system2_stop_head'"
        )
    if checkpoint.get("batch") is not None:
        errors.append(f"batch={checkpoint.get('batch')!r}; final epoch checkpoint required")

    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        errors.append("checkpoint config is missing")
        checkpoint_config = {}
    model_config = checkpoint_config.get("model", {})
    llm_config = model_config.get("llm", {})
    head_config = model_config.get("stop_head", {})
    trajectory_config = checkpoint_config.get("data", {}).get("trajectory", {})
    stage_config = {}
    try:
        stage_config = _first_stage(checkpoint_config)
    except ValueError as exc:
        errors.append(str(exc))

    if head_config.get("enabled") is not True:
        errors.append("model.stop_head.enabled must be true")
    if stage_config.get("train_system2_stop_head") is not True:
        errors.append("training.stages[0].train_system2_stop_head must be true")
    if stage_config.get("base_checkpoint_lora_only") is not True:
        errors.append("training.stages[0].base_checkpoint_lora_only must be true")
    if tuple(stage_config.get("trainable_modules") or ()) != ("stop_head",):
        errors.append("training.stages[0].trainable_modules must be ['stop_head']")
    if trajectory_config.get("sft_include_turns") is not True:
        errors.append("data.trajectory.sft_include_turns must be true")
    if float(trajectory_config.get("system2_stop_path_radius_m", 0.0)) != 3.0:
        errors.append("data.trajectory.system2_stop_path_radius_m must be 3.0")
    if float(
        trajectory_config.get(
            "system2_near_stop_hard_negative_min_goal_distance_m", 0.0
        )
    ) != 4.0:
        errors.append("STOP hard-negative minimum goal distance must be 4.0 m")
    if float(
        trajectory_config.get(
            "system2_near_stop_hard_negative_max_goal_distance_m", 0.0
        )
    ) != 18.0:
        errors.append("STOP hard-negative maximum goal distance must be 18.0 m")
    recorded_base = checkpoint_config.get("runtime", {}).get("base_checkpoint", "")
    if os.path.realpath(str(recorded_base)) != os.path.realpath(str(expected_base_checkpoint)):
        errors.append(
            f"runtime.base_checkpoint={recorded_base!r}, expected {str(expected_base_checkpoint)!r}"
        )

    state = _checkpoint_state(checkpoint, checkpoint_path)
    normalized = {_normalize_state_key(name): value for name, value in state.items()}
    unexpected = sorted(name for name in normalized if not name.startswith("stop_head."))
    if unexpected:
        errors.append(f"unexpected non-head tensors={unexpected[:5]}")
    head_state = {
        name.removeprefix("stop_head."): value
        for name, value in normalized.items()
        if name.startswith("stop_head.")
    }
    _assert_finite(head_state, "System2 STOP-head checkpoint")

    input_dim = int(llm_config.get("hidden_dim", 0) or 0)
    hidden_dim = int(head_config.get("hidden_dim", 0) or 0)
    inner_dim = hidden_dim // 2
    expected_shapes = {
        "classifier.0.weight": (hidden_dim, input_dim),
        "classifier.0.bias": (hidden_dim,),
        "classifier.1.weight": (hidden_dim,),
        "classifier.1.bias": (hidden_dim,),
        "classifier.4.weight": (inner_dim, hidden_dim),
        "classifier.4.bias": (inner_dim,),
        "classifier.5.weight": (inner_dim,),
        "classifier.5.bias": (inner_dim,),
        "classifier.8.weight": (1, inner_dim),
        "classifier.8.bias": (1,),
    }
    actual_shapes = {name: tuple(value.shape) for name, value in head_state.items()}
    if actual_shapes != expected_shapes:
        errors.append(f"STOP-head shapes={actual_shapes!r}, expected={expected_shapes!r}")
    legacy_threshold = float(head_config.get("inference_threshold", 0.5))
    add_threshold = float(head_config.get("add_stop_threshold", legacy_threshold))
    veto_threshold = float(head_config.get("veto_stop_threshold", legacy_threshold))
    if not 0.0 <= veto_threshold < add_threshold <= 1.0:
        errors.append(
            "invalid STOP hysteresis thresholds: "
            f"veto={veto_threshold} add={add_threshold}"
        )
    validation_config = checkpoint_config.get("validation", {})
    if validation_config.get("enabled") is not True:
        errors.append("validation.enabled must be true for calibrated STOP checkpoints")
    if not 0.0 < float(validation_config.get("holdout_clip_fraction", 0.0)) < 1.0:
        errors.append("validation.holdout_clip_fraction must be in (0, 1)")
    metrics = checkpoint.get("metrics", {})
    if "val_stop_add_stop_threshold" not in metrics:
        errors.append("checkpoint metrics lack val_stop_add_stop_threshold")
    if "val_stop_veto_stop_threshold" not in metrics:
        errors.append("checkpoint metrics lack val_stop_veto_stop_threshold")
    if "val_stop_add_stop_threshold" in metrics and not math.isclose(
        float(metrics["val_stop_add_stop_threshold"]),
        add_threshold,
        abs_tol=1e-8,
    ):
        errors.append("checkpoint add threshold does not match validation metrics")
    if "val_stop_veto_stop_threshold" in metrics and not math.isclose(
        float(metrics["val_stop_veto_stop_threshold"]),
        veto_threshold,
        abs_tol=1e-8,
    ):
        errors.append("checkpoint veto threshold does not match validation metrics")

    if errors:
        raise ValueError("System2 STOP-head checkpoint validation failed:\n  - " + "\n  - ".join(errors))
    return {
        "path": str(Path(checkpoint_path).resolve()),
        "stage_name": checkpoint.get("stage_name"),
        "epoch": checkpoint.get("epoch"),
        "head_tensors": len(head_state),
        "head_parameters": sum(value.numel() for value in head_state.values()),
        "inference_threshold": legacy_threshold,
        "add_stop_threshold": add_threshold,
        "veto_stop_threshold": veto_threshold,
    }


def validate_temporal_stop_verifier_checkpoint(
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    from src.models.action.temporal_stop_verifier import (
        TEMPORAL_STOP_FEATURE_NAMES,
        TEMPORAL_STOP_FEATURE_SCHEMA,
    )

    checkpoint = _torch_load(checkpoint_path)
    errors: list[str] = []
    stage_name = checkpoint.get("stage_name")
    supported_stages = {
        "system2_temporal_stop_verifier",
        "system2_temporal_stop_verifier_ensemble",
    }
    if stage_name not in supported_stages:
        errors.append(
            "stage_name="
            f"{stage_name!r}, expected one of {sorted(supported_stages)!r}"
        )
    is_ensemble = stage_name == "system2_temporal_stop_verifier_ensemble"
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        errors.append("checkpoint config is missing")
        config = {}
    verifier_config = config.get("temporal_stop_verifier")
    static_spec = config.get("source_static_stop_head")
    if not isinstance(verifier_config, dict):
        errors.append("temporal_stop_verifier config is missing")
        verifier_config = {}
    if not isinstance(static_spec, dict):
        errors.append("source_static_stop_head config is missing")
        static_spec = {}
    if verifier_config.get("schema") != TEMPORAL_STOP_FEATURE_SCHEMA:
        errors.append("temporal feature schema does not match runtime")
    if tuple(verifier_config.get("feature_names") or ()) != TEMPORAL_STOP_FEATURE_NAMES:
        errors.append("temporal feature names do not match runtime")
    if verifier_config.get("veto_only") is not True:
        errors.append("temporal verifier must be veto_only=true")
    if verifier_config.get("requires_contiguous_zero_based_calls") is not True:
        errors.append("temporal verifier lacks contiguous-call history contract")
    threshold: float | None = None
    thresholds: list[float] = []
    ensemble_size = 0
    if is_ensemble:
        if verifier_config.get("architecture") != "scene_fold_unanimous_ensemble":
            errors.append("temporal ensemble architecture is unsupported")
        if verifier_config.get("aggregation") != "unanimous":
            errors.append("temporal ensemble aggregation must be unanimous")
        ensemble_size = int(verifier_config.get("ensemble_size", 0) or 0)
        if ensemble_size < 2:
            errors.append(f"invalid temporal ensemble_size={ensemble_size}")
        raw_thresholds = verifier_config.get("acceptance_thresholds")
        if not isinstance(raw_thresholds, list):
            errors.append("temporal ensemble acceptance_thresholds must be a list")
        else:
            thresholds = [float(value) for value in raw_thresholds]
            if len(thresholds) != ensemble_size:
                errors.append(
                    "temporal ensemble threshold count does not match ensemble_size"
                )
            if any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in thresholds
            ):
                errors.append("temporal ensemble thresholds must be finite and in [0, 1]")
    else:
        threshold = float(verifier_config.get("acceptance_threshold", float("nan")))
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            errors.append(f"invalid temporal acceptance threshold={threshold}")

    raw_verifier_state = checkpoint.get("trainable_state_dict")
    if not isinstance(raw_verifier_state, dict):
        errors.append("temporal verifier state dict is missing")
        raw_verifier_state = {}
    normalized_verifier = {
        _normalize_state_key(name): value for name, value in raw_verifier_state.items()
    }
    state_prefix = (
        "temporal_stop_ensemble."
        if is_ensemble
        else "temporal_stop_verifier."
    )
    unexpected = sorted(
        name
        for name in normalized_verifier
        if not name.startswith(state_prefix)
    )
    if unexpected:
        errors.append(f"unexpected temporal checkpoint tensors={unexpected[:5]}")
    verifier_state = {
        name.removeprefix(state_prefix): value
        for name, value in normalized_verifier.items()
        if name.startswith(state_prefix)
    }
    _assert_finite(verifier_state, "Temporal STOP verifier")
    feature_dim = len(TEMPORAL_STOP_FEATURE_NAMES)
    if is_ensemble:
        hidden_dim = int(verifier_config.get("member_hidden_dim", 0) or 0)
        expected_verifier_shapes = {"acceptance_thresholds": (ensemble_size,)}
        for member_index in range(ensemble_size):
            prefix = f"members.{member_index}."
            expected_verifier_shapes.update(
                {
                    f"{prefix}feature_mean": (feature_dim,),
                    f"{prefix}feature_scale": (feature_dim,),
                    f"{prefix}classifier.0.weight": (hidden_dim, feature_dim),
                    f"{prefix}classifier.0.bias": (hidden_dim,),
                    f"{prefix}classifier.3.weight": (1, hidden_dim),
                    f"{prefix}classifier.3.bias": (1,),
                }
            )
    else:
        hidden_dim = int(verifier_config.get("hidden_dim", 0) or 0)
        expected_verifier_shapes = {
            "feature_mean": (feature_dim,),
            "feature_scale": (feature_dim,),
            "classifier.0.weight": (hidden_dim, feature_dim),
            "classifier.0.bias": (hidden_dim,),
            "classifier.3.weight": (1, hidden_dim),
            "classifier.3.bias": (1,),
        }
    actual_verifier_shapes = {
        name: tuple(value.shape) for name, value in verifier_state.items()
    }
    if actual_verifier_shapes != expected_verifier_shapes:
        errors.append(
            "temporal verifier shapes="
            f"{actual_verifier_shapes!r}, expected={expected_verifier_shapes!r}"
        )
    if is_ensemble and torch.is_tensor(verifier_state.get("acceptance_thresholds")):
        state_thresholds = verifier_state["acceptance_thresholds"].float().flatten()
        config_thresholds = torch.tensor(thresholds, dtype=torch.float32)
        if state_thresholds.shape != config_thresholds.shape or not torch.allclose(
            state_thresholds,
            config_thresholds,
            rtol=0.0,
            atol=1e-7,
        ):
            errors.append("temporal ensemble state thresholds do not match config")

    raw_static_state = checkpoint.get("source_static_stop_head_state_dict")
    if not isinstance(raw_static_state, dict):
        errors.append("embedded frozen static STOP prior is missing")
        raw_static_state = {}
    normalized_static = {
        _normalize_state_key(name): value for name, value in raw_static_state.items()
    }
    unexpected_static = sorted(
        name for name in normalized_static if not name.startswith("stop_head.")
    )
    if unexpected_static:
        errors.append(f"unexpected embedded static tensors={unexpected_static[:5]}")
    static_state = {
        name.removeprefix("stop_head."): value
        for name, value in normalized_static.items()
        if name.startswith("stop_head.")
    }
    _assert_finite(static_state, "Temporal STOP embedded static prior")
    static_input_dim = int(static_spec.get("input_dim", 0) or 0)
    static_hidden_dim = int(static_spec.get("hidden_dim", 0) or 0)
    static_inner_dim = static_hidden_dim // 2
    expected_static_shapes = {
        "classifier.0.weight": (static_hidden_dim, static_input_dim),
        "classifier.0.bias": (static_hidden_dim,),
        "classifier.1.weight": (static_hidden_dim,),
        "classifier.1.bias": (static_hidden_dim,),
        "classifier.4.weight": (static_inner_dim, static_hidden_dim),
        "classifier.4.bias": (static_inner_dim,),
        "classifier.5.weight": (static_inner_dim,),
        "classifier.5.bias": (static_inner_dim,),
        "classifier.8.weight": (1, static_inner_dim),
        "classifier.8.bias": (1,),
    }
    actual_static_shapes = {
        name: tuple(value.shape) for name, value in static_state.items()
    }
    if actual_static_shapes != expected_static_shapes:
        errors.append(
            "embedded static prior shapes="
            f"{actual_static_shapes!r}, expected={expected_static_shapes!r}"
        )
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("temporal verifier metrics are missing")
    elif is_ensemble:
        oof_metrics = metrics.get("oof")
        fold_metrics = metrics.get("folds")
        if not isinstance(oof_metrics, dict):
            errors.append("temporal ensemble OOF metrics are missing")
        else:
            oof_recall = float(oof_metrics.get("recall", float("nan")))
            oof_fpr = float(
                oof_metrics.get("false_positive_rate", float("nan"))
            )
            if not math.isfinite(oof_recall) or oof_recall < 0.75:
                errors.append(f"temporal ensemble OOF recall is too low: {oof_recall}")
            if not math.isfinite(oof_fpr) or oof_fpr > 0.1:
                errors.append(f"temporal ensemble OOF FPR is too high: {oof_fpr}")
        if not isinstance(fold_metrics, list) or len(fold_metrics) != ensemble_size:
            errors.append("temporal ensemble fold metrics do not match ensemble_size")
    elif not math.isclose(
        float(metrics.get("acceptance_threshold", float("nan"))),
        float(threshold),
        abs_tol=1e-8,
    ):
        errors.append("temporal threshold does not match checkpoint metrics")
    training = checkpoint.get("training")
    if not isinstance(training, dict) or training.get("scene_disjoint") is not True:
        errors.append("temporal verifier must record a scene-disjoint validation split")
    elif is_ensemble and int(training.get("fold_count", 0) or 0) != ensemble_size:
        errors.append("temporal ensemble training fold_count does not match config")

    if errors:
        raise ValueError(
            "System2 temporal STOP verifier validation failed:\n  - "
            + "\n  - ".join(errors)
        )
    result = {
        "path": str(Path(checkpoint_path).resolve()),
        "stage_name": stage_name,
        "epoch": checkpoint.get("epoch"),
        "feature_dim": feature_dim,
        "verifier_tensors": len(verifier_state),
        "static_prior_tensors": len(static_state),
        "veto_only": True,
    }
    if is_ensemble:
        result.update(
            {
                "architecture": "scene_fold_unanimous_ensemble",
                "aggregation": "unanimous",
                "ensemble_size": ensemble_size,
                "acceptance_thresholds": thresholds,
            }
        )
    else:
        result["acceptance_threshold"] = threshold
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--stage3-checkpoint", required=True)
    parser.add_argument("--system2-stop-head-checkpoint", default=None)
    parser.add_argument("--system2-stop-decision-adapter-checkpoint", default=None)
    parser.add_argument("--system2-temporal-stop-verifier-checkpoint", default=None)
    parser.add_argument("--expected-epoch", type=int, default=2)
    parser.add_argument("--expected-adapter-hidden-dim", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.system2_stop_decision_adapter_checkpoint and (
        args.system2_stop_head_checkpoint
        or args.system2_temporal_stop_verifier_checkpoint
    ):
        raise ValueError(
            "STOP-decision adapter cannot be combined with static/temporal STOP policies"
        )
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Config must contain a mapping: {args.config}")

    config_summary = validate_stage3_config(
        config,
        expected_adapter_hidden_dim=args.expected_adapter_hidden_dim,
    )
    result = {
        "status": "passed",
        "config": config_summary,
        "base_checkpoint": validate_base_checkpoint(args.base_checkpoint, config_summary),
        "stage3_checkpoint": validate_stage3_checkpoint(
            args.stage3_checkpoint,
            expected_epoch=args.expected_epoch,
            expected_base_checkpoint=args.base_checkpoint,
            config_summary=config_summary,
        ),
    }
    if args.system2_stop_head_checkpoint:
        result["system2_stop_head_checkpoint"] = validate_stop_head_checkpoint(
            args.system2_stop_head_checkpoint,
            expected_base_checkpoint=args.base_checkpoint,
        )
    if args.system2_stop_decision_adapter_checkpoint:
        result["system2_stop_decision_adapter_checkpoint"] = (
            validate_stop_decision_adapter_checkpoint(
                args.system2_stop_decision_adapter_checkpoint,
                expected_base_checkpoint=args.base_checkpoint,
            )
        )
        result["system2_stop_policy_mode"] = "isolated_stop_decision_lora"
    if args.system2_temporal_stop_verifier_checkpoint:
        result["system2_temporal_stop_verifier_checkpoint"] = (
            validate_temporal_stop_verifier_checkpoint(
                args.system2_temporal_stop_verifier_checkpoint
            )
        )
    if args.system2_stop_head_checkpoint and args.system2_temporal_stop_verifier_checkpoint:
        result["system2_stop_policy_mode"] = "hybrid_static_add_temporal_veto"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

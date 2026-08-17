#!/usr/bin/env python3
"""Strict B=1 two-stage visual-identity pilot for pose-free grounding.

The protocol is deliberately narrower than the earlier joint pilot:

1. ``head-warmup`` starts from the pinned Stage1-S2 LoRA, trains only the
   shared pose-free matcher, and proves that every LoRA tensor stayed exact.
2. ``lora-identity`` and ``lora-heatmap-control`` each start again from the
   original Stage1-S2 LoRA, load the *same* warmup head, freeze that head
   bitwise, and train reachable LoRA tensors only.  The identity arm optimizes
   ``base + global_panorama_pixel_ce + 2 * target_identity``; the control
   receives the same base and panorama terms and omits only identity.

Every K=4 sample is executed as four physically separate B=1 Qwen forwards.
The history query is the equal-weight mean of the four visual occurrences of
that history panorama at LLM layer 20.  Pose, frame index, temporal slot, and
trajectory values never enter the model.  Raw matcher logits are explicitly
requested so target-grounded identity supervision remains connected to LoRA.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import logging
import math
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.tools.train_pose_free_multihistory_pilot as base_pilot
from scripts.training import safe_torch_load

from src.models.heatmap import HeatmapVLNLoss, TargetGroundedPanoramaIdentityLoss

LOGGER = logging.getLogger("pose_free_visual_identity_pilot")

CHECKPOINT_SCHEMA = "pose_free_visual_identity_checkpoint_v3"
REPORT_SCHEMA = "pose_free_visual_identity_report_v3"
PROTOCOL = "strict_b1_visual_identity_two_stage_v3"
HISTORY_QUERY_SOURCE = "history_visual_equal_view_mean_v1"
TRAIN_MODES = ("head-warmup", "lora-identity", "lora-heatmap-control")
EXPECTED_LORA_TENSORS = 224
EXPECTED_TRAINABLE_LORA_TENSORS = 168
EXPECTED_TRAINABLE_LORA_LAYERS = tuple(range(21))
IDENTITY_WEIGHT = 2.0
PANORAMA_WEIGHT = 1.0
MIN_TARGET_SEPARATION = 12.0
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-mode", choices=TRAIN_MODES, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Pinned original Stage1-S2 checkpoint containing all 224 LoRA tensors.",
    )
    parser.add_argument("--warmup-checkpoint", default=None)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--source-inventory-sha256", default=None)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-trainable-lora-layer", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args(argv)


def mode_trains_lora(train_mode: str) -> bool:
    if train_mode not in TRAIN_MODES:
        raise ValueError(f"Unknown train mode: {train_mode}")
    return train_mode != "head-warmup"


def validate_args(args: argparse.Namespace) -> None:
    if args.train_mode not in TRAIN_MODES:
        raise ValueError(f"Unknown train mode: {args.train_mode}")
    if args.train_mode == "head-warmup" and args.warmup_checkpoint is not None:
        raise ValueError("head-warmup must start from Stage1-S2 and forbids --warmup-checkpoint")
    if mode_trains_lora(args.train_mode) and not args.warmup_checkpoint:
        raise ValueError(f"{args.train_mode} requires --warmup-checkpoint from head-warmup")
    if args.train_steps <= 0:
        raise ValueError("The visual-identity pilot requires at least one training step")
    if args.grad_clip <= 0 or args.log_every <= 0:
        raise ValueError("--grad-clip and --log-every must be positive")
    if args.head_learning_rate <= 0 or args.lora_learning_rate <= 0:
        raise ValueError("Learning rates must be positive")
    if args.max_trainable_lora_layer != 20:
        raise ValueError("The strict attribution contract fixes the deepest reachable LoRA layer at 20")


def _legacy_args(args: argparse.Namespace) -> argparse.Namespace:
    """Adapt the new state machine to audited dataset/model construction helpers."""

    adapted = copy.copy(args)
    adapted.branch = "heatmap-lora" if mode_trains_lora(args.train_mode) else "head-only"
    return adapted


def load_visual_identity_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = base_pilot.load_pilot_config(_legacy_args(args))
    heatmap_cfg = cfg["model"]["heatmap"]
    pose_free_cfg = heatmap_cfg.setdefault("pose_free", {})
    pose_free_cfg["history_query_source"] = HISTORY_QUERY_SOURCE
    # Make the intended graph policy explicit even if the input YAML contains
    # stale settings from an earlier pilot.
    heatmap_cfg["heatmap_trains_backbone"] = True
    cfg["model"]["llm"]["gradient_checkpointing"] = mode_trains_lora(args.train_mode)
    return cfg


def visual_identity_config_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    contract = dict(base_pilot.pose_free_config_contract(cfg))
    source = cfg["model"]["heatmap"].get("pose_free", {}).get("history_query_source")
    if source != HISTORY_QUERY_SOURCE:
        raise RuntimeError(f"Visual identity query source mismatch: expected={HISTORY_QUERY_SOURCE} actual={source}")
    contract.update(
        {
            "protocol": PROTOCOL,
            "history_query_source": source,
            "history_query_layer": 20,
            "history_visual_views_per_query": 4,
            "history_visual_view_reduction": "equal_weight_mean",
            "raw_heatmap_logits_required": True,
        }
    )
    return contract


def load_visual_identity_manifest_contract(
    args: argparse.Namespace,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    records, contract = base_pilot.load_manifest_contract(_legacy_args(args))
    with Path(args.selection_manifest).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    parameters = manifest.get("selection_parameters", {})
    minimum = float(parameters.get("min_target_separation_pixels", -1.0))
    if minimum + 1e-12 < MIN_TARGET_SEPARATION:
        raise RuntimeError(
            "Identity supervision requires the manifest to guarantee at least "
            f"{MIN_TARGET_SEPARATION:.1f}px target separation; got {minimum}"
        )
    derivation = manifest.get("derivation_parameters")
    if isinstance(derivation, dict):
        if derivation.get("constraints_relaxed") is not False or derivation.get("constraint_overrides") != {}:
            raise RuntimeError("Derived manifest relaxed or overrode parent constraints")
    contract["minimum_target_separation_pixels"] = minimum
    contract["identity_targets_per_sample"] = 4
    return records, contract


def assert_visual_runtime_contract(model: torch.nn.Module) -> dict[str, Any]:
    contract = dict(base_pilot.assert_runtime_model_contract(model))
    heatmap = model.heatmap_vln
    source = getattr(heatmap, "history_query_source", None)
    feature_source = getattr(heatmap.feat_extractor, "history_query_source", None)
    if source != HISTORY_QUERY_SOURCE or feature_source != HISTORY_QUERY_SOURCE:
        raise RuntimeError(
            f"Materialized visual-query source mismatch: heatmap={source!r} feature_extractor={feature_source!r}"
        )
    forward_parameters = inspect.signature(model.forward).parameters
    if "return_heatmap_logits" not in forward_parameters:
        raise RuntimeError("VLNPipeline.forward does not expose raw heatmap-logit opt-in")
    contract.update(
        {
            "protocol": PROTOCOL,
            "history_query_source": source,
            "history_query_layer": 20,
            "history_visual_views_per_query": 4,
            "qwen_forward_batch_size": 1,
            "qwen_forwards_per_sample": 4,
            "raw_heatmap_logits_opt_in": "return_heatmap_logits=True",
        }
    )
    return contract


def materialize_visual_identity_model(
    args: argparse.Namespace,
    cfg: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    legacy = _legacy_args(args)
    model, stage1_contract, _old_runtime = base_pilot.materialize_model(legacy, cfg)
    runtime_contract = assert_visual_runtime_contract(model)
    return model, stage1_contract, runtime_contract


def configure_training_state(
    model: torch.nn.Module,
    train_mode: str,
    *,
    max_lora_layer: int = 20,
) -> tuple[dict[str, torch.nn.Parameter], dict[str, torch.nn.Parameter]]:
    """Freeze everything, then enable exactly one side of the attribution split."""

    if train_mode not in TRAIN_MODES:
        raise ValueError(f"Unknown train mode: {train_mode}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    matcher = model.heatmap_vln.pose_free_matcher
    if matcher is None:
        raise RuntimeError("Visual identity pilot requires a materialized pose-free matcher")
    head = {name: parameter for name, parameter in matcher.named_parameters()}
    all_lora = base_pilot.normalized_lora_parameters(model)
    if len(all_lora) != EXPECTED_LORA_TENSORS:
        raise RuntimeError(f"Expected {EXPECTED_LORA_TENSORS} LoRA tensors, got {len(all_lora)}")

    trainable_head: dict[str, torch.nn.Parameter] = {}
    trainable_lora: dict[str, torch.nn.Parameter] = {}
    if train_mode == "head-warmup":
        for name, parameter in head.items():
            parameter.requires_grad_(True)
            trainable_head[name] = parameter
    else:
        for name, parameter in all_lora.items():
            match = base_pilot.LORA_LAYER_RE.search(name)
            if match is None:
                raise RuntimeError(f"Cannot parse LoRA layer from {name}")
            if int(match.group(1)) <= max_lora_layer:
                parameter.requires_grad_(True)
                trainable_lora[name] = parameter
        if not trainable_lora:
            raise RuntimeError(f"{train_mode} has no reachable trainable LoRA tensors")
        layers = sorted({int(base_pilot.LORA_LAYER_RE.search(name).group(1)) for name in trainable_lora})
        if len(trainable_lora) != EXPECTED_TRAINABLE_LORA_TENSORS:
            raise RuntimeError(
                "Reachable LoRA tensor count differs from the registered protocol: "
                f"{len(trainable_lora)} != {EXPECTED_TRAINABLE_LORA_TENSORS}"
            )
        if layers != list(EXPECTED_TRAINABLE_LORA_LAYERS):
            raise RuntimeError(
                "Reachable LoRA layers differ from the registered protocol: "
                f"{layers} != {list(EXPECTED_TRAINABLE_LORA_LAYERS)}"
            )

    if any(parameter.requires_grad for parameter in head.values()) != (train_mode == "head-warmup"):
        raise RuntimeError("Head trainability does not match the two-stage state machine")
    if any(parameter.requires_grad for parameter in all_lora.values()) != mode_trains_lora(train_mode):
        raise RuntimeError("LoRA trainability does not match the two-stage state machine")
    return trainable_head, trainable_lora


def trainable_lora_layers(parameters: Mapping[str, torch.nn.Parameter]) -> list[int]:
    layers: set[int] = set()
    for name in parameters:
        match = base_pilot.LORA_LAYER_RE.search(name)
        if match is None:
            raise RuntimeError(f"Cannot parse LoRA layer from {name}")
        layers.add(int(match.group(1)))
    return sorted(layers)


def build_optimization_contract(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    trainable_head: Mapping[str, torch.nn.Parameter],
    trainable_lora: Mapping[str, torch.nn.Parameter],
) -> dict[str, Any]:
    lora_mode = mode_trains_lora(args.train_mode)
    expected_lora_tensors = EXPECTED_TRAINABLE_LORA_TENSORS if lora_mode else 0
    expected_lora_layers = list(EXPECTED_TRAINABLE_LORA_LAYERS) if lora_mode else []
    contract = {
        "optimizer": {
            "name": "AdamW",
            "betas": list(ADAMW_BETAS),
            "eps": ADAMW_EPS,
            "weight_decay": float(args.weight_decay),
            "amsgrad": False,
        },
        "train_steps": int(args.train_steps),
        "seed": int(args.seed),
        "learning_rates": {
            "head": float(args.head_learning_rate),
            "lora": float(args.lora_learning_rate),
            "active_group": "reachable_lora" if lora_mode else "pose_free_matcher_warmup",
            "active": float(args.lora_learning_rate if lora_mode else args.head_learning_rate),
        },
        "grad_clip": float(args.grad_clip),
        "max_trainable_lora_layer": int(args.max_trainable_lora_layer),
        "gradient_checkpointing": bool(cfg["model"]["llm"]["gradient_checkpointing"]),
        "protocol_reachable_lora_tensors": EXPECTED_TRAINABLE_LORA_TENSORS,
        "protocol_reachable_lora_layers": list(EXPECTED_TRAINABLE_LORA_LAYERS),
        "expected_trainable_lora_tensors": expected_lora_tensors,
        "actual_trainable_lora_tensors": len(trainable_lora),
        "expected_trainable_lora_layers": expected_lora_layers,
        "actual_trainable_lora_layers": trainable_lora_layers(trainable_lora),
        "actual_trainable_head_tensors": len(trainable_head),
        "qwen_forward_batch_size": 1,
        "qwen_forwards_per_sample": 4,
    }
    validate_optimization_contract_strict(
        contract,
        expected_train_mode=args.train_mode,
        expected_step=args.train_steps,
    )
    return contract


def validate_optimization_contract_strict(
    contract: Any,
    *,
    expected_train_mode: str,
    expected_step: int | None = None,
) -> None:
    if not isinstance(contract, dict):
        raise RuntimeError("Visual identity checkpoint optimization_contract is missing")
    required = {
        "optimizer",
        "train_steps",
        "seed",
        "learning_rates",
        "grad_clip",
        "max_trainable_lora_layer",
        "gradient_checkpointing",
        "protocol_reachable_lora_tensors",
        "protocol_reachable_lora_layers",
        "expected_trainable_lora_tensors",
        "actual_trainable_lora_tensors",
        "expected_trainable_lora_layers",
        "actual_trainable_lora_layers",
        "actual_trainable_head_tensors",
        "qwen_forward_batch_size",
        "qwen_forwards_per_sample",
    }
    if set(contract) != required:
        raise RuntimeError(
            "Visual identity optimization_contract fields differ from v3: "
            f"missing={sorted(required - set(contract))} extra={sorted(set(contract) - required)}"
        )
    optimizer = contract["optimizer"]
    expected_optimizer_fields = {"name", "betas", "eps", "weight_decay", "amsgrad"}
    if not isinstance(optimizer, dict) or set(optimizer) != expected_optimizer_fields:
        raise RuntimeError("Visual identity optimization_contract has invalid AdamW fields")
    if (
        optimizer["name"] != "AdamW"
        or optimizer["betas"] != list(ADAMW_BETAS)
        or float(optimizer["eps"]) != ADAMW_EPS
        or optimizer["amsgrad"] is not False
        or not math.isfinite(float(optimizer["weight_decay"]))
        or float(optimizer["weight_decay"]) < 0
    ):
        raise RuntimeError("Visual identity optimization_contract has invalid AdamW values")
    learning_rates = contract["learning_rates"]
    if not isinstance(learning_rates, dict) or set(learning_rates) != {
        "head",
        "lora",
        "active_group",
        "active",
    }:
        raise RuntimeError("Visual identity optimization_contract has invalid learning-rate fields")
    if any(
        not math.isfinite(float(learning_rates[key])) or float(learning_rates[key]) <= 0
        for key in ("head", "lora", "active")
    ):
        raise RuntimeError("Visual identity optimization_contract learning rates must be positive finite values")
    lora_mode = mode_trains_lora(expected_train_mode)
    expected_group = "reachable_lora" if lora_mode else "pose_free_matcher_warmup"
    expected_active = float(learning_rates["lora"] if lora_mode else learning_rates["head"])
    expected_tensors = EXPECTED_TRAINABLE_LORA_TENSORS if lora_mode else 0
    expected_layers = list(EXPECTED_TRAINABLE_LORA_LAYERS) if lora_mode else []
    checks = {
        "positive train_steps": int(contract["train_steps"]) > 0,
        "checkpoint step": expected_step is None or int(contract["train_steps"]) == int(expected_step),
        "integer seed": isinstance(contract["seed"], int),
        "active group": learning_rates["active_group"] == expected_group,
        "active lr": float(learning_rates["active"]) == expected_active,
        "positive grad clip": math.isfinite(float(contract["grad_clip"])) and float(contract["grad_clip"]) > 0,
        "max LoRA layer": int(contract["max_trainable_lora_layer"]) == 20,
        "gradient checkpointing": contract["gradient_checkpointing"] is lora_mode,
        "protocol LoRA tensors": int(contract["protocol_reachable_lora_tensors"]) == EXPECTED_TRAINABLE_LORA_TENSORS,
        "protocol LoRA layers": contract["protocol_reachable_lora_layers"] == list(EXPECTED_TRAINABLE_LORA_LAYERS),
        "expected LoRA tensors": int(contract["expected_trainable_lora_tensors"]) == expected_tensors,
        "actual LoRA tensors": int(contract["actual_trainable_lora_tensors"]) == expected_tensors,
        "expected LoRA layers": contract["expected_trainable_lora_layers"] == expected_layers,
        "actual LoRA layers": contract["actual_trainable_lora_layers"] == expected_layers,
        "head tensor count": int(contract["actual_trainable_head_tensors"]) > 0
        if not lora_mode
        else int(contract["actual_trainable_head_tensors"]) == 0,
        "B1 batch": int(contract["qwen_forward_batch_size"]) == 1,
        "four forwards": int(contract["qwen_forwards_per_sample"]) == 4,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Invalid visual identity optimization_contract: " + ", ".join(failed))


def expected_loss_contract(train_mode: str) -> dict[str, Any]:
    return {
        "base_weight": 1.0,
        "identity_weight": IDENTITY_WEIGHT if train_mode == "lora-identity" else 0.0,
        "panorama_weight": PANORAMA_WEIGHT if mode_trains_lora(train_mode) else 0.0,
        "panorama_objective": "global_raw_heatmap_pixel_ce",
        "view_readout": "raw_heatmap_spatial_logsumexp_marginal",
        "control_differs_only_by_identity_term": True,
    }


def regroup_visual_identity_outputs(
    outputs: list[dict[str, torch.Tensor]],
    *,
    num_histories: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Regroup four B=1 predictions without copying or detaching any graph."""

    visibility, heatmaps = base_pilot.regroup_isolated_pair_outputs(
        outputs,
        num_histories=num_histories,
    )
    logits_rows: list[torch.Tensor] = []
    for chain_index, output in enumerate(outputs):
        if "heatmap_logits" not in output:
            raise RuntimeError(f"Isolated chain {chain_index} omitted explicitly requested raw heatmap_logits")
        logits = output["heatmap_logits"]
        if logits.ndim != 5 or tuple(logits.shape[:3]) != (1, 1, 4):
            raise RuntimeError(
                f"Isolated chain {chain_index} heatmap_logits must be [1,1,4,H,W], got {tuple(logits.shape)}"
            )
        if tuple(logits.shape) != tuple(output["heatmaps"].shape):
            raise RuntimeError(f"Isolated chain {chain_index} raw/probability heatmap shapes differ")
        logits_rows.append(logits[:, 0])
    heatmap_logits = torch.stack(logits_rows, dim=1)
    return visibility, heatmaps, heatmap_logits


def compose_training_loss(
    train_mode: str,
    base_total: torch.Tensor,
    identity_output: Mapping[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the preregistered loss contrast exactly, with no hidden weights."""

    if base_total.ndim != 0 or not torch.isfinite(base_total):
        raise ValueError("Base heatmap total must be one finite scalar")
    components = {
        "base": base_total,
        "identity": base_total.new_zeros(()),
        "panorama": base_total.new_zeros(()),
    }
    if mode_trains_lora(train_mode):
        if identity_output is None:
            raise RuntimeError(f"{train_mode} requires target-grounded panorama outputs")
        identity = identity_output.get("identity_loss")
        panorama = identity_output.get("panorama_loss")
        if not torch.is_tensor(identity) or not torch.is_tensor(panorama):
            raise RuntimeError("Identity criterion omitted identity_loss or panorama_loss")
        if identity.ndim != 0 or panorama.ndim != 0 or not torch.isfinite(identity) or not torch.isfinite(panorama):
            raise RuntimeError("Identity and panorama losses must be finite scalars")
        components["identity"] = identity if train_mode == "lora-identity" else identity.new_zeros(())
        components["panorama"] = panorama
        auxiliary = PANORAMA_WEIGHT * panorama
        if train_mode == "lora-identity":
            auxiliary = auxiliary + IDENTITY_WEIGHT * identity
            reported_total = identity_output.get("total")
            if not torch.is_tensor(reported_total) or not torch.allclose(
                reported_total,
                auxiliary,
                rtol=1e-6,
                atol=1e-7,
            ):
                raise RuntimeError(
                    "TargetGroundedPanoramaIdentityLoss weighting differs from the registered 2:1 contract"
                )
        total = base_total + auxiliary
    else:
        if identity_output is not None:
            raise RuntimeError(f"{train_mode} must not consume the identity/panorama objective")
        total = base_total
    components["total"] = total
    return total, components


def audit_lora_objective_gradient(
    objective: torch.Tensor,
    trainable_lora: Mapping[str, torch.nn.Parameter],
    *,
    objective_label: str,
    require_nonzero: bool,
) -> dict[str, Any]:
    """Audit one loss component without consuming the total graph."""

    if objective.ndim != 0 or not torch.isfinite(objective):
        raise RuntimeError(f"{objective_label} audit requires one finite scalar")
    if not trainable_lora:
        raise RuntimeError(f"{objective_label} audit received no trainable LoRA tensors")
    names = list(trainable_lora)
    gradients = torch.autograd.grad(
        objective,
        tuple(trainable_lora.values()),
        retain_graph=True,
        allow_unused=True,
    )
    square_sum = 0.0
    tensors_with_grad = 0
    nonzero_names: list[str] = []
    for name, gradient in zip(names, gradients, strict=True):
        if gradient is None:
            continue
        tensors_with_grad += 1
        norm = float(gradient.detach().float().norm().item())
        square_sum += norm * norm
        if norm > 0:
            nonzero_names.append(name)
    nonzero_layers = sorted({int(base_pilot.LORA_LAYER_RE.search(name).group(1)) for name in nonzero_names})
    summary = {
        "method": "torch.autograd.grad",
        "retain_graph": True,
        "allow_unused": True,
        "objective": objective_label,
        "require_nonzero": require_nonzero,
        "requested_tensors": len(names),
        "tensors_with_grad": tensors_with_grad,
        "tensors_with_nonzero_grad": len(nonzero_names),
        "layers_with_nonzero_grad": nonzero_layers,
        "total_grad_norm": math.sqrt(square_sum),
    }
    summary["nonzero_gradient_reached"] = bool(nonzero_names and summary["total_grad_norm"] > 0)
    if require_nonzero and not summary["nonzero_gradient_reached"]:
        raise RuntimeError(f"{objective_label} produced zero reachable LoRA gradient")
    return summary


def audit_identity_auxiliary_gradient(
    auxiliary: torch.Tensor,
    trainable_lora: Mapping[str, torch.nn.Parameter],
) -> dict[str, Any]:
    """Audit ``2*identity + panorama`` alone without consuming total backward."""

    return audit_lora_objective_gradient(
        auxiliary,
        trainable_lora,
        objective_label="2 * target_grounded_identity + global_panorama_pixel_ce",
        require_nonzero=True,
    )


def assert_four_chain_current_prefix_identity(chains: Mapping[str, Any]) -> dict[str, Any]:
    """Prove that all isolated forwards receive one exact current prefix."""

    checks: dict[str, Any] = {}
    tensors = {
        "current_observation": chains.get("current_observation"),
        "current_views": chains.get("current_views"),
    }
    video_frames = chains.get("video_frames")
    if not torch.is_tensor(video_frames) or tuple(video_frames.shape[:2]) != (4, 2):
        raise RuntimeError("Visual identity requires four two-frame isolated chains")
    tensors["video_current_frame"] = video_frames[:, 1]
    for name, tensor in tensors.items():
        if not torch.is_tensor(tensor) or int(tensor.shape[0]) != 4:
            raise RuntimeError(f"Current-prefix identity gate received invalid {name}")
        if not torch.equal(tensor, tensor[:1].expand_as(tensor)):
            raise RuntimeError(f"Current-prefix differs across isolated B=1 chains: {name}")
        checks[name] = {"shape": list(tensor.shape), "bitwise_identical_across_four_chains": True}
    return {"passed": True, "bitwise_exact": True, "tensors": checks}


def forward_visual_identity_loss(
    model: torch.nn.Module,
    base_criterion: HeatmapVLNLoss,
    identity_criterion: TargetGroundedPanoramaIdentityLoss,
    transformed: dict[str, Any],
    device: torch.device,
    *,
    train_mode: str,
    history_rel_poses: torch.Tensor | None = None,
    audit_current_patches: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Execute 4xB=1 and compute the selected training objective."""

    if history_rel_poses is not None:
        raise ValueError("Visual identity pilot received non-None history_rel_poses")
    chains = base_pilot.flatten_isolated_pair_chains(transformed)
    current_prefix_gate = assert_four_chain_current_prefix_identity(chains)
    num_histories = int(transformed["history_panoramas"].shape[0])
    outputs: list[dict[str, torch.Tensor]] = []
    current_patch_captures: list[torch.Tensor] = []
    matcher = getattr(getattr(model, "heatmap_vln", None), "pose_free_matcher", None)

    def capture_current_patches(
        _module: torch.nn.Module,
        positional: tuple[Any, ...],
        keyword: dict[str, Any],
    ) -> None:
        current = keyword.get("current_patches", positional[0] if positional else None)
        if not torch.is_tensor(current) or current.ndim != 5 or int(current.shape[0]) != 1:
            raise RuntimeError("Current-patch audit did not observe strict B=1 matcher input")
        current_patch_captures.append(current.detach())

    hook = None
    if audit_current_patches:
        if matcher is None:
            raise RuntimeError("Current-patch audit requires a materialized pose-free matcher")
        hook = matcher.register_forward_pre_hook(capture_current_patches, with_kwargs=True)
    try:
        for chain_index in range(num_histories):
            output = model(
                video_frames=chains["video_frames"][chain_index : chain_index + 1].to(device),
                current_observation=chains["current_observation"][chain_index : chain_index + 1].to(device),
                current_views=chains["current_views"][chain_index : chain_index + 1],
                history_panoramas=chains["history_panoramas"][chain_index : chain_index + 1],
                history_rel_poses=None,
                return_heatmaps=True,
                return_heatmap_logits=True,
                return_actions=False,
                return_lm_loss=False,
            )
            outputs.append(output)
    finally:
        if hook is not None:
            hook.remove()
    current_patch_gate = None
    if audit_current_patches:
        if len(current_patch_captures) != 4:
            raise RuntimeError(f"Expected four matcher current-patch captures, got {len(current_patch_captures)}")
        reference = current_patch_captures[0]
        if any(not torch.equal(capture, reference) for capture in current_patch_captures[1:]):
            raise RuntimeError("Current LLM patches differ across the four isolated B=1 forwards")
        current_patch_gate = {
            "passed": True,
            "bitwise_exact": True,
            "captures": 4,
            "shape": list(reference.shape),
            "maximum_abs_difference": 0.0,
        }
    pred_vis, pred_heatmaps, pred_heatmap_logits = regroup_visual_identity_outputs(
        outputs,
        num_histories=num_histories,
    )
    gt_vis = transformed["gt_visibility"].unsqueeze(0).to(device)
    gt_heatmaps = transformed["gt_heatmaps"].unsqueeze(0).to(device)
    history_mask = torch.ones(gt_vis.shape[:2], dtype=torch.bool, device=device)
    base_losses = base_criterion(
        pred_vis,
        pred_heatmaps,
        gt_vis=gt_vis,
        gt_heatmaps=gt_heatmaps,
        history_mask=history_mask,
    )
    identity_output = None
    if mode_trains_lora(train_mode):
        identity_output = identity_criterion(pred_heatmap_logits, gt_vis, gt_heatmaps)
    total, components = compose_training_loss(train_mode, base_losses["total"], identity_output)
    record: dict[str, Any] = {
        "visibility": pred_vis.detach().float().cpu(),
        "heatmaps": pred_heatmaps.detach().float().cpu(),
        "heatmap_logits": pred_heatmap_logits.detach().float().cpu(),
        "gt_visibility": transformed["gt_visibility"].detach().float().cpu(),
        "gt_heatmaps": transformed["gt_heatmaps"].detach().float().cpu(),
        "loss_components": {name: float(value.detach().float().item()) for name, value in components.items()},
        "current_prefix_identity_gate": current_prefix_gate,
        "current_patch_identity_gate": current_patch_gate,
    }
    if mode_trains_lora(train_mode):
        record["_base_term_graph"] = base_losses["total"]
    if identity_output is not None:
        record["minimum_target_separation"] = float(
            identity_output["minimum_target_separation"].detach().float().item()
        )
        # Private graph handle: run_train consumes it with autograd.grad at
        # every recorded step, then still calls total loss.backward().
        record["panorama_view_loss"] = float(identity_output["view_loss"].detach().float().item())
        record["panorama_within_view_loss"] = float(identity_output["within_view_loss"].detach().float().item())
        record["_panorama_term_graph"] = PANORAMA_WEIGHT * identity_output["panorama_loss"]
        if train_mode == "lora-identity":
            record["_identity_auxiliary_graph"] = identity_output["total"]
            record["_identity_term_graph"] = IDENTITY_WEIGHT * identity_output["identity_loss"]
    return total, record


def _require_contract_fields_equal(
    checkpoint: Mapping[str, Any],
    current: Mapping[str, Any],
    fields: tuple[str, ...],
    *,
    label: str,
) -> None:
    for key in fields:
        if checkpoint.get(key) != current.get(key):
            raise RuntimeError(f"Visual identity checkpoint {label} mismatch: {key}")


def validate_visual_identity_checkpoint_payload_strict(
    path: str | Path,
    *,
    expected_train_mode: str,
    stage1_contract: dict[str, Any],
    manifest_contract: dict[str, Any],
    config_contract: dict[str, Any],
    runtime_contract: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Validate v3 provenance and reject all legacy anchor/B=4/joint states."""

    payload = safe_torch_load(path)
    if not isinstance(payload, dict):
        raise RuntimeError("Visual identity checkpoint payload is not a mapping")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError(
            "Refusing legacy anchor/B=4/joint checkpoint: the strict visual-identity v3 schema is required"
        )
    if payload.get("protocol") != PROTOCOL or payload.get("train_mode") != expected_train_mode:
        raise RuntimeError("Visual identity checkpoint protocol/train-mode mismatch")
    if int(payload.get("step", -1)) <= 0:
        raise RuntimeError("Visual identity checkpoint step must be positive")
    if int(payload.get("expected_lora_tensors", -1)) != EXPECTED_LORA_TENSORS:
        raise RuntimeError("Visual identity checkpoint has the wrong all-LoRA tensor-count contract")
    validate_optimization_contract_strict(
        payload.get("optimization_contract"),
        expected_train_mode=expected_train_mode,
        expected_step=int(payload["step"]),
    )
    if payload.get("loss_contract") != expected_loss_contract(expected_train_mode):
        raise RuntimeError("Visual identity checkpoint loss contract mismatch")

    checkpoint_config = payload.get("pose_free_config_contract")
    checkpoint_runtime = payload.get("runtime_contract")
    if checkpoint_config != config_contract or checkpoint_runtime != runtime_contract:
        raise RuntimeError("Visual identity checkpoint config/runtime contract mismatch")
    if checkpoint_config.get("history_query_source") != HISTORY_QUERY_SOURCE:
        raise RuntimeError("Checkpoint used the old text-anchor history query")
    if (
        checkpoint_runtime.get("qwen_forward_batch_size") != 1
        or checkpoint_runtime.get("qwen_forwards_per_sample") != 4
        or checkpoint_runtime.get("history_query_source") != HISTORY_QUERY_SOURCE
    ):
        raise RuntimeError("Checkpoint was not trained with strict 4xB=1 visual queries")

    checkpoint_stage1 = payload.get("stage1_s2_contract")
    checkpoint_manifest = payload.get("manifest_contract")
    if not isinstance(checkpoint_stage1, dict) or not isinstance(checkpoint_manifest, dict):
        raise RuntimeError("Visual identity checkpoint provenance contracts are missing")
    _require_contract_fields_equal(
        checkpoint_stage1,
        stage1_contract,
        ("file_sha256", "loaded_lora_sha256", "matched_lora_tensors"),
        label="Stage1-S2",
    )
    _require_contract_fields_equal(
        checkpoint_manifest,
        manifest_contract,
        (
            "manifest_sha256",
            "file_sha256",
            "source_inventory_sha256",
            "max_clip_id",
            "source_inventory_clips",
            "num_history",
            "train_identity_sha256",
            "val_identity_sha256",
            "train_samples",
            "val_samples",
            "scene_disjoint",
            "split_source_inventories",
            "minimum_target_separation_pixels",
            "identity_targets_per_sample",
        ),
        label="manifest",
    )

    head = payload.get("head_state_dict")
    lora = payload.get("lora_state_dict")
    if not isinstance(head, dict) or not head or not isinstance(lora, dict):
        raise RuntimeError("Visual identity checkpoint is missing head/all-LoRA states")
    if base_pilot.tensor_state_sha256(head) != payload.get("head_state_sha256"):
        raise RuntimeError("Visual identity checkpoint head strong hash mismatch")
    if len(lora) != EXPECTED_LORA_TENSORS or base_pilot.tensor_state_sha256(lora) != payload.get("lora_state_sha256"):
        raise RuntimeError("Visual identity checkpoint LoRA strong hash/count mismatch")
    baseline_lora_hash = stage1_contract["loaded_lora_sha256"]
    if payload.get("initial_lora_sha256") != baseline_lora_hash:
        raise RuntimeError("Training did not initialize from the original pinned Stage1-S2 LoRA")
    if expected_train_mode == "head-warmup":
        if payload.get("lora_state_sha256") != baseline_lora_hash:
            raise RuntimeError("head-warmup changed its frozen Stage1-S2 LoRA")
        if payload.get("warmup_checkpoint_contract") is not None:
            raise RuntimeError("head-warmup checkpoint cannot itself depend on a warmup checkpoint")
    else:
        warmup = payload.get("warmup_checkpoint_contract")
        if not isinstance(warmup, dict) or warmup.get("schema") != CHECKPOINT_SCHEMA:
            raise RuntimeError("LoRA phase lacks a strict v3 head-warmup pairing contract")
        if payload.get("initial_head_sha256") != warmup.get("head_state_sha256"):
            raise RuntimeError("LoRA phase did not initialize from the paired warmup head")
        if payload.get("head_state_sha256") != warmup.get("head_state_sha256"):
            raise RuntimeError("LoRA phase changed the bitwise-frozen warmup head")
    return payload, base_pilot.file_sha256(path)


def load_warmup_head_strict(
    model: torch.nn.Module,
    path: str | Path,
    *,
    stage1_contract: dict[str, Any],
    manifest_contract: dict[str, Any],
    config_contract: dict[str, Any],
    runtime_contract: dict[str, Any],
) -> dict[str, Any]:
    payload, file_hash = validate_visual_identity_checkpoint_payload_strict(
        path,
        expected_train_mode="head-warmup",
        stage1_contract=stage1_contract,
        manifest_contract=manifest_contract,
        config_contract=config_contract,
        runtime_contract=runtime_contract,
    )
    matcher = model.heatmap_vln.pose_free_matcher
    if matcher is None:
        raise RuntimeError("Cannot load warmup head without PoseFreeHistoryMatcher")
    expected: dict[str, torch.Tensor] = dict(matcher.named_parameters())
    expected.update(dict(matcher.named_buffers()))
    base_pilot.strict_load_named_state(expected, payload["head_state_dict"], label="v3 warmup head")
    actual_hash = base_pilot.tensor_state_sha256(base_pilot.pose_free_head_state_dict(model))
    if actual_hash != payload["head_state_sha256"]:
        raise RuntimeError("Warmup head did not load bitwise exactly")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "protocol": PROTOCOL,
        "path": str(Path(path).resolve()),
        "file_sha256": file_hash,
        "head_state_sha256": actual_hash,
        "lora_state_sha256": payload["lora_state_sha256"],
        "step": int(payload["step"]),
        "training_sample_schedule_sha256": payload["training_sample_schedule_sha256"],
        "optimization_contract": payload["optimization_contract"],
    }


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    args: argparse.Namespace,
    stage1_contract: dict[str, Any],
    manifest_contract: dict[str, Any],
    config_contract: dict[str, Any],
    runtime_contract: dict[str, Any],
    warmup_contract: dict[str, Any] | None,
    initial_head_hash: str,
    initial_lora_hash: str,
    schedule_hash: str,
    optimization_contract: dict[str, Any],
    train_log: list[dict[str, Any]],
) -> dict[str, Any]:
    head = base_pilot.pose_free_head_state_dict(model)
    lora = base_pilot.lora_state_dict(model)
    if len(lora) != EXPECTED_LORA_TENSORS:
        raise RuntimeError("Refusing to save an incomplete all-LoRA v3 checkpoint")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "protocol": PROTOCOL,
        "train_mode": args.train_mode,
        "step": args.train_steps,
        "training_pid": os.getpid(),
        "head_state_dict": head,
        "lora_state_dict": lora,
        "head_state_sha256": base_pilot.tensor_state_sha256(head),
        "lora_state_sha256": base_pilot.tensor_state_sha256(lora),
        "initial_head_sha256": initial_head_hash,
        "initial_lora_sha256": initial_lora_hash,
        "expected_lora_tensors": EXPECTED_LORA_TENSORS,
        "stage1_s2_contract": stage1_contract,
        "manifest_contract": manifest_contract,
        "pose_free_config_contract": config_contract,
        "runtime_contract": runtime_contract,
        "warmup_checkpoint_contract": warmup_contract,
        "training_sample_schedule_sha256": schedule_hash,
        "optimization_contract": optimization_contract,
        "loss_contract": expected_loss_contract(args.train_mode),
        "train_log": train_log,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "file_sha256": base_pilot.file_sha256(path),
        "head_state_sha256": payload["head_state_sha256"],
        "lora_state_sha256": payload["lora_state_sha256"],
        "lora_tensors": len(lora),
    }


def _load_pair_checkpoint(path: str | Path, expected_train_mode: str) -> tuple[dict[str, Any], str]:
    payload = safe_torch_load(path)
    if not isinstance(payload, dict):
        raise RuntimeError("Identity/control pair checkpoint is not a mapping")
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("protocol") != PROTOCOL
        or payload.get("train_mode") != expected_train_mode
    ):
        raise RuntimeError(f"Identity/control pair has invalid {expected_train_mode} schema/protocol/mode")
    if int(payload.get("expected_lora_tensors", -1)) != EXPECTED_LORA_TENSORS:
        raise RuntimeError("Identity/control pair has wrong all-LoRA tensor count")
    head = payload.get("head_state_dict")
    lora = payload.get("lora_state_dict")
    if not isinstance(head, dict) or not isinstance(lora, dict):
        raise RuntimeError("Identity/control pair checkpoint is missing states")
    if base_pilot.tensor_state_sha256(head) != payload.get("head_state_sha256"):
        raise RuntimeError("Identity/control pair checkpoint has an invalid head hash")
    if len(lora) != EXPECTED_LORA_TENSORS or base_pilot.tensor_state_sha256(lora) != payload.get("lora_state_sha256"):
        raise RuntimeError("Identity/control pair checkpoint has an invalid LoRA hash/count")
    validate_optimization_contract_strict(
        payload.get("optimization_contract"),
        expected_train_mode=expected_train_mode,
        expected_step=int(payload.get("step", -1)),
    )
    if payload.get("loss_contract") != expected_loss_contract(expected_train_mode):
        raise RuntimeError("Identity/control pair checkpoint has an invalid loss contrast")
    return payload, base_pilot.file_sha256(path)


def validate_identity_control_checkpoint_pair(
    identity_checkpoint: str | Path,
    control_checkpoint: str | Path,
) -> dict[str, Any]:
    """Prove the two LoRA arms differ only in registered auxiliary weights."""

    identity, identity_file_hash = _load_pair_checkpoint(identity_checkpoint, "lora-identity")
    control, control_file_hash = _load_pair_checkpoint(control_checkpoint, "lora-heatmap-control")
    identity_warmup = identity.get("warmup_checkpoint_contract")
    control_warmup = control.get("warmup_checkpoint_contract")
    if not isinstance(identity_warmup, dict) or not isinstance(control_warmup, dict):
        raise RuntimeError("Identity/control pair is missing warmup provenance")

    comparisons = {
        "stage1_s2_contract": identity.get("stage1_s2_contract") == control.get("stage1_s2_contract"),
        "stage1_actual_file_sha256": identity.get("stage1_s2_contract", {}).get("file_sha256")
        == control.get("stage1_s2_contract", {}).get("file_sha256"),
        "manifest_contract": identity.get("manifest_contract") == control.get("manifest_contract"),
        "pose_free_config_contract": identity.get("pose_free_config_contract")
        == control.get("pose_free_config_contract"),
        "runtime_contract": identity.get("runtime_contract") == control.get("runtime_contract"),
        "warmup_actual_file_sha256": identity_warmup.get("file_sha256") == control_warmup.get("file_sha256"),
        "warmup_head_sha256": identity_warmup.get("head_state_sha256") == control_warmup.get("head_state_sha256"),
        "frozen_active_head_sha256": identity.get("head_state_sha256") == control.get("head_state_sha256"),
        "initial_head_sha256": identity.get("initial_head_sha256") == control.get("initial_head_sha256"),
        "initial_lora_sha256": identity.get("initial_lora_sha256") == control.get("initial_lora_sha256"),
        "training_sample_schedule_sha256": identity.get("training_sample_schedule_sha256")
        == control.get("training_sample_schedule_sha256"),
        "optimization_contract": identity.get("optimization_contract") == control.get("optimization_contract"),
    }
    failed = [name for name, passed in comparisons.items() if not passed]
    if failed:
        raise RuntimeError("Identity/control checkpoints are not a causal pair: " + ", ".join(failed))
    identity_loss = identity["loss_contract"]
    control_loss = control["loss_contract"]
    if (
        identity_loss["base_weight"] != control_loss["base_weight"]
        or identity_loss["identity_weight"] != IDENTITY_WEIGHT
        or identity_loss["panorama_weight"] != PANORAMA_WEIGHT
        or control_loss["identity_weight"] != 0.0
        or control_loss["panorama_weight"] != PANORAMA_WEIGHT
    ):
        raise RuntimeError("Identity/control loss contracts do not differ only by 2I")
    return {
        "passed": True,
        "identity_checkpoint": {
            "path": str(Path(identity_checkpoint).resolve()),
            "file_sha256": identity_file_hash,
        },
        "control_checkpoint": {
            "path": str(Path(control_checkpoint).resolve()),
            "file_sha256": control_file_hash,
        },
        "matched_contracts": sorted(comparisons),
        "only_registered_difference": {
            "identity_weight": [IDENTITY_WEIGHT, 0.0],
        },
    }


def _named_gradient_summary(parameters: dict[str, torch.nn.Parameter]) -> dict[str, Any]:
    if not parameters:
        return {
            "tensors_with_grad": 0,
            "tensors_with_nonzero_grad": 0,
            "nonzero_names": [],
            "total_grad_norm": 0.0,
            "per_layer": {},
        }
    return base_pilot.gradient_summary(parameters)


def run_train(args: argparse.Namespace) -> int:
    started = time.time()
    cfg = load_visual_identity_config(args)
    config_contract = visual_identity_config_contract(cfg)
    records, manifest_contract = load_visual_identity_manifest_contract(args)
    dataset = base_pilot.build_explicit_dataset(
        cfg,
        "train",
        records["train"],
        seed=args.seed + 3700,
        reshuffle_slots_each_epoch=True,
        max_clip_id=manifest_contract["max_clip_id"],
        expected_inventory_sha256=manifest_contract["split_source_inventories"]["train"]["inventory_sha256"],
        expected_inventory_clips=manifest_contract["split_source_inventories"]["train"]["clips"],
    )
    model, stage1_contract, runtime_contract = materialize_visual_identity_model(args, cfg)
    device = torch.device(args.device)
    base_criterion = base_pilot.make_criterion(cfg, device)
    identity_criterion = TargetGroundedPanoramaIdentityLoss(
        identity_weight=IDENTITY_WEIGHT,
        panorama_weight=PANORAMA_WEIGHT,
        min_target_separation=MIN_TARGET_SEPARATION,
        expected_num_targets=4,
    ).to(device)

    original_stage1_lora_hash = base_pilot.tensor_state_sha256(base_pilot.lora_state_dict(model))
    if original_stage1_lora_hash != stage1_contract["loaded_lora_sha256"]:
        raise RuntimeError("Fresh process did not initialize from the pinned original Stage1-S2 LoRA")
    warmup_contract = None
    if mode_trains_lora(args.train_mode):
        warmup_contract = load_warmup_head_strict(
            model,
            args.warmup_checkpoint,
            stage1_contract=stage1_contract,
            manifest_contract=manifest_contract,
            config_contract=config_contract,
            runtime_contract=runtime_contract,
        )
        if base_pilot.tensor_state_sha256(base_pilot.lora_state_dict(model)) != original_stage1_lora_hash:
            raise RuntimeError("Loading the warmup head mutated freshly reinitialized Stage1-S2 LoRA")

    initial_head = base_pilot.pose_free_head_state_dict(model)
    initial_lora = base_pilot.lora_state_dict(model)
    initial_head_hash = base_pilot.tensor_state_sha256(initial_head)
    initial_lora_hash = base_pilot.tensor_state_sha256(initial_lora)
    if warmup_contract is not None and initial_head_hash != warmup_contract["head_state_sha256"]:
        raise RuntimeError("LoRA phase did not begin from the exact paired warmup head")

    trainable_head, trainable_lora = configure_training_state(
        model,
        args.train_mode,
        max_lora_layer=args.max_trainable_lora_layer,
    )
    optimization_contract = build_optimization_contract(
        args,
        cfg,
        trainable_head,
        trainable_lora,
    )
    if args.train_mode == "head-warmup":
        groups = [
            {
                "name": "pose_free_matcher_warmup",
                "params": list(trainable_head.values()),
                "lr": args.head_learning_rate,
            }
        ]
    else:
        groups = [
            {
                "name": "reachable_lora",
                "params": list(trainable_lora.values()),
                "lr": args.lora_learning_rate,
            }
        ]
    optimizer = torch.optim.AdamW(
        groups,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        weight_decay=args.weight_decay,
        amsgrad=False,
    )
    model.eval()
    model.qwen2_5_vl.model.train(mode_trains_lora(args.train_mode))
    model.heatmap_vln.pose_free_matcher.train(args.train_mode == "head-warmup")

    train_log: list[dict[str, Any]] = []
    schedule_ids: list[str] = []
    reached_head_names: set[str] = set()
    reached_lora_names: set[str] = set()
    max_head_grad_norm = 0.0
    max_lora_grad_norm = 0.0
    identity_auxiliary_audits: list[dict[str, Any]] = []
    base_term_audits: list[dict[str, Any]] = []
    identity_term_audits: list[dict[str, Any]] = []
    panorama_term_audits: list[dict[str, Any]] = []
    for step in range(1, args.train_steps + 1):
        epoch, index = divmod(step - 1, len(dataset))
        dataset.set_epoch(epoch)
        sample = base_pilot.exact_sample(dataset, index)
        audit = sample["_task36c_audit"]
        history_frames = ",".join(str(value) for value in audit["runtime_history_frames"])
        schedule_ids.append(f"{audit['runtime_sample_id']}:epoch={epoch}:runtime_history={history_frames}")
        transformed = base_pilot.transform_sample(sample, intervention="standard")
        optimizer.zero_grad(set_to_none=True)
        record_step = step == 1 or step % args.log_every == 0 or step == args.train_steps
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss, record = forward_visual_identity_loss(
                model,
                base_criterion,
                identity_criterion,
                transformed,
                device,
                train_mode=args.train_mode,
                history_rel_poses=None,
                audit_current_patches=record_step,
            )
        auxiliary_audit = None
        base_term_audit = None
        identity_term_audit = None
        panorama_term_audit = None
        auxiliary_graph = record.pop("_identity_auxiliary_graph", None)
        identity_term_graph = record.pop("_identity_term_graph", None)
        panorama_term_graph = record.pop("_panorama_term_graph", None)
        base_term_graph = record.pop("_base_term_graph", None)
        if mode_trains_lora(args.train_mode):
            if base_term_graph is None:
                raise RuntimeError(f"{args.train_mode} forward omitted the base-loss graph handle")
            if record_step:
                base_term_audit = {
                    "step": step,
                    **audit_lora_objective_gradient(
                        base_term_graph,
                        trainable_lora,
                        objective_label="base_heatmap_loss",
                        require_nonzero=False,
                    ),
                }
                base_term_audits.append(base_term_audit)
        elif base_term_graph is not None:
            raise RuntimeError("head-warmup unexpectedly exposed a LoRA base-loss graph audit")
        if args.train_mode == "lora-identity":
            if auxiliary_graph is None or identity_term_graph is None or panorama_term_graph is None:
                raise RuntimeError("lora-identity forward omitted component graph handles")
            if record_step:
                auxiliary_audit = {
                    "step": step,
                    **audit_identity_auxiliary_gradient(auxiliary_graph, trainable_lora),
                }
                identity_term_audit = {
                    "step": step,
                    **audit_lora_objective_gradient(
                        identity_term_graph,
                        trainable_lora,
                        objective_label="2 * target_grounded_identity",
                        require_nonzero=False,
                    ),
                }
                panorama_term_audit = {
                    "step": step,
                    **audit_lora_objective_gradient(
                        panorama_term_graph,
                        trainable_lora,
                        objective_label="global_panorama_pixel_ce",
                        require_nonzero=False,
                    ),
                }
                identity_auxiliary_audits.append(auxiliary_audit)
                identity_term_audits.append(identity_term_audit)
                panorama_term_audits.append(panorama_term_audit)
        elif args.train_mode == "lora-heatmap-control":
            if auxiliary_graph is not None or identity_term_graph is not None or panorama_term_graph is None:
                raise RuntimeError("lora-heatmap-control forward component graph contract failed")
            if record_step:
                panorama_term_audit = {
                    "step": step,
                    **audit_lora_objective_gradient(
                        panorama_term_graph,
                        trainable_lora,
                        objective_label="global_panorama_pixel_ce",
                        require_nonzero=False,
                    ),
                }
                panorama_term_audits.append(panorama_term_audit)
        elif any(graph is not None for graph in (auxiliary_graph, identity_term_graph, panorama_term_graph)):
            raise RuntimeError(f"{args.train_mode} unexpectedly produced identity component graphs")
        loss.backward()
        head_grad = _named_gradient_summary(trainable_head)
        lora_grad = _named_gradient_summary(trainable_lora)
        all_head_grad = _named_gradient_summary(
            {name: parameter for name, parameter in model.heatmap_vln.pose_free_matcher.named_parameters()}
        )
        all_lora_grad = _named_gradient_summary(base_pilot.normalized_lora_parameters(model))
        if args.train_mode == "head-warmup" and all_lora_grad["tensors_with_grad"]:
            raise RuntimeError("head-warmup leaked gradients into frozen LoRA")
        if mode_trains_lora(args.train_mode) and all_head_grad["tensors_with_grad"]:
            raise RuntimeError("LoRA phase leaked gradients into the frozen warmup head")
        reached_head_names.update(head_grad["nonzero_names"])
        reached_lora_names.update(lora_grad["nonzero_names"])
        max_head_grad_norm = max(max_head_grad_norm, float(head_grad["total_grad_norm"]))
        max_lora_grad_norm = max(max_lora_grad_norm, float(lora_grad["total_grad_norm"]))
        trainable = [*trainable_head.values(), *trainable_lora.values()]
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        if warmup_contract is not None:
            current_head_hash = base_pilot.tensor_state_sha256(base_pilot.pose_free_head_state_dict(model))
            if current_head_hash != warmup_contract["head_state_sha256"]:
                raise RuntimeError(f"Frozen warmup head changed bitwise at step {step}")
        if record_step:
            gradient_scale_ratios = None
            if mode_trains_lora(args.train_mode):
                base_norm = float(base_term_audit["total_grad_norm"])
                panorama_norm = float(panorama_term_audit["total_grad_norm"])
                identity_norm = (
                    float(identity_term_audit["total_grad_norm"]) if identity_term_audit is not None else None
                )
                gradient_scale_ratios = {
                    "panorama_over_base": panorama_norm / base_norm if base_norm > 0 else None,
                    "panorama_over_weighted_identity": (
                        panorama_norm / identity_norm if identity_norm is not None and identity_norm > 0 else None
                    ),
                    "recommended_identity_balance_band": [0.25, 4.0],
                    "diagnostic_only_no_adaptive_reweighting": True,
                }
            item = {
                "step": step,
                "loss": float(loss.detach().float().item()),
                "loss_components": record["loss_components"],
                "head_gradient": {key: value for key, value in head_grad.items() if key != "nonzero_names"},
                "lora_gradient": {key: value for key, value in lora_grad.items() if key != "nonzero_names"},
                "minimum_target_separation": record.get("minimum_target_separation"),
                "current_prefix_identity_gate": record["current_prefix_identity_gate"],
                "current_patch_identity_gate": record["current_patch_identity_gate"],
                "base_term_lora_gradient": base_term_audit,
                "identity_auxiliary_lora_gradient": auxiliary_audit,
                "identity_term_lora_gradient": identity_term_audit if args.train_mode == "lora-identity" else None,
                "panorama_term_lora_gradient": panorama_term_audit,
                "gradient_scale_ratios": gradient_scale_ratios,
            }
            train_log.append(item)
            LOGGER.info(
                "mode=%s step=%d/%d total=%.6f base=%.6f identity=%.6f panorama=%.6f lora_grad=%.3e",
                args.train_mode,
                step,
                args.train_steps,
                item["loss"],
                item["loss_components"]["base"],
                item["loss_components"]["identity"],
                item["loss_components"]["panorama"],
                lora_grad["total_grad_norm"],
            )

    final_head = base_pilot.pose_free_head_state_dict(model)
    final_lora = base_pilot.lora_state_dict(model)
    final_head_hash = base_pilot.tensor_state_sha256(final_head)
    final_lora_hash = base_pilot.tensor_state_sha256(final_lora)
    head_drift = base_pilot.delta_summary(initial_head, final_head)
    lora_drift = base_pilot.delta_summary(initial_lora, final_lora)
    if args.train_mode == "head-warmup":
        if final_lora_hash != stage1_contract["loaded_lora_sha256"]:
            raise RuntimeError("head-warmup changed frozen Stage1-S2 LoRA")
        if not reached_head_names or head_drift["tensors_with_nonzero_delta"] == 0:
            raise RuntimeError("One-step warmup gate failed: base loss did not reach/change the head")
    else:
        if final_head_hash != warmup_contract["head_state_sha256"]:
            raise RuntimeError("LoRA phase changed the frozen warmup head")
        if not reached_lora_names or lora_drift["tensors_with_nonzero_delta"] == 0:
            raise RuntimeError("One-step LoRA gate failed: selected loss did not reach/change LoRA")
    if args.train_mode == "lora-identity":
        if not identity_auxiliary_audits or identity_auxiliary_audits[0]["step"] != 1:
            raise RuntimeError("Identity auxiliary audit did not cover mandatory step 1")
        if len(identity_auxiliary_audits) != len(train_log):
            raise RuntimeError("Identity auxiliary audit did not cover every recorded training step")
    if mode_trains_lora(args.train_mode) and len(panorama_term_audits) != len(train_log):
        raise RuntimeError("Panorama-term audit did not cover every recorded LoRA training step")
    if mode_trains_lora(args.train_mode) and len(base_term_audits) != len(train_log):
        raise RuntimeError("Base-term audit did not cover every recorded LoRA training step")

    schedule_hash = base_pilot.hash_strings(schedule_ids)
    mode_dir = Path(args.output_dir) / args.train_mode
    checkpoint_contract = save_checkpoint(
        mode_dir / "checkpoint_final.pth",
        model=model,
        args=args,
        stage1_contract=stage1_contract,
        manifest_contract=manifest_contract,
        config_contract=config_contract,
        runtime_contract=runtime_contract,
        warmup_contract=warmup_contract,
        initial_head_hash=initial_head_hash,
        initial_lora_hash=initial_lora_hash,
        schedule_hash=schedule_hash,
        optimization_contract=optimization_contract,
        train_log=train_log,
    )
    _saved_payload, validated_checkpoint_sha = validate_visual_identity_checkpoint_payload_strict(
        checkpoint_contract["path"],
        expected_train_mode=args.train_mode,
        stage1_contract=stage1_contract,
        manifest_contract=manifest_contract,
        config_contract=config_contract,
        runtime_contract=runtime_contract,
    )
    if validated_checkpoint_sha != checkpoint_contract["file_sha256"]:
        raise RuntimeError("Saved v3 checkpoint file hash changed during strict post-save validation")
    pair_contract = None
    if mode_trains_lora(args.train_mode):
        identity_path = Path(args.output_dir) / "lora-identity" / "checkpoint_final.pth"
        control_path = Path(args.output_dir) / "lora-heatmap-control" / "checkpoint_final.pth"
        if identity_path.is_file() and control_path.is_file():
            pair_contract = validate_identity_control_checkpoint_pair(identity_path, control_path)
    report = {
        "schema": REPORT_SCHEMA,
        "protocol": PROTOCOL,
        "train_mode": args.train_mode,
        "train_steps": args.train_steps,
        "duration_seconds": time.time() - started,
        "stage1_s2_contract": stage1_contract,
        "manifest_contract": manifest_contract,
        "pose_free_config_contract": config_contract,
        "runtime_contract": runtime_contract,
        "warmup_checkpoint_contract": warmup_contract,
        "checkpoint": checkpoint_contract,
        "optimization_contract": optimization_contract,
        "identity_control_pair_contract": pair_contract,
        "state_machine_gate": {
            "fresh_stage1_lora_initialized": initial_lora_hash == stage1_contract["loaded_lora_sha256"],
            "head_trainable": args.train_mode == "head-warmup",
            "lora_trainable": mode_trains_lora(args.train_mode),
            "frozen_head_bitwise_preserved": (
                None if args.train_mode == "head-warmup" else final_head_hash == warmup_contract["head_state_sha256"]
            ),
            "frozen_lora_bitwise_preserved": (
                final_lora_hash == stage1_contract["loaded_lora_sha256"] if args.train_mode == "head-warmup" else None
            ),
            "raw_logits_requested_on_all_four_b1_forwards": True,
            "current_prefix_bitwise_identical_on_recorded_steps": all(
                item["current_prefix_identity_gate"].get("passed") is True for item in train_log
            ),
            "current_llm_patches_bitwise_identical_on_recorded_steps": all(
                item["current_patch_identity_gate"].get("passed") is True for item in train_log
            ),
        },
        "gradient_reach": {
            "reachable_trainable_head_tensors": len(trainable_head),
            "ever_nonzero_head_tensors": len(reached_head_names),
            "max_head_grad_norm": max_head_grad_norm,
            "reachable_trainable_lora_tensors": len(trainable_lora),
            "expected_trainable_lora_tensors": (
                EXPECTED_TRAINABLE_LORA_TENSORS if mode_trains_lora(args.train_mode) else 0
            ),
            "actual_trainable_lora_layers": trainable_lora_layers(trainable_lora),
            "expected_trainable_lora_layers": (
                list(EXPECTED_TRAINABLE_LORA_LAYERS) if mode_trains_lora(args.train_mode) else []
            ),
            "ever_nonzero_lora_tensors": len(reached_lora_names),
            "max_lora_grad_norm": max_lora_grad_norm,
        },
        "identity_auxiliary_lora_gradient_audits": identity_auxiliary_audits,
        "base_term_lora_gradient_audits": base_term_audits,
        "identity_term_lora_gradient_audits": identity_term_audits,
        "panorama_term_lora_gradient_audits": panorama_term_audits,
        "head_drift": head_drift,
        "lora_drift": lora_drift,
        "training_sample_schedule_sha256": schedule_hash,
        "loss_contract": {
            "head-warmup": "base",
            "lora-identity": "base + 2 * target_grounded_identity + global_panorama_pixel_ce",
            "lora-heatmap-control": "base + global_panorama_pixel_ce",
        },
        "train_log": train_log,
    }
    base_pilot.json_dump(mode_dir / "train_report.json", report)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    base_pilot.set_seed(args.seed)
    return run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())

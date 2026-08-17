#!/usr/bin/env python3
"""Task-3.6c pose-free multi-history train/eval pilot.

This program intentionally has only two process-level phases:

* ``train`` updates the shared pose-free matcher, optionally with reachable
  Qwen LoRA tensors, and writes a self-contained head + all-LoRA checkpoint;
* ``eval`` starts from the Stage1-S2 checkpoint in a fresh process, loads the
  trained head, optionally loads the trained LoRA, and runs causal image
  interventions.

Exact pose is label-side data only.  The explicit dataset removes every pose
tensor before returning a sample and the forward helper rejects any non-None
pose argument.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training import (
    _load_normalized_state_dict,
    _normalize_state_key,
    assert_complete_lora_checkpoint_match,
    build_model,
    safe_torch_load,
)

from src.data.explicit_multi_history import (
    ExplicitMultiHistoryDataset,
    load_multi_history_records,
)
from src.models.heatmap import HeatmapVLNLoss

LOGGER = logging.getLogger("pose_free_multihistory_pilot")
CHECKPOINT_SCHEMA = "task36c_pose_free_multihistory_checkpoint_v1"
REPORT_SCHEMA = "task36c_pose_free_multihistory_report_v1"
EXPECTED_LORA_TENSORS = 224
INTERVENTIONS = (
    "standard",
    "blank-images",
    "history-shuffle",
    "current-shuffle",
    "single-anchor-swap",
)
LORA_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "eval"), required=True)
    parser.add_argument("--branch", choices=("head-only", "heatmap-lora"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Stage1-S2 checkpoint with exactly 224 LoRA tensors.")
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--source-inventory-sha256", default=None)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-trainable-lora-layer", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--pilot-checkpoint", default=None, help="Required only for --phase eval.")
    parser.add_argument(
        "--eval-head-checkpoint",
        default=None,
        help=(
            "Evaluation-only factorial cell: keep trained LoRA from the heatmap-lora "
            "--pilot-checkpoint, but load the pose-free head from a strictly paired "
            "head-only B=1 pilot checkpoint."
        ),
    )
    parser.add_argument(
        "--eval-lora",
        choices=("trained", "off"),
        default="trained",
        help="Eval with checkpoint LoRA or the untouched Stage1-S2 LoRA.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    eval_head_checkpoint = getattr(args, "eval_head_checkpoint", None)
    if args.phase == "train" and args.pilot_checkpoint is not None:
        raise ValueError("--pilot-checkpoint is evaluation-only")
    if args.phase == "train" and eval_head_checkpoint is not None:
        raise ValueError("--eval-head-checkpoint is evaluation-only")
    if args.phase == "eval" and not args.pilot_checkpoint:
        raise ValueError("--phase eval requires --pilot-checkpoint")
    if eval_head_checkpoint is not None:
        if args.branch != "heatmap-lora":
            raise ValueError("--eval-head-checkpoint requires --branch heatmap-lora")
        if args.eval_lora != "trained":
            raise ValueError("--eval-head-checkpoint requires --eval-lora trained")
    if args.train_steps < 0:
        raise ValueError("--train-steps must be non-negative")
    if args.phase == "train" and args.train_steps == 0:
        raise ValueError("Task-3.6c training requires at least one step")
    if args.grad_clip <= 0 or args.log_every <= 0:
        raise ValueError("--grad-clip and --log-every must be positive")
    if args.max_trainable_lora_layer != 20:
        raise ValueError("The Task-3.6c attribution contract fixes the deepest/reachable LoRA layer at 20")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def hash_strings(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash names, dtype, shape, and exact tensor bytes."""
    digest = hashlib.sha256(b"task36c_tensor_state_v1\0")
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def load_pilot_config(args: argparse.Namespace) -> dict[str, Any]:
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["data"]["root"] = str(Path(args.data_root).resolve())
    cfg["data"]["dataset_type"] = "sliding_window"

    model_cfg = cfg["model"]
    model_cfg["device"] = args.device
    llm_cfg = model_cfg["llm"]
    if args.model_path:
        llm_cfg["model_path"] = str(Path(args.model_path).resolve())
    llm_cfg["attn_implementation"] = "sdpa"
    llm_cfg["enable_compile"] = False
    llm_cfg["lora_dropout"] = 0.0
    llm_cfg["gradient_checkpointing"] = args.branch == "heatmap-lora"

    heatmap_cfg = model_cfg.setdefault("heatmap", {})
    heatmap_cfg["enable"] = True
    heatmap_cfg["decoder_mode"] = "pose_free_matcher"
    heatmap_cfg["trajectory"] = {"enable": False}
    heatmap_cfg["vit_layer_indices"] = []
    heatmap_cfg["llm_layer_indices"] = [20]
    # A normal Qwen forward is required even for head-only: trainable matcher
    # layers cannot save inference tensors. Feature hooks detach in that branch.
    heatmap_cfg["heatmap_trains_backbone"] = True
    heatmap_cfg.setdefault("pose_free", {})["heatmap_size"] = tuple(cfg["data"]["init_hm_size"])
    model_cfg.setdefault("action_head", {})["enable"] = False

    loss_cfg = cfg.setdefault("loss", {}).setdefault("heatmap_vln", {})
    loss_cfg["lambda_coord"] = 0.0
    loss_cfg.setdefault("lambda_vis", 1.0)
    loss_cfg.setdefault("lambda_peak", 1.0)
    loss_cfg.setdefault("lambda_neg", 1.0)
    loss_cfg.setdefault("vis_pos_weight", 7.0)
    return cfg


def pose_free_config_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    heatmap = cfg["model"]["heatmap"]
    contract = {
        "decoder_mode": heatmap.get("decoder_mode"),
        "trajectory_enabled": bool(heatmap.get("trajectory", {}).get("enable", False)),
        "vit_layer_indices": list(heatmap.get("vit_layer_indices", [])),
        "llm_layer_indices": list(heatmap.get("llm_layer_indices", [])),
        "model_pose_input": None,
        "isolated_pair_chains": True,
        "histories_per_qwen_chain": 1,
        "qwen_forward_batch_size": 1,
        "qwen_forwards_per_sample": 4,
    }
    expected = {
        "decoder_mode": "pose_free_matcher",
        "trajectory_enabled": False,
        "vit_layer_indices": [],
        "llm_layer_indices": [20],
        "model_pose_input": None,
        "isolated_pair_chains": True,
        "histories_per_qwen_chain": 1,
        "qwen_forward_batch_size": 1,
        "qwen_forwards_per_sample": 4,
    }
    if contract != expected:
        raise RuntimeError(f"Pose-free config contract mismatch: expected={expected} actual={contract}")
    return contract


def load_manifest_contract(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    records: dict[str, list[dict[str, Any]]] = {}
    manifests = []
    for split in ("train", "val"):
        split_records, manifest = load_multi_history_records(
            args.selection_manifest,
            split,
            expected_source_inventory_sha256=args.source_inventory_sha256,
        )
        if not split_records:
            raise RuntimeError(f"Task-3.6c manifest split {split!r} is empty")
        records[split] = split_records
        manifests.append(manifest)
    if manifests[0] != manifests[1]:
        raise RuntimeError("Manifest changed between strict train/val reads")
    manifest = manifests[0]
    if not bool(manifest.get("ready")):
        raise RuntimeError("Task-3.6c refuses a manifest with ready=false")
    scene_contract = manifest.get("scene_disjoint", {})
    if not bool(scene_contract.get("verified")) or scene_contract.get("overlap"):
        raise RuntimeError(f"Manifest train/val scenes are not disjoint: {scene_contract}")
    if int(manifest["selection_parameters"]["num_history"]) != 4:
        raise RuntimeError("The first Task-3.6c pilot fixes K=4")
    input_contract = manifest.get("model_input_contract", {})
    if "exact_relative_pose" not in set(input_contract.get("forbidden", [])):
        raise RuntimeError("Manifest does not explicitly forbid exact_relative_pose")
    inventory = manifest["source_inventory_contract"]
    max_clip_id = int(inventory.get("max_clip_id", 0))
    if max_clip_id <= 0:
        raise RuntimeError("Manifest source inventory must pin a positive max_clip_id")
    return records, {
        "path": str(Path(args.selection_manifest).resolve()),
        "file_sha256": file_sha256(args.selection_manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_inventory_sha256": inventory["inventory_sha256"],
        "max_clip_id": max_clip_id,
        "source_inventory_clips": int(inventory["clips"]),
        "num_history": 4,
        "train_identity_sha256": manifest["splits"]["train"]["selection_manifest"]["record_identity_sha256"],
        "val_identity_sha256": manifest["splits"]["val"]["selection_manifest"]["record_identity_sha256"],
        "train_samples": len(records["train"]),
        "val_samples": len(records["val"]),
        "scene_disjoint": True,
        "split_source_inventories": {
            split: {
                "inventory_sha256": manifest["splits"][split]["source_inventory"]["inventory_sha256"],
                "clips": int(manifest["splits"][split]["source_inventory"]["clips"]),
            }
            for split in ("train", "val")
        },
    }


def build_explicit_dataset(
    cfg: dict[str, Any],
    split: str,
    records: list[dict[str, Any]],
    *,
    seed: int,
    reshuffle_slots_each_epoch: bool,
    max_clip_id: int,
    expected_inventory_sha256: str,
    expected_inventory_clips: int,
) -> ExplicitMultiHistoryDataset:
    data_cfg = cfg["data"]
    sliding = data_cfg.get("sliding_window", {})
    dataset = ExplicitMultiHistoryDataset(
        root=data_cfg["root"],
        split=split,
        min_history=int(sliding.get("min_history", 5)),
        image_size=tuple(data_cfg["image_size"]),
        hm_size=tuple(data_cfg["init_hm_size"]),
        load_depth=True,
        cache_poses=True,
        sample_stride=2,
        enable_augmentation=False,
        samples_per_clip=8,
        clip_level_sampling=False,
        load_single_view_history_frames=True,
        max_clips=0,
        max_clip_id=max_clip_id,
        selection_records=records,
        slot_seed=seed,
        reshuffle_slots_each_epoch=reshuffle_slots_each_epoch,
        drop_pose_inputs=True,
        verify_runtime_labels=True,
    )
    if len(dataset) != len(records):
        raise RuntimeError(f"Explicit dataset length mismatch for {split}: {len(dataset)} != {len(records)}")
    inventory_rows = []
    for clip_index, clip in enumerate(dataset.clips):
        meta = dataset._load_meta(clip_index)
        try:
            relative_clip = Path(clip).relative_to(dataset.root).as_posix()
        except ValueError:
            relative_clip = Path(clip).as_posix()
        inventory_rows.append(
            f"{relative_clip}\t{meta.get('scene_id')}\t{meta.get('episode_id')}\t"
            f"{meta.get('num_frames')}\t{meta.get('seed')}"
        )
    inventory_rows.sort()
    actual_inventory_hash = hashlib.sha256(
        ("\n".join(inventory_rows) + ("\n" if inventory_rows else "")).encode()
    ).hexdigest()
    if len(inventory_rows) != expected_inventory_clips or actual_inventory_hash != expected_inventory_sha256:
        raise RuntimeError(
            f"Runtime {split} source inventory differs from manifest pin: "
            f"clips={len(inventory_rows)}/{expected_inventory_clips} "
            f"sha256={actual_inventory_hash}/{expected_inventory_sha256}"
        )
    dataset._task36c_source_inventory_sha256 = actual_inventory_hash
    return dataset


def exact_sample(dataset: ExplicitMultiHistoryDataset, index: int) -> dict[str, Any]:
    before = int(getattr(dataset, "_sample_failure_count", 0))
    sample = dataset[index]
    after = int(getattr(dataset, "_sample_failure_count", 0))
    if after != before:
        raise RuntimeError(f"Explicit sample index={index} fell back to a dummy sample")
    # Dataset samples intentionally expose no identity/frame/pose metadata.
    # Build an audit-only sidecar from the already verified manifest index;
    # transform/forward below whitelist only RGB and target tensors.
    runtime_frames = tuple(int(value) for value in dataset._explicit_history_frames[index])
    expected_runtime_frames = tuple(int(value) for value in dataset._explicit_history_frames[index])
    if runtime_frames != expected_runtime_frames:
        raise RuntimeError(f"Explicit runtime history order mismatch at index={index}")
    if sorted(runtime_frames) != sorted(dataset._explicit_canonical_frames[index]):
        raise RuntimeError(f"Explicit runtime anchor set changed at index={index}")
    record = dataset._explicit_records[index]
    _clip_index, current = dataset.sample_index[index]
    current = int(current)
    if current != int(record["current_frame"]):
        raise RuntimeError(f"Explicit runtime current frame changed at index={index}")
    expected_runtime_id = f"{record['relative_clip']}:current={current}:history=" + ",".join(
        str(frame) for frame in runtime_frames
    )
    leaked = [key for key in ("history_rel_poses", "history_poses", "current_pose") if key in sample]
    if leaked:
        raise RuntimeError(f"Pose-free sample leaked model inputs: {leaked}")
    sample["_task36c_audit"] = {
        "manifest_sample_id": dataset._explicit_identities[index],
        "runtime_sample_id": expected_runtime_id,
        "runtime_history_frames": list(runtime_frames),
        "current_frame": current,
        "pose_inputs_removed": True,
    }
    return sample


def normalized_lora_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    result: dict[str, torch.nn.Parameter] = {}
    for name, parameter in model.named_parameters():
        normalized = _normalize_state_key(name)
        if "lora_" not in normalized:
            continue
        if normalized in result and result[normalized] is not parameter:
            raise RuntimeError(f"Distinct LoRA parameters normalize to the same key: {normalized}")
        result[normalized] = parameter
    return result


def lora_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in normalized_lora_parameters(model).items()}


def pose_free_head_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    matcher = model.heatmap_vln.pose_free_matcher
    if matcher is None:
        raise RuntimeError("PoseFreeHistoryMatcher was not materialized")
    state = {name: value.detach().cpu().clone() for name, value in matcher.state_dict().items()}
    if not state or any("pose" in name or "trajectory" in name for name in state):
        raise RuntimeError("Invalid pose-free matcher state")
    return state


def load_stage1_s2_lora_strict(model: torch.nn.Module, checkpoint: str) -> dict[str, Any]:
    payload = safe_torch_load(checkpoint)
    state = payload.get("trainable_state_dict", {})
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Stage1-S2 checkpoint has no trainable_state_dict: {checkpoint}")
    matched = assert_complete_lora_checkpoint_match(model, state, checkpoint_path=checkpoint)
    if matched != EXPECTED_LORA_TENSORS:
        raise RuntimeError(f"Task-3.6c requires exactly {EXPECTED_LORA_TENSORS} matched LoRA tensors, got {matched}")
    _missing, _unexpected, _loaded = _load_normalized_state_dict(model, state)
    current = lora_state_dict(model)
    if len(current) != EXPECTED_LORA_TENSORS:
        raise RuntimeError(f"Materialized model has {len(current)} LoRA tensors, expected {EXPECTED_LORA_TENSORS}")
    checkpoint_lora = {
        _normalize_state_key(name): value.detach().cpu()
        for name, value in state.items()
        if "lora_" in _normalize_state_key(name)
    }
    if set(checkpoint_lora) != set(current):
        raise RuntimeError("Loaded Stage1-S2 LoRA key set differs from the model")
    mismatched = [name for name in current if not torch.equal(current[name], checkpoint_lora[name])]
    if mismatched:
        raise RuntimeError(f"Stage1-S2 LoRA values were not loaded exactly: {mismatched[:5]}")
    return {
        "path": str(Path(checkpoint).resolve()),
        "file_sha256": file_sha256(checkpoint),
        "matched_lora_tensors": matched,
        "loaded_lora_sha256": tensor_state_sha256(current),
    }


def strict_load_named_state(
    expected: dict[str, torch.nn.Parameter],
    state: dict[str, torch.Tensor],
    *,
    label: str,
) -> None:
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(
        name for name in set(expected) & set(state) if tuple(expected[name].shape) != tuple(state[name].shape)
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"Strict {label} load failed: missing={missing[:5]} unexpected={unexpected[:5]} "
            f"shape_mismatch={mismatched[:5]}"
        )
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


def configure_trainable(
    model: torch.nn.Module,
    branch: str,
    max_lora_layer: int,
) -> tuple[list[torch.nn.Parameter], dict[str, torch.nn.Parameter]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    matcher = model.heatmap_vln.pose_free_matcher
    if matcher is None:
        raise RuntimeError("Pose-free matcher is absent")
    head = list(matcher.parameters())
    for parameter in head:
        parameter.requires_grad_(True)

    trainable_lora: dict[str, torch.nn.Parameter] = {}
    all_lora = normalized_lora_parameters(model)
    if len(all_lora) != EXPECTED_LORA_TENSORS:
        raise RuntimeError(f"Expected {EXPECTED_LORA_TENSORS} LoRA tensors, got {len(all_lora)}")
    if branch == "heatmap-lora":
        for name, parameter in all_lora.items():
            match = LORA_LAYER_RE.search(name)
            if match is None:
                raise RuntimeError(f"Cannot parse LoRA layer from {name}")
            if int(match.group(1)) <= max_lora_layer:
                parameter.requires_grad_(True)
                trainable_lora[name] = parameter
        if not trainable_lora:
            raise RuntimeError("heatmap-lora branch has no reachable trainable LoRA")
    return head, trainable_lora


def assert_runtime_model_contract(model: torch.nn.Module) -> dict[str, Any]:
    heatmap = model.heatmap_vln
    if heatmap.decoder_mode != "pose_free_matcher" or heatmap.pose_free_matcher is None:
        raise RuntimeError("Runtime decoder is not PoseFreeHistoryMatcher")
    if heatmap.enable_trajectory or heatmap.vit_layer_indices or heatmap.feat_extractor.vit_layer_indices:
        raise RuntimeError("Pose-free runtime unexpectedly enabled trajectory or ViT hooks")
    if list(heatmap.llm_layer_indices) != [20]:
        raise RuntimeError(f"Pose-free runtime must hook only LLM layer 20, got {heatmap.llm_layer_indices}")
    signature = inspect.signature(heatmap.pose_free_matcher.forward).parameters
    if "history_rel_poses" in signature or heatmap.pose_free_matcher.uses_relative_pose is not False:
        raise RuntimeError("Pose-free matcher exposes a pose input")
    return {
        "decoder_mode": heatmap.decoder_mode,
        "trajectory_enabled": False,
        "vit_hooks": [],
        "llm_hooks": [20],
        "matcher_uses_relative_pose": False,
        "head_trainable_parameters": heatmap.pose_free_matcher.trainable_parameter_count,
        "isolated_pair_chains": True,
        "histories_per_qwen_chain": 1,
        "history_anchor_number_per_chain": 1,
        "qwen_forward_batch_size": 1,
        "qwen_forwards_per_sample": 4,
    }


def transform_sample(
    sample: dict[str, Any],
    *,
    intervention: str,
    partner: dict[str, Any] | None = None,
    target_slot: int | None = None,
) -> dict[str, Any]:
    if intervention not in INTERVENTIONS:
        raise ValueError(f"Unknown intervention: {intervention}")
    audit = sample.get("_task36c_audit")
    if not isinstance(audit, dict) or audit.get("pose_inputs_removed") is not True:
        raise RuntimeError("Pose-free transform requires an exact-sample audit sidecar")
    leaked = [key for key in ("history_rel_poses", "history_poses", "current_pose") if key in sample]
    if leaked:
        raise RuntimeError(f"Pose-free transform received leaked pose inputs: {leaked}")
    current_views = sample["current_views"]
    current_frame = sample["current_frame"]
    histories = sample["history_panoramas"]
    history_frames = sample["history_frames"]
    k = int(histories.shape[0])
    metadata: dict[str, Any] = {"intervention": intervention, "target_slot": None}

    if intervention == "blank-images":
        current_views = torch.zeros_like(current_views)
        current_frame = torch.zeros_like(current_frame)
        histories = torch.zeros_like(histories)
        history_frames = torch.zeros_like(history_frames)
    elif intervention == "history-shuffle":
        order = torch.arange(k - 1, -1, -1)
        histories = histories[order]
        history_frames = history_frames[order]
        metadata["history_permutation"] = order.tolist()
    elif intervention == "current-shuffle":
        if partner is None:
            raise ValueError("current-shuffle requires a partner")
        current_views = partner["current_views"]
        current_frame = partner["current_frame"]
    elif intervention == "single-anchor-swap":
        if partner is None or target_slot is None:
            raise ValueError("single-anchor-swap requires partner and target_slot")
        if target_slot < 0 or target_slot >= k:
            raise ValueError(f"target_slot={target_slot} is outside K={k}")
        histories = histories.clone()
        history_frames = history_frames.clone()
        histories[target_slot] = partner["history_panoramas"][target_slot]
        history_frames[target_slot] = partner["history_frames"][target_slot]
        metadata["target_slot"] = int(target_slot)

    return {
        "current_views": current_views,
        "current_frame": current_frame,
        "history_panoramas": histories,
        "history_frames": history_frames,
        "gt_visibility": sample["gt_visibility"],
        "gt_heatmaps": sample["heatmap"],
        "sample_id": audit["runtime_sample_id"],
        "metadata": metadata,
    }


def flatten_isolated_pair_chains(
    transformed: dict[str, Any],
) -> dict[str, torch.Tensor | list[int]]:
    """Expand one K-history sample into K causally isolated pair chains.

    Each Qwen conversation contains the same current panorama and exactly one
    historical panorama.  Therefore every history query is anchor number 1
    and cannot see another history through its causal prefix.
    """
    histories = transformed["history_panoramas"]
    history_frames = transformed["history_frames"]
    current_views = transformed["current_views"]
    current_frame = transformed["current_frame"]
    if histories.ndim != 5 or history_frames.ndim != 4:
        raise ValueError(
            "Expected histories [K,4,C,H,W] and history_frames [K,C,H,W], "
            f"got {tuple(histories.shape)} and {tuple(history_frames.shape)}"
        )
    k = int(histories.shape[0])
    if k != 4 or int(history_frames.shape[0]) != k:
        raise ValueError(f"Task-3.6c isolated-pair pilot requires K=4, got K={k}")
    return {
        "video_frames": torch.stack((history_frames, current_frame.unsqueeze(0).expand_as(history_frames)), dim=1),
        "current_observation": current_frame.unsqueeze(0).expand(k, -1, -1, -1),
        "current_views": current_views.unsqueeze(0).expand(k, -1, -1, -1, -1),
        "history_panoramas": histories.unsqueeze(1),
        "num_histories": [1] * k,
    }


def assert_blank_chain_input_identity(
    chains: dict[str, torch.Tensor | list[int]],
) -> dict[str, Any]:
    """Require every model-side tensor to be identical across blank B=1 calls."""
    checked = {}
    for key in ("video_frames", "current_observation", "current_views", "history_panoramas"):
        tensor = chains[key]
        if not torch.is_tensor(tensor) or int(tensor.shape[0]) != 4:
            raise RuntimeError(f"Blank-chain contract expected {key} with leading K=4")
        if not torch.equal(tensor, tensor[:1].expand_as(tensor)):
            raise RuntimeError(f"Blank intervention produced non-identical chain input: {key}")
        checked[key] = {
            "shape": list(tensor.shape),
            "four_chains_bitwise_identical": True,
        }
    return checked


def regroup_isolated_pair_outputs(
    outputs: list[dict[str, torch.Tensor]],
    *,
    num_histories: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(outputs) != num_histories:
        raise RuntimeError(f"Expected {num_histories} independent B=1 outputs, got {len(outputs)}")
    visibility_rows = []
    heatmap_rows = []
    for chain_index, output in enumerate(outputs):
        visibility = output["visibility"]
        heatmaps = output["heatmaps"]
        if tuple(visibility.shape) != (1, 1, 4):
            raise RuntimeError(
                f"Isolated chain {chain_index} visibility must be [1,1,4], got {tuple(visibility.shape)}"
            )
        if heatmaps.ndim != 5 or tuple(heatmaps.shape[:3]) != (1, 1, 4):
            raise RuntimeError(
                f"Isolated chain {chain_index} heatmaps must be [1,1,4,H,W], got {tuple(heatmaps.shape)}"
            )
        visibility_rows.append(visibility[:, 0])
        heatmap_rows.append(heatmaps[:, 0])
    return torch.stack(visibility_rows, dim=1), torch.stack(heatmap_rows, dim=1)


def assert_blank_output_identity(
    visibility: torch.Tensor,
    heatmaps: torch.Tensor,
    heatmap_logits: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Fail if four identical blank B=1 chains produce different outputs."""
    if visibility.shape[:2] != (1, 4) or heatmaps.shape[:2] != (1, 4):
        raise RuntimeError("Blank-output identity gate requires regrouped [1,4,...] outputs")
    checks = {
        "visibility": bool(torch.equal(visibility, visibility[:, :1].expand_as(visibility))),
        "heatmaps": bool(torch.equal(heatmaps, heatmaps[:, :1].expand_as(heatmaps))),
    }
    if heatmap_logits is not None:
        if heatmap_logits.shape != heatmaps.shape:
            raise RuntimeError("Blank-output raw/probability heatmap shapes differ")
        checks["heatmap_logits"] = bool(torch.equal(heatmap_logits, heatmap_logits[:, :1].expand_as(heatmap_logits)))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Identical blank B=1 chains produced row-specific outputs: " + ", ".join(failed))
    return {
        "four_blank_chain_outputs_bitwise_identical": True,
        "checked_tensors": sorted(checks),
    }


def forward_loss(
    model: torch.nn.Module,
    criterion: HeatmapVLNLoss,
    transformed: dict[str, Any],
    device: torch.device,
    *,
    history_rel_poses: torch.Tensor | None = None,
    return_heatmap_logits: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if history_rel_poses is not None:
        raise ValueError("Task-3.6c pose-free forward received non-None history_rel_poses")
    chains = flatten_isolated_pair_chains(transformed)
    num_histories = int(transformed["history_panoramas"].shape[0])
    blank_input_gate = None
    if transformed.get("metadata", {}).get("intervention") == "blank-images":
        blank_input_gate = assert_blank_chain_input_identity(chains)
    outputs = []
    for chain_index in range(num_histories):
        # Four physically separate B=1 calls are the attribution contract.
        # No Qwen batch row may encode or perturb history identity.
        model_kwargs = {}
        if return_heatmap_logits:
            model_kwargs["return_heatmap_logits"] = True
        output = model(
            video_frames=chains["video_frames"][chain_index : chain_index + 1].to(device),
            current_observation=chains["current_observation"][chain_index : chain_index + 1].to(device),
            current_views=chains["current_views"][chain_index : chain_index + 1],
            history_panoramas=chains["history_panoramas"][chain_index : chain_index + 1],
            history_rel_poses=None,
            return_heatmaps=True,
            return_actions=False,
            return_lm_loss=False,
            **model_kwargs,
        )
        outputs.append(output)
    pred_vis, pred_heatmaps = regroup_isolated_pair_outputs(
        outputs,
        num_histories=num_histories,
    )
    pred_heatmap_logits = None
    if return_heatmap_logits:
        logits_rows = []
        for chain_index, output in enumerate(outputs):
            logits = output.get("heatmap_logits")
            if not torch.is_tensor(logits):
                raise RuntimeError(f"Isolated chain {chain_index} omitted explicitly requested raw heatmap_logits")
            if logits.ndim != 5 or tuple(logits.shape[:3]) != (1, 1, 4):
                raise RuntimeError(
                    f"Isolated chain {chain_index} heatmap_logits must be [1,1,4,H,W], got {tuple(logits.shape)}"
                )
            if tuple(logits.shape) != tuple(output["heatmaps"].shape):
                raise RuntimeError(f"Isolated chain {chain_index} raw/probability heatmap shapes differ")
            logits_rows.append(logits[:, 0])
        pred_heatmap_logits = torch.stack(logits_rows, dim=1)
    blank_output_gate = None
    if transformed.get("metadata", {}).get("intervention") == "blank-images":
        blank_output_gate = assert_blank_output_identity(
            pred_vis,
            pred_heatmaps,
            pred_heatmap_logits,
        )
    gt_vis = transformed["gt_visibility"].unsqueeze(0).to(device)
    gt_heatmaps = transformed["gt_heatmaps"].unsqueeze(0).to(device)
    history_mask = torch.ones(gt_vis.shape[:2], dtype=torch.bool, device=device)
    losses = criterion(
        pred_vis,
        pred_heatmaps,
        gt_vis=gt_vis,
        gt_heatmaps=gt_heatmaps,
        history_mask=history_mask,
    )
    record = {
        "visibility": pred_vis.detach().float().cpu(),
        "heatmaps": pred_heatmaps.detach().float().cpu(),
        "gt_visibility": transformed["gt_visibility"].detach().float().cpu(),
        "gt_heatmaps": transformed["gt_heatmaps"].detach().float().cpu(),
    }
    if pred_heatmap_logits is not None:
        record["heatmap_logits"] = pred_heatmap_logits.detach().float().cpu()
    if blank_output_gate is not None:
        record["blank_input_identity_gate"] = blank_input_gate
        record["blank_output_identity_gate"] = blank_output_gate
    return losses["total"], record


def gradient_summary(parameters: dict[str, torch.nn.Parameter]) -> dict[str, Any]:
    square_sum = 0.0
    with_grad = 0
    nonzero_names = []
    per_layer: dict[int, dict[str, float | int]] = defaultdict(lambda: {"tensors": 0, "square_sum": 0.0})
    for name, parameter in parameters.items():
        if parameter.grad is None:
            continue
        with_grad += 1
        norm = float(parameter.grad.detach().float().norm().item())
        square_sum += norm * norm
        if norm > 0:
            nonzero_names.append(name)
            match = LORA_LAYER_RE.search(name)
            if match is not None:
                layer = int(match.group(1))
                per_layer[layer]["tensors"] = int(per_layer[layer]["tensors"]) + 1
                per_layer[layer]["square_sum"] = float(per_layer[layer]["square_sum"]) + norm * norm
    return {
        "tensors_with_grad": with_grad,
        "tensors_with_nonzero_grad": len(nonzero_names),
        "nonzero_names": nonzero_names,
        "total_grad_norm": math.sqrt(square_sum),
        "per_layer": {
            str(layer): {
                "nonzero_tensors": int(values["tensors"]),
                "grad_norm": math.sqrt(float(values["square_sum"])),
            }
            for layer, values in sorted(per_layer.items())
        },
    }


def delta_summary(initial: dict[str, torch.Tensor], final: dict[str, torch.Tensor]) -> dict[str, Any]:
    if set(initial) != set(final):
        raise RuntimeError("State key set changed while computing drift")
    square_sum = 0.0
    nonzero = 0
    per_layer_square: dict[int, float] = defaultdict(float)
    per_layer_count: dict[int, int] = defaultdict(int)
    for name in initial:
        delta = final[name].float() - initial[name].float()
        norm = float(delta.norm().item())
        square_sum += norm * norm
        if norm > 0:
            nonzero += 1
            match = LORA_LAYER_RE.search(name)
            if match is not None:
                layer = int(match.group(1))
                per_layer_square[layer] += norm * norm
                per_layer_count[layer] += 1
    return {
        "total_delta_norm": math.sqrt(square_sum),
        "tensors_with_nonzero_delta": nonzero,
        "per_layer": {
            str(layer): {
                "nonzero_tensors": per_layer_count[layer],
                "delta_norm": math.sqrt(per_layer_square[layer]),
            }
            for layer in sorted(per_layer_square)
        },
    }


def make_criterion(cfg: dict[str, Any], device: torch.device) -> HeatmapVLNLoss:
    kwargs = dict(cfg["loss"]["heatmap_vln"])
    kwargs["heatmap_size"] = tuple(cfg["data"]["init_hm_size"])
    kwargs["lambda_coord"] = 0.0
    return HeatmapVLNLoss(**kwargs).to(device)


def binary_curves(scores: np.ndarray, labels: np.ndarray) -> tuple[float | None, float | None]:
    labels = labels.astype(np.int64)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None, None
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    group_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    tp = tp[group_ends]
    fp = fp[group_ends]
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1)
    tpr = np.r_[0.0, recall]
    fpr = np.r_[0.0, fp / negatives]
    return float(np.trapz(tpr, fpr)), float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _peak_xy(heatmap: torch.Tensor) -> tuple[int, int]:
    index = int(heatmap.reshape(-1).argmax().item())
    width = int(heatmap.shape[-1])
    return index % width, index // width


def compute_metrics(
    records: list[dict[str, Any]],
    *,
    slot: int | None = None,
    dynamic_slots: str | None = None,
) -> dict[str, Any]:
    if slot is not None and dynamic_slots is not None:
        raise ValueError("slot and dynamic_slots are mutually exclusive")
    if dynamic_slots not in (None, "targeted", "untargeted"):
        raise ValueError(f"Unknown dynamic_slots mode: {dynamic_slots}")
    visibility_scores = []
    visibility_targets = []
    pixel_errors: list[float] = []
    joint_errors: list[float] = []
    view_correct = 0
    visible_histories = 0
    identity_correct = 0
    identity_total = 0
    k = int(records[0]["gt_visibility"].shape[0]) if records else 0
    confusion = [[0 for _ in range(k)] for _ in range(k)]

    for record in records:
        pred_vis = record["visibility"].squeeze(0)
        pred_heatmaps = record["heatmaps"].squeeze(0)
        gt_vis = record["gt_visibility"]
        gt_heatmaps = record["gt_heatmaps"]
        if slot is not None:
            slots = [slot]
        elif dynamic_slots is not None:
            target_slot = record.get("target_slot")
            if target_slot is None:
                raise ValueError(f"{dynamic_slots} metrics require target_slot on every record")
            slots = (
                [int(target_slot)]
                if dynamic_slots == "targeted"
                else [value for value in range(int(gt_vis.shape[0])) if value != int(target_slot)]
            )
        else:
            slots = list(range(int(gt_vis.shape[0])))
        for history_slot in slots:
            visibility_scores.extend(pred_vis[history_slot].sigmoid().tolist())
            visibility_targets.extend((gt_vis[history_slot] > 0.5).tolist())
            positives = torch.nonzero(gt_vis[history_slot] > 0.5).flatten()
            if positives.numel() == 0:
                continue
            visible_histories += 1
            predicted_view = int(pred_vis[history_slot].argmax().item())
            selected_is_positive = bool((positives == predicted_view).any())
            if selected_is_positive:
                view_correct += 1
                px, py = _peak_xy(pred_heatmaps[history_slot, predicted_view])
                gx, gy = _peak_xy(gt_heatmaps[history_slot, predicted_view])
                joint_errors.append(math.hypot(px - gx, py - gy))
            else:
                joint_errors.append(float("inf"))
            for view in positives.tolist():
                px, py = _peak_xy(pred_heatmaps[history_slot, view])
                gx, gy = _peak_xy(gt_heatmaps[history_slot, view])
                pixel_errors.append(math.hypot(px - gx, py - gy))

            # Anchor identity: classify the output peak by its nearest target
            # among the K anchors on the circular four-view panorama.
            px, py = _peak_xy(pred_heatmaps[history_slot, predicted_view])
            width = int(gt_heatmaps.shape[-1])
            panorama_x = predicted_view * width + px
            candidates = []
            for target_slot in range(k):
                target_views = torch.nonzero(gt_vis[target_slot] > 0.5).flatten()
                if target_views.numel() == 0:
                    continue
                target_view = max(
                    target_views.tolist(),
                    key=lambda view: float(gt_heatmaps[target_slot, view].max().item()),
                )
                gx, gy = _peak_xy(gt_heatmaps[target_slot, target_view])
                target_x = target_view * width + gx
                dx = abs(panorama_x - target_x)
                dx = min(dx, 4 * width - dx)
                candidates.append((math.hypot(dx, py - gy), target_slot))
            if candidates:
                nearest = min(candidates)[1]
                confusion[history_slot][nearest] += 1
                identity_total += 1
                identity_correct += int(nearest == history_slot)

    scores_np = np.asarray(visibility_scores, dtype=np.float64)
    targets_np = np.asarray(visibility_targets, dtype=np.bool_)
    predicted_np = scores_np >= 0.5
    tp = int(np.logical_and(predicted_np, targets_np).sum())
    fp = int(np.logical_and(predicted_np, ~targets_np).sum())
    fn = int(np.logical_and(~predicted_np, targets_np).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    auroc, auprc = binary_curves(scores_np, targets_np)
    errors = np.asarray(pixel_errors, dtype=np.float64)
    joint = np.asarray(joint_errors, dtype=np.float64)
    joint_median = float(np.median(joint)) if joint.size else None
    if joint_median is not None and not math.isfinite(joint_median):
        joint_median = None
    return {
        "visibility_auroc": auroc,
        "visibility_auprc": auprc,
        "visibility_f1": float(f1),
        "visible_view_accuracy": view_correct / max(visible_histories, 1),
        "visible_history_count": visible_histories,
        "median_pixel_error": float(np.median(errors)) if errors.size else None,
        "pck4": float((errors <= 4.0).mean()) if errors.size else None,
        "pck8": float((errors <= 8.0).mean()) if errors.size else None,
        "joint_median_pixel_error": joint_median,
        "joint_pck4": float((joint <= 4.0).mean()) if joint.size else None,
        "joint_pck8": float((joint <= 8.0).mean()) if joint.size else None,
        "anchor_identity": {
            "accuracy": identity_correct / max(identity_total, 1),
            "correct": identity_correct,
            "count": identity_total,
            "chance": 1.0 / k if k else None,
            "confusion_matrix": confusion,
        },
    }


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    pred_vis = record["visibility"].squeeze(0)
    pred_heatmaps = record["heatmaps"].squeeze(0)
    gt_heatmaps = record["gt_heatmaps"]
    return {
        "sample_id": record["sample_id"],
        "target_slot": record.get("target_slot"),
        "visibility_logits": pred_vis.tolist(),
        "gt_visibility": record["gt_visibility"].tolist(),
        "pred_xy": [[list(_peak_xy(pred_heatmaps[k, v])) for v in range(4)] for k in range(pred_heatmaps.shape[0])],
        "gt_xy": [[list(_peak_xy(gt_heatmaps[k, v])) for v in range(4)] for k in range(gt_heatmaps.shape[0])],
    }


def paired_single_swap_output_change(
    standard_records: list[dict[str, Any]],
    swap_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure whether replacing history i changes output i, not the others."""
    standard_by_id = {record["sample_id"]: record for record in standard_records}
    if len(standard_by_id) != len(standard_records):
        raise ValueError("Standard prediction records contain duplicate sample IDs")
    buckets: dict[str, dict[str, list[float]]] = {
        "targeted": {"heatmap_l1": [], "visibility_l1": [], "peak_displacement": []},
        "untargeted": {"heatmap_l1": [], "visibility_l1": [], "peak_displacement": []},
    }
    for swapped in swap_records:
        baseline = standard_by_id.get(swapped["sample_id"])
        target_slot = swapped.get("target_slot")
        if baseline is None or target_slot is None:
            raise ValueError("Swap prediction has no paired standard record/target_slot")
        base_vis = baseline["visibility"].squeeze(0)
        swap_vis = swapped["visibility"].squeeze(0)
        base_hm = baseline["heatmaps"].squeeze(0)
        swap_hm = swapped["heatmaps"].squeeze(0)
        width = int(base_hm.shape[-1])
        for output_slot in range(int(base_vis.shape[0])):
            bucket = buckets["targeted" if output_slot == int(target_slot) else "untargeted"]
            bucket["heatmap_l1"].append(float((swap_hm[output_slot] - base_hm[output_slot]).abs().mean().item()))
            bucket["visibility_l1"].append(float((swap_vis[output_slot] - base_vis[output_slot]).abs().mean().item()))
            base_view = int(base_vis[output_slot].argmax().item())
            swap_view = int(swap_vis[output_slot].argmax().item())
            base_x, base_y = _peak_xy(base_hm[output_slot, base_view])
            swap_x, swap_y = _peak_xy(swap_hm[output_slot, swap_view])
            base_panorama_x = base_view * width + base_x
            swap_panorama_x = swap_view * width + swap_x
            dx = abs(base_panorama_x - swap_panorama_x)
            dx = min(dx, 4 * width - dx)
            bucket["peak_displacement"].append(math.hypot(dx, base_y - swap_y))

    summary: dict[str, Any] = {}
    for label, values in buckets.items():
        summary[label] = {
            "comparisons": len(values["heatmap_l1"]),
            **{f"mean_{name}": float(np.mean(items)) for name, items in values.items()},
        }
    untargeted_l1 = summary["untargeted"]["mean_heatmap_l1"]
    summary["targeted_to_untargeted_heatmap_l1_ratio"] = (
        summary["targeted"]["mean_heatmap_l1"] / untargeted_l1 if untargeted_l1 > 0 else None
    )
    summary["contract"] = "replace history i; compare output i against all output j!=i on the same current"
    return summary


def assert_history_permutation_equivariance(
    standard_records: list[dict[str, Any]],
    shuffled_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Require inverse-permuted B=1 outputs to exactly recover standard."""
    standard_by_id = {record["sample_id"]: record for record in standard_records}
    if len(standard_by_id) != len(standard_records) or len(shuffled_records) != len(standard_records):
        raise RuntimeError("History-permutation gate requires paired unique standard/shuffled records")
    comparisons = 0
    for shuffled in shuffled_records:
        sample_id = shuffled["sample_id"]
        standard = standard_by_id.get(sample_id)
        permutation = shuffled.get("history_permutation")
        if standard is None or not isinstance(permutation, list) or sorted(permutation) != list(range(4)):
            raise RuntimeError(f"History-permutation gate has invalid pair/metadata for {sample_id}")
        inverse = [0] * len(permutation)
        for shuffled_slot, original_slot in enumerate(permutation):
            inverse[int(original_slot)] = shuffled_slot
        tensor_keys = ["visibility", "heatmaps"]
        if "heatmap_logits" in standard or "heatmap_logits" in shuffled:
            if "heatmap_logits" not in standard or "heatmap_logits" not in shuffled:
                raise RuntimeError("History permutation raw-logit records are incomplete")
            tensor_keys.append("heatmap_logits")
        for key in tensor_keys:
            restored = shuffled[key][:, inverse]
            expected = standard[key]
            if not torch.equal(restored, expected):
                maximum = float((restored - expected).abs().max().item())
                raise RuntimeError(
                    "History permutation equivariance failed after inverse reorder: "
                    f"sample={sample_id} tensor={key} max_abs_difference={maximum:.6e}"
                )
            comparisons += 1
    return {
        "passed": True,
        "bitwise_exact": True,
        "samples": len(shuffled_records),
        "tensor_comparisons": comparisons,
        "maximum_abs_difference": 0.0,
        "contract": "inverse(permutation(output(shuffled histories))) == output(standard histories)",
    }


@torch.no_grad()
def evaluate_intervention(
    model: torch.nn.Module,
    criterion: HeatmapVLNLoss,
    dataset: ExplicitMultiHistoryDataset,
    intervention: str,
    device: torch.device,
    *,
    return_heatmap_logits: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    model.heatmap_vln.feat_extractor.detach_features = True
    records: list[dict[str, Any]] = []
    losses = []
    for position in range(len(dataset)):
        sample = exact_sample(dataset, position)
        partner = exact_sample(dataset, (position + 1) % len(dataset)) if len(dataset) > 1 else None
        k = int(sample["history_panoramas"].shape[0])
        target_slots = range(k) if intervention == "single-anchor-swap" else (None,)
        for target_slot in target_slots:
            transformed = transform_sample(
                sample,
                intervention=intervention,
                partner=partner,
                target_slot=target_slot,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss, record = forward_loss(
                    model,
                    criterion,
                    transformed,
                    device,
                    history_rel_poses=None,
                    return_heatmap_logits=return_heatmap_logits,
                )
            record["sample_id"] = transformed["sample_id"]
            record["target_slot"] = transformed["metadata"].get("target_slot")
            record["history_permutation"] = transformed["metadata"].get("history_permutation")
            records.append(record)
            losses.append(float(loss.detach().float().item()))
    metrics = compute_metrics(records)
    metrics["per_slot"] = {str(slot): compute_metrics(records, slot=slot) for slot in range(4)}
    metrics["loss"] = float(np.mean(losses))
    metrics["samples"] = len(records)
    if intervention == "blank-images":
        if not all(
            record.get("blank_input_identity_gate")
            and record.get("blank_output_identity_gate", {}).get("four_blank_chain_outputs_bitwise_identical")
            for record in records
        ):
            raise RuntimeError("Blank input/output identity gates were not recorded for every sample")
        metrics["blank_output_identity_gate"] = {
            "passed": True,
            "bitwise_exact": True,
            "samples": len(records),
            "maximum_abs_difference": 0.0,
        }
        metrics["blank_input_identity_gate"] = {
            "passed": True,
            "bitwise_exact": True,
            "samples": len(records),
        }
    if intervention == "single-anchor-swap":
        metrics["source_samples"] = len(dataset)
        metrics["swap_evaluations_per_sample"] = 4
        metrics["targeted_slot_metrics"] = compute_metrics(records, dynamic_slots="targeted")
        metrics["untargeted_slot_metrics"] = compute_metrics(records, dynamic_slots="untargeted")
    return metrics, records


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    args: argparse.Namespace,
    stage1_contract: dict[str, Any],
    manifest_contract: dict[str, Any],
    config_contract: dict[str, Any],
    runtime_contract: dict[str, Any],
    initial_head_hash: str,
    initial_lora_hash: str,
    train_log: list[dict[str, Any]],
    schedule_hash: str,
) -> dict[str, Any]:
    head = pose_free_head_state_dict(model)
    lora = lora_state_dict(model)
    if len(lora) != EXPECTED_LORA_TENSORS:
        raise RuntimeError("Refusing to save an incomplete all-LoRA checkpoint")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "branch": args.branch,
        "step": args.train_steps,
        "training_pid": os.getpid(),
        "head_state_dict": head,
        "lora_state_dict": lora,
        "head_state_sha256": tensor_state_sha256(head),
        "lora_state_sha256": tensor_state_sha256(lora),
        "initial_head_sha256": initial_head_hash,
        "initial_lora_sha256": initial_lora_hash,
        "expected_lora_tensors": EXPECTED_LORA_TENSORS,
        "stage1_s2_contract": stage1_contract,
        "manifest_contract": manifest_contract,
        "pose_free_config_contract": config_contract,
        "runtime_contract": runtime_contract,
        "training_sample_schedule_sha256": schedule_hash,
        "train_log": train_log,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "head_state_sha256": payload["head_state_sha256"],
        "lora_state_sha256": payload["lora_state_sha256"],
        "lora_tensors": len(lora),
    }


def validate_pilot_checkpoint_payload_strict(
    path: str | Path,
    *,
    branch: str,
    stage1_contract: dict[str, Any],
    manifest_contract: dict[str, Any],
    runtime_contract: dict[str, Any] | None = None,
    config_contract: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate a pilot checkpoint without loading either of its model states."""
    payload = safe_torch_load(path)
    if not isinstance(payload, dict):
        raise RuntimeError("Pilot checkpoint payload is not a mapping")
    if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("branch") != branch:
        raise RuntimeError("Pilot checkpoint schema/branch mismatch")
    if int(payload.get("expected_lora_tensors", -1)) != EXPECTED_LORA_TENSORS:
        raise RuntimeError("Pilot checkpoint has the wrong all-LoRA tensor count contract")

    checkpoint_config = payload.get("pose_free_config_contract")
    if not isinstance(checkpoint_config, dict) or checkpoint_config.get("isolated_pair_chains") is not True:
        raise RuntimeError("Pilot checkpoint was not trained with causally isolated pair chains")
    if config_contract is not None and checkpoint_config != config_contract:
        raise RuntimeError("Pilot checkpoint pose-free config contract mismatch")

    checkpoint_runtime = payload.get("runtime_contract")
    if not isinstance(checkpoint_runtime, dict):
        raise RuntimeError("Pilot checkpoint runtime contract is missing")
    if checkpoint_runtime.get("histories_per_qwen_chain") != 1:
        raise RuntimeError("Pilot checkpoint history-query isolation contract is invalid")
    if (
        checkpoint_runtime.get("qwen_forward_batch_size") != 1
        or checkpoint_runtime.get("qwen_forwards_per_sample") != 4
    ):
        raise RuntimeError("Pilot checkpoint was not trained with four strict B=1 Qwen forwards")
    if runtime_contract is not None and checkpoint_runtime != runtime_contract:
        raise RuntimeError("Pilot checkpoint runtime contract differs from the materialized model")

    checkpoint_stage1 = payload.get("stage1_s2_contract")
    if not isinstance(checkpoint_stage1, dict):
        raise RuntimeError("Pilot checkpoint Stage1-S2 contract is missing")
    for key in ("file_sha256", "loaded_lora_sha256", "matched_lora_tensors"):
        if checkpoint_stage1.get(key) != stage1_contract.get(key):
            raise RuntimeError(f"Pilot checkpoint Stage1-S2 contract mismatch: {key}")

    checkpoint_manifest = payload.get("manifest_contract")
    if not isinstance(checkpoint_manifest, dict):
        raise RuntimeError("Pilot checkpoint manifest contract is missing")
    for key in (
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
    ):
        if checkpoint_manifest.get(key) != manifest_contract.get(key):
            raise RuntimeError(f"Pilot checkpoint manifest contract mismatch: {key}")

    head = payload.get("head_state_dict", {})
    lora = payload.get("lora_state_dict", {})
    if not isinstance(head, dict) or not head:
        raise RuntimeError("Pilot checkpoint has no pose-free head state")
    if not isinstance(lora, dict):
        raise RuntimeError("Pilot checkpoint LoRA state is not a mapping")
    if tensor_state_sha256(head) != payload.get("head_state_sha256"):
        raise RuntimeError("Pilot checkpoint head strong hash mismatch")
    if len(lora) != EXPECTED_LORA_TENSORS or tensor_state_sha256(lora) != payload.get("lora_state_sha256"):
        raise RuntimeError("Pilot checkpoint LoRA strong hash/count mismatch")
    if branch == "head-only":
        baseline_lora_hash = stage1_contract["loaded_lora_sha256"]
        if payload.get("initial_lora_sha256") != baseline_lora_hash:
            raise RuntimeError("Head-only checkpoint did not start from the pinned Stage1-S2 LoRA")
        if payload.get("lora_state_sha256") != baseline_lora_hash:
            raise RuntimeError("Head-only checkpoint changed the supposedly frozen LoRA")
    return payload, file_sha256(path)


def load_pilot_checkpoint_strict(
    model: torch.nn.Module,
    path: str,
    *,
    branch: str,
    stage1_contract: dict[str, Any],
    manifest_contract: dict[str, Any],
    eval_lora: str,
    eval_head_checkpoint: str | None = None,
    runtime_contract: dict[str, Any] | None = None,
    config_contract: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, pilot_file_hash = validate_pilot_checkpoint_payload_strict(
        path,
        branch=branch,
        stage1_contract=stage1_contract,
        manifest_contract=manifest_contract,
        runtime_contract=runtime_contract,
        config_contract=config_contract,
    )

    head_payload = payload
    head_path = path
    head_file_hash = pilot_file_hash
    if eval_head_checkpoint is not None:
        if branch != "heatmap-lora" or eval_lora != "trained":
            raise RuntimeError("An alternate eval head is valid only for trained LoRA from the heatmap-lora branch")
        head_payload, head_file_hash = validate_pilot_checkpoint_payload_strict(
            eval_head_checkpoint,
            branch="head-only",
            stage1_contract=stage1_contract,
            manifest_contract=manifest_contract,
            runtime_contract=runtime_contract,
            config_contract=config_contract,
        )
        head_path = eval_head_checkpoint
        paired_fields = (
            "step",
            "initial_head_sha256",
            "initial_lora_sha256",
            "training_sample_schedule_sha256",
        )
        for key in paired_fields:
            if head_payload.get(key) != payload.get(key):
                raise RuntimeError(f"Factorial head/LoRA checkpoints are not paired: {key}")

    matcher = model.heatmap_vln.pose_free_matcher
    if matcher is None:
        raise RuntimeError("Cannot load pilot head without PoseFreeHistoryMatcher")
    expected_head = {name: parameter for name, parameter in matcher.named_parameters()}
    expected_head.update({name: buffer for name, buffer in matcher.named_buffers()})
    strict_load_named_state(expected_head, head_payload["head_state_dict"], label="pose-free head")
    if eval_lora == "trained":
        strict_load_named_state(
            normalized_lora_parameters(model),
            payload["lora_state_dict"],
            label="trained LoRA",
        )
    elif eval_lora != "off":
        raise ValueError(f"Unknown eval_lora source: {eval_lora}")

    active_head_hash = tensor_state_sha256(pose_free_head_state_dict(model))
    if active_head_hash != head_payload["head_state_sha256"]:
        raise RuntimeError("Evaluation head source did not load exactly")
    current_lora = lora_state_dict(model)
    expected_hash = payload["lora_state_sha256"] if eval_lora == "trained" else stage1_contract["loaded_lora_sha256"]
    actual_hash = tensor_state_sha256(current_lora)
    if actual_hash != expected_hash:
        raise RuntimeError(f"Evaluation LoRA source {eval_lora!r} did not load exactly")

    if eval_lora == "trained":
        lora_source = {
            "path": str(Path(path).resolve()),
            "file_sha256": pilot_file_hash,
            "branch": payload["branch"],
            "head_state_sha256": payload["head_state_sha256"],
            "lora_state_sha256": payload["lora_state_sha256"],
        }
    else:
        lora_source = {
            "path": stage1_contract["path"],
            "file_sha256": stage1_contract["file_sha256"],
            "branch": "stage1-s2",
            "head_state_sha256": None,
            "lora_state_sha256": stage1_contract["loaded_lora_sha256"],
        }
    head_source = {
        "path": str(Path(head_path).resolve()),
        "file_sha256": head_file_hash,
        "branch": head_payload["branch"],
        "head_state_sha256": head_payload["head_state_sha256"],
        "lora_state_sha256": head_payload["lora_state_sha256"],
    }
    return payload, {
        "path": str(Path(path).resolve()),
        "file_sha256": pilot_file_hash,
        "branch": payload["branch"],
        "checkpoint_head_state_sha256": payload["head_state_sha256"],
        "head_state_sha256": active_head_hash,
        "eval_lora": eval_lora,
        "active_lora_sha256": actual_hash,
        "head_override": eval_head_checkpoint is not None,
        "lora_source_checkpoint": lora_source,
        "head_source_checkpoint": head_source,
    }


def materialize_model(
    args: argparse.Namespace,
    cfg: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    model = build_model(cfg, verbose=True, device=args.device, enable_action_head=False)
    model.qwen2_5_vl._load_model()
    stage1_contract = load_stage1_s2_lora_strict(model, args.checkpoint)
    set_seed(args.seed + 991)
    model._ensure_heatmap_vln()
    model.heatmap_vln.feat_extractor.detach_features = args.branch == "head-only"
    runtime_contract = assert_runtime_model_contract(model)
    return model, stage1_contract, runtime_contract


def run_train(args: argparse.Namespace) -> int:
    started = time.time()
    cfg = load_pilot_config(args)
    config_contract = pose_free_config_contract(cfg)
    records, manifest_contract = load_manifest_contract(args)
    dataset = build_explicit_dataset(
        cfg,
        "train",
        records["train"],
        seed=args.seed + 3600,
        reshuffle_slots_each_epoch=True,
        max_clip_id=manifest_contract["max_clip_id"],
        expected_inventory_sha256=manifest_contract["split_source_inventories"]["train"]["inventory_sha256"],
        expected_inventory_clips=manifest_contract["split_source_inventories"]["train"]["clips"],
    )
    model, stage1_contract, runtime_contract = materialize_model(args, cfg)
    device = torch.device(args.device)
    criterion = make_criterion(cfg, device)
    initial_head = pose_free_head_state_dict(model)
    initial_lora = lora_state_dict(model)
    initial_head_hash = tensor_state_sha256(initial_head)
    initial_lora_hash = tensor_state_sha256(initial_lora)
    if initial_lora_hash != stage1_contract["loaded_lora_sha256"]:
        raise RuntimeError("Initial pilot LoRA does not equal the strict Stage1-S2 load")

    head_parameters, trainable_lora = configure_trainable(
        model,
        args.branch,
        args.max_trainable_lora_layer,
    )
    groups = [{"name": "pose_free_matcher", "params": head_parameters, "lr": args.head_learning_rate}]
    if trainable_lora:
        groups.append(
            {"name": "reachable_lora", "params": list(trainable_lora.values()), "lr": args.lora_learning_rate}
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    model.eval()
    # Qwen's gradient-checkpointing blocks are active only in train mode.
    # LoRA dropout is forced to zero above, so this changes memory use rather
    # than injecting adapter dropout into the attribution pilot.
    model.qwen2_5_vl.model.train(args.branch == "heatmap-lora")
    model.heatmap_vln.pose_free_matcher.train()

    train_log = []
    schedule_ids = []
    reached_lora_names: set[str] = set()
    max_lora_grad_norm = 0.0
    for step in range(1, args.train_steps + 1):
        epoch, index = divmod(step - 1, len(dataset))
        dataset.set_epoch(epoch)
        sample = exact_sample(dataset, index)
        runtime_audit = sample["_task36c_audit"]
        history_frames = ",".join(str(value) for value in runtime_audit["runtime_history_frames"])
        schedule_ids.append(f"{runtime_audit['runtime_sample_id']}:epoch={epoch}:runtime_history={history_frames}")
        transformed = transform_sample(sample, intervention="standard")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss, _record = forward_loss(model, criterion, transformed, device, history_rel_poses=None)
        loss.backward()
        all_lora_grad = gradient_summary(normalized_lora_parameters(model))
        if args.branch == "head-only" and all_lora_grad["tensors_with_grad"]:
            raise RuntimeError("Head-only branch leaked gradients into LoRA")
        reached_lora_names.update(all_lora_grad["nonzero_names"])
        max_lora_grad_norm = max(max_lora_grad_norm, float(all_lora_grad["total_grad_norm"]))
        torch.nn.utils.clip_grad_norm_(head_parameters + list(trainable_lora.values()), args.grad_clip)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            item = {
                "step": step,
                "loss": float(loss.detach().float().item()),
                "lora_gradient": {key: value for key, value in all_lora_grad.items() if key != "nonzero_names"},
            }
            train_log.append(item)
            LOGGER.info(
                "branch=%s step=%d/%d loss=%.6f lora_grad=%.3e tensors=%d",
                args.branch,
                step,
                args.train_steps,
                item["loss"],
                all_lora_grad["total_grad_norm"],
                all_lora_grad["tensors_with_nonzero_grad"],
            )

    final_head = pose_free_head_state_dict(model)
    final_lora = lora_state_dict(model)
    lora_drift = delta_summary(initial_lora, final_lora)
    head_drift = delta_summary(initial_head, final_head)
    if args.branch == "head-only" and tensor_state_sha256(final_lora) != initial_lora_hash:
        raise RuntimeError("Head-only branch changed frozen LoRA")
    if args.branch == "heatmap-lora":
        if not reached_lora_names or lora_drift["tensors_with_nonzero_delta"] == 0:
            raise RuntimeError("Heatmap loss did not reach/change any LoRA tensor")

    branch_dir = Path(args.output_dir) / args.branch
    checkpoint_path = branch_dir / "checkpoint_final.pth"
    checkpoint_contract = save_checkpoint(
        checkpoint_path,
        model=model,
        args=args,
        stage1_contract=stage1_contract,
        manifest_contract=manifest_contract,
        config_contract=config_contract,
        runtime_contract=runtime_contract,
        initial_head_hash=initial_head_hash,
        initial_lora_hash=initial_lora_hash,
        train_log=train_log,
        schedule_hash=hash_strings(schedule_ids),
    )
    common_eval = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        "eval",
        "--branch",
        args.branch,
        "--config",
        str(Path(args.config).resolve()),
        "--checkpoint",
        str(Path(args.checkpoint).resolve()),
        "--selection-manifest",
        str(Path(args.selection_manifest).resolve()),
        "--data-root",
        str(Path(args.data_root).resolve()),
        "--output-dir",
        str(Path(args.output_dir).resolve()),
        "--device",
        args.device,
        "--pilot-checkpoint",
        str(checkpoint_path.resolve()),
    ]
    if args.model_path:
        common_eval += ["--model-path", str(Path(args.model_path).resolve())]
    if args.source_inventory_sha256:
        common_eval += ["--source-inventory-sha256", args.source_inventory_sha256]
    report = {
        "schema": REPORT_SCHEMA,
        "phase": "train",
        "branch": args.branch,
        "train_steps": args.train_steps,
        "duration_seconds": time.time() - started,
        "stage1_s2_contract": stage1_contract,
        "manifest_contract": manifest_contract,
        "pose_free_config_contract": config_contract,
        "runtime_contract": runtime_contract,
        "explicit_pose_inputs_removed": True,
        "qwen_training_mode": bool(model.qwen2_5_vl.model.training),
        "gradient_checkpointing_requested": args.branch == "heatmap-lora",
        "checkpoint": checkpoint_contract,
        "gradient_reach": {
            "reachable_trainable_lora_tensors": len(trainable_lora),
            "ever_nonzero_lora_tensors": len(reached_lora_names),
            "max_lora_grad_norm": max_lora_grad_norm,
        },
        "lora_drift": lora_drift,
        "head_drift": head_drift,
        "training_sample_schedule_sha256": hash_strings(schedule_ids),
        "train_log": train_log,
        "fresh_process_evaluation_argv": {
            source: [*common_eval, "--eval-lora", source] for source in ("trained", "off")
        },
        "training_process_runs_evaluation": False,
    }
    json_dump(branch_dir / "train_report.json", report)
    return 0


def run_eval(args: argparse.Namespace) -> int:
    started = time.time()
    cfg = load_pilot_config(args)
    config_contract = pose_free_config_contract(cfg)
    records, manifest_contract = load_manifest_contract(args)
    dataset = build_explicit_dataset(
        cfg,
        "val",
        records["val"],
        seed=args.seed + 3600,
        reshuffle_slots_each_epoch=False,
        max_clip_id=manifest_contract["max_clip_id"],
        expected_inventory_sha256=manifest_contract["split_source_inventories"]["val"]["inventory_sha256"],
        expected_inventory_clips=manifest_contract["split_source_inventories"]["val"]["clips"],
    )
    model, stage1_contract, runtime_contract = materialize_model(args, cfg)
    payload, pilot_contract = load_pilot_checkpoint_strict(
        model,
        args.pilot_checkpoint,
        branch=args.branch,
        stage1_contract=stage1_contract,
        manifest_contract=manifest_contract,
        eval_lora=args.eval_lora,
        eval_head_checkpoint=args.eval_head_checkpoint,
        runtime_contract=runtime_contract,
        config_contract=config_contract,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    model.heatmap_vln.feat_extractor.detach_features = True
    device = torch.device(args.device)
    criterion = make_criterion(cfg, device)

    evaluations: dict[str, Any] = {}
    prediction_records: dict[str, list[dict[str, Any]]] = {}
    standard_raw_records: list[dict[str, Any]] | None = None
    for intervention in INTERVENTIONS:
        LOGGER.info("Evaluating %s with eval_lora=%s", intervention, args.eval_lora)
        metrics, raw_records = evaluate_intervention(model, criterion, dataset, intervention, device)
        if intervention == "standard":
            standard_raw_records = raw_records
        elif intervention == "history-shuffle":
            if standard_raw_records is None:
                raise RuntimeError("history-shuffle evaluation requires standard predictions first")
            metrics["permutation_equivariance_gate"] = assert_history_permutation_equivariance(
                standard_raw_records,
                raw_records,
            )
        elif intervention == "single-anchor-swap":
            if standard_raw_records is None:
                raise RuntimeError("single-anchor-swap evaluation requires standard predictions first")
            metrics["paired_output_change_vs_standard"] = paired_single_swap_output_change(
                standard_raw_records,
                raw_records,
            )
        evaluations[intervention] = metrics
        prediction_records[intervention] = [compact_record(record) for record in raw_records]

    output_name = f"eval_{args.eval_lora}"
    if args.eval_head_checkpoint is not None:
        head_source = pilot_contract["head_source_checkpoint"]
        output_name += f"_head_{head_source['branch']}_{head_source['head_state_sha256'][:12]}"
    output_dir = Path(args.output_dir) / args.branch / output_name
    report = {
        "schema": REPORT_SCHEMA,
        "phase": "eval",
        "branch": args.branch,
        "eval_lora": args.eval_lora,
        "duration_seconds": time.time() - started,
        "fresh_process_contract": {
            "training_pid": payload.get("training_pid"),
            "evaluation_pid": os.getpid(),
            "phase_separated": True,
        },
        "stage1_s2_contract": stage1_contract,
        "manifest_contract": manifest_contract,
        "pose_free_config_contract": config_contract,
        "runtime_contract": runtime_contract,
        "explicit_pose_inputs_removed": True,
        "pilot_checkpoint": pilot_contract,
        "checkpoint_sources": {
            "lora": pilot_contract["lora_source_checkpoint"],
            "head": pilot_contract["head_source_checkpoint"],
        },
        "interventions": list(INTERVENTIONS),
        "evaluations": evaluations,
        "prediction_records": prediction_records,
    }
    json_dump(output_dir / "report.json", report)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    return run_train(args) if args.phase == "train" else run_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())

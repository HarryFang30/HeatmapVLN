#!/usr/bin/env python3
"""Train matched head-only heatmap probes and evaluate input shortcuts.

Three independently launched modes share the same seed, data order, frozen
Stage1-S2 LoRA, and freshly initialised HeatmapVLN head:

* ``full``: normal current/history images plus relative pose;
* ``vision-only``: normal images, no pose input;
* ``pose-only``: constant black images plus relative pose.

The full mode also evaluates history/current shuffles and pose conflicts
without retraining.  This script is intentionally diagnostic: LoRA is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training import (  # noqa: E402
    _load_normalized_state_dict,
    assert_complete_lora_checkpoint_match,
    build_model,
    safe_torch_load,
)
from src.data.sliding_window_dataset import VLNSlidingWindowDataset  # noqa: E402
from src.models.heatmap import HeatmapVLNLoss  # noqa: E402


LOGGER = logging.getLogger("heatmap_shortcut_diagnostic")
MODES = ("full", "vision-only", "pose-only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-history", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--head-checkpoint",
        default=None,
        help="Load a previously trained head and run evaluation only.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["data"]["root"] = str(Path(args.data_root).resolve())
    cfg["data"]["dataset_type"] = "sliding_window"
    sw_cfg = cfg["data"].setdefault("sliding_window", {})
    sw_cfg.update(
        num_history_sample=args.num_history,
        load_depth=True,
        load_history_frames=True,
        cache_poses=True,
        sample_stride=2,
        clip_level_sampling=True,
        samples_per_clip=8,
        defer_heatmap_to_gpu=False,
    )

    model_cfg = cfg["model"]
    model_cfg["device"] = args.device
    llm_cfg = model_cfg["llm"]
    llm_cfg["model_path"] = "/mnt/afs/lixiaoou/intern/fjl/InternNav-Model"
    llm_cfg["attn_implementation"] = "sdpa"
    llm_cfg["gradient_checkpointing"] = False
    llm_cfg["enable_compile"] = False
    llm_cfg["lora_dropout"] = 0.0

    heatmap_cfg = model_cfg.setdefault("heatmap", {})
    heatmap_cfg["enable"] = True
    # Build a normal autograd forward, then detach captured features.  This
    # avoids inference tensors being saved by trainable head layers while
    # still guaranteeing zero LoRA gradient.
    heatmap_cfg["heatmap_trains_backbone"] = True
    heatmap_cfg.setdefault("llm_layer_indices", [6, 13, 20])
    heatmap_cfg.setdefault("vit_layer_indices", [7, 15, 23, 31])
    heatmap_cfg.setdefault("trajectory", {}).setdefault("enable", True)
    model_cfg.setdefault("action_head", {})["enable"] = False

    loss_cfg = cfg.setdefault("loss", {}).setdefault("heatmap_vln", {})
    loss_cfg["lambda_coord"] = 0.0
    loss_cfg.setdefault("lambda_vis", 1.0)
    loss_cfg.setdefault("lambda_peak", 1.0)
    loss_cfg.setdefault("lambda_neg", 1.0)
    loss_cfg.setdefault("vis_pos_weight", 7.0)
    return cfg


def build_dataset(cfg: dict[str, Any], split: str) -> VLNSlidingWindowDataset:
    sw_cfg = cfg["data"]["sliding_window"]
    return VLNSlidingWindowDataset(
        root=cfg["data"]["root"],
        split=split,
        min_history=int(sw_cfg.get("min_history", 5)),
        num_history_sample=int(sw_cfg["num_history_sample"]),
        image_size=tuple(cfg["data"]["image_size"]),
        hm_size=tuple(cfg["data"]["init_hm_size"]),
        load_depth=True,
        cache_poses=True,
        sample_stride=2,
        enable_augmentation=False,
        samples_per_clip=8,
        clip_level_sampling=True,
        load_history_frames=True,
        max_clips=0,
    )


def scene_stratified_indices(
    dataset: VLNSlidingWindowDataset,
    limit: int,
) -> list[int]:
    """Round-robin scenes and exclude the return-to-start final frame."""
    by_scene: dict[str, list[int]] = defaultdict(list)
    for sample_idx, (clip_idx, frame_idx) in enumerate(dataset.sample_index):
        valid_frames = dataset._clip_valid_frames.get(clip_idx, [])
        if valid_frames and frame_idx == valid_frames[-1]:
            continue
        scene = dataset.clips[clip_idx].parent.name
        by_scene[scene].append(sample_idx)

    selected: list[int] = []
    scenes = sorted(by_scene)
    cursor = {scene: 0 for scene in scenes}
    while len(selected) < limit:
        made_progress = False
        for scene in scenes:
            position = cursor[scene]
            if position >= len(by_scene[scene]):
                continue
            selected.append(by_scene[scene][position])
            cursor[scene] += 1
            made_progress = True
            if len(selected) >= limit:
                break
        if not made_progress:
            break
    if not selected:
        raise RuntimeError(f"No non-terminal samples selected for split={dataset.split}")
    return selected


def load_stage1_s2_lora(model: torch.nn.Module, checkpoint: str) -> dict[str, int]:
    ckpt = safe_torch_load(checkpoint)
    state = ckpt.get("trainable_state_dict", {})
    if not state:
        raise RuntimeError(f"Checkpoint has no trainable_state_dict: {checkpoint}")
    matched = assert_complete_lora_checkpoint_match(model, state, checkpoint_path=checkpoint)
    missing, unexpected, loaded = _load_normalized_state_dict(model, state)
    return {
        "checkpoint_tensors": len(state),
        "matched_lora_tensors": matched,
        "loaded_tensors": loaded,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }


def heatmap_head_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return HeatmapVLN state without the shared Qwen module reference."""
    return {
        name: value
        for name, value in module.state_dict().items()
        if not name.startswith("qwen.")
    }


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_loss(cfg: dict[str, Any], device: torch.device) -> HeatmapVLNLoss:
    kwargs = dict(cfg["loss"]["heatmap_vln"])
    kwargs["heatmap_size"] = tuple(cfg["data"]["init_hm_size"])
    kwargs["lambda_coord"] = 0.0
    return HeatmapVLNLoss(**kwargs).to(device)


def transform_sample(
    sample: dict[str, Any],
    *,
    train_mode: str,
    perturbation: str,
    partner: dict[str, Any] | None,
) -> dict[str, Any]:
    current_views = sample["current_views"]
    current_frame = sample["current_frame"]
    histories = sample["history_panoramas"]
    history_frames = sample["history_frames"]
    rel_poses: torch.Tensor | None = sample["history_rel_poses"]

    if train_mode == "pose-only" or perturbation == "blank-images":
        current_views = torch.zeros_like(current_views)
        current_frame = torch.zeros_like(current_frame)
        histories = torch.zeros_like(histories)
        history_frames = torch.zeros_like(history_frames)
    if train_mode == "vision-only" or perturbation == "zero-pose":
        rel_poses = None

    if perturbation == "history-shuffle" and histories.shape[0] > 1:
        order = torch.arange(histories.shape[0] - 1, -1, -1)
        histories = histories[order]
        history_frames = history_frames[order]
    elif perturbation == "current-shuffle":
        if partner is None:
            raise ValueError("current-shuffle requires a partner sample")
        current_views = partner["current_views"]
        current_frame = partner["current_frame"]
    elif perturbation in {"pose-conflict", "pose-conflict-shifted-target"} and rel_poses is not None and rel_poses.shape[0] > 1:
        rel_poses = torch.roll(rel_poses, shifts=1, dims=0)

    gt_visibility = sample["gt_visibility"]
    gt_heatmaps = sample["heatmap"]
    if perturbation == "pose-conflict-shifted-target" and gt_heatmaps.shape[0] > 1:
        gt_visibility = torch.roll(gt_visibility, shifts=1, dims=0)
        gt_heatmaps = torch.roll(gt_heatmaps, shifts=1, dims=0)

    return {
        "history_frames": history_frames,
        "current_frame": current_frame,
        "current_views": current_views,
        "history_panoramas": histories,
        "history_rel_poses": rel_poses,
        "gt_visibility": gt_visibility,
        "gt_heatmaps": gt_heatmaps,
    }


def forward_loss(
    model: torch.nn.Module,
    criterion: HeatmapVLNLoss,
    transformed: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    history = transformed["history_frames"].unsqueeze(0).to(device)
    current = transformed["current_frame"].unsqueeze(0).to(device)
    video = torch.cat([history, current.unsqueeze(1)], dim=1)
    rel_poses = transformed["history_rel_poses"]
    if rel_poses is not None:
        rel_poses = rel_poses.unsqueeze(0).to(device)

    output = model(
        video_frames=video,
        current_observation=current,
        current_views=transformed["current_views"].unsqueeze(0),
        history_panoramas=transformed["history_panoramas"].unsqueeze(0),
        history_rel_poses=rel_poses,
        return_heatmaps=True,
        return_actions=False,
        return_lm_loss=False,
    )
    gt_vis = transformed["gt_visibility"].unsqueeze(0).to(device)
    gt_heatmaps = transformed["gt_heatmaps"].unsqueeze(0).to(device)
    history_mask = torch.ones(
        1,
        gt_heatmaps.shape[1],
        dtype=torch.bool,
        device=device,
    )
    losses = criterion(
        output["visibility"],
        output["heatmaps"],
        gt_vis=gt_vis,
        gt_heatmaps=gt_heatmaps,
        history_mask=history_mask,
    )
    detached = {
        "visibility": output["visibility"].detach().float().cpu(),
        "heatmaps": output["heatmaps"].detach().float().cpu(),
        "gt_visibility": transformed["gt_visibility"].detach().float().cpu(),
        "gt_heatmaps": transformed["gt_heatmaps"].detach().float().cpu(),
    }
    return losses["total"], detached


def binary_curves(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Compute tie-aware AUROC and average precision.

    bf16 visibility logits contain many equal scores.  Treating equal values
    as a sequence of different thresholds makes both metrics depend on input
    order, so cumulative counts are sampled only at the end of each tie group.
    """
    labels = labels.astype(np.int64)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan"), float("nan")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    tp = tp[group_ends]
    fp = fp[group_ends]
    tpr = np.concatenate([[0.0], tp / positives])
    fpr = np.concatenate([[0.0], fp / negatives])
    auroc = float(np.trapz(tpr, fpr))
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return auroc, average_precision


def load_heatmap_head_checkpoint(
    module: torch.nn.Module,
    checkpoint_path: str,
) -> tuple[str, dict[str, Any]]:
    payload = safe_torch_load(checkpoint_path)
    state = payload.get("head_state_dict", {})
    if not state:
        raise RuntimeError(f"Head checkpoint has no head_state_dict: {checkpoint_path}")
    expected = heatmap_head_state_dict(module)
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(
        name
        for name in set(expected) & set(state)
        if tuple(expected[name].shape) != tuple(state[name].shape)
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Incompatible diagnostic head checkpoint: "
            f"missing={missing[:5]} unexpected={unexpected[:5]} "
            f"shape_mismatches={mismatched[:5]}"
        )
    load_missing, load_unexpected = module.load_state_dict(state, strict=False)
    non_qwen_missing = [name for name in load_missing if not name.startswith("qwen.")]
    if non_qwen_missing or load_unexpected:
        raise RuntimeError(
            "Diagnostic head load was incomplete: "
            f"missing={non_qwen_missing[:5]} unexpected={list(load_unexpected)[:5]}"
        )
    initial_hash = payload.get("initial_head_hash")
    if not initial_hash:
        raise RuntimeError(f"Head checkpoint lacks initial_head_hash: {checkpoint_path}")
    return str(initial_hash), payload


def compute_metrics(records: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    pred_vis = torch.cat([record["visibility"].reshape(-1) for record in records])
    gt_vis = torch.cat([record["gt_visibility"].reshape(-1) for record in records])
    probabilities = pred_vis.sigmoid()
    predictions = probabilities >= 0.5
    targets = gt_vis >= 0.5
    tp = int((predictions & targets).sum())
    fp = int((predictions & ~targets).sum())
    fn = int((~predictions & targets).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    auroc, auprc = binary_curves(probabilities.numpy(), targets.numpy())

    view_correct = 0
    view_total = 0
    pixel_errors: list[float] = []
    u_errors: list[float] = []
    for record in records:
        vis_logits = record["visibility"].squeeze(0)
        gt_visibility = record["gt_visibility"]
        pred_heatmaps = record["heatmaps"].squeeze(0)
        gt_heatmaps = record["gt_heatmaps"]
        for history_idx in range(gt_visibility.shape[0]):
            positive_views = torch.nonzero(gt_visibility[history_idx] > 0.5).flatten()
            if positive_views.numel() == 0:
                continue
            view_total += 1
            selected_view = int(vis_logits[history_idx].argmax().item())
            if bool((positive_views == selected_view).any()):
                view_correct += 1
            for view_idx in positive_views.tolist():
                pred_flat_idx = int(pred_heatmaps[history_idx, view_idx].argmax().item())
                gt_flat_idx = int(gt_heatmaps[history_idx, view_idx].argmax().item())
                width = int(gt_heatmaps.shape[-1])
                pred_y, pred_x = divmod(pred_flat_idx, width)
                gt_y, gt_x = divmod(gt_flat_idx, width)
                pixel_errors.append(math.hypot(pred_x - gt_x, pred_y - gt_y))
                u_errors.append(abs(pred_x - gt_x))

    errors = np.asarray(pixel_errors, dtype=np.float64)
    u_values = np.asarray(u_errors, dtype=np.float64)
    return {
        "visibility_auroc": auroc,
        "visibility_auprc": auprc,
        "visibility_f1": float(f1),
        "visibility_precision": float(precision),
        "visibility_recall": float(recall),
        "visible_view_accuracy": view_correct / max(view_total, 1),
        "visible_history_count": view_total,
        "visible_view_count": int(errors.size),
        "median_pixel_error": float(np.median(errors)) if errors.size else float("nan"),
        "median_u_error": float(np.median(u_values)) if u_values.size else float("nan"),
        "pck4": float((errors <= 4.0).mean()) if errors.size else float("nan"),
        "pck8": float((errors <= 8.0).mean()) if errors.size else float("nan"),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: HeatmapVLNLoss,
    dataset: VLNSlidingWindowDataset,
    indices: list[int],
    *,
    train_mode: str,
    perturbation: str,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    model.heatmap_vln.feat_extractor.detach_features = True
    records = []
    loss_values = []
    for position, sample_idx in enumerate(indices):
        sample = dataset[sample_idx]
        partner = dataset[indices[(position + 1) % len(indices)]] if len(indices) > 1 else None
        transformed = transform_sample(
            sample,
            train_mode=train_mode,
            perturbation=perturbation,
            partner=partner,
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss, record = forward_loss(model, criterion, transformed, device)
        loss_values.append(float(loss.detach().float().item()))
        records.append(record)
    metrics = compute_metrics(records)
    metrics["loss"] = float(np.mean(loss_values))
    metrics["samples"] = len(indices)
    return metrics


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cfg = load_config(args)

    train_dataset = build_dataset(cfg, "train")
    val_dataset = build_dataset(cfg, "val")
    train_indices = scene_stratified_indices(train_dataset, args.train_samples)
    val_indices = scene_stratified_indices(val_dataset, args.val_samples)

    model = build_model(cfg, verbose=True, device=args.device, enable_action_head=False)
    model.qwen2_5_vl._load_model()
    load_info = load_stage1_s2_lora(model, args.checkpoint)
    # Make the freshly-created head identical across independently launched modes.
    set_seed(args.seed + 991)
    model._ensure_heatmap_vln()
    model.heatmap_vln.feat_extractor.detach_features = True
    initial_head_hash = state_hash(heatmap_head_state_dict(model.heatmap_vln))
    if args.head_checkpoint:
        initial_head_hash, _head_payload = load_heatmap_head_checkpoint(
            model.heatmap_vln,
            args.head_checkpoint,
        )

    for param in model.parameters():
        param.requires_grad_(False)
    head_named_params = [
        (name, param)
        for name, param in model.heatmap_vln.named_parameters()
        if not name.startswith("qwen.")
    ]
    for _name, param in head_named_params:
        param.requires_grad_(True)
    model.qwen2_5_vl.model.eval()
    model.heatmap_vln.train()

    trainable = [param for _name, param in head_named_params]
    qwen_trainable = [
        name
        for name, param in model.qwen2_5_vl.model.named_parameters()
        if param.requires_grad
    ]
    if qwen_trainable:
        raise RuntimeError(
            "Head-only diagnostic unexpectedly left Qwen parameters trainable: "
            + ", ".join(qwen_trainable[:5])
        )
    optimizer = None
    if not args.head_checkpoint:
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-2)
    criterion = make_loss(cfg, device)
    train_log = []

    for step in range(1, 0 if args.head_checkpoint else args.train_steps + 1):
        sample_idx = train_indices[(step - 1) % len(train_indices)]
        sample = train_dataset[sample_idx]
        transformed = transform_sample(
            sample,
            train_mode=args.mode,
            perturbation="none",
            partner=None,
        )
        model.heatmap_vln.train()
        model.qwen2_5_vl.model.eval()
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss, _record = forward_loss(model, criterion, transformed, device)
        loss.backward()
        lora_nonzero = sum(
            param.grad is not None and float(param.grad.detach().float().norm().item()) > 0.0
            for name, param in model.named_parameters()
            if "lora_" in name
        )
        if lora_nonzero:
            raise RuntimeError(
                f"Head-only diagnostic leaked gradients into {lora_nonzero} LoRA tensors"
            )
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        loss_value = float(loss.detach().float().item())
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            LOGGER.info("mode=%s step=%d/%d loss=%.6f", args.mode, step, args.train_steps, loss_value)
            train_log.append({"step": step, "loss": loss_value})

    standard = evaluate(
        model,
        criterion,
        val_dataset,
        val_indices,
        train_mode=args.mode,
        perturbation="none",
        device=device,
    )
    evaluations = {"standard": standard}
    if args.mode == "full":
        for perturbation in (
            "zero-pose",
            "blank-images",
            "history-shuffle",
            "current-shuffle",
            "pose-conflict",
            "pose-conflict-shifted-target",
        ):
            evaluations[perturbation] = evaluate(
                model,
                criterion,
                val_dataset,
                val_indices,
                train_mode=args.mode,
                perturbation=perturbation,
                device=device,
            )

    head_path = output_dir / "head_final.pth"
    head_state = heatmap_head_state_dict(model.heatmap_vln)
    torch.save(
        {
            "mode": args.mode,
            "head_state_dict": head_state,
            "initial_head_hash": initial_head_hash,
            "config": cfg,
        },
        head_path,
    )
    report = {
        "mode": args.mode,
        "seed": args.seed,
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "load": load_info,
        "initial_head_hash": initial_head_hash,
        "train_steps": args.train_steps,
        "evaluation_only": bool(args.head_checkpoint),
        "loaded_head_checkpoint": args.head_checkpoint,
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "num_history": args.num_history,
        "lambda_coord": 0.0,
        "trainable_head_tensors": len(trainable),
        "trainable_head_numel": int(sum(param.numel() for param in trainable)),
        "trainable_qwen_tensors": 0,
        "train_log": train_log,
        "evaluations": evaluations,
        "head_checkpoint": str(head_path),
    }
    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())

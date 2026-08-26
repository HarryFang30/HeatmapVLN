#!/usr/bin/env python3
"""Audit the real heatmap-loss gradient path into Qwen LoRA parameters.

The audit deliberately uses one real panoramic sample and reports both raw
parameter movement and movement of the effective LoRA update ``(alpha/r)BA``.
It also runs a detached-feature negative control in the same process.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
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


LOGGER = logging.getLogger("heatmap_lora_gradient_audit")
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
LORA_RE = re.compile(r"^(.*)\.lora_([AB])\.[^.]+\.weight$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--gradient-checkpointing",
        choices=("on", "off"),
        default="on",
    )
    parser.add_argument("--num-history", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--overfit-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-search-samples", type=int, default=32)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_audit_config(args: argparse.Namespace) -> dict[str, Any]:
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    cfg["data"]["root"] = str(Path(args.data_root).resolve())
    cfg["data"]["dataset_type"] = "sliding_window"
    sw_cfg = cfg["data"].setdefault("sliding_window", {})
    sw_cfg.update(
        min_history=max(2, int(sw_cfg.get("min_history", 5))),
        num_history_sample=args.num_history,
        load_depth=True,
        load_single_view_history_frames=True,
        cache_poses=True,
        sample_stride=2,
        clip_level_sampling=True,
        samples_per_clip=8,
        defer_heatmap_to_gpu=False,
    )

    model_cfg = cfg["model"]
    model_cfg["device"] = args.device
    llm_cfg = model_cfg["llm"]
    llm_cfg["model_path"] = "/mnt/afs/liwenhao/agent/370910109/InternNav-Model"
    llm_cfg["attn_implementation"] = "sdpa"
    llm_cfg["gradient_checkpointing"] = args.gradient_checkpointing == "on"
    llm_cfg["enable_compile"] = False
    llm_cfg["lora_dropout"] = 0.0

    heatmap_cfg = model_cfg.setdefault("heatmap", {})
    heatmap_cfg["enable"] = True
    heatmap_cfg["heatmap_trains_backbone"] = True
    heatmap_cfg.setdefault("llm_layer_indices", [6, 13, 20])
    heatmap_cfg.setdefault("vit_layer_indices", [7, 15, 23, 31])
    heatmap_cfg.setdefault("trajectory", {}).setdefault("enable", True)
    model_cfg.setdefault("action_head", {})["enable"] = False
    return cfg


def build_dataset(cfg: dict[str, Any], args: argparse.Namespace) -> VLNSlidingWindowDataset:
    sw_cfg = cfg["data"]["sliding_window"]
    return VLNSlidingWindowDataset(
        root=cfg["data"]["root"],
        split="all",
        min_history=sw_cfg["min_history"],
        num_history_sample=sw_cfg["num_history_sample"],
        image_size=tuple(cfg["data"]["image_size"]),
        hm_size=tuple(cfg["data"]["init_hm_size"]),
        load_depth=True,
        cache_poses=True,
        sample_stride=2,
        enable_augmentation=False,
        samples_per_clip=8,
        clip_level_sampling=True,
        load_single_view_history_frames=True,
        max_clips=2,
    )


def choose_real_sample(dataset: VLNSlidingWindowDataset, limit: int) -> tuple[int, dict[str, Any]]:
    fallback = None
    for index in range(min(len(dataset), limit)):
        sample = dataset[index]
        fallback = (index, sample)
        positives = int(sample["gt_visibility"].sum().item())
        negatives = int(sample["gt_visibility"].numel() - positives)
        if positives > 0 and negatives > 0:
            return index, sample
    if fallback is None:
        raise RuntimeError("Audit dataset produced no samples")
    return fallback


def make_loss(cfg: dict[str, Any], device: torch.device) -> HeatmapVLNLoss:
    loss_cfg = dict(cfg.get("loss", {}).get("heatmap_vln", {}))
    loss_cfg["heatmap_size"] = tuple(cfg["data"]["init_hm_size"])
    # The current coordinate branch is intentionally excluded from this graph
    # audit; peak CE, visibility BCE, and negative BCE are sufficient.
    loss_cfg["lambda_coord"] = 0.0
    return HeatmapVLNLoss(**loss_cfg).to(device)


def load_checkpoint(model: torch.nn.Module, checkpoint: str) -> dict[str, int]:
    ckpt = safe_torch_load(checkpoint)
    state = ckpt.get("trainable_state_dict", {})
    if not state:
        raise RuntimeError(f"Checkpoint has no trainable_state_dict: {checkpoint}")
    matched_lora = assert_complete_lora_checkpoint_match(
        model,
        state,
        checkpoint_path=checkpoint,
    )
    missing, unexpected, loaded = _load_normalized_state_dict(model, state)
    return {
        "checkpoint_tensors": len(state),
        "matched_lora_tensors": matched_lora,
        "loaded_tensors": loaded,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }


def lora_named_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    params = {
        name: param
        for name, param in model.named_parameters()
        if "lora_" in name
    }
    if not params:
        raise RuntimeError("Model has no LoRA parameters")
    for param in params.values():
        param.requires_grad_(True)
    return params


def snapshot(params: dict[str, torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    return {name: param.detach().float().cpu().clone() for name, param in params.items()}


def effective_delta_norms(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    *,
    alpha: float,
    rank: int,
) -> dict[str, float]:
    before_pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    after_pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for name, value in before.items():
        match = LORA_RE.match(name)
        if match:
            before_pairs[match.group(1)][match.group(2)] = value
    for name, value in after.items():
        match = LORA_RE.match(name)
        if match:
            after_pairs[match.group(1)][match.group(2)] = value

    # Avoid materialising every dense BA matrix.  For
    # D = B1(A1-A0) + (B1-B0)A0 = U V, use
    # ||D||_F^2 = trace((U^T U)(V V^T)); only 2r x 2r Gram matrices
    # are materialised even for 3584 x 3584 projections.
    scale = float(alpha) / float(rank)
    result: dict[str, float] = {}
    for prefix, pair0 in before_pairs.items():
        pair1 = after_pairs.get(prefix, {})
        if not {"A", "B"}.issubset(pair0) or not {"A", "B"}.issubset(pair1):
            continue
        a0, b0 = pair0["A"].float(), pair0["B"].float()
        a1, b1 = pair1["A"].float(), pair1["B"].float()
        u = torch.cat([b1, b1 - b0], dim=1)
        v = torch.cat([a1 - a0, a0], dim=0)
        gram_u = u.transpose(0, 1).matmul(u)
        gram_v = v.matmul(v.transpose(0, 1))
        norm_sq = (gram_u * gram_v.transpose(0, 1)).sum().clamp(min=0.0)
        result[prefix] = scale * float(norm_sq.sqrt().item())
    return result


def prepare_batch(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    history = sample["history_frames"].unsqueeze(0).to(device)
    current = sample["current_frame"].unsqueeze(0).to(device)
    video = torch.cat([history, current.unsqueeze(1)], dim=1)
    return {
        "video_frames": video,
        "current_observation": current,
        "current_views": sample["current_views"].unsqueeze(0),
        "history_panoramas": sample["history_panoramas"].unsqueeze(0),
        "history_rel_poses": sample["history_rel_poses"].unsqueeze(0).to(device),
        "gt_visibility": sample["gt_visibility"].unsqueeze(0).to(device),
        "gt_heatmaps": sample["heatmap"].unsqueeze(0).to(device),
        "history_mask": torch.ones(
            1,
            sample["heatmap"].shape[0],
            dtype=torch.bool,
            device=device,
        ),
    }


def forward_loss(
    model: torch.nn.Module,
    criterion: HeatmapVLNLoss,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    device_type = batch["video_frames"].device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        output = model(
            video_frames=batch["video_frames"],
            current_observation=batch["current_observation"],
            current_views=batch["current_views"],
            history_panoramas=batch["history_panoramas"],
            history_rel_poses=batch["history_rel_poses"],
            return_heatmaps=True,
            return_actions=False,
            return_lm_loss=False,
        )
        losses = criterion(
            output["visibility"],
            output["heatmaps"],
            gt_vis=batch["gt_visibility"],
            gt_heatmaps=batch["gt_heatmaps"],
            history_mask=batch["history_mask"],
        )
    scalars = {
        key: float(value.detach().float().item())
        for key, value in losses.items()
        if torch.is_tensor(value) and value.numel() == 1
    }
    return losses["total"], scalars


def collect_rows(
    params: dict[str, torch.nn.Parameter],
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    effective_deltas: dict[str, float],
    max_hook_layer: int,
) -> list[dict[str, Any]]:
    rows = []
    for name, param in sorted(params.items()):
        layer_match = LAYER_RE.search(name)
        layer = int(layer_match.group(1)) if layer_match else -1
        grad = param.grad
        if grad is None:
            grad_status = "none"
            grad_norm = 0.0
        else:
            grad_norm = float(grad.detach().float().norm().item())
            grad_status = "nonzero" if grad_norm > 0.0 else "zero"
        param_delta = float((after[name] - before[name]).norm().item())
        pair_match = LORA_RE.match(name)
        prefix = pair_match.group(1) if pair_match else ""
        effective_delta = effective_deltas.get(prefix, 0.0)
        rows.append(
            {
                "name": name,
                "layer": layer,
                "reachable_expected": 0 <= layer <= max_hook_layer,
                "grad_status": grad_status,
                "grad_norm": grad_norm,
                "param_delta": param_delta,
                "effective_deltaW_delta": effective_delta,
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]], max_hook_layer: int) -> dict[str, Any]:
    layer_stats: dict[int, dict[str, Any]] = {}
    for layer in sorted({int(row["layer"]) for row in rows if int(row["layer"]) >= 0}):
        layer_rows = [row for row in rows if int(row["layer"]) == layer]
        layer_stats[layer] = {
            "reachable_expected": layer <= max_hook_layer,
            "tensor_count": len(layer_rows),
            "nonzero_grad_tensors": sum(row["grad_status"] == "nonzero" for row in layer_rows),
            "grad_norm": float(sum(row["grad_norm"] ** 2 for row in layer_rows) ** 0.5),
            "param_delta": float(sum(row["param_delta"] ** 2 for row in layer_rows) ** 0.5),
            "effective_deltaW_delta": float(
                max((row["effective_deltaW_delta"] for row in layer_rows), default=0.0)
            ),
        }
    reachable = [stats for layer, stats in layer_stats.items() if layer <= max_hook_layer]
    unreachable = [stats for layer, stats in layer_stats.items() if layer > max_hook_layer]
    passed = bool(reachable) and all(
        stats["nonzero_grad_tensors"] > 0 and stats["param_delta"] > 0.0
        for stats in reachable
    ) and all(stats["nonzero_grad_tensors"] == 0 for stats in unreachable)
    return {
        "max_hook_layer": max_hook_layer,
        "layers": {str(layer): stats for layer, stats in layer_stats.items()},
        "passed": passed,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_audit_config(args)
    device = torch.device(args.device)
    dataset = build_dataset(cfg, args)
    sample_index, sample = choose_real_sample(dataset, args.max_search_samples)
    LOGGER.info(
        "Using sample %d with %d/%d visible views",
        sample_index,
        int(sample["gt_visibility"].sum().item()),
        sample["gt_visibility"].numel(),
    )

    model = build_model(cfg, verbose=True, device=args.device, enable_action_head=False)
    model.qwen2_5_vl._load_model()
    load_info = load_checkpoint(model, args.checkpoint)
    # Constructing HeatmapVLN registers another reference to the same Qwen
    # module, which intentionally duplicates its keys in ``state_dict``.
    # Verify/load the unique Stage1-S2 LoRA contract before that registration.
    model._ensure_heatmap_vln()
    model.train()
    model.heatmap_vln.feat_extractor.detach_features = False

    lora_params = lora_named_parameters(model)
    for name, param in model.named_parameters():
        if "lora_" not in name and not name.startswith("heatmap_vln."):
            param.requires_grad_(False)

    criterion = make_loss(cfg, device)
    batch = prepare_batch(sample, device)
    llm_cfg = cfg["model"]["llm"]
    max_hook_layer = max(cfg["model"]["heatmap"]["llm_layer_indices"])
    alpha = float(llm_cfg["lora_alpha"])
    rank = int(llm_cfg["lora_rank"])

    positive_before = snapshot(lora_params)
    optimizer = torch.optim.AdamW(lora_params.values(), lr=args.learning_rate, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    positive_loss, positive_losses = forward_loss(model, criterion, batch)
    positive_loss.backward()
    optimizer.step()
    positive_after = snapshot(lora_params)
    effective_deltas = effective_delta_norms(
        positive_before,
        positive_after,
        alpha=alpha,
        rank=rank,
    )
    positive_rows = collect_rows(
        lora_params,
        positive_before,
        positive_after,
        effective_deltas,
        max_hook_layer,
    )
    positive_summary = summarize_rows(positive_rows, max_hook_layer)

    # Negative control: the same real sample and loss, but all hooked features
    # are detached.  Heatmap-head gradients remain valid while LoRA must stay 0.
    model.heatmap_vln.feat_extractor.detach_features = True
    negative_before = snapshot(lora_params)
    optimizer.zero_grad(set_to_none=True)
    negative_loss, negative_losses = forward_loss(model, criterion, batch)
    negative_loss.backward()
    negative_rows_pre_step = []
    for name, param in lora_params.items():
        grad = param.grad
        grad_norm = 0.0 if grad is None else float(grad.detach().float().norm().item())
        negative_rows_pre_step.append((name, grad_norm))
    optimizer.step()
    negative_after = snapshot(lora_params)
    negative_passed = all(grad_norm == 0.0 for _, grad_norm in negative_rows_pre_step) and all(
        torch.equal(negative_before[name], negative_after[name])
        for name in lora_params
    )

    overfit_log: list[dict[str, float | int]] = []
    if args.overfit_steps > 0:
        with torch.no_grad():
            for name, param in lora_params.items():
                param.copy_(positive_before[name].to(device=param.device, dtype=param.dtype))
        model.zero_grad(set_to_none=True)
        model.heatmap_vln.feat_extractor.detach_features = False
        head_params = [
            param
            for name, param in model.heatmap_vln.named_parameters()
            if not name.startswith("qwen.")
        ]
        for param in head_params:
            param.requires_grad_(True)
        overfit_optimizer = torch.optim.AdamW(
            [*lora_params.values(), *head_params],
            lr=args.learning_rate,
            weight_decay=0.0,
        )
        for step in range(1, args.overfit_steps + 1):
            overfit_optimizer.zero_grad(set_to_none=True)
            step_loss, step_losses = forward_loss(model, criterion, batch)
            step_loss.backward()
            torch.nn.utils.clip_grad_norm_([*lora_params.values(), *head_params], 1.0)
            overfit_optimizer.step()
            overfit_log.append({
                "step": step,
                "total": float(step_loss.detach().float().item()),
                "vis_loss": float(step_losses["vis_loss"]),
                "peak_loss": float(step_losses["peak_loss"]),
                "neg_loss": float(step_losses["neg_loss"]),
            })

    mode_name = f"gc_{args.gradient_checkpointing}"
    csv_path = output_dir / f"gradient_audit_{mode_name}.csv"
    json_path = output_dir / f"gradient_audit_{mode_name}.json"
    write_csv(csv_path, positive_rows)
    report = {
        "mode": mode_name,
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "sample_index": sample_index,
        "num_history": int(sample["heatmap"].shape[0]),
        "visible_views": int(sample["gt_visibility"].sum().item()),
        "load": load_info,
        "positive": {
            "losses": positive_losses,
            "summary": positive_summary,
        },
        "detach_negative": {
            "losses": negative_losses,
            "nonzero_grad_tensors": sum(value > 0.0 for _, value in negative_rows_pre_step),
            "passed": negative_passed,
        },
        "overfit": {
            "steps": args.overfit_steps,
            "log": overfit_log,
            "start_loss": overfit_log[0]["total"] if overfit_log else None,
            "end_loss": overfit_log[-1]["total"] if overfit_log else None,
            "decreased": (
                overfit_log[-1]["total"] < overfit_log[0]["total"]
                if len(overfit_log) >= 2
                else None
            ),
        },
        "passed": bool(
            positive_summary["passed"]
            and negative_passed
            and (
                args.overfit_steps < 2
                or overfit_log[-1]["total"] < overfit_log[0]["total"]
            )
        ),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())

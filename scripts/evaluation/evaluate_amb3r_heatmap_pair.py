#!/usr/bin/env python3
"""Paired frozen-head audit: GT pose versus AMB3R-VO pose on identical RGB."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _load_heatmap_without_hash_lock(model, checkpoint_path: str | Path) -> dict:
    """Reuse the strict tensor contract while intentionally not pinning a digest."""

    from scripts.training import frozen_heatmap_checkpoint as loader

    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    payload = loader._load_weights_only(path)
    source = loader._extract_heatmap_state(payload)
    heatmap = loader._resolve_heatmap_module(model)
    target = loader._target_parameters(heatmap)
    source_by_local = loader._validate_exact_coverage(source, target)
    converted = loader._prepare_copies(source_by_local, target)
    loader._copy_and_verify(converted, target)
    heatmap.requires_grad_(False)
    heatmap.eval()
    return {
        "checkpoint_path": str(path),
        "checkpoint_hash_enforced": False,
        "tensor_count": len(target),
        "exact_parameter_coverage": True,
        "weights_only_deserialization": True,
    }


def _visibility_counts(output: dict, gt_visibility: torch.Tensor) -> dict[str, int]:
    predicted = torch.sigmoid(output["visibility"].detach().float()) > 0.5
    target = gt_visibility.detach().to(predicted.device) > 0.5
    return {
        "tp": int((predicted & target).sum().item()),
        "tn": int((~predicted & ~target).sum().item()),
        "fp": int((predicted & ~target).sum().item()),
        "fn": int((~predicted & target).sum().item()),
    }


def _merge_counts(total: dict[str, int], row: dict[str, int]) -> None:
    for key, value in row.items():
        total[key] += int(value)


def _visibility_metrics(counts: dict[str, int]) -> dict[str, float | int]:
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    total = tp + tn + fp + fn
    return {
        **counts,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def _agreement_row(gt_output: dict, vo_output: dict) -> dict[str, float | int]:
    gt_logits = gt_output["heatmap_logits"].detach().float().cpu()
    vo_logits = vo_output["heatmap_logits"].detach().float().cpu()
    width = int(gt_logits.shape[-1])
    gt_peak = gt_logits.reshape(*gt_logits.shape[:-2], -1).argmax(dim=-1)
    vo_peak = vo_logits.reshape(*vo_logits.shape[:-2], -1).argmax(dim=-1)
    gt_y, gt_x = torch.div(gt_peak, width, rounding_mode="floor"), gt_peak % width
    vo_y, vo_x = torch.div(vo_peak, width, rounding_mode="floor"), vo_peak % width
    distance = torch.sqrt(((gt_x - vo_x).float().square() + (gt_y - vo_y).float().square()))
    gt_vis = torch.sigmoid(gt_output["visibility"].detach().float().cpu()) > 0.5
    vo_vis = torch.sigmoid(vo_output["visibility"].detach().float().cpu()) > 0.5
    return {
        "map_count": int(distance.numel()),
        "peak_shift_sum_px": float(distance.sum().item()),
        "peak_shift_le4_count": int((distance <= 4.0).sum().item()),
        "peak_shift_le8_count": int((distance <= 8.0).sum().item()),
        "visibility_field_count": int(gt_vis.numel()),
        "visibility_agree_count": int((gt_vis == vo_vis).sum().item()),
        "logit_abs_sum": float((gt_logits - vo_logits).abs().sum().item()),
        "logit_count": int(gt_logits.numel()),
    }


def _render_comparison(
    output_path: Path,
    *,
    frame_index: int,
    target: torch.Tensor,
    gt_pose_prediction: torch.Tensor,
    vo_pose_prediction: torch.Tensor,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = [
        target.detach().float().amax(dim=0).cpu().numpy(),
        gt_pose_prediction.detach().float().amax(dim=0).cpu().numpy(),
        vo_pose_prediction.detach().float().amax(dim=0).cpu().numpy(),
    ]
    row_names = ("Target", "Frozen head + GT pose", "Frozen head + AMB3R pose")
    view_names = ("Front", "Right", "Back", "Left")
    figure, axes = plt.subplots(3, 4, figsize=(10, 7), constrained_layout=True)
    for row, (name, array) in enumerate(zip(row_names, arrays)):
        vmax = max(float(np.max(array)), 1e-8)
        for view in range(4):
            axes[row, view].imshow(array[view], cmap="magma", vmin=0.0, vmax=vmax)
            axes[row, view].set_xticks([])
            axes[row, view].set_yticks([])
            if row == 0:
                axes[row, view].set_title(view_names[view], fontsize=9)
            if view == 0:
                axes[row, view].set_ylabel(name, fontsize=8)
    figure.suptitle(f"Frame {frame_index}: same RGB/head, pose provider is the only treatment")
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--heatmap-checkpoint", required=True)
    parser.add_argument("--pose-cache", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--visualize-samples", type=int, default=4)
    parser.add_argument(
        "--allow-retrospective-noncausal",
        action="store_true",
        help=(
            "Evaluate earlier current frames from a final optimized trajectory. "
            "This can use future frames and is disabled by default."
        ),
    )
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve(strict=True)
    sys.path.insert(0, str(repo))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clip = Path(args.clip).expanduser().resolve(strict=True)

    from scripts.training.model_builder import build_model
    from scripts.training.utils import load_config, make_autocast_context
    from scripts.training.validate import _HeatmapJointMetricAccumulator
    from src.data.factory import build_sliding_window_dataset
    from src.data.single_view_heatmap_collator import SingleViewHeatmapCollator
    from src.vo.amb3r_pose import history_rel_poses_from_amb3r

    with np.load(Path(args.pose_cache).expanduser().resolve(strict=True)) as cache:
        frame_ids = np.asarray(cache["frame_ids"], dtype=np.int64)
        vo_poses = np.asarray(cache["poses_c2w_opencv"], dtype=np.float32)
        cached_gt = np.asarray(cache["gt_poses_c2w_habitat"], dtype=np.float32)
    if not np.array_equal(frame_ids, np.arange(len(frame_ids))):
        raise ValueError("Pose cache frame_ids are not contiguous from zero")

    # Build a one-clip filesystem view so dataset semantics are reused without
    # indexing all 10,819 expert trajectories.
    view_root = output_dir / "_dataset_view"
    scene_view = view_root / clip.parent.name
    scene_view.mkdir(parents=True, exist_ok=True)
    clip_view = scene_view / clip.name
    if clip_view.is_symlink():
        if clip_view.resolve() != clip:
            raise RuntimeError(f"Existing dataset-view symlink targets {clip_view.resolve()}")
    elif clip_view.exists():
        raise RuntimeError(f"Dataset-view path already exists and is not a symlink: {clip_view}")
    else:
        os.symlink(clip, clip_view, target_is_directory=True)

    config = load_config(args.config)
    dataset = build_sliding_window_dataset(
        config,
        split="all",
        root=str(view_root),
        clip_level_sampling=False,
        sample_stride=1,
        enable_augmentation=False,
        defer_heatmap_to_gpu=False,
    )
    if len(dataset.clips) != 1:
        raise RuntimeError(f"Expected exactly one clip, found {len(dataset.clips)}")
    dataset_gt = np.stack(dataset._load_poses(0), axis=0)
    if (
        dataset_gt.shape[0] < cached_gt.shape[0]
        or dataset_gt.shape[1:] != cached_gt.shape[1:]
        or not np.allclose(dataset_gt[: len(cached_gt)], cached_gt, atol=1e-5)
    ):
        raise RuntimeError("Pose cache does not belong to the selected clip")

    indexed = [
        (dataset_index, int(frame_index))
        for dataset_index, (clip_index, frame_index) in enumerate(dataset.sample_index)
        if int(clip_index) == 0 and int(frame_index) < len(frame_ids)
    ]
    if not args.allow_retrospective_noncausal:
        final_frame = int(frame_ids[-1])
        indexed = [row for row in indexed if row[1] == final_frame]
    indexed = indexed[int(args.sample_offset) :]
    if args.max_samples > 0:
        indexed = indexed[: int(args.max_samples)]
    if not indexed:
        raise RuntimeError("No dataset samples overlap the AMB3R pose cache")

    device = torch.device(args.device)
    model = build_model(config, device=str(device), verbose=False, enable_action_head=False)
    model.qwen2_5_vl._load_model()
    model._ensure_heatmap_vln()
    checkpoint_audit = _load_heatmap_without_hash_lock(model, args.heatmap_checkpoint)
    model = model.to(device).eval()
    collator = SingleViewHeatmapCollator(model.qwen2_5_vl.processor)
    amp_type = config.get("optim", {}).get("amp", "bf16")
    heatmap_size = tuple(config["model"]["heatmap"]["heatmap_size"])
    accumulators = {
        "gt_pose": _HeatmapJointMetricAccumulator(heatmap_size=heatmap_size, device=device),
        "amb3r_pose": _HeatmapJointMetricAccumulator(heatmap_size=heatmap_size, device=device),
    }
    visibility = {
        key: {name: 0 for name in ("tp", "tn", "fp", "fn")}
        for key in accumulators
    }
    agreement = {
        name: 0.0
        for name in (
            "map_count",
            "peak_shift_sum_px",
            "peak_shift_le4_count",
            "peak_shift_le8_count",
            "visibility_field_count",
            "visibility_agree_count",
            "logit_abs_sum",
            "logit_count",
        )
    }
    frame_records = []

    for ordinal, (dataset_index, current_frame) in enumerate(indexed):
        failures_before = int(getattr(dataset, "_sample_failure_count", 0))
        sample = dataset[dataset_index]
        if int(getattr(dataset, "_sample_failure_count", 0)) != failures_before:
            raise RuntimeError(f"Dataset returned a dummy sample at index {dataset_index}")
        batch = collator([sample])
        history_indices = dataset._sample_history_indices(
            0,
            current_frame,
            dataset.num_history_sample,
        )
        vo_rel = history_rel_poses_from_amb3r(
            vo_poses,
            history_indices,
            current_frame,
            translation_scale=args.translation_scale,
        )
        gt_rel = batch["history_rel_poses"]
        if vo_rel.shape != tuple(gt_rel.shape[1:]):
            raise RuntimeError(f"VO/GT history shape mismatch: {vo_rel.shape} vs {tuple(gt_rel.shape)}")

        single_view_inputs = {
            "pixel_values": batch["pixel_values"].to(device, non_blocking=True),
            "image_grid_thw": batch["image_grid_thw"].to(device, non_blocking=True),
        }
        shared = dict(
            video_frames=None,
            instruction_text=[sample.get("text", "")],
            current_observation=None,
            single_view_inputs=single_view_inputs,
            single_view_num_histories=batch["num_histories"],
            return_heatmaps=True,
            return_heatmap_logits=True,
            return_actions=False,
        )
        with torch.inference_mode(), make_autocast_context(device, amp_type):
            outputs = {
                "gt_pose": model(
                    **shared,
                    history_rel_poses=gt_rel.to(device, non_blocking=True),
                ),
                "amb3r_pose": model(
                    **shared,
                    history_rel_poses=torch.from_numpy(vo_rel).unsqueeze(0).to(device),
                ),
            }
        gt_heatmaps = batch["heatmap"].to(device)
        gt_visibility = batch.get("gt_visibility")
        if gt_visibility is None:
            gt_visibility = (gt_heatmaps.amax(dim=(-2, -1)) > 0).float()
        else:
            gt_visibility = gt_visibility.to(device)
        history_mask = batch.get("history_mask")
        if history_mask is not None:
            history_mask = history_mask.to(device)
        for arm, output in outputs.items():
            accumulators[arm].update(
                pred_visibility_logits=output["visibility"],
                pred_heatmaps=output.get("heatmap_logits", output["heatmaps"]),
                gt_visibility=gt_visibility,
                gt_heatmaps=gt_heatmaps,
                history_mask=history_mask,
            )
            _merge_counts(visibility[arm], _visibility_counts(output, gt_visibility))
        row = _agreement_row(outputs["gt_pose"], outputs["amb3r_pose"])
        for key, value in row.items():
            agreement[key] += float(value)

        if ordinal < args.visualize_samples:
            _render_comparison(
                output_dir / f"frame_{current_frame:04d}_gt_vs_amb3r.png",
                frame_index=current_frame,
                target=gt_heatmaps[0].cpu(),
                gt_pose_prediction=outputs["gt_pose"]["heatmaps_gated"][0].cpu(),
                vo_pose_prediction=outputs["amb3r_pose"]["heatmaps_gated"][0].cpu(),
            )
        frame_records.append(
            {
                "current_frame": current_frame,
                "history_indices": history_indices.tolist(),
                "gt_history_rel_poses": gt_rel[0].tolist(),
                "amb3r_history_rel_poses": vo_rel.tolist(),
            }
        )
        torch.cuda.empty_cache()

    arm_metrics = {}
    for arm, accumulator in accumulators.items():
        metrics = accumulator.compute()
        metrics["visibility"] = _visibility_metrics(visibility[arm])
        arm_metrics[arm] = metrics
    map_count = max(agreement["map_count"], 1.0)
    vis_count = max(agreement["visibility_field_count"], 1.0)
    logit_count = max(agreement["logit_count"], 1.0)
    agreement_metrics = {
        "heatmap_map_count": int(agreement["map_count"]),
        "mean_peak_shift_px": agreement["peak_shift_sum_px"] / map_count,
        "peak_shift_le4_rate": agreement["peak_shift_le4_count"] / map_count,
        "peak_shift_le8_rate": agreement["peak_shift_le8_count"] / map_count,
        "visibility_binary_agreement": agreement["visibility_agree_count"] / vis_count,
        "mean_abs_logit_difference": agreement["logit_abs_sum"] / logit_count,
    }
    report = {
        "schema": "heatmapvln-amb3r-paired-frozen-head-audit-v1",
        "causal_treatment": "history_rel_poses only; RGB, frozen head and targets are identical",
        "clip": str(clip),
        "pose_cache": str(Path(args.pose_cache).expanduser().resolve()),
        "samples": len(indexed),
        "causal_evaluation": not bool(args.allow_retrospective_noncausal),
        "evaluated_current_frames": [int(row[1]) for row in indexed],
        "retrospective_future_pose_updates_allowed": bool(
            args.allow_retrospective_noncausal
        ),
        "translation_scale": float(args.translation_scale),
        "per_episode_gt_scale_used": False,
        "checkpoint": checkpoint_audit,
        "metrics": arm_metrics,
        "agreement": agreement_metrics,
        "delta_amb3r_minus_gt": {
            key: arm_metrics["amb3r_pose"][key] - arm_metrics["gt_pose"][key]
            for key in (
                "val_heatmap_joint_pck4",
                "val_heatmap_joint_pck8",
                "val_heatmap_macro_joint_pck8",
                "val_heatmap_view5_accuracy",
            )
        },
        "frames": frame_records,
    }
    report_path = output_dir / "paired_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

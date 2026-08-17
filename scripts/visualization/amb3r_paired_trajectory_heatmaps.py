#!/usr/bin/env python3
"""Render causal GT-pose versus AMB3R-pose heatmaps as trajectory strips.

The heatmap model receives exactly the same front-only RGB in both arms.  The
right/back/left images are loaded only after inference for display.  AMB3R
relative poses come from a cache whose rows were frozen online before later
frames were ingested, so early columns cannot use future RGB/map updates.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


VIEW_NAMES = ("front", "right", "back", "left")


def _parse_window(value: str) -> tuple[str, int, int]:
    try:
        name, start, end = value.rsplit(":", 2)
        start_i, end_i = int(start), int(end)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "window must be NAME:START:END with an inclusive frame range"
        ) from error
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise argparse.ArgumentTypeError(f"unsafe window name: {name!r}")
    if start_i < 0 or end_i < start_i:
        raise argparse.ArgumentTypeError(f"invalid inclusive range [{start_i},{end_i}]")
    return name, start_i, end_i


def _strict_cache(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        current = np.asarray(payload["current_frame_ids"], dtype=np.int64)
        histories = np.asarray(payload["history_frame_ids"], dtype=np.int64)
        relative = np.asarray(payload["history_rel_poses"], dtype=np.float32)
        phases = np.asarray(payload["provider_phases"]).astype(str)
        revisions = np.asarray(payload["trajectory_revisions"], dtype=np.int64)
        last_mapped = np.asarray(payload["last_mapped_frame_ids"], dtype=np.int64)
    count = len(current)
    if count == 0:
        raise ValueError("Causal cache must contain at least one query row")
    if current.shape != (count,) or histories.ndim != 2 or histories.shape[0] != count:
        raise ValueError("Malformed causal cache frame/history arrays")
    if relative.shape != (*histories.shape, 4):
        raise ValueError(f"Malformed causal pose array: {relative.shape}")
    if any(array.shape != (count,) for array in (phases, revisions, last_mapped)):
        raise ValueError("Malformed causal cache provenance arrays")
    if len(set(current.tolist())) != count or np.any(np.diff(current) <= 0):
        raise ValueError("Causal cache current frame IDs must be unique/increasing")
    if np.any(histories < 0) or np.any(histories >= current[:, None]):
        raise ValueError("Every cached history frame must precede its current frame")
    if histories.shape[1] > 1 and np.any(np.diff(histories, axis=1) < 0):
        raise ValueError("Cached history frame IDs must be non-decreasing")
    if np.any(np.diff(revisions) < 0):
        raise ValueError("Trajectory revisions must be non-decreasing")
    if np.any(last_mapped > current):
        raise ValueError("last_mapped_frame_id cannot exceed current_frame_id")
    if any(not value for value in phases.tolist()):
        raise ValueError("Provider phase must be non-empty")
    if not np.isfinite(relative).all():
        raise ValueError("Causal cache contains non-finite poses")
    rows = {
        int(frame): {
            "history_frame_ids": histories[index],
            "history_rel_poses": relative[index],
            "provider_phase": str(phases[index]),
            "trajectory_revision": int(revisions[index]),
            "last_mapped_frame_id": int(last_mapped[index]),
        }
        for index, frame in enumerate(current)
    }
    manifest_path = path.with_suffix(path.suffix + ".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "heatmapvln-amb3r-causal-query-cache-v1":
        raise ValueError(f"Unsupported causal cache schema: {manifest.get('schema')!r}")
    if manifest.get("causal") is not True:
        raise ValueError("Cache manifest does not assert causal=true")
    if manifest.get("per_episode_gt_scale_used") is not False:
        raise ValueError("Refusing a cache with per-episode GT scale")
    if manifest.get("query_every_frame_from_map_initialization") is not True:
        raise ValueError("Cache does not match online query-every-frame cadence")
    if int(manifest.get("query_count", -1)) != count:
        raise ValueError("Cache manifest query_count disagrees with arrays")
    if int(manifest.get("num_history", -1)) != histories.shape[1]:
        raise ValueError("Cache manifest num_history disagrees with arrays")
    return rows, manifest


def _one_clip_dataset_view(output_dir: Path, clip: Path) -> Path:
    root = output_dir / "_dataset_view"
    scene = root / clip.parent.name
    scene.mkdir(parents=True, exist_ok=True)
    link = scene / clip.name
    if link.is_symlink():
        if link.resolve() != clip:
            raise RuntimeError(f"Dataset-view symlink points to {link.resolve()}, expected {clip}")
    elif link.exists():
        raise RuntimeError(f"Dataset-view path exists and is not a symlink: {link}")
    else:
        os.symlink(clip, link, target_is_directory=True)
    return root


def _features_to_decoder(features, decoder_device: torch.device, history_mask: torch.Tensor):
    return type(features)(
        current_vit={
            key: value.to(device=decoder_device, dtype=torch.float32)
            for key, value in features.current_vit.items()
        },
        current_merged=features.current_merged.to(device=decoder_device, dtype=torch.float32),
        history_vit={
            key: value.to(device=decoder_device, dtype=torch.float32)
            for key, value in features.history_vit.items()
        },
        history_merged=features.history_merged.to(device=decoder_device, dtype=torch.float32),
        history_queries=features.history_queries.to(device=decoder_device, dtype=torch.float32),
        history_mask=history_mask.to(device=decoder_device, dtype=torch.bool),
    )


def _forward_pair_shared_visual(model, batch: dict[str, Any], vo_rel: np.ndarray) -> dict[str, dict]:
    extractor = model.single_view_heatmap_extractor
    head = model.heatmap_vln
    if extractor is None or head is None:
        raise RuntimeError("Single-view feature extractor/head is unavailable")
    visual = extractor._visual
    visual.eval()
    head.eval()
    visual_device = next(visual.parameters()).device
    decoder_device = next(head.parameters()).device
    features = extractor.extract_from_pixels(
        pixel_values=batch["pixel_values"].to(visual_device, non_blocking=True),
        image_grid_thw=batch["image_grid_thw"].to(visual_device, non_blocking=True),
        num_histories=batch["num_histories"],
    )
    explicit_mask = batch.get("history_mask", features.history_mask)
    if tuple(explicit_mask.shape) != tuple(features.history_mask.shape):
        raise ValueError("Explicit and visual history masks disagree")
    features = _features_to_decoder(features, decoder_device, explicit_mask)
    gt_rel = batch["history_rel_poses"].to(
        device=decoder_device, dtype=torch.float32, non_blocking=True
    )
    vo_tensor = torch.from_numpy(vo_rel).unsqueeze(0).to(
        device=decoder_device, dtype=torch.float32
    )
    if tuple(gt_rel.shape) != tuple(vo_tensor.shape):
        raise ValueError(f"GT/VO pose shape mismatch: {gt_rel.shape} vs {vo_tensor.shape}")
    return {
        "gt_pose": head(features, gt_rel),
        "amb3r_pose": head(features, vo_tensor),
    }


def _operational_gated(output: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    gated = output["heatmaps_gated"][0].detach().float().cpu()
    logits = output["visibility"][0].detach().float().cpu()
    hard_mask = torch.sigmoid(logits) > 0.5
    # ``heatmaps_gated`` already contains the joint none/F/R/B/L probability
    # gate.  Keep the soft operational map in the primary figure so a
    # low-confidence but correctly located side peak is not made artificially
    # black by an additional visualization-only threshold.
    return gated.amax(dim=0).numpy(), hard_mask.numpy()


def _build_pose_safe_local_strip(renderer, rows, tile: int, gap: int) -> np.ndarray:
    size = 4 * tile
    width = len(rows) * size + max(len(rows) - 1, 0) * gap
    strip = np.zeros((size, width, 3), dtype=np.float32)
    for index, row in enumerate(rows):
        rel = np.asarray(row["gt_history_rel_poses"], dtype=np.float32)
        history_xy = np.column_stack((-rel[:, 1], rel[:, 0]))
        cell = renderer._render_local_traj_bev(history_xy, out_size=size, position_label=index)
        x0 = index * (size + gap)
        strip[:, x0 : x0 + size] = cell
        if index < len(rows) - 1:
            renderer._fill_sep(strip, x0 + size, gap)
    return strip


def _shared_prediction_strips(renderer, rows, tile: int, gap: int) -> tuple[np.ndarray, np.ndarray]:
    import matplotlib.pyplot as plt

    width = len(rows) * (4 * tile) + max(len(rows) - 1, 0) * gap
    outputs = [np.zeros((tile, width, 3), dtype=np.float32) for _ in range(2)]
    cmap = plt.cm.inferno
    for index, row in enumerate(rows):
        arms = (row["gt_pose_agg"], row["amb3r_pose_agg"])
        vmax = max(float(np.max(arms[0])), float(np.max(arms[1])), 1e-8)
        x0 = index * (4 * tile + gap)
        for arm_index, heatmaps in enumerate(arms):
            for view in range(4):
                heatmap = renderer._resize(np.asarray(heatmaps[view]), tile)
                destination = outputs[arm_index][
                    :, x0 + view * tile : x0 + (view + 1) * tile
                ]
                if float(heatmap.max()) >= 1e-8:
                    destination[:] = cmap(np.clip(heatmap / vmax, 0.0, 1.0))[..., :3]
            if index < len(rows) - 1:
                renderer._fill_sep(outputs[arm_index], x0 + 4 * tile, gap)
    return outputs[0], outputs[1]


def _window_metrics(rows: list[dict[str, Any]], heatmap_size, accumulator_cls) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    coverage = np.zeros(4, dtype=np.int64)
    for arm in ("gt_pose", "amb3r_pose"):
        accumulator = accumulator_cls(heatmap_size=heatmap_size, device=torch.device("cpu"))
        counts = {name: 0 for name in ("tp", "tn", "fp", "fn")}
        for row in rows:
            target_heatmap = row["gt_heatmaps"].unsqueeze(0)
            target_visibility = row["gt_visibility"].unsqueeze(0)
            mask = row["history_mask"].unsqueeze(0)
            output = row[f"{arm}_output"]
            accumulator.update(
                pred_visibility_logits=output["visibility"].unsqueeze(0),
                pred_heatmaps=output["heatmap_logits"].unsqueeze(0),
                gt_visibility=target_visibility,
                gt_heatmaps=target_heatmap,
                history_mask=mask,
            )
            pred = torch.sigmoid(output["visibility"]) > 0.5
            target = target_visibility[0] > 0.5
            valid = mask[0, :, None].expand_as(target)
            counts["tp"] += int((pred & target & valid).sum())
            counts["tn"] += int((~pred & ~target & valid).sum())
            counts["fp"] += int((pred & ~target & valid).sum())
            counts["fn"] += int((~pred & target & valid).sum())
        arm_metrics = accumulator.compute()
        denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
        arm_metrics["visibility_f1"] = 2 * counts["tp"] / denominator if denominator else 0.0
        arm_metrics["visibility_counts"] = counts
        metrics[arm] = arm_metrics
    for row in rows:
        visible = (row["gt_visibility"] > 0.5) & (
            row["gt_heatmaps"].amax(dim=(-2, -1)) > 0
        )
        coverage += visible.sum(dim=0).numpy().astype(np.int64)
    metrics["gt_visible_direction_counts"] = {
        name: int(value) for name, value in zip(VIEW_NAMES, coverage)
    }
    return metrics


def _render_window(
    *, renderer, clip: Path, all_poses, rows: list[dict[str, Any]], output: Path,
    title: str, tile: int, gap: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame_ids = [int(row["current_frame_id"]) for row in rows]
    topdown = renderer._build_topdown_strip(
        clip,
        frame_ids,
        [{"history_indices": row["history_frame_ids"].tolist()} for row in rows],
        all_poses,
        tile,
        gap,
    )
    if topdown is None:
        map_strip = _build_pose_safe_local_strip(renderer, rows, tile, gap)
        map_label = "Local trajectory (GT, display only)"
    else:
        map_strip = topdown
        map_label = "Top-down (GT, display only)"
    rgb = renderer._build_rgb_strip(rows, tile, gap)
    target = renderer._build_heatmap_strip(rows, "gt_agg", tile, gap, global_vmax=1.0)
    gt_pose, amb3r_pose = _shared_prediction_strips(renderer, rows, tile, gap)
    strips = (
        (map_strip, map_label),
        (rgb, "RGB display only (F|R|B|L)"),
        (target, "GT target"),
        (gt_pose, "Frozen head + GT pose"),
        (amb3r_pose, "Frozen head + causal AMB3R pose"),
    )
    dpi = 120
    figure = plt.figure(
        figsize=(max(rgb.shape[1] / dpi, 10), max(sum(x.shape[0] for x, _ in strips) / dpi, 4))
    )
    grid = figure.add_gridspec(
        len(strips), 1,
        height_ratios=[strip.shape[0] for strip, _ in strips],
        hspace=0.035,
        left=0.018,
        right=0.997,
        top=0.95,
        bottom=0.025,
    )
    figure.suptitle(title, fontsize=8, y=0.995)
    cell_width = 4 * tile
    tick_positions = [i * (cell_width + gap) + cell_width // 2 for i in range(len(rows))]
    for row_index, (strip, label) in enumerate(strips):
        axis = figure.add_subplot(grid[row_index])
        axis.imshow(strip, aspect="equal", interpolation="nearest")
        axis.set_ylabel(label, fontsize=6, rotation=90, labelpad=8)
        axis.set_yticks([])
        axis.set_xticks(tick_positions)
        axis.set_xticklabels([f"t={frame}" for frame in frame_ids], fontsize=5)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--heatmap-checkpoint", required=True)
    parser.add_argument("--causal-cache", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window", action="append", type=_parse_window, default=[])
    parser.add_argument("--tile-size", type=int, default=56)
    parser.add_argument("--gap", type=int, default=3)
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve(strict=True)
    sys.path.insert(0, str(repo))
    clip = Path(args.clip).expanduser().resolve(strict=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.tile_size <= 0 or args.gap < 0:
        raise ValueError("tile-size must be positive and gap must be non-negative")
    cache_path = Path(args.causal_cache).expanduser().resolve(strict=True)
    cache_rows, cache_manifest = _strict_cache(cache_path)
    windows = args.window or [
        ("all", min(cache_rows), max(cache_rows)),
    ]
    requested_frames = sorted(
        {frame for _, start, end in windows for frame in range(start, end + 1)}
    )
    missing = [frame for frame in requested_frames if frame not in cache_rows]
    if missing:
        raise ValueError(f"Requested frames absent from causal cache: {missing}")
    if Path(cache_manifest["clip"]).resolve() != clip:
        raise ValueError("Causal cache belongs to a different clip")

    from scripts.evaluation.evaluate_amb3r_heatmap_pair import _load_heatmap_without_hash_lock
    from scripts.training.model_builder import build_model
    from scripts.training.utils import load_config, make_autocast_context
    from scripts.training.validate import _HeatmapJointMetricAccumulator
    import scripts.visualization.trajectory_heatmaps as renderer
    from src.data.factory import build_sliding_window_dataset
    from src.data.single_view_heatmap_collator import SingleViewHeatmapCollator

    config = load_config(args.config)
    heatmap_cfg = config.get("model", {}).get("heatmap", {})
    if heatmap_cfg.get("input_mode") != "internnav_single_view":
        raise RuntimeError("Config is not an internnav_single_view heatmap run")
    if config.get("model", {}).get("llm", {}).get("use_lora", False):
        raise RuntimeError("Qualitative audit refuses a LoRA backbone")

    view_root = _one_clip_dataset_view(output_dir, clip)
    dataset = build_sliding_window_dataset(
        config,
        split="all",
        root=str(view_root),
        clip_level_sampling=False,
        sample_stride=1,
        enable_augmentation=False,
        defer_heatmap_to_gpu=False,
    )
    if len(dataset.clips) != 1 or not dataset.single_view_rgb_input:
        raise RuntimeError("Expected one front-only clip dataset")
    lookup = {
        int(frame): index
        for index, (clip_index, frame) in enumerate(dataset.sample_index)
        if int(clip_index) == 0
    }
    absent = [frame for frame in requested_frames if frame not in lookup]
    if absent:
        raise RuntimeError(f"Requested dataset samples are unavailable: {absent}")

    device = torch.device(args.device)
    model = build_model(config, device=str(device), verbose=False, enable_action_head=False)
    model.qwen2_5_vl._load_model()
    model._ensure_heatmap_vln()
    checkpoint_audit = _load_heatmap_without_hash_lock(model, args.heatmap_checkpoint)
    model = model.to(device).eval()
    collator = SingleViewHeatmapCollator(model.qwen2_5_vl.processor)
    amp_type = config.get("optim", {}).get("amp", "bf16")
    all_poses = dataset._load_poses(0)

    records: dict[int, dict[str, Any]] = {}
    for ordinal, current in enumerate(requested_frames, start=1):
        failures = int(getattr(dataset, "_sample_failure_count", 0))
        sample = dataset[lookup[current]]
        if int(getattr(dataset, "_sample_failure_count", 0)) != failures:
            raise RuntimeError(f"Dataset returned a dummy sample for frame {current}")
        leaked = sorted({"current_views", "history_panoramas"}.intersection(sample))
        if leaked:
            raise RuntimeError(f"Panoramic RGB leaked into model sample: {leaked}")
        history_ids = dataset._sample_history_indices(
            0, current, dataset.num_history_sample
        ).astype(np.int64)
        cache_row = cache_rows[current]
        if not np.array_equal(history_ids, cache_row["history_frame_ids"]):
            raise RuntimeError(f"History IDs disagree at frame {current}")
        batch = collator([sample])
        with torch.inference_mode(), make_autocast_context(device, amp_type):
            outputs = _forward_pair_shared_visual(
                model, batch, cache_row["history_rel_poses"]
            )
        gt_agg, gt_mask = _operational_gated(outputs["gt_pose"])
        vo_agg, vo_mask = _operational_gated(outputs["amb3r_pose"])
        display = dataset._load_all_views(dataset.clips[0], current)
        if not torch.equal(display[0], sample["current_frame"]):
            difference = float((display[0] - sample["current_frame"]).abs().max())
            raise RuntimeError(f"Display/model front mismatch at frame {current}: {difference}")
        gt_heatmaps = sample["heatmap"].detach().float().cpu()
        gt_visibility = sample.get("gt_visibility")
        if not torch.is_tensor(gt_visibility):
            gt_visibility = (gt_heatmaps.amax(dim=(-2, -1)) > 0).float()
        else:
            gt_visibility = gt_visibility.detach().float().cpu()
        history_mask = batch.get(
            "history_mask", torch.ones(1, len(history_ids), dtype=torch.bool)
        )[0].detach().bool().cpu()
        record = {
            "current_frame_id": current,
            "history_frame_ids": history_ids,
            "gt_history_rel_poses": sample["history_rel_poses"].detach().float().cpu().numpy(),
            "amb3r_history_rel_poses": cache_row["history_rel_poses"],
            "current_views": display.cpu().numpy().transpose(0, 2, 3, 1),
            "gt_agg": gt_heatmaps.amax(dim=0).numpy(),
            "gt_pose_agg": gt_agg,
            "amb3r_pose_agg": vo_agg,
            "gt_visibility_binary": (gt_visibility > 0.5).numpy(),
            "gt_pose_visibility_binary": gt_mask,
            "amb3r_pose_visibility_binary": vo_mask,
            "gt_heatmaps": gt_heatmaps,
            "gt_visibility": gt_visibility,
            "history_mask": history_mask,
            "gt_pose_output": {
                "visibility": outputs["gt_pose"]["visibility"][0].detach().float().cpu(),
                "heatmap_logits": outputs["gt_pose"]["heatmap_logits"][0].detach().float().cpu(),
            },
            "amb3r_pose_output": {
                "visibility": outputs["amb3r_pose"]["visibility"][0].detach().float().cpu(),
                "heatmap_logits": outputs["amb3r_pose"]["heatmap_logits"][0].detach().float().cpu(),
            },
            "provider_phase": cache_row["provider_phase"],
            "trajectory_revision": cache_row["trajectory_revision"],
        }
        records[current] = record
        print(
            f"[{ordinal}/{len(requested_frames)}] frame={current} "
            f"gt_visible={record['gt_visibility_binary'].sum(axis=0).tolist()} "
            f"gt_pred={gt_mask.sum(axis=0).tolist()} vo_pred={vo_mask.sum(axis=0).tolist()}",
            flush=True,
        )

    heatmap_size = tuple(config["model"]["heatmap"]["heatmap_size"])
    window_reports = []
    image_paths = []
    clip_label = f"{clip.parent.name}/{clip.name}"
    for name, start, end in windows:
        rows = [records[frame] for frame in range(start, end + 1)]
        image_path = output_dir / f"{name}_f{start:04d}-{end:04d}.png"
        _render_window(
            renderer=renderer,
            clip=clip,
            all_poses=all_poses,
            rows=rows,
            output=image_path,
            title=(
                f"{clip_label} | selected-on-GT qualitative | causal online poses | "
                "head input is front-only; F/R/B/L RGB is display-only"
            ),
            tile=args.tile_size,
            gap=args.gap,
        )
        metrics = _window_metrics(rows, heatmap_size, _HeatmapJointMetricAccumulator)
        window_reports.append(
            {
                "name": name,
                "start_inclusive": start,
                "end_inclusive": end,
                "frames": len(rows),
                "image": str(image_path),
                "metrics": metrics,
            }
        )
        image_paths.append(str(image_path))

    report = {
        "schema": "heatmapvln-causal-amb3r-paired-trajectory-strip-v1",
        "clip": str(clip),
        "causal_cache": str(cache_path),
        "cache_semantics": cache_manifest.get("query_semantics"),
        "future_rgb_used_for_earlier_columns": False,
        "same_rgb_and_frozen_head_between_arms": True,
        "only_treatment": "history_rel_poses: habitat GT versus causal AMB3R-VO",
        "model_input_rgb": ["history_front", "current_front"],
        "display_only_rgb": list(VIEW_NAMES),
        "four_view_rgb_passed_to_head": False,
        "selection_warning": "These clips/windows were selected using GT side-view coverage; qualitative, not unbiased metrics.",
        "prediction_display": "operational heatmaps_gated (joint none/F/R/B/L soft gate), max over K histories",
        "binary_visibility_diagnostic": "sigmoid(visibility)>0.5 is reported in JSON but is not multiplied into the primary heatmap figure",
        "prediction_color_scale": "GT-pose and AMB3R-pose share one vmax per frame",
        "target_color_scale": "fixed vmax=1",
        "topdown_warning": "When available, the top-down display can show the complete GT route, including future positions; it is never model input.",
        "cache_per_frame_pose_metrics": cache_manifest.get("per_frame_pose_metrics", []),
        "checkpoint": checkpoint_audit,
        "windows": window_reports,
        "frames": [
            {
                key: (
                    value.tolist() if isinstance(value, np.ndarray) else value
                )
                for key, value in record.items()
                if key
                in {
                    "current_frame_id",
                    "history_frame_ids",
                    "gt_history_rel_poses",
                    "amb3r_history_rel_poses",
                    "gt_visibility_binary",
                    "gt_pose_visibility_binary",
                    "amb3r_pose_visibility_binary",
                    "provider_phase",
                    "trajectory_revision",
                }
            }
            for record in records.values()
        ],
    }
    report_path = output_dir / "trajectory_strip_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "images": image_paths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

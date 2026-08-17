#!/usr/bin/env python3
"""Run AMB3R-VO (DA3) on a continuous R2R RGB clip and cache its poses."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _history_indices(current: int, count: int) -> np.ndarray:
    if current <= 0:
        return np.zeros(0, dtype=np.int64)
    if current <= count:
        return np.arange(current, dtype=np.int64)
    return np.linspace(0, current - 1, count, dtype=np.int64)


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _pose_metrics(
    predicted_c2w_opencv: np.ndarray,
    gt_c2w_habitat: np.ndarray,
    *,
    current_indices: list[int] | range,
    num_history: int,
) -> dict[str, float | int]:
    from src.data.trajectory_utils import compute_history_rel_poses
    from src.vo.amb3r_pose import (
        fit_global_translation_scale,
        history_rel_poses_from_amb3r,
    )

    predicted_rows, target_rows = [], []
    for current in current_indices:
        indices = _history_indices(current, num_history)
        predicted_rows.append(
            history_rel_poses_from_amb3r(
                predicted_c2w_opencv,
                indices,
                current,
            )
        )
        target_rows.append(
            compute_history_rel_poses(
                [gt_c2w_habitat[int(index)] for index in indices],
                gt_c2w_habitat[current],
                camera_forward_axis="-z",
            )
        )
    predicted = np.concatenate(predicted_rows, axis=0)
    target = np.concatenate(target_rows, axis=0)
    delta_xy = predicted[:, :2] - target[:, :2]
    translation_error = np.linalg.norm(delta_xy, axis=-1)
    pred_yaw = np.arctan2(predicted[:, 3], predicted[:, 2])
    target_yaw = np.arctan2(target[:, 3], target[:, 2])
    yaw_error_deg = np.degrees(np.abs(_wrap_angle(pred_yaw - target_yaw)))
    pred_norm = np.linalg.norm(predicted[:, :2], axis=-1)
    gt_norm = np.linalg.norm(target[:, :2], axis=-1)
    nontrivial = (pred_norm > 1e-3) & (gt_norm > 0.05)
    raw_scale_ratio = pred_norm[nontrivial] / gt_norm[nontrivial]
    oracle_scale = fit_global_translation_scale(predicted, target)
    scaled_error = np.linalg.norm(
        predicted[:, :2] * oracle_scale - target[:, :2], axis=-1
    )
    return {
        "relative_pose_rows": int(len(predicted)),
        "translation_mae_m_raw": float(translation_error.mean()),
        "translation_median_m_raw": float(np.median(translation_error)),
        "translation_p90_m_raw": float(np.quantile(translation_error, 0.9)),
        "yaw_mae_deg_raw": float(yaw_error_deg.mean()),
        "yaw_median_deg_raw": float(np.median(yaw_error_deg)),
        "yaw_p90_deg_raw": float(np.quantile(yaw_error_deg, 0.9)),
        "native_scale_ratio_median": (
            float(np.median(raw_scale_ratio)) if raw_scale_ratio.size else math.nan
        ),
        "diagnostic_oracle_per_clip_scale": float(oracle_scale),
        "translation_mae_m_oracle_scaled_diagnostic_only": float(scaled_error.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--amb3r-root", required=True)
    parser.add_argument("--da3-checkpoint", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("backend", "direct"), default="backend")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--resolution", type=int, nargs=2, default=(518, 392), metavar=("W", "H"))
    parser.add_argument("--map-init-window", type=int, default=20)
    parser.add_argument("--map-every", type=int, default=8)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--num-history", type=int, default=8)
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve(strict=True)
    amb3r_root = Path(args.amb3r_root).expanduser().resolve(strict=True)
    checkpoint = Path(args.da3_checkpoint).expanduser().resolve(strict=True)
    sys.path[:0] = [str(repo), str(amb3r_root), str(amb3r_root / "thirdparty")]

    from amb3r.model_zoo import load_model
    from src.vo.clip_io import center_crop_resize_for_amb3r, load_continuous_front_clip

    frame_ids, images_rgb, gt_poses, meta = load_continuous_front_clip(
        args.clip,
        max_frames=args.max_frames,
    )
    if len(frame_ids) <= args.min_history:
        raise ValueError(
            "Pose/heatmap evaluation requires at least one current frame after "
            f"min_history={args.min_history}; loaded {len(frame_ids)} frames"
        )
    processed = center_crop_resize_for_amb3r(
        images_rgb,
        resolution=tuple(args.resolution),
    )
    images = torch.from_numpy(processed).permute(0, 3, 1, 2).float()
    images = images.div_(127.5).sub_(1.0).unsqueeze(0)

    if args.mode == "backend" and len(frame_ids) < args.map_init_window:
        raise ValueError(
            f"backend mode requires at least {args.map_init_window} frames; "
            f"loaded {len(frame_ids)}"
        )
    device = torch.device(args.device)
    start = time.monotonic()
    model = load_model("da3", ckpt_path=str(checkpoint)).to(device).eval()
    load_seconds = time.monotonic() - start

    infer_start = time.monotonic()
    if args.mode == "direct":
        with torch.inference_mode():
            predicted = model.predict_camera_poses(images.to(device))[0].cpu().numpy()
        keyframes = np.arange(len(frame_ids), dtype=np.int64)
    else:
        from slam.pipeline import AMB3R_VO

        pipeline = AMB3R_VO(model, cfg_path=str(amb3r_root / "slam" / "slam_config.yaml"))
        pipeline.cfg.device = str(device)
        pipeline.cfg.map_init_window = int(args.map_init_window)
        pipeline.cfg.map_every = int(args.map_every)
        result = pipeline.run(images)
        predicted = result.poses[: len(frame_ids)].numpy()
        keyframes = result.kf_idx.numpy() if result.kf_idx is not None else np.zeros(0, dtype=np.int64)
    infer_seconds = time.monotonic() - infer_start

    predicted = np.asarray(predicted, dtype=np.float32)
    if predicted.shape != (len(frame_ids), 4, 4) or not np.isfinite(predicted).all():
        raise RuntimeError(f"Invalid AMB3R trajectory shape/content: {predicted.shape}")
    if not np.allclose(
        predicted[:, 3, :],
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        atol=1e-3,
        rtol=0.0,
    ):
        raise RuntimeError("AMB3R returned non-rigid homogeneous poses")

    metrics = {
        "causal_final_current": _pose_metrics(
            predicted,
            gt_poses,
            current_indices=[len(frame_ids) - 1],
            num_history=args.num_history,
        ),
        "retrospective_final_trajectory_diagnostic": _pose_metrics(
            predicted,
            gt_poses,
            current_indices=range(args.min_history, len(frame_ids)),
            num_history=args.num_history,
        ),
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frame_ids=frame_ids,
        poses_c2w_opencv=predicted,
        gt_poses_c2w_habitat=gt_poses,
        keyframe_indices=keyframes,
    )
    manifest = {
        "schema": "heatmapvln-amb3r-vo-pose-cache-v1",
        "clip": str(Path(args.clip).expanduser().resolve()),
        "scene_id": meta.get("scene_id"),
        "episode_id": meta.get("episode_id"),
        "frame_count": int(len(frame_ids)),
        "mode": args.mode,
        "model": "AMB3R-VO (DA3)",
        "da3_checkpoint": str(checkpoint),
        "checkpoint_hash_enforced": False,
        "input": "continuous rgb_front frames",
        "resolution_wh": list(args.resolution),
        "map_init_window": int(args.map_init_window),
        "map_every": int(args.map_every),
        "model_load_seconds": float(load_seconds),
        "inference_seconds": float(infer_seconds),
        "frames_per_second": float(len(frame_ids) / max(infer_seconds, 1e-9)),
        "pose_metrics": metrics,
        "warning": (
            "diagnostic_oracle_per_clip_scale is reported only to diagnose scale; "
            "it is not applied and is forbidden at deployment. Earlier-current "
            "trajectory metrics are retrospective and are not an online causal audit."
        ),
    }
    manifest_path = output.with_suffix(output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"pose_cache": str(output), "manifest": str(manifest_path), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

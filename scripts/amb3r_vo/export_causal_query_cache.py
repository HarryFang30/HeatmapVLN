#!/usr/bin/env python3
"""Cache causal AMB3R-VO pose queries for a continuous qualitative strip.

Unlike an offline final-trajectory export, every row in this cache is frozen
immediately after the corresponding current frame is ingested.  Later map
updates therefore cannot leak future RGB into an earlier visualization frame.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _history_indices(current: int, count: int) -> np.ndarray:
    if current <= 0:
        return np.empty(0, dtype=np.int64)
    if current <= count:
        return np.arange(current, dtype=np.int64)
    return np.linspace(0, current - 1, count, dtype=np.int64)


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--amb3r-root", required=True)
    parser.add_argument("--da3-checkpoint", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--query-start", type=int, default=19)
    parser.add_argument(
        "--query-end",
        type=int,
        default=-1,
        help="Inclusive final current frame; -1 uses the final loaded frame.",
    )
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--map-init-window", type=int, default=20)
    parser.add_argument("--map-every", type=int, default=8)
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=(518, 392),
        metavar=("W", "H"),
    )
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve(strict=True)
    amb3r_root = Path(args.amb3r_root).expanduser().resolve(strict=True)
    checkpoint = Path(args.da3_checkpoint).expanduser().resolve(strict=True)
    clip = Path(args.clip).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().resolve()
    if output.suffix != ".npz":
        output = output.with_suffix(output.suffix + ".npz")
    if args.num_history < 1:
        raise ValueError("num-history must be positive")
    if not np.isfinite(args.translation_scale) or args.translation_scale <= 0:
        raise ValueError("translation-scale must be finite and positive")

    sys.path[:0] = [str(repo), str(amb3r_root), str(amb3r_root / "thirdparty")]
    from amb3r.model_zoo import load_model
    from src.data.trajectory_utils import compute_history_rel_poses
    from src.vo.clip_io import load_continuous_front_clip
    from src.vo.online_amb3r import OnlineAMB3RSession, StatefulAMB3RBackend

    frame_ids, images_rgb, gt_poses, meta = load_continuous_front_clip(
        clip,
        max_frames=args.max_frames,
    )
    if not np.array_equal(frame_ids, np.arange(len(frame_ids), dtype=frame_ids.dtype)):
        raise ValueError("Clip frame IDs must be contiguous from zero")
    query_end = len(frame_ids) - 1 if args.query_end < 0 else int(args.query_end)
    if not 0 <= args.query_start <= query_end < len(frame_ids):
        raise ValueError(
            f"Invalid inclusive query range [{args.query_start}, {query_end}] "
            f"for {len(frame_ids)} loaded frames"
        )
    first_online_query = args.map_init_window - 1
    if args.query_start < first_online_query:
        raise ValueError(
            "This causal strip exporter starts at or after stateful map "
            f"initialization ({first_online_query})"
        )
    if args.query_start < args.num_history:
        raise ValueError(
            "query-start must be at least num-history so every cached row has "
            f"the fixed [K,4] contract; got {args.query_start} < {args.num_history}"
        )

    load_start = time.monotonic()
    model = load_model("da3", ckpt_path=str(checkpoint))
    backend = StatefulAMB3RBackend(
        model,
        cfg_path=amb3r_root / "slam" / "slam_config.yaml",
        device=args.device,
        map_init_window=args.map_init_window,
        map_every=args.map_every,
    )
    session = OnlineAMB3RSession(
        backend,
        map_init_window=args.map_init_window,
        map_every=args.map_every,
        max_history=args.num_history,
        resolution=tuple(args.resolution),
    )
    load_seconds = time.monotonic() - load_start

    session_id = f"qualitative-{meta.get('scene_id', clip.parent.name)}-{clip.name}"
    session.reset(session_id, max_frames=query_end + 1)
    current_rows: list[int] = []
    history_rows: list[np.ndarray] = []
    relative_rows: list[np.ndarray] = []
    phases: list[str] = []
    revisions: list[int] = []
    last_mapped: list[int] = []
    per_frame_metrics: list[dict[str, float | int | str | None]] = []

    inference_start = time.monotonic()
    for current in range(query_end + 1):
        session.ingest(
            session_id,
            frame_id=current,
            frame_rgb=images_rgb[current],
            capture_step=current,
        )
        if current < first_online_query:
            continue
        history = _history_indices(current, args.num_history)
        result = session.query(
            session_id,
            current_frame_id=current,
            history_frame_ids=history.tolist(),
            translation_scale=args.translation_scale,
        )
        if not result.ready or result.history_rel_poses.shape != (len(history), 4):
            raise RuntimeError(
                f"AMB3R query at frame {current} is not ready or has wrong shape"
            )
        # Query every frame from stateful initialization onward, matching the
        # online navigation cadence.  ``query_start`` controls only which rows
        # are retained for visualization; it must not change map updates.
        if current < args.query_start:
            continue
        target = compute_history_rel_poses(
            [gt_poses[int(index)] for index in history],
            gt_poses[current],
            camera_forward_axis="-z",
        )
        prediction = np.asarray(result.history_rel_poses, dtype=np.float32)
        translation_error = np.linalg.norm(prediction[:, :2] - target[:, :2], axis=1)
        pred_yaw = np.arctan2(prediction[:, 3], prediction[:, 2])
        target_yaw = np.arctan2(target[:, 3], target[:, 2])
        yaw_error = np.degrees(np.abs(_wrap_angle(pred_yaw - target_yaw)))
        pred_norm = np.linalg.norm(prediction[:, :2], axis=1)
        target_norm = np.linalg.norm(target[:, :2], axis=1)
        nontrivial = (pred_norm > 1e-3) & (target_norm > 0.05)
        ratios = pred_norm[nontrivial] / target_norm[nontrivial]
        per_frame_metrics.append(
            {
                "current_frame_id": current,
                "translation_mae_m": float(translation_error.mean()),
                "yaw_mae_deg": float(yaw_error.mean()),
                "native_scale_ratio_median": (
                    float(np.median(ratios)) if ratios.size else None
                ),
                "provider_phase": result.provider_phase,
                "trajectory_revision": result.trajectory_revision,
            }
        )
        current_rows.append(current)
        history_rows.append(history)
        relative_rows.append(prediction)
        phases.append(result.provider_phase)
        revisions.append(result.trajectory_revision)
        last_mapped.append(
            -1 if result.last_mapped_frame_id is None else result.last_mapped_frame_id
        )
    inference_seconds = time.monotonic() - inference_start

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        current_frame_ids=np.asarray(current_rows, dtype=np.int64),
        history_frame_ids=np.stack(history_rows).astype(np.int64),
        history_rel_poses=np.stack(relative_rows).astype(np.float32),
        provider_phases=np.asarray(phases),
        trajectory_revisions=np.asarray(revisions, dtype=np.int64),
        last_mapped_frame_ids=np.asarray(last_mapped, dtype=np.int64),
    )
    manifest = {
        "schema": "heatmapvln-amb3r-causal-query-cache-v1",
        "clip": str(clip),
        "scene_id": meta.get("scene_id"),
        "episode_id": meta.get("episode_id"),
        "input": "stored continuous rgb_front frames in increasing frame_id order",
        "pose_provider": "amb3r_vo_da3",
        "pose_convention": "forward_m,left_m,cos_relative_yaw,sin_relative_yaw",
        "query_semantics": (
            "each row was saved immediately at that current frame; later trajectory "
            "revisions cannot update it"
        ),
        "causal": True,
        "query_every_frame_from_map_initialization": True,
        "first_online_query_frame": int(first_online_query),
        "query_start_inclusive": int(args.query_start),
        "query_end_inclusive": int(query_end),
        "query_count": len(current_rows),
        "num_history": int(args.num_history),
        "loaded_frame_count": int(len(frame_ids)),
        "resolution_wh": [int(args.resolution[0]), int(args.resolution[1])],
        "array_shapes": {
            "current_frame_ids": [len(current_rows)],
            "history_frame_ids": [len(current_rows), int(args.num_history)],
            "history_rel_poses": [len(current_rows), int(args.num_history), 4],
        },
        "history_sampling": "numpy.linspace(0,current-1,K,dtype=int64)",
        "translation_scale": float(args.translation_scale),
        "per_episode_gt_scale_used": False,
        "checkpoint_hash_enforced": False,
        "da3_checkpoint": str(checkpoint),
        "map_init_window": int(args.map_init_window),
        "map_every": int(args.map_every),
        "model_load_seconds": float(load_seconds),
        "inference_seconds": float(inference_seconds),
        "per_frame_pose_metrics": per_frame_metrics,
        "warning": (
            "GT pose is used only for offline error reporting. It is never passed "
            "to the AMB3R session or used to scale its output."
        ),
    }
    manifest_path = output.with_suffix(output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "manifest": str(manifest_path),
                "query_count": len(current_rows),
                "inference_seconds": inference_seconds,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

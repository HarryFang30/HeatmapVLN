#!/usr/bin/env python3
"""Exercise reset/continuous-ingest/query against a real AMB3R-VO server.

GT poses are read only after each RPC response to report diagnostic error; no
pose or scale field is ever sent to the VO process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _history_indices(current: int, maximum: int) -> list[int]:
    count = min(int(maximum), int(current))
    if count <= 0:
        return []
    return np.linspace(0, current - 1, num=count, dtype=np.int64).tolist()


def _wrapped_yaw_error_deg(predicted: np.ndarray, target: np.ndarray) -> float:
    pred_yaw = np.arctan2(predicted[:, 3], predicted[:, 2])
    gt_yaw = np.arctan2(target[:, 3], target[:, 2])
    delta = np.arctan2(np.sin(pred_yaw - gt_yaw), np.cos(pred_yaw - gt_yaw))
    return float(np.degrees(np.abs(delta)).mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--rpc-root",
        default=os.environ.get(
            "RPC_ROOT",
            "/mnt/afs/liwenhao/agent/370910109/rpc",
        ),
        help="Repository containing src/vla_rpc",
    )
    parser.add_argument("--clip", required=True)
    parser.add_argument("--server", default="127.0.0.1:50081")
    parser.add_argument("--max-frames", type=int, default=21)
    parser.add_argument("--max-history", type=int, default=8)
    parser.add_argument("--timeout-ms", type=int, default=600000)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--session-id", default="online-rpc-smoke")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve(strict=True)
    rpc_src = (
        Path(args.rpc_root).expanduser().resolve(strict=True) / "src"
    )
    if not (rpc_src / "vla_rpc").is_dir():
        raise FileNotFoundError(f"Expected vla_rpc package under {rpc_src}")
    sys.path[:0] = [str(repo), str(rpc_src)]

    if args.max_frames < 2:
        raise ValueError("--max-frames must be at least two")
    if args.max_history < 1:
        raise ValueError("--max-history must be positive")
    if args.timeout_ms < 1:
        raise ValueError("--timeout-ms must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")

    from src.data.trajectory_utils import compute_history_rel_poses
    from src.vo.clip_io import load_continuous_front_clip
    from vla_rpc.client import VLAClient
    from vla_rpc.core.image import encode_rgb_to_jpeg

    frame_ids, images, gt_c2w, _ = load_continuous_front_clip(
        args.clip,
        max_frames=args.max_frames,
    )
    if len(images) < 2:
        raise ValueError("online RPC smoke requires at least two frames")

    client = VLAClient(
        server_addr=args.server,
        timeout_ms=args.timeout_ms,
        jpeg_quality=args.jpeg_quality,
    )
    client.connect()
    if not client.health_check():
        raise RuntimeError(f"AMB3R-VO server is not healthy: {args.server}")
    info = client.get_server_info()
    if info is None or info.version != "heatmapvln-amb3r-vo-json-v1":
        raise RuntimeError(
            "AMB3R-VO protocol mismatch: "
            f"{getattr(info, 'version', None)!r}"
        )

    reset_result = client.infer_json(
        "reset_episode",
        {"session_id": args.session_id, "max_frames": len(images)},
        [],
    )
    if reset_result is None or not reset_result[0].get("ok", False):
        raise RuntimeError(f"AMB3R-VO reset failed: {reset_result}")

    reports = []
    query_frames = {len(images) - 1}
    if len(images) >= 21:
        query_frames.add(19)
    for frame_id, image in zip(frame_ids.tolist(), images):
        blob = {
            "name": "rgb_front",
            "data": encode_rgb_to_jpeg(image, quality=args.jpeg_quality),
            "mime_type": "image/jpeg",
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
        }
        ingest_result = client.infer_json(
            "ingest_frame",
            {
                "session_id": args.session_id,
                "frame_id": int(frame_id),
                "capture_step": int(frame_id),
            },
            [blob],
        )
        if ingest_result is None or not ingest_result[0].get("ok", False):
            raise RuntimeError(
                f"AMB3R-VO ingest failed at frame {frame_id}: {ingest_result}"
            )
        if frame_id not in query_frames:
            continue

        history = _history_indices(frame_id, args.max_history)
        query_result = client.infer_json(
            "query_relative_poses",
            {
                "session_id": args.session_id,
                "current_frame_id": int(frame_id),
                "history_frame_ids": history,
            },
            [],
        )
        if query_result is None or not query_result[0].get("ok", False):
            raise RuntimeError(
                f"AMB3R-VO query failed at frame {frame_id}: {query_result}"
            )
        response = query_result[0]
        predicted = np.asarray(response["history_rel_poses"], dtype=np.float32)
        target = compute_history_rel_poses(
            [gt_c2w[index] for index in history],
            gt_c2w[frame_id],
            camera_forward_axis="-z",
        )
        reports.append(
            {
                "current_frame_id": int(frame_id),
                "history_frame_ids": history,
                "ready": bool(response["ready"]),
                "provider_phase": response["provider_phase"],
                "trajectory_revision": int(response["trajectory_revision"]),
                "translation_mae_m": float(
                    np.linalg.norm(predicted[:, :2] - target[:, :2], axis=-1).mean()
                ),
                "yaw_mae_deg": _wrapped_yaw_error_deg(predicted, target),
            }
        )

    print(json.dumps({"ok": True, "reports": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

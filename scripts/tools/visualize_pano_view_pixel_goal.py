#!/usr/bin/env python3
"""Sanity-check pano view labels by overlaying goals on 4-view RGB grids."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.pano_view_pixel_goal import (  # noqa: E402
    PANO_HORIZONTAL_VIEWS,
    label_clip_frames,
    load_poses_from_chunks,
    project_to_all_views,
    goal_world_from_pose,
    load_depth_from_chunks,
    load_intrinsics,
    resolve_farthest_pano_pixel_goal,
)


def _load_rgb(clip_dir: Path, frame_id: int, direction: str) -> np.ndarray:
    chunks_dir = clip_dir / "chunks"
    for chunk_path in sorted(chunks_dir.glob("chunk_*.npz")):
        with np.load(chunk_path, allow_pickle=True) as z:
            key = f"rgb_{direction}"
            if key not in z.files:
                raise KeyError(key)
            frame_ids = np.asarray(z["frame_ids"], dtype=np.int32)
            local = np.where(frame_ids == frame_id)[0]
            if local.size == 0:
                continue
            buf = z[key][int(local[0])]
            arr = np.frombuffer(buf.tobytes() if hasattr(buf, "tobytes") else buf, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to decode {direction} frame {frame_id}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raise FileNotFoundError(f"No RGB for frame {frame_id} in {clip_dir}")


def _draw_marker(img: np.ndarray, u: int, v: int, color: tuple[int, int, int], label: str) -> None:
    cv2.circle(img, (u, v), 6, color, 2)
    cv2.drawMarker(img, (u, v), color, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
    cv2.putText(
        img,
        label,
        (max(0, u - 40), max(12, v - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
        cv2.LINE_AA,
    )


def visualize_sample(
    clip_dir: Path,
    frame_id: int,
    output_path: Path,
    *,
    img_size: int = 256,
    min_goal_len: int = 3,
    max_side_dist_m: float = 6.0,
) -> dict:
    intrinsics = load_intrinsics(clip_dir)
    poses_by_view = {d: load_poses_from_chunks(clip_dir, d) for d in PANO_HORIZONTAL_VIEWS}
    depth_front = load_depth_from_chunks(clip_dir, frame_id, "front")

    pano_goal = resolve_farthest_pano_pixel_goal(
        current_t=frame_id,
        num_frames=len(poses_by_view["front"]),
        poses_by_view=poses_by_view,
        depth_front=depth_front,
        img_size=img_size,
        intrinsics=intrinsics,
        min_goal_len=min_goal_len,
        max_side_dist_m=max_side_dist_m,
    )

    tiles = []
    info_lines = [f"{clip_dir.name} frame={frame_id}"]

    if pano_goal is None:
        info_lines.append("no pixel goal")
        for direction in PANO_HORIZONTAL_VIEWS:
            rgb = _load_rgb(clip_dir, frame_id, direction)
            tile = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(tile, direction, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            tiles.append(tile)
    else:
        goal_len, canonical, legacy_uv = pano_goal
        fi = frame_id + goal_len
        world_xyz = goal_world_from_pose(poses_by_view["front"][fi])
        agent_world = goal_world_from_pose(poses_by_view["front"][frame_id])
        current_poses = {d: poses_by_view[d][frame_id] for d in PANO_HORIZONTAL_VIEWS}
        visible = project_to_all_views(
            world_xyz,
            current_poses,
            img_size,
            intrinsics,
            depth_front,
            max_side_dist_m=max_side_dist_m,
            agent_world=agent_world,
        )
        visible_map = {v.view_id: v for v in visible}
        info_lines.append(
            f"canonical={canonical.view_id} uv=[{canonical.u},{canonical.v}] "
            f"goal_len={goal_len} legacy_front={legacy_uv}"
        )

        for direction in PANO_HORIZONTAL_VIEWS:
            rgb = _load_rgb(clip_dir, frame_id, direction)
            tile = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(tile, direction, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if direction in visible_map:
                v = visible_map[direction]
                color = (0, 0, 255) if v.view_id == canonical.view_id else (0, 255, 255)
                _draw_marker(tile, v.u, v.v, color, "CANON" if v.view_id == canonical.view_id else "vis")
            tiles.append(tile)

    top = np.hstack(tiles[:2])
    bottom = np.hstack(tiles[2:4])
    grid = np.vstack([top, bottom])

    banner_h = 48
    banner = np.zeros((banner_h, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        banner,
        info_lines[-1][:120],
        (8, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    out = np.vstack([banner, grid])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out)

    return {
        "clip_dir": str(clip_dir),
        "frame_id": frame_id,
        "info": info_lines,
        "pano_goal": None if pano_goal is None else {
            "view_id": pano_goal[1].view_id,
            "uv": [pano_goal[1].u, pano_goal[1].v],
            "goal_len": pano_goal[0],
            "legacy_front": pano_goal[2],
        },
    }


def _discover_clips(root: Path, split: str) -> list[Path]:
    split_dir = root / split
    clips: list[Path] = []
    for scene_dir in sorted(split_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        for clip_dir in sorted(scene_dir.iterdir()):
            if clip_dir.is_dir() and (clip_dir / "meta.json").exists():
                clips.append(clip_dir)
    return clips


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize pano view pixel goals")
    p.add_argument("--root", default="/home/intern/zhr/fjl/r2r_paronamic_data")
    p.add_argument("--split", default="train")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--output-dir", default="outputs/pano_view_label_sanity")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--min-goal-len", type=int, default=3)
    p.add_argument("--max-side-dist-m", type=float, default=6.0)
    p.add_argument("--ensure-labels", action="store_true", help="Write pano_view_labels.json if missing")
    args = p.parse_args()

    clips = _discover_clips(Path(args.root), args.split)
    rng = random.Random(args.seed)
    rng.shuffle(clips)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    attempts = 0
    clip_idx = 0
    while len(results) < args.num_samples and clip_idx < len(clips):
        clip_dir = clips[clip_idx]
        clip_idx += 1
        with open(clip_dir / "meta.json") as f:
            meta = json.load(f)
        num_frames = int(meta["num_frames"])
        if num_frames < 8:
            continue

        frame_id = rng.randint(5, num_frames - 2)
        attempts += 1
        try:
            if args.ensure_labels and not (clip_dir / "pano_view_labels.json").exists():
                frames = label_clip_frames(
                    clip_dir,
                    img_size=args.image_size,
                    min_goal_len=args.min_goal_len,
                    max_side_dist_m=args.max_side_dist_m,
                )
                from src.data.pano_view_pixel_goal import write_clip_labels

                write_clip_labels(clip_dir, frames)

            out_path = out_dir / f"sanity_{len(results):02d}_{clip_dir.name}_f{frame_id}.jpg"
            rec = visualize_sample(
                clip_dir,
                frame_id,
                out_path,
                img_size=args.image_size,
                min_goal_len=args.min_goal_len,
                max_side_dist_m=args.max_side_dist_m,
            )
            rec["image"] = str(out_path)
            results.append(rec)
            print(f"Wrote {out_path}")
        except Exception as exc:
            print(f"Skip {clip_dir.name} f{frame_id}: {exc}")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"samples": results, "attempts": attempts}, f, indent=2)
    print(f"\nSaved {len(results)} visualizations to {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

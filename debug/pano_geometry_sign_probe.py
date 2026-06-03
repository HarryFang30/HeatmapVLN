#!/usr/bin/env python3
"""Probe panoramic right/left geometry conventions.

This script checks whether three conventions agree:

1. Eval capture yaw offsets in scripts/evaluation/r2r_val_unseen.py.
2. Adapter geometry centers in src/models/adapters/pano_latent_adapter.py.
3. Optional training dataset pose_front/pose_right/pose_back/pose_left.

Important: Habitat yaw offsets and adapter heading angles use opposite-looking
raw signs.  The comparison below converts both into an egocentric heading where
front=0 and right is positive.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


VIEW_NAMES = ("front", "right", "back", "left")

# Raw yaw offsets from r2r_val_unseen.capture_panoramic_views.
EVAL_YAW_OFFSETS_RAD = {
    "front": 0.0,
    "right": -math.pi / 2.0,
    "back": -math.pi,
    "left": -3.0 * math.pi / 2.0,
}

# Adapter centers from pano_latent_adapter._VIEW_CENTER_YAW_RAD.
ADAPTER_CENTER_RAD = {
    "front": 0.0,
    "right": math.pi / 2.0,
    "back": math.pi,
    "left": -math.pi / 2.0,
}

FLIPPED_CENTER_RAD = {
    "front": 0.0,
    "right": -math.pi / 2.0,
    "back": math.pi,
    "left": math.pi / 2.0,
}


def normalize_rad(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def deg(angle: float) -> float:
    return math.degrees(normalize_rad(angle))


def yaw_rot_y(angle: float) -> np.ndarray:
    """Rotation matrix matching np.quaternion(cos(a/2), 0, sin(a/2), 0)."""
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def right_positive_heading_from_rotation(rot: np.ndarray) -> float:
    """Return egocentric heading where front=0 and right is positive.

    Habitat camera local forward is -Z.  We project the world forward vector
    into the XZ plane, then use atan2(x, -z), so +X/right is +90 degrees.
    """
    forward = np.asarray(rot, dtype=np.float64) @ np.asarray([0.0, 0.0, -1.0])
    x = float(forward[0])
    z = float(forward[2])
    return normalize_rad(math.atan2(x, -z))


def eval_capture_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for view in VIEW_NAMES:
        raw_offset = EVAL_YAW_OFFSETS_RAD[view]
        eval_heading = right_positive_heading_from_rotation(yaw_rot_y(raw_offset))
        adapter_heading = normalize_rad(ADAPTER_CENTER_RAD[view])
        error = normalize_rad(eval_heading - adapter_heading)
        rows.append(
            {
                "view": view,
                "eval_raw_yaw_offset_deg": round(math.degrees(raw_offset), 3),
                "eval_heading_right_positive_deg": round(deg(eval_heading), 3),
                "adapter_center_deg": round(deg(adapter_heading), 3),
                "heading_error_deg": round(deg(error), 6),
                "matches_adapter_center": abs(error) < math.radians(1.0e-4),
            }
        )
    return rows


def resolve_clip_dir(data_root: str | None, clip_dir: str | None) -> Path | None:
    if clip_dir:
        path = Path(clip_dir).expanduser().resolve()
        return path
    if not data_root:
        return None
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {root}")
    for chunk in root.rglob("chunk_*.npz"):
        if chunk.parent.name == "chunks":
            return chunk.parent.parent
        return chunk.parent
    return None


def load_pose_table(clip_dir: Path, max_frames: int) -> list[dict[str, Any]]:
    chunks_dir = clip_dir / "chunks"
    chunk_files = sorted(chunks_dir.glob("chunk_*.npz"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.npz under {chunks_dir}")

    deltas: dict[str, list[float]] = {view: [] for view in VIEW_NAMES}
    frames_seen = 0
    for chunk_path in chunk_files:
        with np.load(chunk_path, allow_pickle=True) as z:
            missing = [f"pose_{view}" for view in VIEW_NAMES if f"pose_{view}" not in z.files]
            if missing:
                raise KeyError(f"{chunk_path} is missing keys: {missing}")
            pose_front = np.asarray(z["pose_front"], dtype=np.float64)
            n = pose_front.shape[0]
            for local_i in range(n):
                if frames_seen >= max_frames:
                    break
                front_heading = right_positive_heading_from_rotation(pose_front[local_i, :3, :3])
                for view in VIEW_NAMES:
                    pose = np.asarray(z[f"pose_{view}"], dtype=np.float64)
                    view_heading = right_positive_heading_from_rotation(pose[local_i, :3, :3])
                    deltas[view].append(normalize_rad(view_heading - front_heading))
                frames_seen += 1
        if frames_seen >= max_frames:
            break

    rows: list[dict[str, Any]] = []
    for view in VIEW_NAMES:
        values = deltas[view]
        if not values:
            continue
        median_delta = float(np.median(values))
        adapter_heading = ADAPTER_CENTER_RAD[view]
        err = normalize_rad(median_delta - adapter_heading)
        rows.append(
            {
                "view": view,
                "dataset_pose_delta_deg_median": round(deg(median_delta), 3),
                "adapter_center_deg": round(deg(adapter_heading), 3),
                "heading_error_deg": round(deg(err), 6),
                "num_frames": len(values),
            }
        )
    return rows


def print_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    if not rows:
        print("  <empty>")
        return
    keys = list(rows[0].keys())
    widths = {
        key: max(len(key), *(len(str(row.get(key, ""))) for row in rows))
        for key in keys
    }
    print("  " + "  ".join(key.ljust(widths[key]) for key in keys))
    print("  " + "  ".join("-" * widths[key] for key in keys))
    for row in rows:
        print("  " + "  ".join(str(row.get(key, "")).ljust(widths[key]) for key in keys))


def load_geometry_adapter(path: Path, device: str):
    import torch

    from src.models.adapters import GeometryAwarePanoToNextDiTAdapter

    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    state = ckpt.get("adapter_state_dict")
    if state is None:
        raise KeyError(f"{path} has no adapter_state_dict")
    if "student_proj.weight" not in state:
        raise RuntimeError(
            f"{path} is not a geometry-aware adapter checkpoint "
            "(missing student_proj.weight)"
        )
    saved_args = ckpt.get("args", {}) or {}
    student_dim = int(state["student_proj.weight"].shape[1])
    adapter_dim = int(state["student_proj.weight"].shape[0])
    output_dim = int(state["output_proj.weight"].shape[0])
    num_query = int(state["output_queries"].shape[0])
    ffn_dim = int(state["layers.0.linear1.weight"].shape[0])
    geometry_embed_dim = int(state["view_embedding.weight"].shape[1])
    layer_ids = {
        int(name.split(".")[1])
        for name in state
        if name.startswith("layers.") and name.split(".")[1].isdigit()
    }
    adapter = GeometryAwarePanoToNextDiTAdapter(
        student_dim=student_dim,
        adapter_dim=adapter_dim,
        output_dim=output_dim,
        num_query=num_query,
        num_layers=max(len(layer_ids), 1),
        num_heads=int(saved_args.get("adapter_num_heads", 8)),
        ffn_dim=ffn_dim,
        dropout=float(saved_args.get("adapter_dropout", 0.0)),
        geometry_embed_dim=geometry_embed_dim,
        horizontal_fov_deg=float(saved_args.get("adapter_horizontal_fov_deg", 90.0)),
    )
    adapter.load_state_dict(state)
    adapter.eval().to(device)
    return adapter, {
        "student_dim": student_dim,
        "adapter_dim": adapter_dim,
        "output_dim": output_dim,
        "num_query": num_query,
        "horizontal_fov_deg": adapter.horizontal_fov_deg,
    }


def adapter_sign_sensitivity(
    checkpoint: Path,
    *,
    device: str,
    seed: int,
    batch_size: int,
    image_size: tuple[int, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    import src.models.adapters.pano_latent_adapter as adapter_mod

    adapter, meta = load_geometry_adapter(checkpoint, device)
    torch.manual_seed(seed)
    student_latents = torch.randn(
        batch_size,
        meta["num_query"],
        meta["student_dim"],
        device=device,
        dtype=torch.float32,
    )
    width, height = image_size
    pixels = {
        "center": (width // 2, height // 2),
        "left_quarter": (width // 4, height // 2),
        "right_quarter": (3 * width // 4, height // 2),
    }
    image_hw = torch.tensor([[height, width]] * batch_size, device=device, dtype=torch.float32)

    def run_with_centers(centers: dict[str, float], view: str, pixel: tuple[int, int]):
        old = adapter_mod._VIEW_CENTER_YAW_RAD
        adapter_mod._VIEW_CENTER_YAW_RAD = torch.tensor(
            [centers[v] for v in VIEW_NAMES],
            dtype=torch.float32,
        )
        try:
            view_idx = adapter_mod.view_ids_to_indices([view] * batch_size, device=torch.device(device))
            pixel_xy = torch.tensor([list(pixel)] * batch_size, device=device, dtype=torch.float32)
            with torch.inference_mode():
                return adapter(student_latents, view_idx, pixel_xy, image_hw).float()
        finally:
            adapter_mod._VIEW_CENTER_YAW_RAD = old

    rows: list[dict[str, Any]] = []
    for view in VIEW_NAMES:
        for pixel_name, pixel in pixels.items():
            current = run_with_centers(ADAPTER_CENTER_RAD, view, pixel)
            flipped = run_with_centers(FLIPPED_CENTER_RAD, view, pixel)
            delta = current - flipped
            current_flat = current.flatten(1)
            flipped_flat = flipped.flatten(1)
            cosine = torch.nn.functional.cosine_similarity(current_flat, flipped_flat, dim=1)
            rel_delta = delta.flatten(1).norm(dim=1) / current_flat.norm(dim=1).clamp_min(1.0e-6)
            rows.append(
                {
                    "view": view,
                    "pixel": pixel_name,
                    "cos_current_vs_flipped_mean": round(float(cosine.mean().item()), 6),
                    "rel_delta_norm_mean": round(float(rel_delta.mean().item()), 6),
                    "rms_abs_delta": round(float(delta.pow(2).mean().sqrt().item()), 6),
                }
            )
    return meta, rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Debug panoramic right/left geometry sign conventions."
    )
    parser.add_argument("--data-root", default="", help="Optional panoramic data root.")
    parser.add_argument("--clip-dir", default="", help="Optional single clip directory containing chunks/.")
    parser.add_argument("--max-frames", type=int, default=64, help="Max dataset frames to inspect.")
    parser.add_argument("--adapter-checkpoint", default="", help="Optional geometry-aware adapter .pth.")
    parser.add_argument("--device", default="cpu", help="Device for adapter sensitivity check.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256], metavar=("W", "H"))
    parser.add_argument(
        "--output",
        default="debug/pano_geometry_sign_probe.json",
        help="JSON output path.",
    )
    args = parser.parse_args()

    result: dict[str, Any] = {
        "eval_capture_vs_adapter": eval_capture_table(),
        "dataset_pose_vs_adapter": None,
        "adapter_sign_sensitivity": None,
    }

    print_table("Eval capture offsets converted to right-positive heading", result["eval_capture_vs_adapter"])

    clip = resolve_clip_dir(args.data_root or None, args.clip_dir or None)
    if clip is not None:
        pose_rows = load_pose_table(clip, max_frames=args.max_frames)
        result["dataset_pose_vs_adapter"] = {
            "clip_dir": str(clip),
            "rows": pose_rows,
        }
        print_table(f"Dataset pose deltas vs adapter centers ({clip})", pose_rows)
    else:
        print("\nDataset pose check skipped: pass --data-root or --clip-dir.")

    if args.adapter_checkpoint:
        meta, sensitivity_rows = adapter_sign_sensitivity(
            Path(args.adapter_checkpoint).expanduser().resolve(),
            device=args.device,
            seed=args.seed,
            batch_size=args.batch_size,
            image_size=(int(args.image_size[0]), int(args.image_size[1])),
        )
        result["adapter_sign_sensitivity"] = {
            "checkpoint": str(Path(args.adapter_checkpoint).expanduser().resolve()),
            "meta": meta,
            "rows": sensitivity_rows,
        }
        print_table("Adapter output sensitivity: current centers vs flipped right/left", sensitivity_rows)
    else:
        print("\nAdapter sensitivity skipped: pass --adapter-checkpoint.")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {output}")
    print(
        "\nInterpretation:\n"
        "  - If eval heading_error_deg is ~0, eval raw yaw offsets and adapter centers are not sign-mismatched.\n"
        "  - If dataset pose heading_error_deg is ~0, training pose_right/left also matches adapter centers.\n"
        "  - Adapter sensitivity only says whether flipping signs changes adapter outputs; it does not prove which sign is correct."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Read action-dense R2R front-camera clips for visual odometry."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _decode_rgb(raw: object) -> np.ndarray:
    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        image = raw[..., :3]
        # Collection chunks store OpenCV/BGR arrays when they are uncompressed.
        return cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGR2RGB)
    if isinstance(raw, np.ndarray):
        encoded = np.asarray(raw, dtype=np.uint8).reshape(-1)
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        encoded = np.frombuffer(raw, dtype=np.uint8)
    else:
        encoded = np.asarray(raw, dtype=np.uint8).reshape(-1)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Unable to decode an rgb_front frame")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_continuous_front_clip(
    clip_dir: str | Path,
    *,
    max_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load ``frame_ids, rgb_front, pose_front, meta`` in temporal order.

    The strict contiguous-ID check prevents accidentally evaluating VO on the
    sparse K-history union stored by DAgger rather than an action-dense expert
    video.
    """

    clip = Path(clip_dir).expanduser().resolve(strict=True)
    meta_path = clip / "meta.json"
    chunk_paths = sorted((clip / "chunks").glob("chunk_*.npz"))
    if not meta_path.is_file() or not chunk_paths:
        raise FileNotFoundError(f"Expected meta.json and chunk_*.npz under {clip}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    for chunk_path in chunk_paths:
        with np.load(chunk_path, allow_pickle=True) as payload:
            required = {"frame_ids", "rgb_front", "pose_front"}
            missing = required.difference(payload.files)
            if missing:
                raise KeyError(f"{chunk_path} is missing {sorted(missing)}")
            frame_ids = np.asarray(payload["frame_ids"], dtype=np.int64)
            rgb = payload["rgb_front"]
            poses = np.asarray(payload["pose_front"], dtype=np.float32)
            if not (len(frame_ids) == len(rgb) == len(poses)):
                raise ValueError(f"Chunk field lengths differ in {chunk_path}")
            rows.extend(
                (int(frame_id), _decode_rgb(raw_rgb), pose)
                for frame_id, raw_rgb, pose in zip(frame_ids, rgb, poses)
            )
    rows.sort(key=lambda row: row[0])
    if max_frames > 0:
        rows = rows[: int(max_frames)]
    if not rows:
        raise ValueError(f"No frames loaded from {clip}")

    frame_ids = np.asarray([row[0] for row in rows], dtype=np.int64)
    if frame_ids[0] != 0 or not np.array_equal(
        frame_ids,
        np.arange(len(frame_ids), dtype=np.int64),
    ):
        raise ValueError(
            "AMB3R-VO requires an action-dense sequence with contiguous frame "
            f"IDs starting at zero; got first={frame_ids[0]}, last={frame_ids[-1]}, "
            f"count={len(frame_ids)}"
        )
    images = np.stack([row[1] for row in rows], axis=0)
    poses = np.stack([row[2] for row in rows], axis=0).astype(np.float32)
    if poses.shape != (len(rows), 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"Invalid pose_front trajectory: {poses.shape}")
    return frame_ids, images, poses, meta


def center_crop_resize_for_amb3r(
    images_rgb: np.ndarray,
    *,
    resolution: tuple[int, int] = (518, 392),
) -> np.ndarray:
    """Match released ``slam/datasets/demo.py`` crop-then-resize exactly.

    AMB3R's current demo first takes an integer-rounded centered crop at the
    target aspect ratio, then Lanczos-resizes that crop to the exact model
    resolution.  Doing the mathematically similar resize-then-crop operation
    changes pixels because the two paths round and interpolate differently.
    """

    target_width, target_height = (int(value) for value in resolution)
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"Invalid AMB3R resolution: {resolution}")
    frames = np.asarray(images_rgb)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            "images_rgb must be uint8 [T,H,W,3], "
            f"got dtype={frames.dtype} shape={frames.shape}"
        )
    if len(frames) == 0:
        raise ValueError("images_rgb must contain at least one frame")
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    processed = []
    for array in frames:
        image = Image.fromarray(array, mode="RGB")
        width, height = image.size
        out_width, out_height = target_width, target_height
        if out_width >= out_height and height > 1.1 * width:
            out_width, out_height = out_height, out_width

        target_aspect = out_width / out_height
        image_aspect = width / height
        if image_aspect > target_aspect:
            crop_width = int(round(height * target_aspect))
            left = (width - crop_width) // 2
            image = image.crop((left, 0, left + crop_width, height))
        elif image_aspect < target_aspect:
            crop_height = int(round(width / target_aspect))
            top = (height - crop_height) // 2
            image = image.crop((0, top, width, top + crop_height))

        processed.append(
            np.asarray(image.resize((out_width, out_height), resampling))
        )
    return np.stack(processed, axis=0).astype(np.uint8, copy=False)

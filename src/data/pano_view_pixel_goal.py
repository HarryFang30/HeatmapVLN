"""Panoramic view-aware pixel goal labeling (C3 occlusion policy).

Rules (v1):
  - Project farthest visible future waypoint onto front/right/back/left.
  - Canonical view: use ``front`` when visible (with depth occlusion check);
    otherwise pick the side view closest to image center (geometry only).
  - Side views: no depth occlusion; reject if 3D distance > ``max_side_dist_m``.
  - STOP / TURN frames use ``view_stop`` / ``view_turn`` placeholders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PANO_HORIZONTAL_VIEWS: tuple[str, ...] = ("front", "right", "back", "left")
VIEW_STOP = "view_stop"
VIEW_TURN = "view_turn"  # legacy ambiguous turn — kept for backward compat
VIEW_TURN_LEFT = "view_turn_left"
VIEW_TURN_RIGHT = "view_turn_right"
LABEL_VERSION = 1


@dataclass(frozen=True)
class VisibleProjection:
    view_id: str
    u: int
    v: int
    dist_to_center_px: float
    z_depth_m: float


@dataclass(frozen=True)
class PanoPixelGoalLabel:
    sample_kind: str  # pixel | turn | stop
    pano_view_id: str
    pano_pixel_goal: list[int] | None
    pixel_goal_relative_len: int | None = None
    legacy_front_pixel_goal: list[int] | None = None
    waypoint_dist_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_kind": self.sample_kind,
            "pano_view_id": self.pano_view_id,
            "pano_pixel_goal": self.pano_pixel_goal,
            "pixel_goal_relative_len": self.pixel_goal_relative_len,
            "legacy_front_pixel_goal": self.legacy_front_pixel_goal,
            "waypoint_dist_m": self.waypoint_dist_m,
        }


_intrinsics_cache: dict[str, dict[str, float]] = {}


def load_intrinsics(clip_dir: Path) -> dict[str, float]:
    cache_key = str(clip_dir)
    cached = _intrinsics_cache.get(cache_key)
    if cached is not None:
        return cached

    path = clip_dir / "intrinsics.json"
    with open(path) as f:
        data = json.load(f)

    if "K" in data:
        K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
    else:
        fx = float(data["fx"])
        fy = float(data["fy"])
        cx = float(data["cx"])
        cy = float(data["cy"])

    width = data.get("width", data.get("image_width"))
    height = data.get("height", data.get("image_height"))
    if width is None:
        width = cx * 2.0
    if height is None:
        height = cy * 2.0

    result = {
        "width": float(width),
        "height": float(height),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }
    _intrinsics_cache[cache_key] = result
    return result


def load_poses_from_chunks(clip_dir: Path, direction: str) -> list[np.ndarray]:
    chunks_dir = clip_dir / "chunks"
    chunk_files = sorted(chunks_dir.glob("chunk_*.npz"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunks in {chunks_dir}")

    frame_poses: dict[int, np.ndarray] = {}
    key = f"pose_{direction}"
    for chunk_path in chunk_files:
        with np.load(chunk_path, allow_pickle=True) as z:
            if key not in z.files:
                raise KeyError(f"{key} missing in {chunk_path}")
            frame_ids = np.asarray(z["frame_ids"], dtype=np.int32)
            poses = np.asarray(z[key], dtype=np.float32)
            for local_i, frame_id in enumerate(frame_ids.tolist()):
                frame_poses[int(frame_id)] = poses[local_i]

    if not frame_poses:
        return []
    max_id = max(frame_poses)
    return [frame_poses[i] for i in range(max_id + 1)]


def load_depth_from_chunks(clip_dir: Path, frame_id: int, direction: str) -> np.ndarray | None:
    chunks_dir = clip_dir / "chunks"
    for chunk_path in sorted(chunks_dir.glob("chunk_*.npz")):
        with np.load(chunk_path, allow_pickle=True) as z:
            key = f"depth_{direction}"
            if key not in z.files:
                return None
            frame_ids = np.asarray(z["frame_ids"], dtype=np.int32)
            local = np.where(frame_ids == frame_id)[0]
            if local.size == 0:
                continue
            depth = np.asarray(z[key][int(local[0])], dtype=np.float32)
            return depth
    return None


def goal_world_from_pose(goal_pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(goal_pose, dtype=np.float64)
    return np.array([pose[0, 3], pose[1, 3], pose[2, 3]], dtype=np.float64)


def project_world_point(
    world_xyz: np.ndarray,
    cam_pose_c2w: np.ndarray,
    img_size: int | tuple[int, int] = 256,
    intrinsics: dict[str, float] | None = None,
    depth_map: np.ndarray | None = None,
    depth_tolerance: float = 0.5,
    check_occlusion: bool = True,
) -> tuple[int, int, float, float] | None:
    """Project a world point into a pinhole camera.

    Returns ``(u, v, dist_to_center_px, z_depth_m)`` or ``None``.
    Habitat convention: X right, Y up, -Z forward.
    """
    if isinstance(img_size, (tuple, list)):
        img_w, img_h = int(img_size[0]), int(img_size[1])
    else:
        img_w = img_h = int(img_size)

    if intrinsics is None:
        fx = fy = img_w / 2.0
        cx = cy = img_w / 2.0
    else:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])

    t_inv = np.linalg.inv(np.asarray(cam_pose_c2w, dtype=np.float64))
    p_world = np.array([*world_xyz, 1.0], dtype=np.float64)
    p_cam = t_inv @ p_world
    x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    if z >= -0.1:
        return None

    z_depth = -z
    u_f = fx * x / z_depth + cx
    v_f = fy * (-y) / z_depth + cy
    if u_f < 0 or u_f >= img_w or v_f < 0 or v_f >= img_h:
        return None

    u = max(0, min(img_w - 1, round(u_f)))
    v = max(0, min(img_h - 1, round(v_f)))
    dist_center = float(np.hypot(u_f - cx, v_f - cy))

    if check_occlusion and depth_map is not None:
        dm = depth_map
        if dm.ndim == 3 and dm.shape[-1] == 1:
            dm = dm[:, :, 0]
        dh, dw = dm.shape[:2]
        du = round(u_f * dw / img_w)
        dv = round(v_f * dh / img_h)
        du = max(0, min(dw - 1, du))
        dv = max(0, min(dh - 1, dv))
        pixel_depth = float(dm[dv, du])
        if pixel_depth > 0 and pixel_depth < z_depth - depth_tolerance:
            return None

    return int(u), int(v), dist_center, z_depth


def project_to_all_views(
    world_xyz: np.ndarray,
    poses_by_view: dict[str, np.ndarray],
    img_size: int | tuple[int, int],
    intrinsics: dict[str, float] | None,
    depth_front: np.ndarray | None,
    *,
    max_side_dist_m: float,
    agent_world: np.ndarray,
    depth_tolerance: float = 0.5,
) -> list[VisibleProjection]:
    dist_3d = float(np.linalg.norm(world_xyz - agent_world))
    visible: list[VisibleProjection] = []

    for view_id in PANO_HORIZONTAL_VIEWS:
        pose = poses_by_view[view_id]
        use_depth = view_id == "front"
        if view_id != "front" and dist_3d > max_side_dist_m:
            continue

        proj = project_world_point(
            world_xyz,
            pose,
            img_size=img_size,
            intrinsics=intrinsics,
            depth_map=depth_front if use_depth else None,
            depth_tolerance=depth_tolerance,
            check_occlusion=use_depth,
        )
        if proj is None:
            continue
        u, v, dist_center, z_depth = proj
        visible.append(
            VisibleProjection(
                view_id=view_id,
                u=u,
                v=v,
                dist_to_center_px=dist_center,
                z_depth_m=z_depth,
            )
        )
    return visible


def select_canonical_view(visible: list[VisibleProjection]) -> VisibleProjection | None:
    if not visible:
        return None
    front = [v for v in visible if v.view_id == "front"]
    if front:
        return front[0]
    return min(visible, key=lambda v: v.dist_to_center_px)


def align_internnav_discrete_actions(discrete_actions: np.ndarray) -> np.ndarray:
    if len(discrete_actions) <= 1:
        return discrete_actions
    return np.concatenate([discrete_actions[1:], np.array([0], dtype=discrete_actions.dtype)])


def resolve_farthest_pano_pixel_goal(
    *,
    current_t: int,
    num_frames: int,
    poses_by_view: dict[str, list[np.ndarray]],
    depth_front: np.ndarray | None,
    img_size: int | tuple[int, int] = 256,
    intrinsics: dict[str, float] | None = None,
    min_goal_len: int = 3,
    max_side_dist_m: float = 6.0,
    depth_tolerance: float = 0.5,
) -> tuple[int, VisibleProjection, list[int] | None] | None:
    """Return ``(goal_len, canonical_projection, legacy_front_uv)``."""
    if current_t >= num_frames - 1:
        return None

    agent_world = goal_world_from_pose(poses_by_view["front"][current_t])
    current_poses = {k: v[current_t] for k, v in poses_by_view.items()}
    proj_size = img_size

    for fi in range(num_frames - 1, current_t, -1):
        world_xyz = goal_world_from_pose(poses_by_view["front"][fi])
        visible = project_to_all_views(
            world_xyz,
            current_poses,
            proj_size,
            intrinsics,
            depth_front,
            max_side_dist_m=max_side_dist_m,
            agent_world=agent_world,
            depth_tolerance=depth_tolerance,
        )
        canonical = select_canonical_view(visible)
        if canonical is None:
            continue

        goal_len = fi - current_t
        if goal_len < min_goal_len:
            return None

        legacy_front = project_world_point(
            world_xyz,
            current_poses["front"],
            img_size=proj_size,
            intrinsics=intrinsics,
            depth_map=depth_front,
            depth_tolerance=depth_tolerance,
            check_occlusion=True,
        )
        legacy_uv = [legacy_front[0], legacy_front[1]] if legacy_front is not None else None
        return goal_len, canonical, legacy_uv

    return None


def classify_frame_label(
    *,
    frame_id: int,
    num_frames: int,
    discrete_action: int,
    pano_goal: tuple[int, VisibleProjection, list[int] | None] | None,
) -> PanoPixelGoalLabel:
    if frame_id == num_frames - 1:
        return PanoPixelGoalLabel(
            sample_kind="stop",
            pano_view_id=VIEW_STOP,
            pano_pixel_goal=None,
        )

    if pano_goal is not None:
        goal_len, canonical, legacy_uv = pano_goal
        agent_dist = None
        return PanoPixelGoalLabel(
            sample_kind="pixel",
            pano_view_id=canonical.view_id,
            pano_pixel_goal=[canonical.u, canonical.v],
            pixel_goal_relative_len=goal_len,
            legacy_front_pixel_goal=legacy_uv,
            waypoint_dist_m=agent_dist,
        )

    if discrete_action == 1:
        # Forward-only without pixel goal — skipped in InternNav SFT index.
        return PanoPixelGoalLabel(
            sample_kind="skip",
            pano_view_id=VIEW_TURN,
            pano_pixel_goal=None,
        )

    return PanoPixelGoalLabel(
        sample_kind="turn",
        pano_view_id=VIEW_TURN,
        pano_pixel_goal=None,
    )


def label_clip_frames(
    clip_dir: Path,
    *,
    img_size: int | tuple[int, int] = 256,
    min_goal_len: int = 3,
    max_side_dist_m: float = 6.0,
    depth_tolerance: float = 0.5,
    min_history: int = 5,
) -> dict[str, dict[str, Any]]:
    meta_path = clip_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    num_frames = int(meta["num_frames"])

    intrinsics = load_intrinsics(clip_dir)
    poses_by_view = {
        direction: load_poses_from_chunks(clip_dir, direction)
        for direction in PANO_HORIZONTAL_VIEWS
    }

    discrete_path = clip_dir / "discrete_actions.npy"
    raw_actions = np.load(discrete_path).astype(np.int32)
    actions = align_internnav_discrete_actions(raw_actions)

    frames: dict[str, dict[str, Any]] = {}
    for frame_id in range(num_frames):
        depth_front = load_depth_from_chunks(clip_dir, frame_id, "front")
        pano_goal = resolve_farthest_pano_pixel_goal(
            current_t=frame_id,
            num_frames=num_frames,
            poses_by_view=poses_by_view,
            depth_front=depth_front,
            img_size=img_size,
            intrinsics=intrinsics,
            min_goal_len=min_goal_len,
            max_side_dist_m=max_side_dist_m,
            depth_tolerance=depth_tolerance,
        )
        action = int(actions[frame_id]) if frame_id < len(actions) else 1
        label = classify_frame_label(
            frame_id=frame_id,
            num_frames=num_frames,
            discrete_action=action,
            pano_goal=pano_goal,
        )
        entry = label.to_dict()
        entry["discrete_action"] = action
        entry["eligible_sft"] = (
            frame_id >= min_history
            and label.sample_kind in {"pixel", "turn", "stop"}
        )
        if pano_goal is not None:
            _, canonical, _ = pano_goal
            entry["waypoint_dist_m"] = round(float(canonical.z_depth_m), 4)
        frames[str(frame_id)] = entry

    return frames


def write_clip_labels(clip_dir: Path, frames: dict[str, dict[str, Any]]) -> Path:
    out_path = clip_dir / "pano_view_labels.json"
    payload = {
        "version": LABEL_VERSION,
        "policy": "C3",
        "frames": frames,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def load_clip_labels(clip_dir: Path) -> dict[str, dict[str, Any]] | None:
    path = clip_dir / "pano_view_labels.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("frames", {})

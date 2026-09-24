"""EXP-19 geometry and ground truth on the rerun's own closed-loop states.

Everything that turns a recorded state into a camera pose, a label or an image
position goes through this module; build_records.py only joins and scores.

* Camera pose.  steps.jsonl stores the agent BODY state (position on the navmesh,
  rotation as explicit wxyz).  The eval camera sits at [0, 1.25, 0] in the body
  frame (r2r_val_unseen.py ``build_habitat_config``: RGB_SENSOR.POSITION) and the
  training collector's ``pose_front`` is the same body pose + 1.25 m up, so
  ``camera_c2w = body_c2w @ translate(0, 1.25, 0)`` (Habitat: y up, camera looks
  along -z, x right).
* History affordance-map ground truth = the training label call, not a re-derivation:
  ``src/data/sliding_window_dataset.py::_compute_per_history_multiview_heatmaps``
  calls ``src/data/heatmap_geometry.py::compute_history_heatmap`` once per
  (history slot, view) with one history pose, current pose = front c2w @ R_y(yaw)
  (F 0, R -90, B 180, L +90), depth = front depth in metres for F and None for
  R/B/L (the R2R v2 chunks have no side depth), 256 px HFOV 90 intrinsics
  fx = fy = cx = cy = 128, 64x64 maps, ``depth_normalize=False``.
* Future references use the future-label code path
  (``src/data/future_trajectory_heatmap.py``): System1's selected mean path
  [33, 2] (robot frame, x forward, y left, metres, row 0 = origin) becomes the
  native incremental target [32, 3] (dx, dy scaled by action_scale 4, dyaw 0),
  then ``action_deltas_to_camera_points`` + ``render_future_trajectory_heatmaps``
  (flat, no height).  The executed-path reference is the training recipe on the
  agent's actual later camera poses: ``get_trajectory_relative_to_frame`` +
  ``interpolate_and_resample_trajectory(32, 4.0)`` + ``build_future_target_from_system1_action``
  (height from the poses; no lookdown-visibility truncation, the lookdown depth is not re-rendered).
* Decision images.  The eval camera is 640x480, HFOV 79, square pixels
  (fx = 320 / tan 39.5 deg); the lookdown is the same camera pitched down 30 deg
  (2 x LOOK_DOWN, TILT_ANGLE 15; the sensor does not move).  The front image the
  model saw is that 640x480 frame resized to 384x384 (not aspect-preserving).
  Image positions are returned in pixel INDEX coordinates (pixel (col i, row j)
  is centred at (i, j), matplotlib ``imshow`` default), u = column, v = row.

The three label modules are loaded from ``src/data`` under a stand-in package,
so ``src/data/__init__.py`` (which imports torch and the datasets) never runs:
this module needs numpy (+ scipy for the executed-path spline, as training).
"""
from __future__ import annotations

import importlib
import math
import sys
import types
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.exp18 import geometry as geo  # noqa: E402


def _label_modules():
    """(heatmap_geometry, trajectory_utils, future_trajectory_heatmap) from src/data, torch-free."""
    name = "_exp19_label_code"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(SOURCE_ROOT / "src" / "data")]
        pkg.__package__ = name
        sys.modules[name] = pkg
    return tuple(importlib.import_module(f"{name}.{mod}")
                 for mod in ("heatmap_geometry", "trajectory_utils", "future_trajectory_heatmap"))


HG, TU, FTH = _label_modules()

SENSOR_HEIGHT_M = 1.25
NUM_SLOTS = 8
ACTION_SCALE = 4.0
LABEL_IMG_SIZE = (256, 256)  # (width, height), as compute_history_heatmap takes it
HM_SIZE = (64, 64)
K_LABEL = np.array([[128.0, 0.0, 128.0], [0.0, 128.0, 128.0], [0.0, 0.0, 1.0]], dtype=np.float32)
FUTURE_TIME_RANGES = FTH.FUTURE_TIME_RANGES  # waypoints 1-8, 9-16, 17-24, 25-32

EVAL_IMAGE_WH = (640, 480)
EVAL_HFOV_DEG = 79.0
LOOKDOWN_PITCH_DEG = 30.0
VLM_IMAGE_WH = (384, 384)
DECISION_IMAGES = {"lookdown": {"wh": EVAL_IMAGE_WH, "pitch_deg": LOOKDOWN_PITCH_DEG},
                   "front": {"wh": VLM_IMAGE_WH, "pitch_deg": 0.0}}


# --------------------------------------------------------------------------- #
# Poses
# --------------------------------------------------------------------------- #
def quat_wxyz_to_rot(q) -> np.ndarray:
    """3x3 rotation of a (w, x, y, z) quaternion (normalised first)."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(np.asarray(q, dtype=np.float64))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def body_c2w(position, rotation_wxyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_rot(rotation_wxyz)
    T[:3, 3] = np.asarray(position, dtype=np.float64)
    return T


def camera_c2w(position, rotation_wxyz, height_m: float = SENSOR_HEIGHT_M) -> np.ndarray:
    """Front camera c2w of a body state: body pose with the sensor offset [0, height, 0] in the body frame."""
    T = body_c2w(position, rotation_wxyz)
    T[:3, 3] = T[:3, 3] + T[:3, :3] @ np.array([0.0, height_m, 0.0])
    return T


# --------------------------------------------------------------------------- #
# History affordance-map ground truth
# --------------------------------------------------------------------------- #
def history_labels(history_c2w: Sequence[np.ndarray], current_c2w: np.ndarray,
                   depth_front: Optional[np.ndarray]):
    """GT maps [K, 4, 64, 64] f32 and visibility [K, 4] f32 (views F, R, B, L), label-code exact."""
    views = [geo.view_c2w(current_c2w, v).astype(np.float32) for v in range(4)]
    hm = np.zeros((len(history_c2w), 4) + HM_SIZE, dtype=np.float32)
    vis = np.zeros((len(history_c2w), 4), dtype=np.float32)
    for k, T_h in enumerate(history_c2w):
        for v in range(4):
            m, count = HG.compute_history_heatmap(
                history_poses=[np.asarray(T_h, dtype=np.float32)], current_pose=views[v],
                current_depth=depth_front if v == 0 else None, hm_size=HM_SIZE, img_size=LABEL_IMG_SIZE,
                K=K_LABEL, depth_normalize=False)
            hm[k, v] = m
            vis[k, v] = float(count > 0)
    return hm, vis


def pad_slots(maps: np.ndarray, vis: np.ndarray, num: int = NUM_SLOTS):
    """Pad K <= num history slots to num: (maps [num, 4, H, W], vis [num, 4], mask [num] bool)."""
    k = len(maps)
    if k > num:
        raise ValueError(f"{k} history slots > {num}")
    out_m = np.zeros((num,) + maps.shape[1:], dtype=np.float32)
    out_v = np.zeros((num, 4), dtype=np.float32)
    mask = np.zeros(num, dtype=bool)
    out_m[:k], out_v[:k], mask[:k] = maps, vis, True
    return out_m, out_v, mask


def gt_view_class_and_peak(maps: np.ndarray, vis: np.ndarray):
    """validate.py GT semantics per slot: class (0 none, 1 + view) and the (row, col) peak in that view.

    eligible view = vis > 0.5 and map peak > 0; GT view = first argmax of the peaks over
    eligible views; peak = first-occurrence argmax of that view's map; (-1, -1) when none.
    """
    maps = np.asarray(maps, dtype=np.float32)
    peak = maps.reshape(maps.shape[:-2] + (-1,)).max(-1)
    eligible = (np.asarray(vis) > 0.5) & (peak > 0)
    view = np.where(eligible, peak, -np.inf).argmax(-1)
    visible = eligible.any(-1)
    cls = np.where(visible, view + 1, 0).astype(np.int64)
    row, col = geo.argmax_pixel(np.take_along_axis(maps, view[..., None, None, None], axis=-3)[..., 0, :, :])
    return cls, np.where(visible, row, -1).astype(np.int64), np.where(visible, col, -1).astype(np.int64)


# --------------------------------------------------------------------------- #
# Future references (future-label code path)
# --------------------------------------------------------------------------- #
def path_xy_to_action_deltas(path_xy: np.ndarray, action_scale: float = ACTION_SCALE) -> np.ndarray:
    """Selected mean path [33, 2] (row 0 = origin) -> native incremental target [32, 3] (x4, dyaw 0)."""
    p = np.asarray(path_xy, dtype=np.float64)
    if p.shape != (33, 2):
        raise ValueError(f"selected path must be [33, 2], got {p.shape}")
    if np.abs(p[0]).max() > 1e-6:
        raise ValueError(f"selected path must start at the robot (row 0 = 0), got {p[0].tolist()}")
    out = np.zeros((32, 3), dtype=np.float32)
    out[:, :2] = np.diff(p, axis=0) * action_scale
    return out


def future_reference_from_path(path_xy: np.ndarray):
    """4 bins x 4 views target of System1's selected path (flat), FutureTrajectoryHeatmapTarget."""
    points = FTH.action_deltas_to_camera_points(path_xy_to_action_deltas(path_xy), action_scale=ACTION_SCALE)
    return FTH.render_future_trajectory_heatmaps(points, intrinsics=K_LABEL, image_size=LABEL_IMG_SIZE,
                                                 heatmap_size=HM_SIZE)


def future_reference_from_poses(current_c2w: np.ndarray, future_c2w: Sequence[np.ndarray]):
    """Same target from the camera poses the agent actually reached after this step (training recipe).

    Returns None when there is no later pose (last recorded state).
    """
    if len(future_c2w) == 0:
        return None
    P = np.concatenate([np.asarray(current_c2w, dtype=np.float32)[None],
                        np.asarray(future_c2w, dtype=np.float32).reshape(-1, 4, 4)], axis=0)
    rel = TU.get_trajectory_relative_to_frame(P, camera_deg=0, camera_forward_axis="-z")
    _, traj32 = TU.interpolate_and_resample_trajectory(rel, predict_step_num=32, action_scale=ACTION_SCALE)
    return FTH.build_future_target_from_system1_action(
        traj32, action_scale=ACTION_SCALE, current_camera_c2w=P[0], expert_future_camera_c2w=P,
        intrinsics=K_LABEL, image_size=LABEL_IMG_SIZE, heatmap_size=HM_SIZE)


def predicted_view5(visibility_probability: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """[4 bins, 4 views] probabilities -> view5 per bin: 1 + argmax view, 0 (none) when the max < threshold."""
    p = np.asarray(visibility_probability, dtype=np.float64).reshape(4, 4)
    return np.where(p.max(-1) >= threshold, p.argmax(-1) + 1, 0).astype(np.int64)


# --------------------------------------------------------------------------- #
# System1 path in the world and in the decision image
# --------------------------------------------------------------------------- #
def path_camera_points(path_xy: np.ndarray, pitch_deg: float = 0.0,
                       height_m: float = SENSOR_HEIGHT_M) -> np.ndarray:
    """Floor points of a robot-frame path [N, 2] (x forward, y left) in Habitat camera coords [N, 3].

    Camera at ``height_m`` above the robot, pitched DOWN by ``pitch_deg`` (its c2w
    rotation in the body frame is R_x(-pitch)); pitch 0 = the current front camera.
    """
    p = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    body = np.stack([-p[:, 1], np.full(len(p), -height_m), -p[:, 0]], axis=-1)  # relative to the camera centre
    a = math.radians(pitch_deg)
    rot = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])  # R_x(+pitch)
    return body @ rot.T


def eval_intrinsics(width: int = EVAL_IMAGE_WH[0], height: int = EVAL_IMAGE_WH[1],
                    hfov_deg: float = EVAL_HFOV_DEG):
    """(fx, fy, cx, cy) of the eval camera in continuous pixel units (pixel p covers [p, p + 1))."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return fx, fx, width / 2.0, height / 2.0


def project_to_decision_image(camera_points: np.ndarray, image_wh) -> np.ndarray:
    """Project camera points into a decision image of size ``image_wh`` rendered from the eval camera.

    The pinhole is the 640x480 HFOV-79 eval camera; the result is rescaled to
    ``image_wh`` (384x384 for the front the model saw) and returned in pixel
    index coordinates [N, 2] (u, v).  Points behind the camera (z >= -0.1, the
    label code's test) are NaN; points outside the frame are kept.
    """
    fx, fy, cx, cy = eval_intrinsics()
    pts = np.asarray(camera_points, dtype=np.float64).reshape(-1, 3)
    depth = -pts[:, 2]
    ok = pts[:, 2] < -0.1
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * pts[:, 0] / depth + cx
        v = fy * (-pts[:, 1]) / depth + cy
    sx, sy = image_wh[0] / EVAL_IMAGE_WH[0], image_wh[1] / EVAL_IMAGE_WH[1]
    uv = np.stack([u * sx - 0.5, v * sy - 0.5], axis=-1)
    uv[~ok] = np.nan
    return uv


def path_world_xz(path_xy: np.ndarray, cam_c2w: np.ndarray, height_m: float = SENSOR_HEIGHT_M) -> np.ndarray:
    """World floor (x, z) [N, 2] of a robot-frame path, from the front camera c2w at the call."""
    p = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    world = geo.rel_to_world(cam_c2w, p[:, 0], p[:, 1], -height_m)
    return world[:, [0, 2]]


# --------------------------------------------------------------------------- #
# System2 pixel goal: which response field is the column
# --------------------------------------------------------------------------- #
PIXEL_GOAL_CONVENTIONS = {
    "field_uv": "response pixel_goal = [u (column), v (row)], i.e. the System2 text is 'row col' "
                "(rpc_model_server._parse_internnav_pixel_goal docstring)",
    "field_vu": "response pixel_goal = [v (row), u (column)], i.e. the System2 text is 'col row'",
}
# The statistics must not depend on the lateral base rate.  A raw sign agreement does: on the
# 427 seed-42 log trajectory calls of the 15 candidates (text + System1 endpoint, lookdown assumed),
# 90% of the informative endpoints are to the left and the rows (78..455) mostly sit left of the
# 640 px centre, so the WRONG convention agreed 0.876 (right one 1.000) and a 0.25 margin failed.
# Balanced agreement (1.000 vs 0.524) and the point-biserial correlation (0.825 vs 0.122) separate.
CONVENTION_RULE = {"min_lateral_m": 0.25, "min_calls_per_side": 5, "min_balanced_agreement": 0.75,
                   "min_sign_correlation": 0.5, "min_correlation_margin": 0.3}


class PixelGoalConventionError(RuntimeError):
    def __init__(self, message: str, evidence: dict):
        super().__init__(message)
        self.evidence = evidence


def pixel_goal_uv(pixel_goal, convention: str):
    """(u, v) = (column, row) of a response ``pixel_goal`` under a convention."""
    a, b = (float(x) for x in pixel_goal[:2])
    if convention == "field_uv":
        return a, b
    if convention == "field_vu":
        return b, a
    raise ValueError(f"unknown pixel-goal convention {convention!r}")


def _corr(x: np.ndarray, y: np.ndarray):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else None


def pixel_goal_evidence(samples: Sequence[dict], rule: dict = CONVENTION_RULE) -> dict:
    """Score both conventions on ready calls.

    samples: {"pixel_goal": [a, b] (response field), "image_wh": (W, H) of the decision
    image, "lateral_m": System1 selected-path endpoint y (left positive)}.
    A floor point to the left projects left of the image centre for any camera pitch,
    so the column must fall left of centre (u < (W - 1) / 2) exactly when lateral > 0.
    Informative calls: |lateral| >= min_lateral_m.  Per convention:
      range_violations         u or v outside the decision image (hard evidence)
      sign_correlation         point-biserial correlation of the column's left-of-centre
                               offset (centre - u) / W with sign(lateral), informative calls
      balanced_sign_agreement  mean of the sign agreement on left and on right endpoints
      sign_agreement           plain agreement (descriptive: depends on the left/right mix)
      pearson_left_offset_vs_lateral  offset vs lateral in metres, all calls (descriptive)
    """
    out = {"n_calls": len(samples), "rule": dict(rule), "conventions": {}}
    lat = np.asarray([s["lateral_m"] for s in samples], dtype=np.float64)
    informative = np.abs(lat) >= rule["min_lateral_m"]
    left, right = informative & (lat > 0), informative & (lat < 0)
    out.update(n_informative=int(informative.sum()), n_left=int(left.sum()), n_right=int(right.sum()))
    for name in PIXEL_GOAL_CONVENTIONS:
        uv = np.asarray([pixel_goal_uv(s["pixel_goal"], name) for s in samples], dtype=np.float64).reshape(-1, 2)
        wh = np.asarray([s["image_wh"] for s in samples], dtype=np.float64).reshape(-1, 2)
        bad = (uv[:, 0] < 0) | (uv[:, 0] >= wh[:, 0]) | (uv[:, 1] < 0) | (uv[:, 1] >= wh[:, 1])
        offset = ((wh[:, 0] - 1) / 2.0 - uv[:, 0]) / wh[:, 0]
        agree = np.sign(offset) == np.sign(lat)
        out["conventions"][name] = {
            "meaning": PIXEL_GOAL_CONVENTIONS[name],
            "range_violations": int(bad.sum()),
            "sign_correlation": _corr(offset[informative], np.sign(lat[informative])),
            "balanced_sign_agreement": (float((agree[left].mean() + agree[right].mean()) / 2)
                                        if left.any() and right.any() else None),
            "sign_agreement": float(agree[informative].mean()) if informative.any() else None,
            "pearson_left_offset_vs_lateral": _corr(offset, lat),
        }
    return out


def resolve_pixel_goal_convention(samples: Sequence[dict], rule: dict = CONVENTION_RULE) -> dict:
    """Pick the convention the data supports, or raise PixelGoalConventionError (ambiguous).

    Accepted iff: >= min_calls_per_side informative calls on each side; the convention with
    the larger sign correlation reaches min_sign_correlation, beats the other by
    min_correlation_margin, has balanced sign agreement >= min_balanced_agreement and no
    range violations.
    """
    ev = pixel_goal_evidence(samples, rule)
    conv = ev["conventions"]
    score = {k: (c["sign_correlation"] if c["sign_correlation"] is not None else -1.0) for k, c in conv.items()}
    best = max(score, key=lambda k: (score[k], -conv[k]["range_violations"]))
    other = [k for k in score if k != best][0]
    reasons = []
    for side in ("left", "right"):
        if ev[f"n_{side}"] < rule["min_calls_per_side"]:
            reasons.append(f"only {ev[f'n_{side}']} calls with the endpoint >= {rule['min_lateral_m']} m to the "
                           f"{side} (< {rule['min_calls_per_side']})")
    if score[best] < rule["min_sign_correlation"]:
        reasons.append(f"best sign correlation {score[best]:.3f} < {rule['min_sign_correlation']}")
    if score[best] - score[other] < rule["min_correlation_margin"]:
        reasons.append(f"sign-correlation margin {score[best] - score[other]:.3f} < {rule['min_correlation_margin']}")
    balanced = conv[best]["balanced_sign_agreement"]
    if balanced is None or balanced < rule["min_balanced_agreement"]:
        reasons.append(f"{best} balanced sign agreement {balanced} < {rule['min_balanced_agreement']}")
    if conv[best]["range_violations"]:
        reasons.append(f"{best} puts {conv[best]['range_violations']} pixel goals outside the image")
    ev.update(convention=best if not reasons else None, forced=False, ambiguous_reasons=reasons)
    if reasons:
        raise PixelGoalConventionError("pixel-goal convention is ambiguous: " + "; ".join(reasons), ev)
    return ev

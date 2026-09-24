"""EXP-19 ground truth and geometry (scripts/exp19/gt.py) against hand-computed cases and EXP-18 conventions.

numpy (+ scipy for the executed-path spline, PIL for topdown_io); no torch: gt.py loads the
label modules from src/data without running src/data/__init__.py.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.exp18 import geometry as geo
from scripts.exp19 import gt


def quat_yaw(deg: float) -> list:
    """Habitat yaw quaternion (w, x, y, z) about +y, left-positive."""
    h = math.radians(deg) / 2
    return [math.cos(h), 0.0, math.sin(h), 0.0]


def quat_axis_angle(axis, deg):
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    h = math.radians(deg) / 2
    return [math.cos(h), *(math.sin(h) * axis)]


def rodrigues(axis, deg):
    k = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    a = math.radians(deg)
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


# --------------------------------------------------------------------------- #
# Poses
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("yaw", [0.0, 37.0, -90.0, 135.0, 180.0])
def test_camera_c2w_from_state_matches_exp18_yaw_conventions(yaw):
    pos = [1.5, 0.12, -4.0]
    T = gt.camera_c2w(pos, quat_yaw(yaw))
    np.testing.assert_allclose(T[:3, :3], geo.yaw_rotation(yaw)[:3, :3], atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [1.5, 1.37, -4.0], atol=1e-12)
    assert geo.circular_abs_diff(geo.c2w_yaw_deg(T), yaw) < 1e-9
    # forward of a left-turned camera points to world -x for yaw +90 (habitat: 0 faces -z)
    fwd = geo.c2w_forward(T)
    np.testing.assert_allclose(fwd, [-math.sin(math.radians(yaw)), 0.0, -math.cos(math.radians(yaw))], atol=1e-12)
    from scripts.exp18.topdown import topdown_io
    assert geo.circular_abs_diff(math.degrees(float(topdown_io.yaw_from_quaternion(quat_yaw(yaw)))), yaw) < 1e-9


def test_quaternion_rotation_is_hamilton_wxyz():
    for axis, deg in (([0, 1, 0], 30.0), ([1, 0, 0], -25.0), ([1, 2, 3], 71.0)):
        R = gt.quat_wxyz_to_rot(quat_axis_angle(axis, deg))
        np.testing.assert_allclose(R, rodrigues(axis, deg), atol=1e-12)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    # an unnormalised quaternion is normalised first; the sensor offset follows the body's up axis
    T = gt.camera_c2w([0, 0, 0], np.array(quat_axis_angle([1, 0, 0], 90.0)) * 3.0)
    np.testing.assert_allclose(T[:3, 3], [0.0, 0.0, 1.25], atol=1e-12)


# --------------------------------------------------------------------------- #
# History labels: the label call on a two-pose scene, checked by hand
# --------------------------------------------------------------------------- #
def gaussian(center_uv, sigma, peak):
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    g = peak * np.exp(-((xx - center_uv[0]) ** 2 + (yy - center_uv[1]) ** 2) / (2 * sigma ** 2))
    r = max(1, math.ceil(3 * sigma))  # draw_gaussian_point's window
    x0, x1 = max(0, math.floor(center_uv[0] - r)), min(64, math.ceil(center_uv[0] + r))
    y0, y1 = max(0, math.floor(center_uv[1] - r)), min(64, math.ceil(center_uv[1] + r))
    out = np.zeros((64, 64), np.float32)
    out[y0:y1, x0:x1] = g[y0:y1, x0:x1]
    return out


def test_front_label_reproduces_a_hand_computed_pixel():
    # current body at the origin facing -z; history body 1 m right, 3 m ahead (both 1.25 m cameras)
    cur = gt.camera_c2w([0, 0, 0], quat_yaw(0))
    hist = gt.camera_c2w([1.0, 0, -3.0], quat_yaw(50))  # history heading is irrelevant
    maps, vis = gt.history_labels([hist], cur, depth_front=None)
    assert maps.shape == (1, 4, 64, 64) and vis.tolist() == [[1.0, 0.0, 0.0, 0.0]]
    # pinhole: u = 128 * 1 / 3 + 128 = 170.667 px -> 42.667 in the 64 grid, v = 128 -> 32
    # sigma = clip(1.5 * 128 / 3 * 64 / 256 / 3, 4, 8) = 5.333; peak = 0.7 + 0.3 / (1 + sqrt(10) / 5)
    d = math.sqrt(10.0)
    expected = gaussian((42.0 + 2.0 / 3, 32.0), 16.0 / 3, 0.7 + 0.3 / (1 + d / 5))
    np.testing.assert_allclose(maps[0, 0], expected, atol=1e-5)
    assert geo.argmax_pixel(maps[0, 0]) == (32, 43)
    assert maps[0, 1:].max() == 0.0


@pytest.mark.parametrize("hist_pos,view", [([3.0, 0, 0], geo.RIGHT), ([0, 0, 3.0], geo.BACK), ([-3.0, 0, 0], geo.LEFT)])
def test_side_views_are_fov_only_and_centred(hist_pos, view):
    cur = gt.camera_c2w([0, 0, 0], quat_yaw(0))
    maps, vis = gt.history_labels([gt.camera_c2w(hist_pos, quat_yaw(0))], cur,
                                  depth_front=np.full((256, 256), 0.5, np.float16))  # front depth ignored off-front
    assert vis[0].tolist() == [float(v == view) for v in range(4)]
    assert geo.argmax_pixel(maps[0, view]) == (32, 32)


def test_label_in_a_rotated_two_pose_scene_matches_world_to_rel():
    cur = gt.camera_c2w([1.2, 0.3, -3.4], quat_yaw(37.0))
    world = geo.rel_to_world(cur, 4.0, 1.1, 0.0)  # 4 m ahead, 1.1 m left, same height
    hist = gt.camera_c2w(world - np.array([0, 1.25, 0]), quat_yaw(-120.0))
    maps, vis = gt.history_labels([hist], cur, depth_front=None)
    f, l, up = geo.world_to_rel(cur, geo.c2w_position(hist))
    assert abs(f - 4.0) < 1e-9 and abs(l - 1.1) < 1e-9 and abs(up) < 1e-9
    u = (128.0 * (-l) / f + 128.0) * 64 / 256
    assert vis[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert geo.argmax_pixel(maps[0, 0]) == (32, int(round(u)))


def test_front_depth_occludes_and_distance_gates():
    cur = gt.camera_c2w([0, 0, 0], quat_yaw(0))
    ahead = gt.camera_c2w([0, 0, -3.0], quat_yaw(0))
    _, vis = gt.history_labels([ahead], cur, depth_front=np.full((256, 256), 2.0, np.float16))
    assert vis.tolist() == [[0.0, 0.0, 0.0, 0.0]]  # 2.0 < 3.0 - 0.5
    _, vis = gt.history_labels([ahead], cur, depth_front=np.full((256, 256), 2.6, np.float16))
    assert vis.tolist() == [[1.0, 0.0, 0.0, 0.0]]
    _, vis = gt.history_labels([ahead], cur, depth_front=np.zeros((256, 256), np.float16))  # 0 = no geometry
    assert vis.tolist() == [[0.0, 0.0, 0.0, 0.0]]
    far = gt.camera_c2w([0, 0, -16.0], quat_yaw(0))
    same = gt.camera_c2w([0, 0, 0], quat_yaw(90))
    maps, vis = gt.history_labels([far, same], cur, depth_front=None)
    assert vis.sum() == 0 and maps.max() == 0.0


def test_pad_slots_and_validator_gt_semantics():
    maps = np.zeros((3, 4, 64, 64), np.float32)
    vis = np.zeros((3, 4), np.float32)
    maps[0, 1, 10, 20], maps[0, 3, 5, 6] = 0.8, 0.9  # two eligible views: larger peak (left) wins
    vis[0, 1] = vis[0, 3] = 1.0
    maps[1, 0, 30, 30] = 0.9  # map without the visibility flag: not eligible
    vis[2, 2] = 1.0  # flag without a map: not eligible
    pm, pv, mask = gt.pad_slots(maps, vis)
    assert pm.shape == (8, 4, 64, 64) and pv.shape == (8, 4) and mask.tolist() == [True] * 3 + [False] * 5
    cls, row, col = gt.gt_view_class_and_peak(pm, pv)
    assert cls.tolist() == [1 + geo.LEFT, 0, 0, 0, 0, 0, 0, 0]
    assert (row[0], col[0]) == (5, 6) and row[1:].tolist() == [-1] * 7
    with pytest.raises(ValueError):
        gt.pad_slots(np.zeros((9, 4, 64, 64), np.float32), np.zeros((9, 4), np.float32))


# --------------------------------------------------------------------------- #
# Future references
# --------------------------------------------------------------------------- #
def arc_path(turn_deg: float, straight_m: float = 0.5, total_m: float = 3.2) -> np.ndarray:
    """33-point robot-frame path: straight, then a sharp turn, then straight (x forward, y left)."""
    s = np.linspace(0.0, total_m, 33)
    h = math.radians(turn_deg)
    x = np.where(s <= straight_m, s, straight_m + (s - straight_m) * math.cos(h))
    y = np.where(s <= straight_m, 0.0, (s - straight_m) * math.sin(h))
    return np.stack([x, y], axis=-1)


def test_path_to_action_deltas_round_trips_through_the_label_code():
    path = arc_path(50.0)
    deltas = gt.path_xy_to_action_deltas(path)
    assert deltas.shape == (32, 3) and np.all(deltas[:, 2] == 0)
    pts = gt.FTH.action_deltas_to_camera_points(deltas, action_scale=gt.ACTION_SCALE)
    np.testing.assert_allclose(pts, np.stack([-path[1:, 1], np.zeros(32), -path[1:, 0]], -1), atol=1e-5)
    with pytest.raises(ValueError):
        gt.path_xy_to_action_deltas(path + 0.1)  # must start at the robot


@pytest.mark.parametrize("turn,late_view", [(0.0, 1), (80.0, 1 + geo.LEFT), (-80.0, 1 + geo.RIGHT)])
def test_system1_reference_view_per_bin(turn, late_view):
    ref = gt.future_reference_from_path(arc_path(turn))
    assert ref.view5[0] == 1  # waypoint 8 (0.8 m) is still straight ahead
    assert ref.view5[3] == late_view
    assert ref.heatmap.shape == (4, 4, 64, 64)


def test_turn_in_place_path_has_no_reference():
    ref = gt.future_reference_from_path(np.zeros((33, 2)))
    assert ref.view5.tolist() == [0, 0, 0, 0]


def test_executed_reference_from_later_poses():
    cur = gt.camera_c2w([0, 0, 0], quat_yaw(20.0))
    fwd = geo.c2w_forward(cur)
    straight = [gt.camera_c2w(np.asarray([0, 0, 0]) + 0.25 * i * fwd, quat_yaw(20.0)) for i in range(1, 25)]
    ref = gt.future_reference_from_poses(cur, straight)
    assert ref.view5.tolist() == [1, 1, 1, 1]
    # 1 m ahead, then the agent walks to its right: later bins land in the right view
    right = -geo.c2w_left(cur)
    turn = [gt.camera_c2w(fwd * min(i, 4) * 0.25 + right * max(i - 4, 0) * 0.25, quat_yaw(20.0)) for i in range(1, 25)]
    ref = gt.future_reference_from_poses(cur, turn)
    assert ref.view5[0] == 1 and ref.view5[3] == 1 + geo.RIGHT
    assert gt.future_reference_from_poses(cur, []) is None


def test_predicted_view5_threshold():
    p = np.array([[0.9, 0.1, 0.0, 0.2], [0.2, 0.3, 0.49, 0.1], [0.1, 0.5, 0.2, 0.2], [0.0, 0.0, 0.0, 0.7]])
    assert gt.predicted_view5(p).tolist() == [1, 0, 2, 4]


# --------------------------------------------------------------------------- #
# Decision-image projection of the System1 path
# --------------------------------------------------------------------------- #
def test_eval_intrinsics_hfov79():
    fx, fy, cx, cy = gt.eval_intrinsics()
    assert abs(fx - 320.0 / math.tan(math.radians(39.5))) < 1e-9 and fx == fy
    assert (cx, cy) == (320.0, 240.0)


def test_lookdown_optical_axis_hits_the_floor_at_the_image_centre():
    d = 1.25 / math.tan(math.radians(30.0))
    uv = gt.project_to_decision_image(gt.path_camera_points([[d, 0.0]], pitch_deg=30.0), (640, 480))
    np.testing.assert_allclose(uv[0], [319.5, 239.5], atol=1e-9)


def test_front_projection_scales_to_the_384_image_the_model_saw():
    pts = gt.path_camera_points([[2.0, 0.5]], pitch_deg=0.0)
    np.testing.assert_allclose(pts[0], [-0.5, -1.25, -2.0])
    fx = gt.eval_intrinsics()[0]
    u640, v480 = fx * -0.5 / 2.0 + 320.0, fx * 1.25 / 2.0 + 240.0
    uv = gt.project_to_decision_image(pts, (384, 384))
    np.testing.assert_allclose(uv[0], [u640 * 0.6 - 0.5, v480 * 0.8 - 0.5], atol=1e-9)


@pytest.mark.parametrize("pitch,wh", [(0.0, (384, 384)), (30.0, (640, 480))])
def test_left_is_left_and_behind_is_dropped(pitch, wh):
    uv = gt.project_to_decision_image(gt.path_camera_points([[3.0, 1.0], [3.0, -1.0], [-1.0, 0.0], [0.0, 0.0]], pitch), wh)
    centre = (wh[0] - 1) / 2
    assert uv[0, 0] < centre < uv[1, 0]
    assert np.isnan(uv[2]).all()  # behind the camera
    assert np.isnan(uv[3]).all() == (pitch == 0.0)  # the robot's own floor point: behind a level camera only


def test_path_world_xz_matches_exp18_rel_to_world():
    cam = gt.camera_c2w([2.0, 0.4, 1.0], quat_yaw(-63.0))
    path = arc_path(40.0)
    xz = gt.path_world_xz(path, cam)
    f, l, up = geo.world_to_rel(cam, np.stack([xz[:, 0], np.full(33, 0.4), xz[:, 1]], -1))
    np.testing.assert_allclose(f, path[:, 0], atol=1e-9)
    np.testing.assert_allclose(l, path[:, 1], atol=1e-9)
    np.testing.assert_allclose(up, -1.25, atol=1e-9)
    # the camera-frame points are the same points
    np.testing.assert_allclose(gt.path_camera_points(path), np.stack([-l, up, -f], -1), atol=1e-9)


# --------------------------------------------------------------------------- #
# Pixel-goal convention
# --------------------------------------------------------------------------- #
def samples(n, convention, rng, lookdown=True):
    out = []
    for _ in range(n):
        lat = rng.uniform(-1.5, 1.5)
        wh = (640, 480) if lookdown else (384, 384)
        u = int(np.clip((wh[0] - 1) / 2 - lat * 150 + rng.normal(0, 20), 0, wh[0] - 1))
        v = int(rng.integers(150, wh[1]))
        out.append({"pixel_goal": [u, v] if convention == "field_uv" else [v, u], "image_wh": wh, "lateral_m": lat})
    return out


@pytest.mark.parametrize("convention", ["field_uv", "field_vu"])
def test_convention_is_resolved_from_the_lateral_sign(convention):
    rng = np.random.default_rng(1)
    ev = gt.resolve_pixel_goal_convention(samples(40, convention, rng) + samples(20, convention, rng, lookdown=False))
    assert ev["convention"] == convention and not ev["forced"]
    assert ev["conventions"][convention]["sign_agreement"] > 0.9
    assert ev["conventions"][convention]["range_violations"] == 0
    assert gt.pixel_goal_uv([10, 20], "field_uv") == (10.0, 20.0)
    assert gt.pixel_goal_uv([10, 20], "field_vu") == (20.0, 10.0)


def test_convention_is_not_fooled_by_a_one_sided_lateral_mix():
    # Like the seed-42 log calls: ~90% of endpoints to the left, rows mostly above the image centre, so
    # reading the row as the column also lands "left of centre" most of the time.
    rng = np.random.default_rng(3)
    out = []
    for i in range(200):
        lat = rng.uniform(0.3, 1.5) if i % 10 else -rng.uniform(0.3, 1.5)
        u = int(np.clip(319.5 - lat * 100 + rng.normal(0, 20), 0, 470))  # < 480: no range violation either way
        v = int(rng.integers(80, 330))
        out.append({"pixel_goal": [v, u], "image_wh": (640, 480), "lateral_m": lat})
    ev = gt.pixel_goal_evidence(out)
    wrong, right = ev["conventions"]["field_uv"], ev["conventions"]["field_vu"]
    assert wrong["sign_agreement"] > 0.75 and wrong["range_violations"] == 0  # a plain agreement would be fooled
    assert wrong["balanced_sign_agreement"] < 0.65 and right["balanced_sign_agreement"] > 0.95
    assert gt.resolve_pixel_goal_convention(out)["convention"] == "field_vu"


def test_convention_fails_loudly_when_ambiguous():
    rng = np.random.default_rng(2)
    noise = [{"pixel_goal": [int(rng.integers(0, 480)), int(rng.integers(0, 480))], "image_wh": (640, 480),
              "lateral_m": rng.uniform(-1.5, 1.5)} for _ in range(60)]
    with pytest.raises(gt.PixelGoalConventionError) as err:
        gt.resolve_pixel_goal_convention(noise)
    assert err.value.evidence["convention"] is None and err.value.evidence["ambiguous_reasons"]
    with pytest.raises(gt.PixelGoalConventionError, match=r"only \d calls"):
        gt.resolve_pixel_goal_convention(samples(5, "field_vu", rng))
    # clean evidence, but every endpoint on one side: the column test has no contrast
    one_sided = [s for s in samples(80, "field_vu", rng) if s["lateral_m"] > -0.25]
    with pytest.raises(gt.PixelGoalConventionError, match="to the right"):
        gt.resolve_pixel_goal_convention(one_sided)

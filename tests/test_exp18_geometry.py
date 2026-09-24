"""EXP-18 geometry conventions checked against the real label code (numpy only, no torch).

The labels come from ``src/data/heatmap_geometry.py::compute_history_heatmap``
(loaded by file path so no torch-importing package ``__init__`` runs), called
once per view exactly as ``_compute_per_history_multiview_heatmaps`` does:
current pose = that view's c2w, depth None (the R2R v2 surround views have no
depth), 256 px intrinsics fx = fy = cx = cy = 128, 64x64 maps.

Set ``EXP18_TEST_DUMP=/path/to/clip.npz`` to also check a real EXP-18 dump.
The last three tests cover the E out-and-back turnaround, the H2 attribution
table and the route-pattern key rows (the last two need pandas).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.exp18 import geometry as geo

ROOT = Path(__file__).resolve().parents[1]
K_IMG = np.array([[128.0, 0.0, 128.0], [0.0, 128.0, 128.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _load_by_path(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HG = _load_by_path("_exp18_heatmap_geometry", "src/data/heatmap_geometry.py")
TU = _load_by_path("_exp18_trajectory_utils", "src/data/trajectory_utils.py")


def _pose(yaw_deg: float, position) -> np.ndarray:
    T = geo.yaw_rotation(yaw_deg)
    T[:3, 3] = position
    return T


CURRENT = _pose(37.0, [1.2, 0.5, -3.4])


def _history_at(bearing: float, distance: float, up: float = 0.0, yaw: float = 80.0) -> np.ndarray:
    b = np.radians(bearing)
    return _pose(yaw, geo.rel_to_world(CURRENT, distance * np.cos(b), distance * np.sin(b), up))


def _labels(current: np.ndarray, history: np.ndarray):
    """[4,64,64] maps and [4] visibility, the per-view loop of the dataset label builder."""
    maps, vis = [], []
    for view in range(4):
        hm, count = HG.compute_history_heatmap(
            history_poses=[history.astype(np.float32)],
            current_pose=geo.view_c2w(current, view).astype(np.float32),
            current_depth=None, hm_size=(64, 64), img_size=(256, 256), K=K_IMG)
        maps.append(hm)
        vis.append(count > 0)
    return np.stack(maps), np.asarray(vis)


BOUNDARY = [44.9, 45.0, 45.1, -44.9, -45.0, -45.1, 134.9, 135.1, -134.9, -135.1, 179.9, -179.9, 180.0]
BEARINGS = [0.0, 30.0, -30.0, 90.0, -90.0, 60.0, -60.0, 120.0, -120.0, 150.0, -150.0, 10.0, -170.0] + BOUNDARY


def test_label_grid_index_i_is_continuous_coordinate_i():
    # draw_gaussian_point builds its grid with np.arange: index i sits at coordinate i.
    for centre in (10.0, 10.49, 9.51):
        hm = np.zeros((64, 64), dtype=np.float32)
        HG.draw_gaussian_point(hm, (centre, 20.0), 4.0)
        assert geo.argmax_pixel(hm) == (20, 10)
    maps, vis = _labels(CURRENT, _history_at(0.0, 3.0))
    assert vis.tolist() == [True, False, False, False]
    assert geo.argmax_pixel(maps[0]) == (32, 32)
    assert geo.pixel_to_bearing_elev(geo.FRONT, 32, 32) == (0.0, 0.0)


@pytest.mark.parametrize("bearing", BEARINGS)
@pytest.mark.parametrize("distance,up", [(1.0, 0.0), (3.0, 0.3), (7.0, -0.5), (12.0, 0.0)])
def test_argmax_bearing_of_real_labels_matches_rel_pose_bearing(bearing, distance, up):
    history = _history_at(bearing, distance, up)
    maps, vis = _labels(CURRENT, history)
    if not vis.any():
        pytest.skip("outside every view's vertical field of view")
    rel = TU.compute_history_rel_poses([history.astype(np.float32)], CURRENT.astype(np.float32),
                                       camera_forward_axis="-z")[0]
    gt_bearing = geo.bearing_from_rel_pose(rel[0], rel[1])
    assert geo.circular_abs_diff(gt_bearing, bearing) < 1e-3
    pred, view, row, col = geo.argmax_bearing(maps, return_pixel=True)
    # Half a heatmap pixel is at most atan(0.5/32) = 0.9 deg.
    assert geo.circular_abs_diff(pred, gt_bearing) <= 0.9
    # The peak is the grid index nearest the continuous projection (i <-> i);
    # a projection in (63, 64) can only peak at the last index, 63.
    fwd, left, height = geo.world_to_rel(CURRENT, geo.c2w_position(history))
    elev = np.degrees(np.arctan2(height, np.hypot(fwd, left)))
    _, u, v = geo.bearing_elev_to_pixel(gt_bearing, elev, int(view))
    assert abs(col - min(u, 63.0)) <= 0.5 + 1e-3 and abs(row - min(v, 63.0)) <= 0.5 + 1e-3
    pred_elev = geo.pixel_to_bearing_elev(view, col, row)[1]
    assert abs(pred_elev - elev) <= 0.9
    if up == 0.0 and bearing not in (45.0, -45.0, 135.0, -135.0):
        assert vis.sum() == 1 and int(view) == geo.view_for_bearing(gt_bearing)


def test_view_sectors_and_constant_floor():
    assert [geo.view_for_bearing(b) for b in (0, 45, 45.01, 135, 135.01, 180, -180, -135, -134.99, -45, -44.99)] == \
        [0, 0, 3, 3, 2, 2, 2, 2, 1, 1, 0]
    assert geo.pixel_to_bearing_elev(geo.BACK, 32, 32) == (180.0, 0.0)
    floor = np.zeros((4, 64, 64), dtype=np.float32)
    floor[(geo.BACK,) + geo.FLOOR_PEAK_YX] = 1.0
    assert geo.argmax_bearing(floor) == 180.0
    assert geo.circular_abs_diff(180.0, -170.0) == pytest.approx(10.0)


def test_pixel_direction_round_trip():
    rng = np.random.default_rng(0)
    bearing = rng.uniform(-180, 180, 500)
    elev = rng.uniform(-30, 30, 500)
    for kwargs in ({}, {"fx": 128.0, "cx": 128.0, "pixel_offset": geo.RGB_PIXEL_OFFSET}):
        view, u, v = geo.bearing_elev_to_pixel(bearing, elev, **kwargs)
        b2, e2 = geo.pixel_to_bearing_elev(view, u, v, **kwargs)
        assert np.max(geo.circular_abs_diff(b2, bearing)) < 1e-9
        assert np.max(np.abs(e2 - elev)) < 1e-9


def test_circular_helpers():
    assert geo.wrap_deg(-180.0) == 180.0 and geo.wrap_deg(270.0) == -90.0
    assert geo.circular_range_deg([170.0, -170.0]) == pytest.approx(20.0)
    assert geo.circular_range_deg([0.0, 90.0, 180.0]) == pytest.approx(180.0)
    assert geo.circular_range_deg([10.0, np.nan]) == 0.0
    assert geo.bearing_from_rel_pose(0.0, 1.0) == 90.0 and geo.bearing_from_rel_pose(-1.0, 0.0) == 180.0


def test_ring_column_mapping():
    az = geo.ring_column_azimuths(720)
    assert az[0] == 179.75 and az[-1] == -179.75 and az[359] == 0.25 and az[360] == -0.25
    assert np.allclose(geo.azimuth_to_ring_x(az, 720), np.arange(720))
    assert geo.azimuth_to_ring_x(0.0, 720) == 359.5
    assert geo.ring_extent() == (180.0, -180.0, -45.0, 45.0)


@pytest.mark.parametrize("bearing", [0.0, 23.0, -67.0, 91.0, -135.2, 179.0, 44.95, -45.0, 135.0])
def test_stitch_ring_places_real_label_blob_at_its_bearing(bearing):
    maps, vis = _labels(CURRENT, _history_at(bearing, 4.0))
    assert vis.any()
    ring, valid = geo.stitch_ring(maps, width=720)
    assert ring.shape == (180, 720) and valid[90].all()
    row, col = np.unravel_index(np.nanargmax(ring), ring.shape)
    # half a label pixel (0.9 deg) + half a ring column (0.25 deg)
    assert geo.circular_abs_diff(geo.ring_column_azimuths(720)[col], bearing) <= 1.2
    assert abs(geo.ring_row_elevations(180)[row]) <= 1.2


def test_stitch_ring_is_exact_resampling_of_a_smooth_field_across_seams():
    def field(bearing, elev):
        b, e = np.radians(bearing), np.radians(elev)
        return np.cos(b) + 0.5 * np.sin(2 * b) + 0.3 * np.sin(e)

    rows, cols = np.mgrid[0:64, 0:64]
    maps = np.stack([field(*geo.pixel_to_bearing_elev(v, cols, rows)) for v in range(4)])
    ring, valid = geo.stitch_ring(maps, width=720)
    az = geo.ring_column_azimuths(720)[None, :]
    el = geo.ring_row_elevations(180)[:, None]
    err = np.abs(ring - field(az, el))[valid]
    assert valid[60:120].all()  # +-15 deg band is inside every view, seams included
    assert err.max() < 0.02


def test_stitch_ring_rgb_places_stripes_at_their_azimuths():
    views = np.zeros((4, 256, 256, 3), dtype=np.uint8)
    targets = []
    for v in range(4):
        bearing = geo.VIEW_YAWS_DEG[v] + 20.0
        _, u, _ = geo.bearing_elev_to_pixel(bearing, 0.0, v, fx=128.0, cx=128.0, pixel_offset=geo.RGB_PIXEL_OFFSET)
        views[v, :, int(np.floor(u)):int(np.floor(u)) + 2, v % 3] = 255  # 2 px stripe straddling u
        targets.append(geo.wrap_deg(bearing))
    ring, valid = geo.stitch_ring_rgb(views, width=1440)
    assert ring.dtype == np.uint8 and valid[180].all()
    az = geo.ring_column_azimuths(1440)
    for v, bearing in enumerate(targets):
        lit = az[ring[180, :, v % 3] > 60]
        near = lit[geo.circular_abs_diff(lit, bearing) < 5]
        assert near.size and abs(geo.circular_abs_diff(np.mean(near), bearing)) <= 0.5


def test_world_round_trips_against_compute_history_rel_poses():
    rng = np.random.default_rng(1)
    for _ in range(50):
        cur = _pose(rng.uniform(-180, 180), rng.uniform(-10, 10, 3))
        hists = [_pose(rng.uniform(-180, 180), rng.uniform(-10, 10, 3)) for _ in range(8)]
        ref = TU.compute_history_rel_poses([h.astype(np.float32) for h in hists], cur.astype(np.float32),
                                           camera_forward_axis="-z")
        mine = geo.rel_pose_from_c2w(cur, np.stack(hists))
        assert np.allclose(mine, ref, atol=2e-4)
        fwd, left, up = geo.world_to_rel(cur, geo.c2w_position(np.stack(hists)))
        back = geo.rel_to_world(cur, fwd, left, up)
        assert np.allclose(back, geo.c2w_position(np.stack(hists)), atol=1e-9)
        assert np.allclose(geo.history_bearing_deg(cur, np.stack(hists)), np.degrees(np.arctan2(ref[:, 1], ref[:, 0])),
                           atol=0.05)
        for v in range(4):
            dyaw = geo.c2w_yaw_deg(geo.view_c2w(cur, v)) - geo.c2w_yaw_deg(cur)
            assert geo.circular_abs_diff(dyaw, geo.VIEW_YAWS_DEG[v]) < 1e-9
        f, l = geo.c2w_forward_left_xz(cur)
        assert np.allclose(np.linalg.norm(f), 1) and np.allclose(l, np.cross([0.0, 1.0, 0.0], f))
    assert geo.c2w_yaw_deg(np.eye(4)) == 0.0 and geo.c2w_yaw_deg(geo.yaw_rotation(90.0)) == pytest.approx(90.0)
    assert geo.path_length([[0, 0, 0], [3, 0, 4], [3, 0, 4]]) == 5.0


@pytest.mark.skipif(not os.environ.get("EXP18_TEST_DUMP"), reason="set EXP18_TEST_DUMP to a dump npz")
def test_real_dump_follows_the_conventions():
    z = np.load(os.environ["EXP18_TEST_DUMP"], allow_pickle=False)
    mask = z["history_mask"]
    cur, views, hist = z["current_c2w"], z["current_c2w_views"], z["history_c2w"]
    for v in range(4):
        assert np.allclose(views[:, v], geo.view_c2w(cur, v), atol=1e-4)
    rel = z["gt_rel_poses"]
    assert np.allclose(geo.rel_pose_from_c2w(cur[:, None], hist)[mask], rel[mask], atol=1e-4)
    gt = z["gt_heatmap"].astype(np.float32)
    peak = gt.reshape(gt.shape[:3] + (-1,)).max(-1)
    eligible = (z["gt_visibility"] > 0) & (peak > 0) & mask[..., None]
    visible = eligible.any(-1)
    label_bearing = geo.argmax_bearing(np.where(eligible[..., None, None], gt, 0.0))
    gt_bearing = geo.bearing_from_rel_pose(rel[..., 0], rel[..., 1])
    err = geo.circular_abs_diff(label_bearing, gt_bearing)[visible]
    assert visible.sum() > 0 and err.max() <= 2.0


# --------------------------------------------------------------------------- #
# E out-and-back turnaround, H2 attribution table, route-pattern key rows
# --------------------------------------------------------------------------- #
def _walk(waypoints, step: float = 0.25, cam_height: float = 1.25) -> np.ndarray:
    """Camera centres of an agent walking straight between waypoints, one frame per ``step`` metres."""
    wps = np.asarray(waypoints, dtype=np.float64)
    pts = [wps[0]]
    for a, b in zip(wps[:-1], wps[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        pts.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
    return np.asarray(pts) + np.array([0.0, cam_height, 0.0])


# The outbound leg overshoots east to (6, 0, 0) and then turns back north-west to
# the route end (3, 0, 3): the farthest point from the start is NOT the turnaround.
_REF = [[0.0, 0.2, 0.0], [6.0, 0.2, 0.0], [3.0, 0.2, 3.0]]
_OAB = _REF + _REF[::-1][1:]


def test_out_and_back_turnaround_is_the_route_midpoint_not_the_farthest_point():
    pos = _walk(_OAB)
    res = geo.out_and_back_turnaround(pos, _OAB)
    agent = pos - np.array([0.0, 1.25, 0.0])
    assert res["index"] == 2 and res["waypoints_missed"] == 0
    assert np.allclose(agent[res["frame"]], _REF[-1], atol=1e-9) and res["miss_m"] < 1e-9
    far = int(np.linalg.norm(pos - pos[0], axis=1).argmax())
    assert np.allclose(agent[far], _REF[1]) and far < res["frame"]  # the old rule picks the outbound overshoot
    # a route that crosses its turnaround point earlier: in-order matching takes the later, real visit
    a, b = [3.0, 0.2, 3.0], [6.0, 0.2, 0.0]
    ref = [[0.0, 0.2, 0.0], a, b, a]
    path = ref + ref[::-1][1:]
    pos2 = _walk(path)
    visits = np.nonzero(np.linalg.norm(pos2 - np.array([3.0, 1.45, 3.0]), axis=1) < 1e-9)[0]
    assert len(visits) == 3 and geo.out_and_back_turnaround(pos2, path)["frame"] == visits[1]
    # not an out-and-back -> no frame
    assert geo.out_and_back_turnaround(pos, _REF)["frame"] == -1
    assert geo.out_and_back_turnaround(pos, None)["frame"] == -1


def test_h2_attribution_is_written_only_where_both_verdicts_assert_it():
    cm = pytest.importorskip("scripts.exp18.compute_metrics")
    a = cm.h2_attribution
    assert a("refute", "support")["attribution"] == "泛化瓶颈在里程计，不在热力头"
    assert a("refute", "refute")["attribution"] == "头本身不泛化"
    for vo, gt in (("not_measured", "not_measured"), ("not_measured", "support"), ("not_measured", "refute"),
                   ("refute", "not_measured"), ("refute", "missing"), ("support", "refute"), ("missing", "support")):
        out = a(vo, gt)
        assert out["attribution"] is None and out["attribution_note"], (vo, gt)
    # the builder's fake C-vs-B case: both arms 没测出来 -> no negative claim; literal reading kept for the record
    out = a("not_measured", "not_measured")
    assert out["attribution_note"].startswith("不适用") and out["attribution_literal_reading"] == "头本身不泛化"
    assert a("support", "support")["attribution_literal_reading"] is None


def test_pattern_figure_uses_the_route_turnaround_for_out_and_back():
    pd = pytest.importorskip("pandas")
    sc = pytest.importorskip("scripts.exp18.select_cases")
    pos = _walk(_OAB)
    frame = geo.out_and_back_turnaround(pos, _OAB)["frame"]
    T = len(pos)
    t = sorted(set(range(19, T, 8)) | {T - 1})
    dist = np.linalg.norm(pos[t] - pos[0], axis=1)
    eps, rows = [], []
    for pattern, key, turn_frame in (("out_and_back", "s/clip_1", frame), ("loop", "s/clip_2", -1)):
        eps.append({"tier": "E", "scene": "s", "clip": key[2:], "clip_key": key, "episode_id": key, "order_key": key,
                    "pattern": pattern, "vo_pck8": 0.5, "turnaround_frame": turn_frame, "turnaround_miss_m": 0.0})
        rows.extend({"tier": "E", "clip_key": key, "row": i, "t": ti, "is_final": ti == T - 1,
                     "dist_from_start_m": d} for i, (ti, d) in enumerate(zip(t, dist)))
    eps, rows = pd.DataFrame(eps), pd.DataFrame(rows)
    out = sc.pattern_figure(eps, rows)
    oab, loop = out["out_and_back"], out["loop"]
    roles = {k["role"]: k["t"] for k in oab["key_rows"]}
    assert roles["before_turnaround"] < frame < roles["after_turnaround"]
    assert abs(roles["turnaround"] - frame) <= 4 and oab["turnaround_rule"].startswith("route turnaround")
    # old rule: the overshoot at (6, 0, 0), here on the return leg, far from the real turnaround
    assert abs(oab["farthest_from_start_row_t"] - frame) > 8
    assert loop["turnaround_row_t"] == loop["farthest_from_start_row_t"]
    assert loop["turnaround_rule"].startswith("scored row farthest")
    eps.loc[0, "turnaround_frame"] = -1  # dump without a palindromic reference_path
    assert "FALLBACK" in sc.pattern_figure(eps, rows)["out_and_back"]["turnaround_rule"]

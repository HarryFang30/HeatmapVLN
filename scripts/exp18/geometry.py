"""EXP-18 geometry conventions: bearings, heatmap pixels, heading-centred rings, world poses.

Every EXP-18 tool that turns a heatmap pixel into a direction, or a pose into a
map position, goes through this module.  Each convention is read off the code
that makes the History Head labels, not re-derived:

* ``src/data/sliding_window_dataset.py::_compute_per_history_multiview_heatmaps``
  renders one 64x64 map per view, order front, right, back, left.  The view
  cameras (chunk ``pose_{front,right,back,left}``) share the front camera's
  centre and are turned about its up axis by 0 / -90 / 180 / +90 degrees
  (left-positive; checked on dumped poses, see ``view_c2w``).
* ``src/data/heatmap_geometry.py::compute_history_heatmap`` projects the history
  camera centre with a Habitat pinhole (x right, y up, camera looks along -z):
  ``u = fx*x/(-z) + cx``, ``v = fy*(-y)/(-z) + cy``, fx = fy = cx = cy = 128 at
  256 px (``intrinsics.json``), keeps ``0 <= u < 256``, then draws the Gaussian
  at ``u * 64/256`` with ``draw_gaussian_point``, whose grid is
  ``np.arange(x_min, x_max)``.  So heatmap pixel INDEX i sits at continuous
  heatmap coordinate i, not i + 0.5: index 32 is the optical axis and a point
  straight ahead peaks at (row 32, col 32).  In heatmap units fx = cx = 32.
* ``src/data/trajectory_utils.py::compute_history_rel_poses(..., '-z')`` gives
  (forward, left, cos dyaw, sin dyaw) in the current front-camera frame with
  forward = -z_cam, left = -x_cam and yaw positive to the left.

Bearing = atan2(left, forward) in degrees, left-positive, wrapped to
(-180, 180]: front 0, left +90, right -90, back 180.  A view owns the bearings
(yaw - 45, yaw + 45], matching the label's ``0 <= u < W`` test.

Rendered RGB (habitat-sim / OpenGL) follows the usual image convention
instead: pixel p covers [p, p + 1) and its centre is continuous coordinate
p + 0.5.  ``stitch_ring_rgb`` uses that; the heatmap functions use the label
convention.  numpy only, Python 3.8 compatible (imported from envs/vlnce and
envs/qwen25).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

VIEW_NAMES = ("front", "right", "back", "left")
VIEW_YAWS_DEG = (0.0, -90.0, 180.0, 90.0)
VIEW_CLASS_NAMES = ("none",) + VIEW_NAMES  # validate.py 5-way class: 0 none, 1 + view
FRONT, RIGHT, BACK, LEFT = 0, 1, 2, 3

IMG_SIZE = 256
IMG_FX = 128.0
IMG_CX = 128.0
HM_SIZE = 64
HM_FX = IMG_FX * HM_SIZE / IMG_SIZE  # 32.0
HM_CX = IMG_CX * HM_SIZE / IMG_SIZE  # 32.0
LABEL_PIXEL_OFFSET = 0.0  # heatmap index i <-> continuous coordinate i
RGB_PIXEL_OFFSET = 0.5  # image pixel p <-> continuous coordinate p + 0.5
FLOOR_PEAK_YX = (32, 32)  # constant straight-behind floor: back view, centre pixel

_YAWS = np.asarray(VIEW_YAWS_DEG, dtype=np.float64)


def _out(a: np.ndarray):
    return float(a) if np.ndim(a) == 0 else a


# --------------------------------------------------------------------------- #
# Angles
# --------------------------------------------------------------------------- #
def wrap_deg(angle):
    """Wrap degrees to (-180, 180]."""
    a = np.mod(np.asarray(angle, dtype=np.float64) + 180.0, 360.0) - 180.0
    return _out(np.where(a == -180.0, 180.0, a))


def circular_abs_diff(a, b):
    """|a - b| on the circle, degrees in [0, 180]."""
    d = np.mod(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64) + 180.0, 360.0) - 180.0
    return _out(np.abs(d))


def circular_range_deg(angles) -> float:
    """Width of the smallest arc containing every finite angle (0 for fewer than two)."""
    a = np.asarray(angles, dtype=np.float64).ravel()
    a = np.sort(np.mod(a[np.isfinite(a)], 360.0))
    if a.size < 2:
        return 0.0
    gaps = np.diff(np.concatenate([a, a[:1] + 360.0]))
    return float(360.0 - gaps.max())


def bearing_from_rel_pose(forward, left):
    """GT bearing of a history point from its (forward, left) rel pose, degrees, left-positive."""
    return _out(np.degrees(np.arctan2(np.asarray(left, dtype=np.float64), np.asarray(forward, dtype=np.float64))))


def view_for_bearing(bearing):
    """View index owning each bearing: F (-45, 45], R (-135, -45], B (135, -135], L (45, 135]."""
    b = np.asarray(wrap_deg(bearing), dtype=np.float64)
    v = np.mod(np.floor((45.0 - b) / 90.0), 4).astype(np.int64)
    return int(v) if v.ndim == 0 else v


# --------------------------------------------------------------------------- #
# Heatmap pixels <-> directions (label convention by default)
# --------------------------------------------------------------------------- #
def pixel_to_bearing_elev(view, u, v, *, fx: float = HM_FX, fy: Optional[float] = None, cx: float = HM_CX,
                          cy: Optional[float] = None, pixel_offset: float = LABEL_PIXEL_OFFSET):
    """(bearing, elevation) in degrees of pixel index (u = column, v = row) of ``view``.

    Defaults are the 64x64 label grid; for a 256 px RGB image pass fx = cx = 128
    and ``pixel_offset=RGB_PIXEL_OFFSET``.  Elevation is positive up.
    """
    fy = fx if fy is None else fy
    cy = cx if cy is None else cy
    yaw = _YAWS[np.asarray(view, dtype=np.int64)]
    xn = (np.asarray(u, dtype=np.float64) + pixel_offset - cx) / fx
    yn = (np.asarray(v, dtype=np.float64) + pixel_offset - cy) / fy
    bearing = wrap_deg(yaw - np.degrees(np.arctan(xn)))
    elev = np.degrees(np.arctan2(-yn, np.sqrt(1.0 + xn * xn)))
    return bearing, _out(elev)


def bearing_elev_to_pixel(bearing, elev=0.0, view=None, *, fx: float = HM_FX, fy: Optional[float] = None,
                          cx: float = HM_CX, cy: Optional[float] = None, pixel_offset: float = LABEL_PIXEL_OFFSET):
    """Inverse of ``pixel_to_bearing_elev``: continuous (view, u, v) index coordinates.

    ``view=None`` picks the owning view.  u/v are NaN when the direction is
    behind the chosen view; they may fall outside the grid (caller checks).
    """
    fy = fx if fy is None else fy
    cy = cx if cy is None else cy
    b = np.asarray(bearing, dtype=np.float64)
    view = view_for_bearing(b) if view is None else np.asarray(view, dtype=np.int64)
    az = np.radians(np.asarray(wrap_deg(b - _YAWS[np.asarray(view)]), dtype=np.float64))
    el = np.radians(np.asarray(elev, dtype=np.float64))
    ok = np.cos(az) > 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(ok, cx - fx * np.tan(az) - pixel_offset, np.nan)
        v = np.where(ok, cy - fy * np.tan(el) / np.cos(az) - pixel_offset, np.nan)
    return view, _out(u), _out(v)


def argmax_pixel(maps) -> Tuple[np.ndarray, np.ndarray]:
    """First-occurrence argmax (row, col) over the last two axes, like torch/validate.py."""
    m = np.asarray(maps)
    flat = m.reshape(m.shape[:-2] + (-1,)).argmax(-1)
    return flat // m.shape[-1], flat % m.shape[-1]


def argmax_view_pixel(maps) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint argmax over [..., 4, H, W] -> (view, row, col); -1 where no finite value."""
    m = np.asarray(maps, dtype=np.float32)
    h, w = m.shape[-2:]
    flat = m.reshape(m.shape[:-3] + (-1,))
    finite = np.isfinite(flat)
    idx = np.where(finite, flat, -np.inf).argmax(-1)
    none = ~finite.any(-1)
    view, rem = idx // (h * w), idx % (h * w)
    row, col = rem // w, rem % w
    return tuple(np.where(none, -1, a) for a in (view, row, col))


def argmax_bearing(gated, *, return_pixel: bool = False, **pixel_kwargs):
    """Predicted bearing = joint argmax of ``heatmaps_gated`` [..., 4, 64, 64] (NaN if all NaN)."""
    view, row, col = argmax_view_pixel(gated)
    bearing, _ = pixel_to_bearing_elev(np.maximum(view, 0), col, row, **pixel_kwargs)
    bearing = np.where(view < 0, np.nan, bearing)
    bearing = _out(bearing)
    return (bearing, view, row, col) if return_pixel else bearing


# --------------------------------------------------------------------------- #
# Heading-centred equiangular ring
# --------------------------------------------------------------------------- #
def ring_column_azimuths(width: int) -> np.ndarray:
    """Azimuth of each ring column centre: +180 at the left edge, 0 at the centre, -180 at the right."""
    return 180.0 - (np.arange(width, dtype=np.float64) + 0.5) * 360.0 / width


def ring_row_elevations(height: int, elev_top: float = 45.0, elev_bottom: float = -45.0) -> np.ndarray:
    return elev_top - (np.arange(height, dtype=np.float64) + 0.5) * (elev_top - elev_bottom) / height


def azimuth_to_ring_x(azimuth, width: int):
    """Continuous column index (column j's centre is j) of an azimuth on a ``width`` ring."""
    return _out((180.0 - np.asarray(wrap_deg(azimuth))) * width / 360.0 - 0.5)


def ring_extent(elev_top: float = 45.0, elev_bottom: float = -45.0) -> Tuple[float, float, float, float]:
    """``imshow(ring, extent=ring_extent())`` puts azimuth/elevation degrees on the axes."""
    return (180.0, -180.0, elev_bottom, elev_top)


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray, tol: float = 0.5):
    """Sample img [H, W, C] at index coords; up to ``tol`` px beyond the grid is edge-clamped."""
    h, w = img.shape[:2]
    valid = np.isfinite(x) & np.isfinite(y) & (x >= -tol) & (x <= w - 1 + tol) & (y >= -tol) & (y <= h - 1 + tol)
    xc = np.clip(np.nan_to_num(x), 0, w - 1)
    yc = np.clip(np.nan_to_num(y), 0, h - 1)
    x0 = np.floor(xc).astype(np.int64)
    y0 = np.floor(yc).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (xc - x0)[..., None]
    wy = (yc - y0)[..., None]
    out = (img[y0, x0] * (1 - wx) * (1 - wy) + img[y0, x1] * wx * (1 - wy)
           + img[y1, x0] * (1 - wx) * wy + img[y1, x1] * wx * wy)
    return out, valid


def _stitch(views: np.ndarray, width: int, height: Optional[int], elev_top: float, elev_bottom: float,
            fx: float, fy: float, cx: float, cy: float, pixel_offset: float, seam_column: bool):
    views = np.asarray(views, dtype=np.float64)  # [4, H, W, C]
    height = width // 4 if height is None else int(height)
    az = ring_column_azimuths(width)
    el = ring_row_elevations(height, elev_top, elev_bottom)
    view = view_for_bearing(az)
    az_v = np.radians(np.asarray(wrap_deg(az - _YAWS[view])))  # (-45, 45]
    x = cx - fx * np.tan(az_v) - pixel_offset  # [w]
    y = cy - fy * np.tan(np.radians(el))[:, None] / np.cos(az_v)[None, :] - pixel_offset  # [h, w]
    ring = np.zeros((height, width, views.shape[-1]), dtype=np.float64)
    valid = np.zeros((height, width), dtype=bool)
    for vi in range(4):
        cols = np.nonzero(view == vi)[0]
        if cols.size == 0:
            continue
        img = views[vi]
        if seam_column:
            # Label grid: continuous column W of this view is exactly the ray of
            # column 0 of the clockwise neighbour (same azimuth, same row).
            img = np.concatenate([img, views[(vi + 1) % 4][:, :1]], axis=1)
        vals, ok = _bilinear(img, np.broadcast_to(x[cols], (height, cols.size)), y[:, cols])
        ring[:, cols] = vals
        valid[:, cols] = ok
    return ring, valid


def stitch_ring(maps, width: int = 720, height: Optional[int] = None, elev_top: float = 45.0,
                elev_bottom: float = -45.0, fill: float = np.nan):
    """Resample four 64x64 label-convention maps [4, H, W] onto a heading-centred ring.

    Exact inverse pinhole per ring pixel, bilinear.  Columns span azimuth +180
    (left edge) -> 0 (centre) -> -180 (right edge) with centres at
    ``ring_column_azimuths(width)``; rows span elev_top -> elev_bottom
    (default height = width // 4, i.e. square degrees).  Returns (ring [h, w]
    float32, valid [h, w]); pixels outside every view's vertical FOV get ``fill``.
    """
    m = np.asarray(maps, dtype=np.float64)
    size = m.shape[-1]
    f = HM_FX * size / HM_SIZE
    c = HM_CX * size / HM_SIZE
    ring, valid = _stitch(m[..., None], width, height, elev_top, elev_bottom, f, f, c, c, LABEL_PIXEL_OFFSET, True)
    ring = ring[..., 0]
    ring[~valid] = fill
    return ring.astype(np.float32), valid


def stitch_ring_rgb(views, width: int = 720, height: Optional[int] = None, elev_top: float = 45.0,
                    elev_bottom: float = -45.0, fill: float = 0.0):
    """Same ring for four rendered RGB views [4, H, W, 3] (HFOV 90, pixel centres at p + 0.5).

    Returns (ring [h, w, 3] in the input dtype, valid [h, w]).
    """
    v = np.asarray(views)
    h_img, w_img = v.shape[1:3]
    ring, valid = _stitch(v, width, height, elev_top, elev_bottom, w_img / 2.0, h_img / 2.0, w_img / 2.0,
                          h_img / 2.0, RGB_PIXEL_OFFSET, False)
    ring[~valid] = fill
    if np.issubdtype(v.dtype, np.integer):
        ring = np.clip(np.rint(ring), np.iinfo(v.dtype).min, np.iinfo(v.dtype).max)
    return ring.astype(v.dtype), valid


# --------------------------------------------------------------------------- #
# World <-> current camera (Habitat c2w: y up, camera looks along -z, x right)
# --------------------------------------------------------------------------- #
def c2w_position(c2w) -> np.ndarray:
    return np.asarray(c2w, dtype=np.float64)[..., :3, 3]


def c2w_forward(c2w) -> np.ndarray:
    return -np.asarray(c2w, dtype=np.float64)[..., :3, 2]


def c2w_left(c2w) -> np.ndarray:
    return -np.asarray(c2w, dtype=np.float64)[..., :3, 0]


def c2w_forward_left_xz(c2w) -> Tuple[np.ndarray, np.ndarray]:
    """Unit forward / left vectors projected on the world floor plane (y = 0), shape [..., 3]."""
    out = []
    for vec in (c2w_forward(c2w), c2w_left(c2w)):
        vec = vec.copy()
        vec[..., 1] = 0.0
        out.append(vec / np.linalg.norm(vec, axis=-1, keepdims=True))
    return out[0], out[1]


def c2w_yaw_deg(c2w):
    """World yaw of the camera heading, left-positive about +y; 0 = facing world -z."""
    f = c2w_forward(c2w)
    return _out(np.degrees(np.arctan2(-f[..., 0], -f[..., 2])))


def yaw_rotation(yaw_deg: float) -> np.ndarray:
    """4x4 rotation about +y by ``yaw_deg`` (positive turns a -z camera to the left)."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]])


def view_c2w(front_c2w, view: int) -> np.ndarray:
    """c2w of surround view ``view`` (F, R, B, L) from the front camera's c2w."""
    return np.asarray(front_c2w, dtype=np.float64) @ yaw_rotation(VIEW_YAWS_DEG[view])


def world_to_rel(c2w_current, points) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(forward, left, up) metres of world points [..., 3] in the current camera frame."""
    T = np.asarray(c2w_current, dtype=np.float64)
    d = np.asarray(points, dtype=np.float64) - T[..., :3, 3]
    p = np.einsum("...ji,...j->...i", T[..., :3, :3], d)  # R^T d
    return -p[..., 2], -p[..., 0], p[..., 1]


def rel_to_world(c2w_current, forward, left, up=0.0) -> np.ndarray:
    """World point at (forward, left, up) metres in the current camera frame, shape [..., 3]."""
    T = np.asarray(c2w_current, dtype=np.float64)
    f, l, u = (np.asarray(a, dtype=np.float64)[..., None] for a in (forward, left, up))
    return T[..., :3, 3] + f * c2w_forward(T) + l * c2w_left(T) + u * T[..., :3, 1]


def rel_pose_from_c2w(c2w_current, c2w_history) -> np.ndarray:
    """(forward, left, cos dyaw, sin dyaw) like ``compute_history_rel_poses(..., '-z')``, shape [..., 4]."""
    T = np.asarray(c2w_current, dtype=np.float64)
    H = np.asarray(c2w_history, dtype=np.float64)
    fwd, left, _ = world_to_rel(T, H[..., :3, 3])
    hist_fwd_local = np.einsum("...ji,...j->...i", T[..., :3, :3], c2w_forward(H))
    dyaw = np.arctan2(-hist_fwd_local[..., 0], -hist_fwd_local[..., 2])
    return np.stack([fwd, left, np.cos(dyaw), np.sin(dyaw)], axis=-1)


def history_bearing_deg(c2w_current, c2w_history):
    """GT bearing of history camera centres seen from the current front camera."""
    fwd, left, _ = world_to_rel(c2w_current, c2w_position(c2w_history))
    return bearing_from_rel_pose(fwd, left)


def path_length(positions) -> float:
    """Summed 3D step length of a [T, 3] position sequence (NaN steps skipped)."""
    p = np.asarray(positions, dtype=np.float64)
    if p.shape[0] < 2:
        return 0.0
    steps = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    return float(np.nansum(steps))


def out_and_back_turnaround(positions, reference_path, tolerance: float = 0.5) -> dict:
    """Frame at which an out-and-back clip turns round, read from its route.

    The renderer builds an out-and-back as ``path = ref + ref[::-1][1:]`` and
    records ``turnaround_index = len(ref) - 1`` (render/select_episodes.py
    ``build_e``); the collector copies ``path`` into meta.json as
    ``reference_path``.  The turnaround is therefore ``path[(len(path) - 1) // 2]``,
    which is NOT always the point farthest from the start (an intermediate
    waypoint can lie farther out).

    positions: [T, 3] camera centres of every frame (``clip_c2w``).  The camera
    sits a constant sensor height above the navmesh points of the path (1.25 m
    in the renders); that offset is read from frame 0, which is the route start.
    path[0..mid] are matched in order to their first passage within
    ``tolerance`` (as render/finalize_clip_lists.py ``_route_stats`` does), so
    the return leg, which revisits every outbound point, can never be matched
    first; the turnaround frame is the closest approach to path[mid] inside its
    first passage (ties: earliest).

    Returns {"frame", "index", "miss_m", "waypoints_missed", "note"}; frame is -1
    when ``reference_path`` is not an odd-length palindrome (not an out-and-back).
    """
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    ref = np.asarray(reference_path if reference_path is not None else [], dtype=np.float64).reshape(-1, 3)
    n = len(ref)
    if n < 3 or n % 2 == 0 or not np.allclose(ref, ref[::-1], atol=1e-3):
        return {"frame": -1, "index": -1, "miss_m": float("nan"), "waypoints_missed": 0,
                "note": "reference_path is not an out-and-back palindrome"}
    if len(p) == 0:
        return {"frame": -1, "index": -1, "miss_m": float("nan"), "waypoints_missed": 0, "note": "no frames"}
    agent = p - np.array([0.0, p[0, 1] - ref[0, 1], 0.0])
    mid = (n - 1) // 2
    t0, missed = 0, 0
    for i in range(mid + 1):
        dist = np.linalg.norm(agent[t0:] - ref[i], axis=-1)
        hits = np.nonzero(dist <= tolerance)[0]
        if hits.size:
            start = end = int(hits[0])
            while end + 1 < dist.size and dist[end + 1] <= tolerance:
                end += 1
            best = start + int(dist[start:end + 1].argmin())
        else:  # never within tolerance: closest approach, and do not advance (keeps the order constraint loose)
            missed += 1
            start, best = 0, int(dist.argmin())
        if i < mid:
            t0 += start
    return {"frame": int(t0 + best), "index": int(mid), "miss_m": float(dist[best]), "waypoints_missed": int(missed),
            "note": "in-order first passage of reference_path[(len - 1) // 2]"}

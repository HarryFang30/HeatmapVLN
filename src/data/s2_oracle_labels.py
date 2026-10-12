"""Oracle answers and agreement rules for recorded System2 calls (EXP-23 part B).

For every System2 call in a ``heatmapvln-s2-record-v1`` record this module
decides what the oracle would have answered and whether the model's own answer
was good enough to keep (design doc ``exp23-s2-dagger-design.md`` §2 B):

* ``d <= stop_radius_m``            -> ``STOP``.
* otherwise the oracle route (reference-path following, see
  :func:`last_reached_by_step` / :func:`choose_waypoint`, which deviates from
  the design's literal argmin because that argmin is almost always the goal) is sampled every
  ``sample_step_m``, projected into the lookdown camera and occlusion-tested
  against the recorded lookdown depth; the farthest visible sample at least
  ``min_pixel_ahead_m`` along the route becomes ``↓`` + ``"u v"``.
* nothing visible -> turn towards the first route point ``turn_target_ahead_m``
  along: ``k = clamp(round(|dyaw| / 15), 1, 4)`` arrows; ``|dyaw| < 7.5`` deg
  and nothing visible -> ``skip`` (the oracle has no answer).

Text formats are the deployed server's, checked against code and logs:

* pixel: the native second turn is ``"u v"`` -- column first, then row --
  after a first turn that is exactly ``"↓"``.  Two independent checks: in
  30,277 EXP-20 A0 trajectory answers the first number reaches 601 (1,785 of
  them above 479) and the second never passes 455, on a 640x480 lookdown; and
  InternData-N1 prints its ``goal`` as stored, ``[495, 345]`` for a goal that
  projects to (u, v) = (495.6, 347.1).  ``rpc_model_server._parse_internnav_pixel_goal``
  calls the text ``"row col"`` and returns ``[second, first]``, so the response
  ``pixel_goal`` is really ``[v, u]``.  The server only re-encodes it into the
  same text for System1, so deployment is unaffected, but nothing here reads
  that field: the pixel always comes from the text.
* turn: ``"←"`` / ``"→"`` repeated, at most 4 (``finalize_native_local_actions``
  executes only the first four parsed actions).
* stop: ``"STOP"``.

Geometry follows Habitat (y up, camera looks along -z, x right) and the
deployed client's sensors (``s2_records.CAMERA``): the lookdown camera sits
1.25 m above the agent's navmesh position and is the body frame pitched down
30 deg about its own x axis (two ``LOOK_DOWN`` actions rotate the sensor, they
do not move it).  Depth is Habitat z-depth (along the optical axis).  Image
positions are pixel *index* coordinates (pixel ``(col i, row j)`` is centred
at ``(i, j)``), the convention of ``scripts/exp19/gt.py``.

Everything here is numpy only.  Geodesic distances and navmesh paths are
injected as callables, so the Habitat side (``scripts/tools/label_s2_records.py``)
stays thin and the rules can be tested on a flat synthetic world.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .s2_records import CAMERA


STOP_TEXT = "STOP"
LOOKDOWN_TEXT = "↓"
LEFT_TEXT = "←"
RIGHT_TEXT = "→"
# Same vocabulary and cap as native_internnav_exact.parse_native_actions /
# finalize_native_local_actions (only the first four actions are executed).
_NATIVE_TOKENS = ("STOP", "↑", "←", "→", "↓")
_NATIVE_TOKEN_RE = re.compile("|".join(re.escape(t) for t in _NATIVE_TOKENS))
MAX_EXECUTED_ACTIONS = 4
ORACLE_PATH_ROW_M = 15.0  # route length kept in a label row for gate G4

ORACLE_KINDS = ("stop", "pixel", "turn", "skip")
MODEL_KINDS = ("stop", "pixel", "turn", "other")

Geodesic = Callable[[np.ndarray, np.ndarray], float]
FindPath = Callable[[np.ndarray, np.ndarray], "np.ndarray | None"]


@dataclass(frozen=True)
class OracleConfig:
    stop_radius_m: float = 2.0
    success_radius_m: float = 3.0  # a model STOP inside this is kept (R2R success)
    reached_radius_m: float = 1.0
    rejoin_slack_m: float = 2.0  # see choose_waypoint; 0 = the design's literal argmin
    sample_step_m: float = 0.1
    occlusion_tol_m: float = 0.3
    min_pixel_ahead_m: float = 0.75  # InternNav skips goals within 3 forward steps
    turn_target_ahead_m: float = 0.5
    turn_step_deg: float = 15.0
    max_turns: int = MAX_EXECUTED_ACTIONS
    agree_path_dist_m: float = 1.0
    agree_min_progress_m: float = 0.5

    @property
    def min_turn_deg(self) -> float:
        return self.turn_step_deg / 2.0

    def digest(self) -> str:
        text = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
def quat_wxyz_to_rot(q: Sequence[float]) -> np.ndarray:
    """3x3 rotation of a (w, x, y, z) quaternion (normalised first)."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def yaw_to_quat_wxyz(yaw_rad: float) -> list[float]:
    """Body rotation about +y (positive = turned left); 0 faces -z."""
    return [math.cos(yaw_rad / 2.0), 0.0, math.sin(yaw_rad / 2.0), 0.0]


@dataclass(frozen=True)
class LookdownCamera:
    width: int = int(CAMERA["width"])
    height: int = int(CAMERA["height"])
    hfov_deg: float = float(CAMERA["hfov_deg"])
    height_m: float = float(CAMERA["height_m"])
    pitch_deg: float = float(CAMERA["lookdown_pitch_deg"])

    @property
    def focal(self) -> float:
        """Square pixels: fx = fy = (W / 2) / tan(HFOV / 2)."""
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def centre(self) -> tuple[float, float]:
        """Principal point in pixel index coordinates."""
        return (self.width - 1) / 2.0, (self.height - 1) / 2.0

    def c2w(self, position: Sequence[float], rotation_wxyz: Sequence[float]) -> np.ndarray:
        """Body pose, sensor raised by ``height_m``, then pitched down about its own x."""
        body = quat_wxyz_to_rot(rotation_wxyz)
        a = -math.radians(self.pitch_deg)  # LOOK_DOWN rotates the sensor by -angle about x
        pitch = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])
        out = np.eye(4)
        out[:3, :3] = body @ pitch
        out[:3, 3] = np.asarray(position, dtype=np.float64) + body @ np.array([0.0, self.height_m, 0.0])
        return out

    def project(self, points_world: np.ndarray, c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points [N, 3] -> (pixel index coords [N, 2] (u, v), z-depth [N]).

        Points at or behind the image plane get z-depth <= 0 and NaN pixels.
        """
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        cam = (pts - c2w[:3, 3]) @ c2w[:3, :3]  # R^T (p - o), row-wise
        z = -cam[:, 2]
        f = self.focal
        cx, cy = self.centre
        with np.errstate(divide="ignore", invalid="ignore"):
            u = f * cam[:, 0] / z + cx
            v = f * (-cam[:, 1]) / z + cy
        uv = np.stack([u, v], axis=-1)
        uv[z <= 1e-6] = np.nan
        return uv, z

    def unproject(self, u: float, v: float, z_depth: float, c2w: np.ndarray) -> np.ndarray:
        """Pixel index (u, v) at z-depth -> world point [3]."""
        f = self.focal
        cx, cy = self.centre
        cam = np.array([(u - cx) * z_depth / f, -(v - cy) * z_depth / f, -z_depth])
        return c2w[:3, :3] @ cam + c2w[:3, 3]


def heading_delta_deg(rotation_wxyz: Sequence[float], position: Sequence[float], target: Sequence[float]) -> float:
    """Signed yaw from the agent's heading to ``target`` on the floor plane (left positive)."""
    forward = quat_wxyz_to_rot(rotation_wxyz) @ np.array([0.0, 0.0, -1.0])
    t = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    # Rotating (x, z) by +theta about +y sends -z towards -x (TURN_LEFT).
    cross = forward[2] * t[0] - forward[0] * t[2]
    dot = forward[0] * t[0] + forward[2] * t[2]
    return math.degrees(math.atan2(cross, dot))


# --------------------------------------------------------------------------- #
# Polylines
# --------------------------------------------------------------------------- #
def concat_paths(parts: Sequence[np.ndarray]) -> np.ndarray:
    """Join path pieces, dropping a piece's first point when it repeats the last one."""
    out: list[np.ndarray] = []
    for part in parts:
        part = np.asarray(part, dtype=np.float64).reshape(-1, 3)
        if not len(part):
            continue
        if out and np.linalg.norm(out[-1][-1] - part[0]) < 1e-6:
            part = part[1:]
        if len(part):
            out.append(part)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 3))


def polyline_arclength(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros(0)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])


def resample_polyline(points: np.ndarray, step_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every ``step_m`` of arc length from the start (end point included)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) > 1:  # np.interp needs strictly increasing arc length
        keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9])
        pts = pts[keep]
    arc = polyline_arclength(pts)
    if len(pts) < 2 or arc[-1] <= 0.0:
        return pts[:1].copy(), np.zeros(min(len(pts), 1))
    s = np.arange(0.0, arc[-1], float(step_m))
    if arc[-1] - s[-1] > 1e-9:
        s = np.append(s, arc[-1])
    samples = np.stack([np.interp(s, arc, pts[:, i]) for i in range(3)], axis=-1)
    return samples, s


def nearest_on_polyline(points: np.ndarray, q: Sequence[float]) -> tuple[float, float]:
    """(distance from ``q`` to the polyline, arc length of the nearest point)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    q = np.asarray(q, dtype=np.float64)
    if len(pts) == 1:
        return float(np.linalg.norm(q - pts[0])), 0.0
    a, b = pts[:-1], pts[1:]
    ab = b - a
    denom = np.maximum((ab * ab).sum(-1), 1e-12)
    t = np.clip(((q - a) * ab).sum(-1) / denom, 0.0, 1.0)
    closest = a + t[:, None] * ab
    dist = np.linalg.norm(closest - q, axis=1)
    i = int(np.argmin(dist))
    arc = polyline_arclength(pts)
    return float(dist[i]), float(arc[i] + t[i] * np.sqrt(denom[i]))


# --------------------------------------------------------------------------- #
# Oracle route: follow the reference path from the last waypoint reached
# --------------------------------------------------------------------------- #
def route_waypoints(reference_path: Sequence[Sequence[float]], goal: Sequence[float]) -> np.ndarray:
    """Reference path with the goal appended when it is not already its last point."""
    ref = np.asarray(reference_path, dtype=np.float64).reshape(-1, 3)
    goal = np.asarray(goal, dtype=np.float64).reshape(3)
    if not len(ref) or np.linalg.norm(ref[-1] - goal) > 1e-3:
        ref = np.concatenate([ref, goal[None]], axis=0)
    return ref


def last_reached_by_step(
    positions: np.ndarray,
    waypoints: np.ndarray,
    geodesic: Geodesic,
    radius_m: float,
    *,
    euclid_margin_m: float = 0.5,
) -> np.ndarray:
    """Index of the last waypoint reached at or before each position (-1: none).

    ``w_k`` counts as reached once any visited position is within ``radius_m``
    geodesic of it.  A straight-line distance above ``radius_m + euclid_margin_m``
    rules a pair out without a path query (the margin covers navmesh snapping).
    """
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    wps = np.asarray(waypoints, dtype=np.float64).reshape(-1, 3)
    out = np.full(len(pos), -1, dtype=np.int64)
    best = -1
    for i, p in enumerate(pos):
        near = np.flatnonzero(np.linalg.norm(wps - p, axis=1) <= radius_m + euclid_margin_m)
        for k in near[::-1]:
            if k <= best:
                break
            if geodesic(p, wps[k]) <= radius_m:
                best = int(k)
                break
        out[i] = best
    return out


def remaining_reference_m(waypoints: np.ndarray, geodesic: Geodesic) -> np.ndarray:
    """Route length from each waypoint to the last one (consecutive geodesics)."""
    wps = np.asarray(waypoints, dtype=np.float64).reshape(-1, 3)
    seg = np.array([geodesic(wps[k], wps[k + 1]) for k in range(len(wps) - 1)], dtype=np.float64)
    return np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]]) if len(seg) else np.zeros(len(wps))


def choose_waypoint(
    position: np.ndarray,
    waypoints: np.ndarray,
    last_reached: int,
    remaining_m: np.ndarray,
    geodesic: Geodesic,
    slack_m: float = 0.0,
) -> tuple[int | None, dict[int, float]]:
    """Waypoint at which the oracle rejoins the reference path.

    ``cost_k = geo(p, w_k) + remaining(w_k)`` for ``k > last_reached`` (design
    §2 B.2).  Taken literally, ``argmin cost_k`` is almost always the goal: by
    the triangle inequality ``geo(p, goal) <= geo(p, w_k) + remaining(w_k)``
    (98.9% of on-route states of 300 train episodes), so the oracle would
    ignore the reference path.  The earliest candidate within ``slack_m`` of the
    cheapest one is taken instead: on route that is the next waypoint (the
    reference is near-geodesic: cost gap p90 1.19 m), and a missed waypoint
    is abandoned once going back to it costs more than ``slack_m`` extra.
    ``slack_m = 0`` is the literal argmin.

    With every waypoint reached the last one is the only candidate.  Returns
    ``(None, costs)`` when no candidate is reachable.
    """
    n = len(waypoints)
    candidates = list(range(max(int(last_reached) + 1, 0), n)) or [n - 1]
    costs = {k: float(geodesic(position, waypoints[k]) + remaining_m[k]) for k in candidates}
    finite = {k: c for k, c in costs.items() if math.isfinite(c)}
    if not finite:
        return None, costs
    best = min(finite.values())
    return min(k for k, c in finite.items() if c <= best + float(slack_m)), costs


def route_points(
    position: np.ndarray,
    waypoints: np.ndarray,
    k_star: int,
    find_path: FindPath,
    segment_paths: Sequence["np.ndarray | None"] | None = None,
) -> np.ndarray | None:
    """``find_path(p, w_k*)`` followed by the reference segments to the end.

    ``segment_paths[k]`` (optional cache) is the path from ``w_k`` to ``w_{k+1}``.
    """
    parts = [find_path(np.asarray(position, dtype=np.float64), waypoints[k_star])]
    for k in range(k_star, len(waypoints) - 1):
        seg = segment_paths[k] if segment_paths is not None else find_path(waypoints[k], waypoints[k + 1])
        parts.append(seg)
    if any(part is None or not len(part) for part in parts):
        return None
    return concat_paths(parts)


# --------------------------------------------------------------------------- #
# The model's own answer
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelAnswer:
    kind: str  # one of MODEL_KINDS
    turn1: str
    turn2: str | None
    pixel: tuple[int, int] | None = None  # (u, v)
    turn_direction: int = 0  # +1 left, -1 right, 0 none / cancelling
    note: str = ""


def parse_native_tokens(text: str) -> list[str]:
    return _NATIVE_TOKEN_RE.findall(text or "")


def parse_model_answer(call: Mapping[str, Any]) -> ModelAnswer:
    """Classify a recorded call's answer from its texts, as the server executed it.

    The final text (the second turn when a lookdown turn ran) decides: digits ->
    pixel (``"u v"``, read from the text, never from the response's swapped
    ``pixel_goal``), first
    parsed token ``STOP`` -> stop, executed tokens all ``←``/``→`` -> turn,
    anything else (``↑``, a repeated ``↓``, no token: the server's
    ``fallback_stop``) -> other.
    """
    llm_output = str(call.get("llm_output") or "")
    first = call.get("native_first_output")
    lookdown_turns = int(call.get("native_lookdown_turns") or 0)
    if lookdown_turns >= 1:
        turn1 = str(first) if first else LOOKDOWN_TEXT
        turn2: str | None = llm_output
        final = llm_output
    else:
        turn1 = str(first) if first else llm_output
        turn2 = None
        final = turn1
    if re.search(r"\d", final):
        numbers = [int(x) for x in re.findall(r"\d+", final)]
        if len(numbers) < 2:
            return ModelAnswer("other", turn1, turn2, note="one_number")
        pixel = (numbers[0], numbers[1])
        note = "" if lookdown_turns >= 1 else "pixel_without_lookdown"
        return ModelAnswer("pixel", turn1, turn2, pixel=pixel, note=note)
    tokens = parse_native_tokens(final)
    if not tokens:
        return ModelAnswer("other", turn1, turn2, note="no_action_token")
    if tokens[0] == "STOP":
        return ModelAnswer("stop", turn1, turn2)
    executed = tokens[:MAX_EXECUTED_ACTIONS]
    if all(t in (LEFT_TEXT, RIGHT_TEXT) for t in executed):
        net = executed.count(LEFT_TEXT) - executed.count(RIGHT_TEXT)
        return ModelAnswer("turn", turn1, turn2, turn_direction=int(np.sign(net)))
    return ModelAnswer("other", turn1, turn2, note="non_turn_actions")


def pixel_text(u: int, v: int) -> str:
    """Native second-turn text for image position (u, v): ``"u v"`` (column first)."""
    return f"{int(u)} {int(v)}"


def turn_text(direction: int, count: int) -> str:
    return (LEFT_TEXT if direction > 0 else RIGHT_TEXT) * int(count)


# --------------------------------------------------------------------------- #
# Oracle answer and agreement
# --------------------------------------------------------------------------- #
@dataclass
class OracleAnswer:
    kind: str  # one of ORACLE_KINDS
    turn1: str | None = None
    turn2: str | None = None
    pixel: tuple[int, int] | None = None
    reason: str = ""
    debug: dict[str, Any] = field(default_factory=dict)


def rounded(x: Any, nd: int = 4) -> Any:
    """JSON-friendly copy of a float / array: rounded, non-finite -> None."""
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return [rounded(v, nd) for v in np.asarray(x, dtype=np.float64).tolist()]
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def visible_route_samples(
    route: np.ndarray,
    c2w: np.ndarray,
    depth_m: np.ndarray,
    camera: LookdownCamera,
    config: OracleConfig,
) -> dict[str, np.ndarray]:
    """Route samples, their pixels and which ones the lookdown actually sees.

    A sample is visible when its rounded pixel is inside the image, the depth
    there is a return (> 0) and the sample is not more than ``occlusion_tol_m``
    behind the surface the depth saw.  Depth saturates at the sensor's 10 m,
    so a sample beyond that is never visible (conservative).
    """
    if depth_m.shape != (camera.height, camera.width):
        raise ValueError(f"lookdown depth is {depth_m.shape}, camera is {(camera.height, camera.width)}")
    samples, arc = resample_polyline(route, config.sample_step_m)
    uv, z = camera.project(samples, c2w)
    finite = np.all(np.isfinite(uv), axis=1)
    col = np.full(len(samples), -1, dtype=np.int64)
    row = np.full(len(samples), -1, dtype=np.int64)
    col[finite] = np.floor(uv[finite, 0] + 0.5).astype(np.int64)
    row[finite] = np.floor(uv[finite, 1] + 0.5).astype(np.int64)
    inside = finite & (col >= 0) & (col < camera.width) & (row >= 0) & (row < camera.height)
    seen = np.zeros(len(samples))
    seen[inside] = depth_m[row[inside], col[inside]]
    visible = inside & (seen > 0.0) & (z <= seen + config.occlusion_tol_m)
    return {"samples": samples, "arc": arc, "col": col, "row": row, "z": z, "visible": visible}


def oracle_answer(
    *,
    position: Sequence[float],
    rotation_wxyz: Sequence[float],
    distance_to_goal: float,
    route: np.ndarray | None,
    depth_m: np.ndarray | None,
    config: OracleConfig = OracleConfig(),
    camera: LookdownCamera = LookdownCamera(),
) -> OracleAnswer:
    if not math.isfinite(float(distance_to_goal)):
        return OracleAnswer("skip", reason="goal_unreachable")
    if distance_to_goal <= config.stop_radius_m:
        return OracleAnswer("stop", STOP_TEXT, None, reason="within_stop_radius")
    if route is None or len(route) < 2:
        return OracleAnswer("skip", reason="no_route")
    if depth_m is None:
        return OracleAnswer("skip", reason="no_depth")
    c2w = camera.c2w(position, rotation_wxyz)
    vis = visible_route_samples(route, c2w, depth_m, camera, config)
    arc = vis["arc"]
    debug: dict[str, Any] = {
        "route_length_m": rounded(arc[-1]),
        "n_samples": int(len(arc)),
        "n_visible": int(vis["visible"].sum()),
    }
    usable = np.flatnonzero(vis["visible"] & (arc >= config.min_pixel_ahead_m))
    if len(usable):
        i = int(usable[np.argmax(arc[usable])])
        u, v = int(vis["col"][i]), int(vis["row"][i])
        debug.update(
            visible_arc_m=rounded(arc[i]),
            visible_point_world=rounded(vis["samples"][i]),
            visible_point_zdepth_m=rounded(vis["z"][i]),
        )
        return OracleAnswer("pixel", LOOKDOWN_TEXT, pixel_text(u, v), pixel=(u, v), reason="route_visible", debug=debug)
    ahead = np.flatnonzero(arc >= config.turn_target_ahead_m)
    j = int(ahead[0]) if len(ahead) else len(arc) - 1
    dyaw = heading_delta_deg(rotation_wxyz, position, vis["samples"][j])
    debug.update(turn_point_world=rounded(vis["samples"][j]), delta_yaw_deg=rounded(dyaw, 2))
    if abs(dyaw) < config.min_turn_deg:
        return OracleAnswer("skip", reason="route_ahead_but_occluded", debug=debug)
    count = int(min(max(math.floor(abs(dyaw) / config.turn_step_deg + 0.5), 1), config.max_turns))
    direction = 1 if dyaw > 0 else -1
    return OracleAnswer("turn", turn_text(direction, count), None, reason="route_not_visible", debug=debug)


def _turn_direction_of(text: str | None) -> int:
    tokens = parse_native_tokens(text or "")[:MAX_EXECUTED_ACTIONS]
    return int(np.sign(tokens.count(LEFT_TEXT) - tokens.count(RIGHT_TEXT)))


def judge_model_pixel(
    pixel: tuple[int, int],
    *,
    position: Sequence[float],
    rotation_wxyz: Sequence[float],
    route: np.ndarray | None,
    depth_m: np.ndarray | None,
    config: OracleConfig = OracleConfig(),
    camera: LookdownCamera = LookdownCamera(),
) -> tuple[bool | None, str, dict[str, Any]]:
    """Unproject the model's pixel with depth and test it against the oracle route."""
    if route is None or len(route) < 2:
        return None, "no_route", {}
    if depth_m is None:
        return None, "no_depth", {}
    u, v = int(pixel[0]), int(pixel[1])
    if not (0 <= u < camera.width and 0 <= v < camera.height):
        return False, "model_pixel_outside_image", {}
    z = float(depth_m[v, u])
    if z <= 0.0:
        return False, "model_pixel_no_depth", {}
    q = camera.unproject(u, v, z, camera.c2w(position, rotation_wxyz))
    dist, progress = nearest_on_polyline(route, q)
    debug = {"model_q_world": rounded(q), "model_q_route_dist_m": rounded(dist), "model_q_progress_m": rounded(progress)}
    if dist > config.agree_path_dist_m:
        return False, "model_pixel_off_route", debug
    if progress < config.agree_min_progress_m:
        return False, "model_pixel_no_progress", debug
    return True, "model_pixel_on_route", debug


def label_call(
    *,
    model: ModelAnswer,
    position: Sequence[float],
    rotation_wxyz: Sequence[float],
    distance_to_goal: float,
    route: np.ndarray | None,
    depth_m: np.ndarray | None,
    config: OracleConfig = OracleConfig(),
    camera: LookdownCamera = LookdownCamera(),
) -> dict[str, Any]:
    """Oracle answer, agreement and the SFT target of one call.

    Agreement (design §2 B.4): a model STOP agrees iff ``d <= success_radius_m``;
    any other answer inside ``stop_radius_m`` disagrees (target STOP); a pixel
    agrees iff its depth unprojection lies within ``agree_path_dist_m`` of the
    oracle route at least ``agree_min_progress_m`` along it; a turn agrees iff
    the oracle also turns, the same way.  ``other`` never agrees.

    Target: the model's own turns when it agrees, else the oracle's.  When the
    oracle is ``skip`` there is nothing to substitute, so a call yields a target
    only if a rule that does not need the oracle's answer accepts the model's
    (a STOP inside the success radius, a pixel on the route); otherwise
    ``agree`` and the targets are ``None``.
    """
    d = float(distance_to_goal)
    oracle = oracle_answer(
        position=position,
        rotation_wxyz=rotation_wxyz,
        distance_to_goal=d,
        route=route,
        depth_m=depth_m,
        config=config,
        camera=camera,
    )
    debug = dict(oracle.debug)
    agree: bool | None
    if model.kind == "stop":
        agree = math.isfinite(d) and d <= config.success_radius_m
        reason = "model_stop_within_success" if agree else "model_stop_too_far"
    elif oracle.kind == "stop":
        agree, reason = False, "model_did_not_stop"
    elif model.kind == "pixel":
        assert model.pixel is not None
        agree, reason, extra = judge_model_pixel(
            model.pixel,
            position=position,
            rotation_wxyz=rotation_wxyz,
            route=route,
            depth_m=depth_m,
            config=config,
            camera=camera,
        )
        debug.update(extra)
    elif model.kind == "turn":
        if oracle.kind == "turn":
            same = model.turn_direction != 0 and model.turn_direction == _turn_direction_of(oracle.turn1)
            agree, reason = same, ("turn_same_direction" if same else "turn_wrong_direction")
        elif oracle.kind == "pixel":
            agree, reason = False, "turned_with_route_in_view"
        else:
            agree, reason = None, "turn_unjudgeable"
    else:
        agree, reason = False, f"model_other:{model.note}"

    if agree:
        target = (model.turn1, model.turn2)
    elif oracle.kind == "skip":
        agree, target = None, (None, None)
        reason = f"oracle_skip:{oracle.reason}|{reason}"
    elif agree is None:
        target = (None, None)
    else:
        target = (oracle.turn1, oracle.turn2)
    return {
        "distance_to_goal": rounded(d),
        "oracle_kind": oracle.kind,
        "oracle_reason": oracle.reason,
        "oracle_turn1": oracle.turn1,
        "oracle_turn2": oracle.turn2,
        "oracle_pixel": list(oracle.pixel) if oracle.pixel is not None else None,
        "model_kind": model.kind,
        "model_turn1": model.turn1,
        "model_turn2": model.turn2,
        "model_pixel": list(model.pixel) if model.pixel is not None else None,
        "agree": agree,
        "reason": reason,
        "target_turn1": target[0],
        "target_turn2": target[1],
        "oracle_path": _route_for_row(route),
        "debug": debug,
    }


def _route_for_row(route: np.ndarray | None) -> list[list[float]] | None:
    """The oracle route as the offline gate G4 scores a pixel against it.

    Resampled every 0.25 m and cut at ``ORACLE_PATH_ROW_M``, so a label row stays
    small; G4 only asks whether a predicted pixel lands within ``agree_path_dist_m``
    of the route, and no lookdown pixel is farther away than the depth range.
    """
    if route is None or len(route) < 2:
        return None
    samples, arc = resample_polyline(np.asarray(route, dtype=np.float64), 0.25)
    keep = samples[arc <= ORACLE_PATH_ROW_M]
    return [[round(float(c), 3) for c in point] for point in keep]


__all__ = [
    "LEFT_TEXT",
    "LOOKDOWN_TEXT",
    "LookdownCamera",
    "MAX_EXECUTED_ACTIONS",
    "MODEL_KINDS",
    "ModelAnswer",
    "ORACLE_KINDS",
    "OracleAnswer",
    "OracleConfig",
    "RIGHT_TEXT",
    "STOP_TEXT",
    "choose_waypoint",
    "concat_paths",
    "heading_delta_deg",
    "judge_model_pixel",
    "label_call",
    "last_reached_by_step",
    "nearest_on_polyline",
    "oracle_answer",
    "parse_model_answer",
    "parse_native_tokens",
    "pixel_text",
    "polyline_arclength",
    "quat_wxyz_to_rot",
    "remaining_reference_m",
    "resample_polyline",
    "rounded",
    "route_points",
    "route_waypoints",
    "turn_text",
    "visible_route_samples",
    "yaw_to_quat_wxyz",
]

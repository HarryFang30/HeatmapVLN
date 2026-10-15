"""Route-correction labels for EXP-25 line B: states more than 3 m from the goal only.

Round 1 (EXP-23) lost 13 pt because its oracle contradicted the released model's own
training convention (InternData-N1 r2r, read by ``interndata_replay``): STOP on the
approach, turn directions the model cannot see (34% opposite to the turn it had just
executed, against 0.81% for the model itself, so closed-loop reversals went from 0.35%
to 45% of turn calls), pixel targets beyond 3 m (35%) and on the image's side edges (53%
against 0.7%).  Line B keeps the round-1 route (reference path, rejoined at the earliest
waypoint within ``rejoin_slack_m`` of the cheapest, ``s2_oracle_labels``) and replaces
every rule that broke:

* **No STOP at all.**  A state within ``label_min_distance_m`` (3 m) of the goal is not
  labelled (``skip``); line A (SpinBreak) owns stopping.  Beyond 3 m the oracle walks on.
* **Pixel goal = InternData's.**  The farthest route sample that the lookdown sees
  (depth-checked) and that lies at most ``pixel_max_ahead_m`` (3 m) along the route and
  ``pixel_max_euclid_m`` (3 m) from the agent, at least ``pixel_min_ahead_m`` along
  (InternNav skips goals fewer than 3 frames ahead), inside ``u`` in [100, 540] and
  ``v`` in [32, 455].  InternData-N1's goals: ahead median 2.62 m, p95 2.99 m; u p5-p95
  156-480 with 0.7% outside [100, 540].
* **Turns never flip unseen.**  No usable pixel: aim at the first route sample
  ``turn_target_ahead_m`` along, ``dyaw`` its signed bearing (left positive).

  - ``|dyaw| < turn_step / 2``: the route is ahead but hidden -- no answer (``skip``).
  - the route is seen within 3 m but outside the pixel window (side band): the shorter
    rotation towards it (the direction is in the image);
  - otherwise the route is behind / hidden.  If the previous call executed a turn, keep
    turning that way (hysteresis; ``hysteresis_max_flip_deg`` > 0 lets a flip through
    when the shorter rotation is that small, default never);  else if ``|dyaw| >=
    behind_deg`` turn ``behind_direction`` (left: the released model's habit, both ways
    cost the same to within one call); else the shorter rotation.

  ``k = clamp(round(a / 15), 1, 4)`` arrows for the angle ``a`` the chosen direction has
  to cover, as InternData's expert (1-4 arrows, then forward).
* **Agreement = is the model's own answer making progress.**

  - pixel: its target is the lookdown ray's floor point, cut short by the recorded depth
    when the ray hits something first, snapped to the navmesh; the answer agrees when
    ``(g - g_T) / max(t, 0.25) >= agree_progress_ratio`` (0.5, the EXP-25 research's
    "good"), ``g`` / ``g_T`` the geodesic to the goal from the agent / the target, ``t``
    the horizontal distance to the target.  A ray that hits nothing (above the horizon,
    no depth return) is a pixel without a target: disagree.
  - turn: agrees when it turns towards the route (the oracle pixel's point when the route
    is in view, at least ``turn_step / 2`` off-axis; the aim point otherwise) or the way
    the deterministic rule turns, or either way when the route is ``behind_deg`` behind.
  - a STOP beyond 3 m never agrees; nor does any other answer (``↑``, unparsable).

  Agreeing calls keep the model's own answer as the target (self-distillation, both turns
  when it looked down); a disagreeing call takes the oracle's answer; when the oracle has
  none (``skip``) the call has no target.

Everything is numpy; geodesics and navmesh snapping are injected callables so the Habitat
side (``scripts/tools/label_route_corrections.py``) stays thin and the rules test on a flat
synthetic world.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .s2_oracle_labels import (
    LOOKDOWN_TEXT,
    MAX_EXECUTED_ACTIONS,
    ORACLE_PATH_ROW_M,
    LookdownCamera,
    ModelAnswer,
    OracleAnswer,
    heading_delta_deg,
    parse_native_tokens,
    pixel_text,
    resample_polyline,
    rounded,
    turn_text,
    visible_route_samples,
)
from .s2_records import CAMERA

# Bumped whenever a rule changes, so the config digest -- and the labeller's "already
# current" check -- moves even when no value does.
ROUTE_RULES_VERSION = 1
LABEL_LINE = "B"

Geodesic = Callable[[np.ndarray, np.ndarray], float]
Snap = Callable[[np.ndarray], "np.ndarray | None"]

ROW_KINDS = ("correction", "anchor", "none")


@dataclass(frozen=True)
class RouteLabelConfig:
    label_min_distance_m: float = 3.0  # states at or inside this are line A's (no label)
    # The route (as round 1: s2_oracle_labels.choose_waypoint / route_points).
    reached_radius_m: float = 1.0
    rejoin_slack_m: float = 2.0
    sample_step_m: float = 0.1
    occlusion_tol_m: float = 0.3
    # Pixel goals: InternData-N1's support.
    pixel_min_ahead_m: float = 0.75
    pixel_max_ahead_m: float = 3.0
    pixel_max_euclid_m: float = 3.0
    pixel_u_min: int = 100
    pixel_u_max: int = 540
    pixel_v_min: int = 32
    pixel_v_max: int = 455
    # Turns.
    turn_target_ahead_m: float = 0.5
    turn_step_deg: float = 15.0
    max_turns: int = MAX_EXECUTED_ACTIONS
    hysteresis: bool = True
    hysteresis_max_flip_deg: float = 0.0  # 0: never flip against the executed turn unseen
    behind_deg: float = 150.0
    behind_direction: int = 1  # +1 left, -1 right, 0: the shorter rotation
    # Agreement.
    agree_progress_ratio: float = 0.5
    progress_min_tdist_m: float = 0.25

    @property
    def min_turn_deg(self) -> float:
        return self.turn_step_deg / 2.0

    def digest(self) -> str:
        text = json.dumps({"route_rules": ROUTE_RULES_VERSION, **asdict(self)}, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def executed_turn_direction(answer: ModelAnswer | None) -> int:
    """+1 / -1 when ``answer`` executed a net left / right turn, else 0."""
    if answer is None or answer.kind != "turn":
        return 0
    return int(answer.turn_direction)


def turn_count(angle_deg: float, config: RouteLabelConfig) -> int:
    """Arrows for a rotation of ``angle_deg`` (>= 0): round to 15 deg steps, 1..max_turns."""
    return int(min(max(math.floor(float(angle_deg) / config.turn_step_deg + 0.5), 1), config.max_turns))


def angle_in_direction(dyaw_deg: float, direction: int) -> float:
    """Rotation (deg, [0, 360)) that brings bearing ``dyaw_deg`` ahead turning ``direction``."""
    a = float(dyaw_deg) if direction > 0 else -float(dyaw_deg)
    return a + 360.0 if a < 0.0 else a


def _horizontal(a: Sequence[float], b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    return np.hypot(b[:, 0] - float(a[0]), b[:, 2] - float(a[2]))


# --------------------------------------------------------------------------- #
# Oracle
# --------------------------------------------------------------------------- #
def route_oracle(
    *,
    position: Sequence[float],
    rotation_wxyz: Sequence[float],
    route: np.ndarray | None,
    depth_m: np.ndarray | None,
    previous_turn_direction: int = 0,
    config: RouteLabelConfig = RouteLabelConfig(),
    camera: LookdownCamera = LookdownCamera(),
) -> OracleAnswer:
    """Pixel goal, turn or skip for a state beyond the line-A radius (never STOP)."""
    if route is None or len(route) < 2:
        return OracleAnswer("skip", reason="no_route")
    if depth_m is None:
        return OracleAnswer("skip", reason="no_depth")
    c2w = camera.c2w(position, rotation_wxyz)
    vis = visible_route_samples(route, c2w, depth_m, camera, config)  # type: ignore[arg-type]
    arc, samples, col, row = vis["arc"], vis["samples"], vis["col"], vis["row"]
    euclid = _horizontal(position, samples)
    eps = 1e-6  # resampled arc lengths carry float noise (0.1 * 30 > 3.0)
    near = (arc <= config.pixel_max_ahead_m + eps) & (euclid <= config.pixel_max_euclid_m + eps)
    in_window = (
        (col >= config.pixel_u_min)
        & (col <= config.pixel_u_max)
        & (row >= config.pixel_v_min)
        & (row <= config.pixel_v_max)
    )
    visible = vis["visible"]
    usable = np.flatnonzero(visible & near & in_window & (arc >= config.pixel_min_ahead_m - eps))
    debug: dict[str, Any] = {
        "route_length_m": rounded(arc[-1]),
        "n_visible": int(visible.sum()),
        "n_visible_near": int((visible & near).sum()),
        "n_usable": int(len(usable)),
    }
    if len(usable):
        i = int(usable[np.argmax(arc[usable])])
        u, v = int(col[i]), int(row[i])
        debug.update(
            rule="pixel",
            pixel_ahead_arc_m=rounded(arc[i]),
            pixel_ahead_euclid_m=rounded(euclid[i]),
            pixel_point_world=rounded(samples[i]),
            pixel_dyaw_deg=rounded(heading_delta_deg(rotation_wxyz, position, samples[i]), 2),
        )
        return OracleAnswer("pixel", LOOKDOWN_TEXT, pixel_text(u, v), pixel=(u, v), reason="route_visible", debug=debug)

    ahead = np.flatnonzero(arc >= config.turn_target_ahead_m)
    j = int(ahead[0]) if len(ahead) else len(arc) - 1
    dyaw = heading_delta_deg(rotation_wxyz, position, samples[j])
    shorter = 1 if dyaw > 0 else -1
    debug.update(turn_point_world=rounded(samples[j]), dyaw_deg=rounded(dyaw, 2), shorter_direction=shorter)
    if abs(dyaw) < config.min_turn_deg:
        debug["rule"] = "skip_ahead_hidden"
        return OracleAnswer("skip", reason="route_ahead_but_not_visible", debug=debug)
    seen = bool((visible & near).any())
    prev = int(np.sign(previous_turn_direction))
    if seen:
        direction, rule = shorter, "seen_outside_window"
    elif config.hysteresis and prev != 0 and not (
        shorter != prev and abs(dyaw) <= config.hysteresis_max_flip_deg
    ):
        direction, rule = prev, "hysteresis"
    elif config.behind_direction != 0 and abs(dyaw) >= config.behind_deg:
        direction, rule = int(np.sign(config.behind_direction)), "behind_tiebreak"
    else:
        direction, rule = shorter, "shorter_rotation"
    count = turn_count(angle_in_direction(dyaw, direction), config)
    debug.update(rule=rule, turn_direction=direction, turn_count=count)
    reason = "route_seen_outside_window" if seen else "route_not_visible"
    return OracleAnswer("turn", turn_text(direction, count), None, reason=reason, debug=debug)


# --------------------------------------------------------------------------- #
# Progress of a pixel answer
# --------------------------------------------------------------------------- #
def pixel_floor_target(
    pixel: tuple[int, int],
    *,
    position: Sequence[float],
    rotation_wxyz: Sequence[float],
    depth_m: np.ndarray | None,
    camera: LookdownCamera = LookdownCamera(),
) -> tuple[np.ndarray | None, str]:
    """Where System1 is sent: the ray's floor point, or the surface the depth saw first.

    The floor is the agent's own navmesh height.  The ray is parameterised by z-depth
    (camera-frame direction with unit -z), so the recorded depth cuts it directly; a
    saturated depth (the sensor's max) is no return.  Returns ``(point or None, how)``.
    """
    u, v = int(pixel[0]), int(pixel[1])
    if not (0 <= u < camera.width and 0 <= v < camera.height):
        return None, "outside_image"
    c2w = camera.c2w(position, rotation_wxyz)
    f = camera.focal
    cx, cy = camera.centre
    direction = c2w[:3, :3] @ np.array([(u - cx) / f, -(v - cy) / f, -1.0])
    origin = c2w[:3, 3]
    floor_y = float(position[1])
    t_floor = (floor_y - origin[1]) / direction[1] if direction[1] < -1e-9 else math.inf
    t_depth = math.inf
    if depth_m is not None:
        z = float(depth_m[v, u])
        if 0.0 < z < float(CAMERA["depth_max_m"]) - 1e-3:
            t_depth = z
    t = min(t_floor, t_depth)
    if not math.isfinite(t):
        return None, "no_floor_no_depth"
    point = origin + t * direction
    point[1] = floor_y
    return point, ("floor" if t_floor <= t_depth else "depth")


def pixel_progress(
    pixel: tuple[int, int],
    *,
    position: Sequence[float],
    rotation_wxyz: Sequence[float],
    depth_m: np.ndarray | None,
    distance_to_goal: float,
    goal: Sequence[float],
    geodesic: Geodesic,
    snap: Snap,
    config: RouteLabelConfig = RouteLabelConfig(),
    camera: LookdownCamera = LookdownCamera(),
) -> dict[str, Any]:
    """``{"status": "ok" | "no_target" | "unjudgeable", ...}`` with the progress ratio when ok."""
    target, how = pixel_floor_target(
        pixel, position=position, rotation_wxyz=rotation_wxyz, depth_m=depth_m, camera=camera
    )
    if target is None:
        return {"status": "no_target", "how": how}
    snapped = snap(target)
    if snapped is None or not np.all(np.isfinite(np.asarray(snapped, dtype=np.float64))):
        return {"status": "unjudgeable", "how": how, "why": "snap_failed"}
    snapped = np.asarray(snapped, dtype=np.float64).reshape(3)
    g_target = float(geodesic(snapped, np.asarray(goal, dtype=np.float64)))
    if not math.isfinite(g_target):
        return {"status": "unjudgeable", "how": how, "why": "target_unreachable"}
    tdist = float(_horizontal(position, target)[0])
    ratio = (float(distance_to_goal) - g_target) / max(tdist, config.progress_min_tdist_m)
    return {
        "status": "ok",
        "how": how,
        "g": rounded(distance_to_goal),
        "g_target": rounded(g_target),
        "tdist_m": rounded(tdist),
        "snap_m": rounded(float(np.linalg.norm(snapped - target))),
        "ratio": rounded(ratio),
        "target_world": rounded(target),
    }


# --------------------------------------------------------------------------- #
# One call
# --------------------------------------------------------------------------- #
def _route_for_row(route: np.ndarray | None) -> list[list[float]] | None:
    """The oracle route, resampled every 0.25 m and cut at 15 m (gate G4 reads it)."""
    if route is None or len(route) < 2:
        return None
    samples, arc = resample_polyline(np.asarray(route, dtype=np.float64), 0.25)
    return [[round(float(c), 3) for c in point] for point in samples[arc <= ORACLE_PATH_ROW_M]]


def _judge_turn(model: ModelAnswer, oracle: OracleAnswer, config: RouteLabelConfig) -> tuple[bool | None, str]:
    direction = int(model.turn_direction)
    if direction == 0:
        return False, "turn_cancels_out"
    if oracle.kind == "pixel":
        dyaw = float(oracle.debug["pixel_dyaw_deg"])
        if abs(dyaw) >= config.min_turn_deg and direction == (1 if dyaw > 0 else -1):
            return True, "turn_toward_visible_route"
        return False, "turned_away_from_visible_route"
    if oracle.kind == "turn":
        dyaw = float(oracle.debug["dyaw_deg"])
        if direction == int(oracle.debug["turn_direction"]):
            return True, "turn_matches_rule"
        if direction == int(oracle.debug["shorter_direction"]):
            return True, "turn_toward_route"
        if abs(dyaw) >= config.behind_deg:
            return True, "turn_route_behind_either_way"
        return False, "turn_wrong_direction"
    return None, "turn_unjudgeable"


def label_route_call(
    *,
    model: ModelAnswer,
    previous_model: ModelAnswer | None,
    position: Sequence[float],
    rotation_wxyz: Sequence[float],
    distance_to_goal: float,
    goal: Sequence[float],
    route: np.ndarray | None,
    depth_m: np.ndarray | None,
    geodesic: Geodesic,
    snap: Snap,
    config: RouteLabelConfig = RouteLabelConfig(),
    camera: LookdownCamera = LookdownCamera(),
) -> dict[str, Any]:
    """Oracle answer, agreement, SFT target and row kind of one call (see the module doc)."""
    d = float(distance_to_goal)
    previous_direction = executed_turn_direction(previous_model)
    row: dict[str, Any] = {
        "line": LABEL_LINE,
        "distance_to_goal": rounded(d),
        "model_kind": model.kind,
        "model_turn1": model.turn1,
        "model_turn2": model.turn2,
        "model_pixel": list(model.pixel) if model.pixel is not None else None,
        "model_turn_direction": int(model.turn_direction),
        "previous_model_kind": previous_model.kind if previous_model is not None else None,
        "previous_turn_direction": previous_direction,
    }
    empty = {
        "oracle_kind": "skip",
        "oracle_turn1": None,
        "oracle_turn2": None,
        "oracle_pixel": None,
        "agree": None,
        "target_turn1": None,
        "target_turn2": None,
        "kind": "none",
        "oracle_path": None,
        "debug": {},
    }
    if not math.isfinite(d):
        return {**row, **empty, "skip": True, "oracle_reason": "goal_unreachable", "reason": "goal_unreachable"}
    if d <= config.label_min_distance_m:
        return {**row, **empty, "skip": True, "oracle_reason": "within_line_a_radius", "reason": "within_line_a_radius"}

    oracle = route_oracle(
        position=position,
        rotation_wxyz=rotation_wxyz,
        route=route,
        depth_m=depth_m,
        previous_turn_direction=previous_direction,
        config=config,
        camera=camera,
    )
    debug: dict[str, Any] = dict(oracle.debug)
    if oracle.kind == "pixel":
        point = np.asarray(oracle.debug["pixel_point_world"], dtype=np.float64)
        snapped = snap(point)
        if snapped is not None and np.all(np.isfinite(np.asarray(snapped, dtype=np.float64))):
            g_point = float(geodesic(np.asarray(snapped, dtype=np.float64), np.asarray(goal, dtype=np.float64)))
            tdist = float(_horizontal(position, point)[0])
            if math.isfinite(g_point):
                debug["oracle_progress_ratio"] = rounded((d - g_point) / max(tdist, config.progress_min_tdist_m))

    agree: bool | None
    if model.kind == "stop":
        agree, reason = False, "model_stop_beyond_radius"
    elif model.kind == "pixel":
        assert model.pixel is not None
        progress = pixel_progress(
            model.pixel,
            position=position,
            rotation_wxyz=rotation_wxyz,
            depth_m=depth_m,
            distance_to_goal=d,
            goal=goal,
            geodesic=geodesic,
            snap=snap,
            config=config,
            camera=camera,
        )
        debug["model_progress"] = progress
        if progress["status"] == "ok":
            good = float(progress["ratio"]) >= config.agree_progress_ratio
            agree, reason = good, ("model_pixel_progress" if good else "model_pixel_no_progress")
        elif progress["status"] == "no_target":
            agree, reason = False, f"model_pixel_no_target:{progress['how']}"
        else:
            agree, reason = None, f"model_pixel_unjudgeable:{progress['why']}"
    elif model.kind == "turn":
        agree, reason = _judge_turn(model, oracle, config)
    else:
        agree, reason = False, f"model_other:{model.note}"

    if agree:
        target: tuple[str | None, str | None] = (model.turn1, model.turn2)
        kind = "anchor"
    elif agree is False and oracle.kind != "skip":
        target, kind = (oracle.turn1, oracle.turn2), "correction"
    elif agree is False:
        agree, target, kind = None, (None, None), "none"
        reason = f"oracle_skip:{oracle.reason}|{reason}"
    else:
        target, kind = (None, None), "none"
    return {
        **row,
        "skip": False,
        "oracle_kind": oracle.kind,
        "oracle_reason": oracle.reason,
        "oracle_turn1": oracle.turn1,
        "oracle_turn2": oracle.turn2,
        "oracle_pixel": list(oracle.pixel) if oracle.pixel is not None else None,
        "agree": agree,
        "reason": reason,
        "target_turn1": target[0],
        "target_turn2": target[1],
        "kind": kind,
        "oracle_path": _route_for_row(route),
        "debug": debug,
    }


def answer_direction(turn1: str | None, turn2: str | None) -> int:
    """Net executed turn of a recorded / labelled answer's final turn (0: not a pure turn)."""
    final = turn2 if turn2 is not None else turn1
    if final is None or any(ch.isdigit() for ch in final):
        return 0
    tokens = parse_native_tokens(final)[:MAX_EXECUTED_ACTIONS]
    if not tokens or any(t not in ("←", "→") for t in tokens):
        return 0
    return int(np.sign(tokens.count("←") - tokens.count("→")))


def summary_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """The fields the statistics read, from a line-B row or a round-1 (``s2_oracle_labels``) row."""
    debug = row.get("debug") or {}
    arc = debug.get("pixel_ahead_arc_m", debug.get("visible_arc_m"))
    euclid = debug.get("pixel_ahead_euclid_m")
    if euclid is None and debug.get("visible_point_world") is not None and debug.get("position") is not None:
        p, q = debug["position"], debug["visible_point_world"]
        euclid = math.hypot(q[0] - p[0], q[2] - p[2])
    target1, target2 = row.get("target_turn1"), row.get("target_turn2")
    if row.get("kind") in ROW_KINDS:
        kind = row["kind"]
    elif row.get("agree") is True and target1 is not None:
        kind = "anchor"
    elif row.get("agree") is False and target1 is not None:
        kind = "correction"
    else:
        kind = "none"
    progress = debug.get("model_progress") or {}
    return {
        "d": row.get("distance_to_goal"),
        "skip": bool(row.get("skip")),
        "kind": kind,
        "agree": row.get("agree"),
        "reason": str(row.get("reason") or "").split("|")[0].split(":")[0],
        "model_kind": row.get("model_kind"),
        "model_direction": answer_direction(row.get("model_turn1"), row.get("model_turn2")),
        "oracle_kind": row.get("oracle_kind"),
        "oracle_pixel": row.get("oracle_pixel"),
        "oracle_turn1": row.get("oracle_turn1"),
        "oracle_direction": answer_direction(row.get("oracle_turn1"), row.get("oracle_turn2")),
        "oracle_arrows": len([c for c in (row.get("oracle_turn1") or "") if c in "←→"]),
        "oracle_rule": debug.get("rule"),
        "oracle_dyaw": debug.get("dyaw_deg", debug.get("delta_yaw_deg")),
        "oracle_progress_ratio": debug.get("oracle_progress_ratio"),
        "ahead_arc": arc,
        "ahead_euclid": euclid,
        "target1": target1,
        "target2": target2,
        "target_direction": answer_direction(target1, target2) if target1 is not None else 0,
        "target_is_pixel": target2 is not None and any(ch.isdigit() for ch in target2),
        "target_is_stop": target1 is not None and target1.strip() == "STOP",
        "model_progress_ratio": progress.get("ratio"),
    }


__all__ = [
    "LABEL_LINE",
    "ROUTE_RULES_VERSION",
    "ROW_KINDS",
    "RouteLabelConfig",
    "angle_in_direction",
    "answer_direction",
    "executed_turn_direction",
    "label_route_call",
    "pixel_floor_target",
    "pixel_progress",
    "route_oracle",
    "summary_view",
    "turn_count",
]

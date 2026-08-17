"""Training-free Heatmap-Conditioned Conservative Trajectory Filter.

The module is intentionally independent from Habitat and the model server.  It
turns already-postprocessed native ``TreatmentSpec`` candidates into explicit
candidate/history geometry and applies a conservative, one-step veto rule.

Only deployable inputs are accepted here:

* finalized local action chunks and their native sample mass;
* executed action/odometry history;
* fixed history relative poses;
* frozen heatmap spatial and visibility probabilities.

Goal positions, reference paths, and Habitat outcomes belong in the audit
script as labels only; they must never enter these policy functions.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
VIEW_CENTERS_RAD = np.asarray(
    (0.0, -math.pi / 2.0, math.pi, math.pi / 2.0), dtype=np.float32
)


def _wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(value), np.cos(value))


def _strip_batch(value: np.ndarray, expected_ndim: int) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim == expected_ndim + 1 and result.shape[0] == 1:
        result = result[0]
    if result.ndim != expected_ndim:
        raise ValueError(
            f"expected rank {expected_ndim} (optionally batched), got {result.shape}"
        )
    return result


def action_distribution_entropy(
    base_treatment_ids: Sequence[str], *, sample_total: int | None = None
) -> dict[str, float | int]:
    """Entropy of finalized full action chunks from native diffusion samples.

    Entropy is normalized by ``log(sample_total)``.  This keeps the quantity in
    ``[0,1]`` and, unlike normalization by the observed support size, retains
    the distinction between two equally likely modes and 32 unique samples.
    """

    ids = [str(value) for value in base_treatment_ids]
    if not ids:
        return {
            "entropy_nats": 0.0,
            "normalized_entropy": 0.0,
            "unique_chunks": 0,
            "max_mass": 0.0,
        }
    total = int(sample_total if sample_total is not None else len(ids))
    if total != len(ids) or total <= 0:
        raise ValueError("sample_total must equal the number of sample treatment ids")
    probabilities = np.asarray(
        [count / total for count in Counter(ids).values()], dtype=np.float64
    )
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    denominator = math.log(total) if total > 1 else 1.0
    return {
        "entropy_nats": entropy,
        "normalized_entropy": float(entropy / denominator),
        "unique_chunks": len(probabilities),
        "max_mass": float(probabilities.max()),
    }


def simulate_actions(
    actions: Sequence[int],
    *,
    forward_step_m: float = 0.25,
    turn_deg: float = 15.0,
    interpolation_step_m: float = 0.125,
) -> dict[str, np.ndarray | float]:
    """Kinematically simulate a finalized local action chunk.

    ``forward`` is positive X and ``left`` is positive Y, matching
    ``fixed_history_rel_poses[..., :2]``.  The simulator-free path is used by
    the policy; stored Habitat pose traces are audit labels only.
    """

    forward = 0.0
    left = 0.0
    yaw = 0.0
    points: list[tuple[float, float]] = [(forward, left)]
    turn = math.radians(float(turn_deg))
    subdivisions = max(1, int(math.ceil(forward_step_m / interpolation_step_m)))
    for raw_action in actions:
        action = int(raw_action)
        if action == ACTION_STOP:
            break
        if action == ACTION_LEFT:
            yaw += turn
        elif action == ACTION_RIGHT:
            yaw -= turn
        elif action == ACTION_FORWARD:
            dx = forward_step_m * math.cos(yaw) / subdivisions
            dy = forward_step_m * math.sin(yaw) / subdivisions
            for _ in range(subdivisions):
                forward += dx
                left += dy
                points.append((forward, left))
        else:
            raise ValueError(f"unsupported local action {action}")
    return {
        "points": np.asarray(points, dtype=np.float32),
        "final_yaw": float(_wrap_angle(yaw)),
        "endpoint": np.asarray((forward, left), dtype=np.float32),
    }


def action_edit_fraction(left: Sequence[int], right: Sequence[int]) -> float:
    """Padded Hamming distance between two at-most-four-step chunks."""

    left_values = [int(value) for value in left]
    right_values = [int(value) for value in right]
    width = max(1, min(4, max(len(left_values), len(right_values))))
    pad = 4
    left_padded = (left_values + [pad] * width)[:width]
    right_padded = (right_values + [pad] * width)[:width]
    return float(sum(a != b for a, b in zip(left_padded, right_padded)) / width)


def _bilinear_sample(grid: np.ndarray, x_normalized: float, y_normalized: float) -> float:
    height, width = grid.shape
    x = float(np.clip((x_normalized + 1.0) * 0.5 * (width - 1), 0, width - 1))
    y = float(np.clip((y_normalized + 1.0) * 0.5 * (height - 1), 0, height - 1))
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(width - 1, x0 + 1)
    y1 = min(height - 1, y0 + 1)
    wx = x - x0
    wy = y - y0
    return float(
        (1.0 - wx) * (1.0 - wy) * grid[y0, x0]
        + wx * (1.0 - wy) * grid[y0, x1]
        + (1.0 - wx) * wy * grid[y1, x0]
        + wx * wy * grid[y1, x1]
    )


def bearing_to_view_coordinate(bearing_rad: float) -> tuple[int, float]:
    """Map a panoramic bearing to ``(view_index, normalized_x)``.

    The sign follows the trained tokenizer: rightward image coordinates imply
    a more negative yaw, hence ``bearing = view_center - x*pi/4``.
    """

    deltas = np.asarray(_wrap_angle(float(bearing_rad) - VIEW_CENTERS_RAD))
    view = int(np.argmin(np.abs(deltas)))
    x_normalized = float(
        np.clip((VIEW_CENTERS_RAD[view] - bearing_rad) / (math.pi / 4.0), -1.0, 1.0)
    )
    return view, x_normalized


def _projected_path_points(actions: Sequence[int]) -> tuple[np.ndarray, float]:
    simulated = simulate_actions(actions)
    points = np.asarray(simulated["points"], dtype=np.float32)
    final_yaw = float(simulated["final_yaw"])
    # The start point has undefined bearing and is not a future revisit.  A
    # virtual facing point keeps turn-only candidates geometrically visible.
    usable = points[1:]
    if len(usable) == 0:
        usable = np.asarray(
            [[0.25 * math.cos(final_yaw), 0.25 * math.sin(final_yaw)]],
            dtype=np.float32,
        )
    return usable, final_yaw


def _heatmap_alignment_for_history(
    points: np.ndarray,
    *,
    coarse_probability: np.ndarray,
    spatial_statistics: np.ndarray,
    view_probability: np.ndarray,
    none_probability: float,
) -> float:
    """Candidate raster/heatmap overlap for one history observation."""

    if coarse_probability.shape[:1] != (4,) or spatial_statistics.shape != (4, 7):
        raise ValueError("heatmap context must contain four views")
    best = 0.0
    # ``view_probability`` is already the four non-none entries of the joint
    # five-way softmax, so its sum equals ``1 - p(none)``.  Multiplying by
    # ``1 - p(none)`` again would incorrectly square visibility confidence.
    _ = none_probability
    for point in points:
        distance = float(np.linalg.norm(point))
        if distance <= 1e-6:
            continue
        bearing = math.atan2(float(point[1]), float(point[0]))
        view, x_normalized = bearing_to_view_coordinate(bearing)
        probability = max(0.0, float(view_probability[view]))
        if probability <= 0.0:
            continue

        # Multimodal-safe raster score.  Uniform per-view probability has a
        # likelihood ratio of one and therefore contributes zero evidence.
        grid = np.asarray(coarse_probability[view], dtype=np.float64)
        likelihood_ratio = _bilinear_sample(grid, x_normalized, 0.0) * grid.size
        raster_signal = float(np.clip((likelihood_ratio - 1.0) / 3.0, 0.0, 1.0))

        mean_x, mean_y, var_x, var_y, cov_xy, entropy, _peak = map(
            float, spatial_statistics[view]
        )
        var_x = max(var_x, 1e-3)
        var_y = max(var_y, 1e-3)
        max_cov = math.sqrt(var_x * var_y) * 0.95
        cov_xy = float(np.clip(cov_xy, -max_cov, max_cov))
        determinant = max(var_x * var_y - cov_xy * cov_xy, 1e-6)
        dx = x_normalized - mean_x
        dy = -mean_y
        mahalanobis = (
            var_y * dx * dx - 2.0 * cov_xy * dx * dy + var_x * dy * dy
        ) / determinant
        concentration = float(np.clip((1.0 - entropy) / 0.35, 0.0, 1.0))
        moment_signal = math.exp(-0.5 * max(0.0, mahalanobis)) * concentration
        spatial_signal = 0.5 * raster_signal + 0.5 * moment_signal
        best = max(best, probability * spatial_signal)
    return float(np.clip(best, 0.0, 1.0))


def candidate_history_features(
    actions: Sequence[int],
    context: Mapping[str, np.ndarray],
    *,
    pose_sigma_m: float = 0.45,
) -> dict[str, Any]:
    """Explicit per-history interactions plus aggregate recent-history risk.

    The per-history values are kept because a history observation changes
    meaning between normal and recovery modes.  In normal mode the recent
    slots are obstacles to repetition; in recovery mode one older slot is an
    explicit anchor while only the observations collected after entering the
    loop are penalized.
    """

    mask = _strip_batch(np.asarray(context["fixed_history_mask"]), 1).astype(bool)
    rel = _strip_batch(np.asarray(context["fixed_history_rel_poses"]), 2).astype(
        np.float32
    )
    if rel.shape != (len(mask), 4):
        raise ValueError(f"relative pose/mask mismatch: {rel.shape} vs {mask.shape}")
    points, _final_yaw = _projected_path_points(actions)
    history_points = rel[:, :2]

    rank_value = context.get("history_rank")
    if rank_value is not None:
        rank = _strip_batch(np.asarray(rank_value), 1).astype(np.float32)
    else:
        valid_count = int(mask.sum())
        rank = np.zeros(len(mask), dtype=np.float32)
        if valid_count > 1:
            rank[np.flatnonzero(mask)] = np.linspace(0.0, 1.0, valid_count)
    recent_weight = 0.15 + 0.85 * np.square(np.clip(rank, 0.0, 1.0))

    coarse = context.get("coarse_probabilities")
    stats = context.get("spatial_statistics")
    views = context.get("view_probabilities")
    none = context.get("none_probability")
    heatmap_available = all(value is not None for value in (coarse, stats, views, none))
    if heatmap_available:
        coarse = _strip_batch(np.asarray(coarse), 4).astype(np.float32)
        stats = _strip_batch(np.asarray(stats), 3).astype(np.float32)
        views = _strip_batch(np.asarray(views), 2).astype(np.float32)
        none = _strip_batch(np.asarray(none), 1).astype(np.float32)
        expected = (len(mask), 4)
        if coarse.shape[:2] != expected or stats.shape[:2] != expected:
            raise ValueError("heatmap/history shape mismatch")
        if views.shape != expected or none.shape != (len(mask),):
            raise ValueError("visibility/history shape mismatch")

    raw_pose_by_history = np.zeros(len(mask), dtype=np.float32)
    raw_heatmap_by_history = np.zeros(len(mask), dtype=np.float32)
    pose_by_history = np.zeros(len(mask), dtype=np.float32)
    heatmap_by_history = np.zeros(len(mask), dtype=np.float32)
    hybrid_by_history = np.zeros(len(mask), dtype=np.float32)
    for index in np.flatnonzero(mask):
        distances = np.linalg.norm(points - history_points[index][None, :], axis=1)
        proximity = math.exp(-0.5 * (float(distances.min()) / pose_sigma_m) ** 2)
        weighted_pose = float(recent_weight[index]) * proximity
        alignment = 0.0
        if heatmap_available:
            alignment = _heatmap_alignment_for_history(
                points,
                coarse_probability=coarse[index],
                spatial_statistics=stats[index],
                view_probability=views[index],
                none_probability=float(none[index]),
            )
        weighted_heatmap = float(recent_weight[index]) * alignment
        # Pose supplies the otherwise-unobservable range; heatmap supplies the
        # candidate-specific image-space correspondence.  The floor prevents
        # one poor heatmap from completely erasing a true geometric revisit.
        hybrid = weighted_pose * (0.20 + 0.80 * alignment)
        raw_pose_by_history[index] = proximity
        raw_heatmap_by_history[index] = alignment
        pose_by_history[index] = weighted_pose
        heatmap_by_history[index] = weighted_heatmap
        hybrid_by_history[index] = hybrid

    return {
        "pose_only": float(pose_by_history.max(initial=0.0)),
        "heatmap_only": float(heatmap_by_history.max(initial=0.0)),
        "hybrid": float(hybrid_by_history.max(initial=0.0)),
        "history_count": float(mask.sum()),
        "history_mask": mask.copy(),
        "history_rank": rank.copy(),
        "recent_weight": recent_weight.astype(np.float32, copy=True),
        "raw_pose_by_history": raw_pose_by_history,
        "raw_heatmap_by_history": raw_heatmap_by_history,
        "pose_by_history": pose_by_history,
        "heatmap_by_history": heatmap_by_history,
        "hybrid_by_history": hybrid_by_history,
    }


def candidate_history_risk(
    actions: Sequence[int],
    context: Mapping[str, np.ndarray],
    *,
    pose_sigma_m: float = 0.45,
) -> dict[str, float]:
    """Backward-compatible aggregate normal-mode history risk."""

    features = candidate_history_features(
        actions, context, pose_sigma_m=pose_sigma_m
    )
    return {
        name: float(features[name])
        for name in ("pose_only", "heatmap_only", "hybrid", "history_count")
    }


def heatmap_pose_consistency(context: Mapping[str, np.ndarray]) -> float | None:
    """How well predicted panoramic bearings agree with known relative poses."""

    required = (
        "fixed_history_mask",
        "fixed_history_rel_poses",
        "spatial_statistics",
        "view_probabilities",
        "none_probability",
    )
    if any(name not in context for name in required):
        return None
    mask = _strip_batch(np.asarray(context["fixed_history_mask"]), 1).astype(bool)
    rel = _strip_batch(np.asarray(context["fixed_history_rel_poses"]), 2)
    stats = _strip_batch(np.asarray(context["spatial_statistics"]), 3)
    views = _strip_batch(np.asarray(context["view_probabilities"]), 2)
    none = _strip_batch(np.asarray(context["none_probability"]), 1)
    weighted_similarity = 0.0
    total_weight = 0.0
    for index in np.flatnonzero(mask):
        true_bearing = math.atan2(float(rel[index, 1]), float(rel[index, 0]))
        mean_x = stats[index, :, 0]
        predicted_bearings = VIEW_CENTERS_RAD - mean_x * (math.pi / 4.0)
        weights = np.clip(views[index], 0.0, None) * max(0.0, 1.0 - float(none[index]))
        deltas = np.asarray(_wrap_angle(predicted_bearings - true_bearing))
        similarities = np.exp(-0.5 * np.square(deltas / math.radians(35.0)))
        weighted_similarity += float(np.sum(weights * similarities))
        total_weight += float(np.sum(weights))
    return weighted_similarity / total_weight if total_weight > 0 else None


def deployable_loop_signals(
    executed_actions: Sequence[int],
    visited_body_poses: np.ndarray,
    current_body_pose: np.ndarray,
) -> dict[str, bool | float]:
    """Conservative loop/stall detector using only action and odometry history."""

    actions = [int(value) for value in executed_actions]
    visited = np.asarray(visited_body_poses, dtype=np.float32)
    current = np.asarray(current_body_pose, dtype=np.float32)
    if visited.size == 0:
        visited = np.empty((0, 4, 4), dtype=np.float32)
    if visited.ndim != 3 or visited.shape[1:] != (4, 4) or current.shape != (4, 4):
        raise ValueError("body poses must have shape [N,4,4] and [4,4]")

    recent_actions = actions[-8:]
    alternating = sum(
        (left, right) in {(ACTION_LEFT, ACTION_RIGHT), (ACTION_RIGHT, ACTION_LEFT)}
        for left, right in zip(recent_actions, recent_actions[1:])
    )
    oscillation = len(recent_actions) >= 4 and alternating >= 3

    positions = visited[:, (0, 2), 3] if len(visited) else np.empty((0, 2))
    current_xy = current[(0, 2), 3]
    older = positions[:-4] if len(positions) > 4 else np.empty((0, 2))
    revisit_distance = (
        float(np.linalg.norm(older - current_xy[None, :], axis=1).min())
        if len(older)
        else math.inf
    )
    current_revisit = revisit_distance < 0.35

    recent_positions = np.concatenate((positions[-8:], current_xy[None, :]), axis=0)
    if len(recent_positions) >= 2:
        steps = np.linalg.norm(np.diff(recent_positions, axis=0), axis=1)
        travelled = float(steps.sum())
        displacement = float(np.linalg.norm(recent_positions[-1] - recent_positions[0]))
    else:
        travelled = displacement = 0.0
    inefficient_loop = travelled >= 0.75 and displacement / max(travelled, 1e-6) < 0.35
    forward_count = sum(action == ACTION_FORWARD for action in recent_actions[-5:])
    stalled = forward_count >= 2 and travelled < 0.15
    confirmed = bool(oscillation or current_revisit or inefficient_loop or stalled)
    return {
        "confirmed": confirmed,
        "turn_oscillation": bool(oscillation),
        "current_revisit": bool(current_revisit),
        "inefficient_loop": bool(inefficient_loop),
        "stalled": bool(stalled),
        "recent_travelled_m": travelled,
        "recent_displacement_m": displacement,
        "older_pose_min_distance_m": (
            revisit_distance if math.isfinite(revisit_distance) else -1.0
        ),
    }


def deployable_recovery_partition(
    *,
    fixed_history_mask: np.ndarray,
    fixed_history_age_steps: np.ndarray,
    executed_actions: Sequence[int],
    visited_body_poses: np.ndarray,
    current_body_pose: np.ndarray,
    revisit_radius_m: float = 0.35,
    recent_pose_exclusion: int = 4,
) -> dict[str, Any]:
    """Find a loop-entry anchor and current-loop history using policy inputs.

    ``fixed_history_age_steps`` is already maintained by the online history
    buffer.  No reference path, goal location, simulator outcome, or stored
    candidate endpoint is used.  For a geometric current-position revisit the
    loop entry is the closest sufficiently old odometry pose.  For the other
    deployable loop signals it falls back to the beginning of the detector's
    eight-action window.
    """

    mask = _strip_batch(np.asarray(fixed_history_mask), 1).astype(bool)
    ages = _strip_batch(np.asarray(fixed_history_age_steps), 1).astype(np.int64)
    if ages.shape != mask.shape:
        raise ValueError(f"history age/mask mismatch: {ages.shape} vs {mask.shape}")
    actions = [int(value) for value in executed_actions]
    visited = np.asarray(visited_body_poses, dtype=np.float32)
    current = np.asarray(current_body_pose, dtype=np.float32)
    signals = deployable_loop_signals(actions, visited, current)
    current_step = len(actions)
    capture_steps = current_step - ages
    empty = {
        "ready": False,
        "anchor_index": -1,
        "anchor_capture_step": -1,
        "loop_start_step": -1,
        "loop_history_mask": np.zeros(len(mask), dtype=np.bool_),
        "capture_steps": capture_steps,
        "signals": signals,
        "reason": "loop_not_confirmed",
    }
    if not bool(signals["confirmed"]):
        return empty
    valid_indices = np.flatnonzero(mask)
    if len(valid_indices) == 0:
        return {**empty, "reason": "history_unavailable"}

    positions = visited[:, (0, 2), 3] if len(visited) else np.empty((0, 2))
    current_xy = current[(0, 2), 3]
    older_count = max(0, len(positions) - int(recent_pose_exclusion))
    older = positions[:older_count]
    if len(older):
        distances = np.linalg.norm(older - current_xy[None, :], axis=1)
        nearest_index = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_index])
    else:
        nearest_index = -1
        nearest_distance = math.inf
    if nearest_index >= 0 and nearest_distance < float(revisit_radius_m):
        loop_start_step = nearest_index
        source = "current_revisit"
    else:
        loop_start_step = max(0, current_step - 8)
        source = "detector_window"

    before_loop = valid_indices[capture_steps[valid_indices] <= loop_start_step]
    if len(before_loop):
        anchor_index = int(
            before_loop[np.argmax(capture_steps[before_loop])]
        )
    else:
        # If temporal subsampling skipped the exact entry, the oldest visible
        # slot is the only causal anchor available to the deployed policy.
        anchor_index = int(valid_indices[np.argmin(capture_steps[valid_indices])])
    anchor_step = int(capture_steps[anchor_index])
    loop_mask = mask & (capture_steps > anchor_step)
    loop_mask[anchor_index] = False
    ready = bool(loop_mask.any())
    return {
        "ready": ready,
        "anchor_index": anchor_index,
        "anchor_capture_step": anchor_step,
        "loop_start_step": int(loop_start_step),
        "loop_history_mask": loop_mask,
        "capture_steps": capture_steps,
        "signals": signals,
        "reason": f"{source}_anchor_ready" if ready else "loop_history_unavailable",
    }


def recovery_anchor_risk(
    features: Mapping[str, Any],
    partition: Mapping[str, Any],
    *,
    source: str = "heatmap",
) -> dict[str, float | bool]:
    """Mode-aware recovery energy: attract to anchor, avoid current loop.

    The two terms have equal fixed weight and each lies in ``[0,1]``:

    ``risk = 0.5 * (1 - anchor_overlap) + 0.5 * loop_overlap``.

    Lower is safer.  The formula has no learned or validation-selected sign;
    matched/shuffled heatmaps differ only in the overlap content.
    """

    field = {
        "heatmap": "raw_heatmap_by_history",
        "pose": "raw_pose_by_history",
    }.get(source)
    if field is None:
        raise ValueError(f"unsupported recovery risk source: {source}")
    values = np.asarray(features[field], dtype=np.float32)
    if not bool(partition.get("ready", False)):
        return {
            "risk": 1.0,
            "anchor_overlap": 0.0,
            "loop_overlap": 1.0,
            "ready": False,
        }
    anchor_index = int(partition["anchor_index"])
    loop_mask = np.asarray(partition["loop_history_mask"], dtype=np.bool_)
    if values.shape != loop_mask.shape or not 0 <= anchor_index < len(values):
        raise ValueError("recovery partition/interaction shape mismatch")
    anchor_overlap = float(np.clip(values[anchor_index], 0.0, 1.0))
    loop_overlap = float(np.clip(values[loop_mask].max(initial=0.0), 0.0, 1.0))
    risk = 0.5 * (1.0 - anchor_overlap) + 0.5 * loop_overlap
    return {
        "risk": float(np.clip(risk, 0.0, 1.0)),
        "anchor_overlap": anchor_overlap,
        "loop_overlap": loop_overlap,
        "ready": True,
    }


@dataclass(frozen=True)
class VetoThresholds:
    risk_on: float = 0.20
    risk_margin: float = 0.08
    minimum_native_mass: float = 2.0 / 32.0
    maximum_edit_fraction: float = 0.50
    require_confirmed_loop: bool = True


def select_directional_veto(
    *,
    baseline_id: str,
    candidates: Sequence[Mapping[str, Any]],
    risks: Mapping[str, float],
    loop_confirmed: bool,
    thresholds: VetoThresholds,
) -> dict[str, Any]:
    """Select at most one conservative direction-changing native candidate."""

    by_id = {str(candidate["treatment_id"]): candidate for candidate in candidates}
    if baseline_id not in by_id or baseline_id not in risks:
        raise ValueError("baseline candidate/risk is missing")
    baseline = by_id[baseline_id]
    baseline_actions = tuple(int(value) for value in baseline["spec"]["actions"])
    baseline_risk = float(risks[baseline_id])
    if thresholds.require_confirmed_loop and not loop_confirmed:
        return {
            "treatment_id": baseline_id,
            "intervened": False,
            "reason": "loop_not_confirmed",
            "baseline_risk": baseline_risk,
        }
    if baseline_risk < thresholds.risk_on:
        return {
            "treatment_id": baseline_id,
            "intervened": False,
            "reason": "native_risk_below_threshold",
            "baseline_risk": baseline_risk,
        }
    eligible: list[tuple[float, float, float, str]] = []
    baseline_first = baseline_actions[0] if baseline_actions else None
    for candidate in candidates:
        treatment_id = str(candidate["treatment_id"])
        if treatment_id == baseline_id or treatment_id not in risks:
            continue
        actions = tuple(int(value) for value in candidate["spec"]["actions"])
        if not actions or actions[0] == baseline_first:
            continue
        mass = float(candidate.get("native_sample_mass", 0.0))
        edit = action_edit_fraction(actions, baseline_actions)
        risk = float(risks[treatment_id])
        if mass + 1e-12 < thresholds.minimum_native_mass:
            continue
        if edit > thresholds.maximum_edit_fraction + 1e-12:
            continue
        if risk > baseline_risk - thresholds.risk_margin + 1e-12:
            continue
        eligible.append((risk, edit, -mass, treatment_id))
    if not eligible:
        return {
            "treatment_id": baseline_id,
            "intervened": False,
            "reason": "no_conservative_candidate",
            "baseline_risk": baseline_risk,
        }
    risk, edit, negative_mass, treatment_id = min(eligible)
    return {
        "treatment_id": treatment_id,
        "intervened": True,
        "reason": "history_risk_veto",
        "baseline_risk": baseline_risk,
        "selected_risk": float(risk),
        "risk_reduction": float(baseline_risk - risk),
        "edit_fraction": float(edit),
        "native_sample_mass": float(-negative_mass),
    }


def select_adaptive_prefix(
    *,
    baseline_id: str,
    baseline_execute_len: int,
    prefix_ids_by_length: Mapping[int, str],
    normalized_entropy: float,
    high_threshold: float = 0.80,
    medium_threshold: float = 0.60,
) -> dict[str, Any]:
    """Shorten the native-mean chunk without changing its direction."""

    if not 0.0 <= medium_threshold <= high_threshold <= 1.0:
        raise ValueError("prefix entropy thresholds must satisfy 0 <= medium <= high <= 1")
    full_length = max(0, int(baseline_execute_len))
    if full_length <= 1:
        requested = full_length
    elif normalized_entropy >= high_threshold:
        requested = 1
    elif normalized_entropy >= medium_threshold:
        requested = min(2, full_length)
    else:
        requested = full_length
    treatment_id = str(prefix_ids_by_length.get(requested, baseline_id))
    return {
        "treatment_id": treatment_id,
        "intervened": treatment_id != baseline_id,
        "reason": (
            "adaptive_prefix" if treatment_id != baseline_id else "native_chunk_retained"
        ),
        "normalized_entropy": float(normalized_entropy),
        "native_execute_len": full_length,
        "selected_execute_len": requested,
    }


__all__ = [
    "VetoThresholds",
    "action_distribution_entropy",
    "action_edit_fraction",
    "bearing_to_view_coordinate",
    "candidate_history_features",
    "candidate_history_risk",
    "deployable_loop_signals",
    "deployable_recovery_partition",
    "heatmap_pose_consistency",
    "recovery_anchor_risk",
    "select_adaptive_prefix",
    "select_directional_veto",
    "simulate_actions",
]

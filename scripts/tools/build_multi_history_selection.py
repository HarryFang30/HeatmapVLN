#!/usr/bin/env python3
"""Build a strict, scene-disjoint multi-history localization selection.

Task-3.6 deliberately does not reuse the sliding-window history sampler.  For
each current panorama it searches the complete earlier trajectory for K
non-recent anchors, enforcing temporal/spatial/view diversity and pairwise
target separation.  Exact pose and depth are used only to construct and audit
label metadata; the manifest's model-input contract contains RGB identities
only and explicitly forbids pose-derived fields.

The output is directly consumable by ``ExplicitMultiHistoryDataset`` while the
default ``VLNSlidingWindowDataset`` behaviour remains unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.explicit_multi_history import (
    MULTI_HISTORY_SCHEMA,
    canonical_sha256,
    record_identity,
)
from src.data.sliding_window_dataset import VLNSlidingWindowDataset
from src.data.trajectory_utils import compute_history_rel_poses

LOGGER = logging.getLogger("multi_history_selection")
VIEW_NAMES = ("front", "right", "back", "left")
PINNED_SOURCE_INVENTORY_SHA256 = (
    "658cc81148662efd64ff6c0fb032f49c00f39b2892358c93c9f26c4a0ff1cb66"
)


@dataclass(frozen=True)
class SelectionConstraints:
    num_history: int = 4
    min_temporal_lag: int = 16
    max_temporal_lag: int = 0
    min_spatial_distance: float = 0.75
    max_spatial_distance: float = 15.0
    min_bearing_separation_degrees: float = 30.0
    view_seam_margin_degrees: float = 10.0
    min_distinct_views: int = 3
    min_distinct_lag_bins: int = 2
    min_distinct_distance_bins: int = 2
    min_visible_anchors: int = 4
    min_visible_distinct_views: int = 3
    min_target_separation_pixels: float = 12.0
    min_target_view_fraction: float = 0.15
    max_target_view_fraction: float = 0.35
    beam_width: int = 128
    max_anchor_set_trials: int = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-history", type=int, default=4)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--max-clip-id", type=int, default=2000)
    parser.add_argument("--candidate-currents-per-clip", type=int, default=4)
    parser.add_argument("--min-temporal-lag", type=int, default=16)
    parser.add_argument("--max-temporal-lag", type=int, default=0)
    parser.add_argument("--min-spatial-distance", type=float, default=0.75)
    parser.add_argument("--max-spatial-distance", type=float, default=15.0)
    parser.add_argument("--temporal-lag-edges", default="8,16,32,64")
    parser.add_argument("--spatial-distance-edges", default="0.5,1,2,4,8")
    parser.add_argument("--min-bearing-separation-degrees", type=float, default=30.0)
    parser.add_argument("--view-seam-margin-degrees", type=float, default=10.0)
    parser.add_argument("--min-distinct-views", type=int, default=3)
    parser.add_argument("--min-distinct-lag-bins", type=int, default=2)
    parser.add_argument("--min-distinct-distance-bins", type=int, default=2)
    parser.add_argument("--min-visible-anchors", type=int, default=4)
    parser.add_argument("--min-visible-distinct-views", type=int, default=3)
    parser.add_argument("--min-target-separation-pixels", type=float, default=12.0)
    parser.add_argument("--min-target-view-fraction", type=float, default=0.15)
    parser.add_argument("--max-target-view-fraction", type=float, default=0.35)
    parser.add_argument("--depth-valid-oversample-factor", type=int, default=4)
    parser.add_argument(
        "--expected-source-inventory-sha256",
        default=PINNED_SOURCE_INVENTORY_SHA256,
        help="Optional pinned inventory hash; mismatch aborts before label materialization.",
    )
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--max-anchor-set-trials", type=int, default=32)
    parser.add_argument(
        "--slot-order",
        choices=("canonical", "randomized"),
        default="randomized",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_edges(value: str) -> tuple[float, ...]:
    edges = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not edges or any(not math.isfinite(edge) for edge in edges):
        raise ValueError("Bin edges must contain finite values")
    if any(left >= right for left, right in pairwise(edges)):
        raise ValueError(f"Bin edges must be strictly increasing: {edges}")
    return edges


def numeric_bin(value: float, edges: Sequence[float], prefix: str) -> str:
    for edge in edges:
        if value <= float(edge):
            return f"{prefix}_le_{float(edge):g}"
    return f"{prefix}_gt_{float(edges[-1]):g}"


def normalise_degrees(value: float) -> float:
    result = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if result == -180.0 else result


def bearing_to_view(bearing_degrees: float) -> str:
    """Map forward/left coordinates onto front/right/back/left panorama views."""
    bearing = normalise_degrees(bearing_degrees)
    if -45.0 <= bearing < 45.0:
        return "front"
    if -135.0 <= bearing < -45.0:
        return "right"
    if 45.0 <= bearing < 135.0:
        return "left"
    return "back"


def circular_distance_degrees(left: float, right: float) -> float:
    delta = abs(normalise_degrees(float(left) - float(right)))
    return min(delta, 360.0 - delta)


def distance_to_view_seam_degrees(bearing_degrees: float) -> float:
    """Return distance to the panorama view boundaries (±45°, ±135°)."""
    return min(
        circular_distance_degrees(bearing_degrees, seam)
        for seam in (-135.0, -45.0, 45.0, 135.0)
    )


def _stable_rank(seed: int, *values: Any) -> str:
    material = "\n".join([str(seed), *(str(value) for value in values)])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _relative_clip(dataset: Any, clip_index: int) -> str:
    clip = Path(dataset.clips[clip_index])
    try:
        return clip.relative_to(Path(dataset.root)).as_posix()
    except ValueError:
        return clip.as_posix()


def describe_anchor_candidates(
    poses: Sequence[np.ndarray],
    current_frame: int,
    *,
    constraints: SelectionConstraints,
    temporal_lag_edges: Sequence[float],
    spatial_distance_edges: Sequence[float],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Describe every eligible past frame using label-side pose metadata."""
    rejection_counts: Counter[str] = Counter()
    history_frames = list(range(int(current_frame)))
    if not history_frames:
        return [], Counter({"no_past_frames": 1})
    relative = compute_history_rel_poses(
        [np.asarray(poses[frame], dtype=np.float32) for frame in history_frames],
        np.asarray(poses[int(current_frame)], dtype=np.float32),
    )
    candidates: list[dict[str, Any]] = []
    for frame, rel_pose in zip(history_frames, relative, strict=True):
        lag = int(current_frame) - int(frame)
        if lag < constraints.min_temporal_lag:
            rejection_counts["too_recent"] += 1
            continue
        if constraints.max_temporal_lag > 0 and lag > constraints.max_temporal_lag:
            rejection_counts["too_old"] += 1
            continue
        if not np.isfinite(rel_pose).all():
            rejection_counts["nonfinite_relative_pose"] += 1
            continue
        distance = float(np.linalg.norm(rel_pose[:2]))
        if distance < constraints.min_spatial_distance:
            rejection_counts["too_close"] += 1
            continue
        if constraints.max_spatial_distance > 0 and distance > constraints.max_spatial_distance:
            rejection_counts["too_far"] += 1
            continue
        bearing = normalise_degrees(math.degrees(math.atan2(float(rel_pose[1]), float(rel_pose[0]))))
        seam_distance = distance_to_view_seam_degrees(bearing)
        if seam_distance + 1e-12 < constraints.view_seam_margin_degrees:
            rejection_counts["near_view_seam"] += 1
            continue
        relative_yaw = normalise_degrees(
            math.degrees(math.atan2(float(rel_pose[3]), float(rel_pose[2])))
        )
        candidates.append(
            {
                "history_frame": int(frame),
                "temporal_lag": lag,
                "temporal_lag_bin": numeric_bin(float(lag), temporal_lag_edges, "t"),
                "spatial_distance_m": distance,
                "spatial_distance_bin": numeric_bin(
                    distance,
                    spatial_distance_edges,
                    "d",
                ),
                "bearing_degrees": bearing,
                "bearing_view": bearing_to_view(bearing),
                "view_seam_distance_degrees": seam_distance,
                "relative_yaw_degrees": relative_yaw,
                # This is label metadata, never part of model_inputs.
                "relative_pose_label": [float(value) for value in rel_pose.tolist()],
            }
        )
    candidates.sort(key=lambda item: int(item["history_frame"]))
    return candidates, rejection_counts


def _pairwise_bearing_separated(
    anchors: Sequence[dict[str, Any]],
    minimum_degrees: float,
) -> bool:
    for left_index, left in enumerate(anchors):
        for right in anchors[left_index + 1 :]:
            if circular_distance_degrees(
                float(left["bearing_degrees"]),
                float(right["bearing_degrees"]),
            ) + 1e-12 < minimum_degrees:
                return False
    return True


def _anchor_set_score(anchors: Sequence[dict[str, Any]]) -> float:
    views = {str(anchor["bearing_view"]) for anchor in anchors}
    lag_bins = {str(anchor["temporal_lag_bin"]) for anchor in anchors}
    distance_bins = {str(anchor["spatial_distance_bin"]) for anchor in anchors}
    bearings = [float(anchor["bearing_degrees"]) for anchor in anchors]
    pairwise = [
        circular_distance_degrees(left, right)
        for index, left in enumerate(bearings)
        for right in bearings[index + 1 :]
    ]
    lags = [float(anchor["temporal_lag"]) for anchor in anchors]
    distances = [float(anchor["spatial_distance_m"]) for anchor in anchors]
    separation = min(pairwise, default=180.0) / 180.0
    lag_spread = (max(lags) - min(lags)) / max(max(lags), 1.0)
    distance_spread = (max(distances) - min(distances)) / max(max(distances), 1e-6)
    return (
        10.0 * len(views)
        + 4.0 * len(lag_bins)
        + 4.0 * len(distance_bins)
        + 2.0 * separation
        + lag_spread
        + distance_spread
    )


def _anchor_set_contract_satisfied(
    anchors: Sequence[dict[str, Any]],
    constraints: SelectionConstraints,
) -> bool:
    return bool(
        len(anchors) == constraints.num_history
        and len({str(anchor["bearing_view"]) for anchor in anchors})
        >= constraints.min_distinct_views
        and len({str(anchor["temporal_lag_bin"]) for anchor in anchors})
        >= constraints.min_distinct_lag_bins
        and len({str(anchor["spatial_distance_bin"]) for anchor in anchors})
        >= constraints.min_distinct_distance_bins
        and _pairwise_bearing_separated(
            anchors,
            constraints.min_bearing_separation_degrees,
        )
    )


def rank_anchor_sets(
    candidates: Sequence[dict[str, Any]],
    *,
    constraints: SelectionConstraints,
    seed: int,
    sample_key: str,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Deterministic beam search for diverse K-anchor combinations."""
    ordered = sorted(candidates, key=lambda item: int(item["history_frame"]))
    support = {
        "eligible_anchors": len(ordered),
        "distinct_views": sorted({str(item["bearing_view"]) for item in ordered}),
        "distinct_temporal_lag_bins": sorted(
            {str(item["temporal_lag_bin"]) for item in ordered}
        ),
        "distinct_spatial_distance_bins": sorted(
            {str(item["spatial_distance_bin"]) for item in ordered}
        ),
    }
    immediate_reasons: list[str] = []
    if len(ordered) < constraints.num_history:
        immediate_reasons.append("insufficient_eligible_anchors")
    if len(support["distinct_views"]) < constraints.min_distinct_views:
        immediate_reasons.append("insufficient_view_support")
    if len(support["distinct_temporal_lag_bins"]) < constraints.min_distinct_lag_bins:
        immediate_reasons.append("insufficient_temporal_lag_bin_support")
    if len(support["distinct_spatial_distance_bins"]) < constraints.min_distinct_distance_bins:
        immediate_reasons.append("insufficient_spatial_distance_bin_support")
    if immediate_reasons:
        return [], {"support": support, "failure_reasons": immediate_reasons}

    # State indices are strictly increasing, so every combination is visited
    # at most once.  Beam pruning bounds work for long random-walk clips.
    states: list[tuple[int, ...]] = [()]
    for _depth in range(constraints.num_history):
        expanded: list[tuple[float, str, tuple[int, ...]]] = []
        for state in states:
            start = state[-1] + 1 if state else 0
            for candidate_index in range(start, len(ordered)):
                next_state = (*state, candidate_index)
                anchors = [ordered[index] for index in next_state]
                if not _pairwise_bearing_separated(
                    anchors,
                    constraints.min_bearing_separation_degrees,
                ):
                    continue
                frame_key = ",".join(str(item["history_frame"]) for item in anchors)
                expanded.append(
                    (
                        _anchor_set_score(anchors),
                        _stable_rank(seed, sample_key, frame_key),
                        next_state,
                    )
                )
        expanded.sort(key=lambda item: (-item[0], item[1]))
        states = [item[2] for item in expanded[: constraints.beam_width]]
        if not states:
            break

    ranked: list[tuple[float, str, list[dict[str, Any]]]] = []
    for state in states:
        anchors = [ordered[index] for index in state]
        if not _anchor_set_contract_satisfied(anchors, constraints):
            continue
        frame_key = ",".join(str(item["history_frame"]) for item in anchors)
        ranked.append(
            (
                _anchor_set_score(anchors),
                _stable_rank(seed, sample_key, "final", frame_key),
                anchors,
            )
        )
    ranked.sort(key=lambda item: (-item[0], item[1]))
    anchor_sets = [item[2] for item in ranked[: constraints.max_anchor_set_trials]]
    reasons = [] if anchor_sets else ["no_feasible_separated_anchor_set"]
    return anchor_sets, {"support": support, "failure_reasons": reasons}


def slot_permutation(
    num_history: int,
    *,
    order: str,
    seed: int,
    sample_key: str,
) -> list[int]:
    permutation = list(range(num_history))
    if order == "canonical":
        return permutation
    if order != "randomized":
        raise ValueError(f"Unknown slot order: {order}")
    material = f"{seed}\n{sample_key}\nslot-order".encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))
    rng.shuffle(permutation)
    return permutation


def _heatmap_peak(heatmap: torch.Tensor) -> tuple[int, int, float]:
    flat = heatmap.reshape(-1)
    index = int(flat.argmax().item())
    width = int(heatmap.shape[-1])
    return index % width, index // width, float(flat[index].item())


def compute_label_pool(
    dataset: Any,
    *,
    clip_index: int,
    current_frame: int,
    anchors: Sequence[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], list[int]]:
    """Compute exact depth-aware targets once for the union of trial anchors."""
    by_frame = {int(anchor["history_frame"]): anchor for anchor in anchors}
    frames = sorted(by_frame)
    poses = dataset._load_poses(clip_index)
    history_poses = [poses[frame] for frame in frames]
    current_pose = poses[int(current_frame)]
    clip_dir = dataset.clips[clip_index]
    image_size, intrinsics = dataset._load_intrinsics(clip_index, clip_dir)
    hm_width, hm_height = dataset.hm_size
    heatmaps, visibility = dataset._compute_per_history_multiview_heatmaps(
        clip_idx=clip_index,
        clip_dir=clip_dir,
        history_poses=history_poses,
        current_t=int(current_frame),
        img_size=image_size,
        K=intrinsics,
        hm_size=(hm_height, hm_width),
    )
    if tuple(heatmaps.shape[:2]) != (len(frames), len(VIEW_NAMES)):
        raise ValueError(f"Unexpected heatmap shape: {tuple(heatmaps.shape)}")
    if tuple(visibility.shape) != (len(frames), len(VIEW_NAMES)):
        raise ValueError(f"Unexpected visibility shape: {tuple(visibility.shape)}")

    pose_hash_context = {
        "current_frame": int(current_frame),
        "current_pose": np.asarray(current_pose, dtype=np.float32).tolist(),
    }
    pool: dict[int, dict[str, Any]] = {}
    for pool_index, frame in enumerate(frames):
        target_views: list[dict[str, Any]] = []
        for view_index, view_name in enumerate(VIEW_NAMES):
            visible = bool(float(visibility[pool_index, view_index].item()) > 0.5)
            x: int | None = None
            y: int | None = None
            peak: float | None = None
            if visible:
                x, y, peak = _heatmap_peak(heatmaps[pool_index, view_index])
            target_views.append(
                {
                    "view": view_name,
                    "view_index": view_index,
                    "visible": visible,
                    "x": x,
                    "y": y,
                    "peak_value": peak,
                }
            )
        visible_targets = [target for target in target_views if bool(target["visible"])]
        primary = max(
            visible_targets,
            key=lambda target: (float(target["peak_value"]), -int(target["view_index"])),
            default=None,
        )
        primary_target = None
        if primary is not None:
            primary_target = {
                **primary,
                "panorama_x": int(primary["view_index"]) * hm_width + int(primary["x"]),
            }
        exact_pose_hash = canonical_sha256(
            {
                **pose_hash_context,
                "history_frame": frame,
                "history_pose": np.asarray(poses[frame], dtype=np.float32).tolist(),
            }
        )
        pool[frame] = {
            **by_frame[frame],
            "target_views": target_views,
            "primary_target": primary_target,
            "any_visible": bool(visible_targets),
            "exact_pose_label_sha256": exact_pose_hash,
        }
    return pool, [len(frames), len(VIEW_NAMES), hm_height, hm_width]


def _panorama_target_distance(
    left: dict[str, Any],
    right: dict[str, Any],
    panorama_width: int,
) -> float:
    delta_x = abs(float(left["panorama_x"]) - float(right["panorama_x"]))
    delta_x = min(delta_x, float(panorama_width) - delta_x)
    delta_y = float(left["y"]) - float(right["y"])
    return math.hypot(delta_x, delta_y)


def target_separation_audit(
    anchors: Sequence[dict[str, Any]],
    *,
    heatmap_shape: Sequence[int],
) -> dict[str, Any]:
    primary = [anchor["primary_target"] for anchor in anchors if anchor["primary_target"] is not None]
    panorama_width = int(heatmap_shape[1]) * int(heatmap_shape[3])
    distances = [
        _panorama_target_distance(left, right, panorama_width)
        for index, left in enumerate(primary)
        for right in primary[index + 1 :]
    ]
    return {
        "visible_anchor_count": len(primary),
        "visible_distinct_views": len({str(target["view"]) for target in primary}),
        "pairwise_target_distances_pixels": distances,
        "minimum_target_separation_pixels": min(distances) if distances else None,
        "panorama_width_pixels": panorama_width,
    }


def _label_contract_failure(
    audit: dict[str, Any],
    constraints: SelectionConstraints,
) -> str | None:
    if int(audit["visible_anchor_count"]) < constraints.min_visible_anchors:
        return "insufficient_visible_anchors"
    if int(audit["visible_distinct_views"]) < constraints.min_visible_distinct_views:
        return "insufficient_visible_view_support"
    minimum = audit["minimum_target_separation_pixels"]
    if constraints.min_target_separation_pixels > 0.0:
        if minimum is None or float(minimum) + 1e-12 < constraints.min_target_separation_pixels:
            return "insufficient_target_pixel_separation"
    return None


def assemble_selection_record(
    *,
    relative_clip: str,
    scene: str,
    current_frame: int,
    canonical_anchors: Sequence[dict[str, Any]],
    heatmap_shape: Sequence[int],
    slot_order: str,
    seed: int,
) -> dict[str, Any]:
    canonical = sorted(canonical_anchors, key=lambda item: int(item["history_frame"]))
    provisional_key = f"{relative_clip}:current={int(current_frame)}"
    permutation = slot_permutation(
        len(canonical),
        order=slot_order,
        seed=seed,
        sample_key=provisional_key,
    )
    ordered = [canonical[index] for index in permutation]
    history_frames = [int(anchor["history_frame"]) for anchor in ordered]
    record: dict[str, Any] = {
        "relative_clip": relative_clip,
        "scene": scene,
        "current_frame": int(current_frame),
        "history_frames": history_frames,
        "canonical_history_frames": [int(anchor["history_frame"]) for anchor in canonical],
        "slot_permutation": permutation,
        "slot_order": slot_order,
        "model_inputs": {
            "current_rgb": "current_rgb_panorama",
            "history_rgb": "ordered_history_rgb_observations",
        },
        "loader_alignment": {
            "current": {
                "relative_clip": relative_clip,
                "frame_index": int(current_frame),
            },
            "history": [
                {
                    "loader_position": position,
                    "canonical_index": permutation[position],
                    "relative_clip": relative_clip,
                    "frame_index": frame,
                }
                for position, frame in enumerate(history_frames)
            ],
            "usage": "loader_and_label_alignment_only_never_model_input",
        },
        "label_metadata": {
            "anchors": [{"slot": slot, **anchor} for slot, anchor in enumerate(ordered)],
            "heatmap_pool_shape": [int(value) for value in heatmap_shape],
            "pose_usage": "label_generation_and_audit_only",
        },
    }
    record["sample_id"] = record_identity(record)
    record["label_metadata_sha256"] = canonical_sha256(record["label_metadata"])
    record["record_sha256"] = canonical_sha256(record)
    return record


def _failure(
    *,
    split: str,
    relative_clip: str,
    scene: str,
    current_frame: int,
    stage: str,
    reason: str,
    details: Any = None,
) -> dict[str, Any]:
    return {
        "split": split,
        "relative_clip": relative_clip,
        "scene": scene,
        "current_frame": int(current_frame),
        "stage": stage,
        "reason": reason,
        "details": details,
    }


def candidate_current_frames(
    dataset: Any,
    clip_index: int,
    *,
    minimum_current_frame: int,
    seed: int,
) -> list[int]:
    valid = [
        int(frame)
        for frame in dataset._clip_valid_frames.get(clip_index, [])
        if int(frame) >= minimum_current_frame
    ]
    if valid:
        terminal = int(dataset._clip_valid_frames[clip_index][-1])
        valid = [frame for frame in valid if frame != terminal]
    relative_clip = _relative_clip(dataset, clip_index)
    valid.sort(key=lambda frame: _stable_rank(seed, relative_clip, frame))
    # The caller applies ``per_clip`` to *pose-valid* proposals.  Truncating
    # here would let a few unlucky early/current choices hide feasible later
    # frames from the catalog.
    return valid


def build_pose_catalog(
    dataset: Any,
    *,
    constraints: SelectionConstraints,
    temporal_lag_edges: Sequence[float],
    spatial_distance_edges: Sequence[float],
    candidate_currents_per_clip: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    proposals: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for clip_index in range(len(dataset.clips)):
        relative_clip = _relative_clip(dataset, clip_index)
        scene = Path(dataset.clips[clip_index]).parent.name
        currents = candidate_current_frames(
            dataset,
            clip_index,
            minimum_current_frame=constraints.min_temporal_lag,
            seed=seed,
        )
        try:
            poses = dataset._load_poses(clip_index)
        except Exception as exc:
            failures.append(
                _failure(
                    split=str(dataset.split),
                    relative_clip=relative_clip,
                    scene=scene,
                    current_frame=-1,
                    stage="pose_load",
                    reason="pose_load_failed",
                    details=repr(exc),
                )
            )
            continue
        pose_valid_for_clip = 0
        for current_frame in currents:
            try:
                candidates, rejection_counts = describe_anchor_candidates(
                    poses,
                    current_frame,
                    constraints=constraints,
                    temporal_lag_edges=temporal_lag_edges,
                    spatial_distance_edges=spatial_distance_edges,
                )
                sample_key = f"{relative_clip}:current={current_frame}"
                anchor_sets, diagnostics = rank_anchor_sets(
                    candidates,
                    constraints=constraints,
                    seed=seed,
                    sample_key=sample_key,
                )
                if not anchor_sets:
                    reasons = diagnostics["failure_reasons"] or ["unknown_pose_selection_failure"]
                    for reason in reasons:
                        failures.append(
                            _failure(
                                split=str(dataset.split),
                                relative_clip=relative_clip,
                                scene=scene,
                                current_frame=current_frame,
                                stage="pose_selection",
                                reason=reason,
                                details={
                                    **diagnostics,
                                    "anchor_rejection_counts": dict(rejection_counts),
                                },
                            )
                        )
                    continue
                proposals.append(
                    {
                        "clip_index": clip_index,
                        "relative_clip": relative_clip,
                        "scene": scene,
                        "current_frame": current_frame,
                        "anchor_sets": anchor_sets,
                        "pose_selection_diagnostics": {
                            **diagnostics,
                            "anchor_rejection_counts": dict(rejection_counts),
                        },
                    }
                )
                pose_valid_for_clip += 1
                if (
                    candidate_currents_per_clip > 0
                    and pose_valid_for_clip >= candidate_currents_per_clip
                ):
                    break
            except Exception as exc:
                failures.append(
                    _failure(
                        split=str(dataset.split),
                        relative_clip=relative_clip,
                        scene=scene,
                        current_frame=current_frame,
                        stage="pose_selection",
                        reason="pose_selection_exception",
                        details=repr(exc),
                    )
                )
    proposals.sort(
        key=lambda item: _stable_rank(
            seed,
            item["relative_clip"],
            item["current_frame"],
        )
    )
    return proposals, failures


def scene_round_robin_proposals(
    proposals: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for proposal in proposals:
        by_scene[str(proposal["scene"])].append(proposal)
    for scene in by_scene:
        by_scene[scene].sort(
            key=lambda item: _stable_rank(
                seed,
                item["relative_clip"],
                item["current_frame"],
            )
        )
    output: list[dict[str, Any]] = []
    cursor = Counter()
    while True:
        progressed = False
        for scene in sorted(by_scene):
            position = cursor[scene]
            if position >= len(by_scene[scene]):
                continue
            output.append(by_scene[scene][position])
            cursor[scene] += 1
            progressed = True
        if not progressed:
            return output


def materialize_selection(
    dataset: Any,
    proposals: Sequence[dict[str, Any]],
    *,
    requested_samples: int,
    constraints: SelectionConstraints,
    slot_order: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for proposal in scene_round_robin_proposals(proposals, seed=seed):
        if len(selected) >= requested_samples:
            break
        union: dict[int, dict[str, Any]] = {}
        for anchor_set in proposal["anchor_sets"]:
            for anchor in anchor_set:
                union[int(anchor["history_frame"])] = anchor
        try:
            pool, heatmap_shape = compute_label_pool(
                dataset,
                clip_index=int(proposal["clip_index"]),
                current_frame=int(proposal["current_frame"]),
                anchors=list(union.values()),
            )
        except Exception as exc:
            failures.append(
                _failure(
                    split=str(dataset.split),
                    relative_clip=str(proposal["relative_clip"]),
                    scene=str(proposal["scene"]),
                    current_frame=int(proposal["current_frame"]),
                    stage="label_generation",
                    reason="label_generation_failed",
                    details=repr(exc),
                )
            )
            continue

        trial_failures: Counter[str] = Counter()
        accepted: tuple[list[dict[str, Any]], dict[str, Any]] | None = None
        for anchor_set in proposal["anchor_sets"]:
            labeled = [pool[int(anchor["history_frame"])] for anchor in anchor_set]
            audit = target_separation_audit(labeled, heatmap_shape=heatmap_shape)
            reason = _label_contract_failure(audit, constraints)
            if reason is None:
                accepted = labeled, audit
                break
            trial_failures[reason] += 1
        if accepted is None:
            failures.append(
                _failure(
                    split=str(dataset.split),
                    relative_clip=str(proposal["relative_clip"]),
                    scene=str(proposal["scene"]),
                    current_frame=int(proposal["current_frame"]),
                    stage="label_selection",
                    reason="no_label_valid_anchor_set",
                    details={"trial_failure_counts": dict(trial_failures)},
                )
            )
            continue
        anchors, target_audit = accepted
        record = assemble_selection_record(
            relative_clip=str(proposal["relative_clip"]),
            scene=str(proposal["scene"]),
            current_frame=int(proposal["current_frame"]),
            canonical_anchors=anchors,
            heatmap_shape=heatmap_shape,
            slot_order=slot_order,
            seed=seed,
        )
        record["selection_audit"] = {
            "target_separation": target_audit,
            "pose_selection": proposal["pose_selection_diagnostics"],
        }
        # selection_audit is part of the strict record hash.
        record["record_sha256"] = canonical_sha256(
            {key: value for key, value in record.items() if key != "record_sha256"}
        )
        selected.append(record)
    if len(selected) < requested_samples:
        failures.append(
            {
                "split": str(dataset.split),
                "stage": "depth_valid_pool",
                "reason": "depth_valid_pool_shortfall",
                "details": {
                    "requested_samples": int(requested_samples),
                    "selected_samples": len(selected),
                    "shortfall": int(requested_samples) - len(selected),
                },
            }
        )
    return selected, failures


def _target_view_contributions(record: dict[str, Any]) -> Counter[str]:
    contributions: Counter[str] = Counter()
    for anchor in record["label_metadata"]["anchors"]:
        primary = anchor.get("primary_target")
        if primary is None:
            continue
        contributions[str(primary["view"])] += 1
    return contributions


def deterministic_balanced_selection(
    records: Sequence[dict[str, Any]],
    *,
    requested_samples: int,
    num_history: int,
    min_target_view_fraction: float,
    max_target_view_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scene-aware greedy selection with a non-relaxing final view cap."""
    if requested_samples <= 0:
        return [], {
            "requested_samples": int(requested_samples),
            "selected_samples": 0,
            "selection_complete": True,
            "unmet_constraints": [],
        }
    if not 0.0 <= min_target_view_fraction <= 0.25:
        raise ValueError("min_target_view_fraction must lie in [0,0.25]")
    if not 0.25 <= max_target_view_fraction <= 1.0:
        raise ValueError("max_target_view_fraction must lie in [0.25,1]")
    if min_target_view_fraction > max_target_view_fraction:
        raise ValueError("min_target_view_fraction exceeds max_target_view_fraction")
    final_event_count = int(requested_samples) * int(num_history)
    per_view_floor = math.ceil(min_target_view_fraction * final_event_count - 1e-12)
    per_view_cap = math.floor(max_target_view_fraction * final_event_count + 1e-12)
    remaining = sorted(
        records,
        key=lambda record: _stable_rank(seed, record["sample_id"], "balanced-pool"),
    )
    selected: list[dict[str, Any]] = []
    view_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()

    while len(selected) < requested_samples and remaining:
        best_index: int | None = None
        best_score = -math.inf
        best_tie = ""
        for index, record in enumerate(remaining):
            contribution = _target_view_contributions(record)
            if sum(contribution.values()) != num_history:
                continue
            if any(view_counts[view] + count > per_view_cap for view, count in contribution.items()):
                continue
            prospective = view_counts + contribution
            remaining_event_capacity = (
                requested_samples - len(selected) - 1
            ) * num_history
            if any(
                prospective[view] + remaining_event_capacity < per_view_floor
                for view in VIEW_NAMES
            ):
                continue
            # Prefer under-represented target sectors and under-represented
            # scenes.  The absolute final cap above is never relaxed.
            view_score = sum(
                count
                * (
                    2.0 * max(per_view_floor - view_counts[view], 0)
                    + per_view_cap
                    - view_counts[view]
                )
                / max(per_view_cap, 1)
                for view, count in contribution.items()
            )
            scene = str(record["scene"])
            scene_score = 2.0 / (1.0 + scene_counts[scene])
            score = view_score + scene_score
            tie = _stable_rank(seed, record["sample_id"], "balanced-tie")
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12
                and (best_index is None or tie < best_tie)
            ):
                best_index = index
                best_score = score
                best_tie = tie
        if best_index is None:
            break
        record = remaining.pop(best_index)
        selected.append(record)
        view_counts.update(_target_view_contributions(record))
        scene_counts[str(record["scene"])] += 1

    actual_events = sum(view_counts.values())
    fractions = {
        view: (float(view_counts[view]) / actual_events if actual_events else 0.0)
        for view in VIEW_NAMES
    }
    unmet: list[str] = []
    if len(selected) < requested_samples:
        unmet.append(f"sample_count_shortfall:{requested_samples - len(selected)}")
    if len(selected) == requested_samples:
        for view in VIEW_NAMES:
            if fractions[view] + 1e-12 < min_target_view_fraction:
                unmet.append(
                    f"target_view_fraction:{view}:{fractions[view]:.6f}<"
                    f"{min_target_view_fraction:.6f}"
                )
            if fractions[view] > max_target_view_fraction + 1e-12:
                unmet.append(
                    f"target_view_fraction:{view}:{fractions[view]:.6f}>"
                    f"{max_target_view_fraction:.6f}"
                )
    return selected, {
        "requested_samples": int(requested_samples),
        "pool_samples": len(records),
        "selected_samples": len(selected),
        "selection_complete": len(selected) == requested_samples and not unmet,
        "target_events_at_requested_size": final_event_count,
        "per_view_event_floor": per_view_floor,
        "per_view_event_cap": per_view_cap,
        "min_target_view_fraction": float(min_target_view_fraction),
        "max_target_view_fraction": float(max_target_view_fraction),
        "target_view_counts": {view: int(view_counts[view]) for view in VIEW_NAMES},
        "target_view_fractions": fractions,
        "scene_counts": dict(sorted(scene_counts.items())),
        "unmet_constraints": unmet,
    }


def audit_selection(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scene_counts = Counter(str(record["scene"]) for record in records)
    slot_views: dict[str, Counter[str]] = defaultdict(Counter)
    slot_lags: dict[str, Counter[str]] = defaultdict(Counter)
    slot_distances: dict[str, Counter[str]] = defaultdict(Counter)
    visible_counts = Counter()
    target_view_counts: Counter[str] = Counter()
    minimum_separations: list[float] = []
    seam_distances: list[float] = []
    for record in records:
        anchors = record["label_metadata"]["anchors"]
        for anchor in anchors:
            slot = str(int(anchor["slot"]))
            slot_views[slot][str(anchor["bearing_view"])] += 1
            slot_lags[slot][str(anchor["temporal_lag_bin"])] += 1
            slot_distances[slot][str(anchor["spatial_distance_bin"])] += 1
            seam_distances.append(float(anchor["view_seam_distance_degrees"]))
        target = record["selection_audit"]["target_separation"]
        target_view_counts.update(_target_view_contributions(record))
        visible_counts[str(int(target["visible_anchor_count"]))] += 1
        minimum = target["minimum_target_separation_pixels"]
        if minimum is not None:
            minimum_separations.append(float(minimum))
    return {
        "samples": len(records),
        "scenes": len(scene_counts),
        "scene_counts": dict(sorted(scene_counts.items())),
        "per_slot_bearing_view_counts": {
            slot: dict(sorted(counts.items())) for slot, counts in sorted(slot_views.items())
        },
        "per_slot_temporal_lag_bin_counts": {
            slot: dict(sorted(counts.items())) for slot, counts in sorted(slot_lags.items())
        },
        "per_slot_spatial_distance_bin_counts": {
            slot: dict(sorted(counts.items())) for slot, counts in sorted(slot_distances.items())
        },
        "visible_anchor_count_histogram": dict(sorted(visible_counts.items())),
        "target_view_counts": {view: int(target_view_counts[view]) for view in VIEW_NAMES},
        "target_view_fractions": {
            view: (
                float(target_view_counts[view]) / sum(target_view_counts.values())
                if target_view_counts
                else 0.0
            )
            for view in VIEW_NAMES
        },
        "minimum_target_separation_pixels": (
            min(minimum_separations) if minimum_separations else None
        ),
        "view_seam_distance_degrees": {
            "minimum": min(seam_distances) if seam_distances else None,
            "median": float(np.median(seam_distances)) if seam_distances else None,
        },
    }


def failure_audit(failures: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reason_counts = Counter(str(failure["reason"]) for failure in failures)
    stage_counts = Counter(str(failure["stage"]) for failure in failures)
    return {
        "failures": len(failures),
        "reason_counts": dict(sorted(reason_counts.items())),
        "stage_counts": dict(sorted(stage_counts.items())),
        "failure_sha256": canonical_sha256(list(failures)),
        "records": list(failures),
    }


def selection_manifest(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    identities = [str(record["sample_id"]) for record in records]
    record_hashes = [str(record["record_sha256"]) for record in records]
    scenes = sorted({str(record["scene"]) for record in records})
    return {
        "sample_count": len(records),
        "sample_ids": identities,
        "record_identity_sha256": hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest(),
        "ordered_record_sha256": hashlib.sha256("\n".join(record_hashes).encode("utf-8")).hexdigest(),
        "records_sha256": canonical_sha256(list(records)),
        "scenes": scenes,
        "scene_sha256": hashlib.sha256("\n".join(scenes).encode("utf-8")).hexdigest(),
    }


def assert_scene_disjoint(
    train_records: Sequence[dict[str, Any]],
    val_records: Sequence[dict[str, Any]],
) -> None:
    train_scenes = {str(record["scene"]) for record in train_records}
    val_scenes = {str(record["scene"]) for record in val_records}
    overlap = sorted(train_scenes & val_scenes)
    if overlap:
        raise RuntimeError("Train/validation scenes overlap: " + ", ".join(overlap))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            handle.write("\n")


def _inventory_line(record: dict[str, Any]) -> str:
    return (
        f"{record['relative_clip']}\t{record['scene_id']}\t{record['episode_id']}\t"
        f"{record['num_frames']}\t{record['seed']}"
    )


def _inventory_hash(records: Sequence[dict[str, Any]]) -> str:
    rows = sorted(_inventory_line(record) for record in records)
    payload = ("\n".join(rows) + ("\n" if rows else "")).encode()
    return hashlib.sha256(payload).hexdigest()


def source_inventory(dataset: Any) -> dict[str, Any]:
    """Hash the append-only source scope without reading RGB/depth payloads."""
    records: list[dict[str, Any]] = []
    missing_fields: Counter[str] = Counter()
    for clip_index in range(len(dataset.clips)):
        meta = dataset._load_meta(clip_index)
        relative_clip = _relative_clip(dataset, clip_index)
        values = {
            "relative_clip": relative_clip,
            "scene_id": meta.get("scene_id"),
            "episode_id": meta.get("episode_id"),
            "num_frames": meta.get("num_frames"),
            "seed": meta.get("seed"),
        }
        for field, value in values.items():
            if field != "relative_clip" and value is None:
                missing_fields[field] += 1
        records.append(values)
    records.sort(key=lambda record: str(record["relative_clip"]))
    return {
        "clips": len(records),
        "fields": ["relative_clip", "scene_id", "episode_id", "num_frames", "seed"],
        "sort_key": "relative_clip",
        "serialization": "tab-separated str(fields), lexicographically sorted rows, trailing LF",
        "inventory_sha256": _inventory_hash(records),
        "canonical_records_sha256": canonical_sha256(records),
        "missing_field_counts": dict(sorted(missing_fields.items())),
        "records": records,
    }


def merge_source_inventories(inventories: Sequence[dict[str, Any]]) -> dict[str, Any]:
    records = [record for inventory in inventories for record in inventory["records"]]
    records.sort(key=lambda record: str(record["relative_clip"]))
    duplicates = sorted(
        relative_clip
        for relative_clip, count in Counter(
            str(record["relative_clip"]) for record in records
        ).items()
        if count > 1
    )
    if duplicates:
        raise RuntimeError("Source inventory contains duplicate clips: " + ", ".join(duplicates))
    missing = Counter()
    for inventory in inventories:
        missing.update(inventory.get("missing_field_counts", {}))
    return {
        "clips": len(records),
        "fields": ["relative_clip", "scene_id", "episode_id", "num_frames", "seed"],
        "sort_key": "relative_clip",
        "serialization": "tab-separated str(fields), lexicographically sorted rows, trailing LF",
        "inventory_sha256": _inventory_hash(records),
        "canonical_records_sha256": canonical_sha256(records),
        "missing_field_counts": dict(sorted(missing.items())),
        "records": records,
    }


def _build_dataset(args: argparse.Namespace, config: dict[str, Any], split: str) -> Any:
    data_config = config["data"]
    sliding = data_config.get("sliding_window", {})
    return VLNSlidingWindowDataset(
        root=str(Path(args.data_root).resolve()),
        split=split,
        min_history=int(sliding.get("min_history", 5)),
        num_history_sample=int(args.num_history),
        image_size=tuple(data_config["image_size"]),
        hm_size=tuple(data_config["init_hm_size"]),
        load_depth=True,
        cache_poses=True,
        sample_stride=2,
        enable_augmentation=False,
        clip_level_sampling=False,
        load_history_frames=True,
        max_clips=0,
        max_clip_id=int(args.max_clip_id),
    )


def _validate_constraints(constraints: SelectionConstraints) -> None:
    if constraints.num_history <= 1:
        raise ValueError("Task-3.6 requires at least two history anchors")
    if constraints.min_temporal_lag < 3:
        raise ValueError("Non-recent history requires min_temporal_lag >= 3")
    if constraints.max_temporal_lag and constraints.max_temporal_lag < constraints.min_temporal_lag:
        raise ValueError("max_temporal_lag is smaller than min_temporal_lag")
    if not 1 <= constraints.min_distinct_views <= min(4, constraints.num_history):
        raise ValueError("min_distinct_views is incompatible with K")
    if constraints.min_visible_anchors > constraints.num_history:
        raise ValueError("min_visible_anchors exceeds K")
    if constraints.min_visible_distinct_views > min(4, constraints.min_visible_anchors):
        raise ValueError("min_visible_distinct_views exceeds visible-anchor support")
    if constraints.beam_width <= 0 or constraints.max_anchor_set_trials <= 0:
        raise ValueError("Beam width and trial count must be positive")
    if not 0.0 <= constraints.view_seam_margin_degrees < 45.0:
        raise ValueError("view_seam_margin_degrees must lie in [0,45)")
    if not 0.0 <= constraints.min_target_view_fraction <= 0.25:
        raise ValueError("min_target_view_fraction must lie in [0,0.25]")
    if not 0.25 <= constraints.max_target_view_fraction <= 1.0:
        raise ValueError("max_target_view_fraction must lie in [0.25,1]")
    if constraints.min_target_view_fraction > constraints.max_target_view_fraction:
        raise ValueError("min_target_view_fraction exceeds max_target_view_fraction")


def pose_free_model_input_contract() -> dict[str, Any]:
    """Declare the RGB-only attribution boundary for the Task-3.6 pilot."""
    return {
        "allowed": [
            "current_rgb_panorama",
            "ordered_history_rgb_observations",
        ],
        "forbidden": [
            "history_slot_id",
            "history_frame_index",
            "temporal_lag",
            "exact_relative_pose",
            "absolute_pose",
            "bearing",
            "spatial_distance",
            "target_view",
            "target_pixel",
        ],
        "label_only_metadata_path": "records[*].label_metadata",
        "loader_alignment_metadata_path": "records[*].loader_alignment",
        "record_alignment_fields": [
            "records[*].history_frames",
            "records[*].canonical_history_frames",
            "records[*].slot_permutation",
            "records[*].loader_alignment",
            "records[*].label_metadata.anchors[*].slot",
            "records[*].label_metadata.anchors[*].history_frame",
            "records[*].label_metadata.anchors[*].temporal_lag",
        ],
        "alignment_metadata_policy": (
            "slot/order/frame metadata is loader-and-label alignment state only "
            "and must never be forwarded to the model"
        ),
        "loader": "src.data.explicit_multi_history.ExplicitMultiHistoryDataset",
        "default_dataset_behavior_changed": False,
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    temporal_edges = parse_edges(args.temporal_lag_edges)
    distance_edges = parse_edges(args.spatial_distance_edges)
    constraints = SelectionConstraints(
        num_history=args.num_history,
        min_temporal_lag=args.min_temporal_lag,
        max_temporal_lag=args.max_temporal_lag,
        min_spatial_distance=args.min_spatial_distance,
        max_spatial_distance=args.max_spatial_distance,
        min_bearing_separation_degrees=args.min_bearing_separation_degrees,
        view_seam_margin_degrees=args.view_seam_margin_degrees,
        min_distinct_views=args.min_distinct_views,
        min_distinct_lag_bins=args.min_distinct_lag_bins,
        min_distinct_distance_bins=args.min_distinct_distance_bins,
        min_visible_anchors=args.min_visible_anchors,
        min_visible_distinct_views=args.min_visible_distinct_views,
        min_target_separation_pixels=args.min_target_separation_pixels,
        min_target_view_fraction=args.min_target_view_fraction,
        max_target_view_fraction=args.max_target_view_fraction,
        beam_width=args.beam_width,
        max_anchor_set_trials=args.max_anchor_set_trials,
    )
    _validate_constraints(constraints)
    if args.depth_valid_oversample_factor <= 0:
        raise ValueError("depth_valid_oversample_factor must be positive")
    selection_parameters = {
        **constraints.__dict__,
        "temporal_lag_edges": list(temporal_edges),
        "spatial_distance_edges": list(distance_edges),
        "candidate_currents_per_clip": int(args.candidate_currents_per_clip),
        "slot_order": args.slot_order,
        "seed": int(args.seed),
        "max_clip_id": int(args.max_clip_id),
        "depth_valid_oversample_factor": int(args.depth_valid_oversample_factor),
    }

    split_outputs: dict[str, dict[str, Any]] = {}
    selected_by_split: dict[str, list[dict[str, Any]]] = {}
    requested_by_split = {"train": int(args.train_samples), "val": int(args.val_samples)}
    datasets = {split: _build_dataset(args, config, split) for split in ("train", "val")}
    inventories = {split: source_inventory(dataset) for split, dataset in datasets.items()}
    inventory = merge_source_inventories([inventories["train"], inventories["val"]])
    expected_inventory_hash = str(args.expected_source_inventory_sha256).strip().lower()
    if expected_inventory_hash and inventory["inventory_sha256"] != expected_inventory_hash:
        raise RuntimeError(
            "Source inventory hash mismatch: "
            f"expected={expected_inventory_hash} actual={inventory['inventory_sha256']}"
        )
    for split in ("train", "val"):
        LOGGER.info("Building %s pose catalog", split)
        dataset = datasets[split]
        if not bool(getattr(dataset, "_is_panoramic", False)):
            raise ValueError(f"Task-3.6 requires panoramic data, split={split}")
        proposals, pose_failures = build_pose_catalog(
            dataset,
            constraints=constraints,
            temporal_lag_edges=temporal_edges,
            spatial_distance_edges=distance_edges,
            candidate_currents_per_clip=args.candidate_currents_per_clip,
            seed=args.seed,
        )
        requested = requested_by_split[split]
        pool_target = requested * max(int(args.depth_valid_oversample_factor), 1)
        depth_valid_pool, label_failures = materialize_selection(
            dataset,
            proposals,
            requested_samples=pool_target,
            constraints=constraints,
            slot_order=args.slot_order,
            seed=args.seed,
        )
        selected, balance_diagnostics = deterministic_balanced_selection(
            depth_valid_pool,
            requested_samples=requested,
            num_history=constraints.num_history,
            min_target_view_fraction=constraints.min_target_view_fraction,
            max_target_view_fraction=constraints.max_target_view_fraction,
            seed=args.seed,
        )
        if not balance_diagnostics["selection_complete"]:
            label_failures.append(
                {
                    "split": split,
                    "stage": "dataset_balance",
                    "reason": "balanced_selection_incomplete",
                    "details": balance_diagnostics,
                }
            )
        failures = [*pose_failures, *label_failures]
        selected_by_split[split] = selected
        split_outputs[split] = {
            "selection_manifest": selection_manifest(selected),
            "selection_audit": audit_selection(selected),
            "candidate_catalog": {
                "pose_valid_proposals": len(proposals),
                "proposal_sha256": canonical_sha256(proposals),
                "depth_valid_pool_samples": len(depth_valid_pool),
                "depth_valid_pool_sha256": canonical_sha256(depth_valid_pool),
            },
            "source_inventory": {
                key: value for key, value in inventories[split].items() if key != "records"
            },
            "balanced_selector": balance_diagnostics,
            "failure_audit": failure_audit(failures),
            "selection_complete": bool(balance_diagnostics["selection_complete"]),
            "records": selected,
        }

    assert_scene_disjoint(selected_by_split["train"], selected_by_split["val"])
    train_scenes = set(split_outputs["train"]["selection_manifest"]["scenes"])
    val_scenes = set(split_outputs["val"]["selection_manifest"]["scenes"])
    output: dict[str, Any] = {
        "schema_version": MULTI_HISTORY_SCHEMA,
        "selection_parameters": selection_parameters,
        "source_inventory_contract": {
            "max_clip_id": int(args.max_clip_id),
            "expected_inventory_sha256": expected_inventory_hash or None,
            "verified_against_expected": bool(expected_inventory_hash),
            **inventory,
        },
        "model_input_contract": pose_free_model_input_contract(),
        "scene_disjoint": {
            "verified": not bool(train_scenes & val_scenes),
            "overlap": sorted(train_scenes & val_scenes),
        },
        "ready": all(payload["selection_complete"] for payload in split_outputs.values()),
        "splits": split_outputs,
    }
    output["manifest_sha256"] = canonical_sha256(output)

    output_dir = Path(args.output_dir)
    _write_json(output_dir / "multi_history_selection_manifest.json", output)
    _write_jsonl(output_dir / "train_selection.jsonl", selected_by_split["train"])
    _write_jsonl(output_dir / "val_selection.jsonl", selected_by_split["val"])
    LOGGER.info(
        "Task-3.6 selection ready=%s train=%d/%d val=%d/%d manifest=%s",
        output["ready"],
        len(selected_by_split["train"]),
        args.train_samples,
        len(selected_by_split["val"]),
        args.val_samples,
        output["manifest_sha256"],
    )
    return 0 if output["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

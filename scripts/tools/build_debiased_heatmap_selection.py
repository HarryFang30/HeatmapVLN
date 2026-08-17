#!/usr/bin/env python3
"""Build and audit a deterministic, scene-disjoint Task-3.5b selection.

The original random-walk diagnostic is strongly affected by the immediately
previous history point appearing in the back view near the image centre.  A
plain scene round-robin split does not control that target prior.  This tool
therefore:

1. scans the exact heatmap targets without decoding RGB images;
2. describes every ``(history slot, panorama view)`` target by visibility,
   coordinate bin, temporal lag, and XY spatial lag;
3. greedily covers rare supported strata while enforcing explicit upper
   bounds on recent/back/centre dominance;
4. preserves the dataset's scene-disjoint train/validation split;
5. saves exact ordered identities, hashes, support/selection audits, and the
   empirical-prior strength before and after debiasing.

The selector never silently relaxes dominance bounds.  If the requested
sample count or a supported stratum cannot be covered, the report records the
shortfall and the exact candidate support so that an insufficient collection
cannot be mistaken for a balanced diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools.diagnose_heatmap_priors import (
    evaluate_empirical_prior,
    fit_empirical_prior,
    sample_identity,
)
from scripts.tools.diagnose_heatmap_shortcuts import (
    build_dataset,
    load_config,
    set_seed,
)

from src.data.trajectory_utils import compute_history_rel_poses

LOGGER = logging.getLogger("debiased_heatmap_selection")
SCHEMA_VERSION = "task35b_debiased_selection_v1"
VIEW_NAMES = ("front", "right", "back", "left")
PRIOR_METRICS = (
    "visibility_auroc",
    "visibility_auprc",
    "visible_view_accuracy",
    "median_pixel_error",
    "pck4",
    "pck8",
    "joint_median_pixel_error",
    "joint_pck4",
    "joint_pck8",
)

# Each family contributes one normalised term to the greedy score.  Slot/view
# and slot/view/coordinate coverage receive the largest weights because those
# are exactly the axes exploited by a slot-conditioned empirical prior.
FAMILY_WEIGHTS = {
    "slot_visibility": 2.0,
    "positive_slot": 3.0,
    "positive_view": 2.0,
    "slot_view": 4.0,
    "positive_coordinate": 2.0,
    "slot_coordinate": 3.0,
    "view_coordinate": 2.0,
    "slot_view_coordinate": 4.0,
    "slot_temporal_lag": 1.5,
    "slot_spatial_lag": 1.5,
}
CORE_BALANCE_FAMILIES = (
    "slot_visibility",
    "positive_slot",
    "positive_view",
    "slot_view",
    "positive_coordinate",
    "slot_coordinate",
    "slot_temporal_lag",
    "slot_spatial_lag",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-history", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--max-clip-id", type=int, default=2000)
    parser.add_argument("--candidate-samples-per-scene", type=int, default=0)
    parser.add_argument("--coordinate-grid-size", type=int, default=4)
    parser.add_argument("--center-radius-pixels", type=float, default=4.0)
    parser.add_argument(
        "--temporal-lag-edges",
        default="1,2,4,8,16,32",
        help="Comma-separated inclusive upper edges.",
    )
    parser.add_argument(
        "--spatial-lag-edges",
        default="0.25,0.5,1,2,4",
        help="Comma-separated XY-distance upper edges in metres.",
    )
    parser.add_argument("--max-recent-back-center-fraction", type=float, default=0.25)
    parser.add_argument("--max-recent-positive-fraction", type=float, default=0.60)
    parser.add_argument("--max-back-positive-fraction", type=float, default=0.45)
    parser.add_argument("--max-center-positive-fraction", type=float, default=0.45)
    parser.add_argument(
        "--allow-zero-visible-samples",
        action="store_true",
        help="Allow samples with no visible target in either history slot.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def parse_edges(value: str) -> tuple[float, ...]:
    edges = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not edges or any(not math.isfinite(edge) for edge in edges):
        raise ValueError("Lag-bin edges must contain finite values")
    if any(left >= right for left, right in pairwise(edges)):
        raise ValueError(f"Lag-bin edges must be strictly increasing: {edges}")
    return edges


def _stable_hash(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _format_edge(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def numeric_bin(value: float | None, edges: Sequence[float], prefix: str) -> str:
    if value is None or not math.isfinite(float(value)):
        return "unavailable"
    number = float(value)
    for edge in edges:
        if number <= float(edge):
            return f"{prefix}_le_{_format_edge(float(edge))}"
    return f"{prefix}_gt_{_format_edge(float(edges[-1]))}"


def coordinate_bin(
    x: int,
    y: int,
    *,
    width: int,
    height: int,
    grid_size: int,
) -> str:
    if grid_size <= 0:
        raise ValueError("coordinate grid size must be positive")
    column = min(grid_size - 1, max(0, int(x) * grid_size // max(width, 1)))
    row = min(grid_size - 1, max(0, int(y) * grid_size // max(height, 1)))
    return f"r{row}c{column}"


def _peak_xy(heatmap: torch.Tensor) -> tuple[int, int]:
    width = int(heatmap.shape[-1])
    flat_index = int(heatmap.reshape(-1).argmax().item())
    return flat_index % width, flat_index // width


def candidate_record_from_targets(
    *,
    dataset_index: int,
    sample_id: str,
    scene: str,
    current_frame: int,
    history_frames: Sequence[int] | None,
    gt_visibility: torch.Tensor,
    heatmap: torch.Tensor,
    history_rel_poses: torch.Tensor | None,
    coordinate_grid_size: int,
    center_radius_pixels: float,
    temporal_lag_edges: Sequence[float],
    spatial_lag_edges: Sequence[float],
    view_names: Sequence[str] = VIEW_NAMES,
) -> dict[str, Any]:
    """Create a compact, JSON-safe target descriptor for one sample."""
    visibility = torch.as_tensor(gt_visibility).detach().float().cpu()
    heatmaps = torch.as_tensor(heatmap).detach().float().cpu()
    if visibility.ndim != 2 or heatmaps.ndim != 4:
        raise ValueError(
            "Expected visibility [K,V] and heatmap [K,V,H,W], got "
            f"{tuple(visibility.shape)} and {tuple(heatmaps.shape)}"
        )
    if tuple(heatmaps.shape[:2]) != tuple(visibility.shape):
        raise ValueError("Visibility and heatmap slot/view shapes differ")
    slots, views = map(int, visibility.shape)
    if len(view_names) != views:
        view_names = tuple(f"view_{index}" for index in range(views))
    if history_frames is not None and len(history_frames) != slots:
        raise ValueError("history_frames length does not match heatmap slots")

    rel_poses: torch.Tensor | None = None
    if history_rel_poses is not None:
        rel_poses = torch.as_tensor(history_rel_poses).detach().float().cpu()
        if rel_poses.ndim != 2 or rel_poses.shape[0] != slots or rel_poses.shape[1] < 2:
            raise ValueError("history_rel_poses must have shape [K,>=2]")

    height, width = int(heatmaps.shape[-2]), int(heatmaps.shape[-1])
    centre_x = (width - 1) / 2.0
    centre_y = (height - 1) / 2.0
    slot_records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for slot in range(slots):
        temporal_lag = int(current_frame - int(history_frames[slot])) if history_frames is not None else None
        spatial_lag = None
        if rel_poses is not None:
            raw_spatial_lag = float(torch.linalg.vector_norm(rel_poses[slot, :2]).item())
            spatial_lag = raw_spatial_lag if math.isfinite(raw_spatial_lag) else None
        temporal_label = numeric_bin(temporal_lag, temporal_lag_edges, "t")
        spatial_label = numeric_bin(spatial_lag, spatial_lag_edges, "d")
        slot_record = {
            "slot": slot,
            "history_frame": (int(history_frames[slot]) if history_frames is not None else None),
            "temporal_lag": temporal_lag,
            "temporal_lag_bin": temporal_label,
            "spatial_lag": spatial_lag,
            "spatial_lag_bin": spatial_label,
            "any_visible": bool((visibility[slot] > 0.5).any().item()),
        }
        slot_records.append(slot_record)
        for view in range(views):
            visible = bool(visibility[slot, view].item() > 0.5)
            x: int | None = None
            y: int | None = None
            coord: str | None = None
            is_center = False
            if visible:
                x, y = _peak_xy(heatmaps[slot, view])
                coord = coordinate_bin(
                    x,
                    y,
                    width=width,
                    height=height,
                    grid_size=coordinate_grid_size,
                )
                is_center = bool(
                    abs(float(x) - centre_x) <= center_radius_pixels
                    and abs(float(y) - centre_y) <= center_radius_pixels
                )
            name = str(view_names[view])
            events.append(
                {
                    "slot": slot,
                    "view": view,
                    "view_name": name,
                    "visible": visible,
                    "x": x,
                    "y": y,
                    "coordinate_bin": coord,
                    "is_center": is_center,
                    "temporal_lag_bin": temporal_label,
                    "spatial_lag_bin": spatial_label,
                    "recent_back_center": bool(visible and slot == slots - 1 and name == "back" and is_center),
                }
            )

    return {
        "dataset_index": int(dataset_index),
        "sample_id": str(sample_id),
        "scene": str(scene),
        "current_frame": int(current_frame),
        "history_frames": ([int(value) for value in history_frames] if history_frames is not None else None),
        "heatmap_shape": [slots, views, height, width],
        "coordinate_grid_size": int(coordinate_grid_size),
        "slots": slot_records,
        "events": events,
    }


def load_target_without_rgb(dataset: Any, dataset_index: int) -> dict[str, Any]:
    """Reproduce dataset heatmap targets while skipping all RGB decoding."""
    if not bool(getattr(dataset, "_is_panoramic", False)):
        raise ValueError("Task-3.5b requires a panoramic dataset")
    clip_idx, current_frame = dataset.sample_index[dataset_index]
    clip_dir = dataset.clips[clip_idx]
    history_frames = dataset._sample_history_indices(
        0,
        current_frame,
        dataset.num_history_sample,
    )
    if len(history_frames) != int(dataset.num_history_sample):
        raise ValueError(f"Expected {dataset.num_history_sample} history frames, got {len(history_frames)}")
    poses = dataset._load_poses(clip_idx)
    history_poses = [poses[int(frame)] for frame in history_frames]
    current_pose = poses[int(current_frame)]
    image_size, intrinsics = dataset._load_intrinsics(clip_idx, clip_dir)
    hm_width, hm_height = dataset.hm_size
    heatmap, visibility = dataset._compute_per_history_multiview_heatmaps(
        clip_idx=clip_idx,
        clip_dir=clip_dir,
        history_poses=history_poses,
        current_t=int(current_frame),
        img_size=image_size,
        K=intrinsics,
        hm_size=(hm_height, hm_width),
    )
    relative_poses = torch.from_numpy(compute_history_rel_poses(history_poses, current_pose)).float()
    return {
        "gt_visibility": visibility,
        "heatmap": heatmap,
        "history_rel_poses": relative_poses,
        "history_frames": [int(value) for value in history_frames],
        "current_frame": int(current_frame),
    }


def nonterminal_indices(dataset: Any) -> list[int]:
    output: list[int] = []
    for dataset_index, (clip_idx, frame_idx) in enumerate(dataset.sample_index):
        valid_frames = dataset._clip_valid_frames.get(clip_idx, [])
        if valid_frames and int(frame_idx) == int(valid_frames[-1]):
            continue
        output.append(dataset_index)
    return output


def _limit_metadata_candidates(
    dataset: Any,
    indices: Sequence[int],
    *,
    per_scene: int,
    seed: int,
) -> list[int]:
    if per_scene <= 0:
        return list(indices)
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for dataset_index in indices:
        identity, scene = sample_identity(dataset, dataset_index)
        order = hashlib.sha256(f"{seed}\n{identity}".encode()).hexdigest()
        grouped[scene].append((order, dataset_index))
    selected: list[int] = []
    for scene in sorted(grouped):
        selected.extend(dataset_index for _order, dataset_index in sorted(grouped[scene])[:per_scene])
    return selected


def build_candidate_catalog(
    dataset: Any,
    *,
    coordinate_grid_size: int,
    center_radius_pixels: float,
    temporal_lag_edges: Sequence[float],
    spatial_lag_edges: Sequence[float],
    candidate_samples_per_scene: int = 0,
    seed: int = 42,
    target_loader: Callable[[Any, int], dict[str, Any]] = load_target_without_rgb,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    indices = _limit_metadata_candidates(
        dataset,
        nonterminal_indices(dataset),
        per_scene=candidate_samples_per_scene,
        seed=seed,
    )
    # sample_index is shuffled.  Clip/frame scan order keeps the pose, chunk,
    # and depth caches hot without changing either downstream selection order.
    indices.sort(key=lambda index: tuple(dataset.sample_index[index]))
    for position, dataset_index in enumerate(indices, start=1):
        identity, scene = sample_identity(dataset, dataset_index)
        try:
            target = target_loader(dataset, dataset_index)
            record = candidate_record_from_targets(
                dataset_index=dataset_index,
                sample_id=identity,
                scene=scene,
                current_frame=int(target["current_frame"]),
                history_frames=target.get("history_frames"),
                gt_visibility=target["gt_visibility"],
                heatmap=target["heatmap"],
                history_rel_poses=target.get("history_rel_poses"),
                coordinate_grid_size=coordinate_grid_size,
                center_radius_pixels=center_radius_pixels,
                temporal_lag_edges=temporal_lag_edges,
                spatial_lag_edges=spatial_lag_edges,
            )
            candidates.append(record)
        except Exception as exc:  # keep a full, auditable catalogue failure list
            failures.append(
                {
                    "dataset_index": int(dataset_index),
                    "sample_id": identity,
                    "scene": scene,
                    "error": repr(exc),
                }
            )
        if position % 100 == 0:
            LOGGER.info(
                "Catalogued split=%s %d/%d candidates (%d failures)",
                dataset.split,
                position,
                len(indices),
                len(failures),
            )
    return candidates, failures


def positive_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in record["events"] if bool(event["visible"])]


def record_feature_counts(record: dict[str, Any]) -> dict[str, Counter[str]]:
    families: dict[str, Counter[str]] = defaultdict(Counter)
    for slot in record["slots"]:
        slot_index = int(slot["slot"])
        families["slot_visibility"][f"s{slot_index}|visible={int(bool(slot['any_visible']))}"] += 1
        families["slot_temporal_lag"][f"s{slot_index}|{slot['temporal_lag_bin']}"] += 1
        families["slot_spatial_lag"][f"s{slot_index}|{slot['spatial_lag_bin']}"] += 1
    for event in positive_events(record):
        slot = int(event["slot"])
        view = str(event["view_name"])
        coord = str(event["coordinate_bin"])
        families["positive_slot"][f"s{slot}"] += 1
        families["positive_view"][view] += 1
        families["slot_view"][f"s{slot}|{view}"] += 1
        families["positive_coordinate"][coord] += 1
        families["slot_coordinate"][f"s{slot}|{coord}"] += 1
        families["view_coordinate"][f"{view}|{coord}"] += 1
        families["slot_view_coordinate"][f"s{slot}|{view}|{coord}"] += 1
    return families


def _feature_support(
    records: Sequence[dict[str, Any]],
) -> dict[str, Counter[str]]:
    support: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        features = record_feature_counts(record)
        for family, categories in features.items():
            for category in categories:
                support[family][category] += 1
    return support


def _expected_feature_categories(
    records: Sequence[dict[str, Any]],
) -> dict[str, set[str]]:
    """Return the finite slot/view/coordinate universe implied by the schema."""
    if not records:
        return {}
    first = records[0]
    slots = int(first["heatmap_shape"][0])
    grid_size = int(first.get("coordinate_grid_size", 0))
    view_names = [str(event["view_name"]) for event in first["events"] if int(event["slot"]) == 0]
    coordinates = [f"r{row}c{column}" for row in range(grid_size) for column in range(grid_size)]
    expected: dict[str, set[str]] = {
        "slot_visibility": {f"s{slot}|visible={visible}" for slot in range(slots) for visible in (0, 1)},
        "positive_slot": {f"s{slot}" for slot in range(slots)},
        "positive_view": set(view_names),
        "slot_view": {f"s{slot}|{view}" for slot in range(slots) for view in view_names},
        "positive_coordinate": set(coordinates),
        "slot_coordinate": {f"s{slot}|{coordinate}" for slot in range(slots) for coordinate in coordinates},
        "view_coordinate": {f"{view}|{coordinate}" for view in view_names for coordinate in coordinates},
        "slot_view_coordinate": {
            f"s{slot}|{view}|{coordinate}" for slot in range(slots) for view in view_names for coordinate in coordinates
        },
    }
    return expected


def dominance_counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    visible = 0
    recent = 0
    back = 0
    centre = 0
    recent_back_centre = 0
    for record in records:
        recent_slot = int(record["heatmap_shape"][0]) - 1
        for event in positive_events(record):
            visible += 1
            recent += int(int(event["slot"]) == recent_slot)
            back += int(str(event["view_name"]) == "back")
            centre += int(bool(event["is_center"]))
            recent_back_centre += int(bool(event["recent_back_center"]))
    return {
        "visible_positive_events": visible,
        "recent_positive_events": recent,
        "back_positive_events": back,
        "center_positive_events": centre,
        "recent_back_center_events": recent_back_centre,
    }


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0


def dominance_fractions(records: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    counts = dominance_counts(records)
    total = counts["visible_positive_events"]
    return {
        **counts,
        "recent_positive_fraction": _safe_fraction(counts["recent_positive_events"], total),
        "back_positive_fraction": _safe_fraction(counts["back_positive_events"], total),
        "center_positive_fraction": _safe_fraction(counts["center_positive_events"], total),
        "recent_back_center_fraction": _safe_fraction(counts["recent_back_center_events"], total),
    }


def _dominance_feasible(
    records: Sequence[dict[str, Any]],
    candidate: dict[str, Any],
    constraints: dict[str, float],
) -> bool:
    fractions = dominance_fractions([*records, candidate])
    return bool(
        fractions["recent_back_center_fraction"] <= constraints["max_recent_back_center_fraction"] + 1e-12
        and fractions["recent_positive_fraction"] <= constraints["max_recent_positive_fraction"] + 1e-12
        and fractions["back_positive_fraction"] <= constraints["max_back_positive_fraction"] + 1e-12
        and fractions["center_positive_fraction"] <= constraints["max_center_positive_fraction"] + 1e-12
    )


def _selection_score(
    record: dict[str, Any],
    selected_features: dict[str, Counter[str]],
    support: dict[str, Counter[str]],
    selected_scene_counts: Counter[str],
) -> float:
    score = 0.0
    features = record_feature_counts(record)
    for family, categories in features.items():
        terms: list[float] = []
        for category, contribution in categories.items():
            selected_count = selected_features[family][category]
            support_count = max(int(support[family][category]), 1)
            novelty = 1.0 if selected_count == 0 else 0.0
            deficit = 1.0 / ((selected_count + 1.0) * math.sqrt(support_count))
            terms.append(min(int(contribution), 1) * (deficit + novelty))
        if terms:
            score += FAMILY_WEIGHTS.get(family, 1.0) * sum(terms) / len(terms)
    # Soft scene balancing remains active even after the hard scene cap has to
    # be increased because some scenes lack constraint-compatible examples.
    score += 1.0 / (1.0 + selected_scene_counts[str(record["scene"])])
    score += 0.05 * len(positive_events(record))
    return score


def _add_features(
    aggregate: dict[str, Counter[str]],
    record: dict[str, Any],
) -> None:
    for family, categories in record_feature_counts(record).items():
        aggregate[family].update(categories)


def deterministic_debiased_selection(
    records: Sequence[dict[str, Any]],
    *,
    limit: int,
    seed: int,
    constraints: dict[str, float],
    require_visible_event: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedily cover supported strata without relaxing dominance bounds."""
    if limit <= 0:
        return [], {
            "requested_samples": int(limit),
            "selected_samples": 0,
            "selection_complete": True,
            "unmet_constraints": [],
        }
    eligible = [record for record in records if not require_visible_event or bool(positive_events(record))]
    eligible.sort(key=lambda record: hashlib.sha256(f"{seed}\n{record['sample_id']}".encode()).hexdigest())
    support = _feature_support(eligible)
    scenes = sorted({str(record["scene"]) for record in eligible})
    initial_scene_cap = max(1, math.ceil(limit / max(len(scenes), 1)))
    scene_cap = initial_scene_cap
    max_scene_capacity = max(
        Counter(str(record["scene"]) for record in eligible).values(),
        default=0,
    )
    selected: list[dict[str, Any]] = []
    remaining = list(eligible)
    selected_features: dict[str, Counter[str]] = defaultdict(Counter)
    selected_scene_counts: Counter[str] = Counter()

    while len(selected) < limit and remaining:
        best_index: int | None = None
        best_score = -math.inf
        best_tie = ""
        for index, record in enumerate(remaining):
            scene = str(record["scene"])
            if selected_scene_counts[scene] >= scene_cap:
                continue
            if not _dominance_feasible(selected, record, constraints):
                continue
            score = _selection_score(
                record,
                selected_features,
                support,
                selected_scene_counts,
            )
            tie = hashlib.sha256(f"{seed}\n{record['sample_id']}".encode()).hexdigest()
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12 and (best_index is None or tie < best_tie)
            ):
                best_index = index
                best_score = score
                best_tie = tie
        if best_index is None:
            # Only scene balance may relax.  Target-dominance constraints are
            # deliberately never relaxed because that would invalidate 3.5b.
            if scene_cap < max_scene_capacity:
                scene_cap += 1
                continue
            break
        record = remaining.pop(best_index)
        selected.append(record)
        selected_scene_counts[str(record["scene"])] += 1
        _add_features(selected_features, record)

    fractions = dominance_fractions(selected)
    unmet: list[str] = []
    if len(selected) < limit:
        unmet.append(f"sample_count_shortfall:{limit - len(selected)}")
    hard_checks = {
        "recent_back_center_fraction": (
            float(fractions["recent_back_center_fraction"]),
            constraints["max_recent_back_center_fraction"],
        ),
        "recent_positive_fraction": (
            float(fractions["recent_positive_fraction"]),
            constraints["max_recent_positive_fraction"],
        ),
        "back_positive_fraction": (
            float(fractions["back_positive_fraction"]),
            constraints["max_back_positive_fraction"],
        ),
        "center_positive_fraction": (
            float(fractions["center_positive_fraction"]),
            constraints["max_center_positive_fraction"],
        ),
    }
    for name, (actual, maximum) in hard_checks.items():
        if actual > maximum + 1e-12:
            unmet.append(f"{name}:{actual:.6f}>{maximum:.6f}")

    selected_support = _feature_support(selected)
    expected_categories = _expected_feature_categories(eligible)
    coverage: dict[str, Any] = {}
    for family in sorted(set(support) | set(expected_categories)):
        supported = sorted(support[family])
        covered = sorted(selected_support[family])
        expected = sorted(expected_categories.get(family, set()))
        counts_over_supported = [int(selected_support[family][item]) for item in supported]
        minimum = min(counts_over_supported, default=0)
        maximum = max(counts_over_supported, default=0)
        coverage[family] = {
            "expected_categories": expected,
            "supported_categories": supported,
            "covered_categories": covered,
            "unsupported_expected_categories": sorted(set(expected) - set(supported)),
            "missing_supported_categories": sorted(set(supported) - set(covered)),
            "candidate_support_fraction_of_expected": (
                _safe_fraction(len(supported), len(expected)) if expected else None
            ),
            "coverage_fraction": _safe_fraction(len(covered), len(supported)),
            "selected_count_min_over_supported": minimum,
            "selected_count_max_over_supported": maximum,
            "selected_count_spread_over_supported": maximum - minimum,
            "exact_count_balance_over_supported": bool(supported and maximum - minimum <= 1),
            "candidate_support": dict(sorted(support[family].items())),
            "selected_counts": dict(sorted(selected_support[family].items())),
        }
    missing_slots = coverage.get("positive_slot", {}).get("missing_supported_categories", [])
    if missing_slots:
        unmet.append("missing_positive_slots:" + ",".join(missing_slots))

    diagnostics = {
        "requested_samples": int(limit),
        "eligible_candidates": len(eligible),
        "excluded_zero_visible_candidates": len(records) - len(eligible),
        "selected_samples": len(selected),
        "selection_complete": len(selected) == limit and not unmet,
        "scene_cap_initial": initial_scene_cap,
        "scene_cap_final": scene_cap,
        "scene_cap_relaxed": scene_cap != initial_scene_cap,
        "scene_counts": dict(sorted(selected_scene_counts.items())),
        "dominance": fractions,
        "dominance_constraints": constraints,
        "stratum_coverage": coverage,
        "unmet_constraints": unmet,
    }
    return selected, diagnostics


def scene_round_robin_selection(
    records: Sequence[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Reproduce the existing dataset-order scene-stratified baseline."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in sorted(records, key=lambda item: int(item["dataset_index"])):
        grouped[str(record["scene"])].append(record)
    selected: list[dict[str, Any]] = []
    cursors = {scene: 0 for scene in grouped}
    while len(selected) < limit:
        made_progress = False
        for scene in sorted(grouped):
            cursor = cursors[scene]
            if cursor >= len(grouped[scene]):
                continue
            selected.append(grouped[scene][cursor])
            cursors[scene] += 1
            made_progress = True
            if len(selected) >= limit:
                break
        if not made_progress:
            break
    return selected


def audit_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    slot_stats: dict[str, dict[str, Any]] = {}
    slot_count = max((int(record["heatmap_shape"][0]) for record in records), default=0)
    for slot in range(slot_count):
        slot_records = [item for record in records for item in record["slots"] if int(item["slot"]) == slot]
        events = [event for record in records for event in record["events"] if int(event["slot"]) == slot]
        positives = [event for event in events if bool(event["visible"])]
        slot_stats[str(slot)] = {
            "samples": len(slot_records),
            "samples_with_any_visible": sum(bool(item["any_visible"]) for item in slot_records),
            "positive_view_events": len(positives),
            "view_counts": dict(sorted(Counter(str(item["view_name"]) for item in positives).items())),
            "coordinate_bin_counts": dict(sorted(Counter(str(item["coordinate_bin"]) for item in positives).items())),
            "temporal_lag_bin_counts": dict(
                sorted(Counter(str(item["temporal_lag_bin"]) for item in slot_records).items())
            ),
            "spatial_lag_bin_counts": dict(
                sorted(Counter(str(item["spatial_lag_bin"]) for item in slot_records).items())
            ),
        }
    cross_strata = Counter()
    for record in records:
        for event in record["events"]:
            coordinate = event["coordinate_bin"] if event["visible"] else "none"
            key = (
                f"s{event['slot']}|vis={int(bool(event['visible']))}|{event['view_name']}|"
                f"coord={coordinate}|{event['temporal_lag_bin']}|{event['spatial_lag_bin']}"
            )
            cross_strata[key] += 1
    scene_counts = Counter(str(record["scene"]) for record in records)
    zero_visible = sum(not positive_events(record) for record in records)
    return {
        "samples": len(records),
        "scenes": len(scene_counts),
        "scene_counts": dict(sorted(scene_counts.items())),
        "zero_visible_samples": zero_visible,
        "dominance": dominance_fractions(records),
        "per_history_slot": slot_stats,
        "cross_stratum_counts": dict(sorted(cross_strata.items())),
    }


def selection_manifest(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sample_ids = [str(record["sample_id"]) for record in records]
    scenes = sorted({str(record["scene"]) for record in records})
    indices = [int(record["dataset_index"]) for record in records]
    return {
        "sample_count": len(records),
        "dataset_indices": indices,
        "sample_ids": sample_ids,
        "sample_identity_sha256": _stable_hash(sample_ids),
        "descriptor_sha256": _canonical_hash(list(records)),
        "scenes": scenes,
        "scene_sha256": _stable_hash(scenes),
    }


def assert_scene_disjoint_records(
    train_records: Sequence[dict[str, Any]],
    val_records: Sequence[dict[str, Any]],
) -> None:
    train_scenes = {str(record["scene"]) for record in train_records}
    val_scenes = {str(record["scene"]) for record in val_records}
    overlap = sorted(train_scenes & val_scenes)
    if overlap:
        raise RuntimeError("Train/validation scenes overlap: " + ", ".join(overlap))


class _TargetSubset:
    """Minimal dataset adapter consumed by the existing empirical-prior code."""

    def __init__(
        self,
        source: Any,
        records: Sequence[dict[str, Any]],
        targets: dict[int, dict[str, Any]],
    ) -> None:
        self.root = source.root
        self.clips = source.clips
        self.sample_index = [source.sample_index[int(record["dataset_index"])] for record in records]
        self._samples = [targets[int(record["dataset_index"])] for record in records]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._samples[index]


def _selected_targets(
    dataset: Any,
    groups: Sequence[Sequence[dict[str, Any]]],
    *,
    target_loader: Callable[[Any, int], dict[str, Any]] = load_target_without_rgb,
) -> dict[int, dict[str, Any]]:
    indices = sorted({int(record["dataset_index"]) for records in groups for record in records})
    return {index: target_loader(dataset, index) for index in indices}


def empirical_prior_strength(
    train_dataset: Any,
    val_dataset: Any,
    train_records: Sequence[dict[str, Any]],
    val_records: Sequence[dict[str, Any]],
    *,
    train_targets: dict[int, dict[str, Any]] | None = None,
    val_targets: dict[int, dict[str, Any]] | None = None,
    visibility_alpha: float = 0.5,
    target_loader: Callable[[Any, int], dict[str, Any]] = load_target_without_rgb,
) -> dict[str, Any]:
    if not train_records or not val_records:
        return {"available": False, "reason": "empty train or validation selection"}
    if train_targets is None:
        train_targets = _selected_targets(
            train_dataset,
            [train_records],
            target_loader=target_loader,
        )
    if val_targets is None:
        val_targets = _selected_targets(
            val_dataset,
            [val_records],
            target_loader=target_loader,
        )
    train_subset = _TargetSubset(train_dataset, train_records, train_targets)
    val_subset = _TargetSubset(val_dataset, val_records, val_targets)
    prior = fit_empirical_prior(
        train_subset,
        list(range(len(train_records))),
        visibility_alpha=visibility_alpha,
    )
    metrics, _compact = evaluate_empirical_prior(
        val_subset,
        list(range(len(val_records))),
        prior,
    )
    return {
        "available": True,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "metrics": {metric: metrics.get(metric) for metric in PRIOR_METRICS},
    }


def compare_prior_strength(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    if not before.get("available") or not after.get("available"):
        return {
            "available": False,
            "reason": "before or after empirical-prior evaluation unavailable",
        }
    before_metrics = before["metrics"]
    after_metrics = after["metrics"]
    delta_after_minus_before = {
        metric: (
            float(after_metrics[metric]) - float(before_metrics[metric])
            if after_metrics.get(metric) is not None and before_metrics.get(metric) is not None
            else None
        )
        for metric in PRIOR_METRICS
    }
    weaker = bool(
        delta_after_minus_before["median_pixel_error"] is not None
        and delta_after_minus_before["median_pixel_error"] >= 0.0
        and delta_after_minus_before["pck8"] is not None
        and delta_after_minus_before["pck8"] <= 0.0
        and delta_after_minus_before["joint_pck8"] is not None
        and delta_after_minus_before["joint_pck8"] <= 0.0
    )
    return {
        "available": True,
        "delta_after_minus_before": delta_after_minus_before,
        "shortcut_reduction": {
            "median_error_increase": delta_after_minus_before["median_pixel_error"],
            "pck8_drop": -float(delta_after_minus_before["pck8"]),
            "joint_pck8_drop": -float(delta_after_minus_before["joint_pck8"]),
            "empirical_prior_weaker_on_all_localization_checks": weaker,
        },
        "interpretation": (
            "Descriptive distribution audit only: the before/after validation sets differ by "
            "construction, so this does not estimate a model-treatment effect."
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False)


def _json_safe(value: Any) -> Any:
    """Represent non-finite diagnostic metrics as JSON null, never NaN."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            handle.write("\n")


def _split_report(
    *,
    candidates: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]],
    debiased: Sequence[dict[str, Any]],
    selector: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_catalog": {
            "records": len(candidates),
            "failures": len(failures),
            "record_sha256": _canonical_hash(list(candidates)),
            "failure_records": list(failures),
            "audit": audit_records(candidates),
        },
        "baseline": {
            "manifest": selection_manifest(baseline),
            "audit": audit_records(baseline),
        },
        "debiased": {
            "manifest": selection_manifest(debiased),
            "audit": audit_records(debiased),
            "selector": selector,
        },
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.num_history != 2:
        LOGGER.warning(
            "Task-3.5b was designed for two slots; recent means slot K-1 for K=%d",
            args.num_history,
        )
    if args.coordinate_grid_size <= 0 or args.center_radius_pixels < 0.0:
        raise ValueError("Invalid coordinate grid or centre radius")
    temporal_edges = parse_edges(args.temporal_lag_edges)
    spatial_edges = parse_edges(args.spatial_lag_edges)
    constraints = {
        "max_recent_back_center_fraction": args.max_recent_back_center_fraction,
        "max_recent_positive_fraction": args.max_recent_positive_fraction,
        "max_back_positive_fraction": args.max_back_positive_fraction,
        "max_center_positive_fraction": args.max_center_positive_fraction,
    }
    if any(value < 0.0 or value > 1.0 for value in constraints.values()):
        raise ValueError("Dominance fractions must lie in [0,1]")

    set_seed(args.seed)
    cfg = load_config(args)
    train_dataset = build_dataset(cfg, "train", max_clip_id=args.max_clip_id)
    val_dataset = build_dataset(cfg, "val", max_clip_id=args.max_clip_id)
    train_candidates, train_failures = build_candidate_catalog(
        train_dataset,
        coordinate_grid_size=args.coordinate_grid_size,
        center_radius_pixels=args.center_radius_pixels,
        temporal_lag_edges=temporal_edges,
        spatial_lag_edges=spatial_edges,
        candidate_samples_per_scene=args.candidate_samples_per_scene,
        seed=args.seed,
    )
    val_candidates, val_failures = build_candidate_catalog(
        val_dataset,
        coordinate_grid_size=args.coordinate_grid_size,
        center_radius_pixels=args.center_radius_pixels,
        temporal_lag_edges=temporal_edges,
        spatial_lag_edges=spatial_edges,
        candidate_samples_per_scene=args.candidate_samples_per_scene,
        seed=args.seed,
    )
    assert_scene_disjoint_records(train_candidates, val_candidates)

    train_baseline = scene_round_robin_selection(train_candidates, args.train_samples)
    val_baseline = scene_round_robin_selection(val_candidates, args.val_samples)
    train_debiased, train_selector = deterministic_debiased_selection(
        train_candidates,
        limit=args.train_samples,
        seed=args.seed,
        constraints=constraints,
        require_visible_event=not args.allow_zero_visible_samples,
    )
    val_debiased, val_selector = deterministic_debiased_selection(
        val_candidates,
        limit=args.val_samples,
        seed=args.seed,
        constraints=constraints,
        require_visible_event=not args.allow_zero_visible_samples,
    )
    assert_scene_disjoint_records(train_debiased, val_debiased)

    train_targets = _selected_targets(
        train_dataset,
        [train_baseline, train_debiased],
    )
    val_targets = _selected_targets(
        val_dataset,
        [val_baseline, val_debiased],
    )
    prior_before = empirical_prior_strength(
        train_dataset,
        val_dataset,
        train_baseline,
        val_baseline,
        train_targets=train_targets,
        val_targets=val_targets,
    )
    prior_after = empirical_prior_strength(
        train_dataset,
        val_dataset,
        train_debiased,
        val_debiased,
        train_targets=train_targets,
        val_targets=val_targets,
    )
    prior_baseline_train_debiased_val = empirical_prior_strength(
        train_dataset,
        val_dataset,
        train_baseline,
        val_debiased,
        train_targets=train_targets,
        val_targets=val_targets,
    )
    prior_debiased_train_baseline_val = empirical_prior_strength(
        train_dataset,
        val_dataset,
        train_debiased,
        val_baseline,
        train_targets=train_targets,
        val_targets=val_targets,
    )

    train_report = _split_report(
        candidates=train_candidates,
        failures=train_failures,
        baseline=train_baseline,
        debiased=train_debiased,
        selector=train_selector,
    )
    val_report = _split_report(
        candidates=val_candidates,
        failures=val_failures,
        baseline=val_baseline,
        debiased=val_debiased,
        selector=val_selector,
    )
    limitations = []
    support_limitations = []
    count_balance_limitations = []
    for split, selector in (("train", train_selector), ("val", val_selector)):
        limitations.extend(f"{split}:{value}" for value in selector["unmet_constraints"])
        for family, coverage in selector["stratum_coverage"].items():
            missing = coverage["missing_supported_categories"]
            if missing:
                limitations.append(f"{split}:uncovered_{family}:" + ",".join(missing))
            if family not in CORE_BALANCE_FAMILIES:
                continue
            unsupported = coverage["unsupported_expected_categories"]
            if unsupported:
                support_limitations.append(f"{split}:unsupported_{family}:" + ",".join(unsupported))
            if not coverage["exact_count_balance_over_supported"]:
                count_balance_limitations.append(
                    f"{split}:unequal_{family}:spread={coverage['selected_count_spread_over_supported']}"
                )
    limitations.extend(support_limitations)
    limitations.extend(count_balance_limitations)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "candidate_catalog_train.jsonl", train_candidates)
    _write_jsonl(output_dir / "candidate_catalog_val.jsonl", val_candidates)
    selections = {
        "schema_version": SCHEMA_VERSION,
        "baseline": {
            "train": selection_manifest(train_baseline),
            "val": selection_manifest(val_baseline),
        },
        "debiased": {
            "train": selection_manifest(train_debiased),
            "val": selection_manifest(val_debiased),
        },
    }
    _write_json(output_dir / "selection_manifest.json", selections)
    report = {
        "task": "task35b_debiased_data_diagnostic",
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "config": str(Path(args.config).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "max_clip_id": args.max_clip_id,
        "num_history": args.num_history,
        "scene_disjoint": True,
        "selection_contract": {
            "coordinate_grid_size": args.coordinate_grid_size,
            "center_radius_pixels": args.center_radius_pixels,
            "temporal_lag_edges": list(temporal_edges),
            "spatial_lag_edges": list(spatial_edges),
            "candidate_samples_per_scene": args.candidate_samples_per_scene,
            "require_visible_event": not args.allow_zero_visible_samples,
            "dominance_constraints": constraints,
            "dominance_constraints_relaxed": False,
            "selector": "deterministic_weighted_stratum_coverage_v1",
        },
        "train": train_report,
        "val": val_report,
        "empirical_prior_strength": {
            "before_scene_round_robin": prior_before,
            "after_debiased": prior_after,
            "cross_evaluations": {
                "baseline_train_on_debiased_val": prior_baseline_train_debiased_val,
                "debiased_train_on_baseline_val": prior_debiased_train_baseline_val,
            },
            "comparison": compare_prior_strength(prior_before, prior_after),
        },
        "selection_ready_for_diagnostic": bool(
            train_selector["selection_complete"] and val_selector["selection_complete"]
        ),
        "candidate_support_complete_for_core_marginals": not support_limitations,
        "exact_count_balance_achieved_for_core_marginals": not count_balance_limitations,
        "balance_complete": not limitations,
        "limitations": limitations,
        "artifacts": {
            "selection_manifest": str(output_dir / "selection_manifest.json"),
            "candidate_catalog_train": str(output_dir / "candidate_catalog_train.jsonl"),
            "candidate_catalog_val": str(output_dir / "candidate_catalog_val.jsonl"),
            "report": str(output_dir / "report.json"),
        },
    }
    safe_report = _json_safe(report)
    _write_json(output_dir / "report.json", safe_report)
    print(json.dumps(safe_report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())

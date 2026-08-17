#!/usr/bin/env python3
"""Diagnose scene-disjoint STOP-add signal in frozen trajectory features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

TRAJECTORY_FEATURE_SCHEMA = "heatmapvln-system2-stop-trajectory-feature-v1"


def _auc(targets: list[int], scores: list[float]) -> float:
    positives = [score for target, score in zip(targets, scores) if target == 1]
    negatives = [score for target, score in zip(targets, scores) if target == 0]
    if not positives or not negatives:
        raise ValueError("AUC requires both classes")
    wins = sum(
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _stable_scene_folds(scenes: list[str], fold_count: int, seed: int) -> list[list[str]]:
    if fold_count < 2 or fold_count > len(scenes):
        raise ValueError("fold_count must be between 2 and the number of scenes")
    ordered = sorted(
        scenes,
        key=lambda scene: hashlib.sha256(f"{seed}:{scene}".encode()).digest(),
    )
    folds = [[] for _ in range(fold_count)]
    for index, scene in enumerate(ordered):
        folds[index % fold_count].append(scene)
    if any(not fold for fold in folds):
        raise RuntimeError("Scene fold construction produced an empty fold")
    return folds


def _center_kernel(
    full_kernel: torch.Tensor,
    train_indices: list[int],
    val_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center train and validation features using only the training-fold mean."""
    train = torch.tensor(train_indices, dtype=torch.long)
    val = torch.tensor(val_indices, dtype=torch.long)
    train_kernel = full_kernel.index_select(0, train).index_select(1, train)
    val_train_kernel = full_kernel.index_select(0, val).index_select(1, train)
    train_column_mean = train_kernel.mean(dim=0)
    train_mean = train_kernel.mean()
    centered_train = (
        train_kernel
        - train_kernel.mean(dim=1, keepdim=True)
        - train_column_mean.unsqueeze(0)
        + train_mean
    )
    centered_val_train = (
        val_train_kernel
        - val_train_kernel.mean(dim=1, keepdim=True)
        - train_column_mean.unsqueeze(0)
        + train_mean
    )
    return centered_train, centered_val_train


def _normalize_block(features: torch.Tensor) -> torch.Tensor:
    features = features.float().flatten(1)
    if not bool(torch.isfinite(features).all()):
        raise RuntimeError("Trajectory diagnostic feature block contains non-finite values")
    scale = features.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    return features / scale


def _kernel_from_blocks(blocks: list[torch.Tensor]) -> torch.Tensor:
    if not blocks:
        raise ValueError("At least one trajectory feature block is required")
    normalized = [_normalize_block(block) for block in blocks]
    row_count = normalized[0].shape[0]
    if any(block.shape[0] != row_count for block in normalized):
        raise ValueError("Trajectory feature blocks do not align")
    kernel = sum(
        block @ block.T / float(block.shape[1])
        for block in normalized
    ) / float(len(normalized))
    if not bool(torch.isfinite(kernel).all()):
        raise RuntimeError("Trajectory diagnostic kernel contains non-finite values")
    return kernel.double()


def _ridge_oof_scores(
    kernel: torch.Tensor,
    targets: list[int],
    scene_ids: list[str],
    folds: list[list[str]],
    ridge: float,
) -> list[float]:
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")
    target = torch.tensor([2 * value - 1 for value in targets], dtype=torch.float64)
    scores = torch.empty(len(targets), dtype=torch.float64)
    all_indices = set(range(len(targets)))
    for val_scenes in folds:
        val_scene_set = set(val_scenes)
        val_indices = [i for i, scene in enumerate(scene_ids) if scene in val_scene_set]
        train_indices = sorted(all_indices - set(val_indices))
        if set(target[train_indices].int().tolist()) != {-1, 1}:
            raise RuntimeError("A scene fold lacks one STOP class")
        train_kernel, val_train_kernel = _center_kernel(
            kernel,
            train_indices,
            val_indices,
        )
        system = train_kernel + ridge * torch.eye(
            len(train_indices), dtype=torch.float64
        )
        alpha = torch.linalg.solve(system, target[train_indices])
        scores[val_indices] = val_train_kernel @ alpha
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError("Trajectory diagnostic produced non-finite OOF scores")
    return scores.tolist()


def _group_metrics(
    rows: list[dict[str, Any]],
    scores: list[float],
    aggregation: str,
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["sweep_id"])].append(index)
    group_rows: list[dict[str, Any]] = []
    for sweep_id, indices in grouped.items():
        values = torch.tensor([scores[index] for index in indices], dtype=torch.float64)
        if aggregation == "mean":
            score = float(values.mean().item())
        elif aggregation == "median":
            score = float(values.median().item())
        elif aggregation == "minimum":
            score = float(values.min().item())
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        first = rows[indices[0]]
        if any(
            rows[index]["scene_id"] != first["scene_id"]
            or rows[index]["target"] != first["target"]
            for index in indices
        ):
            raise RuntimeError(f"Inconsistent trajectory probe group: {sweep_id}")
        group_rows.append(
            {
                "sweep_id": sweep_id,
                "scene_id": first["scene_id"],
                "target": first["target"],
                "views": len(indices),
                "score": score,
            }
        )
    group_targets = [int(row["target"]) for row in group_rows]
    group_scores = [float(row["score"]) for row in group_rows]
    max_negative = max(
        score for target, score in zip(group_targets, group_scores) if target == 0
    )
    scenes = sorted({str(row["scene_id"]) for row in group_rows})
    paired_wins = 0
    for scene in scenes:
        scene_rows = [row for row in group_rows if row["scene_id"] == scene]
        positives = [row["score"] for row in scene_rows if row["target"] == 1]
        negatives = [row["score"] for row in scene_rows if row["target"] == 0]
        if len(positives) != 1 or len(negatives) != 1:
            raise RuntimeError(f"Scene {scene} does not have one boundary/goal pair")
        paired_wins += int(positives[0] > negatives[0])
    return {
        "aggregation": aggregation,
        "groups": len(group_rows),
        "auc": _auc(group_targets, group_scores),
        "paired_wins": paired_wins,
        "paired_total": len(scenes),
        "zero_false_positive_goal_catches": sum(
            score > max_negative
            for target, score in zip(group_targets, group_scores)
            if target == 1
        ),
        "max_negative_score": max_negative,
        "records": sorted(group_rows, key=lambda row: (row["scene_id"], row["target"])),
    }


def _view_metrics(rows: list[dict[str, Any]], scores: list[float]) -> dict[str, Any]:
    targets = [int(row["target"]) for row in rows]
    max_negative = max(
        score for target, score in zip(targets, scores) if target == 0
    )
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["target"] == 1:
            grouped[str(row["sweep_id"])].append(index)
    first_view_hits = 0
    first_two_view_hits = 0
    any_view_hits = 0
    for indices in grouped.values():
        ordered = sorted(indices, key=lambda index: int(rows[index]["probe_index"]))
        hits = [scores[index] > max_negative for index in ordered]
        first_view_hits += int(hits[0])
        first_two_view_hits += int(len(hits) >= 2 and hits[0] and hits[1])
        any_view_hits += int(any(hits))
    return {
        "auc": _auc(targets, scores),
        "max_negative_score": max_negative,
        "zero_false_positive_goal_views": sum(
            score > max_negative
            for target, score in zip(targets, scores)
            if target == 1
        ),
        "positive_views": sum(targets),
        "goal_groups": len(grouped),
        "goal_groups_first_view_hit": first_view_hits,
        "goal_groups_first_two_views_hit": first_two_view_hits,
        "goal_groups_any_view_hit": any_view_hits,
    }


def _confirmation_metrics(
    rows: list[dict[str, Any]],
    scores: list[float],
    confirmations: int = 2,
) -> dict[str, Any]:
    """Calibrate the exact consecutive-view event used by closed-loop STOP."""
    if confirmations < 2:
        raise ValueError("Confirmation metrics require at least two views")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["sweep_id"])].append(index)
    events: list[dict[str, Any]] = []
    for sweep_id, indices in grouped.items():
        ordered = sorted(indices, key=lambda index: int(rows[index]["probe_index"]))
        if len(ordered) < confirmations:
            raise RuntimeError(
                f"Probe group {sweep_id} has fewer than {confirmations} views"
            )
        windows = [
            min(scores[index] for index in ordered[start : start + confirmations])
            for start in range(len(ordered) - confirmations + 1)
        ]
        first = rows[ordered[0]]
        events.append(
            {
                "sweep_id": sweep_id,
                "scene_id": first["scene_id"],
                "target": int(first["target"]),
                "first_window_score": float(windows[0]),
                "any_window_score": float(max(windows)),
            }
        )
    targets = [int(event["target"]) for event in events]
    first_scores = [float(event["first_window_score"]) for event in events]
    any_scores = [float(event["any_window_score"]) for event in events]
    first_threshold = max(
        score for target, score in zip(targets, first_scores) if target == 0
    )
    robust_threshold = max(
        score for target, score in zip(targets, any_scores) if target == 0
    )
    return {
        "confirmations": confirmations,
        "events": len(events),
        "first_window_auc": _auc(targets, first_scores),
        "first_window_zero_false_positive_threshold": first_threshold,
        "first_window_zero_false_positive_goal_catches": sum(
            score > first_threshold
            for target, score in zip(targets, first_scores)
            if target == 1
        ),
        "robust_any_boundary_window_threshold": robust_threshold,
        "first_window_goal_catches_at_robust_threshold": sum(
            score > robust_threshold
            for target, score in zip(targets, first_scores)
            if target == 1
        ),
        "any_window_goal_catches_at_robust_threshold": sum(
            score > robust_threshold
            for target, score in zip(targets, any_scores)
            if target == 1
        ),
        "goal_events": sum(targets),
    }


def _read_probe_rows(labels_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with labels_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            boundary = bool(raw.get("boundary_probe_sweep"))
            goal = bool(raw.get("goal_probe_sweep"))
            if not boundary and not goal:
                continue
            if boundary == goal:
                raise RuntimeError(f"Ambiguous probe kind at {labels_path}:{line_number}")
            target = int(raw.get("stop_target", -1))
            expected_target = int(goal)
            if target != expected_target or raw.get("dataset_split") != "train":
                raise RuntimeError(f"Invalid trajectory probe label at {labels_path}:{line_number}")
            if bool(raw.get("original_terminal", False)):
                raise RuntimeError("Trajectory STOP-add probes must be original non-STOP calls")
            sweep_id = raw.get("goal_probe_sweep_id") if goal else raw.get("boundary_probe_sweep_id")
            probe_index = raw.get("goal_probe_index") if goal else raw.get("boundary_probe_index")
            rows.append(
                {
                    **raw,
                    "target": target,
                    "kind": "goal" if goal else "boundary",
                    "sweep_id": str(sweep_id),
                    "probe_index": int(probe_index),
                }
            )
    if not rows:
        raise RuntimeError("No fixed-position trajectory probe rows were found")
    rows.sort(key=lambda row: (row["scene_id"], row["target"], row["probe_index"]))
    keys = [str(row["key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Trajectory probe labels contain duplicate feature keys")
    return rows


def _load_feature_blocks(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    blocks: dict[str, list[torch.Tensor]] = defaultdict(list)
    for row in rows:
        path = Path(str(row.get("path", ""))).expanduser()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, dict)
            or payload.get("key") != row["key"]
            or payload.get("trajectory_feature_schema") != TRAJECTORY_FEATURE_SCHEMA
        ):
            raise RuntimeError(f"Invalid trajectory feature payload: {path}")
        raw = payload.get("raw_traj_latent")
        adapted = payload.get("adapted_traj_latent")
        projected = payload.get("projected_traj_condition")
        trajectory = payload.get("trajectory")
        decision = payload.get("feature")
        if (
            not all(torch.is_tensor(value) for value in (raw, adapted, projected, trajectory, decision))
            or raw.shape != adapted.shape
            or raw.ndim != 2
            or projected.ndim != 2
            or projected.shape[0] != raw.shape[0]
            or trajectory.ndim != 3
            or decision.ndim != 1
        ):
            raise RuntimeError(f"Trajectory feature tensor shape mismatch: {path}")
        metrics = payload.get("trajectory_metrics")
        if not isinstance(metrics, dict):
            raise RuntimeError(f"Missing trajectory metrics: {path}")
        metric = torch.tensor(
            [float(metrics[name]) for name in ("goal_x_m", "goal_y_m", "direct_m", "path_len_m")],
            dtype=torch.float32,
        )
        trajectory_distribution = torch.cat(
            (trajectory.float().mean(dim=0).flatten(), trajectory.float().std(dim=0, unbiased=False).flatten(), metric)
        )
        blocks["decision_hidden"].append(decision.float().flatten())
        blocks["raw_latent"].append(raw.float().flatten())
        blocks["adapted_latent"].append(adapted.float().flatten())
        blocks["adapter_delta"].append((adapted.float() - raw.float()).flatten())
        blocks["projected_condition"].append(projected.float().flatten())
        blocks["trajectory_distribution"].append(trajectory_distribution)
    result = {name: torch.stack(values) for name, values in blocks.items()}
    expected_rows = len(rows)
    if any(value.shape[0] != expected_rows for value in result.values()):
        raise RuntimeError("Loaded trajectory feature blocks do not align")
    return result


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ridge", type=float, action="append", default=None)
    parser.add_argument("--confirmation", type=int, action="append", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.labels_jsonl.is_file() or args.labels_jsonl.stat().st_size == 0:
        raise FileNotFoundError(f"Missing labels JSONL: {args.labels_jsonl}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {args.output_dir}")
    ridge_values = args.ridge or [0.01, 0.1, 1.0, 10.0]
    confirmation_counts = sorted(set(args.confirmation or [2, 3, 4]))
    if any(count < 2 for count in confirmation_counts):
        raise ValueError("All confirmation counts must be >= 2")
    rows = _read_probe_rows(args.labels_jsonl)
    scenes = sorted({str(row["scene_id"]) for row in rows})
    folds = _stable_scene_folds(scenes, args.folds, args.seed)
    blocks = _load_feature_blocks(rows)
    variants = {
        "decision_hidden": ["decision_hidden"],
        "raw_latent": ["raw_latent"],
        "adapted_latent": ["adapted_latent"],
        "adapter_delta": ["adapter_delta"],
        "projected_condition": ["projected_condition"],
        "trajectory_distribution": ["trajectory_distribution"],
        "projected_plus_trajectory": ["projected_condition", "trajectory_distribution"],
        "adapted_projected_trajectory": [
            "adapted_latent",
            "projected_condition",
            "trajectory_distribution",
        ],
    }
    targets = [int(row["target"]) for row in rows]
    scene_ids = [str(row["scene_id"]) for row in rows]
    results: list[dict[str, Any]] = []
    best_scores: list[float] | None = None
    best_rank: tuple[float, ...] | None = None
    best_spec: dict[str, Any] | None = None
    for variant, names in variants.items():
        kernel = _kernel_from_blocks([blocks[name] for name in names])
        for ridge in ridge_values:
            scores = _ridge_oof_scores(kernel, targets, scene_ids, folds, ridge)
            aggregations = {
                name: _group_metrics(rows, scores, name)
                for name in ("mean", "median", "minimum")
            }
            selected = max(
                aggregations.values(),
                key=lambda metrics: (
                    metrics["auc"],
                    metrics["paired_wins"],
                    metrics["zero_false_positive_goal_catches"],
                ),
            )
            confirmation_metrics = {
                str(count): _confirmation_metrics(rows, scores, count)
                for count in confirmation_counts
            }
            selected_confirmation = max(
                confirmation_metrics.values(),
                key=lambda metrics: (
                    metrics["first_window_zero_false_positive_goal_catches"],
                    metrics["first_window_goal_catches_at_robust_threshold"],
                    metrics["first_window_auc"],
                ),
            )
            result = {
                "variant": variant,
                "blocks": names,
                "ridge": float(ridge),
                "view_metrics": _view_metrics(rows, scores),
                "confirmation_metrics": confirmation_metrics,
                "selected_confirmation": selected_confirmation["confirmations"],
                "group_aggregations": aggregations,
                "selected_aggregation": selected["aggregation"],
            }
            results.append(result)
            view_metrics = result["view_metrics"]
            rank = (
                float(
                    selected_confirmation[
                        "first_window_zero_false_positive_goal_catches"
                    ]
                ),
                float(
                    selected_confirmation[
                        "first_window_goal_catches_at_robust_threshold"
                    ]
                ),
                float(view_metrics["goal_groups_first_two_views_hit"]),
                float(view_metrics["goal_groups_first_view_hit"]),
                float(view_metrics["zero_false_positive_goal_views"]),
                int(selected["zero_false_positive_goal_catches"]),
                float(view_metrics["auc"]),
                float(selected["auc"]),
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_scores = scores
                best_spec = result
            print(
                f"variant={variant:30s} ridge={ridge:g} "
                f"view_auc={view_metrics['auc']:.4f} "
                f"view_zero_fp={view_metrics['zero_false_positive_goal_views']}/"
                f"{view_metrics['positive_views']} "
                f"first2={view_metrics['goal_groups_first_two_views_hit']}/"
                f"{view_metrics['goal_groups']} "
                + " ".join(
                    f"confirm{count}="
                    f"{confirmation_metrics[str(count)]['first_window_zero_false_positive_goal_catches']}/"
                    f"{confirmation_metrics[str(count)]['goal_events']}"
                    f"(robust={confirmation_metrics[str(count)]['first_window_goal_catches_at_robust_threshold']})"
                    for count in confirmation_counts
                )
                + " "
                f"group={selected['aggregation']} auc={selected['auc']:.4f} "
                f"paired={selected['paired_wins']}/{selected['paired_total']} "
                f"zero_fp_catch={selected['zero_false_positive_goal_catches']}/{selected['paired_total']}",
                flush=True,
            )

    if best_spec is None or best_scores is None:
        raise RuntimeError("Trajectory diagnostic did not evaluate any variant")
    summary = {
        "stage_name": "system2_stop_trajectory_latent_scene_oof_diagnostic",
        "deployable": False,
        "labels_jsonl": str(args.labels_jsonl.resolve()),
        "rows": len(rows),
        "scenes": len(scenes),
        "folds": folds,
        "feature_schema": TRAJECTORY_FEATURE_SCHEMA,
        "normalization": "per-sample-block-rms_then_train-fold-kernel-centering",
        "exploratory_hyperparameter_search": True,
        "results": results,
        "best": best_spec,
    }
    args.output_dir.mkdir(parents=True)
    _atomic_json(summary, args.output_dir / "summary.json")
    with (args.output_dir / "best_oof_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row, score in zip(rows, best_scores):
            handle.write(
                json.dumps(
                    {
                        "key": row["key"],
                        "scene_id": row["scene_id"],
                        "episode_id": row["episode_id"],
                        "sweep_id": row["sweep_id"],
                        "kind": row["kind"],
                        "probe_index": row["probe_index"],
                        "target": row["target"],
                        "score": float(score),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    print(f"Trajectory latent diagnostic complete: {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

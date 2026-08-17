#!/usr/bin/env python3
"""Cross-fit and train a conservative static System2 STOP-add head.

The OOF predictions are scene-disjoint and are calibrated against the same
consecutive-confirmation policy used by closed-loop evaluation.  The final
head is trained on all train-split rollout rows only after OOF calibration.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.train_stop_head_from_rollout_cache import (
    FEATURE_SCHEMA,
    _atomic_torch_save,
    _build_head,
    _build_sampling_weights,
    _load_checkpoint,
    _load_features,
    _metrics,
    _predict,
    _read_rows,
)
from scripts.training.train_temporal_stop_add_seed_ensemble_diagnostic import (
    _candidate_groups,
    _event_metrics,
    _zero_false_event_threshold,
)
from scripts.training.train_temporal_stop_ensemble_from_rollout_cache import (
    _scene_folds,
)
from scripts.training.train_temporal_stop_verifier_from_rollout_cache import _auc
from src.models.action.stop_head import StopPredictionHead


_SEED_SUFFIX = re.compile(r"_seed(?P<seed>-?\d+)$")


def _annotate_sequence_identity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach stable rollout-source and seed fields used for event grouping."""
    source_roots = sorted(
        {
            str(Path(str(row["path"])).expanduser().resolve().parent.parent)
            for row in rows
        }
    )
    source_indices = {root: index for index, root in enumerate(source_roots)}
    annotated: list[dict[str, Any]] = []
    seen_calls: set[tuple[int, str, int, int, int]] = set()
    for row in rows:
        item = dict(row)
        source_root = str(
            Path(str(item["path"])).expanduser().resolve().parent.parent
        )
        raw_seed = item.get("protocol_seed", item.get("rpc_protocol_seed"))
        if raw_seed is None:
            match = _SEED_SUFFIX.search(str(item.get("key", "")))
            if match is None:
                raise RuntimeError(
                    f"Cannot recover protocol seed from STOP feature key {item.get('key')!r}"
                )
            raw_seed = match.group("seed")
        item["source_index"] = source_indices[source_root]
        item["source_root"] = source_root
        item["protocol_seed"] = int(raw_seed)
        identity = (
            int(item["source_index"]),
            str(item["scene_id"]),
            int(item["episode_id"]),
            int(item["protocol_seed"]),
            int(item["system2_call_index"]),
        )
        if identity in seen_calls:
            raise RuntimeError(f"Duplicate STOP rollout call identity: {identity}")
        seen_calls.add(identity)
        annotated.append(item)
    return annotated


def _candidate_subset(
    rows: list[dict[str, Any]],
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    indices = [
        index
        for index, row in enumerate(rows)
        if not bool(row.get("original_terminal", False))
    ]
    if not indices:
        raise RuntimeError("No original-nonterminal STOP-add candidates were found")
    tensor = torch.tensor(indices, dtype=torch.long)
    candidate_targets = targets[tensor]
    if set(candidate_targets.int().tolist()) != {0, 1}:
        raise RuntimeError("STOP-add candidates must contain both classes")
    return probabilities[tensor], candidate_targets, [rows[index] for index in indices]


def _terminal_confirmation_scores(
    probabilities: torch.Tensor,
    rows: list[dict[str, Any]],
) -> torch.Tensor:
    """Mask scores so only an original Qwen STOP can cast a confirmation vote."""
    if probabilities.shape != (len(rows),):
        raise ValueError("STOP probabilities must align with rollout rows")
    original_terminal = torch.tensor(
        [bool(row.get("original_terminal", False)) for row in rows],
        dtype=torch.bool,
        device=probabilities.device,
    )
    return torch.where(
        original_terminal,
        probabilities,
        torch.full_like(probabilities, float("-inf")),
    )


def _event_calibration(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    confirmations: int,
    minimum_threshold: float,
) -> tuple[float, dict[str, float | int]]:
    raw_threshold = _zero_false_event_threshold(
        probabilities,
        targets,
        rows,
        confirmations=confirmations,
    )
    threshold = minimum_threshold if not math.isfinite(raw_threshold) else max(
        minimum_threshold,
        raw_threshold,
    )
    threshold = min(float(threshold), 1.0)
    metrics = _event_metrics(
        probabilities,
        targets,
        rows,
        threshold=threshold,
        confirmations=confirmations,
    )
    return threshold, metrics


def _build_optimization_targets(
    rows: list[dict[str, Any]],
    evaluation_targets: torch.Tensor,
    *,
    positive_radius_m: float | None,
    negative_radius_m: float | None,
) -> torch.Tensor:
    """Optionally train with a metric margin while evaluating at the 3 m boundary."""
    if evaluation_targets.shape != (len(rows),):
        raise ValueError("Evaluation STOP targets must align with rollout rows")
    if positive_radius_m is None and negative_radius_m is None:
        return evaluation_targets.clone()
    if positive_radius_m is None or negative_radius_m is None:
        raise ValueError("Optimization positive/negative radii must be set together")
    if (
        not math.isfinite(positive_radius_m)
        or not math.isfinite(negative_radius_m)
        or positive_radius_m <= 0.0
        or negative_radius_m <= positive_radius_m
    ):
        raise ValueError("Optimization STOP radii must satisfy 0 < positive < negative")

    targets = torch.full_like(evaluation_targets, -1.0)
    for index, row in enumerate(rows):
        try:
            distance_m = float(row.get("distance_to_goal_m", float("nan")))
        except (TypeError, ValueError):
            distance_m = float("nan")
        if not math.isfinite(distance_m):
            continue
        if distance_m <= positive_radius_m:
            targets[index] = 1.0
        elif distance_m >= negative_radius_m:
            targets[index] = 0.0
    return targets


def _probe_sweep_diagnostics(
    rows: list[dict[str, Any]],
    probabilities: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize fixed-position boundary/goal sweeps from scene-OOF scores."""
    if probabilities.shape != (len(rows),):
        raise ValueError("Probe probabilities must align with probe rows")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("Probe probabilities must be finite")

    grouped: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for row, probability in zip(rows, probabilities.tolist(), strict=True):
        boundary = bool(row.get("boundary_probe_sweep", False))
        goal = bool(row.get("goal_probe_sweep", False))
        if boundary == goal:
            raise ValueError(
                "Each probe row must belong to exactly one boundary or goal sweep"
            )
        kind = "boundary" if boundary else "goal"
        sweep_id = str(row.get(f"{kind}_probe_sweep_id") or "")
        if not sweep_id:
            raise ValueError(f"Probe row is missing {kind}_probe_sweep_id")
        grouped.setdefault(sweep_id, []).append((row, float(probability)))

    group_records: list[dict[str, Any]] = []
    for sweep_id, items in grouped.items():
        first = items[0][0]
        kind = "boundary" if bool(first.get("boundary_probe_sweep")) else "goal"
        index_key = f"{kind}_probe_index"
        items.sort(key=lambda item: int(item[0][index_key]))
        expected_views = int(first.get("boundary_probe_views", len(items)))
        indices = [int(row[index_key]) for row, _ in items]
        if len(items) != expected_views or indices != list(range(expected_views)):
            raise ValueError(
                f"Incomplete {kind} probe sweep {sweep_id}: "
                f"views={len(items)}/{expected_views} indices={indices}"
            )
        target = 0 if kind == "boundary" else 1
        targets = {int(row["stop_target"]) for row, _ in items}
        if targets != {target}:
            raise ValueError(
                f"Invalid targets for {kind} probe sweep {sweep_id}: {targets}"
            )
        distances = [float(row["distance_to_goal_m"]) for row, _ in items]
        if max(distances) - min(distances) > 1e-6:
            raise ValueError(f"Probe sweep moved during collection: {sweep_id}")
        scores = [probability for _, probability in items]
        group_records.append(
            {
                "sweep_id": sweep_id,
                "scene_id": str(first["scene_id"]),
                "episode_id": int(first["episode_id"]),
                "kind": kind,
                "target": target,
                "views": expected_views,
                "distance_to_goal_m": distances[0],
                "mean_probability": sum(scores) / len(scores),
                "min_probability": min(scores),
                "max_probability": max(scores),
                "probabilities": scores,
            }
        )
    group_records.sort(
        key=lambda item: (item["scene_id"], item["episode_id"], item["kind"])
    )
    group_probabilities = torch.tensor(
        [record["mean_probability"] for record in group_records],
        dtype=torch.float32,
    )
    group_targets = torch.tensor(
        [record["target"] for record in group_records],
        dtype=torch.float32,
    )
    if set(group_targets.int().tolist()) != {0, 1}:
        raise ValueError("Probe sweeps must contain boundary and goal groups")
    boundary_means = [
        record["mean_probability"]
        for record in group_records
        if record["kind"] == "boundary"
    ]
    goal_means = [
        record["mean_probability"]
        for record in group_records
        if record["kind"] == "goal"
    ]
    zero_false_threshold = math.nextafter(max(boundary_means), math.inf)
    paired = {
        (record["scene_id"], record["episode_id"]): {}
        for record in group_records
    }
    for record in group_records:
        paired[(record["scene_id"], record["episode_id"])][record["kind"]] = record[
            "mean_probability"
        ]
    complete_pairs = [
        item for item in paired.values() if set(item) == {"boundary", "goal"}
    ]
    summary = {
        "schema": "heatmapvln-system2-stop-probe-scene-oof-v1",
        "rows": len(rows),
        "boundary_groups": len(boundary_means),
        "goal_groups": len(goal_means),
        "group_auc": _auc(group_probabilities, group_targets),
        "max_boundary_mean_probability": max(boundary_means),
        "min_goal_mean_probability": min(goal_means),
        "zero_false_boundary_threshold": zero_false_threshold,
        "zero_false_boundary_goal_groups": sum(
            probability >= zero_false_threshold for probability in goal_means
        ),
        "paired_groups": len(complete_pairs),
        "paired_goal_mean_wins": sum(
            item["goal"] > item["boundary"] for item in complete_pairs
        ),
    }
    return summary, group_records


def _select_probe_subset(
    rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
) -> tuple[list[int], list[dict[str, Any]]]:
    """Select an exact probe subset from the deduplicated training rows."""
    probe_by_key: dict[str, dict[str, Any]] = {}
    for row in probe_rows:
        if not (
            bool(row.get("boundary_probe_sweep", False))
            or bool(row.get("goal_probe_sweep", False))
        ):
            continue
        key = str(row.get("key", ""))
        if not key:
            raise RuntimeError("Cross-fit probe row is missing its feature key")
        if key in probe_by_key:
            raise RuntimeError(f"Duplicate cross-fit probe feature key: {key}")
        probe_by_key[key] = row
    if not probe_by_key:
        raise RuntimeError("Cross-fit probe label files contain no boundary/goal sweeps")

    row_index_by_key = {str(row["key"]): index for index, row in enumerate(rows)}
    missing = sorted(set(probe_by_key) - set(row_index_by_key))
    if missing:
        raise RuntimeError(
            "Cross-fit probe rows were not retained in the training set: "
            f"{missing[:5]}"
        )
    indices = [row_index_by_key[key] for key in probe_by_key]
    selected_rows = [rows[index] for index in indices]
    return indices, selected_rows


def _merge_crossfit_probe_rows(
    base_rows: list[dict[str, Any]],
    candidate_probe_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Append only fixed-position probes and reject overlap with base training data."""
    probe_rows = [
        row
        for row in candidate_probe_rows
        if bool(row.get("boundary_probe_sweep", False))
        or bool(row.get("goal_probe_sweep", False))
    ]
    if not probe_rows:
        raise RuntimeError("Cross-fit probe label files contain no boundary/goal sweeps")
    base_keys = {str(row["key"]) for row in base_rows}
    probe_keys = [str(row["key"]) for row in probe_rows]
    if len(set(probe_keys)) != len(probe_keys):
        raise RuntimeError("Cross-fit probe label files contain duplicate feature keys")
    overlap = sorted(base_keys.intersection(probe_keys))
    if overlap:
        raise RuntimeError(
            "Cross-fit probe rows overlap base training rows: " f"{overlap[:5]}"
        )
    return [*base_rows, *probe_rows], probe_rows


def _train_fixed_epochs(
    *,
    config: dict[str, Any],
    initial_state: dict[str, torch.Tensor],
    features: torch.Tensor,
    targets: torch.Tensor,
    optimization_targets: torch.Tensor,
    rows: list[dict[str, Any]],
    train_indices: list[int],
    initial_probabilities: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    l2_sp_weight: float,
    terminal_negative_weight: float,
    hard_negative_threshold: float,
    hard_negative_weight: float,
    oracle_recovery_positive_weight: float,
    boundary_negative_min_distance_m: float | None,
    boundary_negative_max_distance_m: float | None,
    boundary_negative_weight: float,
    seed: int,
    device: torch.device,
) -> tuple[StopPredictionHead, dict[str, Any]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    head = _build_head(config, initial_state).to(device=device, dtype=torch.float32)
    reference = {
        name: parameter.detach().clone()
        for name, parameter in head.named_parameters()
    }
    if optimization_targets.shape != targets.shape:
        raise ValueError("Optimization and evaluation STOP targets must align")
    optimization_indices = [
        index
        for index in train_indices
        if int(optimization_targets[index].item()) in (0, 1)
    ]
    if not optimization_indices:
        raise RuntimeError("Metric-margin STOP training retained no records")
    weights, sampling_stats = _build_sampling_weights(
        rows,
        optimization_targets,
        optimization_indices,
        initial_probabilities,
        terminal_negative_weight=terminal_negative_weight,
        hard_negative_threshold=hard_negative_threshold,
        hard_negative_weight=hard_negative_weight,
        oracle_recovery_positive_weight=oracle_recovery_positive_weight,
        boundary_negative_min_distance_m=boundary_negative_min_distance_m,
        boundary_negative_max_distance_m=boundary_negative_max_distance_m,
        boundary_negative_weight=boundary_negative_weight,
    )
    train_tensor = torch.tensor(optimization_indices, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(features[train_tensor], optimization_targets[train_tensor]),
        batch_size=batch_size,
        sampler=WeightedRandomSampler(
            weights,
            num_samples=max(len(optimization_indices), batch_size),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        ),
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    epoch_losses: list[float] = []
    for _epoch in range(epochs):
        head.train()
        total_loss = 0.0
        batch_count = 0
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            logits = head.classifier(batch_features).squeeze(-1)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                batch_targets,
            )
            l2_sp = sum(
                (parameter - reference[name]).float().square().mean()
                for name, parameter in head.named_parameters()
            )
            loss = bce + l2_sp_weight * l2_sp
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().item())
            batch_count += 1
        epoch_losses.append(total_loss / max(batch_count, 1))
    return head, {
        "sampling": sampling_stats,
        "optimization_records": len(optimization_indices),
        "epoch_losses": epoch_losses,
        "final_train_loss": epoch_losses[-1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-jsonl", type=Path, action="append", required=True)
    parser.add_argument(
        "--probe-labels-jsonl",
        type=Path,
        action="append",
        help=(
            "Optional fixed-position boundary/goal sweep labels. They are scored only "
            "by the fold that excludes their scene and are never used for training."
        ),
    )
    parser.add_argument(
        "--crossfit-probe-labels-jsonl",
        type=Path,
        action="append",
        help=(
            "Optional fixed-position boundary/goal sweep labels that are added to "
            "training. Their diagnostics remain scene-disjoint because every row is "
            "scored only by the fold that excludes its entire scene."
        ),
    )
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--l2-sp-weight", type=float, default=1e-4)
    parser.add_argument("--terminal-negative-weight", type=float, default=4.0)
    parser.add_argument("--hard-negative-threshold", type=float, default=0.8)
    parser.add_argument("--hard-negative-weight", type=float, default=8.0)
    parser.add_argument("--oracle-recovery-positive-weight", type=float, default=4.0)
    parser.add_argument("--boundary-negative-min-distance-m", type=float)
    parser.add_argument("--boundary-negative-max-distance-m", type=float)
    parser.add_argument("--boundary-negative-weight", type=float, default=1.0)
    parser.add_argument("--optimization-positive-radius-m", type=float)
    parser.add_argument("--optimization-negative-radius-m", type=float)
    parser.add_argument("--relabel-ambiguous-negative-radius-m", type=float, default=3.01)
    parser.add_argument("--confirmations", type=int, default=2)
    parser.add_argument("--minimum-add-threshold", type=float, default=0.9)
    parser.add_argument("--terminal-confirmations", type=int, default=4)
    parser.add_argument("--minimum-terminal-confirm-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--feature-load-workers", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite OOF STOP-add output: {args.output_dir}")
    if args.folds < 2 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("folds must be >= 2; epochs and batch-size must be >= 1")
    if (
        args.confirmations < 1
        or args.terminal_confirmations < 1
        or args.feature_load_workers < 1
    ):
        raise ValueError(
            "confirmations, terminal-confirmations, and feature-load-workers must be >= 1"
        )
    if not 0.0 <= args.minimum_add_threshold <= 1.0:
        raise ValueError("minimum-add-threshold must be in [0, 1]")
    if not 0.0 <= args.minimum_terminal_confirm_threshold <= 1.0:
        raise ValueError("minimum-terminal-confirm-threshold must be in [0, 1]")
    if args.lr <= 0.0 or args.l2_sp_weight < 0.0:
        raise ValueError("lr must be positive and l2-sp-weight must be non-negative")
    boundary_radii = (
        args.boundary_negative_min_distance_m,
        args.boundary_negative_max_distance_m,
    )
    if (boundary_radii[0] is None) != (boundary_radii[1] is None):
        raise ValueError("boundary-negative min/max distances must be set together")
    if boundary_radii[0] is not None and (
        not math.isfinite(boundary_radii[0])
        or not math.isfinite(boundary_radii[1])
        or boundary_radii[0] < 0.0
        or boundary_radii[1] <= boundary_radii[0]
    ):
        raise ValueError("boundary-negative distance range is invalid")
    if (
        not math.isfinite(args.boundary_negative_weight)
        or args.boundary_negative_weight < 1.0
    ):
        raise ValueError("boundary-negative weight must be finite and >= 1")
    optimization_radii = (
        args.optimization_positive_radius_m,
        args.optimization_negative_radius_m,
    )
    if (optimization_radii[0] is None) != (optimization_radii[1] is None):
        raise ValueError("optimization positive/negative radii must be set together")
    if optimization_radii[0] is not None and (
        not math.isfinite(optimization_radii[0])
        or not math.isfinite(optimization_radii[1])
        or optimization_radii[0] <= 0.0
        or optimization_radii[1] <= optimization_radii[0]
    ):
        raise ValueError("optimization radii must satisfy 0 < positive < negative")
    input_paths = [
        *args.labels_jsonl,
        *(args.probe_labels_jsonl or []),
        *(args.crossfit_probe_labels_jsonl or []),
        args.init_checkpoint,
    ]
    for path in input_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing non-empty STOP-add input: {path}")

    checkpoint, initial_state = _load_checkpoint(args.init_checkpoint)
    config = copy.deepcopy(checkpoint["config"])
    training_label_paths = [*args.labels_jsonl]
    base_rows = _read_rows(
        args.labels_jsonl,
        allow_nontrain=False,
        relabel_ambiguous_negative_radius_m=(
            args.relabel_ambiguous_negative_radius_m
        ),
    )
    crossfit_probe_indices: list[int] = []
    crossfit_probe_rows: list[dict[str, Any]] = []
    if args.crossfit_probe_labels_jsonl:
        raw_crossfit_probe_rows = _read_rows(
            args.crossfit_probe_labels_jsonl,
            allow_nontrain=False,
            relabel_ambiguous_negative_radius_m=(
                args.relabel_ambiguous_negative_radius_m
            ),
        )
        base_rows, crossfit_probe_rows = _merge_crossfit_probe_rows(
            base_rows,
            raw_crossfit_probe_rows,
        )
        training_label_paths.extend(args.crossfit_probe_labels_jsonl)
    rows = _annotate_sequence_identity(base_rows)
    if crossfit_probe_rows:
        crossfit_probe_indices, crossfit_probe_rows = _select_probe_subset(
            rows,
            crossfit_probe_rows,
        )
    print(
        f"loading_features=records={len(rows)} workers={args.feature_load_workers}",
        flush=True,
    )
    features, targets = _load_features(rows, workers=args.feature_load_workers)
    probe_rows: list[dict[str, Any]] = []
    probe_features = torch.empty((0, features.shape[1]), dtype=features.dtype)
    probe_oof_probabilities = torch.empty(0, dtype=torch.float32)
    probe_fold_assignments = torch.empty(0, dtype=torch.int64)
    if args.probe_labels_jsonl:
        probe_rows = [
            row
            for row in _read_rows(args.probe_labels_jsonl, allow_nontrain=False)
            if bool(row.get("boundary_probe_sweep", False))
            or bool(row.get("goal_probe_sweep", False))
        ]
        if not probe_rows:
            raise RuntimeError("Probe label files contain no boundary/goal sweep rows")
        probe_features, _probe_targets = _load_features(
            probe_rows,
            workers=args.feature_load_workers,
        )
        probe_oof_probabilities = torch.full(
            (len(probe_rows),),
            float("nan"),
            dtype=torch.float32,
        )
        probe_fold_assignments = torch.zeros(len(probe_rows), dtype=torch.int64)
    optimization_targets = _build_optimization_targets(
        rows,
        targets,
        positive_radius_m=args.optimization_positive_radius_m,
        negative_radius_m=args.optimization_negative_radius_m,
    )
    device = torch.device(args.device)
    initial_head = _build_head(config, initial_state).to(device=device, dtype=torch.float32)
    initial_probabilities = _predict(initial_head, features, device)
    del initial_head

    folds, fold_seed = _scene_folds(
        rows,
        targets,
        fold_count=args.folds,
        seed=args.seed,
    )
    all_indices = set(range(len(rows)))
    oof_probabilities = torch.empty_like(targets)
    fold_summaries: list[dict[str, Any]] = []
    diagnostic_fold_heads: list[dict[str, Any]] = []
    for fold_index, val_indices in enumerate(folds):
        train_indices = sorted(all_indices - set(val_indices))
        head, training = _train_fixed_epochs(
            config=config,
            initial_state=initial_state,
            features=features,
            targets=targets,
            optimization_targets=optimization_targets,
            rows=rows,
            train_indices=train_indices,
            initial_probabilities=initial_probabilities,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            l2_sp_weight=args.l2_sp_weight,
            terminal_negative_weight=args.terminal_negative_weight,
            hard_negative_threshold=args.hard_negative_threshold,
            hard_negative_weight=args.hard_negative_weight,
            oracle_recovery_positive_weight=args.oracle_recovery_positive_weight,
            boundary_negative_min_distance_m=(
                args.boundary_negative_min_distance_m
            ),
            boundary_negative_max_distance_m=(
                args.boundary_negative_max_distance_m
            ),
            boundary_negative_weight=args.boundary_negative_weight,
            seed=args.seed + 1009 * fold_index,
            device=device,
        )
        val_tensor = torch.tensor(val_indices, dtype=torch.long)
        val_probabilities = _predict(head, features[val_tensor], device)
        oof_probabilities[val_tensor] = val_probabilities
        val_targets = targets[val_tensor]
        val_bce = float(
            torch.nn.functional.binary_cross_entropy(
                val_probabilities.clamp(1e-6, 1.0 - 1e-6),
                val_targets,
            ).item()
        )
        fold_summary = {
            "fold": fold_index,
            "train_records": len(train_indices),
            "val_records": len(val_indices),
            "val_scenes": sorted({str(rows[index]["scene_id"]) for index in val_indices}),
            "val_auc": _auc(val_probabilities, val_targets),
            "val_bce": val_bce,
            **training,
        }
        fold_summaries.append(fold_summary)
        if probe_rows:
            val_scenes = set(fold_summary["val_scenes"])
            probe_indices = [
                index
                for index, row in enumerate(probe_rows)
                if str(row["scene_id"]) in val_scenes
            ]
            if probe_indices:
                probe_tensor = torch.tensor(probe_indices, dtype=torch.long)
                probe_fold_assignments[probe_tensor] += 1
                probe_oof_probabilities[probe_tensor] = _predict(
                    head,
                    probe_features[probe_tensor],
                    device,
                )
            diagnostic_fold_heads.append(
                {
                    "fold": fold_index,
                    "val_scenes": sorted(val_scenes),
                    "trainable_state_dict": {
                        f"stop_head.{name}": value.detach().cpu().clone()
                        for name, value in head.state_dict().items()
                    },
                }
            )
        print(
            f"fold={fold_index + 1}/{args.folds} train={len(train_indices)} "
            f"val={len(val_indices)} auc={fold_summary['val_auc']:.4f} "
            f"bce={val_bce:.6f} loss={training['final_train_loss']:.6f}",
            flush=True,
        )
        del head

    crossfit_probe_summary = None
    crossfit_probe_group_records: list[dict[str, Any]] = []
    crossfit_probe_probabilities = torch.empty(0, dtype=torch.float32)
    if crossfit_probe_rows:
        crossfit_probe_tensor = torch.tensor(crossfit_probe_indices, dtype=torch.long)
        crossfit_probe_probabilities = oof_probabilities[crossfit_probe_tensor]
        crossfit_probe_summary, crossfit_probe_group_records = (
            _probe_sweep_diagnostics(
                crossfit_probe_rows,
                crossfit_probe_probabilities,
            )
        )
        print(
            "crossfit_probe_scene_oof="
            + json.dumps(crossfit_probe_summary, sort_keys=True),
            flush=True,
        )

    probe_summary = None
    probe_group_records: list[dict[str, Any]] = []
    if probe_rows:
        invalid_probe_scenes = sorted(
            {
                str(probe_rows[index]["scene_id"])
                for index in (probe_fold_assignments != 1)
                .nonzero()
                .flatten()
                .tolist()
            }
        )
        if invalid_probe_scenes:
            raise RuntimeError(
                "Probe scenes must be assigned to exactly one scene-OOF fold: "
                f"{invalid_probe_scenes}"
            )
        if bool(torch.isnan(probe_oof_probabilities).any()):
            raise RuntimeError("Scene-OOF probe scoring left uninitialized probabilities")
        probe_summary, probe_group_records = _probe_sweep_diagnostics(
            probe_rows,
            probe_oof_probabilities,
        )
        print("probe_scene_oof=" + json.dumps(probe_summary, sort_keys=True), flush=True)

    calibration_indices = sorted(
        set(range(len(rows))) - set(crossfit_probe_indices)
    )
    if not calibration_indices:
        raise RuntimeError("No natural rollout rows remain for OOF calibration")
    calibration_tensor = torch.tensor(calibration_indices, dtype=torch.long)
    calibration_rows = [rows[index] for index in calibration_indices]
    calibration_oof = oof_probabilities[calibration_tensor]
    calibration_initial = initial_probabilities[calibration_tensor]
    calibration_targets = targets[calibration_tensor]
    candidate_oof, candidate_targets, candidate_rows = _candidate_subset(
        calibration_rows,
        calibration_oof,
        calibration_targets,
    )
    candidate_initial, _, _ = _candidate_subset(
        calibration_rows,
        calibration_initial,
        calibration_targets,
    )
    add_threshold, event_metrics = _event_calibration(
        candidate_oof,
        candidate_targets,
        candidate_rows,
        confirmations=args.confirmations,
        minimum_threshold=args.minimum_add_threshold,
    )
    initial_threshold, initial_event_metrics = _event_calibration(
        candidate_initial,
        candidate_targets,
        candidate_rows,
        confirmations=args.confirmations,
        minimum_threshold=args.minimum_add_threshold,
    )
    call_metrics = _metrics(candidate_oof, candidate_targets, add_threshold)
    initial_call_metrics = _metrics(
        candidate_initial,
        candidate_targets,
        initial_threshold,
    )
    terminal_oof_scores = _terminal_confirmation_scores(
        calibration_oof,
        calibration_rows,
    )
    terminal_initial_scores = _terminal_confirmation_scores(
        calibration_initial,
        calibration_rows,
    )
    terminal_confirm_threshold, terminal_event_metrics = _event_calibration(
        terminal_oof_scores,
        calibration_targets,
        calibration_rows,
        confirmations=args.terminal_confirmations,
        minimum_threshold=args.minimum_terminal_confirm_threshold,
    )
    (
        initial_terminal_confirm_threshold,
        initial_terminal_event_metrics,
    ) = _event_calibration(
        terminal_initial_scores,
        calibration_targets,
        calibration_rows,
        confirmations=args.terminal_confirmations,
        minimum_threshold=args.minimum_terminal_confirm_threshold,
    )
    original_terminal_mask = torch.tensor(
        [bool(row.get("original_terminal", False)) for row in calibration_rows],
        dtype=torch.bool,
        device=oof_probabilities.device,
    )
    terminal_call_metrics = _metrics(
        calibration_oof[original_terminal_mask],
        calibration_targets[original_terminal_mask],
        terminal_confirm_threshold,
    )
    terminal_quality_gate = {
        "zero_false_stop_episodes": int(
            terminal_event_metrics["false_stop_episodes"]
        )
        == 0,
        "nonzero_true_stop_episodes": int(
            terminal_event_metrics["true_stop_episodes"]
        )
        > 0,
        "not_worse_than_initial_episode_recall": float(
            terminal_event_metrics["positive_episode_recall"]
        )
        >= float(initial_terminal_event_metrics["positive_episode_recall"]),
    }
    terminal_deployable = all(terminal_quality_gate.values())
    quality_gate = {
        "zero_false_stop_episodes": int(event_metrics["false_stop_episodes"]) == 0,
        "nonzero_true_stop_episodes": int(event_metrics["true_stop_episodes"]) > 0,
        "not_worse_than_initial_episode_recall": float(
            event_metrics["positive_episode_recall"]
        )
        >= float(initial_event_metrics["positive_episode_recall"]),
    }
    deployable = all(quality_gate.values())
    print(
        "oof_event="
        + json.dumps(
            {
                "threshold": add_threshold,
                "metrics": event_metrics,
                "initial_threshold": initial_threshold,
                "initial_metrics": initial_event_metrics,
                "quality_gate": quality_gate,
                "deployable": deployable,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print(
        "oof_terminal_confirmation_event="
        + json.dumps(
            {
                "threshold": terminal_confirm_threshold,
                "confirmations": args.terminal_confirmations,
                "metrics": terminal_event_metrics,
                "initial_threshold": initial_terminal_confirm_threshold,
                "initial_metrics": initial_terminal_event_metrics,
                "quality_gate": terminal_quality_gate,
                "deployable": terminal_deployable,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    final_head, final_training = _train_fixed_epochs(
        config=config,
        initial_state=initial_state,
        features=features,
        targets=targets,
        optimization_targets=optimization_targets,
        rows=rows,
        train_indices=list(range(len(rows))),
        initial_probabilities=initial_probabilities,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        l2_sp_weight=args.l2_sp_weight,
        terminal_negative_weight=args.terminal_negative_weight,
        hard_negative_threshold=args.hard_negative_threshold,
        hard_negative_weight=args.hard_negative_weight,
        oracle_recovery_positive_weight=args.oracle_recovery_positive_weight,
        boundary_negative_min_distance_m=args.boundary_negative_min_distance_m,
        boundary_negative_max_distance_m=args.boundary_negative_max_distance_m,
        boundary_negative_weight=args.boundary_negative_weight,
        seed=args.seed + 7919,
        device=device,
    )

    head_config = config.setdefault("model", {}).setdefault("stop_head", {})
    legacy_threshold = float(head_config.get("inference_threshold", 0.5))
    veto_threshold = float(head_config.get("veto_stop_threshold", legacy_threshold))
    if veto_threshold >= add_threshold:
        veto_threshold = min(0.5, math.nextafter(add_threshold, -math.inf))
    head_config["add_stop_threshold"] = float(add_threshold)
    head_config["veto_stop_threshold"] = float(veto_threshold)
    head_config["terminal_confirm_threshold"] = float(terminal_confirm_threshold)
    head_config["terminal_confirmations"] = int(args.terminal_confirmations)
    head_config["pos_weight"] = 1.0
    head_config["bce_mix"] = 1.0
    oof_summary = {
        "schema": "heatmapvln-system2-stop-add-scene-oof-v1",
        "scene_disjoint": True,
        "fixed_epoch_cross_fit": True,
        "fold_count": int(args.folds),
        "fold_seed": int(fold_seed),
        "confirmations": int(args.confirmations),
        "minimum_add_threshold": float(args.minimum_add_threshold),
        "add_stop_threshold": float(add_threshold),
        "candidate_records": len(candidate_rows),
        "candidate_episodes": len(_candidate_groups(candidate_rows)),
        "candidate_positive_records": int((candidate_targets == 1).sum().item()),
        "candidate_negative_records": int((candidate_targets == 0).sum().item()),
        "oof_auc": _auc(candidate_oof, candidate_targets),
        "oof_call_metrics": call_metrics,
        "oof_event_metrics": event_metrics,
        "initial_add_stop_threshold": float(initial_threshold),
        "initial_call_metrics": initial_call_metrics,
        "initial_event_metrics": initial_event_metrics,
        "quality_gate": quality_gate,
        "deployable": deployable,
        "terminal_confirmation": {
            "original_terminal_records": int(original_terminal_mask.sum().item()),
            "confirmations": int(args.terminal_confirmations),
            "minimum_threshold": float(args.minimum_terminal_confirm_threshold),
            "threshold": float(terminal_confirm_threshold),
            "oof_call_metrics": terminal_call_metrics,
            "oof_event_metrics": terminal_event_metrics,
            "initial_threshold": float(initial_terminal_confirm_threshold),
            "initial_event_metrics": initial_terminal_event_metrics,
            "quality_gate": terminal_quality_gate,
            "deployable": terminal_deployable,
        },
        "folds": fold_summaries,
    }
    if probe_summary is not None:
        oof_summary["external_probe_sweeps"] = probe_summary
    if crossfit_probe_summary is not None:
        oof_summary["crossfit_probe_sweeps"] = crossfit_probe_summary
    config["rollout_stop_training"] = {
        "schema": FEATURE_SCHEMA,
        "training_scope": "all",
        "selection_objective": "add_scene_oof_event",
        "labels_jsonl": [str(path.resolve()) for path in training_label_paths],
        "crossfit_probe_labels_jsonl": [
            str(path.resolve())
            for path in (args.crossfit_probe_labels_jsonl or [])
        ],
        "crossfit_probe_rows_used_for_training": bool(crossfit_probe_rows),
        "crossfit_probe_training_rows": len(crossfit_probe_rows),
        "crossfit_probe_evaluation_scene_disjoint": bool(crossfit_probe_rows),
        "probe_labels_jsonl": [
            str(path.resolve()) for path in (args.probe_labels_jsonl or [])
        ],
        "probe_rows_used_for_training": False,
        "init_checkpoint": str(args.init_checkpoint.resolve()),
        "records": len(rows),
        "positive_records": int((targets == 1).sum().item()),
        "negative_records": int((targets == 0).sum().item()),
        "epochs_run": int(args.epochs),
        "lr": float(args.lr),
        "l2_sp_weight": float(args.l2_sp_weight),
        "terminal_negative_weight": float(args.terminal_negative_weight),
        "hard_negative_threshold": float(args.hard_negative_threshold),
        "hard_negative_weight": float(args.hard_negative_weight),
        "oracle_recovery_positive_weight": float(
            args.oracle_recovery_positive_weight
        ),
        "boundary_negative_min_distance_m": (
            float(args.boundary_negative_min_distance_m)
            if args.boundary_negative_min_distance_m is not None
            else None
        ),
        "boundary_negative_max_distance_m": (
            float(args.boundary_negative_max_distance_m)
            if args.boundary_negative_max_distance_m is not None
            else None
        ),
        "boundary_negative_weight": float(args.boundary_negative_weight),
        "optimization_positive_radius_m": (
            float(args.optimization_positive_radius_m)
            if args.optimization_positive_radius_m is not None
            else None
        ),
        "optimization_negative_radius_m": (
            float(args.optimization_negative_radius_m)
            if args.optimization_negative_radius_m is not None
            else None
        ),
        "optimization_records": int((optimization_targets >= 0.0).sum().item()),
        "optimization_positive_records": int(
            (optimization_targets == 1.0).sum().item()
        ),
        "optimization_negative_records": int(
            (optimization_targets == 0.0).sum().item()
        ),
        "relabel_ambiguous_negative_radius_m": float(
            args.relabel_ambiguous_negative_radius_m
        ),
        "scene_oof": oof_summary,
        "final_training": final_training,
    }
    metrics = {
        "val_stop_add_stop_threshold": float(add_threshold),
        "val_stop_veto_stop_threshold": float(veto_threshold),
        "add_stop_threshold": float(add_threshold),
        "veto_stop_threshold": float(veto_threshold),
        "scene_oof": oof_summary,
    }
    payload = {
        "stage_name": "system2_stop_head",
        "epoch": int(args.epochs),
        "config": config,
        "trainable_state_dict": {
            f"stop_head.{name}": value.detach().cpu()
            for name, value in final_head.state_dict().items()
        },
        "metrics": metrics,
        "source_init_checkpoint": str(args.init_checkpoint.resolve()),
        "diagnostic_oof_probabilities": candidate_oof.detach().cpu(),
        "diagnostic_initial_probabilities": candidate_initial.detach().cpu(),
        "diagnostic_targets": candidate_targets.detach().cpu(),
        "diagnostic_candidate_keys": [str(row["key"]) for row in candidate_rows],
        "diagnostic_all_oof_probabilities": oof_probabilities.detach().cpu(),
        "diagnostic_all_initial_probabilities": initial_probabilities.detach().cpu(),
        "diagnostic_all_targets": targets.detach().cpu(),
        "diagnostic_all_original_terminal": torch.tensor(
            [bool(row.get("original_terminal", False)) for row in rows],
            dtype=torch.bool,
        ),
        "diagnostic_all_keys": [str(row["key"]) for row in rows],
    }
    if probe_rows:
        payload.update(
            {
                "diagnostic_probe_oof_probabilities": probe_oof_probabilities,
                "diagnostic_probe_keys": [str(row["key"]) for row in probe_rows],
                "diagnostic_probe_group_records": probe_group_records,
                "diagnostic_scene_fold_heads": diagnostic_fold_heads,
            }
        )
    if crossfit_probe_rows:
        payload.update(
            {
                "diagnostic_crossfit_probe_oof_probabilities": (
                    crossfit_probe_probabilities.detach().cpu()
                ),
                "diagnostic_crossfit_probe_keys": [
                    str(row["key"]) for row in crossfit_probe_rows
                ],
                "diagnostic_crossfit_probe_group_records": (
                    crossfit_probe_group_records
                ),
            }
        )
    latest_path = args.output_dir / "latest.pth"
    epoch_path = args.output_dir / "checkpoints" / f"epoch_{args.epochs:03d}.pth"
    _atomic_torch_save(payload, epoch_path)
    _atomic_torch_save(payload, latest_path)
    summary = {
        "checkpoint": str(latest_path),
        "source_init_checkpoint": str(args.init_checkpoint.resolve()),
        "records": len(rows),
        "positive_records": int((targets == 1).sum().item()),
        "negative_records": int((targets == 0).sum().item()),
        "add_stop_threshold": float(add_threshold),
        "veto_stop_threshold": float(veto_threshold),
        "scene_oof": oof_summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train a scene-fold temporal ensemble for veto or conservative STOP addition."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.train_temporal_stop_verifier_from_rollout_cache import (
    _auc,
    _add_selection_score,
    _build_candidate_features,
    _calibrate,
    _calibrate_add,
    _load_rollout_rows,
    _load_static_stop_head,
    _predict,
    _predict_static,
    _read_label_rows,
    _selection_score,
    _stable_group_value,
)
from src.models.action.temporal_stop_verifier import (
    TEMPORAL_STOP_FEATURE_NAMES,
    TEMPORAL_STOP_FEATURE_SCHEMA,
    TemporalStopVerifier,
    TemporalStopVerifierEnsemble,
)


def _scene_folds(
    rows: list[dict[str, Any]],
    targets: torch.Tensor,
    *,
    fold_count: int,
    seed: int,
) -> tuple[list[list[int]], int]:
    scenes: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        scenes.setdefault(str(row["scene_id"]), []).append(index)
    if len(scenes) < fold_count:
        raise RuntimeError(
            f"Temporal STOP ensemble needs at least {fold_count} scenes, found {len(scenes)}"
        )
    for candidate_seed in range(seed, seed + 1000):
        ordered = sorted(
            scenes,
            key=lambda scene: _stable_group_value(scene, candidate_seed),
        )
        folds = [[] for _ in range(fold_count)]
        for scene_index, scene in enumerate(ordered):
            folds[scene_index % fold_count].extend(scenes[scene])
        if all(
            set(targets[torch.tensor(fold, dtype=torch.long)].int().tolist()) == {0, 1}
            for fold in folds
        ):
            return folds, candidate_seed
    raise RuntimeError("Could not construct scene-disjoint folds containing both STOP classes")


def _decision_metrics(
    decisions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    decisions = decisions.bool()
    positive = targets == 1
    negative = targets == 0
    tp = int((decisions & positive).sum().item())
    fp = int((decisions & negative).sum().item())
    tn = int((~decisions & negative).sum().item())
    fn = int((~decisions & positive).sum().item())
    return {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _train_member(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    train_indices: list[int],
    val_indices: list[int],
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    objective: str,
) -> tuple[TemporalStopVerifier, float, torch.Tensor, dict[str, Any]]:
    torch.manual_seed(seed)
    train_tensor = torch.tensor(train_indices, dtype=torch.long)
    val_tensor = torch.tensor(val_indices, dtype=torch.long)
    feature_mean = features[train_tensor].mean(dim=0)
    feature_scale = features[train_tensor].std(dim=0, unbiased=False).clamp_min(1e-4)
    member = TemporalStopVerifier(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        hidden_dim=hidden_dim,
        dropout=dropout,
        input_dim=int(features.shape[1]),
    ).to(device)
    train_targets = targets[train_tensor]
    positive_count = int((train_targets == 1).sum().item())
    negative_count = int((train_targets == 0).sum().item())
    weights = torch.where(
        train_targets == 1,
        torch.full_like(train_targets, 0.5 / positive_count),
        torch.full_like(train_targets, 0.5 / negative_count),
    ).double()
    loader = DataLoader(
        TensorDataset(features[train_tensor], train_targets),
        batch_size=batch_size,
        sampler=WeightedRandomSampler(
            weights,
            num_samples=max(len(train_indices), batch_size),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        ),
    )
    optimizer = torch.optim.AdamW(
        member.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = float("inf")
    best_bce = float("inf")
    calibrate = _calibrate_add if objective == "add" else _calibrate
    selection_score = _add_selection_score if objective == "add" else _selection_score
    for epoch in range(1, epochs + 1):
        member.train()
        for batch_features, batch_targets in loader:
            logits = member.logits(batch_features.to(device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                batch_targets.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(member.parameters(), 1.0)
            optimizer.step()
        val_probabilities = _predict(member, features[val_tensor], device)
        val_targets = targets[val_tensor]
        _threshold, val_metrics = calibrate(val_probabilities, val_targets)
        score = selection_score(val_metrics)
        bce = float(
            torch.nn.functional.binary_cross_entropy(
                val_probabilities.clamp(1e-6, 1.0 - 1e-6),
                val_targets,
            ).item()
        )
        if score < best_score - 1e-12 or (
            abs(score - best_score) <= 1e-12 and bce < best_bce
        ):
            best_epoch = epoch
            best_score = score
            best_bce = bce
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in member.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Temporal STOP fold training produced no checkpoint")
    member.load_state_dict(best_state, strict=True)
    val_probabilities = _predict(member, features[val_tensor], device)
    val_targets = targets[val_tensor]
    threshold, val_metrics = calibrate(val_probabilities, val_targets)
    return member, threshold, val_probabilities, {
        "best_epoch": int(best_epoch),
        "threshold": float(threshold),
        "auc": _auc(val_probabilities, val_targets),
        "bce": float(best_bce),
        "selection_score": selection_score(val_metrics),
        "metrics": val_metrics,
        "train_records": len(train_indices),
        "val_records": len(val_indices),
    }


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--static-stop-head-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--feature-load-workers", type=int, default=32)
    parser.add_argument("--relabel-ambiguous-negative-radius-m", type=float, default=3.01)
    parser.add_argument("--objective", choices=("veto", "add"), default="veto")
    parser.add_argument(
        "--candidate-scope",
        choices=("original_terminal", "original_nonterminal"),
        default=None,
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("folds must be >= 2")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite temporal ensemble: {args.output_dir}")
    candidate_scope = args.candidate_scope or (
        "original_nonterminal" if args.objective == "add" else "original_terminal"
    )
    expected_scope = (
        "original_nonterminal" if args.objective == "add" else "original_terminal"
    )
    if candidate_scope != expected_scope:
        raise ValueError(
            f"objective={args.objective} requires candidate-scope={expected_scope}, "
            f"got {candidate_scope}"
        )
    for path in [*args.labels_jsonl, args.static_stop_head_checkpoint]:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing non-empty input file: {path}")

    device = torch.device(args.device)
    static_head, static_spec, static_state, static_checkpoint = _load_static_stop_head(
        args.static_stop_head_checkpoint
    )
    label_rows = _read_label_rows(
        args.labels_jsonl,
        relabel_radius_m=args.relabel_ambiguous_negative_radius_m,
    )
    rows = _load_rollout_rows(label_rows, args.feature_load_workers)
    hidden = torch.stack([row["hidden"] for row in rows])
    static_probabilities = _predict_static(static_head, hidden, device)
    features, targets, candidate_rows = _build_candidate_features(
        rows,
        static_probabilities,
        candidate_scope=candidate_scope,
    )
    folds, fold_seed = _scene_folds(
        candidate_rows,
        targets,
        fold_count=args.folds,
        seed=args.seed,
    )

    members: list[TemporalStopVerifier] = []
    thresholds: list[float] = []
    fold_metrics: list[dict[str, Any]] = []
    oof_probabilities = torch.empty_like(targets)
    oof_decisions = torch.empty_like(targets, dtype=torch.bool)
    all_indices = set(range(len(candidate_rows)))
    for fold_index, val_indices in enumerate(folds):
        train_indices = sorted(all_indices - set(val_indices))
        member, threshold, val_probabilities, metrics = _train_member(
            features,
            targets,
            train_indices=train_indices,
            val_indices=val_indices,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed + 1009 * fold_index,
            device=device,
            objective=args.objective,
        )
        val_tensor = torch.tensor(val_indices, dtype=torch.long)
        oof_probabilities[val_tensor] = val_probabilities
        oof_decisions[val_tensor] = val_probabilities >= threshold
        members.append(member.cpu())
        thresholds.append(threshold)
        metrics["fold"] = fold_index
        metrics["val_scenes"] = sorted(
            {str(candidate_rows[index]["scene_id"]) for index in val_indices}
        )
        fold_metrics.append(metrics)
        print(
            f"fold={fold_index + 1}/{args.folds} "
            f"val={len(val_indices)} auc={metrics['auc']:.4f} "
            f"recall={metrics['metrics']['recall']:.4f} "
            f"fpr={metrics['metrics']['false_positive_rate']:.4f} "
            f"threshold={threshold:.3f} best_epoch={metrics['best_epoch']}",
            flush=True,
        )

    oof_metrics = _decision_metrics(oof_decisions, targets)
    oof_metrics["auc"] = _auc(oof_probabilities, targets)
    if args.objective == "veto":
        oof_metrics["selection_score"] = _selection_score(oof_metrics)
        quality_failed = (
            oof_metrics["false_positive_rate"] > 0.1
            or oof_metrics["recall"] < 0.75
        )
    else:
        oof_metrics["selection_score"] = _add_selection_score(oof_metrics)
        quality_failed = (
            oof_metrics["false_positive_rate"] > 0.0
            or oof_metrics["recall"] <= 0.0
        )
    if quality_failed:
        raise RuntimeError(
            f"Temporal STOP {args.objective} ensemble failed OOF quality gates: "
            f"{json.dumps(oof_metrics, sort_keys=True)}"
        )

    ensemble = TemporalStopVerifierEnsemble(
        members,
        torch.tensor(thresholds, dtype=torch.float32),
    )
    verifier_config = {
        "schema": TEMPORAL_STOP_FEATURE_SCHEMA,
        "feature_names": list(TEMPORAL_STOP_FEATURE_NAMES),
        "input_dim": len(TEMPORAL_STOP_FEATURE_NAMES),
        "architecture": "scene_fold_unanimous_ensemble",
        "ensemble_size": int(args.folds),
        "member_hidden_dim": int(args.hidden_dim),
        "member_dropout": float(args.dropout),
        "acceptance_thresholds": [float(value) for value in thresholds],
        "aggregation": "unanimous",
        "veto_only": args.objective == "veto",
        "add_only": args.objective == "add",
        "candidate_scope": candidate_scope,
        "history_key": ["scene_id", "episode_id", "protocol_seed"],
        "requires_contiguous_zero_based_calls": True,
    }
    payload = {
        "stage_name": (
            "system2_temporal_stop_add_ensemble"
            if args.objective == "add"
            else "system2_temporal_stop_verifier_ensemble"
        ),
        "epoch": max(int(metrics["best_epoch"]) for metrics in fold_metrics),
        "config": {
            "temporal_stop_verifier": verifier_config,
            "source_static_stop_head": static_spec,
        },
        "trainable_state_dict": {
            f"temporal_stop_ensemble.{name}": value.detach().cpu()
            for name, value in ensemble.state_dict().items()
        },
        "source_static_stop_head_state_dict": {
            f"stop_head.{name}": value.detach().cpu()
            for name, value in static_state.items()
        },
        "source_static_stop_head_checkpoint": str(
            args.static_stop_head_checkpoint.resolve()
        ),
        "source_static_stop_head_stage": static_checkpoint.get("stage_name"),
        "metrics": {
            "oof": oof_metrics,
            "folds": fold_metrics,
        },
        "training": {
            "labels_jsonl": [str(path.resolve()) for path in args.labels_jsonl],
            "rollout_rows": len(rows),
            "candidate_rows": len(candidate_rows),
            "positive_candidates": int((targets == 1).sum().item()),
            "negative_candidates": int((targets == 0).sum().item()),
            "fold_count": int(args.folds),
            "fold_seed": int(fold_seed),
            "scene_disjoint": True,
            "objective": args.objective,
            "candidate_scope": candidate_scope,
            "relabel_ambiguous_negative_radius_m": float(
                args.relabel_ambiguous_negative_radius_m
            ),
        },
    }
    latest_path = args.output_dir / "latest.pth"
    _atomic_save(payload, latest_path)
    summary = {
        "checkpoint": str(latest_path),
        "rollout_rows": len(rows),
        "candidate_rows": len(candidate_rows),
        "positive_candidates": int((targets == 1).sum().item()),
        "negative_candidates": int((targets == 0).sum().item()),
        "thresholds": thresholds,
        "objective": args.objective,
        "candidate_scope": candidate_scope,
        "oof": oof_metrics,
        "folds": fold_metrics,
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

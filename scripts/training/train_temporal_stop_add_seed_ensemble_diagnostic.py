#!/usr/bin/env python3
"""Cross-fit a seed ensemble and calibrate STOP addition on closed-loop events."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.train_temporal_stop_ensemble_from_rollout_cache import (
    _scene_folds,
)
from scripts.training.train_temporal_stop_verifier_from_rollout_cache import (
    _auc,
    _build_candidate_features,
    _load_rollout_rows,
    _load_static_stop_head,
    _predict_static,
    _read_label_rows,
)
from src.models.action.temporal_stop_verifier import TemporalStopVerifier


@torch.no_grad()
def _predict_logits(
    verifier: TemporalStopVerifier,
    features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    verifier.eval()
    return torch.cat(
        [verifier.logits(batch.to(device)).float().cpu() for batch in features.split(1024)]
    )


def _train_fixed_member(
    features: torch.Tensor,
    targets: torch.Tensor,
    train_indices: list[int],
    *,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> TemporalStopVerifier:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_tensor = torch.tensor(train_indices, dtype=torch.long)
    train_targets = targets[train_tensor]
    feature_mean = features[train_tensor].mean(dim=0)
    feature_scale = features[train_tensor].std(dim=0, unbiased=False).clamp_min(1e-4)
    verifier = TemporalStopVerifier(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        hidden_dim=hidden_dim,
        dropout=dropout,
        input_dim=int(features.shape[1]),
    ).to(device)
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
        verifier.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    for _epoch in range(epochs):
        verifier.train()
        for batch_features, batch_targets in loader:
            logits = verifier.logits(batch_features.to(device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                batch_targets.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(verifier.parameters(), 1.0)
            optimizer.step()
    return verifier


def _candidate_groups(rows: list[dict[str, Any]]) -> list[list[int]]:
    groups: dict[tuple[int, str, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[
            (
                int(row["source_index"]),
                str(row["scene_id"]),
                int(row["episode_id"]),
                int(row["protocol_seed"]),
            )
        ].append(index)
    return [
        sorted(indices, key=lambda index: int(rows[index]["system2_call_index"]))
        for indices in groups.values()
    ]


def _event_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    confirmations: int,
) -> dict[str, float | int]:
    true_stop = false_stop = missed_stop = true_continue = 0
    for indices in _candidate_groups(rows):
        has_positive = any(int(targets[index].item()) == 1 for index in indices)
        votes = 0
        previous_call: int | None = None
        outcome: int | None = None
        for index in indices:
            call_index = int(rows[index]["system2_call_index"])
            if previous_call is None or call_index != previous_call + 1:
                votes = 0
            previous_call = call_index
            votes = votes + 1 if float(scores[index].item()) >= threshold else 0
            if votes >= confirmations:
                outcome = int(targets[index].item())
                break
        if outcome == 1:
            true_stop += 1
        elif outcome == 0:
            false_stop += 1
        elif has_positive:
            missed_stop += 1
        else:
            true_continue += 1
    positive_episodes = true_stop + missed_stop + false_stop
    negative_episodes = true_continue + false_stop
    return {
        "threshold": float(threshold),
        "confirmations": int(confirmations),
        "true_stop_episodes": true_stop,
        "false_stop_episodes": false_stop,
        "missed_stop_episodes": missed_stop,
        "true_continue_episodes": true_continue,
        "positive_episode_recall": true_stop / max(positive_episodes, 1),
        "false_stop_episode_rate": false_stop / max(negative_episodes, 1),
    }


def _zero_false_event_threshold(
    scores: torch.Tensor,
    targets: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    confirmations: int,
) -> float:
    if confirmations < 1:
        raise ValueError("confirmations must be >= 1")
    unsafe_window_scores: list[float] = []
    for indices in _candidate_groups(rows):
        contiguous: list[int] = []
        previous_call: int | None = None
        for index in indices:
            call_index = int(rows[index]["system2_call_index"])
            if previous_call is None or call_index != previous_call + 1:
                contiguous = []
            previous_call = call_index
            contiguous.append(index)
            contiguous = contiguous[-confirmations:]
            if (
                len(contiguous) == confirmations
                and int(targets[index].item()) == 0
            ):
                unsafe_window_scores.append(
                    min(float(scores[item].item()) for item in contiguous)
                )
    if not unsafe_window_scores:
        return float("-inf")
    maximum_unsafe = max(unsafe_window_scores)
    return math.nextafter(maximum_unsafe, math.inf)


def _aggregate(member_logits: torch.Tensor, strategy: str) -> torch.Tensor:
    if strategy == "mean_logits":
        return member_logits.mean(dim=-1)
    if strategy == "median_logits":
        return member_logits.median(dim=-1).values
    if strategy == "minimum_logits":
        return member_logits.min(dim=-1).values
    raise ValueError(f"Unknown seed-ensemble aggregation: {strategy}")


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
    parser.add_argument("--members-per-fold", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--confirmations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--feature-load-workers", type=int, default=32)
    parser.add_argument("--relabel-ambiguous-negative-radius-m", type=float, default=3.01)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.folds < 2 or args.members_per_fold < 2:
        raise ValueError("folds and members-per-fold must be >= 2")
    if args.epochs < 1 or args.confirmations < 1:
        raise ValueError("epochs and confirmations must be >= 1")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite seed-ensemble diagnostic: {args.output_dir}")
    for path in [*args.labels_jsonl, args.static_stop_head_checkpoint]:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing non-empty input file: {path}")

    device = torch.device(args.device)
    static_head, static_spec, static_state, static_checkpoint = _load_static_stop_head(
        args.static_stop_head_checkpoint
    )
    rows = _load_rollout_rows(
        _read_label_rows(
            args.labels_jsonl,
            relabel_radius_m=args.relabel_ambiguous_negative_radius_m,
        ),
        args.feature_load_workers,
    )
    hidden = torch.stack([row["hidden"] for row in rows])
    static_probabilities = _predict_static(static_head, hidden, device)
    features, targets, candidates = _build_candidate_features(
        rows,
        static_probabilities,
        candidate_scope="original_nonterminal",
    )
    folds, fold_seed = _scene_folds(
        candidates,
        targets,
        fold_count=args.folds,
        seed=args.seed,
    )

    oof_by_strategy = {
        strategy: torch.empty_like(targets)
        for strategy in ("mean_logits", "median_logits", "minimum_logits")
    }
    fold_states: list[dict[str, torch.Tensor]] = []
    fold_summaries: list[dict[str, Any]] = []
    all_indices = set(range(len(candidates)))
    for fold_index, val_indices in enumerate(folds):
        train_indices = sorted(all_indices - set(val_indices))
        val_tensor = torch.tensor(val_indices, dtype=torch.long)
        member_logits: list[torch.Tensor] = []
        fold_member_states: dict[str, torch.Tensor] = {}
        for member_index in range(args.members_per_fold):
            member_seed = args.seed + 1009 * fold_index + 7919 * member_index
            member = _train_fixed_member(
                features,
                targets,
                train_indices,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                seed=member_seed,
                device=device,
            )
            member_logits.append(_predict_logits(member, features[val_tensor], device))
            fold_member_states.update(
                {
                    f"folds.{fold_index}.members.{member_index}.{name}": value.detach().cpu()
                    for name, value in member.state_dict().items()
                }
            )
        stacked_logits = torch.stack(member_logits, dim=-1)
        strategy_auc: dict[str, float] = {}
        for strategy, oof_scores in oof_by_strategy.items():
            aggregated = _aggregate(stacked_logits, strategy)
            oof_scores[val_tensor] = aggregated
            strategy_auc[strategy] = _auc(aggregated, targets[val_tensor])
        fold_states.append(fold_member_states)
        fold_summaries.append(
            {
                "fold": fold_index,
                "train_records": len(train_indices),
                "val_records": len(val_indices),
                "val_scenes": sorted(
                    {str(candidates[index]["scene_id"]) for index in val_indices}
                ),
                "auc": strategy_auc,
            }
        )
        print(
            f"fold={fold_index + 1}/{args.folds} val={len(val_indices)} "
            f"auc={json.dumps(strategy_auc, sort_keys=True)}",
            flush=True,
        )

    strategy_metrics: dict[str, dict[str, Any]] = {}
    for strategy, scores in oof_by_strategy.items():
        threshold = _zero_false_event_threshold(
            scores,
            targets,
            candidates,
            confirmations=args.confirmations,
        )
        event_metrics = _event_metrics(
            scores,
            targets,
            candidates,
            threshold=threshold,
            confirmations=args.confirmations,
        )
        call_predictions = scores >= threshold
        strategy_metrics[strategy] = {
            "auc": _auc(scores, targets),
            "logit_threshold": float(threshold),
            "event": event_metrics,
            "call_true_positives": int(
                (call_predictions & (targets == 1)).sum().item()
            ),
            "call_false_positives": int(
                (call_predictions & (targets == 0)).sum().item()
            ),
        }
    selected_strategy = max(
        strategy_metrics,
        key=lambda strategy: (
            int(strategy_metrics[strategy]["event"]["true_stop_episodes"]),
            float(strategy_metrics[strategy]["auc"]),
        ),
    )
    selected = strategy_metrics[selected_strategy]

    payload = {
        "stage_name": "system2_temporal_stop_add_seed_ensemble_diagnostic",
        "config": {
            "seed_ensemble": {
                "input_dim": int(features.shape[1]),
                "fold_count": int(args.folds),
                "members_per_fold": int(args.members_per_fold),
                "hidden_dim": int(args.hidden_dim),
                "dropout": float(args.dropout),
                "epochs": int(args.epochs),
                "aggregation": selected_strategy,
                "logit_threshold": float(selected["logit_threshold"]),
                "confirmations": int(args.confirmations),
                "deployable": False,
            },
            "source_static_stop_head": static_spec,
        },
        "trainable_state_dict": {
            name: value for fold_state in fold_states for name, value in fold_state.items()
        },
        "source_static_stop_head_state_dict": {
            f"stop_head.{name}": value.detach().cpu()
            for name, value in static_state.items()
        },
        "source_static_stop_head_checkpoint": str(args.static_stop_head_checkpoint.resolve()),
        "source_static_stop_head_stage": static_checkpoint.get("stage_name"),
        "metrics": {
            "selected_strategy": selected_strategy,
            "strategies": strategy_metrics,
            "folds": fold_summaries,
        },
        "training": {
            "labels_jsonl": [str(path.resolve()) for path in args.labels_jsonl],
            "rollout_rows": len(rows),
            "candidate_rows": len(candidates),
            "positive_candidates": int((targets == 1).sum().item()),
            "negative_candidates": int((targets == 0).sum().item()),
            "fold_seed": int(fold_seed),
            "scene_disjoint": True,
            "fixed_epoch_cross_fit": True,
            "relabel_ambiguous_negative_radius_m": float(
                args.relabel_ambiguous_negative_radius_m
            ),
        },
        "diagnostic_oof": {
            strategy: scores.detach().cpu() for strategy, scores in oof_by_strategy.items()
        },
        "diagnostic_targets": targets.detach().cpu(),
        "diagnostic_candidate_keys": [str(row["key"]) for row in candidates],
    }
    latest = args.output_dir / "latest.pth"
    _atomic_save(payload, latest)
    summary = {
        "checkpoint": str(latest),
        "selected_strategy": selected_strategy,
        "selected": selected,
        "strategies": strategy_metrics,
        "folds": fold_summaries,
        "rollout_rows": len(rows),
        "candidate_rows": len(candidates),
        "positive_candidates": int((targets == 1).sum().item()),
        "negative_candidates": int((targets == 0).sum().item()),
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

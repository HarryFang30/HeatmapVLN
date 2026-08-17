#!/usr/bin/env python3
"""Measure scene-disjoint STOP-add quality with temporal semantic content."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.train_temporal_stop_ensemble_from_rollout_cache import (
    _decision_metrics,
    _scene_folds,
    _train_member,
)
from scripts.training.train_temporal_stop_verifier_from_rollout_cache import (
    _auc,
    _load_rollout_rows,
    _load_static_stop_head,
    _predict_static,
    _read_label_rows,
)
from src.models.action.temporal_stop_verifier import (
    TEMPORAL_STOP_FEATURE_NAMES,
    TemporalStopObservation,
    TemporalStopVerifier,
    TemporalStopVerifierEnsemble,
    build_temporal_stop_features,
)
from src.models.heatmap.input_constructor import parse_structured_pano_output


CONTENT_FEATURE_SCHEMA = "heatmapvln-system2-temporal-stop-add-content-v1"
OUTPUT_FEATURE_DIM = 11


@torch.no_grad()
def _predict_static_embeddings(
    head: torch.nn.Module,
    hidden: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    head.to(device=device, dtype=torch.float32)
    encoder = head.classifier[:4]
    result = torch.cat(
        [encoder(batch.to(device)).float().cpu() for batch in hidden.split(1024)]
    )
    if result.ndim != 2 or not bool(torch.isfinite(result).all()):
        raise RuntimeError("Static STOP encoder produced invalid semantic embeddings")
    return result


def _output_features(text: str) -> torch.Tensor:
    parsed = parse_structured_pano_output(text, image_size=(256, 256))
    kind_names = ("pixel", "turn", "stop", "invalid")
    view_names = ("front", "right", "back", "left")
    kind = parsed.kind if parsed.kind in kind_names else "invalid"
    values = [float(kind == name) for name in kind_names]
    values.extend(float(parsed.view_id == name) for name in view_names)
    if parsed.pixel_goal is None:
        values.extend((0.0, 0.0, 0.0))
    else:
        values.extend(
            (
                1.0,
                float(parsed.pixel_goal[0]) / 255.0,
                float(parsed.pixel_goal[1]) / 255.0,
            )
        )
    if len(values) != OUTPUT_FEATURE_DIM:
        raise RuntimeError("Structured STOP output feature dimension changed")
    return torch.tensor(values, dtype=torch.float32)


def _build_content_features(
    rows: list[dict[str, Any]],
    static_probabilities: torch.Tensor,
    static_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    if static_probabilities.shape != (len(rows),):
        raise ValueError("Static probabilities must align with rollout rows")
    if static_embeddings.ndim != 2 or static_embeddings.shape[0] != len(rows):
        raise ValueError("Static embeddings must align with rollout rows")
    grouped: dict[tuple[int, str, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[
            (
                int(row["source_index"]),
                str(row["scene_id"]),
                int(row["episode_id"]),
                int(row["protocol_seed"]),
            )
        ].append(index)

    features: list[torch.Tensor] = []
    targets: list[float] = []
    candidates: list[dict[str, Any]] = []
    for group, indices in grouped.items():
        ordered = sorted(indices, key=lambda index: int(rows[index]["system2_call_index"]))
        observations: list[TemporalStopObservation] = []
        embeddings: list[torch.Tensor] = []
        outputs: list[torch.Tensor] = []
        for expected_call, index in enumerate(ordered):
            row = rows[index]
            call_index = int(row["system2_call_index"])
            if call_index != expected_call:
                raise RuntimeError(
                    "Temporal STOP rollout calls must be contiguous and zero-based: "
                    f"group={group} expected={expected_call} got={call_index}"
                )
            observations.append(
                TemporalStopObservation(
                    call_index=call_index,
                    hidden=row["hidden"],
                    static_stop_probability=float(static_probabilities[index].item()),
                    qwen_stop_log_odds=float(row["qwen_stop_log_odds"]),
                )
            )
            embeddings.append(static_embeddings[index])
            outputs.append(_output_features(str(row.get("llm_output", ""))))
            current = embeddings[-1]
            prev1 = embeddings[-2] if len(embeddings) >= 2 else current
            prev2 = embeddings[-3] if len(embeddings) >= 3 else prev1
            recent_mean = torch.stack(embeddings[-4:]).mean(dim=0)
            output_current = outputs[-1]
            output_prev1 = outputs[-2] if len(outputs) >= 2 else output_current
            output_prev2 = outputs[-3] if len(outputs) >= 3 else output_prev1
            content = torch.cat(
                (
                    current,
                    prev1,
                    prev2,
                    recent_mean,
                    build_temporal_stop_features(observations),
                    output_current,
                    output_prev1,
                    output_prev2,
                )
            )
            target = row.get("stop_target")
            if not bool(row.get("original_terminal", False)) and target in (0, 1):
                features.append(content)
                targets.append(float(target))
                candidates.append(row)
    if not features:
        raise RuntimeError("No labelled original non-STOP content candidates were found")
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    if set(target_tensor.int().tolist()) != {0, 1}:
        raise RuntimeError("Content STOP-add candidates must contain both classes")
    feature_tensor = torch.stack(features)
    if not bool(torch.isfinite(feature_tensor).all()):
        raise RuntimeError("Content STOP-add features contain non-finite values")
    return feature_tensor, target_tensor, candidates


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
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--feature-load-workers", type=int, default=32)
    parser.add_argument("--relabel-ambiguous-negative-radius-m", type=float, default=3.01)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("folds must be >= 2")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite content diagnostic: {args.output_dir}")
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
    static_embeddings = _predict_static_embeddings(static_head, hidden, device)
    features, targets, candidates = _build_content_features(
        rows,
        static_probabilities,
        static_embeddings,
    )
    folds, fold_seed = _scene_folds(
        candidates,
        targets,
        fold_count=args.folds,
        seed=args.seed,
    )

    members: list[TemporalStopVerifier] = []
    thresholds: list[float] = []
    fold_metrics: list[dict[str, Any]] = []
    oof_probabilities = torch.empty_like(targets)
    oof_decisions = torch.empty_like(targets, dtype=torch.bool)
    all_indices = set(range(len(candidates)))
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
            objective="add",
        )
        val_tensor = torch.tensor(val_indices, dtype=torch.long)
        oof_probabilities[val_tensor] = val_probabilities
        oof_decisions[val_tensor] = val_probabilities >= threshold
        members.append(member.cpu())
        thresholds.append(float(threshold))
        metrics["fold"] = fold_index
        metrics["val_scenes"] = sorted(
            {str(candidates[index]["scene_id"]) for index in val_indices}
        )
        fold_metrics.append(metrics)
        print(
            f"fold={fold_index + 1}/{args.folds} val={len(val_indices)} "
            f"auc={metrics['auc']:.4f} recall={metrics['metrics']['recall']:.4f} "
            f"fpr={metrics['metrics']['false_positive_rate']:.4f} "
            f"threshold={threshold:.3f} best_epoch={metrics['best_epoch']}",
            flush=True,
        )

    oof_metrics = _decision_metrics(oof_decisions, targets)
    oof_metrics["auc"] = _auc(oof_probabilities, targets)
    ensemble = TemporalStopVerifierEnsemble(
        members,
        torch.tensor(thresholds, dtype=torch.float32),
    )
    payload = {
        "stage_name": "system2_temporal_stop_add_content_diagnostic",
        "config": {
            "content_features": {
                "schema": CONTENT_FEATURE_SCHEMA,
                "input_dim": int(features.shape[1]),
                "compact_feature_names": list(TEMPORAL_STOP_FEATURE_NAMES),
                "static_embedding_dim": int(static_embeddings.shape[1]),
                "static_history": ["current", "prev1", "prev2", "recent4_mean"],
                "structured_output_history": ["current", "prev1", "prev2"],
            },
            "temporal_stop_verifier": {
                "architecture": "scene_fold_oof_diagnostic",
                "ensemble_size": int(args.folds),
                "member_hidden_dim": int(args.hidden_dim),
                "member_dropout": float(args.dropout),
                "acceptance_thresholds": thresholds,
                "deployable": False,
            },
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
        "source_static_stop_head_checkpoint": str(args.static_stop_head_checkpoint.resolve()),
        "source_static_stop_head_stage": static_checkpoint.get("stage_name"),
        "metrics": {"oof": oof_metrics, "folds": fold_metrics},
        "training": {
            "labels_jsonl": [str(path.resolve()) for path in args.labels_jsonl],
            "rollout_rows": len(rows),
            "candidate_rows": len(candidates),
            "positive_candidates": int((targets == 1).sum().item()),
            "negative_candidates": int((targets == 0).sum().item()),
            "fold_count": int(args.folds),
            "fold_seed": int(fold_seed),
            "scene_disjoint": True,
            "relabel_ambiguous_negative_radius_m": float(
                args.relabel_ambiguous_negative_radius_m
            ),
        },
    }
    latest = args.output_dir / "latest.pth"
    _atomic_save(payload, latest)
    summary = {
        "checkpoint": str(latest),
        "feature_dim": int(features.shape[1]),
        "rollout_rows": len(rows),
        "candidate_rows": len(candidates),
        "positive_candidates": int((targets == 1).sum().item()),
        "negative_candidates": int((targets == 0).sum().item()),
        "thresholds": thresholds,
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

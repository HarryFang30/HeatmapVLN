#!/usr/bin/env python3
"""Train a veto-only temporal verifier from train-split System2 rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.action.stop_head import StopPredictionHead
from src.models.action.temporal_stop_verifier import (
    TEMPORAL_STOP_FEATURE_NAMES,
    TEMPORAL_STOP_FEATURE_SCHEMA,
    TemporalStopObservation,
    TemporalStopVerifier,
    build_temporal_stop_features,
)


ROLLOUT_FEATURE_SCHEMA = "heatmapvln-system2-stop-feature-v1"
TEMPORAL_STOP_CANDIDATE_SCOPES = (
    "original_terminal",
    "original_nonterminal",
    "all",
)


def _normalize_state_name(raw_name: str) -> str:
    name = str(raw_name)
    for prefix in ("module.", "_orig_mod.", "model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def _load_static_stop_head(
    checkpoint_path: Path,
) -> tuple[StopPredictionHead, dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise RuntimeError(f"Invalid static STOP-head checkpoint: {checkpoint_path}")
    config = checkpoint["config"]
    model_config = config.get("model", {})
    llm_config = model_config.get("llm", {})
    head_config = model_config.get("stop_head", {})
    if not bool(head_config.get("enabled", False)):
        raise RuntimeError("Static STOP-head checkpoint does not enable model.stop_head")
    spec = {
        "input_dim": int(llm_config.get("hidden_dim", 3584)),
        "hidden_dim": int(head_config.get("hidden_dim", 512)),
        "dropout": float(head_config.get("dropout", 0.1)),
        "focal_gamma": float(head_config.get("focal_gamma", 2.0)),
        "focal_alpha": float(head_config.get("focal_alpha", 0.5)),
        "pos_weight": float(head_config.get("pos_weight", 1.0)),
        "bce_mix": float(head_config.get("bce_mix", 0.5)),
    }
    head = StopPredictionHead(**spec)
    raw_state = checkpoint.get("trainable_state_dict")
    if not isinstance(raw_state, dict):
        raw_state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if not isinstance(raw_state, dict):
        raise RuntimeError("Static STOP-head checkpoint has no state dict")
    state = {
        name.removeprefix("stop_head."): value.detach().float().cpu()
        for raw_name, value in raw_state.items()
        if (name := _normalize_state_name(raw_name)).startswith("stop_head.")
    }
    expected = head.state_dict()
    if set(state) != set(expected):
        raise RuntimeError(
            "Static STOP-head checkpoint is incomplete: "
            f"found={len(state)} expected={len(expected)} "
            f"missing={sorted(set(expected) - set(state))[:5]}"
        )
    mismatched = [
        name for name in expected if tuple(state[name].shape) != tuple(expected[name].shape)
    ]
    if mismatched:
        raise RuntimeError(f"Static STOP-head tensor shape mismatch: {mismatched[:5]}")
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise RuntimeError("Static STOP-head checkpoint contains non-finite tensors")
    head.load_state_dict(state, strict=True)
    head.requires_grad_(False)
    head.eval()
    return head, spec, state, checkpoint


def _label_target(row: dict[str, Any], relabel_radius_m: float | None) -> int | None:
    target = row.get("stop_target")
    if target in (0, 1):
        return int(target)
    if relabel_radius_m is None:
        return None
    positive_radius = float(row.get("positive_radius_m", 0.0) or 0.0)
    if relabel_radius_m <= positive_radius:
        raise RuntimeError(
            "Ambiguous-negative relabel radius must exceed the positive radius: "
            f"relabel={relabel_radius_m} positive={positive_radius}"
        )
    distance = row.get("distance_to_goal_m")
    if distance is None:
        return None
    distance = float(distance)
    if not math.isfinite(distance):
        raise RuntimeError("Rollout distance_to_goal_m must be finite when present")
    return 0 if distance >= relabel_radius_m else None


def _read_label_rows(
    paths: list[Path],
    *,
    relabel_radius_m: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_index, path in enumerate(paths):
        source_rows: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                if str(row.get("dataset_split", "")) != "train":
                    raise RuntimeError(
                        "Refusing non-train temporal STOP data: "
                        f"{path}:{line_number} split={row.get('dataset_split')!r}"
                    )
                key = str(row.get("key", ""))
                if not key:
                    raise RuntimeError(f"Missing rollout key: {path}:{line_number}")
                row = dict(row)
                row["stop_target"] = _label_target(row, relabel_radius_m)
                row["source_index"] = int(source_index)
                row["source_path"] = str(path.resolve())
                previous = source_rows.get(key)
                if previous is not None:
                    if previous.get("stop_target") != row.get("stop_target"):
                        raise RuntimeError(
                            f"Conflicting labels for duplicate key {key} in {path}"
                        )
                    previous_rank = int(bool(previous.get("oracle_forced_continue", False)))
                    current_rank = int(bool(row.get("oracle_forced_continue", False)))
                    if current_rank < previous_rank:
                        continue
                source_rows[key] = row
        rows.extend(source_rows.values())
    if not rows:
        raise RuntimeError("No temporal STOP rollout rows were found")
    return rows


def _load_rollout_payload(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(row.get("path", ""))).expanduser()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid rollout feature payload: {path}")
    if payload.get("schema") != ROLLOUT_FEATURE_SCHEMA or payload.get("key") != row["key"]:
        raise RuntimeError(f"Rollout feature metadata mismatch: {path}")
    hidden = payload.get("feature")
    if not torch.is_tensor(hidden) or hidden.ndim != 1 or hidden.numel() == 0:
        raise RuntimeError(f"Invalid rollout hidden tensor: {path}")
    hidden = hidden.detach().float().cpu().contiguous()
    if not bool(torch.isfinite(hidden).all()):
        raise RuntimeError(f"Non-finite rollout hidden tensor: {path}")
    metadata_fields = {
        "scene_id": str(payload.get("scene_id", "")),
        "episode_id": int(payload.get("episode_id", -1)),
        "system2_call_index": int(payload.get("system2_call_index", -1)),
    }
    for name, payload_value in metadata_fields.items():
        row_value = str(row.get(name, "")) if name == "scene_id" else int(row.get(name, -1))
        if row_value != payload_value:
            raise RuntimeError(
                f"Rollout {name} mismatch for {path}: row={row_value} payload={payload_value}"
            )
    scores = row.get("system2_decision_scores")
    if not isinstance(scores, dict):
        scores = payload.get("decision_scores")
    if not isinstance(scores, dict):
        raise RuntimeError(f"Rollout lacks System2 decision scores: {path}")
    qwen_log_odds = scores.get("stop_log_odds")
    if qwen_log_odds is None:
        probabilities = scores.get("class_probabilities") or {}
        stop_probability = float(probabilities.get("stop", float("nan")))
        if not math.isfinite(stop_probability) or not 0.0 <= stop_probability <= 1.0:
            raise RuntimeError(f"Rollout lacks valid Qwen STOP score: {path}")
        epsilon = 1e-8
        clipped = min(max(stop_probability, epsilon), 1.0 - epsilon)
        qwen_log_odds = math.log(clipped) - math.log1p(-clipped)
    qwen_log_odds = float(qwen_log_odds)
    if not math.isfinite(qwen_log_odds):
        raise RuntimeError(f"Rollout Qwen STOP log-odds is non-finite: {path}")
    return {
        **row,
        "hidden": hidden,
        "protocol_seed": int(payload.get("protocol_seed", -1)),
        "qwen_stop_log_odds": qwen_log_odds,
    }


def _load_rollout_rows(rows: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("feature-load-workers must be >= 1")
    if workers == 1:
        return [_load_rollout_payload(row) for row in rows]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="temporal-stop-load") as pool:
        return list(pool.map(_load_rollout_payload, rows))


@torch.no_grad()
def _predict_static(
    head: StopPredictionHead,
    hidden: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    head.to(device=device, dtype=torch.float32)
    probabilities = [head(batch.to(device)).float().cpu() for batch in hidden.split(1024)]
    return torch.cat(probabilities)


def _build_candidate_features(
    rows: list[dict[str, Any]],
    static_probabilities: torch.Tensor,
    *,
    candidate_scope: str = "original_terminal",
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    if static_probabilities.shape != (len(rows),):
        raise ValueError("Static probabilities must align with rollout rows")
    if candidate_scope not in TEMPORAL_STOP_CANDIDATE_SCOPES:
        raise ValueError(
            f"Unsupported temporal STOP candidate scope: {candidate_scope!r}"
        )
    grouped: dict[tuple[int, str, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        group = (
            int(row["source_index"]),
            str(row["scene_id"]),
            int(row["episode_id"]),
            int(row["protocol_seed"]),
        )
        grouped[group].append(index)

    candidate_features: list[torch.Tensor] = []
    candidate_targets: list[float] = []
    candidate_rows: list[dict[str, Any]] = []
    for group, indices in grouped.items():
        ordered = sorted(indices, key=lambda index: int(rows[index]["system2_call_index"]))
        observations: list[TemporalStopObservation] = []
        for expected_call, index in enumerate(ordered):
            row = rows[index]
            call_index = int(row["system2_call_index"])
            if call_index != expected_call:
                raise RuntimeError(
                    "Temporal STOP rollout calls must be contiguous and zero-based: "
                    f"group={group} expected={expected_call} got={call_index}"
                )
            observation = TemporalStopObservation(
                call_index=call_index,
                hidden=row["hidden"],
                static_stop_probability=float(static_probabilities[index].item()),
                qwen_stop_log_odds=float(row["qwen_stop_log_odds"]),
            )
            observations.append(observation)
            features = build_temporal_stop_features(observations)
            target = row.get("stop_target")
            original_terminal = bool(row.get("original_terminal", False))
            in_scope = (
                candidate_scope == "all"
                or (candidate_scope == "original_terminal" and original_terminal)
                or (candidate_scope == "original_nonterminal" and not original_terminal)
            )
            if in_scope and target in (0, 1):
                candidate_features.append(features)
                candidate_targets.append(float(target))
                candidate_rows.append(row)
    if not candidate_features:
        raise RuntimeError(
            f"No labelled temporal STOP candidates were found for scope={candidate_scope}"
        )
    targets = torch.tensor(candidate_targets, dtype=torch.float32)
    if set(targets.int().tolist()) != {0, 1}:
        raise RuntimeError("Temporal STOP candidates must contain both positive and negative labels")
    return torch.stack(candidate_features), targets, candidate_rows


def _stable_group_value(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _split_indices(
    rows: list[dict[str, Any]],
    targets: torch.Tensor,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["scene_id"])].append(index)
    if len(groups) < 3:
        raise RuntimeError("Temporal STOP training needs at least three scenes")
    ordered = sorted(groups, key=lambda group: _stable_group_value(group, seed))
    val_count = min(max(1, round(len(ordered) * val_fraction)), len(ordered) - 1)
    for offset in range(len(ordered)):
        rotated = ordered[offset:] + ordered[:offset]
        val_groups = set(rotated[:val_count])
        train = [i for group, values in groups.items() if group not in val_groups for i in values]
        val = [i for group, values in groups.items() if group in val_groups for i in values]
        train_classes = set(targets[torch.tensor(train)].int().tolist())
        val_classes = set(targets[torch.tensor(val)].int().tolist())
        if train_classes == {0, 1} and val_classes == {0, 1}:
            return train, val
    raise RuntimeError("Could not build scene-disjoint temporal STOP splits with both classes")


def _metrics(probabilities: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict[str, float]:
    predictions = probabilities >= threshold
    positive = targets == 1
    negative = targets == 0
    tp = int((predictions & positive).sum().item())
    fp = int((predictions & negative).sum().item())
    tn = int((~predictions & negative).sum().item())
    fn = int((~predictions & positive).sum().item())
    return {
        "threshold": float(threshold),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _selection_score(metrics: dict[str, float]) -> float:
    return (1.0 - float(metrics["recall"])) + 2.0 * float(metrics["false_positive_rate"])


def _calibrate(probabilities: torch.Tensor, targets: torch.Tensor) -> tuple[float, dict[str, float]]:
    candidates = [index / 200.0 for index in range(201)]
    threshold = min(
        candidates,
        key=lambda candidate: (
            _selection_score(_metrics(probabilities, targets, candidate)),
            -candidate,
        ),
    )
    return threshold, _metrics(probabilities, targets, threshold)


def _add_selection_score(metrics: dict[str, float]) -> float:
    """Prioritize zero premature STOPs, then maximize recovered true STOPs."""
    false_positive_rate = float(metrics["false_positive_rate"])
    zero_fpr_penalty = 0.0 if false_positive_rate == 0.0 else 2.0
    return zero_fpr_penalty + false_positive_rate + (1.0 - float(metrics["recall"]))


def _calibrate_add(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    """Select the lowest 0.005-grid threshold with zero validation false adds."""
    candidates = [index / 200.0 for index in range(201)]
    zero_false_positive = [
        threshold
        for threshold in candidates
        if _metrics(probabilities, targets, threshold)["false_positive_rate"] == 0.0
    ]
    threshold = min(zero_false_positive) if zero_false_positive else 1.0
    return threshold, _metrics(probabilities, targets, threshold)


def _auc(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    positive = probabilities[targets == 1]
    negative = probabilities[targets == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean().item())


@torch.no_grad()
def _predict(
    verifier: TemporalStopVerifier,
    features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    verifier.eval()
    return torch.cat(
        [verifier(batch.to(device)).float().cpu() for batch in features.split(1024)]
    )


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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--feature-load-workers", type=int, default=32)
    parser.add_argument("--relabel-ambiguous-negative-radius-m", type=float, default=3.01)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val-fraction must be in (0, 1)")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite temporal STOP output: {args.output_dir}")
    for path in [*args.labels_jsonl, args.static_stop_head_checkpoint]:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing non-empty input file: {path}")

    torch.manual_seed(args.seed)
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
    features, targets, candidate_rows = _build_candidate_features(rows, static_probabilities)
    train_indices, val_indices = _split_indices(
        candidate_rows,
        targets,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    train_tensor = torch.tensor(train_indices, dtype=torch.long)
    val_tensor = torch.tensor(val_indices, dtype=torch.long)
    feature_mean = features[train_tensor].mean(dim=0)
    feature_scale = features[train_tensor].std(dim=0, unbiased=False).clamp_min(1e-4)
    verifier = TemporalStopVerifier(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    train_targets = targets[train_tensor]
    positive_count = int((train_targets == 1).sum().item())
    negative_count = int((train_targets == 0).sum().item())
    weights = torch.where(
        train_targets == 1,
        torch.full_like(train_targets, 0.5 / positive_count),
        torch.full_like(train_targets, 0.5 / negative_count),
    ).double()
    sampler = WeightedRandomSampler(
        weights,
        num_samples=max(len(train_indices), args.batch_size),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        TensorDataset(features[train_tensor], train_targets),
        batch_size=args.batch_size,
        sampler=sampler,
    )
    optimizer = torch.optim.AdamW(
        verifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = float("inf")
    best_bce = float("inf")
    best_threshold = 0.5
    for epoch in range(1, args.epochs + 1):
        verifier.train()
        running_loss = 0.0
        batches = 0
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
            running_loss += float(loss.detach().item())
            batches += 1
        val_probabilities = _predict(verifier, features[val_tensor], device)
        val_targets = targets[val_tensor]
        threshold, val_metrics = _calibrate(val_probabilities, val_targets)
        score = _selection_score(val_metrics)
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
            best_threshold = threshold
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in verifier.state_dict().items()
            }
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch}/{args.epochs} loss={running_loss / max(batches, 1):.6f} "
                f"val_auc={_auc(val_probabilities, val_targets):.4f} "
                f"val_recall={val_metrics['recall']:.4f} "
                f"val_fpr={val_metrics['false_positive_rate']:.4f} "
                f"threshold={threshold:.3f} selection={score:.4f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("Temporal STOP training did not produce a checkpoint")
    verifier.load_state_dict(best_state, strict=True)
    val_probabilities = _predict(verifier, features[val_tensor], device)
    val_targets = targets[val_tensor]
    threshold, val_metrics = _calibrate(val_probabilities, val_targets)
    if not math.isclose(threshold, best_threshold, abs_tol=1e-12):
        raise RuntimeError("Restored temporal STOP threshold does not match best epoch")

    current_static_logits = features[:, TEMPORAL_STOP_FEATURE_NAMES.index("static_logit_current")]
    current_qwen_logits = features[:, TEMPORAL_STOP_FEATURE_NAMES.index("qwen_stop_log_odds_current")]
    metrics = {
        "acceptance_threshold": float(threshold),
        "val": val_metrics,
        "val_auc": _auc(val_probabilities, val_targets),
        "val_selection_score": _selection_score(val_metrics),
        "val_bce": best_bce,
        "val_static_logit_auc": _auc(current_static_logits[val_tensor], val_targets),
        "val_qwen_log_odds_auc": _auc(current_qwen_logits[val_tensor], val_targets),
    }
    temporal_config = {
        "schema": TEMPORAL_STOP_FEATURE_SCHEMA,
        "feature_names": list(TEMPORAL_STOP_FEATURE_NAMES),
        "input_dim": len(TEMPORAL_STOP_FEATURE_NAMES),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "acceptance_threshold": float(threshold),
        "veto_only": True,
        "history_key": ["scene_id", "episode_id", "protocol_seed"],
        "requires_contiguous_zero_based_calls": True,
    }
    payload = {
        "stage_name": "system2_temporal_stop_verifier",
        "epoch": int(best_epoch),
        "config": {
            "temporal_stop_verifier": temporal_config,
            "source_static_stop_head": static_spec,
        },
        "trainable_state_dict": {
            f"temporal_stop_verifier.{name}": value.detach().cpu()
            for name, value in verifier.state_dict().items()
        },
        "source_static_stop_head_state_dict": {
            f"stop_head.{name}": value.detach().cpu()
            for name, value in static_state.items()
        },
        "source_static_stop_head_checkpoint": str(
            args.static_stop_head_checkpoint.resolve()
        ),
        "source_static_stop_head_stage": static_checkpoint.get("stage_name"),
        "metrics": metrics,
        "training": {
            "labels_jsonl": [str(path.resolve()) for path in args.labels_jsonl],
            "rollout_rows": len(rows),
            "candidate_rows": len(candidate_rows),
            "positive_candidates": int((targets == 1).sum().item()),
            "negative_candidates": int((targets == 0).sum().item()),
            "train_candidates": len(train_indices),
            "val_candidates": len(val_indices),
            "scene_disjoint": True,
            "seed": int(args.seed),
            "epochs_run": int(args.epochs),
            "best_epoch": int(best_epoch),
            "relabel_ambiguous_negative_radius_m": float(
                args.relabel_ambiguous_negative_radius_m
            ),
        },
    }
    latest_path = args.output_dir / "latest.pth"
    epoch_path = args.output_dir / "checkpoints" / f"epoch_{best_epoch:03d}.pth"
    _atomic_save(payload, latest_path)
    _atomic_save(payload, epoch_path)
    summary = {
        "checkpoint": str(latest_path),
        "best_epoch": best_epoch,
        "rollout_rows": len(rows),
        "candidate_rows": len(candidate_rows),
        "positive_candidates": int((targets == 1).sum().item()),
        "negative_candidates": int((targets == 0).sum().item()),
        "train_candidates": len(train_indices),
        "val_candidates": len(val_indices),
        "metrics": metrics,
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

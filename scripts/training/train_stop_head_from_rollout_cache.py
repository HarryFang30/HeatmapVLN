#!/usr/bin/env python3
"""Fine-tune the isolated System2 STOP head on train-split rollout features."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.action.stop_head import StopPredictionHead


FEATURE_SCHEMA = "heatmapvln-system2-stop-feature-v1"


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise RuntimeError(f"Invalid STOP-head checkpoint: {path}")
    raw_state = checkpoint.get("trainable_state_dict")
    if not isinstance(raw_state, dict):
        raw_state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if not isinstance(raw_state, dict):
        raise RuntimeError(f"STOP-head checkpoint has no state dict: {path}")

    state: dict[str, torch.Tensor] = {}
    for raw_name, value in raw_state.items():
        name = str(raw_name)
        for prefix in ("module.", "_orig_mod.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        if name.startswith("stop_head."):
            state[name.removeprefix("stop_head.")] = value
    if not state:
        raise RuntimeError(f"STOP-head checkpoint contains no stop_head tensors: {path}")
    return checkpoint, state


def _build_head(config: dict[str, Any], state: dict[str, torch.Tensor]) -> StopPredictionHead:
    model_config = config.get("model", {})
    llm_config = model_config.get("llm", {})
    head_config = model_config.get("stop_head", {})
    head = StopPredictionHead(
        input_dim=int(llm_config.get("hidden_dim", 3584)),
        hidden_dim=int(head_config.get("hidden_dim", 512)),
        dropout=float(head_config.get("dropout", 0.1)),
        focal_gamma=float(head_config.get("focal_gamma", 2.0)),
        focal_alpha=float(head_config.get("focal_alpha", 0.5)),
        pos_weight=1.0,
        bce_mix=1.0,
    )
    missing, unexpected = head.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"STOP-head state mismatch: missing={missing} unexpected={unexpected}")
    return head


def _read_rows(
    paths: list[Path],
    *,
    allow_nontrain: bool,
    relabel_ambiguous_negative_radius_m: float | None = None,
) -> list[dict[str, Any]]:
    deduplicated: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row.get("key", ""))
                if not key:
                    raise RuntimeError(f"Missing feature key at {path}:{line_number}")
                split = str(row.get("dataset_split", ""))
                if split != "train" and not allow_nontrain:
                    raise RuntimeError(
                        f"Refusing non-train STOP feature {key} from split={split!r}; "
                        "val_seen/val_unseen are evaluation-only"
                    )
                raw_feature_path = str(row.get("path", "")).strip()
                if not raw_feature_path:
                    raise RuntimeError(f"Missing feature path at {path}:{line_number}")
                if (
                    relabel_ambiguous_negative_radius_m is not None
                    and row.get("stop_target") is None
                ):
                    try:
                        distance = float(row.get("distance_to_goal_m", float("nan")))
                    except (TypeError, ValueError):
                        distance = float("nan")
                    try:
                        positive_radius = float(row.get("positive_radius_m", 3.0))
                    except (TypeError, ValueError) as error:
                        raise RuntimeError(
                            f"Invalid positive_radius_m at {path}:{line_number}"
                        ) from error
                    if not math.isfinite(positive_radius) or positive_radius <= 0.0:
                        raise RuntimeError(
                            f"Invalid positive_radius_m={positive_radius} at "
                            f"{path}:{line_number}"
                        )
                    if relabel_ambiguous_negative_radius_m <= positive_radius:
                        raise RuntimeError(
                            "Ambiguous-negative relabel radius must exceed the positive "
                            f"radius: {relabel_ambiguous_negative_radius_m} <= "
                            f"{positive_radius} at {path}:{line_number}"
                        )
                    if math.isfinite(distance) and distance >= relabel_ambiguous_negative_radius_m:
                        row = dict(row)
                        row["stop_target"] = 0
                        row["ambiguous_negative_relabelled"] = True
                row = dict(row)
                row["path"] = str(Path(raw_feature_path).expanduser().resolve())
                previous = deduplicated.get(key)
                if previous is not None:
                    previous_target = previous.get("stop_target")
                    current_target = row.get("stop_target")
                    if (
                        previous_target in (0, 1)
                        and current_target in (0, 1)
                        and int(previous_target) != int(current_target)
                    ):
                        raise RuntimeError(
                            f"Conflicting STOP labels for deterministic feature key {key}: "
                            f"{previous_target} vs {current_target}"
                        )
                    previous_identity = (
                        str(previous.get("dataset_split", "")),
                        str(previous.get("scene_id", "")),
                        int(previous.get("episode_id", -1)),
                        int(previous.get("system2_call_index", -1)),
                    )
                    current_identity = (
                        str(row.get("dataset_split", "")),
                        str(row.get("scene_id", "")),
                        int(row.get("episode_id", -1)),
                        int(row.get("system2_call_index", -1)),
                    )
                    if previous_identity != current_identity:
                        raise RuntimeError(
                            f"Conflicting metadata for deterministic feature key {key}: "
                            f"{previous_identity} vs {current_identity}"
                        )
                    previous_rank = (
                        int(previous_target in (0, 1)),
                        int(bool(previous.get("oracle_forced_continue", False))),
                    )
                    current_rank = (
                        int(current_target in (0, 1)),
                        int(bool(row.get("oracle_forced_continue", False))),
                    )
                    if current_rank < previous_rank:
                        continue
                deduplicated[key] = row
    rows = [row for row in deduplicated.values() if row.get("stop_target") in (0, 1)]
    if not rows:
        raise RuntimeError("No non-ambiguous STOP rollout labels were found")
    return rows


def _filter_training_scope(
    rows: list[dict[str, Any]],
    *,
    scope: str,
) -> list[dict[str, Any]]:
    if scope == "all":
        return rows
    if scope != "original-terminal":
        raise ValueError(f"Unsupported STOP rollout training scope: {scope}")
    filtered = [row for row in rows if bool(row.get("original_terminal", False))]
    if not filtered:
        raise RuntimeError("No original System2 STOP candidates were found")
    classes = {int(row["stop_target"]) for row in filtered}
    if classes != {0, 1}:
        raise RuntimeError(
            "Original-terminal STOP training scope must contain both classes"
        )
    return filtered


def _load_feature(row: dict[str, Any]) -> tuple[Path, torch.Tensor, float]:
    path = Path(str(row["path"]))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != FEATURE_SCHEMA or payload.get("key") != row["key"]:
        raise RuntimeError(f"STOP feature metadata mismatch: {path}")
    feature = payload.get("feature")
    if not torch.is_tensor(feature) or feature.ndim != 1:
        raise RuntimeError(f"Invalid STOP feature tensor: {path}")
    feature = feature.float().contiguous()
    if not torch.isfinite(feature).all():
        raise RuntimeError(f"Non-finite STOP feature tensor: {path}")
    return path, feature, float(row["stop_target"])


def _load_features(
    rows: list[dict[str, Any]],
    *,
    workers: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if workers < 1:
        raise ValueError("feature load workers must be >= 1")

    features: list[torch.Tensor] = []
    targets: list[float] = []
    hidden_dim = None
    if workers == 1:
        loaded_features = map(_load_feature, rows)
        executor = None
    else:
        executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="stop-feature-load",
        )
        loaded_features = executor.map(_load_feature, rows)

    try:
        for path, feature, target in loaded_features:
            if hidden_dim is None:
                hidden_dim = int(feature.numel())
            elif feature.numel() != hidden_dim:
                raise RuntimeError(f"Inconsistent STOP feature dimension: {path}")
            features.append(feature)
            targets.append(target)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return torch.stack(features), torch.tensor(targets, dtype=torch.float32)


def _stable_group_value(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _split_indices(
    rows: list[dict[str, Any]],
    targets: torch.Tensor,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], str]:
    scenes = {str(row["scene_id"]) for row in rows}
    use_scene = len(scenes) >= 3
    group_kind = "scene" if use_scene else "episode"
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        group = (
            str(row["scene_id"])
            if use_scene
            else f"{row['scene_id']}:{int(row['episode_id'])}"
        )
        groups.setdefault(group, []).append(index)
    if len(groups) < 2:
        raise RuntimeError("STOP rollout cache needs at least two scene/episode groups")

    ordered = sorted(groups, key=lambda group: _stable_group_value(group, seed))
    val_group_count = min(max(1, round(len(ordered) * val_fraction)), len(ordered) - 1)
    val_groups = set(ordered[:val_group_count])
    train_indices = [i for group, values in groups.items() if group not in val_groups for i in values]
    val_indices = [i for group, values in groups.items() if group in val_groups for i in values]

    def classes(indices: list[int]) -> set[int]:
        return {int(targets[index].item()) for index in indices}

    if classes(train_indices) != {0, 1} or classes(val_indices) != {0, 1}:
        raise RuntimeError(
            "STOP rollout train/validation split must contain both classes; "
            "collect more episodes across scenes"
        )
    return train_indices, val_indices, group_kind


def _metrics(probabilities: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict[str, float]:
    predictions = probabilities >= threshold
    positive = targets == 1
    negative = ~positive
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


def _validation_selection_score(metrics: dict[str, float]) -> float:
    """Penalize premature STOP twice as much as a missed STOP."""
    false_negative_rate = 1.0 - float(metrics["recall"])
    return false_negative_rate + 2.0 * float(metrics["false_positive_rate"])


def _add_validation_selection_score(metrics: dict[str, float]) -> float:
    """Select STOP-add models lexicographically: avoid false adds, then gain recall."""
    false_negative_rate = 1.0 - float(metrics["recall"])
    return 1000.0 * float(metrics["false_positive_rate"]) + false_negative_rate


def _build_sampling_weights(
    rows: list[dict[str, Any]],
    targets: torch.Tensor,
    train_indices: list[int],
    initial_probabilities: torch.Tensor,
    *,
    terminal_negative_weight: float,
    hard_negative_threshold: float,
    hard_negative_weight: float,
    oracle_recovery_positive_weight: float = 1.0,
    boundary_negative_min_distance_m: float | None = None,
    boundary_negative_max_distance_m: float | None = None,
    boundary_negative_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, int]]:
    if initial_probabilities.shape != targets.shape:
        raise ValueError("Initial STOP probabilities must match target shape")
    train_targets = targets[torch.tensor(train_indices, dtype=torch.long)]
    positive_count = int((train_targets == 1).sum().item())
    negative_count = int((train_targets == 0).sum().item())
    if positive_count == 0 or negative_count == 0:
        raise RuntimeError("STOP rollout training split requires both classes")

    positive_multipliers: dict[int, float] = {}
    negative_multipliers: dict[int, float] = {}
    recovery_positive_count = 0
    terminal_negative_count = 0
    hard_negative_count = 0
    boundary_negative_count = 0
    for row_index in train_indices:
        if int(targets[row_index].item()) == 1:
            multiplier = 1.0
            if bool(rows[row_index].get("oracle_recovery_active", False)):
                multiplier *= oracle_recovery_positive_weight
                recovery_positive_count += 1
            positive_multipliers[row_index] = multiplier
            continue
        multiplier = 1.0
        if bool(rows[row_index].get("original_terminal", False)):
            multiplier *= terminal_negative_weight
            terminal_negative_count += 1
        if float(initial_probabilities[row_index].item()) >= hard_negative_threshold:
            multiplier *= hard_negative_weight
            hard_negative_count += 1
        if (
            boundary_negative_min_distance_m is not None
            and boundary_negative_max_distance_m is not None
        ):
            try:
                distance_m = float(rows[row_index].get("distance_to_goal_m", float("nan")))
            except (TypeError, ValueError):
                distance_m = float("nan")
            if (
                math.isfinite(distance_m)
                and boundary_negative_min_distance_m
                <= distance_m
                < boundary_negative_max_distance_m
            ):
                multiplier *= boundary_negative_weight
                boundary_negative_count += 1
        negative_multipliers[row_index] = multiplier

    positive_mass = sum(positive_multipliers.values())
    negative_mass = sum(negative_multipliers.values())
    weights = torch.empty(len(train_indices), dtype=torch.double)
    for local_index, row_index in enumerate(train_indices):
        if int(targets[row_index].item()) == 1:
            weights[local_index] = (
                0.5 * positive_multipliers[row_index] / positive_mass
            )
        else:
            weights[local_index] = 0.5 * negative_multipliers[row_index] / negative_mass
    return weights, {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "recovery_positive_count": recovery_positive_count,
        "terminal_negative_count": terminal_negative_count,
        "hard_negative_count": hard_negative_count,
        "boundary_negative_count": boundary_negative_count,
    }


def _veto_calibration_subset(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    original_terminal: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Use only states where the original policy requested STOP for veto calibration."""
    if original_terminal is None:
        return probabilities, targets, "all_records"
    if original_terminal.shape != targets.shape:
        raise ValueError("original_terminal mask must match STOP targets")
    mask = original_terminal.bool()
    terminal_targets = targets[mask]
    if terminal_targets.numel() >= 2 and set(terminal_targets.int().tolist()) == {0, 1}:
        return probabilities[mask], terminal_targets, "original_terminal"
    return probabilities, targets, "all_records_fallback"


def _calibrate(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    original_terminal: torch.Tensor | None = None,
) -> tuple[float, float]:
    candidates = [index / 200.0 for index in range(201)]
    add_candidates = [
        threshold
        for threshold in candidates
        if threshold >= 0.9
        and _metrics(probabilities, targets, threshold)["false_positive_rate"] == 0.0
    ]
    add_threshold = min(add_candidates) if add_candidates else 1.0
    veto_probabilities, veto_targets, _ = _veto_calibration_subset(
        probabilities,
        targets,
        original_terminal,
    )
    veto_candidates = [threshold for threshold in candidates if threshold < add_threshold]
    veto_threshold = min(
        veto_candidates,
        key=lambda threshold: (
            _validation_selection_score(
                _metrics(veto_probabilities, veto_targets, threshold)
            ),
            # Equal-cost thresholds have identical validation confusion
            # matrices. Prefer the lower one so an uncertain valid STOP is
            # retained instead of being vetoed solely by tie-breaking.
            threshold,
        ),
    )
    return add_threshold, veto_threshold


def _build_checkpoint_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    original_terminal: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build metrics while preserving the original STOP-head checkpoint API."""
    add_threshold, veto_threshold = _calibrate(
        probabilities,
        targets,
        original_terminal,
    )
    veto_probabilities, veto_targets, veto_scope = _veto_calibration_subset(
        probabilities,
        targets,
        original_terminal,
    )
    return {
        "val_at_0_5": _metrics(probabilities, targets, 0.5),
        "val_at_add_threshold": _metrics(probabilities, targets, add_threshold),
        "val_at_veto_threshold": _metrics(
            veto_probabilities,
            veto_targets,
            veto_threshold,
        ),
        "val_all_at_veto_threshold": _metrics(
            probabilities,
            targets,
            veto_threshold,
        ),
        "veto_calibration_scope": veto_scope,
        "veto_calibration_records": int(veto_targets.numel()),
        "add_stop_threshold": add_threshold,
        "veto_stop_threshold": veto_threshold,
        "val_stop_add_stop_threshold": add_threshold,
        "val_stop_veto_stop_threshold": veto_threshold,
    }


@torch.no_grad()
def _predict(head: StopPredictionHead, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    head.eval()
    probabilities: list[torch.Tensor] = []
    for batch in features.split(1024):
        probabilities.append(head(batch.to(device)).float().cpu())
    return torch.cat(probabilities)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--l2-sp-weight", type=float, default=1e-4)
    parser.add_argument("--terminal-negative-weight", type=float, default=4.0)
    parser.add_argument("--hard-negative-threshold", type=float, default=0.8)
    parser.add_argument("--hard-negative-weight", type=float, default=4.0)
    parser.add_argument("--oracle-recovery-positive-weight", type=float, default=1.0)
    parser.add_argument(
        "--relabel-ambiguous-negative-radius-m",
        type=float,
        default=None,
        help=(
            "For rollout rows with stop_target=null, relabel exact Habitat distances "
            "at or above this radius as negatives. Static expert-path labels are unchanged."
        ),
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-load-workers", type=int, default=1)
    parser.add_argument(
        "--training-scope",
        choices=("all", "original-terminal"),
        default="all",
        help=(
            "Use all labelled rollout states, or only states where the original "
            "System2 policy requested STOP. The latter trains a veto-only candidate verifier."
        ),
    )
    parser.add_argument(
        "--selection-objective",
        choices=("veto", "add"),
        default="veto",
        help=(
            "Metric used to select the best epoch. Use add when the checkpoint "
            "will only propose missing STOPs; veto preserves the legacy policy."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-nontrain-diagnostic", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.feature_load_workers < 1:
        raise ValueError("epochs, batch-size, and feature-load-workers must be >= 1")
    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("val-fraction must be in (0, 0.5)")
    if (
        args.lr <= 0.0
        or args.l2_sp_weight < 0.0
        or args.terminal_negative_weight < 1.0
        or not 0.0 <= args.hard_negative_threshold <= 1.0
        or args.hard_negative_weight < 1.0
        or args.oracle_recovery_positive_weight < 1.0
        or (
            args.relabel_ambiguous_negative_radius_m is not None
            and (
                not math.isfinite(args.relabel_ambiguous_negative_radius_m)
                or args.relabel_ambiguous_negative_radius_m <= 0.0
            )
        )
    ):
        raise ValueError("Invalid STOP rollout optimization settings")

    checkpoint, initial_state = _load_checkpoint(args.init_checkpoint)
    config = copy.deepcopy(checkpoint["config"])
    rows = _read_rows(
        args.labels_jsonl,
        allow_nontrain=args.allow_nontrain_diagnostic,
        relabel_ambiguous_negative_radius_m=args.relabel_ambiguous_negative_radius_m,
    )
    rows = _filter_training_scope(rows, scope=args.training_scope)
    feature_load_started = time.monotonic()
    print(
        f"loading_features=records={len(rows)} workers={args.feature_load_workers}",
        flush=True,
    )
    features, targets = _load_features(rows, workers=args.feature_load_workers)
    print(
        f"loaded_features=records={len(rows)} seconds={time.monotonic() - feature_load_started:.1f}",
        flush=True,
    )
    original_terminal = torch.tensor(
        [bool(row.get("original_terminal", False)) for row in rows],
        dtype=torch.bool,
    )
    train_indices, val_indices, group_kind = _split_indices(
        rows,
        targets,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    head = _build_head(config, initial_state).to(device=device, dtype=torch.float32)
    reference = {
        name: parameter.detach().clone()
        for name, parameter in head.named_parameters()
    }
    initial_probabilities = _predict(head, features, device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_tensor = torch.tensor(train_indices, dtype=torch.long)
    train_targets = targets[train_tensor]
    weights, sampling_stats = _build_sampling_weights(
        rows,
        targets,
        train_indices,
        initial_probabilities,
        terminal_negative_weight=args.terminal_negative_weight,
        hard_negative_threshold=args.hard_negative_threshold,
        hard_negative_weight=args.hard_negative_weight,
        oracle_recovery_positive_weight=args.oracle_recovery_positive_weight,
    )
    print(
        "sampling="
        f"positive={sampling_stats['positive_count']} "
        f"negative={sampling_stats['negative_count']} "
        f"recovery_positive={sampling_stats['recovery_positive_count']} "
        f"terminal_negative={sampling_stats['terminal_negative_count']} "
        f"hard_negative={sampling_stats['hard_negative_count']} "
        f"hard_threshold={args.hard_negative_threshold:.3f}",
        flush=True,
    )
    sampler = WeightedRandomSampler(
        weights,
        num_samples=max(len(train_indices), args.batch_size),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    dataset = TensorDataset(features[train_tensor], train_targets)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    best_epoch = 0
    best_selection_score = float("inf")
    best_validation_bce = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        batches = 0
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            logits = head.classifier(batch_features).squeeze(-1)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch_targets)
            l2_sp = sum(
                (parameter - reference[name]).float().square().mean()
                for name, parameter in head.named_parameters()
            )
            loss = bce + args.l2_sp_weight * l2_sp
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().item())
            batches += 1
        val_tensor = torch.tensor(val_indices, dtype=torch.long)
        val_probabilities = _predict(head, features[val_tensor], device)
        val_targets = targets[val_tensor]
        epoch_metrics = _build_checkpoint_metrics(
            val_probabilities,
            val_targets,
            original_terminal[val_tensor],
        )
        if args.selection_objective == "add":
            val_metrics = epoch_metrics["val_at_add_threshold"]
            selection_score = _add_validation_selection_score(val_metrics)
            selected_threshold = epoch_metrics["add_stop_threshold"]
        else:
            val_metrics = epoch_metrics["val_at_veto_threshold"]
            selection_score = _validation_selection_score(val_metrics)
            selected_threshold = epoch_metrics["veto_stop_threshold"]
        validation_bce = float(
            torch.nn.functional.binary_cross_entropy(
                val_probabilities.clamp(1e-6, 1.0 - 1e-6),
                val_targets,
            ).item()
        )
        if (
            selection_score < best_selection_score - 1e-12
            or (
                abs(selection_score - best_selection_score) <= 1e-12
                and validation_bce < best_validation_bce
            )
        ):
            best_epoch = epoch
            best_selection_score = selection_score
            best_validation_bce = validation_bce
            best_state = {
                name: value.detach().clone()
                for name, value in head.state_dict().items()
            }
        print(
            f"epoch={epoch}/{args.epochs} loss={total_loss / max(batches, 1):.6f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} "
            f"val_fpr={val_metrics['false_positive_rate']:.4f} "
            f"selection={selection_score:.4f} "
            f"val_bce={validation_bce:.6f} "
            f"selection_objective={args.selection_objective} "
            f"selected_threshold={selected_threshold:.3f} "
            f"veto_scope={epoch_metrics['veto_calibration_scope']}",
            flush=True,
        )

    if best_state is None or best_epoch < 1:
        raise RuntimeError("STOP rollout training did not produce a best checkpoint")
    head.load_state_dict(best_state, strict=True)
    print(
        f"restored_best_epoch={best_epoch} selection={best_selection_score:.6f} "
        f"val_bce={best_validation_bce:.6f}",
        flush=True,
    )
    val_tensor = torch.tensor(val_indices, dtype=torch.long)
    val_probabilities = _predict(head, features[val_tensor], device)
    val_targets = targets[val_tensor]
    metrics = _build_checkpoint_metrics(
        val_probabilities,
        val_targets,
        original_terminal[val_tensor],
    )
    add_threshold = float(metrics["add_stop_threshold"])
    veto_threshold = float(metrics["veto_stop_threshold"])
    head_config = config.setdefault("model", {}).setdefault("stop_head", {})
    head_config["add_stop_threshold"] = float(add_threshold)
    head_config["veto_stop_threshold"] = float(veto_threshold)
    head_config["pos_weight"] = 1.0
    head_config["bce_mix"] = 1.0
    config["rollout_stop_training"] = {
        "schema": FEATURE_SCHEMA,
        "training_scope": str(args.training_scope),
        "selection_objective": str(args.selection_objective),
        "group_split": group_kind,
        "train_records": len(train_indices),
        "val_records": len(val_indices),
        "labels_jsonl": [str(path) for path in args.labels_jsonl],
        "init_checkpoint": str(args.init_checkpoint),
        "l2_sp_weight": float(args.l2_sp_weight),
        "terminal_negative_weight": float(args.terminal_negative_weight),
        "hard_negative_threshold": float(args.hard_negative_threshold),
        "hard_negative_weight": float(args.hard_negative_weight),
        "oracle_recovery_positive_weight": float(
            args.oracle_recovery_positive_weight
        ),
        "oracle_recovery_positive_records": int(
            sampling_stats["recovery_positive_count"]
        ),
        "hard_negative_records": int(sampling_stats["hard_negative_count"]),
        "relabel_ambiguous_negative_radius_m": (
            float(args.relabel_ambiguous_negative_radius_m)
            if args.relabel_ambiguous_negative_radius_m is not None
            else None
        ),
        "relabelled_negative_records": sum(
            bool(row.get("ambiguous_negative_relabelled", False)) for row in rows
        ),
        "epochs_run": int(args.epochs),
        "best_epoch": int(best_epoch),
        "validation_selection_score": float(best_selection_score),
        "validation_bce": float(best_validation_bce),
        "veto_calibration_scope": str(metrics["veto_calibration_scope"]),
    }
    payload = {
        "stage_name": "system2_stop_head",
        "epoch": int(best_epoch),
        "config": config,
        "trainable_state_dict": {
            f"stop_head.{name}": value.detach().cpu()
            for name, value in head.state_dict().items()
        },
        "metrics": metrics,
        "source_init_checkpoint": str(args.init_checkpoint),
    }
    epoch_path = args.output_dir / "checkpoints" / f"epoch_{best_epoch:03d}.pth"
    latest_path = args.output_dir / "latest.pth"
    _atomic_torch_save(payload, epoch_path)
    _atomic_torch_save(payload, latest_path)
    summary = {
        "checkpoint": str(latest_path),
        "training_scope": str(args.training_scope),
        "selection_objective": str(args.selection_objective),
        "records": len(rows),
        "train_records": len(train_indices),
        "val_records": len(val_indices),
        "epochs_run": int(args.epochs),
        "best_epoch": int(best_epoch),
        "validation_selection_score": float(best_selection_score),
        "validation_bce": float(best_validation_bce),
        "positive_records": int((targets == 1).sum().item()),
        "negative_records": int((targets == 0).sum().item()),
        "relabelled_negative_records": sum(
            bool(row.get("ambiguous_negative_relabelled", False)) for row in rows
        ),
        "metrics": metrics,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train an isolated System2 STOP-decision LoRA without touching navigation LoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training import (
    _load_normalized_state_dict,
    assert_complete_lora_checkpoint_match,
    build_model,
    extract_lora_checkpoint_state,
    load_config,
)

from src.data.factory import build_dataset
from src.data.panoramic_tokenized_collator import PanoramicTokenizedCollator
from src.data.stop_rollout_dataset import System2StopMultimodalDataset
from src.models.qwen2_5_vl.integration import (
    DEFAULT_LORA_ADAPTER_NAME,
    STOP_DECISION_ADAPTER_NAME,
)
from src.models.runtime_compat import ensure_transformers_runtime_compat

LOGGER = logging.getLogger("system2-stop-decision-adapter")
CHECKPOINT_SCHEMA = "heatmapvln-system2-stop-decision-adapter-v1"
ADD_AND_VETO_POLICY = "add_and_veto"
VETO_ONLY_POLICY = "veto_only"
MIN_VETO_RECALL = 0.98
MIN_VETO_NEGATIVE_REJECTION = 0.20
# STOP addition is evaluated at every System2 call. Even a 1% per-call false
# positive rate compounds into an unsafe episode-level premature-STOP rate, so
# deployment requires zero observed additions on held-out regular negatives.
MAX_ADD_FALSE_POSITIVE_RATE = 0.0
MIN_ADD_RECALL = 0.50
MIN_ROC_AUC = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--rollout-root",
        action="append",
        default=[],
        help=(
            "Privileged train-split multimodal rollout root. Repeat to combine "
            "scene-disjoint collection shards. When set, native clip data is not used."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--negative-loss-weight", type=float, default=2.0)
    parser.add_argument(
        "--terminal-negative-sampling-fraction",
        type=float,
        default=0.25,
        help=(
            "Fraction of sampled non-STOP rows reserved for on-policy examples "
            "where the original System2 response incorrectly requested STOP."
        ),
    )
    parser.add_argument(
        "--mined-train-scores-jsonl",
        type=Path,
        help=(
            "Complete train-split score audit from an earlier candidate. The "
            "highest-scoring ordinary negatives are sampled as a separate pool."
        ),
    )
    parser.add_argument(
        "--mined-regular-negative-count",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--mined-regular-negative-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--layers", default="20-27")
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument(
        "--holdout-scene-fraction",
        "--holdout-clip-fraction",
        dest="holdout_scene_fraction",
        type=float,
        default=0.1,
    )
    parser.add_argument("--bce-loss-weight", type=float, default=0.1)
    parser.add_argument("--ranking-loss-weight", type=float, default=1.0)
    parser.add_argument("--ranking-margin", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--validation-adapter-checkpoint",
        type=Path,
        help=(
            "Skip optimization, load a successful or failed STOP-decision "
            "adapter candidate, and replay held-out validation with a "
            "per-example score audit."
        ),
    )
    parser.add_argument(
        "--replay-split",
        choices=("validation", "train"),
        default="validation",
        help=(
            "Dataset split scored by validation replay. Use train only for "
            "hard-negative mining; deployment calibration must use validation."
        ),
    )
    parser.add_argument(
        "--export-veto-only-checkpoint",
        action="store_true",
        help=(
            "During validation replay, export a deployment checkpoint that may "
            "only veto original STOP outputs. The held-out veto and AUC quality "
            "gates must pass."
        ),
    )
    parser.add_argument(
        "--validation-source-training-steps",
        type=int,
        help="Optimization steps used by the replayed candidate.",
    )
    parser.add_argument(
        "--validation-source-learning-rate",
        type=float,
        help="Learning rate used by the replayed candidate.",
    )
    return parser.parse_args()


def _parse_layers(specification: str) -> list[int]:
    values: set[int] = set()
    for part in specification.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            if start > end:
                raise ValueError(f"Invalid layer range: {part}")
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    result = sorted(values)
    if not result or result[0] < 0:
        raise ValueError(f"Invalid STOP-decision layers: {specification!r}")
    return result


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported base checkpoint: {path}")
    for key in ("trainable_state_dict", "model_state_dict", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise KeyError(f"Base checkpoint contains no tensor state dict: {path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _split_by_scene(dataset: Any, fraction: float, seed: int) -> tuple[Any, Any]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("holdout_scene_fraction must be in (0, 1)")
    rollout_scene_ids = getattr(dataset, "sample_scene_ids", None)
    if rollout_scene_ids is not None:
        scenes_to_indices: dict[str, list[int]] = {}
        for index, scene_id in enumerate(rollout_scene_ids):
            scenes_to_indices.setdefault(str(scene_id), []).append(index)
        scenes = sorted(scenes_to_indices)
        if len(scenes) < 2:
            raise RuntimeError("STOP-decision training requires at least two scenes")
        ordered = sorted(
            scenes,
            key=lambda scene_id: hashlib.sha256(
                f"{seed}:{scene_id}".encode()
            ).digest(),
        )
        val_count = min(len(scenes) - 1, max(1, round(len(scenes) * fraction)))
        val_scenes = set(ordered[:val_count])
        train_scenes = set(ordered[val_count:])
        train_indices = [
            index
            for scene_id in sorted(train_scenes)
            for index in scenes_to_indices[scene_id]
        ]
        val_indices = [
            index
            for scene_id in sorted(val_scenes)
            for index in scenes_to_indices[scene_id]
        ]
        train = dataset.subset_by_indices(train_indices)
        val = dataset.subset_by_indices(val_indices)
        if set(train.sample_scene_ids) & set(val.sample_scene_ids):
            raise RuntimeError("STOP-decision train/validation rollout scenes overlap")
        return train, val

    active = sorted({int(clip_idx) for clip_idx, _frame in dataset.sample_index})
    scenes_to_clips: dict[str, set[int]] = {}
    for clip_idx in active:
        scene_id = Path(dataset.clips[clip_idx]).parent.name
        scenes_to_clips.setdefault(scene_id, set()).add(clip_idx)
    scenes = sorted(scenes_to_clips)
    if len(scenes) < 2:
        raise RuntimeError("STOP-decision training requires at least two scenes")
    ordered = sorted(
        scenes,
        key=lambda scene_id: hashlib.sha256(f"{seed}:{scene_id}".encode()).digest(),
    )
    val_count = min(len(scenes) - 1, max(1, round(len(scenes) * fraction)))
    val_scenes = set(ordered[:val_count])
    train_scenes = set(ordered[val_count:])
    val_clips = set().union(*(scenes_to_clips[scene] for scene in val_scenes))
    train_clips = set().union(*(scenes_to_clips[scene] for scene in train_scenes))
    train = dataset.subset_by_clip_indices(train_clips)
    val = dataset.subset_by_clip_indices(val_clips)
    if {
        int(clip_idx) for clip_idx, _frame in train.sample_index
    } & {int(clip_idx) for clip_idx, _frame in val.sample_index}:
        raise RuntimeError("STOP-decision train/validation clips overlap")
    train_scene_ids = {Path(train.clips[idx]).parent.name for idx, _ in train.sample_index}
    val_scene_ids = {Path(val.clips[idx]).parent.name for idx, _ in val.sample_index}
    if train_scene_ids & val_scene_ids:
        raise RuntimeError("STOP-decision train/validation scenes overlap")
    return train, val


def _stop_sample_indices(dataset: Any) -> tuple[list[int], list[int]]:
    rollout_targets = getattr(dataset, "targets", None)
    if rollout_targets is not None:
        positive = [
            index for index, target in enumerate(rollout_targets) if int(target) == 1
        ]
        negative = [
            index for index, target in enumerate(rollout_targets) if int(target) == 0
        ]
        if not positive or not negative:
            raise RuntimeError(
                "Balanced STOP rollout training requires positive and negative "
                f"samples: positive={len(positive)} negative={len(negative)}"
            )
        return positive, negative

    overrides = getattr(dataset, "_system2_sft_kind_override", None)
    if not isinstance(overrides, dict):
        raise RuntimeError("STOP-decision dataset has no SFT-kind metadata")
    positive = sorted(
        int(index)
        for index, kind in overrides.items()
        if str(kind).lower() == "stop"
    )
    positive_set = set(positive)
    negative = [index for index in range(len(dataset)) if index not in positive_set]
    if not positive or not negative:
        raise RuntimeError(
            "Balanced STOP training requires positive and negative samples: "
            f"positive={len(positive)} negative={len(negative)}"
        )
    return positive, negative


def _rollout_policy_counts(dataset: Any) -> dict[str, int]:
    targets = getattr(dataset, "targets", None)
    original_terminals = getattr(dataset, "original_terminals", None)
    if targets is None or original_terminals is None:
        raise RuntimeError("Rollout policy metadata is unavailable")
    if len(targets) != len(original_terminals):
        raise RuntimeError(
            "Rollout target/original-terminal metadata length mismatch: "
            f"targets={len(targets)} terminals={len(original_terminals)}"
        )
    return {
        "add_positive": sum(
            int(target) == 1 and not bool(original_terminal)
            for target, original_terminal in zip(targets, original_terminals)
        ),
        "regular_negative": sum(
            int(target) == 0 and not bool(original_terminal)
            for target, original_terminal in zip(targets, original_terminals)
        ),
        "false_stop_negative": sum(
            int(target) == 0 and bool(original_terminal)
            for target, original_terminal in zip(targets, original_terminals)
        ),
        "original_correct_stop": sum(
            int(target) == 1 and bool(original_terminal)
            for target, original_terminal in zip(targets, original_terminals)
        ),
    }


def _require_rollout_policy_coverage(dataset: Any, *, split_name: str) -> dict[str, int]:
    counts = _rollout_policy_counts(dataset)
    required = ("add_positive", "regular_negative", "false_stop_negative")
    missing = [name for name in required if counts[name] <= 0]
    if missing:
        raise RuntimeError(
            f"STOP-decision {split_name} split lacks policy calibration roles "
            f"{missing}: counts={counts}"
        )
    return counts


class _BalancedStopBatchSampler(Sampler[list[int]]):
    """Yield deterministic batches containing both STOP and non-STOP rows."""

    def __init__(
        self,
        positive_indices: list[int],
        negative_indices: list[int],
        batch_size: int,
        seed: int,
        *,
        priority_negative_indices: list[int] | None = None,
        priority_negative_fraction: float = 0.25,
        mined_negative_indices: list[int] | None = None,
        mined_negative_fraction: float = 0.0,
    ) -> None:
        if batch_size < 2:
            raise ValueError("Balanced STOP batches require batch_size >= 2")
        self.positive_indices = tuple(int(index) for index in positive_indices)
        self.negative_indices = tuple(int(index) for index in negative_indices)
        priority_set = {
            int(index) for index in (priority_negative_indices or [])
        }
        mined_set = {int(index) for index in (mined_negative_indices or [])}
        negative_set = set(self.negative_indices)
        if not priority_set.issubset(negative_set):
            raise ValueError("Priority STOP negatives must be a subset of negatives")
        if not mined_set.issubset(negative_set):
            raise ValueError("Mined STOP negatives must be a subset of negatives")
        if priority_set & mined_set:
            raise ValueError("Recorded false STOP and mined negative pools must be disjoint")
        if priority_set and not 0.0 < priority_negative_fraction < 1.0:
            raise ValueError(
                "priority_negative_fraction must be in (0, 1) when enabled"
            )
        if mined_set and not 0.0 < mined_negative_fraction < 1.0:
            raise ValueError(
                "mined_negative_fraction must be in (0, 1) when enabled"
            )
        self.priority_negative_indices = tuple(sorted(priority_set))
        self.mined_negative_indices = tuple(sorted(mined_set))
        self.regular_negative_indices = tuple(
            index
            for index in self.negative_indices
            if index not in priority_set and index not in mined_set
        )
        self.priority_negative_fraction = (
            float(priority_negative_fraction) if priority_set else 0.0
        )
        self.mined_negative_fraction = (
            float(mined_negative_fraction) if mined_set else 0.0
        )
        if self.regular_negative_indices and (
            self.priority_negative_fraction + self.mined_negative_fraction >= 1.0
        ):
            raise ValueError(
                "Priority and mined negative fractions must sum to less than 1"
            )
        self.positive_per_batch = max(1, batch_size // 2)
        self.negative_per_batch = batch_size - self.positive_per_batch
        self.seed = int(seed)
        self.epoch = 0
        minimum_batches = max(
            math.ceil(len(self.positive_indices) / self.positive_per_batch),
            1,
        )
        while True:
            slot_counts = self._negative_slot_counts(minimum_batches)
            if (
                slot_counts["priority"] >= len(self.priority_negative_indices)
                and slot_counts["mined"] >= len(self.mined_negative_indices)
                and slot_counts["regular"] >= len(self.regular_negative_indices)
            ):
                break
            minimum_batches += 1
        self.batch_count = minimum_batches

    def _negative_slot_counts(self, batch_count: int) -> dict[str, int]:
        total = int(batch_count) * self.negative_per_batch
        active = {
            "priority": bool(self.priority_negative_indices),
            "mined": bool(self.mined_negative_indices),
            "regular": bool(self.regular_negative_indices),
        }
        if sum(active.values()) == 1:
            only = next(name for name, enabled in active.items() if enabled)
            return {name: total if name == only else 0 for name in active}

        if active["regular"]:
            priority = (
                math.ceil(total * self.priority_negative_fraction)
                if active["priority"]
                else 0
            )
            mined = (
                math.ceil(total * self.mined_negative_fraction)
                if active["mined"]
                else 0
            )
            if priority + mined > total:
                mined = max(0, total - priority)
            return {
                "priority": priority,
                "mined": mined,
                "regular": total - priority - mined,
            }

        total_weight = (
            self.priority_negative_fraction + self.mined_negative_fraction
        )
        priority = (
            math.ceil(total * self.priority_negative_fraction / total_weight)
            if active["priority"]
            else 0
        )
        return {
            "priority": priority,
            "mined": total - priority,
            "regular": 0,
        }

    @staticmethod
    def _shuffled_cycle(values: tuple[int, ...], generator: torch.Generator):
        while True:
            for position in torch.randperm(len(values), generator=generator).tolist():
                yield values[position]

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        positives = self._shuffled_cycle(self.positive_indices, generator)
        regular_negatives = (
            self._shuffled_cycle(self.regular_negative_indices, generator)
            if self.regular_negative_indices
            else None
        )
        priority_negatives = (
            self._shuffled_cycle(self.priority_negative_indices, generator)
            if self.priority_negative_indices
            else None
        )
        mined_negatives = (
            self._shuffled_cycle(self.mined_negative_indices, generator)
            if self.mined_negative_indices
            else None
        )
        slot_counts = self._negative_slot_counts(self.batch_count)
        category_schedule = [
            *("priority" for _ in range(slot_counts["priority"])),
            *("mined" for _ in range(slot_counts["mined"])),
            *("regular" for _ in range(slot_counts["regular"])),
        ]
        schedule_order = torch.randperm(
            len(category_schedule), generator=generator
        ).tolist()
        category_schedule = [category_schedule[index] for index in schedule_order]
        for batch_index in range(self.batch_count):
            batch = [next(positives) for _ in range(self.positive_per_batch)]
            start = batch_index * self.negative_per_batch
            categories = category_schedule[start : start + self.negative_per_batch]
            sources = {
                "priority": priority_negatives,
                "mined": mined_negatives,
                "regular": regular_negatives,
            }
            batch.extend(next(sources[category]) for category in categories)
            order = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[index] for index in order]

    def __len__(self) -> int:
        return self.batch_count


def _move_inputs(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for name, value in inputs.items()
    }


def _stop_forward(
    integration: Any,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = _move_inputs(batch["pano_inputs"], device)
    positions = batch["system2_stop_predictor_position"].to(
        device=device,
        dtype=torch.long,
        non_blocking=True,
    )
    targets = batch["system2_stop_target"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    hidden, _vision, _num_images, _traj, _lm = integration._forward_model_inputs(
        inputs,
        return_hidden_states=True,
        skip_lm_head=True,
        return_last_hidden_state_only=True,
        extract_vision_hidden_states=False,
    )
    if hidden is None:
        raise RuntimeError("STOP-decision forward returned no sequence hidden states")
    # Deployment computes the six-way structured-view probabilities in FP32.
    # Keep training/calibration on the same numerical contract so BF16
    # saturation cannot quantize add/veto thresholds near zero or one.
    class_logits = integration.structured_view_class_logits(hidden, positions).float()
    binary_logits = class_logits[:, 0] - torch.logsumexp(class_logits[:, 1:], dim=-1)
    probabilities = torch.sigmoid(binary_logits)
    return binary_logits, probabilities, targets


def _loss(
    binary_logits: torch.Tensor,
    targets: torch.Tensor,
    negative_weight: float,
    *,
    bce_weight: float = 1.0,
    ranking_weight: float = 0.0,
    ranking_margin: float = 0.0,
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(
        binary_logits.float(),
        targets.float(),
        reduction="none",
    )
    weights = torch.where(
        targets >= 0.5,
        torch.ones_like(targets),
        torch.full_like(targets, float(negative_weight)),
    )
    bce = (losses * weights).sum() / weights.sum().clamp_min(1.0)
    ranking = binary_logits.float().sum() * 0.0
    positive_logits = binary_logits[targets >= 0.5].float()
    negative_logits = binary_logits[targets < 0.5].float()
    if ranking_weight > 0 and positive_logits.numel() and negative_logits.numel():
        pairwise_margin = (
            float(ranking_margin)
            - positive_logits[:, None]
            + negative_logits[None, :]
        )
        ranking = F.softplus(pairwise_margin).mean()
    return float(bce_weight) * bce + float(ranking_weight) * ranking


def _confusion(targets: torch.Tensor, probabilities: torch.Tensor, threshold: float) -> dict[str, int]:
    truth = targets >= 0.5
    prediction = probabilities >= threshold
    return {
        "tp": int((prediction & truth).sum().item()),
        "fp": int((prediction & ~truth).sum().item()),
        "tn": int((~prediction & ~truth).sum().item()),
        "fn": int((~prediction & truth).sum().item()),
    }


def _roc_auc(targets: torch.Tensor, probabilities: torch.Tensor) -> float:
    """Compute tie-aware binary ROC-AUC without an sklearn dependency."""
    truth = (targets >= 0.5).to(dtype=torch.bool).flatten()
    scores = probabilities.float().flatten()
    positive_count = int(truth.sum().item())
    negative_count = int((~truth).sum().item())
    if not positive_count or not negative_count:
        raise RuntimeError("ROC-AUC requires both positive and negative samples")

    sorted_scores, order = torch.sort(scores)
    sorted_truth = truth[order]
    positive_rank_sum = 0.0
    start = 0
    sample_count = int(sorted_scores.numel())
    while start < sample_count:
        end = start + 1
        while end < sample_count and bool(sorted_scores[end] == sorted_scores[start]):
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        positive_rank_sum += average_rank * int(sorted_truth[start:end].sum().item())
        start = end
    return float(
        (
            positive_rank_sum
            - positive_count * (positive_count + 1) / 2.0
        )
        / (positive_count * negative_count)
    )


def _probability_quantiles(values: torch.Tensor) -> dict[str, float]:
    quantiles = (0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0)
    values = values.float().flatten()
    if not values.numel():
        raise RuntimeError("Probability summary requires at least one sample")
    result = torch.quantile(values, torch.tensor(quantiles, dtype=values.dtype))
    return {
        f"q{int(round(quantile * 100)):02d}": float(value.item())
        for quantile, value in zip(quantiles, result)
    }


def _validation_score_audit_rows(
    records: list[dict[str, Any]],
    probabilities: torch.Tensor,
) -> list[dict[str, Any]]:
    scores = probabilities.detach().float().cpu().flatten()
    if len(records) != int(scores.numel()):
        raise RuntimeError(
            "Validation score audit length mismatch: "
            f"records={len(records)} scores={scores.numel()}"
        )
    rows = []
    for record, raw_score in zip(records, scores.tolist()):
        score = float(raw_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError(
                f"Invalid validation STOP probability for {record.get('key')}: {score}"
            )
        target = int(record["stop_target"])
        original_terminal = bool(record["original_terminal"])
        if target == 1 and not original_terminal:
            policy_role = "add_positive"
        elif target == 0 and not original_terminal:
            policy_role = "regular_negative"
        elif target == 0:
            policy_role = "false_stop_negative"
        else:
            policy_role = "original_correct_stop"
        original_scores = record.get("system2_decision_scores") or {}
        original_probabilities = original_scores.get("class_probabilities") or {}
        original_stop_probability = original_probabilities.get("stop")
        rows.append(
            {
                "key": str(record["key"]),
                "scene_id": str(record["scene_id"]),
                "episode_id": int(record["episode_id"]),
                "system2_call_index": int(record["system2_call_index"]),
                "stop_target": target,
                "original_terminal": original_terminal,
                "policy_role": policy_role,
                "distance_to_goal_m": float(record["distance_to_goal_m"]),
                "stop_probability": score,
                "original_qwen_stop_probability": (
                    float(original_stop_probability)
                    if original_stop_probability is not None
                    else None
                ),
                "original_output": str(record.get("original_output") or ""),
                "effective_output": str(record.get("effective_output") or ""),
                "oracle_recovery_active": bool(
                    record.get("oracle_recovery_active", False)
                ),
            }
        )
    rows.sort(key=lambda row: (-row["stop_probability"], row["key"]))
    return rows


def _atomic_jsonl_write(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _save_score_audit(
    output_dir: Path,
    dataset: System2StopMultimodalDataset,
    probabilities: torch.Tensor,
    *,
    split_name: str,
) -> dict[str, Any]:
    if split_name not in {"validation", "train"}:
        raise ValueError(f"Unsupported STOP score-audit split: {split_name!r}")
    audit_path = output_dir / f"{split_name}_scores.jsonl"
    audit_rows = _validation_score_audit_rows(dataset.records, probabilities)
    _atomic_jsonl_write(audit_rows, audit_path)
    return {
        "path": str(audit_path),
        "rows": len(audit_rows),
        "sha256": _file_sha256(audit_path),
    }


def _save_validation_score_audit(
    output_dir: Path,
    dataset: System2StopMultimodalDataset,
    probabilities: torch.Tensor,
) -> dict[str, Any]:
    return _save_score_audit(
        output_dir,
        dataset,
        probabilities,
        split_name="validation",
    )


def _load_mined_regular_negative_indices(
    path: Path,
    train_dataset: System2StopMultimodalDataset,
    count: int,
) -> tuple[list[int], dict[str, Any]]:
    audit_path = path.expanduser().resolve()
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing train score audit: {audit_path}")
    rows = []
    seen_keys: set[str] = set()
    with audit_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("key") or "")
            score = float(row.get("stop_probability", float("nan")))
            if not key or key in seen_keys:
                raise RuntimeError(
                    f"Duplicate or missing score-audit key at line {line_number}"
                )
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise RuntimeError(
                    f"Invalid score-audit probability at line {line_number}: {score}"
                )
            seen_keys.add(key)
            rows.append(row)

    train_key_to_index = {
        str(record["key"]): index
        for index, record in enumerate(train_dataset.records)
    }
    if seen_keys != set(train_key_to_index):
        raise RuntimeError(
            "Mined score audit must contain exactly the scene-held-out train split: "
            f"audit={len(seen_keys)} train={len(train_key_to_index)} "
            f"missing={sorted(set(train_key_to_index) - seen_keys)[:3]} "
            f"unexpected={sorted(seen_keys - set(train_key_to_index))[:3]}"
        )
    candidates = [
        row
        for row in rows
        if row.get("policy_role") == "regular_negative"
        and int(row.get("stop_target", -1)) == 0
        and row.get("original_terminal") is False
    ]
    candidates.sort(
        key=lambda row: (-float(row["stop_probability"]), str(row["key"]))
    )
    if len(candidates) < count:
        raise RuntimeError(
            f"Requested {count} mined regular negatives, found {len(candidates)}"
        )
    selected = candidates[:count]
    indices = [train_key_to_index[str(row["key"])] for row in selected]
    return indices, {
        "path": str(audit_path),
        "sha256": _file_sha256(audit_path),
        "audit_rows": len(rows),
        "regular_negative_rows": len(candidates),
        "selected_rows": len(selected),
        "selected_min_probability": min(
            float(row["stop_probability"]) for row in selected
        ),
        "selected_max_probability": max(
            float(row["stop_probability"]) for row in selected
        ),
        "selected_keys_sha256": hashlib.sha256(
            "\n".join(sorted(str(row["key"]) for row in selected)).encode()
        ).hexdigest(),
    }


def _threshold_metrics(
    targets: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    veto_reference_probabilities: torch.Tensor | None = None,
    original_terminal_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    scores = probabilities.detach().float().cpu().flatten().clamp(0.0, 1.0)
    targets = targets.detach().float().cpu().flatten()
    truth = targets >= 0.5
    if original_terminal_mask is not None and veto_reference_probabilities is not None:
        raise ValueError(
            "Pass either recorded original terminals or probability references, not both"
        )
    policy_aware = original_terminal_mask is not None
    if policy_aware:
        original_terminal = original_terminal_mask.detach().bool().cpu().flatten()
        if original_terminal.shape != scores.shape:
            raise ValueError(
                "Original-terminal mask must match validation probabilities: "
                f"reference={tuple(original_terminal.shape)} scores={tuple(scores.shape)}"
            )
        add_positive_mask = truth & ~original_terminal
        add_negative_mask = ~truth & ~original_terminal
        # Any goal-positive state must survive veto, while veto rejection is only
        # meaningful for states where the original generator actually stopped.
        veto_positive_mask = truth
        veto_negative_mask = ~truth & original_terminal
        veto_reference_name = "recorded_original_terminal"
    else:
        if veto_reference_probabilities is None:
            veto_reference_probabilities = probabilities
        reference_scores = (
            veto_reference_probabilities.detach().float().cpu().flatten()
        )
        if reference_scores.shape != scores.shape:
            raise ValueError(
                "Veto reference probabilities must match validation probabilities: "
                f"reference={tuple(reference_scores.shape)} scores={tuple(scores.shape)}"
            )
        add_positive_mask = truth
        add_negative_mask = ~truth
        veto_positive_mask = truth & (reference_scores >= 0.5)
        veto_negative_mask = ~truth
        veto_reference_name = "baseline_system2_stop_positive"

    unique_scores = torch.unique(scores, sorted=True)
    just_above = torch.nextafter(
        unique_scores,
        torch.full_like(unique_scores, float("inf")),
    ).clamp_max(1.0)
    candidates = torch.unique(
        torch.cat(
            [
                torch.tensor([0.0, 1.0], dtype=scores.dtype),
                unique_scores,
                just_above,
            ]
        ),
        sorted=True,
    )
    records = []
    for threshold in candidates.tolist():
        prediction = scores >= float(threshold)
        counts = {
            "tp": int((prediction & add_positive_mask).sum().item()),
            "fp": int((prediction & add_negative_mask).sum().item()),
            "tn": int((~prediction & add_negative_mask).sum().item()),
            "fn": int((~prediction & add_positive_mask).sum().item()),
        }
        positives = int(add_positive_mask.sum().item())
        negatives = int(add_negative_mask.sum().item())
        records.append(
            {
                **counts,
                "threshold": float(threshold),
                "recall": counts["tp"] / max(positives, 1),
                "false_positive_rate": counts["fp"] / max(negatives, 1),
                "negative_rejection_rate": counts["tn"] / max(negatives, 1),
            }
        )

    add_candidates = [
        record
        for record in records
        if record["false_positive_rate"] <= MAX_ADD_FALSE_POSITIVE_RATE
    ]
    add = max(
        add_candidates,
        # With equal held-out recall/FPR, use the highest threshold. This
        # maximizes the deployment margin between the strongest regular
        # negative and the weakest accepted STOP-positive state.
        key=lambda record: (record["recall"], record["threshold"]),
        default=records[-1],
    )

    veto_records = []
    for threshold in candidates.tolist():
        if float(threshold) >= float(add["threshold"]):
            continue
        prediction = scores >= float(threshold)
        tp = int((prediction & veto_positive_mask).sum().item())
        fn = int((~prediction & veto_positive_mask).sum().item())
        fp = int((prediction & veto_negative_mask).sum().item())
        tn = int((~prediction & veto_negative_mask).sum().item())
        veto_records.append(
            {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "threshold": float(threshold),
                "recall": tp / max(tp + fn, 1),
                "false_positive_rate": fp / max(fp + tn, 1),
                "negative_rejection_rate": tn / max(fp + tn, 1),
                "reference_positive_count": int(veto_positive_mask.sum().item()),
                "reference_negative_count": int(veto_negative_mask.sum().item()),
                "reference": veto_reference_name,
            }
        )
    veto_candidates = [
        record
        for record in veto_records
        if record["recall"] >= MIN_VETO_RECALL
        and record["negative_rejection_rate"] >= MIN_VETO_NEGATIVE_REJECTION
    ]
    veto = max(
        veto_candidates,
        key=lambda record: (record["threshold"], -record["false_positive_rate"]),
        default=max(
            (
                record
                for record in veto_records
                if record["recall"] >= MIN_VETO_RECALL
            ),
            key=lambda record: (record["threshold"], -record["false_positive_rate"]),
            default={
                "tp": 0,
                "fp": int(veto_negative_mask.sum().item()),
                "tn": 0,
                "fn": int(veto_positive_mask.sum().item()),
                "threshold": 0.0,
                "recall": 0.0,
                "false_positive_rate": 1.0,
                "negative_rejection_rate": 0.0,
                "reference_positive_count": int(veto_positive_mask.sum().item()),
                "reference_negative_count": int(veto_negative_mask.sum().item()),
                "reference": veto_reference_name,
            },
        ),
    )

    roc_auc = _roc_auc(targets, scores)
    quality_violations = []
    if not int(veto_positive_mask.sum().item()):
        quality_violations.append(
            "validation contains no positive sample for veto retention"
        )
    elif policy_aware and not int(veto_negative_mask.sum().item()):
        quality_violations.append(
            "validation contains no originally-terminal negative sample for veto calibration"
        )
    elif not veto_candidates:
        quality_violations.append(
            "no veto threshold reaches "
            f"recall>={MIN_VETO_RECALL:.2f} with negative rejection>="
            f"{MIN_VETO_NEGATIVE_REJECTION:.2f} and veto<add"
        )
    if policy_aware and not int(add_positive_mask.sum().item()):
        quality_violations.append(
            "validation contains no originally-nonterminal positive sample for STOP addition"
        )
    elif policy_aware and not int(add_negative_mask.sum().item()):
        quality_violations.append(
            "validation contains no originally-nonterminal negative sample for add calibration"
        )
    elif not add_candidates or add["recall"] < MIN_ADD_RECALL:
        quality_violations.append(
            "no add threshold reaches "
            f"recall>={MIN_ADD_RECALL:.2f} with FPR<="
            f"{MAX_ADD_FALSE_POSITIVE_RATE:.2f}"
        )
    if roc_auc < MIN_ROC_AUC:
        quality_violations.append(
            f"ROC-AUC {roc_auc:.4f} is below {MIN_ROC_AUC:.2f}"
        )
    return {
        "policy_kind": ADD_AND_VETO_POLICY,
        "add_enabled": True,
        "add_stop_threshold": add["threshold"],
        "veto_stop_threshold": veto["threshold"],
        "add": add,
        "veto": veto,
        "at_0_5": min(
            records,
            key=lambda record: abs(record["threshold"] - 0.5),
        ),
        "positive_count": int(truth.sum().item()),
        "negative_count": int((~truth).sum().item()),
        "add_reference_positive_count": int(add_positive_mask.sum().item()),
        "add_reference_negative_count": int(add_negative_mask.sum().item()),
        "veto_reference_positive_count": int(veto_positive_mask.sum().item()),
        "veto_reference_negative_count": int(veto_negative_mask.sum().item()),
        "roc_auc": roc_auc,
        "probability_quantiles": {
            "positive": _probability_quantiles(scores[truth]),
            "negative": _probability_quantiles(scores[~truth]),
        },
        "quality_requirements": {
            "min_veto_recall": MIN_VETO_RECALL,
            "min_veto_negative_rejection": MIN_VETO_NEGATIVE_REJECTION,
            "max_add_false_positive_rate": MAX_ADD_FALSE_POSITIVE_RATE,
            "min_add_recall": MIN_ADD_RECALL,
            "min_roc_auc": MIN_ROC_AUC,
        },
        "quality_passed": not quality_violations,
        "quality_violations": quality_violations,
    }


def _veto_only_threshold_metrics(
    threshold_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build an honest deployment contract for a policy that cannot add STOP."""
    veto = threshold_metrics.get("veto")
    if not isinstance(veto, dict):
        raise RuntimeError("Veto-only export requires calibrated veto metrics")

    quality_violations = []
    if int(threshold_metrics.get("veto_reference_positive_count", 0) or 0) <= 0:
        quality_violations.append("veto calibration has no positive reference samples")
    if int(threshold_metrics.get("veto_reference_negative_count", 0) or 0) <= 0:
        quality_violations.append("veto calibration has no false-STOP reference samples")
    if float(veto.get("recall", 0.0)) < MIN_VETO_RECALL:
        quality_violations.append(
            f"veto recall is below {MIN_VETO_RECALL:.2f}"
        )
    if (
        float(veto.get("negative_rejection_rate", 0.0))
        < MIN_VETO_NEGATIVE_REJECTION
    ):
        quality_violations.append(
            "veto false-STOP rejection is below "
            f"{MIN_VETO_NEGATIVE_REJECTION:.2f}"
        )
    if float(threshold_metrics.get("roc_auc", 0.0)) < MIN_ROC_AUC:
        quality_violations.append(f"ROC-AUC is below {MIN_ROC_AUC:.2f}")

    add_positive_count = int(
        threshold_metrics.get("add_reference_positive_count", 0) or 0
    )
    add_negative_count = int(
        threshold_metrics.get("add_reference_negative_count", 0) or 0
    )
    result = dict(threshold_metrics)
    result.update(
        {
            "policy_kind": VETO_ONLY_POLICY,
            "add_enabled": False,
            # This value remains part of the hysteresis schema, but inference also
            # has an explicit control-flow guard. A score of exactly 1 cannot add.
            "add_stop_threshold": 1.0,
            "source_add": threshold_metrics.get("add"),
            "source_quality_passed": bool(
                threshold_metrics.get("quality_passed", False)
            ),
            "source_quality_violations": list(
                threshold_metrics.get("quality_violations") or []
            ),
            "add": {
                "enabled": False,
                "tp": 0,
                "fp": 0,
                "tn": add_negative_count,
                "fn": add_positive_count,
                "threshold": 1.0,
                "recall": 0.0,
                "false_positive_rate": 0.0,
                "negative_rejection_rate": 1.0,
                "reference": "disabled_by_policy",
            },
            "quality_passed": not quality_violations,
            "quality_violations": quality_violations,
        }
    )
    requirements = dict(result.get("quality_requirements") or {})
    requirements.update(
        {
            "policy_kind": VETO_ONLY_POLICY,
            "add_enabled": False,
            "min_add_recall": None,
        }
    )
    result["quality_requirements"] = requirements
    return result


@torch.no_grad()
def _validate(
    integration: Any,
    loader: DataLoader,
    device: torch.device,
    negative_weight: float,
    *,
    veto_reference_probabilities: torch.Tensor | None = None,
    original_terminal_mask: torch.Tensor | None = None,
) -> tuple[float, dict[str, Any], torch.Tensor, torch.Tensor]:
    losses: list[float] = []
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits, batch_probabilities, batch_targets = _stop_forward(
                integration, batch, device
            )
            batch_loss = _loss(logits, batch_targets, negative_weight)
        losses.append(float(batch_loss.item()))
        probabilities.append(batch_probabilities.detach().cpu())
        targets.append(batch_targets.detach().cpu())
    if not losses:
        raise RuntimeError("STOP-decision validation produced no batches")
    all_targets = torch.cat(targets)
    all_probabilities = torch.cat(probabilities)
    return (
        float(np.mean(losses)),
        _threshold_metrics(
            all_targets,
            all_probabilities,
            veto_reference_probabilities=veto_reference_probabilities,
            original_terminal_mask=original_terminal_mask,
        ),
        all_probabilities,
        all_targets,
    )


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.max_steps <= 0 or args.batch_size <= 0 or args.grad_accum_steps <= 0:
        raise ValueError("max_steps, batch_size, and grad_accum_steps must be positive")
    if args.dry_run and args.grad_accum_steps != 1:
        raise ValueError("dry-run requires grad_accum_steps=1 for a real optimizer update")
    if args.dry_run and args.validation_adapter_checkpoint is not None:
        raise ValueError("dry-run cannot be combined with validation replay")
    if args.export_veto_only_checkpoint and args.validation_adapter_checkpoint is None:
        raise ValueError(
            "--export-veto-only-checkpoint requires --validation-adapter-checkpoint"
        )
    if args.export_veto_only_checkpoint and args.replay_split != "validation":
        raise ValueError("veto-only export requires --replay-split validation")
    if args.export_veto_only_checkpoint and (
        args.validation_source_training_steps is None
        or args.validation_source_training_steps <= 0
        or args.validation_source_learning_rate is None
        or args.validation_source_learning_rate <= 0.0
    ):
        raise ValueError(
            "veto-only export requires positive --validation-source-training-steps "
            "and --validation-source-learning-rate"
        )
    if args.max_steps % args.grad_accum_steps:
        raise ValueError("max_steps must be divisible by grad_accum_steps")
    if args.learning_rate <= 0 or args.negative_loss_weight <= 0:
        raise ValueError("learning_rate and negative_loss_weight must be positive")
    if not 0.0 <= args.terminal_negative_sampling_fraction < 1.0:
        raise ValueError("terminal_negative_sampling_fraction must be in [0, 1)")
    if (args.mined_train_scores_jsonl is None) != (
        args.mined_regular_negative_count == 0
    ):
        raise ValueError(
            "--mined-train-scores-jsonl and a positive "
            "--mined-regular-negative-count must be provided together"
        )
    if args.mined_train_scores_jsonl is not None and not (
        args.mined_regular_negative_count > 0
        and 0.0 < args.mined_regular_negative_fraction < 1.0
        and args.terminal_negative_sampling_fraction
        + args.mined_regular_negative_fraction
        < 1.0
    ):
        raise ValueError(
            "Mined-negative count/fraction must be positive and terminal+mined "
            "fractions must sum to less than 1"
        )
    if args.bce_loss_weight < 0 or args.ranking_loss_weight < 0:
        raise ValueError("BCE and ranking loss weights must be non-negative")
    if args.bce_loss_weight + args.ranking_loss_weight <= 0:
        raise ValueError("At least one STOP-decision loss weight must be positive")
    if args.ranking_margin < 0:
        raise ValueError("ranking_margin must be non-negative")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log"),
        ],
    )
    _set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    cfg = load_config(args.config, validate=False)
    if os.environ.get("PANORAMIC_DATA_ROOT"):
        cfg.setdefault("data", {})["root"] = os.environ["PANORAMIC_DATA_ROOT"]
    model_path = (
        os.environ.get("INTERNNAV_MODEL_PATH")
        or cfg.get("paths", {}).get("internnav_model_path")
        or cfg.get("model", {}).get("llm", {}).get("model_path")
    )
    if not model_path:
        raise ValueError("INTERNNAV_MODEL_PATH or model.llm.model_path is required")
    cfg.setdefault("model", {}).setdefault("llm", {})["model_path"] = model_path
    cfg["model"]["llm"]["gradient_checkpointing"] = True
    cfg["model"]["llm"]["lora_dropout"] = 0.0
    cfg["model"].setdefault("heatmap", {})["enable"] = False
    cfg["model"].setdefault("action_head", {})["enable"] = False
    cfg["model"]["action_head"].setdefault("nextdit", {})["enabled"] = False
    cfg["model"].setdefault("stop_head", {})["enabled"] = False
    trajectory_cfg = cfg.setdefault("data", {}).setdefault("trajectory", {})
    trajectory_cfg["max_clips"] = max(0, int(args.max_clips))
    trajectory_cfg["require_sft_target"] = True
    trajectory_cfg["load_traj_images"] = False
    trajectory_cfg["load_history_frames"] = False
    trajectory_cfg["load_history_heatmap"] = False

    layers = _parse_layers(args.layers)
    target_modules = [
        value.strip() for value in args.target_modules.split(",") if value.strip()
    ]
    ensure_transformers_runtime_compat(
        model_path=model_path,
        requested_backbone_type=cfg["model"]["llm"].get("backbone_type", "qwen2_5_vl"),
        requested_attn_implementation=cfg["model"]["llm"].get(
            "attn_implementation", "sdpa"
        ),
        logger=LOGGER,
    )
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    collator = PanoramicTokenizedCollator(
        processor,
        n_traj_query=0,
        sft_mode=True,
        sft_include_turns=True,
        sft_include_forward=False,
        sft_protocol=str(trajectory_cfg.get("system2_sft_protocol", "direct")),
        structured_pano_output=True,
        build_sft_labels=True,
        build_stop_head_targets=True,
        max_seq_length=int(cfg["model"]["llm"].get("max_seq_length", 8192)),
        include_heatmap_targets=False,
        include_history_rel_poses=False,
        retain_raw_panoramic_views=False,
        compute_pano_text_anchor_positions=False,
        heatmap_layout=False,
    )

    rollout_roots = [Path(path).expanduser().resolve() for path in args.rollout_root]
    if args.mined_train_scores_jsonl is not None and not rollout_roots:
        raise ValueError("Mined STOP negatives require on-policy --rollout-root data")
    if rollout_roots:
        if args.max_clips:
            raise ValueError("--max-clips cannot be combined with --rollout-root")
        LOGGER.info(
            "Building on-policy multimodal STOP dataset from %d rollout roots",
            len(rollout_roots),
        )
        dataset = System2StopMultimodalDataset(
            rollout_roots,
            image_size=tuple(int(value) for value in cfg["data"]["image_size"]),
        )
        dataset_source = "on_policy_train_rollout"
        dataset_unit_count = len(set(dataset.sample_scene_ids))
    else:
        LOGGER.info("Building STOP-decision dataset from %s", cfg["data"]["root"])
        dataset = build_dataset(cfg, split="train", load_history_heatmap=False)
        dataset_source = "native_train_clips"
        dataset_unit_count = len(dataset.clips)
    train_dataset, val_dataset = _split_by_scene(
        dataset, args.holdout_scene_fraction, args.seed
    )
    positive_indices, negative_indices = _stop_sample_indices(train_dataset)
    val_positive_indices, val_negative_indices = _stop_sample_indices(val_dataset)
    train_terminal_negative_indices = []
    train_mined_negative_indices = []
    mined_negative_contract = None
    val_terminal_negative_indices = []
    train_policy_counts = None
    val_policy_counts = None
    if rollout_roots:
        train_policy_counts = _require_rollout_policy_coverage(
            train_dataset, split_name="train"
        )
        val_policy_counts = _require_rollout_policy_coverage(
            val_dataset, split_name="validation"
        )
        train_terminal_negative_indices = [
            index
            for index, (target, original_terminal) in enumerate(
                zip(train_dataset.targets, train_dataset.original_terminals)
            )
            if int(target) == 0 and bool(original_terminal)
        ]
        val_terminal_negative_indices = [
            index
            for index, (target, original_terminal) in enumerate(
                zip(val_dataset.targets, val_dataset.original_terminals)
            )
            if int(target) == 0 and bool(original_terminal)
        ]
        if args.mined_train_scores_jsonl is not None:
            (
                train_mined_negative_indices,
                mined_negative_contract,
            ) = _load_mined_regular_negative_indices(
                args.mined_train_scores_jsonl,
                train_dataset,
                args.mined_regular_negative_count,
            )
            if set(train_mined_negative_indices) & set(
                train_terminal_negative_indices
            ):
                raise RuntimeError(
                    "Mined regular negatives overlap recorded false STOP negatives"
                )
    train_batch_sampler = _BalancedStopBatchSampler(
        positive_indices,
        negative_indices,
        args.batch_size,
        args.seed,
        priority_negative_indices=train_terminal_negative_indices,
        priority_negative_fraction=args.terminal_negative_sampling_fraction,
        mined_negative_indices=train_mined_negative_indices,
        mined_negative_fraction=args.mined_regular_negative_fraction,
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": collator,
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **{**loader_kwargs, "persistent_workers": False},
    )
    if not len(train_loader) or not len(val_loader):
        raise RuntimeError("STOP-decision train/validation DataLoader is empty")
    if train_mined_negative_indices and args.max_steps < len(train_batch_sampler):
        raise RuntimeError(
            "Hard-negative mining must cover every training pool at least once: "
            f"max_steps={args.max_steps} sampler_batches={len(train_batch_sampler)}"
        )
    LOGGER.info(
        "Dataset ready: source=%s train=%d val=%d units=%d train_stop=%d "
        "train_non_stop=%d train_false_stop=%d train_mined_negative=%d "
        "val_stop=%d val_non_stop=%d val_false_stop=%d "
        "priority_negative_fraction=%.3f mined_negative_fraction=%.3f "
        "sampler_batches=%d max_steps=%d",
        dataset_source,
        len(train_dataset),
        len(val_dataset),
        dataset_unit_count,
        len(positive_indices),
        len(negative_indices),
        len(train_terminal_negative_indices),
        len(train_mined_negative_indices),
        len(val_positive_indices),
        len(val_negative_indices),
        len(val_terminal_negative_indices),
        train_batch_sampler.priority_negative_fraction,
        train_batch_sampler.mined_negative_fraction,
        len(train_batch_sampler),
        args.max_steps,
    )
    if train_policy_counts is not None and val_policy_counts is not None:
        LOGGER.info(
            "Policy calibration coverage: train=%s validation=%s",
            train_policy_counts,
            val_policy_counts,
        )
    if rollout_roots:
        dataset_contract = {
            "source": dataset_source,
            "dataset_split": "train",
            "roots": [str(path) for path in rollout_roots],
            "labels_sha256": {
                str(path): _file_sha256(
                    path / "system2_stop_multimodal_examples.jsonl"
                )
                for path in rollout_roots
            },
            "scene_count": len(set(dataset.sample_scene_ids)),
            "train_scene_count": len(set(train_dataset.sample_scene_ids)),
            "val_scene_count": len(set(val_dataset.sample_scene_ids)),
        }
        validation_original_terminal_mask = torch.tensor(
            val_dataset.original_terminals,
            dtype=torch.bool,
        )
    else:
        dataset_contract = {
            "source": dataset_source,
            "dataset_split": "train",
            "clip_count": len(dataset.clips),
        }
        validation_original_terminal_mask = None

    model = build_model(cfg, device=str(device), verbose=True).to(device)
    integration = model.qwen2_5_vl
    integration._load_model()
    base_path = Path(args.base_checkpoint).expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"Missing base checkpoint: {base_path}")
    base_state = _checkpoint_state(base_path)
    matched = assert_complete_lora_checkpoint_match(
        model, base_state, checkpoint_path=str(base_path)
    )
    lora_state = extract_lora_checkpoint_state(base_state)
    _missing, _unexpected, loaded = _load_normalized_state_dict(model, lora_state)
    if matched != 224 or loaded != len(lora_state) or loaded != matched:
        raise RuntimeError(
            f"Base LoRA completeness failure: matched={matched} loaded={loaded} "
            f"checkpoint_lora={len(lora_state)}"
        )
    del base_state, lora_state
    original_fingerprint = integration.lora_adapter_fingerprint(
        DEFAULT_LORA_ADAPTER_NAME
    )
    model.requires_grad_(False)
    parameter_count = integration.add_stop_decision_adapter(
        adapter_name=STOP_DECISION_ADAPTER_NAME,
        rank=args.rank,
        alpha=args.alpha,
        layer_indices=layers,
        target_modules=target_modules,
    )
    integration.activate_lora_adapters(
        (DEFAULT_LORA_ADAPTER_NAME, STOP_DECISION_ADAPTER_NAME),
        trainable_adapters=(STOP_DECISION_ADAPTER_NAME,),
    )
    adapter_named_parameters = integration.lora_adapter_named_parameters(
        STOP_DECISION_ADAPTER_NAME
    )
    trainable = [parameter for _name, parameter in adapter_named_parameters]
    adapter_parameter_ids = {id(parameter) for parameter in trainable}
    all_trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    unexpected_trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in adapter_parameter_ids
    ]
    active_adapter_parameter_ids = {
        id(parameter)
        for _name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if (
        unexpected_trainable_names
        or active_adapter_parameter_ids != adapter_parameter_ids
    ):
        raise RuntimeError(
            "STOP-decision scope contains unexpected trainable parameters: "
            f"adapter={len(trainable)} all={len(all_trainable_names)} "
            f"unexpected={unexpected_trainable_names[:8]}"
        )
    LOGGER.info(
        "Verified isolated trainable scope: tensors=%d params=%d active=%s",
        len(trainable),
        parameter_count,
        integration.active_lora_adapters(),
    )
    if args.validation_adapter_checkpoint is not None:
        if not rollout_roots:
            raise ValueError(
                "--validation-adapter-checkpoint requires --rollout-root data"
            )
        replay_path = args.validation_adapter_checkpoint.expanduser().resolve()
        if not replay_path.is_file():
            raise FileNotFoundError(f"Missing validation adapter: {replay_path}")
        replay = torch.load(replay_path, map_location="cpu", weights_only=True)
        if replay.get("schema") not in {
            CHECKPOINT_SCHEMA,
            f"{CHECKPOINT_SCHEMA}-failed-validation",
        }:
            raise RuntimeError(
                f"Invalid validation adapter schema: {replay.get('schema')!r}"
            )
        if replay.get("adapter_name") != STOP_DECISION_ADAPTER_NAME:
            raise RuntimeError("Validation replay has the wrong adapter name")
        replay_config = replay.get("adapter_config") or {
            "rank": args.rank,
            "alpha": args.alpha,
            "layer_indices": layers,
            "target_modules": target_modules,
            "dropout": 0.0,
        }
        expected_config = {
            "rank": args.rank,
            "alpha": args.alpha,
            "layer_indices": layers,
            "target_modules": target_modules,
            "dropout": 0.0,
        }
        if replay_config != expected_config:
            raise RuntimeError(
                "Validation adapter config mismatch: "
                f"checkpoint={replay_config} expected={expected_config}"
            )
        replay_base = replay.get("base_contract") or {}
        replay_default_fingerprint = replay_base.get("default_lora_fingerprint")
        if (
            replay_default_fingerprint is not None
            and replay_default_fingerprint != original_fingerprint
        ):
            raise RuntimeError(
                "Validation adapter was trained against a different navigation LoRA"
            )
        replay_dataset_contract = replay.get("dataset_contract")
        if (
            replay_dataset_contract is not None
            and replay_dataset_contract != dataset_contract
        ):
            raise RuntimeError(
                "Validation adapter dataset contract does not match the replay data"
            )
        replay_state = replay.get("adapter_state_dict")
        if not isinstance(replay_state, dict) or not replay_state:
            raise RuntimeError("Validation adapter has no adapter_state_dict")
        loaded_replay_tensors = integration.load_lora_adapter_state_dict(
            STOP_DECISION_ADAPTER_NAME,
            replay_state,
        )
        if (
            loaded_replay_tensors != len(replay_state)
            or loaded_replay_tensors != len(trainable)
        ):
            raise RuntimeError(
                "Incomplete validation adapter load: "
                f"loaded={loaded_replay_tensors} checkpoint={len(replay_state)}"
            )
        expected_replay_fingerprint = replay.get("adapter_fingerprint")
        current_replay_fingerprint = integration.lora_adapter_fingerprint(
            STOP_DECISION_ADAPTER_NAME
        )
        if (
            expected_replay_fingerprint is not None
            and current_replay_fingerprint != expected_replay_fingerprint
        ):
            raise RuntimeError("Validation adapter fingerprint mismatch after load")
        integration.activate_lora_adapters(
            (DEFAULT_LORA_ADAPTER_NAME, STOP_DECISION_ADAPTER_NAME),
            trainable_adapters=(),
        )
        model.eval()
        replay_dataset = val_dataset
        replay_loader = val_loader
        replay_original_terminal_mask = validation_original_terminal_mask
        if args.replay_split == "train":
            replay_dataset = train_dataset
            replay_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                **{**loader_kwargs, "persistent_workers": False},
            )
            replay_original_terminal_mask = torch.tensor(
                train_dataset.original_terminals,
                dtype=torch.bool,
            )
        replay_loss, replay_metrics, replay_probabilities, _replay_targets = _validate(
            integration,
            replay_loader,
            device,
            args.negative_loss_weight,
            original_terminal_mask=replay_original_terminal_mask,
        )
        if integration.lora_adapter_fingerprint(
            DEFAULT_LORA_ADAPTER_NAME
        ) != original_fingerprint:
            raise RuntimeError("Navigation LoRA changed during validation replay")
        score_audit = _save_score_audit(
            output_dir,
            replay_dataset,
            replay_probabilities,
            split_name=args.replay_split,
        )
        replay_result = {
            "schema": "heatmapvln-system2-stop-decision-validation-replay-v1",
            "replay_split": args.replay_split,
            "adapter_checkpoint": str(replay_path),
            "adapter_fingerprint": current_replay_fingerprint,
            "loss": replay_loss,
            "metrics": replay_metrics,
            "score_audit": score_audit,
        }
        if args.replay_split == "validation":
            replay_result["validation_score_audit"] = score_audit
        if args.export_veto_only_checkpoint:
            veto_only_metrics = _veto_only_threshold_metrics(replay_metrics)
            if not veto_only_metrics["quality_passed"]:
                raise RuntimeError(
                    "Veto-only validation quality gate failed: "
                    + "; ".join(veto_only_metrics["quality_violations"])
                )
            veto_only_checkpoint = {
                "schema": CHECKPOINT_SCHEMA,
                "policy_kind": VETO_ONLY_POLICY,
                "adapter_name": STOP_DECISION_ADAPTER_NAME,
                "adapter_state_dict": replay_state,
                "adapter_fingerprint": current_replay_fingerprint,
                "adapter_config": expected_config,
                "base_contract": {
                    "checkpoint": str(base_path),
                    "checkpoint_file_sha256": _file_sha256(base_path),
                    "default_adapter_name": DEFAULT_LORA_ADAPTER_NAME,
                    "default_lora_tensors": matched,
                    "default_lora_fingerprint": original_fingerprint,
                },
                "token_contract": integration.structured_view_token_contract(),
                "dataset_contract": dataset_contract,
                "thresholds": veto_only_metrics,
                "training": {
                    "steps": args.validation_source_training_steps,
                    "batch_size": args.batch_size,
                    "grad_accum_steps": args.grad_accum_steps,
                    "learning_rate": args.validation_source_learning_rate,
                    "negative_loss_weight": args.negative_loss_weight,
                    "terminal_negative_sampling_fraction": (
                        args.terminal_negative_sampling_fraction
                    ),
                    "mined_regular_negative_fraction": (
                        args.mined_regular_negative_fraction
                    ),
                    "mined_negative_contract": mined_negative_contract,
                    "bce_loss_weight": args.bce_loss_weight,
                    "ranking_loss_weight": args.ranking_loss_weight,
                    "ranking_margin": args.ranking_margin,
                    "holdout_scene_fraction": args.holdout_scene_fraction,
                    "seed": args.seed,
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                    "val_loss": replay_loss,
                    "validation_score_audit": score_audit,
                    "source_candidate": str(replay_path),
                    "source_candidate_sha256": _file_sha256(replay_path),
                    "source_candidate_schema": replay.get("schema"),
                },
            }
            veto_only_path = output_dir / "veto_only_stop_decision_adapter.pth"
            _atomic_torch_save(veto_only_checkpoint, veto_only_path)
            replay_result["veto_only_checkpoint"] = {
                "path": str(veto_only_path),
                "sha256": _file_sha256(veto_only_path),
                "thresholds": veto_only_metrics,
            }
        replay_result_path = output_dir / (
            "validation_replay.json"
            if args.replay_split == "validation"
            else "train_replay.json"
        )
        replay_result_path.write_text(
            json.dumps(replay_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        LOGGER.info(
            "STOP replay complete: split=%s loss=%.5f auc=%.4f audit=%s",
            args.replay_split,
            replay_loss,
            replay_metrics["roc_auc"],
            score_audit["path"],
        )
        return 0
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.max_steps // args.grad_accum_steps, 1),
        eta_min=args.learning_rate * 0.1,
    )

    baseline_validation = None
    baseline_probabilities = None
    baseline_targets = None
    if not args.dry_run:
        model.eval()
        (
            baseline_loss,
            baseline_metrics,
            baseline_probabilities,
            baseline_targets,
        ) = _validate(
            integration,
            val_loader,
            device,
            args.negative_loss_weight,
            original_terminal_mask=validation_original_terminal_mask,
        )
        baseline_validation = {
            "loss": baseline_loss,
            "metrics": baseline_metrics,
        }
        LOGGER.info(
            "Zero-adapter baseline: val_loss=%.5f auc=%.4f add_recall=%.3f "
            "add_fpr=%.3f veto_recall=%.3f veto_negative_rejection=%.3f",
            baseline_loss,
            baseline_metrics["roc_auc"],
            baseline_metrics["add"]["recall"],
            baseline_metrics["add"]["false_positive_rate"],
            baseline_metrics["veto"]["recall"],
            baseline_metrics["veto"]["negative_rejection_rate"],
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_iterator = iter(train_loader)
    dry_run_initial_parameters = (
        [parameter.detach().clone() for parameter in trainable]
        if args.dry_run
        else []
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    rolling_loss = 0.0
    for step in range(1, args.max_steps + 1):
        search_attempts = 0
        while True:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits, probabilities, targets = _stop_forward(
                    integration, batch, device
                )
                loss = _loss(
                    logits,
                    targets,
                    args.negative_loss_weight,
                    bce_weight=args.bce_loss_weight,
                    ranking_weight=args.ranking_loss_weight,
                    ranking_margin=args.ranking_margin,
                )
                scaled_loss = loss / args.grad_accum_steps
            if not args.dry_run or float(loss.detach().item()) > 1e-8:
                break
            search_attempts += 1
            if search_attempts >= len(train_loader):
                raise RuntimeError("REAL dry-run could not find a nonzero STOP loss")
        if not bool(torch.isfinite(loss.detach())):
            raise RuntimeError(f"Non-finite STOP-decision loss at step {step}: {loss}")
        scaled_loss.backward()
        rolling_loss += float(loss.detach().item())
        grad_norm = None
        if step % args.grad_accum_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not bool(torch.isfinite(grad_norm.detach())):
                raise RuntimeError(
                    f"Non-finite STOP-decision gradient norm: {grad_norm}"
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        if step <= 3 or step % args.log_interval == 0:
            counts = _confusion(targets.detach(), probabilities.detach(), 0.5)
            LOGGER.info(
                "step=%d/%d loss=%.5f avg=%.5f lr=%.3e pos=%d neg=%d "
                "tp=%d fp=%d tn=%d fn=%d elapsed=%.1fs",
                step,
                args.max_steps,
                float(loss.detach().item()),
                rolling_loss / step,
                optimizer.param_groups[0]["lr"],
                int((targets >= 0.5).sum().item()),
                int((targets < 0.5).sum().item()),
                counts["tp"],
                counts["fp"],
                counts["tn"],
                counts["fn"],
                time.perf_counter() - started,
            )
        if args.dry_run:
            if grad_norm is None:
                raise RuntimeError("REAL dry-run did not reach an optimizer boundary")
            parameter_delta = max(
                float((parameter.detach() - initial).abs().max().item())
                for parameter, initial in zip(trainable, dry_run_initial_parameters)
            )
            final_default_fingerprint = integration.lora_adapter_fingerprint(
                DEFAULT_LORA_ADAPTER_NAME
            )
            max_memory_gib = (
                torch.cuda.max_memory_allocated(device) / (1024**3)
                if device.type == "cuda"
                else 0.0
            )
            if float(loss.detach().item()) <= 1e-8:
                raise RuntimeError("REAL dry-run positive STOP loss has no learning signal")
            if float(grad_norm.detach().item()) <= 0.0:
                raise RuntimeError("REAL dry-run STOP adapter gradient norm is zero")
            if parameter_delta <= 0.0:
                raise RuntimeError("REAL dry-run optimizer did not update STOP adapter")
            if final_default_fingerprint != original_fingerprint:
                raise RuntimeError("Navigation LoRA changed during REAL dry-run")
            LOGGER.info(
                "REAL STOP-decision dry-run passed: positive_samples=%d loss=%.6f "
                "grad_norm=%.6f max_parameter_delta=%.6g max_allocated=%.2fGiB "
                "default_lora_unchanged=true",
                int((targets >= 0.5).sum().item()),
                float(loss.detach().item()),
                float(grad_norm.detach().item()),
                parameter_delta,
                max_memory_gib,
            )
            return 0

    model.eval()
    val_loss, threshold_metrics, final_probabilities, final_targets = _validate(
        integration,
        val_loader,
        device,
        args.negative_loss_weight,
        original_terminal_mask=validation_original_terminal_mask,
    )
    if baseline_targets is None or not torch.equal(final_targets, baseline_targets):
        raise RuntimeError("STOP-decision validation order changed during training")
    final_fingerprint = integration.lora_adapter_fingerprint(
        DEFAULT_LORA_ADAPTER_NAME
    )
    if final_fingerprint != original_fingerprint:
        raise RuntimeError("Original navigation LoRA changed during STOP-decision training")
    adapter_state = integration.lora_adapter_state_dict(
        STOP_DECISION_ADAPTER_NAME
    )
    adapter_fingerprint = integration.lora_adapter_fingerprint(
        STOP_DECISION_ADAPTER_NAME
    )
    validation_score_audit = None
    if rollout_roots:
        validation_score_audit = _save_validation_score_audit(
            output_dir,
            val_dataset,
            final_probabilities,
        )
    if not threshold_metrics["quality_passed"]:
        failed_validation = {
            "baseline": baseline_validation,
            "final": {"loss": val_loss, "metrics": threshold_metrics},
            "validation_score_audit": validation_score_audit,
        }
        (output_dir / "failed_validation.json").write_text(
            json.dumps(failed_validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _atomic_torch_save(
            {
                "schema": f"{CHECKPOINT_SCHEMA}-failed-validation",
                "policy_kind": ADD_AND_VETO_POLICY,
                "adapter_name": STOP_DECISION_ADAPTER_NAME,
                "adapter_state_dict": adapter_state,
                "adapter_fingerprint": adapter_fingerprint,
                "adapter_config": {
                    "rank": args.rank,
                    "alpha": args.alpha,
                    "layer_indices": layers,
                    "target_modules": target_modules,
                    "dropout": 0.0,
                },
                "thresholds": threshold_metrics,
                "dataset_contract": dataset_contract,
                "base_contract": {
                    "checkpoint": str(base_path),
                    "checkpoint_file_sha256": _file_sha256(base_path),
                    "default_adapter_name": DEFAULT_LORA_ADAPTER_NAME,
                    "default_lora_tensors": matched,
                    "default_lora_fingerprint": original_fingerprint,
                },
                "token_contract": integration.structured_view_token_contract(),
                "training": {
                    "steps": args.max_steps,
                    "batch_size": args.batch_size,
                    "grad_accum_steps": args.grad_accum_steps,
                    "learning_rate": args.learning_rate,
                    "negative_loss_weight": args.negative_loss_weight,
                    "terminal_negative_sampling_fraction": (
                        args.terminal_negative_sampling_fraction
                    ),
                    "mined_regular_negative_fraction": (
                        args.mined_regular_negative_fraction
                    ),
                    "mined_negative_contract": mined_negative_contract,
                    "bce_loss_weight": args.bce_loss_weight,
                    "ranking_loss_weight": args.ranking_loss_weight,
                    "ranking_margin": args.ranking_margin,
                    "holdout_scene_fraction": args.holdout_scene_fraction,
                    "seed": args.seed,
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                    "val_loss": val_loss,
                    "validation_score_audit": validation_score_audit,
                },
            },
            output_dir / "failed_adapter_candidate.pth",
        )
        raise RuntimeError(
            "STOP-decision validation quality gate failed: "
            + "; ".join(threshold_metrics["quality_violations"])
        )
    integration.activate_lora_adapters(
        (DEFAULT_LORA_ADAPTER_NAME,),
        trainable_adapters=(),
    )
    if integration.active_lora_adapters() != (DEFAULT_LORA_ADAPTER_NAME,):
        raise RuntimeError("Could not restore navigation-only LoRA after training")

    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "policy_kind": ADD_AND_VETO_POLICY,
        "adapter_name": STOP_DECISION_ADAPTER_NAME,
        "adapter_state_dict": adapter_state,
        "adapter_fingerprint": adapter_fingerprint,
        "adapter_config": {
            "rank": args.rank,
            "alpha": args.alpha,
            "layer_indices": layers,
            "target_modules": target_modules,
            "dropout": 0.0,
        },
        "base_contract": {
            "checkpoint": str(base_path),
            "checkpoint_file_sha256": _file_sha256(base_path),
            "default_adapter_name": DEFAULT_LORA_ADAPTER_NAME,
            "default_lora_tensors": matched,
            "default_lora_fingerprint": original_fingerprint,
        },
        "token_contract": integration.structured_view_token_contract(),
        "dataset_contract": dataset_contract,
        "thresholds": threshold_metrics,
        "training": {
            "steps": args.max_steps,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "learning_rate": args.learning_rate,
            "negative_loss_weight": args.negative_loss_weight,
            "terminal_negative_sampling_fraction": (
                args.terminal_negative_sampling_fraction
            ),
            "mined_regular_negative_fraction": (
                args.mined_regular_negative_fraction
            ),
            "mined_negative_contract": mined_negative_contract,
            "bce_loss_weight": args.bce_loss_weight,
            "ranking_loss_weight": args.ranking_loss_weight,
            "ranking_margin": args.ranking_margin,
            "holdout_scene_fraction": args.holdout_scene_fraction,
            "seed": args.seed,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "val_loss": val_loss,
            "baseline_validation": baseline_validation,
            "validation_score_audit": validation_score_audit,
        },
    }
    checkpoint_path = output_dir / "stop_decision_adapter.pth"
    _atomic_torch_save(checkpoint, checkpoint_path)
    summary = {
        key: value for key, value in checkpoint.items() if key != "adapter_state_dict"
    }
    summary["checkpoint"] = str(checkpoint_path)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "STOP-decision training complete: val_loss=%.5f add>=%.3f veto>=%.3f checkpoint=%s",
        val_loss,
        threshold_metrics["add_stop_threshold"],
        threshold_metrics["veto_stop_threshold"],
        checkpoint_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

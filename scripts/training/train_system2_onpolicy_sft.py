#!/usr/bin/env python3
"""Continue the existing navigation LoRA on native plus on-policy System2 SFT."""

from __future__ import annotations

import argparse
import atexit
import copy
import hashlib
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training import (
    _load_normalized_state_dict,
    assert_complete_lora_checkpoint_match,
    build_model,
    extract_lora_checkpoint_state,
    load_config,
)
from scripts.training.utils import make_autocast_context

from src.data.factory import build_dataset
from src.data.panoramic_tokenized_collator import PanoramicTokenizedCollator
from src.data.stop_rollout_dataset import (
    MixedSystem2SFTDataset,
    System2StopMultimodalDataset,
)
from src.models.qwen2_5_vl.integration import (
    DEFAULT_LORA_ADAPTER_NAME,
    STRUCTURED_VIEW_CLASSES,
)
from src.models.runtime_compat import ensure_transformers_runtime_compat

LOGGER = logging.getLogger("system2-onpolicy-sft")
CHECKPOINT_SCHEMA = "heatmapvln-system2-onpolicy-sft-v1"
_MIXED_SFT_ROLES = (
    "native",
    "onpolicy_positive",
    "onpolicy_regular_negative",
    "onpolicy_false_stop_negative",
)
_PAIRED_POSITIVE_ROLE = "onpolicy_paired_positive"
_SYSTEM2_SFT_SAMPLE_KEYS = frozenset(
    {
        "history_frames",
        "current_frame",
        "action",
        "action_valid",
        "discrete_action",
        "is_stop",
        "text",
        "current_views",
        "history_panoramas",
        "pixel_goal",
        "pano_view_id",
        "pano_pixel_goal",
        "pano_sample_kind",
        "turn_action_text",
        "turn_actions",
        "stop_rollout_key",
        "system2_replay_role",
        "system2_original_terminal",
        "system2_oracle_stop_target",
        "system2_stop_pair_id",
    }
)


@dataclass(frozen=True)
class _DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()


def _destroy_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _initialize_distributed(requested_device: str) -> _DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return _DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device(requested_device),
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed System2 continuation requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        raise RuntimeError(
            "WORLD_SIZE > 1 requires torchrun to provide LOCAL_RANK"
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=device,
    )
    atexit.register(_destroy_process_group)
    rank = dist.get_rank()
    actual_world_size = dist.get_world_size()
    if actual_world_size != world_size:
        raise RuntimeError(
            f"Distributed world-size mismatch: env={world_size} actual={actual_world_size}"
        )
    return _DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=actual_world_size,
        device=device,
    )


def _distributed_loss_means(
    loss_groups: tuple[list[torch.Tensor], ...],
    zero: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Return globally normalized losses with correct data-parallel gradients."""
    if not dist.is_available() or not dist.is_initialized():
        return tuple(
            torch.stack(losses).mean() if losses else zero
            for losses in loss_groups
        )
    counts = torch.tensor(
        [len(losses) for losses in loss_groups],
        device=zero.device,
        dtype=torch.float32,
    )
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    world_size = float(dist.get_world_size())
    means = []
    for losses, global_count in zip(loss_groups, counts):
        if not losses or float(global_count.item()) <= 0:
            means.append(zero)
            continue
        # Gradients are averaged again below. Scaling each rank's local sum by
        # world_size/global_count therefore produces the exact global mean.
        means.append(
            torch.stack(losses).sum() * (world_size / global_count)
        )
    return tuple(means)


def _synchronize_gradients(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    context: _DistributedContext,
) -> None:
    if not context.enabled:
        return
    missing = [name for name, parameter in named_parameters if parameter.grad is None]
    missing_count = torch.tensor(
        [len(missing)],
        device=context.device,
        dtype=torch.int64,
    )
    dist.all_reduce(missing_count, op=dist.ReduceOp.SUM)
    if int(missing_count.item()) != 0:
        raise RuntimeError(
            "All System2 LoRA tensors must receive gradients on every rank: "
            f"rank={context.rank} local_missing={missing[:8]} "
            f"global_missing_count={int(missing_count.item())}"
        )

    grouped: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for _name, parameter in named_parameters:
        gradient = parameter.grad
        assert gradient is not None
        grouped.setdefault((gradient.device, gradient.dtype), []).append(gradient)
    with torch.no_grad():
        for gradients in grouped.values():
            flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.mul_(1.0 / context.world_size)
            offset = 0
            for gradient in gradients:
                count = gradient.numel()
                gradient.copy_(flat[offset : offset + count].view_as(gradient))
                offset += count


def _reduce_training_window(
    values: tuple[float, ...],
    role_counts: dict[str, int],
    context: _DistributedContext,
) -> tuple[tuple[float, ...], dict[str, int]]:
    if not context.enabled:
        return values, role_counts
    roles = (*_MIXED_SFT_ROLES, _PAIRED_POSITIVE_ROLE)
    payload = torch.tensor(
        [*values, *(float(role_counts.get(role, 0)) for role in roles)],
        device=context.device,
        dtype=torch.float64,
    )
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    reduced_values = tuple(
        float(value.item()) / context.world_size
        for value in payload[: len(values)]
    )
    reduced_roles = {
        role: int(round(payload[len(values) + index].item()))
        for index, role in enumerate(roles)
        if payload[len(values) + index].item() > 0
    }
    return reduced_values, reduced_roles


def _reduce_role_counts(
    role_counts: dict[str, int],
    context: _DistributedContext,
) -> dict[str, int]:
    if not context.enabled:
        return dict(sorted(role_counts.items()))
    roles = (*_MIXED_SFT_ROLES, _PAIRED_POSITIVE_ROLE)
    payload = torch.tensor(
        [role_counts.get(role, 0) for role in roles],
        device=context.device,
        dtype=torch.int64,
    )
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    return {
        role: int(payload[index].item())
        for index, role in enumerate(roles)
        if int(payload[index].item()) > 0
    }


def _assert_fingerprint_synchronized(
    fingerprint: str,
    context: _DistributedContext,
    *,
    label: str,
) -> None:
    if not context.enabled:
        return
    fingerprints: list[str | None] = [None] * context.world_size
    dist.all_gather_object(fingerprints, fingerprint)
    if len(set(fingerprints)) != 1:
        raise RuntimeError(
            f"System2 {label} LoRA differs across ranks: {fingerprints}"
        )


class _System2SFTCollator:
    """Normalize native and on-policy rows to the shared System2 SFT contract."""

    def __init__(self, tokenized_collator: PanoramicTokenizedCollator) -> None:
        self.tokenized_collator = tokenized_collator

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        expanded = []
        for raw_sample in batch:
            sample = dict(raw_sample)
            paired_positive = sample.pop("_system2_paired_positive", None)
            expanded.append(sample)
            if paired_positive is not None:
                expanded.append(paired_positive)
        normalized = [
            {
                key: value
                for key, value in sample.items()
                if key in _SYSTEM2_SFT_SAMPLE_KEYS
            }
            for sample in expanded
        ]
        return self.tokenized_collator(normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    rollout_source = parser.add_mutually_exclusive_group(required=True)
    rollout_source.add_argument("--rollout-root", action="append")
    rollout_source.add_argument("--rollout-report", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--l2-sp-weight", type=float, default=1.0)
    parser.add_argument("--native-slots", type=int, default=14)
    parser.add_argument("--onpolicy-positive-slots", type=int, default=3)
    parser.add_argument("--onpolicy-regular-negative-slots", type=int, default=1)
    parser.add_argument("--onpolicy-false-stop-negative-slots", type=int, default=2)
    parser.add_argument("--regular-negative-min-stop-log-odds", type=float)
    parser.add_argument("--pairwise-stop-margin-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-stop-margin-gap", type=float, default=1.0)
    parser.add_argument("--holdout-scene-fraction", type=float, default=0.2)
    parser.add_argument("--max-validation-samples", type=int, default=128)
    parser.add_argument("--max-train-evaluation-samples", type=int, default=48)
    parser.add_argument("--validation-interval", type=int, default=0)
    parser.add_argument("--save-validation-checkpoints", action="store_true")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint: {path}")
    for key in ("trainable_state_dict", "model_state_dict", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise KeyError(f"Checkpoint contains no tensor state dict: {path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _resolve_rollout_roots(
    args: argparse.Namespace,
) -> tuple[list[Path], dict[str, Any] | None]:
    if args.rollout_report is None:
        roots = [Path(root).expanduser().resolve() for root in args.rollout_root]
        report_contract = None
    else:
        report_path = args.rollout_report.expanduser().resolve()
        if not report_path.is_file():
            raise FileNotFoundError(f"Missing rollout validation report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "passed":
            raise RuntimeError(
                f"Rollout validation report did not pass: {report.get('status')!r}"
            )
        entries = report.get("roots")
        if not isinstance(entries, list) or not entries:
            raise RuntimeError("Rollout validation report contains no roots")
        raw_roots = [
            entry.get("root") if isinstance(entry, dict) else None
            for entry in entries
        ]
        if any(not isinstance(root, str) or not root for root in raw_roots):
            raise RuntimeError("Rollout validation report contains an invalid root")
        expected_count = int(report.get("root_count", -1))
        if expected_count != len(raw_roots):
            raise RuntimeError(
                "Rollout validation report root count mismatch: "
                f"reported={expected_count} actual={len(raw_roots)}"
            )
        roots = [Path(root).expanduser().resolve() for root in raw_roots]
        report_contract = {
            "path": str(report_path),
            "sha256": _file_sha256(report_path),
            "status": report["status"],
            "root_count": expected_count,
        }
    if len(set(roots)) != len(roots):
        raise RuntimeError("Duplicate rollout roots are not allowed")
    missing = [str(root) for root in roots if not root.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing rollout roots: {missing}")
    return roots, report_contract


def _move_inputs(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for name, value in inputs.items()
    }


def _sft_objective(
    integration: Any,
    batch: dict[str, Any],
    device: torch.device,
    *,
    structured_class_token_ids: tuple[int, ...],
    pairwise_stop_margin_weight: float = 0.0,
    pairwise_stop_margin_gap: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    stop_token_id = int(structured_class_token_ids[0])
    inputs = _move_inputs(batch["pano_inputs"], device)
    _hidden, _vision, _images, _traj, lm_output = integration._forward_model_inputs(
        inputs,
        return_hidden_states=False,
        return_lm_correct_logprobs=True,
        structured_class_token_ids=structured_class_token_ids,
        extract_vision_hidden_states=False,
    )
    if not isinstance(lm_output, dict):
        raise RuntimeError("Sparse System2 SFT forward returned no LM output")
    alignment = lm_output.get("alignment")
    logprobs = lm_output.get("correct_label_logprobs")
    rejection_log_odds = lm_output.get("correct_label_rejection_log_odds")
    structured_logits = lm_output.get("structured_class_logits")
    if (
        not isinstance(alignment, dict)
        or not torch.is_tensor(logprobs)
        or not torch.is_tensor(rejection_log_odds)
        or not torch.is_tensor(structured_logits)
    ):
        raise RuntimeError("Sparse System2 SFT output violates its alignment contract")

    counts = [int(count) for count in alignment["sample_label_tokens"]]
    if any(count <= 0 for count in counts):
        raise RuntimeError(f"Every SFT row must have labelled tokens: {counts}")
    if sum(counts) != int(logprobs.numel()):
        raise RuntimeError(
            "Sparse LM alignment mismatch: "
            f"counts={sum(counts)} logprobs={logprobs.numel()}"
        )
    if rejection_log_odds.shape != logprobs.shape:
        raise RuntimeError(
            "Sparse LM rejection log-odds mismatch: "
            f"logprobs={tuple(logprobs.shape)} "
            f"log_odds={tuple(rejection_log_odds.shape)}"
        )
    roles = [str(role) for role in batch["system2_replay_role"]]
    if structured_logits.shape != (len(roles), len(structured_class_token_ids)):
        raise RuntimeError(
            "Structured System2 class-logit mismatch: "
            f"logits={tuple(structured_logits.shape)} "
            f"expected={(len(roles), len(structured_class_token_ids))}"
        )
    token_ids = alignment["sample_correct_token_ids"]
    if len(roles) != len(counts) or len(token_ids) != len(counts):
        raise RuntimeError(
            "System2 mixed-batch alignment mismatch: "
            f"roles={len(roles)} counts={len(counts)} ids={len(token_ids)}"
        )
    pair_ids = list(batch.get("system2_stop_pair_id") or [None] * len(roles))
    if len(pair_ids) != len(roles):
        raise RuntimeError(
            "System2 pair metadata is not batch-aligned: "
            f"pairs={len(pair_ids)} roles={len(roles)}"
        )
    stop_margins = structured_logits[:, 0].float() - torch.logsumexp(
        structured_logits[:, 1:].float(),
        dim=1,
    )

    sample_losses = []
    token_ce_losses = []
    stop_rejection_losses = []
    offset = 0
    for row, (count, role, row_token_ids) in enumerate(zip(counts, roles, token_ids)):
        row_logprobs = logprobs[offset : offset + count].float()
        offset += count
        if role != "native":
            stop_positions = [
                position
                for position, token_id in enumerate(row_token_ids)
                if int(token_id) == int(stop_token_id)
            ]
            if len(stop_positions) != 1 or batch["sft_target_text"][row] != ["view: stop"]:
                raise RuntimeError(
                    "On-policy STOP supervision must expose exactly one labelled "
                    f"STOP token: row={row} role={role!r} positions={stop_positions} "
                    f"target={batch['sft_target_text'][row]!r}"
                )
            stop_position = stop_positions[0]
        if role in {
            "onpolicy_regular_negative",
            "onpolicy_false_stop_negative",
        }:
            rejection = F.softplus(stop_margins[row])
            stop_rejection_losses.append(rejection)
            sample_losses.append(rejection)
        elif role in {"onpolicy_positive", _PAIRED_POSITIVE_ROLE}:
            if role == _PAIRED_POSITIVE_ROLE and pairwise_stop_margin_weight <= 0:
                raise RuntimeError(
                    "Paired positive row requires pairwise_stop_margin_weight > 0"
                )
            # Train exactly the six-way decision used by online System2.
            # Native replay still preserves the full structured text protocol.
            token_ce = F.cross_entropy(
                structured_logits[row : row + 1].float(),
                torch.zeros(1, device=structured_logits.device, dtype=torch.long),
            )
            token_ce_losses.append(token_ce)
            sample_losses.append(token_ce)
        else:
            token_ce = -row_logprobs.mean()
            token_ce_losses.append(token_ce)
            sample_losses.append(token_ce)

    zero = logprobs.sum() * 0.0
    pair_rows: dict[str, dict[str, int]] = {}
    for row, (role, pair_id) in enumerate(zip(roles, pair_ids)):
        if pair_id is None:
            continue
        if role not in {"onpolicy_false_stop_negative", _PAIRED_POSITIVE_ROLE}:
            raise RuntimeError(
                f"Unexpected System2 pair role: row={row} role={role!r}"
            )
        members = pair_rows.setdefault(str(pair_id), {})
        if role in members:
            raise RuntimeError(f"Duplicate {role!r} row for pair {pair_id!r}")
        members[role] = row

    pair_losses = []
    for pair_id, members in pair_rows.items():
        expected = {"onpolicy_false_stop_negative", _PAIRED_POSITIVE_ROLE}
        if set(members) != expected:
            raise RuntimeError(
                f"Incomplete System2 STOP pair {pair_id!r}: roles={sorted(members)}"
            )
        positive_margin = stop_margins[members[_PAIRED_POSITIVE_ROLE]]
        false_stop_margin = stop_margins[
            members["onpolicy_false_stop_negative"]
        ]
        pair_losses.append(
            F.softplus(
                float(pairwise_stop_margin_gap)
                - (positive_margin - false_stop_margin)
            )
        )
    base_loss, token_ce_mean, rejection_mean, pairwise_mean = (
        _distributed_loss_means(
            (
                sample_losses,
                token_ce_losses,
                stop_rejection_losses,
                pair_losses,
            ),
            zero,
        )
    )
    total_loss = base_loss + float(pairwise_stop_margin_weight) * pairwise_mean
    return total_loss, token_ce_mean, rejection_mean, pairwise_mean


def _target_class_indices(target_texts: list[list[str]], device: torch.device) -> torch.Tensor:
    indices = []
    for texts in target_texts:
        if len(texts) != 1 or not texts[0].startswith("view: "):
            raise RuntimeError(f"Invalid structured System2 target: {texts!r}")
        class_name = texts[0].splitlines()[0].removeprefix("view: ").strip()
        try:
            indices.append(STRUCTURED_VIEW_CLASSES.index(class_name))
        except ValueError as exc:
            raise RuntimeError(f"Unknown structured System2 class: {class_name!r}") from exc
    return torch.tensor(indices, device=device, dtype=torch.long)


def _structured_class_forward(
    integration: Any,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = _move_inputs(batch["pano_inputs"], device)
    positions = batch["system2_stop_predictor_position"].to(
        device=device,
        dtype=torch.long,
        non_blocking=True,
    )
    hidden, _vision, _images, _traj, _lm = integration._forward_model_inputs(
        inputs,
        return_hidden_states=True,
        skip_lm_head=True,
        return_last_hidden_state_only=True,
        extract_vision_hidden_states=False,
    )
    if hidden is None:
        raise RuntimeError("Structured System2 validation returned no hidden states")
    logits = integration.structured_view_class_logits(hidden, positions).float()
    targets = _target_class_indices(batch["sft_target_text"], device)
    return logits, targets


def _stratified_validation_subset(
    dataset: System2StopMultimodalDataset,
    max_samples: int,
    seed: int,
) -> System2StopMultimodalDataset:
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    pools = [
        [index for index, target in enumerate(dataset.targets) if int(target) == 1],
        [
            index
            for index, (target, terminal) in enumerate(
                zip(dataset.targets, dataset.original_terminals)
            )
            if int(target) == 0 and not bool(terminal)
        ],
        [
            index
            for index, (target, terminal) in enumerate(
                zip(dataset.targets, dataset.original_terminals)
            )
            if int(target) == 0 and bool(terminal)
        ],
    ]
    for pool_index, pool in enumerate(pools):
        pool.sort(
            key=lambda index: hashlib.sha256(
                f"{seed}:{pool_index}:{dataset.records[index]['key']}".encode()
            ).digest()
        )
    selected: list[int] = []
    positions = [0] * len(pools)
    while len(selected) < max_samples:
        made_progress = False
        for pool_index, pool in enumerate(pools):
            if positions[pool_index] >= len(pool):
                continue
            selected.append(pool[positions[pool_index]])
            positions[pool_index] += 1
            made_progress = True
            if len(selected) >= max_samples:
                break
        if not made_progress:
            break
    return dataset.subset_by_indices(selected)


def _evaluate(
    integration: Any,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor, list[str]]:
    predictions = []
    targets = []
    stop_probabilities = []
    stop_margins = []
    oracle_stop_targets: list[int] = []
    original_terminals: list[bool] = []
    keys: list[str] = []
    integration.model.eval()
    with torch.no_grad():
        for batch in loader:
            with make_autocast_context(device, "bf16"):
                logits, batch_targets = _structured_class_forward(
                    integration, batch, device
                )
            probabilities = torch.softmax(logits.float(), dim=-1)
            predictions.append(logits.argmax(dim=-1).cpu())
            targets.append(batch_targets.cpu())
            stop_probabilities.append(probabilities[:, 0].cpu())
            stop_margins.append(
                (logits[:, 0] - logits[:, 1:].amax(dim=-1)).float().cpu()
            )
            oracle_stop_targets.extend(
                int(value) for value in batch["system2_oracle_stop_target"]
            )
            original_terminals.extend(
                bool(value) for value in batch["system2_original_terminal"]
            )
            keys.extend(str(value) for value in batch["stop_rollout_key"])

    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    stop_probability = torch.cat(stop_probabilities)
    stop_margin = torch.cat(stop_margins)
    oracle_stop_target = torch.tensor(oracle_stop_targets, dtype=torch.bool)
    original_terminal = torch.tensor(original_terminals, dtype=torch.bool)
    is_stop = oracle_stop_target
    predicted_stop = prediction == 0
    regular_negative = ~is_stop & ~original_terminal
    false_stop_negative = ~is_stop & original_terminal
    class_target_known = ~false_stop_negative

    def _rate(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
        count = int(denominator.sum().item())
        return float((numerator & denominator).sum().item() / count) if count else 0.0

    def _masked_stat(
        values: torch.Tensor,
        mask: torch.Tensor,
        *,
        quantile: float | None = None,
    ) -> float:
        selected = values[mask].float()
        if not selected.numel():
            return 0.0
        if quantile is None:
            return float(selected.mean().item())
        return float(torch.quantile(selected, quantile).item())

    metrics = {
        "samples": float(target.numel()),
        "stop_samples": float(is_stop.sum().item()),
        "regular_negative_samples": float(regular_negative.sum().item()),
        "false_stop_negative_samples": float(false_stop_negative.sum().item()),
        "known_class_samples": float(class_target_known.sum().item()),
        "class_accuracy": _rate(prediction == target, class_target_known),
        "stop_recall": _rate(predicted_stop, is_stop),
        "regular_negative_stop_fpr": _rate(predicted_stop, regular_negative),
        "false_stop_negative_stop_fpr": _rate(predicted_stop, false_stop_negative),
        "non_stop_class_accuracy": _rate(prediction == target, regular_negative),
        "mean_stop_probability_positive": float(stop_probability[is_stop].mean().item()),
        "mean_stop_probability_negative": float(stop_probability[~is_stop].mean().item()),
        "mean_stop_margin_positive": _masked_stat(stop_margin, is_stop),
        "median_stop_margin_positive": _masked_stat(
            stop_margin, is_stop, quantile=0.5
        ),
        "p90_stop_margin_positive": _masked_stat(stop_margin, is_stop, quantile=0.9),
        "mean_stop_margin_regular_negative": _masked_stat(
            stop_margin, regular_negative
        ),
        "mean_stop_margin_false_stop_negative": _masked_stat(
            stop_margin, false_stop_negative
        ),
    }
    metrics["positive_regular_margin_gap"] = (
        metrics["mean_stop_margin_positive"]
        - metrics["mean_stop_margin_regular_negative"]
    )
    metrics["positive_false_stop_margin_gap"] = (
        metrics["mean_stop_margin_positive"]
        - metrics["mean_stop_margin_false_stop_negative"]
    )
    return metrics, prediction, keys


def _with_quality_metrics(
    metrics: dict[str, float],
    predictions: torch.Tensor,
    *,
    baseline_metrics: dict[str, float],
    baseline_predictions: torch.Tensor,
    targets: list[int],
    original_terminals: list[bool],
) -> dict[str, float | bool]:
    enriched: dict[str, float | bool] = dict(metrics)
    regular_non_stop_mask = torch.tensor(
        [target == 0 and not terminal for target, terminal in zip(targets, original_terminals)],
        dtype=torch.bool,
    )
    if regular_non_stop_mask.numel() != predictions.numel():
        raise RuntimeError("System2 validation target count changed")
    enriched["non_stop_prediction_retention"] = float(
        (
            predictions[regular_non_stop_mask]
            == baseline_predictions[regular_non_stop_mask]
        )
        .float()
        .mean()
        .item()
    )
    enriched["stop_recall_improvement"] = (
        float(enriched["stop_recall"]) - baseline_metrics["stop_recall"]
    )
    enriched["false_stop_fpr_improvement"] = (
        baseline_metrics["false_stop_negative_stop_fpr"]
        - float(enriched["false_stop_negative_stop_fpr"])
    )
    enriched["quality_passed"] = bool(
        enriched["stop_recall_improvement"] > 0
        and enriched["false_stop_fpr_improvement"] > 0
        and enriched["regular_negative_stop_fpr"] <= 0.02
        and enriched["non_stop_class_accuracy"] >= 0.85
        and enriched["non_stop_prediction_retention"] >= 0.90
    )
    return enriched


def _relative_l2_sp(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    references: dict[str, torch.Tensor],
    denominator: torch.Tensor,
) -> torch.Tensor:
    numerator = torch.zeros((), device=denominator.device, dtype=torch.float32)
    for name, parameter in named_parameters:
        numerator = numerator + (parameter.float() - references[name]).square().sum()
    return numerator / denominator.clamp_min(1e-12)


def _current_lora_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }
    if len(state) != 224:
        raise RuntimeError(f"Expected 224 trained LoRA tensors, found {len(state)}")
    return dict(sorted(state.items()))


def main() -> int:
    args = parse_args()
    if args.max_steps <= 0 or args.batch_size <= 0 or args.grad_accum_steps <= 0:
        raise ValueError("max_steps, batch_size, and grad_accum_steps must be positive")
    if args.learning_rate <= 0 or args.min_learning_rate < 0:
        raise ValueError("Learning rates must be non-negative and base LR must be positive")
    if args.min_learning_rate > args.learning_rate:
        raise ValueError("min_learning_rate cannot exceed learning_rate")
    if args.l2_sp_weight < 0:
        raise ValueError("l2_sp_weight must be non-negative")
    if args.pairwise_stop_margin_weight < 0:
        raise ValueError("pairwise_stop_margin_weight must be non-negative")
    if args.pairwise_stop_margin_gap < 0:
        raise ValueError("pairwise_stop_margin_gap must be non-negative")
    if args.validation_interval < 0:
        raise ValueError("validation_interval must be non-negative")
    if args.max_validation_samples < 0 or args.max_train_evaluation_samples < 0:
        raise ValueError("Evaluation sample limits must be non-negative")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")

    distributed = _initialize_distributed(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    if distributed.is_main:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    log_path = (
        output_dir / "train.log"
        if distributed.is_main
        else output_dir / f"train.rank{distributed.rank}.log"
    )
    handlers: list[logging.Handler] = [logging.FileHandler(log_path)]
    if distributed.is_main:
        handlers.insert(0, logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s rank={distributed.rank} %(levelname)s %(message)s",
        handlers=handlers,
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _set_seed(args.seed)
    device = distributed.device
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    LOGGER.info(
        "Distributed runtime: enabled=%s rank=%d local_rank=%d world_size=%d "
        "device=%s batch_per_rank=%d global_batch=%d",
        distributed.enabled,
        distributed.rank,
        distributed.local_rank,
        distributed.world_size,
        device,
        args.batch_size,
        args.batch_size * distributed.world_size,
    )

    cfg = load_config(args.config, validate=False)
    model_path = os.environ.get("INTERNNAV_MODEL_PATH") or cfg["model"]["llm"].get(
        "model_path"
    )
    if not model_path:
        raise ValueError("INTERNNAV_MODEL_PATH or model.llm.model_path is required")
    cfg["model"]["llm"]["model_path"] = model_path
    cfg["model"]["llm"]["gradient_checkpointing"] = True
    cfg["model"]["llm"]["lora_dropout"] = 0.0
    cfg["model"].setdefault("heatmap", {})["enable"] = False
    cfg["model"].setdefault("action_head", {})["enable"] = False
    cfg["model"]["action_head"].setdefault("nextdit", {})["enabled"] = False
    cfg["model"]["action_head"]["nextdit"].setdefault(
        "pano_latent_adapter", {}
    )["enabled"] = False
    cfg["model"].setdefault("stop_head", {})["enabled"] = False
    if args.dataset_root is not None:
        cfg["data"]["root"] = str(args.dataset_root.expanduser().resolve())
    elif os.environ.get("PANORAMIC_DATA_ROOT"):
        cfg["data"]["root"] = os.environ["PANORAMIC_DATA_ROOT"]

    native_cfg = copy.deepcopy(cfg)
    trajectory_cfg = native_cfg["data"].setdefault("trajectory", {})
    trajectory_cfg.update(
        {
            "max_clips": max(0, int(args.max_clips)),
            "require_sft_target": True,
            "load_traj_images": False,
            "load_history_frames": False,
            "load_history_heatmap": False,
            "system2_sample_step": 4,
            "system2_stop_oversample": 5,
            "system2_stop_path_radius_m": 0.0,
            "system2_near_stop_hard_negative_oversample": 0,
            "system2_near_stop_hard_negative_min_path_m": 0.0,
            "system2_near_stop_hard_negative_max_path_m": 0.0,
            "system2_near_stop_hard_negative_min_goal_distance_m": 0.0,
            "system2_near_stop_hard_negative_max_goal_distance_m": 0.0,
        }
    )

    ensure_transformers_runtime_compat(
        model_path=model_path,
        requested_backbone_type=cfg["model"]["llm"].get(
            "backbone_type", "qwen2_5_vl"
        ),
        requested_attn_implementation=cfg["model"]["llm"].get(
            "attn_implementation", "sdpa"
        ),
        logger=LOGGER,
    )
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    collator = _System2SFTCollator(
        PanoramicTokenizedCollator(
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
    )

    LOGGER.info("Building native Stage1-S2 replay from %s", native_cfg["data"]["root"])
    native_dataset = build_dataset(
        native_cfg,
        split="train",
        load_history_heatmap=False,
    )
    rollout_roots, rollout_report_contract = _resolve_rollout_roots(args)
    rollout_dataset = System2StopMultimodalDataset(
        rollout_roots,
        image_size=tuple(int(value) for value in cfg["data"]["image_size"]),
    )
    rollout_train, rollout_validation = rollout_dataset.split_by_scene(
        holdout_fraction=args.holdout_scene_fraction,
        seed=args.seed,
    )
    validation_dataset = _stratified_validation_subset(
        rollout_validation,
        args.max_validation_samples,
        args.seed,
    )
    train_evaluation_dataset = _stratified_validation_subset(
        rollout_train,
        args.max_train_evaluation_samples,
        args.seed + 1,
    )
    mixed_dataset = MixedSystem2SFTDataset(
        native_dataset,
        rollout_train,
        native_slots=args.native_slots,
        positive_slots=args.onpolicy_positive_slots,
        regular_negative_slots=args.onpolicy_regular_negative_slots,
        false_stop_negative_slots=args.onpolicy_false_stop_negative_slots,
        regular_negative_min_stop_log_odds=(
            args.regular_negative_min_stop_log_odds
        ),
        pair_false_stops=args.pairwise_stop_margin_weight > 0,
    )
    LOGGER.info(
        "Mixed SFT data: native=%d rollout_train=%d rollout_validation=%d "
        "validation_used=%d train_evaluation_used=%d virtual_epoch=%d "
        "source_counts=%s pool_sizes=%s regular_negative_mining=%s "
        "false_stop_pairing=%s",
        len(native_dataset),
        len(rollout_train),
        len(rollout_validation),
        len(validation_dataset),
        len(train_evaluation_dataset),
        len(mixed_dataset),
        mixed_dataset.source_counts(),
        mixed_dataset.pool_sizes(),
        mixed_dataset.regular_negative_mining_contract(),
        mixed_dataset.false_stop_pairing_contract(),
    )
    mixed_preflight_indices = []
    for role in _MIXED_SFT_ROLES:
        try:
            mixed_preflight_indices.append(mixed_dataset.slot_pattern.index(role))
        except ValueError as exc:
            raise RuntimeError(f"Mixed-data preflight cannot find role {role!r}") from exc
    mixed_collate_preflight = collator(
        [mixed_dataset[index] for index in mixed_preflight_indices]
    )
    observed_preflight_roles = set(
        mixed_collate_preflight.get("system2_replay_role") or []
    )
    expected_preflight_roles = set(_MIXED_SFT_ROLES)
    if args.pairwise_stop_margin_weight > 0:
        expected_preflight_roles.add(_PAIRED_POSITIVE_ROLE)
    if observed_preflight_roles != expected_preflight_roles:
        raise RuntimeError(
            "Mixed System2 collate preflight lost source roles: "
            f"expected={expected_preflight_roles} observed={observed_preflight_roles}"
        )
    del mixed_collate_preflight
    LOGGER.info(
        "Mixed System2 collate preflight passed: roles=%s",
        tuple(sorted(expected_preflight_roles)),
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": collator,
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
    }
    generator = torch.Generator().manual_seed(args.seed + distributed.rank)
    train_sampler = (
        DistributedSampler(
            mixed_dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        if distributed.enabled
        else None
    )
    train_loader = DataLoader(
        mixed_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        generator=generator,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
        **loader_kwargs,
    )
    train_evaluation_loader = DataLoader(
        train_evaluation_dataset,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
        **loader_kwargs,
    )

    model = build_model(
        cfg,
        device=str(device),
        verbose=distributed.is_main,
        enable_action_head=False,
    ).to(device)
    integration = model.qwen2_5_vl
    integration._load_model()
    structured_token_contract = integration.structured_view_token_contract()
    if structured_token_contract["classes"][0] != "stop":
        raise RuntimeError(
            f"Structured System2 class order changed: {structured_token_contract}"
        )
    structured_class_token_ids = tuple(
        int(token_id) for token_id in structured_token_contract["class_token_ids"]
    )
    base_path = args.base_checkpoint.expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"Missing base checkpoint: {base_path}")
    base_state = _checkpoint_state(base_path)
    matched = assert_complete_lora_checkpoint_match(
        model, base_state, checkpoint_path=str(base_path)
    )
    lora_state = extract_lora_checkpoint_state(base_state)
    _missing, _unexpected, loaded = _load_normalized_state_dict(model, lora_state)
    if matched != 224 or loaded != 224 or len(lora_state) != 224:
        raise RuntimeError(
            f"All-layer LoRA load failed: matched={matched} loaded={loaded} "
            f"checkpoint={len(lora_state)}"
        )
    del base_state, lora_state

    model.requires_grad_(False)
    integration.activate_lora_adapters(
        (DEFAULT_LORA_ADAPTER_NAME,),
        trainable_adapters=(DEFAULT_LORA_ADAPTER_NAME,),
    )
    named_parameters = integration.lora_adapter_named_parameters(
        DEFAULT_LORA_ADAPTER_NAME
    )
    trainable_ids = {id(parameter) for _name, parameter in named_parameters}
    unexpected_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in trainable_ids
    ]
    if len(named_parameters) != 224 or unexpected_trainable:
        raise RuntimeError(
            "System2 continuation trainable scope is not exactly the 224 default "
            f"LoRA tensors: lora={len(named_parameters)} unexpected={unexpected_trainable[:8]}"
        )
    initial_fingerprint = integration.lora_adapter_fingerprint(
        DEFAULT_LORA_ADAPTER_NAME
    )
    _assert_fingerprint_synchronized(
        initial_fingerprint,
        distributed,
        label="initial",
    )
    references = {
        name: parameter.detach().float().clone()
        for name, parameter in named_parameters
    }
    reference_denominator = sum(
        value.square().sum() for value in references.values()
    ).clamp_min(1e-12)
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in named_parameters
    }
    LOGGER.info(
        "Verified trainable scope: tensors=%d params=%d active=%s",
        len(named_parameters),
        sum(parameter.numel() for _name, parameter in named_parameters),
        integration.active_lora_adapters(),
    )
    distributed.barrier()

    baseline_metrics, baseline_predictions, validation_keys = _evaluate(
        integration, validation_loader, device
    )
    train_baseline_metrics, _train_baseline_predictions, train_evaluation_keys = (
        _evaluate(integration, train_evaluation_loader, device)
    )
    LOGGER.info("Baseline structured validation: %s", baseline_metrics)
    LOGGER.info("Baseline train-scene evaluation: %s", train_baseline_metrics)

    optimizer = torch.optim.AdamW(
        [parameter for _name, parameter in named_parameters],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    max_steps = 1 if args.dry_run else args.max_steps
    if args.dry_run:
        training_batches = (
            collator([mixed_dataset[index]]) for index in mixed_preflight_indices
        )
        effective_grad_accum_steps = len(mixed_preflight_indices)
    else:
        training_batches = iter(train_loader)
        effective_grad_accum_steps = args.grad_accum_steps
    warmup_steps = round(max_steps * args.warmup_ratio)
    min_ratio = args.min_learning_rate / args.learning_rate

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, min_ratio)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    integration.model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    micro_steps = 0
    last_sft = last_ce = last_rejection = last_pair = last_l2 = last_grad = 0.0
    window_sft = window_ce = window_rejection = window_pair = window_l2 = 0.0
    window_roles: dict[str, int] = {}
    observed_source_counts: dict[str, int] = {}
    validation_history: list[dict[str, Any]] = []
    for batch in training_batches:
        with make_autocast_context(device, "bf16"):
            (
                sft_loss,
                token_ce_loss,
                stop_rejection_loss,
                pairwise_stop_margin_loss,
            ) = _sft_objective(
                integration,
                batch,
                device,
                structured_class_token_ids=structured_class_token_ids,
                pairwise_stop_margin_weight=args.pairwise_stop_margin_weight,
                pairwise_stop_margin_gap=args.pairwise_stop_margin_gap,
            )
            l2_sp_loss = _relative_l2_sp(
                named_parameters,
                references,
                reference_denominator,
            )
            loss = (
                sft_loss + args.l2_sp_weight * l2_sp_loss
            ) / effective_grad_accum_steps
        if not bool(torch.isfinite(loss.detach())):
            raise RuntimeError(f"Non-finite System2 SFT loss: {loss.detach().item()}")
        loss.backward()
        micro_steps += 1
        window_sft += float(sft_loss.detach().item())
        window_ce += float(token_ce_loss.detach().item())
        window_rejection += float(stop_rejection_loss.detach().item())
        window_pair += float(pairwise_stop_margin_loss.detach().item())
        window_l2 += float(l2_sp_loss.detach().item())
        for role in batch.get("system2_replay_role") or []:
            window_roles[str(role)] = window_roles.get(str(role), 0) + 1
            observed_source_counts[str(role)] = (
                observed_source_counts.get(str(role), 0) + 1
            )
        if micro_steps % effective_grad_accum_steps:
            continue
        _synchronize_gradients(named_parameters, distributed)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _name, parameter in named_parameters],
            args.grad_clip,
        )
        if not bool(torch.isfinite(grad_norm.detach())) or grad_norm.item() <= 0:
            raise RuntimeError(f"Invalid System2 LoRA gradient norm: {grad_norm.item()}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        reduced_values, reduced_window_roles = _reduce_training_window(
            (
                window_sft / effective_grad_accum_steps,
                window_ce / effective_grad_accum_steps,
                window_rejection / effective_grad_accum_steps,
                window_pair / effective_grad_accum_steps,
                window_l2 / effective_grad_accum_steps,
            ),
            window_roles,
            distributed,
        )
        last_sft, last_ce, last_rejection, last_pair, last_l2 = reduced_values
        last_grad = float(grad_norm.detach().item())
        if distributed.is_main and (
            optimizer_steps <= 3 or optimizer_steps % args.log_interval == 0
        ):
            LOGGER.info(
                "Step %d/%d sft=%.6f token_ce=%.6f reject_stop=%.6f "
                "pair_rank=%.6f "
                "l2sp=%.8f grad=%.5f lr=%.3g roles=%s",
                optimizer_steps,
                max_steps,
                last_sft,
                last_ce,
                last_rejection,
                last_pair,
                last_l2,
                last_grad,
                scheduler.get_last_lr()[0],
                dict(sorted(reduced_window_roles.items())),
            )
        window_sft = window_ce = window_rejection = window_pair = window_l2 = 0.0
        window_roles = {}
        if (
            args.validation_interval > 0
            and optimizer_steps % args.validation_interval == 0
            and optimizer_steps < max_steps
        ):
            interval_metrics, interval_predictions, interval_keys = _evaluate(
                integration,
                validation_loader,
                device,
            )
            if interval_keys != validation_keys:
                raise RuntimeError("System2 interval validation order changed")
            interval_metrics = _with_quality_metrics(
                interval_metrics,
                interval_predictions,
                baseline_metrics=baseline_metrics,
                baseline_predictions=baseline_predictions,
                targets=validation_dataset.targets,
                original_terminals=validation_dataset.original_terminals,
            )
            validation_history.append(
                {"optimizer_step": optimizer_steps, "metrics": interval_metrics}
            )
            if distributed.is_main:
                LOGGER.info(
                    "Validation at step %d: %s",
                    optimizer_steps,
                    interval_metrics,
                )
            if args.save_validation_checkpoints and distributed.is_main:
                interval_state = _current_lora_state(model)
                interval_checkpoint = {
                    "schema": CHECKPOINT_SCHEMA,
                    "trainable_state_dict": interval_state,
                    "config": cfg,
                    "base_contract": {
                        "checkpoint": str(base_path),
                        "lora_tensors": matched,
                        "initial_lora_fingerprint": initial_fingerprint,
                    },
                    "training": {
                        "optimizer_steps": optimizer_steps,
                        "scheduled_max_steps": max_steps,
                        "learning_rate": args.learning_rate,
                        "min_learning_rate": args.min_learning_rate,
                        "l2_sp_weight": args.l2_sp_weight,
                        "pairwise_stop_margin_weight": (
                            args.pairwise_stop_margin_weight
                        ),
                        "pairwise_stop_margin_gap": args.pairwise_stop_margin_gap,
                        "seed": args.seed,
                    },
                    "validation": {
                        "baseline": baseline_metrics,
                        "at_checkpoint": interval_metrics,
                        "keys": validation_keys,
                    },
                }
                interval_dir = output_dir / "validation_checkpoints"
                interval_dir.mkdir(parents=True, exist_ok=True)
                interval_path = interval_dir / f"step_{optimizer_steps:06d}.pth"
                _atomic_torch_save(interval_checkpoint, interval_path)
                LOGGER.info(
                    "Saved 224-LoRA validation checkpoint: %s",
                    interval_path,
                )
            distributed.barrier()
            integration.model.train()
        if optimizer_steps >= max_steps:
            break
    if optimizer_steps != max_steps:
        raise RuntimeError(
            f"Training loader exhausted at {optimizer_steps}/{max_steps} optimizer steps"
        )

    max_parameter_delta = max(
        float((parameter.detach() - initial_parameters[name]).abs().max().item())
        for name, parameter in named_parameters
    )
    final_fingerprint = integration.lora_adapter_fingerprint(
        DEFAULT_LORA_ADAPTER_NAME
    )
    _assert_fingerprint_synchronized(
        final_fingerprint,
        distributed,
        label="final",
    )
    if max_parameter_delta <= 0 or final_fingerprint == initial_fingerprint:
        raise RuntimeError("System2 continuation did not update the default navigation LoRA")
    local_peak_memory_gib = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    peak_memory_tensor = torch.tensor(
        [local_peak_memory_gib],
        device=device,
        dtype=torch.float64,
    )
    if distributed.enabled:
        dist.all_reduce(peak_memory_tensor, op=dist.ReduceOp.MAX)
    peak_memory_gib = float(peak_memory_tensor.item())
    if args.dry_run:
        if distributed.is_main:
            LOGGER.info(
                "REAL System2 on-policy SFT dry-run passed: steps=%d sft=%.6f "
                "token_ce=%.6f reject_stop=%.6f pair_rank=%.6f "
                "l2sp=%.8f grad=%.5f "
                "max_delta=%.6g peak_memory=%.2fGiB roles=%s",
                optimizer_steps,
                last_sft,
                last_ce,
                last_rejection,
                last_pair,
                last_l2,
                last_grad,
                max_parameter_delta,
                peak_memory_gib,
                sorted(expected_preflight_roles),
            )
        distributed.barrier()
        return 0

    final_metrics, final_predictions, final_keys = _evaluate(
        integration, validation_loader, device
    )
    train_final_metrics, _train_final_predictions, train_final_keys = _evaluate(
        integration, train_evaluation_loader, device
    )
    if final_keys != validation_keys:
        raise RuntimeError("System2 validation order changed between baseline and final")
    if train_final_keys != train_evaluation_keys:
        raise RuntimeError("System2 train-scene evaluation order changed")
    final_metrics = _with_quality_metrics(
        final_metrics,
        final_predictions,
        baseline_metrics=baseline_metrics,
        baseline_predictions=baseline_predictions,
        targets=validation_dataset.targets,
        original_terminals=validation_dataset.original_terminals,
    )
    quality_passed = bool(final_metrics["quality_passed"])
    validation_history.append(
        {"optimizer_step": optimizer_steps, "metrics": final_metrics}
    )
    observed_source_counts = _reduce_role_counts(
        observed_source_counts,
        distributed,
    )
    if distributed.is_main:
        LOGGER.info("Observed mixed source counts: %s", observed_source_counts)
        LOGGER.info("Final structured validation: %s", final_metrics)
        LOGGER.info("Final train-scene evaluation: %s", train_final_metrics)

    trained_state = _current_lora_state(model)
    if assert_complete_lora_checkpoint_match(model, trained_state) != 224:
        raise RuntimeError("Saved System2 continuation is not a complete all-layer LoRA")
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "trainable_state_dict": trained_state,
        "config": cfg,
        "base_contract": {
            "checkpoint": str(base_path),
            "checkpoint_file_sha256": _file_sha256(base_path),
            "lora_tensors": matched,
            "initial_lora_fingerprint": initial_fingerprint,
            "final_lora_fingerprint": final_fingerprint,
        },
        "data_contract": {
            "native_root": str(native_cfg["data"]["root"]),
            "native_samples": len(native_dataset),
            "rollout_roots": [str(path) for path in rollout_roots],
            "rollout_validation_report": rollout_report_contract,
            "rollout_train_samples": len(rollout_train),
            "rollout_validation_samples": len(rollout_validation),
            "train_evaluation_samples": len(train_evaluation_dataset),
            "mixed_source_counts": mixed_dataset.source_counts(),
            "rollout_pool_sizes": mixed_dataset.pool_sizes(),
            "regular_negative_mining": (
                mixed_dataset.regular_negative_mining_contract()
            ),
            "false_stop_pairing": mixed_dataset.false_stop_pairing_contract(),
            "observed_source_counts": dict(sorted(observed_source_counts.items())),
            "slot_ratio": {
                "native": args.native_slots,
                "onpolicy_positive": args.onpolicy_positive_slots,
                "onpolicy_regular_negative": args.onpolicy_regular_negative_slots,
                "onpolicy_false_stop_negative": (
                    args.onpolicy_false_stop_negative_slots
                ),
            },
            "holdout_scene_fraction": args.holdout_scene_fraction,
        },
        "training": {
            "optimizer_steps": optimizer_steps,
            "batch_size": args.batch_size,
            "batch_size_per_rank": args.batch_size,
            "global_batch_size": args.batch_size * distributed.world_size,
            "world_size": distributed.world_size,
            "grad_accum_steps": args.grad_accum_steps,
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "l2_sp_weight": args.l2_sp_weight,
            "pairwise_stop_margin_weight": args.pairwise_stop_margin_weight,
            "pairwise_stop_margin_gap": args.pairwise_stop_margin_gap,
            "last_sft_loss": last_sft,
            "last_ce_loss": last_ce,
            "last_stop_rejection_loss": last_rejection,
            "last_pairwise_stop_margin_loss": last_pair,
            "last_relative_l2_sp": last_l2,
            "max_parameter_delta": max_parameter_delta,
            "peak_memory_gib": peak_memory_gib,
            "seed": args.seed,
        },
        "validation": {
            "baseline": baseline_metrics,
            "final": final_metrics,
            "train_scene_baseline": train_baseline_metrics,
            "train_scene_final": train_final_metrics,
            "history": validation_history,
            "keys": validation_keys,
            "train_scene_keys": train_evaluation_keys,
        },
    }
    if not distributed.is_main:
        distributed.barrier()
        return 0
    checkpoint_path = output_dir / "latest.pth"
    _atomic_torch_save(checkpoint, checkpoint_path)
    summary = {key: value for key, value in checkpoint.items() if key != "trainable_state_dict"}
    summary["checkpoint"] = str(checkpoint_path)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "System2 on-policy SFT complete: quality_passed=%s checkpoint=%s",
        quality_passed,
        checkpoint_path,
    )
    distributed.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

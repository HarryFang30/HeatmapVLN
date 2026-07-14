#!/usr/bin/env python3
"""Task-4 pilot: preserve Stage1-S2 while adapting heatmap head and LoRA.

The pilot deliberately uses two independent data streams:

* random-walk projected-history heatmaps;
* R2R panoramic System2 structured waypoint SFT rehearsal.

Each optimizer step sees the same heatmap sample in every branch.  The joint
branch can either accumulate the legacy token-pooled hard-label CE gradient or
preserve the initial model's correct-label log probabilities with a token-
pooled FP32 MSE.  The latter caches the teacher values for the exact planned
batches before any update.  The rehearsal stream walks the full scene-
partitioned candidate pool without replacement inside each deterministic
epoch.  Only heatmap-head parameters and Qwen LoRA layers 0--20 are trainable;
later LoRA layers are carried through unchanged because the deepest heatmap
hook is layer 20.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import pickle
import random
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.system2_sft_sanity_check import (
    generate_from_messages,
    image_size_from_cfg,
    make_generation_messages,
    parse_generated_text,
    parse_target,
    target_texts_for_sample,
    update_metrics,
)
from scripts.tools.audit_heatmap_lora_gradient import (
    effective_delta_norms,
    snapshot,
)
from scripts.tools.build_balanced_sft_view import MANIFEST_NAME
from scripts.tools.diagnose_heatmap_shortcuts import (
    build_dataset as build_heatmap_dataset,
)
from scripts.tools.diagnose_heatmap_shortcuts import (
    evaluate as evaluate_heatmap,
)
from scripts.tools.diagnose_heatmap_shortcuts import (
    forward_loss as forward_heatmap_loss,
)
from scripts.tools.diagnose_heatmap_shortcuts import (
    heatmap_head_state_dict,
    load_heatmap_head_checkpoint,
    load_stage1_s2_lora,
    make_loss,
    scene_stratified_indices,
    selection_contract,
    set_seed,
    state_hash,
    transform_sample,
)
from scripts.tools.diagnose_heatmap_shortcuts import (
    load_config as load_heatmap_config,
)
from scripts.training import (
    build_model,
)
from scripts.training import (
    load_config as load_training_config,
)

from src.data.factory import build_dataset as build_training_dataset
from src.data.panoramic_tokenized_collator import (
    IGNORE_INDEX,
    PanoramicTokenizedCollator,
)

LOGGER = logging.getLogger("heatmap_joint_pilot")
MODES = ("head-only", "heatmap-lora", "joint-rehearsal")
REHEARSAL_OBJECTIVES = ("hard-ce", "correct-label-logprob-mse")
SPARSE_CORRECT_LOGPROB_BACKENDS = (
    "hf_logits_to_keep_tensor_predictor_union_v1",
    "lm_head_pre_hook_predictor_union_v1",
)
LORA_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--config", required=True, help="Task-3 heatmap config.")
    parser.add_argument("--checkpoint", required=True, help="Current Stage1-S2 checkpoint.")
    parser.add_argument("--data-root", required=True, help="Random-walk heatmap dataset.")
    parser.add_argument("--sft-config", required=True, help="Manifest config from the Stage1-S2 run.")
    parser.add_argument("--sft-data-root", required=True, help="R2R panoramic training root.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-history", type=int, default=2)
    parser.add_argument("--max-clip-id", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--heatmap-train-samples", type=int, default=128)
    parser.add_argument("--heatmap-val-samples", type=int, default=64)
    parser.add_argument(
        "--sft-train-samples",
        type=int,
        default=0,
        help="Optional capped rehearsal pool for smoke tests; 0 uses every candidate.",
    )
    parser.add_argument(
        "--sft-batch-size",
        type=int,
        default=4,
        help="SFT examples collated into one token-pooled rehearsal forward.",
    )
    parser.add_argument("--sft-val-samples", type=int, default=32)
    parser.add_argument("--sft-generation-samples", type=int, default=16)
    parser.add_argument("--sft-holdout-scenes", type=int, default=7)
    parser.add_argument("--sft-max-clips", type=int, default=0)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--rehearsal-objective",
        choices=REHEARSAL_OBJECTIVES,
        default="hard-ce",
        help=(
            "hard-ce preserves the existing pilot exactly; correct-label-logprob-mse "
            "matches the initial model's cached correct-label log probabilities."
        ),
    )
    parser.add_argument("--rehearsal-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-trainable-lora-layer", type=int, default=20)
    parser.add_argument("--gradient-cosine-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--milestone-steps",
        default="0,25,50,100",
        help=(
            "Comma-separated isolated-checkpoint steps. Mid-run evaluation is never "
            "performed in the training process; milestones.json records fresh-process "
            "evaluation argv for each checkpoint."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--coord-tolerance", type=float, default=15.0)
    parser.add_argument(
        "--interventions",
        default="blank-images,history-shuffle,current-shuffle",
        help="Comma-separated final heatmap interventions; empty disables them.",
    )
    parser.add_argument(
        "--head-checkpoint",
        default=None,
        help="Evaluation-only reuse of the matched Task-3 Full head (requires train-steps=0).",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip autoregressive retention sanity; teacher-forced CE is still evaluated.",
    )
    return parser.parse_args()


def _hash_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _tensor_state_hash(state: dict[str, torch.Tensor]) -> str:
    return state_hash(state)


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fp32_tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(b"task4_fp32_tensor_v1\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class RNGState:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: dict[int, torch.Tensor]


def capture_rng_state(cuda_devices: Iterable[int] = ()) -> RNGState:
    cuda_device_indices = sorted({int(index) for index in cuda_devices})
    return RNGState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.random.get_rng_state().clone(),
        # Capture only devices used by this process. get_rng_state_all() can
        # initialize CUDA contexts on unrelated GPUs in a shared 8-GPU node.
        torch_cuda={index: torch.cuda.get_rng_state(index).clone() for index in cuda_device_indices}
        if torch.cuda.is_available()
        else {},
    )


def restore_rng_state(state: RNGState) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.random.set_rng_state(state.torch_cpu)
    for index, cuda_state in state.torch_cuda.items():
        torch.cuda.set_rng_state(cuda_state, device=index)


def rng_state_sha256(state: RNGState) -> str:
    digest = hashlib.sha256()
    digest.update(b"task4_rng_state_v1\0")
    digest.update(pickle.dumps(state.python, protocol=4))
    numpy_algorithm, numpy_keys, numpy_position, numpy_gaussian, numpy_cached = state.numpy
    digest.update(str(numpy_algorithm).encode("ascii"))
    digest.update(np.asarray(numpy_keys).tobytes(order="C"))
    digest.update(
        pickle.dumps(
            (numpy_position, numpy_gaussian, numpy_cached),
            protocol=4,
        )
    )
    digest.update(state.torch_cpu.detach().cpu().numpy().tobytes(order="C"))
    for index, cuda_state in sorted(state.torch_cuda.items()):
        digest.update(str(index).encode("ascii"))
        digest.update(cuda_state.detach().cpu().numpy().tobytes(order="C"))
    return digest.hexdigest()


def module_mode_contract(model: torch.nn.Module) -> dict[str, bool]:
    return {name: bool(module.training) for name, module in model.named_modules()}


def restore_module_modes(
    model: torch.nn.Module,
    contract: dict[str, bool],
) -> None:
    modules = dict(model.named_modules())
    if modules.keys() != contract.keys():
        raise RuntimeError("Model module topology changed while restoring training modes")
    # Direct assignment preserves deliberately mixed train/eval submodule modes;
    # calling model.train(...) recursively would erase that distinction.
    for name, training in contract.items():
        modules[name].training = bool(training)


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    temporary.replace(path)


def requested_milestone_steps(specification: str) -> list[int]:
    values: set[int] = set()
    for raw_value in specification.split(","):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        value = int(raw_value)
        if value < 0:
            raise ValueError(f"Milestone steps must be non-negative, got {value}")
        values.add(value)
    return sorted(values)


def effective_milestone_steps(specification: str, train_steps: int) -> list[int]:
    if train_steps < 0:
        raise ValueError(f"Train steps must be non-negative, got {train_steps}")
    return [step for step in requested_milestone_steps(specification) if step <= train_steps]


def load_pilot_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_heatmap_config(args)
    llm_cfg = cfg["model"]["llm"]
    llm_cfg["gradient_checkpointing"] = args.mode != "head-only"
    llm_cfg["lora_dropout"] = 0.0
    heatmap_cfg = cfg["model"]["heatmap"]
    heatmap_cfg["enable"] = True
    heatmap_cfg["heatmap_trains_backbone"] = args.mode != "head-only"
    cfg.setdefault("loss", {}).setdefault("heatmap_vln", {})["lambda_coord"] = 0.0
    return cfg


def lora_named_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    parameters = {name: parameter for name, parameter in model.named_parameters() if "lora_" in name}
    if not parameters:
        raise RuntimeError("No Qwen LoRA parameters were materialized")
    return parameters


def lora_layer(name: str) -> int:
    match = LORA_LAYER_RE.search(name)
    if match is None:
        raise RuntimeError(f"Cannot determine LoRA layer from parameter name: {name}")
    return int(match.group(1))


def configure_trainable_parameters(
    model: torch.nn.Module,
    mode: str,
    max_lora_layer: int,
) -> tuple[list[torch.nn.Parameter], dict[str, torch.nn.Parameter], list[dict[str, Any]]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    head_parameters = [
        parameter for name, parameter in model.heatmap_vln.named_parameters() if not name.startswith("qwen.")
    ]
    for parameter in head_parameters:
        parameter.requires_grad_(True)

    all_lora = lora_named_parameters(model)
    trainable_lora: dict[str, torch.nn.Parameter] = {}
    if mode != "head-only":
        for name, parameter in all_lora.items():
            if lora_layer(name) <= max_lora_layer:
                parameter.requires_grad_(True)
                trainable_lora[name] = parameter

    if mode != "head-only" and not trainable_lora:
        raise RuntimeError("LoRA branch has no trainable LoRA tensors")
    groups = [{"name": "heatmap_head", "params": head_parameters}]
    if trainable_lora:
        groups.append({"name": "reachable_lora", "params": list(trainable_lora.values())})
    return head_parameters, trainable_lora, groups


def build_optimizer(
    args: argparse.Namespace,
    head_parameters: list[torch.nn.Parameter],
    trainable_lora: dict[str, torch.nn.Parameter],
) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = [
        {
            "name": "heatmap_head",
            "params": head_parameters,
            "lr": args.head_learning_rate,
        }
    ]
    if trainable_lora:
        groups.append(
            {
                "name": "reachable_lora",
                "params": list(trainable_lora.values()),
                "lr": args.lora_learning_rate,
            }
        )
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def select_scene_partition(
    dataset: Any,
    *,
    seed: int,
    holdout_scene_count: int,
) -> tuple[list[str], list[str]]:
    scenes = sorted({clip.parent.name for clip in dataset.clips})
    if len(scenes) < 2:
        raise RuntimeError("S1-S2 rehearsal requires at least two scenes")
    holdout_scene_count = min(max(1, holdout_scene_count), len(scenes) - 1)
    ranked = sorted(
        scenes,
        key=lambda scene: hashlib.sha256(f"{seed}:{scene}".encode()).hexdigest(),
    )
    holdout = sorted(ranked[:holdout_scene_count])
    rehearsal = sorted(set(scenes) - set(holdout))
    return rehearsal, holdout


def select_indices_from_scenes(
    dataset: Any,
    scenes: list[str],
    limit: int,
    *,
    stop_fraction: float = 0.25,
) -> list[int]:
    if not 0.0 <= stop_fraction <= 1.0:
        raise ValueError(f"stop_fraction must be in [0, 1], got {stop_fraction}")
    allowed = set(scenes)
    by_category_scene: dict[str, dict[str, list[int]]] = {
        "pixel": defaultdict(list),
        "stop": defaultdict(list),
    }
    seen_stop_keys: set[tuple[int, int]] = set()
    for sample_idx, (clip_idx, frame_idx) in enumerate(dataset.sample_index):
        scene = dataset.clips[clip_idx].parent.name
        if scene not in allowed:
            continue
        category = sft_index_category(dataset, clip_idx, frame_idx)
        if category == "stop":
            # The source index repeats each terminal STOP several times. Keep
            # one copy per physical clip/frame, then control its weight through
            # an explicit category quota.
            sample_key = (int(clip_idx), int(frame_idx))
            if sample_key in seen_stop_keys:
                continue
            seen_stop_keys.add(sample_key)
        by_category_scene[category][scene].append(sample_idx)

    def scene_round_robin(category: str) -> list[int]:
        by_scene = by_category_scene[category]
        ordered: list[int] = []
        cursors = {scene: 0 for scene in sorted(by_scene)}
        while True:
            progressed = False
            for scene in sorted(by_scene):
                cursor = cursors[scene]
                if cursor >= len(by_scene[scene]):
                    continue
                ordered.append(by_scene[scene][cursor])
                cursors[scene] += 1
                progressed = True
            if not progressed:
                return ordered

    pixel_candidates = scene_round_robin("pixel")
    stop_candidates = scene_round_robin("stop")
    if limit >= 2 and pixel_candidates and stop_candidates:
        stop_target = min(max(1, round(limit * stop_fraction)), limit - 1)
    else:
        stop_target = min(round(limit * stop_fraction), limit)
    pixel_target = limit - stop_target
    selected_pixel = pixel_candidates[:pixel_target]
    selected_stop = stop_candidates[:stop_target]

    # If one category is genuinely exhausted, fill the requested diagnostic
    # size from the other category without duplicating a physical sample.
    deficit = limit - len(selected_pixel) - len(selected_stop)
    if deficit > 0:
        remaining = pixel_candidates[len(selected_pixel) :] + stop_candidates[len(selected_stop) :]
        selected_extra = remaining[:deficit]
    else:
        selected_extra = []
    # The underlying SFT index is already deterministically shuffled. Sorting
    # the selected indices spreads both categories through the training cycle
    # while preserving the category-wise scene-balanced membership.
    selected = sorted(selected_pixel + selected_stop + selected_extra)
    if not selected:
        raise RuntimeError(f"No S1-S2 samples selected from scenes={sorted(allowed)}")
    return selected


def all_indices_from_scenes(dataset: Any, scenes: list[str]) -> list[int]:
    """Return the complete SFT index stream for ``scenes``.

    Unlike the small diagnostic selector above, this intentionally preserves
    the source dataset's STOP oversampling.  The corrected pilot should match
    the original Stage1-S2 sample distribution and let the batched Qwen loss
    pool all non-ignored tokens, rather than assigning every physical example
    one equally weighted optimizer step.
    """
    allowed = set(scenes)
    indices = [
        sample_idx
        for sample_idx, (clip_idx, _frame_idx) in enumerate(dataset.sample_index)
        if dataset.clips[clip_idx].parent.name in allowed
    ]
    if not indices:
        raise RuntimeError(f"No S1-S2 candidates found for scenes={sorted(allowed)}")
    return indices


@dataclass(frozen=True)
class StreamBatch:
    epoch: int
    start_position: int
    indices: tuple[int, ...]


class DeterministicEpochBatchStream:
    """Deterministic, process-local batches with no replacement per epoch.

    Ordering is SHA256-ranked instead of using a global RNG.  Consequently
    planning or advancing this stream cannot perturb LoRA/dropout randomness,
    and heatmap-only and joint branches can prove that they contracted the
    exact same rehearsal schedule.
    """

    algorithm = "sha256_epoch_rank_no_replacement_v1"

    def __init__(self, indices: Iterable[int], *, batch_size: int, seed: int):
        self.indices = tuple(int(index) for index in indices)
        if not self.indices:
            raise ValueError("DeterministicEpochBatchStream requires at least one index")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("Dataset indices must be unique even when physical samples repeat")
        if batch_size <= 0:
            raise ValueError(f"SFT batch size must be positive, got {batch_size}")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.position = 0
        self._order = self._order_for_epoch(0)

    def _order_for_epoch(self, epoch: int) -> tuple[int, ...]:
        return tuple(
            sorted(
                self.indices,
                key=lambda index: (
                    hashlib.sha256(f"{self.seed}:{epoch}:{index}".encode()).digest(),
                    index,
                ),
            )
        )

    def next_batch(self) -> StreamBatch:
        if self.position >= len(self._order):
            self.epoch += 1
            self.position = 0
            self._order = self._order_for_epoch(self.epoch)
        start = self.position
        stop = min(start + self.batch_size, len(self._order))
        self.position = stop
        return StreamBatch(
            epoch=self.epoch,
            start_position=start,
            indices=self._order[start:stop],
        )

    def planned_batches(self, steps: int) -> list[StreamBatch]:
        if steps < 0:
            raise ValueError(f"Planned steps must be non-negative, got {steps}")
        clone = DeterministicEpochBatchStream(
            self.indices,
            batch_size=self.batch_size,
            seed=self.seed,
        )
        return [clone.next_batch() for _ in range(steps)]


def sft_index_category(dataset: Any, clip_idx: int, frame_idx: int) -> str:
    valid_frames = getattr(dataset, "_clip_valid_frames", {}).get(clip_idx, [])
    if valid_frames and int(frame_idx) == int(valid_frames[-1]):
        return "stop"
    return "pixel"


def sft_sample_identity(dataset: Any, index: int) -> str:
    clip_idx, frame_idx = dataset.sample_index[index]
    clip = dataset.clips[clip_idx]
    try:
        relative = clip.relative_to(dataset.root).as_posix()
    except ValueError:
        relative = clip.as_posix()
    return f"{relative}:frame={frame_idx}"


def generic_selection_contract(dataset: Any, indices: list[int]) -> dict[str, Any]:
    sample_ids = []
    scenes = set()
    category_counts: Counter = Counter()
    for index in indices:
        clip_idx, frame_idx = dataset.sample_index[index]
        clip = dataset.clips[clip_idx]
        sample_ids.append(sft_sample_identity(dataset, index))
        scenes.add(clip.parent.name)
        category_counts[sft_index_category(dataset, clip_idx, frame_idx)] += 1
    return {
        "sample_count": len(sample_ids),
        "unique_physical_sample_count": len(set(sample_ids)),
        "duplicate_physical_sample_count": len(sample_ids) - len(set(sample_ids)),
        "sample_identity_sha256": _hash_strings(sample_ids),
        "sample_identities": sample_ids,
        "scenes": sorted(scenes),
        "category_counts": {category: int(category_counts[category]) for category in ("pixel", "stop")},
    }


def rehearsal_stream_contract(
    dataset: Any,
    indices: list[int],
    *,
    batch_size: int,
    seed: int,
    train_steps: int,
) -> dict[str, Any]:
    stream = DeterministicEpochBatchStream(indices, batch_size=batch_size, seed=seed)
    batches = stream.planned_batches(train_steps)
    flattened = [index for batch in batches for index in batch.indices]
    identities = [sft_sample_identity(dataset, index) for index in flattened]
    category_counts: Counter = Counter()
    batch_records = []
    for batch in batches:
        categories: Counter = Counter()
        batch_identities = []
        for index in batch.indices:
            clip_idx, frame_idx = dataset.sample_index[index]
            category = sft_index_category(dataset, clip_idx, frame_idx)
            categories[category] += 1
            category_counts[category] += 1
            batch_identities.append(sft_sample_identity(dataset, index))
        batch_records.append(
            {
                "epoch": batch.epoch,
                "start_position": batch.start_position,
                "dataset_indices": list(batch.indices),
                "sample_identities": batch_identities,
                "category_counts": {category: int(categories[category]) for category in ("pixel", "stop")},
            }
        )
    return {
        "algorithm": stream.algorithm,
        "seed": int(seed),
        "batch_size": int(batch_size),
        "no_replacement_within_epoch": True,
        "candidate_count": len(indices),
        "candidate_dataset_index_sha256": _hash_strings(str(index) for index in indices),
        "planned_steps": int(train_steps),
        "planned_sample_count": len(flattened),
        "planned_epoch_count": len({batch.epoch for batch in batches}),
        "planned_dataset_index_sha256": _hash_strings(str(index) for index in flattened),
        "planned_sample_identity_sha256": _hash_strings(identities),
        "planned_category_counts": {category: int(category_counts[category]) for category in ("pixel", "stop")},
        "planned_batches": batch_records,
    }


def sft_dataset_contract(dataset: Any) -> dict[str, Any]:
    root = Path(dataset.root).resolve()
    identities = sorted(clip.relative_to(root).as_posix() for clip in dataset.clips)
    per_scene = Counter(clip.parent.name for clip in dataset.clips)
    contract: dict[str, Any] = {
        "clip_count": len(identities),
        "scene_count": len(per_scene),
        "scenes": sorted(per_scene),
        "per_scene_clip_counts": {scene: int(per_scene[scene]) for scene in sorted(per_scene)},
        "clip_identities": identities,
        "clip_identity_sha256": _hash_strings(identities),
        "balanced_view_manifest": None,
    }

    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest_identities = sorted(manifest.get("selected_clip_identities", []))
        if manifest_identities != identities:
            raise RuntimeError(
                "Loaded SFT clips do not exactly match the balanced-view manifest; "
                "do not combine --sft-max-clips truncation with an already balanced view"
            )
        contract["balanced_view_manifest"] = {
            "path": str(manifest_path),
            "source_root": manifest.get("source_root"),
            "selection_algorithm": manifest.get("selection_algorithm"),
            "selected_clip_identity_sha256": manifest.get("selected_clip_identity_sha256"),
            "total_clips": manifest.get("total_clips"),
            "scene_count": manifest.get("scene_count"),
        }
    return contract


def build_sft_dataset_and_collator(
    args: argparse.Namespace,
    model: torch.nn.Module,
) -> tuple[Any, PanoramicTokenizedCollator, dict[str, Any]]:
    cfg = load_training_config(args.sft_config)
    cfg["data"]["root"] = str(Path(args.sft_data_root).resolve())
    trajectory_cfg = cfg["data"].setdefault("trajectory", {})
    dataset = build_training_dataset(
        cfg,
        split="train",
        enable_augmentation=False,
        enable_trajectory_augmentation=False,
        load_traj_images=False,
        # Match the original Stage1-S2 non-heatmap training override. The
        # rehearsal stream only consumes Qwen tokens and LM labels.
        load_history_heatmap=False,
        require_sft_target=True,
        max_clips=args.sft_max_clips,
    )
    stage_cfg = cfg["training"]["stages"][0]
    collator = PanoramicTokenizedCollator(
        model.qwen2_5_vl.processor,
        n_traj_query=0,
        sft_mode=True,
        sft_include_turns=stage_cfg.get("sft_include_turns", False),
        sft_include_forward=stage_cfg.get("sft_include_forward", False),
        sft_protocol=trajectory_cfg.get("system2_sft_protocol", "direct"),
        structured_pano_output=trajectory_cfg.get("structured_pano_output", True),
        build_sft_labels=True,
        max_seq_length=int(cfg["model"]["llm"].get("max_seq_length", 8192)),
        include_heatmap_targets=False,
        include_history_rel_poses=False,
        retain_raw_panoramic_views=False,
        compute_pano_text_anchor_positions=False,
        heatmap_layout=False,
    )
    return dataset, collator, cfg


def exact_sft_sample(dataset: Any, index: int) -> dict[str, Any]:
    """Load one contracted SFT index without dataset-level fallback.

    ``VLNTrajectoryDataset.__getitem__`` retries arbitrary global indices when
    a projection or target fails. That is useful for stochastic training, but
    can silently cross the pilot's rehearsal/holdout scene boundary.
    """
    build_sample = getattr(dataset, "_build_sample", None)
    validate_target = getattr(dataset, "_result_has_system2_sft_target", None)
    if not callable(build_sample) or not callable(validate_target):
        raise TypeError(
            "Task-4 exact SFT retrieval requires a VLNTrajectoryDataset-like "
            "_build_sample/_result_has_system2_sft_target contract"
        )
    try:
        sample = build_sample(index)
    except Exception as exc:
        raise RuntimeError(f"Failed to load exact contracted SFT index={index}") from exc
    if not validate_target(sample):
        raise RuntimeError(f"Exact contracted SFT index={index} has no valid System2 target")
    return sample


def sft_forward_loss(
    model: torch.nn.Module,
    collator: PanoramicTokenizedCollator,
    samples: list[dict[str, Any]] | tuple[dict[str, Any], ...] | dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, int, list[int]]:
    """Run one genuinely batched, token-pooled Qwen rehearsal objective.

    Qwen's causal-LM ``outputs.loss`` is a single cross-entropy mean over every
    non-ignored shifted label in the complete batch.  Returning per-row token
    counts makes that reduction auditable and prevents a future refactor from
    silently reverting to equal-weight, one-example microsteps.
    """
    if isinstance(samples, dict):
        samples = [samples]
    samples = list(samples)
    if not samples:
        raise ValueError("S1-S2 rehearsal batch must contain at least one sample")
    batch = collator(samples)
    labels = batch["pano_inputs"]["labels"]
    if labels.ndim != 2 or labels.shape[0] != len(samples):
        raise RuntimeError(
            f"S1-S2 collator returned incompatible labels: shape={tuple(labels.shape)} batch_size={len(samples)}"
        )
    shifted_labels = labels[:, 1:]
    sample_label_tokens = [int(value) for value in (shifted_labels != IGNORE_INDEX).sum(dim=1).tolist()]
    label_tokens = sum(sample_label_tokens)
    if label_tokens <= 0:
        raise RuntimeError("S1-S2 rehearsal batch has no labeled assistant tokens")
    current = batch["current_frame"].to(device, non_blocking=True)
    output = model(
        video_frames=current.unsqueeze(1),
        current_observation=current,
        panoramic_inputs=batch["pano_inputs"],
        panoramic_num_histories=batch["pano_num_histories"],
        panoramic_text_anchor_positions=None,
        return_heatmaps=False,
        return_actions=False,
        return_lm_loss=True,
    )
    loss = output.get("lm_loss")
    if loss is None:
        raise RuntimeError("S1-S2 rehearsal forward returned no lm_loss")
    if loss.ndim != 0 or not torch.isfinite(loss.detach()).item():
        raise RuntimeError(f"S1-S2 rehearsal returned invalid pooled lm_loss={loss}")
    return loss, label_tokens, sample_label_tokens


def expected_correct_label_alignment(labels: torch.Tensor) -> dict[str, Any]:
    if labels.ndim != 2:
        raise ValueError(f"Expected rank-2 SFT labels, got {tuple(labels.shape)}")
    shifted_valid = labels[:, 1:] != IGNORE_INDEX
    sample_predictor_positions = [
        torch.nonzero(shifted_valid[row], as_tuple=False).flatten().tolist() for row in range(labels.shape[0])
    ]
    sample_correct_token_ids = [
        labels[row, torch.tensor(positions, dtype=torch.long) + 1].tolist() if positions else []
        for row, positions in enumerate(sample_predictor_positions)
    ]
    sample_label_tokens = [len(positions) for positions in sample_predictor_positions]
    label_tokens = sum(sample_label_tokens)
    if label_tokens <= 0:
        raise RuntimeError("S1-S2 preservation batch has no labeled assistant tokens")
    return {
        "schema": "shifted_correct_label_predictors_v1",
        "ignore_index": int(IGNORE_INDEX),
        "batch_size": int(labels.shape[0]),
        "sequence_length": int(labels.shape[1]),
        "sample_predictor_positions": sample_predictor_positions,
        "sample_correct_token_ids": sample_correct_token_ids,
        "sample_label_tokens": sample_label_tokens,
        "label_tokens": label_tokens,
    }


def sft_correct_label_logprobs(
    model: torch.nn.Module,
    collator: PanoramicTokenizedCollator,
    samples: list[dict[str, Any]] | tuple[dict[str, Any], ...] | dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, int, list[int], dict[str, Any]]:
    """Return row-major correct-label log probabilities and strict alignment.

    The integration computes only the union of predictor positions, using
    either HF's native tensor ``logits_to_keep`` API or a temporary input-slice
    pre-hook on the physical conditional-generation LM head for Transformers
    4.51. It then evaluates correct-label log probabilities in FP32. This
    wrapper independently reconstructs shifted-label alignment from the
    collator output and refuses any mismatch.
    """
    if isinstance(samples, dict):
        samples = [samples]
    samples = list(samples)
    if not samples:
        raise ValueError("S1-S2 preservation batch must contain at least one sample")
    batch = collator(samples)
    labels = batch["pano_inputs"]["labels"]
    expected = expected_correct_label_alignment(labels)
    current = batch["current_frame"].to(device, non_blocking=True)
    output = model(
        video_frames=current.unsqueeze(1),
        current_observation=current,
        panoramic_inputs=batch["pano_inputs"],
        panoramic_num_histories=batch["pano_num_histories"],
        panoramic_text_anchor_positions=None,
        return_heatmaps=False,
        return_actions=False,
        return_lm_loss=False,
        return_lm_correct_logprobs=True,
    )
    logprobs = output.get("lm_correct_label_logprobs")
    reported = output.get("lm_correct_label_alignment")
    if logprobs is None or reported is None:
        raise RuntimeError("S1-S2 preservation forward returned no correct-label log probabilities")
    if logprobs.ndim != 1 or int(logprobs.numel()) != expected["label_tokens"]:
        raise RuntimeError(
            "Correct-label log-prob count disagrees with shifted labels: "
            f"shape={tuple(logprobs.shape)} expected={expected['label_tokens']}"
        )
    if logprobs.dtype != torch.float32 or not torch.isfinite(logprobs.detach()).all().item():
        raise RuntimeError(f"Correct-label log probabilities must be finite FP32 values, got dtype={logprobs.dtype}")
    alignment_fields = (
        "schema",
        "ignore_index",
        "batch_size",
        "sequence_length",
        "sample_predictor_positions",
        "sample_correct_token_ids",
        "sample_label_tokens",
        "label_tokens",
    )
    reported_semantic = {field: reported.get(field) for field in alignment_fields}
    if reported_semantic != expected:
        raise RuntimeError(
            "Qwen correct-label alignment does not exactly match collator labels: "
            f"expected={expected} reported={reported_semantic}"
        )
    backend = reported.get("backend")
    if backend not in SPARSE_CORRECT_LOGPROB_BACKENDS:
        raise RuntimeError(f"Correct-label preservation requires the explicit sparse-logits backend, got {backend!r}")
    alignment_sha256 = _canonical_json_sha256(expected)
    telemetry = {
        **expected,
        "alignment_sha256": alignment_sha256,
        "backend": backend,
        "predictor_position_union": reported.get("predictor_position_union"),
        "returned_logits_shape": reported.get("returned_logits_shape"),
        "returned_logprob_dtype": reported.get("returned_logprob_dtype"),
        "conditional_generation_module": reported.get(
            "conditional_generation_module"
        ),
        "lm_head_module": reported.get("lm_head_module"),
        "native_logits_to_keep_explicit_signature": reported.get(
            "native_logits_to_keep_explicit_signature"
        ),
        "lm_head_hook_call_count": reported.get("lm_head_hook_call_count"),
        "lm_head_input_shape_before": reported.get("lm_head_input_shape_before"),
        "lm_head_input_shape_after": reported.get("lm_head_input_shape_after"),
        "lm_head_hook_removed": reported.get("lm_head_hook_removed"),
    }
    return (
        logprobs,
        expected["label_tokens"],
        expected["sample_label_tokens"],
        telemetry,
    )


def _logprob_stats(values: torch.Tensor) -> dict[str, float | int]:
    detached = values.detach().to(device="cpu", dtype=torch.float32)
    return {
        "count": int(detached.numel()),
        "mean": float(detached.mean().item()),
        "minimum": float(detached.min().item()),
        "maximum": float(detached.max().item()),
    }


def precompute_teacher_correct_logprob_cache(
    *,
    model: torch.nn.Module,
    dataset: Any,
    collator: PanoramicTokenizedCollator,
    planned_batches: list[StreamBatch],
    device: torch.device,
    output_path: Path,
    source_lora_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cache the initial teacher on every exact planned rehearsal batch.

    Values are stored as CPU FP32 tensors.  Python, NumPy, CPU/CUDA Torch RNG
    and every module's possibly mixed train/eval flag are restored exactly so
    caching cannot perturb the optimization trajectory.
    """
    if not planned_batches:
        raise ValueError("Teacher correct-label cache requires planned batches")
    cuda_rng_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )
    rng_before = capture_rng_state(cuda_rng_devices)
    rng_before_sha256 = rng_state_sha256(rng_before)
    modes_before = module_mode_contract(model)
    modes_before_sha256 = _canonical_json_sha256(modes_before)
    records: list[dict[str, Any]] = []
    nonzero_qwen_dropout = [
        {"module": name, "p": float(module.p)}
        for name, module in model.qwen2_5_vl.named_modules()
        if isinstance(module, torch.nn.Dropout) and float(module.p) != 0.0
    ]
    functional_dropout_settings = []
    seen_configs: set[int] = set()
    for module_name, module in model.qwen2_5_vl.named_modules():
        config = getattr(module, "config", None)
        for config_name, candidate in (
            ("config", config),
            ("config.text_config", getattr(config, "text_config", None)),
        ):
            if candidate is None or id(candidate) in seen_configs:
                continue
            seen_configs.add(id(candidate))
            for attribute in (
                "attention_dropout",
                "hidden_dropout",
                "hidden_dropout_prob",
                "dropout",
            ):
                value = getattr(candidate, attribute, None)
                if isinstance(value, (float, int)) and float(value) != 0.0:
                    functional_dropout_settings.append(
                        {
                            "module": module_name,
                            "config": config_name,
                            "attribute": attribute,
                            "value": float(value),
                        }
                    )
    if nonzero_qwen_dropout or functional_dropout_settings:
        raise RuntimeError(
            "Correct-label preservation requires deterministic train-mode Qwen "
            "for gradient checkpointing, but found nonzero dropout: "
            f"modules={nonzero_qwen_dropout} config={functional_dropout_settings}"
        )
    try:
        # Cache without a graph in eval mode. The live student stays in train
        # mode so Qwen gradient checkpointing remains active and activation
        # memory stays bounded. LoRA dropout is forced to zero by
        # load_pilot_config and all remaining Qwen Dropout modules are audited
        # above; step 1 additionally requires bit-exact teacher/student values.
        model.eval()
        extractor = model.heatmap_vln.feat_extractor
        with torch.no_grad(), extractor.suspend_capture():
            for step, batch in enumerate(planned_batches, start=1):
                indices = list(batch.indices)
                identities = [sft_sample_identity(dataset, index) for index in indices]
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    values, label_tokens, sample_label_tokens, alignment = sft_correct_label_logprobs(
                        model,
                        collator,
                        [exact_sft_sample(dataset, index) for index in indices],
                        device,
                    )
                teacher_values = (
                    values.detach()
                    .to(
                        device="cpu",
                        dtype=torch.float32,
                    )
                    .contiguous()
                )
                value_sha256 = _fp32_tensor_sha256(teacher_values)
                record_identity = {
                    "schema": "task4_teacher_correct_logprob_record_v2",
                    "step": step,
                    "epoch": int(batch.epoch),
                    "start_position": int(batch.start_position),
                    "dataset_indices": indices,
                    "sample_identities": identities,
                    "label_tokens": int(label_tokens),
                    "sample_label_tokens": sample_label_tokens,
                    "alignment_sha256": alignment["alignment_sha256"],
                    "backend": alignment["backend"],
                    "values_sha256": value_sha256,
                }
                record_sha256 = _canonical_json_sha256(record_identity)
                records.append(
                    {
                        **record_identity,
                        "record_sha256": record_sha256,
                        "alignment": alignment,
                        "teacher_logprobs": teacher_values,
                        "teacher_stats": _logprob_stats(teacher_values),
                    }
                )
    finally:
        try:
            restore_module_modes(model, modes_before)
        finally:
            restore_rng_state(rng_before)

    rng_after_sha256 = rng_state_sha256(capture_rng_state(cuda_rng_devices))
    modes_after = module_mode_contract(model)
    modes_after_sha256 = _canonical_json_sha256(modes_after)
    if rng_after_sha256 != rng_before_sha256:
        raise RuntimeError("Teacher cache failed to restore RNG state exactly")
    if modes_after != modes_before:
        raise RuntimeError("Teacher cache failed to restore model modes exactly")
    backend_counts = Counter(record["backend"] for record in records)
    if len(backend_counts) != 1:
        raise RuntimeError(
            "Teacher cache changed sparse-logits backend across planned batches: "
            f"{dict(backend_counts)}"
        )
    cache_identity = {
        "schema": "task4_teacher_correct_logprob_cache_v2",
        "source_lora_sha256": source_lora_sha256,
        "record_count": len(records),
        "planned_sample_count": sum(len(record["dataset_indices"]) for record in records),
        "total_label_tokens": sum(record["label_tokens"] for record in records),
        "sparse_backend": next(iter(backend_counts)),
        "backend_counts": dict(sorted(backend_counts.items())),
        "record_sha256": [record["record_sha256"] for record in records],
    }
    cache_sha256 = _canonical_json_sha256(cache_identity)
    artifact = {
        **cache_identity,
        "cache_sha256": cache_sha256,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(output_path)
    record_contracts = [
        {key: value for key, value in record.items() if key != "teacher_logprobs"} for record in records
    ]
    contract = {
        **cache_identity,
        "cache_sha256": cache_sha256,
        "artifact": str(output_path.resolve()),
        "artifact_file_sha256": _file_sha256(output_path),
        "storage_device": "cpu",
        "storage_dtype": "torch.float32",
        "teacher_forward_mode": "eval_student_train_dropout_zero",
        "nonzero_qwen_dropout_modules": nonzero_qwen_dropout,
        "nonzero_qwen_functional_dropout_settings": functional_dropout_settings,
        "rng_sha256_before": rng_before_sha256,
        "rng_sha256_after_restore": rng_after_sha256,
        "rng_restored_exactly": True,
        "module_modes_sha256_before": modes_before_sha256,
        "module_modes_sha256_after_restore": modes_after_sha256,
        "module_modes_restored_exactly": True,
        "records": record_contracts,
    }
    return records, contract


@torch.no_grad()
def evaluate_sft_ce(
    model: torch.nn.Module,
    dataset: Any,
    collator: PanoramicTokenizedCollator,
    indices: list[int],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    extractor = model.heatmap_vln.feat_extractor
    weighted_loss = 0.0
    total_tokens = 0
    records = []
    for index in indices:
        with (
            extractor.suspend_capture(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ),
        ):
            loss, label_tokens, _sample_label_tokens = sft_forward_loss(
                model,
                collator,
                [exact_sft_sample(dataset, index)],
                device,
            )
        value = float(loss.detach().float().item())
        weighted_loss += value * label_tokens
        total_tokens += label_tokens
        records.append({"dataset_index": index, "loss": value, "label_tokens": label_tokens})
    mean_loss = weighted_loss / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "samples": len(indices),
        "label_tokens": total_tokens,
        "records": records,
    }


def _generation_summary(metrics: Counter, *, requested_samples: int) -> dict[str, Any]:
    evaluated = int(metrics["total"])
    attempted = int(metrics["attempted"])
    errors = int(metrics["errors"])
    skipped = int(metrics["skipped_no_target"])
    total = max(evaluated, 1)
    coord_targets = max(int(metrics["coord_targets"]), 1)
    stop_targets = max(int(metrics["stop_targets"]), 1)
    turn_targets = max(int(metrics["turn_targets"]), 1)
    view_targets = max(int(metrics["view_coord_targets"]), 1)
    return {
        "samples": evaluated,
        "requested_samples": int(requested_samples),
        "attempted_samples": attempted,
        "errors": errors,
        "skipped_no_target": skipped,
        "complete_coverage": bool(
            attempted == requested_samples and evaluated == requested_samples and errors == 0 and skipped == 0
        ),
        "format_valid": float(metrics["format_valid"]) / total,
        "action_valid": float(metrics["action_valid"]) / total,
        "category_match": float(metrics["category_match"]) / total,
        "coord_hit": float(metrics["coord_hit"]) / coord_targets,
        "stop_hit": float(metrics["stop_hit"]) / stop_targets,
        "turn_hit": float(metrics["turn_hit"]) / turn_targets,
        "view_hit": float(metrics["view_hit"]) / view_targets,
        "counts": {key: float(value) for key, value in sorted(metrics.items())},
    }


@torch.no_grad()
def evaluate_sft_generation(
    model: torch.nn.Module,
    dataset: Any,
    indices: list[int],
    cfg: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    trajectory_cfg = cfg["data"]["trajectory"]
    protocol = str(trajectory_cfg.get("system2_sft_protocol", "direct")).lower()
    include_turns = bool(trajectory_cfg.get("sft_include_turns", False))
    include_forward = bool(trajectory_cfg.get("sft_include_forward", False))
    image_size = image_size_from_cfg(cfg)
    generation_args = SimpleNamespace(max_new_tokens=args.max_new_tokens)
    extractor = model.heatmap_vln.feat_extractor
    metrics: Counter = Counter()
    records = []
    for index in indices:
        metrics["attempted"] += 1
        try:
            sample = exact_sft_sample(dataset, index)
            target_texts = target_texts_for_sample(
                sample,
                include_turns,
                include_forward,
                protocol,
            )
            if not target_texts:
                metrics["skipped_no_target"] += 1
                continue
            messages = make_generation_messages(sample, "target_instruction", protocol)
            with extractor.suspend_capture():
                prediction_text = generate_from_messages(
                    model,
                    messages,
                    device,
                    generation_args,
                )
            target = parse_target(target_texts[0], image_size)
            prediction = parse_generated_text(prediction_text, image_size)
            sample_metrics = update_metrics(
                metrics,
                target,
                prediction,
                args.coord_tolerance,
            )
            records.append(
                {
                    "dataset_index": index,
                    "target_text": target_texts[0],
                    "prediction_text": prediction_text,
                    **sample_metrics,
                }
            )
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:  # keep a finite diagnostic set auditable
            metrics["errors"] += 1
            records.append({"dataset_index": index, "error": repr(exc)})
    summary = _generation_summary(metrics, requested_samples=len(indices))
    summary["records"] = records
    return summary


def gradient_layer_summary(parameters: dict[str, torch.nn.Parameter]) -> dict[str, Any]:
    layers: dict[int, dict[str, Any]] = defaultdict(lambda: {"tensor_count": 0, "nonzero_tensors": 0, "norm_sq": 0.0})
    for name, parameter in parameters.items():
        layer = lora_layer(name)
        grad = parameter.grad
        norm = 0.0 if grad is None else float(grad.detach().float().norm().item())
        layers[layer]["tensor_count"] += 1
        layers[layer]["nonzero_tensors"] += int(norm > 0.0)
        layers[layer]["norm_sq"] += norm * norm
    return {
        str(layer): {
            "tensor_count": values["tensor_count"],
            "nonzero_tensors": values["nonzero_tensors"],
            "grad_norm": math.sqrt(values["norm_sq"]),
        }
        for layer, values in sorted(layers.items())
    }


def parameter_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    norm_sq = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        norm_sq += float(grad.square().sum().item())
    return math.sqrt(norm_sq)


def clone_gradients(parameters: dict[str, torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone() for name, parameter in parameters.items() if parameter.grad is not None
    }


def gradient_conflict(
    heatmap_gradients: dict[str, torch.Tensor],
    parameters: dict[str, torch.nn.Parameter],
) -> dict[str, float | int]:
    dot = 0.0
    heatmap_norm_sq = 0.0
    rehearsal_norm_sq = 0.0
    negative_tensors = 0
    compared_tensors = 0
    for name, parameter in parameters.items():
        heatmap_gradient = heatmap_gradients.get(name)
        if heatmap_gradient is None or parameter.grad is None:
            continue
        rehearsal_gradient = parameter.grad.detach() - heatmap_gradient
        hm = heatmap_gradient.float()
        lm = rehearsal_gradient.float()
        tensor_dot = float((hm * lm).sum().item())
        dot += tensor_dot
        heatmap_norm_sq += float(hm.square().sum().item())
        rehearsal_norm_sq += float(lm.square().sum().item())
        negative_tensors += int(tensor_dot < 0.0)
        compared_tensors += 1
    denominator = math.sqrt(heatmap_norm_sq * rehearsal_norm_sq)
    return {
        "cosine": dot / denominator if denominator > 0.0 else float("nan"),
        "heatmap_norm": math.sqrt(heatmap_norm_sq),
        "weighted_rehearsal_norm": math.sqrt(rehearsal_norm_sq),
        "rehearsal_to_heatmap_norm_ratio": (
            math.sqrt(rehearsal_norm_sq / heatmap_norm_sq) if heatmap_norm_sq > 0.0 else float("nan")
        ),
        "negative_tensor_fraction": negative_tensors / max(compared_tensors, 1),
        "compared_tensors": compared_tensors,
    }


def explicit_rehearsal_gradient_telemetry(
    parameters: dict[str, torch.nn.Parameter],
    rehearsal_gradients: tuple[torch.Tensor | None, ...],
) -> tuple[dict[str, float | int], dict[str, Any], float]:
    """Compare heatmap grads in ``parameter.grad`` with explicit SFT grads.

    The correct-logprob objective uses ``autograd.grad`` so every step can
    report the preservation-only norm and conflict without cloning the entire
    LoRA gradient set.  Returned gradients are still added to ``.grad`` by the
    caller before global clipping.
    """
    if len(parameters) != len(rehearsal_gradients):
        raise ValueError("Explicit rehearsal gradients do not match LoRA tensors")
    dot = 0.0
    heatmap_norm_sq = 0.0
    rehearsal_norm_sq = 0.0
    negative_tensors = 0
    compared_tensors = 0
    per_layer: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"tensor_count": 0, "nonzero_tensors": 0, "norm_sq": 0.0}
    )
    for (name, parameter), rehearsal_gradient in zip(
        parameters.items(),
        rehearsal_gradients,
    ):
        layer = lora_layer(name)
        per_layer[layer]["tensor_count"] += 1
        if rehearsal_gradient is None:
            continue
        lm = rehearsal_gradient.detach().float()
        lm_norm_sq = float(lm.square().sum().item())
        per_layer[layer]["norm_sq"] += lm_norm_sq
        per_layer[layer]["nonzero_tensors"] += int(lm_norm_sq > 0.0)
        rehearsal_norm_sq += lm_norm_sq
        if parameter.grad is None:
            continue
        hm = parameter.grad.detach().float()
        tensor_dot = float((hm * lm).sum().item())
        dot += tensor_dot
        heatmap_norm_sq += float(hm.square().sum().item())
        negative_tensors += int(tensor_dot < 0.0)
        compared_tensors += 1
    denominator = math.sqrt(heatmap_norm_sq * rehearsal_norm_sq)
    conflict = {
        "cosine": dot / denominator if denominator > 0.0 else float("nan"),
        "heatmap_norm": math.sqrt(heatmap_norm_sq),
        "weighted_rehearsal_norm": math.sqrt(rehearsal_norm_sq),
        "rehearsal_to_heatmap_norm_ratio": (
            math.sqrt(rehearsal_norm_sq / heatmap_norm_sq) if heatmap_norm_sq > 0.0 else float("nan")
        ),
        "negative_tensor_fraction": negative_tensors / max(compared_tensors, 1),
        "compared_tensors": compared_tensors,
    }
    layer_summary = {
        str(layer): {
            "tensor_count": values["tensor_count"],
            "nonzero_tensors": values["nonzero_tensors"],
            "weighted_rehearsal_grad_norm": math.sqrt(values["norm_sq"]),
        }
        for layer, values in sorted(per_layer.items())
    }
    return conflict, layer_summary, math.sqrt(rehearsal_norm_sq)


def accumulate_explicit_gradients(
    parameters: dict[str, torch.nn.Parameter],
    gradients: tuple[torch.Tensor | None, ...],
) -> None:
    if len(parameters) != len(gradients):
        raise ValueError("Explicit gradients do not match LoRA tensors")
    for parameter, gradient in zip(parameters.values(), gradients):
        if gradient is None:
            continue
        detached = gradient.detach()
        if parameter.grad is None:
            parameter.grad = detached
        else:
            parameter.grad.add_(detached)


def full_lora_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if "lora_" in name}
    if len(state) != 224:
        raise RuntimeError(f"Task-4 checkpoint must preserve all 224 Stage1-S2 LoRA tensors, got {len(state)}")
    return state


def lora_drift_summary(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    *,
    alpha: float,
    rank: int,
    max_trainable_layer: int,
) -> dict[str, Any]:
    effective = effective_delta_norms(before, after, alpha=alpha, rank=rank)
    layers: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"tensor_count": 0, "changed_tensors": 0, "delta_sq": 0.0, "effective": []}
    )
    for name, initial in before.items():
        layer = lora_layer(name)
        delta = float((after[name].float() - initial.float()).norm().item())
        layers[layer]["tensor_count"] += 1
        layers[layer]["changed_tensors"] += int(delta > 0.0)
        layers[layer]["delta_sq"] += delta * delta
        pair = re.match(r"^(.*)\.lora_[AB]\.[^.]+\.weight$", name)
        if pair and pair.group(1) in effective:
            layers[layer]["effective"].append(effective[pair.group(1)])
    result = {
        str(layer): {
            "reachable_trainable": layer <= max_trainable_layer,
            "tensor_count": values["tensor_count"],
            "changed_tensors": values["changed_tensors"],
            "parameter_delta_norm": math.sqrt(values["delta_sq"]),
            "max_effective_deltaW_norm": max(values["effective"], default=0.0),
        }
        for layer, values in sorted(layers.items())
    }
    frozen_late_unchanged = all(
        values["changed_tensors"] == 0 for layer, values in layers.items() if layer > max_trainable_layer
    )
    return {"layers": result, "frozen_late_layers_unchanged": frozen_late_unchanged}


def save_pilot_checkpoint(
    path: Path,
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    initial_head_hash: str,
    train_log: list[dict[str, Any]],
    checkpoint_contract: dict[str, Any] | None = None,
) -> None:
    lora_state = full_lora_state(model)
    head_state = heatmap_head_state_dict(model.heatmap_vln)
    trainable_state = dict(lora_state)
    trainable_state.update({f"heatmap_vln.{name}": value for name, value in head_state.items()})
    serialized_contract = dict(checkpoint_contract or {})
    serialized_contract.update(
        {
            "checkpoint_step": int(step),
            "train_log_record_count": len(train_log),
            "train_log_is_per_step_prefix": len(train_log) == int(step),
        }
    )
    payload = {
        "task": "task4_joint_pilot",
        "mode": args.mode,
        "step": step,
        "initial_head_hash": initial_head_hash,
        "head_state_dict": head_state,
        "lora_state_dict": lora_state,
        "trainable_state_dict": trainable_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "train_log": train_log,
        "contract": serialized_contract,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def milestone_evaluation_argv(
    args: argparse.Namespace,
    checkpoint_path: Path,
    evaluation_root: Path,
) -> list[str]:
    """Build a fresh-process, evaluation-only invocation for one milestone."""
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "head-only",
        "--config",
        str(Path(args.config).resolve()),
        "--checkpoint",
        str(checkpoint_path.resolve()),
        "--head-checkpoint",
        str(checkpoint_path.resolve()),
        "--data-root",
        str(Path(args.data_root).resolve()),
        "--sft-config",
        str(Path(args.sft_config).resolve()),
        "--sft-data-root",
        str(Path(args.sft_data_root).resolve()),
        "--output-dir",
        str(evaluation_root.resolve()),
        "--device",
        str(args.device),
        "--num-history",
        str(args.num_history),
        "--max-clip-id",
        str(args.max_clip_id),
        "--train-steps",
        "0",
        "--heatmap-train-samples",
        str(args.heatmap_train_samples),
        "--heatmap-val-samples",
        str(args.heatmap_val_samples),
        "--sft-train-samples",
        str(args.sft_train_samples),
        "--sft-batch-size",
        str(args.sft_batch_size),
        "--sft-val-samples",
        str(args.sft_val_samples),
        "--sft-generation-samples",
        str(args.sft_generation_samples),
        "--sft-holdout-scenes",
        str(args.sft_holdout_scenes),
        "--sft-max-clips",
        str(args.sft_max_clips),
        "--head-learning-rate",
        str(args.head_learning_rate),
        "--lora-learning-rate",
        str(args.lora_learning_rate),
        "--rehearsal-objective",
        str(args.rehearsal_objective),
        "--rehearsal-weight",
        str(args.rehearsal_weight),
        "--weight-decay",
        str(args.weight_decay),
        "--grad-clip",
        str(args.grad_clip),
        "--max-trainable-lora-layer",
        str(args.max_trainable_lora_layer),
        "--gradient-cosine-every",
        str(args.gradient_cosine_every),
        "--log-every",
        str(args.log_every),
        "--save-every",
        "0",
        "--milestone-steps",
        "",
        "--seed",
        str(args.seed),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--coord-tolerance",
        str(args.coord_tolerance),
        "--interventions",
        str(args.interventions),
    ]
    if args.skip_generation:
        argv.append("--skip-generation")
    return argv


def write_milestone_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    requested_steps: list[int],
    effective_steps: list[int],
    checkpoints: dict[int, Path],
) -> Path:
    entries = []
    for step, checkpoint_path in sorted(checkpoints.items()):
        evaluation_root = output_dir / "milestone_evaluations" / f"step_{step:06d}"
        entries.append(
            {
                "step": int(step),
                "checkpoint": str(checkpoint_path.resolve()),
                "evaluation_argv": milestone_evaluation_argv(
                    args,
                    checkpoint_path,
                    evaluation_root,
                ),
                "expected_report": str((evaluation_root / "head-only" / "report.json").resolve()),
            }
        )
    path = output_dir / "milestones.json"
    _json_dump(
        path,
        {
            "schema": "task4_isolated_milestones_v1",
            "requested_steps": requested_steps,
            "effective_steps": effective_steps,
            "training_process_runs_midpoint_evaluation": False,
            "evaluation_isolation": (
                "Run evaluation_argv in a fresh process. Checkpoint serialization "
                "does not execute model forwards and therefore cannot perturb the "
                "training RNG or optimizer state."
            ),
            "entries": entries,
        },
    )
    return path


def main() -> int:
    args = parse_args()
    started_at = time.time()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.train_steps < 0:
        raise ValueError(f"--train-steps must be non-negative, got {args.train_steps}")
    if args.sft_train_samples < 0:
        raise ValueError(f"--sft-train-samples must be non-negative, got {args.sft_train_samples}")
    if args.sft_batch_size <= 0:
        raise ValueError(f"--sft-batch-size must be positive, got {args.sft_batch_size}")
    if args.grad_clip <= 0:
        raise ValueError(f"--grad-clip must be positive, got {args.grad_clip}")
    if not math.isfinite(args.rehearsal_weight) or args.rehearsal_weight < 0.0:
        raise ValueError(f"--rehearsal-weight must be finite and non-negative, got {args.rehearsal_weight}")
    if args.gradient_cosine_every <= 0:
        raise ValueError(f"--gradient-cosine-every must be positive, got {args.gradient_cosine_every}")
    if args.log_every <= 0:
        raise ValueError(f"--log-every must be positive, got {args.log_every}")
    requested_milestones = requested_milestone_steps(args.milestone_steps)
    effective_milestones = effective_milestone_steps(
        args.milestone_steps,
        args.train_steps,
    )
    if args.head_checkpoint and args.train_steps != 0:
        raise ValueError("--head-checkpoint is evaluation-only and requires --train-steps 0")
    if args.mode == "head-only" and args.train_steps > 0:
        LOGGER.warning("Task-3 Full already provides the matched 500-step head-only branch")
    if args.mode != "joint-rehearsal" and args.rehearsal_weight != 1.0:
        LOGGER.warning("rehearsal-weight is ignored outside joint-rehearsal mode")
    if args.mode != "joint-rehearsal" and args.rehearsal_objective != "hard-ce":
        LOGGER.warning("rehearsal-objective is ignored outside joint-rehearsal mode")

    set_seed(args.seed)
    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cfg = load_pilot_config(args)

    heatmap_train_dataset = build_heatmap_dataset(
        cfg,
        "train",
        max_clip_id=args.max_clip_id,
    )
    heatmap_val_dataset = build_heatmap_dataset(
        cfg,
        "val",
        max_clip_id=args.max_clip_id,
    )
    heatmap_train_indices = scene_stratified_indices(
        heatmap_train_dataset,
        args.heatmap_train_samples,
    )
    heatmap_val_indices = scene_stratified_indices(
        heatmap_val_dataset,
        args.heatmap_val_samples,
    )
    heatmap_train_contract = selection_contract(heatmap_train_dataset, heatmap_train_indices)
    heatmap_val_contract = selection_contract(heatmap_val_dataset, heatmap_val_indices)
    if set(heatmap_train_contract["scenes"]) & set(heatmap_val_contract["scenes"]):
        raise RuntimeError("Heatmap pilot train/val scenes are not disjoint")

    model = build_model(
        cfg,
        verbose=True,
        device=args.device,
        enable_action_head=False,
    )
    model.qwen2_5_vl._load_model()
    load_info = load_stage1_s2_lora(model, args.checkpoint)
    set_seed(args.seed + 991)
    model._ensure_heatmap_vln()
    fresh_initial_head_hash = _tensor_state_hash(heatmap_head_state_dict(model.heatmap_vln))
    initial_head_hash = fresh_initial_head_hash
    if args.head_checkpoint:
        initial_head_hash, _head_payload = load_heatmap_head_checkpoint(
            model.heatmap_vln,
            args.head_checkpoint,
        )
    starting_head_hash = _tensor_state_hash(heatmap_head_state_dict(model.heatmap_vln))

    head_parameters, trainable_lora, _groups = configure_trainable_parameters(
        model,
        args.mode,
        args.max_trainable_lora_layer,
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = build_optimizer(args, head_parameters, trainable_lora)
    all_lora = lora_named_parameters(model)
    initial_lora = snapshot(all_lora)
    initial_lora_hash = _tensor_state_hash(initial_lora)
    criterion = make_loss(cfg, device)

    sft_dataset, sft_collator, sft_cfg = build_sft_dataset_and_collator(args, model)
    sft_clips_contract = sft_dataset_contract(sft_dataset)
    rehearsal_scenes, holdout_scenes = select_scene_partition(
        sft_dataset,
        seed=args.seed,
        holdout_scene_count=args.sft_holdout_scenes,
    )
    if len(holdout_scenes) != args.sft_holdout_scenes:
        raise RuntimeError(
            "SFT dataset cannot satisfy the requested scene holdout: "
            f"requested={args.sft_holdout_scenes} actual={len(holdout_scenes)} "
            f"dataset_scenes={sft_clips_contract['scene_count']}"
        )
    all_sft_train_indices = all_indices_from_scenes(sft_dataset, rehearsal_scenes)
    if args.sft_train_samples > 0:
        sft_train_indices = select_indices_from_scenes(
            sft_dataset,
            rehearsal_scenes,
            args.sft_train_samples,
        )
        sft_pool_mode = "capped_scene_category_diagnostic"
    else:
        sft_train_indices = all_sft_train_indices
        sft_pool_mode = "full_source_index_including_stop_oversampling"
    sft_val_indices = select_indices_from_scenes(
        sft_dataset,
        holdout_scenes,
        args.sft_val_samples,
    )
    sft_generation_indices = sft_val_indices[: args.sft_generation_samples]
    sft_train_contract = generic_selection_contract(sft_dataset, sft_train_indices)
    sft_train_contract["pool_mode"] = sft_pool_mode
    sft_train_contract["full_candidate_count_before_optional_cap"] = len(all_sft_train_indices)
    sft_val_contract = generic_selection_contract(sft_dataset, sft_val_indices)
    if set(sft_train_contract["scenes"]) & set(sft_val_contract["scenes"]):
        raise RuntimeError("S1-S2 rehearsal/retention scenes are not disjoint")

    rehearsal_stream_seed = args.seed + 4004
    rehearsal_stream = DeterministicEpochBatchStream(
        sft_train_indices,
        batch_size=args.sft_batch_size,
        seed=rehearsal_stream_seed,
    )
    planned_rehearsal_batches = rehearsal_stream.planned_batches(args.train_steps)
    sft_stream_contract = rehearsal_stream_contract(
        sft_dataset,
        sft_train_indices,
        batch_size=args.sft_batch_size,
        seed=rehearsal_stream_seed,
        train_steps=args.train_steps,
    )
    sft_loss_reduction = (
        "mean_over_all_nonignored_shifted_batch_tokens"
        if args.rehearsal_objective == "hard-ce"
        else ("fp32_mse_over_all_nonignored_shifted_correct_label_logprobs_from_initial_teacher")
    )
    checkpoint_contract = {
        "initial_head_hash": initial_head_hash,
        "initial_lora_hash": initial_lora_hash,
        "heatmap_train_sample_identity_sha256": heatmap_train_contract["sample_identity_sha256"],
        "heatmap_val_sample_identity_sha256": heatmap_val_contract["sample_identity_sha256"],
        "sft_rehearsal_sample_identity_sha256": sft_train_contract["sample_identity_sha256"],
        "sft_retention_sample_identity_sha256": sft_val_contract["sample_identity_sha256"],
        "sft_rehearsal_stream": sft_stream_contract,
        "sft_rehearsal_objective": args.rehearsal_objective,
        "sft_loss_reduction": sft_loss_reduction,
    }
    teacher_cache_records: list[dict[str, Any]] = []
    teacher_cache_contract: dict[str, Any] | None = None
    if (
        args.mode == "joint-rehearsal"
        and args.rehearsal_objective == "correct-label-logprob-mse"
        and args.train_steps > 0
    ):
        LOGGER.info(
            "Precomputing initial-teacher correct-label log probabilities for "
            "%d exact planned batches before any update",
            len(planned_rehearsal_batches),
        )
        teacher_cache_records, teacher_cache_contract = precompute_teacher_correct_logprob_cache(
            model=model,
            dataset=sft_dataset,
            collator=sft_collator,
            planned_batches=planned_rehearsal_batches,
            device=device,
            output_path=output_dir / "teacher_correct_label_logprobs.pt",
            source_lora_sha256=initial_lora_hash,
        )
        if len(teacher_cache_records) != args.train_steps:
            raise RuntimeError(
                "Teacher cache does not cover every planned optimizer step: "
                f"records={len(teacher_cache_records)} steps={args.train_steps}"
            )
        post_cache_lora_hash = _tensor_state_hash(snapshot(all_lora))
        post_cache_head_hash = _tensor_state_hash(
            heatmap_head_state_dict(model.heatmap_vln)
        )
        if post_cache_lora_hash != initial_lora_hash:
            raise RuntimeError("Teacher precomputation changed initial LoRA parameters")
        if post_cache_head_hash != starting_head_hash:
            raise RuntimeError(
                "Teacher precomputation changed initial heatmap head parameters"
            )
        teacher_cache_contract["source_lora_sha256_after_cache"] = (
            post_cache_lora_hash
        )
        teacher_cache_contract["source_head_sha256_before_cache"] = (
            starting_head_hash
        )
        teacher_cache_contract["source_head_sha256_after_cache"] = (
            post_cache_head_hash
        )
        teacher_cache_contract["model_parameters_unchanged"] = True
        checkpoint_contract["teacher_correct_label_logprob_cache"] = teacher_cache_contract
    milestone_checkpoints: dict[int, Path] = {}
    if 0 in effective_milestones:
        milestone_path = output_dir / "checkpoint_step_000000.pth"
        save_pilot_checkpoint(
            milestone_path,
            args=args,
            model=model,
            optimizer=optimizer,
            step=0,
            initial_head_hash=initial_head_hash,
            train_log=[],
            checkpoint_contract=checkpoint_contract,
        )
        milestone_checkpoints[0] = milestone_path
    milestone_manifest_path = write_milestone_manifest(
        args,
        output_dir,
        requested_steps=requested_milestones,
        effective_steps=effective_milestones,
        checkpoints=milestone_checkpoints,
    )

    LOGGER.info("Evaluating fixed S1-S2 retention set before Task-4 updates")
    sft_before_ce = evaluate_sft_ce(
        model,
        sft_dataset,
        sft_collator,
        sft_val_indices,
        device,
    )
    sft_before_generation = None
    if not args.skip_generation:
        sft_before_generation = evaluate_sft_generation(
            model,
            sft_dataset,
            sft_generation_indices,
            sft_cfg,
            args,
            device,
        )

    train_log: list[dict[str, Any]] = []
    conflict_log: list[dict[str, Any]] = []
    for step in range(1, args.train_steps + 1):
        scheduled_sft_batch = rehearsal_stream.next_batch()
        if scheduled_sft_batch != planned_rehearsal_batches[step - 1]:
            raise RuntimeError(f"Live rehearsal stream diverged from its recorded deterministic plan at step={step}")
        scheduled_sft_indices = list(scheduled_sft_batch.indices)
        scheduled_sft_identities = [sft_sample_identity(sft_dataset, index) for index in scheduled_sft_indices]
        scheduled_sft_categories: Counter = Counter()
        for index in scheduled_sft_indices:
            clip_idx, frame_idx = sft_dataset.sample_index[index]
            scheduled_sft_categories[sft_index_category(sft_dataset, clip_idx, frame_idx)] += 1

        model.train()
        model.heatmap_vln.feat_extractor.detach_features = args.mode == "head-only"
        optimizer.zero_grad(set_to_none=True)

        heatmap_sample_index = heatmap_train_indices[(step - 1) % len(heatmap_train_indices)]
        transformed = transform_sample(
            heatmap_train_dataset[heatmap_sample_index],
            train_mode="full",
            perturbation="none",
            partner=None,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            heatmap_loss, _record = forward_heatmap_loss(
                model,
                criterion,
                transformed,
                device,
            )
        heatmap_loss.backward()
        heatmap_grad_layers = gradient_layer_summary(trainable_lora) if trainable_lora else {}
        heatmap_lora_grad_norm = math.sqrt(
            sum(float(values["grad_norm"]) ** 2 for values in heatmap_grad_layers.values())
        )
        heatmap_head_grad_norm = parameter_grad_norm(head_parameters)

        diagnose_conflict = (
            args.mode == "joint-rehearsal"
            and trainable_lora
            and (
                args.rehearsal_objective == "correct-label-logprob-mse"
                or step == 1
                or step % args.gradient_cosine_every == 0
            )
        )
        heatmap_gradient_snapshot = (
            clone_gradients(trainable_lora) if diagnose_conflict and args.rehearsal_objective == "hard-ce" else {}
        )
        lm_loss_value: float | None = (
            0.0 if args.rehearsal_objective == "hard-ce" else None
        )
        rehearsal_objective_loss_value = 0.0
        label_tokens = 0
        sample_label_tokens: list[int] = []
        conflict = None
        preservation_telemetry = None
        rehearsal_grad_layers: dict[str, Any] = {}
        weighted_rehearsal_grad_norm = 0.0
        if args.mode == "joint-rehearsal":
            sft_samples = [exact_sft_sample(sft_dataset, index) for index in scheduled_sft_indices]
            if args.rehearsal_objective == "hard-ce":
                # Keep capture suspended through backward: non-reentrant gradient
                # checkpointing recomputes hooked layers during this call.
                with model.heatmap_vln.feat_extractor.suspend_capture():
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        lm_loss, label_tokens, sample_label_tokens = sft_forward_loss(
                            model,
                            sft_collator,
                            sft_samples,
                            device,
                        )
                        weighted_lm_loss = args.rehearsal_weight * lm_loss
                    weighted_lm_loss.backward()
                    lm_loss_value = float(lm_loss.detach().float().item())
                    rehearsal_objective_loss_value = lm_loss_value
                if diagnose_conflict:
                    conflict = gradient_conflict(
                        heatmap_gradient_snapshot,
                        trainable_lora,
                    )
                    conflict["step"] = step
                    conflict_log.append(conflict)
                    weighted_rehearsal_grad_norm = float(conflict["weighted_rehearsal_norm"])
            else:
                if len(teacher_cache_records) != args.train_steps:
                    raise RuntimeError("Correct-label objective has no complete teacher cache")
                teacher_record = teacher_cache_records[step - 1]
                if (
                    teacher_record["step"] != step
                    or teacher_record["epoch"] != scheduled_sft_batch.epoch
                    or teacher_record["start_position"] != scheduled_sft_batch.start_position
                    or teacher_record["dataset_indices"] != scheduled_sft_indices
                    or teacher_record["sample_identities"] != scheduled_sft_identities
                ):
                    raise RuntimeError(f"Teacher cache/live rehearsal alignment mismatch at step={step}")
                teacher_identity = {
                    key: teacher_record[key]
                    for key in (
                        "schema",
                        "step",
                        "epoch",
                        "start_position",
                        "dataset_indices",
                        "sample_identities",
                        "label_tokens",
                        "sample_label_tokens",
                        "alignment_sha256",
                        "backend",
                        "values_sha256",
                    )
                }
                if (
                    _canonical_json_sha256(teacher_identity) != teacher_record["record_sha256"]
                    or _fp32_tensor_sha256(teacher_record["teacher_logprobs"]) != teacher_record["values_sha256"]
                ):
                    raise RuntimeError(f"Teacher cache strong-hash verification failed at step={step}")
                teacher_logprobs = teacher_record["teacher_logprobs"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                # The live student uses deterministic train mode (all Qwen
                # dropout is audited as zero). Keep capture suspended through
                # autograd.grad: non-reentrant gradient checkpointing
                # recomputes Qwen layers.
                with model.heatmap_vln.feat_extractor.suspend_capture():
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        (
                            student_logprobs,
                            label_tokens,
                            sample_label_tokens,
                            student_alignment,
                        ) = sft_correct_label_logprobs(
                            model,
                            sft_collator,
                            sft_samples,
                            device,
                        )
                        if student_alignment["alignment_sha256"] != teacher_record["alignment_sha256"]:
                            raise RuntimeError(f"Student/teacher correct-label alignment hash mismatch at step={step}")
                        if student_alignment["backend"] != teacher_record["backend"]:
                            raise RuntimeError(
                                "Student/teacher sparse-logits backend mismatch at "
                                f"step={step}: student={student_alignment['backend']} "
                                f"teacher={teacher_record['backend']}"
                            )
                        difference = student_logprobs.float() - teacher_logprobs
                        preservation_loss = difference.square().mean(dtype=torch.float32)
                        weighted_preservation_loss = args.rehearsal_weight * preservation_loss
                    difference_cpu = difference.detach().to(
                        device="cpu",
                        dtype=torch.float32,
                    )
                    student_cpu = student_logprobs.detach().to(
                        device="cpu",
                        dtype=torch.float32,
                    )
                    initial_exact_match = bool(
                        torch.equal(
                            student_cpu,
                            teacher_record["teacher_logprobs"],
                        )
                    )
                    if step == 1 and not initial_exact_match:
                        raise RuntimeError(
                            "Initial correct-label preservation loss is not exactly zero; "
                            "teacher/student forward modes or alignment diverged"
                        )
                    explicit_gradients = torch.autograd.grad(
                        weighted_preservation_loss,
                        tuple(trainable_lora.values()),
                        allow_unused=True,
                    )
                conflict, rehearsal_grad_layers, weighted_rehearsal_grad_norm = explicit_rehearsal_gradient_telemetry(
                    trainable_lora,
                    explicit_gradients,
                )
                conflict["step"] = step
                conflict_log.append(conflict)
                accumulate_explicit_gradients(trainable_lora, explicit_gradients)
                rehearsal_objective_loss_value = float(preservation_loss.detach().item())
                preservation_telemetry = {
                    "sparse_backend": student_alignment["backend"],
                    "teacher_cache_record_sha256": teacher_record["record_sha256"],
                    "teacher_alignment_sha256": teacher_record["alignment_sha256"],
                    "student_alignment_sha256": student_alignment["alignment_sha256"],
                    "teacher_values_sha256": teacher_record["values_sha256"],
                    "student_values_sha256": _fp32_tensor_sha256(student_cpu),
                    "teacher": teacher_record["teacher_stats"],
                    "student": _logprob_stats(student_cpu),
                    "difference": {
                        "mse": rehearsal_objective_loss_value,
                        "mean_absolute": float(difference_cpu.abs().mean().item()),
                        "maximum_absolute": float(difference_cpu.abs().max().item()),
                        "exact_match": initial_exact_match,
                    },
                    "initial_zero_check": (
                        {
                            "required": True,
                            "exact_match": initial_exact_match,
                            "loss_is_exact_zero": rehearsal_objective_loss_value == 0.0,
                        }
                        if step == 1
                        else None
                    ),
                    "alignment": student_alignment,
                }

            if len(sample_label_tokens) != len(scheduled_sft_indices):
                raise RuntimeError(
                    "SFT per-sample token telemetry does not match the contracted "
                    f"batch: tokens={sample_label_tokens} "
                    f"indices={scheduled_sft_indices}"
                )
            if sum(sample_label_tokens) != label_tokens:
                raise RuntimeError(
                    "SFT pooled token count disagrees with per-sample counts: "
                    f"pooled={label_tokens} per_sample={sample_label_tokens}"
                )

        combined_lora_grad_norm = parameter_grad_norm(trainable_lora.values())
        # The SFT-only forward bypasses HeatmapVLN, so its backward cannot add
        # head gradients. Avoid another full parameter scan in every step.
        combined_head_grad_norm = heatmap_head_grad_norm
        combined_trainable_grad_norm = math.hypot(
            combined_lora_grad_norm,
            combined_head_grad_norm,
        )
        returned_preclip_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip).detach().float().item()
        )
        clip_coefficient = min(
            1.0,
            float(args.grad_clip) / (returned_preclip_norm + 1e-6),
        )
        postclip_trainable_grad_norm = returned_preclip_norm * clip_coefficient
        optimizer.step()

        record = {
            "step": step,
            "heatmap_loss": float(heatmap_loss.detach().float().item()),
            "lm_loss": lm_loss_value,
            "lm_loss_semantics": (
                "token_pooled_hard_label_cross_entropy"
                if args.rehearsal_objective == "hard-ce"
                else None
            ),
            "rehearsal_objective": args.rehearsal_objective,
            "rehearsal_objective_loss": rehearsal_objective_loss_value,
            "rehearsal_weight": args.rehearsal_weight,
            "lm_label_tokens": label_tokens,
            "lm_sample_label_tokens": sample_label_tokens,
            "correct_label_logprob_preservation": preservation_telemetry,
            "sft_rehearsal_batch": {
                "executed": args.mode == "joint-rehearsal",
                "epoch": scheduled_sft_batch.epoch,
                "start_position": scheduled_sft_batch.start_position,
                "dataset_indices": scheduled_sft_indices,
                "sample_identities": scheduled_sft_identities,
                "category_counts": {
                    category: int(scheduled_sft_categories[category]) for category in ("pixel", "stop")
                },
            },
            "gradient_norms": {
                "heatmap_lora_before_rehearsal": heatmap_lora_grad_norm,
                "heatmap_head": heatmap_head_grad_norm,
                "weighted_rehearsal_lora": weighted_rehearsal_grad_norm,
                "combined_lora_before_clip": combined_lora_grad_norm,
                "combined_head_before_clip": combined_head_grad_norm,
                "combined_trainable_before_clip": combined_trainable_grad_norm,
                "clip_grad_norm_returned_preclip": returned_preclip_norm,
                "clip_coefficient": clip_coefficient,
                "combined_trainable_after_clip": postclip_trainable_grad_norm,
            },
            "heatmap_lora_grad_layers": heatmap_grad_layers,
            "rehearsal_lora_grad_layers": rehearsal_grad_layers,
            "gradient_conflict": conflict,
        }
        train_log.append(record)
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            LOGGER.info(
                "mode=%s step=%d/%d heatmap=%.6f objective=%s value=%.6f "
                "sft_batch=%d tokens=%d categories=%s cosine=%s",
                args.mode,
                step,
                args.train_steps,
                record["heatmap_loss"],
                args.rehearsal_objective,
                rehearsal_objective_loss_value,
                len(scheduled_sft_indices),
                label_tokens,
                dict(record["sft_rehearsal_batch"]["category_counts"]),
                "n/a" if conflict is None else f"{conflict['cosine']:.4f}",
            )
            _json_dump(output_dir / "progress.json", {"latest": record, "log": train_log})
        if args.save_every > 0 and step % args.save_every == 0:
            save_pilot_checkpoint(
                output_dir / "checkpoint_latest.pth",
                args=args,
                model=model,
                optimizer=optimizer,
                step=step,
                initial_head_hash=initial_head_hash,
                train_log=train_log,
                checkpoint_contract=checkpoint_contract,
            )
        if step in effective_milestones and step != args.train_steps:
            milestone_path = output_dir / f"checkpoint_step_{step:06d}.pth"
            save_pilot_checkpoint(
                milestone_path,
                args=args,
                model=model,
                optimizer=optimizer,
                step=step,
                initial_head_hash=initial_head_hash,
                train_log=train_log,
                checkpoint_contract=checkpoint_contract,
            )
            milestone_checkpoints[step] = milestone_path
            write_milestone_manifest(
                args,
                output_dir,
                requested_steps=requested_milestones,
                effective_steps=effective_milestones,
                checkpoints=milestone_checkpoints,
            )

    executed_sft_steps = sum(int(record["sft_rehearsal_batch"]["executed"]) for record in train_log)
    expected_executed_sft_steps = args.train_steps if args.mode == "joint-rehearsal" else 0
    telemetry_contract = {
        "record_count": len(train_log),
        "expected_record_count": args.train_steps,
        "every_optimizer_step_recorded": len(train_log) == args.train_steps,
        "executed_sft_steps": executed_sft_steps,
        "expected_executed_sft_steps": expected_executed_sft_steps,
        "total_sft_label_tokens": sum(int(record["lm_label_tokens"]) for record in train_log),
        "rehearsal_objective": args.rehearsal_objective,
        "correct_label_preservation_steps": sum(
            int(record["correct_label_logprob_preservation"] is not None) for record in train_log
        ),
        "teacher_cache_record_count": len(teacher_cache_records),
        "teacher_cache_sha256": (None if teacher_cache_contract is None else teacher_cache_contract["cache_sha256"]),
        "initial_correct_label_logprob_loss_exactly_zero": (
            None
            if not train_log
            or args.mode != "joint-rehearsal"
            or args.rehearsal_objective != "correct-label-logprob-mse"
            else bool(train_log[0]["correct_label_logprob_preservation"]["initial_zero_check"]["loss_is_exact_zero"])
        ),
    }
    if not telemetry_contract["every_optimizer_step_recorded"]:
        raise RuntimeError(f"Incomplete per-step telemetry: {telemetry_contract}")
    if executed_sft_steps != expected_executed_sft_steps:
        raise RuntimeError(f"SFT execution contract mismatch: {telemetry_contract}")
    if (
        args.mode == "joint-rehearsal"
        and args.rehearsal_objective == "correct-label-logprob-mse"
        and args.train_steps > 0
        and (
            telemetry_contract["correct_label_preservation_steps"] != args.train_steps
            or telemetry_contract["teacher_cache_record_count"] != args.train_steps
            or not telemetry_contract["initial_correct_label_logprob_loss_exactly_zero"]
        )
    ):
        raise RuntimeError(f"Incomplete correct-label preservation telemetry: {telemetry_contract}")
    checkpoint_contract["training_telemetry"] = telemetry_contract

    # Persist the terminal training state before any final evaluation.  Final
    # evaluation is read-only, but it can be long-running; a failure there
    # must not discard the step-100 artifact needed by an isolated evaluator.
    checkpoint_path = output_dir / "checkpoint_final.pth"
    save_pilot_checkpoint(
        checkpoint_path,
        args=args,
        model=model,
        optimizer=optimizer,
        step=args.train_steps,
        initial_head_hash=initial_head_hash,
        train_log=train_log,
        checkpoint_contract=checkpoint_contract,
    )
    if args.train_steps in effective_milestones:
        milestone_checkpoints.setdefault(args.train_steps, checkpoint_path)
    milestone_manifest_path = write_milestone_manifest(
        args,
        output_dir,
        requested_steps=requested_milestones,
        effective_steps=effective_milestones,
        checkpoints=milestone_checkpoints,
    )

    final_lora = snapshot(all_lora)
    final_lora_hash = _tensor_state_hash(final_lora)
    llm_cfg = cfg["model"]["llm"]
    drift = lora_drift_summary(
        initial_lora,
        final_lora,
        alpha=float(llm_cfg["lora_alpha"]),
        rank=int(llm_cfg["lora_rank"]),
        max_trainable_layer=args.max_trainable_lora_layer,
    )
    if not drift["frozen_late_layers_unchanged"]:
        raise RuntimeError("Frozen LoRA layers above the deepest heatmap hook changed")

    final_head_hash = _tensor_state_hash(heatmap_head_state_dict(model.heatmap_vln))
    LOGGER.info("Evaluating final heatmap and fixed S1-S2 retention sets")
    standard_heatmap = evaluate_heatmap(
        model,
        criterion,
        heatmap_val_dataset,
        heatmap_val_indices,
        train_mode="full",
        perturbation="none",
        device=device,
        sample_ids=heatmap_val_contract["sample_identities"],
    )
    heatmap_evaluations = {"standard": standard_heatmap}
    interventions = [value.strip() for value in args.interventions.split(",") if value.strip()]
    for perturbation in interventions:
        heatmap_evaluations[perturbation] = evaluate_heatmap(
            model,
            criterion,
            heatmap_val_dataset,
            heatmap_val_indices,
            train_mode="full",
            perturbation=perturbation,
            device=device,
        )

    sft_after_ce = evaluate_sft_ce(
        model,
        sft_dataset,
        sft_collator,
        sft_val_indices,
        device,
    )
    sft_after_generation = None
    if not args.skip_generation:
        sft_after_generation = evaluate_sft_generation(
            model,
            sft_dataset,
            sft_generation_indices,
            sft_cfg,
            args,
            device,
        )

    report = {
        "task": "task4_joint_pilot",
        "mode": args.mode,
        "seed": args.seed,
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "sft_config": str(Path(args.sft_config).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "sft_data_root": str(Path(args.sft_data_root).resolve()),
        "max_clip_id": args.max_clip_id,
        "train_steps": args.train_steps,
        "elapsed_seconds": time.time() - started_at,
        "load": load_info,
        "contract": {
            "initial_head_hash": initial_head_hash,
            "fresh_initial_head_hash": fresh_initial_head_hash,
            "starting_head_hash": starting_head_hash,
            "final_head_hash": final_head_hash,
            "initial_lora_hash": initial_lora_hash,
            "final_lora_hash": final_lora_hash,
            "all_lora_tensors": len(all_lora),
            "trainable_lora_tensors": len(trainable_lora),
            "trainable_lora_layers": sorted({lora_layer(name) for name in trainable_lora}),
            "max_trainable_lora_layer": args.max_trainable_lora_layer,
            "frozen_late_layers_unchanged": drift["frozen_late_layers_unchanged"],
            "heatmap_train": heatmap_train_contract,
            "heatmap_val": heatmap_val_contract,
            "sft_dataset": sft_clips_contract,
            "sft_scene_partition": {
                "requested_holdout_scene_count": args.sft_holdout_scenes,
                "rehearsal_scenes": rehearsal_scenes,
                "holdout_scenes": holdout_scenes,
            },
            "sft_rehearsal": sft_train_contract,
            "sft_rehearsal_stream": sft_stream_contract,
            "teacher_correct_label_logprob_cache": teacher_cache_contract,
            "sft_retention": sft_val_contract,
            "milestones": {
                "requested_steps": requested_milestones,
                "effective_steps": effective_milestones,
                "manifest": str(milestone_manifest_path),
                "midpoint_evaluation_in_training_process": False,
            },
            "training_telemetry": telemetry_contract,
        },
        "optimization": {
            "head_learning_rate": args.head_learning_rate,
            "lora_learning_rate": args.lora_learning_rate,
            "rehearsal_objective": args.rehearsal_objective,
            "rehearsal_weight": args.rehearsal_weight,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "lambda_coord": 0.0,
            "sft_batch_size": args.sft_batch_size,
            "sft_pool_mode": sft_pool_mode,
            "sft_stream_algorithm": sft_stream_contract["algorithm"],
            "sft_loss_reduction": sft_loss_reduction,
            "correct_label_logprob_backend": (
                None
                if teacher_cache_contract is None
                else teacher_cache_contract["sparse_backend"]
            ),
        },
        "train_log": train_log,
        "gradient_conflict": conflict_log,
        "lora_drift": drift,
        "heatmap_evaluations": heatmap_evaluations,
        "sft_retention": {
            "teacher_forced_before": sft_before_ce,
            "teacher_forced_after": sft_after_ce,
            "generation_before": sft_before_generation,
            "generation_after": sft_after_generation,
        },
        "checkpoint_final": str(checkpoint_path),
    }
    _json_dump(output_dir / "report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())

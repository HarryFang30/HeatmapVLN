#!/usr/bin/env python3
"""
Train a panoramic latent-space adapter for InternNav System1.

This is intentionally narrower than Stage2 bridge training:
  student: panoramic Qwen TRAJ hidden states from HeatmapVLN / Stage1-S2
  output:  adapted 3584-dim latents before InternNav's frozen cond_projector
  loss:    native InternNav teacher latent/cond distillation + GT trajectory
           loss through frozen cond_projector + NextDiT
  train:   adapter only; Pano-System2 and InternNav System1 stay frozen

Frozen VLM + frozen InternNav System1 let this test answer one question:
can a small residual MLP translate the student VLM latent "dialect" into the
InternNav latent "dialect" that frozen cond_projector + NextDiT can execute?

Native teacher sidecars must come from InternNav's front/history + lookdown
protocol (``collect_internnav_teacher_sidecar.py --coord-source dataset``).
They are matched to panoramic student samples by ``(clip_idx, current_t)``,
not by the integer dataset index.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import queue
import random
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from scripts.training.model_builder import build_model
from scripts.training.utils import (
    _load_normalized_state_dict,
    _normalize_state_key,
    assert_complete_lora_checkpoint_match,
    load_config,
    safe_torch_load,
)
from src.data.factory import build_trajectory_dataset
from src.data.pano_teacher_alignment import (
    compute_aligned_teacher_latents_3584_batch,
    has_structured_pano_pixel_goal,
    make_teacher_turn_args,
)
from src.data.panoramic_tokenized_collator import PanoramicTokenizedCollator
from src.models.adapters import (
    PanoLatentSpaceAdapter,
    view_ids_to_indices,
)

LOGGER = logging.getLogger("pano_latent_adapter")
REQUIRED_SYSTEM1_PREFIXES = (
    "cond_projector.",
    "traj_dit.",
    "memory_encoder.",
    "rgb_model.",
    "rgb_resampler.",
    "action_encoder.",
    "action_decoder.",
)


@dataclass
class AdapterTrainBatch:
    student_latents: torch.Tensor
    teacher_latents: torch.Tensor | None
    teacher_cond: torch.Tensor | None
    records: list[dict[str, Any]]
    trajectory: torch.Tensor | None = None
    trajectory_valid: torch.Tensor | None = None
    traj_images: torch.Tensor | None = None


@dataclass
class AdapterCpuBatch:
    collated: dict[str, Any]
    teacher_latents: torch.Tensor | None
    teacher_cond: torch.Tensor | None
    records: list[dict[str, Any]]
    samples: list[dict[str, Any]] | None = None


class NativeTeacherTargetCache:
    """Per-process CPU RAM cache for native teacher sidecar tensors."""

    def __init__(self, *, mode: str = "none", max_items: int = 0) -> None:
        mode = str(mode or "none").lower()
        if mode not in {"none", "lru", "unbounded"}:
            raise ValueError(f"Unsupported teacher cache mode: {mode}")
        self.mode = mode
        self.max_items = max(0, int(max_items))
        self._items: OrderedDict[str, tuple[torch.Tensor, torch.Tensor | None]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def _key(self, rec: dict[str, Any]) -> str:
        path = rec.get("_tensor_path")
        if not path:
            raise RuntimeError(f"Missing tensor sidecar for dataset_index={rec.get('dataset_index')}")
        return str(path)

    def get(self, rec: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.enabled:
            payload = _load_validated_tensor_sidecar_payload(rec)
            path = rec.get("_tensor_path")
            raw = _native_raw_tensor_from_payload(payload, path)
            cond = _native_cond_tensor_from_payload(payload)
            return raw, cond

        key = self._key(rec)
        with self._lock:
            cached = self._items.get(key)
            if cached is not None:
                self.hits += 1
                if self.mode == "lru":
                    self._items.move_to_end(key)
                return cached

        payload = _load_validated_tensor_sidecar_payload(rec)
        raw = _native_raw_tensor_from_payload(payload, rec.get("_tensor_path"))
        cond = _native_cond_tensor_from_payload(payload)

        with self._lock:
            existing = self._items.get(key)
            if existing is not None:
                self.hits += 1
                if self.mode == "lru":
                    self._items.move_to_end(key)
                return existing
            self.misses += 1
            self._items[key] = (raw, cond)
            if self.mode == "lru" and self.max_items > 0:
                while len(self._items) > self.max_items:
                    self._items.popitem(last=False)
        return raw, cond

    def stats(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "mode": self.mode,
                "items": len(self._items),
                "hits": self.hits,
                "misses": self.misses,
            }


class PanoToInternNavLatentAdapter(nn.Module):
    """Per-query adapter from student pano hidden states to teacher latents.

    Defaults are tuned for "cross-interface projection": pure projector
    (``residual=False``), no leading LayerNorm (``pre_norm=False``) so the
    student latent scale is preserved into the MLP, plus a per-dim output
    affine that lets the adapter rescale to the teacher latent norm without
    asking the MLP to memorise both direction and magnitude in its weights.

    The ``norm`` LayerNorm submodule is always instantiated for state-dict
    compatibility with older checkpoints; it is only applied when
    ``pre_norm=True``.
    """

    def __init__(
        self,
        dim: int = 3584,
        hidden_dim: int = 2048,
        dropout: float = 0.0,
        residual: bool = False,
        zero_init: bool = False,
        pre_norm: bool = False,
        n_layers: int = 1,
        output_affine: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.pre_norm = pre_norm
        self.n_layers = max(int(n_layers), 1)

        self.norm = nn.LayerNorm(dim)

        layers: list[nn.Module] = []
        in_dim = dim
        for _ in range(self.n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, dim))
        self.mlp = nn.Sequential(*layers)

        self.gate = nn.Parameter(torch.tensor(1.0))

        if output_affine:
            self.out_scale = nn.Parameter(torch.ones(dim))
            self.out_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("out_scale", None)
            self.register_parameter("out_bias", None)

        if zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        if self.pre_norm:
            h = self.norm(h)
        h = self.mlp(h).to(dtype=x.dtype)
        if self.out_scale is not None:
            h = h * self.out_scale.to(dtype=h.dtype)
        if self.out_bias is not None:
            h = h + self.out_bias.to(dtype=h.dtype)
        if self.residual:
            return x + self.gate.to(dtype=x.dtype) * h
        return h


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _distributed_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank0() -> bool:
    return not _distributed_available() or dist.get_rank() == 0


def _init_distributed(args: argparse.Namespace) -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = args.ddp_backend
        if backend == "auto":
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
        return device, rank, local_rank, world_size

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    return device, rank, local_rank, world_size


def _cleanup_distributed() -> None:
    if _distributed_available():
        dist.destroy_process_group()


def _unwrap_adapter(adapter: nn.Module) -> nn.Module:
    return adapter.module if isinstance(adapter, DistributedDataParallel) else adapter


def _extract_checkpoint_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    ckpt = safe_torch_load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    for key in ("model_state_dict", "trainable_state_dict", "state_dict"):
        state = ckpt.get(key)
        if isinstance(state, dict):
            return state
    if all(torch.is_tensor(value) for value in ckpt.values()):
        return ckpt
    raise KeyError(f"No model_state_dict/trainable_state_dict/state_dict in {path}")


def _assert_internnav_system1_loaded(model: Any) -> None:
    """Require the frozen local System1 port to come entirely from InternNav."""
    head = getattr(model, "nextdit_action_head", None)
    audit = getattr(model, "_internnav_system1_load_audit", None)
    if head is None:
        raise RuntimeError("Model has no NextDiT action head")
    if not audit:
        raise RuntimeError(
            "InternNav System1 weights were not loaded. Set --internnav-model-path "
            "to the released InternNav model directory."
        )

    loaded_keys = set(audit["loaded_keys"])
    required_keys = {
        key
        for key in head.state_dict()
        if key.startswith(REQUIRED_SYSTEM1_PREFIXES)
    }
    missing_keys = sorted(required_keys - loaded_keys)
    if not audit["latent_queries_loaded"] or missing_keys:
        missing_preview = ", ".join(missing_keys[:10])
        raise RuntimeError(
            "InternNav System1 load is incomplete; refusing to freeze random weights. "
            f"source={audit['source']} latent_queries_loaded={audit['latent_queries_loaded']} "
            f"missing_required={len(missing_keys)}"
            + (f" first_missing=[{missing_preview}]" if missing_preview else "")
        )
    LOGGER.info(
        "Verified complete frozen InternNav System1 load from %s: %d required tensors + latent_queries",
        audit["source"],
        len(required_keys),
    )


def _compatible_lora_checkpoint_keys(
    model: nn.Module,
    state: dict[str, torch.Tensor],
) -> list[str]:
    current_state = {
        _normalize_state_key(name): value
        for name, value in model.state_dict().items()
    }
    return [
        name
        for name, value in state.items()
        if "lora_" in _normalize_state_key(name)
        and _normalize_state_key(name) in current_state
        and current_state[_normalize_state_key(name)].shape == value.shape
    ]


def _lora_checkpoint_state(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in state.items()
        if "lora_" in _normalize_state_key(name)
    }


def _load_student_model(cfg: dict[str, Any], args: argparse.Namespace, device: torch.device):
    if args.internnav_model_path:
        os.environ["INTERNNAV_MODEL_PATH"] = args.internnav_model_path
        cfg.setdefault("paths", {})["internnav_model_path"] = args.internnav_model_path
        cfg.setdefault("model", {}).setdefault("llm", {})["model_path"] = args.internnav_model_path
        cfg.setdefault("model", {}).setdefault("action_head", {}).setdefault("nextdit", {})[
            "internnav_model_path"
        ] = args.internnav_model_path

    model = build_model(cfg, device=str(device), verbose=False)
    model = model.to(device)

    # Processor/model must be loaded before LoRA checkpoint weights are applied.
    model.qwen2_5_vl._load_model()
    if model.qwen2_5_vl.processor is None:
        raise RuntimeError("Qwen processor is None after _load_model()")
    if model.latent_queries is None:
        raise RuntimeError("Model has no latent_queries; enable NextDiT action head in config")
    _assert_internnav_system1_loaded(model)

    if not args.base_checkpoint:
        raise RuntimeError("--base-checkpoint is required for the frozen Stage1-S2 student")
    state = _extract_checkpoint_state_dict(args.base_checkpoint)
    compatible_lora_keys = _compatible_lora_checkpoint_keys(model, state)
    lora_state = _lora_checkpoint_state(state)
    matched_lora = assert_complete_lora_checkpoint_match(
        model,
        lora_state,
        checkpoint_path=str(args.base_checkpoint),
    )
    missing, unexpected, loaded = _load_normalized_state_dict(model, lora_state)
    LOGGER.info(
        "Loaded Stage1-S2 LoRA checkpoint %s: checkpoint_tensors=%d lora_tensors=%d "
        "loaded=%d missing=%d unexpected=%d compatible_lora=%d complete_lora=%d",
        args.base_checkpoint,
        len(state),
        len(lora_state),
        loaded,
        len(missing),
        len(unexpected),
        len(compatible_lora_keys),
        matched_lora,
    )
    if loaded == 0:
        raise RuntimeError(
            f"Stage1-S2 checkpoint {args.base_checkpoint} loaded zero compatible LoRA tensors"
        )

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _resolve_tensor_path(jsonl_path: Path, path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_file():
        return path
    candidate = (jsonl_path.parent / path_value).expanduser()
    if candidate.is_file():
        return candidate
    return path


def _normalized_clip_dir_key(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(Path(str(value)).expanduser().resolve())
    except Exception:
        return str(value)


def _stable_sample_key(clip_dir: Any, current_t: Any) -> str | None:
    if clip_dir is None or current_t is None:
        return None
    clip_key = _normalized_clip_dir_key(clip_dir)
    if not clip_key:
        return None
    return f"{clip_key}|t={int(current_t)}"


def _stable_sample_key_from_mapping(mapping: dict[str, Any]) -> str | None:
    return (
        mapping.get("stable_sample_key")
        or _stable_sample_key(mapping.get("clip_dir"), mapping.get("current_t"))
    )


def _stable_sample_keys_match(payload: dict[str, Any], rec: dict[str, Any]) -> bool:
    payload_key = _stable_sample_key_from_mapping(payload)
    record_key = _stable_sample_key_from_mapping(rec)
    return bool(payload_key and record_key and payload_key == record_key)


def _resolve_record_clip_idx(dataset: Any, rec: dict[str, Any]) -> int | None:
    """Resolve a sidecar record against the current dataset, preferring clip_dir.

    Older sidecars store ``clip_idx`` from the collection-time dataset order.
    Expanding the dataset can shift those indices, while ``clip_dir`` remains
    stable.  Prefer the path whenever it is available.
    """
    clip_dir = rec.get("clip_dir")
    clips = getattr(dataset, "clips", None)
    if clip_dir is not None and clips is not None:
        target_key = _normalized_clip_dir_key(clip_dir)
        lookup = getattr(dataset, "_clip_dir_to_idx", None)
        if isinstance(lookup, dict):
            for raw_key in (str(clip_dir), str(Path(str(clip_dir)).expanduser()), target_key):
                if raw_key and raw_key in lookup:
                    return int(lookup[raw_key])
        if target_key:
            for idx, current_clip_dir in enumerate(clips):
                if _normalized_clip_dir_key(current_clip_dir) == target_key:
                    return int(idx)

    clip_idx = rec.get("clip_idx")
    if clip_idx is None:
        return None
    return int(clip_idx)


def _load_validated_tensor_sidecar_payload(rec: dict[str, Any]) -> dict[str, Any]:
    """Load one tensor sidecar and require it to belong to the JSONL record."""
    path = rec.get("_tensor_path")
    expected_idx = rec.get("dataset_index")
    if not path:
        raise RuntimeError(f"Missing tensor sidecar for dataset_index={expected_idx}")
    payload = safe_torch_load(path, map_location="cpu", trust_checkpoint=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Tensor sidecar {path} is not a dict payload")
    payload_idx = payload.get("dataset_index")
    if payload_idx is None:
        raise RuntimeError(
            f"Tensor sidecar {path} has no dataset_index; re-collect the sidecar"
        )
    if int(payload_idx) != int(expected_idx) and not _stable_sample_keys_match(payload, rec):
        raise RuntimeError(
            f"Tensor sidecar dataset_index mismatch: path={path} "
            f"payload={payload_idx} record={expected_idx}"
        )
    return payload


def _load_teacher_records(
    jsonl_path: Path,
    *,
    require_tensor: bool = True,
    require_coord_uv: bool = True,
    require_native_teacher: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            if rec.get("dataset_index") is None:
                continue
            teacher = rec.get("teacher", {})
            if require_native_teacher:
                coord_source = str(teacher.get("coord_source") or "").lower()
                mode = str(teacher.get("mode") or "").lower()
                if coord_source != "dataset" or mode != "dataset_coord":
                    continue
            tensor_info = teacher.get("system1", {}).get("tensor_sidecar", {})
            tensor_path = _resolve_tensor_path(jsonl_path, tensor_info.get("path"))
            if require_tensor and (tensor_path is None or not tensor_path.is_file()):
                continue
            if require_coord_uv and teacher.get("coord_uv") is None:
                continue
            rec["_tensor_path"] = str(tensor_path) if tensor_path is not None else None
            rec["_sidecar_coord_source"] = teacher.get("coord_source")
            rec["_sidecar_mode"] = teacher.get("mode")
            # Extract pano alignment metadata for sidecar validation.
            ds_label = rec.get("dataset_label", {}) or {}
            rec["_sidecar_pano_view_id"] = (
                teacher.get("pano_view_id")
                or ds_label.get("pano_view_id")
                or None
            )
            pixel_goal = ds_label.get("pano_pixel_goal")
            rec["_sidecar_pano_pixel_goal"] = (
                [int(pixel_goal[0]), int(pixel_goal[1])]
                if pixel_goal and len(pixel_goal) >= 2
                else None
            )
            records.append(rec)
    return records


def _validate_native_teacher_sidecar_record(rec: dict[str, Any]) -> bool:
    teacher = rec.get("teacher", {}) or {}
    if str(teacher.get("coord_source") or "").lower() != "dataset":
        return False
    if str(teacher.get("mode") or "").lower() != "dataset_coord":
        return False
    tensor_path = rec.get("_tensor_path")
    if not tensor_path or not Path(str(tensor_path)).is_file():
        return False
    try:
        payload = _load_validated_tensor_sidecar_payload(rec)
    except Exception as exc:
        LOGGER.warning(
            "Skip native teacher record dataset_index=%s: invalid tensor sidecar (%r)",
            rec.get("dataset_index"),
            exc,
        )
        return False
    if "traj_latents" not in payload:
        LOGGER.warning(
            "Skip native teacher record dataset_index=%s: tensor sidecar has no traj_latents",
            rec.get("dataset_index"),
        )
        return False
    if not any(key in payload for key in ("traj_latents_768", "traj_cond_768", "traj_cond")):
        LOGGER.warning(
            "Native teacher record dataset_index=%s has no saved 768-dim cond; "
            "will project raw latents through the frozen student cond_projector.",
            rec.get("dataset_index"),
        )
    return True


def _load_alignment_teacher(args: argparse.Namespace, device: torch.device):
    import types

    from scripts.evaluation.collect_internnav_teacher_sidecar import _load_teacher

    model_path = str(args.internnav_model_path or "").strip()
    if not model_path:
        raise RuntimeError(
            "--teacher-target-mode aligned requires --internnav-model-path "
            "(or INTERNAV_MODEL_PATH)"
        )
    sub = types.SimpleNamespace(
        internnav_repo=str(args.internnav_repo),
        model_path=model_path,
        flash_attn_stub=bool(args.teacher_flash_attn_stub),
        torch_dtype=str(args.teacher_torch_dtype),
        attn_implementation=str(args.teacher_attn_implementation),
        require_nextdit=True,
    )
    model, processor, _traj_to_actions = _load_teacher(sub, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, processor


def _prepare_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.student_config)
    cfg["data"]["root"] = args.root
    if "paths" in cfg:
        cfg["paths"]["dataset_root"] = args.root
    traj_cfg = cfg.setdefault("data", {}).setdefault("trajectory", {})
    traj_cfg["panoramic_vlm_input"] = True
    traj_cfg["compute_pixel_goal"] = True
    traj_cfg["compute_pano_view_pixel_goal"] = True
    traj_cfg["pano_max_side_dist_m"] = float(getattr(args, "pano_max_side_dist_m", 6.0))
    traj_cfg["load_lookdown_for_system2"] = False
    traj_cfg["load_traj_images"] = True
    traj_cfg["traj_image_size"] = [224, 224]
    traj_cfg["enable_trajectory_augmentation"] = False
    traj_cfg["require_sft_target"] = False
    return cfg


def _copy_sample_for_collator(sample: dict[str, Any], coord_uv: list[int]) -> dict[str, Any]:
    copied = {k: v for k, v in sample.items()}
    copied["pixel_goal"] = [int(coord_uv[0]), int(coord_uv[1])]
    return copied


def _sample_from_record(dataset: Any, rec: dict[str, Any]) -> dict[str, Any]:
    """Load the exact sidecar state, preferring stable ``clip_dir/current_t``.

    The collector can run on the fast generic dataset index while older
    sidecars may use the InternNav SFT index.  Using the recorded clip/frame
    pair avoids coupling adapter training to whichever index mode was used;
    using ``clip_dir`` first also survives dataset expansion that shifts
    integer clip indices.
    """
    idx = int(rec["dataset_index"])
    clip_idx = _resolve_record_clip_idx(dataset, rec)
    current_t = rec.get("current_t")
    if clip_idx is None or current_t is None:
        return dataset[idx]

    target = (int(clip_idx), int(current_t))
    if 0 <= idx < len(dataset.sample_index) and tuple(dataset.sample_index[idx]) == target:
        if hasattr(dataset, "_build_sample"):
            return dataset._build_sample(idx)
        return dataset[idx]

    # Temporarily append the requested state.  VLNTrajectoryDataset builds
    # samples from `self.sample_index[idx]`, so this avoids an expensive exact
    # global index rebuild while still using the normal loader path.
    temp_idx = len(dataset.sample_index)
    dataset.sample_index.append(target)
    old_range = getattr(dataset, "_sample_subsequence_range", None)
    try:
        if old_range is not None:
            try:
                meta = dataset._load_meta(target[0])
                num_frames = int(meta.get("num_frames", target[1] + 1))
            except Exception:
                num_frames = target[1] + 1
            old_range[temp_idx] = (0, num_frames)
        if hasattr(dataset, "_build_sample"):
            return dataset._build_sample(temp_idx)
        return dataset[temp_idx]
    finally:
        dataset.sample_index.pop()
        if old_range is not None:
            old_range.pop(temp_idx, None)


def _try_fast_pano_goal_from_record(
    dataset: Any,
    rec: dict[str, Any],
) -> tuple[bool, dict[str, Any] | None]:
    """Check the C3 pano pixel label without materializing image tensors.

    Full sample construction loads history/current/pano/trajectory images.  The
    pre-filter only needs to know whether this frame has a trainable pano pixel
    goal, so use the dataset's projection resolver directly when the sidecar
    contains a stable clip/frame pair.
    """
    clip_idx = _resolve_record_clip_idx(dataset, rec)
    current_t = rec.get("current_t")
    if clip_idx is None or current_t is None:
        return False, None
    if not hasattr(dataset, "_resolve_farthest_pano_pixel_goal"):
        return False, None
    if not hasattr(dataset, "_load_meta") or not hasattr(dataset, "clips"):
        return False, None

    clip_i = int(clip_idx)
    frame_i = int(current_t)
    clips = getattr(dataset, "clips")
    if clip_i < 0 or clip_i >= len(clips):
        return False, None

    try:
        meta = dataset._load_meta(clip_i)
        num_frames = int(meta["num_frames"])
        result = dataset._resolve_farthest_pano_pixel_goal(
            clip_idx=clip_i,
            clip_dir=Path(clips[clip_i]),
            current_t=frame_i,
            num_frames=num_frames,
            img_size=getattr(dataset, "image_size", (224, 224)),
        )
    except Exception as exc:
        LOGGER.debug(
            "Fast pano-goal check unavailable for dataset_index=%s clip=%s t=%s: %r",
            rec.get("dataset_index"),
            clip_idx,
            current_t,
            exc,
        )
        return False, None

    if result is None:
        return True, None

    goal_len, view_id, pano_pg, legacy_uv = result
    goal: dict[str, Any] = {
        "pano_view_id": view_id,
        "pano_pixel_goal": [int(pano_pg[0]), int(pano_pg[1])],
        "pano_pixel_goal_relative_len": int(goal_len),
        "pano_sample_kind": "pixel",
    }
    if legacy_uv is not None:
        goal["legacy_front_pixel_goal"] = [int(legacy_uv[0]), int(legacy_uv[1])]
    return True, goal


def _move_pano_inputs_to_device(pano_inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in pano_inputs.items()
    }


def _make_pano_collator(
    processor: Any,
    n_traj_query: int,
    *,
    sft_protocol: str = "direct",
) -> PanoramicTokenizedCollator:
    return PanoramicTokenizedCollator(
        processor,
        n_traj_query=n_traj_query,
        sft_mode=True,
        sft_protocol=sft_protocol,
        structured_pano_output=True,
        build_sft_labels=False,
    )


def _collate_student_batch_cpu(
    collator: PanoramicTokenizedCollator,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    # The collator clears each sample dict after packing to release large image
    # references. Keep the originals intact for the aligned teacher pass.
    return collator([{key: value for key, value in sample.items()} for sample in samples])


@torch.no_grad()
def _extract_student_latents_from_collated(
    model,
    batch: dict[str, Any],
    device: torch.device,
    *,
    batch_size: int,
) -> torch.Tensor:
    pano_inputs = _move_pano_inputs_to_device(batch["pano_inputs"], device)
    histories = batch["history_frames"].to(device, non_blocking=True)
    current = batch["current_frame"].to(device, non_blocking=True)
    lq = model.latent_queries.expand(batch_size, -1, -1).to(
        device=device,
        dtype=model.config.dtype,
    )
    out = model.qwen2_5_vl(
        history_frames=histories,
        current_frame=current,
        panoramic_inputs=pano_inputs,
        panoramic_num_histories=batch["pano_num_histories"],
        latent_queries=lq,
        return_hidden_states=False,
    )
    traj_hs = out.get("traj_hidden_states")
    if traj_hs is None:
        raise RuntimeError("Student Qwen forward returned no traj_hidden_states")
    return traj_hs.detach()


@torch.no_grad()
def _extract_student_latents(
    model,
    processor,
    samples: list[dict[str, Any]],
    device: torch.device,
    n_traj_query: int,
    *,
    sft_protocol: str = "direct",
    return_batch: bool = False,
    collator: PanoramicTokenizedCollator | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    if collator is None:
        collator = _make_pano_collator(
            processor,
            n_traj_query,
            sft_protocol=sft_protocol,
        )
    batch = _collate_student_batch_cpu(collator, samples)
    traj_hs = _extract_student_latents_from_collated(
        model,
        batch,
        device,
        batch_size=len(samples),
    )
    if return_batch:
        return traj_hs, batch
    return traj_hs


@torch.no_grad()
def _project_teacher_latents_to_dim(
    latents: torch.Tensor,
    *,
    model: Any,
    target_dim: int,
) -> torch.Tensor:
    if latents.shape[-1] == target_dim:
        return latents
    head = getattr(model, "nextdit_action_head", None)
    if head is None:
        raise RuntimeError("Cannot project teacher latents: model has no nextdit_action_head")
    if target_dim != int(head.config.latent_emb_size):
        raise RuntimeError(
            f"Unsupported teacher target_dim={target_dim}; expected {head.config.latent_emb_size}"
        )
    expected_in = int(head.config.vlm_hidden_dim)
    if latents.shape[-1] != expected_in:
        raise RuntimeError(
            f"Cannot project teacher latents from dim {latents.shape[-1]} to {target_dim}; "
            f"expected source dim {expected_in}"
        )
    projector_dtype = next(head.cond_projector.parameters()).dtype
    projected = head.cond_projector(latents.to(dtype=projector_dtype))
    return projected.detach()


def _load_teacher_latents(
    records: list[dict[str, Any]],
    device: torch.device,
    *,
    model: Any | None = None,
    target_dim: int = 768,
) -> torch.Tensor:
    latents = []
    for rec in records:
        path = rec.get("_tensor_path")
        payload = _load_validated_tensor_sidecar_payload(rec)
        latent = None
        if target_dim == 768:
            for key in ("traj_latents_768", "traj_cond_768", "traj_cond"):
                if key in payload:
                    latent = payload[key].detach()
                    break
        if latent is None:
            if "traj_latents" not in payload:
                raise RuntimeError(f"{path} has no traj_latents")
            latent = payload["traj_latents"].detach()
        if latent.dim() == 3 and latent.shape[0] == 1:
            latent = latent.squeeze(0)
        latents.append(latent)
    stacked = torch.stack(latents, dim=0).to(device)
    if stacked.shape[-1] != target_dim:
        if model is None:
            raise RuntimeError(
                f"Teacher latents are dim {stacked.shape[-1]}, need {target_dim}; "
                "pass model to project through cond_projector"
            )
        stacked = _project_teacher_latents_to_dim(
            stacked,
            model=model,
            target_dim=target_dim,
        )
    return stacked


def _squeeze_teacher_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


def _native_cond_tensor_from_payload(payload: dict[str, Any]) -> torch.Tensor | None:
    for key in ("traj_latents_768", "traj_cond_768", "traj_cond"):
        if key in payload:
            value = payload[key]
            if torch.is_tensor(value):
                return _squeeze_teacher_tensor(value)
    return None


def _native_raw_tensor_from_payload(payload: dict[str, Any], path: Any) -> torch.Tensor:
    if "traj_latents" not in payload or not torch.is_tensor(payload["traj_latents"]):
        raise RuntimeError(f"{path} has no traj_latents")
    return _squeeze_teacher_tensor(payload["traj_latents"])


def _load_native_teacher_targets_cpu(
    records: list[dict[str, Any]],
    *,
    need_raw: bool,
    need_cond: bool,
    teacher_cache: NativeTeacherTargetCache | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    raw_latents: list[torch.Tensor] = []
    cond_latents: list[torch.Tensor] = []

    if not need_raw and not need_cond:
        return None, None

    for rec in records:
        if teacher_cache is not None:
            raw_tensor, cond_tensor = teacher_cache.get(rec)
        else:
            path = rec.get("_tensor_path")
            payload = _load_validated_tensor_sidecar_payload(rec)
            raw_tensor = _native_raw_tensor_from_payload(payload, path)
            cond_tensor = _native_cond_tensor_from_payload(payload)

        if need_raw:
            raw_latents.append(raw_tensor)

        if need_cond:
            cond_latents.append(cond_tensor if cond_tensor is not None else raw_tensor)

    raw = torch.stack(raw_latents, dim=0) if need_raw else None
    cond = torch.stack(cond_latents, dim=0) if need_cond else None
    return raw, cond


def _build_records_from_dataset(
    dataset: Any,
    *,
    max_samples: int = 0,
) -> list[dict[str, Any]]:
    """Auto-generate minimal teacher records from dataset for aligned mode.

    Uses a fast path that only loads poses + depth + metadata (no RGB images).
    Each sample's pano pixel goal is re-computed via the same C3 occlusion rule
    used during dataset building, so there is no drift.  Typically finishes in
    single-digit minutes even for 100k+ samples.
    """
    records: list[dict[str, Any]] = []
    total = len(dataset)
    if _rank0():
        from tqdm import tqdm as _tqdm

        it = _tqdm(range(total), desc="Building records from dataset (fast)", unit="samples")
    else:
        it = range(total)

    # Cache metadata lookups that are shared across samples within the same clip.
    _clip_meta: dict[int, dict[str, Any]] = {}
    _clip_dir: dict[int, Any] = {}
    _clip_img_size: dict[int, tuple[int, int]] = {}

    for idx in it:
        clip_idx, current_t = (
            tuple(dataset.sample_index[idx])
            if idx < len(dataset.sample_index)
            else (None, None)
        )
        if clip_idx is None:
            continue

        # Lazy-load per-clip metadata.
        if clip_idx not in _clip_meta:
            try:
                _clip_meta[clip_idx] = dataset._load_meta(clip_idx)
                _clip_dir[clip_idx] = dataset.clips[clip_idx]
                _clip_img_size[clip_idx] = dataset._load_intrinsics(clip_idx, dataset.clips[clip_idx])[0]
            except Exception:
                _clip_meta[clip_idx] = None  # type: ignore[assignment]
                continue
        meta = _clip_meta.get(clip_idx)
        if meta is None:
            continue
        clip_dir = _clip_dir[clip_idx]
        img_size = _clip_img_size[clip_idx]
        num_frames = int(meta.get("num_frames", 0))
        if current_t >= num_frames - 1:
            continue  # last frame → no pixel goal possible

        # Fast pano label computation — no images loaded.
        try:
            pano_result = dataset._resolve_farthest_pano_pixel_goal(
                clip_idx=clip_idx,
                clip_dir=clip_dir,
                current_t=current_t,
                num_frames=num_frames,
                img_size=img_size,
            )
        except Exception:
            continue

        if pano_result is None:
            continue
        # pano_result = (goal_len, view_id, [u, v], legacy_front_uv)
        _goal_len, view_id, pano_pg, _legacy = pano_result
        if pano_pg is None:
            continue

        records.append({
            "status": "ok",
            "dataset_index": idx,
            "clip_idx": clip_idx,
            "current_t": current_t,
            "_tensor_path": None,
        })
        if max_samples > 0 and len(records) >= max_samples:
            break
    return records


def _filter_records_with_pano_goals(
    records: list[dict[str, Any]],
    *,
    dataset: Any,
    validate_sidecar_metadata: bool = False,
    require_native_teacher_sidecar: bool = False,
) -> list[dict[str, Any]]:
    """Keep records whose exact frame has a structured pano pixel goal.

    DDP needs every rank to execute the same number of backward calls.  Filtering
    once before sharding avoids rank-local batch skips when a teacher sidecar
    record has no student pano pixel target under the current C3 rule.

    When ``validate_sidecar_metadata`` is True, also verifies that the sidecar
    record's pano metadata matches the dataset sample's computed labels.  This
    catches stale or misaligned sidecars before training starts.
    """
    filtered: list[dict[str, Any]] = []
    skipped = 0
    failed = 0
    mismatched = 0
    missing_metadata = 0
    invalid_tensor = 0
    invalid_native = 0
    fast_checked = 0
    fallback_loaded = 0
    total = len(records)
    for rec_i, rec in enumerate(records, start=1):
        idx = rec.get("dataset_index")
        if rec_i % 1000 == 0:
            LOGGER.info(
                "Filtering teacher records... %d/%d kept=%d skipped=%d failed=%d "
                "fast=%d fallback=%d",
                rec_i,
                total,
                len(filtered),
                skipped,
                failed,
                fast_checked,
                fallback_loaded,
            )
        if require_native_teacher_sidecar:
            if str(rec.get("_sidecar_coord_source") or "").lower() != "dataset":
                invalid_native += 1
                continue
            if str(rec.get("_sidecar_mode") or "").lower() != "dataset_coord":
                invalid_native += 1
                continue
            if rec.get("clip_idx") is None or rec.get("current_t") is None:
                invalid_native += 1
                continue
            tensor_path = rec.get("_tensor_path")
            if not tensor_path or not Path(str(tensor_path)).is_file():
                invalid_native += 1
                continue

        fast_available, fast_goal = _try_fast_pano_goal_from_record(dataset, rec)
        if fast_available:
            fast_checked += 1
            if fast_goal is None:
                skipped += 1
                continue
            sample: dict[str, Any] | None = None
            pano_view_id = fast_goal["pano_view_id"]
            pano_pixel_goal = fast_goal["pano_pixel_goal"]
        else:
            try:
                sample = _sample_from_record(dataset, rec)
            except Exception as exc:
                failed += 1
                LOGGER.warning(
                    "Skip teacher record dataset_index=%s: failed to load exact sample (%r)",
                    idx,
                    exc,
                )
                continue
            fallback_loaded += 1
            if not _has_trainable_pano_goal(sample):
                skipped += 1
                continue
            pano_view_id = sample.get("pano_view_id")
            pano_pixel_goal = sample.get("pano_pixel_goal")

        # Validate sidecar metadata against the current dataset labels.
        # This runs globally before DDP sharding so a mismatch is surfaced
        # synchronously on every rank.
        if validate_sidecar_metadata:
            sv = rec.get("_sidecar_pano_view_id")
            sp = rec.get("_sidecar_pano_pixel_goal")
            if sv is None or sp is None:
                missing_metadata += 1
                LOGGER.warning(
                    "Skip sidecar record dataset_index=%s: missing pano metadata "
                    "view_id=%r pixel_goal=%r",
                    idx, sv, sp,
                )
                continue
            if str(sv).lower() != str(pano_view_id or "").lower():
                mismatched += 1
                LOGGER.warning(
                    "Skip sidecar record dataset_index=%s: pano_view_id mismatch "
                    "sidecar=%r dataset=%r",
                    idx, sv, pano_view_id,
                )
                continue
            dp = pano_pixel_goal
            if dp is None or int(sp[0]) != int(dp[0]) or int(sp[1]) != int(dp[1]):
                mismatched += 1
                LOGGER.warning(
                    "Skip sidecar record dataset_index=%s: pano_pixel_goal mismatch "
                    "sidecar=%s dataset=%s",
                    idx, sp, dp,
                )
                continue
            try:
                _load_validated_tensor_sidecar_payload(rec)
            except Exception as exc:
                invalid_tensor += 1
                LOGGER.warning(
                    "Skip sidecar record dataset_index=%s: invalid tensor sidecar (%r)",
                    idx, exc,
                )
                continue
        filtered.append(rec)
    LOGGER.info(
        "Filtered teacher records for pano pixel goals: "
        "kept=%d skipped_no_pano_goal=%d failed=%d mismatched=%d "
        "missing_metadata=%d invalid_tensor=%d invalid_native=%d "
        "fast_checked=%d fallback_loaded=%d",
        len(filtered),
        skipped,
        failed,
        mismatched,
        missing_metadata,
        invalid_tensor,
        invalid_native,
        fast_checked,
        fallback_loaded,
    )
    if mismatched or missing_metadata or invalid_tensor or invalid_native:
        LOGGER.warning(
            "%d sidecar records failed strict/native validation and were dropped. "
            "Re-collect sidecars if this count is unexpected.",
            mismatched + missing_metadata + invalid_tensor + invalid_native,
        )
    return filtered


def _latent_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    cosine_weight: float,
    mse_weight: float,
    norm_weight: float,
    norm_loss_type: str = "log_ratio",
    front_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Legacy latent-space diagnostic loss, gated to front-view samples only.

    The current PanoLatentSpaceAdapter training path does not call this helper;
    it is retained for older experiments that compare against teacher latents.
    """
    pred_f = pred.float()
    target_f = target.float()
    cos = F.cosine_similarity(pred_f.flatten(1), target_f.flatten(1), dim=1)
    cos_loss_per_sample = 1.0 - cos  # [B]
    mse_loss_per_sample = F.mse_loss(pred_f, target_f, reduction="none").flatten(1).mean(dim=1)  # [B]
    pred_norm = pred_f.norm(dim=-1)
    target_norm = target_f.norm(dim=-1).clamp_min(1.0e-6)
    norm_ratio = pred_norm / target_norm
    if norm_loss_type == "log_ratio":
        norm_loss_per_sample = (torch.log(norm_ratio.clamp_min(1.0e-6))) ** 2
        norm_loss_per_sample = norm_loss_per_sample.mean(dim=-1)  # [B]
    elif norm_loss_type == "ratio":
        norm_loss_per_sample = ((norm_ratio - 1.0) ** 2).mean(dim=-1)
    else:
        raise ValueError(f"Unknown norm_loss_type: {norm_loss_type!r}")

    if front_mask is not None:
        mask = front_mask.to(device=pred_f.device, dtype=torch.float32)
        denom = mask.sum().clamp_min(1.0)
        cos_loss = (cos_loss_per_sample * mask).sum() / denom
        mse_loss = (mse_loss_per_sample * mask).sum() / denom
        norm_loss = (norm_loss_per_sample * mask).sum() / denom
        # Diagnostics: aggregate over front-masked subset
        cos_mean = (cos * mask).sum() / denom
        pn_mean = (pred_norm.mean(dim=-1) * mask).sum() / denom
        tn_mean = (target_norm.mean(dim=-1) * mask).sum() / denom
        nr_mean = (norm_ratio.mean(dim=-1) * mask).sum() / denom
    else:
        cos_loss = cos_loss_per_sample.mean()
        mse_loss = mse_loss_per_sample.mean()
        norm_loss = norm_loss_per_sample.mean()
        cos_mean = cos.mean()
        pn_mean = pred_norm.mean()
        tn_mean = target_norm.mean()
        nr_mean = norm_ratio.mean()

    loss = cosine_weight * cos_loss + mse_weight * mse_loss + norm_weight * norm_loss
    return loss, {
        "loss": float(loss.detach().item()),
        "cosine": float(cos_mean.detach().item()),
        "cos_loss": float(cos_loss.detach().item()),
        "mse_loss": float(mse_loss.detach().item()),
        "norm_loss": float(norm_loss.detach().item()),
        "pred_norm": float(pn_mean.detach().item()),
        "target_norm": float(tn_mean.detach().item()),
        "norm_ratio": float(nr_mean.detach().item()),
    }


def _cosine_and_norm_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    norm_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_f = pred.float()
    target_f = target.to(device=pred.device, dtype=torch.float32)
    cos = F.cosine_similarity(pred_f.flatten(1), target_f.flatten(1), dim=1)
    cos_loss = (1.0 - cos).mean()
    pred_norm = pred_f.norm(dim=-1)
    target_norm = target_f.norm(dim=-1).clamp_min(1.0e-6)
    norm_ratio = pred_norm / target_norm
    norm_loss = (torch.log(norm_ratio.clamp_min(1.0e-6)) ** 2).mean()
    loss = cos_loss + float(norm_weight) * norm_loss
    return loss, {
        "cos": float(cos.mean().detach().item()),
        "cos_loss": float(cos_loss.detach().item()),
        "norm_loss": float(norm_loss.detach().item()),
        "pred_norm": float(pred_norm.mean().detach().item()),
        "target_norm": float(target_norm.mean().detach().item()),
        "norm_ratio": float(norm_ratio.mean().detach().item()),
    }


def _cond_distill_loss(
    pred_cond: torch.Tensor,
    teacher_cond: torch.Tensor,
    *,
    cosine_weight: float = 1.0,
    beta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_f = pred_cond.float()
    target_f = teacher_cond.to(device=pred_cond.device, dtype=torch.float32)
    smooth_l1 = F.smooth_l1_loss(pred_f, target_f, beta=float(beta))
    cos = F.cosine_similarity(pred_f.flatten(1), target_f.flatten(1), dim=1)
    cos_loss = (1.0 - cos).mean()
    mse = F.mse_loss(pred_f, target_f)
    loss = smooth_l1 + float(cosine_weight) * cos_loss
    return loss, {
        "cos": float(cos.mean().detach().item()),
        "cos_loss": float(cos_loss.detach().item()),
        "smooth_l1": float(smooth_l1.detach().item()),
        "mse": float(mse.detach().item()),
        "pred_norm": float(pred_f.norm(dim=-1).mean().detach().item()),
        "target_norm": float(target_f.norm(dim=-1).mean().detach().item()),
    }


def _gt_flow_loss_from_projected(
    *,
    model: Any,
    pred_cond: torch.Tensor,
    batch: AdapterTrainBatch,
) -> torch.Tensor:
    if batch.trajectory is None or model.nextdit_action_head is None:
        return pred_cond.sum() * 0.0
    head = model.nextdit_action_head
    dit_dtype = next(head.action_encoder.parameters()).dtype
    gt = batch.trajectory.to(device=pred_cond.device, dtype=dit_dtype)
    images = batch.traj_images.to(device=pred_cond.device) if batch.traj_images is not None else None
    valid = batch.trajectory_valid.to(device=pred_cond.device) if batch.trajectory_valid is not None else None
    pred_exp, gt_exp, images_exp, valid_exp = head._expand_sequence_training_inputs(
        pred_cond.to(dtype=dit_dtype),
        gt,
        images,
        valid,
    )
    noisy, timesteps, target_vel = head.sample_flow_matching_inputs(gt_exp)
    pred_vel = head.predict_velocity_from_projected(
        pred_exp,
        noisy,
        timesteps,
        traj_images=images_exp,
    )
    return head.masked_velocity_mse(pred_vel, target_vel, valid_exp)


def _compute_adapter_objective(
    *,
    model: Any,
    pred_raw: torch.Tensor,
    batch: AdapterTrainBatch,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    head = getattr(model, "nextdit_action_head", None)
    if head is None:
        raise RuntimeError("Adapter training requires model.nextdit_action_head")
    cond_projector = head.cond_projector
    proj_dtype = next(cond_projector.parameters()).dtype
    pred_cond = cond_projector(pred_raw.to(dtype=proj_dtype))

    zero = pred_raw.sum() * 0.0
    raw_loss = zero
    raw_stats = {
        "cos": 0.0,
        "cos_loss": 0.0,
        "norm_loss": 0.0,
        "pred_norm": 0.0,
        "target_norm": 0.0,
        "norm_ratio": 0.0,
    }
    if batch.teacher_latents is not None and float(args.raw_distill_weight) > 0:
        raw_loss, raw_stats = _cosine_and_norm_loss(
            pred_raw,
            batch.teacher_latents,
            norm_weight=float(args.raw_norm_weight),
        )

    cond_loss = zero
    cond_stats = {
        "cos": 0.0,
        "cos_loss": 0.0,
        "smooth_l1": 0.0,
        "mse": 0.0,
        "pred_norm": 0.0,
        "target_norm": 0.0,
    }
    if batch.teacher_cond is not None and float(args.cond_distill_weight) > 0:
        cond_loss, cond_stats = _cond_distill_loss(
            pred_cond,
            batch.teacher_cond,
            cosine_weight=float(args.cond_cosine_weight),
            beta=float(args.cond_smooth_l1_beta),
        )

    gt_loss = _gt_flow_loss_from_projected(model=model, pred_cond=pred_cond, batch=batch)
    loss = (
        float(args.raw_distill_weight) * raw_loss
        + float(args.cond_distill_weight) * cond_loss
        + float(args.gt_weight) * gt_loss
    )
    metrics = {
        "loss": float(loss.detach().item()),
        "raw": float(raw_loss.detach().item()),
        "raw_cos": raw_stats["cos"],
        "raw_cos_loss": raw_stats["cos_loss"],
        "raw_norm_loss": raw_stats["norm_loss"],
        "raw_norm_ratio": raw_stats["norm_ratio"],
        "cond": float(cond_loss.detach().item()),
        "cond_cos": cond_stats["cos"],
        "cond_cos_loss": cond_stats["cos_loss"],
        "cond_smooth_l1": cond_stats["smooth_l1"],
        "cond_mse": cond_stats["mse"],
        "gt": float(gt_loss.detach().item()),
    }
    return loss, metrics, pred_cond


def _has_trainable_pano_goal(sample: dict[str, Any]) -> bool:
    return has_structured_pano_pixel_goal(sample)


def _goal_tensors_from_samples(
    samples: list[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    view_ids = [str(sample["pano_view_id"]).lower() for sample in samples]
    view_indices = view_ids_to_indices(view_ids, device=device)
    pixels = torch.tensor(
        [[int(sample["pano_pixel_goal"][0]), int(sample["pano_pixel_goal"][1])] for sample in samples],
        device=device,
        dtype=torch.float32,
    )
    image_hw_values: list[list[int]] = []
    for sample in samples:
        image_tensor = sample.get("current_views")
        if image_tensor is None:
            image_tensor = sample.get("current_frame")
        if image_tensor is None or not torch.is_tensor(image_tensor):
            raise RuntimeError("Cannot infer image size for pano geometry token")
        height = int(image_tensor.shape[-2])
        width = int(image_tensor.shape[-1])
        image_hw_values.append([height, width])
    image_hw = torch.tensor(image_hw_values, device=device, dtype=torch.float32)
    return view_indices, pixels, image_hw


def _collated_tensor(
    batch: dict[str, Any],
    key: str,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    value = batch.get(key)
    if value is None or not torch.is_tensor(value):
        return None
    if dtype is None:
        return value.to(device, non_blocking=True)
    return value.to(device=device, dtype=dtype, non_blocking=True)


def _split_train_val(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic split based on a stable record ordering and the seed."""
    if not records:
        return [], []
    if args.val_records > 0:
        val_size = min(int(args.val_records), len(records) - 1)
    else:
        val_size = int(round(max(args.val_ratio, 0.0) * len(records)))
        val_size = max(val_size, 0)
        val_size = min(val_size, len(records) - 1) if len(records) > 1 else 0
    if val_size <= 0:
        return list(records), []

    ordered = sorted(records, key=lambda r: int(r["dataset_index"]))
    rng = random.Random(args.seed)
    indices = list(range(len(ordered)))
    rng.shuffle(indices)
    val_pos = set(indices[:val_size])
    train_records = [ordered[i] for i in range(len(ordered)) if i not in val_pos]
    val_records = [ordered[i] for i in indices[:val_size]]
    return train_records, val_records


def _epoch_rank_records(
    records: list[dict[str, Any]],
    *,
    seed: int,
    epoch: int,
    rank: int,
    world_size: int,
) -> list[dict[str, Any]]:
    """Return the same number of batches per rank for DDP.

    DDP requires every rank to execute the same number of backward calls.  We
    deterministically shuffle once per epoch, pad to a multiple of world_size,
    then strided-shard the list.
    """
    if world_size <= 1:
        shuffled = list(records)
        random.Random(seed + epoch).shuffle(shuffled)
        return shuffled
    if not records:
        return []

    shuffled = list(records)
    random.Random(seed + epoch).shuffle(shuffled)
    total_size = int(math.ceil(len(shuffled) / float(world_size)) * world_size)
    if total_size > len(shuffled):
        shuffled.extend(shuffled[: total_size - len(shuffled)])
    return shuffled[rank:total_size:world_size]


def _reduce_metrics(
    sums: dict[str, float],
    count: int,
    device: torch.device,
) -> dict[str, float]:
    keys = sorted(sums)
    if not keys:
        # In distributed mode every rank must participate in the all_reduce
        # collective; returning early would deadlock the other ranks.
        # Callers must ensure every rank processes ≥1 batch (see
        # _evaluate_adapter for validation and _epoch_rank_records for training).
        if _distributed_available():
            raise RuntimeError(
                "_reduce_metrics called with empty sums in distributed mode — "
                "this would deadlock other ranks.  Pad your data to a multiple "
                "of world_size before sharding."
            )
        return {}
    values = [sums[key] for key in keys] + [float(count)]
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if _distributed_available():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total_count = max(float(tensor[-1].item()), 1.0)
    return {key: float(tensor[i].item() / total_count) for i, key in enumerate(keys)}


def _build_batch(
    batch_records: list[dict[str, Any]],
    *,
    dataset: Any,
    model: Any,
    processor: Any,
    device: torch.device,
    n_traj_query: int,
    teacher_target_mode: str = "aligned",
    teacher_model: Any | None = None,
    teacher_processor: Any | None = None,
    teacher_turn_args: Any | None = None,
    collator: PanoramicTokenizedCollator | None = None,
    teacher_cache: NativeTeacherTargetCache | None = None,
) -> AdapterTrainBatch:
    if collator is None:
        collator = _make_pano_collator(
            processor,
            n_traj_query,
            sft_protocol="direct",
        )
    cpu_batch = _prepare_batch_cpu(
        batch_records,
        dataset=dataset,
        collator=collator,
        teacher_target_mode=teacher_target_mode,
        keep_samples=teacher_model is not None and teacher_processor is not None,
        need_raw=True,
        need_cond=True,
        teacher_cache=teacher_cache,
    )
    return _finalize_batch_on_device(
        cpu_batch,
        model=model,
        device=device,
        teacher_model=teacher_model,
        teacher_processor=teacher_processor,
        teacher_turn_args=teacher_turn_args,
    )


def _record_batches(
    records: list[dict[str, Any]],
    batch_size: int,
) -> list[list[dict[str, Any]]]:
    return [
        records[start:start + batch_size]
        for start in range(0, len(records), batch_size)
    ]


def _records_for_rank_remaining_epochs(
    train_records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    start_epoch: int,
) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for epoch in range(start_epoch, int(args.epochs)):
        for rec in _epoch_rank_records(
            train_records,
            seed=int(args.seed),
            epoch=epoch,
            rank=rank,
            world_size=world_size,
        ):
            key = str(rec.get("_tensor_path") or rec.get("dataset_index"))
            by_path.setdefault(key, rec)
    return list(by_path.values())


def _preload_teacher_cache(
    records: list[dict[str, Any]],
    *,
    teacher_cache: NativeTeacherTargetCache,
    workers: int,
    rank: int,
) -> None:
    if not teacher_cache.enabled or not records:
        return
    worker_count = max(1, int(workers))
    task_q: queue.Queue[Any] = queue.Queue(maxsize=worker_count * 4)
    done = object()
    first_error: list[BaseException] = []
    stats = {"done": 0}
    lock = threading.Lock()

    def worker() -> None:
        while True:
            item = task_q.get()
            try:
                if item is done:
                    return
                try:
                    teacher_cache.get(item)
                except BaseException as exc:
                    with lock:
                        if not first_error:
                            first_error.append(exc)
                finally:
                    with lock:
                        stats["done"] += 1
                        done_count = stats["done"]
                    if rank == 0 and done_count % 10000 == 0:
                        LOGGER.info(
                            "Preloading teacher sidecar cache... %d/%d cache=%s",
                            done_count,
                            len(records),
                            teacher_cache.stats(),
                        )
            finally:
                task_q.task_done()

    threads = [
        threading.Thread(target=worker, name=f"stage2-teacher-cache-preload-{idx}", daemon=True)
        for idx in range(worker_count)
    ]
    for thread in threads:
        thread.start()

    t0 = time.perf_counter()
    for rec in records:
        task_q.put(rec)
    for _ in threads:
        task_q.put(done)
    task_q.join()
    for thread in threads:
        thread.join(timeout=1.0)

    if first_error:
        raise first_error[0]
    if rank == 0:
        elapsed = time.perf_counter() - t0
        LOGGER.info(
            "Preloaded teacher sidecar cache: records=%d workers=%d elapsed=%.1fs cache=%s",
            len(records),
            worker_count,
            elapsed,
            teacher_cache.stats(),
        )


def _iter_prepared_cpu_batches(
    record_batches: list[list[dict[str, Any]]],
    *,
    dataset: Any,
    dataset_factory: Any | None = None,
    collator: PanoramicTokenizedCollator,
    collator_factory: Any | None = None,
    teacher_target_mode: str,
    keep_samples: bool,
    need_raw: bool,
    need_cond: bool,
    teacher_cache: NativeTeacherTargetCache | None,
    prefetch_batches: int,
    prefetch_workers: int = 1,
) -> Any:
    def build(
        batch_records: list[dict[str, Any]],
        batch_dataset: Any,
        batch_collator: PanoramicTokenizedCollator,
    ) -> AdapterCpuBatch:
        return _prepare_batch_cpu(
            batch_records,
            dataset=batch_dataset,
            collator=batch_collator,
            teacher_target_mode=teacher_target_mode,
            keep_samples=keep_samples,
            need_raw=need_raw,
            need_cond=need_cond,
            teacher_cache=teacher_cache,
        )

    worker_count = max(1, int(prefetch_workers))
    if prefetch_batches <= 0:
        for batch_records in record_batches:
            yield build(batch_records, dataset, collator)
        return

    if worker_count <= 1:
        q: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(prefetch_batches)))
        done = object()

        def worker() -> None:
            batch_dataset = dataset_factory() if dataset_factory is not None else dataset
            try:
                for batch_records in record_batches:
                    q.put(build(batch_records, batch_dataset, collator))
            except BaseException as exc:
                q.put(exc)
            finally:
                q.put(done)

        thread = threading.Thread(target=worker, name="stage2-batch-prefetch", daemon=True)
        thread.start()
        try:
            while True:
                item = q.get()
                if item is done:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            thread.join(timeout=1.0)
        return

    total = len(record_batches)
    task_q: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(prefetch_batches)) + worker_count)
    q: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(prefetch_batches)))
    done = object()

    def producer() -> None:
        for idx, batch_records in enumerate(record_batches):
            task_q.put((idx, batch_records))
        for _ in range(worker_count):
            task_q.put(done)

    def worker(worker_idx: int) -> None:
        batch_dataset = dataset_factory() if dataset_factory is not None else dataset
        batch_collator = collator_factory() if collator_factory is not None else collator
        try:
            while True:
                item = task_q.get()
                try:
                    if item is done:
                        return
                    idx, batch_records = item
                    q.put((idx, build(batch_records, batch_dataset, batch_collator), None))
                except BaseException as exc:
                    if item is done:
                        raise
                    idx = item[0] if isinstance(item, tuple) else -1
                    q.put((idx, None, exc))
                finally:
                    task_q.task_done()
        except BaseException as exc:
            q.put((-1, None, exc))

    producer_thread = threading.Thread(target=producer, name="stage2-batch-prefetch-producer", daemon=True)
    workers = [
        threading.Thread(target=worker, args=(idx,), name=f"stage2-batch-prefetch-{idx}", daemon=True)
        for idx in range(worker_count)
    ]
    producer_thread.start()
    for thread in workers:
        thread.start()

    next_idx = 0
    buffered: dict[int, AdapterCpuBatch] = {}
    try:
        while next_idx < total:
            ready = buffered.pop(next_idx, None)
            if ready is not None:
                yield ready
                next_idx += 1
                continue

            idx, batch, exc = q.get()
            if exc is not None:
                raise exc
            if idx == next_idx:
                yield batch
                next_idx += 1
            else:
                buffered[int(idx)] = batch
    finally:
        producer_thread.join(timeout=1.0)
        for thread in workers:
            thread.join(timeout=1.0)


def _prepare_batch_cpu(
    batch_records: list[dict[str, Any]],
    *,
    dataset: Any,
    collator: PanoramicTokenizedCollator,
    teacher_target_mode: str = "aligned",
    keep_samples: bool = False,
    need_raw: bool = True,
    need_cond: bool = True,
    teacher_cache: NativeTeacherTargetCache | None = None,
) -> AdapterCpuBatch:
    batch_samples: list[dict[str, Any]] = []
    usable_records: list[dict[str, Any]] = []
    for rec in batch_records:
        idx = int(rec["dataset_index"])
        if (idx < 0 or idx >= len(dataset)) and (
            rec.get("clip_idx") is None or rec.get("current_t") is None
        ):
            raise RuntimeError(
                f"Out-of-range dataset_index={idx} without clip/frame fallback after prefilter"
            )
        sample = _sample_from_record(dataset, rec)
        if not _has_trainable_pano_goal(sample):
            raise RuntimeError(
                "Lost trainable pano pixel goal after prefilter: "
                f"dataset_index={idx} kind={sample.get('pano_sample_kind')} "
                f"view={sample.get('pano_view_id')}"
            )
        batch_samples.append(sample)
        usable_records.append(rec)
    if not usable_records:
        raise RuntimeError("Cannot build an empty adapter batch")

    collated = _collate_student_batch_cpu(collator, batch_samples)

    teacher_latents: torch.Tensor | None = None
    teacher_cond: torch.Tensor | None = None
    if teacher_target_mode == "native_sidecar":
        teacher_latents, teacher_cond = _load_native_teacher_targets_cpu(
            usable_records,
            need_raw=need_raw,
            need_cond=need_cond,
            teacher_cache=teacher_cache,
        )

    return AdapterCpuBatch(
        collated=collated,
        teacher_latents=teacher_latents,
        teacher_cond=teacher_cond,
        records=usable_records,
        samples=batch_samples if keep_samples else None,
    )


def _finalize_batch_on_device(
    cpu_batch: AdapterCpuBatch,
    *,
    model: Any,
    device: torch.device,
    teacher_model: Any | None = None,
    teacher_processor: Any | None = None,
    teacher_turn_args: Any | None = None,
) -> AdapterTrainBatch:
    student_latents = _extract_student_latents_from_collated(
        model,
        cpu_batch.collated,
        device,
        batch_size=len(cpu_batch.records),
    )

    teacher_latents: torch.Tensor | None = None
    teacher_cond: torch.Tensor | None = None

    if cpu_batch.teacher_latents is not None:
        teacher_latents = cpu_batch.teacher_latents.to(device, non_blocking=True)
        if teacher_latents.shape[-1] != int(student_latents.shape[-1]):
            teacher_latents = _project_teacher_latents_to_dim(
                teacher_latents,
                model=model,
                target_dim=int(student_latents.shape[-1]),
            )

    if cpu_batch.teacher_cond is not None:
        teacher_cond = cpu_batch.teacher_cond.to(device, non_blocking=True)
        cond_dim = int(model.nextdit_action_head.config.latent_emb_size)
        if teacher_cond.shape[-1] != cond_dim:
            teacher_cond = _project_teacher_latents_to_dim(
                teacher_cond,
                model=model,
                target_dim=cond_dim,
            )

    if teacher_model is not None and teacher_processor is not None:
        if cpu_batch.samples is None:
            raise RuntimeError("Aligned teacher diagnostics require CPU samples")
        try:
            teacher_device = next(teacher_model.parameters()).device
        except StopIteration:
            teacher_device = device
        teacher_latents = compute_aligned_teacher_latents_3584_batch(
            teacher_model, teacher_processor, cpu_batch.samples,
            teacher_device,
            turn_args=teacher_turn_args or make_teacher_turn_args(),
        ).to(device)
        if model.nextdit_action_head is not None:
            cond_projector = model.nextdit_action_head.cond_projector
            proj_dtype = next(cond_projector.parameters()).dtype
            with torch.no_grad():
                teacher_cond = cond_projector(teacher_latents.to(dtype=proj_dtype)).detach()

    trajectory = _collated_tensor(cpu_batch.collated, "trajectory", device)
    trajectory_valid = _collated_tensor(cpu_batch.collated, "trajectory_valid", device)
    traj_images = _collated_tensor(cpu_batch.collated, "traj_images", device)

    batch = AdapterTrainBatch(
        student_latents=student_latents,
        teacher_latents=teacher_latents,  # type: ignore[arg-type]
        teacher_cond=teacher_cond,
        records=cpu_batch.records,
        trajectory=trajectory,
        trajectory_valid=trajectory_valid,
        traj_images=traj_images,
    )
    return batch


@torch.no_grad()
def _evaluate_adapter(
    adapter: nn.Module,
    val_records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    dataset: Any,
    dataset_factory: Any | None = None,
    model: Any,
    processor: Any,
    device: torch.device,
    n_traj_query: int,
    batch_size: int,
    collator: PanoramicTokenizedCollator,
    teacher_target_mode: str = "aligned",
    teacher_model: Any | None = None,
    teacher_processor: Any | None = None,
    teacher_turn_args: Any | None = None,
    rank: int = 0,
    world_size: int = 1,
    prefetch_batches: int = 0,
    prefetch_workers: int = 1,
    teacher_cache: NativeTeacherTargetCache | None = None,
) -> tuple[dict[str, float], int]:
    """Evaluate adapter, optionally sharding val_records across ranks.

    When ``world_size > 1`` each rank processes a strided subset of
    ``val_records``.  Returns ``(raw_sums, count)`` so callers can reduce
    across ranks with ``_reduce_metrics``.
    """
    adapter.eval()
    shard = val_records
    if world_size > 1:
        # Pad to a multiple of world_size so every rank gets ≥1 batch.
        # Mirrors _epoch_rank_records (training) so that _reduce_metrics
        # never receives an empty dict on any rank — which would skip the
        # all_reduce collective and deadlock the other ranks.
        total_size = int(math.ceil(len(val_records) / float(world_size)) * world_size)
        if total_size > len(val_records):
            padded = list(val_records)
            padded.extend(padded[: total_size - len(padded)])
            shard = padded[rank::world_size]
        else:
            shard = val_records[rank::world_size]
    running: dict[str, float] = {}
    count = 0
    num_batches = (len(shard) + batch_size - 1) // batch_size
    pbar = tqdm(
        total=num_batches,
        desc=f"Validating (rank {rank})",
        unit="step",
        ncols=100,
        disable=rank != 0,
    )
    try:
        need_raw = teacher_target_mode == "native_sidecar" and float(args.raw_distill_weight) > 0
        need_cond = teacher_target_mode == "native_sidecar" and float(args.cond_distill_weight) > 0
        keep_samples = teacher_model is not None and teacher_processor is not None
        for cpu_batch in _iter_prepared_cpu_batches(
            _record_batches(shard, batch_size),
            dataset=dataset,
            dataset_factory=dataset_factory,
            collator=collator,
            collator_factory=lambda: _make_pano_collator(
                processor,
                n_traj_query,
                sft_protocol="direct",
            ),
            teacher_target_mode=teacher_target_mode,
            keep_samples=keep_samples,
            need_raw=need_raw,
            need_cond=need_cond,
            teacher_cache=teacher_cache,
            prefetch_batches=prefetch_batches,
            prefetch_workers=prefetch_workers,
        ):
            batch = _finalize_batch_on_device(
                cpu_batch,
                model=model,
                device=device,
                teacher_model=teacher_model,
                teacher_processor=teacher_processor,
                teacher_turn_args=teacher_turn_args,
            )
            pred = adapter(batch.student_latents)
            _loss, metrics, _pred_cond = _compute_adapter_objective(
                model=model,
                pred_raw=pred,
                batch=batch,
                args=args,
            )
            count += 1
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value
            if rank == 0:
                pbar.set_postfix(
                    loss=f"{running.get('loss', 0.0)/count:.5f}",
                    cond_cos=f"{running.get('cond_cos', 0.0)/count:.3f}",
                    gt=f"{running.get('gt', 0.0)/count:.5f}",
                )
            pbar.update(1)
    finally:
        pbar.close()
        adapter.train()
    return running, count


def _save_checkpoint(
    path: Path,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    *,
    epoch: int,
    step: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    adapter_to_save = _unwrap_adapter(adapter)
    torch.save(
        {
            "adapter_type": "pano_latent_space",
            "adapter_state_dict": adapter_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def _load_adapter_config_defaults(config_path: str | None) -> dict[str, Any]:
    """Load adapter-training defaults from a YAML config ``adapter:`` section."""
    if not config_path:
        return {}
    path = Path(config_path).expanduser()
    if not path.is_file():
        LOGGER.warning("Adapter config not found: %s; using CLI defaults.", path)
        return {}
    import yaml

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    adapter_cfg: dict[str, Any] = cfg.get("adapter", {}) or {}
    if not adapter_cfg:
        LOGGER.warning("No 'adapter:' section in %s; using CLI defaults.", path)
        return {}

    defaults: dict[str, Any] = {}

    # Architecture (PanoLatentSpaceAdapter — simple MLP)
    defaults["adapter_hidden_dim"] = int(adapter_cfg.get("hidden_dim", 2048))
    defaults["adapter_dropout"] = float(adapter_cfg.get("dropout", 0.0))

    # Training
    training = adapter_cfg.get("training", {}) or {}
    defaults["epochs"] = int(training.get("epochs", 5))
    defaults["batch_size"] = int(training.get("batch_size", 2))
    defaults["lr"] = float(training.get("lr", 1.0e-4))
    defaults["weight_decay"] = float(training.get("weight_decay", 0.01))
    defaults["grad_clip"] = float(training.get("grad_clip", 1.0))
    defaults["max_samples"] = int(training.get("max_samples", 0))
    defaults["index_mode"] = str(training.get("index_mode", "generic"))
    defaults["val_ratio"] = float(training.get("val_ratio", 0.1))
    defaults["prefetch_batches"] = int(training.get("prefetch_batches", 2))
    defaults["prefetch_workers"] = int(training.get("prefetch_workers", 1))
    defaults["teacher_cache_mode"] = str(training.get("teacher_cache_mode", "none"))
    defaults["teacher_cache_max_items"] = int(training.get("teacher_cache_max_items", 0))
    defaults["teacher_preload_cache"] = bool(training.get("teacher_preload_cache", False))
    defaults["teacher_preload_workers"] = int(training.get("teacher_preload_workers", 4))

    # Teacher
    teacher = adapter_cfg.get("teacher", {}) or {}
    defaults["teacher_target_mode"] = str(teacher.get("target_mode", "aligned"))
    defaults["teacher_torch_dtype"] = str(teacher.get("torch_dtype", "bfloat16"))
    defaults["teacher_attn_implementation"] = str(teacher.get("attn_implementation", "sdpa"))
    defaults["teacher_flash_attn_stub"] = bool(teacher.get("flash_attn_stub", True))

    # Loss
    loss_cfg = adapter_cfg.get("loss", {}) or {}
    defaults["raw_distill_weight"] = float(loss_cfg.get("raw_weight", 0.1))
    defaults["cond_distill_weight"] = float(loss_cfg.get("cond_weight", 1.0))
    defaults["gt_weight"] = float(loss_cfg.get("gt_weight", 0.2))
    defaults["raw_norm_weight"] = float(loss_cfg.get("raw_norm_weight", 0.1))
    defaults["cond_cosine_weight"] = float(loss_cfg.get("cond_cosine_weight", 1.0))
    defaults["cond_smooth_l1_beta"] = float(loss_cfg.get("cond_smooth_l1_beta", 1.0))

    return defaults


def _parse_args_with_config() -> argparse.Namespace:
    """Two-pass parse: load YAML defaults, then let CLI override."""
    # First pass: discover which config files were requested.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--student-config", default="")
    pre.add_argument("--config", default="")  # legacy alias for --student-config
    pre.add_argument("--adapter-config", default="")
    pre_args, _ = pre.parse_known_args()

    student_config = pre_args.student_config or pre_args.config or "configs/train_pano_adapter_stage2_8gpu.yaml"
    adapter_config = pre_args.adapter_config or student_config
    adapter_defaults = _load_adapter_config_defaults(adapter_config)

    p = argparse.ArgumentParser(
        description="Train pano-to-InternNav latent adapter (Stage2)",
    )
    # Register full versions after the lightweight config-discovery parse.
    p.add_argument("--student-config", default=student_config,
                   help="Student model config for build_model (default: %(default)s)")
    p.add_argument("--config", default=student_config, dest="student_config_legacy",
                   help=argparse.SUPPRESS)
    p.add_argument("--adapter-config", default=adapter_config,
                   help="Config for adapter/training defaults (default: same as --student-config)")

    p.add_argument("--root", default=os.environ.get("PANORAMIC_DATA_ROOT", "/workspace/r2r_panoramic_data"))
    p.add_argument("--split", default="train")
    p.add_argument("--teacher-jsonl", default="",
                   help=(
                       "Teacher sidecar JSONL. Required for native_sidecar mode; collect "
                       "with collect_internnav_teacher_sidecar.py --coord-source dataset "
                       "--tensor-output-dir. In legacy aligned mode it may supply a "
                       "prefiltered record subset."
                   ))
    p.add_argument("--base-checkpoint", default="checkpoints/stage1-s2_latest.pth")
    p.add_argument("--internnav-model-path", default=os.environ.get("INTERNNAV_MODEL_PATH", ""))
    p.add_argument("--internnav-repo", default=os.environ.get("INTERNNAV_REPO", "~/InternNav"))
    p.add_argument(
        "--teacher-target-mode",
        choices=["aligned", "sidecar", "native_sidecar"],
        default=adapter_defaults.get("teacher_target_mode", "aligned"),
        help=(
            "native_sidecar: use InternNav native front/lookdown teacher tensors "
            "collected with coord_source=dataset and align by (clip_idx,current_t). "
            "aligned: legacy synthetic pano teacher diagnostic path. "
            "sidecar: legacy pre-collected sidecars."
        ),
    )
    p.add_argument(
        "--compute-teacher-mse", action="store_true", default=False,
        help="Load teacher model and compute MSE against teacher latents for diagnostic logging (adds ~7 GB VRAM).",
    )
    p.add_argument("--teacher-device", default="", help="Device for aligned teacher (default: same as --device)")
    p.add_argument("--teacher-torch-dtype", default=adapter_defaults.get("teacher_torch_dtype", "bfloat16"))
    p.add_argument("--teacher-attn-implementation", default=adapter_defaults.get("teacher_attn_implementation", "sdpa"))
    p.add_argument("--teacher-flash-attn-stub", dest="teacher_flash_attn_stub", action="store_true",
                   default=adapter_defaults.get("teacher_flash_attn_stub", True))
    p.add_argument("--no-teacher-flash-attn-stub", dest="teacher_flash_attn_stub", action="store_false")
    p.add_argument(
        "--index-mode",
        choices=["generic", "internnav_sft"],
        default=adapter_defaults.get("index_mode", "generic"),
        help="Use generic for sidecars collected with the fast default index; internnav_sft exactly rebuilds the old SFT index.",
    )
    p.add_argument("--output-dir", default="outputs/pano_latent_adapter")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--ddp-backend",
        choices=["auto", "nccl", "gloo"],
        default="auto",
        help="Distributed backend when launched with torchrun. auto uses nccl on CUDA.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=adapter_defaults.get("epochs", 5))
    p.add_argument("--batch-size", type=int, default=adapter_defaults.get("batch_size", 2))
    p.add_argument(
        "--prefetch-batches",
        type=int,
        default=adapter_defaults.get("prefetch_batches", 2),
        help="CPU batch prefetch queue depth. 0 disables threaded prefetch.",
    )
    p.add_argument(
        "--prefetch-workers",
        type=int,
        default=adapter_defaults.get("prefetch_workers", 1),
        help="Number of CPU workers preparing future batches when prefetch is enabled.",
    )
    p.add_argument(
        "--teacher-cache-mode",
        choices=["none", "lru", "unbounded"],
        default=adapter_defaults.get("teacher_cache_mode", "none"),
        help="CPU RAM cache for native teacher sidecar tensors.",
    )
    p.add_argument(
        "--teacher-cache-max-items",
        type=int,
        default=adapter_defaults.get("teacher_cache_max_items", 0),
        help="Max cached sidecars in lru mode. 0 means no explicit cap.",
    )
    p.add_argument(
        "--teacher-preload-cache",
        action=argparse.BooleanOptionalAction,
        default=adapter_defaults.get("teacher_preload_cache", False),
        help="Preload this rank's remaining-epoch native teacher sidecars into RAM before training.",
    )
    p.add_argument(
        "--teacher-preload-workers",
        type=int,
        default=adapter_defaults.get("teacher_preload_workers", 4),
        help="CPU workers used by --teacher-preload-cache.",
    )
    p.add_argument("--max-samples", type=int, default=adapter_defaults.get("max_samples", 0))
    p.add_argument("--lr", type=float, default=adapter_defaults.get("lr", 1.0e-4))
    p.add_argument("--weight-decay", type=float, default=adapter_defaults.get("weight_decay", 0.01))
    p.add_argument("--grad-clip", type=float, default=adapter_defaults.get("grad_clip", 1.0))
    p.add_argument("--adapter-hidden-dim", type=int, default=adapter_defaults.get("adapter_hidden_dim", 2048))
    p.add_argument("--adapter-dropout", type=float, default=adapter_defaults.get("adapter_dropout", 0.0))
    p.add_argument("--raw-distill-weight", type=float, default=adapter_defaults.get("raw_distill_weight", 0.1))
    p.add_argument("--cond-distill-weight", type=float, default=adapter_defaults.get("cond_distill_weight", 1.0))
    p.add_argument("--gt-weight", type=float, default=adapter_defaults.get("gt_weight", 0.2))
    p.add_argument("--raw-norm-weight", type=float, default=adapter_defaults.get("raw_norm_weight", 0.1))
    p.add_argument("--cond-cosine-weight", type=float, default=adapter_defaults.get("cond_cosine_weight", 1.0))
    p.add_argument("--cond-smooth-l1-beta", type=float, default=adapter_defaults.get("cond_smooth_l1_beta", 1.0))
    p.add_argument("--pano-max-side-dist-m", type=float, default=6.0)
    p.add_argument(
        "--val-ratio",
        type=float,
        default=adapter_defaults.get("val_ratio", 0.1),
        help="Fraction of records held out as a deterministic validation split (after --max-samples).",
    )
    p.add_argument(
        "--val-records",
        type=int,
        default=0,
        help="Override --val-ratio with an absolute count. 0 disables the override.",
    )
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-every-epochs", type=int, default=1)
    p.add_argument("--resume-adapter", default="")
    return p.parse_args()


def main() -> int:
    args = _parse_args_with_config()
    # Normalise: --config is a legacy alias for --student-config
    if not args.student_config and args.student_config_legacy:
        args.student_config = args.student_config_legacy
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device, rank, local_rank, world_size = _init_distributed(args)
    if not _rank0():
        logging.getLogger().setLevel(logging.WARNING)
    _set_seed(args.seed + rank)

    try:
        cfg = _prepare_config(args)
        teacher_jsonl_str = str(args.teacher_jsonl).strip()

        # Build dataset first — needed both for record filtering AND for
        # auto-generating records when no teacher JSONL is provided.
        dataset = build_trajectory_dataset(
            cfg,
            split=args.split,
            enable_augmentation=False,
            enable_trajectory_augmentation=False,
            load_history_heatmap=False,
            panoramic_vlm_input=True,
            compute_pixel_goal=True,
            compute_pano_view_pixel_goal=True,
            pano_max_side_dist_m=float(args.pano_max_side_dist_m),
            load_lookdown_for_system2=False,
            load_traj_images=True,
        )
        if _rank0():
            LOGGER.info("Dataset samples=%d", len(dataset))

        def make_prefetch_dataset():
            return build_trajectory_dataset(
                cfg,
                split=args.split,
                enable_augmentation=False,
                enable_trajectory_augmentation=False,
                load_history_heatmap=False,
                panoramic_vlm_input=True,
                compute_pixel_goal=True,
                compute_pano_view_pixel_goal=True,
                pano_max_side_dist_m=float(args.pano_max_side_dist_m),
                load_lookdown_for_system2=False,
                load_traj_images=True,
            )

        if teacher_jsonl_str:
            teacher_jsonl = Path(teacher_jsonl_str).expanduser()
            use_sidecar_tensors = args.teacher_target_mode in {"sidecar", "native_sidecar"}
            records = _load_teacher_records(
                teacher_jsonl,
                require_tensor=use_sidecar_tensors,
                require_coord_uv=use_sidecar_tensors,
                require_native_teacher=args.teacher_target_mode == "native_sidecar",
            )
            if args.max_samples > 0:
                records = records[: args.max_samples]
            if not records:
                raise RuntimeError(f"No usable teacher records found in {teacher_jsonl}")
            if _rank0():
                LOGGER.info(
                    "Loaded %d teacher records from %s (world_size=%d teacher_target_mode=%s)",
                    len(records),
                    teacher_jsonl,
                    world_size,
                    args.teacher_target_mode,
                )
            records = _filter_records_with_pano_goals(
                records,
                dataset=dataset,
                validate_sidecar_metadata=args.teacher_target_mode == "sidecar",
                require_native_teacher_sidecar=args.teacher_target_mode == "native_sidecar",
            )
        elif args.teacher_target_mode == "aligned":
            if _rank0():
                LOGGER.info(
                    "No --teacher-jsonl provided; auto-generating records from dataset "
                    "(aligned mode records; teacher loads only with --compute-teacher-mse)"
                )
            records = _build_records_from_dataset(
                dataset,
                max_samples=args.max_samples,
            )
            if _rank0():
                LOGGER.info("Auto-generated %d records from dataset", len(records))
        else:
            raise RuntimeError(
                f"--teacher-jsonl is required for teacher-target-mode={args.teacher_target_mode}"
            )

        if not records:
            raise RuntimeError(
                "No teacher records remain after filtering for structured pano pixel goals"
            )
        if _rank0():
            LOGGER.info("Usable pano pixel teacher records=%d", len(records))

        model = _load_student_model(cfg, args, device)
        processor = model.qwen2_5_vl.processor
        if processor is None:
            raise RuntimeError("Missing Qwen processor")

        teacher_model = None
        teacher_processor = None
        teacher_turn_args = make_teacher_turn_args(seed=args.seed)
        if args.compute_teacher_mse:
            if args.teacher_target_mode == "aligned":
                teacher_device = device
                if str(args.teacher_device).strip():
                    teacher_device = torch.device(str(args.teacher_device).strip())
                teacher_model, teacher_processor = _load_alignment_teacher(args, teacher_device)
                if _rank0():
                    LOGGER.info("Loaded aligned InternNav teacher on %s (MSE diagnostic)", teacher_device)
            elif args.teacher_target_mode == "native_sidecar":
                if _rank0():
                    LOGGER.info(
                        "Using native teacher tensor sidecars; no teacher model is loaded "
                        "during adapter training."
                    )
            elif args.teacher_target_mode == "sidecar":
                raise RuntimeError(
                    "Use --teacher-target-mode=native_sidecar for native InternNav tensor sidecars."
                )
        else:
            if args.teacher_target_mode == "sidecar":
                raise RuntimeError(
                    "Use --teacher-target-mode=native_sidecar for native InternNav tensor sidecars."
                )
            if _rank0() and args.teacher_target_mode == "native_sidecar":
                LOGGER.info(
                    "Using native teacher tensor sidecars with weights: raw=%.3f cond=%.3f gt=%.3f",
                    args.raw_distill_weight,
                    args.cond_distill_weight,
                    args.gt_weight,
                )
            elif _rank0():
                LOGGER.info(
                    "No native teacher sidecar targets active; teacher model not loaded "
                    "(legacy aligned mode can still use --compute-teacher-mse)."
                )

        n_traj_query = int(cfg.get("model", {}).get("action_head", {}).get("nextdit", {}).get("n_query", 4))
        hidden_dim = int(cfg.get("model", {}).get("llm", {}).get("hidden_dim", 3584))
        collator = _make_pano_collator(
            processor,
            n_traj_query,
            sft_protocol="direct",
        )

        adapter = PanoLatentSpaceAdapter(
            dim=hidden_dim,
            hidden_dim=int(args.adapter_hidden_dim),
            dropout=float(args.adapter_dropout),
        ).to(device)

        start_epoch = 0
        global_step = 0
        resume_ckpt: dict[str, Any] | None = None
        if args.resume_adapter:
            resume_ckpt = safe_torch_load(args.resume_adapter, map_location=str(device), trust_checkpoint=True)
            adapter.load_state_dict(resume_ckpt["adapter_state_dict"])
            start_epoch = int(resume_ckpt.get("epoch", 0))
            global_step = int(resume_ckpt.get("step", 0))

        train_adapter: nn.Module = adapter
        if world_size > 1:
            if device.type == "cuda":
                train_adapter = DistributedDataParallel(
                    adapter,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=False,
                )
            else:
                train_adapter = DistributedDataParallel(adapter, find_unused_parameters=False)

        if _rank0():
            LOGGER.info(
                "Adapter: PanoLatentSpaceAdapter dim=%d hidden_dim=%d dropout=%.2f "
                "ddp=%s rank=%d local_rank=%d",
                hidden_dim,
                args.adapter_hidden_dim,
                args.adapter_dropout,
                world_size > 1,
                rank,
                local_rank,
            )
            LOGGER.info("CPU batch prefetch depth=%d", max(0, int(args.prefetch_batches)))
            LOGGER.info("CPU batch prefetch workers=%d", max(1, int(args.prefetch_workers)))
            LOGGER.info(
                "Teacher cache config: mode=%s max_items=%d preload=%s preload_workers=%d",
                args.teacher_cache_mode,
                int(args.teacher_cache_max_items),
                bool(args.teacher_preload_cache),
                int(args.teacher_preload_workers),
            )
        optimizer = torch.optim.AdamW(train_adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if resume_ckpt is not None and "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            if _rank0():
                LOGGER.info(
                    "Resumed adapter from %s at epoch=%d step=%d",
                    args.resume_adapter,
                    start_epoch,
                    global_step,
                )

        out_dir = Path(args.output_dir).expanduser()
        if _rank0():
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / "train_args.json").open("w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2, ensure_ascii=False)

        train_records, val_records = _split_train_val(records, args)
        if _rank0():
            LOGGER.info("Split records: train=%d val=%d", len(train_records), len(val_records))
            with (out_dir / "split.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "train_indices": [int(rec["dataset_index"]) for rec in train_records],
                        "val_indices": [int(rec["dataset_index"]) for rec in val_records],
                    },
                    f,
                    indent=2,
                )
        if _distributed_available():
            dist.barrier()

        teacher_cache = NativeTeacherTargetCache(
            mode=str(args.teacher_cache_mode),
            max_items=int(args.teacher_cache_max_items),
        )
        if args.teacher_target_mode == "native_sidecar" and teacher_cache.enabled:
            if _rank0():
                LOGGER.info("Native teacher RAM cache enabled: %s", teacher_cache.stats())
            if bool(args.teacher_preload_cache):
                preload_records = _records_for_rank_remaining_epochs(
                    train_records,
                    args=args,
                    rank=rank,
                    world_size=world_size,
                    start_epoch=start_epoch,
                )
                if _rank0():
                    LOGGER.info(
                        "Preloading native teacher cache for rank-local remaining epochs: "
                        "records=%d workers=%d",
                        len(preload_records),
                        int(args.teacher_preload_workers),
                    )
                _preload_teacher_cache(
                    preload_records,
                    teacher_cache=teacher_cache,
                    workers=int(args.teacher_preload_workers),
                    rank=rank,
                )
                if _distributed_available():
                    dist.barrier()

        for epoch in range(start_epoch, args.epochs):
            epoch_records = _epoch_rank_records(
                train_records,
                seed=args.seed,
                epoch=epoch,
                rank=rank,
                world_size=world_size,
            )
            running: dict[str, float] = {}
            count = 0
            train_adapter.train()

            num_batches = (len(epoch_records) + args.batch_size - 1) // args.batch_size
            pbar = tqdm(
                total=num_batches,
                desc=f"Epoch {epoch + 1}/{args.epochs}",
                unit="step",
                disable=not _rank0(),
                ncols=140,
            )
            need_raw = args.teacher_target_mode == "native_sidecar" and float(args.raw_distill_weight) > 0
            need_cond = args.teacher_target_mode == "native_sidecar" and float(args.cond_distill_weight) > 0
            keep_samples = teacher_model is not None and teacher_processor is not None
            cpu_batch_iter = iter(_iter_prepared_cpu_batches(
                _record_batches(epoch_records, args.batch_size),
                dataset=dataset,
                dataset_factory=make_prefetch_dataset,
                collator=collator,
                collator_factory=lambda: _make_pano_collator(
                    processor,
                    n_traj_query,
                    sft_protocol="direct",
                ),
                teacher_target_mode=args.teacher_target_mode,
                keep_samples=keep_samples,
                need_raw=need_raw,
                need_cond=need_cond,
                teacher_cache=teacher_cache,
                prefetch_batches=max(0, int(args.prefetch_batches)),
                prefetch_workers=max(1, int(args.prefetch_workers)),
            ))
            for _step_idx in range(num_batches):
                wait_t0 = time.perf_counter()
                cpu_batch = next(cpu_batch_iter)
                prefetch_wait_s = time.perf_counter() - wait_t0

                finalize_t0 = time.perf_counter()
                batch = _finalize_batch_on_device(
                    cpu_batch,
                    model=model,
                    device=device,
                    teacher_model=teacher_model,
                    teacher_processor=teacher_processor,
                    teacher_turn_args=teacher_turn_args,
                )
                finalize_s = time.perf_counter() - finalize_t0

                train_t0 = time.perf_counter()
                pred = train_adapter(batch.student_latents)
                loss, metrics, _pred_cond = _compute_adapter_objective(
                    model=model,
                    pred_raw=pred,
                    batch=batch,
                    args=args,
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    clip_grad_norm_(train_adapter.parameters(), args.grad_clip)
                optimizer.step()
                train_step_s = time.perf_counter() - train_t0

                global_step += world_size
                count += 1
                metrics["prefetch_wait_s"] = prefetch_wait_s
                metrics["finalize_s"] = finalize_s
                metrics["train_step_s"] = train_step_s
                for key, value in metrics.items():
                    running[key] = running.get(key, 0.0) + value

                # Update tqdm postfix with running averages.
                avg = {k: v / max(count, 1) for k, v in running.items()}
                pbar.set_postfix(
                    loss=f"{avg.get('loss', 0):.5f}",
                    raw_cos=f"{avg.get('raw_cos', 0):.3f}",
                    cond_cos=f"{avg.get('cond_cos', 0):.3f}",
                    gt=f"{avg.get('gt', 0):.5f}",
                )
                pbar.update(1)

                if _rank0() and args.log_interval > 0 and count % args.log_interval == 0:
                    LOGGER.info(
                        "epoch=%d local_step=%d global_step=%d loss=%.6f "
                        "raw=%.6f raw_cos=%.4f cond=%.6f cond_cos=%.4f "
                        "cond_smooth_l1=%.6f gt=%.6f wait=%.3fs finalize=%.3fs train=%.3fs",
                        epoch + 1, count, global_step,
                        avg.get("loss", 0.0),
                        avg.get("raw", 0.0),
                        avg.get("raw_cos", 0.0),
                        avg.get("cond", 0.0),
                        avg.get("cond_cos", 0.0),
                        avg.get("cond_smooth_l1", 0.0),
                        avg.get("gt", 0.0),
                        avg.get("prefetch_wait_s", 0.0),
                        avg.get("finalize_s", 0.0),
                        avg.get("train_step_s", 0.0),
                    )
            try:
                next(cpu_batch_iter)
            except StopIteration:
                pass
            pbar.close()

            epoch_metrics = _reduce_metrics(running, count, device)
            if _rank0():
                LOGGER.info("epoch=%d train metrics=%s", epoch + 1, epoch_metrics)

            is_last_epoch = epoch == args.epochs - 1

            # Distributed validation: each rank processes a strided subset of
            # val_records, then metrics are all_reduced.  DDP stays alive —
            # since validation uses @torch.no_grad() there are no pending
            # gradient syncs, so NCCL watchdogs are not at risk.
            val_metrics: dict[str, float] | None = None
            if val_records and is_last_epoch:
                raw_adapter = _unwrap_adapter(train_adapter)
                val_sums, val_count = _evaluate_adapter(
                    raw_adapter,
                    val_records,
                    args=args,
                    dataset=dataset,
                    dataset_factory=make_prefetch_dataset,
                    model=model,
                    processor=processor,
                    device=device,
                    n_traj_query=n_traj_query,
                    batch_size=args.batch_size,
                    collator=collator,
                    teacher_target_mode=args.teacher_target_mode,
                    teacher_model=teacher_model,
                    teacher_processor=teacher_processor,
                    teacher_turn_args=teacher_turn_args,
                    rank=rank,
                    world_size=world_size,
                    prefetch_batches=max(0, int(args.prefetch_batches)),
                    prefetch_workers=max(1, int(args.prefetch_workers)),
                    teacher_cache=teacher_cache,
                )
                val_metrics = _reduce_metrics(val_sums, val_count, device)
                if _rank0():
                    LOGGER.info("epoch=%d val   metrics=%s", epoch + 1, val_metrics)

            if rank == 0:
                combined_metrics = dict(epoch_metrics)
                if val_metrics is not None:
                    for key, value in val_metrics.items():
                        combined_metrics[f"val_{key}"] = value
                _save_checkpoint(
                    out_dir / "latest.pth",
                    train_adapter,
                    optimizer,
                    args,
                    epoch=epoch + 1,
                    step=global_step,
                    metrics=combined_metrics,
                )
                if args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
                    _save_checkpoint(
                        out_dir / f"epoch_{epoch + 1:03d}.pth",
                        train_adapter,
                        optimizer,
                        args,
                        epoch=epoch + 1,
                        step=global_step,
                        metrics=combined_metrics,
                    )
            if _distributed_available():
                dist.barrier()

        if rank == 0:
            LOGGER.info("Saved adapter to %s", out_dir / "latest.pth")
        return 0
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())

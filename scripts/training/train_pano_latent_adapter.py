#!/usr/bin/env python3
"""
Train a geometry-aware panoramic-to-InternNav latent adapter.

This is intentionally narrower than Stage2 bridge training:
  student: panoramic Qwen TRAJ hidden states from HeatmapVLN / Stage1-S2
  target:  InternNav teacher traj_latents projected through cond_projector to 768
  train:   adapter only; Pano-System2 and InternNav System1 stay frozen

Frozen VLM + frozen InternNav System1 let this test answer one question:
can a teacher-guided translator map panoramic latent queries plus structured
goal geometry into the NextDiT condition space that System1 actually consumes?

..  warning::
    The aligned teacher latent targets use a **synthetic single-turn structured
    pano protocol** (``view: front\\npixel: u v``).  InternNav's native training
    protocol is a **two-turn lookdown + raw coordinate** conditioning path.
    Even for front-view samples the teacher latent is not strictly in the
    InternNav native System1 distribution.

    The primary supervision signal for ALL views is the **GT trajectory loss**
    through the frozen System1 NextDiT, which is native.  Teacher latent
    distillation (cosine + MSE + policy) is a supplementary signal gated to
    front-view samples only, accepting a small distribution gap.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
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
    load_config,
    safe_torch_load,
)
from src.data.factory import build_trajectory_dataset
from src.data.pano_teacher_alignment import (
    compute_aligned_teacher_latents_3584_batch,
    compute_aligned_teacher_latents_768_batch,
    has_structured_pano_pixel_goal,
    make_teacher_turn_args,
)
from src.data.panoramic_tokenized_collator import PanoramicTokenizedCollator
from src.models.adapters import (
    GeometryAwarePanoToNextDiTAdapter,
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
    records: list[dict[str, Any]]
    trajectory: torch.Tensor | None = None
    trajectory_valid: torch.Tensor | None = None
    traj_images: torch.Tensor | None = None


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
    missing, unexpected, loaded = _load_normalized_state_dict(model, lora_state)
    LOGGER.info(
        "Loaded Stage1-S2 LoRA checkpoint %s: checkpoint_tensors=%d lora_tensors=%d "
        "loaded=%d missing=%d unexpected=%d compatible_lora=%d",
        args.base_checkpoint,
        len(state),
        len(lora_state),
        loaded,
        len(missing),
        len(unexpected),
        len(compatible_lora_keys),
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
    if int(payload_idx) != int(expected_idx):
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
            tensor_info = teacher.get("system1", {}).get("tensor_sidecar", {})
            tensor_path = _resolve_tensor_path(jsonl_path, tensor_info.get("path"))
            if require_tensor and (tensor_path is None or not tensor_path.is_file()):
                continue
            if require_coord_uv and teacher.get("coord_uv") is None:
                continue
            rec["_tensor_path"] = str(tensor_path) if tensor_path is not None else None
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
    """Load the exact `(clip_idx, current_t)` state recorded in the sidecar.

    The collector can run on the fast generic dataset index while older
    sidecars may use the InternNav SFT index.  Using the recorded clip/frame
    pair avoids coupling adapter training to whichever index mode was used.
    """
    idx = int(rec["dataset_index"])
    clip_idx = rec.get("clip_idx")
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


def _move_pano_inputs_to_device(pano_inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in pano_inputs.items()
    }


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
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    collator = PanoramicTokenizedCollator(
        processor,
        n_traj_query=n_traj_query,
        sft_mode=True,
        sft_protocol=sft_protocol,
        structured_pano_output=True,
    )
    # The collator clears each sample dict after packing to release large image
    # references. Keep the originals intact for the aligned teacher pass.
    batch = collator([{key: value for key, value in sample.items()} for sample in samples])
    pano_inputs = _move_pano_inputs_to_device(batch["pano_inputs"], device)
    histories = batch["history_frames"].to(device)
    current = batch["current_frame"].to(device)
    lq = model.latent_queries.expand(len(samples), -1, -1).to(
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
    traj_hs = traj_hs.detach()
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
    for rec in records:
        idx = rec.get("dataset_index")
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
        if not _has_trainable_pano_goal(sample):
            skipped += 1
            continue
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
            if str(sv).lower() != str(sample.get("pano_view_id") or "").lower():
                mismatched += 1
                LOGGER.warning(
                    "Skip sidecar record dataset_index=%s: pano_view_id mismatch "
                    "sidecar=%r dataset=%r",
                    idx, sv, sample.get("pano_view_id"),
                )
                continue
            dp = sample.get("pano_pixel_goal")
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
        "missing_metadata=%d invalid_tensor=%d",
        len(filtered),
        skipped,
        failed,
        mismatched,
        missing_metadata,
        invalid_tensor,
    )
    if mismatched or missing_metadata or invalid_tensor:
        LOGGER.warning(
            "%d sidecar records failed strict alignment validation and were dropped. "
            "Re-collect sidecars if this count is unexpected.",
            mismatched + missing_metadata + invalid_tensor,
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
    """Latent-space distillation loss, gated to front-view samples only.

    Non-front samples have out-of-distribution teacher latents (the teacher
    never saw the corresponding image).  ``front_mask`` zeros their contribution
    so they only receive GT trajectory supervision.
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
        return value.to(device)
    return value.to(device=device, dtype=dtype)


def _policy_and_gt_losses(
    *,
    model: Any,
    pred_cond: torch.Tensor,
    teacher_cond: torch.Tensor,
    batch: AdapterTrainBatch,
    policy_weight: float,
    gt_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if policy_weight <= 0 and gt_weight <= 0:
        zero = pred_cond.sum() * 0.0
        return zero, {"policy_loss": 0.0, "gt_loss": 0.0}
    if batch.trajectory is None:
        zero = pred_cond.sum() * 0.0
        return zero, {"policy_loss": 0.0, "gt_loss": 0.0}

    head = getattr(model, "nextdit_action_head", None)
    if head is None:
        raise RuntimeError("Policy/GT loss requires model.nextdit_action_head")

    dit_dtype = next(head.action_encoder.parameters()).dtype
    gt = batch.trajectory.to(device=pred_cond.device, dtype=dit_dtype)
    images = batch.traj_images.to(device=pred_cond.device) if batch.traj_images is not None else None
    valid = (
        batch.trajectory_valid.to(device=pred_cond.device)
        if batch.trajectory_valid is not None
        else None
    )

    pred_exp, gt_exp, images_exp, valid_exp = head._expand_sequence_training_inputs(
        pred_cond.to(dtype=dit_dtype),
        gt,
        images,
        valid,
    )
    teacher_exp, _gt_t, _images_t, _valid_t = head._expand_sequence_training_inputs(
        teacher_cond.to(device=pred_cond.device, dtype=dit_dtype),
        gt,
        images,
        valid,
    )
    noisy, timesteps, target_velocity = head.sample_flow_matching_inputs(gt_exp)

    # Build front-view mask for policy loss gating.  When traj_images has the
    # multi-current shape (B, N, H, W, 3), _expand_sequence_training_inputs
    # flattens B → B×N; the mask must be expanded to match.
    fm = batch.front_mask
    front_mask_exp: torch.Tensor | None = None
    if fm is not None and policy_weight > 0:
        expanded = (
            images is not None
            and images.ndim == 5
            and gt.ndim == 4
        )
        if expanded:
            num_frames = images.shape[1]  # N from [B, N, H, W, 3]
            front_mask_exp = fm.to(device=pred_cond.device, dtype=torch.float32)
            front_mask_exp = front_mask_exp.unsqueeze(1).expand(-1, num_frames).reshape(-1)
            if valid_exp is not None:
                front_mask_exp = front_mask_exp * valid_exp.float()
        else:
            front_mask_exp = fm.to(device=pred_cond.device, dtype=torch.float32)
            if valid_exp is not None:
                front_mask_exp = front_mask_exp * valid_exp.float()

    policy_loss = pred_cond.sum() * 0.0
    if policy_weight > 0:
        with torch.no_grad():
            teacher_velocity = head.predict_velocity_from_projected(
                teacher_exp,
                noisy,
                timesteps,
                traj_images=images_exp,
            )
        student_velocity = head.predict_velocity_from_projected(
            pred_exp,
            noisy,
            timesteps,
            traj_images=images_exp,
        )
        if front_mask_exp is not None and front_mask_exp.sum() > 0:
            policy_loss = head.masked_velocity_mse(
                student_velocity,
                teacher_velocity.detach(),
                trajectory_valid=front_mask_exp,
            )
        elif front_mask_exp is not None:
            policy_loss = pred_cond.sum() * 0.0
        else:
            policy_loss = head.masked_velocity_mse(
                student_velocity,
                teacher_velocity.detach(),
                trajectory_valid=valid_exp,
            )
    else:
        student_velocity = None

    gt_loss = pred_cond.sum() * 0.0
    if gt_weight > 0:
        if student_velocity is None:
            student_velocity = head.predict_velocity_from_projected(
                pred_exp,
                noisy,
                timesteps,
                traj_images=images_exp,
            )
        gt_loss = head.masked_velocity_mse(
            student_velocity,
            target_velocity,
            trajectory_valid=valid_exp,
        )

    total = policy_weight * policy_loss + gt_weight * gt_loss
    return total, {
        "policy_loss": float(policy_loss.detach().item()),
        "gt_loss": float(gt_loss.detach().item()),
    }


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
) -> AdapterTrainBatch:
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

    student_latents, collated = _extract_student_latents(
        model, processor, batch_samples, device, n_traj_query,
        sft_protocol="direct", return_batch=True,
    )

    # Teacher latents are only needed for MSE loss (legacy).  Pure GT-loss
    # training skips teacher inference entirely.
    teacher_latents: torch.Tensor | None = None
    if teacher_model is not None and teacher_processor is not None:
        try:
            teacher_device = next(teacher_model.parameters()).device
        except StopIteration:
            teacher_device = device
        teacher_latents = compute_aligned_teacher_latents_3584_batch(
            teacher_model, teacher_processor, batch_samples,
            teacher_device,
            turn_args=teacher_turn_args or make_teacher_turn_args(),
        ).to(device)

    trajectory = _collated_tensor(collated, "trajectory", device)
    trajectory_valid = _collated_tensor(collated, "trajectory_valid", device)
    traj_images = _collated_tensor(collated, "traj_images", device)

    batch = AdapterTrainBatch(
        student_latents=student_latents,
        teacher_latents=teacher_latents,  # type: ignore[arg-type]
        records=usable_records,
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
    dataset: Any,
    model: Any,
    processor: Any,
    device: torch.device,
    n_traj_query: int,
    batch_size: int,
    teacher_target_mode: str = "aligned",
    teacher_model: Any | None = None,
    teacher_processor: Any | None = None,
    teacher_turn_args: Any | None = None,
) -> dict[str, float]:
    adapter.eval()
    running: dict[str, float] = {}
    count = 0
    num_batches = (len(val_records) + batch_size - 1) // batch_size
    pbar = tqdm(total=num_batches, desc="Validating", unit="step", ncols=100, disable=not _rank0())
    try:
        for start in range(0, len(val_records), batch_size):
            batch_records = val_records[start:start + batch_size]
            batch = _build_batch(
                batch_records,
                dataset=dataset,
                model=model,
                processor=processor,
                device=device,
                n_traj_query=n_traj_query,
                teacher_target_mode=teacher_target_mode,
                teacher_model=teacher_model,
                teacher_processor=teacher_processor,
                teacher_turn_args=teacher_turn_args,
            )
            pred = adapter(batch.student_latents)
            mse = float(F.mse_loss(
                pred.float(),
                batch.teacher_latents.to(device=pred.device, dtype=torch.float32),
            ).item()) if batch.teacher_latents is not None else 0.0
            gt_val = 0.0
            if batch.trajectory is not None and model.nextdit_action_head is not None:
                head = model.nextdit_action_head
                proj_dtype = next(head.cond_projector.parameters()).dtype
                projected = head.cond_projector(pred.to(dtype=proj_dtype))
                gt_t = batch.trajectory.to(device=pred.device, dtype=proj_dtype)
                images_t = batch.traj_images.to(device=pred.device) if batch.traj_images is not None else None
                valid_t = batch.trajectory_valid.to(device=pred.device) if batch.trajectory_valid is not None else None
                pe, ge, ie, ve = head._expand_sequence_training_inputs(projected, gt_t, images_t, valid_t)
                noisy, ts, tv = head.sample_flow_matching_inputs(ge)
                pv = head.predict_velocity_from_projected(pe, noisy, ts, traj_images=ie)
                gt_val = float(head.masked_velocity_mse(pv, tv, ve).item())
            val_loss = gt_val
            count += 1
            running["loss"] = running.get("loss", 0.0) + val_loss
            running["mse"] = running.get("mse", 0.0) + mse
            running["gt"] = running.get("gt", 0.0) + gt_val
            avg_mse = running["mse"] / count
            pbar.set_postfix(mse=f"{avg_mse:.6f}", gt=f"{running['gt']/count:.5f}")
            pbar.update(1)
    finally:
        pbar.close()
        adapter.train()
    return {k: v / max(count, 1) for k, v in running.items()}


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

    # Teacher
    teacher = adapter_cfg.get("teacher", {}) or {}
    defaults["teacher_target_mode"] = str(teacher.get("target_mode", "aligned"))
    defaults["teacher_torch_dtype"] = str(teacher.get("torch_dtype", "bfloat16"))
    defaults["teacher_attn_implementation"] = str(teacher.get("attn_implementation", "sdpa"))
    defaults["teacher_flash_attn_stub"] = bool(teacher.get("flash_attn_stub", True))

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
                   help="Teacher sidecar JSONL. When empty and teacher-target-mode=aligned, auto-generate records from dataset.")
    p.add_argument("--base-checkpoint", default="checkpoints/stage1-s2_latest.pth")
    p.add_argument("--internnav-model-path", default=os.environ.get("INTERNNAV_MODEL_PATH", ""))
    p.add_argument("--internnav-repo", default=os.environ.get("INTERNNAV_REPO", "~/InternNav"))
    p.add_argument(
        "--teacher-target-mode",
        choices=["aligned", "sidecar"],
        default=adapter_defaults.get("teacher_target_mode", "aligned"),
        help=(
            "aligned: on-the-fly InternNav teacher 768 latents from dataset pano goals "
            "(recommended; no traj_latents_768 sidecar required). "
            "sidecar: load pre-collected traj_latents_768 tensors from --teacher-jsonl."
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
    p.add_argument("--max-samples", type=int, default=adapter_defaults.get("max_samples", 0))
    p.add_argument("--lr", type=float, default=adapter_defaults.get("lr", 1.0e-4))
    p.add_argument("--weight-decay", type=float, default=adapter_defaults.get("weight_decay", 0.01))
    p.add_argument("--grad-clip", type=float, default=adapter_defaults.get("grad_clip", 1.0))
    p.add_argument("--adapter-hidden-dim", type=int, default=adapter_defaults.get("adapter_hidden_dim", 2048))
    p.add_argument("--adapter-dropout", type=float, default=adapter_defaults.get("adapter_dropout", 0.0))
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

        if teacher_jsonl_str:
            teacher_jsonl = Path(teacher_jsonl_str).expanduser()
            use_sidecar_tensors = args.teacher_target_mode == "sidecar"
            records = _load_teacher_records(
                teacher_jsonl,
                require_tensor=use_sidecar_tensors,
                require_coord_uv=use_sidecar_tensors,
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
                validate_sidecar_metadata=use_sidecar_tensors,
            )
        elif args.teacher_target_mode == "aligned":
            if _rank0():
                LOGGER.info(
                    "No --teacher-jsonl provided; auto-generating records from dataset "
                    "(aligned mode — teacher runs on-the-fly during training)"
                )
            records = _build_records_from_dataset(
                dataset,
                max_samples=args.max_samples,
            )
            if _rank0():
                LOGGER.info("Auto-generated %d records from dataset", len(records))
        else:
            raise RuntimeError(
                "--teacher-jsonl is required for teacher-target-mode=sidecar"
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
            elif args.teacher_target_mode == "sidecar":
                raise RuntimeError(
                    "sidecar mode stores 768-dim latents but PanoLatentSpaceAdapter "
                    "targets 3584-dim. Use --teacher-target-mode=aligned or omit "
                    "--compute-teacher-mse."
                )
        else:
            if args.teacher_target_mode == "sidecar":
                raise RuntimeError(
                    "PanoLatentSpaceAdapter uses pure GT loss. "
                    "sidecar mode (768-dim teacher latents) is not supported."
                )
            if _rank0():
                LOGGER.info("Pure GT loss — teacher model not loaded "
                            "(use --compute-teacher-mse for diagnostic MSE)")

        n_traj_query = int(cfg.get("model", {}).get("action_head", {}).get("nextdit", {}).get("n_query", 4))
        hidden_dim = int(cfg.get("model", {}).get("llm", {}).get("hidden_dim", 3584))

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
            for start in range(0, len(epoch_records), args.batch_size):
                batch_records = epoch_records[start:start + args.batch_size]
                batch = _build_batch(
                    batch_records,
                    dataset=dataset,
                    model=model,
                    processor=processor,
                    device=device,
                    n_traj_query=n_traj_query,
                    teacher_target_mode=args.teacher_target_mode,
                    teacher_model=teacher_model,
                    teacher_processor=teacher_processor,
                    teacher_turn_args=teacher_turn_args,
                )
                pred = train_adapter(batch.student_latents)
                mse = torch.tensor(0.0, device=pred.device)
                if batch.teacher_latents is not None:
                    mse = F.mse_loss(
                        pred.float(),
                        batch.teacher_latents.to(device=pred.device, dtype=torch.float32),
                    )
                gt_loss = torch.tensor(0.0, device=pred.device)
                if batch.trajectory is not None and model.nextdit_action_head is not None:
                    head = model.nextdit_action_head
                    proj_dtype = next(head.cond_projector.parameters()).dtype
                    projected = head.cond_projector(pred.to(dtype=proj_dtype))  # (B,Q,3584)→(B,Q,768)
                    gt = batch.trajectory.to(device=pred.device, dtype=proj_dtype)
                    images = batch.traj_images.to(device=pred.device) if batch.traj_images is not None else None
                    valid = batch.trajectory_valid.to(device=pred.device) if batch.trajectory_valid is not None else None
                    pred_exp, gt_exp, images_exp, valid_exp = head._expand_sequence_training_inputs(
                        projected, gt, images, valid,
                    )
                    noisy, timesteps, target_vel = head.sample_flow_matching_inputs(gt_exp)
                    pred_vel = head.predict_velocity_from_projected(
                        pred_exp, noisy, timesteps, traj_images=images_exp,
                    )
                    gt_loss = head.masked_velocity_mse(pred_vel, target_vel, valid_exp)
                loss = gt_loss
                metrics = {
                    "loss": float(loss.detach().item()),
                    "mse": float(mse.detach().item()),
                    "gt": float(gt_loss.detach().item()),
                }

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    clip_grad_norm_(train_adapter.parameters(), args.grad_clip)
                optimizer.step()

                global_step += world_size
                count += 1
                for key, value in metrics.items():
                    running[key] = running.get(key, 0.0) + value

                # Update tqdm postfix with running averages.
                avg = {k: v / max(count, 1) for k, v in running.items()}
                pbar.set_postfix(
                    loss=f"{avg.get('loss', 0):.5f}",
                    mse=f"{avg.get('mse', 0):.6f}",
                    gt=f"{avg.get('gt', 0):.5f}",
                )
                pbar.update(1)

                if _rank0() and args.log_interval > 0 and count % args.log_interval == 0:
                    LOGGER.info(
                        "epoch=%d local_step=%d global_step=%d loss=%.6f mse=%.8f gt=%.6f",
                        epoch + 1, count, global_step,
                        avg.get("loss", 0.0), avg.get("mse", 0.0), avg.get("gt", 0.0),
                    )
            pbar.close()

            epoch_metrics = _reduce_metrics(running, count, device)
            if _rank0():
                LOGGER.info("epoch=%d train metrics=%s", epoch + 1, epoch_metrics)

            is_last_epoch = epoch == args.epochs - 1
            if _distributed_available() and is_last_epoch:
                # Validation can exceed NCCL's watchdog timeout. Tear down the
                # process group before the final rank-0-only validation pass.
                dist.barrier()
                _cleanup_distributed()

            val_metrics: dict[str, float] | None = None
            if rank == 0 and val_records and is_last_epoch:
                val_metrics = _evaluate_adapter(
                    _unwrap_adapter(train_adapter),
                    val_records,
                    dataset=dataset,
                    model=model,
                    processor=processor,
                    device=device,
                    n_traj_query=n_traj_query,
                    batch_size=args.batch_size,
                    teacher_target_mode=args.teacher_target_mode,
                    teacher_model=teacher_model,
                    teacher_processor=teacher_processor,
                    teacher_turn_args=teacher_turn_args,
                )
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

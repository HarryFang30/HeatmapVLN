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

from scripts.training.model_builder import build_model
from scripts.training.utils import (
    _load_normalized_state_dict,
    load_config,
    safe_torch_load,
)
from src.data.factory import build_trajectory_dataset
from src.data.panoramic_tokenized_collator import PanoramicTokenizedCollator
from src.models.adapters import GeometryAwarePanoToNextDiTAdapter, view_ids_to_indices

LOGGER = logging.getLogger("pano_latent_adapter")


@dataclass
class AdapterTrainBatch:
    student_latents: torch.Tensor
    teacher_latents: torch.Tensor
    view_indices: torch.Tensor
    goal_pixels: torch.Tensor
    image_hw: torch.Tensor
    trajectory: torch.Tensor | None
    trajectory_valid: torch.Tensor | None
    traj_images: torch.Tensor | None
    records: list[dict[str, Any]]


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
    ckpt = safe_torch_load(str(path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    for key in ("model_state_dict", "trainable_state_dict", "state_dict"):
        state = ckpt.get(key)
        if isinstance(state, dict):
            return state
    if all(torch.is_tensor(value) for value in ckpt.values()):
        return ckpt
    raise KeyError(f"No model_state_dict/trainable_state_dict/state_dict in {path}")


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

    if args.base_checkpoint:
        state = _extract_checkpoint_state_dict(args.base_checkpoint)
        missing, unexpected, loaded = _load_normalized_state_dict(model, state)
        LOGGER.info(
            "Loaded base/student checkpoint %s: loaded=%d missing=%d unexpected=%d",
            args.base_checkpoint,
            loaded,
            len(missing),
            len(unexpected),
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


def _load_teacher_records(jsonl_path: Path, *, require_tensor: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            teacher = rec.get("teacher", {})
            tensor_info = teacher.get("system1", {}).get("tensor_sidecar", {})
            tensor_path = _resolve_tensor_path(jsonl_path, tensor_info.get("path"))
            if require_tensor and (tensor_path is None or not tensor_path.is_file()):
                continue
            if teacher.get("coord_uv") is None:
                continue
            rec["_tensor_path"] = str(tensor_path) if tensor_path is not None else None
            records.append(rec)
    return records


def _prepare_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    cfg["data"]["root"] = args.root
    if "paths" in cfg:
        cfg["paths"]["dataset_root"] = args.root
    traj_cfg = cfg.setdefault("data", {}).setdefault("trajectory", {})
    use_traj_images = bool(getattr(args, "use_traj_images", args.index_mode == "internnav_sft"))
    traj_cfg["panoramic_vlm_input"] = True
    traj_cfg["compute_pixel_goal"] = use_traj_images
    traj_cfg["compute_pano_view_pixel_goal"] = True
    traj_cfg["pano_max_side_dist_m"] = float(getattr(args, "pano_max_side_dist_m", 6.0))
    traj_cfg["load_lookdown_for_system2"] = False
    traj_cfg["load_traj_images"] = use_traj_images
    traj_cfg["enable_trajectory_augmentation"] = False
    traj_cfg["require_sft_target"] = args.index_mode == "internnav_sft"
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
    batch = collator(samples)
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
        if not path:
            raise RuntimeError(f"Missing tensor sidecar for dataset_index={rec.get('dataset_index')}")
        payload = safe_torch_load(path, map_location="cpu", trust_checkpoint=True)
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


def _filter_records_with_pano_goals(
    records: list[dict[str, Any]],
    *,
    dataset: Any,
) -> list[dict[str, Any]]:
    """Keep records whose exact frame has a structured pano pixel goal.

    DDP needs every rank to execute the same number of backward calls.  Filtering
    once before sharding avoids rank-local batch skips when a teacher sidecar
    record has no student pano pixel target under the current C3 rule.
    """
    filtered: list[dict[str, Any]] = []
    skipped = 0
    failed = 0
    for rec in records:
        try:
            sample = _sample_from_record(dataset, rec)
        except Exception as exc:
            failed += 1
            LOGGER.warning(
                "Skip teacher record dataset_index=%s: failed to load exact sample (%r)",
                rec.get("dataset_index"),
                exc,
            )
            continue
        if _has_trainable_pano_goal(sample):
            filtered.append(rec)
        else:
            skipped += 1
    LOGGER.info(
        "Filtered teacher records for pano pixel goals: kept=%d skipped_no_pano_goal=%d failed=%d",
        len(filtered),
        skipped,
        failed,
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
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_f = pred.float()
    target_f = target.float()
    cos = F.cosine_similarity(pred_f.flatten(1), target_f.flatten(1), dim=1)
    cos_loss = (1.0 - cos).mean()
    mse_loss = F.mse_loss(pred_f, target_f)
    pred_norm = pred_f.norm(dim=-1)
    target_norm = target_f.norm(dim=-1).clamp_min(1.0e-6)
    norm_ratio = pred_norm / target_norm
    if norm_loss_type == "log_ratio":
        # Symmetric in pred_norm/target_norm vs target_norm/pred_norm and finite at 1.
        norm_loss = (torch.log(norm_ratio.clamp_min(1.0e-6))) ** 2
        norm_loss = norm_loss.mean()
    elif norm_loss_type == "ratio":
        norm_loss = ((norm_ratio - 1.0) ** 2).mean()
    else:
        raise ValueError(f"Unknown norm_loss_type: {norm_loss_type!r}")
    loss = cosine_weight * cos_loss + mse_weight * mse_loss + norm_weight * norm_loss
    return loss, {
        "loss": float(loss.detach().item()),
        "cosine": float(cos.mean().detach().item()),
        "cos_loss": float(cos_loss.detach().item()),
        "mse_loss": float(mse_loss.detach().item()),
        "norm_loss": float(norm_loss.detach().item()),
        "pred_norm": float(pred_norm.mean().detach().item()),
        "target_norm": float(target_norm.mean().detach().item()),
        "norm_ratio": float(norm_ratio.mean().detach().item()),
    }


def _has_trainable_pano_goal(sample: dict[str, Any]) -> bool:
    kind = str(sample.get("pano_sample_kind") or "").lower()
    if kind and kind != "pixel":
        return False
    if sample.get("pano_pixel_goal") is None:
        return False
    view_id = str(sample.get("pano_view_id") or "").lower()
    return view_id in {"front", "right", "back", "left"}


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
) -> AdapterTrainBatch | None:
    batch_samples: list[dict[str, Any]] = []
    usable_records: list[dict[str, Any]] = []
    for rec in batch_records:
        idx = int(rec["dataset_index"])
        if (idx < 0 or idx >= len(dataset)) and (
            rec.get("clip_idx") is None or rec.get("current_t") is None
        ):
            LOGGER.warning("Skip out-of-range dataset_index=%s without clip/frame fallback", idx)
            continue
        sample = _sample_from_record(dataset, rec)
        if not _has_trainable_pano_goal(sample):
            LOGGER.warning(
                "Skip dataset_index=%s without trainable pano pixel goal after prefilter (kind=%s view=%s)",
                idx,
                sample.get("pano_sample_kind"),
                sample.get("pano_view_id"),
            )
            continue
        batch_samples.append(sample)
        usable_records.append(rec)
    if not usable_records:
        return None

    view_indices, goal_pixels, image_hw = _goal_tensors_from_samples(batch_samples, device)

    student_latents, collated = _extract_student_latents(
        model,
        processor,
        batch_samples,
        device,
        n_traj_query,
        sft_protocol="direct",
        return_batch=True,
    )
    target_dim = int(model.nextdit_action_head.config.latent_emb_size)
    teacher_latents = _load_teacher_latents(
        usable_records,
        device,
        model=model,
        target_dim=target_dim,
    )
    trajectory = _collated_tensor(collated, "trajectory", device)
    trajectory_valid = _collated_tensor(collated, "trajectory_valid", device)
    traj_images = _collated_tensor(collated, "traj_images", device)
    return AdapterTrainBatch(
        student_latents=student_latents,
        teacher_latents=teacher_latents,
        view_indices=view_indices,
        goal_pixels=goal_pixels,
        image_hw=image_hw,
        trajectory=trajectory,
        trajectory_valid=trajectory_valid,
        traj_images=traj_images,
        records=usable_records,
    )


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
    cosine_weight: float,
    mse_weight: float,
    norm_weight: float,
    norm_loss_type: str,
    policy_weight: float,
    gt_weight: float,
) -> dict[str, float]:
    adapter.eval()
    running: dict[str, float] = {}
    count = 0
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
            )
            if batch is None:
                continue
            pred = adapter(
                batch.student_latents,
                batch.view_indices,
                batch.goal_pixels,
                batch.image_hw,
            )
            latent_loss, metrics = _latent_loss(
                pred,
                batch.teacher_latents.to(device=pred.device, dtype=pred.dtype),
                cosine_weight=cosine_weight,
                mse_weight=mse_weight,
                norm_weight=norm_weight,
                norm_loss_type=norm_loss_type,
            )
            policy_loss, policy_metrics = _policy_and_gt_losses(
                model=model,
                pred_cond=pred,
                teacher_cond=batch.teacher_latents,
                batch=batch,
                policy_weight=policy_weight,
                gt_weight=gt_weight,
            )
            metrics.update(policy_metrics)
            metrics["loss"] = float((latent_loss + policy_loss).detach().item())
            count += 1
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value
    finally:
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
            "adapter_state_dict": adapter_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train pano-to-InternNav latent adapter")
    p.add_argument("--config", default="configs/train_config_internnav_8gpu_stage2_wider.yaml")
    p.add_argument("--root", default=os.environ.get("PANORAMIC_DATA_ROOT", "/workspace/r2r_panoramic_data"))
    p.add_argument("--split", default="train")
    p.add_argument("--teacher-jsonl", required=True)
    p.add_argument("--base-checkpoint", default="checkpoints/stage1-s2_latest.pth")
    p.add_argument("--internnav-model-path", default=os.environ.get("INTERNNAV_MODEL_PATH", ""))
    p.add_argument(
        "--index-mode",
        choices=["generic", "internnav_sft"],
        default="generic",
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
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--lr", type=float, default=1.0e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--adapter-hidden-dim", type=int, default=2048)
    p.add_argument("--adapter-dim", type=int, default=768)
    p.add_argument("--adapter-output-dim", type=int, default=768)
    p.add_argument("--adapter-num-heads", type=int, default=8)
    p.add_argument("--adapter-geometry-embed-dim", type=int, default=64)
    p.add_argument("--adapter-horizontal-fov-deg", type=float, default=90.0)
    p.add_argument("--adapter-dropout", type=float, default=0.0)
    p.add_argument(
        "--adapter-n-layers",
        type=int,
        default=1,
        help="Number of decoder-style Transformer adapter layers.",
    )
    p.add_argument("--pano-max-side-dist-m", type=float, default=6.0)
    p.add_argument(
        "--use-traj-images",
        dest="use_traj_images",
        action="store_true",
        help="Load System1 front_down visual-memory frames for policy/GT losses.",
    )
    p.add_argument("--no-use-traj-images", dest="use_traj_images", action="store_false")
    p.set_defaults(use_traj_images=True)
    p.add_argument(
        "--residual",
        dest="residual",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--no-residual", dest="residual", action="store_false", help=argparse.SUPPRESS)
    p.set_defaults(residual=False)
    p.add_argument(
        "--zero-init",
        dest="zero_init",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--no-zero-init", dest="zero_init", action="store_false", help=argparse.SUPPRESS)
    p.set_defaults(zero_init=False)
    p.add_argument(
        "--pre-norm",
        dest="pre_norm",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--no-pre-norm",
        dest="pre_norm",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    p.set_defaults(pre_norm=False)
    p.add_argument(
        "--output-affine",
        dest="output_affine",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--no-output-affine",
        dest="output_affine",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    p.set_defaults(output_affine=True)
    p.add_argument("--cosine-weight", type=float, default=0.1)
    p.add_argument("--mse-weight", type=float, default=1.0)
    p.add_argument("--policy-weight", type=float, default=1.0)
    p.add_argument("--gt-weight", type=float, default=1.0)
    p.add_argument(
        "--norm-weight",
        type=float,
        default=0.0,
        help="Penalty on pred/teacher latent norm ratio. Set to 0 to disable.",
    )
    p.add_argument(
        "--norm-loss-type",
        choices=["log_ratio", "ratio"],
        default="log_ratio",
        help="log_ratio is symmetric around 1.0 and well-behaved when pred_norm is too large.",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
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
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device, rank, local_rank, world_size = _init_distributed(args)
    if not _rank0():
        logging.getLogger().setLevel(logging.WARNING)
    _set_seed(args.seed + rank)

    try:
        cfg = _prepare_config(args)
        teacher_jsonl = Path(args.teacher_jsonl).expanduser()
        records = _load_teacher_records(teacher_jsonl)
        if args.max_samples > 0:
            records = records[: args.max_samples]
        if not records:
            raise RuntimeError(f"No usable teacher records found in {teacher_jsonl}")

        if _rank0():
            LOGGER.info(
                "Loaded %d teacher records from %s (world_size=%d)",
                len(records),
                teacher_jsonl,
                world_size,
            )
        dataset = build_trajectory_dataset(
            cfg,
            split=args.split,
            enable_augmentation=False,
            enable_trajectory_augmentation=False,
            load_history_heatmap=False,
            panoramic_vlm_input=True,
            compute_pixel_goal=bool(args.use_traj_images),
            compute_pano_view_pixel_goal=True,
            pano_max_side_dist_m=float(args.pano_max_side_dist_m),
            load_lookdown_for_system2=False,
            load_traj_images=bool(args.use_traj_images),
        )
        if _rank0():
            LOGGER.info("Dataset samples=%d", len(dataset))

        records = _filter_records_with_pano_goals(records, dataset=dataset)
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
        n_traj_query = int(cfg.get("model", {}).get("action_head", {}).get("nextdit", {}).get("n_query", 4))
        hidden_dim = int(cfg.get("model", {}).get("llm", {}).get("hidden_dim", 3584))

        target_dim = int(model.nextdit_action_head.config.latent_emb_size)
        if int(args.adapter_output_dim) != target_dim:
            raise RuntimeError(
                f"Adapter output dim must match NextDiT latent dim {target_dim}; "
                f"got {args.adapter_output_dim}"
            )

        adapter = GeometryAwarePanoToNextDiTAdapter(
            student_dim=hidden_dim,
            adapter_dim=int(args.adapter_dim),
            output_dim=target_dim,
            num_query=n_traj_query,
            num_layers=int(args.adapter_n_layers),
            num_heads=int(args.adapter_num_heads),
            ffn_dim=int(args.adapter_hidden_dim),
            dropout=float(args.adapter_dropout),
            geometry_embed_dim=int(args.adapter_geometry_embed_dim),
            horizontal_fov_deg=float(args.adapter_horizontal_fov_deg),
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
                    find_unused_parameters=True,
                )
            else:
                train_adapter = DistributedDataParallel(adapter, find_unused_parameters=True)

        if _rank0():
            LOGGER.info(
                "Adapter: geometry-aware decoder layers=%d student_dim=%d adapter_dim=%d output_dim=%d "
                "heads=%d ffn_dim=%d use_traj_images=%s loss_weights(mse=%.3g cos=%.3g policy=%.3g gt=%.3g) "
                "ddp=%s rank=%d local_rank=%d",
                args.adapter_n_layers,
                hidden_dim,
                args.adapter_dim,
                target_dim,
                args.adapter_num_heads,
                args.adapter_hidden_dim,
                args.use_traj_images,
                args.mse_weight,
                args.cosine_weight,
                args.policy_weight,
                args.gt_weight,
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

            for start in range(0, len(epoch_records), args.batch_size):
                batch_records = epoch_records[start:start + args.batch_size]
                batch = _build_batch(
                    batch_records,
                    dataset=dataset,
                    model=model,
                    processor=processor,
                    device=device,
                    n_traj_query=n_traj_query,
                )
                if batch is None:
                    continue

                pred = train_adapter(
                    batch.student_latents,
                    batch.view_indices,
                    batch.goal_pixels,
                    batch.image_hw,
                )
                latent_loss, metrics = _latent_loss(
                    pred,
                    batch.teacher_latents.to(device=pred.device, dtype=pred.dtype),
                    cosine_weight=args.cosine_weight,
                    mse_weight=args.mse_weight,
                    norm_weight=args.norm_weight,
                    norm_loss_type=args.norm_loss_type,
                )
                policy_loss, policy_metrics = _policy_and_gt_losses(
                    model=model,
                    pred_cond=pred,
                    teacher_cond=batch.teacher_latents,
                    batch=batch,
                    policy_weight=args.policy_weight,
                    gt_weight=args.gt_weight,
                )
                loss = latent_loss + policy_loss
                metrics.update(policy_metrics)
                metrics["loss"] = float(loss.detach().item())

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    clip_grad_norm_(train_adapter.parameters(), args.grad_clip)
                optimizer.step()

                global_step += world_size
                count += 1
                for key, value in metrics.items():
                    running[key] = running.get(key, 0.0) + value

                if _rank0() and args.log_interval > 0 and count % args.log_interval == 0:
                    avg = {k: v / max(count, 1) for k, v in running.items()}
                    LOGGER.info(
                        "epoch=%d local_step=%d global_step=%d loss=%.5f cosine=%.5f mse=%.6f "
                        "policy=%.6f gt=%.6f norm_ratio=%.3f pred_norm=%.3f target_norm=%.3f",
                        epoch + 1,
                        count,
                        global_step,
                        avg.get("loss", 0.0),
                        avg.get("cosine", 0.0),
                        avg.get("mse_loss", 0.0),
                        avg.get("policy_loss", 0.0),
                        avg.get("gt_loss", 0.0),
                        avg.get("norm_ratio", 0.0),
                        avg.get("pred_norm", 0.0),
                        avg.get("target_norm", 0.0),
                    )

            epoch_metrics = _reduce_metrics(running, count, device)
            if _rank0():
                LOGGER.info("epoch=%d train metrics=%s", epoch + 1, epoch_metrics)

            val_metrics: dict[str, float] | None = None
            if _rank0() and val_records:
                val_metrics = _evaluate_adapter(
                    _unwrap_adapter(train_adapter),
                    val_records,
                    dataset=dataset,
                    model=model,
                    processor=processor,
                    device=device,
                    n_traj_query=n_traj_query,
                    batch_size=args.batch_size,
                    cosine_weight=args.cosine_weight,
                    mse_weight=args.mse_weight,
                    norm_weight=args.norm_weight,
                    norm_loss_type=args.norm_loss_type,
                    policy_weight=args.policy_weight,
                    gt_weight=args.gt_weight,
                )
                LOGGER.info("epoch=%d val   metrics=%s", epoch + 1, val_metrics)

            if _rank0():
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

        if _rank0():
            LOGGER.info("Saved adapter to %s", out_dir / "latest.pth")
        return 0
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())

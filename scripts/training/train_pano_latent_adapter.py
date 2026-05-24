#!/usr/bin/env python3
"""
Train a small panoramic-to-InternNav latent adapter.

This is intentionally narrower than Stage2 bridge training:
  student: panoramic Qwen TRAJ hidden states from HeatmapVLN / Stage1-S2
  target:  InternNav teacher traj_latents saved by collect_internnav_teacher_sidecar.py
  train:   a small adapter only

Frozen VLM + frozen InternNav System1 let this test answer one question:
can a lightweight interface adapter map panoramic latent queries back onto the
InternNav latent manifold that System1 already understands?
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from scripts.training.model_builder import build_model
from scripts.training.utils import (
    _load_normalized_state_dict,
    load_config,
    safe_torch_load,
)
from src.data.factory import build_trajectory_dataset
from src.data.panoramic_tokenized_collator import PanoramicTokenizedCollator

LOGGER = logging.getLogger("pano_latent_adapter")


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
    traj_cfg["panoramic_vlm_input"] = True
    traj_cfg["load_lookdown_for_system2"] = True
    traj_cfg["load_traj_images"] = args.index_mode == "internnav_sft"
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
) -> torch.Tensor:
    collator = PanoramicTokenizedCollator(
        processor,
        n_traj_query=n_traj_query,
        sft_mode=True,
        sft_protocol="internnav",
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
    return traj_hs.detach()


def _load_teacher_latents(records: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    latents = []
    for rec in records:
        path = rec.get("_tensor_path")
        if not path:
            raise RuntimeError(f"Missing tensor sidecar for dataset_index={rec.get('dataset_index')}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "traj_latents" not in payload:
            raise RuntimeError(f"{path} has no traj_latents")
        latent = payload["traj_latents"].detach()
        if latent.dim() == 3 and latent.shape[0] == 1:
            latent = latent.squeeze(0)
        latents.append(latent)
    return torch.stack(latents, dim=0).to(device)


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


def _build_batch(
    batch_records: list[dict[str, Any]],
    *,
    dataset: Any,
    model: Any,
    processor: Any,
    device: torch.device,
    n_traj_query: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    batch_samples: list[dict[str, Any]] = []
    usable_records: list[dict[str, Any]] = []
    for rec in batch_records:
        idx = int(rec["dataset_index"])
        if idx < 0 or idx >= len(dataset):
            LOGGER.warning("Skip out-of-range dataset_index=%s", idx)
            continue
        sample = _sample_from_record(dataset, rec)
        coord_uv = rec["teacher"]["coord_uv"]
        batch_samples.append(_copy_sample_for_collator(sample, coord_uv))
        usable_records.append(rec)
    if not usable_records:
        empty = torch.empty(0)
        return empty, empty, []

    student_latents = _extract_student_latents(
        model,
        processor,
        batch_samples,
        device,
        n_traj_query,
    )
    teacher_latents = _load_teacher_latents(usable_records, device).to(dtype=student_latents.dtype)
    return student_latents, teacher_latents, usable_records


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
) -> dict[str, float]:
    adapter.eval()
    running: dict[str, float] = {}
    count = 0
    try:
        for start in range(0, len(val_records), batch_size):
            batch_records = val_records[start:start + batch_size]
            student_latents, teacher_latents, usable = _build_batch(
                batch_records,
                dataset=dataset,
                model=model,
                processor=processor,
                device=device,
                n_traj_query=n_traj_query,
            )
            if not usable:
                continue
            pred = adapter(student_latents)
            _, metrics = _latent_loss(
                pred,
                teacher_latents,
                cosine_weight=cosine_weight,
                mse_weight=mse_weight,
                norm_weight=norm_weight,
                norm_loss_type=norm_loss_type,
            )
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
    torch.save(
        {
            "adapter_state_dict": adapter.state_dict(),
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--lr", type=float, default=1.0e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--adapter-hidden-dim", type=int, default=2048)
    p.add_argument("--adapter-dropout", type=float, default=0.0)
    p.add_argument(
        "--adapter-n-layers",
        type=int,
        default=1,
        help="Number of hidden GELU layers in the projector MLP (1 means Linear-GELU-Linear).",
    )
    p.add_argument(
        "--residual",
        dest="residual",
        action="store_true",
        help="Use student_latent + adapter_delta. Off by default because pano and InternNav latents have different scales.",
    )
    p.add_argument("--no-residual", dest="residual", action="store_false")
    p.set_defaults(residual=False)
    p.add_argument(
        "--zero-init",
        dest="zero_init",
        action="store_true",
        help="Zero-initialize the final projection layer. Useful mainly with --residual.",
    )
    p.add_argument("--no-zero-init", dest="zero_init", action="store_false")
    p.set_defaults(zero_init=False)
    p.add_argument(
        "--pre-norm",
        dest="pre_norm",
        action="store_true",
        help="Apply a LayerNorm to the student latent before the MLP (strips per-token scale).",
    )
    p.add_argument(
        "--no-pre-norm",
        dest="pre_norm",
        action="store_false",
        help="Default: feed raw student latents into the MLP so per-token scale is preserved.",
    )
    p.set_defaults(pre_norm=False)
    p.add_argument(
        "--output-affine",
        dest="output_affine",
        action="store_true",
        help="Add a per-dim learnable scale/bias on the adapter output (helps match teacher norm).",
    )
    p.add_argument(
        "--no-output-affine",
        dest="output_affine",
        action="store_false",
        help="Disable the per-dim output affine.",
    )
    p.set_defaults(output_affine=True)
    p.add_argument("--cosine-weight", type=float, default=1.0)
    p.add_argument("--mse-weight", type=float, default=0.1)
    p.add_argument(
        "--norm-weight",
        type=float,
        default=0.1,
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
    _set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = _prepare_config(args)
    teacher_jsonl = Path(args.teacher_jsonl).expanduser()
    records = _load_teacher_records(teacher_jsonl)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    if not records:
        raise RuntimeError(f"No usable teacher records found in {teacher_jsonl}")

    LOGGER.info("Loaded %d teacher records from %s", len(records), teacher_jsonl)
    dataset = build_trajectory_dataset(
        cfg,
        split=args.split,
        enable_augmentation=False,
        enable_trajectory_augmentation=False,
        load_history_heatmap=False,
        panoramic_vlm_input=True,
        load_lookdown_for_system2=True,
        load_traj_images=args.index_mode == "internnav_sft",
    )
    LOGGER.info("Dataset samples=%d", len(dataset))

    model = _load_student_model(cfg, args, device)
    processor = model.qwen2_5_vl.processor
    if processor is None:
        raise RuntimeError("Missing Qwen processor")
    n_traj_query = int(cfg.get("model", {}).get("action_head", {}).get("nextdit", {}).get("n_query", 4))
    hidden_dim = int(cfg.get("model", {}).get("llm", {}).get("hidden_dim", 3584))

    adapter = PanoToInternNavLatentAdapter(
        dim=hidden_dim,
        hidden_dim=args.adapter_hidden_dim,
        dropout=args.adapter_dropout,
        residual=args.residual,
        zero_init=args.zero_init,
        pre_norm=args.pre_norm,
        n_layers=args.adapter_n_layers,
        output_affine=args.output_affine,
    ).to(device)
    LOGGER.info(
        "Adapter: residual=%s pre_norm=%s output_affine=%s n_layers=%d hidden_dim=%d dim=%d",
        args.residual,
        args.pre_norm,
        args.output_affine,
        args.adapter_n_layers,
        args.adapter_hidden_dim,
        hidden_dim,
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    global_step = 0
    if args.resume_adapter:
        ckpt = torch.load(args.resume_adapter, map_location=device, weights_only=False)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("step", 0))
        LOGGER.info("Resumed adapter from %s at epoch=%d step=%d", args.resume_adapter, start_epoch, global_step)

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    train_records, val_records = _split_train_val(records, args)
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

    rng = random.Random(args.seed)
    for epoch in range(start_epoch, args.epochs):
        rng.shuffle(train_records)
        running: dict[str, float] = {}
        count = 0
        adapter.train()

        for start in range(0, len(train_records), args.batch_size):
            batch_records = train_records[start:start + args.batch_size]
            student_latents, teacher_latents, usable = _build_batch(
                batch_records,
                dataset=dataset,
                model=model,
                processor=processor,
                device=device,
                n_traj_query=n_traj_query,
            )
            if not usable:
                continue

            pred = adapter(student_latents)
            loss, metrics = _latent_loss(
                pred,
                teacher_latents,
                cosine_weight=args.cosine_weight,
                mse_weight=args.mse_weight,
                norm_weight=args.norm_weight,
                norm_loss_type=args.norm_loss_type,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                clip_grad_norm_(adapter.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1
            count += 1
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value

            if args.log_interval > 0 and global_step % args.log_interval == 0:
                avg = {k: v / max(count, 1) for k, v in running.items()}
                LOGGER.info(
                    "epoch=%d step=%d loss=%.5f cosine=%.5f mse=%.6f norm_ratio=%.3f pred_norm=%.3f target_norm=%.3f",
                    epoch + 1,
                    global_step,
                    avg.get("loss", 0.0),
                    avg.get("cosine", 0.0),
                    avg.get("mse_loss", 0.0),
                    avg.get("norm_ratio", 0.0),
                    avg.get("pred_norm", 0.0),
                    avg.get("target_norm", 0.0),
                )

        epoch_metrics = {k: v / max(count, 1) for k, v in running.items()}
        LOGGER.info("epoch=%d train metrics=%s", epoch + 1, epoch_metrics)

        val_metrics: dict[str, float] | None = None
        if val_records:
            val_metrics = _evaluate_adapter(
                adapter,
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
            )
            LOGGER.info("epoch=%d val   metrics=%s", epoch + 1, val_metrics)

        combined_metrics = dict(epoch_metrics)
        if val_metrics is not None:
            for key, value in val_metrics.items():
                combined_metrics[f"val_{key}"] = value
        _save_checkpoint(
            out_dir / "latest.pth",
            adapter,
            optimizer,
            args,
            epoch=epoch + 1,
            step=global_step,
            metrics=combined_metrics,
        )
        if args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
            _save_checkpoint(
                out_dir / f"epoch_{epoch + 1:03d}.pth",
                adapter,
                optimizer,
                args,
                epoch=epoch + 1,
                step=global_step,
                metrics=combined_metrics,
            )

    LOGGER.info("Saved adapter to %s", out_dir / "latest.pth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Base-only vs Stage2 bridge comparison on gold pixel_goal samples (train path B).

- base-only: InternNav System1 + Stage1-S2 LoRA (no stage2_latest bridge weights)
- stage2:    base-only + latent_queries + cond_projector from stage2 checkpoint

Uses PanoramicTokenizedCollator + train-aligned ``extract_traj_hidden_states`` path,
then ``get_trajectory`` and ``compute_loss`` vs GT.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from scripts.evaluation.latent_parity_train_vs_eval import (
    PATH_LEN_RE,
    _build_train_batch,
    _gt_trajectory_for_summary,
    _parse_path_len,
    _run_get_trajectory,
)
from scripts.evaluation.r2r_val_unseen import (
    _extract_checkpoint_state_dict,
    _load_compatible_state_dict,
    _resolve_internnav_model_path,
    _trajectory_debug_summary,
    _verify_internnav_system1_loaded,
)
from scripts.training.model_builder import build_model
from scripts.training.utils import load_config
from src.data.factory import build_trajectory_dataset

LOGGER = logging.getLogger("bridge_ab")


def _cast_traj_images(
    traj_images: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """DepthAnything path expects activations in the same dtype as rgb_model (bf16)."""
    return traj_images.to(device=device, dtype=dtype)


def _build_traj_images_for_infer(
    sample: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Training-style [anchor, current] pair for get_trajectory."""
    ti = sample.get("traj_images")
    if ti is None:
        return None
    if not torch.is_tensor(ti):
        ti = torch.as_tensor(ti)
    if ti.dim() != 4:
        return None
    if ti.shape[0] == 1:
        pair = torch.stack([ti[0], ti[0]], dim=0)
    else:
        pair = torch.stack([ti[0], ti[-1]], dim=0)
    return _cast_traj_images(pair.unsqueeze(0), device, dtype)


def _prepare_gt_for_loss(
    sample: dict,
    device: torch.device,
    dtype: torch.dtype,
    traj_images_infer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Single forward step → match (B, T, 3) GT and (B, 2, H, W, 3) images (no N-expand)."""
    traj = sample.get("trajectory")
    if traj is None:
        raise RuntimeError("Sample missing trajectory for loss")
    if not torch.is_tensor(traj):
        traj = torch.as_tensor(traj)
    if traj.dim() == 3:
        gt = traj[0].unsqueeze(0)
        tv = sample.get("trajectory_valid")
        if tv is not None:
            if not torch.is_tensor(tv):
                tv = torch.as_tensor(tv)
            traj_valid = tv[0].reshape(1) if tv.dim() > 0 else tv.reshape(1)
        else:
            traj_valid = None
    elif traj.dim() == 2:
        gt = traj.unsqueeze(0)
        tv = sample.get("trajectory_valid")
        if tv is not None:
            traj_valid = (
                tv.reshape(1) if torch.is_tensor(tv) else torch.as_tensor([float(tv)])
            )
        else:
            traj_valid = None
    else:
        raise RuntimeError(f"Unexpected trajectory shape {tuple(traj.shape)}")

    gt = gt.to(device=device, dtype=dtype)
    if traj_valid is not None:
        traj_valid = traj_valid.to(device)
    traj_images = traj_images_infer
    return gt, traj_valid, traj_images


def _snapshot_bridge_params(model) -> dict[str, torch.Tensor]:
    snap = {"latent_queries": model.latent_queries.detach().cpu().clone()}
    for name, param in model.nextdit_action_head.named_parameters():
        if name.startswith("cond_projector."):
            snap[f"nextdit_action_head.{name}"] = param.detach().cpu().clone()
    return snap


def _bridge_delta(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, float]:
    out = {}
    for key in before:
        if key not in after:
            continue
        diff = (after[key].float() - before[key].float()).abs()
        out[f"{key}_max_abs"] = float(diff.max().item())
        out[f"{key}_mean_abs"] = float(diff.mean().item())
    return out


def load_bridge_model(
    cfg: dict,
    base_ckpt: Path,
    device: torch.device,
    *,
    stage2_ckpt: Path | None = None,
    label: str,
):
    internnav_path = _resolve_internnav_model_path(cfg)
    if internnav_path:
        print(f"[{label}] InternNav model path: {internnav_path}", flush=True)

    model = build_model(cfg, device=str(device), verbose=False)
    model = model.to(device)
    _verify_internnav_system1_loaded(model, internnav_path)

    model.qwen2_5_vl._load_model()
    if model.qwen2_5_vl.processor is None:
        raise RuntimeError("Qwen processor is None after _load_model()")
    if model.nextdit_action_head is None or model.latent_queries is None:
        raise RuntimeError("nextdit_action_head / latent_queries not enabled")

    bridge_before = _snapshot_bridge_params(model)

    base_sd = _extract_checkpoint_state_dict(str(base_ckpt))
    _load_compatible_state_dict(model, base_sd, str(base_ckpt), label=f"[{label}] Base checkpoint")

    bridge_after_base = _snapshot_bridge_params(model)
    stage2_delta = {}

    if stage2_ckpt is not None:
        stage2_sd = _extract_checkpoint_state_dict(str(stage2_ckpt))
        _load_compatible_state_dict(
            model, stage2_sd, str(stage2_ckpt), label=f"[{label}] Stage2 bridge",
        )
        bridge_after_stage2 = _snapshot_bridge_params(model)
        stage2_delta = _bridge_delta(bridge_after_base, bridge_after_stage2)
        print(f"[{label}] Bridge delta after stage2 load: {stage2_delta}", flush=True)
    else:
        print(f"[{label}] Skipping stage2 bridge (base-only / InternNav bridge)", flush=True)

    internnav_delta = _bridge_delta(bridge_before, bridge_after_base)
    model.eval()
    return model, internnav_delta, stage2_delta


def train_path_eval_one(
    model,
    sample: dict,
    device: torch.device,
    *,
    n_traj_query: int,
    num_sample_trajs: int,
    action_scale: float,
    traj_image_size: tuple[int, int],
) -> dict:
    processor = model.qwen2_5_vl.processor
    pano_inputs, pano_num_histories = _build_train_batch(sample, processor, n_traj_query)
    pano_inputs = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in pano_inputs.items()
    }
    lq = model.latent_queries.expand(1, -1, -1).to(device=device, dtype=model.config.dtype)

    with torch.no_grad():
        qwen_out = model.qwen2_5_vl(
            history_frames=sample["history_frames"].unsqueeze(0).to(device),
            current_frame=sample["current_frame"].unsqueeze(0).to(device),
            panoramic_inputs=pano_inputs,
            panoramic_num_histories=pano_num_histories,
            latent_queries=lq,
            return_hidden_states=False,
        )
    traj_hs = qwen_out["traj_hidden_states"]
    if traj_hs is None:
        raise RuntimeError("No traj_hidden_states from train forward")

    img_dtype = model.config.dtype
    traj_images = _build_traj_images_for_infer(sample, device, img_dtype)
    if traj_images is None:
        from scripts.evaluation.latent_parity_train_vs_eval import _build_eval_traj_images

        traj_images = _build_eval_traj_images(sample, device, traj_image_size)

    traj_out = _run_get_trajectory(
        model, traj_hs, traj_images,
        num_sample_trajs=num_sample_trajs,
        action_scale=action_scale,
    )

    gt_traj, traj_valid, traj_images_loss = _prepare_gt_for_loss(
        sample, device, img_dtype, traj_images,
    )
    with torch.cuda.amp.autocast(dtype=img_dtype, enabled=(device.type == "cuda")):
        loss_out = model.nextdit_action_head.compute_loss(
            traj_hs,
            gt_traj,
            traj_images=traj_images_loss,
            trajectory_valid=traj_valid,
        )
    traj_loss = float(loss_out["loss"].detach().cpu().item())

    return {
        **traj_out,
        "trajectory_loss": traj_loss,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Base-only vs Stage2 bridge on train path B")
    p.add_argument("--config", default="configs/train_config_internnav_8gpu.yaml")
    p.add_argument("--base-checkpoint", default="checkpoints/stage1-s2_latest.pth")
    p.add_argument("--stage2-checkpoint", default="checkpoints/stage2_latest.pth")
    p.add_argument("--root", default="/workspace/r2r_panoramic_audit_v2")
    p.add_argument("--split", default="train")
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="logs/bridge_base_vs_stage2.jsonl")
    return p.parse_args()


def _summarize_branch(records: list[dict], prefix: str) -> dict[str, float]:
    path_lens = [r[f"{prefix}_path_len"] for r in records if r.get(f"{prefix}_path_len") is not None]
    losses = [r[f"{prefix}_trajectory_loss"] for r in records if r.get(f"{prefix}_trajectory_loss") is not None]
    forwards = [r.get(f"{prefix}_forward_count", 0) for r in records]
    n = len(records)
    return {
        "n": n,
        "path_len_mean": float(np.mean(path_lens)) if path_lens else float("nan"),
        "path_len_median": float(np.median(path_lens)) if path_lens else float("nan"),
        "loss_mean": float(np.mean(losses)) if losses else float("nan"),
        "loss_median": float(np.median(losses)) if losses else float("nan"),
        "forward_any_pct": 100.0 * sum(1 for f in forwards if f > 0) / n if n else 0.0,
        "path_short_pct": 100.0 * sum(1 for p in path_lens if p < 0.5) / len(path_lens) if path_lens else 0.0,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    cfg = load_config(args.config)
    cfg["data"]["root"] = args.root
    if "paths" in cfg:
        cfg["paths"]["dataset_root"] = args.root

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    traj_cfg = cfg.get("model", {}).get("action_head", {}).get("nextdit", {})
    num_sample_trajs = int(traj_cfg.get("num_sample_trajs", 32))
    action_scale = float(cfg.get("data", {}).get("action_scale", 4.0))
    n_traj_query = int(traj_cfg.get("n_query", 4))
    traj_size = tuple(cfg.get("data", {}).get("traj_image_size", [224, 224]))

    dataset = build_trajectory_dataset(cfg, split=args.split)
    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    # Pick indices first so both runs use identical samples.
    chosen: list[tuple[int, dict]] = []
    skipped = 0
    for idx in indices:
        if len(chosen) >= args.num_samples:
            break
        sample = dataset[idx]
        if sample.get("pixel_goal") is None:
            skipped += 1
            continue
        if float(sample.get("is_stop", 0.0)) > 0.5:
            skipped += 1
            continue
        if sample.get("trajectory") is None:
            skipped += 1
            continue
        chosen.append((idx, sample))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")

    records: list[dict] = []

    print("\n=== Loading base-only (no stage2 bridge) ===", flush=True)
    model_base, _, _ = load_bridge_model(
        cfg, Path(args.base_checkpoint), device,
        stage2_ckpt=None, label="base-only",
    )
    for idx, sample in chosen:
        sample_copy = {k: v for k, v in sample.items()}
        try:
            base_out = train_path_eval_one(
                model_base, sample_copy, device,
                n_traj_query=n_traj_query,
                num_sample_trajs=num_sample_trajs,
                action_scale=action_scale,
                traj_image_size=traj_size,
            )
        except Exception as exc:
            LOGGER.exception("base-only failed idx=%s: %s", idx, exc)
            continue

        gt_t = _gt_trajectory_for_summary(sample)
        gt_summary = _trajectory_debug_summary(gt_t, 1, action_scale) if gt_t is not None else None
        gt_path_len = _parse_path_len(gt_summary) if gt_summary else None

        rec = {
            "dataset_index": idx,
            "pixel_goal": [int(sample["pixel_goal"][0]), int(sample["pixel_goal"][1])],
            "gt_path_len": gt_path_len,
            "base_only_path_len": base_out["path_len"],
            "base_only_forward_count": base_out["forward_count"],
            "base_only_trajectory_loss": base_out["trajectory_loss"],
            "base_only_trajectory_summary": base_out["trajectory_summary"],
        }
        records.append(rec)

    del model_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n=== Loading stage2 (base + bridge) ===", flush=True)
    model_s2, _, stage2_bridge_delta = load_bridge_model(
        cfg, Path(args.base_checkpoint), device,
        stage2_ckpt=Path(args.stage2_checkpoint), label="stage2",
    )
    for rec in records:
        idx = rec["dataset_index"]
        sample = dataset[idx]
        sample_copy = {k: v for k, v in sample.items()}
        try:
            s2_out = train_path_eval_one(
                model_s2, sample_copy, device,
                n_traj_query=n_traj_query,
                num_sample_trajs=num_sample_trajs,
                action_scale=action_scale,
                traj_image_size=traj_size,
            )
        except Exception as exc:
            LOGGER.exception("stage2 failed idx=%s: %s", idx, exc)
            continue
        rec["stage2_path_len"] = s2_out["path_len"]
        rec["stage2_forward_count"] = s2_out["forward_count"]
        rec["stage2_trajectory_loss"] = s2_out["trajectory_loss"]
        rec["stage2_trajectory_summary"] = s2_out["trajectory_summary"]
        rec["path_len_delta_stage2_minus_base"] = (
            None if rec.get("base_only_path_len") is None or s2_out["path_len"] is None
            else s2_out["path_len"] - rec["base_only_path_len"]
        )
        rec["loss_delta_stage2_minus_base"] = (
            None if rec.get("base_only_trajectory_loss") is None
            else s2_out["trajectory_loss"] - rec["base_only_trajectory_loss"]
        )

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    n = len(records)
    if n == 0:
        print("No records.", flush=True)
        return 1

    base_sum = _summarize_branch(records, "base_only")
    s2_sum = _summarize_branch(records, "stage2")
    gt_lens = [r["gt_path_len"] for r in records if r.get("gt_path_len") is not None]

    s2_better_path = sum(
        1 for r in records
        if r.get("base_only_path_len") is not None and r.get("stage2_path_len") is not None
        and r["stage2_path_len"] > r["base_only_path_len"] + 0.05
    )
    s2_worse_path = sum(
        1 for r in records
        if r.get("base_only_path_len") is not None and r.get("stage2_path_len") is not None
        and r["stage2_path_len"] < r["base_only_path_len"] - 0.05
    )
    s2_better_loss = sum(
        1 for r in records
        if r.get("base_only_trajectory_loss") is not None and r.get("stage2_trajectory_loss") is not None
        and r["stage2_trajectory_loss"] < r["base_only_trajectory_loss"] - 1e-4
    )

    print("\n===== Base-only vs Stage2 (train path B, gold pixel_goal) =====", flush=True)
    print(f"samples: {n} (skipped while picking: {skipped})", flush=True)
    if gt_lens:
        print(f"GT path_len:          mean={np.mean(gt_lens):.3f} median={np.median(gt_lens):.3f}", flush=True)
    print(
        f"base-only path_len:   mean={base_sum['path_len_mean']:.3f} "
        f"median={base_sum['path_len_median']:.3f} short<0.5m={base_sum['path_short_pct']:.1f}%",
        flush=True,
    )
    print(
        f"stage2 path_len:      mean={s2_sum['path_len_mean']:.3f} "
        f"median={s2_sum['path_len_median']:.3f} short<0.5m={s2_sum['path_short_pct']:.1f}%",
        flush=True,
    )
    print(
        f"base-only traj_loss:  mean={base_sum['loss_mean']:.4f} median={base_sum['loss_median']:.4f}",
        flush=True,
    )
    print(
        f"stage2 traj_loss:     mean={s2_sum['loss_mean']:.4f} median={s2_sum['loss_median']:.4f}",
        flush=True,
    )
    print(
        f"forward>0:            base-only={base_sum['forward_any_pct']:.1f}% "
        f"stage2={s2_sum['forward_any_pct']:.1f}%",
        flush=True,
    )
    print(f"stage2 path_len > base+0.05m: {s2_better_path}/{n}", flush=True)
    print(f"stage2 path_len < base-0.05m: {s2_worse_path}/{n}", flush=True)
    print(f"stage2 lower loss:            {s2_better_loss}/{n}", flush=True)
    print(f"stage2 bridge param delta:     {stage2_bridge_delta}", flush=True)
    print(f"jsonl: {out_path}", flush=True)

    med_gt = float(np.median(gt_lens)) if gt_lens else 0.0
    med_b = base_sum["path_len_median"]
    med_s2 = s2_sum["path_len_median"]
    if med_s2 > med_b + 0.1 and med_s2 < med_gt * 0.5:
        print(
            "\nVERDICT: Stage2 improves over base-only but trajectories still short vs GT "
            "→ bridge learned partially; consider capacity / training recipe.",
            flush=True,
        )
    elif abs(med_s2 - med_b) < 0.08:
        print(
            "\nVERDICT: Stage2 ≈ base-only → bridge checkpoint likely did not change behavior; "
            "inspect training loss curve and whether 5 bridge keys actually updated.",
            flush=True,
        )
    elif med_s2 < med_b - 0.08:
        print(
            "\nVERDICT: Stage2 worse than base-only → training direction / labels / "
            "traj_images / action_scale misalignment suspected.",
            flush=True,
        )
    elif med_s2 > med_b + 0.1 and med_s2 >= med_gt * 0.5:
        print(
            "\nVERDICT: Stage2 clearly better and approaching GT scale on this slice.",
            flush=True,
        )
    else:
        print("\nVERDICT: Mixed — see per-sample jsonl.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

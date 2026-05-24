#!/usr/bin/env python3
"""
Base-only vs Stage2 bridge comparison on gold pixel_goal (train path B latents).

Two rollout metrics (apples-to-apples aligned):
  - initial_current: traj_images=[ti0, ti0], GT path_len from trajectory[0]
  - train_full_loss: compute_loss(B,N,...) with training expand logic

RNG is fixed per (sample_index, branch, metric) before stochastic ops.
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
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from scripts.evaluation.latent_parity_train_vs_eval import _build_train_batch
from scripts.evaluation.r2r_val_unseen import (
    TRAJECTORY_SELECTION_CHOICES,
    _extract_checkpoint_state_dict,
    _finalize_local_actions,
    _load_compatible_state_dict,
    _resolve_internnav_model_path,
    _trajectory_debug_summary,
    _verify_internnav_system1_loaded,
    reconstruct_xy_from_delta,
    select_trajectory_xy,
    traj_to_actions,
    trajectory_xy_path_len,
)
from scripts.training.model_builder import build_model
from scripts.training.utils import load_config

from src.data.factory import build_trajectory_dataset

LOGGER = logging.getLogger("bridge_ab")
PATH_LEN_RE = re.compile(r"path_len=([0-9.]+)")


def _parse_path_len(summary: str) -> float | None:
    m = PATH_LEN_RE.search(summary)
    return float(m.group(1)) if m else None


def _set_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cast_traj_images(traj_images: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return traj_images.to(device=device, dtype=dtype)


def _path_len_from_delta_np(traj: np.ndarray, action_scale: float) -> float:
    """Relative delta trajectory (T, 3) → path length in meters."""
    if traj.ndim != 2 or traj.shape[0] == 0:
        return 0.0
    deltas = traj[:, :2].astype(np.float64) / float(action_scale)
    cumsum_xy = np.cumsum(deltas, axis=0)
    xy = np.concatenate([np.zeros((1, 2), dtype=np.float64), cumsum_xy], axis=0)
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def _build_initial_current_images(sample: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ti = sample.get("traj_images")
    if ti is None:
        raise RuntimeError("Sample missing traj_images")
    if not torch.is_tensor(ti):
        ti = torch.as_tensor(ti)
    anchor = ti[0]
    pair = torch.stack([anchor, anchor], dim=0)
    return _cast_traj_images(pair.unsqueeze(0), device, dtype)


def _build_train_full_tensors(
    sample: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    traj = sample["trajectory"]
    if not torch.is_tensor(traj):
        traj = torch.as_tensor(traj)
    if traj.dim() == 2:
        traj = traj.unsqueeze(0)
    if traj.dim() != 3:
        raise RuntimeError(f"Expected trajectory (N,T,3), got {tuple(traj.shape)}")

    ti = sample.get("traj_images")
    if ti is None:
        raise RuntimeError("Sample missing traj_images")
    if not torch.is_tensor(ti):
        ti = torch.as_tensor(ti)
    if ti.dim() != 4:
        raise RuntimeError(f"Expected traj_images (N,H,W,3), got {tuple(ti.shape)}")

    tv = sample.get("trajectory_valid")
    if tv is None:
        traj_valid = None
    else:
        if not torch.is_tensor(tv):
            tv = torch.as_tensor(tv)
        traj_valid = tv.unsqueeze(0) if tv.dim() == 1 else tv
        if traj_valid.dim() == 1:
            traj_valid = traj_valid.unsqueeze(0)

    return (
        _cast_traj_images(ti.unsqueeze(0), device, dtype),
        traj.unsqueeze(0).to(device=device, dtype=dtype),
        traj_valid.to(device) if traj_valid is not None else None,
    )


def _continuous_trajectory_stats(
    trajectory: torch.Tensor,
    num_sample_trajs: int,
    action_scale: float,
) -> dict[str, Any]:
    """Stats over num_sample_trajs parallel flow-matching samples."""
    n = min(int(num_sample_trajs), int(trajectory.shape[0]))
    trajs = trajectory[:n].float().detach().cpu().numpy()
    trajs[:, :, :2] /= float(action_scale)
    all_trajectory = reconstruct_xy_from_delta(trajs)
    per_sample_path_lens = [trajectory_xy_path_len(all_trajectory[i]) for i in range(n)]

    endpoints = []
    delta_norms = []
    for i in range(n):
        deltas = np.diff(all_trajectory[i, :, :2], axis=0)
        endpoints.append(all_trajectory[i, -1, :2] if len(all_trajectory[i]) else np.zeros(2))
        delta_norms.append(float(np.linalg.norm(deltas)))

    endpoints_arr = np.stack(endpoints, axis=0) if endpoints else np.zeros((0, 2))
    endpoint_std = float(np.std(endpoints_arr, axis=0).mean()) if len(endpoints_arr) else 0.0

    mean_summary = _trajectory_debug_summary(trajectory, n, action_scale)
    mean_path_len = _parse_path_len(mean_summary)

    per_sample_forward = []
    for i in range(n):
        _set_rng(10_000 + i)
        acts = _finalize_local_actions(
            traj_to_actions(
                trajectory[i : i + 1],
                num_sample_trajs=1,
                action_scale=action_scale,
                trajectory_selection="mean",
            )
        )
        per_sample_forward.append(sum(1 for a in acts if a == 1))

    selection_results: dict[str, Any] = {}
    for selection in TRAJECTORY_SELECTION_CHOICES:
        _set_rng(20_000)
        selected_xy, selected_idx = select_trajectory_xy(all_trajectory, selection)
        actions = _finalize_local_actions(
            traj_to_actions(
                trajectory[:n],
                num_sample_trajs=n,
                action_scale=action_scale,
                trajectory_selection=selection,
            )
        )
        selection_results[selection] = {
            "selected_index": selected_idx,
            "path_len": trajectory_xy_path_len(selected_xy),
            "actions_first4": actions[:4],
            "forward_count": sum(1 for a in actions if a == 1),
            "zero_pad": actions[:4] == [0, 0, 0, 0],
        }
    mean_result = selection_results["mean"]

    return {
        "trajectory_summary_mean": mean_summary,
        "path_len_mean": mean_path_len,
        "per_sample_path_lens": per_sample_path_lens,
        "per_sample_path_len_median": float(np.median(per_sample_path_lens)) if per_sample_path_lens else None,
        "per_sample_path_len_std": float(np.std(per_sample_path_lens)) if per_sample_path_lens else None,
        "endpoint_std_xy_scalar": endpoint_std,
        "delta_xy_norm_mean": float(np.mean(delta_norms)) if delta_norms else None,
        "delta_xy_norm_std": float(np.std(delta_norms)) if delta_norms else None,
        "per_sample_forward_in_decode": per_sample_forward,
        "forward_any_per_sample_pct": 100.0 * sum(1 for f in per_sample_forward if f > 0) / max(n, 1),
        "selection_results": selection_results,
        "actions_from_mean_traj_first4": mean_result["actions_first4"],
        "forward_from_mean_traj": mean_result["forward_count"],
        "zero_pad_mean_traj": mean_result["zero_pad"],
    }


def _extract_traj_hidden(model, sample: dict, device: torch.device, n_traj_query: int) -> torch.Tensor:
    processor = model.qwen2_5_vl.processor
    pano_inputs, pano_num_histories = _build_train_batch(sample, processor, n_traj_query)
    pano_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in pano_inputs.items()}
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
        raise RuntimeError("No traj_hidden_states from train path B")
    return traj_hs


def _eval_branch(
    model,
    traj_hs: torch.Tensor,
    sample: dict,
    device: torch.device,
    *,
    sample_index: int,
    branch: str,
    num_sample_trajs: int,
    action_scale: float,
) -> dict[str, Any]:
    dtype = model.config.dtype
    out: dict[str, Any] = {}

    # --- initial_current: [ti0, ti0] + GT traj[0] ---
    traj_images_ic = _build_initial_current_images(sample, device, dtype)
    gt_ic = sample["trajectory"]
    if not torch.is_tensor(gt_ic):
        gt_ic = torch.as_tensor(gt_ic)
    if gt_ic.dim() == 3:
        gt_ic_t = gt_ic[0]
    else:
        gt_ic_t = gt_ic
    out["gt_initial_current_path_len"] = _path_len_from_delta_np(
        gt_ic_t.detach().float().cpu().numpy(), action_scale,
    )

    seed_rollout = sample_index * 10_000 + (0 if branch == "base_only" else 1) * 1000
    _set_rng(seed_rollout)
    with torch.no_grad():
        traj_rollout = model.nextdit_action_head.get_trajectory(traj_hs, traj_images=traj_images_ic)
    ic_stats = _continuous_trajectory_stats(traj_rollout, num_sample_trajs, action_scale)
    out["initial_current"] = ic_stats

    seed_loss_ic = sample_index * 10_000 + (0 if branch == "base_only" else 1) * 1000 + 100
    _set_rng(seed_loss_ic)
    gt_loss = gt_ic_t.unsqueeze(0).to(device=device, dtype=dtype)
    with torch.cuda.amp.autocast(dtype=dtype, enabled=(device.type == "cuda")):
        loss_ic = model.nextdit_action_head.compute_loss(
            traj_hs, gt_loss, traj_images=traj_images_ic, trajectory_valid=None,
        )
    out["initial_current"]["trajectory_loss_single_frame"] = float(loss_ic["loss"].detach().cpu().item())

    # --- train_full_loss: training (B,N,...) expand ---
    traj_images_full, gt_full, tv_full = _build_train_full_tensors(sample, device, dtype)
    gt_frame_path_lens = [
        _path_len_from_delta_np(gt_full[0, i].detach().float().cpu().numpy(), action_scale)
        for i in range(gt_full.shape[1])
    ]
    out["gt_train_per_frame_path_lens"] = gt_frame_path_lens
    out["gt_train_per_frame_path_len_mean"] = float(np.mean(gt_frame_path_lens)) if gt_frame_path_lens else None

    seed_loss_full = sample_index * 10_000 + (0 if branch == "base_only" else 1) * 1000 + 200
    _set_rng(seed_loss_full)
    with torch.cuda.amp.autocast(dtype=dtype, enabled=(device.type == "cuda")):
        loss_full = model.nextdit_action_head.compute_loss(
            traj_hs, gt_full, traj_images=traj_images_full, trajectory_valid=tv_full,
        )
    out["train_full_loss"] = float(loss_full["loss"].detach().cpu().item())

    return out


def _snapshot_bridge_params(model) -> dict[str, torch.Tensor]:
    snap = {"latent_queries": model.latent_queries.detach().cpu().clone()}
    for name, param in model.nextdit_action_head.named_parameters():
        if name.startswith("cond_projector."):
            snap[f"nextdit_action_head.{name}"] = param.detach().cpu().clone()
    return snap


def _bridge_delta(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> dict[str, float]:
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
    stage2_ckpt: Path | None,
    label: str,
):
    internnav_path = _resolve_internnav_model_path(cfg)
    if internnav_path:
        print(f"[{label}] InternNav: {internnav_path}", flush=True)

    model = build_model(cfg, device=str(device), verbose=False).to(device)
    _verify_internnav_system1_loaded(model, internnav_path)
    model.qwen2_5_vl._load_model()
    if model.qwen2_5_vl.processor is None:
        raise RuntimeError("processor is None")
    if model.nextdit_action_head is None or model.latent_queries is None:
        raise RuntimeError("nextdit_action_head / latent_queries disabled")

    base_sd = _extract_checkpoint_state_dict(str(base_ckpt))
    _load_compatible_state_dict(model, base_sd, str(base_ckpt), label=f"[{label}] Base")
    bridge_after_base = _snapshot_bridge_params(model)

    stage2_delta = {}
    if stage2_ckpt is not None:
        stage2_sd = _extract_checkpoint_state_dict(str(stage2_ckpt))
        _load_compatible_state_dict(model, stage2_sd, str(stage2_ckpt), label=f"[{label}] Stage2")
        stage2_delta = _bridge_delta(bridge_after_base, _snapshot_bridge_params(model))
        print(f"[{label}] bridge delta: {stage2_delta}", flush=True)
    else:
        print(f"[{label}] base-only (no stage2 ckpt)", flush=True)

    model.eval()
    return model, stage2_delta


def _agg_metric(records: list[dict], *keys: str) -> float | None:
    vals = []
    for r in records:
        cur = r
        for k in keys:
            if cur is None:
                cur = None
                break
            cur = cur.get(k) if isinstance(cur, dict) else None
        if cur is not None and isinstance(cur, (int, float)):
            vals.append(float(cur))
    return float(np.mean(vals)) if vals else None


def parse_args():
    p = argparse.ArgumentParser(description="Base-only vs Stage2 bridge (aligned metrics)")
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


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    _set_rng(args.seed)

    cfg = load_config(args.config)
    cfg["data"]["root"] = args.root
    if "paths" in cfg:
        cfg["paths"]["dataset_root"] = args.root

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    traj_cfg = cfg.get("model", {}).get("action_head", {}).get("nextdit", {})
    num_sample_trajs = int(traj_cfg.get("num_sample_trajs", 32))
    action_scale = float(cfg.get("data", {}).get("action_scale", 4.0))
    n_traj_query = int(traj_cfg.get("n_query", 4))

    dataset = build_trajectory_dataset(cfg, split=args.split)
    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    chosen: list[tuple[int, dict]] = []
    skipped = 0
    for idx in indices:
        if len(chosen) >= args.num_samples:
            break
        sample = dataset[idx]
        if sample.get("pixel_goal") is None or float(sample.get("is_stop", 0.0)) > 0.5:
            skipped += 1
            continue
        if sample.get("trajectory") is None or sample.get("traj_images") is None:
            skipped += 1
            continue
        chosen.append((idx, sample))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    print("\n=== base-only ===", flush=True)
    model_base, _ = load_bridge_model(
        cfg, Path(args.base_checkpoint), device, stage2_ckpt=None, label="base-only",
    )
    for idx, sample in chosen:
        try:
            hs = _extract_traj_hidden(model_base, {k: v for k, v in sample.items()}, device, n_traj_query)
            base_metrics = _eval_branch(
                model_base, hs, sample, device,
                sample_index=idx, branch="base_only",
                num_sample_trajs=num_sample_trajs, action_scale=action_scale,
            )
            records.append({
                "dataset_index": idx,
                "pixel_goal": [int(sample["pixel_goal"][0]), int(sample["pixel_goal"][1])],
                "base_only": base_metrics,
            })
        except Exception as exc:
            LOGGER.exception("base-only idx=%s: %s", idx, exc)
    del model_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n=== stage2 ===", flush=True)
    model_s2, stage2_delta = load_bridge_model(
        cfg, Path(args.base_checkpoint), device,
        stage2_ckpt=Path(args.stage2_checkpoint), label="stage2",
    )
    for rec in records:
        idx = rec["dataset_index"]
        sample = dataset[idx]
        try:
            hs = _extract_traj_hidden(model_s2, {k: v for k, v in sample.items()}, device, n_traj_query)
            rec["stage2"] = _eval_branch(
                model_s2, hs, sample, device,
                sample_index=idx, branch="stage2",
                num_sample_trajs=num_sample_trajs, action_scale=action_scale,
            )
        except Exception as exc:
            LOGGER.exception("stage2 idx=%s: %s", idx, exc)
    for rec in records:
        rec["stage2_bridge_delta"] = stage2_delta

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    n = len(records)
    if n == 0:
        print("No records.", flush=True)
        return 1

    def ic_path(rec, branch):
        return rec.get(branch, {}).get("initial_current", {}).get("path_len_mean")

    def ic_forward(rec, branch):
        return rec.get(branch, {}).get("initial_current", {}).get("forward_from_mean_traj", 0)

    def full_loss(rec, branch):
        return rec.get(branch, {}).get("train_full_loss")

    gt_ic = [r["base_only"]["gt_initial_current_path_len"] for r in records if "base_only" in r]
    gt_train = [r["base_only"]["gt_train_per_frame_path_len_mean"] for r in records if "base_only" in r]

    print("\n===== Base-only vs Stage2 (aligned) =====", flush=True)
    print(f"samples: {n} (skipped: {skipped})", flush=True)
    print(f"GT initial_current path_len:  mean={np.mean(gt_ic):.3f}", flush=True)
    print(f"GT train frames path_len:   mean={np.mean(gt_train):.3f}", flush=True)

    for label in ("base_only", "stage2"):
        ic_pl = [ic_path(r, label) for r in records if ic_path(r, label) is not None]
        ic_ps_std = [
            r[label]["initial_current"]["per_sample_path_len_std"]
            for r in records if label in r
        ]
        ic_ep = [
            r[label]["initial_current"]["endpoint_std_xy_scalar"]
            for r in records if label in r
        ]
        ic_fwd = [ic_forward(r, label) for r in records]
        fl = [full_loss(r, label) for r in records if full_loss(r, label) is not None]
        print(f"\n[{label}]", flush=True)
        if ic_pl:
            print(f"  initial_current path_len_mean: {np.mean(ic_pl):.3f} median={np.median(ic_pl):.3f}", flush=True)
            print(f"  per_sample path_len std(avg): {np.mean(ic_ps_std):.3f}", flush=True)
            print(f"  endpoint_std_xy (avg):       {np.mean(ic_ep):.3f}", flush=True)
            print(f"  forward_from_mean_traj>0:  {sum(1 for f in ic_fwd if f > 0)}/{n}", flush=True)
            for selection in TRAJECTORY_SELECTION_CHOICES:
                sel = [
                    r[label]["initial_current"]["selection_results"][selection]
                    for r in records
                    if label in r and selection in r[label]["initial_current"].get("selection_results", {})
                ]
                if not sel:
                    continue
                sel_fwd = [item["forward_count"] for item in sel]
                sel_path = [item["path_len"] for item in sel]
                print(
                    f"  selection[{selection}]: "
                    f"forward>0={sum(1 for f in sel_fwd if f > 0)}/{len(sel)} "
                    f"path_len_mean={np.mean(sel_path):.3f}",
                    flush=True,
                )
        if fl:
            print(f"  train_full_loss:             mean={np.mean(fl):.4f} median={np.median(fl):.4f}", flush=True)

    b_ic = [ic_path(r, "base_only") for r in records if ic_path(r, "base_only") is not None]
    s_ic = [ic_path(r, "stage2") for r in records if ic_path(r, "stage2") is not None]
    b_fl = [full_loss(r, "base_only") for r in records if full_loss(r, "base_only") is not None]
    s_fl = [full_loss(r, "stage2") for r in records if full_loss(r, "stage2") is not None]

    print(f"\nstage2 bridge delta: {stage2_delta}", flush=True)
    print(f"jsonl: {out_path}", flush=True)

    if b_ic and s_ic and np.mean(s_ic) < np.mean(b_ic) - 0.05 and sum(ic_forward(r, "stage2") for r in records) == 0:
        print(
            "\nVERDICT: initial_current — stage2 shorter / no FORWARD vs base-only "
            "→ bridge training may bias rollout conservative (after aligned test).",
            flush=True,
        )
    elif b_fl and s_fl and np.mean(s_fl) < np.mean(b_fl) - 0.01 and np.mean(s_ic or [0]) < np.mean(gt_ic) * 0.5:
        print(
            "\nVERDICT: train_full_loss lower but initial_current still short vs GT "
            "→ optimizes training objective, not rollout distance.",
            flush=True,
        )
    else:
        print("\nVERDICT: see jsonl per-sample initial_current + train_full_loss.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
End-to-end sanity check for a trained pano-to-InternNav latent adapter.

For each teacher sidecar record this script:

  1. Runs the (frozen) panoramic student VLM to extract ``traj_hidden_states``.
  2. Maps them through the trained adapter to get adapter-projected latents.
  3. Loads the saved InternNav teacher ``traj_latents`` + ``dp_actions``.
  4. Runs the frozen InternNav System1 (``model.generate_traj``) on the
     adapter-projected latents to get *adapter dp_actions*.
  5. Compares adapter vs teacher in two spaces:
        - latent space: cosine / mse / norm_ratio
        - trajectory space: per-step L2, path length, endpoint distance,
          mean-trajectory cosine, discrete-action overlap.

This answers the real question Stage2 cares about: does the adapter project
panoramic latents back onto the manifold that the frozen System1 actually
understands, or does the adapter merely fit the training-set teacher latents
in isolation?
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
import torch.nn.functional as F

from scripts.evaluation.collect_internnav_teacher_sidecar import (
    _build_images_dp,
    _load_teacher,
    _torch_dtype,
)
from scripts.training.train_pano_latent_adapter import (
    PanoToInternNavLatentAdapter,
    _copy_sample_for_collator,
    _extract_student_latents,
    _load_student_model,
    _load_teacher_records,
    _prepare_config,
    _sample_from_record,
)
from src.data.factory import build_trajectory_dataset

LOGGER = logging.getLogger("eval_pano_latent_adapter")


def _infer_n_layers_from_state(state_dict: dict[str, torch.Tensor]) -> int:
    """Each MLP hidden block contributes one ``mlp.<idx>.weight`` Linear.

    Layout per layer: ``Linear, GELU, Dropout`` then a final ``Linear``,
    i.e. the number of Linears is ``n_layers + 1``.
    """
    linear_indices = sorted(
        int(name.split(".")[1])
        for name in state_dict
        if name.startswith("mlp.") and name.endswith(".weight") and name.count(".") == 2
    )
    n_linear = len(linear_indices)
    return max(n_linear - 1, 1)


def _load_adapter_from_checkpoint(
    path: Path,
    *,
    dim: int,
    fallback_args: argparse.Namespace,
    device: torch.device,
) -> tuple[PanoToInternNavLatentAdapter, dict[str, Any]]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("adapter_state_dict")
    if state_dict is None:
        raise KeyError(f"{path} has no adapter_state_dict")
    saved_args = ckpt.get("args", {}) or {}

    def _get(key: str, default: Any) -> Any:
        if key in saved_args:
            return saved_args[key]
        return getattr(fallback_args, key, default)

    has_output_affine = "out_scale" in state_dict
    has_new_flags = any(k in saved_args for k in ("pre_norm", "output_affine", "adapter_n_layers"))

    inferred_n_layers = _infer_n_layers_from_state(state_dict)
    if "adapter_n_layers" in saved_args:
        n_layers = int(saved_args["adapter_n_layers"])
    else:
        n_layers = inferred_n_layers

    if "pre_norm" in saved_args:
        pre_norm = bool(saved_args["pre_norm"])
    elif has_new_flags:
        # New-style checkpoint that simply omitted pre_norm: trust current CLI default.
        pre_norm = bool(getattr(fallback_args, "pre_norm", False))
    else:
        # Legacy checkpoint: the old adapter always applied the leading LayerNorm.
        pre_norm = True

    adapter = PanoToInternNavLatentAdapter(
        dim=dim,
        hidden_dim=int(_get("adapter_hidden_dim", 2048)),
        dropout=float(_get("adapter_dropout", 0.0)),
        residual=bool(_get("residual", False)),
        zero_init=False,  # weights are loaded right away
        pre_norm=pre_norm,
        n_layers=n_layers,
        output_affine=has_output_affine,
    )
    missing, unexpected = adapter.load_state_dict(state_dict, strict=False)
    if missing:
        LOGGER.warning("Adapter state_dict missing keys: %s", missing)
    if unexpected:
        LOGGER.warning("Adapter state_dict unexpected keys: %s", unexpected)
    adapter.eval()
    adapter.to(device)
    LOGGER.info(
        "Loaded adapter from %s residual=%s pre_norm=%s output_affine=%s n_layers=%d hidden_dim=%d",
        path,
        bool(_get("residual", False)),
        pre_norm,
        has_output_affine,
        n_layers,
        int(_get("adapter_hidden_dim", 2048)),
    )
    return adapter, saved_args


def _restrict_records_by_split(
    records: list[dict[str, Any]],
    split_file: Path | None,
    split_key: str,
) -> list[dict[str, Any]]:
    if split_file is None or not split_file.exists():
        return records
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    wanted = set(int(idx) for idx in payload.get(split_key, []))
    if not wanted:
        return records
    return [rec for rec in records if int(rec["dataset_index"]) in wanted]


def _mean_trajectory(dp_actions: torch.Tensor) -> torch.Tensor:
    """``dp_actions`` is ``[num_sample, steps, action_dim]``; mean over samples."""
    return dp_actions.float().mean(dim=0)


def _cumulative_xy(mean_traj: torch.Tensor, action_scale: float) -> torch.Tensor:
    delta_xy = mean_traj[:, :2].detach().cpu() / float(action_scale)
    xy = torch.cat(
        [torch.zeros((1, 2), dtype=delta_xy.dtype), torch.cumsum(delta_xy, dim=0)],
        dim=0,
    )
    return xy


def _trajectory_metrics(
    adapter_dp: torch.Tensor,
    teacher_dp: torch.Tensor,
    *,
    traj_to_actions_fn: Any,
    action_scale: float,
) -> dict[str, float]:
    adapter_mean = _mean_trajectory(adapter_dp)
    teacher_mean = _mean_trajectory(teacher_dp)

    step_l2 = torch.linalg.norm(adapter_mean - teacher_mean, dim=-1).mean().item()
    cosine = F.cosine_similarity(
        adapter_mean.flatten().unsqueeze(0),
        teacher_mean.flatten().unsqueeze(0),
        dim=-1,
    ).item()

    adapter_xy = _cumulative_xy(adapter_mean, action_scale)
    teacher_xy = _cumulative_xy(teacher_mean, action_scale)
    adapter_path_len = torch.linalg.norm(torch.diff(adapter_xy, dim=0), dim=-1).sum().item()
    teacher_path_len = torch.linalg.norm(torch.diff(teacher_xy, dim=0), dim=-1).sum().item()
    endpoint_distance = torch.linalg.norm(adapter_xy[-1] - teacher_xy[-1]).item()

    adapter_actions = traj_to_actions_fn(adapter_dp.float().cpu().clone())
    teacher_actions = traj_to_actions_fn(teacher_dp.float().cpu().clone())
    overlap = _action_overlap(adapter_actions, teacher_actions)
    first_action_match = int(bool(
        adapter_actions and teacher_actions and adapter_actions[0] == teacher_actions[0]
    ))

    return {
        "traj_step_l2": float(step_l2),
        "traj_cosine": float(cosine),
        "adapter_path_len_m": float(adapter_path_len),
        "teacher_path_len_m": float(teacher_path_len),
        "path_len_diff_m": float(abs(adapter_path_len - teacher_path_len)),
        "endpoint_distance_m": float(endpoint_distance),
        "action_overlap_at_min_len": overlap,
        "first_action_match": first_action_match,
        "adapter_actions": adapter_actions,
        "teacher_actions": teacher_actions,
    }


def _action_overlap(a: list[int], b: list[int]) -> float:
    if not a or not b:
        return 0.0
    common = min(len(a), len(b))
    matches = sum(1 for i in range(common) if a[i] == b[i])
    return matches / common


def _latent_metrics_for_one(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.detach().float().cpu()
    target = target.detach().float().cpu()
    if pred.dim() == 3 and pred.shape[0] == 1:
        pred = pred.squeeze(0)
    if target.dim() == 3 and target.shape[0] == 1:
        target = target.squeeze(0)
    cos = F.cosine_similarity(pred.flatten().unsqueeze(0), target.flatten().unsqueeze(0), dim=-1).item()
    mse = F.mse_loss(pred, target).item()
    pred_norm = pred.norm(dim=-1).mean().item()
    target_norm = target.norm(dim=-1).mean().item()
    return {
        "latent_cosine": float(cos),
        "latent_mse": float(mse),
        "latent_pred_norm": float(pred_norm),
        "latent_target_norm": float(target_norm),
        "latent_norm_ratio": float(pred_norm / max(target_norm, 1.0e-6)),
    }


def _aggregate(records: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    if not records:
        return out
    for key in keys:
        values = [r[key] for r in records if isinstance(r.get(key), (int, float))]
        if not values:
            continue
        out[f"mean_{key}"] = float(np.mean(values))
        out[f"median_{key}"] = float(np.median(values))
        out[f"min_{key}"] = float(np.min(values))
        out[f"max_{key}"] = float(np.max(values))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end sanity check for pano-to-InternNav adapter")
    p.add_argument("--config", default="configs/train_config_internnav_8gpu_stage2_wider.yaml")
    p.add_argument("--root", required=True, help="Panoramic dataset root")
    p.add_argument("--split", default="train")
    p.add_argument("--teacher-jsonl", required=True)
    p.add_argument("--adapter-checkpoint", required=True)
    p.add_argument("--base-checkpoint", default="checkpoints/stage1-s2_latest.pth")
    p.add_argument("--internnav-repo", default=os.environ.get("INTERNNAV_REPO", "~/InternNav"))
    p.add_argument("--model-path", default=os.environ.get("INTERNNAV_MODEL_PATH", "/workspace/InternNav_Model"))
    p.add_argument("--internnav-model-path", default=os.environ.get("INTERNNAV_MODEL_PATH", ""))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--flash-attn-stub", dest="flash_attn_stub", action="store_true", default=True)
    p.add_argument("--no-flash-attn-stub", dest="flash_attn_stub", action="store_false")
    p.add_argument("--require-nextdit", dest="require_nextdit", action="store_true", default=True)
    p.add_argument("--no-require-nextdit", dest="require_nextdit", action="store_false")
    p.add_argument(
        "--index-mode",
        choices=["generic", "internnav_sft"],
        default="generic",
    )
    p.add_argument("--num-samples", type=int, default=0, help="0 = use all matched records")
    p.add_argument("--split-file", default="", help="Path to split.json produced by train_pano_latent_adapter")
    p.add_argument(
        "--split-key",
        choices=["val_indices", "train_indices"],
        default="val_indices",
    )
    p.add_argument("--traj-image-size", type=int, default=224)
    p.add_argument("--predict-steps", type=int, default=32)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--num-sample-trajs", type=int, default=32)
    p.add_argument("--action-scale", type=float, default=0.0, help="0 = read from config; otherwise override")
    p.add_argument("--output", default="", help="Optional JSONL output path (default: <adapter-dir>/e2e_sanity.jsonl)")
    p.add_argument("--max-prints", type=int, default=10)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = _torch_dtype(args.torch_dtype)
    cfg = _prepare_config(args)
    action_scale = float(args.action_scale) if args.action_scale > 0 else float(
        cfg.get("data", {}).get("trajectory", {}).get("action_scale", 4.0)
    )

    teacher_jsonl = Path(args.teacher_jsonl).expanduser()
    records = _load_teacher_records(teacher_jsonl)
    split_file = Path(args.split_file).expanduser() if args.split_file else None
    if split_file is not None:
        records = _restrict_records_by_split(records, split_file, args.split_key)
        LOGGER.info("Restricted records via split=%s key=%s -> %d", split_file, args.split_key, len(records))
    if args.num_samples > 0:
        records = records[: args.num_samples]
    if not records:
        raise RuntimeError("No usable teacher records after filtering")
    LOGGER.info("Evaluating on %d records", len(records))

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

    student_model = _load_student_model(cfg, args, device)
    processor = student_model.qwen2_5_vl.processor
    n_traj_query = int(cfg.get("model", {}).get("action_head", {}).get("nextdit", {}).get("n_query", 4))
    hidden_dim = int(cfg.get("model", {}).get("llm", {}).get("hidden_dim", 3584))

    adapter, _saved_args = _load_adapter_from_checkpoint(
        Path(args.adapter_checkpoint).expanduser(),
        dim=hidden_dim,
        fallback_args=args,
        device=device,
    )

    teacher_model, _teacher_processor, traj_to_actions_fn = _load_teacher(args, device)

    output_path = Path(args.output).expanduser() if args.output else (
        Path(args.adapter_checkpoint).expanduser().parent / "e2e_sanity.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing per-sample report to %s", output_path)

    summary_records: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as fout:
        for i, rec in enumerate(records):
            idx = int(rec["dataset_index"])
            sample = _sample_from_record(dataset, rec)
            coord_uv = rec["teacher"]["coord_uv"]
            sample_for_collator = _copy_sample_for_collator(sample, coord_uv)

            with torch.no_grad():
                student_latents = _extract_student_latents(
                    student_model,
                    processor,
                    [sample_for_collator],
                    device,
                    n_traj_query,
                )
                adapter_latents = adapter(student_latents)

            payload = torch.load(rec["_tensor_path"], map_location="cpu", weights_only=False)
            teacher_latents = payload["traj_latents"].to(device=device, dtype=adapter_latents.dtype)
            teacher_dp_saved = payload.get("dp_actions")
            if teacher_dp_saved is None:
                LOGGER.warning("Record idx=%s has no saved dp_actions; skipping", idx)
                continue

            traj_images = _build_images_dp(
                sample,
                device=device,
                dtype=dtype,
                image_size=(args.traj_image_size, args.traj_image_size),
            )

            adapter_latents_for_dit = adapter_latents.to(dtype=teacher_latents.dtype)
            if adapter_latents_for_dit.dim() == 2:
                adapter_latents_for_dit = adapter_latents_for_dit.unsqueeze(0)
            with torch.inference_mode():
                adapter_dp_actions = teacher_model.generate_traj(
                    adapter_latents_for_dit,
                    traj_images,
                    None,
                    predict_step_nums=args.predict_steps,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    num_sample_trajs=args.num_sample_trajs,
                )

            adapter_dp_cpu = adapter_dp_actions.detach().cpu()
            teacher_dp_cpu = teacher_dp_saved.detach().cpu().float()

            latent_metrics = _latent_metrics_for_one(adapter_latents, teacher_latents)
            traj_metrics = _trajectory_metrics(
                adapter_dp_cpu,
                teacher_dp_cpu,
                traj_to_actions_fn=traj_to_actions_fn,
                action_scale=action_scale,
            )

            report = {
                "dataset_index": idx,
                "clip_idx": rec.get("clip_idx"),
                "current_t": rec.get("current_t"),
                **latent_metrics,
                **traj_metrics,
            }
            fout.write(json.dumps(report, ensure_ascii=False) + "\n")
            summary_records.append(report)

            if i < args.max_prints:
                LOGGER.info(
                    "[%d/%d] idx=%d cos=%.3f norm_ratio=%.3f traj_cos=%.3f step_l2=%.4f "
                    "endpoint_d=%.3fm path_len adapter=%.3f teacher=%.3f overlap=%.2f first_match=%d",
                    i + 1,
                    len(records),
                    idx,
                    latent_metrics["latent_cosine"],
                    latent_metrics["latent_norm_ratio"],
                    traj_metrics["traj_cosine"],
                    traj_metrics["traj_step_l2"],
                    traj_metrics["endpoint_distance_m"],
                    traj_metrics["adapter_path_len_m"],
                    traj_metrics["teacher_path_len_m"],
                    traj_metrics["action_overlap_at_min_len"],
                    traj_metrics["first_action_match"],
                )

    summary = _aggregate(
        summary_records,
        keys=[
            "latent_cosine",
            "latent_norm_ratio",
            "latent_mse",
            "traj_cosine",
            "traj_step_l2",
            "endpoint_distance_m",
            "path_len_diff_m",
            "action_overlap_at_min_len",
            "first_action_match",
        ],
    )
    summary["num_records"] = len(summary_records)
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    LOGGER.info("Summary: %s", json.dumps(summary, ensure_ascii=False))
    LOGGER.info("Wrote summary to %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

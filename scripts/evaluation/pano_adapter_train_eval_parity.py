#!/usr/bin/env python3
"""
Train-path vs eval-oracle latent parity for PanoLatentSpaceAdapter.

For the same dataset sample and the same gold panoramic view/pixel answer:

  A. train_path:
     PanoramicTokenizedCollator(sft_mode=True) -> Qwen latent-query forward

  B. eval_oracle_path:
     online-style prompt -> Qwen generate -> replace generated answer suffix
     with gold panoramic answer -> generate_latents(...)

Both produce [B, 4, 3584] TRAJ-query hidden states.  This script compares the
raw query states, adapter outputs, cond_projector outputs, and NextDiT rollouts.
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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from scripts.evaluation.oracle_pano_adapter_bridge_test import (
    _build_dataset,
    _choose_indices,
    _condition_suffix_text,
    _generate_system2_text,
    _load_cfg_for_dataset,
    _parse_summary,
    _prepare_prompt_inputs,
    _traj_images_from_sample,
)
from scripts.evaluation.r2r_val_unseen import (
    _condition_output_ids_for_pixel_goal,
    _finalize_local_actions,
    _load_pano_latent_adapter,
    _trajectory_debug_summary,
    _trajectory_from_condition,
    load_model,
    traj_to_actions,
)
from scripts.training.train_pano_latent_adapter import _extract_student_latents
from src.models.heatmap.input_constructor import structured_condition_text

LOGGER = logging.getLogger("pano_adapter_parity")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pano adapter train/eval latent parity")
    p.add_argument("--config", default="configs/train_pano_adapter_stage2_8gpu.yaml")
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--adapter-checkpoint", required=True)
    p.add_argument("--root", default="/workspace/r2r_panoramic_audit_v2")
    p.add_argument("--split", default="train")
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument(
        "--trajectory-selection",
        default="mean",
        choices=(
            "mean",
            "endpoint_medoid",
            "path_medoid",
            "median_endpoint_nearest",
            "forward_or_medoid",
            "longest_forward",
        ),
    )
    p.add_argument("--output", default="logs/pano_adapter_train_eval_parity.jsonl")
    p.add_argument("--max-prints", type=int, default=8)
    return p.parse_args()


def _tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    af = a.detach().float().reshape(1, -1)
    bf = b.detach().float().reshape(1, -1)
    if af.shape != bf.shape:
        return {
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "cosine": None,
            "l2": None,
            "mean_abs": None,
            "max_abs": None,
            "norm_a": float(af.norm().item()),
            "norm_b": float(bf.norm().item()),
            "norm_ratio_a_over_b": None,
        }
    diff = af - bf
    norm_b = float(bf.norm().item())
    return {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "cosine": float(F.cosine_similarity(af, bf, dim=1).item()),
        "l2": float(diff.norm().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "norm_a": float(af.norm().item()),
        "norm_b": norm_b,
        "norm_ratio_a_over_b": float(af.norm().item() / (norm_b + 1.0e-8)),
    }


def _per_query_cosine(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    aq = a.detach().float().squeeze(0)
    bq = b.detach().float().squeeze(0)
    if aq.shape != bq.shape or aq.ndim != 2:
        return []
    return [
        float(F.cosine_similarity(aq[i : i + 1], bq[i : i + 1], dim=1).item())
        for i in range(aq.shape[0])
    ]


def _project_adapter_and_cond(model, adapter, traj_hs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    adapter_param = next(adapter.parameters(), None)
    adapter_dtype = adapter_param.dtype if adapter_param is not None else traj_hs.dtype
    adapted = adapter(traj_hs.to(dtype=adapter_dtype))
    cond_projector = model.nextdit_action_head.cond_projector
    proj_dtype = next(cond_projector.parameters()).dtype
    cond = cond_projector(adapted.to(dtype=proj_dtype))
    return adapted, cond


def _rollout_from_cond(
    model,
    cond: torch.Tensor,
    traj_images: torch.Tensor,
    *,
    num_sample_trajs: int,
    action_scale: float,
    trajectory_selection: str,
) -> dict[str, Any]:
    with torch.inference_mode():
        trajectory = _trajectory_from_condition(
            model.nextdit_action_head,
            cond,
            traj_images=traj_images,
        )
    actions = _finalize_local_actions(
        traj_to_actions(
            trajectory,
            num_sample_trajs=num_sample_trajs,
            action_scale=action_scale,
            trajectory_selection=trajectory_selection,
        )
    )
    summary = _trajectory_debug_summary(trajectory, num_sample_trajs, action_scale)
    parsed = _parse_summary(summary)
    return {
        "trajectory_summary": summary,
        **parsed,
        "actions_first4": [int(a) for a in actions[:4]],
        "forward_count_first4": sum(1 for a in actions[:4] if int(a) == 1),
        "forward_count": sum(1 for a in actions if int(a) == 1),
        "no_forward_first4": all(int(a) != 1 for a in actions[:4]),
        "turn_only_first4": all(int(a) in (2, 3) for a in actions[:4]),
    }


def _eval_oracle_latents(
    model,
    processor,
    sample: dict[str, Any],
    device: torch.device,
    *,
    internnav_protocol: bool,
    structured_pano_output: bool,
    max_new_tokens: int,
    gold_view: str,
    gold_pixel: list[int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    inputs = _prepare_prompt_inputs(
        processor,
        sample,
        device,
        internnav_protocol=internnav_protocol,
        structured_pano_output=structured_pano_output,
    )
    student_text, output_ids, prompt_len = _generate_system2_text(
        model,
        processor,
        inputs,
        max_new_tokens=max_new_tokens,
    )
    condition_output_ids = _condition_output_ids_for_pixel_goal(
        output_ids=output_ids,
        prompt_len=prompt_len,
        tokenizer=processor.tokenizer,
        pixel_goal=gold_pixel,
        llm_output=student_text,
        coord_order="generated",
        view_id=gold_view,
        structured_output=True,
    )
    lq = model.latent_queries.expand(1, -1, -1).to(
        device=device,
        dtype=model.config.dtype,
    )
    with torch.inference_mode():
        traj_hs = model.qwen2_5_vl.generate_latents(
            output_ids=condition_output_ids,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            latent_queries=lq,
            attention_mask=inputs.get("attention_mask"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )
    suffix = _condition_suffix_text(processor.tokenizer, condition_output_ids, prompt_len)
    return traj_hs, {
        "student_text": student_text,
        "condition_suffix_text": suffix,
        "prompt_unchanged": bool(torch.equal(condition_output_ids[:, :prompt_len], output_ids[:, :prompt_len])),
        "prompt_len": int(prompt_len),
        "condition_seq_len": int(condition_output_ids.shape[1]),
    }


def _train_path_latents(
    model,
    processor,
    sample: dict[str, Any],
    device: torch.device,
    *,
    n_traj_query: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    traj_hs, batch = _extract_student_latents(
        model,
        processor,
        [sample],
        device,
        n_traj_query,
        sft_protocol="direct",
        return_batch=True,
    )
    pano_inputs = batch.get("pano_inputs", {})
    return traj_hs, {
        "pano_input_ids_shape": (
            list(pano_inputs["input_ids"].shape)
            if isinstance(pano_inputs, dict) and "input_ids" in pano_inputs
            else None
        ),
        "pano_num_histories": (
            int(batch["pano_num_histories"][0])
            if "pano_num_histories" in batch and batch["pano_num_histories"]
            else None
        ),
    }


def _record_one(
    *,
    model,
    processor,
    adapter,
    sample: dict[str, Any],
    dataset_index: int,
    device: torch.device,
    action_scale: float,
    num_sample_trajs: int,
    n_traj_query: int,
    internnav_protocol: bool,
    structured_pano_output: bool,
    max_new_tokens: int,
    trajectory_selection: str,
) -> dict[str, Any]:
    gold_view = str(sample["pano_view_id"]).lower()
    gold_pixel = [int(sample["pano_pixel_goal"][0]), int(sample["pano_pixel_goal"][1])]
    oracle_text = structured_condition_text(gold_view, gold_pixel)

    train_hs, train_meta = _train_path_latents(
        model,
        processor,
        sample,
        device,
        n_traj_query=n_traj_query,
    )
    eval_hs, eval_meta = _eval_oracle_latents(
        model,
        processor,
        sample,
        device,
        internnav_protocol=internnav_protocol,
        structured_pano_output=structured_pano_output,
        max_new_tokens=max_new_tokens,
        gold_view=gold_view,
        gold_pixel=gold_pixel,
    )

    train_adapted, train_cond = _project_adapter_and_cond(model, adapter, train_hs)
    eval_adapted, eval_cond = _project_adapter_and_cond(model, adapter, eval_hs)

    traj_images = _traj_images_from_sample(sample, device, model.config.dtype)
    train_rollout = _rollout_from_cond(
        model,
        train_cond,
        traj_images,
        num_sample_trajs=num_sample_trajs,
        action_scale=action_scale,
        trajectory_selection=trajectory_selection,
    )
    eval_rollout = _rollout_from_cond(
        model,
        eval_cond,
        traj_images,
        num_sample_trajs=num_sample_trajs,
        action_scale=action_scale,
        trajectory_selection=trajectory_selection,
    )

    return {
        "dataset_index": int(dataset_index),
        "gold_view": gold_view,
        "gold_pixel": gold_pixel,
        "oracle_text": oracle_text,
        "train_meta": train_meta,
        "eval_meta": eval_meta,
        "raw_train_vs_eval": _tensor_metrics(train_hs, eval_hs),
        "raw_train_vs_eval_per_query_cosine": _per_query_cosine(train_hs, eval_hs),
        "adapted_train_vs_eval": _tensor_metrics(train_adapted, eval_adapted),
        "adapted_train_vs_eval_per_query_cosine": _per_query_cosine(train_adapted, eval_adapted),
        "cond_train_vs_eval": _tensor_metrics(train_cond, eval_cond),
        "cond_train_vs_eval_per_query_cosine": _per_query_cosine(train_cond, eval_cond),
        "train_rollout": train_rollout,
        "eval_rollout": eval_rollout,
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _summarize(records: list[dict[str, Any]], output: Path, args: argparse.Namespace) -> dict[str, Any]:
    raw_cos = [r["raw_train_vs_eval"]["cosine"] for r in records if r["raw_train_vs_eval"]["cosine"] is not None]
    adapted_cos = [
        r["adapted_train_vs_eval"]["cosine"]
        for r in records
        if r["adapted_train_vs_eval"]["cosine"] is not None
    ]
    cond_cos = [r["cond_train_vs_eval"]["cosine"] for r in records if r["cond_train_vs_eval"]["cosine"] is not None]
    train_path = [
        r["train_rollout"]["path_len"]
        for r in records
        if isinstance(r["train_rollout"].get("path_len"), (int, float))
    ]
    eval_path = [
        r["eval_rollout"]["path_len"]
        for r in records
        if isinstance(r["eval_rollout"].get("path_len"), (int, float))
    ]
    n = len(records)
    train_forward = sum(1 for r in records if r["train_rollout"].get("forward_count_first4", 0) > 0)
    eval_forward = sum(1 for r in records if r["eval_rollout"].get("forward_count_first4", 0) > 0)
    suffix_ok = sum(1 for r in records if r["eval_meta"].get("condition_suffix_text") == r["oracle_text"])
    prompt_ok = sum(1 for r in records if r["eval_meta"].get("prompt_unchanged"))

    return {
        "num_samples": n,
        "trajectory_selection": args.trajectory_selection,
        "output": str(output),
        "suffix_matches_oracle": suffix_ok,
        "prompt_unchanged": prompt_ok,
        "raw_cosine_mean": _mean(raw_cos),
        "raw_cosine_median": _median(raw_cos),
        "raw_cosine_min": float(min(raw_cos)) if raw_cos else None,
        "adapted_cosine_mean": _mean(adapted_cos),
        "cond_cosine_mean": _mean(cond_cos),
        "train_path_len_mean": _mean(train_path),
        "train_path_len_median": _median(train_path),
        "eval_path_len_mean": _mean(eval_path),
        "eval_path_len_median": _median(eval_path),
        "train_forward_first4": train_forward,
        "eval_forward_first4": eval_forward,
        "train_forward_first4_rate": float(train_forward / n) if n else None,
        "eval_forward_first4_rate": float(eval_forward / n) if n else None,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_args = argparse.Namespace(
        config=args.config,
        checkpoint=None,
        base_checkpoint=args.base_checkpoint,
    )
    model, model_cfg = load_model(model_args, device)
    if model.nextdit_action_head is None or model.latent_queries is None:
        raise RuntimeError("Config/model must enable NextDiT action_head and latent_queries")
    processor = model.qwen2_5_vl.processor
    if processor is None:
        raise RuntimeError("Qwen processor is None after load_model")

    hidden_dim = int(model_cfg.get("model", {}).get("llm", {}).get("hidden_dim", 3584))
    adapter = _load_pano_latent_adapter(args.adapter_checkpoint, hidden_dim, device)

    dataset_cfg = _load_cfg_for_dataset(args)
    dataset = _build_dataset(dataset_cfg, args.split)
    indices = _choose_indices(dataset, args.num_samples, args.seed)
    if not indices:
        raise RuntimeError("No valid pano pixel samples found")

    traj_cfg = dataset_cfg.get("data", {}).get("trajectory", {})
    action_scale = float(traj_cfg.get("action_scale", dataset_cfg.get("data", {}).get("action_scale", 4.0)))
    num_sample_trajs = int(
        model_cfg.get("model", {})
        .get("action_head", {})
        .get("nextdit", {})
        .get("num_sample_trajs", 32)
    )
    n_traj_query = int(
        model_cfg.get("model", {})
        .get("action_head", {})
        .get("nextdit", {})
        .get("n_query", 4)
    )
    protocol = str(traj_cfg.get("system2_sft_protocol", "direct")).lower()
    structured_output = bool(traj_cfg.get("structured_pano_output", True))
    internnav_protocol = protocol == "internnav"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as fout:
        for i, idx in enumerate(indices, start=1):
            sample = dataset[idx]
            rec = _record_one(
                model=model,
                processor=processor,
                adapter=adapter,
                sample=sample,
                dataset_index=idx,
                device=device,
                action_scale=action_scale,
                num_sample_trajs=num_sample_trajs,
                n_traj_query=n_traj_query,
                internnav_protocol=internnav_protocol,
                structured_pano_output=structured_output,
                max_new_tokens=args.max_new_tokens,
                trajectory_selection=args.trajectory_selection,
            )
            records.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i <= args.max_prints:
                LOGGER.info(
                    "[%d/%d idx=%s] raw_cos=%.4f cond_cos=%.4f "
                    "train_actions=%s train_path=%.3f eval_actions=%s eval_path=%.3f "
                    "suffix_ok=%s",
                    i,
                    len(indices),
                    idx,
                    rec["raw_train_vs_eval"]["cosine"],
                    rec["cond_train_vs_eval"]["cosine"],
                    rec["train_rollout"]["actions_first4"],
                    rec["train_rollout"].get("path_len") or -1.0,
                    rec["eval_rollout"]["actions_first4"],
                    rec["eval_rollout"].get("path_len") or -1.0,
                    rec["eval_meta"].get("condition_suffix_text") == rec["oracle_text"],
                )

    summary = _summarize(records, out_path, args)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Summary: %s", json.dumps(summary, ensure_ascii=False))
    LOGGER.info("Wrote %s and %s", out_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

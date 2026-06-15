#!/usr/bin/env python3
"""
Compare native InternNav teacher, train-path adapter, and eval-oracle adapter.

For the same dataset indices this script runs:

  1. Native InternNav teacher:
     front/history + lookdown dataset coord -> InternNav generate_latents
     -> InternNav generate_traj

  2. Adapter train path:
     PanoramicTokenizedCollator(sft_mode=True) gold pano answer
     -> Qwen TRAJ hidden states -> adapter -> frozen cond_projector -> NextDiT

  3. Adapter eval oracle path:
     online-style pano prompt -> generate text -> replace answer suffix with
     gold pano answer -> generate_latents -> adapter -> frozen cond_projector
     -> NextDiT

The purpose is to separate a trusted native InternNav/System1 baseline from
adapter-latent dialect issues.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import types
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

from scripts.evaluation.collect_internnav_teacher_sidecar import (
    _build_first_turn,
    _build_images_dp,
    _condition_on_dataset_coord,
    _find_cond_projector,
    _load_teacher,
    _normalize_image_grid_thw,
)
from scripts.evaluation.oracle_pano_adapter_bridge_test import (
    _build_dataset,
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
from scripts.training.utils import load_config
from src.data.factory import build_trajectory_dataset
from src.models.heatmap.input_constructor import structured_condition_text

LOGGER = logging.getLogger("pano_adapter_teacher_bridge_compare")
VALID_PANO_VIEWS = {"front", "right", "back", "left"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Native teacher vs pano adapter bridge comparison")
    p.add_argument("--config", default="configs/train_pano_adapter_stage2_8gpu.yaml")
    p.add_argument("--teacher-config", default="configs/train_config_internnav_8gpu_stage2_wider.yaml")
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--adapter-checkpoint", required=True)
    p.add_argument("--root", default="/workspace/r2r_panoramic_audit_v2")
    p.add_argument("--split", default="train")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--student-device", default="cuda:0")
    p.add_argument("--teacher-device", default="cuda:1")
    p.add_argument("--internnav-repo", default=os.environ.get("INTERNNAV_REPO", "/workspace/InternNav"))
    p.add_argument("--teacher-model-path", default=os.environ.get("INTERNNAV_MODEL_PATH", "/workspace/InternNav_Model"))
    p.add_argument("--teacher-torch-dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    p.add_argument("--teacher-attn-implementation", default="sdpa")
    p.add_argument("--teacher-flash-attn-stub", action="store_true", default=True)
    p.add_argument("--no-teacher-flash-attn-stub", dest="teacher_flash_attn_stub", action="store_false")
    p.add_argument("--teacher-front-width", type=int, default=0)
    p.add_argument("--teacher-front-height", type=int, default=0)
    p.add_argument("--teacher-traj-image-size", type=int, default=224)
    p.add_argument("--teacher-predict-steps", type=int, default=32)
    p.add_argument("--teacher-guidance-scale", type=float, default=1.0)
    p.add_argument("--teacher-num-inference-steps", type=int, default=10)
    p.add_argument("--teacher-num-sample-trajs", type=int, default=32)
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
    p.add_argument("--output", default="logs/pano_adapter_teacher_bridge_compare.jsonl")
    p.add_argument("--max-prints", type=int, default=8)
    return p.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _load_native_cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.teacher_config)
    cfg["data"]["root"] = args.root
    if "paths" in cfg:
        cfg["paths"]["dataset_root"] = args.root
    traj_cfg = cfg.setdefault("data", {}).setdefault("trajectory", {})
    traj_cfg["panoramic_vlm_input"] = False
    traj_cfg["load_lookdown_for_system2"] = True
    traj_cfg["load_traj_images"] = True
    traj_cfg["enable_trajectory_augmentation"] = False
    traj_cfg["require_sft_target"] = False
    return cfg


def _build_native_dataset(cfg: dict[str, Any], split: str):
    return build_trajectory_dataset(
        cfg,
        split=split,
        enable_augmentation=False,
        enable_trajectory_augmentation=False,
        load_history_heatmap=False,
        panoramic_vlm_input=False,
        compute_pixel_goal=False,
        compute_pano_view_pixel_goal=False,
        load_lookdown_for_system2=True,
        load_traj_images=True,
        require_sft_target=False,
    )


def _sample_index_pair(dataset: Any, idx: int) -> tuple[int, int] | None:
    if hasattr(dataset, "sample_index"):
        clip_idx, current_t = dataset.sample_index[idx]
        return int(clip_idx), int(current_t)
    return None


def _choose_index_pairs(
    pano_dataset: Any,
    native_dataset: Any,
    num_samples: int,
    seed: int,
) -> list[tuple[int, int]]:
    native_by_pair: dict[tuple[int, int], int] = {}
    for native_idx in range(len(native_dataset)):
        pair = _sample_index_pair(native_dataset, native_idx)
        if pair is not None and pair not in native_by_pair:
            native_by_pair[pair] = native_idx

    rng = random.Random(seed)
    indices = list(range(len(pano_dataset)))
    rng.shuffle(indices)
    chosen: list[tuple[int, int]] = []
    for pano_idx in indices:
        pano_pair = _sample_index_pair(pano_dataset, pano_idx)
        if pano_pair is None or pano_pair not in native_by_pair:
            continue
        native_idx = native_by_pair[pano_pair]
        pano_sample = pano_dataset[pano_idx]
        native_sample = native_dataset[native_idx]
        view_id = str(pano_sample.get("pano_view_id") or "").lower()
        if view_id not in VALID_PANO_VIEWS:
            continue
        if pano_sample.get("pano_pixel_goal") is None:
            continue
        if native_sample.get("pixel_goal") is None:
            continue
        if pano_sample.get("traj_images") is None or native_sample.get("traj_images") is None:
            continue
        chosen.append((pano_idx, native_idx))
        if len(chosen) >= num_samples:
            break
    return chosen


def _tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    af = a.detach().float().cpu().reshape(1, -1)
    bf = b.detach().float().cpu().reshape(1, -1)
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
    norm_a = float(af.norm().item())
    norm_b = float(bf.norm().item())
    return {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "cosine": float(F.cosine_similarity(af, bf, dim=1).item()),
        "l2": float(diff.norm().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "norm_a": norm_a,
        "norm_b": norm_b,
        "norm_ratio_a_over_b": float(norm_a / (norm_b + 1.0e-8)),
    }


def _per_query_cosine(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    aq = a.detach().float().cpu().squeeze(0)
    bq = b.detach().float().cpu().squeeze(0)
    if aq.shape != bq.shape or aq.ndim != 2:
        return []
    return [
        float(F.cosine_similarity(aq[i : i + 1], bq[i : i + 1], dim=1).item())
        for i in range(aq.shape[0])
    ]


def _pad_actions(actions: list[int], n: int = 8) -> list[int]:
    out = [int(a) for a in actions[:n]]
    if len(out) < n:
        out.extend([0] * (n - len(out)))
    return out


def _rollout_fields(
    dp_actions: torch.Tensor,
    *,
    num_sample_trajs: int,
    action_scale: float,
    trajectory_selection: str,
) -> dict[str, Any]:
    actions = _finalize_local_actions(
        traj_to_actions(
            dp_actions,
            num_sample_trajs=num_sample_trajs,
            action_scale=action_scale,
            trajectory_selection=trajectory_selection,
        )
    )
    summary = _trajectory_debug_summary(dp_actions, num_sample_trajs, action_scale)
    parsed = _parse_summary(summary)
    actions8 = _pad_actions([int(a) for a in actions], 8)
    return {
        "trajectory_summary": summary,
        **parsed,
        "actions8": actions8,
        "actions_first4": actions8[:4],
        "forward_count_first4": int(sum(1 for a in actions8[:4] if a == 1)),
        "forward_count8": int(sum(1 for a in actions8 if a == 1)),
        "no_forward_first4": bool(all(a != 1 for a in actions8[:4])),
        "turn_only_first4": bool(all(a in (2, 3) for a in actions8[:4])),
    }


def _adapter_project_and_rollout(
    student_model: Any,
    adapter: torch.nn.Module,
    traj_hs: torch.Tensor,
    traj_images: torch.Tensor,
    *,
    num_sample_trajs: int,
    action_scale: float,
    trajectory_selection: str,
) -> dict[str, Any]:
    adapter_param = next(adapter.parameters(), None)
    adapter_dtype = adapter_param.dtype if adapter_param is not None else traj_hs.dtype
    cond_projector = student_model.nextdit_action_head.cond_projector
    proj_dtype = next(cond_projector.parameters()).dtype
    with torch.inference_mode():
        adapted = adapter(traj_hs.to(dtype=adapter_dtype))
        cond = cond_projector(adapted.to(dtype=proj_dtype))
        dp_actions = _trajectory_from_condition(
            student_model.nextdit_action_head,
            cond,
            traj_images=traj_images,
        )
    return {
        "adapted": adapted,
        "cond": cond,
        "dp_actions": dp_actions,
        "rollout": _rollout_fields(
            dp_actions,
            num_sample_trajs=num_sample_trajs,
            action_scale=action_scale,
            trajectory_selection=trajectory_selection,
        ),
    }


def _student_rollout_from_teacher_raw(
    student_model: Any,
    teacher_traj_hs: torch.Tensor,
    traj_images: torch.Tensor,
    *,
    num_sample_trajs: int,
    action_scale: float,
    trajectory_selection: str,
) -> dict[str, Any]:
    cond_projector = student_model.nextdit_action_head.cond_projector
    proj_dtype = next(cond_projector.parameters()).dtype
    raw = teacher_traj_hs.to(device=traj_images.device, dtype=proj_dtype)
    with torch.inference_mode():
        cond = cond_projector(raw)
        dp_actions = _trajectory_from_condition(
            student_model.nextdit_action_head,
            cond,
            traj_images=traj_images,
        )
    return {
        "cond": cond,
        "dp_actions": dp_actions,
        "rollout": _rollout_fields(
            dp_actions,
            num_sample_trajs=num_sample_trajs,
            action_scale=action_scale,
            trajectory_selection=trajectory_selection,
        ),
    }


def _train_path_latents(
    student_model: Any,
    student_processor: Any,
    sample: dict[str, Any],
    device: torch.device,
    *,
    n_traj_query: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    traj_hs, batch = _extract_student_latents(
        student_model,
        student_processor,
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


def _eval_oracle_latents(
    student_model: Any,
    student_processor: Any,
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
        student_processor,
        sample,
        device,
        internnav_protocol=internnav_protocol,
        structured_pano_output=structured_pano_output,
    )
    student_text, output_ids, prompt_len = _generate_system2_text(
        student_model,
        student_processor,
        inputs,
        max_new_tokens=max_new_tokens,
    )
    condition_output_ids = _condition_output_ids_for_pixel_goal(
        output_ids=output_ids,
        prompt_len=prompt_len,
        tokenizer=student_processor.tokenizer,
        pixel_goal=gold_pixel,
        llm_output=student_text,
        coord_order="generated",
        view_id=gold_view,
        structured_output=True,
    )
    lq = student_model.latent_queries.expand(1, -1, -1).to(
        device=device,
        dtype=student_model.config.dtype,
    )
    with torch.inference_mode():
        traj_hs = student_model.qwen2_5_vl.generate_latents(
            output_ids=condition_output_ids,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            latent_queries=lq,
            attention_mask=inputs.get("attention_mask"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )
    suffix = _condition_suffix_text(student_processor.tokenizer, condition_output_ids, prompt_len)
    return traj_hs, {
        "student_text": student_text,
        "condition_suffix_text": suffix,
        "prompt_unchanged": bool(torch.equal(condition_output_ids[:, :prompt_len], output_ids[:, :prompt_len])),
        "prompt_len": int(prompt_len),
        "condition_seq_len": int(condition_output_ids.shape[1]),
    }


def _native_teacher_branch(
    teacher_model: Any,
    teacher_processor: Any,
    teacher_traj_to_actions_fn: Any,
    sample: dict[str, Any],
    dataset_index: int,
    args: argparse.Namespace,
    teacher_device: torch.device,
    action_scale: float,
) -> dict[str, Any]:
    turn_args = types.SimpleNamespace(
        front_width=int(args.teacher_front_width),
        front_height=int(args.teacher_front_height),
        conjunction_mode="fixed",
        fixed_conjunction="you can see ",
    )
    rng = random.Random(int(args.seed) + int(dataset_index) * 1009)
    first_messages, first_images = _build_first_turn(sample, turn_args, rng)
    (
        coord_text,
        output_ids,
        inputs,
        prompt_len,
        coord_uv,
        goal_yx,
    ) = _condition_on_dataset_coord(
        teacher_processor,
        first_messages,
        first_images,
        sample,
        turn_args,
        rng,
        teacher_device,
    )
    image_grid_thw = _normalize_image_grid_thw(inputs)
    traj_images = _build_images_dp(
        sample,
        device=teacher_device,
        dtype=_torch_dtype(args.teacher_torch_dtype),
        image_size=(int(args.teacher_traj_image_size), int(args.teacher_traj_image_size)),
    )
    with torch.inference_mode():
        traj_hs = teacher_model.generate_latents(
            output_ids,
            inputs.pixel_values,
            image_grid_thw,
        )
        cond_projector = _find_cond_projector(teacher_model)
        if cond_projector is None:
            raise RuntimeError("Teacher model has no cond_projector")
        proj_dtype = next(cond_projector.parameters()).dtype
        cond = cond_projector(traj_hs.to(dtype=proj_dtype))
        dp_actions = teacher_model.generate_traj(
            traj_hs,
            traj_images,
            None,
            predict_step_nums=int(args.teacher_predict_steps),
            guidance_scale=float(args.teacher_guidance_scale),
            num_inference_steps=int(args.teacher_num_inference_steps),
            num_sample_trajs=int(args.teacher_num_sample_trajs),
        )
    teacher_native_actions = _pad_actions(
        [int(a) for a in teacher_traj_to_actions_fn(dp_actions.clone())],
        8,
    )
    rollout = _rollout_fields(
        dp_actions,
        num_sample_trajs=int(args.teacher_num_sample_trajs),
        action_scale=action_scale,
        trajectory_selection=args.trajectory_selection,
    )
    return {
        "coord_text": coord_text,
        "coord_uv": coord_uv,
        "internnav_pixel_goal_yx": goal_yx,
        "prompt_len": int(prompt_len),
        "traj_hs": traj_hs,
        "cond": cond,
        "dp_actions": dp_actions,
        "teacher_native_actions8": teacher_native_actions,
        "teacher_native_actions_first4": teacher_native_actions[:4],
        "rollout": rollout,
    }


def _record_one(
    *,
    pano_idx: int,
    native_idx: int,
    pano_sample: dict[str, Any],
    native_sample: dict[str, Any],
    pano_dataset: Any,
    native_dataset: Any,
    student_model: Any,
    student_processor: Any,
    adapter: torch.nn.Module,
    teacher_model: Any,
    teacher_processor: Any,
    teacher_traj_to_actions_fn: Any,
    student_device: torch.device,
    teacher_device: torch.device,
    action_scale: float,
    student_num_sample_trajs: int,
    n_traj_query: int,
    internnav_protocol: bool,
    structured_pano_output: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    gold_view = str(pano_sample["pano_view_id"]).lower()
    gold_pixel = [int(pano_sample["pano_pixel_goal"][0]), int(pano_sample["pano_pixel_goal"][1])]
    oracle_text = structured_condition_text(gold_view, gold_pixel)

    teacher = _native_teacher_branch(
        teacher_model,
        teacher_processor,
        teacher_traj_to_actions_fn,
        native_sample,
        native_idx,
        args,
        teacher_device,
        action_scale,
    )

    train_hs, train_meta = _train_path_latents(
        student_model,
        student_processor,
        pano_sample,
        student_device,
        n_traj_query=n_traj_query,
    )
    eval_hs, eval_meta = _eval_oracle_latents(
        student_model,
        student_processor,
        pano_sample,
        student_device,
        internnav_protocol=internnav_protocol,
        structured_pano_output=structured_pano_output,
        max_new_tokens=args.max_new_tokens,
        gold_view=gold_view,
        gold_pixel=gold_pixel,
    )
    traj_images = _traj_images_from_sample(pano_sample, student_device, student_model.config.dtype)
    train = _adapter_project_and_rollout(
        student_model,
        adapter,
        train_hs,
        traj_images,
        num_sample_trajs=student_num_sample_trajs,
        action_scale=action_scale,
        trajectory_selection=args.trajectory_selection,
    )
    eval_oracle = _adapter_project_and_rollout(
        student_model,
        adapter,
        eval_hs,
        traj_images,
        num_sample_trajs=student_num_sample_trajs,
        action_scale=action_scale,
        trajectory_selection=args.trajectory_selection,
    )
    teacher_on_student = _student_rollout_from_teacher_raw(
        student_model,
        teacher["traj_hs"],
        traj_images,
        num_sample_trajs=student_num_sample_trajs,
        action_scale=action_scale,
        trajectory_selection=args.trajectory_selection,
    )

    pano_pair = _sample_index_pair(pano_dataset, pano_idx)
    native_pair = _sample_index_pair(native_dataset, native_idx)
    suffix_ok = eval_meta.get("condition_suffix_text") == oracle_text

    return {
        "pano_dataset_index": int(pano_idx),
        "native_dataset_index": int(native_idx),
        "dataset_index": int(pano_idx),
        "pano_sample_index": pano_pair,
        "native_sample_index": native_pair,
        "sample_index_matches": pano_pair == native_pair,
        "gold_pano": {
            "view": gold_view,
            "pixel_uv": gold_pixel,
            "oracle_text": oracle_text,
        },
        "native_dataset": {
            "pixel_goal_uv": [int(native_sample["pixel_goal"][0]), int(native_sample["pixel_goal"][1])],
        },
        "teacher_native": {
            "coord_text": teacher["coord_text"],
            "coord_uv": teacher["coord_uv"],
            "internnav_pixel_goal_yx": teacher["internnav_pixel_goal_yx"],
            "prompt_len": teacher["prompt_len"],
            "traj_hs_norm": float(teacher["traj_hs"].float().norm().item()),
            "cond_norm": float(teacher["cond"].float().norm().item()),
            "teacher_native_actions8": teacher["teacher_native_actions8"],
            "teacher_native_actions_first4": teacher["teacher_native_actions_first4"],
            "rollout": teacher["rollout"],
        },
        "teacher_raw_on_student_system1": {
            "cond_norm": float(teacher_on_student["cond"].float().norm().item()),
            "rollout": teacher_on_student["rollout"],
        },
        "adapter_train": {
            "meta": train_meta,
            "traj_hs_norm": float(train_hs.float().norm().item()),
            "adapted_norm": float(train["adapted"].float().norm().item()),
            "cond_norm": float(train["cond"].float().norm().item()),
            "rollout": train["rollout"],
        },
        "adapter_eval_oracle": {
            "meta": eval_meta,
            "suffix_matches_oracle": suffix_ok,
            "traj_hs_norm": float(eval_hs.float().norm().item()),
            "adapted_norm": float(eval_oracle["adapted"].float().norm().item()),
            "cond_norm": float(eval_oracle["cond"].float().norm().item()),
            "rollout": eval_oracle["rollout"],
        },
        "metrics": {
            "raw_train_vs_eval": _tensor_metrics(train_hs, eval_hs),
            "raw_train_vs_eval_per_query_cosine": _per_query_cosine(train_hs, eval_hs),
            "train_adapted_vs_teacher_raw": _tensor_metrics(train["adapted"], teacher["traj_hs"]),
            "train_adapted_vs_teacher_raw_per_query_cosine": _per_query_cosine(train["adapted"], teacher["traj_hs"]),
            "eval_adapted_vs_teacher_raw": _tensor_metrics(eval_oracle["adapted"], teacher["traj_hs"]),
            "eval_adapted_vs_teacher_raw_per_query_cosine": _per_query_cosine(eval_oracle["adapted"], teacher["traj_hs"]),
            "train_cond_vs_teacher_cond": _tensor_metrics(train["cond"], teacher["cond"]),
            "train_cond_vs_teacher_cond_per_query_cosine": _per_query_cosine(train["cond"], teacher["cond"]),
            "eval_cond_vs_teacher_cond": _tensor_metrics(eval_oracle["cond"], teacher["cond"]),
            "eval_cond_vs_teacher_cond_per_query_cosine": _per_query_cosine(eval_oracle["cond"], teacher["cond"]),
            "teacher_student_cond_vs_teacher_cond": _tensor_metrics(teacher_on_student["cond"], teacher["cond"]),
            "teacher_student_cond_vs_teacher_cond_per_query_cosine": _per_query_cosine(
                teacher_on_student["cond"],
                teacher["cond"],
            ),
        },
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _collect_float(records: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for rec in records:
        cur: Any = rec
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        if isinstance(cur, (int, float)):
            values.append(float(cur))
    return values


def _forward_count(records: list[dict[str, Any]], branch: str) -> int:
    return sum(
        1
        for rec in records
        if rec[branch]["rollout"].get("forward_count_first4", 0) > 0
    )


def _summarize(records: list[dict[str, Any]], out_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    n = len(records)
    summary: dict[str, Any] = {
        "num_samples": n,
        "output": str(out_path),
        "trajectory_selection": args.trajectory_selection,
        "sample_index_matches": sum(1 for rec in records if rec.get("sample_index_matches")),
        "suffix_matches_oracle": sum(
            1 for rec in records if rec["adapter_eval_oracle"].get("suffix_matches_oracle")
        ),
        "prompt_unchanged": sum(
            1 for rec in records if rec["adapter_eval_oracle"]["meta"].get("prompt_unchanged")
        ),
        "branches": {},
        "cosines": {},
    }
    for branch in ("teacher_native", "teacher_raw_on_student_system1", "adapter_train", "adapter_eval_oracle"):
        path_lens = _collect_float(records, (branch, "rollout", "path_len"))
        direct = _collect_float(records, (branch, "rollout", "direct"))
        fwd = _forward_count(records, branch)
        summary["branches"][branch] = {
            "forward_first4": fwd,
            "forward_first4_rate": float(fwd / n) if n else None,
            "path_len_mean": _mean(path_lens),
            "path_len_median": _median(path_lens),
            "direct_mean": _mean(direct),
            "direct_median": _median(direct),
        }
    for name in (
        "raw_train_vs_eval",
        "train_adapted_vs_teacher_raw",
        "eval_adapted_vs_teacher_raw",
        "train_cond_vs_teacher_cond",
        "eval_cond_vs_teacher_cond",
        "teacher_student_cond_vs_teacher_cond",
    ):
        vals = _collect_float(records, ("metrics", name, "cosine"))
        l2_vals = _collect_float(records, ("metrics", name, "l2"))
        summary["cosines"][name] = {
            "mean": _mean(vals),
            "median": _median(vals),
            "min": float(min(vals)) if vals else None,
            "l2_mean": _mean(l2_vals),
        }
    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    student_device = torch.device(args.student_device if torch.cuda.is_available() else "cpu")
    teacher_device = torch.device(args.teacher_device if torch.cuda.is_available() else "cpu")

    model_args = argparse.Namespace(
        config=args.config,
        checkpoint=None,
        base_checkpoint=args.base_checkpoint,
    )
    student_model, model_cfg = load_model(model_args, student_device)
    if student_model.nextdit_action_head is None or student_model.latent_queries is None:
        raise RuntimeError("Student config/model must enable NextDiT action_head and latent_queries")
    student_processor = student_model.qwen2_5_vl.processor
    if student_processor is None:
        raise RuntimeError("Student Qwen processor is missing")

    hidden_dim = int(model_cfg.get("model", {}).get("llm", {}).get("hidden_dim", 3584))
    adapter = _load_pano_latent_adapter(args.adapter_checkpoint, hidden_dim, student_device)

    teacher_load_args = types.SimpleNamespace(
        internnav_repo=str(args.internnav_repo),
        model_path=str(args.teacher_model_path),
        flash_attn_stub=bool(args.teacher_flash_attn_stub),
        torch_dtype=str(args.teacher_torch_dtype),
        attn_implementation=str(args.teacher_attn_implementation),
        require_nextdit=True,
    )
    teacher_model, teacher_processor, teacher_traj_to_actions_fn = _load_teacher(
        teacher_load_args,
        teacher_device,
    )

    pano_cfg = _load_cfg_for_dataset(args)
    pano_dataset = _build_dataset(pano_cfg, args.split)
    native_cfg = _load_native_cfg(args)
    native_dataset = _build_native_dataset(native_cfg, args.split)
    index_pairs = _choose_index_pairs(pano_dataset, native_dataset, args.num_samples, args.seed)
    if not index_pairs:
        raise RuntimeError("No overlapping pano/native pixel samples found")

    traj_cfg = pano_cfg.get("data", {}).get("trajectory", {})
    action_scale = float(traj_cfg.get("action_scale", pano_cfg.get("data", {}).get("action_scale", 4.0)))
    student_num_sample_trajs = int(
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
        for i, (pano_idx, native_idx) in enumerate(index_pairs, start=1):
            rec = _record_one(
                pano_idx=pano_idx,
                native_idx=native_idx,
                pano_sample=pano_dataset[pano_idx],
                native_sample=native_dataset[native_idx],
                pano_dataset=pano_dataset,
                native_dataset=native_dataset,
                student_model=student_model,
                student_processor=student_processor,
                adapter=adapter,
                teacher_model=teacher_model,
                teacher_processor=teacher_processor,
                teacher_traj_to_actions_fn=teacher_traj_to_actions_fn,
                student_device=student_device,
                teacher_device=teacher_device,
                action_scale=action_scale,
                student_num_sample_trajs=student_num_sample_trajs,
                n_traj_query=n_traj_query,
                internnav_protocol=internnav_protocol,
                structured_pano_output=structured_output,
                args=args,
            )
            records.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            if i <= args.max_prints:
                LOGGER.info(
                    "[%d/%d idx=%s] teacher=%s path=%.3f teacher->student=%s path=%.3f "
                    "train=%s path=%.3f eval=%s path=%.3f "
                    "cond_cos(train/eval)=%.4f/%.4f suffix_ok=%s",
                    i,
                    len(index_pairs),
                    f"{pano_idx}/{native_idx}",
                    rec["teacher_native"]["rollout"]["actions_first4"],
                    rec["teacher_native"]["rollout"].get("path_len") or -1.0,
                    rec["teacher_raw_on_student_system1"]["rollout"]["actions_first4"],
                    rec["teacher_raw_on_student_system1"]["rollout"].get("path_len") or -1.0,
                    rec["adapter_train"]["rollout"]["actions_first4"],
                    rec["adapter_train"]["rollout"].get("path_len") or -1.0,
                    rec["adapter_eval_oracle"]["rollout"]["actions_first4"],
                    rec["adapter_eval_oracle"]["rollout"].get("path_len") or -1.0,
                    rec["metrics"]["train_cond_vs_teacher_cond"]["cosine"],
                    rec["metrics"]["eval_cond_vs_teacher_cond"]["cosine"],
                    rec["adapter_eval_oracle"]["suffix_matches_oracle"],
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = _summarize(records, out_path, args)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Summary: %s", json.dumps(summary, ensure_ascii=False))
    LOGGER.info("Wrote %s and %s", out_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

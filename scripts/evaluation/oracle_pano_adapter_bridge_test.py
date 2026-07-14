#!/usr/bin/env python3
"""
Offline oracle-replacement bridge test for the pano latent adapter.

This intentionally mirrors the online panoramic eval sequence:

  images + prompt -> autoregressive System2 text generation
  -> replace only the generated assistant-answer suffix with gold pano text
  -> generate_latents(... latent_queries ...)
  -> pano adapter -> cond_projector -> frozen NextDiT -> discrete actions

The replacement happens before the latent-query Qwen forward.  Prompt tokens and
image tensors are kept identical between the generated and oracle branches.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from collections import Counter
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

from scripts.evaluation.r2r_val_unseen import (
    _condition_output_ids_for_pixel_goal,
    _finalize_local_actions,
    _load_pano_latent_adapter,
    _maybe_apply_pano_latent_adapter,
    _normalize_multimodal_inputs,
    _parse_pano_view_id,
    _parse_pixel_goal,
    _trajectory_debug_summary,
    _trajectory_from_condition,
    load_model,
    reconstruct_xy_from_delta,
    select_trajectory_xy,
    traj_to_actions,
)
from scripts.evaluation.system2_sft_sanity_check import (
    current_views_from_sample,
    history_panoramas_from_sample,
)
from scripts.training.utils import load_config
from src.data.factory import build_trajectory_dataset
from src.models.heatmap.input_constructor import construct_input, structured_condition_text
from src.utils.trajectory_direction import (
    VIEW_TARGET_ANGLE_DEG,
    angular_error_deg,
    pairwise_representation_stats,
    summarize_direction_response,
)

LOGGER = logging.getLogger("oracle_pano_adapter_bridge")
VALID_PANO_VIEWS = {"front", "right", "back", "left"}
COUNTERFACTUAL_VIEWS = ("front", "right", "back", "left")
TRAJECTORY_SELECTIONS = (
    "mean",
    "endpoint_medoid",
    "path_medoid",
    "median_endpoint_nearest",
    "forward_or_medoid",
    "longest_forward",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run generated-vs-oracle text replacement through "
            "generate_latents -> pano adapter -> frozen NextDiT."
        ),
    )
    p.add_argument("--config", default="configs/train_pano_adapter_stage2_8gpu.yaml")
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--adapter-checkpoint", required=True)
    p.add_argument("--root", default="/workspace/r2r_panoramic_audit_v2")
    p.add_argument("--split", default="train")
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument(
        "--trajectory-selection",
        default="mean",
        choices=TRAJECTORY_SELECTIONS,
    )
    p.add_argument(
        "--output",
        default="logs/oracle_pano_adapter_bridge_test.jsonl",
        help="Per-sample JSONL output.",
    )
    p.add_argument("--max-prints", type=int, default=8)
    p.add_argument(
        "--counterfactual-all-views",
        action="store_true",
        help=(
            "For each fixed image/prompt sample, replace the assistant suffix "
            "with front/right/back/left center goals and report latent and "
            "trajectory direction separation."
        ),
    )
    p.add_argument(
        "--counterfactual-pixel",
        type=int,
        nargs=2,
        metavar=("U", "V"),
        default=None,
        help="Pixel used for all counterfactual views; defaults to image center.",
    )
    p.add_argument(
        "--trajectory-seed",
        type=int,
        default=20260714,
        help="Shared diffusion seed across counterfactual views for each sample.",
    )
    p.add_argument(
        "--dataset-max-clips",
        type=int,
        default=0,
        help="Limit dataset indexing for diagnostics; 0 keeps all clips.",
    )
    return p.parse_args()


def _load_cfg_for_dataset(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    cfg["data"]["root"] = args.root
    if "paths" in cfg:
        cfg["paths"]["dataset_root"] = args.root
    traj_cfg = cfg.setdefault("data", {}).setdefault("trajectory", {})
    traj_cfg["panoramic_vlm_input"] = True
    traj_cfg["compute_pixel_goal"] = True
    traj_cfg["compute_pano_view_pixel_goal"] = True
    traj_cfg["load_lookdown_for_system2"] = False
    traj_cfg["load_traj_images"] = True
    traj_cfg["enable_trajectory_augmentation"] = False
    traj_cfg["require_sft_target"] = False
    return cfg


def _build_dataset(cfg: dict[str, Any], split: str, *, max_clips: int = 0):
    return build_trajectory_dataset(
        cfg,
        split=split,
        enable_trajectory_augmentation=False,
        load_depth=False,
        load_history_heatmap=False,
        panoramic_vlm_input=True,
        compute_pixel_goal=True,
        compute_pano_view_pixel_goal=True,
        load_lookdown_for_system2=False,
        load_traj_images=True,
        require_sft_target=False,
        max_clips=max(0, int(max_clips)),
    )


def _move_inputs_to_device(inputs: Any, device: torch.device) -> dict[str, torch.Tensor]:
    if not hasattr(inputs, "items"):
        raise TypeError(f"Unexpected processor output type: {type(inputs)}")
    moved = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    _normalize_multimodal_inputs(moved)
    return moved


def _prepare_prompt_inputs(
    processor,
    sample: dict[str, Any],
    device: torch.device,
    *,
    internnav_protocol: bool,
    structured_pano_output: bool,
) -> dict[str, torch.Tensor]:
    messages = construct_input(
        current_views=current_views_from_sample(sample),
        history_panoramas=history_panoramas_from_sample(sample),
        instruction=sample.get("text"),
        pixel_goal=[0, 0],
        internnav_protocol=internnav_protocol,
        structured_pano_output=structured_pano_output,
    )
    messages = [m for m in messages if m.get("role") != "assistant"]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return _move_inputs_to_device(inputs, device)


def _generate_system2_text(
    model,
    processor,
    inputs: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
) -> tuple[str, torch.Tensor, int]:
    with torch.inference_mode():
        output_ids = model.qwen2_5_vl.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        ).sequences
    prompt_len = int(inputs["input_ids"].shape[1])
    text = processor.tokenizer.decode(
        output_ids[0][prompt_len:],
        skip_special_tokens=True,
    ).strip()
    return text, output_ids, prompt_len


def _traj_images_from_sample(
    sample: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    traj_images = sample.get("traj_images")
    if traj_images is None:
        raise RuntimeError("Sample is missing traj_images; build dataset with load_traj_images=True")
    if not torch.is_tensor(traj_images):
        traj_images = torch.as_tensor(traj_images)
    if traj_images.ndim == 5:
        image = traj_images[0, 0]
    elif traj_images.ndim == 4:
        image = traj_images[0]
    elif traj_images.ndim == 3:
        image = traj_images
    else:
        raise RuntimeError(f"Unexpected traj_images shape: {tuple(traj_images.shape)}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise RuntimeError(f"Expected HWC traj image, got {tuple(image.shape)}")
    image = image.detach().float().clamp(0.0, 1.0)
    pair = torch.stack([image, image], dim=0).unsqueeze(0)
    return pair.to(device=device, dtype=dtype)


def _parse_summary(summary: str) -> dict[str, float | None]:
    match = re.search(r"direct=([-+0-9.eE]+), path_len=([-+0-9.eE]+)", summary)
    if not match:
        return {"direct": None, "path_len": None}
    return {
        "direct": float(match.group(1)),
        "path_len": float(match.group(2)),
    }


def _condition_suffix_text(tokenizer, condition_output_ids: torch.Tensor, prompt_len: int) -> str:
    return tokenizer.decode(
        condition_output_ids[0][prompt_len:],
        skip_special_tokens=True,
    ).strip()


def _run_condition_branch(
    *,
    branch: str,
    model,
    processor,
    adapter,
    inputs: dict[str, torch.Tensor],
    output_ids: torch.Tensor,
    prompt_len: int,
    llm_output: str,
    pixel_goal: list[int],
    view_id: str,
    structured_output: bool,
    traj_images: torch.Tensor,
    num_sample_trajs: int,
    action_scale: float,
    trajectory_selection: str,
    trajectory_seed: int | None = None,
    diagnostic_detail: bool = False,
) -> dict[str, Any]:
    tokenizer = processor.tokenizer

    condition_output_ids = _condition_output_ids_for_pixel_goal(
        output_ids=output_ids,
        prompt_len=prompt_len,
        tokenizer=tokenizer,
        pixel_goal=pixel_goal,
        llm_output=llm_output,
        coord_order="generated",
        view_id=view_id,
        structured_output=structured_output,
    )

    prompt_unchanged = bool(torch.equal(condition_output_ids[:, :prompt_len], output_ids[:, :prompt_len]))
    suffix_text = _condition_suffix_text(tokenizer, condition_output_ids, prompt_len)

    lq = model.latent_queries.expand(1, -1, -1).to(
        device=condition_output_ids.device,
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
        traj_hs_norm = float(traj_hs.float().norm().item())
        conditioned = _maybe_apply_pano_latent_adapter(
            traj_hs,
            adapter,
            view_id=view_id,
            pixel_goal=pixel_goal,
            image_size=None,
            cond_projector=(
                model.nextdit_action_head.cond_projector
                if model.nextdit_action_head is not None
                else None
            ),
        )
        conditioned_norm = float(conditioned.float().norm().item())
        if trajectory_seed is not None:
            torch.manual_seed(int(trajectory_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(trajectory_seed))
        trajectory = _trajectory_from_condition(
            model.nextdit_action_head,
            conditioned,
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
    parsed_summary = _parse_summary(summary)

    result: dict[str, Any] = {
        "branch": branch,
        "condition_suffix_text": suffix_text,
        "prompt_unchanged": prompt_unchanged,
        "traj_hs_norm": traj_hs_norm,
        "conditioned_norm": conditioned_norm,
        "trajectory_summary": summary,
        **parsed_summary,
        "actions_first4": [int(a) for a in actions[:4]],
        "forward_count_first4": sum(1 for a in actions[:4] if int(a) == 1),
        "forward_count": sum(1 for a in actions if int(a) == 1),
        "turn_only_first4": all(int(a) in (2, 3) for a in actions[:4]),
        "no_forward_first4": all(int(a) != 1 for a in actions[:4]),
    }
    if diagnostic_detail:
        trajectory_np = trajectory[:num_sample_trajs].float().detach().cpu().numpy()
        direction = summarize_direction_response(
            trajectory_np,
            view_id=view_id,
            action_scale=action_scale,
        )
        scaled = trajectory_np.copy()
        scaled[:, :, :2] /= float(action_scale)
        all_xy = reconstruct_xy_from_delta(scaled)
        selections: dict[str, Any] = {}
        for selection in TRAJECTORY_SELECTIONS:
            selected_xy, selected_idx = select_trajectory_xy(all_xy, selection)
            endpoint = selected_xy[-1, :2]
            direct = float(np.linalg.norm(endpoint))
            angle = (
                float(np.degrees(np.arctan2(endpoint[1], endpoint[0])))
                if direct > 1.0e-6
                else None
            )
            selected_actions = _finalize_local_actions(
                traj_to_actions(
                    trajectory,
                    num_sample_trajs=num_sample_trajs,
                    action_scale=action_scale,
                    trajectory_selection=selection,
                )
            )
            selections[selection] = {
                "selected_index": int(selected_idx) if selected_idx is not None else None,
                "endpoint_xy_m": [float(endpoint[0]), float(endpoint[1])],
                "endpoint_angle_deg": angle,
                "endpoint_angle_error_deg": (
                    float(angular_error_deg(angle, VIEW_TARGET_ANGLE_DEG[view_id]))
                    if angle is not None
                    else 180.0
                ),
                "actions_first4": [int(action) for action in selected_actions[:4]],
            }
        result["direction"] = direction
        result["selections"] = selections
        result["_traj_hs_repr"] = traj_hs.float().detach().cpu().numpy()
        result["_conditioned_repr"] = conditioned.float().detach().cpu().numpy()
    return result


def _generated_branch_goal(llm_output: str, image_size: tuple[int, int]) -> tuple[str, list[int]] | None:
    pixel_goal = _parse_pixel_goal(llm_output, image_size)
    if pixel_goal is None:
        return None
    return _parse_pano_view_id(llm_output) or "front", pixel_goal


def _choose_indices(dataset, num_samples: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    chosen: list[int] = []
    for idx in indices:
        sample = dataset[idx]
        view_id = str(sample.get("pano_view_id") or "").lower()
        pixel_goal = sample.get("pano_pixel_goal")
        if view_id not in VALID_PANO_VIEWS or pixel_goal is None:
            continue
        if sample.get("traj_images") is None:
            continue
        chosen.append(idx)
        if len(chosen) >= num_samples:
            break
    return chosen


def _run_counterfactual_sample(
    *,
    dataset,
    sample: dict[str, Any],
    dataset_index: int,
    model,
    processor,
    adapter,
    device: torch.device,
    internnav_protocol: bool,
    structured_output: bool,
    image_size: tuple[int, int],
    counterfactual_pixel: list[int] | None,
    max_new_tokens: int,
    num_sample_trajs: int,
    action_scale: float,
    trajectory_seed: int,
) -> dict[str, Any]:
    inputs = _prepare_prompt_inputs(
        processor,
        sample,
        device,
        internnav_protocol=internnav_protocol,
        structured_pano_output=structured_output,
    )
    student_text, output_ids, prompt_len = _generate_system2_text(
        model,
        processor,
        inputs,
        max_new_tokens=max_new_tokens,
    )
    traj_images = _traj_images_from_sample(sample, device, model.config.dtype)
    pixel = (
        [int(counterfactual_pixel[0]), int(counterfactual_pixel[1])]
        if counterfactual_pixel is not None
        else [int(image_size[0] // 2), int(image_size[1] // 2)]
    )

    raw_representations: dict[str, np.ndarray] = {}
    conditioned_representations: dict[str, np.ndarray] = {}
    view_results: dict[str, Any] = {}
    shared_seed = int(trajectory_seed) + int(dataset_index)
    for view_id in COUNTERFACTUAL_VIEWS:
        result = _run_condition_branch(
            branch=f"counterfactual_{view_id}",
            model=model,
            processor=processor,
            adapter=adapter,
            inputs=inputs,
            output_ids=output_ids,
            prompt_len=prompt_len,
            llm_output=student_text,
            pixel_goal=pixel,
            view_id=view_id,
            structured_output=True,
            traj_images=traj_images,
            num_sample_trajs=num_sample_trajs,
            action_scale=action_scale,
            trajectory_selection="mean",
            trajectory_seed=shared_seed,
            diagnostic_detail=True,
        )
        raw_representations[view_id] = result.pop("_traj_hs_repr")
        conditioned_representations[view_id] = result.pop("_conditioned_repr")
        result["expected_suffix_text"] = structured_condition_text(view_id, pixel)
        result["suffix_matches_expected"] = (
            result["condition_suffix_text"] == result["expected_suffix_text"]
        )
        view_results[view_id] = result

    clip_idx = current_t = None
    if hasattr(dataset, "sample_index"):
        clip_idx, current_t = dataset.sample_index[dataset_index]
    return {
        "dataset_index": int(dataset_index),
        "clip_idx": int(clip_idx) if clip_idx is not None else None,
        "current_t": int(current_t) if current_t is not None else None,
        "source_view": str(sample.get("pano_view_id") or ""),
        "source_pixel": (
            [int(value) for value in sample["pano_pixel_goal"]]
            if sample.get("pano_pixel_goal") is not None
            else None
        ),
        "counterfactual_pixel": pixel,
        "student_text": student_text,
        "prompt_len": prompt_len,
        "trajectory_seed": shared_seed,
        "representations": {
            "raw": pairwise_representation_stats(raw_representations),
            "conditioned": pairwise_representation_stats(conditioned_representations),
        },
        "views": view_results,
    }


def _summarize_counterfactual(
    records: list[dict[str, Any]],
    *,
    output: Path,
    adapter_checkpoint: str,
) -> dict[str, Any]:
    direction_keys = (
        "candidate_angle_error_mean_deg",
        "candidate_angle_error_median_deg",
        "candidate_within_45_rate",
        "candidate_within_90_rate",
        "candidate_positive_progress_rate",
        "candidate_alignment_mean",
        "candidate_progress_mean_m",
        "mean_endpoint_angle_error_deg",
        "mean_endpoint_progress_m",
    )
    views: dict[str, Any] = {}
    for view_id in COUNTERFACTUAL_VIEWS:
        view_summary: dict[str, Any] = {}
        for key in direction_keys:
            values = [float(rec["views"][view_id]["direction"][key]) for rec in records]
            view_summary[f"{key}_mean"] = float(np.mean(values))
            view_summary[f"{key}_median"] = float(np.median(values))
        selection_summary: dict[str, Any] = {}
        for selection in TRAJECTORY_SELECTIONS:
            errors = [
                float(rec["views"][view_id]["selections"][selection]["endpoint_angle_error_deg"])
                for rec in records
            ]
            selection_summary[selection] = {
                "endpoint_angle_error_deg_mean": float(np.mean(errors)),
                "endpoint_angle_error_deg_median": float(np.median(errors)),
                "within_45_rate": float(np.mean(np.asarray(errors) <= 45.0)),
            }
        view_summary["selections"] = selection_summary
        views[view_id] = view_summary

    representation_summary: dict[str, Any] = {}
    for stage in ("raw", "conditioned"):
        cosines = [float(rec["representations"][stage]["cosine_mean"]) for rec in records]
        relative_l2 = [
            float(rec["representations"][stage]["relative_l2_mean"])
            for rec in records
        ]
        representation_summary[stage] = {
            "pairwise_cosine_mean": float(np.mean(cosines)),
            "pairwise_cosine_min": float(np.min(cosines)),
            "pairwise_relative_l2_mean": float(np.mean(relative_l2)),
            "pairwise_relative_l2_max": float(np.max(relative_l2)),
        }

    return {
        "mode": "counterfactual_all_views",
        "num_samples": len(records),
        "adapter_checkpoint": str(adapter_checkpoint),
        "views": views,
        "representations": representation_summary,
        "output": str(output),
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

    hidden_dim = int(model_cfg.get("model", {}).get("llm", {}).get("hidden_dim", 3584))
    adapter = _load_pano_latent_adapter(args.adapter_checkpoint, hidden_dim, device)

    dataset_cfg = _load_cfg_for_dataset(args)
    dataset = _build_dataset(
        dataset_cfg,
        args.split,
        max_clips=args.dataset_max_clips,
    )
    indices = _choose_indices(dataset, args.num_samples, args.seed)
    if not indices:
        raise RuntimeError("No pixel pano samples found for oracle bridge test")

    traj_cfg = dataset_cfg.get("data", {}).get("trajectory", {})
    action_scale = float(traj_cfg.get("action_scale", dataset_cfg.get("data", {}).get("action_scale", 4.0)))
    image_size = tuple(int(v) for v in dataset_cfg.get("data", {}).get("image_size", [256, 256]))
    if args.counterfactual_pixel is not None:
        u, v = (int(value) for value in args.counterfactual_pixel)
        width, height = image_size
        if not (0 <= u < width and 0 <= v < height):
            raise ValueError(
                "--counterfactual-pixel must be inside the configured image: "
                f"pixel={(u, v)} image_size={image_size}"
            )
    num_sample_trajs = int(
        model_cfg.get("model", {})
        .get("action_head", {})
        .get("nextdit", {})
        .get("num_sample_trajs", 32)
    )
    protocol = str(traj_cfg.get("system2_sft_protocol", "direct")).lower()
    structured_output = bool(traj_cfg.get("structured_pano_output", True))
    internnav_protocol = protocol == "internnav"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    counterfactual_records: list[dict[str, Any]] = []
    branch_metrics: dict[str, Counter[str]] = {
        "generated": Counter(),
        "oracle_replace": Counter(),
    }
    numeric: dict[str, dict[str, list[float]]] = {
        "generated": {"direct": [], "path_len": []},
        "oracle_replace": {"direct": [], "path_len": []},
    }

    processor = model.qwen2_5_vl.processor
    assert processor is not None

    with out_path.open("w", encoding="utf-8") as fout:
        for n, idx in enumerate(indices, start=1):
            sample = dataset[idx]
            if args.counterfactual_all_views:
                counterfactual = _run_counterfactual_sample(
                    dataset=dataset,
                    sample=sample,
                    dataset_index=idx,
                    model=model,
                    processor=processor,
                    adapter=adapter,
                    device=device,
                    internnav_protocol=internnav_protocol,
                    structured_output=structured_output,
                    image_size=image_size,
                    counterfactual_pixel=args.counterfactual_pixel,
                    max_new_tokens=args.max_new_tokens,
                    num_sample_trajs=num_sample_trajs,
                    action_scale=action_scale,
                    trajectory_seed=args.trajectory_seed,
                )
                fout.write(json.dumps(counterfactual, ensure_ascii=False) + "\n")
                counterfactual_records.append(counterfactual)
                counters["processed"] += 1
                if n <= args.max_prints:
                    LOGGER.info(
                        "[%d/%d idx=%s] counterfactual mean endpoint errors: %s",
                        n,
                        len(indices),
                        idx,
                        {
                            view: round(
                                float(
                                    counterfactual["views"][view]["direction"][
                                        "mean_endpoint_angle_error_deg"
                                    ]
                                ),
                                2,
                            )
                            for view in COUNTERFACTUAL_VIEWS
                        },
                    )
                continue

            gold_view = str(sample["pano_view_id"]).lower()
            gold_pixel = [int(sample["pano_pixel_goal"][0]), int(sample["pano_pixel_goal"][1])]
            oracle_text = structured_condition_text(gold_view, gold_pixel)

            inputs = _prepare_prompt_inputs(
                processor,
                sample,
                device,
                internnav_protocol=internnav_protocol,
                structured_pano_output=structured_output,
            )
            student_text, output_ids, prompt_len = _generate_system2_text(
                model,
                processor,
                inputs,
                max_new_tokens=args.max_new_tokens,
            )
            traj_images = _traj_images_from_sample(sample, device, model.config.dtype)

            rec: dict[str, Any] = {
                "dataset_index": int(idx),
                "clip_idx": int(dataset.sample_index[idx][0]) if hasattr(dataset, "sample_index") else None,
                "current_t": int(dataset.sample_index[idx][1]) if hasattr(dataset, "sample_index") else None,
                "gold_view": gold_view,
                "gold_pixel": gold_pixel,
                "oracle_text": oracle_text,
                "student_text": student_text,
                "prompt_len": prompt_len,
            }

            generated_goal = _generated_branch_goal(student_text, image_size)
            if generated_goal is None:
                rec["generated"] = {
                    "ran": False,
                    "reason": "student output did not parse as pixel goal",
                }
                branch_metrics["generated"]["not_parseable"] += 1
            else:
                gen_view, gen_pixel = generated_goal
                gen_result = _run_condition_branch(
                    branch="generated",
                    model=model,
                    processor=processor,
                    adapter=adapter,
                    inputs=inputs,
                    output_ids=output_ids,
                    prompt_len=prompt_len,
                    llm_output=student_text,
                    pixel_goal=gen_pixel,
                    view_id=gen_view,
                    structured_output=structured_output,
                    traj_images=traj_images,
                    num_sample_trajs=num_sample_trajs,
                    action_scale=action_scale,
                    trajectory_selection=args.trajectory_selection,
                )
                gen_result["ran"] = True
                gen_result["view"] = gen_view
                gen_result["pixel"] = gen_pixel
                rec["generated"] = gen_result

            oracle_result = _run_condition_branch(
                branch="oracle_replace",
                model=model,
                processor=processor,
                adapter=adapter,
                inputs=inputs,
                output_ids=output_ids,
                prompt_len=prompt_len,
                llm_output=student_text,
                pixel_goal=gold_pixel,
                view_id=gold_view,
                structured_output=True,
                traj_images=traj_images,
                num_sample_trajs=num_sample_trajs,
                action_scale=action_scale,
                trajectory_selection=args.trajectory_selection,
            )
            oracle_result["ran"] = True
            oracle_result["expected_suffix_text"] = oracle_text
            oracle_result["suffix_matches_oracle"] = (
                oracle_result["condition_suffix_text"] == oracle_text
            )
            rec["oracle_replace"] = oracle_result

            for branch in ("generated", "oracle_replace"):
                br = rec.get(branch) or {}
                if not br.get("ran"):
                    continue
                branch_metrics[branch]["ran"] += 1
                if br.get("prompt_unchanged"):
                    branch_metrics[branch]["prompt_unchanged"] += 1
                if br.get("suffix_matches_oracle"):
                    branch_metrics[branch]["suffix_matches_oracle"] += 1
                if br.get("forward_count_first4", 0) > 0:
                    branch_metrics[branch]["forward_first4"] += 1
                if br.get("no_forward_first4"):
                    branch_metrics[branch]["no_forward_first4"] += 1
                if br.get("turn_only_first4"):
                    branch_metrics[branch]["turn_only_first4"] += 1
                for key in ("direct", "path_len"):
                    value = br.get(key)
                    if isinstance(value, (int, float)):
                        numeric[branch][key].append(float(value))

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counters["processed"] += 1
            if n <= args.max_prints:
                LOGGER.info(
                    "[%d/%d idx=%s] gold=%s %s student=%r oracle_actions=%s "
                    "oracle_summary=%s suffix_ok=%s prompt_ok=%s",
                    n,
                    len(indices),
                    idx,
                    gold_view,
                    gold_pixel,
                    student_text,
                    oracle_result["actions_first4"],
                    oracle_result["trajectory_summary"],
                    oracle_result["suffix_matches_oracle"],
                    oracle_result["prompt_unchanged"],
                )

    if args.counterfactual_all_views:
        summary = _summarize_counterfactual(
            counterfactual_records,
            output=out_path,
            adapter_checkpoint=args.adapter_checkpoint,
        )
        summary_path = out_path.with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        LOGGER.info("Counterfactual summary: %s", json.dumps(summary, ensure_ascii=False))
        LOGGER.info("Wrote %s and %s", out_path, summary_path)
        return 0

    summary: dict[str, Any] = {
        "num_samples": int(counters["processed"]),
        "trajectory_selection": args.trajectory_selection,
        "num_sample_trajs": num_sample_trajs,
        "action_scale": action_scale,
        "branches": {},
        "output": str(out_path),
    }
    for branch, metrics in branch_metrics.items():
        ran = int(metrics.get("ran", 0))
        branch_summary: dict[str, Any] = dict(metrics)
        for key, vals in numeric[branch].items():
            if vals:
                branch_summary[f"{key}_mean"] = float(np.mean(vals))
                branch_summary[f"{key}_median"] = float(np.median(vals))
        if ran:
            branch_summary["forward_first4_rate"] = float(metrics["forward_first4"] / ran)
            branch_summary["no_forward_first4_rate"] = float(metrics["no_forward_first4"] / ran)
            branch_summary["turn_only_first4_rate"] = float(metrics["turn_only_first4"] / ran)
        summary["branches"][branch] = branch_summary

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Summary: %s", json.dumps(summary, ensure_ascii=False))
    LOGGER.info("Wrote %s and %s", out_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

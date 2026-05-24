#!/usr/bin/env python3
"""
Oracle pixel-goal bridge test (offline, no Habitat, no VLM coordinate generation).

Uses gold pixel_goal from the dataset, eval-aligned InternNav two-turn prompts
(construct_input(..., pixel_goal=[0, 0])), then generate_latents + get_trajectory.
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

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image
from scripts.evaluation.r2r_val_unseen import (
    _condition_output_ids_for_pixel_goal,
    _extract_checkpoint_state_dict,
    _finalize_local_actions,
    _load_compatible_state_dict,
    _lookdown_to_traj_tensor,
    _normalize_multimodal_inputs,
    _resolve_internnav_model_path,
    _system1_coord_order,
    _trajectory_debug_summary,
    _verify_internnav_system1_loaded,
    traj_to_actions,
)
from scripts.evaluation.system2_sft_sanity_check import make_generation_messages
from scripts.training.model_builder import build_model
from scripts.training.utils import load_config

from src.data.factory import build_trajectory_dataset
from src.models.heatmap.input_constructor import INTERNAV_CONJUNCTIONS

LOGGER = logging.getLogger("oracle_bridge")


def _ensure_pil_from_sample_tensor(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if torch.is_tensor(img):
        arr = img.detach().cpu()
        if arr.dim() == 3 and arr.shape[0] in (1, 3):
            arr = arr.permute(1, 2, 0)
        arr = (arr.float().clamp(0, 1) * 255).byte().numpy()
        return Image.fromarray(arr)
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(img)
    raise TypeError(type(img))


def build_eval_aligned_messages(sample: dict) -> list[dict]:
    """Match eval: construct_input pixel_goal=[0,0], then InternNav ↓ + lookdown user."""
    messages = make_generation_messages(
        sample,
        prompt_mode="target_instruction",
        protocol="internnav",
    )
    lookdown = sample.get("lookdown_frame")
    if lookdown is None:
        raise RuntimeError("Sample missing lookdown_frame")
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": "↓"}],
    })
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": random.choice(INTERNAV_CONJUNCTIONS)},
            {"type": "image", "image": _ensure_pil_from_sample_tensor(lookdown)},
        ],
    })
    return messages


def load_eval_model(cfg: dict, base_ckpt: Path, stage2_ckpt: Path, device: torch.device):
    internnav_path = _resolve_internnav_model_path(cfg)
    if internnav_path:
        print(f"InternNav model path: {internnav_path}", flush=True)

    model = build_model(cfg, device=str(device), verbose=False)
    model = model.to(device)
    _verify_internnav_system1_loaded(model, internnav_path)

    # Qwen processor must exist before LoRA / bridge weights are applied (same as eval).
    model.qwen2_5_vl._load_model()
    if model.qwen2_5_vl.processor is None:
        raise RuntimeError("Qwen processor is still None after _load_model()")

    if model.nextdit_action_head is None:
        raise RuntimeError("nextdit_action_head is not enabled in config")

    base_sd = _extract_checkpoint_state_dict(str(base_ckpt))
    _load_compatible_state_dict(model, base_sd, str(base_ckpt), label="Base checkpoint")
    stage2_sd = _extract_checkpoint_state_dict(str(stage2_ckpt))
    _load_compatible_state_dict(model, stage2_sd, str(stage2_ckpt), label="Main checkpoint")

    model.eval()
    return model


def oracle_forward_one(
    model,
    sample: dict,
    device: torch.device,
    *,
    num_sample_trajs: int,
    action_scale: float,
    traj_image_size: tuple[int, int],
    coord_order: str,
) -> dict:
    pixel_goal = [int(sample["pixel_goal"][0]), int(sample["pixel_goal"][1])]
    coord_text = f"{pixel_goal[0]} {pixel_goal[1]}"

    messages = build_eval_aligned_messages(sample)
    processor = model.qwen2_5_vl.processor
    if processor is None:
        raise RuntimeError("processor is None — call load_eval_model() first")

    prefill = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_len = int(prefill["input_ids"].shape[1])

    messages_with_gold = [
        *messages,
        {
            "role": "assistant",
            "content": [{"type": "text", "text": coord_text}],
        },
    ]
    full = processor.apply_chat_template(
        messages_with_gold,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    output_ids = full["input_ids"].to(device)
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in full.items()}
    _normalize_multimodal_inputs(inputs)

    condition_output_ids = _condition_output_ids_for_pixel_goal(
        output_ids=output_ids,
        prompt_len=prompt_len,
        tokenizer=processor.tokenizer,
        pixel_goal=pixel_goal,
        llm_output=coord_text,
        coord_order=coord_order,
    )

    lq = model.latent_queries.expand(1, -1, -1).to(device=device, dtype=model.config.dtype)
    with torch.no_grad():
        traj_hs = model.qwen2_5_vl.generate_latents(
            output_ids=condition_output_ids,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            latent_queries=lq,
        )

    lookdown_pil = _ensure_pil_from_sample_tensor(sample["lookdown_frame"])
    if lookdown_pil.size != traj_image_size:
        lookdown_pil = lookdown_pil.resize(traj_image_size)
    traj_t = _lookdown_to_traj_tensor(lookdown_pil, device)
    traj_images = torch.stack([traj_t, traj_t]).unsqueeze(0).to(device)

    with torch.no_grad():
        trajectory = model.nextdit_action_head.get_trajectory(
            traj_hs,
            traj_images=traj_images,
        )

    actions = _finalize_local_actions(
        traj_to_actions(
            trajectory,
            num_sample_trajs=num_sample_trajs,
            action_scale=action_scale,
        )
    )
    summary = _trajectory_debug_summary(trajectory, num_sample_trajs, action_scale)
    return {
        "pixel_goal": pixel_goal,
        "coord_text": coord_text,
        "trajectory_summary": summary,
        "actions": actions,
        "forward_count": sum(1 for a in actions if a == 1),
        "zero_pad_pattern": actions[:4] == [0, 0, 0, 0],
        "turn_only": all(a in (2, 3) for a in actions if a != 0),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Oracle gold pixel_goal bridge test")
    p.add_argument("--config", default="configs/train_config_internnav_8gpu.yaml")
    p.add_argument("--base-checkpoint", default="checkpoints/stage1-s2_latest.pth")
    p.add_argument("--stage2-checkpoint", default="checkpoints/stage2_latest.pth")
    p.add_argument("--root", default="/workspace/r2r_panoramic_audit_v2")
    p.add_argument("--split", default="train")
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="logs/oracle_pixel_goal_bridge_audit.jsonl")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    cfg = load_config(args.config)
    cfg["data"]["root"] = args.root
    if "paths" in cfg:
        cfg["paths"]["dataset_root"] = args.root

    device = torch.device(args.device)
    traj_cfg = cfg["data"].get("trajectory", {})
    num_sample_trajs = int(traj_cfg.get("num_sample_trajs", cfg["model"]["action_head"]["nextdit"].get("num_sample_trajs", 32)))
    action_scale = float(traj_cfg.get("action_scale", 4.0))
    traj_size = tuple(traj_cfg.get("traj_image_size", [224, 224]))
    coord_order = _system1_coord_order(argparse.Namespace(system1_coord_order="auto"), panoramic_internnav_protocol=True)
    print(f"System1 coordinate text order: {coord_order}", flush=True)

    model = load_eval_model(
        cfg,
        Path(args.base_checkpoint),
        Path(args.stage2_checkpoint),
        device,
    )

    dataset = build_trajectory_dataset(cfg, split=args.split)
    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")

    records = []
    direct_list = []
    path_len_list = []
    skipped = 0

    for idx in indices:
        if len(records) >= args.num_samples:
            break
        sample = dataset[idx]
        if sample.get("pixel_goal") is None:
            skipped += 1
            continue
        if float(sample.get("is_stop", 0.0)) > 0.5:
            skipped += 1
            continue

        try:
            result = oracle_forward_one(
                model,
                sample,
                device,
                num_sample_trajs=num_sample_trajs,
                action_scale=action_scale,
                traj_image_size=traj_size,
                coord_order=coord_order,
            )
        except Exception as exc:
            LOGGER.exception("Failed idx=%s: %s", idx, exc)
            continue

        rec = {"dataset_index": idx, **result}
        records.append(rec)

        m = re.search(r"direct=([0-9.]+), path_len=([0-9.]+)", result["trajectory_summary"])
        if m:
            direct_list.append(float(m.group(1)))
            path_len_list.append(float(m.group(2)))

        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(records)
    zero_pat = sum(1 for r in records if r["zero_pad_pattern"])
    has_forward = sum(1 for r in records if r["forward_count"] > 0)
    path_ge_1 = sum(1 for p in path_len_list if p >= 1.0)
    path_ge_05 = sum(1 for p in path_len_list if p >= 0.5)

    print("\n===== Oracle Pixel-Goal Bridge Summary =====", flush=True)
    print(f"samples:           {n} (skipped non-coord/stop: {skipped})", flush=True)
    if path_len_list:
        print(f"path_len mean:     {np.mean(path_len_list):.3f} m", flush=True)
        print(f"path_len median:   {np.median(path_len_list):.3f} m", flush=True)
        print(f"direct mean:       {np.mean(direct_list):.3f} m", flush=True)
    if n:
        print(f"actions=[0,0,0,0]:  {zero_pat}/{n} ({100*zero_pat/n:.1f}%)", flush=True)
        print(f"any forward(1):    {has_forward}/{n} ({100*has_forward/n:.1f}%)", flush=True)
    print(f"path_len >= 0.5m:  {path_ge_05}/{n}", flush=True)
    print(f"path_len >= 1.0m:  {path_ge_1}/{n}", flush=True)
    action_hist = Counter()
    for r in records:
        for a in r["actions"][:4]:
            action_hist[a] += 1
    print(f"first-4 action hist: {dict(sorted(action_hist.items()))}", flush=True)
    print(f"jsonl:             {out_path}", flush=True)

    if n and np.mean(path_len_list) < 0.5:
        print(
            "\nVERDICT: Gold pixel_goal still yields short trajectories → "
            "likely Stage2 bridge / latent_queries / cond_projector / generate_latents path.",
            flush=True,
        )
    elif n and path_ge_1 >= n * 0.5:
        print(
            "\nVERDICT: Gold pixel_goal yields reasonable trajectories → "
            "closed-loop System2 (prompt/history/off-policy) is the main suspect.",
            flush=True,
        )
    else:
        print("\nVERDICT: Mixed / borderline — inspect per-sample jsonl.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

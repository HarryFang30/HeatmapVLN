#!/usr/bin/env python3
"""H7 verification: does the coord-condition shift student latents enough
to explain the ~36pt closed-loop SR gap?

For each sample:
  * latent_A: extract student traj_hidden_states with **teacher** coord wired
    into the collator prompt (training/offline distribution).
  * latent_B: extract student traj_hidden_states with the **student-generated**
    coord wired into the collator prompt (closed-loop distribution).
  * report cosine(A, B) plus the pixel distance between teacher_coord and
    student_coord.

Hypothesis H7 says adapter was trained on (A) but at eval sees (B). If A and B
are far apart in latent space, the adapter is OOD at closed-loop time.

Run roughly 30-60 samples for a 30-minute experiment on one GPU.
TEMPORARY DIAGNOSTIC SCRIPT; delete after the question is settled.
"""

from __future__ import annotations

import argparse
import copy
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
import torch.nn.functional as F
from PIL import Image

from scripts.training.train_pano_latent_adapter import (
    _copy_sample_for_collator,
    _extract_student_latents,
    _load_student_model,
    _load_teacher_records,
    _prepare_config,
    _sample_from_record,
)
from src.data.factory import build_trajectory_dataset
from src.models.heatmap.input_constructor import VIEW_NAMES, construct_input

LOGGER = logging.getLogger("verify_h7")

LEGACY_CONJUNCTIONS = [
    "you can see ",
    "in front of you is ",
    "there is ",
    "you can spot ",
    "you are toward the ",
    "ahead of you is ",
    "in your sight is ",
]
LOOKDOWN_TURN_TOKEN = "\u2193"


def _normalize_multimodal_inputs(inputs: dict[str, torch.Tensor]) -> None:
    if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None:
        vgt = inputs["video_grid_thw"]
        if vgt.shape[0] > 0 and vgt[:, 0].max() > 1:
            inputs["video_grid_thw"] = torch.repeat_interleave(vgt, vgt[:, 0], dim=0)
            inputs["video_grid_thw"][:, 0] = 1


def _tensor_chw_to_pil(t: torch.Tensor, resize: tuple[int, int] | None = None) -> Image.Image:
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    arr = (t.float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    if resize is not None:
        img = img.resize(resize)
    return img


def _parse_pixel_goal(llm_output: str, image_size: tuple[int, int]) -> list[int] | None:
    if not re.search(r"\d", llm_output):
        return None
    coord = [int(c) for c in re.findall(r"\d+", llm_output)]
    if len(coord) < 2:
        return None
    w, h = int(image_size[0]), int(image_size[1])
    u, v = int(coord[0]), int(coord[1])
    if not (0 <= u < w and 0 <= v < h):
        u = max(0, min(w - 1, u))
        v = max(0, min(h - 1, v))
    return [u, v]


def _run_vlm_once(
    model,
    processor,
    messages: list[dict[str, Any]],
    *,
    device: torch.device,
    max_new_tokens: int = 128,
) -> str:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    _normalize_multimodal_inputs(inputs)
    with torch.no_grad():
        outputs = model.qwen2_5_vl.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )
    output_ids = outputs.sequences
    return processor.tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )


def _generate_student_coord(
    model,
    processor,
    sample: dict[str, Any],
    *,
    image_size: tuple[int, int],
    device: torch.device,
    internnav_protocol: bool = True,
    max_new_tokens: int = 128,
) -> tuple[list[int] | None, dict[str, Any]]:
    """Run student VLM in closed-loop style (InternNav two-turn protocol).

    Turn 1: panorama -> expect either coord directly, or LOOKDOWN_TURN_TOKEN.
    Turn 2 (only if turn 1 returned LOOKDOWN_TURN_TOKEN): re-issue with lookdown
    frame appended -> expect coord.

    Returns ``(coord, info)`` where ``info`` contains both turns' raw text and
    which turn produced the final coord.
    """
    current_views_tensor = sample.get("current_views")
    history_panoramas_tensor = sample.get("history_panoramas")
    if current_views_tensor is None or history_panoramas_tensor is None:
        raise RuntimeError(
            "Sample lacks panoramic views; build dataset with panoramic_vlm_input=True"
        )

    current_views = {
        name: _tensor_chw_to_pil(current_views_tensor[i], resize=image_size)
        for i, name in enumerate(VIEW_NAMES)
    }
    history_panoramas = [
        {
            name: _tensor_chw_to_pil(history_panoramas_tensor[t, i], resize=image_size)
            for i, name in enumerate(VIEW_NAMES)
        }
        for t in range(history_panoramas_tensor.shape[0])
    ]

    instruction = str(sample.get("text", ""))

    base_messages = construct_input(
        current_views=current_views,
        history_panoramas=history_panoramas,
        instruction=instruction,
        pixel_goal=[0, 0],
        internnav_protocol=internnav_protocol,
    )
    base_messages = [m for m in base_messages if m["role"] != "assistant"]

    turn1_text = _run_vlm_once(
        model, processor, base_messages, device=device, max_new_tokens=max_new_tokens,
    )

    info: dict[str, Any] = {"turn1_text": turn1_text, "turn2_text": None, "used_turn": 1}

    coord = _parse_pixel_goal(turn1_text, image_size)
    if coord is not None:
        return coord, info

    if LOOKDOWN_TURN_TOKEN not in turn1_text:
        return None, info

    lookdown_tensor = sample.get("lookdown_frame")
    if lookdown_tensor is None:
        info["error"] = "no lookdown_frame in sample; cannot do turn 2"
        return None, info

    lookdown_pil = _tensor_chw_to_pil(lookdown_tensor, resize=image_size)

    messages_turn2 = copy.deepcopy(base_messages)
    messages_turn2.append({
        "role": "assistant",
        "content": [{"type": "text", "text": turn1_text}],
    })
    messages_turn2.append({
        "role": "user",
        "content": [
            {"type": "text", "text": random.choice(LEGACY_CONJUNCTIONS)},
            {"type": "image", "image": lookdown_pil},
        ],
    })

    turn2_text = _run_vlm_once(
        model, processor, messages_turn2, device=device, max_new_tokens=max_new_tokens,
    )
    info["turn2_text"] = turn2_text
    info["used_turn"] = 2

    coord = _parse_pixel_goal(turn2_text, image_size)
    return coord, info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/train_config_internnav_8gpu_stage2_wider.yaml")
    p.add_argument("--root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--teacher-jsonl", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--internnav-model-path", default=os.environ.get("INTERNNAV_MODEL_PATH", ""))
    p.add_argument("--index-mode", choices=["generic", "internnav_sft"], default="generic")
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--vlm-image-size", type=int, default=384)
    p.add_argument(
        "--output",
        default="logs/h7_verify.jsonl",
        help="Per-sample JSONL output.",
    )
    p.add_argument(
        "--internnav-protocol",
        action="store_true",
        default=True,
        help="Use InternNav-style coord prompt (matches closed-loop eval default).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = _prepare_config(args)

    teacher_jsonl = Path(args.teacher_jsonl).expanduser()
    records = _load_teacher_records(teacher_jsonl)
    if args.num_samples > 0:
        records = records[: args.num_samples]
    if not records:
        raise RuntimeError("No teacher records loaded")
    LOGGER.info("Loaded %d teacher records", len(records))

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
    n_traj_query = int(
        cfg.get("model", {}).get("action_head", {}).get("nextdit", {}).get("n_query", 4)
    )
    image_size = (int(args.vlm_image_size), int(args.vlm_image_size))

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cosines: list[float] = []
    coord_diffs: list[float] = []
    coord_failures = 0
    valid = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for i, rec in enumerate(records):
            try:
                sample = _sample_from_record(dataset, rec)
            except Exception as exc:
                LOGGER.warning("[%d] sample load failed: %s", i, exc)
                continue

            teacher_coord = list(rec["teacher"]["coord_uv"])

            try:
                student_coord, gen_info = _generate_student_coord(
                    student_model, processor, sample,
                    image_size=image_size, device=device,
                    internnav_protocol=args.internnav_protocol,
                )
            except Exception as exc:
                LOGGER.warning("[%d] coord generation failed: %s", i, exc)
                coord_failures += 1
                continue

            turn1_text = gen_info.get("turn1_text", "") or ""
            turn2_text = gen_info.get("turn2_text") or ""

            if student_coord is None:
                LOGGER.info(
                    "[%d] student VLM did not return a coord (turn1=%r turn2=%r)",
                    i, turn1_text[:60], turn2_text[:60],
                )
                coord_failures += 1
                continue

            with torch.no_grad():
                sample_A = _copy_sample_for_collator(sample, teacher_coord)
                latent_A = _extract_student_latents(
                    student_model, processor, [sample_A], device, n_traj_query
                )
                sample_B = _copy_sample_for_collator(sample, student_coord)
                latent_B = _extract_student_latents(
                    student_model, processor, [sample_B], device, n_traj_query
                )

            la = latent_A.float().flatten(1)
            lb = latent_B.float().flatten(1)
            cos = float(F.cosine_similarity(la, lb, dim=1).mean().item())
            coord_diff = float(np.linalg.norm(
                np.array(teacher_coord, dtype=np.float64)
                - np.array(student_coord, dtype=np.float64)
            ))

            cosines.append(cos)
            coord_diffs.append(coord_diff)
            valid += 1

            entry = {
                "i": i,
                "dataset_index": int(rec["dataset_index"]),
                "clip_idx": rec.get("clip_idx"),
                "current_t": rec.get("current_t"),
                "teacher_coord_uv": teacher_coord,
                "student_coord_uv": student_coord,
                "student_turn1_text": turn1_text,
                "student_turn2_text": turn2_text,
                "used_turn": gen_info.get("used_turn"),
                "coord_diff_pixels": coord_diff,
                "latent_A_norm": float(latent_A.float().norm().item()),
                "latent_B_norm": float(latent_B.float().norm().item()),
                "latent_A_B_cosine": cos,
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()

            LOGGER.info(
                "[%d/%d] idx=%s turn=%d teach=%s stud=%s diff=%.1fpx cos(A,B)=%.4f",
                i + 1, len(records), entry["dataset_index"],
                int(gen_info.get("used_turn", 0)),
                teacher_coord, student_coord, coord_diff, cos,
            )

    if not cosines:
        print("\nNo valid samples; aborting.", flush=True)
        return 1

    cos_arr = np.array(cosines)
    diff_arr = np.array(coord_diffs)

    print("\n" + "=" * 78)
    print("H7 verification summary")
    print("=" * 78)
    print(f"Valid samples:   {valid} / {len(records)}  (coord_failures: {coord_failures})")
    print()
    print("Latent A (teacher coord) vs B (student coord) cosine:")
    print(f"  mean              {cos_arr.mean():.4f}")
    print(f"  median            {np.median(cos_arr):.4f}")
    print(f"  min               {cos_arr.min():.4f}")
    print(f"  max               {cos_arr.max():.4f}")
    print(f"  std               {cos_arr.std():.4f}")
    print(f"  fraction < 0.99   {(cos_arr < 0.99).mean():.2%}")
    print(f"  fraction < 0.95   {(cos_arr < 0.95).mean():.2%}")
    print(f"  fraction < 0.90   {(cos_arr < 0.90).mean():.2%}")
    print()
    print("Coord diff (pixels) teacher vs student:")
    print(f"  mean              {diff_arr.mean():.1f}")
    print(f"  median            {np.median(diff_arr):.1f}")
    print(f"  max               {diff_arr.max():.1f}")
    print(f"  fraction within 5px:  {(diff_arr <= 5.0).mean():.2%}")
    print(f"  fraction within 20px: {(diff_arr <= 20.0).mean():.2%}")
    print()

    mean_cos = float(cos_arr.mean())
    if mean_cos >= 0.99:
        print(
            "VERDICT: H7 RULED OUT. Coord condition has negligible effect on latent. "
            "Adapter sees nearly identical inputs in training and closed-loop. "
            "The SR gap is from another source (closed-loop dynamics, NextDiT brittleness, etc)."
        )
    elif mean_cos >= 0.95:
        print(
            "VERDICT: H7 PARTIALLY HIT. Coord condition shifts latent ~%.2f cosine on average. "
            "This contributes to OOD at closed-loop time. Retraining adapter with "
            "student-generated coord may recover 5-15pt SR." % mean_cos
        )
    elif mean_cos >= 0.85:
        print(
            "VERDICT: H7 LIKELY CONFIRMED. Coord condition shifts latent meaningfully "
            "(avg cosine %.2f). Adapter is partially OOD at closed-loop time. "
            "Retraining with student-generated coord likely needed; expect 10-25pt SR gain." % mean_cos
        )
    else:
        print(
            "VERDICT: H7 STRONGLY CONFIRMED. Coord condition collapses latent quality "
            "(avg cosine %.2f). Adapter is heavily OOD at closed-loop time; this is the "
            "dominant root cause of the SR gap. Retraining required." % mean_cos
        )

    print(f"\nPer-sample JSONL: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

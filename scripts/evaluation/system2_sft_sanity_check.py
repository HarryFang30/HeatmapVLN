#!/usr/bin/env python
"""
Lightweight generation sanity check for Stage1-S2 panoramic System2 LoRA SFT.

This script intentionally evaluates autoregressive generation, not teacher-
forced LM loss.  It samples dataset items, builds the same panoramic prompt
without leaking the assistant target, runs Qwen generate, and reports whether
the output is a valid System2 target: pixel coordinate, STOP, or turn arrow.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from scripts.training.checkpoint import load_checkpoint_for_resume
from scripts.training.model_builder import build_model
from scripts.training.utils import load_config

from src.data.factory import build_dataset
from src.models.heatmap.input_constructor import (
    DIRECT_WAYPOINT_TASK_SUFFIX,
    INTERNAV_LOOKDOWN_TASK_SUFFIX,
    INTERNAV_TURN_TASK_SUFFIX,
    VIEW_NAMES,
    _ensure_pil,
    construct_input,
)

LOGGER = logging.getLogger("system2_sft_sanity")
TURN_TEXT = {
    0: "STOP",
    1: "↑",
    2: "←",
    3: "→",
    5: "↓",
}
ACTION_TEXT_TO_IDS = OrderedDict(
    {
        "STOP": [0],
        "↑": [1],
        "←": [2],
        "→": [3],
        "↓": [5],
    }
)
TARGET_PROMPT = DIRECT_WAYPOINT_TASK_SUFFIX + INTERNAV_TURN_TASK_SUFFIX
INTERNNAV_TARGET_PROMPT = INTERNAV_LOOKDOWN_TASK_SUFFIX


@dataclass
class ParsedText:
    kind: str
    text: str
    coord: list[int] | None = None
    coord_valid: bool | None = None
    action_seq: list[int] | None = None
    format_valid: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity-check Stage1-S2 panoramic System2 SFT generation.",
    )
    parser.add_argument(
        "--config",
        default="configs/train_system2_panoramic_sft_2gpu.yaml",
        help="Training config used by Stage1-S2.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Stage1-S2 checkpoint to evaluate.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to sample. The SFT config uses train by default.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Optional dataset root override.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Dataset epoch seed for random-subsequence resampling.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Generation length cap. Targets should be very short.",
    )
    parser.add_argument(
        "--coord-tolerance",
        type=float,
        default=15.0,
        help="Pixel L-infinity tolerance for coordinate target hits.",
    )
    parser.add_argument(
        "--focus",
        choices=("random", "subseq_end", "non_subseq_end"),
        default="random",
        help="Optionally focus random-subsequence endpoint samples.",
    )
    parser.add_argument(
        "--output",
        default="system2_sft_sanity.jsonl",
        help="JSONL file for per-sample generation records.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("target_instruction", "current_inference"),
        default="target_instruction",
        help=(
            "target_instruction uses the SFT waypoint prompt without the answer. "
            "current_inference mimics construct_input(..., pixel_goal=None)."
        ),
    )
    parser.add_argument(
        "--protocol",
        choices=("config", "direct", "internnav"),
        default="config",
        help=(
            "System2 protocol to sanity-check. 'internnav' expects pixel-goal "
            "samples to generate ↓ first, then coordinates after a lookdown image."
        ),
    )
    parser.add_argument(
        "--print-examples",
        type=int,
        default=12,
        help="Number of examples to print to stdout.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def resolve_checkpoint_path(path: str | None, cfg: dict[str, Any]) -> Path:
    if path:
        ckpt = Path(path)
    else:
        out_dir = (
            cfg.get("log", {}).get("out_dir")
            or cfg.get("logging", {}).get("out_dir")
        )
        if not out_dir:
            raise FileNotFoundError(
                "Checkpoint not provided and config.log.out_dir is missing."
            )
        ckpt = Path(out_dir) / "latest" / "checkpoints" / "latest.pth"
    if ckpt.exists():
        return ckpt
    best = ckpt.with_name("best.pth")
    if best.exists():
        LOGGER.warning("Checkpoint %s not found; using %s", ckpt, best)
        return best
    raise FileNotFoundError(f"Checkpoint not found: {ckpt}")


def prepare_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    if args.root:
        cfg["data"]["root"] = args.root

    # This script checks VLM text generation only.  Avoid building unused heads.
    cfg.setdefault("model", {}).setdefault("heatmap", {})["enable"] = False
    cfg.setdefault("model", {}).setdefault("action_head", {})["enable"] = False
    cfg.setdefault("model", {}).setdefault("llm", {})["gradient_checkpointing"] = False
    return cfg


def build_sanity_dataset(cfg: dict[str, Any], args: argparse.Namespace):
    traj_cfg = cfg.setdefault("data", {}).setdefault("trajectory", {})
    protocol = resolve_protocol(cfg, args)
    overrides = {
        "enable_augmentation": False,
        "enable_trajectory_augmentation": False,
        "compute_pixel_goal": True,
        "load_lookdown_for_system2": protocol == "internnav",
        "pixel_goal_direction": (
            traj_cfg.get("pixel_goal_direction", "front_down")
            if protocol == "internnav" else traj_cfg.get("pixel_goal_direction", "front")
        ),
        "require_sft_target": True,
        "sft_include_turns": traj_cfg.get("sft_include_turns", True),
        "sft_include_forward": traj_cfg.get("sft_include_forward", False),
        "panoramic_vlm_input": True,
        "load_traj_images": False,
    }
    dataset = build_dataset(cfg, split=args.split, **overrides)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(args.epoch)
    return dataset


def load_sft_model(cfg: dict[str, Any], checkpoint: Path, device: torch.device):
    LOGGER.info("Building model on %s", device)
    model = build_model(cfg, device=str(device), verbose=False, enable_action_head=False)
    model = model.to(device)

    # Qwen/LoRA is lazy; LoRA modules must exist before checkpoint loading.
    if hasattr(model, "qwen2_5_vl"):
        model.qwen2_5_vl._load_model()

    info = load_checkpoint_for_resume(str(checkpoint), model, logger=LOGGER)
    LOGGER.info(
        "Loaded checkpoint epoch=%s stage=%s metrics=%s",
        info.get("epoch"),
        info.get("stage_name"),
        info.get("metrics", {}),
    )
    model.eval()
    return model


def resolve_protocol(cfg: dict[str, Any], args: argparse.Namespace) -> str:
    if args.protocol != "config":
        return args.protocol
    return (
        cfg.get("data", {})
        .get("trajectory", {})
        .get("system2_sft_protocol", "direct")
    ).lower()


def target_texts_for_sample(
    sample: dict[str, Any],
    include_turns: bool,
    include_forward: bool,
    protocol: str,
) -> list[str]:
    discrete_action = int(sample.get("discrete_action", 1))
    if float(sample.get("is_stop", 0.0)) > 0.5 or discrete_action == 0:
        return ["STOP"]

    pixel_goal = sample.get("pixel_goal")
    if pixel_goal is not None:
        coord_text = f"{int(pixel_goal[0])} {int(pixel_goal[1])}"
        if protocol == "internnav":
            return ["↓", coord_text]
        return [coord_text]

    turn_action_text = sample.get("turn_action_text")
    if include_turns and isinstance(turn_action_text, str) and turn_action_text:
        return [turn_action_text]
    turn_actions = sample.get("turn_actions")
    if include_turns and isinstance(turn_actions, list) and turn_actions:
        return ["".join(TURN_TEXT.get(int(action_code), "") for action_code in turn_actions)]

    if include_turns and discrete_action in (2, 3, 5):
        return [TURN_TEXT[discrete_action]]
    if include_forward and discrete_action == 1:
        return [TURN_TEXT[1]]
    return []


def parse_target(text: str | None, image_size: tuple[int, int]) -> ParsedText:
    if text is None:
        return ParsedText(kind="none", text="")
    return parse_generated_text(text, image_size=image_size, target_mode=True)


def _parse_action_sequence(raw: str) -> tuple[list[int], bool]:
    compact = re.sub(r"[\s\t\r\n。.!！,，;；:：]+", "", raw or "")
    if not compact:
        return [], False
    pattern = re.compile("|".join(re.escape(token) for token in ACTION_TEXT_TO_IDS))
    matches = pattern.findall(compact)
    if not matches:
        return [], False
    reconstructed = "".join(matches)
    if reconstructed != compact:
        return [], False
    action_seq = []
    for token in matches:
        action_seq.extend(ACTION_TEXT_TO_IDS[token])
    return action_seq, True


def parse_generated_text(
    text: str,
    image_size: tuple[int, int],
    target_mode: bool = False,
) -> ParsedText:
    raw = (text or "").strip()
    upper = raw.upper()
    canonical = raw.strip(" \t\r\n。.!！,，;；:：")

    action_seq, action_seq_valid = _parse_action_sequence(raw)
    contains_stop = "STOP" in upper
    direction_hits = {
        "left": (
            "←" in raw
            or re.search(r"\bleft\b", raw, flags=re.I)
            or any(token in raw for token in ("向左", "往左", "左转", "转左"))
        ),
        "right": (
            "→" in raw
            or re.search(r"\bright\b", raw, flags=re.I)
            or any(token in raw for token in ("向右", "往右", "右转", "转右"))
        ),
        "down": (
            "↓" in raw
            or re.search(r"\bdown\b", raw, flags=re.I)
            or any(token in raw for token in ("向下", "往下", "低头", "下看"))
        ),
        "forward": (
            "↑" in raw
            or re.search(r"\bforward\b", raw, flags=re.I)
            or any(token in raw for token in ("向前", "往前", "前进"))
        ),
    }
    direction_kinds = [kind for kind, hit in direction_hits.items() if hit]

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", raw)
    has_numbers = bool(numbers)

    if canonical.upper() == "STOP":
        return ParsedText(kind="stop", text=raw, action_seq=[0], format_valid=True)

    if action_seq:
        if len(action_seq) == 1 and action_seq[0] in (1, 2, 3, 5):
            kind = {
                1: "forward",
                2: "left",
                3: "right",
                5: "down",
            }[action_seq[0]]
            return ParsedText(
                kind=kind,
                text=raw,
                action_seq=action_seq,
                format_valid=action_seq_valid,
            )
        return ParsedText(
            kind="action_seq",
            text=raw,
            action_seq=action_seq,
            format_valid=action_seq_valid,
        )

    if contains_stop:
        return ParsedText(kind="mixed", text=raw, action_seq=action_seq or None, format_valid=False)

    if len(direction_kinds) == 1 and not has_numbers:
        kind = direction_kinds[0]
        arrow = {"left": "←", "right": "→", "down": "↓", "forward": "↑"}[kind]
        strict = canonical == arrow
        action_map = {"left": [2], "right": [3], "down": [5], "forward": [1]}
        return ParsedText(kind=kind, text=raw, action_seq=action_map[kind], format_valid=strict)
    if direction_kinds:
        return ParsedText(kind="mixed", text=raw, action_seq=None, format_valid=False)

    if len(numbers) >= 2:
        x = round(float(numbers[0]))
        y = round(float(numbers[1]))
        width, height = image_size
        coord_valid = 0 <= x < width and 0 <= y < height
        strict = len(numbers) == 2
        return ParsedText(
            kind="coord",
            text=raw,
            coord=[x, y],
            coord_valid=coord_valid,
            action_seq=None,
            format_valid=strict,
        )

    return ParsedText(kind="invalid", text=raw, action_seq=None, format_valid=False)


def current_views_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        name: sample["current_views"][idx]
        for idx, name in enumerate(VIEW_NAMES)
    }


def history_panoramas_from_sample(sample: dict[str, Any]) -> list[dict[str, Any]]:
    history = sample["history_panoramas"]
    return [
        {
            name: history[hist_idx, view_idx]
            for view_idx, name in enumerate(VIEW_NAMES)
        }
        for hist_idx in range(history.shape[0])
    ]


def make_generation_messages(
    sample: dict[str, Any],
    prompt_mode: str,
    protocol: str,
) -> list[dict[str, Any]]:
    current_views = current_views_from_sample(sample)
    history_panoramas = history_panoramas_from_sample(sample)
    internnav_protocol = protocol == "internnav"

    if prompt_mode == "current_inference":
        return construct_input(
            current_views=current_views,
            history_panoramas=history_panoramas,
            instruction=sample.get("text"),
            pixel_goal=None,
            assistant_text=None,
            internnav_protocol=internnav_protocol,
        )

    # Passing a dummy pixel_goal makes construct_input include the target
    # instruction.  We then remove the assistant answer, so no label leaks.
    messages = construct_input(
        current_views=current_views,
        history_panoramas=history_panoramas,
        instruction=sample.get("text"),
        pixel_goal=[0, 0],
        assistant_text=None,
        internnav_protocol=internnav_protocol,
    )
    messages = [m for m in messages if m.get("role") != "assistant"]
    user_content = messages[0]["content"]
    target_prompt = INTERNNAV_TARGET_PROMPT if internnav_protocol else TARGET_PROMPT
    for item in reversed(user_content):
        if item.get("type") == "text":
            item["text"] = target_prompt
            break
    return messages


def make_second_turn_messages(
    sample: dict[str, Any],
    first_turn_text: str,
    protocol: str,
) -> list[dict[str, Any]]:
    messages = make_generation_messages(
        sample,
        prompt_mode="target_instruction",
        protocol=protocol,
    )
    lookdown_frame = sample.get("lookdown_frame")
    if lookdown_frame is None:
        raise RuntimeError("InternNav sanity check sample is missing lookdown_frame.")
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": first_turn_text}],
    })
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": random.choice(INTERNNAV_CONJUNCTIONS)},
            {"type": "image", "image": _ensure_pil(lookdown_frame)},
        ],
    })
    return messages


def normalize_multimodal_inputs(inputs: dict[str, torch.Tensor]) -> None:
    if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None:
        vgt = inputs["video_grid_thw"]
        if vgt.shape[0] > 0 and vgt[:, 0].max() > 1:
            inputs["video_grid_thw"] = torch.repeat_interleave(vgt, vgt[:, 0], dim=0)
            inputs["video_grid_thw"][:, 0] = 1


def move_inputs_to_device(inputs: Any, device: torch.device) -> dict[str, torch.Tensor]:
    if hasattr(inputs, "items"):
        items = inputs.items()
    else:
        raise TypeError(f"Unexpected processor output type: {type(inputs)}")

    moved = {}
    for key, value in items:
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    normalize_multimodal_inputs(moved)
    return moved


def generate_from_messages(
    model,
    messages: list[dict[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
) -> str:
    processor = model.qwen2_5_vl.processor
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = move_inputs_to_device(inputs, device)

    with torch.inference_mode():
        output = model.qwen2_5_vl.model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )
    sequences = output.sequences
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = sequences[:, prompt_len:]
    return processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def generate_one(
    model,
    sample: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    protocol: str,
) -> str:
    messages = make_generation_messages(
        sample,
        prompt_mode=args.prompt_mode,
        protocol=protocol,
    )
    return generate_from_messages(model, messages, device, args)


def image_size_from_cfg(cfg: dict[str, Any]) -> tuple[int, int]:
    size = cfg["data"].get("image_size", [256, 256])
    return int(size[0]), int(size[1])


def is_subsequence_end(dataset, idx: int) -> bool | None:
    sample_index = getattr(dataset, "sample_index", None)
    ranges = getattr(dataset, "_sample_subsequence_range", None)
    if sample_index is None or ranges is None or idx >= len(sample_index):
        return None
    _clip_idx, current_t = sample_index[idx]
    start_end = ranges.get(idx)
    if not start_end:
        return None
    _start, end = start_end
    return int(current_t) == int(end) - 1


def choose_indices(dataset, args: argparse.Namespace) -> list[int]:
    all_indices = list(range(len(dataset)))
    if args.focus != "random":
        want_end = args.focus == "subseq_end"
        all_indices = [
            idx for idx in all_indices
            if is_subsequence_end(dataset, idx) is want_end
        ]
    rng = random.Random(args.seed)
    rng.shuffle(all_indices)
    return all_indices[: min(args.num_samples, len(all_indices))]


def update_metrics(
    metrics: Counter,
    target: ParsedText,
    pred: ParsedText,
    coord_tolerance: float,
    *,
    include_turn_metrics: bool = True,
) -> dict[str, Any]:
    metrics["total"] += 1
    metrics[f"target_{target.kind}"] += 1
    metrics[f"pred_{pred.kind}"] += 1

    format_valid = bool(pred.format_valid)
    action_valid = format_valid and (pred.kind != "coord" or bool(pred.coord_valid))
    if format_valid:
        metrics["format_valid"] += 1
    if action_valid:
        metrics["action_valid"] += 1

    category_match = target.kind == pred.kind
    if category_match:
        metrics["category_match"] += 1

    coord_linf = None
    coord_l1 = None
    coord_hit = None
    if target.kind == "coord":
        metrics["coord_targets"] += 1
        if pred.kind == "coord" and pred.coord is not None and target.coord is not None:
            dx = abs(pred.coord[0] - target.coord[0])
            dy = abs(pred.coord[1] - target.coord[1])
            coord_linf = max(dx, dy)
            coord_l1 = dx + dy
            coord_hit = coord_linf <= coord_tolerance and bool(pred.coord_valid)
            metrics["coord_pred_on_coord_target"] += 1
            metrics["coord_l1_sum"] += coord_l1
            metrics["coord_linf_sum"] += coord_linf
            if coord_hit:
                metrics["coord_hit"] += 1

    if target.kind == "stop":
        metrics["stop_targets"] += 1
        if pred.kind == "stop":
            metrics["stop_hit"] += 1

    if include_turn_metrics and target.kind in {"left", "right", "down", "forward", "action_seq"}:
        metrics["turn_targets"] += 1
        if pred.action_seq is not None and target.action_seq is not None and pred.action_seq == target.action_seq:
            metrics["turn_hit"] += 1

    return {
        "format_valid": format_valid,
        "action_valid": action_valid,
        "category_match": category_match,
        "coord_linf": coord_linf,
        "coord_l1": coord_l1,
        "coord_hit": coord_hit,
        "pred_action_seq": pred.action_seq,
        "target_action_seq": target.action_seq,
    }


def print_summary(metrics: Counter, output_path: Path) -> None:
    total = max(int(metrics["total"]), 1)
    coord_targets = max(int(metrics["coord_targets"]), 1)
    stop_targets = max(int(metrics["stop_targets"]), 1)
    turn_targets = max(int(metrics["turn_targets"]), 1)
    coord_pred_count = max(int(metrics["coord_pred_on_coord_target"]), 1)

    def pct(name: str, denom: int = total) -> str:
        return f"{100.0 * float(metrics[name]) / max(denom, 1):.1f}%"

    print("\n===== Stage1-S2 System2 SFT Sanity Summary =====")
    print(f"records:            {int(metrics['total'])}")
    print(f"format_valid:       {pct('format_valid')}")
    print(f"action_valid:       {pct('action_valid')}")
    print(f"category_match:     {pct('category_match')}")
    print(f"coord_hit@tol:      {pct('coord_hit', coord_targets)}  ({int(metrics['coord_hit'])}/{int(metrics['coord_targets'])})")
    print(f"stop_hit:           {pct('stop_hit', stop_targets)}  ({int(metrics['stop_hit'])}/{int(metrics['stop_targets'])})")
    print(f"turn_hit:           {pct('turn_hit', turn_targets)}  ({int(metrics['turn_hit'])}/{int(metrics['turn_targets'])})")
    if metrics["first_down_targets"] > 0:
        print(
            f"first_down_hit:     {pct('first_down_hit', int(metrics['first_down_targets']))}  "
            f"({int(metrics['first_down_hit'])}/{int(metrics['first_down_targets'])})"
        )
    if metrics["second_coord_attempted"] > 0:
        print(
            f"second_coord_hit:   {pct('second_coord_hit', int(metrics['second_coord_attempted']))}  "
            f"({int(metrics['second_coord_hit'])}/{int(metrics['second_coord_attempted'])})"
        )
    if metrics["first_down_targets"] > 0:
        print(
            f"second_coord_overall:{pct('second_coord_hit', int(metrics['first_down_targets']))}  "
            f"({int(metrics['second_coord_hit'])}/{int(metrics['first_down_targets'])})"
        )
    if metrics["coord_pred_on_coord_target"] > 0:
        print(f"coord_mean_L1:      {float(metrics['coord_l1_sum']) / coord_pred_count:.2f}")
        print(f"coord_mean_Linf:    {float(metrics['coord_linf_sum']) / coord_pred_count:.2f}")
    if metrics["second_turn_skipped_bad_first"] > 0:
        print(f"second_skipped:     {int(metrics['second_turn_skipped_bad_first'])}")

    target_counts = {k.removeprefix("target_"): int(v) for k, v in metrics.items() if k.startswith("target_")}
    pred_counts = {k.removeprefix("pred_"): int(v) for k, v in metrics.items() if k.startswith("pred_")}
    print(f"target_counts:      {json.dumps(target_counts, ensure_ascii=False, sort_keys=True)}")
    print(f"pred_counts:        {json.dumps(pred_counts, ensure_ascii=False, sort_keys=True)}")
    print(f"jsonl:              {output_path}")


def main() -> None:
    configure_logging()
    args = parse_args()
    cfg = prepare_config(args)
    checkpoint = resolve_checkpoint_path(args.checkpoint, cfg)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    image_size = image_size_from_cfg(cfg)

    dataset = build_sanity_dataset(cfg, args)
    indices = choose_indices(dataset, args)
    protocol = resolve_protocol(cfg, args)
    LOGGER.info(
        "Dataset split=%s size=%d selected=%d focus=%s protocol=%s",
        args.split,
        len(dataset),
        len(indices),
        args.focus,
        protocol,
    )

    model = load_sft_model(cfg, checkpoint, device)
    include_turns = bool(cfg["data"].get("trajectory", {}).get("sft_include_turns", True))
    include_forward = bool(cfg["data"].get("trajectory", {}).get("sft_include_forward", False))

    metrics: Counter = Counter()
    printed = 0

    with output_path.open("w", encoding="utf-8") as f:
        for ordinal, idx in enumerate(indices, start=1):
            try:
                sample = dataset[idx]
                target_texts = target_texts_for_sample(
                    sample,
                    include_turns,
                    include_forward,
                    protocol,
                )
                if not target_texts:
                    metrics["skipped_no_target"] += 1
                    continue

                first_target_text = target_texts[0]
                pred_text = generate_one(model, sample, device, args, protocol)
                target = parse_target(first_target_text, image_size)
                pred = parse_generated_text(pred_text, image_size)
                is_internnav_pixel_goal = protocol == "internnav" and len(target_texts) > 1
                if is_internnav_pixel_goal:
                    metrics["first_down_targets"] += 1
                    if pred.action_seq == [5] and pred.format_valid:
                        metrics["first_down_hit"] += 1
                extra = update_metrics(
                    metrics,
                    target,
                    pred,
                    args.coord_tolerance,
                    include_turn_metrics=not is_internnav_pixel_goal,
                )

                record = {
                    "ordinal": ordinal,
                    "dataset_index": idx,
                    "turn": "first",
                    "protocol": protocol,
                    "is_requested_subsequence_end": is_subsequence_end(dataset, idx),
                    "instruction": sample.get("text", ""),
                    "target_text": first_target_text,
                    "target_texts": target_texts,
                    "target": asdict(target),
                    "prediction_text": pred_text,
                    "prediction": asdict(pred),
                    **extra,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

                if printed < args.print_examples:
                    status = "OK" if extra["action_valid"] else "BAD"
                    print(
                        f"[{ordinal:03d}] {status} "
                        f"target={first_target_text!r} pred={pred_text!r} "
                        f"kind={target.kind}->{pred.kind}"
                    )
                    printed += 1

                if is_internnav_pixel_goal:
                    if pred.kind != "down":
                        metrics["second_turn_skipped_bad_first"] += 1
                        f.write(json.dumps({
                            "ordinal": ordinal,
                            "dataset_index": idx,
                            "turn": "second",
                            "protocol": protocol,
                            "skipped": "first_turn_not_down",
                            "first_prediction_text": pred_text,
                        }, ensure_ascii=False) + "\n")
                        f.flush()
                        continue

                    second_messages = make_second_turn_messages(
                        sample,
                        first_turn_text=pred_text,
                        protocol=protocol,
                    )
                    second_pred_text = generate_from_messages(
                        model,
                        second_messages,
                        device,
                        args,
                    )
                    second_target_text = target_texts[1]
                    second_target = parse_target(second_target_text, image_size)
                    second_pred = parse_generated_text(second_pred_text, image_size)
                    metrics["second_coord_attempted"] += 1
                    second_extra = update_metrics(
                        metrics,
                        second_target,
                        second_pred,
                        args.coord_tolerance,
                        include_turn_metrics=False,
                    )
                    if second_extra["coord_hit"]:
                        metrics["second_coord_hit"] += 1
                    f.write(json.dumps({
                        "ordinal": ordinal,
                        "dataset_index": idx,
                        "turn": "second",
                        "protocol": protocol,
                        "target_text": second_target_text,
                        "target": asdict(second_target),
                        "prediction_text": second_pred_text,
                        "prediction": asdict(second_pred),
                        **second_extra,
                    }, ensure_ascii=False) + "\n")
                    f.flush()

                    if printed < args.print_examples:
                        status = "OK" if second_extra["action_valid"] else "BAD"
                        print(
                            f"[{ordinal:03d}.2] {status} "
                            f"target={second_target_text!r} pred={second_pred_text!r} "
                            f"kind={second_target.kind}->{second_pred.kind}"
                        )
                        printed += 1

            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:
                metrics["errors"] += 1
                error_record = {
                    "ordinal": ordinal,
                    "dataset_index": idx,
                    "error": repr(exc),
                }
                f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                f.flush()
                LOGGER.exception("Failed sample idx=%s", idx)
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if metrics["errors"]:
        LOGGER.warning("Encountered %d sample errors", int(metrics["errors"]))
    print_summary(metrics, output_path)


if __name__ == "__main__":
    main()

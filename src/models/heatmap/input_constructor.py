"""
Input Constructor for HeatmapVLN
==================================

Constructs text-guided multi-image input for Qwen2.5-VL.
Each panoramic position provides 4 views (front/right/back/left at 256x256).
Text annotations encode scene context, group structure, and spatial orientation.

Prompt wording follows ``NavPixelGoalDataset`` in InternNav
(``internnav/dataset/internvla_n1_lerobot_dataset.py``).
"""

from __future__ import annotations

import random
from typing import Union

import numpy as np
import torch
from PIL import Image

VIEW_NAMES = ["front", "right", "back", "left"]

# InternNav ``NavPixelGoalDataset.conjunctions``
INTERNAV_CONJUNCTIONS = [
    "you can see ",
    "in front of you is ",
    "there is ",
    "you can spot ",
    "you are toward the ",
    "ahead of you is ",
    "in your sight is ",
]

# Appendix A.1 / NavPixelGoalDataset user prompt (without <history> / <image> placeholders).
INTERNAV_BASE_PROMPT = (
    "You are an autonomous navigation assistant. Your task is to {instruction}. "
    "Where should you go next to stay on track? Please output the next waypoint's "
    "coordinates in the image. Please output STOP when you have successfully completed the task."
)

# VL-LN bench variant in ``vlln_lerobot_dataset.py`` (look-down + turn hints).
INTERNAV_LOOKDOWN_TASK_SUFFIX = (
    " When you want to output a waypoint you need to TILT DOWN (↓) by 30 degrees "
    "then output the next waypoint's coordinates in the look-down image. "
    "In case the next waypoint is out of view, utilize the turn actions: "
    "TURN LEFT (←) or TURN RIGHT (→) by 30 degrees."
)

INTERNAV_TURN_TASK_SUFFIX = (
    " In case the next waypoint is out of view, utilize the turn actions: "
    "TURN LEFT (←) or TURN RIGHT (→) by 30 degrees."
)

DIRECT_WAYPOINT_TASK_SUFFIX = (
    " Output the next waypoint coordinates in the front view of the current observation."
)

HISTORY_PROJECTION_TASK = (
    "Project each historical location into the current panoramic views."
)

_ANCHOR_TOKEN_CACHE: dict[tuple[int, int], list[list[int]]] = {}


def construct_input(
    current_views: dict[str, Union[Image.Image, torch.Tensor]],
    history_panoramas: list[dict[str, Union[Image.Image, torch.Tensor]]],
    instruction: str | None = None,
    pixel_goal: list[int] | None = None,
    assistant_text: str | None = None,
    lookdown_frame: Union[Image.Image, torch.Tensor, np.ndarray] | None = None,
    internnav_protocol: bool = False,
) -> list[dict]:
    """
    Construct text-annotated multi-image messages for Qwen2.5-VL.

    Panoramic layout is HeatmapVLN-specific; user-facing task text matches InternNav System2 SFT.
    """
    content: list[dict] = []
    instruction_text = instruction or ""
    has_history = len(history_panoramas) > 0

    prompt_text = INTERNAV_BASE_PROMPT.format(instruction=instruction_text)
    if has_history:
        prompt_text += " These are your historical observations:"
    content.append({"type": "text", "text": prompt_text})

    for hist_idx, hist in enumerate(history_panoramas):
        content.append({"type": "text", "text": _build_history_anchor_text(hist_idx)})
        for view_name in VIEW_NAMES:
            content.append({"type": "image", "image": _ensure_pil(hist[view_name])})

    content.append({
        "type": "text",
        "text": (
            f"Current panoramic observation "
            f"(views: {', '.join(VIEW_NAMES)}):"
        ),
    })
    for view_name in VIEW_NAMES:
        content.append({"type": "image", "image": _ensure_pil(current_views[view_name])})

    nav_target_text = assistant_text
    if nav_target_text is None and pixel_goal is not None:
        nav_target_text = f"{pixel_goal[0]} {pixel_goal[1]}"

    if nav_target_text is not None or pixel_goal is not None:
        if internnav_protocol:
            content.append({"type": "text", "text": INTERNAV_LOOKDOWN_TASK_SUFFIX})
        else:
            content.append({
                "type": "text",
                "text": DIRECT_WAYPOINT_TASK_SUFFIX + INTERNAV_TURN_TASK_SUFFIX,
            })
    else:
        content.append({"type": "text", "text": HISTORY_PROJECTION_TASK})

    messages = [{"role": "user", "content": content}]

    if nav_target_text is not None:
        if internnav_protocol and pixel_goal is not None:
            if lookdown_frame is None:
                return messages
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": "↓"}],
            })
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": random.choice(INTERNAV_CONJUNCTIONS)},
                    {"type": "image", "image": _ensure_pil(lookdown_frame)},
                ],
            })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": nav_target_text}],
        })

    return messages


def _ensure_pil(img: Union[Image.Image, torch.Tensor, np.ndarray]) -> Image.Image:
    """Convert tensor / ndarray to PIL Image if necessary."""
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu()
        if arr.dim() == 3 and arr.shape[0] in (1, 3):
            arr = arr.permute(1, 2, 0)
        arr = (arr.float().clamp(0, 1) * 255).byte().numpy()
        return Image.fromarray(arr)
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(img)
    raise TypeError(f"Unsupported image type: {type(img)}")


def find_text_anchor_positions(
    input_ids: torch.Tensor,
    tokenizer,
    num_history: int,
) -> dict[int, int]:
    """
    Locate the end token of each exact history-anchor annotation.

    After LLM attention, these token positions aggregate visual information
    from the following 4 images, serving as compact query vectors.
    """
    ids = input_ids.squeeze().tolist()

    anchors: dict[int, int] = {}
    i = 0
    cache_key = (id(tokenizer), num_history)
    anchor_token_ids = _ANCHOR_TOKEN_CACHE.get(cache_key)
    if anchor_token_ids is None:
        anchor_token_ids = []
        for hist_idx in range(num_history):
            anchor_text = _build_history_anchor_text(hist_idx)
            anchor_ids = tokenizer.encode(anchor_text, add_special_tokens=False)
            if not anchor_ids:
                raise RuntimeError(f"Failed to tokenize anchor text: {anchor_text}")
            anchor_token_ids.append(anchor_ids)
        _ANCHOR_TOKEN_CACHE[cache_key] = anchor_token_ids

    for hist_idx in range(num_history):
        anchor_ids = anchor_token_ids[hist_idx]

        found = False
        while i < len(ids):
            if _sublist_match(ids, i, anchor_ids):
                anchors[hist_idx] = i + len(anchor_ids) - 1
                i += len(anchor_ids)
                found = True
                break
            i += 1
        if not found:
            raise RuntimeError(
                f"Failed to locate history anchor {hist_idx + 1} in tokenized prompt. "
                "Prompt layout and tokenizer output are inconsistent."
            )

    return anchors


def construct_input_stage2(
    history_frames: list[Union[Image.Image, torch.Tensor]],
    current_frame: Union[Image.Image, torch.Tensor],
    lookdown_frame: Union[Image.Image, torch.Tensor],
    instruction: str | None = None,
    pixel_goal: list[int] | None = None,
    assistant_text: str | None = None,
) -> list[dict]:
    """Construct InternNav-aligned Stage 2 input (front-view + lookdown)."""
    all_frames = [_ensure_pil(f) for f in history_frames]
    all_frames.append(_ensure_pil(current_frame))

    instruction_text = instruction or ""
    prompt_text = INTERNAV_BASE_PROMPT.format(instruction=instruction_text)
    if len(history_frames) > 0:
        prompt_text += " These are your historical observations in the following video."

    user_content: list[dict] = [
        {"type": "text", "text": prompt_text},
        {"type": "video", "video": all_frames},
    ]
    messages = [{"role": "user", "content": user_content}]

    nav_target_text = assistant_text
    if nav_target_text is None and pixel_goal is not None:
        nav_target_text = f"{pixel_goal[0]} {pixel_goal[1]}"

    if pixel_goal is not None:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": "↓"}],
        })
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": random.choice(INTERNAV_CONJUNCTIONS)},
                {"type": "image", "image": _ensure_pil(lookdown_frame)},
            ],
        })

    if nav_target_text is not None:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": nav_target_text}],
        })

    return messages


def _build_history_anchor_text(hist_idx: int) -> str:
    return (
        f"Historical observation {hist_idx + 1} "
        f"(panoramic views: {', '.join(VIEW_NAMES)}):"
    )


def _sublist_match(seq: list, start: int, pattern: list) -> bool:
    if start + len(pattern) > len(seq):
        return False
    return seq[start : start + len(pattern)] == pattern

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
import re
from dataclasses import dataclass
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

STRUCTURED_PANO_OUTPUT_SUFFIX = (
    " Output the next waypoint using exactly two lines when applicable: "
    "`view: <front|right|back|left>` and `pixel: <u> <v>`. "
    "Output `view: stop` when you have completed the task. "
    "Output `view: turn` when the waypoint is not visible in any panoramic view."
)

HISTORY_PROJECTION_TASK = (
    "Project each historical location into the current panoramic views."
)

_ANCHOR_TOKEN_CACHE: dict[tuple[int, int], list[list[int]]] = {}


def format_structured_pano_assistant_text(
    pano_view_id: str | None,
    pano_pixel_goal: list[int] | None,
    *,
    sample_kind: str | None = None,
    is_stop: bool = False,
) -> str | None:
    """Build Stage1-S2 structured assistant target text."""
    if is_stop or sample_kind == "stop" or pano_view_id == "view_stop":
        return "view: stop"
    if pano_pixel_goal is not None and pano_view_id in VIEW_NAMES:
        u, v = int(pano_pixel_goal[0]), int(pano_pixel_goal[1])
        return f"view: {pano_view_id}\npixel: {u} {v}"
    if sample_kind in {"turn_left", "turn_right"}:
        return f"view: {sample_kind}"
    if pano_view_id in {"view_turn_left", "view_turn_right"}:
        direction = pano_view_id.replace("view_turn_", "")
        return f"view: turn_{direction}"
    if sample_kind == "turn" or pano_view_id == "view_turn":
        return "view: turn"
    return None


@dataclass(frozen=True)
class StructuredPanoParseResult:
    kind: str  # "pixel", "turn_left", "turn_right", "turn", "stop", "legacy_coord", "invalid"
    view_id: str | None = None
    pixel_goal: list[int] | None = None
    turn_direction: str | None = None  # "left" or "right" when kind is turn_left/turn_right


_CLAMP_WARNED = False


def parse_structured_pano_output(
    text: str,
    image_size: tuple[int, int] | None = None,
) -> StructuredPanoParseResult:
    """Parse Stage1-S2 structured output or fall back to legacy ``u v`` coords."""
    if not text or not str(text).strip():
        return StructuredPanoParseResult(kind="invalid")

    # Recognise turn_left / turn_right as well as the legacy ambiguous "turn".
    view_match = re.search(
        r"\bview\s*:\s*(front|right|back|left|stop|turn_left|turn_right|turn)\b",
        text,
        flags=re.I,
    )
    if view_match is not None:
        view = view_match.group(1).lower()
        if view == "stop":
            return StructuredPanoParseResult(kind="stop")
        if view == "turn_left":
            return StructuredPanoParseResult(kind="turn", turn_direction="left")
        if view == "turn_right":
            return StructuredPanoParseResult(kind="turn", turn_direction="right")
        if view == "turn":
            return StructuredPanoParseResult(kind="turn")
        pixel_match = re.search(r"\bpixel\s*:\s*(\d+)\s+(\d+)\b", text, flags=re.I)
        if pixel_match is not None:
            u, v = int(pixel_match.group(1)), int(pixel_match.group(2))
            u, v = _clamp_coord(u, v, image_size)
            return StructuredPanoParseResult(
                kind="pixel",
                view_id=view,
                pixel_goal=[u, v],
            )
        return StructuredPanoParseResult(kind="invalid", view_id=view)

    if re.search(r"\bSTOP\b", text, flags=re.I):
        return StructuredPanoParseResult(kind="stop")

    if re.search(r"\d", text):
        nums = [int(c) for c in re.findall(r"\d+", text)]
        if len(nums) >= 2:
            u, v = nums[0], nums[1]
            u, v = _clamp_coord(u, v, image_size)
            return StructuredPanoParseResult(
                kind="legacy_coord",
                view_id="front",
                pixel_goal=[u, v],
            )
    return StructuredPanoParseResult(kind="invalid")


def _clamp_coord(
    u: int, v: int, image_size: tuple[int, int] | None
) -> tuple[int, int]:
    """Clamp (u, v) to image bounds, warning once if clamping occurred."""
    global _CLAMP_WARNED
    if image_size is None:
        return u, v
    w, h = int(image_size[0]), int(image_size[1])
    cu, cv = u, v
    u = max(0, min(w - 1, u))
    v = max(0, min(h - 1, v))
    if (cu != u or cv != v) and not _CLAMP_WARNED:
        import logging
        _logger = logging.getLogger(__name__)
        _logger.warning(
            "Clamped pixel_goal from [%d, %d] to [%d, %d] for image_size=%s",
            cu, cv, u, v, (w, h),
        )
        _CLAMP_WARNED = True
    return u, v


def vlm_output_requests_stop(text: str) -> bool:
    parsed = parse_structured_pano_output(text, image_size=None)
    if parsed.kind == "stop":
        return True
    return bool(re.search(r"\bSTOP\b", text or "", flags=re.I))


def vlm_output_requests_turn(text: str) -> str | None:
    """Return turn direction ("left" / "right") if the output is a turn request.

    Returns ``None`` when the output is not a turn.
    When the structured output uses the ambiguous ``view: turn`` (no direction),
    returns ``None`` so the caller can fall back to ground-truth inference.
    """
    parsed = parse_structured_pano_output(text, image_size=None)
    if parsed.kind == "turn":
        return parsed.turn_direction  # may be None for ambiguous "view: turn"
    return None


def structured_condition_text(
    view_id: str,
    pixel_goal: list[int],
) -> str:
    """Canonical structured assistant text for System1 latent conditioning."""
    text = format_structured_pano_assistant_text(view_id, pixel_goal)
    if text is None:
        raise ValueError(f"Cannot build structured condition for view={view_id} pixel={pixel_goal}")
    return text


def construct_input(
    current_views: dict[str, Union[Image.Image, torch.Tensor]],
    history_panoramas: list[dict[str, Union[Image.Image, torch.Tensor]]],
    instruction: str | None = None,
    pixel_goal: list[int] | None = None,
    assistant_text: str | None = None,
    lookdown_frame: Union[Image.Image, torch.Tensor, np.ndarray] | None = None,
    internnav_protocol: bool = False,
    structured_pano_output: bool = False,
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
        if structured_pano_output:
            content.append({"type": "text", "text": STRUCTURED_PANO_OUTPUT_SUFFIX})
        elif internnav_protocol:
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
        if (
            internnav_protocol
            and pixel_goal is not None
            and not structured_pano_output
        ):
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
        {"type": "video", "video": all_frames, "nframes": len(all_frames)},
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

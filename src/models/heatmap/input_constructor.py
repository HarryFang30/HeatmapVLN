"""
Input Constructor for HeatmapVLN
==================================

Constructs text-guided multi-image input for Qwen2.5-VL.
Each panoramic position provides 4 views (front/right/back/left at 256x256).
Text annotations encode scene context, group structure, and spatial orientation.

Reference: HeatmapVLN设计文档 Section 3
"""

from typing import Union

import numpy as np
import torch
from PIL import Image

VIEW_NAMES = ["front", "right", "back", "left"]
VIEW_ANGLES = ["0°正前方", "90°右侧", "180°正后方", "270°左侧"]
ORIENTATION_STR = "、".join(VIEW_ANGLES)
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

    Args:
        current_views: dict with keys 'front', 'right', 'back', 'left',
            values are PIL Images or tensors (C,H,W) in [0,1].
        history_panoramas: list of dicts with same structure, ordered by time.
        instruction: optional navigation instruction for task grounding.
        pixel_goal: optional [x, y] pixel coordinates of the next waypoint
            in the front view.
        assistant_text: optional explicit assistant response.  When omitted
            and ``pixel_goal`` is provided, the response defaults to
            ``"{x} {y}"``.
        lookdown_frame: optional current lookdown observation.  Required for
            teacher-forced pixel-goal samples when ``internnav_protocol`` is
            enabled.
        internnav_protocol: when true, pixel-goal samples use the InternNav
            two-turn protocol: assistant ``↓``, user lookdown image,
            assistant coordinates.

    Returns:
        messages: list of message dicts compatible with the Qwen2.5-VL processor.
    """
    content = []

    content.append({
        "type": "text",
        "text": "以下是一个室内导航场景。",
    })
    if instruction:
        content.append({
            "type": "text",
            "text": f"导航指令：{instruction}",
        })
    content.append({
        "type": "text",
        "text": f"当前位置的全景观测（朝向{ORIENTATION_STR}）：",
    })
    for view_name in VIEW_NAMES:
        img = _ensure_pil(current_views[view_name])
        content.append({"type": "image", "image": img})

    for i, hist in enumerate(history_panoramas):
        content.append({
            "type": "text",
            "text": _build_history_anchor_text(i),
        })
        for view_name in VIEW_NAMES:
            img = _ensure_pil(hist[view_name])
            content.append({"type": "image", "image": img})

    nav_target_text = assistant_text
    if nav_target_text is None and pixel_goal is not None:
        nav_target_text = f"{pixel_goal[0]} {pixel_goal[1]}"

    if nav_target_text is not None or pixel_goal is not None:
        if internnav_protocol:
            prompt_text = (
                "判断每个历史位置在当前视图中的投影位置。"
                "如果已经完成导航，请输出 STOP；如果目标不在前视图中，"
                "请输出 ← 或 → 调整朝向；如果需要在前视图中定位下一个导航目标，"
                "请先输出 ↓，收到下视图后再输出下视图中的像素坐标。"
            )
        else:
            prompt_text = (
                "判断每个历史位置在当前视图中的投影位置，"
                "并输出下一个导航目标在前视图中的像素坐标。"
                "如果已经完成导航，请输出 STOP；如果目标不在前视图中，"
                "请输出 ← 或 → 调整朝向。"
            )
        content.append({
            "type": "text",
            "text": prompt_text,
        })
    else:
        content.append({
            "type": "text",
            "text": "判断每个历史位置在当前视图中的投影位置。",
        })

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
                "content": [{"type": "image", "image": _ensure_pil(lookdown_frame)}],
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

    Returns:
        dict mapping history_index -> token position in the sequence.
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
    """Construct InternNav-aligned Stage 2 input (front-view + lookdown).

    Pixel-goal samples mirror the InternNav paper::

        User:   [video: K past + current front frames] + instruction
        Assistant: ↓
        User:   [lookdown image]
        Assistant: (x, y)   ← pixel-goal coordinates (teacher forcing)

    STOP / turn samples use a single assistant response without the lookdown
    turn, matching InternNav's non-pixel-goal branches.

    Args:
        history_frames: K front-view history images.
        current_frame:  Current front-view observation.
        lookdown_frame: Current lookdown (pitch=30°) observation.
        instruction:    Navigation instruction text.
        pixel_goal:     [x, y] pixel coordinates of next waypoint.
        assistant_text: optional explicit assistant response.  Defaults to
            ``"{x} {y}"`` when ``pixel_goal`` is provided.

    Returns:
        messages: list of message dicts for the Qwen2.5-VL processor.
    """
    all_frames = [_ensure_pil(f) for f in history_frames]
    all_frames.append(_ensure_pil(current_frame))

    user_content: list = []
    if instruction:
        user_content.append({"type": "text", "text": instruction})
    user_content.append({"type": "video", "video": all_frames})

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
            "content": [{"type": "image", "image": _ensure_pil(lookdown_frame)}],
        })

    if nav_target_text is not None:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": nav_target_text}],
        })

    return messages


def _build_history_anchor_text(hist_idx: int) -> str:
    return f"历史位置{hist_idx + 1}的全景观测（朝向{ORIENTATION_STR}）："


def _sublist_match(seq: list, start: int, pattern: list) -> bool:
    if start + len(pattern) > len(seq):
        return False
    return seq[start : start + len(pattern)] == pattern

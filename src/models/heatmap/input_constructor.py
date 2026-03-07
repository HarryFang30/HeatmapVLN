"""
Input Constructor for HeatmapVLN
==================================

Constructs text-guided multi-image input for Qwen3.5-9B.
Each panoramic position provides 4 views (front/right/back/left at 256x256).
Text annotations encode scene context, group structure, and spatial orientation.

Reference: HeatmapVLN设计文档 Section 3
"""

from typing import Dict, List, Optional, Union
from PIL import Image
import torch
import numpy as np


VIEW_NAMES = ["front", "right", "back", "left"]
VIEW_ANGLES = ["0°正前方", "90°右侧", "180°正后方", "270°左侧"]
ORIENTATION_STR = "、".join(VIEW_ANGLES)


def construct_input(
    current_views: Dict[str, Union[Image.Image, torch.Tensor]],
    history_panoramas: List[Dict[str, Union[Image.Image, torch.Tensor]]],
    instruction: Optional[str] = None,
) -> List[Dict]:
    """
    Construct text-annotated multi-image messages for Qwen3.5.

    Args:
        current_views: dict with keys 'front', 'right', 'back', 'left',
            values are PIL Images or tensors (C,H,W) in [0,1].
        history_panoramas: list of dicts with same structure, ordered by time.
        instruction: optional navigation instruction for task grounding.

    Returns:
        messages: list of message dicts compatible with Qwen3.5 processor.
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

    content.append({
        "type": "text",
        "text": "判断每个历史位置在当前视图中的投影位置。",
    })

    messages = [{"role": "user", "content": content}]
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
) -> Dict[int, int]:
    """
    Locate the end token of each exact history-anchor annotation.

    After LLM attention, these token positions aggregate visual information
    from the following 4 images, serving as compact query vectors.

    Returns:
        dict mapping history_index -> token position in the sequence.
    """
    ids = input_ids.squeeze().tolist()

    anchors: Dict[int, int] = {}
    i = 0
    for hist_idx in range(num_history):
        anchor_text = _build_history_anchor_text(hist_idx)
        anchor_ids = tokenizer.encode(anchor_text, add_special_tokens=False)
        if not anchor_ids:
            raise RuntimeError(f"Failed to tokenize anchor text: {anchor_text}")

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


def _build_history_anchor_text(hist_idx: int) -> str:
    return f"历史位置{hist_idx + 1}的全景观测（朝向{ORIENTATION_STR}）："


def _sublist_match(seq: list, start: int, pattern: list) -> bool:
    if start + len(pattern) > len(seq):
        return False
    return seq[start : start + len(pattern)] == pattern

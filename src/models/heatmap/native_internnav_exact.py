"""Byte-exact port of the certified native InternNav System2 front end.

The 62.5%-SR R2R val-unseen baseline was produced by the locked replica stack
in ``evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802/tools/
rpc_internnav_native_server.py``.  The v3 evaluation stack instead built its
System2 prompts with the training-side ``construct_input_stage2`` and its own
output parsing, which diverges from the replica in six measured ways (384x384
conversational lookdown instead of 640x480, per-image ``"\\n"`` separators,
a doubled instruction period, single-step turn execution, no ``↑``/mixed
arrow handling, and lookdown-request fallbacks that stop the episode).  Under
the frozen System2 those token- and image-level differences killed 41% of
episodes outright in both PPA evaluations.

This module ports the replica's prompt construction and action parsing
verbatim so the v3 server can drive System2 byte-identically to the certified
baseline.  It must never be used by training code: the training prompt
contract stays in ``construct_input_stage2``.
"""

from __future__ import annotations

import copy
import re
from collections import OrderedDict

from PIL import Image


DEFAULT_IMAGE_TOKEN = "<image>"

NATIVE_PROMPT = (
    "You are an autonomous navigation assistant. Your task is to <instruction>. "
    "Where should you go next to stay on track? Please output the next waypoint's "
    "coordinates in the image. Please output STOP when you have successfully completed the task."
)
NATIVE_CONJUNCTION = "you can see "
NATIVE_LOOKDOWN_SIZE = (640, 480)
NATIVE_MAX_STEPS = 8
NATIVE_MAX_LOCAL_STEPS = 4

NATIVE_ACTION_STOP = 0
NATIVE_ACTION_FORWARD = 1
NATIVE_ACTION_LEFT = 2
NATIVE_ACTION_RIGHT = 3
NATIVE_ACTION_LOOKDOWN = 5

NATIVE_ACTIONS = OrderedDict(
    {
        "STOP": [NATIVE_ACTION_STOP],
        "↑": [NATIVE_ACTION_FORWARD],
        "←": [NATIVE_ACTION_LEFT],
        "→": [NATIVE_ACTION_RIGHT],
        "↓": [NATIVE_ACTION_LOOKDOWN],
    }
)


def _split_and_clean(text: str) -> list[str]:
    parts = re.split(r"(<image>)", text)
    result: list[str] = []
    for part in parts:
        if part == DEFAULT_IMAGE_TOKEN:
            result.append(part)
        else:
            cleaned = part.replace("\n", "").strip()
            if cleaned:
                result.append(cleaned)
    return result


def _content_from_prompt(
    prompt: str,
    images: list[Image.Image],
    start_index: int = 0,
) -> tuple[list[dict], int]:
    content: list[dict] = []
    image_index = start_index
    for part in _split_and_clean(prompt):
        if part == DEFAULT_IMAGE_TOKEN:
            if not 0 <= image_index < len(images):
                raise RuntimeError(
                    "Native prompt/image mismatch: requested index "
                    f"{image_index}, images={len(images)}"
                )
            content.append({"type": "image", "image": images[image_index]})
            image_index += 1
        else:
            content.append({"type": "text", "text": part})
    return content, image_index


def build_native_messages(
    instruction: str,
    history_front: list[Image.Image],
    current_front: Image.Image,
) -> tuple[list[dict], list[Image.Image]]:
    """Reproduce InternNav's front-only R2R System2 prompt deterministically."""
    prompt = NATIVE_PROMPT.replace("<instruction>.", instruction)
    if history_front:
        placeholders = (DEFAULT_IMAGE_TOKEN + "\n") * len(history_front)
        prompt += f" These are your historical observations: {placeholders}."
    prompt += f" {NATIVE_CONJUNCTION}{DEFAULT_IMAGE_TOKEN}."
    images = list(history_front) + [current_front]
    content, consumed = _content_from_prompt(prompt, images)
    if consumed != len(images):
        raise RuntimeError(
            f"Native prompt consumed {consumed}/{len(images)} images"
        )
    return [{"role": "user", "content": content}], images


def append_native_lookdown_turn(
    messages: list[dict],
    images: list[Image.Image],
    first_output: str,
    lookdown: Image.Image,
) -> tuple[list[dict], list[Image.Image]]:
    """Reproduce the official second conversational turn after a down-arrow."""
    messages = copy.deepcopy(messages)
    messages.append(
        {"role": "assistant", "content": [{"type": "text", "text": first_output}]}
    )
    images = list(images) + [lookdown]
    content, consumed = _content_from_prompt(
        f"{NATIVE_CONJUNCTION}{DEFAULT_IMAGE_TOKEN}.",
        images,
        start_index=len(images) - 1,
    )
    if consumed != len(images):
        raise RuntimeError(
            f"Native lookdown turn consumed {consumed}/{len(images)} images"
        )
    messages.append({"role": "user", "content": content})
    return messages, images


def parse_native_actions(output: str) -> list[int]:
    pattern = "|".join(re.escape(token) for token in NATIVE_ACTIONS)
    matches = re.findall(pattern, output or "")
    return [action for token in matches for action in NATIVE_ACTIONS[token]]


def finalize_native_local_actions(actions: list[int]) -> list[int]:
    """Pad to eight steps with STOP, then cap to the four-step local queue."""
    actions = list(actions)
    if len(actions) < NATIVE_MAX_STEPS:
        actions += [NATIVE_ACTION_STOP] * (NATIVE_MAX_STEPS - len(actions))
    return [int(action) for action in actions[:NATIVE_MAX_LOCAL_STEPS]]


def native_requests_lookdown(llm_output: str) -> bool:
    """First-turn LOOKDOWN request: no digits and the first action is ``↓``."""
    if re.search(r"\d", llm_output or ""):
        return False
    return parse_native_actions(llm_output)[:1] == [NATIVE_ACTION_LOOKDOWN]


__all__ = [
    "DEFAULT_IMAGE_TOKEN",
    "NATIVE_ACTIONS",
    "NATIVE_ACTION_FORWARD",
    "NATIVE_ACTION_LEFT",
    "NATIVE_ACTION_LOOKDOWN",
    "NATIVE_ACTION_RIGHT",
    "NATIVE_ACTION_STOP",
    "NATIVE_CONJUNCTION",
    "NATIVE_LOOKDOWN_SIZE",
    "NATIVE_MAX_LOCAL_STEPS",
    "NATIVE_MAX_STEPS",
    "NATIVE_PROMPT",
    "append_native_lookdown_turn",
    "build_native_messages",
    "finalize_native_local_actions",
    "native_requests_lookdown",
    "parse_native_actions",
]

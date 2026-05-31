from types import SimpleNamespace

import pytest
import torch

from src.data.pano_teacher_alignment import (
    append_structured_pano_suffix,
    compute_aligned_teacher_latents_768_batch,
    has_structured_pano_pixel_goal,
    structured_assistant_from_sample,
)


def test_has_structured_pano_pixel_goal():
    sample = {
        "pano_sample_kind": "pixel",
        "pano_view_id": "right",
        "pano_pixel_goal": [211, 128],
    }
    assert has_structured_pano_pixel_goal(sample)
    assert structured_assistant_from_sample(sample) == "view: right\npixel: 211 128"


def test_append_structured_pano_suffix():
    messages = [{"role": "user", "content": [{"type": "text", "text": "Navigate."}]}]
    updated = append_structured_pano_suffix(messages)
    assert "view:" in updated[0]["content"][0]["text"]
    assert messages[0]["content"][0]["text"] == "Navigate."


def test_aligned_teacher_batch_rejects_invalid_sample():
    with pytest.raises(RuntimeError, match="without a structured pano pixel goal"):
        compute_aligned_teacher_latents_768_batch(
            teacher_model=None,
            processor=None,
            samples=[{}],
            device=torch.device("cpu"),
            turn_args=SimpleNamespace(seed=42),
        )

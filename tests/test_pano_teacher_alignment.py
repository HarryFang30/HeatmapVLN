import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.data.pano_teacher_alignment import (
    NATIVE_TEACHER_ALIGNMENT_VERSION,
    NATIVE_TEACHER_SIDECAR_SCHEMA,
    aligned_native_sidecar_contract,
    append_structured_pano_suffix,
    compute_aligned_teacher_latents_768_batch,
    has_structured_pano_pixel_goal,
    structured_assistant_from_sample,
    sidecar_alignment_metadata,
    validate_aligned_native_sidecar_contract_fields,
)


def test_aligned_native_contract_binds_the_same_goal_frame_and_yx_text():
    sample = {
        "pano_sample_kind": "pixel",
        "pano_view_id": "front",
        "pano_pixel_goal": [123, 111],
        "pano_pixel_goal_relative_len": 7,
        "pano_goal_frame_idx": 12,
        "aligned_native_pixel_goal_uv": [151, 202],
        "aligned_native_goal_frame_idx": 12,
        "aligned_native_visible": True,
    }
    contract = aligned_native_sidecar_contract(
        sample,
        stable_sample_key="scene/clip|t=5",
        current_t=5,
    )

    assert contract["sidecar_schema"] == NATIVE_TEACHER_SIDECAR_SCHEMA
    assert contract["alignment_version"] == NATIVE_TEACHER_ALIGNMENT_VERSION
    assert len(contract["alignment_fingerprint"]) == 64
    assert contract["alignment_contract"]["goal_frame_idx"] == 12
    assert contract["alignment_contract"]["native_goal"]["pixel_uv"] == [151, 202]
    assert contract["alignment_contract"]["native_goal"]["text_yx"] == [202, 151]


def test_aligned_native_contract_rejects_different_goal_frames():
    sample = {
        "pano_sample_kind": "pixel",
        "pano_view_id": "front",
        "pano_pixel_goal": [123, 111],
        "pano_pixel_goal_relative_len": 7,
        "pano_goal_frame_idx": 12,
        "aligned_native_pixel_goal_uv": [151, 202],
        "aligned_native_goal_frame_idx": 11,
    }
    with pytest.raises(ValueError, match="goal-frame mismatch"):
        aligned_native_sidecar_contract(
            sample,
            stable_sample_key="scene/clip|t=5",
            current_t=5,
        )


def test_native_contract_field_validator_binds_teacher_and_dataset_labels():
    sample = {
        "pano_sample_kind": "pixel",
        "pano_view_id": "front",
        "pano_pixel_goal": [123, 111],
        "pano_pixel_goal_relative_len": 7,
        "pano_goal_frame_idx": 12,
        "aligned_native_pixel_goal_uv": [151, 202],
        "aligned_native_goal_frame_idx": 12,
        "aligned_native_visible": True,
    }
    stable_key = "scene/clip|t=5"
    record = {
        "stable_sample_key": stable_key,
        "current_t": 5,
        **aligned_native_sidecar_contract(
            sample,
            stable_sample_key=stable_key,
            current_t=5,
        ),
        "teacher": {
            "coord_uv": [151, 202],
            "internnav_pixel_goal_yx": [202, 151],
            "conditioned_coord_text": "202 151",
            "pano_view_id": "front",
            "goal_frame_idx": 12,
        },
        "dataset_label": sidecar_alignment_metadata(sample),
    }

    contract = validate_aligned_native_sidecar_contract_fields(record)
    assert contract["stable_sample_key"] == stable_key

    broken = copy.deepcopy(record)
    broken["teacher"]["coord_uv"] = [202, 151]
    with pytest.raises(ValueError, match="teacher.coord_uv"):
        validate_aligned_native_sidecar_contract_fields(broken)

    broken = copy.deepcopy(record)
    broken["dataset_label"]["pano_goal_frame_idx"] = 13
    with pytest.raises(ValueError, match="pano goal frame"):
        validate_aligned_native_sidecar_contract_fields(broken)


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


def test_aligned_teacher_batch_keeps_full_context_and_runs_teacher_sequentially(monkeypatch):
    from scripts.evaluation import collect_internnav_teacher_sidecar as sidecar

    def build_first_turn(sample, _turn_args, _rng):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Navigate sample {sample['sample_id']}."}],
            }
        ]
        return messages, [object()]

    monkeypatch.setattr(sidecar, "_build_first_turn", build_first_turn)

    class Inputs:
        def __init__(self, sample_id):
            self.input_ids = torch.tensor([[101, sample_id, 202]])
            self.pixel_values = torch.zeros(1)
            self.image_grid_thw = torch.tensor([[1, 1, 1]])

        def to(self, _device):
            return self

    class Processor:
        def apply_chat_template(self, messages, **_kwargs):
            return " ".join(
                item["text"]
                for message in messages
                for item in message["content"]
                if item["type"] == "text"
            )

        def __call__(self, *, text, images, **_kwargs):
            assert len(text) == 1
            assert len(images) == 1
            assert "Navigate sample" in text[0]
            sample_id = 1 if "sample 1." in text[0] else 2
            return Inputs(sample_id)

    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.cond_projector = nn.Linear(1, 1, bias=False)
            nn.init.ones_(self.cond_projector.weight)
            self.calls = []

        def generate_latents(self, input_ids, _pixel_values, _image_grid_thw):
            self.calls.append(input_ids.clone())
            return input_ids[:, -1:].float().unsqueeze(-1)

    teacher = Teacher()
    samples = [
        {
            "sample_id": 1,
            "pano_sample_kind": "pixel",
            "pano_view_id": "front",
            "pano_pixel_goal": [10, 20],
        },
        {
            "sample_id": 2,
            "pano_sample_kind": "pixel",
            "pano_view_id": "right",
            "pano_pixel_goal": [30, 40],
        },
    ]
    latents = compute_aligned_teacher_latents_768_batch(
        teacher_model=teacher,
        processor=Processor(),
        samples=samples,
        device=torch.device("cpu"),
        turn_args=SimpleNamespace(seed=42),
    )

    assert latents.shape == (2, 1, 1)
    assert [call.tolist() for call in teacher.calls] == [[[101, 1, 202]], [[101, 2, 202]]]

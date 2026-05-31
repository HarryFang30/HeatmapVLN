import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import scripts.training.train_pano_latent_adapter as train_adapter
from scripts.training.train_pano_latent_adapter import (
    _assert_internnav_system1_loaded,
    _build_batch,
    _compatible_lora_checkpoint_keys,
    _extract_student_latents,
    _lora_checkpoint_state,
    _parse_args_with_config,
)


class _FakeSystem1Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cond_projector = nn.Linear(2, 2)
        self.traj_dit = nn.Linear(2, 2)
        self.memory_encoder = nn.Linear(2, 2)
        self.rgb_model = nn.Linear(2, 2)
        self.rgb_resampler = nn.Linear(2, 2)
        self.action_encoder = nn.Linear(2, 2)
        self.action_decoder = nn.Linear(2, 2)


def _fake_loaded_model():
    head = _FakeSystem1Head()
    return SimpleNamespace(
        nextdit_action_head=head,
        _internnav_system1_load_audit={
            "source": "fake-internnav",
            "latent_queries_loaded": True,
            "loaded_keys": tuple(head.state_dict()),
        },
    )


def test_assert_internnav_system1_loaded_accepts_complete_audit():
    _assert_internnav_system1_loaded(_fake_loaded_model())


def test_assert_internnav_system1_loaded_rejects_missing_frozen_tensor():
    model = _fake_loaded_model()
    model._internnav_system1_load_audit["loaded_keys"] = tuple(
        key
        for key in model._internnav_system1_load_audit["loaded_keys"]
        if key != "rgb_model.weight"
    )

    with pytest.raises(RuntimeError, match="missing_required=1"):
        _assert_internnav_system1_loaded(model)


def test_compatible_lora_checkpoint_keys_normalizes_ddp_prefix():
    model = nn.Module()
    model.register_parameter("lora_A", nn.Parameter(torch.zeros(2, 3)))
    state = {
        "module.lora_A": torch.ones(2, 3),
        "module.lora_B": torch.ones(3, 2),
    }

    assert _compatible_lora_checkpoint_keys(model, state) == ["module.lora_A"]


def test_lora_checkpoint_state_does_not_override_internnav_system1():
    state = {
        "module.qwen.lora_A": torch.ones(2, 3),
        "latent_queries": torch.ones(1, 4, 2),
        "nextdit_action_head.cond_projector.weight": torch.ones(2, 2),
    }

    assert list(_lora_checkpoint_state(state)) == ["module.qwen.lora_A"]


def test_extract_student_latents_preserves_samples_for_aligned_teacher(monkeypatch):
    class _ClearingCollator:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __call__(self, samples):
            for sample in samples:
                sample.clear()
            return {
                "pano_inputs": {},
                "history_frames": torch.zeros(1, 1),
                "current_frame": torch.zeros(1, 1),
                "pano_num_histories": [0],
            }

    class _FakeQwen:
        def __call__(self, **_kwargs):
            return {"traj_hidden_states": torch.ones(1, 4, 2)}

    monkeypatch.setattr(train_adapter, "PanoramicTokenizedCollator", _ClearingCollator)
    sample = {"marker": "must-survive"}
    model = SimpleNamespace(
        latent_queries=torch.zeros(1, 4, 2),
        config=SimpleNamespace(dtype=torch.float32),
        qwen2_5_vl=_FakeQwen(),
    )

    _extract_student_latents(
        model,
        processor=None,
        samples=[sample],
        device=torch.device("cpu"),
        n_traj_query=4,
    )

    assert sample == {"marker": "must-survive"}


def test_build_batch_fails_fast_when_prefilter_contract_is_broken():
    with pytest.raises(RuntimeError, match="after prefilter"):
        _build_batch(
            [{"dataset_index": 3}],
            dataset=[],
            model=None,
            processor=None,
            device=torch.device("cpu"),
            n_traj_query=4,
        )


def test_stage2_cli_parser_registers_config_flags_once(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_pano_latent_adapter.py",
            "--student-config",
            "configs/train_pano_adapter_stage2_8gpu.yaml",
        ],
    )

    args = _parse_args_with_config()

    assert args.student_config == "configs/train_pano_adapter_stage2_8gpu.yaml"

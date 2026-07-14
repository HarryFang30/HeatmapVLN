from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.qwen2_5_vl.integration import (
    TRAJ_TOKEN_INDEX,
    Qwen2_5VLConfig,
    Qwen2_5VLIntegration,
)


class _MergedBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)


class _FakePeftBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.ones(2, 1))
        self.base = nn.Linear(2, 2)
        self.merge_args = None

    def merge_and_unload(self, *, progressbar, safe_merge):
        self.merge_args = (progressbar, safe_merge)
        return _MergedBackbone()


def test_merge_lora_for_frozen_forward_removes_adapter_tensors():
    integration = Qwen2_5VLIntegration(Qwen2_5VLConfig(device="cpu", use_lora=True))
    peft_model = _FakePeftBackbone()
    integration.model = peft_model
    integration._model_loaded = True

    merged_count = integration.merge_lora_for_frozen_forward(safe_merge=True)

    assert merged_count == 1
    assert peft_model.merge_args == (False, True)
    assert integration.config.use_lora is False
    assert not integration.model.training
    assert not any(param.requires_grad for param in integration.model.parameters())
    assert not any("lora_" in name for name, _ in integration.model.named_parameters())


def test_inference_tensor_clone_can_feed_trainable_adapter():
    with torch.inference_mode():
        frozen_hidden = torch.randn(2, 4, 8)
    materialized = frozen_hidden.clone()
    adapter = nn.Linear(8, 8)

    adapter(materialized).sum().backward()

    assert frozen_hidden.is_inference()
    assert not materialized.is_inference()
    assert adapter.weight.grad is not None


class _NoLastHiddenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = nn.Identity()
        self.output_hidden_states_calls = []

    def forward(self, input_ids, output_hidden_states, **_kwargs):
        self.output_hidden_states_calls.append(output_hidden_states)
        hidden = torch.ones((*input_ids.shape, 8), device=input_ids.device)
        return SimpleNamespace(
            last_hidden_state=None,
            hidden_states=(hidden,) if output_hidden_states else None,
        )


class _NoLastHiddenWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _NoLastHiddenModel()


def test_last_hidden_only_falls_back_to_hidden_state_tuple():
    integration = Qwen2_5VLIntegration(
        Qwen2_5VLConfig(
            device="cpu",
            frozen_traj_inference_mode=True,
            traj_last_hidden_state_only=True,
        )
    )
    wrapper = _NoLastHiddenWrapper()
    integration.model = wrapper
    integration._model_loaded = True
    integration.image_token_id = 999
    input_ids = torch.tensor([[11, 12, TRAJ_TOKEN_INDEX, TRAJ_TOKEN_INDEX]])
    latent_queries = torch.zeros(1, 2, 8)

    _hidden, _vision, _n_img, traj_hidden, _loss = integration._forward_model_inputs(
        {"input_ids": input_ids},
        return_hidden_states=False,
        skip_lm_head=True,
        latent_queries=latent_queries,
    )

    assert wrapper.model.output_hidden_states_calls == [False, True]
    assert integration.config.traj_last_hidden_state_only is False
    assert traj_hidden.shape == (1, 2, 8)
    assert not traj_hidden.is_inference()


def test_stage3_default_inference_path_uses_hidden_state_tuple_once():
    integration = Qwen2_5VLIntegration(
        Qwen2_5VLConfig(
            device="cpu",
            frozen_traj_inference_mode=True,
            traj_last_hidden_state_only=False,
        )
    )
    wrapper = _NoLastHiddenWrapper()
    integration.model = wrapper
    integration._model_loaded = True
    integration.image_token_id = 999
    input_ids = torch.tensor([[11, 12, TRAJ_TOKEN_INDEX, TRAJ_TOKEN_INDEX]])

    _hidden, _vision, _n_img, traj_hidden, _loss = integration._forward_model_inputs(
        {"input_ids": input_ids},
        return_hidden_states=False,
        skip_lm_head=True,
        latent_queries=torch.zeros(1, 2, 8),
    )

    assert wrapper.model.output_hidden_states_calls == [True]
    assert traj_hidden.shape == (1, 2, 8)
    assert not traj_hidden.is_inference()


class _NativeTrackingLMHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.input_shapes = []

    def forward(self, hidden_states):
        self.input_shapes.append(list(hidden_states.shape))
        vocabulary = 5
        token = torch.arange(vocabulary, dtype=torch.float32).view(1, 1, vocabulary)
        return hidden_states + token.square() * self.scale


class _SparseCorrectLogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_head = _NativeTrackingLMHead()
        self.kept_positions = None

    def forward(
        self,
        input_ids,
        output_hidden_states,
        return_dict,
        use_cache,
        logits_to_keep,
        **_kwargs,
    ):
        del output_hidden_states, return_dict, use_cache
        self.kept_positions = logits_to_keep.detach().cpu().tolist()
        batch, sequence = input_ids.shape
        row = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1)
        position = torch.arange(sequence, dtype=torch.float32).view(1, sequence, 1)
        hidden_states = (row + position)[:, logits_to_keep]
        logits = self.lm_head(hidden_states)
        return SimpleNamespace(logits=logits, loss=None, hidden_states=None)


def test_correct_label_logprobs_use_sparse_predictor_union_and_fp32():
    integration = Qwen2_5VLIntegration(Qwen2_5VLConfig(device="cpu"))
    model = _SparseCorrectLogitModel()
    integration.model = model
    integration._model_loaded = True
    integration.image_token_id = 999
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]])
    labels = torch.tensor(
        [
            [-100, -100, 2, 1, -100, -100],
            [-100, 0, -100, -100, -100, 3],
        ]
    )

    _hidden, _vision, _n_img, _traj, lm_output = integration._forward_model_inputs(
        {"input_ids": input_ids, "labels": labels},
        return_hidden_states=False,
        return_lm_correct_logprobs=True,
    )

    assert isinstance(lm_output, dict)
    assert model.kept_positions == [0, 1, 2, 4]
    assert model.lm_head.input_shapes == [[2, 4, 1]]
    assert lm_output["alignment"]["sample_predictor_positions"] == [[1, 2], [0, 4]]
    assert lm_output["alignment"]["sample_correct_token_ids"] == [[2, 1], [0, 3]]
    assert lm_output["alignment"]["backend"] == ("hf_logits_to_keep_tensor_predictor_union_v1")
    assert lm_output["correct_label_logprobs"].dtype == torch.float32
    token_logits = torch.arange(5, dtype=torch.float32).square()
    expected = F.log_softmax(token_logits, dim=-1)[torch.tensor([2, 1, 0, 3])]
    torch.testing.assert_close(lm_output["correct_label_logprobs"], expected)


class _LegacyTrackingLMHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "weight",
            torch.tensor(
                [
                    [0.0, -2.0],
                    [0.5, -1.0],
                    [1.0, 0.0],
                    [1.5, 1.0],
                    [2.0, 2.0],
                ]
            ),
        )
        self.input_shapes = []

    def forward(self, hidden_states):
        self.input_shapes.append(list(hidden_states.shape))
        return hidden_states @ self.weight.t()


class _LegacyConditionalGeneration(nn.Module):
    """Transformers-4.51-like Qwen: no logits_to_keep parameter."""

    def __init__(self, *, raise_after_lm_head: bool = False) -> None:
        super().__init__()
        self.hidden_scale = nn.Parameter(torch.tensor(0.25))
        self.lm_head = _LegacyTrackingLMHead()
        self.raise_after_lm_head = raise_after_lm_head

    def forward(
        self,
        input_ids,
        output_hidden_states,
        return_dict,
        use_cache,
        **_kwargs,
    ):
        del output_hidden_states, return_dict, use_cache
        batch, sequence = input_ids.shape
        row = torch.arange(batch, dtype=torch.float32).view(batch, 1)
        position = torch.arange(sequence, dtype=torch.float32).view(1, sequence)
        first = row + position
        second = (row + 1.0) * (position + 1.0) * self.hidden_scale
        hidden_states = torch.stack((first, second), dim=-1)
        logits = self.lm_head(hidden_states)
        if self.raise_after_lm_head:
            raise RuntimeError("legacy forward failed after lm_head")
        return SimpleNamespace(logits=logits, loss=None, hidden_states=None)


class _LegacyPeftLikeWrapper(nn.Module):
    """Mimic PEFT's kwargs-forward wrapper around the physical Qwen module."""

    def __init__(self, conditional_generation):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = conditional_generation

    def forward(self, **kwargs):
        return self.base_model.model(**kwargs)


def _legacy_sparse_fixture(*, raise_after_lm_head=False):
    integration = Qwen2_5VLIntegration(Qwen2_5VLConfig(device="cpu"))
    conditional = _LegacyConditionalGeneration(
        raise_after_lm_head=raise_after_lm_head,
    )
    integration.model = _LegacyPeftLikeWrapper(conditional)
    integration._model_loaded = True
    integration.image_token_id = 999
    inputs = {
        "input_ids": torch.tensor(
            [[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]]
        ),
        "labels": torch.tensor(
            [
                [-100, -100, 2, 1, -100, -100],
                [-100, 0, -100, -100, -100, 3],
            ]
        ),
    }
    return integration, conditional, inputs


def test_legacy_qwen_uses_lm_head_pre_hook_sparsely_and_preserves_gradient():
    integration, conditional, inputs = _legacy_sparse_fixture()
    assert len(conditional.lm_head._forward_pre_hooks) == 0

    _hidden, _vision, _n_img, _traj, lm_output = integration._forward_model_inputs(
        inputs,
        return_hidden_states=False,
        return_lm_correct_logprobs=True,
    )

    alignment = lm_output["alignment"]
    assert alignment["backend"] == "lm_head_pre_hook_predictor_union_v1"
    assert not alignment["native_logits_to_keep_explicit_signature"]
    assert alignment["conditional_generation_module"] == "base_model.model"
    assert alignment["lm_head_input_shape_before"] == [2, 6, 2]
    assert alignment["lm_head_input_shape_after"] == [2, 4, 2]
    assert alignment["lm_head_hook_call_count"] == 1
    assert alignment["lm_head_hook_removed"]
    assert conditional.lm_head.input_shapes == [[2, 4, 2]]
    assert len(conditional.lm_head._forward_pre_hooks) == 0
    lm_output["correct_label_logprobs"].sum().backward()
    assert conditional.hidden_scale.grad is not None
    assert float(conditional.hidden_scale.grad.abs().item()) > 0.0


def test_legacy_lm_head_hook_is_removed_when_model_forward_raises():
    integration, conditional, inputs = _legacy_sparse_fixture(
        raise_after_lm_head=True,
    )
    with pytest.raises(RuntimeError, match="legacy forward failed"):
        integration._forward_model_inputs(
            inputs,
            return_hidden_states=False,
            return_lm_correct_logprobs=True,
        )
    assert conditional.lm_head.input_shapes == [[2, 4, 2]]
    assert len(conditional.lm_head._forward_pre_hooks) == 0


class _NoPhysicalLMHeadModel(nn.Module):
    def forward(self, **_kwargs):
        raise AssertionError("forward body should not run")


def test_correct_label_logprobs_refuse_any_full_logits_fallback_without_lm_head():
    integration = Qwen2_5VLIntegration(Qwen2_5VLConfig(device="cpu"))
    integration.model = _NoPhysicalLMHeadModel()
    integration._model_loaded = True
    integration.image_token_id = 999
    with pytest.raises(RuntimeError, match="exactly one physical"):
        integration._forward_model_inputs(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "labels": torch.tensor([[-100, 2, -100]]),
            },
            return_hidden_states=False,
            return_lm_correct_logprobs=True,
        )

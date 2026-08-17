"""Focused regression tests for the post-load NextDiT heatmap control path."""

from __future__ import annotations

import pytest
import torch

from src.models.action.heatmap_control import HeatmapControlAdapter
from src.models.action.nextdit.nextdit_crossattn import (
    NextDiTCrossAttn,
    NextDiTCrossAttnConfig,
)
from src.models.action.nextdit.nextdit_traj import LuminaNextDiTBlock
from src.models.action.nextdit_action_head import NextDiTActionHead


def _small_crossattn(*, num_layers: int = 2) -> NextDiTCrossAttn:
    config = NextDiTCrossAttnConfig(
        input_size=8,
        patch_size=1,
        in_channels=64,
        dim=64,
        n_layers=num_layers,
        n_heads=4,
        n_kv_heads=4,
        multiple_of=32,
        ffn_dim_multiplier=1.0,
        norm_eps=1e-5,
        latent_embedding_size=64,
        learn_sigma=False,
        qk_norm=True,
        _gradient_checkpointing=False,
    )
    return NextDiTCrossAttn(config)


def _small_block() -> LuminaNextDiTBlock:
    return LuminaNextDiTBlock(
        dim=64,
        num_attention_heads=4,
        num_kv_heads=4,
        multiple_of=32,
        ffn_dim_multiplier=1.0,
        norm_eps=1e-5,
        qk_norm=True,
        cross_attention_dim=64,
    )


def _block_inputs(batch_size: int = 2):
    return {
        "hidden_states": torch.randn(batch_size, 8, 64),
        "attention_mask": torch.ones(batch_size, 8, dtype=torch.int32),
        "image_rotary_emb": None,
        "encoder_hidden_states": torch.randn(batch_size, 5, 64),
        "encoder_mask": torch.ones(batch_size, 5, dtype=torch.bool),
        "temb": torch.randn(batch_size, 64),
        "cross_attention_kwargs": {},
    }


def _model_inputs(batch_size: int = 2):
    return {
        "x": torch.randn(batch_size, 8, 64),
        "timestep": torch.randint(0, 1000, (batch_size,), dtype=torch.long),
        "z_latents": torch.randn(batch_size, 5, 64),
    }


def test_control_is_attached_only_after_native_construction_and_stays_fp32():
    model = _small_crossattn(num_layers=2)
    native_keys = tuple(model.state_dict())
    assert native_keys
    assert not any("heatmap_control" in key for key in native_keys)

    adapters = model.enable_heatmap_control(
        token_dim=32,
        control_dim=32,
        num_heads=4,
    )
    assert len(adapters) == 2
    assert adapters[0] is not adapters[1]
    assert all(adapter.gate.shape == (4,) for adapter in adapters)
    assert all(torch.count_nonzero(adapter.gate) == 0 for adapter in adapters)
    assert any("heatmap_control" in key for key in model.state_dict())

    model.to(dtype=torch.bfloat16)
    assert all(
        parameter.dtype == torch.float32
        for adapter in adapters
        for parameter in adapter.parameters()
        if parameter.is_floating_point()
    )

    same_adapters = model.enable_heatmap_control(
        token_dim=32,
        control_dim=32,
        num_heads=4,
    )
    assert same_adapters == adapters
    with pytest.raises(ValueError, match="must match"):
        model.enable_heatmap_control(token_dim=16, control_dim=32, num_heads=4)


def test_adapter_zero_gate_all_padding_and_empty_memory_are_exact_noops():
    adapter = HeatmapControlAdapter(model_dim=64, control_dim=32, num_heads=4)
    hidden_states = torch.randn(3, 7, 64)
    tokens = torch.randn(3, 5, 32)

    zero_gate_delta = adapter(
        hidden_states,
        tokens,
        heatmap_mask=torch.ones(3, 5, dtype=torch.bool),
    )
    assert zero_gate_delta.dtype == torch.float32
    assert torch.equal(zero_gate_delta, torch.zeros_like(zero_gate_delta))

    with torch.no_grad():
        adapter.gate.fill_(0.4)
    all_padding_delta = adapter(
        hidden_states,
        tokens,
        heatmap_mask=torch.zeros(3, 5, dtype=torch.bool),
    )
    assert torch.isfinite(all_padding_delta).all()
    assert torch.equal(all_padding_delta, torch.zeros_like(all_padding_delta))

    empty_delta = adapter(
        hidden_states,
        tokens[:, :0],
        heatmap_mask=torch.zeros(3, 0, dtype=torch.bool),
    )
    assert torch.isfinite(empty_delta).all()
    assert torch.equal(empty_delta, torch.zeros_like(empty_delta))


def test_block_runtime_off_and_zero_gate_preserve_bitwise_native_output():
    torch.manual_seed(7)
    block = _small_block().eval()
    inputs = _block_inputs()
    native_output = block(**inputs)

    adapter = block.enable_heatmap_control(control_dim=32, num_heads=4)
    runtime_off_output = block(**inputs)
    assert torch.equal(runtime_off_output, native_output)

    tokens = torch.randn(2, 6, 32)
    zero_gate_output = block(
        **inputs,
        heatmap_hidden_states=tokens,
        heatmap_mask=torch.ones(2, 6, dtype=torch.bool),
        heatmap_valid=torch.ones(2, dtype=torch.bool),
    )
    assert torch.equal(zero_gate_output, native_output)
    assert torch.count_nonzero(adapter.gate) == 0


def test_nonzero_control_changes_block_output_and_backpropagates():
    torch.manual_seed(11)
    block = _small_block().eval()
    adapter = block.enable_heatmap_control(control_dim=32, num_heads=4)
    with torch.no_grad():
        adapter.gate.fill_(0.3)

    inputs = _block_inputs()
    native_output = block(**inputs)
    tokens = torch.randn(2, 6, 32, requires_grad=True)
    controlled_output = block(
        **inputs,
        heatmap_hidden_states=tokens,
        heatmap_mask=torch.ones(2, 6, dtype=torch.bool),
        heatmap_valid=torch.ones(2, dtype=torch.bool),
    )
    assert not torch.equal(controlled_output, native_output)

    controlled_output.square().mean().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert torch.count_nonzero(tokens.grad) > 0
    assert adapter.gate.grad is not None
    assert torch.isfinite(adapter.gate.grad).all()
    assert torch.count_nonzero(adapter.gate.grad) > 0
    control_weight_grads = [
        parameter.grad
        for name, parameter in adapter.named_parameters()
        if name != "gate"
    ]
    assert any(
        grad is not None
        and torch.isfinite(grad).all()
        and torch.count_nonzero(grad) > 0
        for grad in control_weight_grads
    )


def test_gradient_checkpointing_receives_heatmap_inputs_and_backpropagates():
    torch.manual_seed(19)
    model = _small_crossattn(num_layers=2)
    adapters = model.enable_heatmap_control(
        token_dim=32,
        control_dim=32,
        num_heads=4,
    )
    model.train()
    model.model.gradient_checkpointing = True
    for adapter in adapters:
        with torch.no_grad():
            adapter.gate.fill_(0.2)

    inputs = _model_inputs()
    tokens = torch.randn(2, 6, 32, requires_grad=True)
    output = model(
        **inputs,
        heatmap_tokens=tokens,
        heatmap_mask=torch.tensor(
            [[True, True, True, False, False, False], [False] * 6]
        ),
        heatmap_valid=torch.tensor([True, True]),
    )
    assert torch.isfinite(output).all()
    output.square().mean().backward()

    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert torch.count_nonzero(tokens.grad[0]) > 0
    assert torch.equal(tokens.grad[1], torch.zeros_like(tokens.grad[1]))
    assert all(adapter.gate.grad is not None for adapter in adapters)


def test_full_model_zero_gate_is_bitwise_equal_to_runtime_off():
    torch.manual_seed(23)
    model = _small_crossattn(num_layers=2).eval()
    inputs = _model_inputs()
    native_output = model(**inputs)

    model.enable_heatmap_control(token_dim=32, control_dim=32, num_heads=4)
    runtime_off_output = model(**inputs)
    with_tokens_output = model(
        **inputs,
        heatmap_tokens=torch.randn(2, 4, 32),
        heatmap_mask=torch.ones(2, 4, dtype=torch.bool),
        heatmap_valid=torch.ones(2, dtype=torch.bool),
    )
    assert torch.equal(runtime_off_output, native_output)
    assert torch.equal(with_tokens_output, native_output)


def test_cfg_heatmap_order_matches_native_unconditional_conditional_order():
    tokens = torch.tensor(
        [
            [[1.0, 1.5], [2.0, 2.5]],
            [[3.0, 3.5], [4.0, 4.5]],
        ]
    )
    mask = torch.tensor([[True, False], [True, True]])
    valid = torch.tensor([True, False])
    cfg_tokens, cfg_mask, cfg_valid = NextDiTActionHead._prepare_cfg_heatmap_inputs(
        tokens,
        mask,
        valid,
        batch_size=2,
        num_sample_trajs=3,
    )

    expected_tokens = torch.cat((torch.zeros_like(tokens), tokens), dim=0)
    expected_tokens = expected_tokens.repeat_interleave(3, dim=0)
    expected_mask = torch.cat((mask, mask), dim=0).repeat_interleave(3, dim=0)
    expected_valid = torch.tensor(
        [False, False, False, False, False, False, True, True, True, False, False, False]
    )
    assert torch.equal(cfg_tokens, expected_tokens)
    assert torch.equal(cfg_mask, expected_mask)
    assert torch.equal(cfg_valid, expected_valid)
    assert torch.equal(cfg_tokens[:6], torch.zeros_like(cfg_tokens[:6]))


def test_sequence_heatmap_alignment_flattens_bn_without_silent_repetition():
    tokens = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).float()
    mask = torch.arange(2 * 3 * 4).reshape(2, 3, 4) % 2 == 0
    valid = torch.tensor([[True, False, True], [False, True, True]])
    gt_trajectory = torch.randn(2, 3, 8, 3)
    traj_images = torch.randn(2, 3, 3, 12, 12)

    flat_tokens, flat_mask, flat_valid = NextDiTActionHead._expand_heatmap_sequence_inputs(
        tokens,
        mask,
        valid,
        gt_trajectory,
        traj_images,
    )
    assert torch.equal(flat_tokens, tokens.flatten(0, 1))
    assert torch.equal(flat_mask, mask.flatten(0, 1))
    assert torch.equal(flat_valid, valid.flatten(0, 1))

    with pytest.raises(ValueError, match="multi-current"):
        NextDiTActionHead._expand_heatmap_sequence_inputs(
            tokens[:, 0],
            mask[:, 0],
            valid[:, 0],
            gt_trajectory,
            traj_images,
        )

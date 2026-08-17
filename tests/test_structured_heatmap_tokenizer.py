from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/models/heatmap/structured_heatmap_tokenizer.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_structured_heatmap_tokenizer_under_test", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
SPATIAL_STATISTIC_NAMES = _MODULE.SPATIAL_STATISTIC_NAMES
STRUCTURED_FEATURE_DIM = _MODULE.STRUCTURED_FEATURE_DIM
StructuredHeatmapTokenizer = _MODULE.StructuredHeatmapTokenizer


def _tokenizer() -> StructuredHeatmapTokenizer:
    return StructuredHeatmapTokenizer(
        token_dim=128,
        mlp_hidden_dim=64,
        temporal_num_heads=4,
        temporal_ffn_dim=128,
        dropout=0.0,
        age_scale_steps=32.0,
    )


def test_shape_probability_mass_mask_rank_and_fp32_contract() -> None:
    torch.manual_seed(4)
    tokenizer = _tokenizer()
    heatmap_logits = torch.randn(2, 4, 4, 64, 64).to(torch.bfloat16)
    visibility_logits = torch.randn(2, 4, 4).to(torch.bfloat16)
    history_mask = torch.tensor(
        [[1, 1, 1, 1], [1, 0, 1, 0]],
        dtype=torch.float32,
    )
    history_age_steps = torch.tensor(
        [[20, 8, 4, 1], [12, 99, 3, 0]],
        dtype=torch.long,
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = tokenizer(
            heatmap_logits,
            visibility_logits,
            history_mask,
            history_age_steps,
        )

    assert output["tokens"].shape == (2, 16, 128)
    assert output["token_mask"].shape == (2, 16)
    assert output["structured_features"].shape == (
        2,
        4,
        4,
        STRUCTURED_FEATURE_DIM,
    )
    assert output["spatial_statistics"].shape == (
        2,
        4,
        4,
        len(SPATIAL_STATISTIC_NAMES),
    )
    assert output["coarse_probabilities"].shape == (2, 4, 4, 8, 8)
    assert output["view_probabilities"].shape == (2, 4, 4)
    assert output["none_probability"].shape == (2, 4)
    assert output["tokens"].dtype == torch.float32
    assert output["coarse_probabilities"].dtype == torch.float32

    expected_token_mask = history_mask.bool().repeat_interleave(4, dim=1)
    assert torch.equal(output["token_mask"], expected_token_mask)
    assert torch.count_nonzero(output["tokens"][~expected_token_mask]) == 0

    coarse_mass = output["coarse_probabilities"].sum(dim=(-2, -1))
    valid_views = history_mask.bool().unsqueeze(-1).expand_as(coarse_mass)
    torch.testing.assert_close(
        coarse_mass[valid_views],
        torch.ones_like(coarse_mass[valid_views]),
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.count_nonzero(coarse_mass[~valid_views]) == 0

    categorical_mass = (
        output["view_probabilities"].sum(dim=-1)
        + output["none_probability"]
    )
    torch.testing.assert_close(
        categorical_mass,
        torch.ones_like(categorical_mass),
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.equal(
        output["none_probability"][~history_mask.bool()],
        torch.ones_like(output["none_probability"][~history_mask.bool()]),
    )
    assert torch.count_nonzero(
        output["view_probabilities"][~history_mask.bool()]
    ) == 0

    torch.testing.assert_close(
        output["history_rank"][0],
        torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]),
    )
    torch.testing.assert_close(
        output["history_rank"][1],
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
    )
    assert output["normalized_age"][1, 1] == 0
    assert output["normalized_age"][0, 0] == pytest.approx(
        math.log1p(20.0) / math.log1p(32.0)
    )


def test_uniform_and_peaked_spatial_statistics_and_five_way_probability() -> None:
    tokenizer = _tokenizer().eval()
    logits = torch.zeros(1, 1, 4, 64, 64)
    logits[0, 0, 1].fill_(-40.0)
    peak_y, peak_x = 10, 20
    logits[0, 0, 1, peak_y, peak_x] = 40.0
    visibility = torch.zeros(1, 1, 4)
    output = tokenizer(
        logits,
        visibility,
        torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([[32]]),
    )

    # Uniform spatial softmax becomes uniform 8x8 probability mass.
    torch.testing.assert_close(
        output["coarse_probabilities"][0, 0, 0],
        torch.full((8, 8), 1.0 / 64.0),
        rtol=1e-5,
        atol=1e-7,
    )
    uniform = output["spatial_statistics"][0, 0, 0]
    coordinate = torch.linspace(-1.0, 1.0, 64)
    expected_variance = coordinate.square().mean()
    torch.testing.assert_close(uniform[0], torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(uniform[1], torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(uniform[2], expected_variance, atol=1e-6, rtol=0)
    torch.testing.assert_close(uniform[3], expected_variance, atol=1e-6, rtol=0)
    torch.testing.assert_close(uniform[4], torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(uniform[5], torch.tensor(1.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        uniform[6],
        torch.tensor(1.0 / (64.0 * 64.0)),
        atol=1e-8,
        rtol=0,
    )

    peaked = output["spatial_statistics"][0, 0, 1]
    torch.testing.assert_close(peaked[0], coordinate[peak_x], atol=1e-5, rtol=0)
    torch.testing.assert_close(peaked[1], coordinate[peak_y], atol=1e-5, rtol=0)
    assert peaked[2] < 1e-5
    assert peaked[3] < 1e-5
    assert peaked[5] < 1e-5
    assert peaked[6] > 0.99999

    # [none, front, right, back, left] are equiprobable for zero logits.
    torch.testing.assert_close(
        output["none_probability"],
        torch.full((1, 1), 0.2),
    )
    torch.testing.assert_close(
        output["view_probabilities"],
        torch.full((1, 1, 4), 0.2),
    )

    # Structured geometry uses exact front/right/back/left sin/cos values.
    yaw_start = 64 + len(SPATIAL_STATISTIC_NAMES) + 3
    expected_yaw = torch.tensor(
        [[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]]
    )
    torch.testing.assert_close(
        output["structured_features"][0, 0, :, yaw_start : yaw_start + 2],
        expected_yaw,
    )
    assert output["normalized_age"].item() == pytest.approx(1.0)
    assert output["history_rank"].item() == pytest.approx(0.0)


def test_non_prefix_invalid_slots_cannot_change_valid_tokens() -> None:
    torch.manual_seed(11)
    tokenizer = _tokenizer().eval()
    logits = torch.randn(1, 4, 4, 64, 64)
    visibility = torch.randn(1, 4, 4)
    mask = torch.tensor([[True, False, True, False]])
    ages = torch.tensor([[8, 7, 2, 1]])

    baseline = tokenizer(logits, visibility, mask, ages)

    perturbed_logits = logits.clone()
    perturbed_visibility = visibility.clone()
    perturbed_ages = ages.clone()
    perturbed_logits[:, 1] = torch.randn_like(perturbed_logits[:, 1]) * 100.0
    perturbed_logits[:, 3] = torch.randn_like(perturbed_logits[:, 3]) * 100.0
    perturbed_visibility[:, 1] = 1000.0
    perturbed_visibility[:, 3] = -1000.0
    perturbed_ages[:, 1] = 1000
    perturbed_ages[:, 3] = 999

    changed = tokenizer(
        perturbed_logits,
        perturbed_visibility,
        mask,
        perturbed_ages,
    )
    valid_token_mask = mask.repeat_interleave(4, dim=1)
    torch.testing.assert_close(
        baseline["tokens"][valid_token_mask],
        changed["tokens"][valid_token_mask],
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(changed["tokens"][~valid_token_mask]) == 0
    torch.testing.assert_close(
        baseline["history_rank"],
        changed["history_rank"],
    )


def test_all_invalid_history_is_finite_and_semantically_neutral() -> None:
    torch.manual_seed(19)
    tokenizer = _tokenizer()
    output = tokenizer(
        torch.randn(2, 3, 4, 64, 64) * 100.0,
        torch.randn(2, 3, 4) * 100.0,
        torch.zeros(2, 3, dtype=torch.bool),
        torch.tensor([[8, 4, 1], [7, 3, 0]]),
    )

    for value in output.values():
        if value.is_floating_point():
            assert torch.isfinite(value).all()
    assert not output["token_mask"].any()
    assert torch.count_nonzero(output["tokens"]) == 0
    assert torch.count_nonzero(output["coarse_probabilities"]) == 0
    assert torch.count_nonzero(output["spatial_statistics"]) == 0
    assert torch.count_nonzero(output["view_probabilities"]) == 0
    assert torch.equal(
        output["none_probability"],
        torch.ones_like(output["none_probability"]),
    )
    assert torch.count_nonzero(output["normalized_age"]) == 0
    assert torch.count_nonzero(output["history_rank"]) == 0


def test_backward_reaches_tokenizer_and_valid_raw_logits_only() -> None:
    torch.manual_seed(23)
    tokenizer = _tokenizer().train()
    logits = torch.randn(
        2,
        3,
        4,
        64,
        64,
        requires_grad=True,
    )
    visibility = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor(
        [[True, False, True], [True, True, True]]
    )
    ages = torch.tensor([[8, 4, 1], [9, 3, 2]])
    output = tokenizer(logits, visibility, mask, ages)

    weights = torch.randn_like(output["tokens"])
    loss = (output["tokens"] * weights).sum()
    loss.backward()

    assert logits.grad is not None
    assert visibility.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(visibility.grad).all()
    assert logits.grad[mask].abs().sum() > 0
    assert visibility.grad[mask].abs().sum() > 0
    assert torch.count_nonzero(logits.grad[~mask]) == 0
    assert torch.count_nonzero(visibility.grad[~mask]) == 0

    core_parameters = (
        tokenizer.shared_mlp[1].weight,
        tokenizer.shared_mlp[4].weight,
        tokenizer.temporal_transformer.self_attn.in_proj_weight,
        tokenizer.temporal_transformer.linear1.weight,
    )
    for parameter in core_parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0

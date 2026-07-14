import inspect

import pytest
import torch

from src.models.heatmap.pose_free_matching import (
    PoseFreeHistoryMatcher,
    pad_history_queries,
)


def _make_inputs(*, batch_size=2, num_histories=4, coarse_size=3):
    torch.manual_seed(7)
    current = torch.randn(batch_size, 4, coarse_size, coarse_size, 12)
    histories = torch.randn(batch_size, num_histories, 10)
    return current, histories


def _make_matcher(*, heatmap_size=(11, 13)):
    torch.manual_seed(11)
    return PoseFreeHistoryMatcher(
        current_dim=12,
        query_dim=10,
        match_dim=6,
        heatmap_size=heatmap_size,
        visibility_hidden_dim=5,
    )


def test_pose_free_matcher_output_contract_and_no_pose_api():
    matcher = _make_matcher()
    current, histories = _make_inputs()

    output = matcher(current, histories)

    assert output["heatmaps"].shape == (2, 4, 4, 11, 13)
    assert output["heatmap_logits"].shape == (2, 4, 4, 11, 13)
    assert output["coarse_heatmap_logits"].shape == (2, 4, 4, 3, 3)
    assert output["visibility"].shape == (2, 4, 4)
    assert output["history_mask"].shape == (2, 4)
    assert output["history_mask"].dtype == torch.bool
    assert output["history_mask"].all()
    assert torch.all((output["heatmaps"] >= 0) & (output["heatmaps"] <= 1))

    parameters = inspect.signature(matcher.forward).parameters
    assert "history_rel_poses" not in parameters
    assert "relative_pose" not in parameters
    assert matcher.uses_relative_pose is False
    assert all("pose" not in name and "trajectory" not in name for name in matcher.state_dict())

    with pytest.raises(TypeError, match="unexpected keyword"):
        matcher(current, histories, history_rel_poses=torch.randn(2, 4, 4))


def test_history_permutation_equivariance_is_exact_up_to_float_tolerance():
    matcher = _make_matcher().eval()
    current, histories = _make_inputs()
    permutation = torch.tensor([2, 0, 3, 1])

    original = matcher(current, histories)
    permuted = matcher(current, histories[:, permutation])

    for key in (
        "heatmaps",
        "heatmap_logits",
        "coarse_heatmap_logits",
        "visibility",
    ):
        torch.testing.assert_close(
            permuted[key],
            original[key][:, permutation],
            rtol=1e-6,
            atol=1e-6,
        )
    torch.testing.assert_close(permuted["history_mask"], original["history_mask"][:, permutation])


def test_changing_one_history_query_cannot_change_other_outputs():
    matcher = _make_matcher().eval()
    current, histories = _make_inputs(batch_size=1)
    changed_histories = histories.clone()
    changed_histories[:, 1] = torch.randn_like(changed_histories[:, 1]) * 5.0

    original = matcher(current, histories)
    changed = matcher(current, changed_histories)

    unchanged_indices = torch.tensor([0, 2, 3])
    for key in (
        "heatmaps",
        "heatmap_logits",
        "coarse_heatmap_logits",
        "visibility",
    ):
        torch.testing.assert_close(
            changed[key][:, unchanged_indices],
            original[key][:, unchanged_indices],
            rtol=0,
            atol=0,
        )
    assert not torch.allclose(changed["heatmaps"][:, 1], original["heatmaps"][:, 1])


def test_history_mask_zeros_padded_outputs_and_preserves_real_outputs():
    matcher = _make_matcher().eval()
    current, histories = _make_inputs()
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])

    unmasked = matcher(current, histories)
    masked = matcher(current, histories, history_mask=mask)

    torch.testing.assert_close(masked["heatmaps"][mask], unmasked["heatmaps"][mask])
    torch.testing.assert_close(masked["visibility"][mask], unmasked["visibility"][mask])
    assert torch.count_nonzero(masked["heatmaps"][~mask]) == 0
    assert torch.count_nonzero(masked["heatmap_logits"][~mask]) == 0
    assert torch.count_nonzero(masked["coarse_heatmap_logits"][~mask]) == 0
    assert torch.count_nonzero(masked["visibility"][~mask]) == 0


def test_gradients_reach_current_and_history_vlm_features():
    matcher = _make_matcher()
    current, histories = _make_inputs(batch_size=1)
    current.requires_grad_(True)
    histories.requires_grad_(True)

    output = matcher(current, histories)
    loss = output["heatmap_logits"].square().mean() + output["visibility"].square().mean()
    loss.backward()

    assert current.grad is not None
    assert histories.grad is not None
    assert torch.isfinite(current.grad).all()
    assert torch.isfinite(histories.grad).all()
    assert current.grad.abs().sum() > 0
    assert histories.grad.abs().sum() > 0
    assert matcher.current_projection.weight.grad is not None
    assert matcher.query_projection.weight.grad is not None


def test_pad_history_queries_keeps_gradient_and_builds_true_length_mask():
    q00 = torch.randn(5, requires_grad=True)
    q01 = torch.randn(5, requires_grad=True)
    q10 = torch.randn(5, requires_grad=True)

    padded, mask = pad_history_queries([[q00, q01], [q10]])

    assert padded.shape == (2, 2, 5)
    assert mask.tolist() == [[True, True], [True, False]]
    assert torch.count_nonzero(padded[1, 1]) == 0

    padded[mask].square().sum().backward()
    for query in (q00, q01, q10):
        assert query.grad is not None
        assert query.grad.abs().sum() > 0


def test_empty_history_dimension_has_stable_output_shapes():
    matcher = _make_matcher(heatmap_size=(8, 9))
    current, _ = _make_inputs(batch_size=3)
    histories = torch.empty(3, 0, 10)

    output = matcher(current, histories)

    assert output["heatmaps"].shape == (3, 0, 4, 8, 9)
    assert output["visibility"].shape == (3, 0, 4)
    assert output["coarse_heatmap_logits"].shape == (3, 0, 4, 3, 3)


@pytest.mark.parametrize(
    ("current_shape", "history_shape", "mask_shape", "message"),
    [
        ((2, 3, 3, 3, 12), (2, 4, 10), None, "4 panoramic views"),
        ((2, 4, 3, 3, 11), (2, 4, 10), None, "current feature dim"),
        ((2, 4, 3, 3, 12), (1, 4, 10), None, "Batch mismatch"),
        ((2, 4, 3, 3, 12), (2, 4, 9), None, "history query dim"),
        ((2, 4, 3, 3, 12), (2, 4, 10), (2, 3), "history_mask"),
    ],
)
def test_invalid_shapes_fail_loudly(current_shape, history_shape, mask_shape, message):
    matcher = _make_matcher()
    current = torch.randn(*current_shape)
    histories = torch.randn(*history_shape)
    mask = None if mask_shape is None else torch.ones(*mask_shape)

    with pytest.raises(ValueError, match=message):
        matcher(current, histories, history_mask=mask)


def test_decoder_parameter_budget_is_lightweight():
    matcher = PoseFreeHistoryMatcher(
        current_dim=3584,
        query_dim=3584,
        match_dim=64,
        visibility_hidden_dim=16,
    )

    # Two 3584x64 projections dominate.  The complete readout stays below
    # 0.5M parameters, far smaller than the legacy DPT + transformer + deconv
    # heatmap stack.
    assert matcher.trainable_parameter_count < 500_000

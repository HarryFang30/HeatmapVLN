import torch
import torch.nn.functional as F

from src.models.heatmap.fine_localization import FineLocalization
from src.models.heatmap.heatmap_vln_loss import HeatmapVLNLoss


def _one_hot_heatmap(
    num_histories: int,
    height: int,
    width: int,
) -> torch.Tensor:
    return torch.zeros(num_histories, 4, height, width)


def test_raw_logits_keep_spatial_gradient_after_sigmoid_saturates():
    height = width = 4
    raw_logits = torch.full(
        (1, 1, 4, height, width),
        80.0,
        requires_grad=True,
    )
    probabilities = raw_logits.sigmoid()
    assert torch.equal(probabilities, torch.ones_like(probabilities))

    gt_heatmaps = torch.zeros_like(probabilities)
    gt_heatmaps[0, 0, 0, 0, 0] = 1.0
    gt_visibility = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    criterion = HeatmapVLNLoss(
        heatmap_size=(height, width),
        lambda_vis=0.0,
        lambda_coord=0.0,
        lambda_neg=0.0,
        allow_probability_fallback=False,
    )
    losses = criterion(
        torch.zeros_like(gt_visibility),
        probabilities,
        gt_visibility,
        gt_heatmaps,
        pred_heatmap_logits=raw_logits,
    )
    losses["total"].backward()

    assert torch.isfinite(losses["total"])
    assert raw_logits.grad is not None
    assert torch.isfinite(raw_logits.grad).all()
    assert raw_logits.grad[0, 0, 0].abs().sum() > 0


def test_raw_negative_loss_has_gradient_for_saturated_false_positive():
    raw_logits = torch.full((1, 1, 4, 3, 3), 80.0, requires_grad=True)
    probabilities = raw_logits.sigmoid()
    gt_heatmaps = torch.zeros_like(probabilities)
    gt_visibility = torch.zeros(1, 1, 4)
    criterion = HeatmapVLNLoss(
        heatmap_size=(3, 3),
        lambda_vis=0.0,
        lambda_peak=0.0,
        lambda_coord=0.0,
        lambda_neg=1.0,
        allow_probability_fallback=False,
    )
    losses = criterion(
        torch.zeros_like(gt_visibility),
        probabilities,
        gt_visibility,
        gt_heatmaps,
        pred_heatmap_logits=raw_logits,
    )
    losses["total"].backward()

    assert losses["neg_loss"] > 79.0
    assert raw_logits.grad is not None
    assert torch.all(raw_logits.grad > 0)


def test_dsnt_smooth_l1_handles_corner_targets_in_normalized_coordinates():
    height, width = 5, 9
    target = torch.zeros(1, height, width)
    target[0, 0, 0] = 1.0
    good = torch.zeros_like(target)
    bad = torch.zeros_like(target)
    good[0, 0, 0] = 12.0
    bad[0, -1, -1] = 12.0

    criterion = HeatmapVLNLoss(
        heatmap_size=(height, width),
        temperature=0.25,
        coord_smooth_l1_beta=0.1,
    )
    good_loss = criterion.soft_argmax_coord_loss(good, target)
    bad_loss = criterion.soft_argmax_coord_loss(bad, target)

    assert good_loss < 1e-4
    assert bad_loss > 3.0


def test_view_macro_auxiliary_amplifies_minority_direction_gradient():
    height = width = 3
    view_indices = torch.tensor([0] * 8 + [1, 2, 3])
    num_histories = int(view_indices.numel())
    raw_logits = torch.zeros(
        num_histories,
        4,
        height,
        width,
        requires_grad=True,
    )
    probabilities = raw_logits.sigmoid()
    gt_visibility = torch.zeros(num_histories, 4)
    gt_heatmaps = _one_hot_heatmap(num_histories, height, width)
    for row, view in enumerate(view_indices.tolist()):
        gt_visibility[row, view] = 1.0
        gt_heatmaps[row, view, view // 2, view % 2] = 1.0

    criterion = HeatmapVLNLoss(
        heatmap_size=(height, width),
        lambda_vis=0.0,
        lambda_peak=1.0,
        lambda_coord=0.0,
        lambda_neg=0.0,
        lambda_view_macro=1.0,
        allow_probability_fallback=False,
    )
    losses = criterion(
        torch.zeros_like(gt_visibility),
        probabilities,
        gt_visibility,
        gt_heatmaps,
        pred_heatmap_logits=raw_logits,
    )
    losses["total"].backward()

    per_history_grad = torch.stack(
        [
            raw_logits.grad[row, view].norm()
            for row, view in enumerate(view_indices.tolist())
        ]
    )
    majority_per_sample = per_history_grad[:8].mean()
    minority_per_sample = per_history_grad[8:].mean()
    assert minority_per_sample > 2.0 * majority_per_sample
    assert losses["view_macro_loss"] > 0


def test_direction_macro_auxiliary_balances_view_classification_without_resampling():
    height = width = 3
    # Natural distribution is back-heavy; the three minority directions occur
    # once each. The macro auxiliary must amplify their per-sample gradients.
    view_indices = torch.tensor([2] * 8 + [0, 1, 3])
    num_histories = int(view_indices.numel())
    pred_visibility = torch.zeros(num_histories, 4, requires_grad=True)
    raw_logits = torch.zeros(
        num_histories,
        4,
        height,
        width,
        requires_grad=True,
    )
    gt_visibility = torch.zeros(num_histories, 4)
    gt_heatmaps = torch.zeros_like(raw_logits)
    for row, view in enumerate(view_indices.tolist()):
        gt_visibility[row, view] = 1
        gt_heatmaps[row, view, 1, 1] = 1

    criterion = HeatmapVLNLoss(
        heatmap_size=(height, width),
        lambda_vis=0.0,
        lambda_peak=0.0,
        lambda_coord=0.0,
        lambda_neg=0.0,
        lambda_panoramic_view=1.0,
        lambda_direction_macro=1.0,
        allow_probability_fallback=False,
    )
    losses = criterion(
        pred_visibility,
        raw_logits.sigmoid(),
        gt_visibility,
        gt_heatmaps,
        pred_heatmap_logits=raw_logits,
    )
    losses["total"].backward()

    per_history_grad = pred_visibility.grad.norm(dim=-1)
    back_grad = per_history_grad[:8].mean()
    minority_grad = per_history_grad[8:].mean()
    assert minority_grad > 2.0 * back_grad
    assert losses["direction_macro_loss"] > 0


def test_hierarchical_panorama_view_loss_covers_none_and_four_views():
    gt_visibility = torch.zeros(5, 4)
    gt_heatmaps = torch.zeros(5, 4, 2, 2)
    for row in range(1, 5):
        view = row - 1
        gt_visibility[row, view] = 1.0
        gt_heatmaps[row, view, 0, 0] = 1.0

    good = torch.full((5, 4), -8.0)
    for row in range(1, 5):
        good[row, row - 1] = 8.0
    bad = good.roll(shifts=1, dims=0)

    criterion = HeatmapVLNLoss(heatmap_size=(2, 2))
    good_loss = criterion.panoramic_view_loss(
        good,
        gt_visibility,
        gt_heatmaps,
        valid=None,
    )
    bad_loss = criterion.panoramic_view_loss(
        bad,
        gt_visibility,
        gt_heatmaps,
        valid=None,
    )
    assert good_loss < 1e-2
    assert bad_loss > 5.0


def test_fine_localization_exposes_raw_logits_with_opt_in_coarse_residual():
    torch.manual_seed(0)
    legacy = FineLocalization(c_fused=2, coarse_logit_residual=False)
    residual = FineLocalization(c_fused=2, coarse_logit_residual=True)
    residual.load_state_dict(legacy.state_dict(), strict=True)
    for parameter in legacy.refine.parameters():
        parameter.data.zero_()
    residual.load_state_dict(legacy.state_dict(), strict=True)

    vit_fused = torch.zeros(4, 2, 16, 16)
    coarse_logits = torch.linspace(-2.0, 2.0, 8 * 8).reshape(1, 1, 8, 8)
    coarse_logits = coarse_logits.expand(1, 4, 8, 8).clone()
    spatial_out = torch.zeros(1, 4 * 8 * 8, 2)

    legacy_prob, legacy_logits = legacy(
        vit_fused,
        coarse_logits,
        spatial_out,
        return_logits=True,
    )
    residual_prob, residual_logits = residual(
        vit_fused,
        coarse_logits,
        spatial_out,
        return_logits=True,
    )
    expected = F.interpolate(
        coarse_logits.reshape(4, 1, 8, 8),
        size=(64, 64),
        mode="bilinear",
        align_corners=False,
    ).reshape(1, 4, 64, 64)

    assert torch.equal(legacy_logits, torch.zeros_like(legacy_logits))
    assert torch.equal(legacy_prob, torch.full_like(legacy_prob, 0.5))
    assert torch.allclose(residual_logits, expected)
    assert torch.allclose(residual_prob, residual_logits.sigmoid())
    assert legacy.state_dict().keys() == residual.state_dict().keys()

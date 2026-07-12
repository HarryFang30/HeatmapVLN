import pytest
import torch

from src.models.heatmap.heatmap_vln_loss import HeatmapVLNLoss


def test_padded_histories_do_not_contribute_invisible_negative_loss():
    loss_fn = HeatmapVLNLoss(
        lambda_vis=0.0,
        lambda_coord=0.0,
        lambda_peak=0.0,
        lambda_neg=1.0,
        heatmap_size=(2, 2),
    )
    pred_vis = torch.zeros(1, 3, 4)
    gt_vis = torch.zeros_like(pred_vis)
    gt_heatmaps = torch.zeros(1, 3, 4, 2, 2)
    pred_heatmaps = torch.full_like(gt_heatmaps, 0.1)
    pred_heatmaps[:, 2] = 0.99  # Deliberately bad prediction in padded K slot.

    masked = loss_fn(
        pred_vis,
        pred_heatmaps,
        gt_vis,
        gt_heatmaps,
        history_mask=torch.tensor([[1.0, 1.0, 0.0]]),
    )["total"]
    unmasked = loss_fn(
        pred_vis,
        pred_heatmaps,
        gt_vis,
        gt_heatmaps,
    )["total"]

    expected_real_only = -torch.log(torch.tensor(0.9))
    assert torch.allclose(masked, expected_real_only)
    assert masked < unmasked


def test_history_mask_shape_mismatch_fails_instead_of_dropping_mask():
    loss_fn = HeatmapVLNLoss(heatmap_size=(2, 2))
    pred_vis = torch.zeros(1, 3, 4)
    pred_heatmaps = torch.zeros(1, 3, 4, 2, 2)

    with pytest.raises(ValueError, match="history_mask shape mismatch"):
        loss_fn(
            pred_vis,
            pred_heatmaps,
            torch.zeros_like(pred_vis),
            torch.zeros_like(pred_heatmaps),
            history_mask=torch.ones(1, 1),
        )

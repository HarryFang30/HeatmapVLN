import math

import torch

from scripts.training import validate as validate_module
from scripts.training.validate import _HeatmapJointMetricAccumulator


def _metric_batch():
    height = width = 12
    num_histories = 6
    pred_visibility = torch.full((1, num_histories, 4), -5.0)
    pred_heatmaps = torch.full((1, num_histories, 4, height, width), -10.0)
    gt_visibility = torch.zeros(1, num_histories, 4)
    gt_heatmaps = torch.zeros(1, num_histories, 4, height, width)

    # Row 0: GT none, prediction none.
    # Row 1: front, correct view and exact pixel.
    gt_visibility[0, 1, 0] = 1
    gt_heatmaps[0, 1, 0, 1, 1] = 1
    pred_visibility[0, 1, 0] = 5
    pred_heatmaps[0, 1, 0, 1, 1] = 10

    # Row 2: right, correct view, five-pixel error.
    gt_visibility[0, 2, 1] = 1
    gt_heatmaps[0, 2, 1, 1, 1] = 1
    pred_visibility[0, 2, 1] = 5
    pred_heatmaps[0, 2, 1, 1, 6] = 10

    # Row 3: back spatial peak is exact, but the selected view is front.
    gt_visibility[0, 3, 2] = 1
    gt_heatmaps[0, 3, 2, 2, 2] = 1
    pred_visibility[0, 3, 0] = 5
    pred_heatmaps[0, 3, 2, 2, 2] = 10

    # Row 4: left, correct view, nine-pixel error.
    gt_visibility[0, 4, 3] = 1
    gt_heatmaps[0, 4, 3, 1, 1] = 1
    pred_visibility[0, 4, 3] = 5
    pred_heatmaps[0, 4, 3, 1, 10] = 10

    # Row 5 deliberately looks valid but is padding and must not contribute.
    gt_visibility[0, 5, 0] = 1
    gt_heatmaps[0, 5, 0, 0, 0] = 1
    pred_visibility[0, 5, 1] = 5
    pred_heatmaps[0, 5, 0, 11, 11] = 10

    history_mask = torch.tensor([[True, True, True, True, True, False]])
    return {
        "pred_visibility_logits": pred_visibility,
        "pred_heatmaps": pred_heatmaps,
        "gt_visibility": gt_visibility,
        "gt_heatmaps": gt_heatmaps,
        "history_mask": history_mask,
    }


def test_joint_metrics_use_natural_counts_and_include_none_in_view5_accuracy():
    state = _HeatmapJointMetricAccumulator(
        heatmap_size=(12, 12),
        device=torch.device("cpu"),
    )
    state.update(**_metric_batch())
    metrics = state.compute()

    assert metrics["val_heatmap_valid_count"] == 5
    assert metrics["val_heatmap_visible_count"] == 4
    assert metrics["val_heatmap_none_count"] == 1
    assert metrics["val_heatmap_view5_accuracy"] == 4 / 5
    assert metrics["val_heatmap_joint_pck4"] == 1 / 4
    assert metrics["val_heatmap_joint_pck8"] == 2 / 4
    assert metrics["val_heatmap_macro_joint_pck8"] == 0.5
    assert metrics["val_heatmap_supported_direction_count"] == 4

    # Conditional spatial errors are [0, 5, 0, 9]. Wrong-view selection is
    # reflected by joint PCK rather than corrupting the spatial-head quantile.
    assert metrics["val_heatmap_pixel_error_median"] == 2.5
    assert math.isclose(
        metrics["val_heatmap_pixel_error_p90"],
        7.8,
        rel_tol=0,
        abs_tol=1e-12,
    )

    expected = {
        "front": (1.0, 1),
        "right": (1.0, 1),
        "back": (0.0, 1),
        "left": (0.0, 1),
    }
    for view, (pck8, count) in expected.items():
        assert metrics[f"val_heatmap_{view}_pck8"] == pck8
        assert metrics[f"val_heatmap_{view}_count"] == count


def test_macro_joint_pck8_does_not_reweight_natural_overall_metric():
    state = _HeatmapJointMetricAccumulator(
        heatmap_size=(4, 4),
        device=torch.device("cpu"),
    )
    state.counts[state._VISIBLE] = 5
    state.counts[state._JOINT_PCK8_CORRECT] = 4
    state.counts[state._PER_VIEW_COUNT_START + 0] = 4
    state.counts[state._PER_VIEW_PCK8_START + 0] = 4
    state.counts[state._PER_VIEW_COUNT_START + 1] = 1
    state.counts[state._PER_VIEW_PCK8_START + 1] = 0

    metrics = state.compute()
    assert metrics["val_heatmap_joint_pck8"] == 0.8
    assert metrics["val_heatmap_macro_joint_pck8"] == 0.5
    assert metrics["val_heatmap_supported_direction_count"] == 2


def test_joint_metrics_all_reduce_adds_counts_and_histogram_before_quantiles(
    monkeypatch,
):
    batch = _metric_batch()
    first_rows = slice(0, 3)
    second_rows = slice(3, 6)

    first = _HeatmapJointMetricAccumulator(
        heatmap_size=(12, 12),
        device=torch.device("cpu"),
    )
    second = _HeatmapJointMetricAccumulator(
        heatmap_size=(12, 12),
        device=torch.device("cpu"),
    )
    full = _HeatmapJointMetricAccumulator(
        heatmap_size=(12, 12),
        device=torch.device("cpu"),
    )
    first.update(
        pred_visibility_logits=batch["pred_visibility_logits"][:, first_rows],
        pred_heatmaps=batch["pred_heatmaps"][:, first_rows],
        gt_visibility=batch["gt_visibility"][:, first_rows],
        gt_heatmaps=batch["gt_heatmaps"][:, first_rows],
        history_mask=batch["history_mask"][:, first_rows],
    )
    second.update(
        pred_visibility_logits=batch["pred_visibility_logits"][:, second_rows],
        pred_heatmaps=batch["pred_heatmaps"][:, second_rows],
        gt_visibility=batch["gt_visibility"][:, second_rows],
        gt_heatmaps=batch["gt_heatmaps"][:, second_rows],
        history_mask=batch["history_mask"][:, second_rows],
    )
    full.update(**batch)

    remote_tensors = [
        second.counts.clone(),
        second.pixel_error_histogram.clone(),
    ]

    def fake_all_reduce(tensor):
        tensor.add_(remote_tensors.pop(0))
        return tensor

    monkeypatch.setattr(
        validate_module,
        "_dist_all_reduce_in_place",
        fake_all_reduce,
    )
    first.all_reduce()

    assert not remote_tensors
    assert torch.equal(first.counts, full.counts)
    assert torch.equal(
        first.pixel_error_histogram,
        full.pixel_error_histogram,
    )
    assert first.compute() == full.compute()


def test_all_none_validation_has_finite_selection_metric():
    state = _HeatmapJointMetricAccumulator(
        heatmap_size=(4, 4),
        device=torch.device("cpu"),
    )
    pred_visibility = torch.tensor(
        [[[-2.0, -2.0, -2.0, -2.0], [3.0, -2.0, -2.0, -2.0]]]
    )
    state.update(
        pred_visibility_logits=pred_visibility,
        pred_heatmaps=torch.zeros(1, 2, 4, 4, 4),
        gt_visibility=torch.zeros(1, 2, 4),
        gt_heatmaps=torch.zeros(1, 2, 4, 4, 4),
    )
    metrics = state.compute()

    assert metrics["val_heatmap_view5_accuracy"] == 0.5
    assert metrics["val_heatmap_joint_pck8"] == 0.0
    assert metrics["val_heatmap_pixel_error_median"] == 0.0
    assert metrics["val_heatmap_pixel_error_p90"] == 0.0

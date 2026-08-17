import logging

import torch

from scripts.training.train_loop import (
    _heatmap_diagnostic_distribution,
    _log_heatmap_diagnostics,
)


class _RecordingWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, name, value, step):
        self.scalars[name] = (value, step)


def test_heatmap_diagnostics_prefer_raw_logits_over_sigmoid_reconstruction():
    probabilities = torch.full((1, 1, 1, 2, 2), 0.5)
    logits = torch.tensor(
        [[[[[20.0, 0.0], [0.0, 0.0]]]]],
        dtype=torch.float32,
    )

    returned_probabilities, distribution, used_raw = (
        _heatmap_diagnostic_distribution(
            {
                "heatmaps": probabilities,
                "heatmap_logits": logits,
            }
        )
    )

    assert used_raw is True
    torch.testing.assert_close(
        returned_probabilities,
        torch.full((1, 1, 2, 2), 0.5),
    )
    assert distribution[0, 0, 0, 0] > 0.999


def test_heatmap_diagnostics_keep_legacy_probability_fallback():
    logits = torch.tensor([[[[2.0, 0.0], [-1.0, 1.0]]]])
    probabilities = torch.sigmoid(logits).unsqueeze(1)

    _, distribution, used_raw = _heatmap_diagnostic_distribution(
        {"heatmaps": probabilities}
    )

    assert used_raw is False
    expected = torch.softmax(logits.reshape(1, 1, -1), dim=-1).reshape_as(logits)
    torch.testing.assert_close(distribution, expected, rtol=1e-5, atol=1e-6)


def test_heatmap_diagnostics_allow_frozen_prediction_without_gt():
    writer = _RecordingWriter()
    _log_heatmap_diagnostics(
        {
            "heatmaps": torch.full((1, 1, 1, 2, 2), 0.5),
            "visibility": torch.tensor([[10.0, -10.0]]),
        },
        None,
        {"gt_visibility": torch.tensor([[1.0, 0.0]])},
        writer,
        actual_step=400,
        cfg={"log": {}},
        logger=logging.getLogger(__name__),
    )

    assert "diag/pred_heatmap_mean" not in writer.scalars
    assert writer.scalars["diag/vis_accuracy"] == (1.0, 400)

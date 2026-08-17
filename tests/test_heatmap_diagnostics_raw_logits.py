import torch

from scripts.training.train_loop import _heatmap_diagnostic_distribution


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


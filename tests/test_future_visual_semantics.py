from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


MODULE_PATH = (
    Path(__file__).parents[1] / "src" / "models" / "past_plan_action.py"
)
SPEC = importlib.util.spec_from_file_location(
    "future_visual_semantics_past_plan_action",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
PPA = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PPA
SPEC.loader.exec_module(PPA)


def _gate(
    logits: torch.Tensor,
    confidence: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is None:
        mask = torch.ones(logits.shape[:2], dtype=torch.bool)
    return PPA._confidence_gated_future_heatmaps(logits, confidence, mask)


class FutureVisualSemanticsTest(unittest.TestCase):
    def test_nonempty_view_peak_equals_independent_visibility_confidence(self) -> None:
        logits = torch.tensor(
            [[[[[-4.0, -1.0, 0.5], [1.0, 2.0, -2.0]],
               [[0.4, 0.1, 0.2], [0.3, 0.8, 0.15]]]]]
        )
        confidence = torch.tensor([[[0.23, 0.81]]])

        gated = _gate(logits, confidence)

        torch.testing.assert_close(gated.amax(dim=(-2, -1)), confidence)

    def test_confidence_changes_brightness_without_changing_spatial_shape(self) -> None:
        logits = torch.tensor(
            [[[[[-4.0, -1.0, 0.5], [1.0, 2.0, -2.0]]]]]
        )
        low = torch.tensor([[[0.20]]])
        high = torch.tensor([[[0.85]]])

        low_gated = _gate(logits, low)
        high_gated = _gate(logits, high)
        expected_shape = torch.exp(
            logits - logits.amax(dim=(-2, -1), keepdim=True)
        )

        torch.testing.assert_close(low_gated / low[..., None, None], expected_shape)
        torch.testing.assert_close(high_gated / high[..., None, None], expected_shape)

    def test_additive_logit_offset_preserves_exact_display_shape(self) -> None:
        logits = torch.tensor(
            [[[[[-4.0, -1.0, 0.0], [1.0, 2.0, -2.0]]]]],
            dtype=torch.float64,
        )
        confidence = torch.tensor([[[0.67]]], dtype=torch.float64)

        baseline = _gate(logits, confidence)
        offset = _gate(logits + 8.0, confidence)

        self.assertTrue(torch.equal(offset, baseline))

    def test_extreme_empty_and_masked_maps_are_finite_or_hard_zero(self) -> None:
        logits = torch.tensor(
            [
                [
                    [
                        [[-torch.inf, -torch.inf], [-torch.inf, -torch.inf]],
                        [[-1.0e30, 1.0e30], [3.0, -4.0]],
                    ],
                    [
                        [[9.0, 1.0], [2.0, 4.0]],
                        [[3.0, 6.0], [5.0, 2.0]],
                    ],
                ]
            ]
        )
        original = logits.clone()
        confidence = torch.tensor([[[0.90, 0.60], [0.70, 0.80]]])
        mask = torch.tensor([[True, False]])

        gated = _gate(logits, confidence, mask)

        self.assertEqual(torch.count_nonzero(gated[0, 0, 0]).item(), 0)
        torch.testing.assert_close(gated[0, 0, 1].amax(), confidence[0, 0, 1])
        self.assertEqual(torch.count_nonzero(gated[0, 1]).item(), 0)
        self.assertTrue(torch.isfinite(gated).all().item())
        self.assertTrue(torch.equal(logits, original))


if __name__ == "__main__":
    unittest.main()

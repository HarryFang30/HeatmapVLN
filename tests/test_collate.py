"""Tests for the collate_fn — shape and dtype verification."""

import torch
from tests.conftest import _make_dummy_sample

from scripts.training.collate import collate_fn


class TestCollateBasic:
    def test_collate_basic_shapes(self):
        """Collated batch has correct shapes and dtypes."""
        batch = [_make_dummy_sample(K=8), _make_dummy_sample(K=8)]
        result = collate_fn(batch)

        assert result["history_frames"].shape == (2, 8, 3, 256, 256)
        assert result["current_frame"].shape == (2, 3, 256, 256)
        assert result["heatmap"].shape[0] == 2
        assert result["action"].shape == (2, 2)
        assert result["action_valid"].shape == (2,)
        assert result["is_stop"].shape == (2,)
        assert len(result["text"]) == 2
        assert result["history_mask"].shape == (2, 8)

    def test_collate_with_trajectory(self):
        """Trajectory fields are correctly stacked."""
        batch = [_make_dummy_sample(with_trajectory=True) for _ in range(3)]
        result = collate_fn(batch)

        assert result["trajectory"].shape == (3, 24, 3)
        assert result["trajectory_valid"].shape == (3,)
        assert result["progress"].shape == (3,)

    def test_collate_variable_history(self):
        """Different history lengths are padded to the longest."""
        s1 = _make_dummy_sample(K=4)
        s2 = _make_dummy_sample(K=8)
        result = collate_fn([s1, s2])

        assert result["history_frames"].shape == (2, 8, 3, 256, 256)
        assert result["history_mask"][0, :4].sum() == 4
        assert result["history_mask"][0, 4:].sum() == 0
        assert result["history_mask"][1].sum() == 8

    def test_collate_preserves_dtypes(self):
        """Key tensors have expected dtypes."""
        batch = [_make_dummy_sample()]
        result = collate_fn(batch)

        assert result["history_frames"].dtype == torch.float32
        assert result["action_valid"].dtype in (torch.float32, torch.float64)
        assert result["history_mask"].dtype == torch.float32

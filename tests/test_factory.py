"""Tests for the dataset factory functions."""

from unittest.mock import patch

from src.data.factory import (
    build_dataset,
    build_sliding_window_dataset,
    build_trajectory_dataset,
)


class TestFactory:
    def test_build_sliding_window_extracts_params(self, minimal_cfg):
        """Factory extracts correct params from config and passes to constructor."""
        with patch("src.data.sliding_window_dataset.VLNSlidingWindowDataset") as MockDS:
            MockDS.return_value = "mock_dataset"
            result = build_sliding_window_dataset(minimal_cfg, split="train")
            assert result == "mock_dataset"
            call_kwargs = MockDS.call_args[1]
            assert call_kwargs["root"] == "/tmp/fake_data"
            assert call_kwargs["split"] == "train"
            assert call_kwargs["min_history"] == 5
            assert call_kwargs["image_size"] == (256, 256)

    def test_build_trajectory_extracts_params(self, minimal_cfg):
        """Factory extracts trajectory-specific params."""
        minimal_cfg["data"]["dataset_type"] = "trajectory"
        minimal_cfg["data"]["trajectory"]["enable_augmentation"] = False
        with patch("src.data.trajectory_dataset.VLNTrajectoryDataset") as MockDS:
            MockDS.return_value = "mock_traj"
            result = build_trajectory_dataset(minimal_cfg, split="val")
            assert result == "mock_traj"
            call_kwargs = MockDS.call_args[1]
            assert call_kwargs["split"] == "val"
            assert call_kwargs["min_history"] == 5
            assert call_kwargs["enable_augmentation"] is False

    def test_build_trajectory_can_skip_duplicate_history_frames(self, minimal_cfg):
        minimal_cfg["data"]["trajectory"]["load_history_frames"] = False
        with patch("src.data.trajectory_dataset.VLNTrajectoryDataset") as mock_ds:
            build_trajectory_dataset(minimal_cfg, split="train")

        assert mock_ds.call_args.kwargs["load_history_frames"] is False

    def test_build_dataset_dispatches_sliding_window(self, minimal_cfg):
        """build_dataset dispatches to sliding_window when dataset_type matches."""
        minimal_cfg["data"]["dataset_type"] = "sliding_window"
        with patch("src.data.sliding_window_dataset.VLNSlidingWindowDataset") as MockSW:
            MockSW.return_value = "sw"
            result = build_dataset(minimal_cfg, split="train")
            assert result == "sw"
            MockSW.assert_called_once()

    def test_build_dataset_dispatches_trajectory(self, minimal_cfg):
        """build_dataset dispatches to trajectory when dataset_type matches."""
        minimal_cfg["data"]["dataset_type"] = "trajectory"
        with patch("src.data.trajectory_dataset.VLNTrajectoryDataset") as MockTD:
            MockTD.return_value = "td"
            result = build_dataset(minimal_cfg, split="val")
            assert result == "td"
            MockTD.assert_called_once()

    def test_override_propagates(self, minimal_cfg):
        """Override kwargs take precedence over config values."""
        with patch("src.data.sliding_window_dataset.VLNSlidingWindowDataset") as MockDS:
            MockDS.return_value = "mock"
            build_sliding_window_dataset(
                minimal_cfg, split="val",
                samples_per_clip=99,
                defer_heatmap_to_gpu=True,
            )
            call_kwargs = MockDS.call_args[1]
            assert call_kwargs["samples_per_clip"] == 99
            assert call_kwargs["defer_heatmap_to_gpu"] is True

    def test_override_root(self, minimal_cfg):
        """The root parameter can be overridden."""
        with patch("src.data.sliding_window_dataset.VLNSlidingWindowDataset") as MockDS:
            MockDS.return_value = "mock"
            build_sliding_window_dataset(minimal_cfg, root="/other/path")
            assert MockDS.call_args[1]["root"] == "/other/path"

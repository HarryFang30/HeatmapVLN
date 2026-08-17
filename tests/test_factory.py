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
            assert call_kwargs["enable_augmentation"] is False

    def test_build_sliding_window_explicitly_enables_augmentation(self, minimal_cfg):
        minimal_cfg["data"]["sliding_window"]["enable_augmentation"] = True
        with patch("src.data.sliding_window_dataset.VLNSlidingWindowDataset") as mock_ds:
            build_sliding_window_dataset(minimal_cfg, split="train")

        assert mock_ds.call_args.kwargs["enable_augmentation"] is True

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
            assert call_kwargs["compute_aligned_native_pixel_goal"] is False

    def test_build_trajectory_aligned_native_projection_is_explicit_opt_in(self, minimal_cfg):
        minimal_cfg["data"]["dataset_type"] = "trajectory"
        minimal_cfg["data"]["trajectory"]["compute_aligned_native_pixel_goal"] = True
        with patch("src.data.trajectory_dataset.VLNTrajectoryDataset") as mock_ds:
            build_trajectory_dataset(minimal_cfg, split="train")

        assert mock_ds.call_args.kwargs["compute_aligned_native_pixel_goal"] is True

    def test_build_trajectory_can_skip_duplicate_history_frames(self, minimal_cfg):
        minimal_cfg["data"]["trajectory"]["load_single_view_history_frames"] = False
        with patch("src.data.trajectory_dataset.VLNTrajectoryDataset") as mock_ds:
            build_trajectory_dataset(minimal_cfg, split="train")

        assert mock_ds.call_args.kwargs["load_single_view_history_frames"] is False

    def test_factory_accepts_deprecated_history_key(self, minimal_cfg):
        minimal_cfg["data"]["sliding_window"]["load_history_frames"] = False
        with patch("src.data.sliding_window_dataset.VLNSlidingWindowDataset") as mock_ds:
            with __import__("pytest").warns(
                FutureWarning, match="load_single_view_history_frames"
            ):
                build_sliding_window_dataset(minimal_cfg, split="train")

        assert mock_ds.call_args.kwargs["load_single_view_history_frames"] is False

    def test_factory_rejects_conflicting_history_keys(self, minimal_cfg):
        section = minimal_cfg["data"]["trajectory"]
        section["load_single_view_history_frames"] = False
        section["load_history_frames"] = True
        with patch("src.data.trajectory_dataset.VLNTrajectoryDataset"):
            with __import__("pytest").raises(
                ValueError, match="Conflicting history settings"
            ):
                build_trajectory_dataset(minimal_cfg, split="train")

    def test_deprecated_history_override_takes_precedence(self, minimal_cfg):
        minimal_cfg["data"]["sliding_window"][
            "load_single_view_history_frames"
        ] = True
        with patch("src.data.sliding_window_dataset.VLNSlidingWindowDataset") as mock_ds:
            with __import__("pytest").warns(FutureWarning):
                build_sliding_window_dataset(
                    minimal_cfg,
                    split="train",
                    load_history_frames=False,
                )

        assert mock_ds.call_args.kwargs["load_single_view_history_frames"] is False

    def test_panoramic_dummy_keeps_full_history_when_single_view_is_disabled(self):
        from src.data.sliding_window_dataset import VLNSlidingWindowDataset

        dataset = object.__new__(VLNSlidingWindowDataset)
        dataset.image_size = (8, 6)
        dataset.hm_size = (4, 3)
        dataset.num_history_sample = 5
        dataset.load_single_view_history_frames = False
        dataset.defer_heatmap_to_gpu = False
        dataset._is_panoramic = True

        sample = dataset._get_dummy_sample()
        assert sample["history_frames"].shape == (1, 3, 6, 8)
        assert sample["history_panoramas"].shape == (5, 4, 3, 6, 8)

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

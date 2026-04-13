"""Unit tests for data sub-modules: augmentation, heatmap geometry, trajectory utils."""

import numpy as np
import pytest

from src.data.augmentation import (
    ColorJitterAugmentation,
    GaussianNoiseAugmentation,
    InternNavStyleAugmentation,
)
from src.data.heatmap_geometry import (
    compute_adaptive_sigma_pinhole,
    compute_history_heatmap,
    draw_gaussian_point,
    project_point_pinhole,
)
from src.data.trajectory_utils import (
    apply_trajectory_augmentation,
    interpolate_and_resample_trajectory,
    smooth_and_resample_trajectory,
    xy_to_delta_xyt,
)


class TestAugmentation:
    def test_color_jitter_preserves_shape(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        aug = ColorJitterAugmentation(p=1.0)
        out = aug(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_color_jitter_noop_when_disabled(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        aug = ColorJitterAugmentation(p=0.0)
        out = aug(img)
        np.testing.assert_array_equal(out, img)

    def test_gaussian_noise_noop_when_disabled(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        aug = GaussianNoiseAugmentation(p=0.0)
        out = aug(img)
        np.testing.assert_array_equal(out, img)

    def test_gaussian_noise_preserves_shape(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        aug = GaussianNoiseAugmentation(p=1.0)
        out = aug(img)
        assert out.shape == img.shape

    def test_internnav_style_preserves_shape(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        aug = InternNavStyleAugmentation(p=1.0)
        out = aug(img)
        assert out.shape == img.shape


class TestHeatmapGeometry:
    def test_draw_gaussian_peak(self):
        hm = np.zeros((64, 64), dtype=np.float32)
        draw_gaussian_point(hm, (32.0, 32.0), sigma=3.0, peak_value=1.0)
        assert hm[32, 32] == pytest.approx(1.0, abs=0.01)
        assert hm.max() <= 1.0

    def test_draw_gaussian_out_of_bounds(self):
        hm = np.zeros((64, 64), dtype=np.float32)
        draw_gaussian_point(hm, (-10.0, -10.0), sigma=2.0)
        assert hm.sum() == 0.0

    def test_compute_history_heatmap_empty(self):
        pose = np.eye(4, dtype=np.float32)
        hm, vis = compute_history_heatmap([], pose, None, hm_size=(64, 64))
        assert hm.shape == (64, 64)
        assert hm.sum() == 0.0
        assert vis == 0

    def test_compute_adaptive_sigma_range(self):
        sigma = compute_adaptive_sigma_pinhole(5.0, 320.0, min_sigma=1.0, max_sigma=8.0)
        assert 1.0 <= sigma <= 8.0

    def test_project_behind_camera(self):
        K = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]], dtype=np.float32)
        result = project_point_pinhole(np.array([0, 0, 1.0]), K, 640, 480)
        assert result is None

    def test_project_in_front(self):
        K = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]], dtype=np.float32)
        result = project_point_pinhole(np.array([0, 0, -5.0]), K, 640, 480)
        assert result is not None
        _u, _v, z = result
        assert z == pytest.approx(5.0)


class TestTrajectoryUtils:
    def test_xy_to_delta_xyt_shape(self):
        xy = np.array([[0, 0], [1, 0], [2, 1], [3, 2]], dtype=np.float32)
        delta = xy_to_delta_xyt(xy)
        assert delta.shape == (3, 3)

    def test_xy_to_delta_xyt_single_point(self):
        xy = np.array([[5, 3]], dtype=np.float32)
        delta = xy_to_delta_xyt(xy)
        assert delta.shape == (0, 3)

    def test_smooth_and_resample(self):
        pts = np.array([[0, 0], [1, 0], [2, 1], [3, 3]], dtype=np.float32)
        out = smooth_and_resample_trajectory(pts, sample_length=10)
        assert out.shape == (10, 2)

    def test_smooth_and_resample_single_point(self):
        pts = np.array([[5, 3]], dtype=np.float32)
        out = smooth_and_resample_trajectory(pts, sample_length=5)
        assert out.shape == (5, 2)
        np.testing.assert_array_equal(out[0], [5, 3])

    def test_apply_augmentation_shape(self):
        traj = np.random.randn(24, 3).astype(np.float32)
        out = apply_trajectory_augmentation(traj, p=1.0)
        assert out.shape == (24, 3)

    def test_apply_augmentation_noop(self):
        traj = np.random.randn(24, 3).astype(np.float32)
        out = apply_trajectory_augmentation(traj, p=0.0)
        np.testing.assert_array_equal(out, traj)

    def test_interpolate_and_resample(self):
        abs_traj = np.column_stack([
            np.linspace(0, 5, 20),
            np.linspace(0, 3, 20),
            np.zeros(20),
        ]).astype(np.float32)
        resampled_abs, resampled_rel = interpolate_and_resample_trajectory(abs_traj, predict_step_num=16)
        assert resampled_abs.shape == (17, 2)
        assert resampled_rel.shape == (16, 3)

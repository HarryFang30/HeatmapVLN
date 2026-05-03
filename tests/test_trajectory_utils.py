import numpy as np

from src.data.trajectory_utils import (
    interpolate_and_resample_trajectory,
    smooth_and_resample_trajectory,
)


def test_smooth_resample_handles_duplicate_points():
    points = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.5, 0.0],
            [1.0, 0.2],
            [1.0, 0.2],
            [1.5, 0.4],
        ],
        dtype=np.float32,
    )

    resampled = smooth_and_resample_trajectory(points, sample_length=8, interval=0.1)

    assert resampled.shape == (8, 2)
    assert np.isfinite(resampled).all()
    np.testing.assert_allclose(resampled[0], [0.0, 0.0], atol=1e-6)


def test_interpolate_resample_handles_stationary_segments():
    absolute_trajectory = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.35, 0.0, 0.0],
            [0.35, 0.0, 0.0],
            [0.75, 0.1, 0.0],
            [0.75, 0.1, 0.0],
            [1.2, 0.2, 0.0],
        ],
        dtype=np.float32,
    )

    resampled_xy, delta_xyt = interpolate_and_resample_trajectory(
        absolute_trajectory,
        predict_step_num=4,
        action_scale=4.0,
    )

    assert resampled_xy.shape == (5, 2)
    assert delta_xyt.shape == (4, 3)
    assert np.isfinite(resampled_xy).all()
    assert np.isfinite(delta_xyt).all()


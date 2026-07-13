import numpy as np

from src.data.trajectory_utils import (
    get_trajectory_relative_to_frame,
    interpolate_and_resample_trajectory,
    smooth_and_resample_trajectory,
)


def _habitat_pose(yaw_rad=0.0, translation=(0.0, 0.0, 0.0)):
    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )
    pose[:3, 3] = np.asarray(translation, dtype=np.float64)
    return pose


def test_habitat_negative_z_forward_maps_to_internnav_positive_x():
    poses = np.stack(
        [
            _habitat_pose(),
            _habitat_pose(translation=(0.0, 0.0, -0.25)),
        ]
    )

    relative = get_trajectory_relative_to_frame(
        poses,
        camera_forward_axis="-z",
    )

    np.testing.assert_allclose(relative[1], [0.25, 0.0, 0.0], atol=1e-7)


def test_habitat_left_turn_keeps_left_positive_lateral_axis():
    yaw = np.deg2rad(15.0)
    left_rotation = _habitat_pose(yaw_rad=yaw)[:3, :3]
    forward_after_left = left_rotation @ np.array([0.0, 0.0, -0.25])
    poses = np.stack(
        [
            _habitat_pose(),
            _habitat_pose(yaw_rad=yaw),
            _habitat_pose(yaw_rad=yaw, translation=forward_after_left),
        ]
    )

    relative = get_trajectory_relative_to_frame(
        poses,
        camera_forward_axis="-z",
    )

    assert relative[1, 2] > 0.0
    assert relative[2, 0] > 0.0
    assert relative[2, 1] > 0.0
    np.testing.assert_allclose(relative[2, 2], yaw, atol=1e-7)


def test_trajectory_forward_axis_rejects_unknown_value():
    with np.testing.assert_raises_regex(ValueError, "camera_forward_axis"):
        get_trajectory_relative_to_frame(
            np.stack([np.eye(4), np.eye(4)]),
            camera_forward_axis="negative-z",
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

from __future__ import annotations

import numpy as np
import pytest

from src.vo.amb3r_pose import (
    fit_global_translation_scale,
    history_rel_poses_from_amb3r,
    opencv_c2w_to_habitat_c2w,
)


_S = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def _habitat_c2w(*, x: float = 0.0, z: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float32,
    )
    pose[0, 3] = x
    pose[2, 3] = z
    return pose


def _to_opencv(pose_habitat: np.ndarray) -> np.ndarray:
    return pose_habitat @ _S


def test_opencv_to_habitat_basis_change_round_trips_known_poses() -> None:
    habitat = np.stack(
        [_habitat_c2w(z=-1.0), _habitat_c2w(x=-2.0, yaw_deg=90.0)]
    )
    opencv = np.stack([_to_opencv(pose) for pose in habitat])
    np.testing.assert_allclose(
        opencv_c2w_to_habitat_c2w(opencv), habitat, rtol=0.0, atol=1e-6
    )


def test_amb3r_history_adapter_reuses_heatmap_forward_left_yaw_contract() -> None:
    habitat = np.stack(
        [
            _habitat_c2w(z=-1.0),
            _habitat_c2w(x=-1.0),
            _habitat_c2w(yaw_deg=90.0),
            _habitat_c2w(),
        ]
    )
    opencv = np.stack([_to_opencv(pose) for pose in habitat])
    actual = history_rel_poses_from_amb3r(opencv, [0, 1, 2], 3)
    expected = np.asarray(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_fixed_translation_scale_only_changes_xy() -> None:
    habitat = np.stack([_habitat_c2w(z=-1.0, yaw_deg=90.0), _habitat_c2w()])
    opencv = np.stack([_to_opencv(pose) for pose in habitat])
    native = history_rel_poses_from_amb3r(opencv, [0], 1)
    scaled = history_rel_poses_from_amb3r(opencv, [0], 1, translation_scale=2.5)
    np.testing.assert_allclose(scaled[:, :2], native[:, :2] * 2.5)
    np.testing.assert_allclose(scaled[:, 2:], native[:, 2:])


def test_amb3r_global_gauge_cancels_from_relative_pose_tokens() -> None:
    habitat = np.stack([_habitat_c2w(z=-1.0), _habitat_c2w()])
    opencv = np.stack([_to_opencv(pose) for pose in habitat])
    gauge = _to_opencv(_habitat_c2w(x=3.0, z=7.0, yaw_deg=37.0))
    gauged = gauge[None] @ opencv
    expected = history_rel_poses_from_amb3r(opencv, [0], 1)
    actual = history_rel_poses_from_amb3r(gauged, [0], 1)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_global_scale_fit_uses_one_dataset_level_scalar() -> None:
    predicted = np.asarray(
        [[0.5, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]], dtype=np.float32
    )
    target = predicted.copy()
    target[:, :2] *= 2.0
    assert fit_global_translation_scale(predicted, target) == pytest.approx(2.0)


def test_future_history_index_is_rejected() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0)
    with pytest.raises(ValueError, match="no later"):
        history_rel_poses_from_amb3r(poses, [3], 2)


def test_repeated_current_frame_has_identity_relative_pose() -> None:
    habitat = np.stack([_habitat_c2w(z=-1.0), _habitat_c2w(x=2.0)])
    opencv = np.stack([_to_opencv(pose) for pose in habitat])
    actual = history_rel_poses_from_amb3r(opencv, [0, 1, 1], 1)
    np.testing.assert_allclose(
        actual[-2:],
        np.asarray([[0.0, 0.0, 1.0, 0.0]] * 2, dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )

"""Regression locks for Habitat c2w -> heatmap trajectory semantics."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.trajectory_utils import compute_history_rel_poses
from src.models.heatmap.single_view_panorama_conditioner import (
    VIEW_ANGLES_DEGREES,
    VIEW_NAMES,
)


def _habitat_c2w(*, x: float = 0.0, z: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    """Create a Habitat camera-to-world pose (camera forward is local -Z)."""
    yaw = np.deg2rad(yaw_deg)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.asarray(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=np.float32,
    )
    pose[0, 3] = x
    pose[2, 3] = z
    return pose


def test_minus_z_fields_are_forward_left_and_left_positive_yaw() -> None:
    current = _habitat_c2w()
    histories = [
        _habitat_c2w(z=-1.0),
        _habitat_c2w(z=1.0),
        _habitat_c2w(x=-1.0),
        _habitat_c2w(x=1.0),
        _habitat_c2w(yaw_deg=90.0),
        _habitat_c2w(yaw_deg=-90.0),
    ]
    actual = compute_history_rel_poses(
        histories,
        current,
        camera_forward_axis="-z",
    )
    expected = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],   # forward
            [-1.0, 0.0, 1.0, 0.0],  # back
            [0.0, 1.0, 1.0, 0.0],   # left
            [0.0, -1.0, 1.0, 0.0],  # right
            [0.0, 0.0, 0.0, 1.0],   # left yaw
            [0.0, 0.0, 0.0, -1.0],  # right yaw
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_legacy_plus_z_is_the_deterministic_mirror_not_the_physical_fields() -> None:
    current = _habitat_c2w()
    histories = [_habitat_c2w(z=-1.0), _habitat_c2w(yaw_deg=90.0)]
    physical = compute_history_rel_poses(
        histories,
        current,
        camera_forward_axis="-z",
    )
    legacy = compute_history_rel_poses(
        histories,
        current,
        camera_forward_axis="+z",
    )
    # old_pose = [-new_forward, new_left, new_cos, -new_sin]
    np.testing.assert_allclose(
        legacy,
        physical * np.asarray([-1.0, 1.0, 1.0, -1.0], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )


def test_four_decoder_slots_lock_front_right_back_left_yaw_signs() -> None:
    assert VIEW_NAMES == ("front", "right", "back", "left")
    assert VIEW_ANGLES_DEGREES == (0.0, -90.0, 180.0, 90.0)


def test_unknown_camera_forward_axis_is_rejected() -> None:
    with pytest.raises(ValueError, match="camera_forward_axis"):
        compute_history_rel_poses(
            [_habitat_c2w(z=-1.0)],
            _habitat_c2w(),
            camera_forward_axis="z",
        )

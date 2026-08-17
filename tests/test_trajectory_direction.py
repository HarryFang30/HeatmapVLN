import numpy as np
import pytest

from src.utils.trajectory_direction import (
    align_trajectory_endpoint_heading,
    pano_recenter_turn,
    pairwise_representation_stats,
    summarize_direction_response,
    view_pixel_target_angle_deg,
)


@pytest.mark.parametrize(
    ("view_id", "direction", "count"),
    [
        ("front", None, 0),
        ("right", "right", 6),
        ("back", "right", 12),
        ("left", "left", 6),
    ],
)
def test_pano_recenter_turn_for_habitat_15_degrees(view_id, direction, count):
    assert pano_recenter_turn(view_id, turn_angle_deg=15.0) == (direction, count)


def test_pano_recenter_turn_rejects_residual_heading():
    with pytest.raises(ValueError, match="not divisible"):
        pano_recenter_turn("right", turn_angle_deg=14.0)


def _constant_delta(dx: float, dy: float, *, candidates: int = 4, steps: int = 3) -> np.ndarray:
    result = np.zeros((candidates, steps, 3), dtype=np.float32)
    result[:, :, 0] = dx
    result[:, :, 1] = dy
    return result


def test_direction_response_matches_front_and_right_axes():
    front = summarize_direction_response(
        _constant_delta(1.0, 0.0),
        view_id="front",
        action_scale=1.0,
    )
    right = summarize_direction_response(
        _constant_delta(0.0, -1.0),
        view_id="right",
        action_scale=1.0,
    )

    assert front["candidate_within_45_rate"] == 1.0
    assert front["mean_endpoint_angle_error_deg"] == 0.0
    assert right["candidate_within_45_rate"] == 1.0
    assert right["mean_endpoint_angle_deg"] == -90.0


def test_direction_response_exposes_mean_cancellation():
    deltas = _constant_delta(0.0, -1.0, candidates=2)
    deltas[1, :, 1] = 1.0

    stats = summarize_direction_response(
        deltas,
        view_id="right",
        action_scale=1.0,
    )

    assert stats["candidate_within_45_rate"] == 0.5
    assert stats["mean_endpoint_direct_m"] == 0.0
    assert stats["mean_endpoint_angle_error_deg"] == 180.0


def test_pairwise_representation_stats_reports_view_separation():
    stats = pairwise_representation_stats({
        "front": np.array([1.0, 0.0]),
        "right": np.array([0.0, 1.0]),
        "back": np.array([1.0, 0.0]),
    })

    assert len(stats["pairs"]) == 3
    assert 0.0 < stats["cosine_mean"] < 1.0
    assert stats["relative_l2_mean"] > 0.0


def test_view_pixel_target_angle_includes_horizontal_pixel_offset():
    assert view_pixel_target_angle_deg("front", [128, 128], [256, 256]) == 0.0
    assert view_pixel_target_angle_deg("front", [192, 128], [256, 256]) == -22.5
    assert view_pixel_target_angle_deg("right", [64, 128], [256, 256]) == -67.5
    assert view_pixel_target_angle_deg("back", [0, 128], [256, 256]) == -135.0


def test_endpoint_heading_alignment_preserves_path_shape_and_length():
    path = np.array([[0.0, 0.0], [1.0, 0.25], [2.0, 0.0]], dtype=np.float32)
    aligned, rotation_deg = align_trajectory_endpoint_heading(
        path,
        target_angle_deg=-90.0,
    )

    assert rotation_deg == -90.0
    assert np.allclose(aligned[-1], [0.0, -2.0], atol=1.0e-7)
    assert np.allclose(
        np.linalg.norm(np.diff(aligned, axis=0), axis=1),
        np.linalg.norm(np.diff(path, axis=0), axis=1),
    )


def test_endpoint_heading_alignment_leaves_zero_path_unchanged():
    path = np.zeros((3, 2), dtype=np.float32)
    aligned, rotation_deg = align_trajectory_endpoint_heading(path, target_angle_deg=180.0)

    assert rotation_deg == 0.0
    assert np.array_equal(aligned, path)

import numpy as np

from src.utils.trajectory_direction import (
    pairwise_representation_stats,
    summarize_direction_response,
)


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

from __future__ import annotations

import numpy as np

from src.models.action.treatment_spec import (
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STOP,
    TRAJECTORY_SELECTION_CHOICES,
    TrajectoryPostprocessConfig,
    build_treatment_spec,
)


def _straight_deltas(
    *, samples: int = 4, steps: int = 8, scaled_step: float = 1.0
) -> np.ndarray:
    values = np.zeros((samples, steps, 3), dtype=np.float32)
    values[:, :, 0] = scaled_step
    return values


def test_all_trajectory_selection_modes_are_deterministic() -> None:
    values = _straight_deltas(samples=4)
    values[1, :, 1] = 0.25
    values[2, :, 0] = 0.5
    values[3, :, 1] = -0.25
    for selection in TRAJECTORY_SELECTION_CHOICES:
        config = TrajectoryPostprocessConfig(
            num_sample_trajs=4,
            trajectory_selection=selection,
        )
        assert build_treatment_spec(values, config) == build_treatment_spec(
            values.copy(), config
        )


def test_short_path_stop_padding_means_local_replan() -> None:
    spec = build_treatment_spec(
        _straight_deltas(samples=1, steps=1),
        TrajectoryPostprocessConfig(num_sample_trajs=1),
    )
    assert spec.raw_discrete_actions == (ACTION_FORWARD,)
    assert spec.padded_capped_actions == (
        ACTION_FORWARD,
        ACTION_STOP,
        ACTION_STOP,
        ACTION_STOP,
    )
    assert spec.response_actions == spec.padded_capped_actions
    assert spec.habitat_actions == (ACTION_FORWARD,)
    assert spec.execute_len == 1
    assert spec.end_reason == "local_stop_replan"
    assert spec.replan_after is True
    assert spec.trigger_anti_deadlock is False


def test_first_stop_becomes_one_left_anti_deadlock_action() -> None:
    values = np.zeros((1, 8, 3), dtype=np.float32)
    spec = build_treatment_spec(
        values,
        TrajectoryPostprocessConfig(num_sample_trajs=1),
    )
    assert spec.raw_discrete_actions == (ACTION_STOP,)
    assert spec.padded_capped_actions == (ACTION_STOP,) * 4
    assert spec.response_actions == (ACTION_LEFT,)
    assert spec.habitat_actions == (ACTION_LEFT,)
    assert spec.execute_len == 1
    assert spec.end_reason == "anti_deadlock_replan"
    assert spec.trigger_anti_deadlock is True


def test_long_queue_exhausts_at_four_and_replans() -> None:
    spec = build_treatment_spec(
        _straight_deltas(samples=1, steps=16),
        TrajectoryPostprocessConfig(num_sample_trajs=1),
    )
    assert spec.response_actions == (ACTION_FORWARD,) * 4
    assert spec.habitat_actions == (ACTION_FORWARD,) * 4
    assert spec.execute_len == 4
    assert spec.end_reason == "queue_exhausted_replan"


def test_action_scale_and_x_sign_are_in_the_spec_and_change_actions() -> None:
    values = _straight_deltas(samples=1, steps=4, scaled_step=1.0)
    native = build_treatment_spec(
        values,
        TrajectoryPostprocessConfig(num_sample_trajs=1, action_scale=4.0),
    )
    mirrored = build_treatment_spec(
        values,
        TrajectoryPostprocessConfig(
            num_sample_trajs=1,
            action_scale=4.0,
            trajectory_x_sign=-1.0,
        ),
    )
    assert native.action_scale == 4.0
    assert native.trajectory_x_sign == 1.0
    assert ACTION_RIGHT in mirrored.raw_discrete_actions or ACTION_LEFT in mirrored.raw_discrete_actions
    assert native != mirrored


def test_heading_alignment_is_applied_before_discretization() -> None:
    values = _straight_deltas(samples=1, steps=8)
    spec = build_treatment_spec(
        values,
        TrajectoryPostprocessConfig(
            num_sample_trajs=1,
            target_heading_deg=90.0,
        ),
    )
    assert np.isclose(spec.heading_rotation_deg, 90.0)
    assert spec.target_heading_deg == 90.0
    assert spec.raw_discrete_actions[0] == ACTION_LEFT


def test_json_payload_contains_every_control_boundary() -> None:
    spec = build_treatment_spec(
        _straight_deltas(samples=1),
        TrajectoryPostprocessConfig(num_sample_trajs=1),
    ).to_dict()
    assert {
        "trajectory_selection",
        "selected_trajectory_index",
        "action_scale",
        "trajectory_x_sign",
        "target_heading_deg",
        "heading_rotation_deg",
        "raw_discrete_actions",
        "padded_capped_actions",
        "response_actions",
        "habitat_actions",
        "execute_len",
        "end_reason",
        "replan_after",
        "trigger_anti_deadlock",
    }.issubset(spec)

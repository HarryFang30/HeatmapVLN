import pytest

from scripts.evaluation.closed_loop_guard import (
    STOP_ACCEPT,
    STOP_CONTINUE,
    STOP_PROBE,
    ClosedLoopGuard,
    ClosedLoopGuardConfig,
)


FORWARD = 1
LEFT = 2
RIGHT = 3


def _guard(**overrides) -> ClosedLoopGuard:
    config = ClosedLoopGuardConfig(**overrides)
    guard = ClosedLoopGuard(
        config,
        forward_action=FORWARD,
        left_action=LEFT,
        right_action=RIGHT,
    )
    guard.reset_episode((0.0, 0.0, 0.0))
    return guard


def test_action_chunk_limits_each_system2_plan():
    guard = _guard(action_chunk_size=2)

    assert guard.limit_actions([1, 2, 3, 1]) == [1, 2]
    assert guard.limit_actions([1]) == [1]


def test_stop_requires_consecutive_votes_and_resets_on_trajectory():
    guard = _guard(stop_confirmations=2)

    assert guard.observe_system2_terminal(True) == STOP_PROBE
    assert guard.next_stop_probe_action() == LEFT
    assert guard.observe_system2_terminal(False) == STOP_CONTINUE
    assert guard.observe_system2_terminal(True) == STOP_PROBE
    assert guard.next_stop_probe_action() == RIGHT
    assert guard.observe_system2_terminal(True) == STOP_ACCEPT


def test_original_single_vote_stop_behavior_is_preserved():
    guard = _guard(stop_confirmations=1)

    assert guard.observe_system2_terminal(True) == STOP_ACCEPT


def test_collision_recovery_alternates_turn_direction():
    guard = _guard(
        loop_guard_enabled=True,
        collision_forward_limit=3,
        recovery_cooldown_steps=0,
    )

    assert guard.observe_action(FORWARD, (0, 0, 0), (0, 0, 0)) is None
    assert guard.observe_action(FORWARD, (0, 0, 0), (0, 0, 0)) is None
    first = guard.observe_action(FORWARD, (0, 0, 0), (0, 0, 0))
    assert first is not None
    assert first.reason == "collision"
    assert first.actions == (LEFT, LEFT, LEFT)

    guard.observe_action(FORWARD, (0, 0, 0), (0, 0, 0))
    guard.observe_action(FORWARD, (0, 0, 0), (0, 0, 0))
    second = guard.observe_action(FORWARD, (0, 0, 0), (0, 0, 0))
    assert second is not None
    assert second.actions == (RIGHT, RIGHT, RIGHT)


def test_motion_loop_detects_return_after_real_travel():
    guard = _guard(
        loop_guard_enabled=True,
        collision_forward_limit=99,
        motion_window_steps=4,
        motion_min_path_m=3.5,
        motion_max_net_m=0.1,
        recovery_cooldown_steps=0,
    )
    points = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
    ]

    event = None
    for before, after in zip(points, points[1:]):
        event = guard.observe_action(FORWARD, before, after)

    assert event is not None
    assert event.reason.startswith("motion_loop(")


def test_plan_loop_requires_dominant_view_and_spatial_return():
    guard = _guard(
        loop_guard_enabled=True,
        plan_window_calls=4,
        plan_view_dominance=1.0,
        plan_min_path_m=3.0,
        plan_max_net_m=0.1,
        recovery_cooldown_steps=0,
    )
    points = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
    ]

    event = None
    for point in points:
        event = guard.observe_plan("right", point)

    assert event is not None
    assert event.reason.startswith("plan_loop(")


def test_straight_progress_does_not_trigger_loop_recovery():
    guard = _guard(
        loop_guard_enabled=True,
        collision_forward_limit=99,
        motion_window_steps=4,
        motion_min_path_m=2.0,
        motion_max_net_m=0.75,
    )

    for index in range(1, 8):
        event = guard.observe_action(
            FORWARD,
            (float(index - 1), 0.0, 0.0),
            (float(index), 0.0, 0.0),
        )
        assert event is None


def test_invalid_guard_configuration_fails_closed():
    with pytest.raises(ValueError, match="action_chunk_size"):
        ClosedLoopGuardConfig(action_chunk_size=0)
    with pytest.raises(ValueError, match="stop_confirmations"):
        ClosedLoopGuardConfig(stop_confirmations=0)
    with pytest.raises(ValueError, match="plan_view_dominance"):
        ClosedLoopGuardConfig(plan_view_dominance=0.5)

import pytest

from scripts.evaluation.closed_loop_guard import (
    STOP_ACCEPT,
    STOP_CONTINUE,
    STOP_PROBE,
    ClosedLoopGuard,
    ClosedLoopGuardConfig,
    should_trust_temporal_terminal,
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


def test_trusted_terminal_bypasses_confirmation_and_resets_pending_vote():
    guard = _guard(
        stop_confirmations=2,
        stop_confirmation_max_gap_calls=1,
    )

    assert guard.observe_system2_terminal(True) == STOP_PROBE
    assert guard.stop_votes == 1
    assert (
        guard.observe_system2_terminal(True, trusted_terminal=True)
        == STOP_ACCEPT
    )
    assert guard.stop_votes == 0
    assert guard.stop_gap_calls == 0


def test_trusted_terminal_rejects_nonterminal_output():
    guard = _guard(stop_confirmations=2)

    with pytest.raises(ValueError, match="trusted_terminal requires terminal=True"):
        guard.observe_system2_terminal(False, trusted_terminal=True)


def test_temporal_terminal_trust_requires_calibrated_margin():
    assert should_trust_temporal_terminal(
        enabled=True,
        decision="temporal_confirms_original_stop",
        observed_margin=0.0123,
        min_margin=0.005,
    )
    assert not should_trust_temporal_terminal(
        enabled=True,
        decision="temporal_confirms_original_stop",
        observed_margin=0.0016,
        min_margin=0.005,
    )
    assert not should_trust_temporal_terminal(
        enabled=False,
        decision="temporal_confirms_original_stop",
        observed_margin=0.5,
        min_margin=0.005,
    )
    assert not should_trust_temporal_terminal(
        enabled=True,
        decision="hybrid_static_adds_stop",
        observed_margin=0.5,
        min_margin=0.005,
    )


def test_temporal_terminal_trust_fails_closed_without_margin():
    assert not should_trust_temporal_terminal(
        enabled=True,
        decision="temporal_confirms_original_stop",
        observed_margin=None,
        min_margin=0.005,
    )
    with pytest.raises(ValueError, match="min_margin"):
        should_trust_temporal_terminal(
            enabled=True,
            decision="temporal_confirms_original_stop",
            observed_margin=0.1,
            min_margin=float("nan"),
        )


def test_stop_confirmation_does_not_mix_terminal_sources():
    guard = _guard(
        stop_confirmations=2,
        stop_confirmation_max_gap_calls=2,
        stop_confirmation_view_sweep=True,
    )

    assert guard.observe_system2_terminal(
        True,
        terminal_source="temporal_confirms_original_stop",
    ) == STOP_PROBE
    assert guard.stop_vote_source == "temporal_confirms_original_stop"
    assert guard.observe_system2_terminal(False) == STOP_PROBE
    assert guard.observe_system2_terminal(
        True,
        terminal_source="hybrid_static_adds_stop",
    ) == STOP_PROBE
    assert guard.stop_votes == 1
    assert guard.stop_vote_source == "hybrid_static_adds_stop"
    assert guard.observe_system2_terminal(
        True,
        terminal_source="hybrid_static_adds_stop",
    ) == STOP_ACCEPT
    assert guard.stop_vote_source is None


def test_terminal_source_rejects_nonterminal_observation():
    guard = _guard(stop_confirmations=2)

    with pytest.raises(ValueError, match="terminal_source requires terminal=True"):
        guard.observe_system2_terminal(
            False,
            terminal_source="hybrid_static_adds_stop",
        )


def test_stop_confirmation_can_bridge_one_nonterminal_replan():
    guard = _guard(
        stop_confirmations=2,
        stop_confirmation_max_gap_calls=1,
    )

    assert guard.observe_system2_terminal(True) == STOP_PROBE
    assert guard.observe_system2_terminal(False) == STOP_CONTINUE
    assert guard.stop_votes == 1
    assert guard.stop_gap_calls == 1
    assert guard.observe_system2_terminal(True) == STOP_ACCEPT


def test_stop_confirmation_view_sweep_probes_without_executing_gap_plan():
    guard = _guard(
        stop_confirmations=2,
        stop_confirmation_max_gap_calls=1,
        stop_confirmation_view_sweep=True,
        stop_probe_turn="left",
    )

    assert guard.observe_system2_terminal(True) == STOP_PROBE
    assert guard.next_stop_probe_action() == LEFT
    assert guard.observe_system2_terminal(False) == STOP_PROBE
    assert guard.stop_votes == 1
    assert guard.stop_gap_calls == 1
    assert guard.next_stop_probe_action() == LEFT
    assert guard.observe_system2_terminal(True) == STOP_ACCEPT


def test_stop_confirmation_view_sweep_resumes_after_gap_budget():
    guard = _guard(
        stop_confirmations=2,
        stop_confirmation_max_gap_calls=1,
        stop_confirmation_view_sweep=True,
    )

    assert guard.observe_system2_terminal(True) == STOP_PROBE
    assert guard.observe_system2_terminal(False) == STOP_PROBE
    assert guard.observe_system2_terminal(False) == STOP_CONTINUE
    assert guard.stop_votes == 0
    assert guard.stop_gap_calls == 0


def test_stop_confirmation_resets_after_gap_budget_is_exhausted():
    guard = _guard(
        stop_confirmations=2,
        stop_confirmation_max_gap_calls=1,
    )

    assert guard.observe_system2_terminal(True) == STOP_PROBE
    assert guard.observe_system2_terminal(False) == STOP_CONTINUE
    assert guard.observe_system2_terminal(False) == STOP_CONTINUE
    assert guard.stop_votes == 0
    assert guard.stop_gap_calls == 0
    assert guard.observe_system2_terminal(True) == STOP_PROBE


def test_high_confidence_stop_bypasses_low_confidence_confirmation():
    guard = _guard(
        stop_confirmations=2,
        stop_high_confidence_threshold=0.8,
    )

    assert (
        guard.observe_system2_terminal(True, stop_probability=0.9)
        == STOP_ACCEPT
    )


def test_low_confidence_stop_requires_confirmation_and_resets():
    guard = _guard(
        stop_confirmations=2,
        stop_high_confidence_threshold=0.8,
    )

    assert guard.observe_system2_terminal(True, stop_probability=0.6) == STOP_PROBE
    assert guard.observe_system2_terminal(False, stop_probability=0.1) == STOP_CONTINUE
    assert guard.observe_system2_terminal(True, stop_probability=0.6) == STOP_PROBE
    assert guard.observe_system2_terminal(True, stop_probability=0.7) == STOP_ACCEPT


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


def test_collision_recovery_follows_last_turn_and_translates():
    guard = _guard(
        loop_guard_enabled=True,
        collision_forward_limit=1,
        recovery_turns=2,
        recovery_forward_steps=2,
        recovery_follow_last_turn=True,
        recovery_cooldown_steps=12,
    )

    assert guard.observe_action(RIGHT, (0, 0, 0), (0, 0, 0)) is None
    first = guard.observe_action(FORWARD, (0, 0, 0), (0, 0, 0))
    assert first is not None
    assert first.actions == (RIGHT, RIGHT, FORWARD, FORWARD)

    # A blocked escape must still retrigger collision recovery during cooldown.
    second = guard.observe_action(FORWARD, (0, 0, 0), (0, 0, 0))
    assert second is not None
    assert second.actions == (RIGHT, RIGHT, FORWARD, FORWARD)


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
    with pytest.raises(ValueError, match="stop_confirmation_max_gap_calls"):
        ClosedLoopGuardConfig(stop_confirmation_max_gap_calls=-1)
    with pytest.raises(ValueError, match="stop_confirmation_view_sweep"):
        ClosedLoopGuardConfig(stop_confirmation_view_sweep=True)
    with pytest.raises(ValueError, match="stop_high_confidence_threshold"):
        ClosedLoopGuardConfig(
            stop_confirmations=2,
            stop_high_confidence_threshold=1.1,
        )
    with pytest.raises(ValueError, match="requires stop_confirmations"):
        ClosedLoopGuardConfig(
            stop_confirmations=1,
            stop_high_confidence_threshold=0.8,
        )
    with pytest.raises(ValueError, match="recovery_forward_steps"):
        ClosedLoopGuardConfig(recovery_forward_steps=-1)
    with pytest.raises(ValueError, match="plan_view_dominance"):
        ClosedLoopGuardConfig(plan_view_dominance=0.5)

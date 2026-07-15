"""Non-privileged execution guards for closed-loop navigation evaluation.

The guard uses only issued actions, agent self-motion, and model waypoint
outputs. It never consumes the simulator goal position or distance-to-goal.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from math import hypot
from typing import Sequence


STOP_CONTINUE = "continue"
STOP_PROBE = "probe"
STOP_ACCEPT = "accept"


@dataclass(frozen=True)
class RecoveryEvent:
    reason: str
    actions: tuple[int, ...]


@dataclass(frozen=True)
class ClosedLoopGuardConfig:
    action_chunk_size: int = 4
    stop_confirmations: int = 1
    stop_probe_turn: str = "left"
    loop_guard_enabled: bool = False
    collision_epsilon_m: float = 0.03
    collision_forward_limit: int = 3
    motion_window_steps: int = 32
    motion_min_path_m: float = 2.0
    motion_max_net_m: float = 0.75
    plan_window_calls: int = 20
    plan_view_dominance: float = 0.9
    plan_min_path_m: float = 3.0
    plan_max_net_m: float = 1.5
    recovery_turns: int = 3
    recovery_cooldown_steps: int = 12

    def __post_init__(self) -> None:
        if self.action_chunk_size < 1:
            raise ValueError("action_chunk_size must be >= 1")
        if self.stop_confirmations < 1:
            raise ValueError("stop_confirmations must be >= 1")
        if self.stop_probe_turn not in {"left", "right"}:
            raise ValueError("stop_probe_turn must be 'left' or 'right'")
        if self.collision_epsilon_m < 0.0:
            raise ValueError("collision_epsilon_m must be >= 0")
        if self.collision_forward_limit < 1:
            raise ValueError("collision_forward_limit must be >= 1")
        if self.motion_window_steps < 2:
            raise ValueError("motion_window_steps must be >= 2")
        if self.motion_min_path_m < 0.0 or self.motion_max_net_m < 0.0:
            raise ValueError("motion loop thresholds must be >= 0")
        if self.plan_window_calls < 2:
            raise ValueError("plan_window_calls must be >= 2")
        if not 0.5 < self.plan_view_dominance <= 1.0:
            raise ValueError("plan_view_dominance must be in (0.5, 1]")
        if self.plan_min_path_m < 0.0 or self.plan_max_net_m < 0.0:
            raise ValueError("plan loop thresholds must be >= 0")
        if self.recovery_turns < 1:
            raise ValueError("recovery_turns must be >= 1")
        if self.recovery_cooldown_steps < 0:
            raise ValueError("recovery_cooldown_steps must be >= 0")


def _xz(position: Sequence[float]) -> tuple[float, float]:
    if len(position) < 3:
        raise ValueError(f"Expected XYZ position, got {position!r}")
    return float(position[0]), float(position[2])


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _path_length(points: Sequence[tuple[float, float]]) -> float:
    return sum(_distance(a, b) for a, b in zip(points, points[1:]))


class ClosedLoopGuard:
    """Stateful STOP verification and self-motion loop detection."""

    def __init__(
        self,
        config: ClosedLoopGuardConfig,
        *,
        forward_action: int,
        left_action: int,
        right_action: int,
    ) -> None:
        self.config = config
        self.forward_action = int(forward_action)
        self.left_action = int(left_action)
        self.right_action = int(right_action)
        self._motion_positions: deque[tuple[float, float]] = deque(
            maxlen=config.motion_window_steps + 1
        )
        self._plan_records: deque[tuple[str, tuple[float, float]]] = deque(
            maxlen=config.plan_window_calls
        )
        self._stop_votes = 0
        self._stop_probe_count = 0
        self._collision_streak = 0
        self._cooldown_steps = 0
        self._recovery_count = 0

    @property
    def recovery_count(self) -> int:
        return self._recovery_count

    def reset_episode(self, position: Sequence[float]) -> None:
        self._motion_positions.clear()
        self._motion_positions.append(_xz(position))
        self._plan_records.clear()
        self._stop_votes = 0
        self._stop_probe_count = 0
        self._collision_streak = 0
        self._cooldown_steps = 0
        self._recovery_count = 0

    def limit_actions(self, actions: Sequence[int]) -> list[int]:
        return [int(action) for action in actions[: self.config.action_chunk_size]]

    def observe_system2_terminal(self, terminal: bool) -> str:
        if not terminal:
            self._stop_votes = 0
            return STOP_CONTINUE
        self._stop_votes += 1
        if self._stop_votes >= self.config.stop_confirmations:
            self._stop_votes = 0
            return STOP_ACCEPT
        return STOP_PROBE

    def next_stop_probe_action(self) -> int:
        prefer_left = self.config.stop_probe_turn == "left"
        use_left = prefer_left if self._stop_probe_count % 2 == 0 else not prefer_left
        self._stop_probe_count += 1
        return self.left_action if use_left else self.right_action

    def observe_action(
        self,
        action: int,
        before_position: Sequence[float],
        after_position: Sequence[float],
    ) -> RecoveryEvent | None:
        before = _xz(before_position)
        after = _xz(after_position)
        displacement = _distance(before, after)
        self._motion_positions.append(after)

        if self._cooldown_steps > 0:
            self._cooldown_steps -= 1

        if int(action) == self.forward_action:
            if displacement <= self.config.collision_epsilon_m:
                self._collision_streak += 1
            else:
                self._collision_streak = 0
        else:
            self._collision_streak = 0

        if not self.config.loop_guard_enabled or self._cooldown_steps > 0:
            return None
        if self._collision_streak >= self.config.collision_forward_limit:
            return self._make_recovery("collision", after)

        if len(self._motion_positions) == self._motion_positions.maxlen:
            points = list(self._motion_positions)
            path_m = _path_length(points)
            net_m = _distance(points[0], points[-1])
            if path_m >= self.config.motion_min_path_m and net_m <= self.config.motion_max_net_m:
                return self._make_recovery(
                    f"motion_loop(path={path_m:.2f},net={net_m:.2f})",
                    after,
                )
        return None

    def observe_plan(
        self,
        view_name: str | None,
        position: Sequence[float],
    ) -> RecoveryEvent | None:
        if not self.config.loop_guard_enabled or not view_name:
            return None
        point = _xz(position)
        self._plan_records.append((str(view_name), point))
        if self._cooldown_steps > 0 or len(self._plan_records) < self.config.plan_window_calls:
            return None

        records = list(self._plan_records)
        dominant_count = Counter(view for view, _ in records).most_common(1)[0][1]
        dominance = dominant_count / len(records)
        points = [position for _, position in records]
        path_m = _path_length(points)
        net_m = _distance(points[0], points[-1])
        if (
            dominance >= self.config.plan_view_dominance
            and path_m >= self.config.plan_min_path_m
            and net_m <= self.config.plan_max_net_m
        ):
            return self._make_recovery(
                f"plan_loop(dominance={dominance:.2f},path={path_m:.2f},net={net_m:.2f})",
                point,
            )
        return None

    def _make_recovery(
        self,
        reason: str,
        current_position: tuple[float, float],
    ) -> RecoveryEvent:
        use_left = self._recovery_count % 2 == 0
        turn_action = self.left_action if use_left else self.right_action
        actions = (turn_action,) * self.config.recovery_turns
        self._recovery_count += 1
        self._collision_streak = 0
        self._stop_votes = 0
        self._cooldown_steps = self.config.recovery_cooldown_steps
        self._motion_positions.clear()
        self._motion_positions.append(current_position)
        self._plan_records.clear()
        return RecoveryEvent(reason=reason, actions=actions)

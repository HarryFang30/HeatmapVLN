"""Fail-closed helpers for privileged STOP DAgger collection."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HistoricalFalseStopTrigger:
    """Audited historical STOP event used only to start offline recovery."""

    system2_call_index: int
    step: int
    distance_m: float
    protocol_seed: int
    source_labels: str


def prune_stop_collection_jsonl_for_resume(
    path: str | Path,
    completed_episodes: set[tuple[str, int]],
) -> tuple[int, int]:
    """Atomically discard rows from episodes not committed to progress.json."""
    labels_path = Path(path)
    if not labels_path.exists():
        return 0, 0
    if not labels_path.is_file():
        raise ValueError(f"STOP collection labels are not a file: {labels_path}")

    completed = {
        (str(scene_id), int(episode_id))
        for scene_id, episode_id in completed_episodes
    }
    kept = 0
    dropped = 0
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=labels_path.parent,
            prefix=f".{labels_path.name}.resume.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            with labels_path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        scene_id = str(row["scene_id"])
                        episode_id = int(row["episode_id"])
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise ValueError(
                            f"Invalid STOP collection row at "
                            f"{labels_path}:{line_number}"
                        ) from exc
                    if (scene_id, episode_id) not in completed:
                        dropped += 1
                        continue
                    output.write(line if line.endswith("\n") else line + "\n")
                    kept += 1
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, labels_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return kept, dropped


def parse_historical_false_stop_trigger(
    metadata: Mapping[str, Any],
    *,
    expected_protocol_seed: int,
    negative_radius_m: float,
) -> HistoricalFalseStopTrigger:
    """Fail closed when privileged cohort-trigger provenance is incomplete."""

    def required_int(name: str) -> int:
        value = metadata.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be an integer >= 0")
        return value

    call_index = required_int("historical_false_stop_system2_call_index")
    step = required_int("historical_false_stop_step")
    protocol_seed = required_int("historical_false_stop_rpc_protocol_seed")
    if protocol_seed != expected_protocol_seed:
        raise ValueError(
            "historical false-STOP seed mismatch: "
            f"cohort={protocol_seed} runtime={expected_protocol_seed}"
        )
    raw_distance = metadata.get("historical_false_stop_distance_m")
    if isinstance(raw_distance, bool) or not isinstance(raw_distance, (int, float)):
        raise ValueError("historical_false_stop_distance_m must be numeric")
    distance_m = float(raw_distance)
    if not math.isfinite(distance_m) or distance_m < float(negative_radius_m):
        raise ValueError(
            "historical false-STOP distance must be finite and at least the "
            f"negative radius ({negative_radius_m:g} m)"
        )
    source_labels = metadata.get("historical_false_stop_source_labels")
    if not isinstance(source_labels, str) or not source_labels.strip():
        raise ValueError("historical_false_stop_source_labels must be non-empty")
    return HistoricalFalseStopTrigger(
        system2_call_index=call_index,
        step=step,
        distance_m=distance_m,
        protocol_seed=protocol_seed,
        source_labels=source_labels,
    )


def validate_historical_false_stop_source(
    trigger: HistoricalFalseStopTrigger,
    *,
    scene_id: str,
    episode_id: int,
) -> dict[str, Any]:
    """Verify the cohort trigger against its immutable rollout evidence."""
    labels_path = Path(trigger.source_labels)
    if not labels_path.is_file() or labels_path.stat().st_size == 0:
        raise ValueError(f"historical false-STOP labels are missing: {labels_path}")

    matches: list[dict[str, Any]] = []
    with labels_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                str(row.get("scene_id")) == str(scene_id)
                and int(row.get("episode_id", -1)) == int(episode_id)
                and int(row.get("system2_call_index", -1))
                == trigger.system2_call_index
            ):
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            "historical false-STOP source must contain exactly one matching "
            f"row for {(scene_id, int(episode_id), trigger.system2_call_index)}; "
            f"found {len(matches)}"
        )
    row = matches[0]
    if row.get("original_terminal") is not True or row.get("stop_target") != 0:
        raise ValueError("historical source row is not a terminal negative STOP")
    if int(row.get("step", -1)) != trigger.step:
        raise ValueError("historical source row step does not match cohort metadata")
    if not math.isclose(
        float(row.get("distance_to_goal_m", float("nan"))),
        trigger.distance_m,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("historical source row distance does not match cohort metadata")
    feature_path = Path(str(row.get("path", "")))
    if not feature_path.is_file() or feature_path.stat().st_size == 0:
        raise ValueError(f"historical source feature tensor is missing: {feature_path}")

    progress_path = labels_path.parent / "progress.json"
    if not progress_path.is_file():
        raise ValueError(f"historical source progress is missing: {progress_path}")
    progress_matches: list[dict[str, Any]] = []
    with progress_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            progress = json.loads(line)
            if (
                str(progress.get("scene_id")) == str(scene_id)
                and int(progress.get("episode_id", -1)) == int(episode_id)
            ):
                progress_matches.append(progress)
    if len(progress_matches) != 1:
        raise ValueError("historical source progress row is missing or duplicated")
    if int(progress_matches[0].get("rpc_protocol_seed", -1)) != trigger.protocol_seed:
        raise ValueError("historical source progress seed does not match cohort metadata")
    return row


def should_force_continue_negative(
    *,
    collection_enabled: bool,
    force_continue_negatives: bool,
    terminal: bool,
    rollout_label: int | None,
) -> bool:
    """Return true only for a labelled far-away STOP during collection."""
    if rollout_label not in (None, 0, 1):
        raise ValueError(f"Invalid STOP rollout label: {rollout_label!r}")
    if force_continue_negatives and not collection_enabled:
        raise ValueError(
            "Forced negative continuation requires STOP feature collection"
        )
    return bool(
        collection_enabled
        and force_continue_negatives
        and terminal
        and rollout_label == 0
    )


def should_record_stop_multimodal_example(
    *,
    rollout_label: int | None,
    original_terminal: bool,
    stop_log_odds: float | None,
    regular_min_stop_log_odds: float | None,
    episode_has_record: bool,
) -> bool:
    """Keep STOP labels, hard regular negatives, and one row per episode."""
    if rollout_label not in (None, 0, 1):
        raise ValueError(f"Invalid STOP rollout label: {rollout_label!r}")
    if regular_min_stop_log_odds is None:
        return True
    threshold = float(regular_min_stop_log_odds)
    if not math.isfinite(threshold):
        raise ValueError("Multimodal regular-negative threshold must be finite")
    if rollout_label is None:
        return False
    if rollout_label == 1 or bool(original_terminal):
        return True
    try:
        score = float(stop_log_odds)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Selective multimodal STOP collection requires finite stop_log_odds"
        ) from exc
    if not math.isfinite(score):
        raise ValueError(
            "Selective multimodal STOP collection requires finite stop_log_odds"
        )
    return score > threshold or not bool(episode_has_record)


def validate_oracle_recovery_collection(
    *,
    collection_enabled: bool,
    force_continue_negatives: bool,
    oracle_recovery_after_negative: bool,
) -> bool:
    """Fail closed unless oracle recovery is nested inside DAgger collection."""
    if oracle_recovery_after_negative and not (
        collection_enabled and force_continue_negatives
    ):
        raise ValueError(
            "Oracle STOP recovery requires feature collection and forced "
            "negative continuation"
        )
    return bool(oracle_recovery_after_negative)


def validate_oracle_path_collection(
    *,
    collection_enabled: bool,
    force_continue_negatives: bool,
    oracle_path_from_start: bool,
) -> bool:
    """Restrict start-to-goal oracle paths to labelled offline collection."""
    if oracle_path_from_start and not (
        collection_enabled and force_continue_negatives
    ):
        raise ValueError(
            "Oracle path-from-start collection requires feature collection and "
            "forced negative continuation"
        )
    return bool(oracle_path_from_start)


def validate_boundary_probe_collection(
    *,
    collection_enabled: bool,
    force_continue_negatives: bool,
    oracle_path_from_start: bool,
    boundary_probe_sweep: bool,
    min_distance_m: float,
    max_distance_m: float,
    probes: int,
) -> bool:
    """Restrict metric-triggered view sweeps to privileged oracle-path collection."""
    if isinstance(probes, bool) or not isinstance(probes, int) or probes < 2:
        raise ValueError("Boundary probe sweep requires probes >= 2")
    if (
        not math.isfinite(float(min_distance_m))
        or not math.isfinite(float(max_distance_m))
        or float(min_distance_m) < 0.0
        or float(max_distance_m) <= float(min_distance_m)
    ):
        raise ValueError("Boundary probe sweep requires 0 <= min < max distance")
    if boundary_probe_sweep and not (
        collection_enabled and force_continue_negatives and oracle_path_from_start
    ):
        raise ValueError(
            "Boundary probe sweep requires forced oracle path-from-start feature collection"
        )
    return bool(boundary_probe_sweep)


def validate_oracle_recovery_actions_per_call(value: int) -> int:
    """Validate the number of privileged primitive actions per System2 query."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("Oracle recovery actions_per_call must be an integer >= 1")
    return value


def should_finish_oracle_recovery_collection(
    *,
    goal_probe_count: int,
    max_goal_probes: int,
) -> bool:
    """Bound privileged near-goal probing so collection cannot spin forever."""
    if isinstance(goal_probe_count, bool) or goal_probe_count < 0:
        raise ValueError("Oracle recovery goal_probe_count must be >= 0")
    if isinstance(max_goal_probes, bool) or max_goal_probes < 1:
        raise ValueError("Oracle recovery max_goal_probes must be >= 1")
    return goal_probe_count >= max_goal_probes


@dataclass
class OracleRecoveryState:
    """Keep oracle recovery active until the bounded collector completes."""

    active: bool = False
    activations: int = 0
    activation_reason: str | None = None

    def _activate(self, reason: str) -> bool:
        if self.active:
            return True
        self.active = True
        self.activations += 1
        self.activation_reason = reason
        return True

    def activate_from_cohort(
        self,
        *,
        rollout_label: int | None,
        reason: str,
    ) -> bool:
        """Start explicitly privileged collection for an audited cohort."""
        if rollout_label not in (0, 1):
            raise ValueError(
                "Cohort-triggered recovery requires an unambiguous STOP label"
            )
        if reason not in {
            "historical_false_stop_call",
            "current_positive_stop",
        }:
            raise ValueError(f"Invalid cohort recovery reason: {reason!r}")
        return self._activate(reason)

    def activate_from_start(self) -> bool:
        """Start an explicitly privileged shortest-path collection episode."""
        return self._activate("oracle_path_from_start")

    def observe(self, *, terminal: bool, rollout_label: int | None) -> bool:
        """Return whether the real action must be replaced by an oracle action."""
        if rollout_label not in (None, 0, 1):
            raise ValueError(f"Invalid STOP rollout label: {rollout_label!r}")

        if self.active:
            return True

        if terminal and rollout_label == 0:
            return self._activate("current_false_stop")

        return False

    def complete(self) -> None:
        """End recovery only after the caller has collected every goal probe."""
        if not self.active:
            raise RuntimeError("Cannot complete inactive oracle recovery")
        self.active = False


@dataclass
class BoundaryProbeSweepState:
    """Collect one fixed-position negative view sweep per oracle-path episode."""

    enabled: bool
    min_distance_m: float
    max_distance_m: float
    max_probes: int
    active: bool = False
    completed: bool = False
    probe_index: int = 0
    activation_distance_m: float | None = None

    def __post_init__(self) -> None:
        validate_boundary_probe_collection(
            collection_enabled=True,
            force_continue_negatives=True,
            oracle_path_from_start=True,
            boundary_probe_sweep=self.enabled,
            min_distance_m=self.min_distance_m,
            max_distance_m=self.max_distance_m,
            probes=self.max_probes,
        )

    def observe(self, *, distance_m: float, rollout_label: int | None) -> int | None:
        if rollout_label not in (None, 0, 1):
            raise ValueError(f"Invalid STOP rollout label: {rollout_label!r}")
        distance_m = float(distance_m)
        if not math.isfinite(distance_m) or distance_m < 0.0:
            raise ValueError("Boundary probe distance must be finite and non-negative")
        if not self.enabled or self.completed:
            return None
        if not self.active:
            if not (
                rollout_label != 1
                and self.min_distance_m <= distance_m < self.max_distance_m
            ):
                return None
            self.active = True
            self.activation_distance_m = distance_m
        return self.probe_index

    def finish_current_probe(self) -> bool:
        """Advance after recording one view; return whether the sweep is complete."""
        if not self.active:
            raise RuntimeError("Cannot finish an inactive boundary probe sweep")
        self.probe_index += 1
        if self.probe_index >= self.max_probes:
            self.active = False
            self.completed = True
        return self.completed

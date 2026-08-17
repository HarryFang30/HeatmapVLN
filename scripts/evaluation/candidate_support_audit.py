"""Counterfactual candidate-support and identifiability audit primitives.

This module is deliberately independent from Habitat, Qwen, and the model
server. It owns the durable audit contract shared by the RPC candidate
exporter, the Habitat fork collector, and later identifiability probes.

The central unit is a :class:`TreatmentSpec`, not a raw diffusion trajectory.
It represents the exact state transition deployment would execute: actions,
execution length, replan timing, STOP semantics, and anti-deadlock effects.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


AUDIT_SCHEMA_VERSION = "counterfactual-candidate-audit-v1"
TREATMENT_SCHEMA_VERSION = "candidate-treatment-v1"
COMPACT_FEATURE_SCHEMA_VERSION = "candidate-compact-features-v1"

ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
VALID_LOCAL_ACTIONS = (ACTION_STOP, ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT)

END_EARLY_REPLAN = "early_replan"
END_QUEUE_EXHAUSTED = "queue_exhausted_replan"
END_LOCAL_STOP = "local_stop_replan"
END_ANTI_DEADLOCK = "anti_deadlock_replan"
END_REPLAN_NOW = "replan_now"
VALID_END_REASONS = (
    END_EARLY_REPLAN,
    END_QUEUE_EXHAUSTED,
    END_LOCAL_STOP,
    END_ANTI_DEADLOCK,
    END_REPLAN_NOW,
)


@dataclass(frozen=True)
class RNGSnapshot:
    """In-memory RNG state for isolated closed-loop counterfactual branches."""

    python_state: Any
    numpy_state: Any
    torch_cpu_state: Any | None
    torch_cuda_states: tuple[Any, ...]


def capture_rng_snapshot() -> RNGSnapshot:
    torch_cpu_state = None
    torch_cuda_states: tuple[Any, ...] = ()
    try:
        import torch

        torch_cpu_state = torch.random.get_rng_state().clone()
        if torch.cuda.is_available():
            torch_cuda_states = tuple(
                state.clone() for state in torch.cuda.get_rng_state_all()
            )
    except (ImportError, RuntimeError):
        # The pure schema/analysis environment is allowed to be torch-free.
        pass
    return RNGSnapshot(
        python_state=copy.deepcopy(random.getstate()),
        numpy_state=copy.deepcopy(np.random.get_state()),
        torch_cpu_state=torch_cpu_state,
        torch_cuda_states=torch_cuda_states,
    )


def restore_rng_snapshot(snapshot: RNGSnapshot) -> None:
    if not isinstance(snapshot, RNGSnapshot):
        raise TypeError("snapshot must be RNGSnapshot")
    random.setstate(copy.deepcopy(snapshot.python_state))
    np.random.set_state(copy.deepcopy(snapshot.numpy_state))
    if snapshot.torch_cpu_state is None:
        return
    import torch

    torch.random.set_rng_state(snapshot.torch_cpu_state.clone())
    if snapshot.torch_cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG snapshot cannot be restored without CUDA")
        torch.cuda.set_rng_state_all(
            [state.clone() for state in snapshot.torch_cuda_states]
        )


@dataclass(frozen=True)
class ClosedLoopSnapshot:
    """Complete in-memory branch point: simulator, runtime, history, and RNG."""

    simulator_state: Any
    runtime_state: Any
    rng_state: RNGSnapshot


class ClosedLoopFork:
    """Transactional branch helper for future continuation rollouts.

    Both objects must expose ``snapshot()`` and ``restore(snapshot)``. The
    simulator component can be :class:`HabitatShadowBackend`; the runtime
    component owns conversation/history queues, control state, counters, and
    any model cache. Every branch is restored even when evaluation raises.
    """

    def __init__(self, simulator_component: Any, runtime_component: Any) -> None:
        for name, component in (
            ("simulator_component", simulator_component),
            ("runtime_component", runtime_component),
        ):
            if not callable(getattr(component, "snapshot", None)) or not callable(
                getattr(component, "restore", None)
            ):
                raise TypeError(f"{name} must expose snapshot() and restore()")
        self.simulator_component = simulator_component
        self.runtime_component = runtime_component

    def snapshot(self) -> ClosedLoopSnapshot:
        return ClosedLoopSnapshot(
            simulator_state=copy.deepcopy(self.simulator_component.snapshot()),
            runtime_state=copy.deepcopy(self.runtime_component.snapshot()),
            rng_state=capture_rng_snapshot(),
        )

    def restore(self, snapshot: ClosedLoopSnapshot) -> None:
        if not isinstance(snapshot, ClosedLoopSnapshot):
            raise TypeError("snapshot must be ClosedLoopSnapshot")
        self.simulator_component.restore(copy.deepcopy(snapshot.simulator_state))
        self.runtime_component.restore(copy.deepcopy(snapshot.runtime_state))
        restore_rng_snapshot(snapshot.rng_state)

    @contextmanager
    def branch(self):
        snapshot = self.snapshot()
        try:
            yield snapshot
        finally:
            self.restore(snapshot)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = _canonical_json(
        {"dtype": value.dtype.str, "shape": list(value.shape)}
    ).encode("utf-8")
    return sha256_bytes(header + b"\0" + value.tobytes(order="C"))


def _validate_action(action: int) -> int:
    if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
        raise TypeError(f"action must be an integer, got {type(action).__name__}")
    value = int(action)
    if value not in VALID_LOCAL_ACTIONS:
        raise ValueError(f"unsupported local action {value}")
    return value


def finalize_local_actions(
    actions: Sequence[int],
    *,
    max_steps: int = 8,
    max_local_steps: int = 4,
) -> tuple[int, ...]:
    """Mirror InternNav's pad-to-eight then cap-to-four post-processing."""

    if isinstance(max_steps, bool) or int(max_steps) <= 0:
        raise ValueError("max_steps must be a positive integer")
    if isinstance(max_local_steps, bool) or not 0 < int(max_local_steps) <= int(max_steps):
        raise ValueError("max_local_steps must be in [1,max_steps]")
    result = [_validate_action(action) for action in actions]
    if len(result) < int(max_steps):
        result.extend([ACTION_STOP] * (int(max_steps) - len(result)))
    return tuple(result[: int(max_local_steps)])


@dataclass(frozen=True)
class TreatmentSpec:
    """Exact deployable local transition semantics for one candidate.

    ``actions`` contains only actions actually sent to Habitat. A local STOP
    is represented by ``end_reason=local_stop_replan`` and is never included
    in ``actions``. The first-action STOP anti-deadlock path is represented as
    one LEFT action with ``trigger_anti_deadlock=True``.
    """

    actions: tuple[int, ...]
    execute_len: int
    end_reason: str
    replan_after: bool = True
    update_local_stop_counter: bool = False
    trigger_anti_deadlock: bool = False

    def __post_init__(self) -> None:
        canonical_actions = tuple(_validate_action(action) for action in self.actions)
        if ACTION_STOP in canonical_actions:
            raise ValueError("TreatmentSpec.actions may not contain local STOP")
        object.__setattr__(self, "actions", canonical_actions)
        if isinstance(self.execute_len, bool) or int(self.execute_len) != len(canonical_actions):
            raise ValueError("execute_len must equal the number of executable actions")
        if self.end_reason not in VALID_END_REASONS:
            raise ValueError(f"unsupported end_reason {self.end_reason!r}")
        if self.end_reason == END_REPLAN_NOW and canonical_actions:
            raise ValueError("replan_now treatment must execute no actions")
        if self.trigger_anti_deadlock:
            if canonical_actions != (ACTION_LEFT,) or self.end_reason != END_ANTI_DEADLOCK:
                raise ValueError("anti-deadlock treatment must be one LEFT action")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": TREATMENT_SCHEMA_VERSION,
            "actions": list(self.actions),
            "execute_len": int(self.execute_len),
            "end_reason": self.end_reason,
            "replan_after": bool(self.replan_after),
            "update_local_stop_counter": bool(self.update_local_stop_counter),
            "trigger_anti_deadlock": bool(self.trigger_anti_deadlock),
        }

    @property
    def signature(self) -> str:
        return sha256_bytes(_canonical_json(self.canonical_dict()).encode("utf-8"))


def treatments_from_finalized_chunk(
    finalized_actions: Sequence[int],
    *,
    prefix_lengths: Sequence[int] = (1, 2, 3, 4),
    include_replan_now: bool = False,
) -> tuple[TreatmentSpec, ...]:
    """Expand a deployed four-action chunk into exact prefix treatments."""

    chunk = tuple(_validate_action(action) for action in finalized_actions)
    if not chunk:
        raise ValueError("finalized_actions may not be empty")
    if len(chunk) > 4:
        raise ValueError("finalized action chunk may contain at most four actions")

    results: list[TreatmentSpec] = []
    if include_replan_now:
        results.append(
            TreatmentSpec(actions=(), execute_len=0, end_reason=END_REPLAN_NOW)
        )

    if chunk[0] == ACTION_STOP:
        results.append(
            TreatmentSpec(
                actions=(ACTION_LEFT,),
                execute_len=1,
                end_reason=END_ANTI_DEADLOCK,
                trigger_anti_deadlock=True,
            )
        )
        return tuple(results)

    stop_index = chunk.index(ACTION_STOP) if ACTION_STOP in chunk else len(chunk)
    executable = chunk[:stop_index]
    if not executable:
        raise AssertionError("first STOP should have been handled above")
    base_end_reason = END_LOCAL_STOP if stop_index < len(chunk) else END_QUEUE_EXHAUSTED
    requested_lengths = sorted(
        {
            int(length)
            for length in prefix_lengths
            if not isinstance(length, bool) and 1 <= int(length) <= len(executable)
        }
    )
    if len(executable) not in requested_lengths:
        requested_lengths.append(len(executable))

    for length in requested_lengths:
        is_full = length == len(executable)
        results.append(
            TreatmentSpec(
                actions=tuple(executable[:length]),
                execute_len=length,
                end_reason=base_end_reason if is_full else END_EARLY_REPLAN,
            )
        )
    deduped: dict[str, TreatmentSpec] = {}
    for treatment in results:
        deduped.setdefault(treatment.signature, treatment)
    return tuple(deduped.values())


@dataclass(frozen=True)
class CandidateProvenance:
    arm: str
    aggregation: str
    sample_index: int | None
    trajectory_sha256: str

    def __post_init__(self) -> None:
        if self.arm not in {"native", "heatmap_control"}:
            raise ValueError(f"unsupported candidate arm {self.arm!r}")
        if not self.aggregation:
            raise ValueError("aggregation must be non-empty")
        if self.sample_index is not None and (
            isinstance(self.sample_index, bool) or int(self.sample_index) < 0
        ):
            raise ValueError("sample_index must be None or a non-negative integer")
        if not re.fullmatch(r"[0-9a-f]{64}", self.trajectory_sha256):
            raise ValueError("trajectory_sha256 must be lowercase SHA256")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class AggregatedTreatment:
    treatment: TreatmentSpec
    provenances: tuple[CandidateProvenance, ...]
    native_sample_count: int
    heatmap_sample_count: int
    native_sample_total: int
    heatmap_sample_total: int

    @property
    def native_sample_mass(self) -> float:
        return (
            float(self.native_sample_count) / float(self.native_sample_total)
            if self.native_sample_total
            else 0.0
        )

    @property
    def heatmap_sample_mass(self) -> float:
        return (
            float(self.heatmap_sample_count) / float(self.heatmap_sample_total)
            if self.heatmap_sample_total
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "treatment_id": self.treatment.signature,
            "spec": self.treatment.canonical_dict(),
            "provenances": [provenance.to_dict() for provenance in self.provenances],
            "native_sample_count": int(self.native_sample_count),
            "native_sample_total": int(self.native_sample_total),
            "native_sample_mass": self.native_sample_mass,
            "heatmap_sample_count": int(self.heatmap_sample_count),
            "heatmap_sample_total": int(self.heatmap_sample_total),
            "heatmap_sample_mass": self.heatmap_sample_mass,
        }


def _validate_trajectory_samples(name: str, trajectories: Any) -> np.ndarray:
    value = np.asarray(trajectories)
    if value.ndim != 3 or value.shape[0] < 1 or value.shape[1] < 1 or value.shape[2] != 3:
        raise ValueError(f"{name} must have shape [K,T,3], got {value.shape}")
    if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite floating point")
    return np.ascontiguousarray(value.astype(np.float32, copy=False))


def _trajectory_paths(trajectories: np.ndarray) -> np.ndarray:
    return np.cumsum(trajectories[..., :2], axis=1, dtype=np.float64)


def trajectory_medoid_index(trajectories: np.ndarray) -> int:
    value = _validate_trajectory_samples("trajectories", trajectories)
    flat = _trajectory_paths(value).reshape(value.shape[0], -1)
    distances = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    return int(np.argmin(distances.sum(axis=1)))


def nearest_to_mean_index(trajectories: np.ndarray) -> int:
    value = _validate_trajectory_samples("trajectories", trajectories)
    paths = _trajectory_paths(value)
    mean_path = paths.mean(axis=0)
    distances = np.linalg.norm(paths - mean_path[None, ...], axis=(1, 2))
    return int(np.argmin(distances))


@dataclass(frozen=True)
class CandidateSet:
    treatments: tuple[AggregatedTreatment, ...]
    source_entries: tuple[dict[str, Any], ...]
    baselines: Mapping[str, str]
    native_sample_total: int
    heatmap_sample_total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": AUDIT_SCHEMA_VERSION,
            "treatments": [treatment.to_dict() for treatment in self.treatments],
            "source_entries": list(self.source_entries),
            "baselines": dict(self.baselines),
            "native_sample_total": int(self.native_sample_total),
            "heatmap_sample_total": int(self.heatmap_sample_total),
            "unique_treatment_count": len(self.treatments),
        }


def build_candidate_set(
    native_trajectories: Any,
    *,
    trajectory_to_actions: Callable[[np.ndarray], Sequence[int]],
    heatmap_trajectories: Any | None = None,
    prefix_lengths: Sequence[int] = (1, 2, 3, 4),
    include_replan_now: bool = False,
    max_steps: int = 8,
    max_local_steps: int = 4,
) -> CandidateSet:
    """Build and deduplicate deployment-semantic treatments.

    ``trajectory_to_actions`` must convert exactly one ``[T,3]`` delta
    trajectory with the authoritative InternNav postprocessor prior to the
    pad/cap step implemented here.
    """

    native = _validate_trajectory_samples("native_trajectories", native_trajectories)
    heatmap = (
        None
        if heatmap_trajectories is None
        else _validate_trajectory_samples("heatmap_trajectories", heatmap_trajectories)
    )
    if heatmap is not None and heatmap.shape != native.shape:
        raise ValueError(
            "native and heatmap trajectory sample shapes must match for paired noise: "
            f"{native.shape} != {heatmap.shape}"
        )

    grouped_specs: dict[str, TreatmentSpec] = {}
    grouped_provenance: dict[str, list[CandidateProvenance]] = defaultdict(list)
    sample_counts: dict[str, Counter[str]] = {
        "native": Counter(),
        "heatmap_control": Counter(),
    }
    source_entries: list[dict[str, Any]] = []

    def add_candidate(
        trajectory: np.ndarray,
        *,
        arm: str,
        aggregation: str,
        sample_index: int | None,
    ) -> tuple[str, tuple[str, ...]]:
        trajectory = np.ascontiguousarray(trajectory.astype(np.float32, copy=False))
        raw_actions = trajectory_to_actions(trajectory)
        finalized = finalize_local_actions(
            raw_actions,
            max_steps=max_steps,
            max_local_steps=max_local_steps,
        )
        treatments = treatments_from_finalized_chunk(
            finalized,
            prefix_lengths=prefix_lengths,
            include_replan_now=include_replan_now,
        )
        provenance = CandidateProvenance(
            arm=arm,
            aggregation=aggregation,
            sample_index=sample_index,
            trajectory_sha256=sha256_array(trajectory),
        )
        treatment_ids: list[str] = []
        for treatment in treatments:
            treatment_id = treatment.signature
            grouped_specs.setdefault(treatment_id, treatment)
            grouped_provenance[treatment_id].append(provenance)
            treatment_ids.append(treatment_id)
            if aggregation == "sample":
                sample_counts[arm][treatment_id] += 1
        base_id = treatment_ids[-1]
        source_entries.append(
            {
                "arm": arm,
                "aggregation": aggregation,
                "sample_index": sample_index,
                "trajectory_sha256": provenance.trajectory_sha256,
                "finalized_actions": list(finalized),
                "treatment_ids": treatment_ids,
                "base_treatment_id": base_id,
            }
        )
        return base_id, tuple(treatment_ids)

    native_sample_base_ids: list[str] = []
    for index, trajectory in enumerate(native):
        base_id, _ = add_candidate(
            trajectory,
            arm="native",
            aggregation="sample",
            sample_index=index,
        )
        native_sample_base_ids.append(base_id)
    native_mean_id, _ = add_candidate(
        native.mean(axis=0),
        arm="native",
        aggregation="trajectory_mean",
        sample_index=None,
    )

    medoid_index = trajectory_medoid_index(native)
    nearest_index = nearest_to_mean_index(native)
    native_mode_id = Counter(native_sample_base_ids).most_common(1)[0][0]
    baselines: dict[str, str] = {
        "native_trajectory_mean": native_mean_id,
        "native_action_mode": native_mode_id,
        "native_trajectory_medoid": native_sample_base_ids[medoid_index],
        "native_nearest_to_mean": native_sample_base_ids[nearest_index],
        "native_random_sample_0": native_sample_base_ids[0],
    }

    if heatmap is not None:
        for index, trajectory in enumerate(heatmap):
            add_candidate(
                trajectory,
                arm="heatmap_control",
                aggregation="sample",
                sample_index=index,
            )
        heatmap_mean_id, _ = add_candidate(
            heatmap.mean(axis=0),
            arm="heatmap_control",
            aggregation="trajectory_mean",
            sample_index=None,
        )
        baselines["heatmap_trajectory_mean"] = heatmap_mean_id

    native_total = int(native.shape[0])
    heatmap_total = int(heatmap.shape[0]) if heatmap is not None else 0
    treatments = tuple(
        AggregatedTreatment(
            treatment=grouped_specs[treatment_id],
            provenances=tuple(grouped_provenance[treatment_id]),
            native_sample_count=int(sample_counts["native"][treatment_id]),
            heatmap_sample_count=int(sample_counts["heatmap_control"][treatment_id]),
            native_sample_total=native_total,
            heatmap_sample_total=heatmap_total,
        )
        for treatment_id in sorted(grouped_specs)
    )
    return CandidateSet(
        treatments=treatments,
        source_entries=tuple(source_entries),
        baselines=baselines,
        native_sample_total=native_total,
        heatmap_sample_total=heatmap_total,
    )


def candidate_count_sensitivity(
    candidate_set: CandidateSet,
    *,
    arm: str = "native",
    ks: Sequence[int] = (1, 4, 8, 16, 32),
) -> list[dict[str, int]]:
    """Count unique deployment treatments for deterministic sample prefixes.

    Each stochastic trajectory can yield multiple prefix/replan treatments.  The
    full treatment set, rather than only the final chunk, is therefore the
    relevant support size at a given candidate count K.
    """

    if arm not in {"native", "heatmap_control"}:
        raise ValueError(f"unsupported arm {arm!r}")
    sample_entries = [
        entry
        for entry in candidate_set.source_entries
        if entry["arm"] == arm and entry["aggregation"] == "sample"
    ]
    sample_entries.sort(key=lambda entry: int(entry["sample_index"]))
    result: list[dict[str, int]] = []
    for raw_k in ks:
        if isinstance(raw_k, bool) or int(raw_k) <= 0:
            raise ValueError("candidate count K must be positive")
        requested_k = int(raw_k)
        effective_k = min(requested_k, len(sample_entries))
        if effective_k == 0:
            continue
        unique_base = {
            entry["base_treatment_id"]
            for entry in sample_entries[:effective_k]
        }
        unique = {
            treatment_id
            for entry in sample_entries[:effective_k]
            for treatment_id in entry["treatment_ids"]
        }
        result.append(
            {
                "requested_k": requested_k,
                "effective_k": effective_k,
                "unique_treatment_count": len(unique),
                "unique_base_treatment_count": len(unique_base),
            }
        )
    return result


@dataclass(frozen=True)
class LocalOutcome:
    treatment_id: str
    actions_executed: tuple[int, ...]
    travelled_m: float
    endpoint_offpath_m: float
    endpoint_route_progress_m: float
    route_progress_delta_m: float
    endpoint_euclidean_goal_distance_m: float
    min_euclidean_goal_distance_m: float
    collision_or_stuck_count: int
    revisit: bool
    entered_euclidean_success_radius: bool
    left_euclidean_success_radius: bool
    endpoint_pose: np.ndarray
    pose_trace: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "treatment_id": self.treatment_id,
            "actions_executed": list(self.actions_executed),
            "travelled_m": float(self.travelled_m),
            "endpoint_offpath_m": float(self.endpoint_offpath_m),
            "endpoint_route_progress_m": float(self.endpoint_route_progress_m),
            "route_progress_delta_m": float(self.route_progress_delta_m),
            "endpoint_euclidean_goal_distance_m": float(
                self.endpoint_euclidean_goal_distance_m
            ),
            "min_euclidean_goal_distance_m": float(
                self.min_euclidean_goal_distance_m
            ),
            "collision_or_stuck_count": int(self.collision_or_stuck_count),
            "revisit": bool(self.revisit),
            "entered_euclidean_success_radius": bool(
                self.entered_euclidean_success_radius
            ),
            "left_euclidean_success_radius": bool(
                self.left_euclidean_success_radius
            ),
        }


def evaluate_local_treatment(
    backend: Any,
    treatment: TreatmentSpec,
    *,
    start_pose: Any,
    route_tracker: Any,
    route_progress_m: float,
    goal_position: Any,
    older_poses: Any | None = None,
    success_radius_m: float = 3.0,
    revisit_radius_m: float = 0.35,
    collision_translation_m: float = 0.05,
) -> LocalOutcome:
    """Shadow-execute a treatment and compute explicitly local outcomes."""

    if not math.isfinite(float(success_radius_m)) or float(success_radius_m) <= 0:
        raise ValueError("success_radius_m must be positive")
    start = np.asarray(start_pose, dtype=np.float64)
    if start.shape != (4, 4) or not np.isfinite(start).all():
        raise ValueError("start_pose must be finite [4,4]")
    goal = np.asarray(goal_position, dtype=np.float64).reshape(-1)
    if goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError("goal_position must be finite xyz")

    poses = backend.simulate_actions(
        list(treatment.actions),
        start_pose=start,
        max_actions=max(1, len(treatment.actions)),
    )
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise RuntimeError(f"shadow backend returned invalid poses {poses.shape}")
    positions = poses[:, :3, 3]
    step_distance = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    travelled = float(step_distance.sum())
    endpoint = positions[-1]
    offpath, raw_progress = route_tracker.project(endpoint)
    route_delta = float(raw_progress) - float(route_progress_m)
    goal_distances = np.linalg.norm(positions - goal[None, :], axis=1)

    collision_count = 0
    for action, distance in zip(treatment.actions, step_distance):
        if int(action) == ACTION_FORWARD and float(distance) < float(collision_translation_m):
            collision_count += 1

    revisit = False
    if older_poses is not None:
        older = np.asarray(older_poses, dtype=np.float64)
        if older.size:
            if older.ndim == 3 and older.shape[1:] == (4, 4):
                older_positions = older[:, :3, 3]
            elif older.ndim == 2 and older.shape[1] == 3:
                older_positions = older
            else:
                raise ValueError("older_poses must be [N,4,4] or [N,3]")
            revisit = bool(
                np.linalg.norm(older_positions - endpoint[None, :], axis=1).min()
                <= float(revisit_radius_m)
            )

    inside = goal_distances <= float(success_radius_m)
    entered = bool(inside.any())
    first_inside = int(np.argmax(inside)) if entered else -1
    left = bool(entered and (~inside[first_inside:]).any())
    return LocalOutcome(
        treatment_id=treatment.signature,
        actions_executed=treatment.actions,
        travelled_m=travelled,
        endpoint_offpath_m=float(offpath),
        endpoint_route_progress_m=float(raw_progress),
        route_progress_delta_m=route_delta,
        endpoint_euclidean_goal_distance_m=float(goal_distances[-1]),
        min_euclidean_goal_distance_m=float(goal_distances.min()),
        collision_or_stuck_count=collision_count,
        revisit=revisit,
        entered_euclidean_success_radius=entered,
        left_euclidean_success_radius=left,
        endpoint_pose=poses[-1].astype(np.float32),
        pose_trace=poses.astype(np.float32),
    )


def validate_compact_arrays(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("compact arrays must be a non-empty mapping")
    result: dict[str, np.ndarray] = {}
    for name, raw_value in arrays.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"invalid compact array name {name!r}")
        value = np.asarray(raw_value)
        if value.dtype.hasobject:
            raise TypeError(f"compact array {name} may not use object dtype")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"compact array {name} contains non-finite values")
        result[name] = np.ascontiguousarray(value)
    return result


def compact_array_manifest(arrays: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_compact_arrays(arrays)
    return {
        "schema": COMPACT_FEATURE_SCHEMA_VERSION,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "sha256": sha256_array(value),
                "nbytes": int(value.nbytes),
            }
            for name, value in sorted(validated.items())
        },
    }


class AuditShardWriter:
    """Atomic, resumable writer for one independent audit worker shard."""

    def __init__(
        self,
        root: str | Path,
        *,
        shard_id: int,
        max_bytes: int,
    ) -> None:
        if isinstance(shard_id, bool) or int(shard_id) < 0:
            raise ValueError("shard_id must be a non-negative integer")
        if isinstance(max_bytes, bool) or int(max_bytes) <= 0:
            raise ValueError("max_bytes must be positive")
        self.root = Path(root).expanduser().resolve()
        self.shard_id = int(shard_id)
        self.max_bytes = int(max_bytes)
        self.shard_dir = self.root / f"shard_{self.shard_id:02d}"
        self.arrays_dir = self.shard_dir / "arrays"
        self.index_path = self.shard_dir / "records.jsonl"
        self.manifest_path = self.shard_dir / "manifest.json"
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict[str, Any]] = {}
        self._bytes = 0
        if self.index_path.exists():
            with self.index_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = str(row.get("state_key") or "")
                    if not key or key in self._records:
                        raise RuntimeError(
                            f"invalid/reused state_key at {self.index_path}:{line_number}"
                        )
                    self._records[key] = row
                    self._bytes += int(row.get("array_file_bytes", 0))

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def bytes_written(self) -> int:
        return self._bytes

    def contains(self, state_key: str) -> bool:
        return str(state_key) in self._records

    @staticmethod
    def _state_filename(state_key: str) -> str:
        digest = sha256_bytes(str(state_key).encode("utf-8"))
        return f"{digest}.npz"

    def commit(
        self,
        *,
        state_key: str,
        record: Mapping[str, Any],
        arrays: Mapping[str, Any],
    ) -> dict[str, Any]:
        key = str(state_key).strip()
        if not key:
            raise ValueError("state_key must be non-empty")
        if key in self._records:
            return dict(self._records[key])
        record_payload = dict(record)
        reserved_record_keys = {
            "schema",
            "state_key",
            "compact_features",
            "array_file",
            "array_file_sha256",
            "array_file_bytes",
        }
        collisions = sorted(reserved_record_keys.intersection(record_payload))
        if collisions:
            raise ValueError(
                "record may not override reserved audit fields: "
                + ", ".join(collisions)
            )
        # Fail before creating an array artifact if metadata is not durable JSON.
        _canonical_json(record_payload)
        validated = validate_compact_arrays(arrays)
        filename = self._state_filename(key)
        final_path = self.arrays_dir / filename
        if final_path.exists():
            raise RuntimeError(f"orphan/colliding audit array file exists: {final_path}")

        with tempfile.NamedTemporaryFile(
            dir=self.arrays_dir,
            prefix=f".{filename}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        try:
            with temp_path.open("wb") as handle:
                np.savez_compressed(handle, **validated)
                handle.flush()
                os.fsync(handle.fileno())
            file_bytes = int(temp_path.stat().st_size)
            if self._bytes + file_bytes > self.max_bytes:
                raise RuntimeError(
                    "audit shard byte quota would be exceeded: "
                    f"current={self._bytes} candidate={file_bytes} max={self.max_bytes}"
                )
            array_sha = sha256_bytes(temp_path.read_bytes())
            os.replace(temp_path, final_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        row = {
            "schema": AUDIT_SCHEMA_VERSION,
            "state_key": key,
            **record_payload,
            "compact_features": compact_array_manifest(validated),
            "array_file": f"arrays/{filename}",
            "array_file_sha256": array_sha,
            "array_file_bytes": file_bytes,
        }
        encoded = (_canonical_json(row) + "\n").encode("utf-8")
        with self.index_path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self._records[key] = row
        self._bytes += file_bytes
        return dict(row)

    def seal(self, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        extra_payload = dict(extra or {})
        reserved_manifest_keys = {
            "schema",
            "shard_id",
            "record_count",
            "array_bytes",
            "max_bytes",
            "records_jsonl_sha256",
        }
        collisions = sorted(reserved_manifest_keys.intersection(extra_payload))
        if collisions:
            raise ValueError(
                "manifest extra may not override reserved fields: "
                + ", ".join(collisions)
            )
        _canonical_json(extra_payload)
        manifest = {
            "schema": AUDIT_SCHEMA_VERSION,
            "shard_id": self.shard_id,
            "record_count": self.record_count,
            "array_bytes": self.bytes_written,
            "max_bytes": self.max_bytes,
            "records_jsonl_sha256": (
                sha256_bytes(self.index_path.read_bytes())
                if self.index_path.exists()
                else sha256_bytes(b"")
            ),
            **extra_payload,
        }
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=self.shard_dir,
            prefix=".manifest.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temp_path, self.manifest_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        return manifest


__all__ = [
    "ACTION_FORWARD",
    "ACTION_LEFT",
    "ACTION_RIGHT",
    "ACTION_STOP",
    "AUDIT_SCHEMA_VERSION",
    "AggregatedTreatment",
    "AuditShardWriter",
    "CandidateSet",
    "ClosedLoopFork",
    "ClosedLoopSnapshot",
    "LocalOutcome",
    "RNGSnapshot",
    "TreatmentSpec",
    "build_candidate_set",
    "candidate_count_sensitivity",
    "compact_array_manifest",
    "capture_rng_snapshot",
    "evaluate_local_treatment",
    "finalize_local_actions",
    "nearest_to_mean_index",
    "restore_rng_snapshot",
    "sha256_array",
    "trajectory_medoid_index",
    "treatments_from_finalized_chunk",
    "validate_compact_arrays",
]

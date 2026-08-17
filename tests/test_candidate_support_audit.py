from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation.candidate_support_audit import (
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STOP,
    END_ANTI_DEADLOCK,
    END_EARLY_REPLAN,
    END_LOCAL_STOP,
    END_QUEUE_EXHAUSTED,
    AuditShardWriter,
    ClosedLoopFork,
    TreatmentSpec,
    build_candidate_set,
    candidate_count_sensitivity,
    compact_array_manifest,
    evaluate_local_treatment,
    finalize_local_actions,
    nearest_to_mean_index,
    trajectory_medoid_index,
    treatments_from_finalized_chunk,
)


def _pose(x: float = 0.0, z: float = 0.0) -> np.ndarray:
    value = np.eye(4, dtype=np.float32)
    value[0, 3] = x
    value[2, 3] = z
    return value


def test_finalize_local_actions_matches_pad_then_cap_contract():
    assert finalize_local_actions([ACTION_FORWARD, ACTION_LEFT]) == (
        ACTION_FORWARD,
        ACTION_LEFT,
        ACTION_STOP,
        ACTION_STOP,
    )
    assert finalize_local_actions([ACTION_FORWARD] * 9) == (ACTION_FORWARD,) * 4


def test_treatment_prefixes_distinguish_early_replan_and_local_stop():
    values = treatments_from_finalized_chunk(
        [ACTION_FORWARD, ACTION_FORWARD, ACTION_STOP, ACTION_STOP]
    )
    assert [(item.actions, item.end_reason) for item in values] == [
        ((ACTION_FORWARD,), END_EARLY_REPLAN),
        ((ACTION_FORWARD, ACTION_FORWARD), END_LOCAL_STOP),
    ]
    assert len({item.signature for item in values}) == 2


def test_treatment_prefixes_enumerate_one_through_four():
    values = treatments_from_finalized_chunk(
        [ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT, ACTION_FORWARD]
    )
    assert [item.execute_len for item in values] == [1, 2, 3, 4]
    assert [item.end_reason for item in values] == [
        END_EARLY_REPLAN,
        END_EARLY_REPLAN,
        END_EARLY_REPLAN,
        END_QUEUE_EXHAUSTED,
    ]


def test_first_local_stop_is_explicit_anti_deadlock_treatment():
    (value,) = treatments_from_finalized_chunk(
        [ACTION_STOP, ACTION_STOP, ACTION_STOP, ACTION_STOP]
    )
    assert value.actions == (ACTION_LEFT,)
    assert value.end_reason == END_ANTI_DEADLOCK
    assert value.trigger_anti_deadlock is True


def test_treatment_signature_includes_replan_semantics():
    early = TreatmentSpec(
        actions=(ACTION_FORWARD,),
        execute_len=1,
        end_reason=END_EARLY_REPLAN,
    )
    exhausted = TreatmentSpec(
        actions=(ACTION_FORWARD,),
        execute_len=1,
        end_reason=END_QUEUE_EXHAUSTED,
    )
    assert early.signature != exhausted.signature


def _candidate_trajectories(codes: list[int]) -> np.ndarray:
    values = np.zeros((len(codes), 4, 3), dtype=np.float32)
    values[:, 0, 0] = np.asarray(codes, dtype=np.float32)
    return values


def _code_converter(trajectory: np.ndarray) -> list[int]:
    code = int(round(float(trajectory[0, 0])))
    mapping = {
        1: [ACTION_FORWARD, ACTION_FORWARD, ACTION_FORWARD, ACTION_FORWARD],
        2: [ACTION_LEFT, ACTION_FORWARD, ACTION_FORWARD, ACTION_FORWARD],
        3: [ACTION_RIGHT, ACTION_FORWARD, ACTION_FORWARD, ACTION_FORWARD],
        4: [ACTION_FORWARD, ACTION_FORWARD, ACTION_STOP],
    }
    return mapping.get(code, [ACTION_STOP])


def test_candidate_set_tracks_mass_baselines_and_paired_arms():
    native = _candidate_trajectories([1, 1, 1, 2])
    control = _candidate_trajectories([2, 2, 3, 4])
    result = build_candidate_set(
        native,
        trajectory_to_actions=_code_converter,
        heatmap_trajectories=control,
    )
    payload = result.to_dict()
    assert result.native_sample_total == 4
    assert result.heatmap_sample_total == 4
    assert "native_trajectory_mean" in result.baselines
    assert "native_action_mode" in result.baselines
    assert "native_trajectory_medoid" in result.baselines
    assert "native_nearest_to_mean" in result.baselines
    assert "heatmap_trajectory_mean" in result.baselines
    mode_id = result.baselines["native_action_mode"]
    mode = next(item for item in payload["treatments"] if item["treatment_id"] == mode_id)
    assert mode["native_sample_count"] == 3
    assert mode["native_sample_mass"] == pytest.approx(0.75)


def test_candidate_set_rejects_unpaired_native_control_shapes():
    with pytest.raises(ValueError, match="paired noise"):
        build_candidate_set(
            _candidate_trajectories([1, 2]),
            trajectory_to_actions=_code_converter,
            heatmap_trajectories=_candidate_trajectories([1]),
        )


def test_candidate_count_sensitivity_counts_all_prefix_treatments():
    result = build_candidate_set(
        _candidate_trajectories([1, 1, 2, 3]),
        trajectory_to_actions=_code_converter,
    )
    assert candidate_count_sensitivity(result, ks=(1, 2, 4, 32)) == [
        {
            "requested_k": 1,
            "effective_k": 1,
            "unique_treatment_count": 4,
            "unique_base_treatment_count": 1,
        },
        {
            "requested_k": 2,
            "effective_k": 2,
            "unique_treatment_count": 4,
            "unique_base_treatment_count": 1,
        },
        {
            "requested_k": 4,
            "effective_k": 4,
            "unique_treatment_count": 12,
            "unique_base_treatment_count": 3,
        },
        {
            "requested_k": 32,
            "effective_k": 4,
            "unique_treatment_count": 12,
            "unique_base_treatment_count": 3,
        },
    ]


def test_medoid_and_nearest_to_mean_are_real_samples():
    trajectories = np.zeros((3, 2, 3), dtype=np.float32)
    trajectories[1, :, 0] = 1.0
    trajectories[2, :, 0] = 10.0
    assert trajectory_medoid_index(trajectories) == 1
    assert nearest_to_mean_index(trajectories) in {0, 1}


class _FakeBackend:
    def simulate_actions(self, actions, *, start_pose, max_actions):
        poses = [np.asarray(start_pose, dtype=np.float32).copy()]
        current = poses[0].copy()
        for action in actions:
            current = current.copy()
            if action == ACTION_FORWARD:
                current[0, 3] += 0.25
            poses.append(current)
        return np.stack(poses)


class _FakeRouteTracker:
    @staticmethod
    def project(position):
        point = np.asarray(position)
        return abs(float(point[2])), float(point[0])


def test_local_outcome_is_vector_valued_and_does_not_mutate_treatment():
    treatment = TreatmentSpec(
        actions=(ACTION_FORWARD, ACTION_LEFT, ACTION_FORWARD),
        execute_len=3,
        end_reason=END_QUEUE_EXHAUSTED,
    )
    outcome = evaluate_local_treatment(
        _FakeBackend(),
        treatment,
        start_pose=_pose(),
        route_tracker=_FakeRouteTracker(),
        route_progress_m=0.0,
        goal_position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        older_poses=np.stack([_pose(-2.0, 0.0)]),
        success_radius_m=0.6,
    )
    assert outcome.actions_executed == treatment.actions
    assert outcome.travelled_m == pytest.approx(0.5)
    assert outcome.route_progress_delta_m == pytest.approx(0.5)
    assert outcome.collision_or_stuck_count == 0
    assert outcome.entered_euclidean_success_radius is True
    assert outcome.endpoint_pose.shape == (4, 4)
    assert outcome.pose_trace.shape == (4, 4, 4)


def test_compact_manifest_hashes_shape_dtype_and_bytes():
    arrays = {
        "native_trajectories": np.zeros((32, 32, 3), dtype=np.float16),
        "heatmap_token_mask": np.ones((32,), dtype=np.bool_),
    }
    manifest = compact_array_manifest(arrays)
    assert manifest["arrays"]["native_trajectories"]["shape"] == [32, 32, 3]
    assert manifest["arrays"]["native_trajectories"]["nbytes"] == 32 * 32 * 3 * 2
    assert len(manifest["arrays"]["native_trajectories"]["sha256"]) == 64


def test_audit_writer_is_atomic_resumable_and_sealed(tmp_path: Path):
    writer = AuditShardWriter(tmp_path, shard_id=3, max_bytes=10_000_000)
    row = writer.commit(
        state_key="scene:1:call:0",
        record={"split": "train", "candidate_set": {"unique": 2}},
        arrays={"native_trajectories": np.zeros((2, 4, 3), dtype=np.float16)},
    )
    assert row["state_key"] == "scene:1:call:0"
    assert writer.record_count == 1
    assert (writer.shard_dir / row["array_file"]).is_file()
    assert writer.commit(
        state_key="scene:1:call:0",
        record={"ignored": True},
        arrays={"different": np.ones((1,), dtype=np.float32)},
    ) == row

    resumed = AuditShardWriter(tmp_path, shard_id=3, max_bytes=10_000_000)
    assert resumed.contains("scene:1:call:0")
    manifest = resumed.seal(extra={"policy": "native"})
    assert manifest["record_count"] == 1
    assert manifest["policy"] == "native"
    assert json.loads(resumed.manifest_path.read_text())["record_count"] == 1


def test_audit_writer_enforces_quota_without_partial_commit(tmp_path: Path):
    writer = AuditShardWriter(tmp_path, shard_id=0, max_bytes=1)
    with pytest.raises(RuntimeError, match="quota"):
        writer.commit(
            state_key="too-large",
            record={},
            arrays={"value": np.ones((16,), dtype=np.float32)},
        )
    assert writer.record_count == 0
    assert not writer.index_path.exists()
    assert list(writer.arrays_dir.iterdir()) == []


def test_audit_writer_rejects_reserved_metadata_overrides(tmp_path: Path):
    writer = AuditShardWriter(tmp_path, shard_id=0, max_bytes=10_000)
    with pytest.raises(ValueError, match="reserved audit fields"):
        writer.commit(
            state_key="state",
            record={"schema": "forged"},
            arrays={"value": np.ones((1,), dtype=np.float32)},
        )
    with pytest.raises(ValueError, match="reserved fields"):
        writer.seal(extra={"record_count": 999})


class _SnapshotComponent:
    def __init__(self, value):
        self.value = value

    def snapshot(self):
        return {"value": self.value}

    def restore(self, snapshot):
        self.value = snapshot["value"]


def test_closed_loop_fork_restores_sim_runtime_and_rng():
    simulator = _SnapshotComponent("base-sim")
    runtime = _SnapshotComponent({"history": [1], "queue": []})
    random.seed(17)
    np.random.seed(17)

    with ClosedLoopFork(simulator, runtime).branch():
        simulator.value = "branch-sim"
        runtime.value["history"].append(2)
        branch_python = random.random()
        branch_numpy = float(np.random.random())

    assert simulator.value == "base-sim"
    assert runtime.value == {"history": [1], "queue": []}
    assert random.random() == branch_python
    assert float(np.random.random()) == branch_numpy

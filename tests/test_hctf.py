import math

import numpy as np

from scripts.evaluation.hctf import (
    VetoThresholds,
    action_distribution_entropy,
    bearing_to_view_coordinate,
    candidate_history_risk,
    deployable_loop_signals,
    deployable_recovery_partition,
    recovery_anchor_risk,
    select_adaptive_prefix,
    select_directional_veto,
)


def _pose(x: float, z: float) -> np.ndarray:
    value = np.eye(4, dtype=np.float32)
    value[0, 3] = x
    value[2, 3] = z
    return value


def _context(view_index: int) -> dict[str, np.ndarray]:
    coarse = np.full((1, 1, 4, 8, 8), 1.0 / 64.0, dtype=np.float32)
    coarse[0, 0, view_index] = 0.0
    coarse[0, 0, view_index, 3:5, 3:5] = 0.25
    stats = np.zeros((1, 1, 4, 7), dtype=np.float32)
    stats[..., 2] = 0.04
    stats[..., 3] = 0.04
    stats[..., 5] = 0.20
    stats[..., 6] = 0.25
    views = np.zeros((1, 1, 4), dtype=np.float32)
    views[0, 0, view_index] = 0.99
    return {
        "fixed_history_mask": np.asarray([[True]]),
        "fixed_history_rel_poses": np.asarray(
            [[[0.50, 0.0, 1.0, 0.0]]], dtype=np.float32
        ),
        "history_rank": np.asarray([[1.0]], dtype=np.float32),
        "coarse_probabilities": coarse,
        "spatial_statistics": stats,
        "view_probabilities": views,
        "none_probability": np.asarray([[0.01]], dtype=np.float32),
    }


def test_action_entropy_preserves_sample_support_scale():
    concentrated = action_distribution_entropy(["a"] * 32)
    two_modes = action_distribution_entropy(["a"] * 16 + ["b"] * 16)
    unique = action_distribution_entropy([str(i) for i in range(32)])
    assert concentrated["normalized_entropy"] == 0.0
    assert 0.0 < two_modes["normalized_entropy"] < unique["normalized_entropy"]
    assert math.isclose(unique["normalized_entropy"], 1.0)


def test_panorama_bearing_coordinate_contract():
    assert bearing_to_view_coordinate(0.0) == (0, 0.0)
    view, x = bearing_to_view_coordinate(-math.pi / 2.0)
    assert view == 1 and abs(x) < 1e-6
    view, x = bearing_to_view_coordinate(math.pi / 2.0)
    assert view == 3 and abs(x) < 1e-6


def test_matched_heatmap_modulates_true_pose_revisit():
    matched = candidate_history_risk([1, 1], _context(0))
    shuffled = candidate_history_risk([1, 1], _context(2))
    assert matched["pose_only"] == shuffled["pose_only"]
    assert matched["heatmap_only"] > shuffled["heatmap_only"]
    assert matched["hybrid"] > shuffled["hybrid"]


def test_loop_detector_uses_only_actions_and_odometry():
    actions = [2, 3, 2, 3]
    visited = np.stack([_pose(0.0, 0.0) for _ in actions])
    signals = deployable_loop_signals(actions, visited, _pose(0.0, 0.0))
    assert signals["confirmed"]
    assert signals["turn_oscillation"]


def test_veto_requires_loop_risk_margin_mass_and_direction_change():
    baseline = {
        "treatment_id": "base",
        "spec": {"actions": [1, 1], "execute_len": 2},
        "native_sample_mass": 0.50,
    }
    alternative = {
        "treatment_id": "alt",
        "spec": {"actions": [2, 1], "execute_len": 2},
        "native_sample_mass": 0.25,
    }
    thresholds = VetoThresholds(
        risk_on=0.20,
        risk_margin=0.10,
        minimum_native_mass=0.10,
        maximum_edit_fraction=0.50,
    )
    no_loop = select_directional_veto(
        baseline_id="base",
        candidates=[baseline, alternative],
        risks={"base": 0.50, "alt": 0.10},
        loop_confirmed=False,
        thresholds=thresholds,
    )
    assert not no_loop["intervened"]
    selected = select_directional_veto(
        baseline_id="base",
        candidates=[baseline, alternative],
        risks={"base": 0.50, "alt": 0.10},
        loop_confirmed=True,
        thresholds=thresholds,
    )
    assert selected["intervened"] and selected["treatment_id"] == "alt"


def test_adaptive_prefix_never_changes_direction():
    selected = select_adaptive_prefix(
        baseline_id="full",
        baseline_execute_len=4,
        prefix_ids_by_length={1: "p1", 2: "p2", 4: "full"},
        normalized_entropy=0.90,
    )
    assert selected["treatment_id"] == "p1"
    assert selected["selected_execute_len"] == 1


def test_recovery_partition_uses_loop_entry_and_history_ages_only():
    visited = np.stack(
        [
            _pose(0.0, 0.0),
            _pose(0.25, 0.0),
            _pose(0.50, 0.0),
            _pose(0.50, 0.25),
            _pose(0.50, 0.50),
            _pose(0.25, 0.50),
            _pose(0.0, 0.50),
            _pose(0.0, 0.25),
        ]
    )
    partition = deployable_recovery_partition(
        fixed_history_mask=np.asarray([[True, True, True, True]]),
        fixed_history_age_steps=np.asarray([[8, 6, 4, 2]]),
        executed_actions=[1] * 8,
        visited_body_poses=visited,
        current_body_pose=_pose(0.0, 0.0),
    )
    assert partition["ready"]
    assert partition["loop_start_step"] == 0
    assert partition["anchor_index"] == 0
    assert partition["loop_history_mask"].tolist() == [False, True, True, True]


def test_recovery_energy_rewards_anchor_and_penalizes_loop_history():
    partition = {
        "ready": True,
        "anchor_index": 0,
        "loop_history_mask": np.asarray([False, True, True]),
    }
    matched = recovery_anchor_risk(
        {"raw_heatmap_by_history": np.asarray([0.8, 0.2, 0.1])},
        partition,
    )
    shuffled = recovery_anchor_risk(
        {"raw_heatmap_by_history": np.asarray([0.1, 0.8, 0.7])},
        partition,
    )
    assert math.isclose(matched["risk"], 0.2, abs_tol=1e-6)
    assert math.isclose(shuffled["risk"], 0.85, abs_tol=1e-6)
    assert matched["risk"] < shuffled["risk"]

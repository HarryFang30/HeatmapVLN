from collections import defaultdict

import numpy as np
import torch

from scripts.evaluation.probe_candidate_identifiability import (
    CandidateRanker,
    ProbeState,
    _collate,
    build_dev_episode_split,
    build_scene_split,
    candidate_feature_names,
    candidate_features,
    local_priority,
    ranking_loss,
    shuffle_heatmap_context,
    tune_abstention_threshold,
)


def _outcome(progress, *, collision=0):
    return {
        "entered_euclidean_success_radius": False,
        "left_euclidean_success_radius": False,
        "route_progress_delta_m": progress,
        "endpoint_offpath_m": 0.0,
        "collision_or_stuck_count": collision,
        "revisit": False,
        "min_euclidean_goal_distance_m": 4.0 - progress,
        "endpoint_euclidean_goal_distance_m": 4.0 - progress,
    }


def _treatment(treatment_id, actions, *, native=0, heatmap=0, mean=False):
    provenances = []
    if native:
        provenances.append(
            {
                "arm": "native",
                "aggregation": "trajectory_mean" if mean else "sample",
            }
        )
    if heatmap:
        provenances.append(
            {
                "arm": "heatmap_control",
                "aggregation": "sample",
            }
        )
    return {
        "treatment_id": treatment_id,
        "spec": {
            "actions": actions,
            "execute_len": len(actions),
            "replan_after": True,
            "trigger_anti_deadlock": False,
            "update_local_stop_counter": False,
            "end_reason": "queue_exhausted_replan",
        },
        "native_sample_mass": native / 32,
        "heatmap_sample_mass": heatmap / 32,
        "native_sample_count": native,
        "heatmap_sample_count": heatmap,
        "native_sample_total": 32,
        "heatmap_sample_total": 32,
        "provenances": provenances,
    }


def _state(key, scene, priorities):
    count = len(priorities)
    baseline = 0
    baseline_priority = priorities[baseline]
    return ProbeState(
        state_key=key,
        scene_id=scene,
        episode_id=key,
        candidate=np.zeros((count, len(candidate_feature_names())), dtype=np.float32),
        priorities=tuple(priorities),
        exact_priorities=tuple(priorities),
        best_mask=np.asarray(
            [value == max(priorities) for value in priorities], dtype=np.bool_
        ),
        baseline_preference=np.asarray(
            [
                int(value > baseline_priority) - int(value < baseline_priority)
                for value in priorities
            ],
            dtype=np.int8,
        ),
        baseline_index=baseline,
        system2_tokens=np.zeros((4, 6), dtype=np.float16),
        metadata=np.full((7,), float(key[-1]), dtype=np.float32),
        heatmap_tokens=np.full((3, 5), float(key[-1]), dtype=np.float16),
        heatmap_mask=np.ones((3,), dtype=np.bool_),
    )


def test_scene_split_is_strictly_scene_disjoint():
    records = [
        {"scene_id": f"scene_{scene}", "state_key": f"{scene}_{state}"}
        for scene in range(12)
        for state in range(scene + 1)
    ]
    mapping, summary = build_scene_split(
        records, seed=17, ratios=(0.7, 0.15, 0.15)
    )
    assert set(mapping) == {f"scene_{scene}" for scene in range(12)}
    split_scenes = [
        set(summary["splits"][name]["scenes"])
        for name in ("train", "validation", "test")
    ]
    assert all(split_scenes)
    assert not (split_scenes[0] & split_scenes[1])
    assert not (split_scenes[0] & split_scenes[2])
    assert not (split_scenes[1] & split_scenes[2])


def test_development_fallback_is_episode_disjoint_and_invalid_for_decisions():
    records = [
        {
            "scene_id": "one_scene",
            "episode_id": str(episode),
            "state_key": f"state_{episode}_{state}",
        }
        for episode in range(8)
        for state in range(episode + 1)
    ]
    mapping, summary = build_dev_episode_split(
        records, seed=17, ratios=(0.7, 0.15, 0.15)
    )
    assert set(mapping) == {record["state_key"] for record in records}
    assert summary["episode_disjoint"] is True
    assert summary["scene_disjoint"] is False
    assert summary["decision_valid"] is False
    episode_splits = defaultdict(set)
    for record in records:
        episode_splits[record["episode_id"]].add(mapping[record["state_key"]])
    assert all(len(splits) == 1 for splits in episode_splits.values())


def test_local_priority_uses_resolution_without_scalarizing():
    assert local_priority(_outcome(0.011), resolution_m=0.05) == local_priority(
        _outcome(0.019), resolution_m=0.05
    )
    assert local_priority(_outcome(0.08), resolution_m=0.05) > local_priority(
        _outcome(0.01), resolution_m=0.05
    )


def test_candidate_features_mark_baseline_and_relative_actions():
    baseline = _treatment("base", [1, 1], native=32, mean=True)
    candidate = _treatment("candidate", [2, 1], native=4, heatmap=8)
    names = candidate_feature_names()
    baseline_value = candidate_features(
        baseline, baseline_treatment_id="base", baseline_actions=[1, 1]
    )
    candidate_value = candidate_features(
        candidate, baseline_treatment_id="base", baseline_actions=[1, 1]
    )
    assert baseline_value[names.index("is_native_mean_baseline")] == 1.0
    assert candidate_value[names.index("is_native_mean_baseline")] == 0.0
    assert candidate_value[names.index("baseline_hamming_fraction")] > 0.0
    assert candidate_value[names.index("has_heatmap_provenance")] == 1.0


def test_ranker_forward_and_loss_cover_candidate_context_interaction():
    priorities = ((0.0,), (1.0,), (-1.0,))
    batch = _collate([_state("state1", "scene", priorities), _state("state2", "scene", priorities)])
    model = CandidateRanker(
        variant="candidate_system2_heatmap_tokens",
        candidate_width=batch["candidate"].shape[-1],
        system2_width=batch["system2_tokens"].shape[-1],
        metadata_width=batch["metadata"].shape[-1],
        heatmap_width=batch["heatmap_tokens"].shape[-1],
        hidden_width=16,
        dropout=0.0,
    )
    scores = model(batch)
    assert scores.shape == (2, 3)
    assert torch.isfinite(ranking_loss(scores, batch))

    invalid = _state("state3", "scene", priorities)
    invalid.heatmap_tokens = np.zeros((0, 5), dtype=np.float16)
    invalid.heatmap_mask = np.zeros((0,), dtype=np.bool_)
    invalid_batch = _collate([invalid])
    invalid_scores = model(invalid_batch)
    assert invalid_scores.shape == (1, 3)
    assert torch.all(torch.isfinite(invalid_scores))


def test_threshold_tuning_respects_destroy_constraint_and_shuffle_keeps_labels():
    priorities = [((0.0,), (1.0,), (-1.0,)), ((0.0,), (-1.0,), (1.0,))]
    states = [
        _state("state1", "scene", priorities[0]),
        _state("state2", "scene", priorities[1]),
    ]
    scores = [np.asarray([0.0, 2.0, -1.0]), np.asarray([0.0, 1.0, 0.5])]
    threshold, summary = tune_abstention_threshold(
        states, scores, max_destroy_state_rate=0.0
    )
    assert summary["destroyed_states"] == 0
    assert summary["positive_states"] == 1
    assert threshold >= 1.0

    shuffled, shuffle_summary = shuffle_heatmap_context(states, seed=11)
    assert shuffle_summary["changed_states"] == 2
    assert shuffled[0].priorities == states[0].priorities
    assert shuffled[1].priorities == states[1].priorities
    assert np.array_equal(shuffled[0].metadata, states[1].metadata)
    assert np.array_equal(shuffled[1].metadata, states[0].metadata)

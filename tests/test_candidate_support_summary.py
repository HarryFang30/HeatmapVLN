import numpy as np

from scripts.evaluation.candidate_support_audit import build_candidate_set
from scripts.evaluation.summarize_candidate_support_audit import summarize


def _trajectories(codes):
    value = np.zeros((len(codes), 2, 3), dtype=np.float32)
    value[:, 0, 0] = np.asarray(codes, dtype=np.float32)
    return value


def _converter(trajectory):
    code = int(round(float(trajectory[0, 0])))
    return {1: [1], 2: [2], 3: [3]}[code]


def _outcome(treatment_id, progress):
    return {
        "treatment_id": treatment_id,
        "actions_executed": [],
        "travelled_m": 0.0,
        "endpoint_offpath_m": 0.0,
        "endpoint_route_progress_m": progress,
        "route_progress_delta_m": progress,
        "endpoint_euclidean_goal_distance_m": 5.0 - progress,
        "min_euclidean_goal_distance_m": 5.0 - progress,
        "collision_or_stuck_count": 0,
        "revisit": False,
        "entered_euclidean_success_radius": False,
        "left_euclidean_success_radius": False,
    }


def test_summary_separates_native_and_heatmap_proposal_support():
    candidates = build_candidate_set(
        _trajectories([1, 1]),
        heatmap_trajectories=_trajectories([1, 2]),
        trajectory_to_actions=_converter,
    ).to_dict()
    arms = {}
    for treatment in candidates["treatments"]:
        arms[treatment["treatment_id"]] = {
            provenance["arm"] for provenance in treatment["provenances"]
        }
    heatmap_only = next(
        treatment_id
        for treatment_id, source_arms in arms.items()
        if source_arms == {"heatmap_control"}
    )
    outcomes = [
        _outcome(treatment["treatment_id"], 0.0)
        for treatment in candidates["treatments"]
    ]
    for outcome in outcomes:
        if outcome["treatment_id"] == heatmap_only:
            outcome["route_progress_delta_m"] = 1.0

    result = summarize(
        [
            {
                "schema": "counterfactual-candidate-audit-v1",
                "state_key": "state",
                "candidate_set": candidates,
                "local_outcomes": outcomes,
                "state_strata": {"primary_native_distribution": True},
            }
        ],
        [{"array_bytes": 123}],
    )
    assert result["decision_status"] == "insufficient_local_only"
    assert result["overall"]["local_positive_support_native_rate"] == 0.0
    assert result["overall"]["local_positive_support_union_rate"] == 1.0
    assert result["overall"]["heatmap_adds_positive_support_rate"] == 1.0
    assert (
        result["candidate_count_sensitivity"]["native"]["1"][
            "positive_support_rate"
        ]
        == 0.0
    )
    assert (
        result["candidate_count_sensitivity"]["heatmap_control"]["1"][
            "positive_support_rate"
        ]
        == 1.0
    )
    assert (
        result["candidate_count_sensitivity"]["paired_union"]["1"][
            "positive_support_rate"
        ]
        == 1.0
    )
    assert result["storage"]["compressed_array_bytes"] == 123

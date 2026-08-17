import json

import pytest
from scripts.evaluation.stop_dagger import (
    BoundaryProbeSweepState,
    OracleRecoveryState,
    parse_historical_false_stop_trigger,
    prune_stop_collection_jsonl_for_resume,
    should_finish_oracle_recovery_collection,
    should_force_continue_negative,
    should_record_stop_multimodal_example,
    validate_boundary_probe_collection,
    validate_historical_false_stop_source,
    validate_oracle_path_collection,
    validate_oracle_recovery_actions_per_call,
    validate_oracle_recovery_collection,
)


def test_prune_stop_collection_jsonl_for_resume_keeps_only_committed_episodes(
    tmp_path,
):
    labels = tmp_path / "labels.jsonl"
    rows = [
        {"scene_id": "scene-a", "episode_id": 1, "call": 0},
        {"scene_id": "scene-a", "episode_id": 1, "call": 1},
        {"scene_id": "scene-b", "episode_id": 2, "call": 0},
    ]
    labels.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    kept, dropped = prune_stop_collection_jsonl_for_resume(
        labels,
        {("scene-a", 1)},
    )

    assert (kept, dropped) == (2, 1)
    assert [json.loads(line) for line in labels.read_text().splitlines()] == rows[:2]


def test_prune_stop_collection_jsonl_for_resume_is_fail_closed_and_atomic(tmp_path):
    labels = tmp_path / "labels.jsonl"
    original = '{"scene_id":"scene-a","episode_id":1}\nnot-json\n'
    labels.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid STOP collection row"):
        prune_stop_collection_jsonl_for_resume(labels, {("scene-a", 1)})

    assert labels.read_text(encoding="utf-8") == original


def test_force_continue_only_applies_to_labelled_terminal_negatives():
    common = {
        "collection_enabled": True,
        "force_continue_negatives": True,
        "terminal": True,
    }

    assert should_force_continue_negative(**common, rollout_label=0) is True
    assert should_force_continue_negative(**common, rollout_label=1) is False
    assert should_force_continue_negative(**common, rollout_label=None) is False
    assert should_force_continue_negative(
        **{**common, "terminal": False},
        rollout_label=0,
    ) is False


def test_force_continue_requires_collection_and_valid_label():
    with pytest.raises(ValueError, match="requires STOP feature collection"):
        should_force_continue_negative(
            collection_enabled=False,
            force_continue_negatives=True,
            terminal=True,
            rollout_label=0,
        )
    with pytest.raises(ValueError, match="Invalid STOP rollout label"):
        should_force_continue_negative(
            collection_enabled=True,
            force_continue_negatives=True,
            terminal=True,
            rollout_label=2,
        )


def test_selective_multimodal_collection_keeps_stop_labels_and_hard_regulars():
    common = {
        "regular_min_stop_log_odds": -10.0,
        "episode_has_record": True,
    }

    assert should_record_stop_multimodal_example(
        **common,
        rollout_label=1,
        original_terminal=False,
        stop_log_odds=-30.0,
    )
    assert should_record_stop_multimodal_example(
        **common,
        rollout_label=0,
        original_terminal=True,
        stop_log_odds=4.0,
    )
    assert should_record_stop_multimodal_example(
        **common,
        rollout_label=0,
        original_terminal=False,
        stop_log_odds=-9.5,
    )
    assert not should_record_stop_multimodal_example(
        **common,
        rollout_label=0,
        original_terminal=False,
        stop_log_odds=-10.0,
    )
    assert should_record_stop_multimodal_example(
        rollout_label=0,
        original_terminal=False,
        stop_log_odds=-10.0,
        regular_min_stop_log_odds=-10.0,
        episode_has_record=False,
    )
    assert not should_record_stop_multimodal_example(
        **common,
        rollout_label=None,
        original_terminal=False,
        stop_log_odds=2.0,
    )


def test_selective_multimodal_collection_is_fail_closed():
    with pytest.raises(ValueError, match="finite stop_log_odds"):
        should_record_stop_multimodal_example(
            rollout_label=0,
            original_terminal=False,
            stop_log_odds=None,
            regular_min_stop_log_odds=-10.0,
            episode_has_record=False,
        )
    with pytest.raises(ValueError, match="threshold must be finite"):
        should_record_stop_multimodal_example(
            rollout_label=0,
            original_terminal=False,
            stop_log_odds=-5.0,
            regular_min_stop_log_odds=float("nan"),
            episode_has_record=False,
        )


def test_unfiltered_multimodal_collection_preserves_legacy_behavior():
    assert should_record_stop_multimodal_example(
        rollout_label=None,
        original_terminal=False,
        stop_log_odds=None,
        regular_min_stop_log_odds=None,
        episode_has_record=False,
    )


def test_oracle_recovery_is_restricted_to_forced_dagger_collection():
    assert validate_oracle_recovery_collection(
        collection_enabled=True,
        force_continue_negatives=True,
        oracle_recovery_after_negative=True,
    ) is True
    assert validate_oracle_recovery_collection(
        collection_enabled=False,
        force_continue_negatives=False,
        oracle_recovery_after_negative=False,
    ) is False
    with pytest.raises(ValueError, match="requires feature collection"):
        validate_oracle_recovery_collection(
            collection_enabled=True,
            force_continue_negatives=False,
            oracle_recovery_after_negative=True,
        )


def test_oracle_recovery_persists_through_positive_stop_until_complete():
    state = OracleRecoveryState()

    assert state.observe(terminal=False, rollout_label=0) is False
    assert state.observe(terminal=True, rollout_label=0) is True
    assert state.active is True
    assert state.activations == 1

    assert state.observe(terminal=False, rollout_label=None) is True
    assert state.observe(terminal=False, rollout_label=1) is True
    assert state.observe(terminal=True, rollout_label=0) is True
    assert state.activations == 1

    assert state.observe(terminal=True, rollout_label=1) is True
    assert state.active is True
    state.complete()
    assert state.active is False


def test_oracle_path_from_start_is_restricted_and_activates_once():
    assert validate_oracle_path_collection(
        collection_enabled=True,
        force_continue_negatives=True,
        oracle_path_from_start=True,
    ) is True
    with pytest.raises(ValueError, match="path-from-start"):
        validate_oracle_path_collection(
            collection_enabled=True,
            force_continue_negatives=False,
            oracle_path_from_start=True,
        )

    state = OracleRecoveryState()
    assert state.activate_from_start() is True
    assert state.active is True
    assert state.activations == 1
    assert state.activation_reason == "oracle_path_from_start"
    assert state.activate_from_start() is True
    assert state.activations == 1


def test_boundary_probe_sweep_is_privileged_and_bounded():
    assert validate_boundary_probe_collection(
        collection_enabled=True,
        force_continue_negatives=True,
        oracle_path_from_start=True,
        boundary_probe_sweep=True,
        min_distance_m=3.01,
        max_distance_m=6.0,
        probes=3,
    ) is True
    with pytest.raises(ValueError, match="requires forced oracle path"):
        validate_boundary_probe_collection(
            collection_enabled=True,
            force_continue_negatives=True,
            oracle_path_from_start=False,
            boundary_probe_sweep=True,
            min_distance_m=3.01,
            max_distance_m=6.0,
            probes=3,
        )

    state = BoundaryProbeSweepState(
        enabled=True,
        min_distance_m=3.01,
        max_distance_m=6.0,
        max_probes=3,
    )
    assert state.observe(distance_m=6.5, rollout_label=0) is None
    assert state.observe(distance_m=4.5, rollout_label=None) == 0
    assert state.finish_current_probe() is False
    assert state.observe(distance_m=4.5, rollout_label=0) == 1
    assert state.finish_current_probe() is False
    assert state.observe(distance_m=4.5, rollout_label=0) == 2
    assert state.finish_current_probe() is True
    assert state.observe(distance_m=4.5, rollout_label=0) is None


def test_oracle_recovery_action_chunk_is_positive_integer():
    assert validate_oracle_recovery_actions_per_call(1) == 1
    assert validate_oracle_recovery_actions_per_call(4) == 4
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="actions_per_call"):
            validate_oracle_recovery_actions_per_call(value)


def test_oracle_recovery_rejects_invalid_labels():
    state = OracleRecoveryState()
    with pytest.raises(ValueError, match="Invalid STOP rollout label"):
        state.observe(terminal=True, rollout_label=2)


def test_oracle_recovery_can_start_from_audited_cohort_trigger():
    state = OracleRecoveryState()

    assert state.activate_from_cohort(
        rollout_label=0,
        reason="historical_false_stop_call",
    ) is True
    assert state.active is True
    assert state.activations == 1
    assert state.activation_reason == "historical_false_stop_call"
    assert state.observe(terminal=True, rollout_label=1) is True


def test_historical_false_stop_trigger_requires_exact_seed_and_provenance():
    metadata = {
        "historical_false_stop_system2_call_index": 22,
        "historical_false_stop_step": 88,
        "historical_false_stop_distance_m": 4.428,
        "historical_false_stop_rpc_protocol_seed": 42,
        "historical_false_stop_source_labels": "/data/labels.jsonl",
    }

    trigger = parse_historical_false_stop_trigger(
        metadata,
        expected_protocol_seed=42,
        negative_radius_m=3.01,
    )

    assert trigger.system2_call_index == 22
    assert trigger.step == 88
    assert trigger.distance_m == pytest.approx(4.428)

    with pytest.raises(ValueError, match="seed mismatch"):
        parse_historical_false_stop_trigger(
            metadata,
            expected_protocol_seed=43,
            negative_radius_m=3.01,
        )


def test_historical_false_stop_source_matches_terminal_negative_evidence(tmp_path):
    feature_path = tmp_path / "feature.pt"
    feature_path.write_bytes(b"tensor")
    labels_path = tmp_path / "system2_stop_rollout_labels.jsonl"
    labels_path.write_text(
        json.dumps(
            {
                "scene_id": "scene-a",
                "episode_id": 7,
                "system2_call_index": 22,
                "step": 88,
                "distance_to_goal_m": 4.428,
                "original_terminal": True,
                "stop_target": 0,
                "path": str(feature_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "scene_id": "scene-a",
                "episode_id": 7,
                "rpc_protocol_seed": 42,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trigger = parse_historical_false_stop_trigger(
        {
            "historical_false_stop_system2_call_index": 22,
            "historical_false_stop_step": 88,
            "historical_false_stop_distance_m": 4.428,
            "historical_false_stop_rpc_protocol_seed": 42,
            "historical_false_stop_source_labels": str(labels_path),
        },
        expected_protocol_seed=42,
        negative_radius_m=3.01,
    )

    row = validate_historical_false_stop_source(
        trigger,
        scene_id="scene-a",
        episode_id=7,
    )

    assert row["original_terminal"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("historical_false_stop_system2_call_index", True),
        ("historical_false_stop_step", -1),
        ("historical_false_stop_distance_m", 2.0),
        ("historical_false_stop_source_labels", ""),
    ],
)
def test_historical_false_stop_trigger_rejects_invalid_metadata(field, value):
    metadata = {
        "historical_false_stop_system2_call_index": 22,
        "historical_false_stop_step": 88,
        "historical_false_stop_distance_m": 4.428,
        "historical_false_stop_rpc_protocol_seed": 42,
        "historical_false_stop_source_labels": "/data/labels.jsonl",
    }
    metadata[field] = value

    with pytest.raises(ValueError, match="historical|source|distance"):
        parse_historical_false_stop_trigger(
            metadata,
            expected_protocol_seed=42,
            negative_radius_m=3.01,
        )


def test_oracle_recovery_collection_stops_at_probe_limit():
    assert should_finish_oracle_recovery_collection(
        goal_probe_count=7,
        max_goal_probes=8,
    ) is False
    assert should_finish_oracle_recovery_collection(
        goal_probe_count=8,
        max_goal_probes=8,
    ) is True


@pytest.mark.parametrize(
    ("goal_probe_count", "max_goal_probes"),
    [(-1, 8), (0, 0), (True, 8), (0, False)],
)
def test_oracle_recovery_collection_rejects_invalid_limits(
    goal_probe_count,
    max_goal_probes,
):
    with pytest.raises(ValueError, match="Oracle recovery"):
        should_finish_oracle_recovery_collection(
            goal_probe_count=goal_probe_count,
            max_goal_probes=max_goal_probes,
        )

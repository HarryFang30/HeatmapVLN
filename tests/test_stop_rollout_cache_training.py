import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.training.train_stop_head_from_rollout_cache import (
    FEATURE_SCHEMA,
    _build_checkpoint_metrics,
    _build_sampling_weights,
    _add_validation_selection_score,
    _calibrate,
    _filter_training_scope,
    _load_features,
    _read_rows,
    _split_indices,
    _validation_selection_score,
)
from scripts.training.train_stop_head_add_scene_oof import (
    _annotate_sequence_identity,
    _build_optimization_targets,
    _event_calibration,
    _merge_crossfit_probe_rows,
    _probe_sweep_diagnostics,
    _select_probe_subset,
    _terminal_confirmation_scores,
)
from scripts.training.train_temporal_stop_verifier_from_rollout_cache import (
    _build_candidate_features as _build_temporal_candidate_features,
    _calibrate_add as _calibrate_temporal_add,
)
from scripts.training.train_temporal_stop_add_seed_ensemble_diagnostic import (
    _event_metrics as _temporal_add_event_metrics,
    _zero_false_event_threshold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_rollout_training_script_runs_as_direct_entrypoint():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/training/train_stop_head_from_rollout_cache.py"),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "--labels-jsonl" in result.stdout


def _write_feature(tmp_path, key, value):
    path = tmp_path / f"{key}.pth"
    torch.save(
        {
            "schema": FEATURE_SCHEMA,
            "key": key,
            "feature": torch.full((8,), float(value)),
        },
        path,
    )
    return path


def test_rollout_cache_rejects_eval_split_by_default(tmp_path):
    feature_path = _write_feature(tmp_path, "sample", 1.0)
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "key": "sample",
                "path": str(feature_path),
                "dataset_split": "val_unseen",
                "stop_target": 1,
            }
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="Refusing non-train"):
        _read_rows([labels], allow_nontrain=False)


def test_rollout_cache_can_relabel_exact_metric_boundary_negatives(tmp_path):
    rows = []
    for key, distance in (("inside_margin", 3.20), ("outside_margin", 3.25)):
        rows.append(
            {
                "key": key,
                "path": str(_write_feature(tmp_path, key, distance)),
                "dataset_split": "train",
                "scene_id": "scene",
                "episode_id": 1,
                "stop_target": None,
                "distance_to_goal_m": distance,
                "positive_radius_m": 3.0,
            }
        )
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps(row) + "\n" for row in rows))

    loaded = _read_rows(
        [labels],
        allow_nontrain=False,
        relabel_ambiguous_negative_radius_m=3.25,
    )

    assert [row["key"] for row in loaded] == ["outside_margin"]
    assert loaded[0]["stop_target"] == 0
    assert loaded[0]["ambiguous_negative_relabelled"] is True


def test_rollout_cache_deduplicates_deterministic_keys_and_prefers_intervention(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    rows = [
        {
            "key": "duplicate",
            "path": str(_write_feature(first_dir, "duplicate", 1.0)),
            "dataset_split": "train",
            "scene_id": "scene",
            "episode_id": 1,
            "system2_call_index": 2,
            "stop_target": 0,
            "original_terminal": True,
        },
        {
            "key": "duplicate",
            "path": str(_write_feature(second_dir, "duplicate", 1.0)),
            "dataset_split": "train",
            "scene_id": "scene",
            "episode_id": 1,
            "system2_call_index": 2,
            "stop_target": 0,
            "original_terminal": True,
            "oracle_forced_continue": True,
        },
    ]
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps(row) + "\n" for row in rows))

    loaded = _read_rows([labels], allow_nontrain=False)

    assert len(loaded) == 1
    assert loaded[0]["oracle_forced_continue"] is True
    assert Path(loaded[0]["path"]).parent == second_dir


def test_rollout_cache_rejects_conflicting_deterministic_key_labels(tmp_path):
    rows = []
    for target in (0, 1):
        feature_dir = tmp_path / str(target)
        feature_dir.mkdir()
        rows.append(
            {
                "key": "conflict",
                "path": str(_write_feature(feature_dir, "conflict", target)),
                "dataset_split": "train",
                "scene_id": "scene",
                "episode_id": 1,
                "system2_call_index": 2,
                "stop_target": target,
            }
        )
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(RuntimeError, match="Conflicting STOP labels"):
        _read_rows([labels], allow_nontrain=False)


def test_original_terminal_training_scope_keeps_only_stop_candidates():
    rows = [
        {"key": "ordinary_negative", "stop_target": 0, "original_terminal": False},
        {"key": "terminal_negative", "stop_target": 0, "original_terminal": True},
        {"key": "terminal_positive", "stop_target": 1, "original_terminal": True},
        {"key": "ordinary_positive", "stop_target": 1, "original_terminal": False},
    ]

    filtered = _filter_training_scope(rows, scope="original-terminal")

    assert [row["key"] for row in filtered] == [
        "terminal_negative",
        "terminal_positive",
    ]


def test_temporal_candidate_scope_separates_veto_and_add_examples():
    rows = []
    for call_index, (terminal, target) in enumerate(
        ((False, 0), (True, 0), (False, 1), (True, 1))
    ):
        rows.append(
            {
                "key": f"call_{call_index}",
                "source_index": 0,
                "scene_id": "scene",
                "episode_id": 7,
                "protocol_seed": 11,
                "system2_call_index": call_index,
                "hidden": torch.full((8,), float(call_index + 1)),
                "qwen_stop_log_odds": float(call_index),
                "original_terminal": terminal,
                "stop_target": target,
            }
        )
    static_probabilities = torch.tensor([0.1, 0.9, 0.8, 0.95])

    _, veto_targets, veto_rows = _build_temporal_candidate_features(
        rows,
        static_probabilities,
    )
    _, add_targets, add_rows = _build_temporal_candidate_features(
        rows,
        static_probabilities,
        candidate_scope="original_nonterminal",
    )

    assert [row["key"] for row in veto_rows] == ["call_1", "call_3"]
    assert veto_targets.tolist() == [0.0, 1.0]
    assert [row["key"] for row in add_rows] == ["call_0", "call_2"]
    assert add_targets.tolist() == [0.0, 1.0]


def test_temporal_add_calibration_requires_zero_false_adds():
    probabilities = torch.tensor([0.10, 0.81, 0.20, 0.95])
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])

    threshold, metrics = _calibrate_temporal_add(probabilities, targets)

    assert threshold == pytest.approx(0.815)
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["recall"] == pytest.approx(0.5)


def test_temporal_add_event_calibration_matches_two_confirmation_policy():
    rows = [
        {
            "source_index": 0,
            "scene_id": "negative",
            "episode_id": 1,
            "protocol_seed": 7,
            "system2_call_index": call,
        }
        for call in range(2)
    ] + [
        {
            "source_index": 0,
            "scene_id": "positive",
            "episode_id": 2,
            "protocol_seed": 7,
            "system2_call_index": call,
        }
        for call in range(2)
    ]
    scores = torch.tensor([0.9, 0.8, 0.85, 0.9])
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])

    threshold = _zero_false_event_threshold(
        scores,
        targets,
        rows,
        confirmations=2,
    )
    metrics = _temporal_add_event_metrics(
        scores,
        targets,
        rows,
        threshold=threshold,
        confirmations=2,
    )

    assert threshold > 0.8
    assert metrics["false_stop_episodes"] == 0
    assert metrics["true_stop_episodes"] == 1


def test_static_add_oof_sequence_identity_separates_rollout_sources(tmp_path):
    roots = [tmp_path / "first", tmp_path / "second"]
    rows = []
    for root_index, root in enumerate(roots):
        path = root / "system2_stop_features" / f"sample_{root_index}.pth"
        path.parent.mkdir(parents=True)
        rows.append(
            {
                "key": f"scene_ep000001_call00000_seed{40 + root_index}",
                "path": str(path),
                "scene_id": "scene",
                "episode_id": 1,
                "system2_call_index": 0,
            }
        )

    annotated = _annotate_sequence_identity(rows)

    assert [row["source_index"] for row in annotated] == [0, 1]
    assert [row["protocol_seed"] for row in annotated] == [40, 41]


def test_static_add_oof_event_calibration_respects_minimum_threshold():
    rows = [
        {
            "source_index": 0,
            "scene_id": scene,
            "episode_id": episode,
            "protocol_seed": 7,
            "system2_call_index": call,
        }
        for scene, episode in (("negative", 1), ("positive", 2))
        for call in range(2)
    ]
    probabilities = torch.tensor([0.2, 0.3, 0.92, 0.94])
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])

    threshold, metrics = _event_calibration(
        probabilities,
        targets,
        rows,
        confirmations=2,
        minimum_threshold=0.9,
    )

    assert threshold == pytest.approx(0.9)
    assert metrics["false_stop_episodes"] == 0
    assert metrics["true_stop_episodes"] == 1


def test_static_oof_terminal_confirmation_masks_nonterminal_calls():
    rows = []
    probabilities = []
    targets = []
    for scene, episode, terminal_scores, target, nonterminal_call in (
        ("negative", 1, [0.8, 0.7, 0.9, 0.6], 0.0, 1),
        ("positive", 2, [0.75, 0.8, 0.85, 0.9], 1.0, None),
    ):
        for call, score in enumerate(terminal_scores):
            rows.append(
                {
                    "source_index": 0,
                    "scene_id": scene,
                    "episode_id": episode,
                    "protocol_seed": 7,
                    "system2_call_index": call,
                    "original_terminal": call != nonterminal_call,
                }
            )
            probabilities.append(score if call != nonterminal_call else 0.99)
            targets.append(target)
    probability_tensor = torch.tensor(probabilities)
    target_tensor = torch.tensor(targets)

    scores = _terminal_confirmation_scores(probability_tensor, rows)
    threshold, metrics = _event_calibration(
        scores,
        target_tensor,
        rows,
        confirmations=3,
        minimum_threshold=0.0,
    )

    assert torch.isneginf(scores[1])
    assert not torch.isneginf(scores[5])
    assert threshold == pytest.approx(0.0)
    assert metrics["false_stop_episodes"] == 0
    assert metrics["true_stop_episodes"] == 1


def test_static_oof_probe_diagnostics_aggregate_complete_fixed_sweeps():
    rows = []
    probabilities = []
    for episode, kind, target, scores in (
        (1, "boundary", 0, [0.1, 0.2]),
        (1, "goal", 1, [0.8, 0.9]),
        (2, "boundary", 0, [0.3, 0.4]),
        (2, "goal", 1, [0.7, 0.6]),
    ):
        for index, score in enumerate(scores):
            rows.append(
                {
                    "scene_id": f"scene_{episode}",
                    "episode_id": episode,
                    "stop_target": target,
                    "distance_to_goal_m": 5.0 if kind == "boundary" else 2.0,
                    "boundary_probe_views": 2,
                    "boundary_probe_sweep": kind == "boundary",
                    "boundary_probe_index": index if kind == "boundary" else None,
                    "boundary_probe_sweep_id": (
                        f"scene_{episode}:{episode}:boundary"
                        if kind == "boundary"
                        else None
                    ),
                    "goal_probe_sweep": kind == "goal",
                    "goal_probe_index": index if kind == "goal" else None,
                    "goal_probe_sweep_id": (
                        f"scene_{episode}:{episode}:goal" if kind == "goal" else None
                    ),
                }
            )
            probabilities.append(score)

    summary, groups = _probe_sweep_diagnostics(
        rows,
        torch.tensor(probabilities),
    )

    assert len(groups) == 4
    assert summary["group_auc"] == 1.0
    assert summary["paired_goal_mean_wins"] == 2
    assert summary["zero_false_boundary_goal_groups"] == 2


def test_static_oof_selects_crossfit_probe_rows_from_training_set():
    rows = [
        {"key": "ordinary", "stop_target": 0},
        {"key": "boundary", "boundary_probe_sweep": True, "stop_target": 0},
        {"key": "goal", "goal_probe_sweep": True, "stop_target": 1},
    ]
    indices, selected = _select_probe_subset(
        rows,
        [rows[2], rows[0], rows[1]],
    )

    assert indices == [2, 1]
    assert [row["key"] for row in selected] == ["goal", "boundary"]


def test_static_oof_crossfit_training_uses_only_fixed_sweep_rows():
    base = [{"key": "base", "stop_target": 0}]
    merged, probes = _merge_crossfit_probe_rows(
        base,
        [
            {"key": "navigate", "stop_target": 0},
            {"key": "boundary", "boundary_probe_sweep": True, "stop_target": 0},
            {"key": "goal", "goal_probe_sweep": True, "stop_target": 1},
        ],
    )

    assert [row["key"] for row in merged] == ["base", "boundary", "goal"]
    assert [row["key"] for row in probes] == ["boundary", "goal"]


def test_rollout_cache_relabel_radius_must_exceed_positive_radius(tmp_path):
    feature_path = _write_feature(tmp_path, "sample", 3.1)
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "key": "sample",
                "path": str(feature_path),
                "dataset_split": "train",
                "scene_id": "scene",
                "episode_id": 1,
                "stop_target": None,
                "distance_to_goal_m": 3.1,
                "positive_radius_m": 3.0,
            }
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="must exceed the positive radius"):
        _read_rows(
            [labels],
            allow_nontrain=False,
            relabel_ambiguous_negative_radius_m=3.0,
        )


def test_rollout_cache_relabel_keeps_missing_distance_ambiguous(tmp_path):
    rows = [
        {
            "key": "missing_distance",
            "path": str(_write_feature(tmp_path, "missing_distance", 0.0)),
            "dataset_split": "train",
            "scene_id": "scene",
            "episode_id": 1,
            "stop_target": None,
            "distance_to_goal_m": None,
            "positive_radius_m": 3.0,
        },
        {
            "key": "labelled_negative",
            "path": str(_write_feature(tmp_path, "labelled_negative", 0.0)),
            "dataset_split": "train",
            "scene_id": "scene",
            "episode_id": 1,
            "stop_target": 0,
        },
    ]
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps(row) + "\n" for row in rows))

    loaded = _read_rows(
        [labels],
        allow_nontrain=False,
        relabel_ambiguous_negative_radius_m=3.25,
    )

    assert [row["key"] for row in loaded] == ["labelled_negative"]


def test_rollout_cache_loads_features_and_splits_scene_disjoint(tmp_path):
    rows = []
    for scene_index in range(6):
        for target in (0, 1):
            key = f"scene{scene_index}_{target}"
            rows.append(
                {
                    "key": key,
                    "path": str(_write_feature(tmp_path, key, target)),
                    "dataset_split": "train",
                    "scene_id": f"scene{scene_index}",
                    "episode_id": scene_index,
                    "stop_target": target,
                    "original_terminal": bool(target),
                }
            )

    features, targets = _load_features(rows, workers=4)
    train_indices, val_indices, group_kind = _split_indices(
        rows,
        targets,
        val_fraction=0.25,
        seed=42,
    )

    assert features.shape == (12, 8)
    assert group_kind == "scene"
    assert {int(targets[index]) for index in train_indices} == {0, 1}
    assert {int(targets[index]) for index in val_indices} == {0, 1}
    train_scenes = {rows[index]["scene_id"] for index in train_indices}
    val_scenes = {rows[index]["scene_id"] for index in val_indices}
    assert train_scenes.isdisjoint(val_scenes)


def test_rollout_cache_parallel_feature_load_preserves_input_order(tmp_path):
    rows = []
    for index in range(16):
        key = f"sample_{index}"
        rows.append(
            {
                "key": key,
                "path": str(_write_feature(tmp_path, key, index)),
                "stop_target": index % 2,
            }
        )

    sequential_features, sequential_targets = _load_features(rows)
    parallel_features, parallel_targets = _load_features(rows, workers=8)

    assert torch.equal(parallel_features, sequential_features)
    assert torch.equal(parallel_targets, sequential_targets)


def test_validation_selection_penalizes_premature_stop_more_than_missed_stop():
    missed_stop = {"recall": 0.8, "false_positive_rate": 0.0}
    premature_stop = {"recall": 1.0, "false_positive_rate": 0.2}

    assert _validation_selection_score(missed_stop) == pytest.approx(0.2)
    assert _validation_selection_score(premature_stop) == pytest.approx(0.4)


def test_add_selection_prioritizes_zero_false_adds_then_recall():
    no_false_adds = {"recall": 0.2, "false_positive_rate": 0.0}
    more_recall_with_false_add = {"recall": 1.0, "false_positive_rate": 0.01}
    better_zero_fpr_recall = {"recall": 0.7, "false_positive_rate": 0.0}

    assert _add_validation_selection_score(no_false_adds) < (
        _add_validation_selection_score(more_recall_with_false_add)
    )
    assert _add_validation_selection_score(better_zero_fpr_recall) < (
        _add_validation_selection_score(no_false_adds)
    )


def test_rollout_checkpoint_metrics_preserve_preflight_threshold_contract():
    probabilities = torch.tensor([0.01, 0.2, 0.95, 0.99])
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])

    metrics = _build_checkpoint_metrics(probabilities, targets)

    assert metrics["val_stop_add_stop_threshold"] == metrics["add_stop_threshold"]
    assert metrics["val_stop_veto_stop_threshold"] == metrics["veto_stop_threshold"]


def test_veto_calibration_uses_closed_loop_cost_and_conservative_tie_break():
    probabilities = torch.tensor([0.10, 0.20, 0.55, 0.10, 0.70])
    targets = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])

    add_threshold, veto_threshold = _calibrate(probabilities, targets)

    assert add_threshold == pytest.approx(0.9)
    assert veto_threshold == pytest.approx(0.555)
    veto_metrics = _build_checkpoint_metrics(probabilities, targets)["val_at_veto_threshold"]
    assert veto_metrics["recall"] == pytest.approx(0.5)
    assert veto_metrics["false_positive_rate"] == pytest.approx(0.0)


def test_veto_calibration_uses_only_original_terminal_states():
    probabilities = torch.tensor([0.89, 0.88, 0.70, 0.80])
    targets = torch.tensor([0.0, 0.0, 0.0, 1.0])
    original_terminal = torch.tensor([False, False, True, True])

    _, all_records_veto = _calibrate(probabilities, targets)
    _, terminal_veto = _calibrate(probabilities, targets, original_terminal)
    metrics = _build_checkpoint_metrics(
        probabilities,
        targets,
        original_terminal,
    )

    assert all_records_veto == pytest.approx(0.895)
    assert terminal_veto == pytest.approx(0.705)
    assert metrics["veto_calibration_scope"] == "original_terminal"
    assert metrics["veto_calibration_records"] == 2
    assert metrics["val_at_veto_threshold"]["false_positive_rate"] == 0.0
    assert metrics["val_at_veto_threshold"]["recall"] == 1.0


def test_veto_calibration_falls_back_when_terminal_subset_has_one_class():
    probabilities = torch.tensor([0.10, 0.20, 0.80, 0.90])
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])
    original_terminal = torch.tensor([False, False, False, True])

    metrics = _build_checkpoint_metrics(
        probabilities,
        targets,
        original_terminal,
    )

    assert metrics["veto_calibration_scope"] == "all_records_fallback"
    assert metrics["veto_calibration_records"] == 4


def test_add_calibration_requires_zero_validation_false_positives():
    probabilities = torch.tensor([0.10, 0.91, 0.20, 0.95])
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])

    add_threshold, _ = _calibrate(probabilities, targets)

    assert add_threshold == pytest.approx(0.915)
    add_metrics = _build_checkpoint_metrics(probabilities, targets)["val_at_add_threshold"]
    assert add_metrics["false_positive_rate"] == pytest.approx(0.0)


def test_sampling_weights_focus_hard_negatives_without_changing_class_balance():
    rows = [
        {"original_terminal": False},
        {"original_terminal": False},
        {"original_terminal": False},
        {"original_terminal": True},
    ]
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    initial_probabilities = torch.tensor([0.9, 0.8, 0.2, 0.95])

    weights, stats = _build_sampling_weights(
        rows,
        targets,
        [0, 1, 2, 3],
        initial_probabilities,
        terminal_negative_weight=2.0,
        hard_negative_threshold=0.8,
        hard_negative_weight=3.0,
    )

    assert weights[:2].sum().item() == pytest.approx(0.5)
    assert weights[2:].sum().item() == pytest.approx(0.5)
    assert weights[3].item() > weights[2].item()
    assert stats == {
        "positive_count": 2,
        "negative_count": 2,
        "recovery_positive_count": 0,
        "terminal_negative_count": 1,
        "hard_negative_count": 1,
        "boundary_negative_count": 0,
    }


def test_sampling_weights_upweight_oracle_recovery_positives_within_class():
    rows = [
        {"oracle_recovery_active": False},
        {"oracle_recovery_active": True},
        {"original_terminal": False},
        {"original_terminal": False},
    ]
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    initial_probabilities = torch.tensor([0.2, 0.2, 0.2, 0.2])

    weights, stats = _build_sampling_weights(
        rows,
        targets,
        [0, 1, 2, 3],
        initial_probabilities,
        terminal_negative_weight=1.0,
        hard_negative_threshold=0.8,
        hard_negative_weight=1.0,
        oracle_recovery_positive_weight=4.0,
    )

    assert weights[:2].sum().item() == pytest.approx(0.5)
    assert weights[2:].sum().item() == pytest.approx(0.5)
    assert weights[1].item() == pytest.approx(4.0 * weights[0].item())
    assert stats["recovery_positive_count"] == 1


def test_sampling_weights_upweight_boundary_negatives_within_class():
    rows = [
        {"distance_to_goal_m": 2.9},
        {"distance_to_goal_m": 3.5},
        {"distance_to_goal_m": 7.0},
    ]
    targets = torch.tensor([1.0, 0.0, 0.0])
    initial_probabilities = torch.tensor([0.2, 0.2, 0.2])

    weights, stats = _build_sampling_weights(
        rows,
        targets,
        [0, 1, 2],
        initial_probabilities,
        terminal_negative_weight=1.0,
        hard_negative_threshold=0.8,
        hard_negative_weight=1.0,
        boundary_negative_min_distance_m=3.01,
        boundary_negative_max_distance_m=6.0,
        boundary_negative_weight=4.0,
    )

    assert weights[0].item() == pytest.approx(0.5)
    assert weights[1].item() == pytest.approx(4.0 * weights[2].item())
    assert stats["boundary_negative_count"] == 1


def test_metric_margin_optimization_targets_preserve_evaluation_targets():
    rows = [
        {"distance_to_goal_m": 1.5},
        {"distance_to_goal_m": 2.5},
        {"distance_to_goal_m": 3.5},
        {"distance_to_goal_m": 4.5},
    ]
    evaluation_targets = torch.tensor([1.0, 1.0, 0.0, 0.0])

    optimization_targets = _build_optimization_targets(
        rows,
        evaluation_targets,
        positive_radius_m=2.0,
        negative_radius_m=4.0,
    )

    assert optimization_targets.tolist() == [1.0, -1.0, -1.0, 0.0]
    assert evaluation_targets.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_metric_margin_optimization_targets_require_both_radii():
    with pytest.raises(ValueError, match="must be set together"):
        _build_optimization_targets(
            [{"distance_to_goal_m": 1.0}],
            torch.tensor([1.0]),
            positive_radius_m=2.0,
            negative_radius_m=None,
        )

import pytest
import torch
from scripts.training.diagnose_stop_trajectory_latent_scene_oof import (
    _center_kernel,
    _confirmation_metrics,
    _group_metrics,
    _kernel_from_blocks,
    _ridge_oof_scores,
    _stable_scene_folds,
    _view_metrics,
)


def test_center_kernel_matches_explicit_train_mean_centering():
    features = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
            [0.0, 2.0, 3.0],
        ],
        dtype=torch.float64,
    )
    kernel = features @ features.T
    train_indices = [0, 1, 2]
    val_indices = [3, 4]

    centered_train, centered_val = _center_kernel(
        kernel,
        train_indices,
        val_indices,
    )

    train = features[train_indices]
    val = features[val_indices]
    mean = train.mean(dim=0)
    assert torch.allclose(centered_train, (train - mean) @ (train - mean).T)
    assert torch.allclose(centered_val, (val - mean) @ (train - mean).T)


def test_scene_folds_are_stable_disjoint_and_complete():
    scenes = [f"scene-{index}" for index in range(7)]

    first = _stable_scene_folds(scenes, fold_count=3, seed=123)
    second = _stable_scene_folds(list(reversed(scenes)), fold_count=3, seed=123)

    assert first == second
    assert sorted(scene for fold in first for scene in fold) == sorted(scenes)
    assert sum(len(fold) for fold in first) == len(set(scene for fold in first for scene in fold))


def test_ridge_oof_scores_preserve_scene_disjoint_class_signal():
    scenes = [f"scene-{index}" for index in range(4)]
    scene_ids = [scene for scene in scenes for _ in range(2)]
    targets = [target for _ in scenes for target in (0, 1)]
    features = torch.tensor(
        [[-1.0, float(index)] if target == 0 else [1.0, float(index)]
         for index in range(4) for target in (0, 1)]
    )
    kernel = _kernel_from_blocks([features])
    folds = [[scene] for scene in scenes]

    scores = _ridge_oof_scores(
        kernel,
        targets,
        scene_ids,
        folds,
        ridge=0.1,
    )

    for index in range(0, len(scores), 2):
        assert scores[index + 1] > scores[index]


def test_group_metrics_require_global_calibration_beyond_paired_ranking():
    rows = []
    scores = []
    for scene_index, scene in enumerate(("a", "b", "c")):
        for target, base in ((0, float(scene_index)), (1, float(scene_index) + 0.5)):
            for view in range(2):
                rows.append(
                    {
                        "scene_id": scene,
                        "target": target,
                        "sweep_id": f"{scene}:{target}",
                        "probe_index": view,
                    }
                )
                scores.append(base + 0.01 * view)

    metrics = _group_metrics(rows, scores, "mean")

    assert metrics["paired_wins"] == 3
    assert metrics["paired_total"] == 3
    assert metrics["zero_false_positive_goal_catches"] == 1
    assert metrics["auc"] == pytest.approx(2.0 / 3.0)

    views = _view_metrics(rows, scores)
    assert views["zero_false_positive_goal_views"] == 2
    assert views["positive_views"] == 6
    assert views["goal_groups_first_view_hit"] == 1
    assert views["goal_groups_first_two_views_hit"] == 1
    assert views["goal_groups_any_view_hit"] == 1


def test_confirmation_metrics_calibrate_consecutive_events_not_single_views():
    rows = []
    scores = []
    groups = {
        "a:0": ("a", 0, [10.0, 0.0, 9.0]),
        "a:1": ("a", 1, [5.0, 5.0, 5.0]),
        "b:0": ("b", 0, [8.0, -1.0, 7.0]),
        "b:1": ("b", 1, [4.0, 4.0, 4.0]),
    }
    for sweep_id, (scene, target, values) in groups.items():
        for probe_index, score in enumerate(values):
            rows.append(
                {
                    "scene_id": scene,
                    "target": target,
                    "sweep_id": sweep_id,
                    "probe_index": probe_index,
                }
            )
            scores.append(score)

    metrics = _confirmation_metrics(rows, scores, confirmations=2)

    assert metrics["first_window_zero_false_positive_threshold"] == pytest.approx(0.0)
    assert metrics["first_window_zero_false_positive_goal_catches"] == 2
    assert metrics["robust_any_boundary_window_threshold"] == pytest.approx(0.0)
    assert metrics["first_window_goal_catches_at_robust_threshold"] == 2

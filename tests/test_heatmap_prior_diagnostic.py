from __future__ import annotations

from pathlib import Path

import pytest
import torch
from scripts.tools.diagnose_heatmap_priors import (
    assert_scene_disjoint,
    evaluate_empirical_prior,
    fit_empirical_prior,
    selection_manifest,
)
from scripts.tools.summarize_task35 import (
    build_per_history_slot_diagnostic,
    build_verdict,
    paired_scene_cluster_bootstrap,
    scene_cluster_indices,
    scene_from_sample_id,
)


class _TinyPriorDataset:
    def __init__(self, root: Path, samples: list[dict], identities: list[tuple[str, str, int]]):
        self.root = root
        self.samples = samples
        self.clips = []
        clip_lookup: dict[tuple[str, str], int] = {}
        self.sample_index = []
        for scene, clip, frame in identities:
            key = (scene, clip)
            if key not in clip_lookup:
                clip_lookup[key] = len(self.clips)
                self.clips.append(root / scene / clip)
            self.sample_index.append((clip_lookup[key], frame))

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def _target(
    visibility: torch.Tensor,
    peaks: list[tuple[int, int, int, int]],
    *,
    height: int = 3,
    width: int = 3,
) -> dict:
    heatmap = torch.zeros(2, 4, height, width)
    for history, view, x, y in peaks:
        heatmap[history, view, y, x] = 1.0
    return {"gt_visibility": visibility.float(), "heatmap": heatmap}


def test_empirical_prior_uses_only_selected_training_targets(tmp_path):
    first = _target(
        torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]]),
        [(0, 0, 1, 0), (1, 1, 2, 1)],
    )
    second = _target(
        torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]]),
        [(0, 0, 1, 0), (0, 1, 0, 2)],
    )
    held_out = _target(torch.ones(2, 4), [(0, 0, 2, 2), (1, 3, 2, 2)])
    dataset = _TinyPriorDataset(
        tmp_path,
        [first, second, held_out],
        [("train_a", "clip_0", 5), ("train_b", "clip_1", 6), ("val", "clip_2", 7)],
    )

    prior = fit_empirical_prior(dataset, [0, 1], visibility_alpha=0.5)
    expected_probability = (
        first["gt_visibility"] + second["gt_visibility"] + 0.5
    ) / 3.0
    assert torch.allclose(prior["visibility_probability"], expected_probability)
    assert torch.allclose(
        prior["visibility_logits"].sigmoid(),
        expected_probability,
    )
    assert torch.allclose(
        prior["mean_heatmap"],
        (first["heatmap"] + second["heatmap"]) / 2.0,
    )

    # A held-out target mutation cannot alter a prior fit from indices [0, 1].
    baseline_logits = prior["visibility_logits"].clone()
    baseline_heatmap = prior["mean_heatmap"].clone()
    dataset.samples[2] = _target(torch.zeros(2, 4), [])
    refit = fit_empirical_prior(dataset, [0, 1], visibility_alpha=0.5)
    assert torch.equal(refit["visibility_logits"], baseline_logits)
    assert torch.equal(refit["mean_heatmap"], baseline_heatmap)


def test_selection_manifest_hashes_ordered_samples_and_checks_scene_disjoint(tmp_path):
    sample = _target(torch.zeros(2, 4), [])
    dataset = _TinyPriorDataset(
        tmp_path,
        [sample, sample, sample],
        [("scene_a", "clip_0", 5), ("scene_b", "clip_1", 6), ("scene_a", "clip_0", 7)],
    )
    manifest = selection_manifest(dataset, [0, 1, 2])
    reversed_manifest = selection_manifest(dataset, [2, 1, 0])
    assert manifest["sample_ids"] == [
        "scene_a/clip_0:frame=5",
        "scene_b/clip_1:frame=6",
        "scene_a/clip_0:frame=7",
    ]
    assert manifest["sample_identity_hash"] != reversed_manifest["sample_identity_hash"]
    assert manifest["scenes"] == ["scene_a", "scene_b"]
    assert manifest["scene_hash"] == reversed_manifest["scene_hash"]

    assert_scene_disjoint(
        {"scenes": ["scene_a"]},
        {"scenes": ["scene_b"]},
    )
    with pytest.raises(RuntimeError, match="scene_a"):
        assert_scene_disjoint(
            {"scenes": ["scene_a"]},
            {"scenes": ["scene_a", "scene_b"]},
        )


def test_prior_evaluation_reuses_logits_and_emits_compact_records(tmp_path):
    train = _target(
        torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]]),
        [(0, 0, 1, 0), (1, 1, 2, 1)],
    )
    dataset = _TinyPriorDataset(
        tmp_path,
        [train, train],
        [("train", "clip_0", 5), ("val", "clip_1", 6)],
    )
    prior = fit_empirical_prior(dataset, [0], visibility_alpha=0.5)
    metrics, records = evaluate_empirical_prior(dataset, [1], prior)

    assert metrics["samples"] == 1
    assert metrics["visible_history_count"] == 2
    assert metrics["median_pixel_error"] == 0.0
    assert len(records) == 1
    assert records[0]["sample_id"] == "val/clip_1:frame=6"
    assert records[0]["pred_xy"][0][0] == [1, 0]
    assert records[0]["gt_xy"][1][1] == [2, 1]
    assert torch.allclose(
        torch.tensor(records[0]["visibility_probability"]),
        prior["visibility_logits"].sigmoid(),
    )


def _compact_record(
    sample_id: str,
    *,
    pred_x: int,
    gt_x: int = 0,
) -> dict:
    return {
        "sample_id": sample_id,
        "visibility_logits": [[2.0, -2.0, -2.0, -2.0]],
        "gt_visibility": [[1.0, 0.0, 0.0, 0.0]],
        "pred_xy": [[[pred_x, 0], [0, 0], [0, 0], [0, 0]]],
        "gt_xy": [[[gt_x, 0], [0, 0], [0, 0], [0, 0]]],
    }


def test_scene_cluster_bootstrap_groups_split_qualified_sample_ids():
    records = [
        _compact_record("val/scene_a/clip_000001:frame=5", pred_x=0),
        _compact_record("val/scene_a/clip_000002:frame=7", pred_x=1),
        _compact_record("scene_b/clip_000003:frame=9", pred_x=2),
    ]
    assert scene_from_sample_id(records[0]["sample_id"]) == "scene_a"
    assert scene_from_sample_id(records[2]["sample_id"]) == "scene_b"
    clusters = scene_cluster_indices(records)
    assert list(clusters) == ["scene_a", "scene_b"]
    assert clusters["scene_a"].tolist() == [0, 1]
    assert clusters["scene_b"].tolist() == [2]


def test_paired_bootstrap_resamples_scene_clusters_and_reports_contract():
    baseline = [
        _compact_record("scene_a/clip_000001:frame=5", pred_x=12),
        _compact_record("scene_a/clip_000002:frame=7", pred_x=12),
        _compact_record("scene_b/clip_000003:frame=9", pred_x=10),
        _compact_record("scene_b/clip_000004:frame=11", pred_x=10),
    ]
    full = [
        _compact_record("scene_a/clip_000001:frame=5", pred_x=0),
        _compact_record("scene_a/clip_000002:frame=7", pred_x=0),
        _compact_record("scene_b/clip_000003:frame=9", pred_x=0),
        _compact_record("scene_b/clip_000004:frame=11", pred_x=0),
    ]
    result = paired_scene_cluster_bootstrap(
        full,
        baseline,
        samples=50,
        seed=42,
    )
    assert result["bootstrap_contract"]["resampling_unit"] == "scene"
    assert result["bootstrap_contract"]["scene_count"] == 2
    assert result["bootstrap_contract"]["scene_sample_counts"] == {
        "scene_a": 2,
        "scene_b": 2,
    }
    assert result["median_relative_improvement"]["ci95"][0] > 0.0
    assert result["pck8_delta"]["ci95"][0] > 0.0
    assert result["joint_pck8_delta"]["ci95"][0] > 0.0


def test_task35_verdict_requires_joint_pck8_effect_and_scene_cluster_ci():
    positive_metric = {"ci95": [0.01, 0.30], "mean": 0.15}
    bootstrap = {
        baseline: {
            "median_relative_improvement": positive_metric,
            "pck8_delta": positive_metric,
            "joint_pck8_delta": positive_metric,
        }
        for baseline in ("no-input", "empirical-prior")
    }
    effect = {
        "median_relative_improvement_over_stronger_null": 0.25,
        "pck8_delta_over_stronger_null": 0.15,
        "joint_pck8_delta_over_stronger_null": 0.09,
    }
    failed = build_verdict(effect, bootstrap)
    assert failed["checks"]["joint_pck8_effect"] is False
    assert failed["sample_specific_localization_passed"] is False

    effect["joint_pck8_delta_over_stronger_null"] = 0.10
    passed = build_verdict(effect, bootstrap)
    assert passed["thresholds"]["paired_ci_resampling_unit"] == "scene"
    assert passed["checks"]["joint_pck8_effect"] is True
    assert passed["sample_specific_localization_passed"] is True


def _two_slot_record(
    sample_id: str,
    *,
    errors: tuple[int, int],
    selected_views: tuple[int, int],
) -> dict:
    visibility_logits = []
    gt_visibility = []
    pred_xy = []
    gt_xy = []
    positive_views = (0, 1)
    for slot, positive_view in enumerate(positive_views):
        logits = [-2.0] * 4
        logits[selected_views[slot]] = 2.0
        visibility_logits.append(logits)
        visibility = [0.0] * 4
        visibility[positive_view] = 1.0
        gt_visibility.append(visibility)
        pred_views = [[0, 0] for _ in range(4)]
        pred_views[positive_view] = [errors[slot], 0]
        pred_xy.append(pred_views)
        gt_xy.append([[0, 0] for _ in range(4)])
    return {
        "sample_id": sample_id,
        "visibility_logits": visibility_logits,
        "gt_visibility": gt_visibility,
        "pred_xy": pred_xy,
        "gt_xy": gt_xy,
    }


def _report_with_records(records: list[dict]) -> dict:
    return {"evaluations": {"standard": {"prediction_records": records}}}


def test_post_hoc_history_slot_diagnostic_reports_modes_and_stronger_null_effects():
    sample_id = "scene_a/clip_000001:frame=5"
    reports = {
        "full": _report_with_records(
            [_two_slot_record(sample_id, errors=(0, 0), selected_views=(0, 1))]
        ),
        "no-input": _report_with_records(
            [_two_slot_record(sample_id, errors=(12, 12), selected_views=(0, 0))]
        ),
        "empirical-prior": _report_with_records(
            [_two_slot_record(sample_id, errors=(10, 4), selected_views=(0, 1))]
        ),
    }
    diagnostic = build_per_history_slot_diagnostic(reports)

    assert diagnostic["post_hoc"] is True
    assert diagnostic["affects_aggregate_verdict"] is False
    assert diagnostic["num_history_slots"] == 2
    full_slot_1 = diagnostic["modes"]["full"]["slot_1"]
    assert full_slot_1["visible_history_count"] == 1
    assert full_slot_1["visible_view_accuracy"] == 1.0
    assert full_slot_1["median_pixel_error"] == 0.0
    assert full_slot_1["pck8"] == 1.0
    assert full_slot_1["joint_pck8"] == 1.0

    slot_0 = diagnostic["full_vs_stronger_null"]["slot_0"]
    assert slot_0["stronger_null"]["median_pixel_error"] == {
        "value": 10.0,
        "source_modes": ["empirical-prior"],
    }
    assert slot_0["effect"]["median_relative_improvement_over_stronger_null"] == 1.0
    assert slot_0["effect"]["pck8_delta_over_stronger_null"] == 1.0
    assert slot_0["effect"]["joint_pck8_delta_over_stronger_null"] == 1.0

    # Slot 1's empirical prior already succeeds end-to-end, so Full has no
    # PCK advantage there even though its median error is lower.
    slot_1 = diagnostic["full_vs_stronger_null"]["slot_1"]
    assert slot_1["stronger_null"]["joint_pck8"]["source_modes"] == [
        "empirical-prior"
    ]
    assert slot_1["effect"]["median_relative_improvement_over_stronger_null"] == 1.0
    assert slot_1["effect"]["pck8_delta_over_stronger_null"] == 0.0
    assert slot_1["effect"]["joint_pck8_delta_over_stronger_null"] == 0.0

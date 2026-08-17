from __future__ import annotations

import pytest
import torch
from scripts.tools.build_debiased_heatmap_selection import (
    assert_scene_disjoint_records,
    audit_records,
    candidate_record_from_targets,
    compare_prior_strength,
    deterministic_debiased_selection,
    empirical_prior_strength,
    scene_round_robin_selection,
    selection_manifest,
)

CONSTRAINTS = {
    "max_recent_back_center_fraction": 0.25,
    "max_recent_positive_fraction": 0.60,
    "max_back_positive_fraction": 0.45,
    "max_center_positive_fraction": 0.45,
}


def _record(
    index: int,
    scene: str,
    visible_events: list[tuple[int, int, int, int]],
) -> dict:
    visibility = torch.zeros(2, 4)
    heatmap = torch.zeros(2, 4, 8, 8)
    for slot, view, x, y in visible_events:
        visibility[slot, view] = 1.0
        heatmap[slot, view, y, x] = 1.0
    return candidate_record_from_targets(
        dataset_index=index,
        sample_id=f"train/{scene}/clip_{index:06d}:frame=10",
        scene=scene,
        current_frame=10,
        history_frames=[0, 9],
        gt_visibility=visibility,
        heatmap=heatmap,
        history_rel_poses=torch.tensor([[2.0 + index * 0.01, 0.0, 1.0, 0.0], [0.2, 0.0, 1.0, 0.0]]),
        coordinate_grid_size=4,
        center_radius_pixels=1.0,
        temporal_lag_edges=(1, 2, 4, 8, 16),
        spatial_lag_edges=(0.25, 0.5, 1.0, 2.0, 4.0),
    )


def test_candidate_descriptor_records_slot_view_coordinate_and_lags():
    visibility = torch.zeros(2, 4)
    heatmap = torch.zeros(2, 4, 8, 8)
    visibility[0, 0] = 1.0
    heatmap[0, 0, 7, 0] = 1.0
    visibility[1, 2] = 1.0
    heatmap[1, 2, 4, 4] = 1.0

    record = candidate_record_from_targets(
        dataset_index=7,
        sample_id="train/scene_a/clip_000007:frame=10",
        scene="scene_a",
        current_frame=10,
        history_frames=[0, 9],
        gt_visibility=visibility,
        heatmap=heatmap,
        history_rel_poses=torch.tensor([[3.0, 4.0, 1.0, 0.0], [0.1, 0.0, 1.0, 0.0]]),
        coordinate_grid_size=4,
        center_radius_pixels=1.0,
        temporal_lag_edges=(1, 4, 8),
        spatial_lag_edges=(0.25, 1.0, 5.0),
    )

    assert record["history_frames"] == [0, 9]
    assert record["slots"][0]["temporal_lag"] == 10
    assert record["slots"][0]["temporal_lag_bin"] == "t_gt_8"
    assert record["slots"][0]["spatial_lag"] == pytest.approx(5.0)
    assert record["slots"][0]["spatial_lag_bin"] == "d_le_5"
    assert record["slots"][1]["temporal_lag_bin"] == "t_le_1"
    assert record["slots"][1]["spatial_lag_bin"] == "d_le_0p25"

    old_front = next(event for event in record["events"] if event["slot"] == 0 and event["view_name"] == "front")
    recent_back = next(event for event in record["events"] if event["slot"] == 1 and event["view_name"] == "back")
    assert old_front["coordinate_bin"] == "r3c0"
    assert old_front["recent_back_center"] is False
    assert recent_back["coordinate_bin"] == "r2c2"
    assert recent_back["is_center"] is True
    assert recent_back["recent_back_center"] is True


def test_debiased_selection_is_deterministic_and_limits_recent_back_center():
    candidates = []
    # Dataset-order baseline is intentionally dominated by recent/back/centre.
    for index in range(8):
        scene = "scene_a" if index % 2 == 0 else "scene_b"
        candidates.append(_record(index, scene, [(1, 2, 4, 4)]))

    # Each diverse sample exposes both slots, uses non-centre coordinates, and
    # spans all views.  Four such samples are available in each scene.
    for offset in range(8):
        index = 100 + offset
        scene = "scene_a" if offset % 2 == 0 else "scene_b"
        old_view = offset % 4
        recent_view = (offset + 1) % 4
        candidates.append(
            _record(
                index,
                scene,
                [
                    (0, old_view, 0 if offset % 2 == 0 else 7, offset % 8),
                    (1, recent_view, 7 if offset % 2 == 0 else 0, 7 - offset % 8),
                ],
            )
        )

    baseline = scene_round_robin_selection(candidates, 8)
    baseline_dominance = audit_records(baseline)["dominance"]
    assert baseline_dominance["recent_back_center_fraction"] == 1.0

    selected, diagnostics = deterministic_debiased_selection(
        candidates,
        limit=8,
        seed=42,
        constraints=CONSTRAINTS,
    )
    reversed_selected, reversed_diagnostics = deterministic_debiased_selection(
        list(reversed(candidates)),
        limit=8,
        seed=42,
        constraints=CONSTRAINTS,
    )
    selected_ids = [record["sample_id"] for record in selected]
    assert selected_ids == [record["sample_id"] for record in reversed_selected]
    assert diagnostics["selection_complete"] is True
    assert reversed_diagnostics["selection_complete"] is True
    assert diagnostics["unmet_constraints"] == []
    assert diagnostics["scene_counts"] == {"scene_a": 4, "scene_b": 4}

    dominance = audit_records(selected)["dominance"]
    assert dominance["recent_back_center_fraction"] <= 0.25
    assert dominance["recent_positive_fraction"] <= 0.60
    assert dominance["back_positive_fraction"] <= 0.45
    assert dominance["center_positive_fraction"] <= 0.45
    assert set(diagnostics["stratum_coverage"]["positive_slot"]["covered_categories"]) == {
        "s0",
        "s1",
    }
    assert diagnostics["stratum_coverage"]["positive_coordinate"]["unsupported_expected_categories"]
    assert "exact_count_balance_over_supported" in diagnostics["stratum_coverage"]["slot_view"]


def test_infeasible_collection_reports_shortfall_without_relaxing_dominance():
    candidates = [_record(index, "scene_a", [(1, 2, 4, 4)]) for index in range(6)]
    selected, diagnostics = deterministic_debiased_selection(
        candidates,
        limit=5,
        seed=42,
        constraints=CONSTRAINTS,
    )

    assert selected == []
    assert diagnostics["selection_complete"] is False
    assert "sample_count_shortfall:5" in diagnostics["unmet_constraints"]
    assert diagnostics["dominance_constraints"] == CONSTRAINTS
    assert diagnostics["dominance"]["recent_back_center_fraction"] == 0.0


def test_selection_manifest_is_order_sensitive_and_scenes_must_be_disjoint():
    first = _record(0, "scene_a", [(0, 0, 0, 0)])
    second = _record(1, "scene_b", [(1, 1, 7, 7)])
    manifest = selection_manifest([first, second])
    reversed_manifest = selection_manifest([second, first])
    assert manifest["sample_identity_sha256"] != reversed_manifest["sample_identity_sha256"]
    assert manifest["descriptor_sha256"] != reversed_manifest["descriptor_sha256"]

    assert_scene_disjoint_records([first], [second])
    with pytest.raises(RuntimeError, match="scene_a"):
        assert_scene_disjoint_records([first], [first, second])


class _TargetSource:
    def __init__(self, root, scene: str, targets: list[dict]):
        self.root = root
        self.clips = [root / scene / "clip_000000"]
        self.sample_index = [(0, 10 + index) for index in range(len(targets))]
        self.targets = targets


def _prior_target(slot: int, view: int, x: int, y: int) -> dict:
    visibility = torch.zeros(2, 4)
    heatmap = torch.zeros(2, 4, 16, 16)
    visibility[slot, view] = 1.0
    heatmap[slot, view, y, x] = 1.0
    return {"gt_visibility": visibility, "heatmap": heatmap}


def test_empirical_prior_strength_reuses_selected_target_only_samples(tmp_path):
    train_source = _TargetSource(
        tmp_path,
        "train_scene",
        [_prior_target(0, 0, 2, 2), _prior_target(1, 1, 12, 12)],
    )
    val_source = _TargetSource(
        tmp_path,
        "val_scene",
        [_prior_target(0, 0, 2, 2), _prior_target(1, 1, 12, 12)],
    )
    train_records = [
        _record(0, "train_scene", [(0, 0, 0, 0)]),
        _record(1, "train_scene", [(1, 1, 7, 7)]),
    ]
    val_records = [
        _record(0, "val_scene", [(0, 0, 0, 0)]),
        _record(1, "val_scene", [(1, 1, 7, 7)]),
    ]

    result = empirical_prior_strength(
        train_source,
        val_source,
        train_records,
        val_records,
        target_loader=lambda source, index: source.targets[index],
    )
    comparison = compare_prior_strength(result, result)

    assert result["available"] is True
    assert result["train_samples"] == 2
    assert result["val_samples"] == 2
    assert set(result["metrics"]) >= {"median_pixel_error", "pck8", "joint_pck8"}
    assert comparison["available"] is True
    assert comparison["delta_after_minus_before"]["pck8"] == 0.0

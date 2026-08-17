from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest
import scripts.tools.build_multi_history_selection as multi_history_builder
import torch
from scripts.tools.build_multi_history_selection import (
    SelectionConstraints,
    assemble_selection_record,
    assert_scene_disjoint,
    bearing_to_view,
    describe_anchor_candidates,
    deterministic_balanced_selection,
    distance_to_view_seam_degrees,
    pose_free_model_input_contract,
    rank_anchor_sets,
    selection_manifest,
    slot_permutation,
    source_inventory,
    target_separation_audit,
)

from src.data.explicit_multi_history import (
    MULTI_HISTORY_SCHEMA,
    ExplicitMultiHistoryDataset,
    canonical_sha256,
    load_multi_history_records,
    verify_runtime_targets,
    verify_selection_record,
)
from src.data.sliding_window_dataset import VLNSlidingWindowDataset


def _anchor(
    frame: int,
    *,
    lag: int,
    distance: float,
    bearing: float,
    view: str,
    lag_bin: str,
    distance_bin: str,
    target_x: int,
) -> dict:
    view_index = ("front", "right", "back", "left").index(view)
    return {
        "history_frame": frame,
        "temporal_lag": lag,
        "temporal_lag_bin": lag_bin,
        "spatial_distance_m": distance,
        "spatial_distance_bin": distance_bin,
        "bearing_degrees": bearing,
        "bearing_view": view,
        "view_seam_distance_degrees": distance_to_view_seam_degrees(bearing),
        "relative_yaw_degrees": 0.0,
        "relative_pose_label": [distance, 0.0, 1.0, 0.0],
        "target_views": [
            {
                "view": name,
                "view_index": index,
                "visible": index == view_index,
                "x": target_x if index == view_index else None,
                "y": 4 if index == view_index else None,
                "peak_value": 1.0 if index == view_index else None,
            }
            for index, name in enumerate(("front", "right", "back", "left"))
        ],
        "primary_target": {
            "view": view,
            "view_index": view_index,
            "visible": True,
            "x": target_x,
            "y": 4,
            "peak_value": 1.0,
            "panorama_x": view_index * 8 + target_x,
        },
        "any_visible": True,
        "exact_pose_label_sha256": f"pose-{frame}",
    }


def _diverse_anchors() -> list[dict]:
    return [
        _anchor(
            4,
            lag=36,
            distance=4.0,
            bearing=0.0,
            view="front",
            lag_bin="t_le_64",
            distance_bin="d_le_4",
            target_x=2,
        ),
        _anchor(
            12,
            lag=28,
            distance=3.0,
            bearing=-90.0,
            view="right",
            lag_bin="t_le_32",
            distance_bin="d_le_4",
            target_x=3,
        ),
        _anchor(
            20,
            lag=20,
            distance=1.5,
            bearing=180.0,
            view="back",
            lag_bin="t_le_32",
            distance_bin="d_le_2",
            target_x=4,
        ),
        _anchor(
            24,
            lag=16,
            distance=0.8,
            bearing=90.0,
            view="left",
            lag_bin="t_le_16",
            distance_bin="d_le_1",
            target_x=5,
        ),
    ]


def test_view_mapping_and_seam_margin_are_explicit():
    assert bearing_to_view(0.0) == "front"
    assert bearing_to_view(-90.0) == "right"
    assert bearing_to_view(180.0) == "back"
    assert bearing_to_view(90.0) == "left"
    assert distance_to_view_seam_degrees(45.0) == 0.0
    assert distance_to_view_seam_degrees(35.0) == pytest.approx(10.0)
    assert distance_to_view_seam_degrees(0.0) == pytest.approx(45.0)


def test_candidate_filter_rejects_view_seam_before_selection(monkeypatch):
    relative = np.asarray(
        [
            [1.0, 1.0, 1.0, 0.0],  # exactly +45 degrees: ambiguous seam
            [1.0, 0.0, 1.0, 0.0],  # front sector centre
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        multi_history_builder,
        "compute_history_rel_poses",
        lambda history, current: relative,
    )
    poses = [np.eye(4, dtype=np.float32) for _ in range(3)]
    constraints = SelectionConstraints(
        num_history=2,
        min_temporal_lag=1,
        min_spatial_distance=0.0,
        max_spatial_distance=0.0,
        view_seam_margin_degrees=10.0,
        min_distinct_views=1,
        min_distinct_lag_bins=1,
        min_distinct_distance_bins=1,
        min_visible_anchors=1,
        min_visible_distinct_views=1,
    )
    candidates, rejected = describe_anchor_candidates(
        poses,
        2,
        constraints=constraints,
        temporal_lag_edges=(1.0, 2.0),
        spatial_distance_edges=(1.0, 2.0),
    )
    assert [candidate["history_frame"] for candidate in candidates] == [1]
    assert candidates[0]["view_seam_distance_degrees"] == pytest.approx(45.0)
    assert rejected["near_view_seam"] == 1


def test_beam_selection_is_deterministic_and_meets_all_pose_gates():
    anchors = _diverse_anchors()
    # Add tempting but seam-adjacent/redundant candidates; the pure selector
    # still has to choose a fully diverse four-anchor set.
    anchors.extend(
        [
            _anchor(
                26,
                lag=14,
                distance=0.7,
                bearing=5.0,
                view="front",
                lag_bin="t_le_16",
                distance_bin="d_le_1",
                target_x=6,
            ),
            _anchor(
                28,
                lag=12,
                distance=0.6,
                bearing=10.0,
                view="front",
                lag_bin="t_le_16",
                distance_bin="d_le_1",
                target_x=7,
            ),
        ]
    )
    constraints = SelectionConstraints(
        num_history=4,
        min_temporal_lag=12,
        min_spatial_distance=0.5,
        max_spatial_distance=15.0,
        min_bearing_separation_degrees=30.0,
        view_seam_margin_degrees=10.0,
        min_distinct_views=3,
        min_distinct_lag_bins=2,
        min_distinct_distance_bins=2,
        min_visible_anchors=4,
        min_visible_distinct_views=3,
        min_target_separation_pixels=4.0,
        beam_width=64,
        max_anchor_set_trials=8,
    )
    selected, diagnostics = rank_anchor_sets(
        anchors,
        constraints=constraints,
        seed=42,
        sample_key="scene/clip:current=40",
    )
    reversed_selected, reversed_diagnostics = rank_anchor_sets(
        list(reversed(anchors)),
        constraints=constraints,
        seed=42,
        sample_key="scene/clip:current=40",
    )

    assert diagnostics["failure_reasons"] == []
    assert reversed_diagnostics["failure_reasons"] == []
    assert [[item["history_frame"] for item in group] for group in selected] == [
        [item["history_frame"] for item in group] for group in reversed_selected
    ]
    best = selected[0]
    assert len(best) == 4
    assert len({item["bearing_view"] for item in best}) >= 3
    assert len({item["temporal_lag_bin"] for item in best}) >= 2
    assert len({item["spatial_distance_bin"] for item in best}) >= 2


def test_beam_selection_reports_structural_support_failure():
    anchors = [_diverse_anchors()[0], copy.deepcopy(_diverse_anchors()[0])]
    anchors[1]["history_frame"] = 5
    selected, diagnostics = rank_anchor_sets(
        anchors,
        constraints=SelectionConstraints(),
        seed=42,
        sample_key="missing-support",
    )
    assert selected == []
    assert "insufficient_eligible_anchors" in diagnostics["failure_reasons"]
    assert "insufficient_view_support" in diagnostics["failure_reasons"]


def test_record_keeps_pose_in_labels_only_and_hashes_exact_frames():
    anchors = _diverse_anchors()
    record = assemble_selection_record(
        relative_clip="scene_a/clip_000001",
        scene="scene_a",
        current_frame=40,
        canonical_anchors=anchors,
        heatmap_shape=[4, 4, 8, 8],
        slot_order="randomized",
        seed=42,
    )
    verify_selection_record(record, expected_k=4)

    assert sorted(record["history_frames"]) == [4, 12, 20, 24]
    assert record["sample_id"].endswith(
        "history=" + ",".join(str(frame) for frame in record["history_frames"])
    )
    assert record["model_inputs"] == {
        "current_rgb": "current_rgb_panorama",
        "history_rgb": "ordered_history_rgb_observations",
    }
    assert [item["frame_index"] for item in record["loader_alignment"]["history"]] == record[
        "history_frames"
    ]
    assert "relative_pose" not in json.dumps(record["model_inputs"])
    assert "frame" not in json.dumps(record["model_inputs"])
    assert "slot" not in json.dumps(record["model_inputs"])
    assert "lag" not in json.dumps(record["model_inputs"])
    assert "relative_pose_label" in json.dumps(record["label_metadata"])
    assert record["label_metadata"]["pose_usage"] == "label_generation_and_audit_only"

    changed = copy.deepcopy(record)
    changed["history_frames"][0], changed["history_frames"][1] = (
        changed["history_frames"][1],
        changed["history_frames"][0],
    )
    with pytest.raises(ValueError, match=r"slot_permutation|sample_id|record_sha256"):
        verify_selection_record(changed, expected_k=4)


def test_pose_free_contract_allows_only_rgb_and_forbids_alignment_metadata():
    contract = pose_free_model_input_contract()
    assert contract["allowed"] == [
        "current_rgb_panorama",
        "ordered_history_rgb_observations",
    ]
    assert {
        "history_slot_id",
        "history_frame_index",
        "temporal_lag",
        "exact_relative_pose",
        "absolute_pose",
        "bearing",
        "spatial_distance",
        "target_view",
        "target_pixel",
    }.issubset(contract["forbidden"])
    assert "records[*].history_frames" in contract["record_alignment_fields"]
    assert "records[*].label_metadata.anchors[*].temporal_lag" in contract[
        "record_alignment_fields"
    ]
    assert "must never be forwarded to the model" in contract["alignment_metadata_policy"]


def test_target_separation_wraps_at_panorama_boundary():
    anchors = _diverse_anchors()[:2]
    anchors[0]["primary_target"]["panorama_x"] = 1
    anchors[1]["primary_target"]["panorama_x"] = 31
    anchors[0]["primary_target"]["y"] = 4
    anchors[1]["primary_target"]["y"] = 4
    audit = target_separation_audit(anchors, heatmap_shape=[2, 4, 8, 8])
    assert audit["minimum_target_separation_pixels"] == pytest.approx(2.0)


def test_dataset_level_view_cap_is_hard_and_deterministic():
    pool = []
    for index in range(5):
        pool.append(
            assemble_selection_record(
                relative_clip=f"scene_{index % 3}/clip_{index:06d}",
                scene=f"scene_{index % 3}",
                current_frame=40,
                canonical_anchors=_diverse_anchors(),
                heatmap_shape=[4, 4, 8, 8],
                slot_order="canonical",
                seed=42,
            )
        )
    selected, diagnostics = deterministic_balanced_selection(
        pool,
        requested_samples=3,
        num_history=4,
        min_target_view_fraction=0.15,
        max_target_view_fraction=0.35,
        seed=42,
    )
    reversed_selected, reversed_diagnostics = deterministic_balanced_selection(
        list(reversed(pool)),
        requested_samples=3,
        num_history=4,
        min_target_view_fraction=0.15,
        max_target_view_fraction=0.35,
        seed=42,
    )
    assert diagnostics["selection_complete"] is True
    assert reversed_diagnostics["selection_complete"] is True
    assert [record["sample_id"] for record in selected] == [
        record["sample_id"] for record in reversed_selected
    ]
    assert max(diagnostics["target_view_fractions"].values()) <= 0.35

    skewed = copy.deepcopy(pool[0])
    for anchor in skewed["label_metadata"]["anchors"][:3]:
        anchor["primary_target"]["view"] = "back"
    failed, failed_diagnostics = deterministic_balanced_selection(
        [skewed],
        requested_samples=1,
        num_history=4,
        min_target_view_fraction=0.15,
        max_target_view_fraction=0.35,
        seed=42,
    )
    assert failed == []
    assert failed_diagnostics["selection_complete"] is False
    assert failed_diagnostics["unmet_constraints"] == ["sample_count_shortfall:1"]


def test_manifest_loader_verifies_scene_split_and_top_level_hash(tmp_path):
    train = assemble_selection_record(
        relative_clip="scene_train/clip_000001",
        scene="scene_train",
        current_frame=40,
        canonical_anchors=_diverse_anchors(),
        heatmap_shape=[4, 4, 8, 8],
        slot_order="canonical",
        seed=42,
    )
    val = assemble_selection_record(
        relative_clip="scene_val/clip_000002",
        scene="scene_val",
        current_frame=40,
        canonical_anchors=_diverse_anchors(),
        heatmap_shape=[4, 4, 8, 8],
        slot_order="canonical",
        seed=42,
    )
    assert_scene_disjoint([train], [val])
    with pytest.raises(RuntimeError, match="scene_train"):
        assert_scene_disjoint([train], [train])

    payload = {
        "schema_version": MULTI_HISTORY_SCHEMA,
        "selection_parameters": {"num_history": 4},
        "source_inventory_contract": {
            "max_clip_id": 2000,
            "clips": 2,
            "records": [
                {
                    "relative_clip": "scene_train/clip_000001",
                    "scene_id": "scene_train",
                    "episode_id": 1,
                    "num_frames": 50,
                    "seed": 101,
                },
                {
                    "relative_clip": "scene_val/clip_000002",
                    "scene_id": "scene_val",
                    "episode_id": 2,
                    "num_frames": 50,
                    "seed": 102,
                },
            ],
        },
        "splits": {
            "train": {"selection_manifest": selection_manifest([train]), "records": [train]},
            "val": {"selection_manifest": selection_manifest([val]), "records": [val]},
        },
    }
    inventory_lines = sorted(
        (
            f"{record['relative_clip']}\t{record['scene_id']}\t{record['episode_id']}\t"
            f"{record['num_frames']}\t{record['seed']}"
        )
        for record in payload["source_inventory_contract"]["records"]
    )
    payload["source_inventory_contract"]["inventory_sha256"] = hashlib.sha256(
        ("\n".join(inventory_lines) + "\n").encode()
    ).hexdigest()
    payload["manifest_sha256"] = canonical_sha256(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded, manifest = load_multi_history_records(path, "train")
    assert [record["sample_id"] for record in loaded] == [train["sample_id"]]
    assert manifest["manifest_sha256"] == payload["manifest_sha256"]
    with pytest.raises(ValueError, match="requested snapshot"):
        load_multi_history_records(
            path,
            "train",
            expected_source_inventory_sha256="0" * 64,
        )

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["selection_parameters"]["num_history"] = 3
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_sha256 mismatch"):
        load_multi_history_records(path, "train")


def test_source_inventory_hash_is_sorted_and_seed_sensitive(tmp_path):
    class FakeDataset:
        def __init__(self, entries):
            self.root = tmp_path
            self.clips = [tmp_path / relative for relative, _meta in entries]
            self._metas = [_meta for _relative, _meta in entries]

        def _load_meta(self, index):
            return self._metas[index]

    entries = [
        (
            "scene_b/clip_000002",
            {"scene_id": "scene_b", "episode_id": 2, "num_frames": 50, "seed": 12},
        ),
        (
            "scene_a/clip_000001",
            {"scene_id": "scene_a", "episode_id": 1, "num_frames": 40, "seed": 11},
        ),
    ]
    forward = source_inventory(FakeDataset(entries))
    reversed_inventory = source_inventory(FakeDataset(list(reversed(entries))))
    assert forward["inventory_sha256"] == reversed_inventory["inventory_sha256"]
    assert [record["relative_clip"] for record in forward["records"]] == [
        "scene_a/clip_000001",
        "scene_b/clip_000002",
    ]
    expected_payload = (
        "scene_a/clip_000001\tscene_a\t1\t40\t11\n"
        "scene_b/clip_000002\tscene_b\t2\t50\t12\n"
    )
    assert forward["inventory_sha256"] == hashlib.sha256(expected_payload.encode()).hexdigest()

    changed = copy.deepcopy(entries)
    changed[0][1]["seed"] = 999
    assert source_inventory(FakeDataset(changed))["inventory_sha256"] != forward[
        "inventory_sha256"
    ]


def test_runtime_target_verification_detects_visibility_and_peak_drift():
    record = assemble_selection_record(
        relative_clip="scene_a/clip_000001",
        scene="scene_a",
        current_frame=40,
        canonical_anchors=_diverse_anchors(),
        heatmap_shape=[4, 4, 8, 8],
        slot_order="canonical",
        seed=42,
    )
    visibility = torch.zeros(4, 4)
    heatmaps = torch.zeros(4, 4, 8, 8)
    for slot, anchor in enumerate(record["label_metadata"]["anchors"]):
        target = anchor["primary_target"]
        view = int(target["view_index"])
        visibility[slot, view] = 1.0
        heatmaps[slot, view, int(target["y"]), int(target["x"])] = 1.0
    sample = {"gt_visibility": visibility, "heatmap": heatmaps}
    verify_runtime_targets(sample, record, record["history_frames"])

    visibility_drift = visibility.clone()
    first_view = int(record["label_metadata"]["anchors"][0]["primary_target"]["view_index"])
    visibility_drift[0, first_view] = 0.0
    with pytest.raises(RuntimeError, match="Runtime visibility drift"):
        verify_runtime_targets(
            {"gt_visibility": visibility_drift, "heatmap": heatmaps},
            record,
            record["history_frames"],
        )

    peak_drift = heatmaps.clone()
    peak_drift[0, first_view].zero_()
    peak_drift[0, first_view, 0, 0] = 1.0
    with pytest.raises(RuntimeError, match="Runtime heatmap peak drift"):
        verify_runtime_targets(
            {"gt_visibility": visibility, "heatmap": peak_drift},
            record,
            record["history_frames"],
        )


def test_explicit_dataset_override_reuses_base_getitem_and_aligns_slots(monkeypatch):
    dataset = ExplicitMultiHistoryDataset.__new__(ExplicitMultiHistoryDataset)
    dataset.sample_index = [(0, 40)]
    dataset._explicit_history_frames = [(24, 4, 20, 12)]
    dataset._explicit_identities = ["scene/clip:current=40:history=24,4,20,12"]
    dataset._active_explicit_history = None
    dataset._active_explicit_current = None
    dataset._explicit_verify_runtime_labels = False

    def fake_base_getitem(self, index):
        _clip, current = self.sample_index[index]
        exact = self._sample_history_indices(0, current, 4)
        return {
            "base_history_indices": exact.copy(),
            "history_rel_poses": "must-not-reach-model",
        }

    monkeypatch.setattr(VLNSlidingWindowDataset, "__getitem__", fake_base_getitem)
    sample = ExplicitMultiHistoryDataset.__getitem__(dataset, 0)

    assert sample["base_history_indices"].tolist() == [24, 4, 20, 12]
    assert "history_rel_poses" not in sample
    assert not any(key.startswith("explicit_") for key in sample)
    assert dataset._active_explicit_history is None


def test_epoch_slot_shuffle_is_deterministic_and_preserves_anchor_set():
    def make_dataset():
        dataset = ExplicitMultiHistoryDataset.__new__(ExplicitMultiHistoryDataset)
        dataset._explicit_identities = ["scene/clip:current=40:history=4,12,20,24"]
        dataset._explicit_canonical_frames = [(4, 12, 20, 24)]
        dataset._explicit_initial_frames = [(4, 12, 20, 24)]
        dataset._explicit_history_frames = [(4, 12, 20, 24)]
        dataset._explicit_slot_seed = 42
        dataset._explicit_reshuffle_slots_each_epoch = True
        return dataset

    first = make_dataset()
    second = make_dataset()
    first.set_epoch(7)
    second.set_epoch(7)
    assert first._explicit_history_frames == second._explicit_history_frames
    assert sorted(first._explicit_history_frames[0]) == [4, 12, 20, 24]
    assert slot_permutation(4, order="canonical", seed=42, sample_key="x") == [0, 1, 2, 3]

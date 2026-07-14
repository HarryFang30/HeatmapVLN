from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.tools.build_multi_history_selection import (
    assemble_selection_record,
    deterministic_balanced_selection,
    selection_manifest,
)
from scripts.tools.derive_multi_history_subset import derive_multi_history_subset

from src.data.explicit_multi_history import (
    MULTI_HISTORY_SCHEMA,
    canonical_sha256,
    load_multi_history_records,
)

VIEWS = ("front", "right", "back", "left")


def _anchor(frame: int, view_index: int, *, offset: int = 0) -> dict:
    view = VIEWS[view_index]
    bearing = (0.0, -90.0, 180.0, 90.0)[view_index]
    target_x = 8 + view_index * 20 + offset
    return {
        "history_frame": frame,
        "temporal_lag": 80 - frame,
        "temporal_lag_bin": "t_le_64" if frame < 20 else "t_le_32",
        "spatial_distance_m": 4.0 - 0.7 * view_index,
        "spatial_distance_bin": "d_le_4" if view_index < 2 else "d_le_2",
        "bearing_degrees": bearing,
        "bearing_view": view,
        "view_seam_distance_degrees": 45.0,
        "relative_yaw_degrees": 0.0,
        "relative_pose_label": [4.0 - view_index, 0.0, 1.0, 0.0],
        "target_views": [
            {
                "view": candidate_view,
                "view_index": candidate_index,
                "visible": candidate_index == view_index,
                "x": target_x if candidate_index == view_index else None,
                "y": 12 if candidate_index == view_index else None,
                "peak_value": 1.0 if candidate_index == view_index else None,
            }
            for candidate_index, candidate_view in enumerate(VIEWS)
        ],
        "primary_target": {
            "view": view,
            "view_index": view_index,
            "visible": True,
            "x": target_x,
            "y": 12,
            "peak_value": 1.0,
            "panorama_x": view_index * 128 + target_x,
        },
        "any_visible": True,
        "exact_pose_label_sha256": hashlib.sha256(f"pose-{frame}".encode()).hexdigest(),
    }


def _record(relative_clip: str, scene: str, current: int, seed: int) -> dict:
    anchors = [_anchor(frame, view_index, offset=seed % 5) for view_index, frame in enumerate((4, 20, 36, 52))]
    record = assemble_selection_record(
        relative_clip=relative_clip,
        scene=scene,
        current_frame=current,
        canonical_anchors=anchors,
        heatmap_shape=[4, 4, 128, 128],
        slot_order="randomized",
        seed=seed,
    )
    record["selection_audit"] = {
        "target_separation": {
            "visible_anchor_count": 4,
            "visible_distinct_views": 4,
            "pairwise_target_distances_pixels": [20.0, 40.0, 60.0],
            "minimum_target_separation_pixels": 20.0,
            "panorama_width_pixels": 512,
        },
        "pose_selection": {"source": "unit-test-strict-parent"},
    }
    record["record_sha256"] = canonical_sha256({key: value for key, value in record.items() if key != "record_sha256"})
    return record


def _inventory(records: list[dict]) -> dict:
    entries = [
        {
            "relative_clip": record["relative_clip"],
            "scene_id": record["scene"],
            "episode_id": index,
            "num_frames": 100,
            "seed": 1000 + index,
        }
        for index, record in enumerate(records)
    ]
    entries.sort(key=lambda item: item["relative_clip"])
    rows = sorted(
        f"{entry['relative_clip']}\t{entry['scene_id']}\t{entry['episode_id']}\t{entry['num_frames']}\t{entry['seed']}"
        for entry in entries
    )
    return {
        "max_clip_id": 2000,
        "clips": len(entries),
        "inventory_sha256": hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest(),
        "records": entries,
    }


def _write_parent(path: Path) -> tuple[dict, list[dict], list[dict]]:
    # Deliberately non-lexical train order: equal-size derivation must retain it.
    train = [
        _record("scene_train/clip_000003", "scene_train", 80, 3),
        _record("scene_train/clip_000001", "scene_train", 80, 1),
        _record("scene_train/clip_000004", "scene_train", 80, 4),
        _record("scene_train/clip_000002", "scene_train", 80, 2),
    ]
    val = [_record(f"scene_val/clip_{index:06d}", "scene_val", 80, 10 + index) for index in range(1, 5)]
    all_records = [*train, *val]
    parameters = {
        "num_history": 4,
        "min_target_view_fraction": 0.15,
        "max_target_view_fraction": 0.35,
        "seed": 42,
        "min_temporal_lag": 16,
        "min_target_separation_pixels": 12.0,
    }

    def split_payload(records: list[dict], *, complete: bool) -> dict:
        return {
            "selection_manifest": selection_manifest(records),
            "selection_audit": {"parent": True, "samples": len(records)},
            "candidate_catalog": {
                "pose_valid_proposals": 777,
                "depth_valid_pool_samples": 88,
            },
            "source_inventory": {"inventory_sha256": "parent-split-inventory"},
            "balanced_selector": {
                "requested_samples": 64,
                "selected_samples": len(records),
                "selection_complete": complete,
            },
            "failure_audit": {
                "failures": 1,
                "reason_counts": {"sample_count_shortfall": 1},
                "records": [{"reason": "sample_count_shortfall"}],
            },
            "selection_complete": complete,
            "records": records,
        }

    parent = {
        "schema_version": MULTI_HISTORY_SCHEMA,
        "selection_parameters": parameters,
        "source_inventory_contract": _inventory(all_records),
        "model_input_contract": {"allowed": ["current_rgb_panorama"]},
        "scene_disjoint": {"verified": True, "overlap": []},
        # The motivating case: original requested count was infeasible even
        # though every retained record is individually strict and hash-valid.
        "ready": False,
        "splits": {
            "train": split_payload(train, complete=True),
            "val": split_payload(val, complete=False),
        },
    }
    parent["manifest_sha256"] = canonical_sha256(parent)
    path.write_text(json.dumps(parent, indent=2), encoding="utf-8")
    return parent, train, val


def test_derivation_preserves_equal_train_and_strictly_balances_smaller_val(tmp_path):
    parent_path = tmp_path / "parent.json"
    parent, parent_train, parent_val = _write_parent(parent_path)
    output_dir = tmp_path / "derived"

    derived = derive_multi_history_subset(
        parent_path,
        output_dir,
        train_samples=4,
        val_samples=2,
    )

    assert derived["ready"] is True
    assert derived["derived_from"]["ready"] is False
    assert "no record or dataset-level constraint was relaxed" in derived["readiness_reason"]
    assert derived["selection_parameters"] == parent["selection_parameters"]
    assert derived["derivation_parameters"]["constraint_overrides"] == {}
    assert derived["derivation_parameters"]["constraints_relaxed"] is False
    assert derived["scene_disjoint"] == {"verified": True, "overlap": []}

    derived_train = derived["splits"]["train"]["records"]
    assert [record["sample_id"] for record in derived_train] == [record["sample_id"] for record in parent_train]
    assert [record["record_sha256"] for record in derived_train] == [record["record_sha256"] for record in parent_train]
    assert (
        derived["splits"]["train"]["balanced_selector"]["selection_strategy"]
        == "preserve_parent_exact_order_and_record_hashes"
    )

    expected_val, expected_balance = deterministic_balanced_selection(
        parent_val,
        requested_samples=2,
        num_history=4,
        min_target_view_fraction=0.15,
        max_target_view_fraction=0.35,
        seed=42,
    )
    assert expected_balance["selection_complete"] is True
    assert [record["sample_id"] for record in derived["splits"]["val"]["records"]] == [
        record["sample_id"] for record in expected_val
    ]

    for split in ("train", "val"):
        loaded, loaded_manifest = load_multi_history_records(
            output_dir / "multi_history_selection_manifest.json",
            split,
        )
        assert loaded == derived["splits"][split]["records"]
        assert loaded_manifest["manifest_sha256"] == derived["manifest_sha256"]
        jsonl_path = output_dir / f"{split}_selection.jsonl"
        artifact = derived["artifacts"][f"{split}_selection"]
        assert hashlib.sha256(jsonl_path.read_bytes()).hexdigest() == artifact["sha256"]
        assert [json.loads(line) for line in jsonl_path.read_text().splitlines()] == loaded

    val_payload = derived["splits"]["val"]
    assert val_payload["failure_audit"]["failures"] == 0
    assert val_payload["parent_failure_audit"] == parent["splits"]["val"]["failure_audit"]
    assert "not current derivation shortfalls" in val_payload["parent_failure_audit_scope"]
    assert val_payload["candidate_catalog"] == parent["splits"]["val"]["candidate_catalog"]
    assert derived["derived_from"]["file_sha256"] == hashlib.sha256(parent_path.read_bytes()).hexdigest()
    assert derived["derived_from"]["manifest_sha256"] == parent["manifest_sha256"]

    manifest_before = (output_dir / "multi_history_selection_manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        derive_multi_history_subset(
            parent_path,
            output_dir,
            train_samples=4,
            val_samples=2,
        )
    assert (output_dir / "multi_history_selection_manifest.json").read_bytes() == manifest_before


def test_derivation_rejects_expansion_without_publishing_partial_output(tmp_path):
    parent_path = tmp_path / "parent.json"
    _write_parent(parent_path)
    output_dir = tmp_path / "must_not_exist"

    with pytest.raises(ValueError, match="Cannot derive 5 train samples from only 4"):
        derive_multi_history_subset(
            parent_path,
            output_dir,
            train_samples=5,
            val_samples=2,
        )
    assert not output_dir.exists()


def test_parent_manifest_must_pass_strict_hash_verification(tmp_path):
    parent_path = tmp_path / "parent.json"
    parent, _train, _val = _write_parent(parent_path)
    parent["selection_parameters"]["num_history"] = 3
    parent_path.write_text(json.dumps(parent), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_sha256 mismatch"):
        derive_multi_history_subset(
            parent_path,
            tmp_path / "must_not_exist",
            train_samples=4,
            val_samples=2,
        )

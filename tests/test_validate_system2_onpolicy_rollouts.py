import json
from pathlib import Path

import pytest
from PIL import Image
from scripts.training.validate_system2_onpolicy_rollouts import (
    SCHEMA,
    _expected_target,
    _policy_bucket,
    validate_rollouts,
)


def _write_collection(root: Path, *, scenes: int = 10, prefix: str = "") -> None:
    root.mkdir()
    rows = []
    for scene_index in range(scenes):
        scene_id = f"scene_{scene_index:02d}"
        for call_index, (target, terminal, distance) in enumerate(
            ((1, False, 2.5), (0, False, 4.0), (0, True, 4.0))
        ):
            key = f"{prefix}{scene_id}_ep{scene_index:03d}_call{call_index:02d}"
            view_paths = {}
            for view in ("front", "right", "back", "left"):
                relative = Path("system2_stop_multimodal_examples") / key / f"{view}.jpg"
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color=(scene_index, call_index, 1)).save(
                    destination,
                    format="JPEG",
                )
                view_paths[view] = str(relative)
            rows.append(
                {
                    "schema": SCHEMA,
                    "key": key,
                    "dataset_split": "train",
                    "scene_id": scene_id,
                    "episode_id": scene_index,
                    "system2_call_index": call_index,
                    "protocol_seed": 140,
                    "instruction": "walk to the goal",
                    "distance_to_goal_m": distance,
                    "stop_target": target,
                    "original_terminal": terminal,
                    "current_views": view_paths,
                    "history_views": [],
                }
            )
    labels = root / "system2_stop_multimodal_examples.jsonl"
    labels.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (root / "eval_manifest.json").write_text(
        json.dumps(
            {
                "data_path": "/data/train/train.json.gz",
                "rpc_require_deterministic_sampling": True,
                "system2_stop_feature_collection": True,
                "system2_stop_multimodal_example_collection": True,
                "system2_stop_collect_force_continue_negatives": True,
                "system2_stop_collect_oracle_recovery_after_negative": True,
                "system2_stop_policy_mode": "original_system2",
                "system2_stop_head_checkpoint": "",
                "system2_stop_decision_adapter_checkpoint": "",
                "system2_temporal_stop_verifier_checkpoint": "",
                "system2_stop_positive_radius_m": 3.0,
                "system2_stop_negative_radius_m": 3.01,
            }
        ),
        encoding="utf-8",
    )
    (root / "result.json").write_text(
        json.dumps({"total_episodes": scenes}),
        encoding="utf-8",
    )


def _write_cohort(path: Path, *, scenes: int) -> None:
    path.write_text(
        json.dumps(
            {
                "episodes": [
                    {"scene_id": f"scene_{index:02d}", "episode_id": index}
                    for index in range(scenes)
                ]
            }
        ),
        encoding="utf-8",
    )


def test_target_boundaries_and_policy_buckets():
    assert _expected_target(3.0, 3.0, 3.01) == 1
    assert _expected_target(3.005, 3.0, 3.01) is None
    assert _expected_target(3.01, 3.0, 3.01) == 0
    assert _policy_bucket(1, False) == "add_positive"
    assert _policy_bucket(0, False) == "regular_negative"
    assert _policy_bucket(0, True) == "false_stop_negative"


def test_validates_new_images_then_trusts_immutable_base_report(tmp_path):
    root = tmp_path / "rollout"
    _write_collection(root)
    cohort = tmp_path / "cohort.json"
    _write_cohort(cohort, scenes=10)

    report = validate_rollouts(
        base_report=None,
        new_roots=[root],
        new_cohorts=[cohort],
        split_seed=7,
        holdout_fraction=0.2,
        decode_workers=2,
    )

    assert report["status"] == "passed"
    assert report["rows"] == 30
    assert report["decoded_images"] == 120
    assert report["roots"][0]["cohort"]["episodes"] == 10
    assert report["train_policy_counts"] == {
        "add_positive": 8,
        "false_stop_negative": 8,
        "regular_negative": 8,
    }
    assert report["validation_policy_counts"] == {
        "add_positive": 2,
        "false_stop_negative": 2,
        "regular_negative": 2,
    }

    base_report = tmp_path / "validated.json"
    base_report.write_text(json.dumps(report), encoding="utf-8")
    merged = validate_rollouts(
        base_report=base_report,
        new_roots=[],
        split_seed=7,
        holdout_fraction=0.2,
        decode_workers=2,
    )
    assert merged["decoded_images"] == 0
    assert merged["base_images_trusted_from_prior_report"] == 120


def test_rejects_episode_overlap_between_new_roots(tmp_path):
    root_a = tmp_path / "rollout_a"
    root_b = tmp_path / "rollout_b"
    _write_collection(root_a, scenes=2, prefix="a_")
    _write_collection(root_b, scenes=2, prefix="b_")

    with pytest.raises(RuntimeError, match="new rollout episode overlaps"):
        validate_rollouts(
            base_report=None,
            new_roots=[root_a, root_b],
            split_seed=7,
            holdout_fraction=0.5,
            decode_workers=2,
        )


def test_rejects_rollout_cohort_episode_mismatch(tmp_path):
    root = tmp_path / "rollout"
    _write_collection(root, scenes=2)
    cohort = tmp_path / "cohort.json"
    _write_cohort(cohort, scenes=1)

    with pytest.raises(RuntimeError, match="Rollout/cohort episode mismatch"):
        validate_rollouts(
            base_report=None,
            new_roots=[root],
            new_cohorts=[cohort],
            split_seed=7,
            holdout_fraction=0.5,
            decode_workers=2,
        )

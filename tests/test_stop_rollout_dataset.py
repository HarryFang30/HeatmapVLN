import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.stop_rollout_dataset import (
    STOP_MULTIMODAL_EXAMPLE_SCHEMA,
    MixedSystem2SFTDataset,
    System2StopMultimodalDataset,
)


def _write_views(root: Path, key: str, value: int) -> dict[str, str]:
    directory = root / "system2_stop_multimodal_examples" / key
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for view in ("front", "right", "back", "left"):
        path = directory / f"{view}.jpg"
        Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(path)
        result[view] = str(path.relative_to(root))
    return result


def _write_root(root: Path, *, split: str = "train") -> None:
    rows = []
    for scene_index, scene in enumerate(("scene_a", "scene_b")):
        history = _write_views(root, f"{scene}_history", 10 + scene_index)
        for target in (0, 1):
            key = f"{scene}_{target}"
            rows.append(
                {
                    "schema": STOP_MULTIMODAL_EXAMPLE_SCHEMA,
                    "key": key,
                    "dataset_split": split,
                    "scene_id": scene,
                    "episode_id": scene_index,
                    "system2_call_index": target,
                    "protocol_seed": 42,
                    "instruction": "walk to the pantry",
                    "distance_to_goal_m": 2.0 if target else 8.0,
                    "stop_target": target,
                    "original_output": (
                        "view: stop" if target else "view: right\npixel: 70 80"
                    ),
                    "original_terminal": bool(target),
                    "effective_output": (
                        "view: stop" if target else "view: right\npixel: 70 80"
                    ),
                    "system2_decision_scores": {
                        "class_probabilities": {"front": 0.1, "right": 0.8},
                        "stop_log_odds": -20.0 if scene_index == 0 else -3.0,
                    },
                    "current_views": _write_views(root, key, 30 + target),
                    "history_views": [history],
                    "image_size": [8, 8],
                }
            )
    labels = root / "system2_stop_multimodal_examples.jsonl"
    labels.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_rollout_dataset_reconstructs_prompt_and_targets(tmp_path):
    _write_root(tmp_path)
    dataset = System2StopMultimodalDataset([tmp_path], image_size=(8, 8))

    negative = dataset[0]
    positive = dataset[1]

    assert len(dataset) == 4
    assert dataset.original_terminals == (False, True, False, True)
    assert negative["current_views"].shape == (4, 3, 8, 8)
    assert negative["history_panoramas"].shape == (1, 4, 3, 8, 8)
    assert negative["action"].shape == (2,)
    assert negative["pano_view_id"] == "right"
    assert negative["pano_pixel_goal"] == [70, 80]
    assert negative["is_stop"] == 0.0
    assert positive["pano_sample_kind"] == "stop"
    assert positive["is_stop"] == 1.0


def test_false_stop_uses_rejected_stop_text_without_fake_waypoint(tmp_path):
    _write_root(tmp_path)
    dataset = System2StopMultimodalDataset([tmp_path], image_size=(8, 8))
    dataset.records[0]["original_terminal"] = True
    dataset.original_terminals = (True, *dataset.original_terminals[1:])

    false_stop = dataset[0]

    assert false_stop["pano_view_id"] == "view_stop"
    assert false_stop["pano_pixel_goal"] is None
    assert false_stop["pano_sample_kind"] == "stop_reject"
    assert false_stop["is_stop"] == 0.0
    assert false_stop["system2_oracle_stop_target"] == 0


def test_rollout_dataset_rejects_val_unseen_leakage(tmp_path):
    _write_root(tmp_path, split="val_unseen")

    with pytest.raises(ValueError, match="Refusing non-train"):
        System2StopMultimodalDataset([tmp_path], image_size=(8, 8))


def test_rollout_dataset_scene_subset_preserves_metadata(tmp_path):
    _write_root(tmp_path)
    dataset = System2StopMultimodalDataset([tmp_path], image_size=(8, 8))

    subset = dataset.subset_by_indices([0, 1])

    assert len(subset) == 2
    assert set(subset.sample_scene_ids) == {"scene_a"}
    assert subset.targets == (0, 1)
    assert subset.original_terminals == (False, True)


def test_rollout_dataset_scene_split_is_deterministic_and_disjoint(tmp_path):
    _write_root(tmp_path)
    dataset = System2StopMultimodalDataset([tmp_path], image_size=(8, 8))

    train_a, validation_a = dataset.split_by_scene(
        holdout_fraction=0.5,
        seed=17,
    )
    train_b, validation_b = dataset.split_by_scene(
        holdout_fraction=0.5,
        seed=17,
    )

    assert train_a.sample_scene_ids == train_b.sample_scene_ids
    assert validation_a.sample_scene_ids == validation_b.sample_scene_ids
    assert set(train_a.sample_scene_ids).isdisjoint(validation_a.sample_scene_ids)
    assert set(train_a.targets) == {0, 1}
    assert set(validation_a.targets) == {0, 1}


class _NativeDataset:
    _is_panoramic = True

    def __len__(self):
        return 5

    def __getitem__(self, index):
        return {"native_index": int(index)}


def test_mixed_system2_sft_dataset_covers_native_and_onpolicy_roles(tmp_path):
    _write_root(tmp_path)
    rollout = System2StopMultimodalDataset([tmp_path], image_size=(8, 8))
    records = list(rollout.records)
    records[0] = {**records[0], "original_terminal": True}
    rollout.records = records
    rollout.original_terminals = tuple(
        bool(record["original_terminal"]) for record in records
    )

    mixed = MixedSystem2SFTDataset(
        _NativeDataset(),
        rollout,
        native_slots=2,
        positive_slots=1,
        regular_negative_slots=1,
        false_stop_negative_slots=1,
    )

    assert len(mixed) == 15
    assert mixed.source_counts() == {
        "native": 6,
        "onpolicy_positive": 3,
        "onpolicy_regular_negative": 3,
        "onpolicy_false_stop_negative": 3,
    }
    assert mixed.pool_sizes() == {
        "onpolicy_positive": 2,
        "onpolicy_regular_negative": 1,
        "onpolicy_false_stop_negative": 1,
    }
    assert mixed[0]["system2_replay_role"] == "native"
    assert mixed[2]["system2_replay_role"] == "onpolicy_positive"
    assert mixed[3]["system2_replay_role"] == "onpolicy_regular_negative"
    assert mixed[3]["pano_view_id"] == "view_stop"
    assert mixed[3]["pano_pixel_goal"] is None
    assert mixed[3]["pano_sample_kind"] == "stop_reject"
    assert mixed[4]["system2_replay_role"] == "onpolicy_false_stop_negative"


def test_mixed_system2_sft_dataset_mines_hard_regular_negatives(tmp_path):
    _write_root(tmp_path)
    rollout = System2StopMultimodalDataset([tmp_path], image_size=(8, 8))
    records = list(rollout.records)
    records[1] = {
        **records[1],
        "stop_target": 0,
        "original_terminal": True,
    }
    rollout.records = records
    rollout.targets = tuple(int(record["stop_target"]) for record in records)
    rollout.original_terminals = tuple(
        bool(record["original_terminal"]) for record in records
    )

    mixed = MixedSystem2SFTDataset(
        _NativeDataset(),
        rollout,
        native_slots=2,
        positive_slots=1,
        regular_negative_slots=1,
        false_stop_negative_slots=1,
        regular_negative_min_stop_log_odds=-10.0,
    )

    assert mixed.regular_negative_indices == (2,)
    assert mixed.regular_negative_mining_contract() == {
        "min_stop_log_odds": -10.0,
        "candidate_count": 2,
        "selected_count": 1,
    }


def test_mixed_dataset_attaches_nearest_same_episode_positive_to_false_stop(
    tmp_path,
):
    _write_root(tmp_path)
    rollout = System2StopMultimodalDataset([tmp_path], image_size=(8, 8))
    records = list(rollout.records)
    records[0] = {**records[0], "original_terminal": True}
    rollout.records = records
    rollout.original_terminals = tuple(
        bool(record["original_terminal"]) for record in records
    )
    mixed = MixedSystem2SFTDataset(
        _NativeDataset(),
        rollout,
        native_slots=2,
        positive_slots=1,
        regular_negative_slots=1,
        false_stop_negative_slots=1,
        pair_false_stops=True,
    )

    false_stop = mixed[4]
    paired_positive = false_stop["_system2_paired_positive"]

    assert false_stop["system2_replay_role"] == "onpolicy_false_stop_negative"
    assert paired_positive["system2_replay_role"] == "onpolicy_paired_positive"
    assert false_stop["system2_stop_pair_id"] == paired_positive["system2_stop_pair_id"]
    assert paired_positive["system2_oracle_stop_target"] == 1
    next_false_stop = mixed[9]
    assert next_false_stop["system2_stop_pair_id"] != false_stop["system2_stop_pair_id"]
    assert mixed.false_stop_pairing_contract() == {
        "enabled": True,
        "candidate_count": 1,
        "available_paired_count": 1,
        "selected_count": 1,
    }

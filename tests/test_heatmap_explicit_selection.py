import json
from pathlib import Path

import pytest
import torch

from scripts.tools.diagnose_heatmap_shortcuts import (
    history_mask_for_slots,
    load_explicit_selection,
    parse_history_slots,
    selection_contract,
)


class _Dataset:
    def __init__(self, root: Path, split: str, scenes: list[str]) -> None:
        self.root = root
        self.clips = [root / split / scene / f"clip_{index:06d}" for index, scene in enumerate(scenes)]
        self.sample_index = [(index, 10 + index) for index in range(len(self.clips))]

    def __len__(self) -> int:
        return len(self.sample_index)


def _manifest_entry(dataset: _Dataset, indices: list[int]) -> dict:
    contract = selection_contract(dataset, indices)
    return {
        "sample_count": contract["sample_count"],
        "dataset_indices": indices,
        "sample_ids": contract["sample_identities"],
        "sample_identity_sha256": contract["sample_identity_sha256"],
        "scenes": contract["scenes"],
    }


def _write_manifest(path: Path, train: _Dataset, val: _Dataset) -> None:
    payload = {
        "schema_version": "task35b_debiased_selection_v1",
        "baseline": {
            "train": _manifest_entry(train, [0]),
            "val": _manifest_entry(val, [0]),
        },
        "debiased": {
            "train": _manifest_entry(train, [1, 0]),
            "val": _manifest_entry(val, [1, 0]),
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_explicit_selection_verifies_ordered_identities(tmp_path):
    train = _Dataset(tmp_path, "train", ["scene_a", "scene_b"])
    val = _Dataset(tmp_path, "val", ["scene_c", "scene_d"])
    path = tmp_path / "selection_manifest.json"
    _write_manifest(path, train, val)

    train_indices, val_indices, verified = load_explicit_selection(
        path,
        "debiased",
        train,
        val,
    )

    assert train_indices == [1, 0]
    assert val_indices == [1, 0]
    assert verified["selection_name"] == "debiased"
    assert verified["manifest_path"] == str(path.resolve())
    assert verified["train"] == selection_contract(train, train_indices)
    assert verified["val"] == selection_contract(val, val_indices)


def test_load_explicit_selection_rejects_stale_identity(tmp_path):
    train = _Dataset(tmp_path, "train", ["scene_a", "scene_b"])
    val = _Dataset(tmp_path, "val", ["scene_c", "scene_d"])
    path = tmp_path / "selection_manifest.json"
    _write_manifest(path, train, val)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["debiased"]["val"]["sample_ids"][0] = "stale/clip:frame=0"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match dataset"):
        load_explicit_selection(path, "debiased", train, val)


def test_load_explicit_selection_rejects_duplicate_indices(tmp_path):
    train = _Dataset(tmp_path, "train", ["scene_a", "scene_b"])
    val = _Dataset(tmp_path, "val", ["scene_c", "scene_d"])
    path = tmp_path / "selection_manifest.json"
    _write_manifest(path, train, val)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["debiased"]["train"]["dataset_indices"] = [0, 0]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="duplicate dataset indices"):
        load_explicit_selection(path, "debiased", train, val)


def test_history_slot_loss_mask_is_explicit_and_bounded():
    assert parse_history_slots("", 2) is None
    assert parse_history_slots("1,0,1", 2) == (0, 1)
    assert parse_history_slots("0", 2) == (0,)
    with pytest.raises(ValueError, match="outside"):
        parse_history_slots("2", 2)

    all_slots = history_mask_for_slots(
        2,
        device=torch.device("cpu"),
        active_slots=None,
    )
    oldest_only = history_mask_for_slots(
        2,
        device=torch.device("cpu"),
        active_slots=(0,),
    )
    assert all_slots.tolist() == [[True, True]]
    assert oldest_only.tolist() == [[True, False]]
    with pytest.raises(ValueError, match="outside"):
        history_mask_for_slots(
            2,
            device=torch.device("cpu"),
            active_slots=(2,),
        )

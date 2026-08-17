"""Dataset wiring tests for heatmap-control trajectory training."""

from __future__ import annotations

from unittest.mock import patch

import torch

from src.data.factory import (
    build_dataset,
    build_trajectory_dagger_dataset,
)
from src.data.trajectory_dataset import (
    HABITAT_HISTORY_POSE_CONVENTION,
    LEGACY_HISTORY_POSE_CONVENTION,
    _history_pose_contract,
    _history_temporal_metadata,
)


def _mixture_config() -> dict:
    return {
        "data": {
            "root": "/expert",
            "train_split": "train",
            "image_size": [224, 224],
            "init_hm_size": [64, 64],
            "dataset_type": "expert_dagger_mixture",
            "trajectory": {},
            "trajectory_dagger": {
                "collection_roots": ["/dagger/train"],
                "val_collection_roots": ["/dagger/val"],
                "num_history": 8,
            },
            "mixture": {
                "profile": "expert50_normal20_hard30",
            },
        }
    }


def test_factory_reuses_zero_copy_expert_dagger_builder() -> None:
    cfg = _mixture_config()
    with (
        patch(
            "src.data.factory.build_trajectory_dataset",
            return_value="expert",
        ) as expert_builder,
        patch(
            "src.data.factory.build_trajectory_dagger_dataset",
            return_value="dagger",
        ) as dagger_builder,
        patch(
            "src.data.trajectory_dagger_dataset.build_expert_dagger_mixture",
            return_value="mixture",
        ) as mixture_builder,
    ):
        result = build_dataset(
            cfg,
            split="train",
            expert_overrides={"enable_augmentation": False},
            dagger_overrides={"verify_tar_sha256": True},
        )

    assert result == "mixture"
    expert_builder.assert_called_once_with(
        cfg,
        "train",
        enable_augmentation=False,
    )
    dagger_builder.assert_called_once_with(
        cfg,
        "train",
        verify_tar_sha256=True,
    )
    mixture_builder.assert_called_once_with("expert", "dagger")


def test_dagger_factory_selects_validation_collection_roots() -> None:
    cfg = _mixture_config()
    with patch(
        "src.data.trajectory_dagger_dataset.TrajectoryDaggerDataset"
    ) as dataset_type:
        build_trajectory_dagger_dataset(cfg, split="val")

    assert dataset_type.call_args.kwargs["collection_roots"] == [
        "/dagger/val"
    ]


def test_expert_history_metadata_matches_dagger_contract() -> None:
    frame_ids, valid, mask, ages = _history_temporal_metadata(
        [2, 5, 9],
        current_t=12,
    )
    assert torch.equal(frame_ids, torch.tensor([2, 5, 9]))
    assert valid.dtype == torch.bool
    assert valid.tolist() == [True, True, True]
    assert mask.dtype == torch.float32
    assert torch.equal(ages, torch.tensor([10, 7, 3]))


def test_expert_pose_contract_uses_habitat_minus_z_only_for_native_targets() -> None:
    assert _history_pose_contract("internnav_habitat") == (
        "-z",
        HABITAT_HISTORY_POSE_CONVENTION,
    )
    assert _history_pose_contract("legacy_pitched_camera") == (
        "+z",
        LEGACY_HISTORY_POSE_CONVENTION,
    )

"""Deployment boundary tests for automatic epoch-3 control evaluation."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from scripts.training.heatmap_control_deployment import (
    HeatmapControlDeploymentError,
    validate_heatmap_control_deployment_checkpoint,
)
from scripts.training.native_internnav_dependency import (
    NATIVE_DEPENDENCY_SCHEMA,
    NATIVE_MODEL_FILE_COUNT,
    NATIVE_MODEL_MANIFEST_PATH,
    NATIVE_MODEL_MANIFEST_SHA256,
    NATIVE_MODEL_PATH,
)


HEATMAP_SHA = "a" * 64
POLICY_FINGERPRINT = "internnav-native-v1:" + "b" * 64
ROOTS = [f"/datasets/dagger/shard_{index:02d}" for index in range(4)]


def _payload() -> dict:
    state = {"heatmap_tokenizer.weight": torch.ones(2, 2)}
    state.update(
        {
            (
                "nextdit_action_head.traj_dit.model.layers."
                f"{index}.heatmap_control.gate"
            ): torch.zeros(4)
            for index in range(12)
        }
    )
    return {
        "epoch": 3,
        "stage_idx": 0,
        "batch": None,
        "stage_name": "heatmap_system1_control",
        "weight_semantics": {"trainable_state_dict": "ema"},
        "trainable_state_dict": state,
        "ema_state_dict": {
            "shadow": copy.deepcopy(state),
            "target_decay": 0.999,
            "warmup_steps": 500,
            "step_count": 100,
        },
        "config": {
            "model": {
                "llm": {"model_path": NATIVE_MODEL_PATH},
                "action_head": {
                    "nextdit": {
                        "internnav_model_path": NATIVE_MODEL_PATH,
                        "heatmap_control": {
                            "heatmap_checkpoint_sha256": HEATMAP_SHA,
                        }
                    }
                }
            },
            "runtime": {
                "native_internnav_dependency": {
                    "schema": NATIVE_DEPENDENCY_SCHEMA,
                    "model_path": NATIVE_MODEL_PATH,
                    "manifest_path": NATIVE_MODEL_MANIFEST_PATH,
                    "manifest_sha256": NATIVE_MODEL_MANIFEST_SHA256,
                    "file_count": NATIVE_MODEL_FILE_COUNT,
                    "verified": True,
                },
                "frozen_heatmap_dependency": {
                    "checkpoint_sha256": HEATMAP_SHA,
                }
            },
            "data": {
                "dataset_type": "expert_dagger_mixture",
                "in_order": True,
                "mixture": {
                    "profile": "expert50_normal20_hard30",
                    "seed": 42,
                    "epoch_size": 72000,
                },
                "trajectory_dagger": {
                    "collection_roots": ROOTS,
                    "expected_policy_fingerprint": POLICY_FINGERPRINT,
                },
            },
        },
    }


def _validate(tmp_path: Path, payload: dict, name: str = "epoch_003.pth") -> dict:
    path = tmp_path / name
    torch.save(payload, path)
    return validate_heatmap_control_deployment_checkpoint(
        path,
        expected_heatmap_sha256=HEATMAP_SHA,
        expected_policy_fingerprint=POLICY_FINGERPRINT,
        expected_collection_roots=ROOTS,
    )


def test_complete_epoch_three_ema_control_checkpoint_is_accepted(
    tmp_path: Path,
) -> None:
    report = _validate(tmp_path, _payload())
    assert report["epoch"] == 3
    assert report["control_layers"] == list(range(12))
    assert report["tokenizer_tensor_count"] == 1
    assert len(report["checkpoint_sha256"]) == 64


def test_deployment_boundary_rejects_wrong_stage_index(tmp_path: Path) -> None:
    payload = _payload()
    payload["stage_idx"] = 1
    with pytest.raises(HeatmapControlDeploymentError, match="stage_idx=0"):
        _validate(tmp_path, payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.__setitem__("batch", 1000), "complete epoch"),
        (
            lambda payload: payload["weight_semantics"].__setitem__(
                "trainable_state_dict", "online"
            ),
            "EMA",
        ),
        (
            lambda payload: payload["trainable_state_dict"].__setitem__(
                "nextdit_action_head.traj_dit.model.native.weight",
                torch.ones(1),
            ),
            "forbidden deployment tensor",
        ),
        (
            lambda payload: payload["trainable_state_dict"].pop(
                "nextdit_action_head.traj_dit.model.layers.11.heatmap_control.gate"
            ),
            "layers 0..11",
        ),
        (
            lambda payload: payload["trainable_state_dict"].__setitem__(
                "heatmap_tokenizer.weight", torch.tensor([1], dtype=torch.int64)
            ),
            "not floating point",
        ),
        (
            lambda payload: payload["trainable_state_dict"].__setitem__(
                "heatmap_tokenizer.weight", torch.tensor([float("nan")])
            ),
            "non-finite",
        ),
        (
            lambda payload: payload["config"]["data"]["mixture"].__setitem__(
                "epoch_size", 72160
            ),
            "72k mixture",
        ),
        (
            lambda payload: payload["config"]["data"][
                "trajectory_dagger"
            ].__setitem__("collection_roots", list(reversed(ROOTS))),
            "DAgger roots",
        ),
        (
            lambda payload: payload["config"]["runtime"][
                "frozen_heatmap_dependency"
            ].__setitem__("checkpoint_sha256", "c" * 64),
            "heatmap SHA-256",
        ),
        (
            lambda payload: payload["config"]["runtime"][
                "native_internnav_dependency"
            ].__setitem__("verified", False),
            "released InternNav closure",
        ),
        (
            lambda payload: payload["ema_state_dict"]["shadow"]
            .get("heatmap_tokenizer.weight")
            .add_(1),
            "exact EMA tensor",
        ),
    ],
)
def test_deployment_boundary_rejects_invalid_eval_artifacts(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload = copy.deepcopy(_payload())
    mutation(payload)
    with pytest.raises(HeatmapControlDeploymentError, match=message):
        _validate(tmp_path, payload, "invalid.pth")

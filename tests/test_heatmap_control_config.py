"""Fail-closed configuration tests for frozen-InternNav heatmap control."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from src.config_schema import HeatmapControlConfig, TrainConfig


NATIVE_FINGERPRINT = "internnav-native-v1:" + "b" * 64


def _control_config() -> dict:
    return {
        "seed": 42,
        "data": {
            "root": "/datasets/r2r_expert",
            "image_size": [224, 224],
            "init_hm_size": [64, 64],
            "dataset_type": "expert_dagger_mixture",
            "in_order": True,
            "trajectory": {
                "trajectory_target_convention": "internnav_habitat",
                "predict_horizon": 32,
                "load_traj_images": True,
                "load_single_view_history_frames": True,
                "panoramic_vlm_input": False,
                "pixel_goal_direction": "front_down",
            },
            "trajectory_dagger": {
                "collection_roots": ["/datasets/dagger"],
                "expected_policy_mode": "internnav_native",
                "expected_policy_fingerprint": NATIVE_FINGERPRINT,
            },
            "mixture": {
                "profile": "expert50_normal20_hard30",
                "epoch_size": 1000,
                "seed": 42,
            },
        },
        "model": {
            "llm": {
                "model_path": "/models/original-internnav",
                "use_lora": False,
                "gradient_checkpointing": False,
            },
            "heatmap": {
                "enable": True,
                "input_mode": "internnav_single_view",
                "feature_source": "vit_only",
                "architecture_id": (
                    "internnav_single_view_vision_only_four_direction_v2"
                ),
                "output_direction_order": [
                    "front",
                    "right",
                    "back",
                    "left",
                ],
                "history_pose_convention": (
                    "habitat_c2w_minus_z__forward_left_cos_yaw_sin_yaw__v1"
                ),
                "restore_vit_spatial_layout": True,
                "heatmap_trains_backbone": False,
                "trajectory": {
                    "enable": True,
                    "num_freqs": 16,
                    "d_attn": 256,
                    "num_heads": 4,
                    "num_layers": 2,
                    "max_spatial_range": 10.0,
                },
            },
            "action_head": {
                "enable": True,
                "nextdit": {
                    "enabled": True,
                    "internnav_model_path": "/models/original-internnav",
                    "internnav_system1_path": "",
                    "pretrained_system1_path": None,
                    "dav2_ckpt_path": "",
                    "warmup_steps": 0,
                    "pano_latent_adapter": {"enabled": False},
                    "heatmap_control": {
                        "enabled": True,
                        "heatmap_checkpoint_path": "/models/heatmap.pth",
                        "heatmap_checkpoint_sha256": "a" * 64,
                    },
                },
            },
        },
        "loss": {
            "trajectory_weight": 1.0,
            "heatmap_weight": 0.0,
            "lm_weight": 0.0,
        },
        "training": {
            "stages": [
                {
                    "name": "heatmap_control",
                    "epochs": 1,
                    "train_heatmap": False,
                    "train_history": False,
                    "train_future": False,
                    "train_lm": False,
                    "train_system2_sft": False,
                    "train_action": True,
                    "strict_trainable_modules": True,
                    "trainable_modules": [
                        "heatmap_tokenizer",
                        "heatmap_control",
                    ],
                }
            ]
        },
        "log": {"out_dir": "/tmp/heatmap-control"},
    }


def _set_nested(config: dict, dotted_path: str, value: object) -> None:
    current = config
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def test_control_config_accepts_only_frozen_original_internnav() -> None:
    parsed = TrainConfig.model_validate(_control_config())
    control = parsed.model.action_head.nextdit.heatmap_control
    assert control.schema_version == "heatmap-control-v1"
    assert control.token_dim == control.control_dim == 128
    assert control.num_heads == control.temporal_heads == 4
    assert control.coarse_size == 8
    assert control.age_normalizer_steps == 32.0
    assert parsed.data.mixture.profile == "expert50_normal20_hard30"
    assert parsed.data.mixture.epoch_size == 1000
    assert parsed.data.in_order is True


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            "model.action_head.nextdit.internnav_model_path",
            "/models/different",
            "same original InternNav model path",
        ),
        ("model.llm.use_lora", True, "forbids System2 LoRA"),
        (
            "model.action_head.nextdit.pano_latent_adapter",
            {"enabled": True},
            "panoramic latent adapter",
        ),
        ("model.action_head.nextdit.warmup_steps", 10, "warmup_steps=0"),
        (
            "training.stages.0.trainable_modules",
            ["heatmap_control"],
            "must train exactly",
        ),
        ("training.stages.0.train_action", False, "train_action=true"),
        ("training.stages.0.train_history", True, "train_history=false"),
        ("training.stages.0.train_lm", True, "train_lm=false"),
        ("loss.trajectory_weight", 0.0, "trajectory_weight > 0"),
        ("loss.heatmap_weight", 1.0, "heatmap_weight=0"),
        (
            "model.action_head.nextdit.heatmap_control.control_dim",
            64,
            "token_dim == control_dim",
        ),
        (
            "model.action_head.nextdit.heatmap_control.heatmap_checkpoint_sha256",
            "not-a-digest",
            "SHA-256",
        ),
        (
            "data.trajectory.load_traj_images",
            False,
            "load_traj_images=true",
        ),
        (
            "data.in_order",
            False,
            "data.in_order=true",
        ),
    ],
)
def test_control_config_rejects_contract_drift(
    path: str,
    value: object,
    message: str,
) -> None:
    config = copy.deepcopy(_control_config())
    if ".0." in path:
        prefix, suffix = path.split(".0.", 1)
        container = config
        for part in prefix.split("."):
            container = container[part]
        container[0][suffix] = value
    else:
        _set_nested(config, path, value)
    with pytest.raises(ValidationError, match=message):
        TrainConfig.model_validate(config)


def test_control_mixture_requires_explicit_epoch_size() -> None:
    config = _control_config()
    del config["data"]["mixture"]["epoch_size"]
    with pytest.raises(ValidationError, match="explicit mixture.epoch_size"):
        TrainConfig.model_validate(config)


def test_pure_dagger_config_does_not_require_expert_root() -> None:
    config = {
        "data": {
            "dataset_type": "trajectory_dagger",
            "image_size": [224, 224],
            "init_hm_size": [64, 64],
            "trajectory_dagger": {
                "collection_roots": ["/datasets/dagger"],
                "expected_policy_mode": "internnav_native",
                "expected_policy_fingerprint": NATIVE_FINGERPRINT,
            },
        },
        "training": {"stages": [{"name": "dagger", "epochs": 1}]},
        "log": {"out_dir": "/tmp/dagger"},
    }
    parsed = TrainConfig.model_validate(config)
    assert parsed.data.root is None
    assert parsed.data.trajectory_dagger.require_lookdown is True


def test_custom_mixture_weights_are_typed_and_complete() -> None:
    config = _control_config()
    config["data"]["mixture"] = {
        "profile": None,
        "weights": {
            "expert": 0.5,
            "dagger_normal": 0.2,
            "dagger_hard": 0.3,
        },
        "epoch_size": 1000,
        "seed": 42,
    }
    parsed = TrainConfig.model_validate(config)
    assert parsed.data.mixture.weights["dagger_hard"] == 0.3


def test_enabled_control_requires_checkpoint_provenance() -> None:
    with pytest.raises(ValidationError, match="heatmap_checkpoint_path"):
        HeatmapControlConfig(enabled=True)

"""Contracts for the internnav_single_view arm of the shortcut diagnostic."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts.tools.diagnose_heatmap_shortcuts import (
    CONSTANT_REL_POSE,
    constant_rel_poses,
    is_single_view,
    load_config,
    transform_sample,
)
from scripts.tools.summarize_heatmap_shortcuts import (
    expected_lora_tensors,
    matched_contract,
)


def _sample(num_history: int = 3) -> dict:
    return {
        "history_frames": torch.rand(num_history, 3, 8, 8),
        "current_frame": torch.rand(3, 8, 8),
        "current_views": torch.rand(4, 3, 8, 8),
        "history_panoramas": torch.rand(num_history, 4, 3, 8, 8),
        "history_rel_poses": torch.rand(num_history, 4),
        "gt_visibility": torch.rand(num_history, 4),
        "heatmap": torch.rand(num_history, 4, 8, 8),
        "heatmap_direction_order": ("front", "right", "back", "left"),
        "history_pose_convention": "habitat_c2w_minus_z__forward_left_cos_yaw_sin_yaw__v1",
        "history_pose_provider": "habitat_gt",
        "action": torch.zeros(2),
        "action_valid": 1.0,
        "discrete_action": 1,
        "is_stop": 0.0,
    }


SINGLE_VIEW = "internnav_single_view"


class TestSingleViewTransform:
    def test_panoramic_rgb_never_reaches_the_single_view_collator(self) -> None:
        out = transform_sample(
            _sample(),
            train_mode="full",
            perturbation="none",
            partner=None,
            architecture=SINGLE_VIEW,
        )
        assert "current_views" not in out
        assert "history_panoramas" not in out
        # Provenance metadata the collator validates fail-closed.
        for key in (
            "heatmap_direction_order",
            "history_pose_convention",
            "history_pose_provider",
            "heatmap",
        ):
            assert key in out

    def test_pose_ablation_supplies_a_constant_instead_of_none(self) -> None:
        # The deployed head has no absent-pose state, so the ablation must feed
        # a constant that carries no information about history layout.
        for mode in ("vision-only", "no-input"):
            out = transform_sample(
                _sample(),
                train_mode=mode,
                perturbation="none",
                partner=None,
                architecture=SINGLE_VIEW,
            )
            poses = out["history_rel_poses"]
            assert poses is not None
            assert poses.shape == (3, 4)
            assert torch.equal(poses[0], torch.tensor(CONSTANT_REL_POSE))
            assert poses.unique(dim=0).shape[0] == 1

    def test_legacy_pose_ablation_still_removes_the_input(self) -> None:
        out = transform_sample(
            _sample(),
            train_mode="vision-only",
            perturbation="none",
            partner=None,
        )
        assert out["history_rel_poses"] is None
        assert out["current_views"] is not None

    def test_blank_image_modes_zero_every_rgb_input(self) -> None:
        for mode in ("pose-only", "no-input"):
            out = transform_sample(
                _sample(),
                train_mode=mode,
                perturbation="none",
                partner=None,
                architecture=SINGLE_VIEW,
            )
            assert not out["history_frames"].any()
            assert not out["current_frame"].any()

    def test_history_shuffle_reverses_only_the_images(self) -> None:
        sample = _sample()
        out = transform_sample(
            sample,
            train_mode="full",
            perturbation="history-shuffle",
            partner=None,
            architecture=SINGLE_VIEW,
        )
        assert torch.equal(out["history_frames"], sample["history_frames"].flip(0))
        assert torch.equal(out["history_rel_poses"], sample["history_rel_poses"])

    def test_current_shuffle_takes_the_partner_front_image(self) -> None:
        sample, partner = _sample(), _sample()
        out = transform_sample(
            sample,
            train_mode="full",
            perturbation="current-shuffle",
            partner=partner,
            architecture=SINGLE_VIEW,
        )
        assert torch.equal(out["current_frame"], partner["current_frame"])

    def test_shifted_target_rolls_the_collator_target_too(self) -> None:
        sample = _sample()
        out = transform_sample(
            sample,
            train_mode="full",
            perturbation="pose-conflict-shifted-target",
            partner=None,
            architecture=SINGLE_VIEW,
        )
        rolled = torch.roll(sample["heatmap"], shifts=1, dims=0)
        assert torch.equal(out["gt_heatmaps"], rolled)
        # The collator validates sample["heatmap"]; it must agree with the target.
        assert torch.equal(out["heatmap"], rolled)
        assert torch.equal(
            out["gt_visibility"], torch.roll(sample["gt_visibility"], shifts=1, dims=0)
        )

    def test_constant_poses_match_reference_dtype_and_device(self) -> None:
        reference = torch.zeros(5, 4, dtype=torch.float64)
        poses = constant_rel_poses(reference)
        assert poses.shape == reference.shape
        assert poses.dtype == torch.float64


def _write_config(tmp_path, *, input_mode: str, use_lora: bool):
    cfg = {
        "data": {"image_size": [384, 384], "init_hm_size": [64, 64]},
        "model": {
            "llm": {"use_lora": use_lora},
            "heatmap": {"input_mode": input_mode},
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _args(tmp_path, config, architecture: str):
    return SimpleNamespace(
        config=str(config),
        data_root=str(tmp_path),
        device="cpu",
        num_history=8,
        architecture=architecture,
        internnav_model_path="/models/internnav",
    )


class TestLoadConfigArchitecture:
    def test_single_view_keeps_the_backbone_ungraded_and_front_only(self, tmp_path) -> None:
        config = _write_config(tmp_path, input_mode="internnav_single_view", use_lora=False)
        cfg = load_config(_args(tmp_path, config, "internnav_single_view"))
        assert cfg["model"]["heatmap"]["heatmap_trains_backbone"] is False
        assert cfg["data"]["sliding_window"]["single_view_rgb_input"] is True
        assert cfg["model"]["llm"]["model_path"] == "/models/internnav"

    def test_single_view_refuses_a_panoramic_config(self, tmp_path) -> None:
        config = _write_config(tmp_path, input_mode="panoramic", use_lora=False)
        with pytest.raises(ValueError, match="input_mode=internnav_single_view"):
            load_config(_args(tmp_path, config, "internnav_single_view"))

    def test_single_view_refuses_lora(self, tmp_path) -> None:
        config = _write_config(tmp_path, input_mode="internnav_single_view", use_lora=True)
        with pytest.raises(ValueError, match="forbids LoRA"):
            load_config(_args(tmp_path, config, "internnav_single_view"))

    def test_legacy_path_is_unchanged(self, tmp_path) -> None:
        config = _write_config(tmp_path, input_mode="panoramic", use_lora=True)
        cfg = load_config(_args(tmp_path, config, "legacy_panoramic"))
        assert cfg["model"]["heatmap"]["heatmap_trains_backbone"] is True
        assert cfg["data"]["sliding_window"]["single_view_rgb_input"] is False
        assert cfg["model"]["heatmap"]["llm_layer_indices"] == [6, 13, 20]

    def test_namespaces_without_the_flag_stay_legacy(self, tmp_path) -> None:
        # Sibling probes reuse load_config with their own argument namespaces.
        config = _write_config(tmp_path, input_mode="panoramic", use_lora=True)
        legacy_args = SimpleNamespace(
            config=str(config),
            data_root=str(tmp_path),
            device="cpu",
            num_history=8,
        )
        assert is_single_view(legacy_args) is False
        cfg = load_config(legacy_args)
        assert cfg["model"]["heatmap"]["heatmap_trains_backbone"] is True


def _report(mode: str, architecture: str, lora: int, seed: int = 42):
    return {
        "mode": mode,
        "architecture": architecture,
        "seed": seed,
        "checkpoint": None,
        "initial_head_hash": "head-hash",
        "train_steps": 100,
        "trainable_head_numel": 17,
        "trainable_qwen_tensors": 0,
        "load": {"matched_lora_tensors": lora},
        "selection_contract": {
            "train": {"sample_identity_sha256": "train-hash"},
            "val": {"sample_identity_sha256": "val-hash"},
        },
        "evaluations": {"standard": {"loss": 1.0}},
    }


class TestSummarizerContract:
    def test_single_view_expects_no_adapter_tensors(self) -> None:
        assert expected_lora_tensors("internnav_single_view") == 0
        assert expected_lora_tensors("legacy_panoramic") == 224

    def test_four_matched_single_view_probes_pass(self) -> None:
        reports = {
            mode: _report(mode, "internnav_single_view", 0)
            for mode in ("full", "vision-only", "pose-only", "no-input")
        }
        contract = matched_contract(reports)
        assert contract["passed"], contract["checks"]
        assert contract["architecture"] == ["internnav_single_view"]

    def test_mixed_architectures_are_refused(self) -> None:
        reports = {
            "full": _report("full", "internnav_single_view", 0),
            "vision-only": _report("vision-only", "legacy_panoramic", 224),
            "pose-only": _report("pose-only", "internnav_single_view", 0),
        }
        contract = matched_contract(reports)
        assert not contract["passed"]
        assert contract["checks"]["same_architecture"] is False

    def test_mismatched_seed_or_selection_is_refused(self) -> None:
        reports = {
            "full": _report("full", "internnav_single_view", 0),
            "vision-only": _report("vision-only", "internnav_single_view", 0, seed=7),
            "pose-only": _report("pose-only", "internnav_single_view", 0),
        }
        contract = matched_contract(reports)
        assert not contract["passed"]
        assert contract["checks"]["same_seed"] is False

"""Unit contracts for the v2 trust-region PPA bridge retraining stack."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from src.models.past_plan_action import (
    PastToPlanBridge,
    compute_shared_plan_action_losses,
)


def _memory_inputs(batch: int, tokens: int, memory_dim: int):
    memory = torch.randn(batch, tokens, memory_dim)
    memory_mask = torch.ones(batch, tokens, dtype=torch.bool)
    return memory, memory_mask


class TestBridgeTrustRegion:
    def test_invalid_ratio_is_rejected(self) -> None:
        for ratio in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="max_delta_ratio"):
                PastToPlanBridge(
                    plan_dim=8, memory_dim=4, num_heads=2, max_delta_ratio=ratio
                )

    def test_zero_bridge_stays_bitwise_identity_under_trust_region(self) -> None:
        bridge = PastToPlanBridge(
            plan_dim=8, memory_dim=4, num_heads=2, max_delta_ratio=0.05
        )
        plan_z0 = torch.randn(2, 4, 8)
        memory, memory_mask = _memory_inputs(2, 3, 4)
        result, diagnostics = bridge(
            plan_z0, memory, memory_mask, return_diagnostics=True
        )
        assert torch.equal(result, plan_z0)
        assert diagnostics["delta_token_ratio"].shape == (2, 4)
        assert torch.equal(
            diagnostics["delta_token_ratio"],
            torch.zeros(2, 4),
        )

    def test_trust_region_caps_per_token_norm_and_keeps_direction(self) -> None:
        torch.manual_seed(7)
        unclamped = PastToPlanBridge(plan_dim=8, memory_dim=4, num_heads=2)
        clamped = PastToPlanBridge(
            plan_dim=8, memory_dim=4, num_heads=2, max_delta_ratio=0.05
        )
        with torch.no_grad():
            unclamped.cross_attention.out_proj.weight.fill_(0.5)
            unclamped.cross_attention.out_proj.bias.fill_(0.1)
        clamped.load_state_dict(unclamped.state_dict())

        plan_z0 = torch.randn(2, 4, 8)
        memory, memory_mask = _memory_inputs(2, 3, 4)
        raw = unclamped(plan_z0, memory, memory_mask) - plan_z0
        result, diagnostics = clamped(
            plan_z0, memory, memory_mask, return_diagnostics=True
        )
        delta = result - plan_z0

        z0_norm = plan_z0.norm(dim=-1)
        assert (delta.norm(dim=-1) <= 0.05 * z0_norm * (1 + 1e-4)).all()
        assert (diagnostics["delta_token_ratio"] <= 0.05 * (1 + 1e-4)).all()
        cosine = torch.nn.functional.cosine_similarity(
            delta.reshape(-1, 8), raw.reshape(-1, 8), dim=-1
        )
        assert (cosine > 1 - 1e-5).all()

    def test_trust_region_leaves_small_deltas_untouched(self) -> None:
        torch.manual_seed(11)
        bridge = PastToPlanBridge(
            plan_dim=8, memory_dim=4, num_heads=2, max_delta_ratio=1.0
        )
        reference = PastToPlanBridge(plan_dim=8, memory_dim=4, num_heads=2)
        with torch.no_grad():
            bridge.cross_attention.out_proj.weight.normal_(0.0, 1e-4)
            bridge.cross_attention.out_proj.bias.zero_()
        reference.load_state_dict(bridge.state_dict())

        plan_z0 = torch.randn(2, 4, 8)
        memory, memory_mask = _memory_inputs(2, 3, 4)
        assert torch.equal(
            bridge(plan_z0, memory, memory_mask),
            reference(plan_z0, memory, memory_mask),
        )

    def test_no_memory_bypass_reports_zero_ratio(self) -> None:
        bridge = PastToPlanBridge(
            plan_dim=8, memory_dim=4, num_heads=2, max_delta_ratio=0.05
        )
        plan_z0 = torch.randn(1, 4, 8)
        result, diagnostics = bridge(
            plan_z0, None, None, return_diagnostics=True
        )
        assert torch.equal(result, plan_z0)
        assert torch.equal(
            diagnostics["delta_token_ratio"], torch.zeros(1, 4)
        )


class _FakeProjectedActionHead(nn.Module):
    """Deterministic frozen stand-in exposing the projected-flow APIs.

    ``predict_velocity_from_projected`` returns each sample's condition mean
    broadcast over the trajectory, so action/preserve values are exactly
    hand-computable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()), requires_grad=False)

    @staticmethod
    def _expand_sequence_training_inputs(cond, gt, images, valid):
        return cond, gt, images, valid

    @staticmethod
    def sample_flow_matching_inputs(gt):
        return gt.clone(), torch.zeros(gt.shape[0]), torch.zeros_like(gt)

    def predict_velocity_from_projected(
        self, cond, noisy, timesteps, traj_images=None
    ):
        del timesteps, traj_images
        per_sample = cond.float().mean(dim=(1, 2))
        return per_sample[:, None, None].expand_as(noisy).clone()

    @staticmethod
    def masked_velocity_mse(pred, target, trajectory_valid=None):
        loss = torch.nn.functional.mse_loss(
            pred.float(), target.float(), reduction="none"
        )
        if trajectory_valid is None:
            return loss.mean()
        mask = trajectory_valid.float()
        if mask.sum() <= 0:
            return loss.sum() * 0.0
        per_sample = loss.mean(dim=(1, 2))
        return (per_sample * mask).sum() / mask.sum()


class TestSharedPlanActionLossesV2:
    def _base_inputs(self):
        head = _FakeProjectedActionHead().eval()
        plan_z0 = torch.stack(
            [
                torch.full((2, 4), 1.0),
                torch.full((2, 4), 3.0),
            ]
        )
        gt = torch.zeros(2, 3, 3)
        return head, plan_z0, gt

    def test_advantage_weighting_scales_only_where_native_is_wrong(self) -> None:
        head, plan_z0, gt = self._base_inputs()
        result = compute_shared_plan_action_losses(
            action_head=head,
            plan_z0=plan_z0,
            plan_z=plan_z0.clone(),
            gt_trajectory=gt,
            trajectory_valid=None,
            traj_images=None,
            advantage_reference_mse=1.0,
            advantage_max_weight=4.0,
        )
        # Native per-sample velocity errors are [1, 9] against the zero
        # target, so the weights are [1, min(9, 4)] and the weighted action
        # loss over the unweighted denominator is (1*1 + 9*4) / 2.
        assert result["action"].item() == pytest.approx(18.5)
        assert result["advantage_weight_mean"].item() == pytest.approx(2.5)
        assert result["preserve"].item() == pytest.approx(0.0)

    def test_advantage_disabled_matches_uniform_mean(self) -> None:
        head, plan_z0, gt = self._base_inputs()
        result = compute_shared_plan_action_losses(
            action_head=head,
            plan_z0=plan_z0,
            plan_z=plan_z0.clone(),
            gt_trajectory=gt,
            trajectory_valid=None,
            traj_images=None,
        )
        assert result["action"].item() == pytest.approx(5.0)
        assert "advantage_weight_mean" not in result

    def test_invalid_advantage_reference_is_rejected(self) -> None:
        head, plan_z0, gt = self._base_inputs()
        with pytest.raises(ValueError, match="advantage_reference_mse"):
            compute_shared_plan_action_losses(
                action_head=head,
                plan_z0=plan_z0,
                plan_z=plan_z0.clone(),
                gt_trajectory=gt,
                trajectory_valid=None,
                traj_images=None,
                advantage_reference_mse=0.0,
            )

    def test_relative_delta_penalty_is_scale_free(self) -> None:
        head = _FakeProjectedActionHead().eval()
        plan_z0 = torch.full((1, 2, 4), 2.0)
        plan_z = plan_z0 + 1.0
        gt = torch.zeros(1, 3, 3)
        relative = compute_shared_plan_action_losses(
            action_head=head,
            plan_z0=plan_z0,
            plan_z=plan_z,
            gt_trajectory=gt,
            trajectory_valid=None,
            traj_images=None,
            delta_relative=True,
        )
        absolute = compute_shared_plan_action_losses(
            action_head=head,
            plan_z0=plan_z0,
            plan_z=plan_z,
            gt_trajectory=gt,
            trajectory_valid=None,
            traj_images=None,
        )
        # ||delta||^2 = 4 per token against ||z0||^2 = 16 per token.
        assert relative["delta_z_l2"].item() == pytest.approx(0.25)
        assert absolute["delta_z_l2"].item() == pytest.approx(1.0)


class TestRolloutMetrics:
    def _config(self):
        from src.models.action.treatment_spec import TrajectoryPostprocessConfig

        return TrajectoryPostprocessConfig(num_sample_trajs=2, action_scale=4.0)

    def test_gt_endpoint_uses_deployment_scaling(self) -> None:
        from src.models.action.rollout_metrics import gt_endpoint_xy

        gt = torch.zeros(6, 3)
        gt[:, 0] = 4.0
        endpoint = gt_endpoint_xy(gt, self._config())
        assert endpoint == pytest.approx([6.0, 0.0])

    def test_identical_banks_agree_exactly(self) -> None:
        from src.models.action.rollout_metrics import compute_rollout_pair_metrics

        bank = torch.zeros(2, 6, 3)
        bank[:, :, 0] = 4.0
        gt = torch.zeros(6, 3)
        gt[:, 0] = 4.0
        metrics = compute_rollout_pair_metrics(
            bank_bridged=bank,
            bank_native=bank.clone(),
            gt_trajectory=gt,
            config=self._config(),
        )
        assert metrics["endpoint_error"] == pytest.approx(0.0)
        assert metrics["endpoint_error_native"] == pytest.approx(0.0)
        assert metrics["endpoint_gap_to_native"] == pytest.approx(0.0)
        assert metrics["action_agreement"] == 1.0

    def test_diverging_banks_are_scored_against_native(self) -> None:
        from src.models.action.rollout_metrics import compute_rollout_pair_metrics

        forward_bank = torch.zeros(2, 6, 3)
        forward_bank[:, :, 0] = 4.0
        stop_bank = torch.zeros(2, 6, 3)
        gt = torch.zeros(6, 3)
        gt[:, 0] = 4.0
        metrics = compute_rollout_pair_metrics(
            bank_bridged=stop_bank,
            bank_native=forward_bank,
            gt_trajectory=gt,
            config=self._config(),
        )
        assert metrics["endpoint_error"] == pytest.approx(6.0)
        assert metrics["endpoint_error_native"] == pytest.approx(0.0)
        assert metrics["endpoint_gap_to_native"] == pytest.approx(6.0)
        assert metrics["action_agreement"] == 0.0


class TestConfigSchemaV2:
    def test_max_delta_ratio_bounds(self) -> None:
        from src.config_schema import PastPlanActionConfig

        accepted = PastPlanActionConfig.model_validate(
            {"enabled": True, "max_delta_ratio": 0.05}
        )
        assert accepted.max_delta_ratio == pytest.approx(0.05)
        with pytest.raises(ValueError, match="max_delta_ratio"):
            PastPlanActionConfig.model_validate(
                {"enabled": True, "max_delta_ratio": 1.5}
            )

    def test_reset_bridge_requires_bridge_only_refinement(self) -> None:
        from src.config_schema import TrainingStageConfig

        base = {
            "name": "ppa_action_refine_v2",
            "epochs": 1,
            "past_plan_action_stage": "stage2_joint",
            "heatmap_pose_adaptation_init": True,
            "required_history_pose_provider": "amb3r_vo_cache",
            "past_plan_action_bridge_only": True,
            "past_plan_action_reset_bridge": True,
        }
        stage = TrainingStageConfig.model_validate(base)
        assert stage.past_plan_action_reset_bridge is True

        with pytest.raises(ValueError, match="bridge-only action refinement"):
            TrainingStageConfig.model_validate(
                {**base, "past_plan_action_bridge_only": False}
            )
        with pytest.raises(ValueError, match="requires a Past->Plan->Action"):
            TrainingStageConfig.model_validate(
                {
                    "name": "plain",
                    "epochs": 1,
                    "past_plan_action_reset_bridge": True,
                }
            )

    def test_action_advantage_and_rollout_knobs_are_validated(self) -> None:
        from src.config_schema import LossConfig, ValidationConfig

        with pytest.raises(ValueError, match="action_advantage_reference_mse"):
            LossConfig.model_validate({"action_advantage_reference_mse": -1.0})
        with pytest.raises(ValueError, match="action_advantage_max_weight"):
            LossConfig.model_validate({"action_advantage_max_weight": 0.5})
        with pytest.raises(ValueError, match="val_rollout_batches"):
            ValidationConfig.model_validate({"val_rollout_batches": -1})

    def test_action_refine_v2_config_file_passes_schema(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.training.utils import load_config

        monkeypatch.setenv("PPA_DATA_ROOT", "/tmp/ppa-data")
        monkeypatch.setenv("PPA_AMB3R_CACHE_ROOT", "/tmp/ppa-cache")
        monkeypatch.setenv("PPA_ACTION_REFINE_OUTPUT_ROOT", "/tmp/ppa-out")
        monkeypatch.setenv("PPA_TENSORBOARD_ROOT", "/tmp/ppa-tb")
        monkeypatch.setenv("INTERNNAV_MODEL_PATH", "/tmp/internnav-model")
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "ppa_action_refine_v2_8gpu.yaml"
        )
        cfg = load_config(str(config_path))

        assert cfg["model"]["past_plan_action"]["max_delta_ratio"] == pytest.approx(
            0.05
        )
        stage = cfg["training"]["stages"][0]
        assert stage["past_plan_action_bridge_only"] is True
        assert stage["past_plan_action_reset_bridge"] is True
        loss_cfg = cfg["loss"]
        assert loss_cfg["delta_z_relative"] is True
        assert loss_cfg["action_advantage_enabled"] is True
        assert loss_cfg["preserve_weight"] == pytest.approx(2.0)
        validation_cfg = cfg["validation"]
        assert validation_cfg["save_best_metric"] == "val_rollout_endpoint_error"
        assert validation_cfg["val_rollout_batches"] == 8

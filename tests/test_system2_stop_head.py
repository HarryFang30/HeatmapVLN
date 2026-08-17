from pathlib import Path

import torch
import yaml
import pytest

from src.models.action.stop_head import StopPredictionHead
from scripts.training.validate import _select_stop_hysteresis_thresholds


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stop_head_config_freezes_original_lora_and_trains_only_head():
    config = yaml.safe_load(
        (REPO_ROOT / "configs/train_system2_panoramic_stop_head_8gpu.yaml").read_text()
    )
    trajectory = config["data"]["trajectory"]
    llm = config["model"]["llm"]
    stop_head = config["model"]["stop_head"]
    stage = config["training"]["stages"][0]

    assert trajectory["system2_stop_oversample"] == 1
    assert trajectory["system2_stop_path_radius_m"] == 3.0
    assert trajectory["system2_near_stop_hard_negative_oversample"] == 2
    assert trajectory["system2_near_stop_hard_negative_min_goal_distance_m"] == 4.0
    assert trajectory["system2_near_stop_hard_negative_max_goal_distance_m"] == 18.0
    assert trajectory["sft_include_turns"] is True
    assert llm["lora_rank"] == 32
    assert llm["lora_layer_indices"] == list(range(28))
    assert stop_head["enabled"] is True
    assert stop_head["pos_weight"] == 1.0
    assert stop_head["bce_mix"] == 1.0
    assert stop_head["add_stop_threshold"] == 0.9
    assert stop_head["veto_stop_threshold"] == 0.2
    assert stage["epochs"] == 1
    assert stage["train_lm"] is False
    assert stage["train_system2_stop_head"] is True
    assert stage["base_checkpoint_lora_only"] is True
    assert stage["trainable_modules"] == ["stop_head"]
    assert config["optim"]["stop_head_lr"] == 1.0e-4
    assert config["validation"]["enabled"] is True
    assert config["validation"]["holdout_clip_fraction"] == 0.05


def test_stop_head_launcher_checks_complete_checkpoint():
    launcher = (REPO_ROOT / "scripts/run_system2_stop_head_8gpu_mxc500.sh").read_text()

    assert "len(lora) != 224" in launcher
    assert "torch.isfinite" in launcher
    assert "trainable=stop_head" in launcher
    assert "run_stage1_s2_8gpu_mxc500_launcher.sh" in launcher
    stage_launcher = (REPO_ROOT / "scripts/run_stage1_s2_8gpu.sh").read_text()
    common = (REPO_ROOT / "scripts/stage_training_common.sh").read_text()
    assert "STAGE1_S2_MAX_CLIPS" in stage_launcher
    assert 'set_int(trajectory, "max_clips", env("MAX_CLIPS"))' in common
    assert 'multi_gpu", {})["enabled"] = visible_device_count > 1' in common


def test_eval_launcher_guards_both_stop_hysteresis_thresholds():
    launcher = (
        REPO_ROOT / "scripts/run_stage3_r2r_val_unseen_rpc_mxc500.sh"
    ).read_text()

    assert "tensors=10 add_threshold=" in launcher
    assert '"veto_threshold="' in launcher
    assert "tensors=10 threshold=" not in launcher
    assert "STAGE3_EVAL_COLLECT_STOP_FEATURES" in launcher
    assert "STAGE3_EVAL_STOP_COLLECT_FORCE_CONTINUE_NEGATIVES" in launcher
    assert "STAGE3_EVAL_STOP_COLLECT_ORACLE_PATH_FROM_START" in launcher
    assert "STAGE3_EVAL_STOP_COLLECT_BOUNDARY_PROBE_SWEEP" in launcher
    assert "STAGE3_EVAL_STOP_ORACLE_RECOVERY_ACTIONS_PER_CALL" in launcher
    assert "STAGE3_EVAL_EXPECTED_EPISODES" in launcher
    assert "STAGE3_EVAL_STOP_TEMPORAL_TRUST_MIN_MARGIN" in launcher
    assert "--system2_stop_temporal_trust_min_margin" in launcher
    assert "--system2_stop_feature_dump_dir" in launcher
    assert "--collect_system2_stop_features" in launcher
    assert "--system2_stop_collect_force_continue_negatives" in launcher
    assert "--system2_stop_collect_oracle_path_from_start" in launcher
    assert "--system2_stop_collect_boundary_probe_sweep" in launcher
    assert "--system2_stop_oracle_recovery_actions_per_call" in launcher
    assert "must use the unmodified original System2 policy" in launcher
    assert "System2 STOP DAgger feature collection is ACTIVE" in launcher

    evaluator = (REPO_ROOT / "scripts/evaluation/r2r_val_unseen.py").read_text()
    rpc_body = evaluator.split("def run_eval_rpc_panoramic(args):", 1)[1].split(
        "\ndef ", 1
    )[0]
    assert "collect_stop_features = bool(args.collect_system2_stop_features)" in rpc_body
    assert "if collect_stop_features:" in rpc_body


def test_stop_head_loss_does_not_apply_legacy_ten_x_positive_weight():
    head = StopPredictionHead(
        input_dim=4,
        hidden_dim=4,
        dropout=0.0,
        focal_gamma=0.0,
        focal_alpha=0.5,
        pos_weight=1.0,
        bce_mix=1.0,
    )
    logits = torch.zeros(2)
    targets = torch.tensor([0.0, 1.0])

    loss = head.compute_loss(logits, targets)

    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())


def test_stop_head_rejects_invalid_loss_balance():
    with pytest.raises(ValueError, match="pos_weight"):
        StopPredictionHead(input_dim=4, hidden_dim=4, pos_weight=0.0)


def test_stop_threshold_calibration_uses_asymmetric_constraints():
    thresholds = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    # TP, FP, TN, FN. The add threshold must satisfy FPR <= 1% and the 0.9
    # safety floor; the veto threshold is capped at 0.5 while retaining >=98%.
    confusion = torch.tensor(
        [
            [100, 1000, 0, 0],
            [100, 100, 900, 0],
            [99, 20, 980, 1],
            [98, 5, 995, 2],
            [95, 0, 1000, 5],
            [0, 0, 1000, 100],
        ],
        dtype=torch.float64,
    )

    result = _select_stop_hysteresis_thresholds(
        thresholds,
        confusion,
        max_add_false_positive_rate=0.01,
        min_veto_recall=0.98,
        min_add_threshold=0.9,
        max_veto_threshold=0.5,
    )

    assert result["add_stop_threshold"] == pytest.approx(0.9)
    assert result["add_false_positive_rate"] == pytest.approx(0.0)
    assert result["veto_stop_threshold"] == pytest.approx(0.5)
    assert result["veto_recall"] == pytest.approx(0.99)

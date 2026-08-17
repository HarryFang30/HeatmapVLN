import json
import sys
from pathlib import Path

import pytest
import torch
import yaml
from scripts.evaluation.preflight_stage3_rpc_eval import (
    _adapter_fingerprint,
    _file_sha256,
    validate_base_checkpoint,
    validate_stage3_checkpoint,
    validate_stage3_config,
    validate_stop_decision_adapter_checkpoint,
    validate_stop_head_checkpoint,
    validate_temporal_stop_verifier_checkpoint,
)
from scripts.evaluation.preflight_stage3_rpc_eval import (
    main as preflight_main,
)

from src.models.action import StopPredictionHead
from src.models.action.temporal_stop_verifier import (
    TEMPORAL_STOP_FEATURE_NAMES,
    TEMPORAL_STOP_FEATURE_SCHEMA,
    TemporalStopVerifier,
    TemporalStopVerifierEnsemble,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(*, dim: int = 8, adapter_hidden_dim: int = 4) -> dict:
    return {
        "data": {
            "trajectory": {
                "panoramic_vlm_input": True,
                "structured_pano_output": True,
                "trajectory_target_convention": "internnav_habitat",
            }
        },
        "model": {
            "llm": {
                "hidden_dim": dim,
                "use_lora": True,
                "lora_rank": 32,
                "lora_layer_indices": list(range(28)),
                "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "action_head": {
                "nextdit": {
                    "enabled": True,
                    "pano_latent_adapter": {
                        "enabled": True,
                        "hidden_dim": adapter_hidden_dim,
                    },
                }
            },
        },
        "training": {
            "stages": [
                {
                    "name": "stage3",
                    "strict_trainable_modules": True,
                    "requires_base_checkpoint": True,
                    "require_complete_internnav_system1": True,
                    "base_checkpoint_lora_only": True,
                    "trainable_modules": ["pano_latent_adapter"],
                }
            ]
        },
    }


def _base_state() -> dict[str, torch.Tensor]:
    state = {}
    for layer in range(28):
        for module in ("q_proj", "k_proj", "v_proj", "o_proj"):
            prefix = (
                "qwen2_5_vl.model.base_model.model.model."
                f"layers.{layer}.self_attn.{module}"
            )
            state[f"{prefix}.lora_A.default.weight"] = torch.zeros(32, 8)
            state[f"{prefix}.lora_B.default.weight"] = torch.zeros(8, 32)
    return state


def _write_base(path: Path) -> None:
    torch.save(
        {
            "stage_name": "stage1_s2_panoramic_sft",
            "epoch": 5,
            "trainable_state_dict": _base_state(),
        },
        path,
    )


def _write_stage3(path: Path, base_path: Path, *, nonfinite: bool = False) -> None:
    cfg = _config()
    cfg["runtime"] = {"base_checkpoint": str(base_path.resolve())}
    state = {
        "pano_latent_adapter.mlp.0.weight": torch.zeros(4, 8),
        "pano_latent_adapter.mlp.0.bias": torch.zeros(4),
        "pano_latent_adapter.mlp.3.weight": torch.zeros(8, 4),
        "pano_latent_adapter.mlp.3.bias": torch.zeros(8),
    }
    if nonfinite:
        state["pano_latent_adapter.mlp.3.bias"][0] = float("nan")
    torch.save(
        {
            "stage_name": "stage3",
            "epoch": 2,
            "batch": None,
            "config": cfg,
            "trainable_state_dict": state,
        },
        path,
    )


def _write_stop_decision_adapter(path: Path, base_path: Path) -> None:
    state = {}
    for layer in range(20, 28):
        for module in ("q_proj", "k_proj", "v_proj", "o_proj"):
            prefix = (
                "base_model.model.model.layers."
                f"{layer}.self_attn.{module}"
            )
            state[f"{prefix}.lora_A.weight"] = torch.zeros(8, 8)
            state[f"{prefix}.lora_B.weight"] = torch.zeros(8, 8)
    classes = ("stop", "front", "right", "back", "left", "turn")
    class_ids = list(range(20, 26))
    torch.save(
        {
            "schema": "heatmapvln-system2-stop-decision-adapter-v1",
            "adapter_name": "stop_decision",
            "adapter_state_dict": state,
            "adapter_fingerprint": _adapter_fingerprint(state),
            "adapter_config": {
                "rank": 8,
                "alpha": 16,
                "layer_indices": list(range(20, 28)),
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "dropout": 0.0,
            },
            "base_contract": {
                "checkpoint": str(base_path.resolve()),
                "checkpoint_file_sha256": _file_sha256(base_path),
                "default_adapter_name": "default",
                "default_lora_tensors": 224,
                "default_lora_fingerprint": "a" * 64,
            },
            "token_contract": {
                "schema": "heatmapvln-structured-view-token-contract-v1",
                "classes": list(classes),
                "prefix_token_ids": [10, 11],
                "class_token_ids": class_ids,
                "patterns": {
                    name: [10, 11, token_id]
                    for name, token_id in zip(classes, class_ids)
                },
            },
            "thresholds": {
                "add_stop_threshold": 0.9,
                "veto_stop_threshold": 0.2,
                "quality_passed": True,
                "quality_violations": [],
                "roc_auc": 0.9,
                "veto_reference_positive_count": 10,
                "add": {"recall": 0.6, "false_positive_rate": 0.0},
                "veto": {"recall": 1.0, "negative_rejection_rate": 0.8},
            },
            "training": {
                "holdout_scene_fraction": 0.1,
                "ranking_loss_weight": 1.0,
            },
        },
        path,
    )


def _write_stop_head(path: Path, base_path: Path, *, add_unexpected: bool = False) -> None:
    head = StopPredictionHead(input_dim=8, hidden_dim=4, dropout=0.0)
    state = {
        f"stop_head.{name}": value
        for name, value in head.state_dict().items()
    }
    if add_unexpected:
        state["qwen2_5_vl.bad"] = torch.zeros(1)
    torch.save(
        {
            "stage_name": "system2_stop_head",
            "epoch": 1,
            "batch": None,
            "config": {
                "runtime": {"base_checkpoint": str(base_path.resolve())},
                "data": {
                    "trajectory": {
                        "sft_include_turns": True,
                        "system2_stop_path_radius_m": 3.0,
                        "system2_near_stop_hard_negative_min_goal_distance_m": 4.0,
                        "system2_near_stop_hard_negative_max_goal_distance_m": 18.0,
                    }
                },
                "model": {
                    "llm": {"hidden_dim": 8},
                    "stop_head": {
                        "enabled": True,
                        "hidden_dim": 4,
                        "inference_threshold": 0.5,
                        "add_stop_threshold": 0.9,
                        "veto_stop_threshold": 0.2,
                    },
                },
                "training": {
                    "stages": [
                        {
                            "name": "system2_stop_head",
                            "train_system2_stop_head": True,
                            "base_checkpoint_lora_only": True,
                            "trainable_modules": ["stop_head"],
                        }
                    ]
                },
                "validation": {
                    "enabled": True,
                    "holdout_clip_fraction": 0.05,
                },
            },
            "trainable_state_dict": state,
            "metrics": {
                "val_stop_add_stop_threshold": 0.9,
                "val_stop_veto_stop_threshold": 0.2,
            },
        },
        path,
    )


def _write_temporal_stop_verifier(path: Path, *, wrong_feature_order: bool = False) -> None:
    static_head = StopPredictionHead(input_dim=8, hidden_dim=4, dropout=0.0)
    verifier = TemporalStopVerifier(
        feature_mean=torch.zeros(len(TEMPORAL_STOP_FEATURE_NAMES)),
        feature_scale=torch.ones(len(TEMPORAL_STOP_FEATURE_NAMES)),
        hidden_dim=4,
        dropout=0.0,
    )
    feature_names = list(TEMPORAL_STOP_FEATURE_NAMES)
    if wrong_feature_order:
        feature_names[:2] = reversed(feature_names[:2])
    torch.save(
        {
            "stage_name": "system2_temporal_stop_verifier",
            "epoch": 7,
            "config": {
                "temporal_stop_verifier": {
                    "schema": TEMPORAL_STOP_FEATURE_SCHEMA,
                    "feature_names": feature_names,
                    "hidden_dim": 4,
                    "dropout": 0.0,
                    "acceptance_threshold": 0.6,
                    "veto_only": True,
                    "requires_contiguous_zero_based_calls": True,
                },
                "source_static_stop_head": {
                    "input_dim": 8,
                    "hidden_dim": 4,
                    "dropout": 0.0,
                    "focal_gamma": 2.0,
                    "focal_alpha": 0.5,
                    "pos_weight": 1.0,
                    "bce_mix": 0.5,
                },
            },
            "trainable_state_dict": {
                f"temporal_stop_verifier.{name}": value
                for name, value in verifier.state_dict().items()
            },
            "source_static_stop_head_state_dict": {
                f"stop_head.{name}": value
                for name, value in static_head.state_dict().items()
            },
            "metrics": {"acceptance_threshold": 0.6},
            "training": {"scene_disjoint": True},
        },
        path,
    )


def _write_temporal_stop_ensemble(path: Path) -> None:
    static_head = StopPredictionHead(input_dim=8, hidden_dim=4, dropout=0.0)
    dimension = len(TEMPORAL_STOP_FEATURE_NAMES)
    ensemble = TemporalStopVerifierEnsemble(
        [
            TemporalStopVerifier(
                feature_mean=torch.full((dimension,), float(index)),
                feature_scale=torch.ones(dimension),
                hidden_dim=4,
                dropout=0.0,
            )
            for index in range(2)
        ],
        torch.tensor([0.55, 0.7]),
    )
    torch.save(
        {
            "stage_name": "system2_temporal_stop_verifier_ensemble",
            "epoch": 9,
            "config": {
                "temporal_stop_verifier": {
                    "schema": TEMPORAL_STOP_FEATURE_SCHEMA,
                    "feature_names": list(TEMPORAL_STOP_FEATURE_NAMES),
                    "architecture": "scene_fold_unanimous_ensemble",
                    "ensemble_size": 2,
                    "member_hidden_dim": 4,
                    "member_dropout": 0.0,
                    "acceptance_thresholds": [0.55, 0.7],
                    "aggregation": "unanimous",
                    "veto_only": True,
                    "requires_contiguous_zero_based_calls": True,
                },
                "source_static_stop_head": {
                    "input_dim": 8,
                    "hidden_dim": 4,
                    "dropout": 0.0,
                    "focal_gamma": 2.0,
                    "focal_alpha": 0.5,
                    "pos_weight": 1.0,
                    "bce_mix": 0.5,
                },
            },
            "trainable_state_dict": {
                f"temporal_stop_ensemble.{name}": value
                for name, value in ensemble.state_dict().items()
            },
            "source_static_stop_head_state_dict": {
                f"stop_head.{name}": value
                for name, value in static_head.state_dict().items()
            },
            "metrics": {
                "oof": {"recall": 0.9, "false_positive_rate": 0.05},
                "folds": [{"fold": 0}, {"fold": 1}],
            },
            "training": {
                "scene_disjoint": True,
                "fold_count": 2,
            },
        },
        path,
    )


def test_stage3_rpc_preflight_accepts_exact_all_layer_checkpoints(tmp_path):
    base_path = tmp_path / "base.pth"
    stage3_path = tmp_path / "stage3.pth"
    _write_base(base_path)
    _write_stage3(stage3_path, base_path)

    summary = validate_stage3_config(_config(), expected_adapter_hidden_dim=4)
    base = validate_base_checkpoint(base_path, summary)
    stage3 = validate_stage3_checkpoint(
        stage3_path,
        expected_epoch=2,
        expected_base_checkpoint=base_path,
        config_summary=summary,
    )

    assert base["lora_tensors"] == 224
    assert stage3["adapter_tensors"] == 4
    assert stage3["adapter_parameters"] == 76


def test_stage3_rpc_preflight_rejects_partial_lora_checkpoint(tmp_path):
    base_path = tmp_path / "base.pth"
    state = _base_state()
    state.pop(next(iter(state)))
    torch.save({"trainable_state_dict": state}, base_path)
    summary = validate_stage3_config(_config(), expected_adapter_hidden_dim=4)

    with pytest.raises(ValueError, match="Base LoRA checkpoint validation failed"):
        validate_base_checkpoint(base_path, summary)


def test_stage3_rpc_preflight_rejects_legacy_trajectory_targets():
    config = _config()
    config["data"]["trajectory"]["trajectory_target_convention"] = (
        "legacy_pitched_camera"
    )

    with pytest.raises(ValueError, match="trajectory_target_convention"):
        validate_stage3_config(config, expected_adapter_hidden_dim=4)


def test_stage3_rpc_preflight_rejects_nonfinite_adapter(tmp_path):
    base_path = tmp_path / "base.pth"
    stage3_path = tmp_path / "stage3.pth"
    _write_base(base_path)
    _write_stage3(stage3_path, base_path, nonfinite=True)
    summary = validate_stage3_config(_config(), expected_adapter_hidden_dim=4)

    with pytest.raises(ValueError, match="non-finite"):
        validate_stage3_checkpoint(
            stage3_path,
            expected_epoch=2,
            expected_base_checkpoint=base_path,
            config_summary=summary,
        )


def test_stage3_rpc_preflight_accepts_isolated_stop_head(tmp_path):
    base_path = tmp_path / "base.pth"
    stop_head_path = tmp_path / "stop_head.pth"
    _write_base(base_path)
    _write_stop_head(stop_head_path, base_path)

    result = validate_stop_head_checkpoint(
        stop_head_path,
        expected_base_checkpoint=base_path,
    )

    assert result["head_tensors"] == 10
    assert result["inference_threshold"] == pytest.approx(0.5)
    assert result["add_stop_threshold"] == pytest.approx(0.9)
    assert result["veto_stop_threshold"] == pytest.approx(0.2)


def test_stage3_rpc_preflight_accepts_isolated_stop_decision_adapter(tmp_path):
    base_path = tmp_path / "base.pth"
    adapter_path = tmp_path / "stop_decision.pth"
    _write_base(base_path)
    _write_stop_decision_adapter(adapter_path, base_path)

    result = validate_stop_decision_adapter_checkpoint(
        adapter_path,
        expected_base_checkpoint=base_path,
    )

    assert result["adapter_tensors"] == 64
    assert result["policy_kind"] == "add_and_veto"
    assert result["add_enabled"] is True
    assert result["add_stop_threshold"] == pytest.approx(0.9)
    assert result["veto_stop_threshold"] == pytest.approx(0.2)


def test_stage3_rpc_preflight_accepts_explicit_veto_only_adapter(tmp_path):
    base_path = tmp_path / "base.pth"
    adapter_path = tmp_path / "veto_only_stop_decision.pth"
    _write_base(base_path)
    _write_stop_decision_adapter(adapter_path, base_path)
    checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=True)
    checkpoint["policy_kind"] = "veto_only"
    checkpoint["thresholds"].update(
        {
            "policy_kind": "veto_only",
            "add_enabled": False,
            "add_stop_threshold": 1.0,
        }
    )
    checkpoint["thresholds"]["add"] = {
        "recall": 0.0,
        "false_positive_rate": 0.0,
    }
    torch.save(checkpoint, adapter_path)

    result = validate_stop_decision_adapter_checkpoint(
        adapter_path,
        expected_base_checkpoint=base_path,
    )

    assert result["policy_kind"] == "veto_only"
    assert result["add_enabled"] is False
    assert result["add_stop_threshold"] == pytest.approx(1.0)


def test_stage3_rpc_preflight_rejects_stop_decision_add_false_positives(tmp_path):
    base_path = tmp_path / "base.pth"
    adapter_path = tmp_path / "stop_decision.pth"
    _write_base(base_path)
    _write_stop_decision_adapter(adapter_path, base_path)
    checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=True)
    checkpoint["thresholds"]["add"]["false_positive_rate"] = 0.005
    torch.save(checkpoint, adapter_path)

    with pytest.raises(ValueError, match="false-positive rate must be zero"):
        validate_stop_decision_adapter_checkpoint(
            adapter_path,
            expected_base_checkpoint=base_path,
        )


def test_stage3_rpc_preflight_rejects_stop_head_with_policy_weights(tmp_path):
    base_path = tmp_path / "base.pth"
    stop_head_path = tmp_path / "stop_head.pth"
    _write_base(base_path)
    _write_stop_head(stop_head_path, base_path, add_unexpected=True)

    with pytest.raises(ValueError, match="unexpected non-head tensors"):
        validate_stop_head_checkpoint(
            stop_head_path,
            expected_base_checkpoint=base_path,
        )


def test_stage3_rpc_preflight_accepts_temporal_veto_only_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "temporal.pth"
    _write_temporal_stop_verifier(checkpoint_path)

    result = validate_temporal_stop_verifier_checkpoint(checkpoint_path)

    assert result["veto_only"] is True
    assert result["acceptance_threshold"] == pytest.approx(0.6)
    assert result["feature_dim"] == len(TEMPORAL_STOP_FEATURE_NAMES)
    assert result["static_prior_tensors"] == 10


def test_stage3_rpc_preflight_accepts_static_add_with_temporal_veto(tmp_path):
    base_path = tmp_path / "base.pth"
    stop_head_path = tmp_path / "stop_head.pth"
    temporal_path = tmp_path / "temporal.pth"
    _write_base(base_path)
    _write_stop_head(stop_head_path, base_path)
    _write_temporal_stop_verifier(temporal_path)

    assert validate_stop_head_checkpoint(
        stop_head_path,
        expected_base_checkpoint=base_path,
    )["add_stop_threshold"] == pytest.approx(0.9)
    assert validate_temporal_stop_verifier_checkpoint(temporal_path)[
        "veto_only"
    ] is True


def test_stage3_rpc_preflight_main_reports_hybrid_policy(
    tmp_path,
    monkeypatch,
    capsys,
):
    config_path = tmp_path / "config.yaml"
    base_path = tmp_path / "base.pth"
    stage3_path = tmp_path / "stage3.pth"
    stop_head_path = tmp_path / "stop_head.pth"
    temporal_path = tmp_path / "temporal.pth"
    config_path.write_text(yaml.safe_dump(_config()), encoding="utf-8")
    _write_base(base_path)
    _write_stage3(stage3_path, base_path)
    _write_stop_head(stop_head_path, base_path)
    _write_temporal_stop_verifier(temporal_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight_stage3_rpc_eval.py",
            "--config",
            str(config_path),
            "--base-checkpoint",
            str(base_path),
            "--stage3-checkpoint",
            str(stage3_path),
            "--system2-stop-head-checkpoint",
            str(stop_head_path),
            "--system2-temporal-stop-verifier-checkpoint",
            str(temporal_path),
            "--expected-adapter-hidden-dim",
            "4",
        ],
    )

    assert preflight_main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["system2_stop_policy_mode"] == (
        "hybrid_static_add_temporal_veto"
    )


def test_stage3_rpc_preflight_accepts_unanimous_temporal_ensemble(tmp_path):
    checkpoint_path = tmp_path / "temporal_ensemble.pth"
    _write_temporal_stop_ensemble(checkpoint_path)

    result = validate_temporal_stop_verifier_checkpoint(checkpoint_path)

    assert result["veto_only"] is True
    assert result["architecture"] == "scene_fold_unanimous_ensemble"
    assert result["aggregation"] == "unanimous"
    assert result["ensemble_size"] == 2
    assert result["acceptance_thresholds"] == pytest.approx([0.55, 0.7])


def test_stage3_rpc_preflight_rejects_temporal_feature_order_drift(tmp_path):
    checkpoint_path = tmp_path / "temporal.pth"
    _write_temporal_stop_verifier(checkpoint_path, wrong_feature_order=True)

    with pytest.raises(ValueError, match="feature names"):
        validate_temporal_stop_verifier_checkpoint(checkpoint_path)


def test_stage3_rpc_launcher_routes_habitat_to_configured_sim_gpu():
    launcher = (
        PROJECT_ROOT / "scripts/run_stage3_r2r_val_unseen_rpc_mxc500.sh"
    ).read_text(encoding="utf-8")

    assert 'STAGE3_EVAL_SIM_GPU="${STAGE3_EVAL_SIM_GPU:-$STAGE3_EVAL_MODEL_GPU}"' in launcher
    assert 'CUDA_VISIBLE_DEVICES="$STAGE3_EVAL_SIM_GPU"' in launcher
    assert "CUDA_VISIBLE_DEVICES=0" not in launcher
    assert "--sim_gpu_id 0" in launcher

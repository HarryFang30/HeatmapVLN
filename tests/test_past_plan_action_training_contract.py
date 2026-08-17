from __future__ import annotations

import logging

import pytest
import torch
from torch import nn

from scripts.training.checkpoint import CheckpointManager, load_checkpoint_for_resume
from scripts.training.distributed import _get_supported_trainable_sync_modules
from scripts.training.model_builder import _is_allowed_trainable_name
from scripts.training.optimizer import build_optimizer
from src.config_schema import normalize_config
from src.models.future_trajectory_objective import (
    future_tube_metrics_from_statistics,
    future_tube_sufficient_statistics,
)
from src.models.past_plan_action import PastPlanActionChain
from src.models.past_plan_action_training import configure_past_plan_action_stage


class _Coarse(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_history = nn.Linear(8, 4)
        self.proj_traj = nn.Linear(4, 4)
        self.pos_embed = nn.Parameter(torch.zeros(3, 4))
        self.pos_embed_shadow = nn.Parameter(torch.zeros(3, 4))
        self.self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(4, 2, batch_first=True), 1
        )
        self.heatmap_head = nn.Linear(4, 1)
        self.vis_head = nn.Linear(4, 4)
        self.unrelated = nn.Linear(4, 4)


class _Past(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.coarse = _Coarse()
        self.fine = nn.Sequential(nn.Linear(4, 4))
        self.vit_dpt_fusion = nn.Linear(4, 4)


class _Pipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.past_plan_action = PastPlanActionChain()
        self.heatmap_vln = _Past()
        self.nextdit_action_head = nn.Sequential(nn.Linear(4, 4))
        self.nextdit_action_head.cond_projector = nn.Linear(8, 4)


def _configure(model: _Pipeline, stage: str) -> None:
    configure_past_plan_action_stage(
        stage=stage,
        chain=model.past_plan_action,
        past_head=model.heatmap_vln,
        native_action_head=model.nextdit_action_head,
        native_cond_projector=model.nextdit_action_head.cond_projector,
    )


def _minimal_config(*, enabled: bool) -> dict:
    cfg = {
        "data": {
            "root": "/tmp/data",
            "image_size": [256, 256],
            "init_hm_size": [64, 64],
            "dataset_type": "trajectory",
            "trajectory": {
                "predict_horizon": 32,
                "enable_trajectory_augmentation": not enabled,
                "trajectory_target_convention": (
                    "internnav_habitat" if enabled else "legacy_pitched_camera"
                ),
                "future_heatmap": {"enabled": enabled},
            },
        },
        "model": {
            "heatmap": {
                "enable": True,
                "c_fused": 256,
                "decoder_mode": "legacy",
                "trajectory": {"enable": True},
            },
            "action_head": {
                "enable": True,
                "nextdit": {
                    "enabled": True,
                    "latent_emb_size": 768,
                    "n_query": 4,
                    "predict_steps": 32,
                },
            },
        },
        "training": {
            "stages": [
                {
                    "name": "legacy",
                    "epochs": 1,
                    "train_action": True,
                    "trainable_modules": ["nextdit_action_head"],
                }
            ]
        },
        "log": {"out_dir": "/tmp/out"},
    }
    if enabled:
        cfg["loss"] = {
            "future_heatmap": {},
            "past_plan_action": {},
        }
        cfg["model"]["action_head"]["nextdit"]["past_plan_action"] = {
            "enabled": True,
            "stage": "stage2_joint",
        }
        cfg["training"]["stages"] = [
            {
                "name": "stage2_joint",
                "epochs": 1,
                "train_history": True,
                "train_future": True,
                "train_action": True,
                "past_plan_action_stage": "stage2_joint",
                "trajectory_sequence_mode": "first_only",
                "strict_trainable_modules": True,
                "trainable_modules": [
                    "future_heatmap_head",
                    "heatmap_memory_and_decoder",
                    "past_plan_bridge",
                ],
            }
        ]
    return cfg


def test_disabled_schema_does_not_materialize_ppa() -> None:
    normalized = normalize_config(_minimal_config(enabled=False))
    nextdit = normalized["model"]["action_head"]["nextdit"]
    assert "past_plan_action" not in nextdit


def test_enabled_schema_accepts_exact_stage2_contract() -> None:
    normalized = normalize_config(_minimal_config(enabled=True))
    ppa = normalized["model"]["action_head"]["nextdit"]["past_plan_action"]
    assert ppa["enabled"] is True
    assert ppa["plan_dim"] == 768
    assert normalized["loss"]["past_plan_action"]["preserve"] == 0.5


def test_schema_rejects_future_action_augmentation_mismatch() -> None:
    cfg = _minimal_config(enabled=True)
    cfg["data"]["trajectory"]["enable_trajectory_augmentation"] = True
    with pytest.raises(ValueError, match="augmentation=false"):
        normalize_config(cfg)


def test_pos_embed_prefix_does_not_allow_shadow_parameter() -> None:
    scopes = {"heatmap_memory_and_decoder"}
    assert _is_allowed_trainable_name("heatmap_vln.coarse.pos_embed", scopes)
    assert not _is_allowed_trainable_name(
        "heatmap_vln.coarse.pos_embed_shadow", scopes
    )


def test_main_optimizer_and_ddp_scopes_cover_each_trainable_once() -> None:
    model = _Pipeline()
    _configure(model, "stage2_joint")
    stage = {
        "past_plan_action_stage": "stage2_joint",
        "trainable_modules": [
            "future_heatmap_head",
            "past_plan_bridge",
            "heatmap_memory_and_decoder",
        ],
    }
    optimizer = build_optimizer(
        model,
        {
            "optim": {
                "past_plan_action_future_lr": 1e-4,
                "past_plan_action_bridge_lr": 2e-5,
                "past_plan_action_shared_map_lr": 2e-5,
                "weight_decay": 0.01,
            }
        },
        stage,
    )
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(ids) == len(set(ids))
    assert set(ids) == expected
    assert [group["family"] for group in optimizer.param_groups] == [
        "future",
        "bridge",
        "shared_map",
    ]
    sync = _get_supported_trainable_sync_modules(model, stage)
    assert [name for name, _ in sync] == [
        "past_plan_action.future_head",
        "past_plan_action.bridge",
        "heatmap_vln.coarse",
        "heatmap_vln.fine",
    ]


def test_checkpoint_is_hash_free_and_refuses_stage_scope_resume(tmp_path) -> None:
    model = _Pipeline()
    _configure(model, "stage1_map_pretrain")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    cfg = _minimal_config(enabled=True)
    cfg["model"]["action_head"]["nextdit"]["past_plan_action"]["stage"] = (
        "stage1_map_pretrain"
    )
    manager = CheckpointManager(str(tmp_path))
    path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        stage_idx=0,
        stage_name="stage1_map_pretrain",
        metrics={"val_loss": 1.0},
        cfg=cfg,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    contract = checkpoint["past_plan_action_contract"]
    assert contract["checkpoint_digest_enforced"] is False
    assert contract["file_lock_used"] is False
    assert not any("sha256" in key.lower() for key in contract)

    _configure(model, "stage2_joint")
    with pytest.raises(RuntimeError, match="Stage-1.*Stage-2 warm start"):
        load_checkpoint_for_resume(str(path), model)


def test_future_tube_statistics_are_additive_across_ranks() -> None:
    generator = torch.Generator().manual_seed(7)
    pred_logits = torch.randn(2, 4, 4, generator=generator)
    pred_heatmaps = torch.rand(2, 4, 4, 64, 64, generator=generator)
    gt_heatmaps = torch.zeros_like(pred_heatmaps)
    gt_heatmaps[0, :, 0, 20:24, 20:24] = 1.0
    gt_heatmaps[1, :, 1, 30:34, 30:34] = 1.0
    gt_visibility = (gt_heatmaps.amax(dim=(-2, -1)) > 0).float()
    time_mask = torch.ones(2, 4, dtype=torch.bool)
    kwargs = {
        "pred_visibility_logits": pred_logits,
        "pred_heatmaps": pred_heatmaps,
        "gt_visibility": gt_visibility,
        "gt_heatmaps": gt_heatmaps,
        "future_time_mask": time_mask,
    }
    joint = future_tube_sufficient_statistics(**kwargs)
    split = sum(
        (
            future_tube_sufficient_statistics(
                **{key: value[index : index + 1] for key, value in kwargs.items()}
            )
            for index in range(2)
        ),
        torch.zeros_like(joint),
    )
    assert torch.allclose(joint, split)
    joint_metrics = future_tube_metrics_from_statistics(joint)
    split_metrics = future_tube_metrics_from_statistics(split)
    assert joint_metrics.soft_iou == pytest.approx(split_metrics.soft_iou, abs=1e-9)
    assert joint_metrics.topk_support_recall == pytest.approx(
        split_metrics.topk_support_recall, abs=1e-9
    )
    assert joint_metrics.visibility_f1 == split_metrics.visibility_f1
    assert joint_metrics.per_view_support == split_metrics.per_view_support

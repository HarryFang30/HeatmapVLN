from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from scripts.training.checkpoint import CheckpointManager, load_checkpoint_for_resume
from scripts.training.ema import EMAModel
from scripts.training.model_builder import set_trainable_modules
from scripts.training.optimizer import build_optimizer
from scripts.training.pose_adaptation import (
    EXPECTED_COMPLETE_HEATMAP_HEAD_TENSORS,
    EXPECTED_POSE_ADAPTATION_TENSORS,
    POSE_ADAPTATION_PREFIXES,
    complete_heatmap_head_state,
    load_pose_adaptation_initialization,
    assert_required_history_pose_provider,
)
from scripts.training.train_loop import _apply_bridge_only_train_mode
from scripts.training.validate import _HeatmapJointMetricAccumulator
from src.config_schema import TrainingStageConfig
from src.models.heatmap.trajectory_attention import TrajectoryGuidedAttention


class _TensorFamily(nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.values = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(index))) for index in range(count)]
        )


class _Heatmap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vit_dpt_fusion = _TensorFamily(12)
        self.vit_panorama_conditioner = _TensorFamily(12)
        self.coarse_panorama_conditioner = _TensorFamily(12)
        self.coarse = TrajectoryGuidedAttention(
            c_llm=8,
            c_fused=4,
            num_freqs=1,
            d_attn=8,
            num_heads=2,
            num_layers=2,
        )
        self.fine = _TensorFamily(6)
        self.pose_free_matcher = None

    def trainable_head_modules(self):
        return (
            self.vit_dpt_fusion,
            self.vit_panorama_conditioner,
            self.coarse_panorama_conditioner,
            self.coarse,
            self.fine,
        )


class _Pipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heatmap_vln = _Heatmap()


def _stage() -> dict:
    return {
        "name": "amb3r_pose_adapt",
        "trainable_modules": ["heatmap_vln"],
        "strict_trainable_modules": True,
        "train_action": False,
        "heatmap_pose_adaptation_init": True,
        "heatmap_trainable_parameter_prefixes": list(POSE_ADAPTATION_PREFIXES),
        "required_history_pose_provider": "amb3r_vo_cache",
    }


def _optimizer(model: nn.Module):
    return build_optimizer(
        model,
        {
            "optim": {
                "heatmap_lr": 2e-5,
                "heatmap_coarse_lr": 2e-5,
                "heatmap_proj_traj_lr": 1e-4,
                "weight_decay": 0.01,
            }
        },
        _stage(),
    )


def test_schema_rejects_any_scope_broader_than_four_prefixes() -> None:
    TrainingStageConfig(epochs=2, **_stage())
    bad = _stage()
    bad["heatmap_trainable_parameter_prefixes"] = list(POSE_ADAPTATION_PREFIXES[:3])
    try:
        TrainingStageConfig(epochs=2, **bad)
    except ValueError as exc:
        assert "four audited" in str(exc)
    else:
        raise AssertionError("three-prefix adaptation scope should be rejected")

    assert_required_history_pose_provider(
        {"history_pose_provider": ["amb3r_vo_cache", "amb3r_vo_cache"]},
        _stage(),
    )
    try:
        assert_required_history_pose_provider(
            {"history_pose_provider": ["amb3r_vo_cache", "gt"]}, _stage()
        )
    except RuntimeError as exc:
        assert "100% provider" in str(exc)
    else:
        raise AssertionError("mixed GT/AMB3R batch should be rejected")


def test_exact_34_tensor_scope_lrs_and_train_modes() -> None:
    model = _Pipeline()
    set_trainable_modules(model, _stage(), logging.getLogger("test"))
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    assert len(trainable) == EXPECTED_POSE_ADAPTATION_TENSORS
    assert all(name.startswith(POSE_ADAPTATION_PREFIXES) for name in trainable)
    assert not any(value.requires_grad for value in model.heatmap_vln.fine.parameters())
    assert not any(
        value.requires_grad for value in model.heatmap_vln.coarse.proj_history.parameters()
    )

    optimizer = _optimizer(model)
    lrs_by_name = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert lrs_by_name["heatmap_proj_traj_decay"] == 1e-4
    assert lrs_by_name["heatmap_proj_traj_no_decay"] == 1e-4
    assert lrs_by_name["heatmap_pose_adaptation_rest_decay"] == 2e-5
    assert lrs_by_name["heatmap_pose_adaptation_rest_no_decay"] == 2e-5

    model.train()
    _apply_bridge_only_train_mode(model, _stage(), logging.getLogger("test"))
    assert not model.heatmap_vln.training
    assert model.heatmap_vln.coarse.self_attn.training
    assert model.heatmap_vln.coarse.proj_traj.training
    assert model.heatmap_vln.coarse.vis_head.training
    assert model.heatmap_vln.coarse.heatmap_head.training
    assert not model.heatmap_vln.coarse.proj_history.training
    assert not model.heatmap_vln.fine.training


def test_selective_checkpoint_is_full_79_but_resume_uses_online_34(tmp_path: Path) -> None:
    model = _Pipeline()
    set_trainable_modules(model, _stage(), logging.getLogger("test"))
    optimizer = _optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    ema = EMAModel(model)
    with torch.no_grad():
        model.heatmap_vln.coarse.proj_traj.weight.fill_(3.0)
        model.heatmap_vln.fine.values[0].fill_(8.0)
        ema.shadow["heatmap_vln.coarse.proj_traj.weight"].fill_(2.0)

    manager = CheckpointManager(str(tmp_path))
    manager.configure_best_metric("val_heatmap_joint_pck4", "max")
    path = manager.save(
        model,
        optimizer,
        scheduler,
        epoch=1,
        stage_idx=0,
        stage_name="amb3r_pose_adapt",
        metrics={"val_loss": 1.0, "val_heatmap_joint_pck4": 0.7},
        cfg={"training": {"stages": [_stage()]}},
        is_best=True,
        ema=ema,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert len(payload["trainable_state_dict"]) == EXPECTED_COMPLETE_HEATMAP_HEAD_TENSORS
    assert len(payload["online_trainable_state_dict"]) == EXPECTED_POSE_ADAPTATION_TENSORS
    assert payload["weight_semantics"]["trainable_state_dict"] == (
        "ema_trainable_plus_frozen_heatmap_head"
    )

    # train.py performs one metadata-only resume pass before the lazy Head is
    # constructed, then a full restore after optimizer/EMA construction.
    lazy = nn.Module()
    metadata = load_checkpoint_for_resume(str(path), lazy, metadata_only=True)
    assert metadata["epoch"] == 1

    fresh = _Pipeline()
    report = load_pose_adaptation_initialization(fresh, path)
    assert report["loaded_tensor_count"] == EXPECTED_COMPLETE_HEATMAP_HEAD_TENSORS
    torch.testing.assert_close(
        fresh.heatmap_vln.coarse.proj_traj.weight,
        torch.full_like(fresh.heatmap_vln.coarse.proj_traj.weight, 2.0),
    )

    resumed = _Pipeline()
    with torch.no_grad():
        resumed.heatmap_vln.fine.values[0].fill_(-8.0)
    set_trainable_modules(resumed, _stage(), logging.getLogger("test"))
    resumed_optimizer = _optimizer(resumed)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
        resumed_optimizer, lambda _step: 1.0
    )
    resumed_ema = EMAModel(resumed)
    load_checkpoint_for_resume(
        str(path),
        resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        ema=resumed_ema,
    )
    torch.testing.assert_close(
        resumed.heatmap_vln.coarse.proj_traj.weight,
        torch.full_like(resumed.heatmap_vln.coarse.proj_traj.weight, 3.0),
    )
    torch.testing.assert_close(
        resumed_ema.shadow["heatmap_vln.coarse.proj_traj.weight"],
        torch.full_like(resumed.heatmap_vln.coarse.proj_traj.weight, 2.0),
    )
    torch.testing.assert_close(
        resumed.heatmap_vln.fine.values[0],
        torch.tensor(8.0),
    )


def test_real_head_checkpoint_is_exact_79_parameters_and_preserves_buffers(
    tmp_path: Path,
) -> None:
    from src.models.heatmap.single_view_heatmap_decoder import (
        SingleViewFourDirectionHeatmapHead,
    )

    class _RealPipeline(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heatmap_vln = SingleViewFourDirectionHeatmapHead(
                c_vit=4,
                c_merged=8,
                c_fused=8,
                # Production Head has four ViT fusion inputs; the exact 79
                # learned-tensor contract includes all four projections.
                vit_layer_indices=(0, 1, 2, 3),
                trajectory_num_freqs=1,
                trajectory_num_heads=2,
                trajectory_num_layers=2,
            )

    source = _RealPipeline()
    full_module_state = source.heatmap_vln.state_dict()
    learned_head_state = complete_heatmap_head_state(source)
    assert len(full_module_state) == 81
    assert len(dict(source.heatmap_vln.named_parameters())) == 79
    assert len(learned_head_state) == EXPECTED_COMPLETE_HEATMAP_HEAD_TENSORS
    assert set(full_module_state) - {
        name.removeprefix("heatmap_vln.") for name in learned_head_state
    } == {
        "vit_panorama_conditioner.direction_angles_degrees",
        "coarse_panorama_conditioner.direction_angles_degrees",
    }

    with torch.no_grad():
        source.heatmap_vln.coarse.proj_traj.weight.fill_(7.0)
    checkpoint_state = complete_heatmap_head_state(source)
    checkpoint_path = tmp_path / "real_head_79.pth"
    torch.save(
        {
            "trainable_state_dict": checkpoint_state,
            "weight_semantics": {"trainable_state_dict": "deployment"},
        },
        checkpoint_path,
    )

    destination = _RealPipeline()
    fixed_buffers_before = {
        name: value.detach().clone()
        for name, value in destination.heatmap_vln.named_buffers()
        if name.endswith("direction_angles_degrees")
    }
    report = load_pose_adaptation_initialization(destination, checkpoint_path)
    assert report["loaded_tensor_count"] == EXPECTED_COMPLETE_HEATMAP_HEAD_TENSORS
    torch.testing.assert_close(
        destination.heatmap_vln.coarse.proj_traj.weight,
        torch.full_like(destination.heatmap_vln.coarse.proj_traj.weight, 7.0),
    )
    fixed_buffers_after = {
        name: value.detach().clone()
        for name, value in destination.heatmap_vln.named_buffers()
        if name.endswith("direction_angles_degrees")
    }
    assert fixed_buffers_after.keys() == fixed_buffers_before.keys()
    for name in fixed_buffers_before:
        assert torch.equal(fixed_buffers_after[name], fixed_buffers_before[name])


def test_pose_adaptation_single_view_pipeline_keeps_head_graph_and_exact_grad_scope() -> None:
    from src.models.heatmap.native_single_view_feature_extractor import (
        NativeSingleViewFeatures,
    )
    from src.models.heatmap.single_view_heatmap_decoder import (
        SingleViewFourDirectionHeatmapHead,
    )
    from src.models.pipeline import VLNPipeline

    head = SingleViewFourDirectionHeatmapHead(
        c_vit=4,
        c_merged=8,
        c_fused=8,
        vit_layer_indices=(0,),
        trajectory_num_freqs=1,
        trajectory_num_heads=2,
        trajectory_num_layers=2,
    )
    head.requires_grad_(False)
    for module_name in ("proj_traj", "self_attn", "vis_head", "heatmap_head"):
        getattr(head.coarse, module_name).requires_grad_(True)
    head.eval()
    for module_name in ("proj_traj", "self_attn", "vis_head", "heatmap_head"):
        getattr(head.coarse, module_name).train()

    class _Extractor:
        def __init__(self) -> None:
            self._visual = nn.Linear(1, 1).requires_grad_(False)

        def extract_from_pixels(self, **_kwargs):
            return NativeSingleViewFeatures(
                current_vit={0: torch.randn(1, 4, 16, 16)},
                current_merged=torch.randn(1, 8, 8, 8),
                history_vit={0: torch.randn(1, 2, 4, 16, 16)},
                history_merged=torch.randn(1, 2, 8, 8, 8),
                history_queries=torch.randn(1, 2, 8),
                history_mask=torch.ones(1, 2, dtype=torch.bool),
            )

    pipeline = VLNPipeline.__new__(VLNPipeline)
    nn.Module.__init__(pipeline)
    pipeline.heatmap_vln = head
    pipeline.single_view_heatmap_extractor = _Extractor()
    output = pipeline._forward_frozen_single_view_heatmap(
        inputs={
            "pixel_values": torch.zeros(1),
            "image_grid_thw": torch.zeros(1, 3, dtype=torch.long),
        },
        num_histories=[2],
        history_rel_poses=torch.tensor(
            [[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]]
        ),
        explicit_history_mask=None,
    )
    assert output["heatmap_logits"].requires_grad
    assert output["visibility"].requires_grad
    (output["heatmap_logits"].mean() + output["visibility"].mean()).backward()
    selected = [
        (name, parameter)
        for name, parameter in head.named_parameters()
        if parameter.requires_grad
    ]
    assert len(selected) == EXPECTED_POSE_ADAPTATION_TENSORS
    assert not [name for name, parameter in selected if parameter.grad is None]
    assert all(parameter.grad is None for parameter in head.fine.parameters())
    assert all(parameter.grad is None for parameter in head.vit_dpt_fusion.parameters())
    assert all(parameter.grad is None for parameter in head.coarse.proj_history.parameters())
    assert all(
        parameter.grad is None
        for parameter in head.vit_panorama_conditioner.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in head.coarse_panorama_conditioner.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in pipeline.single_view_heatmap_extractor._visual.parameters()
    )


def test_per_view_pck4_is_history_masked() -> None:
    metric = _HeatmapJointMetricAccumulator(
        heatmap_size=(16, 16),
        device=torch.device("cpu"),
    )
    gt_hm = torch.zeros(5, 4, 16, 16)
    pred_hm = torch.zeros_like(gt_hm)
    gt_vis = torch.zeros(5, 4)
    pred_vis = torch.full_like(gt_vis, -10.0)
    for view in range(4):
        gt_hm[view, view, 4, 4] = 1
        pred_hm[view, view, 5, 5] = 1
        gt_vis[view, view] = 1
        pred_vis[view, view] = 10
    # A deliberately wrong padded row must not affect any count.
    gt_hm[4, 0, 2, 2] = 1
    gt_vis[4, 0] = 1
    metric.update(
        pred_visibility_logits=pred_vis,
        pred_heatmaps=pred_hm,
        gt_visibility=gt_vis,
        gt_heatmaps=gt_hm,
        history_mask=torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool),
    )
    values = metric.compute()
    for view in ("front", "right", "back", "left"):
        assert values[f"val_heatmap_{view}_pck4"] == 1.0
        assert values[f"val_heatmap_{view}_count"] == 1.0

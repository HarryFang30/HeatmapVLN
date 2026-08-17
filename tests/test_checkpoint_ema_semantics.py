from __future__ import annotations

import torch
import torch.nn as nn

from scripts.training.checkpoint import (
    CheckpointManager,
    load_checkpoint_for_resume,
)
from scripts.training.ema import EMAModel
from scripts.training.utils import assert_complete_lora_checkpoint_match


class _TinyModel(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([value], dtype=torch.float32))


class _DependentHeatmapModel(nn.Module):
    def __init__(
        self,
        *,
        lora_value: float,
        heatmap_value: float,
    ) -> None:
        super().__init__()
        self.qwen2_5_vl = nn.Module()
        self.qwen2_5_vl.block = nn.Module()
        self.qwen2_5_vl.block.lora_A = nn.Parameter(
            torch.tensor([lora_value], dtype=torch.float32),
            requires_grad=False,
        )
        self.qwen2_5_vl.block.lora_B = nn.Parameter(
            torch.tensor([lora_value + 1.0], dtype=torch.float32),
            requires_grad=False,
        )
        # Mirror the physical aliases present in VLNPipeline.  The deployment
        # checkpoint must still contain exactly two LoRA tensors.
        self.vlm_backbone = self.qwen2_5_vl
        self.heatmap_vln = nn.Linear(1, 1)
        self.heatmap_vln.weight.data.fill_(heatmap_value)
        self.heatmap_vln.bias.data.fill_(heatmap_value + 1.0)


def _optimizer_and_scheduler(model: nn.Module):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    return optimizer, scheduler


def _dependent_cfg(
    *,
    requires_base_checkpoint: bool = True,
    deployment_include_frozen_lora: bool | None = None,
) -> dict:
    stage = {
        "name": "heatmap",
        "requires_base_checkpoint": requires_base_checkpoint,
        "merge_frozen_lora": False,
    }
    if deployment_include_frozen_lora is not None:
        stage["deployment_include_frozen_lora"] = (
            deployment_include_frozen_lora
        )
    return {
        "training": {"stages": [stage]},
        "runtime": {"base_checkpoint": "/models/base.pth"},
    }


def test_checkpoint_keeps_deployment_ema_and_optimizer_matched_online_weights(
    tmp_path,
) -> None:
    model = _TinyModel(3.0)
    optimizer, scheduler = _optimizer_and_scheduler(model)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.weight.data.fill_(3.0)

    ema = EMAModel(model, decay=0.9, warmup_steps=10)
    ema.shadow["weight"].fill_(2.0)
    ema.step_count = 7

    manager = CheckpointManager(str(tmp_path))
    manager.configure_best_metric("val_heatmap_joint_pck8", "max")
    assert manager.is_better(0.42)
    path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=4,
        stage_idx=0,
        stage_name="heatmap",
        metrics={"val_loss": 1.25, "val_heatmap_joint_pck8": 0.42},
        cfg={},
        is_best=True,
        ema=ema,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    torch.testing.assert_close(
        checkpoint["trainable_state_dict"]["weight"],
        torch.tensor([2.0]),
    )
    torch.testing.assert_close(
        checkpoint["online_trainable_state_dict"]["weight"],
        torch.tensor([3.0]),
    )
    assert checkpoint["best_val_loss"] == 1.25
    assert checkpoint["best_metric_name"] == "val_heatmap_joint_pck8"
    assert checkpoint["best_metric_mode"] == "max"
    assert checkpoint["best_metric_value"] == 0.42

    resumed = _TinyModel(-1.0)
    resumed_optimizer, resumed_scheduler = _optimizer_and_scheduler(resumed)
    resumed_ema = EMAModel(resumed, decay=0.5, warmup_steps=1)
    load_checkpoint_for_resume(
        str(path),
        resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        ema=resumed_ema,
    )

    torch.testing.assert_close(resumed.weight, torch.tensor([3.0]))
    torch.testing.assert_close(resumed_ema.shadow["weight"], torch.tensor([2.0]))
    assert resumed_ema.step_count == 7
    assert resumed_ema.target_decay == 0.9
    assert resumed_ema.warmup_steps == 10


def test_base_dependent_checkpoint_is_self_contained_but_resume_stays_online(
    tmp_path,
) -> None:
    model = _DependentHeatmapModel(lora_value=7.0, heatmap_value=3.0)
    optimizer, scheduler = _optimizer_and_scheduler(model)
    ema = EMAModel(model, decay=0.9, warmup_steps=10)
    ema.shadow["heatmap_vln.weight"].fill_(2.0)
    ema.shadow["heatmap_vln.bias"].fill_(2.5)

    manager = CheckpointManager(str(tmp_path))
    manager.configure_best_metric("val_heatmap_joint_pck8", "max")
    path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        stage_idx=0,
        stage_name="heatmap",
        metrics={"val_loss": 1.0, "val_heatmap_joint_pck8": 0.4},
        cfg=_dependent_cfg(),
        is_best=True,
        ema=ema,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    deployment = checkpoint["trainable_state_dict"]
    online = checkpoint["online_trainable_state_dict"]

    assert set(deployment) == {
        "qwen2_5_vl.block.lora_A",
        "qwen2_5_vl.block.lora_B",
        "heatmap_vln.weight",
        "heatmap_vln.bias",
    }
    assert set(online) == {"heatmap_vln.weight", "heatmap_vln.bias"}
    torch.testing.assert_close(
        deployment["qwen2_5_vl.block.lora_A"],
        torch.tensor([7.0]),
    )
    torch.testing.assert_close(
        deployment["qwen2_5_vl.block.lora_B"],
        torch.tensor([8.0]),
    )
    torch.testing.assert_close(
        deployment["heatmap_vln.weight"],
        torch.tensor([[2.0]]),
    )
    torch.testing.assert_close(
        online["heatmap_vln.weight"],
        torch.tensor([[3.0]]),
    )
    assert assert_complete_lora_checkpoint_match(model, deployment) == 2
    assert checkpoint["weight_semantics"]["trainable_state_dict"] == (
        "ema_trainable_plus_frozen_lora"
    )
    assert checkpoint["deployment_state_manifest"] == {
        "requires_base_checkpoint": True,
        "base_checkpoint": "/models/base.pth",
        "included_frozen_lora": True,
        "frozen_lora_tensor_count": 2,
        "deployment_trainable_tensor_count": 2,
        "deployment_tensor_count": 4,
        "online_trainable_tensor_count": 2,
    }

    # Training resume must not overwrite the already-loaded base LoRA with the
    # deployment copy; it restores only optimizer-matched online heatmap state.
    resumed = _DependentHeatmapModel(lora_value=99.0, heatmap_value=-1.0)
    resumed_optimizer, resumed_scheduler = _optimizer_and_scheduler(resumed)
    resumed_ema = EMAModel(resumed)
    load_checkpoint_for_resume(
        str(path),
        resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        ema=resumed_ema,
    )
    torch.testing.assert_close(
        resumed.qwen2_5_vl.block.lora_A,
        torch.tensor([99.0]),
    )
    torch.testing.assert_close(
        resumed.heatmap_vln.weight,
        torch.tensor([[3.0]]),
    )
    torch.testing.assert_close(
        resumed_ema.shadow["heatmap_vln.weight"],
        torch.tensor([[2.0]]),
    )


def test_non_dependent_or_explicit_opt_out_keeps_legacy_deployment_scope(
    tmp_path,
) -> None:
    for name, cfg in (
        ("independent", _dependent_cfg(requires_base_checkpoint=False)),
        (
            "opt_out",
            _dependent_cfg(deployment_include_frozen_lora=False),
        ),
    ):
        out_dir = tmp_path / name
        model = _DependentHeatmapModel(
            lora_value=7.0,
            heatmap_value=3.0,
        )
        optimizer, scheduler = _optimizer_and_scheduler(model)
        ema = EMAModel(model)
        checkpoint_path = CheckpointManager(str(out_dir)).save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            stage_idx=0,
            stage_name="heatmap",
            metrics={"val_loss": 1.0},
            cfg=cfg,
            ema=ema,
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        assert set(checkpoint["trainable_state_dict"]) == {
            "heatmap_vln.weight",
            "heatmap_vln.bias",
        }
        assert "deployment_state_manifest" not in checkpoint


def test_best_only_writes_incumbent_without_latest_or_epoch_artifacts(
    tmp_path,
) -> None:
    model = _DependentHeatmapModel(lora_value=7.0, heatmap_value=3.0)
    optimizer, scheduler = _optimizer_and_scheduler(model)
    ema = EMAModel(model)
    manager = CheckpointManager(str(tmp_path))
    manager.configure_best_metric("val_heatmap_joint_pck8", "max")

    path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        stage_idx=0,
        stage_name="heatmap",
        metrics={"val_loss": 1.0, "val_heatmap_joint_pck8": 0.4},
        cfg=_dependent_cfg(),
        is_best=True,
        ema=ema,
        best_only=True,
        extra_state={"checkpoint_kind": "pre_training_baseline"},
    )

    assert path == tmp_path / "best.pth"
    assert path.exists()
    assert not (tmp_path / "latest.pth").exists()
    assert not (tmp_path / "epoch_000.pth").exists()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_kind"] == "pre_training_baseline"
    assert len(checkpoint["trainable_state_dict"]) == 4
    assert len(checkpoint["online_trainable_state_dict"]) == 2


def test_legacy_checkpoint_initializes_ema_from_loaded_weights(tmp_path) -> None:
    path = tmp_path / "legacy.pth"
    torch.save(
        {
            "epoch": 2,
            "trainable_state_dict": {"weight": torch.tensor([5.0])},
        },
        path,
    )
    model = _TinyModel(-1.0)
    ema = EMAModel(model)

    load_checkpoint_for_resume(str(path), model, ema=ema)

    torch.testing.assert_close(model.weight, torch.tensor([5.0]))
    torch.testing.assert_close(ema.shadow["weight"], torch.tensor([5.0]))
    assert ema.step_count == 0

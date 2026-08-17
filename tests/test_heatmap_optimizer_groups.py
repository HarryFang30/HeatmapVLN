import torch
from torch import nn

from scripts.training.optimizer import build_optimizer


class _Coarse(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = nn.Linear(2, 2)
        self.vis_head = nn.Linear(2, 4)


class _Heatmap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pose_free_matcher = None
        self.vit_dpt_fusion = nn.Linear(2, 2)
        self.llm_dpt_fusion = nn.Linear(2, 2)
        self.coarse = _Coarse()
        self.fine = nn.Linear(2, 2)


class _Pipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heatmap_vln = _Heatmap()


def _group_lrs(optimizer):
    return {
        group["name"]: group["lr"]
        for group in optimizer.param_groups
    }


def test_heatmap_optimizer_supports_per_module_learning_rates():
    optimizer = build_optimizer(
        _Pipeline(),
        cfg={
            "optim": {
                "heatmap_lr": 3e-5,
                "heatmap_vit_lr": 5e-5,
                "heatmap_fine_lr": 1e-4,
                "heatmap_llm_lr": 2e-5,
                "heatmap_coarse_lr": 2e-5,
                "vis_head_lr": 2e-5,
                "weight_decay": 1e-2,
            }
        },
        stage_cfg={"trainable_modules": ["heatmap_vln"]},
    )
    lrs = _group_lrs(optimizer)

    assert lrs["heatmap_vit_dpt_fusion_decay"] == 5e-5
    assert lrs["heatmap_fine_decay"] == 1e-4
    assert lrs["heatmap_llm_dpt_fusion_decay"] == 2e-5
    assert lrs["heatmap_coarse_decay"] == 2e-5
    assert lrs["heatmap_vis_head_decay"] == 2e-5


def test_heatmap_optimizer_group_learning_rates_fall_back_to_heatmap_lr():
    optimizer = build_optimizer(
        _Pipeline(),
        cfg={
            "optim": {
                "heatmap_lr": 7e-5,
                # Schema-normalized configs materialize optional keys as None.
                "heatmap_vit_lr": None,
                "heatmap_fine_lr": None,
                "heatmap_llm_lr": None,
                "heatmap_coarse_lr": None,
                "vis_head_lr": None,
                "weight_decay": 1e-2,
            }
        },
        stage_cfg={"trainable_modules": ["heatmap_vln"]},
    )

    assert {
        group["lr"]
        for group in optimizer.param_groups
        if group["name"].startswith("heatmap_")
    } == {7e-5}

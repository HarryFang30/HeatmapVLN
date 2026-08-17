import logging

from torch import nn

from scripts.training.train_loop import _apply_bridge_only_train_mode


class _Pipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared_qwen = nn.Sequential(
            nn.Linear(2, 2),
            nn.Dropout(0.5),
        )
        self.vlm_backbone = nn.Module()
        self.vlm_backbone.model = shared_qwen
        self.heatmap_vln = nn.Module()
        self.heatmap_vln.qwen = shared_qwen
        self.heatmap_vln.decoder = nn.Sequential(
            nn.Linear(2, 2), nn.Dropout(0.5)
        )


def test_heatmap_only_mode_keeps_frozen_qwen_eval_and_head_train():
    model = _Pipeline()
    model.train()

    _apply_bridge_only_train_mode(
        model,
        {
            "trainable_modules": ["heatmap_vln"],
            "train_action": False,
            "train_lm": False,
        },
        logging.getLogger("test.heatmap_only_mode"),
    )

    assert model.vlm_backbone.training is False
    assert all(module.training is False for module in model.vlm_backbone.modules())
    assert model.heatmap_vln.training is True
    assert model.heatmap_vln.decoder.training is True
    assert all(module.training is True for module in model.heatmap_vln.decoder.modules())
    assert model.heatmap_vln.qwen is model.vlm_backbone.model
    assert model.heatmap_vln.qwen.training is False

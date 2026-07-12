import ast
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from scripts.training.checkpoint import load_checkpoint_model_state


class _LazyHeatmapModel(nn.Module):
    def __init__(self, *, enabled: bool = True) -> None:
        super().__init__()
        self.heatmap_vln = None
        self.enabled = enabled
        self.ensure_calls = 0

    def _ensure_heatmap_vln(self) -> None:
        self.ensure_calls += 1
        if self.enabled:
            self.heatmap_vln = nn.Linear(3, 2)


def test_checkpoint_loader_materializes_and_loads_lazy_heatmap_head():
    model = _LazyHeatmapModel()
    expected_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    expected_bias = torch.tensor([1.5, -2.0])

    _missing, _unexpected, loaded = load_checkpoint_model_state(
        model,
        {
            "module.heatmap_vln.weight": expected_weight,
            "module.heatmap_vln.bias": expected_bias,
        },
        checkpoint_path="stage1.pth",
    )

    assert model.ensure_calls == 1
    assert loaded == 2
    assert torch.equal(model.heatmap_vln.weight, expected_weight)
    assert torch.equal(model.heatmap_vln.bias, expected_bias)


def test_checkpoint_loader_refuses_heatmap_shape_mismatch():
    model = _LazyHeatmapModel()

    with pytest.raises(RuntimeError, match="Incomplete HeatmapVLN checkpoint load refused"):
        load_checkpoint_model_state(
            model,
            {"heatmap_vln.weight": torch.zeros(4, 3)},
            checkpoint_path="bad.pth",
        )


def test_checkpoint_loader_refuses_heatmap_state_when_head_is_disabled():
    model = _LazyHeatmapModel(enabled=False)

    with pytest.raises(RuntimeError, match="heatmap head is disabled"):
        load_checkpoint_model_state(
            model,
            {"heatmap_vln.weight": torch.zeros(2, 3)},
            checkpoint_path="stage1.pth",
        )


@pytest.mark.parametrize(
    "script_path",
    [
        "scripts/evaluation/heatmap.py",
        "scripts/evaluation/general.py",
        "scripts/visualization/heatmap.py",
    ],
)
def test_heatmap_entrypoints_forward_history_relative_poses(script_path):
    """Keep geometry conditioning wired through both user-facing entrypoints."""
    project_root = Path(__file__).resolve().parent.parent
    tree = ast.parse((project_root / script_path).read_text())

    heatmap_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and any(keyword.arg == "return_heatmaps" for keyword in node.keywords)
    ]

    assert heatmap_calls, f"No heatmap model call found in {script_path}"
    assert all(
        any(keyword.arg == "history_rel_poses" for keyword in call.keywords)
        for call in heatmap_calls
    )
    assert all(
        any(keyword.arg == "panoramic_num_histories" for keyword in call.keywords)
        for call in heatmap_calls
    )

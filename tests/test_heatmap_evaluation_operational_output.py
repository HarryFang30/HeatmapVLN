from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from scripts.evaluation.heatmap import (
    evaluate_heatmap,
    select_evaluation_heatmaps,
)


class _OneBatchLoader:
    batch_size = 1

    def __init__(self, batch: dict[str, Any]) -> None:
        self.batch = batch

    def __iter__(self):
        yield self.batch


class _FixedHeatmapModel:
    def __init__(self, output: dict[str, torch.Tensor]) -> None:
        self.output = output
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, **_kwargs):
        return self.output


def _normalized_joint_output() -> dict[str, torch.Tensor]:
    raw = torch.full((1, 1, 4, 2, 2), 0.99)
    gated = torch.zeros_like(raw)
    gated[0, 0, 0, 0, 0] = 0.70
    gated[0, 0, 1, 0, 0] = 0.10
    gated[0, 0, 2, 0, 0] = 0.05
    gated[0, 0, 3, 0, 0] = 0.05
    none_probability = torch.tensor([[0.10]])
    return {
        "heatmaps": raw,
        "heatmaps_gated": gated,
        "none_probability": none_probability,
    }


def test_joint_config_selects_operational_gated_heatmaps() -> None:
    output = _normalized_joint_output()

    selected, none_probability, source = select_evaluation_heatmaps(
        output,
        joint_panorama_inference=True,
    )

    assert source == "heatmaps_gated"
    assert torch.equal(selected, output["heatmaps_gated"])
    assert none_probability is output["none_probability"]
    assert not torch.equal(selected, output["heatmaps"])


def test_legacy_config_keeps_raw_heatmap_compatibility() -> None:
    raw = torch.rand(2, 3, 8, 8)

    selected, none_probability, source = select_evaluation_heatmaps(
        {"heatmaps": raw},
        joint_panorama_inference=False,
    )

    assert source == "heatmaps"
    assert selected is raw
    assert none_probability is None


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (
            {"heatmaps": torch.zeros(1, 1, 4, 2, 2)},
            "heatmaps_gated",
        ),
        (
            {
                "heatmaps_gated": torch.full((1, 1, 4, 2, 2), 1 / 16),
            },
            "none_probability",
        ),
        (
            {
                "heatmaps_gated": torch.full((1, 1, 4, 2, 2), 0.1),
                "none_probability": torch.tensor([[0.1]]),
            },
            "not normalized",
        ),
    ],
)
def test_joint_output_selection_fails_closed(
    output: dict[str, torch.Tensor],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        select_evaluation_heatmaps(
            output,
            joint_panorama_inference=True,
        )


def _joint_metric_fixture():
    height = width = 12
    num_histories = 4
    gt_heatmaps = torch.zeros(1, num_histories, 4, height, width)
    gt_visibility = torch.zeros(1, num_histories, 4)
    gated = torch.zeros_like(gt_heatmaps)
    none_probability = torch.full((1, num_histories), 0.09)

    targets = [
        # target view, target y/x, predicted view, predicted y/x
        (0, (1, 1), 0, (1, 1)),
        (1, (1, 1), 1, (1, 6)),
        (2, (2, 2), 0, (2, 2)),
        (3, (1, 1), 3, (1, 10)),
    ]
    for history_idx, (target_view, target_xy, pred_view, pred_xy) in enumerate(
        targets
    ):
        target_y, target_x = target_xy
        pred_y, pred_x = pred_xy
        gt_visibility[0, history_idx, target_view] = 1
        gt_heatmaps[0, history_idx, target_view, target_y, target_x] = 1
        for view_idx in range(4):
            gated[0, history_idx, view_idx, target_y, target_x] = 0.02
        gated[0, history_idx, pred_view].zero_()
        gated[0, history_idx, pred_view, pred_y, pred_x] = 0.85

    # Deliberately wrong raw maps: the integration assertion fails if the
    # evaluator accidentally selects these instead of operational probabilities.
    raw = torch.zeros_like(gated)
    raw[..., -1, -1] = 1
    output = {
        "heatmaps": raw,
        "heatmaps_gated": gated,
        "none_probability": none_probability,
        "visibility": torch.zeros(1, num_histories, 4),
    }
    batch = {
        "current_frame": torch.zeros(1, 3, 8, 8),
        "history_frames": torch.zeros(1, 1, 3, 8, 8),
        "text": ["go"],
        "current_views": torch.zeros(1, 4, 3, 8, 8),
        "history_panoramas": torch.zeros(
            1,
            num_histories,
            4,
            3,
            8,
            8,
        ),
        "history_rel_poses": torch.zeros(1, num_histories, 4),
        "heatmap": gt_heatmaps,
        "gt_visibility": gt_visibility,
        "history_mask": torch.ones(1, num_histories, dtype=torch.bool),
    }
    return output, batch


def test_evaluate_heatmap_reports_joint_pck_and_direction_counts(
    tmp_path: Path,
) -> None:
    output, batch = _joint_metric_fixture()
    model = _FixedHeatmapModel(output)
    save_dir = tmp_path / "heatmap_eval"

    results = evaluate_heatmap(
        model=model,
        dataloader=_OneBatchLoader(batch),
        gpu_heatmap_computer=object(),
        device=torch.device("cpu"),
        save_dir=save_dir,
        max_samples=0,
        num_vis=0,
        joint_panorama_inference=True,
        amp_mode="none",
    )

    assert model.eval_called
    assert results["heatmap_output_source"] == "heatmaps_gated"
    joint = results["joint_panorama"]
    assert joint["visible_samples"] == 4
    assert joint["joint_pck4"] == 1 / 4
    assert joint["joint_pck8"] == 2 / 4
    assert joint["view5_accuracy"] == 3 / 4
    assert joint["per_direction"] == {
        "front": {"count": 1, "pck8": 1.0},
        "right": {"count": 1, "pck8": 1.0},
        "back": {"count": 1, "pck8": 0.0},
        "left": {"count": 1, "pck8": 0.0},
    }
    assert save_dir.is_dir()
    assert (save_dir / "visualizations").is_dir()
    assert list(tmp_path.iterdir()) == [save_dir]

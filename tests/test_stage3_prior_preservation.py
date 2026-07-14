from __future__ import annotations

import pytest
import torch

from scripts.training.train_loop import (
    _prepare_trajectory_sequence_inputs,
    _trajectory_view_sample_weights,
)
from scripts.training.utils import build_l2_sp_reference, compute_l2_sp_loss


class _AdapterOnlyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pano_latent_adapter = torch.nn.Linear(3, 2, bias=False)
        self.frozen = torch.nn.Linear(2, 1)
        self.frozen.requires_grad_(False)


def test_first_only_matches_goal_freeze_eval_pair() -> None:
    trajectories = torch.arange(2 * 3 * 4 * 3).reshape(2, 3, 4, 3).float()
    valid = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    images = torch.arange(2 * 3 * 2 * 2 * 3).reshape(2, 3, 2, 2, 3).float()

    selected_traj, selected_valid, image_pair = _prepare_trajectory_sequence_inputs(
        trajectories,
        valid,
        images,
        mode="first_only",
    )

    torch.testing.assert_close(selected_traj, trajectories[:, 0])
    torch.testing.assert_close(selected_valid, valid[:, 0])
    assert image_pair.shape == (2, 2, 2, 2, 3)
    torch.testing.assert_close(image_pair[:, 0], images[:, 0])
    torch.testing.assert_close(image_pair[:, 1], images[:, 0])


def test_view_weights_are_ordered_per_sample() -> None:
    weights = _trajectory_view_sample_weights(
        ["left", "front", "back", "right"],
        {
            "enabled": True,
            "weights": {"front": 1.0, "right": 2.0, "back": 16.0, "left": 3.0},
        },
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(weights, torch.tensor([3.0, 1.0, 16.0, 2.0]))


def test_view_weights_reject_non_pixel_samples() -> None:
    with pytest.raises(RuntimeError, match="pixel-goal views"):
        _trajectory_view_sample_weights(
            ["front", "stop"],
            {"enabled": True, "weights": {}},
            device=torch.device("cpu"),
        )


def test_relative_l2_sp_tracks_loaded_stage2_adapter() -> None:
    model = _AdapterOnlyModel()
    cfg = {
        "loss": {
            "l2_sp": {
                "enabled": True,
                "weight": 5.0,
                "modules": ["pano_latent_adapter"],
            }
        }
    }
    reference = build_l2_sp_reference(model, cfg)
    assert set(reference) == {"pano_latent_adapter.weight"}
    assert compute_l2_sp_loss(
        model,
        reference,
        device=torch.device("cpu"),
        normalization="relative_l2",
    ).item() == 0.0

    with torch.no_grad():
        model.pano_latent_adapter.weight.mul_(1.1)
    relative = compute_l2_sp_loss(
        model,
        reference,
        device=torch.device("cpu"),
        normalization="relative_l2",
    )
    assert relative.item() == pytest.approx(0.01, rel=1e-5)


def test_invalid_l2_sp_normalization_fails_closed() -> None:
    model = _AdapterOnlyModel()
    reference = {"pano_latent_adapter.weight": model.pano_latent_adapter.weight.detach().clone()}
    with pytest.raises(ValueError, match="normalization"):
        compute_l2_sp_loss(
            model,
            reference,
            device=torch.device("cpu"),
            normalization="silent_noop",
        )

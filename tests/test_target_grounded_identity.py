import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.models import pipeline as pipeline_module
from src.models.heatmap import (
    TargetGroundedIdentityLoss,
    TargetGroundedPanoramaIdentityLoss,
    circular_pairwise_distances,
    extract_primary_panorama_targets,
    target_grounded_panorama_losses,
    target_grounded_score_matrix,
)


def _ground_truth(
    *,
    targets: list[tuple[int, int, int]] | None = None,
    height: int = 16,
    width: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_values = targets or [
        (0, 5, 8),
        (1, 5, 8),
        (2, 5, 8),
        (3, 5, 8),
    ]
    visibility = torch.zeros(1, 4, 4)
    heatmaps = torch.zeros(1, 4, 4, height, width)
    for history_index, (view, y, x) in enumerate(target_values):
        visibility[0, history_index, view] = 1
        heatmaps[0, history_index, view, y, x] = 1
    return visibility, heatmaps


def _diagonal_predictions(
    gt_visibility: torch.Tensor,
    gt_heatmaps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    visibility_logits = torch.full_like(gt_visibility, -8.0)
    heatmap_logits = torch.zeros_like(gt_heatmaps)
    targets = extract_primary_panorama_targets(gt_visibility, gt_heatmaps)
    for batch_index in range(int(gt_visibility.shape[0])):
        for history_index in range(4):
            view = int(targets.view_indices[batch_index, history_index])
            y = int(targets.y[batch_index, history_index])
            x = int(targets.x[batch_index, history_index])
            visibility_logits[batch_index, history_index, view] = 8.0
            heatmap_logits[batch_index, history_index, view, y, x] = 12.0
    return visibility_logits, heatmap_logits


def test_primary_targets_use_visible_gt_only_and_ties_choose_lower_view():
    visibility, heatmaps = _ground_truth()
    visibility[0, 0, 2] = 1
    heatmaps[0, 0, 2, 3, 4] = 1
    heatmaps[0, 0, 1, 1, 1] = 100  # Invisible and therefore forbidden.

    targets = extract_primary_panorama_targets(visibility, heatmaps)

    assert targets.view_indices.tolist() == [[0, 1, 2, 3]]
    assert targets.x.tolist() == [[8, 8, 8, 8]]
    assert targets.y.tolist() == [[5, 5, 5, 5]]
    assert targets.panorama_x.tolist() == [[8, 24, 40, 56]]
    assert targets.panorama_width == 64


def test_primary_targets_fail_closed_when_a_history_has_no_positive_target():
    visibility, heatmaps = _ground_truth()
    visibility[:, 2] = 0
    heatmaps[:, 2] = 0

    with pytest.raises(ValueError, match="requires a visible GT target"):
        extract_primary_panorama_targets(visibility, heatmaps)


def test_circular_distance_wraps_across_last_and_first_panorama_pixels():
    panorama_x = torch.tensor([[0, 63, 24, 40]])
    y = torch.tensor([[8, 8, 8, 8]])

    distances = circular_pairwise_distances(panorama_x, y, panorama_width=64)

    assert distances.shape == (1, 4, 4)
    assert distances[0, 0, 1].item() == 1
    assert distances[0, 1, 0].item() == 1


def test_score_matrix_uses_view_conditional_spatial_log_prob_and_has_diagonal_signal():
    visibility, heatmaps = _ground_truth()
    _visibility_logits, heatmap_logits = _diagonal_predictions(visibility, heatmaps)
    targets = extract_primary_panorama_targets(visibility, heatmaps)

    scores = target_grounded_score_matrix(heatmap_logits, targets)
    biased = heatmap_logits + torch.tensor([0.0, 7.0, -3.0, 12.0]).reshape(1, 1, 4, 1, 1)
    biased_scores = target_grounded_score_matrix(biased, targets)

    assert scores.shape == (1, 4, 4)
    assert torch.equal(scores.argmax(dim=-1), torch.arange(4).unsqueeze(0))
    torch.testing.assert_close(scores, biased_scores, rtol=1e-6, atol=1e-6)


def test_global_panorama_loss_decomposes_into_view_and_within_view_terms():
    gt_visibility, gt_heatmaps = _ground_truth()
    _pred_visibility, pred_heatmaps = _diagonal_predictions(gt_visibility, gt_heatmaps)
    losses = target_grounded_panorama_losses(
        pred_heatmaps,
        gt_visibility,
        gt_heatmaps,
    )

    assert losses["panorama_loss"].item() < 0.03
    assert losses["view_loss"].item() < 0.03
    assert losses["within_view_loss"].item() < 0.03
    torch.testing.assert_close(
        losses["panorama_loss"],
        losses["view_loss"] + losses["within_view_loss"],
        rtol=1e-5,
        atol=1e-5,
    )
    assert losses["view_logits"].argmax(dim=-1).tolist() == [[0, 1, 2, 3]]


def test_soft_heatmap_panorama_decomposition_is_exact_for_batched_non_square_maps():
    gt_visibility, gt_heatmaps = _ground_truth(
        height=3,
        width=7,
        targets=[
            (0, 1, 1),
            (1, 1, 2),
            (2, 1, 3),
            (3, 1, 4),
        ],
    )
    gt_heatmaps[:, :, :, 1, 5] += 0.25 * gt_visibility
    gt_visibility = gt_visibility.repeat(2, 1, 1)
    gt_heatmaps = gt_heatmaps.repeat(2, 1, 1, 1, 1)
    torch.manual_seed(91)
    logits = torch.randn_like(gt_heatmaps)

    losses = target_grounded_panorama_losses(logits, gt_visibility, gt_heatmaps)

    torch.testing.assert_close(
        losses["panorama_loss"],
        losses["view_loss"] + losses["within_view_loss"],
        rtol=1e-6,
        atol=1e-6,
    )
    assert losses["target_view_distribution"].shape == (2, 4, 4)


def test_global_panorama_loss_is_view_sensitive_while_identity_is_view_bias_invariant():
    gt_visibility, gt_heatmaps = _ground_truth()
    _pred_visibility, pred_heatmaps = _diagonal_predictions(gt_visibility, gt_heatmaps)
    criterion = TargetGroundedPanoramaIdentityLoss()

    standard = criterion(pred_heatmaps, gt_visibility, gt_heatmaps)
    wrong_view_bias = torch.tensor([-20.0, 20.0, -20.0, -20.0]).reshape(1, 1, 4, 1, 1)
    biased = criterion(pred_heatmaps + wrong_view_bias, gt_visibility, gt_heatmaps)

    torch.testing.assert_close(standard["identity_loss"], biased["identity_loss"])
    assert biased["panorama_loss"] > standard["panorama_loss"] + 10


def test_panorama_identity_blank_logits_have_exact_chance_losses_and_raw_logit_gradients():
    gt_visibility, gt_heatmaps = _ground_truth()
    pred_heatmaps = torch.zeros_like(gt_heatmaps, requires_grad=True)
    criterion = TargetGroundedPanoramaIdentityLoss()

    losses = criterion(pred_heatmaps, gt_visibility, gt_heatmaps)

    torch.testing.assert_close(losses["identity_loss"], torch.tensor(math.log(4.0)))
    torch.testing.assert_close(losses["view_loss"], torch.tensor(math.log(4.0)))
    torch.testing.assert_close(losses["within_view_loss"], torch.tensor(math.log(16 * 16.0)))
    torch.testing.assert_close(losses["panorama_loss"], torch.tensor(math.log(4 * 16 * 16.0)))
    losses["total"].backward()
    assert pred_heatmaps.grad is not None
    assert (pred_heatmaps.grad.abs().sum(dim=(0, 2, 3, 4)) > 0).all()


def test_strict_panorama_identity_rejects_multiple_visible_views_and_negative_gt():
    gt_visibility, gt_heatmaps = _ground_truth()
    pred_heatmaps = torch.zeros_like(gt_heatmaps)
    gt_visibility[0, 0, 1] = 1
    gt_heatmaps[0, 0, 1, 4, 4] = 1

    with pytest.raises(ValueError, match="exactly one visible GT view"):
        TargetGroundedPanoramaIdentityLoss()(pred_heatmaps, gt_visibility, gt_heatmaps)

    gt_visibility, gt_heatmaps = _ground_truth()
    gt_heatmaps[0, 0, 0, 0, 0] = -1
    with pytest.raises(ValueError, match="must be non-negative"):
        target_grounded_panorama_losses(pred_heatmaps, gt_visibility, gt_heatmaps)


def test_symmetric_identity_and_view_losses_are_low_for_correct_predictions():
    gt_visibility, gt_heatmaps = _ground_truth()
    pred_visibility, pred_heatmaps = _diagonal_predictions(gt_visibility, gt_heatmaps)
    criterion = TargetGroundedIdentityLoss()

    losses = criterion(pred_visibility, pred_heatmaps, gt_visibility, gt_heatmaps)

    assert losses["score_matrix"].argmax(dim=-1).tolist() == [[0, 1, 2, 3]]
    assert losses["identity_loss"].item() < 0.02
    assert losses["view_loss"].item() < 0.01
    assert losses["minimum_target_separation"].item() == 16


def test_identity_handles_same_view_negatives_and_batches_without_visibility_shortcut():
    gt_visibility, gt_heatmaps = _ground_truth(
        targets=[
            (0, 8, 2),
            (0, 8, 18),
            (0, 8, 34),
            (0, 8, 50),
        ],
        height=16,
        width=64,
    )
    gt_visibility = gt_visibility.repeat(2, 1, 1)
    gt_heatmaps = gt_heatmaps.repeat(2, 1, 1, 1, 1)
    pred_visibility, pred_heatmaps = _diagonal_predictions(
        gt_visibility,
        gt_heatmaps,
    )
    pred_heatmaps.requires_grad_(True)
    criterion = TargetGroundedIdentityLoss()

    standard = criterion(
        pred_visibility,
        pred_heatmaps,
        gt_visibility,
        gt_heatmaps,
    )
    visibility_changed = criterion(
        torch.randn_like(pred_visibility) * 100,
        pred_heatmaps,
        gt_visibility,
        gt_heatmaps,
    )

    assert standard["score_matrix"].shape == (2, 4, 4)
    assert torch.equal(
        standard["score_matrix"].argmax(dim=-1),
        torch.arange(4).repeat(2, 1),
    )
    assert standard["identity_loss"].item() < 0.02
    torch.testing.assert_close(
        standard["identity_loss"],
        visibility_changed["identity_loss"],
        rtol=0,
        atol=0,
    )
    standard["identity_loss"].backward()
    assert pred_heatmaps.grad is not None
    assert (pred_heatmaps.grad.abs().sum(dim=(0, 2, 3, 4)) > 0).all()


def test_identical_blank_outputs_are_exactly_chance_and_backpropagate_to_every_row():
    gt_visibility, gt_heatmaps = _ground_truth()
    pred_visibility = torch.zeros_like(gt_visibility, requires_grad=True)
    pred_heatmaps = torch.zeros_like(gt_heatmaps, requires_grad=True)
    criterion = TargetGroundedIdentityLoss()

    losses = criterion(pred_visibility, pred_heatmaps, gt_visibility, gt_heatmaps)

    torch.testing.assert_close(
        losses["identity_loss"],
        torch.tensor(math.log(4.0)),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        losses["view_loss"],
        torch.tensor(math.log(4.0)),
        rtol=1e-6,
        atol=1e-6,
    )
    losses["total"].backward()
    assert pred_visibility.grad is not None
    assert pred_heatmaps.grad is not None
    assert (pred_visibility.grad.abs().sum(dim=(0, 2)) > 0).all()
    assert (pred_heatmaps.grad.abs().sum(dim=(0, 2, 3, 4)) > 0).all()


def test_joint_history_permutation_conjugates_scores_and_preserves_losses():
    torch.manual_seed(81)
    gt_visibility, gt_heatmaps = _ground_truth()
    pred_visibility = torch.randn_like(gt_visibility)
    pred_heatmaps = torch.randn_like(gt_heatmaps)
    criterion = TargetGroundedIdentityLoss()
    standard = criterion(pred_visibility, pred_heatmaps, gt_visibility, gt_heatmaps)
    permutation = torch.tensor([2, 0, 3, 1])

    permuted = criterion(
        pred_visibility[:, permutation],
        pred_heatmaps[:, permutation],
        gt_visibility[:, permutation],
        gt_heatmaps[:, permutation],
    )

    expected_scores = standard["score_matrix"][:, permutation][:, :, permutation]
    torch.testing.assert_close(permuted["score_matrix"], expected_scores)
    torch.testing.assert_close(permuted["identity_loss"], standard["identity_loss"])
    torch.testing.assert_close(permuted["view_loss"], standard["view_loss"])
    torch.testing.assert_close(permuted["total"], standard["total"])


def test_loss_rejects_targets_that_violate_circular_minimum_separation():
    gt_visibility, gt_heatmaps = _ground_truth(
        targets=[
            (0, 8, 0),
            (3, 8, 15),
            (1, 8, 8),
            (2, 8, 8),
        ]
    )
    pred_visibility = torch.zeros_like(gt_visibility)
    pred_heatmaps = torch.zeros_like(gt_heatmaps)

    with pytest.raises(ValueError, match=r"minimum=1\.000000 required=12\.000000"):
        TargetGroundedIdentityLoss()(
            pred_visibility,
            pred_heatmaps,
            gt_visibility,
            gt_heatmaps,
        )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("identity_weight", math.nan),
        ("view_weight", math.inf),
        ("temperature", math.nan),
        ("min_target_separation", -math.inf),
    ],
)
def test_loss_rejects_nonfinite_hyperparameters(argument: str, value: float):
    with pytest.raises(ValueError, match="must be finite"):
        TargetGroundedIdentityLoss(**{argument: value})


class _QwenOutputStub(nn.Module):
    def __init__(self, *, include_logits: bool = True) -> None:
        super().__init__()
        self.include_logits = include_logits
        self.register_buffer("visibility", torch.arange(4, dtype=torch.float32).reshape(1, 1, 4))
        self.register_buffer("heatmaps", torch.full((1, 1, 4, 3, 5), 0.25))
        self.register_buffer("heatmap_logits", torch.full((1, 1, 4, 3, 5), -math.log(3.0)))

    def forward(self, **_kwargs):
        output = {
            "visibility": self.visibility,
            "heatmaps": self.heatmaps,
            "num_image_tokens": 0,
        }
        if self.include_logits:
            output["heatmap_logits"] = self.heatmap_logits
        return output


def _pipeline_stub(*, include_logits: bool = True):
    pipeline = pipeline_module.VLNPipeline.__new__(pipeline_module.VLNPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.config = SimpleNamespace(enable_runtime_timing=False, dtype=torch.float32)
    pipeline.heatmap_vln = nn.Identity()
    pipeline._heatmap_enabled = True
    pipeline.qwen2_5_vl = _QwenOutputStub(include_logits=include_logits)
    pipeline.nextdit_action_head = None
    pipeline.latent_queries = None
    pipeline.llm_projector = nn.Identity()
    return pipeline


def _pipeline_inputs():
    return {
        "video_frames": torch.zeros(1, 2, 3, 2, 2),
        "current_views": torch.zeros(1, 4, 3, 2, 2),
        "history_panoramas": torch.zeros(1, 1, 4, 3, 2, 2),
        "return_actions": False,
    }


def test_pipeline_raw_heatmap_logits_are_strictly_opt_in_and_default_keys_do_not_change():
    pipeline = _pipeline_stub()

    default = pipeline(**_pipeline_inputs())
    diagnostic = pipeline(return_heatmap_logits=True, **_pipeline_inputs())

    assert set(default) == {"processing_metadata", "visibility", "heatmaps"}
    assert set(diagnostic) == {
        "processing_metadata",
        "visibility",
        "heatmaps",
        "heatmap_logits",
    }
    torch.testing.assert_close(default["visibility"], diagnostic["visibility"], rtol=0, atol=0)
    torch.testing.assert_close(default["heatmaps"], diagnostic["heatmaps"], rtol=0, atol=0)
    torch.testing.assert_close(
        diagnostic["heatmap_logits"],
        pipeline.qwen2_5_vl.heatmap_logits,
        rtol=0,
        atol=0,
    )

    pipeline.qwen2_5_vl.heatmap_logits.requires_grad_(True)
    pipeline(return_heatmap_logits=True, **_pipeline_inputs())["heatmap_logits"].sum().backward()
    assert pipeline.qwen2_5_vl.heatmap_logits.grad is not None


def test_pipeline_raw_logit_opt_in_fails_if_decoder_does_not_supply_logits():
    pipeline = _pipeline_stub(include_logits=False)

    with pytest.raises(RuntimeError, match="explicitly requested"):
        pipeline(return_heatmap_logits=True, **_pipeline_inputs())


def test_pipeline_raw_logit_opt_in_requires_an_active_heatmap_path():
    pipeline = _pipeline_stub()

    with pytest.raises(ValueError, match="active panoramic heatmap path"):
        pipeline(
            return_heatmap_logits=True,
            return_heatmaps=False,
            **_pipeline_inputs(),
        )

    with pytest.raises(ValueError, match="active panoramic heatmap path"):
        pipeline(
            video_frames=torch.zeros(1, 2, 3, 2, 2),
            return_actions=False,
            return_heatmap_logits=True,
        )

import argparse
import json

import pytest
import torch
from scripts.training.train_system2_onpolicy_sft import (
    _resolve_rollout_roots,
    _sft_objective,
    _System2SFTCollator,
    _with_quality_metrics,
)


class _CaptureCollator:
    def __call__(self, batch):
        return batch


def test_rollout_report_is_the_single_source_of_root_paths(tmp_path):
    roots = [tmp_path / "root_a", tmp_path / "root_b"]
    for root in roots:
        root.mkdir()
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "root_count": 2,
                "roots": [{"root": str(root)} for root in roots],
            }
        ),
        encoding="utf-8",
    )

    resolved, contract = _resolve_rollout_roots(
        argparse.Namespace(rollout_report=report_path, rollout_root=None)
    )

    assert resolved == [root.resolve() for root in roots]
    assert contract is not None
    assert contract["root_count"] == 2
    assert len(contract["sha256"]) == 64


def test_rollout_report_rejects_count_mismatch(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "root_count": 2,
                "roots": [{"root": str(root)}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="root count mismatch"):
        _resolve_rollout_roots(
            argparse.Namespace(rollout_report=report_path, rollout_root=None)
        )


def test_quality_metrics_require_both_stop_improvements_and_prior_retention():
    metrics = {
        "stop_recall": 0.5,
        "false_stop_negative_stop_fpr": 0.25,
        "regular_negative_stop_fpr": 0.0,
        "non_stop_class_accuracy": 1.0,
    }
    enriched = _with_quality_metrics(
        metrics,
        torch.tensor([0, 1, 2]),
        baseline_metrics={
            "stop_recall": 0.25,
            "false_stop_negative_stop_fpr": 0.75,
        },
        baseline_predictions=torch.tensor([0, 1, 2]),
        targets=[1, 0, 0],
        original_terminals=[False, False, True],
    )

    assert enriched["stop_recall_improvement"] == pytest.approx(0.25)
    assert enriched["false_stop_fpr_improvement"] == pytest.approx(0.5)
    assert enriched["non_stop_prediction_retention"] == pytest.approx(1.0)
    assert enriched["quality_passed"] is True


def test_system2_collator_strips_unrelated_native_targets():
    collator = _System2SFTCollator(_CaptureCollator())

    normalized = collator(
        [
            {
                "text": "go to the desk",
                "action": torch.zeros(2),
                "trajectory": torch.zeros(32, 3),
                "heatmap": torch.zeros(1, 8, 8),
                "gt_visibility": torch.ones(1),
            }
        ]
    )

    assert set(normalized[0]) == {"text", "action"}


def test_system2_collator_flattens_false_stop_and_paired_positive():
    collator = _System2SFTCollator(_CaptureCollator())
    paired = {
        "text": "walk to the desk",
        "action": torch.zeros(2),
        "system2_replay_role": "onpolicy_paired_positive",
        "system2_stop_pair_id": "pair-1",
    }

    normalized = collator(
        [
            {
                "text": "walk to the desk",
                "action": torch.zeros(2),
                "system2_replay_role": "onpolicy_false_stop_negative",
                "system2_stop_pair_id": "pair-1",
                "_system2_paired_positive": paired,
            }
        ]
    )

    assert [row["system2_replay_role"] for row in normalized] == [
        "onpolicy_false_stop_negative",
        "onpolicy_paired_positive",
    ]
    assert [row["system2_stop_pair_id"] for row in normalized] == [
        "pair-1",
        "pair-1",
    ]


class _SparseLogprobIntegration:
    def __init__(self, logprobs, rejection_log_odds, token_ids, structured_logits):
        self.logprobs = logprobs
        self.rejection_log_odds = rejection_log_odds
        self.token_ids = token_ids
        self.structured_logits = structured_logits

    def _forward_model_inputs(self, inputs, **kwargs):
        del inputs, kwargs
        alignment = {
            "sample_label_tokens": [len(row) for row in self.token_ids],
            "sample_correct_token_ids": self.token_ids,
        }
        return None, None, None, None, {
            "alignment": alignment,
            "correct_label_logprobs": self.logprobs,
            "correct_label_rejection_log_odds": self.rejection_log_odds,
            "structured_class_logits": self.structured_logits,
        }


def test_false_stop_uses_unlikelihood_without_counterfactual_ce():
    class_token_ids = (7, 8, 9, 10, 11, 12)
    stop_token_id = class_token_ids[0]
    logprobs = torch.tensor(
        [-0.2, -0.4, -0.1, torch.log(torch.tensor(0.8))],
        requires_grad=True,
    )
    rejection_log_odds = torch.tensor(
        [0.0, 0.0, 0.0, torch.log(torch.tensor(4.0))],
        requires_grad=True,
    )
    structured_logits = torch.tensor(
        [[0.0] * 6, [2.0, 0.5, -0.5, -1.0, -1.5, -2.0]],
        requires_grad=True,
    )
    integration = _SparseLogprobIntegration(
        logprobs,
        rejection_log_odds,
        [[class_token_ids[1], 4], [3, stop_token_id]],
        structured_logits,
    )
    batch = {
        "pano_inputs": {"input_ids": torch.ones((2, 1), dtype=torch.long)},
        "system2_replay_role": [
            "native",
            "onpolicy_false_stop_negative",
        ],
        "sft_target_text": [["view: front\npixel: 10 20"], ["view: stop"]],
    }

    loss, token_ce, rejection, pair_rank = _sft_objective(
        integration,
        batch,
        torch.device("cpu"),
        structured_class_token_ids=class_token_ids,
    )

    expected_ce = torch.tensor(0.3)
    expected_rejection = torch.nn.functional.softplus(
        structured_logits[1, 0]
        - torch.logsumexp(structured_logits[1, 1:], dim=0)
    )
    torch.testing.assert_close(token_ce, expected_ce)
    torch.testing.assert_close(rejection, expected_rejection)
    torch.testing.assert_close(pair_rank, torch.tensor(0.0))
    torch.testing.assert_close(loss, (expected_ce + expected_rejection) / 2)

    loss.backward()
    assert structured_logits.grad is not None
    assert structured_logits.grad[1, 0] > 0
    assert logprobs.grad is not None
    assert logprobs.grad[-1] == 0
    assert rejection_log_odds.grad is None


def test_onpolicy_positive_targets_stop_token_without_prefix_dilution():
    class_token_ids = (7, 8, 9, 10, 11, 12)
    stop_token_id = class_token_ids[0]
    logprobs = torch.tensor([-0.01, -0.02, -12.0], requires_grad=True)
    rejection_log_odds = torch.zeros(3, requires_grad=True)
    structured_logits = torch.tensor(
        [[-12.0, 0.0, -1.0, -2.0, -3.0, -4.0]],
        requires_grad=True,
    )
    integration = _SparseLogprobIntegration(
        logprobs,
        rejection_log_odds,
        [[3, 4, stop_token_id]],
        structured_logits,
    )
    batch = {
        "pano_inputs": {"input_ids": torch.ones((1, 1), dtype=torch.long)},
        "system2_replay_role": ["onpolicy_positive"],
        "sft_target_text": [["view: stop"]],
    }

    loss, token_ce, rejection, pair_rank = _sft_objective(
        integration,
        batch,
        torch.device("cpu"),
        structured_class_token_ids=class_token_ids,
    )

    expected = torch.nn.functional.cross_entropy(
        structured_logits,
        torch.zeros(1, dtype=torch.long),
    )
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(token_ce, expected)
    torch.testing.assert_close(rejection, torch.tensor(0.0))
    torch.testing.assert_close(pair_rank, torch.tensor(0.0))

    loss.backward()
    assert structured_logits.grad is not None
    assert structured_logits.grad[0, 0] < 0
    torch.testing.assert_close(logprobs.grad, torch.zeros_like(logprobs))
    assert rejection_log_odds.grad is None


def test_regular_negative_rejects_stop_without_waypoint_ce():
    class_token_ids = (7, 8, 9, 10, 11, 12)
    stop_token_id = class_token_ids[0]
    logprobs = torch.tensor([-0.01, -0.02, -0.3], requires_grad=True)
    rejection_log_odds = torch.tensor(
        [0.0, 0.0, torch.log(torch.tensor(3.0))],
        requires_grad=True,
    )
    structured_logits = torch.tensor(
        [[1.0, 0.5, 0.0, -0.5, -1.0, -1.5]],
        requires_grad=True,
    )
    integration = _SparseLogprobIntegration(
        logprobs,
        rejection_log_odds,
        [[3, 4, stop_token_id]],
        structured_logits,
    )
    batch = {
        "pano_inputs": {"input_ids": torch.ones((1, 1), dtype=torch.long)},
        "system2_replay_role": ["onpolicy_regular_negative"],
        "sft_target_text": [["view: stop"]],
    }

    loss, token_ce, rejection, pair_rank = _sft_objective(
        integration,
        batch,
        torch.device("cpu"),
        structured_class_token_ids=class_token_ids,
    )

    expected = torch.nn.functional.softplus(
        structured_logits[0, 0]
        - torch.logsumexp(structured_logits[0, 1:], dim=0)
    )
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(token_ce, torch.tensor(0.0))
    torch.testing.assert_close(rejection, expected)
    torch.testing.assert_close(pair_rank, torch.tensor(0.0))

    loss.backward()
    torch.testing.assert_close(logprobs.grad, torch.zeros_like(logprobs))
    assert rejection_log_odds.grad is None
    assert structured_logits.grad is not None
    assert structured_logits.grad[0, 0] > 0


def test_pairwise_stop_margin_ranks_and_anchors_same_episode_positive():
    class_token_ids = (7, 8, 9, 10, 11, 12)
    stop_token_id = class_token_ids[0]
    logprobs = torch.tensor([-0.1, -0.1], requires_grad=True)
    rejection_log_odds = torch.zeros(2, requires_grad=True)
    structured_logits = torch.tensor(
        [
            [2.0, 0.0, -1.0, -1.0, -1.0, -1.0],
            [-2.0, 0.0, -1.0, -1.0, -1.0, -1.0],
        ],
        requires_grad=True,
    )
    integration = _SparseLogprobIntegration(
        logprobs,
        rejection_log_odds,
        [[stop_token_id], [stop_token_id]],
        structured_logits,
    )
    batch = {
        "pano_inputs": {"input_ids": torch.ones((2, 1), dtype=torch.long)},
        "system2_replay_role": [
            "onpolicy_false_stop_negative",
            "onpolicy_paired_positive",
        ],
        "system2_stop_pair_id": ["episode-pair", "episode-pair"],
        "sft_target_text": [["view: stop"], ["view: stop"]],
    }

    loss, token_ce, rejection, pair_rank = _sft_objective(
        integration,
        batch,
        torch.device("cpu"),
        structured_class_token_ids=class_token_ids,
        pairwise_stop_margin_weight=0.5,
        pairwise_stop_margin_gap=1.0,
    )

    margins = structured_logits[:, 0] - torch.logsumexp(
        structured_logits[:, 1:], dim=1
    )
    expected_rejection = torch.nn.functional.softplus(margins[0])
    expected_positive_ce = torch.nn.functional.cross_entropy(
        structured_logits[1:2], torch.zeros(1, dtype=torch.long)
    )
    expected_pair = torch.nn.functional.softplus(1.0 - (margins[1] - margins[0]))
    torch.testing.assert_close(rejection, expected_rejection)
    torch.testing.assert_close(token_ce, expected_positive_ce)
    torch.testing.assert_close(pair_rank, expected_pair)
    torch.testing.assert_close(
        loss,
        0.5 * (expected_rejection + expected_positive_ce) + 0.5 * expected_pair,
    )

    loss.backward()
    assert structured_logits.grad[0, 0] > 0
    assert structured_logits.grad[1, 0] < 0

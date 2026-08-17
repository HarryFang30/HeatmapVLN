import math
from types import SimpleNamespace

import pytest
import torch
from scripts.training.train_system2_stop_decision_adapter import (
    _BalancedStopBatchSampler,
    _loss,
    _require_rollout_policy_coverage,
    _split_by_scene,
    _threshold_metrics,
    _validation_score_audit_rows,
    _veto_only_threshold_metrics,
)
from torch import nn

from src.models.qwen2_5_vl.integration import Qwen2_5VLIntegration


class _FakeTuner(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.ModuleDict(
            {
                "default": nn.Linear(3, 2, bias=False),
                "stop_decision": nn.Linear(3, 2, bias=False),
            }
        )
        self.lora_B = nn.ModuleDict(
            {
                "default": nn.Linear(2, 3, bias=False),
                "stop_decision": nn.Linear(2, 3, bias=False),
            }
        )
        self.active_adapters = ["default"]

    def set_adapter(self, adapter_name, inference_mode=False):
        del inference_mode
        self.active_adapters = list(adapter_name)


class _FakePeftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = _FakeTuner()
        self.peft_config = {"default": object(), "stop_decision": object()}


def _integration_with_fake_peft() -> Qwen2_5VLIntegration:
    integration = object.__new__(Qwen2_5VLIntegration)
    nn.Module.__init__(integration)
    integration._model_loaded = True
    integration.model = _FakePeftModel()
    return integration


def test_adapter_stack_freezes_original_and_canonicalizes_checkpoint_keys():
    integration = _integration_with_fake_peft()
    integration.activate_lora_adapters(
        ("default", "stop_decision"),
        trainable_adapters=("stop_decision",),
    )

    assert integration.active_lora_adapters() == ("default", "stop_decision")
    parameters = dict(integration.model.named_parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in parameters.items()
        if ".default." in name
    )
    assert all(
        parameter.requires_grad
        for name, parameter in parameters.items()
        if ".stop_decision." in name
    )

    state = integration.lora_adapter_state_dict("stop_decision")
    assert len(state) == 2
    assert all("stop_decision" not in name for name in state)
    assert {name.rsplit(".", 2)[-2] for name in state} == {"lora_A", "lora_B"}

    for value in state.values():
        value.fill_(0.25)
    assert integration.load_lora_adapter_state_dict("stop_decision", state) == 2
    assert all(
        torch.equal(parameter.detach().cpu(), torch.full_like(parameter.detach().cpu(), 0.25))
        for _name, parameter in integration.lora_adapter_named_parameters(
            "stop_decision"
        )
    )


class _Tokenizer:
    classes = ("stop", "front", "right", "back", "left", "turn")

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        name = text.removeprefix("view: ")
        return [10, 11, 20 + self.classes.index(name)]


def test_structured_view_logits_project_only_six_class_rows(monkeypatch):
    integration = object.__new__(Qwen2_5VLIntegration)
    nn.Module.__init__(integration)
    integration.processor = SimpleNamespace(tokenizer=_Tokenizer())
    lm_head = nn.Linear(3, 40, bias=False)
    with torch.no_grad():
        lm_head.weight.zero_()
        for offset in range(6):
            lm_head.weight[20 + offset] = torch.tensor(
                [float(offset + 1), 0.0, 0.0]
            )
    monkeypatch.setattr(
        integration,
        "_locate_conditional_generation_lm_head",
        lambda: ("model", object(), lm_head),
    )
    hidden = torch.zeros(2, 4, 3)
    hidden[0, 1, 0] = 2.0
    hidden[1, 3, 0] = 3.0
    logits = integration.structured_view_class_logits(
        hidden, torch.tensor([1, 3])
    )

    assert logits.shape == (2, 6)
    torch.testing.assert_close(logits[0], torch.arange(1, 7).float() * 2)
    torch.testing.assert_close(logits[1], torch.arange(1, 7).float() * 3)
    contract = integration.structured_view_token_contract()
    assert contract["prefix_token_ids"] == [10, 11]
    assert contract["class_token_ids"] == list(range(20, 26))


def test_weighted_stop_loss_penalizes_negative_false_stop_more():
    logits = torch.tensor([0.0, 0.0])
    targets = torch.tensor([1.0, 0.0])
    loss = _loss(logits, targets, negative_weight=3.0)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())


def test_threshold_calibration_keeps_add_false_positive_rate_bounded():
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    probabilities = torch.tensor([0.95, 0.75, 0.70, 0.10])
    metrics = _threshold_metrics(targets, probabilities)

    assert metrics["add"]["false_positive_rate"] == 0.0
    assert metrics["add_stop_threshold"] > 0.70
    assert metrics["veto"]["recall"] == 1.0
    assert metrics["veto_stop_threshold"] < metrics["add_stop_threshold"]
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["quality_passed"] is True


def test_threshold_calibration_requires_zero_heldout_add_false_positives():
    targets = torch.tensor([1.0, 1.0] + [0.0] * 200)
    probabilities = torch.tensor([0.90, 0.60] + [0.70] + [0.10] * 199)

    metrics = _threshold_metrics(targets, probabilities)

    assert metrics["add"]["false_positive_rate"] == 0.0
    assert metrics["add"]["recall"] == pytest.approx(0.5)
    assert metrics["add_stop_threshold"] == pytest.approx(0.90)


def test_threshold_calibration_rejects_weak_add_recall_even_with_valid_veto():
    targets = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    probabilities = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    metrics = _threshold_metrics(targets, probabilities)

    assert 0.0 < metrics["veto_stop_threshold"] < metrics["add_stop_threshold"]
    assert metrics["veto"]["negative_rejection_rate"] == 1.0
    assert metrics["add"]["recall"] == pytest.approx(0.25)
    assert metrics["quality_passed"] is False
    assert len(metrics["quality_violations"]) >= 2


def test_veto_only_contract_can_pass_without_add_recall():
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    probabilities = torch.tensor([0.7, 0.95, 0.9, 0.1])
    original_terminal = torch.tensor([False, True, False, True])
    metrics = _threshold_metrics(
        targets,
        probabilities,
        original_terminal_mask=original_terminal,
    )

    veto_only = _veto_only_threshold_metrics(metrics)

    assert metrics["quality_passed"] is False
    assert veto_only["quality_passed"] is True
    assert veto_only["policy_kind"] == "veto_only"
    assert veto_only["add_enabled"] is False
    assert veto_only["add_stop_threshold"] == pytest.approx(1.0)
    assert veto_only["add"]["false_positive_rate"] == 0.0


def test_validation_score_audit_identifies_ranked_policy_outliers():
    records = [
        {
            "key": "positive",
            "scene_id": "scene-a",
            "episode_id": 1,
            "system2_call_index": 2,
            "stop_target": 1,
            "original_terminal": False,
            "distance_to_goal_m": 2.5,
            "system2_decision_scores": {
                "class_probabilities": {"stop": 0.1}
            },
            "original_output": "view: front",
            "effective_output": "view: front",
        },
        {
            "key": "outlier-negative",
            "scene_id": "scene-b",
            "episode_id": 3,
            "system2_call_index": 4,
            "stop_target": 0,
            "original_terminal": False,
            "distance_to_goal_m": 3.1,
            "system2_decision_scores": {
                "class_probabilities": {"stop": 0.2}
            },
            "original_output": "view: right",
            "effective_output": "view: right",
            "oracle_recovery_active": True,
        },
    ]

    rows = _validation_score_audit_rows(records, torch.tensor([0.8, 0.99]))

    assert [row["key"] for row in rows] == ["outlier-negative", "positive"]
    assert rows[0]["policy_role"] == "regular_negative"
    assert rows[0]["original_qwen_stop_probability"] == pytest.approx(0.2)
    assert rows[0]["oracle_recovery_active"] is True
    assert rows[1]["policy_role"] == "add_positive"


def test_threshold_calibration_resolves_sub_milliprobability_scores():
    targets = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    probabilities = torch.tensor(
        [1.0e-9, 8.0e-10, 7.0e-10, 6.0e-10, 5.0e-10, 4.0e-10, 3.0e-10, 2.0e-10]
    )

    metrics = _threshold_metrics(
        targets,
        probabilities,
        veto_reference_probabilities=torch.tensor(
            [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]
        ),
    )

    assert 0.0 < metrics["veto_stop_threshold"] < 1.0e-8
    assert metrics["veto"]["recall"] == 1.0
    assert metrics["veto"]["negative_rejection_rate"] >= 0.75
    assert metrics["add_stop_threshold"] > metrics["veto_stop_threshold"]
    assert metrics["add"]["recall"] >= 0.5
    assert metrics["add"]["false_positive_rate"] == 0.0
    assert metrics["quality_passed"] is True


def test_policy_aware_thresholds_use_recorded_generation_decisions():
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    probabilities = torch.tensor([0.9, 0.8, 0.6, 0.7, 0.2, 0.1])
    original_terminal = torch.tensor([False, True, True, False, True, False])

    metrics = _threshold_metrics(
        targets,
        probabilities,
        original_terminal_mask=original_terminal,
    )

    assert metrics["add_reference_positive_count"] == 1
    assert metrics["add_reference_negative_count"] == 2
    assert metrics["veto_reference_positive_count"] == 2
    assert metrics["veto_reference_negative_count"] == 2
    assert metrics["add"]["recall"] == 1.0
    assert metrics["add"]["false_positive_rate"] == 0.0
    assert metrics["veto"]["recall"] == 1.0
    assert metrics["veto"]["negative_rejection_rate"] == 1.0
    assert metrics["veto"]["reference"] == "recorded_original_terminal"
    assert metrics["quality_passed"] is True


def test_pairwise_ranking_loss_rewards_positive_ordering():
    targets = torch.tensor([1.0, 0.0])
    ordered = _loss(
        torch.tensor([2.0, -2.0]),
        targets,
        negative_weight=1.0,
        bce_weight=0.0,
        ranking_weight=1.0,
        ranking_margin=2.0,
    )
    reversed_order = _loss(
        torch.tensor([-2.0, 2.0]),
        targets,
        negative_weight=1.0,
        bce_weight=0.0,
        ranking_weight=1.0,
        ranking_margin=2.0,
    )

    assert ordered.item() < reversed_order.item()


class _FakeSceneDataset:
    def __init__(self):
        self.clips = [
            "/data/train/scene_a/clip_0",
            "/data/train/scene_a/clip_1",
            "/data/train/scene_b/clip_0",
            "/data/train/scene_c/clip_0",
        ]
        self.sample_index = [(index, 0) for index in range(len(self.clips))]

    def subset_by_clip_indices(self, clip_indices):
        subset = _FakeSceneDataset()
        subset.sample_index = [
            item for item in self.sample_index if item[0] in clip_indices
        ]
        return subset


def test_scene_holdout_has_no_scene_leakage():
    train, val = _split_by_scene(_FakeSceneDataset(), fraction=1 / 3, seed=7)
    train_scenes = {train.clips[index].split("/")[-2] for index, _ in train.sample_index}
    val_scenes = {val.clips[index].split("/")[-2] for index, _ in val.sample_index}

    assert train_scenes
    assert val_scenes
    assert train_scenes.isdisjoint(val_scenes)


class _FakeRolloutDataset:
    def __init__(self, scene_ids=("scene_a", "scene_a", "scene_b", "scene_b")):
        self.sample_scene_ids = tuple(scene_ids)
        self.targets = tuple(index % 2 for index in range(len(scene_ids)))

    def subset_by_indices(self, indices):
        return _FakeRolloutDataset([self.sample_scene_ids[index] for index in indices])


def test_rollout_scene_holdout_has_no_scene_leakage():
    train, val = _split_by_scene(_FakeRolloutDataset(), fraction=0.5, seed=7)

    assert set(train.sample_scene_ids).isdisjoint(val.sample_scene_ids)
    assert set(train.targets) == {0, 1}
    assert set(val.targets) == {0, 1}


def test_balanced_batch_sampler_emits_both_classes():
    sampler = _BalancedStopBatchSampler([0, 1], [2, 3, 4, 5], batch_size=2, seed=11)

    batches = list(sampler)

    assert len(batches) == len(sampler)
    assert all(len(set(batch) & {0, 1}) == 1 for batch in batches)
    assert all(len(set(batch) & {2, 3, 4, 5}) == 1 for batch in batches)


def test_balanced_batch_sampler_oversamples_recorded_false_stops_without_dropping_regular_negatives():
    regular_negatives = set(range(20, 32))
    priority_negatives = {40, 41}
    sampler = _BalancedStopBatchSampler(
        list(range(8)),
        sorted(regular_negatives | priority_negatives),
        batch_size=4,
        seed=17,
        priority_negative_indices=sorted(priority_negatives),
        priority_negative_fraction=0.25,
    )

    batches = list(sampler)
    sampled_negatives = [
        index
        for batch in batches
        for index in batch
        if index in regular_negatives or index in priority_negatives
    ]

    assert regular_negatives.issubset(sampled_negatives)
    assert priority_negatives.issubset(sampled_negatives)
    assert sum(index in priority_negatives for index in sampled_negatives) == math.ceil(
        len(sampled_negatives) * 0.25
    )
    assert all(sum(index < 20 for index in batch) == 2 for batch in batches)


def test_balanced_batch_sampler_keeps_false_stop_and_mined_negative_quotas_separate():
    positives = set(range(8))
    regular_negatives = set(range(20, 32))
    priority_negatives = {40, 41}
    mined_negatives = {50, 51, 52}
    all_negatives = regular_negatives | priority_negatives | mined_negatives
    sampler = _BalancedStopBatchSampler(
        sorted(positives),
        sorted(all_negatives),
        batch_size=4,
        seed=23,
        priority_negative_indices=sorted(priority_negatives),
        priority_negative_fraction=0.25,
        mined_negative_indices=sorted(mined_negatives),
        mined_negative_fraction=0.25,
    )

    batches = list(sampler)
    sampled_negatives = [
        index for batch in batches for index in batch if index in all_negatives
    ]

    assert regular_negatives.issubset(sampled_negatives)
    assert priority_negatives.issubset(sampled_negatives)
    assert mined_negatives.issubset(sampled_negatives)
    assert sum(index in priority_negatives for index in sampled_negatives) == math.ceil(
        len(sampled_negatives) * 0.25
    )
    assert sum(index in mined_negatives for index in sampled_negatives) == math.ceil(
        len(sampled_negatives) * 0.25
    )
    assert all(sum(index in positives for index in batch) == 2 for batch in batches)


def test_rollout_policy_coverage_requires_real_false_stop_negatives():
    complete = SimpleNamespace(
        targets=(1, 1, 0, 0),
        original_terminals=(False, True, False, True),
    )
    counts = _require_rollout_policy_coverage(complete, split_name="validation")
    assert counts == {
        "add_positive": 1,
        "regular_negative": 1,
        "false_stop_negative": 1,
        "original_correct_stop": 1,
    }

    missing_veto = SimpleNamespace(
        targets=(1, 0),
        original_terminals=(False, False),
    )
    with pytest.raises(RuntimeError, match="false_stop_negative"):
        _require_rollout_policy_coverage(missing_veto, split_name="validation")

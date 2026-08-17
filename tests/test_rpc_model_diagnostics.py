import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("MACA_HOME", "/opt/maca-3.3.0")
os.environ.setdefault("MACA_PATH", os.environ["MACA_HOME"])
os.environ.setdefault("MACA_DIR", os.environ["MACA_PATH"])

from scripts.evaluation.rpc_model_server import (
    _augment_system2_stop_feature_with_trajectory,
    _fallback_replan_action,
    _load_system2_stop_decision_adapter,
    _load_system2_stop_head,
    _load_system2_temporal_stop_verifier,
    _project_trajectory_condition,
    _StructuredViewPrefixLogitsProcessor,
    _system2_decision_scores,
    _system2_generation_decision_hidden,
    _system2_hybrid_stop_decision,
    _system2_non_stop_output_or_fallback,
    _system2_stop_bad_words_ids,
    _system2_stop_decision_adapter_probe,
    _system2_stop_head_decision,
    _system2_stop_hidden_alignment,
    _system2_stop_probability,
    _system2_temporal_stop_decision,
    _trajectory_debug_metrics,
    _validate_system2_force_non_stop_request,
    _validate_system2_stop_threshold_overrides,
    _write_system2_stop_feature,
)

from src.models.action import StopPredictionHead
from src.models.action.temporal_stop_verifier import (
    TEMPORAL_STOP_FEATURE_NAMES,
    TEMPORAL_STOP_FEATURE_SCHEMA,
    TemporalStopVerifier,
    TemporalStopVerifierEnsemble,
)


class _StructuredTokenizer:
    _ids = {
        "stop": 30,
        "front": 31,
        "left": 32,
        "right": 33,
        "back": 34,
        "turn": 35,
    }

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [10, 11, self._ids[text.removeprefix("view: ")]]


class _FakeStopDecisionIntegration:
    def __init__(self):
        self.active = ("default",)
        self.add_kwargs = None
        self.loaded_state = None
        self.model = torch.nn.Module()

    def lora_adapter_fingerprint(self, adapter_name):
        return "default-fingerprint" if adapter_name == "default" else "stop-fingerprint"

    def structured_view_token_contract(self):
        tokenizer = _StructuredTokenizer()
        classes = ["stop", "front", "right", "back", "left", "turn"]
        return {
            "schema": "heatmapvln-structured-view-token-contract-v1",
            "classes": classes,
            "prefix_token_ids": [10, 11],
            "class_token_ids": [tokenizer._ids[name] for name in classes],
            "patterns": {
                name: [10, 11, tokenizer._ids[name]] for name in classes
            },
        }

    def add_stop_decision_adapter(self, **kwargs):
        self.add_kwargs = kwargs
        return 12

    def load_lora_adapter_state_dict(self, adapter_name, state_dict):
        assert adapter_name == "stop_decision"
        self.loaded_state = state_dict
        return len(state_dict)

    def activate_lora_adapters(self, adapter_names, *, trainable_adapters=()):
        assert not trainable_adapters
        self.active = tuple(adapter_names)

    def active_lora_adapters(self):
        return self.active

    def _forward_model_inputs(self, inputs, **kwargs):
        assert self.active == ("default", "stop_decision")
        assert kwargs == {
            "return_hidden_states": True,
            "skip_lm_head": True,
            "return_last_hidden_state_only": True,
            "extract_vision_hidden_states": False,
        }
        assert inputs["input_ids"].tolist() == [[1, 2, 10, 11]]
        assert inputs["attention_mask"].tolist() == [[1, 1, 1, 1]]
        return torch.zeros(1, 4, 3), None, 0, None, None

    def structured_view_class_logits(self, hidden, positions):
        assert hidden.shape == (1, 4, 3)
        assert positions.tolist() == [3]
        return torch.tensor([[3.0, 1.0, 0.0, -1.0, -2.0, -3.0]])


def _stop_decision_checkpoint(integration):
    return {
        "schema": "heatmapvln-system2-stop-decision-adapter-v1",
        "adapter_name": "stop_decision",
        "adapter_state_dict": {"layer.lora_A.weight": torch.ones(2, 3)},
        "adapter_fingerprint": "stop-fingerprint",
        "adapter_config": {
            "rank": 2,
            "alpha": 4,
            "layer_indices": [20, 21],
            "target_modules": ["q_proj"],
            "dropout": 0.0,
        },
        "base_contract": {
            "checkpoint": "/tmp/base.pth",
            "default_adapter_name": "default",
            "default_lora_tensors": 224,
            "default_lora_fingerprint": "default-fingerprint",
        },
        "token_contract": integration.structured_view_token_contract(),
        "thresholds": {
            "add_stop_threshold": 0.9,
            "veto_stop_threshold": 0.2,
            "quality_passed": True,
            "quality_violations": [],
            "roc_auc": 0.9,
            "veto_reference_positive_count": 10,
            "add": {"recall": 0.6, "false_positive_rate": 0.0},
            "veto": {"recall": 1.0, "negative_rejection_rate": 0.8},
        },
        "training": {
            "holdout_scene_fraction": 0.1,
            "ranking_loss_weight": 1.0,
        },
    }


def test_stop_decision_adapter_load_is_exact_and_restores_navigation_lora(tmp_path):
    integration = _FakeStopDecisionIntegration()
    checkpoint_path = tmp_path / "stop_decision.pth"
    torch.save(_stop_decision_checkpoint(integration), checkpoint_path)

    metadata = _load_system2_stop_decision_adapter(
        str(checkpoint_path),
        integration=integration,
        expected_base_checkpoint="/tmp/base.pth",
    )

    assert integration.active == ("default",)
    assert integration.add_kwargs["layer_indices"] == [20, 21]
    assert metadata["adapter_tensors"] == 1
    assert metadata["policy_kind"] == "add_and_veto"
    assert metadata["add_enabled"] is True
    assert metadata["add_stop_threshold"] == pytest.approx(0.9)
    assert metadata["veto_stop_threshold"] == pytest.approx(0.2)


def test_stop_decision_adapter_rejects_nonzero_add_false_positive_rate(tmp_path):
    integration = _FakeStopDecisionIntegration()
    checkpoint = _stop_decision_checkpoint(integration)
    checkpoint["thresholds"]["add"]["false_positive_rate"] = 0.005
    checkpoint_path = tmp_path / "unsafe_stop_decision.pth"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match="calibration metrics are below contract"):
        _load_system2_stop_decision_adapter(
            str(checkpoint_path),
            integration=integration,
            expected_base_checkpoint="/tmp/base.pth",
        )


def test_stop_decision_veto_only_adapter_disables_addition_by_contract(tmp_path):
    integration = _FakeStopDecisionIntegration()
    checkpoint = _stop_decision_checkpoint(integration)
    checkpoint["policy_kind"] = "veto_only"
    checkpoint["thresholds"].update(
        {
            "policy_kind": "veto_only",
            "add_enabled": False,
            "add_stop_threshold": 1.0,
            "quality_passed": True,
        }
    )
    checkpoint["thresholds"]["add"] = {
        "recall": 0.0,
        "false_positive_rate": 0.0,
    }
    checkpoint_path = tmp_path / "veto_only_stop_decision.pth"
    torch.save(checkpoint, checkpoint_path)

    metadata = _load_system2_stop_decision_adapter(
        str(checkpoint_path),
        integration=integration,
        expected_base_checkpoint="/tmp/base.pth",
    )

    assert metadata["policy_kind"] == "veto_only"
    assert metadata["add_enabled"] is False
    assert metadata["add_stop_threshold"] == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match="forbids add threshold overrides"):
        _load_system2_stop_decision_adapter(
            str(checkpoint_path),
            integration=_FakeStopDecisionIntegration(),
            expected_base_checkpoint="/tmp/base.pth",
            add_threshold_override=0.9,
        )


def test_stop_decision_adapter_probe_uses_extra_lora_only_for_scoring():
    integration = _FakeStopDecisionIntegration()

    result = _system2_stop_decision_adapter_probe(
        qwen_integration=integration,
        inputs={
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        },
    )

    assert integration.active == ("default",)
    assert result["selected"] == "stop"
    assert result["stop_probability"] > 0.8
    assert sum(result["class_probabilities"].values()) == pytest.approx(1.0)


def test_trajectory_debug_metrics_are_structured_and_scaled():
    trajectory = torch.tensor(
        [
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[0.0, 2.0, 0.0], [2.0, 0.0, 0.0]],
        ]
    )

    metrics = _trajectory_debug_metrics(
        trajectory,
        num_sample_trajs=2,
        action_scale=2.0,
        trajectory_x_sign=-1.0,
    )

    assert metrics == pytest.approx(
        {
            "goal_x_m": -1.0,
            "goal_y_m": 1.0,
            "direct_m": 2.0**0.5,
            "path_len_m": 2.0**0.5,
        }
    )


def test_project_trajectory_condition_does_not_project_768_twice():
    class _Projector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            return value[..., :4]

    projector = _Projector()
    action_head = SimpleNamespace(
        config=SimpleNamespace(latent_emb_size=4),
        cond_projector=projector,
    )
    projected = torch.zeros(1, 3, 4)

    assert _project_trajectory_condition(action_head, projected) is projected
    assert projector.calls == 0

    raw = torch.zeros(1, 3, 8)
    result = _project_trajectory_condition(action_head, raw)
    assert result.shape == (1, 3, 4)
    assert projector.calls == 1


def test_structured_view_generation_rejects_out_of_protocol_class():
    processor = _StructuredViewPrefixLogitsProcessor(
        tokenizer=_StructuredTokenizer(),
        prompt_len=2,
    )
    scores = torch.zeros(1, 64)

    first = processor(torch.tensor([[90, 91]]), scores)
    assert first.argmax(dim=-1).item() == 10
    assert torch.isneginf(first[0, 11])

    second = processor(torch.tensor([[90, 91, 10]]), scores)
    assert second.argmax(dim=-1).item() == 11

    class_scores = scores.clone()
    class_scores[0, 40] = 100.0  # Simulates the observed invalid `view: side`.
    class_scores[0, _StructuredTokenizer._ids["front"]] = 5.0
    decision = processor(torch.tensor([[90, 91, 10, 11]]), class_scores)
    assert decision.argmax(dim=-1).item() == _StructuredTokenizer._ids["front"]
    assert torch.isneginf(decision[0, 40])

    unconstrained = processor(
        torch.tensor([[90, 91, 10, 11, _StructuredTokenizer._ids["front"]]]),
        class_scores,
    )
    assert unconstrained is class_scores


def test_structured_view_generation_can_exclude_stop_class():
    processor = _StructuredViewPrefixLogitsProcessor(
        tokenizer=_StructuredTokenizer(),
        prompt_len=2,
        excluded_labels=("stop",),
    )
    scores = torch.zeros(1, 64)
    scores[0, _StructuredTokenizer._ids["stop"]] = 100.0
    scores[0, _StructuredTokenizer._ids["right"]] = 5.0

    decision = processor(torch.tensor([[90, 91, 10, 11]]), scores)

    assert decision.argmax(dim=-1).item() == _StructuredTokenizer._ids["right"]
    assert torch.isneginf(decision[0, _StructuredTokenizer._ids["stop"]])


def test_structured_view_generation_rejects_invalid_exclusions():
    with pytest.raises(ValueError, match="Unknown structured view labels"):
        _StructuredViewPrefixLogitsProcessor(
            tokenizer=_StructuredTokenizer(),
            prompt_len=2,
            excluded_labels=("side",),
        )
    with pytest.raises(ValueError, match="At least one structured view label"):
        _StructuredViewPrefixLogitsProcessor(
            tokenizer=_StructuredTokenizer(),
            prompt_len=2,
            excluded_labels=("stop", "front", "left", "right", "back", "turn"),
        )


@pytest.mark.parametrize(
    ("output", "call_index", "expected", "fallback"),
    [
        ("view: front\npixel: 10 20", 2, "view: front\npixel: 10 20", False),
        ("view: turn_left", 3, "view: turn_left", False),
        ("view: stop", 2, "view: turn_left", True),
        ("malformed", 3, "view: turn_right", True),
    ],
)
def test_non_stop_output_fallback_is_valid_and_deterministic(
    output,
    call_index,
    expected,
    fallback,
):
    sanitized, used_fallback = _system2_non_stop_output_or_fallback(
        output,
        system2_call_index=call_index,
    )

    assert sanitized == expected
    assert used_fallback is fallback
    assert "stop" not in sanitized


def test_system2_decision_scores_normalize_structured_classes():
    tokenizer = _StructuredTokenizer()
    scores = [torch.zeros(1, 64) for _ in range(3)]
    class_logits = {
        "stop": 3.0,
        "front": 1.0,
        "left": 0.0,
        "right": -1.0,
        "back": -2.0,
        "turn": -3.0,
    }
    for label, value in class_logits.items():
        scores[2][0, tokenizer._ids[label]] = value
    sequence = torch.tensor([[90, 91, 10, 11, tokenizer._ids["stop"]]])

    result = _system2_decision_scores(
        tokenizer=tokenizer,
        sequence=sequence,
        prompt_len=2,
        generation_scores=scores,
    )

    assert result["selected"] == "stop"
    assert sum(result["class_probabilities"].values()) == pytest.approx(1.0)
    assert result["class_probabilities"]["stop"] > 0.8
    assert result["stop_log_odds"] > 1.0


def test_force_non_stop_uses_only_the_structured_stop_class_token():
    assert _system2_stop_bad_words_ids(_StructuredTokenizer()) == [[30]]


def test_temporal_stop_policy_is_strictly_veto_only():
    assert _system2_temporal_stop_decision(
        verifier_probability=1.0,
        acceptance_threshold=0.5,
        original_output="view: front\npixel: 128 128",
    ) == "temporal_keeps_original_non_stop"
    assert _system2_temporal_stop_decision(
        verifier_probability=0.7,
        acceptance_threshold=0.5,
        original_output="view: stop",
    ) == "temporal_confirms_original_stop"
    assert _system2_temporal_stop_decision(
        verifier_probability=0.2,
        acceptance_threshold=0.5,
        original_output="view: stop",
    ) == "temporal_requests_stop_veto"


def test_hybrid_stop_policy_keeps_add_and_veto_roles_disjoint():
    assert _system2_hybrid_stop_decision(
        original_output="view: stop",
        temporal_decision="temporal_vetoes_original_stop",
        static_add_decision="head_adds_stop",
    ) == "temporal_vetoes_original_stop"
    assert _system2_hybrid_stop_decision(
        original_output="view: front\npixel: 100 128",
        temporal_decision="temporal_keeps_original_non_stop",
        static_add_decision="head_adds_stop",
    ) == "hybrid_static_adds_stop"
    assert _system2_hybrid_stop_decision(
        original_output="view: front\npixel: 100 128",
        temporal_decision="temporal_keeps_original_non_stop",
        static_add_decision="head_keeps_original_non_stop",
    ) == "hybrid_keeps_original_non_stop"


def test_hybrid_stop_policy_allows_static_add_threshold_override():
    _validate_system2_stop_threshold_overrides(
        static_head_enabled=True,
        temporal_verifier_enabled=True,
        add_threshold_override=0.98,
        veto_threshold_override=None,
    )


def test_temporal_stop_policy_rejects_static_veto_threshold_override():
    with pytest.raises(ValueError, match="temporal verifier owns veto decisions"):
        _validate_system2_stop_threshold_overrides(
            static_head_enabled=True,
            temporal_verifier_enabled=True,
            add_threshold_override=0.98,
            veto_threshold_override=0.2,
        )


def test_pure_temporal_stop_policy_rejects_static_add_threshold_override():
    with pytest.raises(ValueError, match="requires a static STOP head"):
        _validate_system2_stop_threshold_overrides(
            static_head_enabled=False,
            temporal_verifier_enabled=True,
            add_threshold_override=0.98,
            veto_threshold_override=None,
        )


def test_static_stop_policy_allows_static_threshold_overrides():
    _validate_system2_stop_threshold_overrides(
        static_head_enabled=True,
        temporal_verifier_enabled=False,
        add_threshold_override=0.98,
        veto_threshold_override=0.2,
    )


def test_temporal_stop_checkpoint_loads_embedded_frozen_prior(tmp_path):
    static_head = StopPredictionHead(input_dim=8, hidden_dim=4, dropout=0.0)
    verifier = TemporalStopVerifier(
        feature_mean=torch.zeros(len(TEMPORAL_STOP_FEATURE_NAMES)),
        feature_scale=torch.ones(len(TEMPORAL_STOP_FEATURE_NAMES)),
        hidden_dim=4,
        dropout=0.0,
    )
    checkpoint_path = tmp_path / "temporal.pth"
    torch.save(
        {
            "stage_name": "system2_temporal_stop_verifier",
            "config": {
                "temporal_stop_verifier": {
                    "schema": TEMPORAL_STOP_FEATURE_SCHEMA,
                    "feature_names": list(TEMPORAL_STOP_FEATURE_NAMES),
                    "hidden_dim": 4,
                    "dropout": 0.0,
                    "acceptance_threshold": 0.65,
                    "veto_only": True,
                    "requires_contiguous_zero_based_calls": True,
                },
                "source_static_stop_head": {
                    "input_dim": 8,
                    "hidden_dim": 4,
                    "dropout": 0.0,
                    "focal_gamma": 2.0,
                    "focal_alpha": 0.5,
                    "pos_weight": 1.0,
                    "bce_mix": 0.5,
                },
            },
            "trainable_state_dict": {
                f"temporal_stop_verifier.{name}": value
                for name, value in verifier.state_dict().items()
            },
            "source_static_stop_head_state_dict": {
                f"stop_head.{name}": value
                for name, value in static_head.state_dict().items()
            },
        },
        checkpoint_path,
    )

    loaded_static, loaded_verifier, threshold = _load_system2_temporal_stop_verifier(
        str(checkpoint_path),
        device=torch.device("cpu"),
    )

    assert threshold == pytest.approx(0.65)
    assert loaded_static.training is False
    assert loaded_verifier.training is False
    assert not any(parameter.requires_grad for parameter in loaded_static.parameters())
    assert not any(parameter.requires_grad for parameter in loaded_verifier.parameters())


def test_temporal_stop_ensemble_checkpoint_loads_exact_members(tmp_path):
    static_head = StopPredictionHead(input_dim=8, hidden_dim=4, dropout=0.0)
    dimension = len(TEMPORAL_STOP_FEATURE_NAMES)
    ensemble = TemporalStopVerifierEnsemble(
        [
            TemporalStopVerifier(
                feature_mean=torch.full((dimension,), float(index)),
                feature_scale=torch.ones(dimension),
                hidden_dim=4,
                dropout=0.0,
            )
            for index in range(2)
        ],
        torch.tensor([0.55, 0.7]),
    )
    checkpoint_path = tmp_path / "temporal_ensemble.pth"
    torch.save(
        {
            "stage_name": "system2_temporal_stop_verifier_ensemble",
            "config": {
                "temporal_stop_verifier": {
                    "schema": TEMPORAL_STOP_FEATURE_SCHEMA,
                    "feature_names": list(TEMPORAL_STOP_FEATURE_NAMES),
                    "architecture": "scene_fold_unanimous_ensemble",
                    "ensemble_size": 2,
                    "member_hidden_dim": 4,
                    "member_dropout": 0.0,
                    "acceptance_thresholds": [0.55, 0.7],
                    "aggregation": "unanimous",
                    "veto_only": True,
                    "requires_contiguous_zero_based_calls": True,
                },
                "source_static_stop_head": {
                    "input_dim": 8,
                    "hidden_dim": 4,
                    "dropout": 0.0,
                    "focal_gamma": 2.0,
                    "focal_alpha": 0.5,
                    "pos_weight": 1.0,
                    "bce_mix": 0.5,
                },
            },
            "trainable_state_dict": {
                f"temporal_stop_ensemble.{name}": value
                for name, value in ensemble.state_dict().items()
            },
            "source_static_stop_head_state_dict": {
                f"stop_head.{name}": value
                for name, value in static_head.state_dict().items()
            },
        },
        checkpoint_path,
    )

    loaded_static, loaded_ensemble, threshold = (
        _load_system2_temporal_stop_verifier(
            str(checkpoint_path),
            device=torch.device("cpu"),
        )
    )

    assert threshold is None
    assert isinstance(loaded_ensemble, TemporalStopVerifierEnsemble)
    assert loaded_ensemble.acceptance_thresholds.tolist() == pytest.approx([0.55, 0.7])
    assert loaded_static.training is False
    assert loaded_ensemble.training is False
    assert not any(parameter.requires_grad for parameter in loaded_ensemble.parameters())


def test_force_non_stop_is_restricted_to_unmodified_dagger_collection():
    assert _validate_system2_force_non_stop_request(
        force_non_stop=True,
        feature_dump_enabled=True,
        stop_head_enabled=False,
        oracle_system2_enabled=False,
    ) is True
    assert _validate_system2_force_non_stop_request(
        force_non_stop=False,
        feature_dump_enabled=False,
        stop_head_enabled=True,
        oracle_system2_enabled=True,
    ) is False

    invalid = (
        ({"force_non_stop": "true"}, "must be a boolean"),
        ({"feature_dump_enabled": False}, "restricted to STOP feature collection"),
        ({"stop_head_enabled": True}, "unmodified original System2 policy"),
        ({"oracle_system2_enabled": True}, "cannot be combined with oracle"),
    )
    common = {
        "force_non_stop": True,
        "feature_dump_enabled": True,
        "stop_head_enabled": False,
        "oracle_system2_enabled": False,
    }
    for override, message in invalid:
        with pytest.raises(ValueError, match=message):
            _validate_system2_force_non_stop_request(**{**common, **override})


def test_stop_head_uses_hidden_state_before_class_token():
    hidden_steps = (
        (torch.full((1, 4, 8), 0.0), torch.full((1, 4, 8), 0.1)),
        (torch.full((1, 1, 8), 1.0), torch.full((1, 1, 8), 1.1)),
        (torch.full((1, 1, 8), 2.0), torch.full((1, 1, 8), 2.1)),
    )
    generation = SimpleNamespace(
        sequences=torch.tensor([[90, 91, 10, 11, 31]]),
        hidden_states=hidden_steps,
    )

    hidden = _system2_generation_decision_hidden(
        generation=generation,
        tokenizer=_StructuredTokenizer(),
        prompt_len=2,
    )

    assert hidden.shape == (1, 8)
    assert torch.equal(hidden, torch.full((1, 8), 2.1))


def test_stop_hidden_alignment_reports_training_generation_drift():
    generated = torch.tensor([[1.0, 0.0, 1.0]])
    teacher_forced = torch.tensor([[1.0, 0.0, 0.0]])

    result = _system2_stop_hidden_alignment(generated, teacher_forced)

    assert result["cosine_mean"] == pytest.approx(2**-0.5)
    assert result["max_abs_error"] == pytest.approx(1.0)
    assert result["mean_abs_error"] == pytest.approx(1.0 / 3.0)


def test_stop_hidden_alignment_rejects_shape_mismatch():
    with pytest.raises(RuntimeError, match="shape mismatch"):
        _system2_stop_hidden_alignment(torch.zeros(1, 4), torch.zeros(1, 5))


def test_stop_feature_dump_is_atomic_and_path_sanitized(tmp_path):
    result = _write_system2_stop_feature(
        tmp_path,
        decision_hidden=torch.arange(8, dtype=torch.float32).reshape(1, 8),
        sampling_metadata={
            "scene_id": "mp3d/scene-with-space",
            "episode_id": 7,
            "system2_call_index": 3,
            "protocol_seed": 42,
        },
        original_output="view: stop",
        decision_scores={"selected": "stop"},
    )

    path = tmp_path / f"{result['key']}.pth"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert path.is_file()
    assert path.parent == tmp_path
    assert "/" not in result["key"]
    assert result["hidden_dim"] == 8
    assert saved["schema"] == "heatmapvln-system2-stop-feature-v1"
    assert saved["collection_namespace"] == result["collection_namespace"]
    assert saved["collection_root"] == str(tmp_path.resolve().parent)
    assert torch.equal(saved["feature"], torch.arange(8, dtype=torch.float32))
    assert not list(tmp_path.glob("*.tmp"))


def test_stop_feature_key_is_namespaced_by_collection_root(tmp_path):
    metadata = {
        "scene_id": "scene",
        "episode_id": 7,
        "system2_call_index": 3,
        "protocol_seed": 42,
    }
    first = _write_system2_stop_feature(
        tmp_path / "first" / "system2_stop_features",
        decision_hidden=torch.zeros(1, 8),
        sampling_metadata=metadata,
        original_output="view: front\npixel: 128 128",
        decision_scores={"selected": "front"},
    )
    second = _write_system2_stop_feature(
        tmp_path / "second" / "system2_stop_features",
        decision_hidden=torch.ones(1, 8),
        sampling_metadata=metadata,
        original_output="view: front\npixel: 128 128",
        decision_scores={"selected": "front"},
    )

    assert first["key"] != second["key"]
    assert first["collection_namespace"] != second["collection_namespace"]
    assert first["key"].endswith("scene_ep000007_call00003_seed42")


def test_stop_feature_trajectory_augmentation_is_atomic(tmp_path):
    record = _write_system2_stop_feature(
        tmp_path,
        decision_hidden=torch.arange(8, dtype=torch.float32).reshape(1, 8),
        sampling_metadata={
            "scene_id": "scene",
            "episode_id": 7,
            "system2_call_index": 3,
            "protocol_seed": 42,
        },
        original_output="view: front\npixel: 128 128",
        decision_scores={"selected": "front"},
    )

    augmented = _augment_system2_stop_feature_with_trajectory(
        record,
        raw_traj_latent=torch.arange(24, dtype=torch.float32).reshape(1, 3, 8),
        adapted_traj_latent=torch.arange(24, dtype=torch.float32).reshape(1, 3, 8) + 1,
        projected_traj_condition=torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
        trajectory=torch.arange(36, dtype=torch.float32).reshape(2, 6, 3),
        trajectory_metrics={
            "goal_x_m": 1.0,
            "goal_y_m": 2.0,
            "direct_m": 3.0,
            "path_len_m": 4.0,
        },
        local_actions=[1, 2],
        pixel_goal=(128, 96),
        pano_goal_view="left",
    )

    path = Path(record["path"])
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["feature"].shape == (8,)
    assert saved["raw_traj_latent"].shape == (3, 8)
    assert saved["adapted_traj_latent"].shape == (3, 8)
    assert saved["projected_traj_condition"].shape == (3, 4)
    assert saved["trajectory"].shape == (2, 6, 3)
    assert saved["trajectory_metrics"]["path_len_m"] == pytest.approx(4.0)
    assert saved["local_actions"] == [1, 2]
    assert saved["pixel_goal"] == [128, 96]
    assert saved["pano_goal_view"] == "left"
    assert augmented["trajectory_feature_schema"].endswith("-v1")
    assert augmented["raw_traj_latent_shape"] == [3, 8]
    assert augmented["adapted_traj_latent_shape"] == [3, 8]
    assert augmented["projected_traj_condition_shape"] == [3, 4]
    assert not list(tmp_path.glob("*.tmp"))


def test_stop_feature_trajectory_augmentation_rejects_non_finite(tmp_path):
    record = _write_system2_stop_feature(
        tmp_path,
        decision_hidden=torch.zeros(1, 8),
        sampling_metadata={
            "scene_id": "scene",
            "episode_id": 7,
            "system2_call_index": 3,
            "protocol_seed": 42,
        },
        original_output="view: front\npixel: 128 128",
        decision_scores={"selected": "front"},
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        _augment_system2_stop_feature_with_trajectory(
            record,
            raw_traj_latent=torch.full((1, 3, 8), float("nan")),
            adapted_traj_latent=torch.zeros(1, 3, 8),
            projected_traj_condition=torch.zeros(1, 3, 4),
            trajectory=torch.zeros(2, 6, 3),
            trajectory_metrics={"direct_m": 1.0},
            local_actions=[1],
            pixel_goal=(128, 96),
            pano_goal_view="front",
        )


@pytest.mark.parametrize(
    ("probability", "original", "constrained", "expected"),
    [
        (0.8, "view: stop", None, "head_confirms_original_stop"),
        (0.8, "view: front\npixel: 100 128", None, "head_adds_stop"),
        (0.6, "view: stop", None, "head_confirms_original_stop"),
        (0.6, "view: front\npixel: 100 128", None, "head_keeps_original_non_stop"),
        (0.2, "view: front\npixel: 100 128", None, "head_keeps_original_non_stop"),
        (0.2, "view: stop", None, "head_requests_stop_veto"),
        (0.2, "view: stop", "view: left\npixel: 180 128", "head_vetoes_stop"),
        (0.2, "view: stop", "invalid output", "head_veto_fallback_replan"),
    ],
)
def test_system2_stop_head_preserves_original_waypoint_policy(
    probability,
    original,
    constrained,
    expected,
):
    assert _system2_stop_head_decision(
        stop_probability=probability,
        add_stop_threshold=0.7,
        veto_stop_threshold=0.3,
        original_output=original,
        constrained_output=constrained,
        image_size=(256, 256),
    ) == expected


def test_malformed_non_stop_output_falls_back_to_replan_not_stop():
    assert _fallback_replan_action("front") == 2
    assert _fallback_replan_action("left") == 2
    assert _fallback_replan_action("right") == 3
    assert _fallback_replan_action("back") == 3


def test_stop_head_add_requires_qwen_corroboration_when_enabled():
    common = {
        "stop_probability": 0.95,
        "add_stop_threshold": 0.9,
        "veto_stop_threshold": 0.55,
        "original_output": "view: front\npixel: 100 128",
        "add_min_qwen_stop_probability": 1e-4,
    }

    assert _system2_stop_head_decision(
        **common,
        original_stop_probability=1e-8,
    ) == "head_rejects_uncorroborated_stop"
    assert _system2_stop_head_decision(
        **common,
        original_stop_probability=1e-3,
    ) == "head_adds_stop"


def test_veto_only_stop_decision_never_adds_even_at_probability_one():
    assert _system2_stop_head_decision(
        stop_probability=1.0,
        add_stop_threshold=1.0,
        veto_stop_threshold=0.2,
        original_output="view: front\npixel: 100 128",
        allow_add_stop=False,
    ) == "head_keeps_original_non_stop"


def test_stop_head_probability_is_not_sigmoided_twice():
    class ProbabilityHead(torch.nn.Module):
        def forward(self, _hidden):
            return torch.tensor([0.9])

    probability = _system2_stop_probability(
        ProbabilityHead(),
        torch.zeros(1, 8),
    )

    assert probability == pytest.approx(0.9)


def test_stop_head_checkpoint_loader_requires_exact_isolated_state(tmp_path):
    head = StopPredictionHead(input_dim=8, hidden_dim=4, dropout=0.0)
    checkpoint = {
        "config": {
            "model": {
                "llm": {"hidden_dim": 8},
                "stop_head": {
                    "enabled": True,
                    "hidden_dim": 4,
                    "dropout": 0.0,
                    "add_stop_threshold": 0.9,
                    "veto_stop_threshold": 0.2,
                },
            }
        },
        "trainable_state_dict": {
            f"stop_head.{name}": value
            for name, value in head.state_dict().items()
        },
    }
    path = tmp_path / "head.pth"
    torch.save(checkpoint, path)

    loaded, add_threshold, veto_threshold = _load_system2_stop_head(
        str(path),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert add_threshold == pytest.approx(0.9)
    assert veto_threshold == pytest.approx(0.2)
    assert set(loaded.state_dict()) == set(head.state_dict())
    assert {parameter.dtype for parameter in loaded.parameters()} == {torch.float32}
    assert all(not parameter.requires_grad for parameter in loaded.parameters())

    checkpoint["trainable_state_dict"]["pano_latent_adapter.bad"] = torch.zeros(1)
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="non-head trainable tensors"):
        _load_system2_stop_head(
            str(path),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

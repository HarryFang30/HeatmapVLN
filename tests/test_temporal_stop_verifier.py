import math

import pytest
import torch

from src.models.action.temporal_stop_verifier import (
    TEMPORAL_STOP_FEATURE_NAMES,
    TemporalStopEpisodeHistory,
    TemporalStopObservation,
    TemporalStopVerifier,
    TemporalStopVerifierEnsemble,
    build_temporal_stop_features,
    probability_to_logit,
)


def _observation(call_index, value, static_probability=0.5, qwen_log_odds=0.0):
    return TemporalStopObservation(
        call_index=call_index,
        hidden=torch.tensor([value, value + 1.0, value + 2.0]),
        static_stop_probability=static_probability,
        qwen_stop_log_odds=qwen_log_odds,
    )


def test_temporal_features_have_stable_named_contract():
    observations = [
        _observation(0, 0.0, 0.2, -2.0),
        _observation(1, 1.0, 0.8, 1.0),
        _observation(2, 2.0, 0.9, 3.0),
    ]

    features = build_temporal_stop_features(observations)
    values = dict(zip(TEMPORAL_STOP_FEATURE_NAMES, features.tolist()))

    assert features.shape == (len(TEMPORAL_STOP_FEATURE_NAMES),)
    assert torch.isfinite(features).all()
    assert values["static_logit_current"] == pytest.approx(probability_to_logit(0.9))
    assert values["static_logit_delta_prev1"] == pytest.approx(
        probability_to_logit(0.9) - probability_to_logit(0.8)
    )
    assert values["qwen_stop_log_odds_delta_first"] == pytest.approx(5.0)
    assert values["has_prev1"] == 1.0
    assert values["has_prev2"] == 1.0
    assert values["log1p_call_index"] == pytest.approx(math.log(3.0))


def test_temporal_first_call_uses_neutral_history_fallbacks():
    features = build_temporal_stop_features([_observation(0, 2.0, 0.7, 1.5)])
    values = dict(zip(TEMPORAL_STOP_FEATURE_NAMES, features.tolist()))

    assert values["static_logit_delta_prev1"] == pytest.approx(0.0)
    assert values["qwen_stop_log_odds_delta_prev2"] == pytest.approx(0.0)
    assert values["hidden_cosine_prev1"] == pytest.approx(1.0)
    assert values["hidden_rms_delta_first"] == pytest.approx(0.0)
    assert values["has_prev1"] == 0.0
    assert values["has_prev2"] == 0.0


def test_temporal_history_resets_only_on_zero_call():
    history = TemporalStopEpisodeHistory()
    first = history.observe(
        episode_key=("scene-a", 1, 42),
        observation=_observation(0, 0.0),
    )
    second = history.observe(
        episode_key=("scene-a", 1, 42),
        observation=_observation(1, 1.0),
    )

    assert first.shape == second.shape
    assert history.length == 2
    with pytest.raises(RuntimeError, match="episode changed"):
        history.observe(
            episode_key=("scene-b", 2, 42),
            observation=_observation(2, 2.0),
        )
    with pytest.raises(RuntimeError, match="contiguous and unique"):
        history.observe(
            episode_key=("scene-a", 1, 42),
            observation=_observation(3, 3.0),
        )

    history.observe(
        episode_key=("scene-b", 2, 42),
        observation=_observation(0, 4.0),
    )
    assert history.episode_key == ("scene-b", 2, 42)
    assert history.length == 1


def test_temporal_feature_builder_rejects_noncontiguous_calls():
    with pytest.raises(ValueError, match="contiguous and zero-based"):
        build_temporal_stop_features(
            [_observation(0, 0.0), _observation(2, 1.0)]
        )


def test_temporal_verifier_standardizes_and_returns_probabilities():
    dimension = len(TEMPORAL_STOP_FEATURE_NAMES)
    verifier = TemporalStopVerifier(
        feature_mean=torch.zeros(dimension),
        feature_scale=torch.ones(dimension),
        hidden_dim=8,
        dropout=0.0,
    )
    probabilities = verifier(torch.zeros(3, dimension))

    assert probabilities.shape == (3,)
    assert torch.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_temporal_verifier_supports_explicit_content_dimension():
    verifier = TemporalStopVerifier(
        feature_mean=torch.zeros(19),
        feature_scale=torch.ones(19),
        hidden_dim=4,
        dropout=0.0,
        input_dim=19,
    )

    assert verifier(torch.ones(3, 19)).shape == (3,)


def test_temporal_ensemble_requires_unanimous_member_acceptance():
    dimension = len(TEMPORAL_STOP_FEATURE_NAMES)
    members = [
        TemporalStopVerifier(
            feature_mean=torch.zeros(dimension),
            feature_scale=torch.ones(dimension),
            hidden_dim=2,
            dropout=0.0,
        )
        for _ in range(2)
    ]
    with torch.no_grad():
        for parameter in members[0].parameters():
            parameter.zero_()
        for parameter in members[1].parameters():
            parameter.zero_()
        members[0].classifier[-1].bias.fill_(2.0)
        members[1].classifier[-1].bias.fill_(-2.0)
    ensemble = TemporalStopVerifierEnsemble(
        members,
        torch.tensor([0.5, 0.5]),
    )
    features = torch.zeros(3, dimension)

    probabilities = ensemble.member_probabilities(features)

    assert probabilities.shape == (3, 2)
    assert (probabilities[:, 0] > 0.5).all()
    assert (probabilities[:, 1] < 0.5).all()
    assert not ensemble.accepts(features).any()

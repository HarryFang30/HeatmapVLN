"""Non-privileged temporal verification for original System2 STOP requests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


TEMPORAL_STOP_FEATURE_SCHEMA = "heatmapvln-system2-temporal-stop-features-v1"


def _series_feature_names(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}_current",
        f"{prefix}_prev1",
        f"{prefix}_prev2",
        f"{prefix}_first",
        f"{prefix}_recent_mean",
        f"{prefix}_recent_min",
        f"{prefix}_recent_max",
        f"{prefix}_delta_prev1",
        f"{prefix}_delta_prev2",
        f"{prefix}_delta_first",
    )


TEMPORAL_STOP_FEATURE_NAMES = (
    *_series_feature_names("static_logit"),
    *_series_feature_names("qwen_stop_log_odds"),
    "hidden_rms_current",
    "hidden_rms_prev1",
    "hidden_rms_first",
    "hidden_cosine_prev1",
    "hidden_cosine_prev2",
    "hidden_cosine_first",
    "hidden_rms_delta_prev1",
    "hidden_rms_delta_prev2",
    "hidden_rms_delta_first",
    "log1p_call_index",
    "log1p_history_length",
    "has_prev1",
    "has_prev2",
)


@dataclass(frozen=True)
class TemporalStopObservation:
    call_index: int
    hidden: torch.Tensor
    static_stop_probability: float
    qwen_stop_log_odds: float

    def __post_init__(self) -> None:
        if self.call_index < 0:
            raise ValueError("Temporal STOP call_index must be >= 0")
        if not torch.is_tensor(self.hidden) or self.hidden.ndim != 1:
            raise ValueError("Temporal STOP hidden state must be a rank-1 tensor")
        if self.hidden.numel() == 0 or not bool(torch.isfinite(self.hidden.float()).all()):
            raise ValueError("Temporal STOP hidden state must be non-empty and finite")
        if not (
            math.isfinite(float(self.static_stop_probability))
            and 0.0 <= float(self.static_stop_probability) <= 1.0
        ):
            raise ValueError("Temporal STOP static probability must be finite and in [0, 1]")
        if not math.isfinite(float(self.qwen_stop_log_odds)):
            raise ValueError("Temporal STOP Qwen log-odds must be finite")


def probability_to_logit(probability: float, epsilon: float = 1e-6) -> float:
    probability = min(max(float(probability), epsilon), 1.0 - epsilon)
    return math.log(probability) - math.log1p(-probability)


def _series_features(values: Sequence[float]) -> list[float]:
    current = float(values[-1])
    prev1 = float(values[-2]) if len(values) >= 2 else current
    prev2 = float(values[-3]) if len(values) >= 3 else prev1
    first = float(values[0])
    recent = [float(value) for value in values[-4:]]
    return [
        current,
        prev1,
        prev2,
        first,
        sum(recent) / len(recent),
        min(recent),
        max(recent),
        current - prev1,
        current - prev2,
        current - first,
    ]


def _rms(tensor: torch.Tensor) -> float:
    return float(tensor.float().square().mean().sqrt().item())


def _cosine(current: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            current.float().unsqueeze(0),
            reference.float().unsqueeze(0),
            dim=-1,
            eps=1e-8,
        ).item()
    )


def _rms_delta(current: torch.Tensor, reference: torch.Tensor) -> float:
    return _rms(current.float() - reference.float())


def build_temporal_stop_features(
    observations: Sequence[TemporalStopObservation],
) -> torch.Tensor:
    """Build the exact compact feature vector used by training and RPC inference."""
    if not observations:
        raise ValueError("Temporal STOP features require at least one observation")
    hidden_dim = int(observations[0].hidden.numel())
    previous_call_index = -1
    for observation in observations:
        if int(observation.hidden.numel()) != hidden_dim:
            raise ValueError("Temporal STOP history contains inconsistent hidden dimensions")
        if observation.call_index != previous_call_index + 1:
            raise ValueError(
                "Temporal STOP history must be contiguous and zero-based: "
                f"expected={previous_call_index + 1} got={observation.call_index}"
            )
        previous_call_index = observation.call_index

    static_logits = [
        probability_to_logit(observation.static_stop_probability)
        for observation in observations
    ]
    qwen_log_odds = [float(observation.qwen_stop_log_odds) for observation in observations]
    current = observations[-1].hidden.detach().float().cpu().contiguous()
    prev1 = observations[-2].hidden if len(observations) >= 2 else current
    prev2 = observations[-3].hidden if len(observations) >= 3 else prev1
    first = observations[0].hidden

    values = [
        *_series_features(static_logits),
        *_series_features(qwen_log_odds),
        _rms(current),
        _rms(prev1),
        _rms(first),
        _cosine(current, prev1),
        _cosine(current, prev2),
        _cosine(current, first),
        _rms_delta(current, prev1),
        _rms_delta(current, prev2),
        _rms_delta(current, first),
        math.log1p(observations[-1].call_index),
        math.log1p(len(observations)),
        float(len(observations) >= 2),
        float(len(observations) >= 3),
    ]
    if len(values) != len(TEMPORAL_STOP_FEATURE_NAMES):
        raise RuntimeError(
            "Temporal STOP feature contract mismatch: "
            f"values={len(values)} names={len(TEMPORAL_STOP_FEATURE_NAMES)}"
        )
    features = torch.tensor(values, dtype=torch.float32)
    if not bool(torch.isfinite(features).all()):
        raise RuntimeError("Temporal STOP feature builder produced non-finite values")
    return features


class TemporalStopEpisodeHistory:
    """Maintain one deterministic episode history and reject state contamination."""

    def __init__(self) -> None:
        self._episode_key: tuple[str, int, int] | None = None
        self._observations: list[TemporalStopObservation] = []

    @property
    def episode_key(self) -> tuple[str, int, int] | None:
        return self._episode_key

    @property
    def length(self) -> int:
        return len(self._observations)

    def reset(self) -> None:
        self._episode_key = None
        self._observations.clear()

    def observe(
        self,
        *,
        episode_key: tuple[str, int, int],
        observation: TemporalStopObservation,
    ) -> torch.Tensor:
        normalized_key = (str(episode_key[0]), int(episode_key[1]), int(episode_key[2]))
        if observation.call_index == 0:
            self._episode_key = normalized_key
            self._observations = []
        elif self._episode_key != normalized_key:
            raise RuntimeError(
                "Temporal STOP episode changed without a zero-index reset: "
                f"current={self._episode_key} received={normalized_key} "
                f"call={observation.call_index}"
            )
        expected_call = len(self._observations)
        if observation.call_index != expected_call:
            raise RuntimeError(
                "Temporal STOP RPC calls must be contiguous and unique: "
                f"expected={expected_call} got={observation.call_index}"
            )
        self._observations.append(observation)
        return build_temporal_stop_features(self._observations)


class TemporalStopVerifier(nn.Module):
    """A tiny standardized MLP over compact temporal System2 features."""

    def __init__(
        self,
        *,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        hidden_dim: int = 16,
        dropout: float = 0.1,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        feature_mean = feature_mean.detach().float().flatten()
        feature_scale = feature_scale.detach().float().flatten()
        expected_dim = (
            len(TEMPORAL_STOP_FEATURE_NAMES) if input_dim is None else int(input_dim)
        )
        if expected_dim < 1:
            raise ValueError("Temporal STOP input_dim must be >= 1")
        if feature_mean.numel() != expected_dim or feature_scale.numel() != expected_dim:
            raise ValueError(
                "Temporal STOP normalization dimension mismatch: "
                f"mean={feature_mean.numel()} scale={feature_scale.numel()} "
                f"expected={expected_dim}"
            )
        if not bool(torch.isfinite(feature_mean).all()):
            raise ValueError("Temporal STOP feature mean must be finite")
        if not bool(torch.isfinite(feature_scale).all()) or bool((feature_scale <= 0).any()):
            raise ValueError("Temporal STOP feature scale must be finite and positive")
        if hidden_dim < 1:
            raise ValueError("Temporal STOP hidden_dim must be >= 1")
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_scale", feature_scale)
        self.classifier = nn.Sequential(
            nn.Linear(expected_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        expected_dim = int(self.feature_mean.numel())
        if features.shape[-1] != expected_dim:
            raise ValueError(
                "Temporal STOP input dimension mismatch: "
                f"got={features.shape[-1]} expected={expected_dim}"
            )
        normalized = (features.float() - self.feature_mean) / self.feature_scale
        return self.classifier(normalized).squeeze(-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(features))


class TemporalStopVerifierEnsemble(nn.Module):
    """Scene-fold verifier ensemble with a conservative unanimous decision."""

    def __init__(
        self,
        members: Sequence[TemporalStopVerifier],
        acceptance_thresholds: torch.Tensor,
    ) -> None:
        super().__init__()
        if len(members) < 2:
            raise ValueError("Temporal STOP ensemble requires at least two members")
        thresholds = acceptance_thresholds.detach().float().flatten()
        if thresholds.numel() != len(members):
            raise ValueError("Temporal STOP ensemble thresholds must match member count")
        if not bool(torch.isfinite(thresholds).all()) or bool(
            ((thresholds < 0.0) | (thresholds > 1.0)).any()
        ):
            raise ValueError("Temporal STOP ensemble thresholds must be in [0, 1]")
        self.members = nn.ModuleList(members)
        self.register_buffer("acceptance_thresholds", thresholds)

    def member_probabilities(self, features: torch.Tensor) -> torch.Tensor:
        return torch.stack([member(features) for member in self.members], dim=-1)

    def accepts(self, features: torch.Tensor) -> torch.Tensor:
        probabilities = self.member_probabilities(features)
        return (probabilities >= self.acceptance_thresholds).all(dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the minimum calibrated margin for logging, not a probability."""
        probabilities = self.member_probabilities(features)
        return (probabilities - self.acceptance_thresholds).min(dim=-1).values

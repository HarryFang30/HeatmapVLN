"""Checkpoint selection policies.

The default policy exactly matches the historical single-metric strict
comparison.  The optional constrained policy keeps a step-0 baseline as the
incumbent, protects natural-distribution/back-view quality, and then performs
lexicographic macro -> overall -> val-loss selection.  An optional
all-direction retention gate prevents the macro average from hiding a collapse
in one named direction.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


class BestCheckpointSelector:
    """Stateful, serializable checkpoint selector.

    Args:
        primary_metric: Validation metric used for the first comparison.
        primary_mode: ``"min"`` or ``"max"``.
        baseline_as_incumbent: Enable constrained selection and require a
            step-0 baseline before candidates are considered.
        overall_metric: Natural-distribution metric used both as a baseline
            gate and the first tie-breaker.
        overall_tolerance: Maximum absolute regression from step 0.
        back_metric: Back-direction metric protected by a second baseline gate.
        back_tolerance: Maximum absolute regression from step 0.
        direction_metrics: Optional ``label -> metric`` mapping.  Every listed
            direction must stay within ``direction_tolerance`` of step 0.
        loss_metric: Final, lower-is-better tie-breaker.
    """

    _EPSILON = 1e-12

    def __init__(
        self,
        *,
        primary_metric: str = "val_total_loss",
        primary_mode: str = "min",
        baseline_as_incumbent: bool = False,
        overall_metric: str = "val_heatmap_joint_pck8",
        overall_tolerance: float = 0.02,
        back_metric: str = "val_heatmap_back_pck8",
        back_tolerance: float = 0.03,
        direction_metrics: Mapping[str, str] | None = None,
        direction_tolerance: float = 0.03,
        loss_metric: str = "val_loss",
    ):
        if primary_mode not in {"min", "max"}:
            raise ValueError(
                f"primary_mode must be 'min' or 'max', got {primary_mode!r}"
            )
        if (
            overall_tolerance < 0
            or back_tolerance < 0
            or direction_tolerance < 0
        ):
            raise ValueError("Baseline gate tolerances must be non-negative")
        for name, value in (
            ("primary_metric", primary_metric),
            ("overall_metric", overall_metric),
            ("back_metric", back_metric),
            ("loss_metric", loss_metric),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} must be a non-empty metric key")
        normalized_direction_metrics: dict[str, str] = {}
        for raw_label, raw_metric in (direction_metrics or {}).items():
            label = str(raw_label).strip()
            metric = str(raw_metric).strip()
            if not label or not label.replace("_", "").isalnum():
                raise ValueError(
                    "direction_metrics labels must contain only letters, "
                    f"numbers, or underscores, got {raw_label!r}"
                )
            if not metric:
                raise ValueError(
                    f"direction_metrics[{raw_label!r}] must be a metric key"
                )
            normalized_direction_metrics[label] = metric

        self.primary_metric = str(primary_metric)
        self.primary_mode = primary_mode
        self.baseline_as_incumbent = bool(baseline_as_incumbent)
        self.overall_metric = str(overall_metric)
        self.overall_tolerance = float(overall_tolerance)
        self.back_metric = str(back_metric)
        self.back_tolerance = float(back_tolerance)
        self.direction_metrics = dict(
            sorted(normalized_direction_metrics.items())
        )
        self.direction_tolerance = float(direction_tolerance)
        self.loss_metric = str(loss_metric)

        self.baseline_metrics: dict[str, float] | None = None
        self.incumbent_metrics: dict[str, float] | None = None
        self.incumbent_epoch: int | None = None
        self.incumbent_source: str | None = None

    @staticmethod
    def _finite_metric(
        metrics: Mapping[str, Any],
        key: str,
    ) -> float | None:
        value = metrics.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _required_baseline_metrics(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
            self.primary_metric,
            self.overall_metric,
            self.back_metric,
            self.loss_metric,
                    *self.direction_metrics.values(),
                )
            )
        )

    def set_baseline(
        self,
        metrics: Mapping[str, Any],
        *,
        epoch: int = 0,
    ) -> dict[str, Any]:
        """Register the exact step-0 evaluation and make it the incumbent."""
        missing = [
            key
            for key in self._required_baseline_metrics()
            if self._finite_metric(metrics, key) is None
        ]
        if missing:
            raise ValueError(
                "Step-0 baseline lacks finite checkpoint-selection metrics: "
                f"{missing}"
            )
        baseline = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        self.baseline_metrics = baseline
        if self.baseline_as_incumbent:
            self.incumbent_metrics = dict(baseline)
            self.incumbent_epoch = int(epoch)
            self.incumbent_source = "step0_baseline"
        return {
            "record_type": "checkpoint_selection_baseline",
            "epoch": int(epoch),
            "source": "step0_baseline",
            "primary_metric": self.primary_metric,
            "primary_value": baseline[self.primary_metric],
            "overall_metric": self.overall_metric,
            "overall_value": baseline[self.overall_metric],
            "back_metric": self.back_metric,
            "back_value": baseline[self.back_metric],
            "direction_values": {
                label: baseline[metric]
                for label, metric in self.direction_metrics.items()
            },
            "baseline_as_incumbent": self.baseline_as_incumbent,
        }

    def set_incumbent(
        self,
        metrics: Mapping[str, Any],
        *,
        epoch: int | None,
        source: str,
    ) -> None:
        """Install a known incumbent, primarily for legacy loss-only resume."""
        if self._finite_metric(metrics, self.primary_metric) is None:
            raise ValueError(
                "Incumbent lacks finite primary metric "
                f"{self.primary_metric!r}"
            )
        self.incumbent_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        self.incumbent_epoch = int(epoch) if epoch is not None else None
        self.incumbent_source = str(source)

    def _strictly_better_primary(self, candidate: float, reference: float) -> bool:
        if self.primary_mode == "max":
            return candidate > reference + self._EPSILON
        return candidate < reference - self._EPSILON

    def _candidate_tuple(self, metrics: Mapping[str, Any]) -> tuple[float, ...]:
        primary = self._finite_metric(metrics, self.primary_metric)
        if primary is None:
            raise ValueError(
                f"Candidate lacks finite primary metric {self.primary_metric!r}"
            )
        primary_score = primary if self.primary_mode == "max" else -primary
        if not self.baseline_as_incumbent:
            # Preserve historical strict single-metric behavior by omitting
            # all tie-breakers unless constrained selection is explicitly on.
            return (primary_score,)
        overall = self._finite_metric(metrics, self.overall_metric)
        loss = self._finite_metric(metrics, self.loss_metric)
        if overall is None or loss is None:
            raise ValueError(
                "Constrained candidate lacks finite tie-break metrics: "
                f"{self.overall_metric!r}, {self.loss_metric!r}"
            )
        return (primary_score, overall, -loss)

    def _gate_rejections(
        self,
        metrics: Mapping[str, Any],
    ) -> tuple[list[str], list[str]]:
        if not self.baseline_as_incumbent:
            return [], []
        if self.baseline_metrics is None:
            raise RuntimeError(
                "Constrained checkpoint selection requires set_baseline() first"
            )

        codes: list[str] = []
        details: list[str] = []
        candidate_primary = self._finite_metric(metrics, self.primary_metric)
        baseline_primary = self.baseline_metrics[self.primary_metric]
        if candidate_primary is None:
            codes.append("non_finite_primary")
            details.append(
                f"{self.primary_metric} is missing or non-finite"
            )
        elif not self._strictly_better_primary(
            candidate_primary,
            baseline_primary,
        ):
            codes.append("primary_not_above_step0")
            details.append(
                f"{self.primary_metric}={candidate_primary:.8g} did not strictly "
                f"improve step0={baseline_primary:.8g} ({self.primary_mode})"
            )

        for metric, tolerance, code in (
            (
                self.overall_metric,
                self.overall_tolerance,
                "overall_below_step0_tolerance",
            ),
            (
                self.back_metric,
                self.back_tolerance,
                "back_below_step0_tolerance",
            ),
        ):
            candidate = self._finite_metric(metrics, metric)
            baseline = self.baseline_metrics[metric]
            floor = baseline - tolerance
            if candidate is None:
                codes.append(f"non_finite_{code.split('_', 1)[0]}")
                details.append(f"{metric} is missing or non-finite")
            elif candidate < floor - self._EPSILON:
                codes.append(code)
                details.append(
                    f"{metric}={candidate:.8g} below step0 floor "
                    f"{floor:.8g} ({baseline:.8g}-{tolerance:.8g})"
                )
        for label, metric in self.direction_metrics.items():
            candidate = self._finite_metric(metrics, metric)
            baseline = self.baseline_metrics[metric]
            floor = baseline - self.direction_tolerance
            if candidate is None:
                codes.append(f"non_finite_direction_{label}")
                details.append(f"{metric} is missing or non-finite")
            elif candidate < floor - self._EPSILON:
                codes.append(
                    f"direction_{label}_below_step0_tolerance"
                )
                details.append(
                    f"{metric}={candidate:.8g} below step0 floor "
                    f"{floor:.8g} ({baseline:.8g}-"
                    f"{self.direction_tolerance:.8g})"
                )
        return codes, details

    def consider(
        self,
        metrics: Mapping[str, Any],
        *,
        epoch: int,
    ) -> dict[str, Any]:
        """Evaluate one epoch and update the incumbent when appropriate."""
        candidate_primary = self._finite_metric(metrics, self.primary_metric)
        if candidate_primary is None:
            gate_codes = ["non_finite_primary"]
            gate_details = [
                f"{self.primary_metric} is missing or non-finite"
            ]
        else:
            gate_codes, gate_details = self._gate_rejections(metrics)

        accepted = False
        reason_codes = list(gate_codes)
        reason_details = list(gate_details)
        if not reason_codes:
            candidate_tuple = self._candidate_tuple(metrics)
            if self.incumbent_metrics is None:
                accepted = True
            else:
                incumbent_tuple = self._candidate_tuple(self.incumbent_metrics)
                accepted = candidate_tuple > incumbent_tuple
                if not accepted:
                    reason_codes.append("not_better_than_incumbent")
                    reason_details.append(
                        "candidate lexicographic score did not strictly exceed "
                        f"incumbent epoch={self.incumbent_epoch} "
                        f"source={self.incumbent_source}"
                    )

        previous_epoch = self.incumbent_epoch
        previous_source = self.incumbent_source
        if accepted:
            self.incumbent_metrics = {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
                and math.isfinite(float(value))
            }
            self.incumbent_epoch = int(epoch)
            self.incumbent_source = "training_epoch"

        return {
            "record_type": "checkpoint_selection_decision",
            "epoch": int(epoch),
            "accepted_as_best": accepted,
            "eligible": not gate_codes,
            "reason_codes": reason_codes,
            "reason_details": reason_details,
            "primary_metric": self.primary_metric,
            "primary_mode": self.primary_mode,
            "candidate_primary": candidate_primary,
            "candidate_overall": self._finite_metric(
                metrics,
                self.overall_metric,
            ),
            "candidate_back": self._finite_metric(metrics, self.back_metric),
            "candidate_directions": {
                label: self._finite_metric(metrics, metric)
                for label, metric in self.direction_metrics.items()
            },
            "candidate_loss": self._finite_metric(metrics, self.loss_metric),
            "previous_incumbent_epoch": previous_epoch,
            "previous_incumbent_source": previous_source,
            "incumbent_epoch": self.incumbent_epoch,
            "incumbent_source": self.incumbent_source,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "primary_metric": self.primary_metric,
                "primary_mode": self.primary_mode,
                "baseline_as_incumbent": self.baseline_as_incumbent,
                "overall_metric": self.overall_metric,
                "overall_tolerance": self.overall_tolerance,
                "back_metric": self.back_metric,
                "back_tolerance": self.back_tolerance,
                "direction_metrics": self.direction_metrics,
                "direction_tolerance": self.direction_tolerance,
                "loss_metric": self.loss_metric,
            },
            "baseline_metrics": self.baseline_metrics,
            "incumbent_metrics": self.incumbent_metrics,
            "incumbent_epoch": self.incumbent_epoch,
            "incumbent_source": self.incumbent_source,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected_config = self.state_dict()["config"]
        actual_config = dict(state.get("config", {}))
        if actual_config != expected_config:
            raise ValueError(
                "Checkpoint selection config changed across resume: "
                f"checkpoint={actual_config}, current={expected_config}"
            )
        baseline = state.get("baseline_metrics")
        incumbent = state.get("incumbent_metrics")
        self.baseline_metrics = (
            {str(key): float(value) for key, value in baseline.items()}
            if isinstance(baseline, Mapping)
            else None
        )
        self.incumbent_metrics = (
            {str(key): float(value) for key, value in incumbent.items()}
            if isinstance(incumbent, Mapping)
            else None
        )
        epoch = state.get("incumbent_epoch")
        self.incumbent_epoch = int(epoch) if epoch is not None else None
        source = state.get("incumbent_source")
        self.incumbent_source = str(source) if source is not None else None

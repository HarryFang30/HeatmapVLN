"""Dependency-light aggregation for Habitat navigation metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def aggregate_navigation_metrics(
    successes: Sequence[float],
    spls: Sequence[float],
    oracle_successes: Sequence[float],
    navigation_errors: Sequence[float],
) -> dict[str, float | int]:
    if not successes:
        return {"SR": 0.0, "SPL": 0.0, "OS": 0.0, "NE": 0.0, "total_episodes": 0}

    successes_array = np.asarray(successes, dtype=np.float64)
    spls_array = np.nan_to_num(
        np.asarray(spls, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    oracle_array = np.asarray(oracle_successes, dtype=np.float64)
    errors_array = np.asarray(navigation_errors, dtype=np.float64)
    finite_errors = errors_array[np.isfinite(errors_array)]

    return {
        "SR": float(successes_array.mean()),
        "SPL": float(spls_array.mean()),
        "OS": float(oracle_array.mean()),
        "NE": float(finite_errors.mean()) if finite_errors.size else 0.0,
        "total_episodes": len(successes),
    }

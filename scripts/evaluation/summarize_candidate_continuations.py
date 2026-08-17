#!/usr/bin/env python3
"""Integrity-check and summarize one-deviation continuation rollouts."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluation.summarize_candidate_support_audit import _read_records


SCHEMA = "candidate-continuation-summary-v1"
ROLL_OUT_SCHEMA = "candidate-continuation-rollout-v1"
ROLES = (
    "native_mean",
    "system2_selector",
    "heatmap_token_selector",
    "native_local_oracle",
    "union_local_oracle",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _quantize(value: float, resolution: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite outcome value: {value}")
    return float(round(value / resolution)) if resolution > 0 else value


def _short_quality(outcome: dict[str, Any]) -> tuple[float, ...]:
    entered = bool(outcome["entered_euclidean_success_radius"])
    left = bool(outcome["left_euclidean_success_radius"])
    return (
        float(outcome["habitat_success"] > 0.5),
        float(entered and not left),
        float(entered),
        float(not left),
        _quantize(outcome["route_progress_delta_m"], 0.05),
        -_quantize(outcome["endpoint_offpath_m"], 0.05),
        -float(outcome["collision_or_stuck_count"]),
        -float(outcome["revisit_count"]),
        -_quantize(outcome["min_euclidean_goal_distance_m"], 0.05),
        -_quantize(outcome["euclidean_goal_distance_m"], 0.05),
    )


def _end_quality(outcome: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(outcome["habitat_success"] > 0.5),
        float(outcome["habitat_oracle_success"] > 0.5),
        _quantize(outcome["habitat_spl"], 0.001),
        -_quantize(outcome["habitat_distance_to_goal_m"], 0.05),
        _quantize(outcome["route_progress_delta_m"], 0.05),
        -_quantize(outcome["endpoint_offpath_m"], 0.05),
        -float(outcome["collision_or_stuck_count"]),
        -float(outcome["revisit_count"]),
        -float(outcome["absolute_navigation_step_id"]),
    )


def _sign(left: tuple[float, ...], right: tuple[float, ...]) -> int:
    return int(left > right) - int(left < right)


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _comparison_summary(counter: Counter[str]) -> dict[str, Any]:
    total = int(counter["total"])
    return {
        "comparisons": total,
        "better": int(counter["better"]),
        "equal": int(counter["equal"]),
        "worse": int(counter["worse"]),
        "positive_rate": _rate(counter["better"], total),
        "destroy_rate": _rate(counter["worse"], total),
        "noninferior_rate": _rate(counter["better"] + counter["equal"], total),
    }


def _add_sign(counter: Counter[str], sign: int) -> None:
    counter["total"] += 1
    counter[{1: "better", 0: "equal", -1: "worse"}[int(sign)]] += 1


def _role_rows(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        for role in row.get("selector_roles") or []:
            if role in result:
                raise RuntimeError(
                    f"duplicate role {role} at {row.get('source_state_key')}"
                )
            result[str(role)] = row
    return result


def summarize(
    records: list[dict[str, Any]], manifests: list[dict[str, Any]]
) -> dict[str, Any]:
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scene_splits: dict[str, set[str]] = defaultdict(set)
    replay_translation_max = 0.0
    replay_rotation_max = 0.0
    authoritative_end_rows = 0
    requested_end_rows = 0
    termination_reasons: Counter[str] = Counter()
    for row in records:
        if row.get("continuation_schema") != ROLL_OUT_SCHEMA:
            raise RuntimeError(
                f"unexpected continuation schema at {row.get('state_key')}"
            )
        replay = row.get("replay_verification") or {}
        if replay.get("status") != "exact_prefix_replay_verified":
            raise RuntimeError(f"unverified replay at {row.get('state_key')}")
        replay_translation_max = max(
            replay_translation_max,
            float(replay.get("max_translation_error_m", math.inf)),
        )
        replay_rotation_max = max(
            replay_rotation_max,
            float(replay.get("max_rotation_max_abs", math.inf)),
        )
        state_key = str(row["source_state_key"])
        by_state[state_key].append(row)
        scene_splits[str(row["scene_split"])].add(str(row["scene_id"]))
        requested_end_rows += int(bool(row.get("run_to_episode_end")))
        authoritative_end_rows += int(bool(row.get("episode_end_authoritative")))
        termination_reasons[str((row.get("termination") or {}).get("reason"))] += 1
    split_names = sorted(scene_splits)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = scene_splits[left] & scene_splits[right]
            if overlap:
                raise RuntimeError(
                    f"scene leakage between {left}/{right}: {sorted(overlap)}"
                )

    role_horizon: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    pair_horizon: dict[str, Counter[str]] = defaultdict(Counter)
    stability: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    role_coverage: Counter[str] = Counter()
    state_split_counts: Counter[str] = Counter()
    end_state_count = 0
    for state_key, rows in by_state.items():
        roles = _role_rows(rows)
        missing = [role for role in ROLES if role not in roles]
        if missing:
            # Smoke intentionally limits branches; formal data must be complete.
            if not all(bool(row.get("smoke_mode")) for row in rows):
                raise RuntimeError(f"formal state {state_key} lacks roles {missing}")
        baseline = roles.get("native_mean")
        if baseline is None:
            raise RuntimeError(f"state {state_key} lacks native_mean")
        split = str(baseline["scene_split"])
        state_split_counts[split] += 1
        for role in roles:
            role_coverage[role] += 1

        baseline_horizons = baseline.get("horizon_outcomes") or {}
        horizon_names = sorted(
            baseline_horizons,
            key=lambda value: int(value),
        )
        for role, row in roles.items():
            horizons = row.get("horizon_outcomes") or {}
            for horizon in horizon_names:
                if horizon not in horizons:
                    continue
                sign = _sign(
                    _short_quality(horizons[horizon]),
                    _short_quality(baseline_horizons[horizon]),
                )
                _add_sign(role_horizon[role][horizon], sign)
            left_end = row.get("episode_end_outcome")
            base_end = baseline.get("episode_end_outcome")
            if (
                isinstance(left_end, dict)
                and isinstance(base_end, dict)
                and bool(row.get("episode_end_authoritative"))
                and bool(baseline.get("episode_end_authoritative"))
            ):
                end_sign = _sign(_end_quality(left_end), _end_quality(base_end))
                _add_sign(role_horizon[role]["episode_end"], end_sign)
                for horizon in horizon_names:
                    if horizon not in horizons:
                        continue
                    short_sign = _sign(
                        _short_quality(horizons[horizon]),
                        _short_quality(baseline_horizons[horizon]),
                    )
                    bucket = stability[role][horizon]
                    bucket["total"] += 1
                    bucket["same_sign"] += int(short_sign == end_sign)
                    bucket["opposite_nonzero"] += int(
                        short_sign != 0 and end_sign != 0 and short_sign != end_sign
                    )
                    bucket["short_positive_end_positive"] += int(
                        short_sign > 0 and end_sign > 0
                    )
                    bucket["short_positive_end_nonpositive"] += int(
                        short_sign > 0 and end_sign <= 0
                    )
        if all(
            isinstance(row.get("episode_end_outcome"), dict)
            and bool(row.get("episode_end_authoritative"))
            for row in rows
        ):
            end_state_count += 1

        system2 = roles.get("system2_selector")
        heatmap = roles.get("heatmap_token_selector")
        if system2 is not None and heatmap is not None:
            for horizon in horizon_names:
                if horizon in system2["horizon_outcomes"] and horizon in heatmap["horizon_outcomes"]:
                    sign = _sign(
                        _short_quality(heatmap["horizon_outcomes"][horizon]),
                        _short_quality(system2["horizon_outcomes"][horizon]),
                    )
                    _add_sign(pair_horizon[horizon], sign)
            if (
                isinstance(system2.get("episode_end_outcome"), dict)
                and isinstance(heatmap.get("episode_end_outcome"), dict)
                and bool(system2.get("episode_end_authoritative"))
                and bool(heatmap.get("episode_end_authoritative"))
            ):
                sign = _sign(
                    _end_quality(heatmap["episode_end_outcome"]),
                    _end_quality(system2["episode_end_outcome"]),
                )
                _add_sign(pair_horizon["episode_end"], sign)

    stability_summary: dict[str, dict[str, Any]] = {}
    for role, horizons in stability.items():
        stability_summary[role] = {}
        for horizon, counts in horizons.items():
            total = int(counts["total"])
            short_positive = int(
                counts["short_positive_end_positive"]
                + counts["short_positive_end_nonpositive"]
            )
            stability_summary[role][horizon] = {
                "comparisons": total,
                "same_sign": int(counts["same_sign"]),
                "same_sign_rate": _rate(counts["same_sign"], total),
                "opposite_nonzero": int(counts["opposite_nonzero"]),
                "opposite_nonzero_rate": _rate(counts["opposite_nonzero"], total),
                "short_positive": short_positive,
                "short_positive_end_positive": int(
                    counts["short_positive_end_positive"]
                ),
                "short_positive_precision_for_end_positive": _rate(
                    counts["short_positive_end_positive"], short_positive
                ),
            }

    array_bytes = sum(int(manifest["array_bytes"]) for manifest in manifests)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "storage": {
            "shards": len(manifests),
            "records": len(records),
            "compressed_array_bytes": array_bytes,
            "compressed_array_gb_decimal": array_bytes / 1_000_000_000.0,
        },
        "coverage": {
            "states": len(by_state),
            "authoritative_episode_end_states": end_state_count,
            "requested_episode_end_rows": requested_end_rows,
            "authoritative_episode_end_rows": authoritative_end_rows,
            "role_state_counts": dict(sorted(role_coverage.items())),
            "scene_split_state_counts": dict(sorted(state_split_counts.items())),
            "scene_split_scene_counts": {
                split: len(scenes) for split, scenes in sorted(scene_splits.items())
            },
            "scene_disjoint_verified": True,
            "termination_reasons": dict(sorted(termination_reasons.items())),
        },
        "replay_integrity": {
            "all_verified": True,
            "max_translation_error_m": replay_translation_max,
            "max_rotation_max_abs": replay_rotation_max,
        },
        "against_native_mean": {
            role: {
                horizon: _comparison_summary(counts)
                for horizon, counts in sorted(
                    horizons.items(),
                    key=lambda item: (
                        item[0] == "episode_end",
                        int(item[0]) if item[0] != "episode_end" else 10**9,
                    ),
                )
            }
            for role, horizons in sorted(role_horizon.items())
        },
        "heatmap_selector_vs_system2_selector": {
            horizon: _comparison_summary(counts)
            for horizon, counts in sorted(
                pair_horizon.items(),
                key=lambda item: (
                    item[0] == "episode_end",
                    int(item[0]) if item[0] != "episode_end" else 10**9,
                ),
            )
        },
        "short_horizon_sign_stability_against_episode_end": stability_summary,
        "decision_note": (
            "Use authoritative episode-end rows and scene-disjoint splits for the "
            "next selector; H1/H3/H5 are acceptable labels only if their signs are "
            "stable against episode end."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=8)
    args = parser.parse_args()
    records, manifests = _read_records(
        args.audit_root.expanduser().resolve(), args.expected_shards
    )
    result = summarize(records, manifests)
    _atomic_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

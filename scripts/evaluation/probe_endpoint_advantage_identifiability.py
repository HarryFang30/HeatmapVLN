#!/usr/bin/env python3
"""Endpoint-advantage identifiability benchmark for candidate interventions.

The benchmark deliberately answers a narrower question than policy training:
given the same endpoint-labelled native treatments at a state, does matched
heatmap context improve the ability of a low-capacity model to decide whether
to replace the native-mean treatment?

It also emits three static audits required before interpreting the probe:

* action-level disagreement between the existing System2 and heatmap selectors;
* complete/incomplete endpoint-label missingness;
* observed-treatment endpoint-oracle proposal support, explicitly marked as a
  lower bound whenever not every source candidate has an endpoint rollout.

No simulator-only goal/path fields enter model inputs.  They are used only for
endpoint labels and offline one-deviation policy evaluation.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.probe_candidate_identifiability import (
    _atomic_json,
    decode_bfloat16_bits,
    heatmap_metadata,
)


SCHEMA = "endpoint-advantage-identifiability-benchmark-v1"
MODEL_VARIANTS = (
    "candidate_only",
    "candidate_system2",
    "candidate_system2_heatmap_metadata",
    "candidate_system2_heatmap_geometry",
    "candidate_system2_heatmap_tokens",
)
HEATMAP_VARIANTS = {
    "candidate_system2_heatmap_metadata",
    "candidate_system2_heatmap_geometry",
    "candidate_system2_heatmap_tokens",
}
CONTEXT_MODES = ("matched", "zeroed", "shuffled")
ACTION_NAMES = ("stop", "forward", "left", "right", "pad")
END_REASONS = (
    "early_replan",
    "queue_exhausted_replan",
    "local_stop_replan",
    "anti_deadlock_replan",
    "replan_now",
)
VIEW_CENTERS_RAD = np.asarray(
    (0.0, -math.pi / 2.0, math.pi, math.pi / 2.0), dtype=np.float32
)
DEFAULT_COVERAGES = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_rate(numerator: float, denominator: float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _std(values: Sequence[float]) -> float | None:
    return float(np.std(values)) if values else None


def _quantile(values: Sequence[float], probability: float) -> float | None:
    return float(np.quantile(values, probability)) if values else None


def _json_lines(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}") from exc


def _load_targets(root: Path, expected_shards: int) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    targets: dict[str, dict[str, Any]] = {}
    payloads: list[dict[str, Any]] = []
    for shard_id in range(expected_shards):
        path = root / f"targets_shard_{shard_id:02d}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "candidate-continuation-targets-v1":
            raise RuntimeError(f"unexpected target schema: {path}")
        if int(payload.get("shard_id", -1)) != shard_id:
            raise RuntimeError(f"target shard id mismatch: {path}")
        for row in payload.get("targets") or []:
            key = str(row["state_key"])
            if key in targets:
                raise RuntimeError(f"duplicate continuation target: {key}")
            targets[key] = row
        payloads.append(payload)
    if not targets:
        raise RuntimeError("continuation target plan is empty")
    return targets, payloads


def _load_continuations(
    root: Path,
    expected_shards: int,
    *,
    verify_integrity: bool,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    manifests: list[dict[str, Any]] = []
    seen_state_keys: set[str] = set()
    for shard_id in range(expected_shards):
        shard = root / f"shard_{shard_id:02d}"
        manifest_path = shard / "manifest.json"
        records_path = shard / "records.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("shard_id", -1)) != shard_id:
            raise RuntimeError(f"continuation shard mismatch: {manifest_path}")
        if verify_integrity:
            actual = _sha256_file(records_path)
            expected = str(manifest.get("records_jsonl_sha256"))
            if actual != expected:
                raise RuntimeError(
                    f"continuation record hash mismatch: {records_path}: {actual} != {expected}"
                )
        rows = list(_json_lines(records_path))
        if len(rows) != int(manifest.get("record_count", -1)):
            raise RuntimeError(f"continuation record count mismatch: {records_path}")
        for row in rows:
            state_key = str(row["state_key"])
            if state_key in seen_state_keys:
                raise RuntimeError(f"duplicate continuation branch: {state_key}")
            seen_state_keys.add(state_key)
            if row.get("continuation_schema") != "candidate-continuation-rollout-v1":
                raise RuntimeError(f"unexpected continuation schema: {state_key}")
            replay = row.get("replay_verification") or {}
            if replay.get("status") != "exact_prefix_replay_verified":
                raise RuntimeError(f"unverified prefix replay: {state_key}")
            grouped[str(row["source_state_key"])].append(row)
        manifests.append(manifest)
    if not grouped:
        raise RuntimeError("continuation audit is empty")
    return grouped, manifests


def _load_source_targets(
    root: Path,
    target_keys: set[str],
    expected_shards: int,
    *,
    verify_integrity: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    manifests: list[dict[str, Any]] = []
    for shard_id in range(expected_shards):
        shard = root / f"shard_{shard_id:02d}"
        manifest_path = shard / "manifest.json"
        records_path = shard / "records.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("shard_id", -1)) != shard_id:
            raise RuntimeError(f"source shard mismatch: {manifest_path}")
        digest = hashlib.sha256() if verify_integrity else None
        count = 0
        with records_path.open("rb") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                count += 1
                if digest is not None:
                    digest.update(raw)
                row = json.loads(raw)
                key = str(row["state_key"])
                if key in target_keys:
                    if key in records:
                        raise RuntimeError(f"duplicate source target state: {key}")
                    row["__shard_dir"] = str(shard)
                    records[key] = row
        if count != int(manifest.get("record_count", -1)):
            raise RuntimeError(f"source record count mismatch: {records_path}")
        if digest is not None:
            actual = digest.hexdigest()
            expected = str(manifest.get("records_jsonl_sha256"))
            if actual != expected:
                raise RuntimeError(
                    f"source record hash mismatch: {records_path}: {actual} != {expected}"
                )
        manifests.append(manifest)
        print(
            f"[endpoint-probe] indexed source shard {shard_id + 1}/{expected_shards}; "
            f"joined={len(records)}/{len(target_keys)}",
            flush=True,
        )
    missing = sorted(target_keys - set(records))
    if missing:
        raise RuntimeError(f"source audit lacks {len(missing)} targets; first={missing[:3]}")
    return records, manifests


def _load_episode_results(worker_root: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if worker_root is None:
        return {}
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(worker_root.glob("shard_*/progress.json")):
        for row in _json_lines(path):
            key = (str(row["scene_id"]), str(row["episode_id"]))
            if key in results:
                raise RuntimeError(f"duplicate baseline episode result: {key}")
            results[key] = row
    return results


def _has_arm(treatment: dict[str, Any], arm: str) -> bool:
    count_key = "native_sample_count" if arm == "native" else "heatmap_sample_count"
    return int(treatment.get(count_key, 0)) > 0 or any(
        str(item.get("arm")) == arm for item in treatment.get("provenances") or []
    )


def native_candidate_feature_names() -> list[str]:
    names: list[str] = []
    for position in range(4):
        names.extend(f"action_{position}_{name}" for name in ACTION_NAMES)
    names.extend(f"action_fraction_{name}" for name in ACTION_NAMES[:4])
    names.extend(
        (
            "execute_fraction",
            "replan_after",
            "trigger_anti_deadlock",
            "update_local_stop_counter",
        )
    )
    names.extend(f"end_reason_{name}" for name in END_REASONS)
    names.extend(
        (
            "native_sample_mass",
            "native_sample_fraction",
            "has_native_mean_provenance",
            "is_native_mean_baseline",
            "baseline_common_prefix_fraction",
            "baseline_hamming_fraction",
            "baseline_length_delta_fraction",
            "forward_count",
            "left_count",
            "right_count",
            "net_turn_fraction",
        )
    )
    return names


def native_candidate_features(
    treatment: dict[str, Any],
    *,
    baseline_id: str,
    baseline_actions: Sequence[int],
) -> np.ndarray:
    """Candidate features with all heatmap proposal provenance removed."""

    spec = treatment["spec"]
    actions = [int(value) for value in spec["actions"]]
    if len(actions) > 4 or any(value not in (0, 1, 2, 3) for value in actions):
        raise RuntimeError(f"invalid treatment actions: {actions}")
    padded = actions + [4] * (4 - len(actions))
    values: list[float] = []
    for value in padded:
        values.extend(float(value == category) for category in range(5))
    values.extend(float(actions.count(action)) / 4.0 for action in range(4))
    values.extend(
        (
            float(spec["execute_len"]) / 4.0,
            float(bool(spec["replan_after"])),
            float(bool(spec["trigger_anti_deadlock"])),
            float(bool(spec["update_local_stop_counter"])),
        )
    )
    end_reason = str(spec["end_reason"])
    if end_reason not in END_REASONS:
        raise RuntimeError(f"unsupported end reason: {end_reason}")
    values.extend(float(end_reason == name) for name in END_REASONS)
    common_prefix = 0
    for left, right in zip(actions, map(int, baseline_actions)):
        if left != right:
            break
        common_prefix += 1
    baseline_padded = list(map(int, baseline_actions)) + [4] * (
        4 - len(baseline_actions)
    )
    hamming = sum(left != right for left, right in zip(padded, baseline_padded))
    provenances = list(treatment.get("provenances") or [])
    native_total = max(1, int(treatment.get("native_sample_total", 0)))
    values.extend(
        (
            float(treatment.get("native_sample_mass", 0.0)),
            float(treatment.get("native_sample_count", 0)) / native_total,
            float(
                any(
                    item.get("arm") == "native"
                    and item.get("aggregation") == "trajectory_mean"
                    for item in provenances
                )
            ),
            float(str(treatment["treatment_id"]) == baseline_id),
            float(common_prefix) / 4.0,
            float(hamming) / 4.0,
            float(len(actions) - len(baseline_actions)) / 4.0,
            float(actions.count(1)) / 4.0,
            float(actions.count(2)) / 4.0,
            float(actions.count(3)) / 4.0,
            float(actions.count(2) - actions.count(3)) / 4.0,
        )
    )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(native_candidate_feature_names()),):
        raise AssertionError("native candidate feature width mismatch")
    return result


def _strip_leading_batch(value: np.ndarray, expected_ndim: int) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim == expected_ndim + 1 and result.shape[0] == 1:
        result = result[0]
    if result.ndim != expected_ndim:
        raise RuntimeError(
            f"unexpected array rank {result.shape}; expected ndim={expected_ndim}"
        )
    return result


def _load_context(
    source: dict[str, Any], *, verify_integrity: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    shard = Path(str(source["__shard_dir"]))
    array_path = shard / str(source["array_file"])
    if verify_integrity:
        actual = _sha256_file(array_path)
        expected = str(source.get("array_file_sha256"))
        if actual != expected:
            raise RuntimeError(f"source array hash mismatch: {array_path}")
    with np.load(array_path, allow_pickle=False) as arrays:
        required = (
            "system2_latent_bf16_bits",
            "heatmap_tokens",
            "heatmap_token_mask",
            "heatmap_sample_valid",
        )
        missing = [name for name in required if name not in arrays.files]
        if missing:
            raise RuntimeError(f"{array_path} lacks arrays: {missing}")
        system2 = np.asarray(
            decode_bfloat16_bits(arrays["system2_latent_bf16_bits"])[0],
            dtype=np.float16,
        )
        tokens = np.asarray(arrays["heatmap_tokens"][0], dtype=np.float16)
        token_mask = np.asarray(arrays["heatmap_token_mask"][0], dtype=np.bool_)
        metadata = heatmap_metadata(arrays)
        valid = bool(np.asarray(arrays["heatmap_sample_valid"]).reshape(-1)[0])
        geometry_context: dict[str, np.ndarray] = {
            "valid": np.asarray([valid], dtype=np.bool_),
        }
        geometry_arrays = (
            ("spatial_statistics", 3),
            ("view_probabilities", 2),
            ("none_probability", 1),
            ("normalized_age", 1),
            ("fixed_history_mask", 1),
            ("fixed_history_rel_poses", 2),
        )
        if valid:
            absent = [name for name, _ in geometry_arrays if name not in arrays.files]
            if absent:
                raise RuntimeError(f"valid heatmap sample lacks geometry arrays: {absent}")
            for name, ndim in geometry_arrays:
                geometry_context[name] = np.asarray(
                    _strip_leading_batch(arrays[name], ndim), dtype=np.float32
                )
    if system2.ndim != 2 or tokens.ndim != 2:
        raise RuntimeError(f"unexpected context shapes at {array_path}")
    if token_mask.shape != (tokens.shape[0],):
        raise RuntimeError(f"heatmap token mask mismatch at {array_path}")
    return system2, metadata, tokens, token_mask, geometry_context


GEOMETRY_BASE_NAMES = (
    "context_available",
    "valid_history_fraction",
    "visibility_mass",
    "endpoint_min_history_distance",
    "path_min_history_distance",
    "endpoint_visibility_weighted_inverse_distance",
    "path_visibility_weighted_inverse_distance",
    "endpoint_revisit_risk",
    "path_revisit_risk",
    "moves_toward_visible_history_fraction",
    "final_bearing_heatmap_overlap",
    "motion_bearing_heatmap_overlap",
    "recent_final_bearing_overlap",
    "recent_motion_bearing_overlap",
    "peak_weighted_final_overlap",
    "peak_weighted_motion_overlap",
)


def geometry_feature_names() -> list[str]:
    base = list(GEOMETRY_BASE_NAMES)
    return base + [f"delta_from_native_mean/{name}" for name in base]


def _simulate_candidate(
    actions: Sequence[int], *, forward_step_m: float = 0.25, turn_deg: float = 15.0
) -> tuple[np.ndarray, float, float]:
    forward = left = yaw = 0.0
    points = [(forward, left)]
    motion_bearings: list[float] = []
    turn = math.radians(turn_deg)
    for action in map(int, actions):
        if action == 1:
            forward += forward_step_m * math.cos(yaw)
            left += forward_step_m * math.sin(yaw)
            points.append((forward, left))
            motion_bearings.append(yaw)
        elif action == 2:
            yaw += turn
        elif action == 3:
            yaw -= turn
        elif action == 0:
            break
    if motion_bearings:
        motion_yaw = math.atan2(
            sum(math.sin(value) for value in motion_bearings),
            sum(math.cos(value) for value in motion_bearings),
        )
    else:
        motion_yaw = yaw
    return np.asarray(points, dtype=np.float32), float(yaw), float(motion_yaw)


def _angular_overlap(candidate_yaw: float, bearings: np.ndarray, weights: np.ndarray) -> float:
    if bearings.size == 0 or float(weights.sum()) <= 0.0:
        return 0.0
    delta = np.arctan2(
        np.sin(bearings - candidate_yaw), np.cos(bearings - candidate_yaw)
    )
    similarity = np.exp(-0.5 * np.square(delta / math.radians(35.0)))
    return float(np.sum(similarity * weights) / max(1e-8, float(weights.sum())))


def _geometry_base(
    spec: dict[str, Any], context: dict[str, np.ndarray]
) -> np.ndarray:
    if not bool(np.asarray(context.get("valid", [False])).reshape(-1)[0]):
        return np.zeros(len(GEOMETRY_BASE_NAMES), dtype=np.float32)
    mask = np.asarray(context["fixed_history_mask"], dtype=np.float32).reshape(-1) > 0.5
    rel = np.asarray(context["fixed_history_rel_poses"], dtype=np.float32)
    stats = np.asarray(context["spatial_statistics"], dtype=np.float32)
    view = np.asarray(context["view_probabilities"], dtype=np.float32)
    none = np.asarray(context["none_probability"], dtype=np.float32).reshape(-1)
    age = np.asarray(context["normalized_age"], dtype=np.float32).reshape(-1)
    k = len(mask)
    if rel.shape != (k, 4) or stats.shape[:2] != (k, 4) or view.shape != (k, 4):
        raise RuntimeError(
            f"geometry context mismatch: mask={mask.shape} rel={rel.shape} "
            f"stats={stats.shape} view={view.shape}"
        )
    points, final_yaw, motion_yaw = _simulate_candidate(spec["actions"])
    endpoint = points[-1]
    if not mask.any():
        return np.asarray(
            [1.0] + [0.0] * (len(GEOMETRY_BASE_NAMES) - 1), dtype=np.float32
        )
    history_xy = rel[mask, :2]
    reliability = np.max(view[mask], axis=1) * (1.0 - none[mask])
    recent = np.clip(1.0 - age[mask], 0.0, 1.0)
    endpoint_distance = np.linalg.norm(history_xy - endpoint[None, :], axis=1)
    path_distance = np.min(
        np.linalg.norm(
            history_xy[:, None, :] - points[None, :, :], axis=-1
        ),
        axis=1,
    )
    initial_distance = np.linalg.norm(history_xy, axis=1)
    denom = max(1e-8, float(reliability.sum()))
    inv_endpoint = float(np.sum(reliability / (0.25 + endpoint_distance)) / denom)
    inv_path = float(np.sum(reliability / (0.25 + path_distance)) / denom)

    mean_x = stats[mask, :, 0]
    peak = np.clip(stats[mask, :, 6], 0.0, None)
    bearings = VIEW_CENTERS_RAD[None, :] - mean_x * (math.pi / 4.0)
    view_weight = view[mask] * (1.0 - none[mask, None])
    recent_weight = view_weight * recent[:, None]
    peak_weight = view_weight * peak
    result = np.asarray(
        (
            1.0,
            float(mask.mean()),
            float(view_weight.sum()) / max(1, k),
            float(endpoint_distance.min()),
            float(path_distance.min()),
            inv_endpoint,
            inv_path,
            float(np.max(reliability * (endpoint_distance < 0.5))),
            float(np.max(reliability * (path_distance < 0.5))),
            float(np.sum(reliability * (endpoint_distance < initial_distance)) / denom),
            _angular_overlap(final_yaw, bearings.reshape(-1), view_weight.reshape(-1)),
            _angular_overlap(motion_yaw, bearings.reshape(-1), view_weight.reshape(-1)),
            _angular_overlap(final_yaw, bearings.reshape(-1), recent_weight.reshape(-1)),
            _angular_overlap(motion_yaw, bearings.reshape(-1), recent_weight.reshape(-1)),
            _angular_overlap(final_yaw, bearings.reshape(-1), peak_weight.reshape(-1)),
            _angular_overlap(motion_yaw, bearings.reshape(-1), peak_weight.reshape(-1)),
        ),
        dtype=np.float32,
    )
    if not np.all(np.isfinite(result)):
        raise RuntimeError("candidate/heatmap geometry contains non-finite values")
    return result


def candidate_heatmap_geometry(
    specs: Sequence[dict[str, Any]],
    context: dict[str, np.ndarray],
    *,
    baseline_index: int,
) -> np.ndarray:
    base = np.stack([_geometry_base(spec, context) for spec in specs])
    delta = base - base[baseline_index : baseline_index + 1]
    result = np.concatenate((base, delta), axis=1).astype(np.float32, copy=False)
    if result.shape[1] != len(geometry_feature_names()):
        raise AssertionError("geometry feature width mismatch")
    return result


def _endpoint_label(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> tuple[int, float, str]:
    candidate_success = bool(float(candidate["habitat_success"]) > 0.5)
    baseline_success = bool(float(baseline["habitat_success"]) > 0.5)
    if candidate_success != baseline_success:
        return (
            (1 if candidate_success else -1),
            4.0,
            ("rescue" if candidate_success else "destroy"),
        )
    if candidate_success and baseline_success:
        delta = float(candidate["habitat_spl"]) - float(baseline["habitat_spl"])
        if abs(delta) >= 0.001:
            return (int(delta > 0.0) - int(delta < 0.0), 1.0, "both_success_spl")
        return 0, 0.0, "both_success_tie"
    return 0, 0.0, "both_fail_low_confidence"


def _endpoint_key(outcome: dict[str, Any]) -> tuple[float, float]:
    success = float(float(outcome["habitat_success"]) > 0.5)
    return success, (float(outcome["habitat_spl"]) if success else 0.0)


@dataclass
class EndpointState:
    state_key: str
    scene_id: str
    episode_id: str
    step_id: int
    candidate: np.ndarray
    geometry: np.ndarray
    candidate_ids: tuple[str, ...]
    candidate_specs: tuple[dict[str, Any], ...]
    outcomes: tuple[dict[str, Any], ...]
    preference: np.ndarray
    preference_weight: np.ndarray
    baseline_index: int
    system2_tokens: np.ndarray
    metadata: np.ndarray
    heatmap_tokens: np.ndarray
    heatmap_mask: np.ndarray
    geometry_context: dict[str, np.ndarray]


def _build_endpoint_states(
    targets: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    continuations: dict[str, list[dict[str, Any]]],
    *,
    verify_integrity: bool,
    max_states: int,
) -> tuple[list[EndpointState], dict[str, Any]]:
    states: list[EndpointState] = []
    exclusions: Counter[str] = Counter()
    label_semantics: Counter[str] = Counter()
    context_coverage: Counter[str] = Counter()
    active_token_counts: list[int] = []
    complete_keys = [
        key
        for key in sorted(targets)
        if key in continuations
        and continuations[key]
        and all(bool(row.get("episode_end_authoritative")) for row in continuations[key])
    ]
    if max_states > 0:
        # A lexicographic prefix can collapse a smoke test onto one or two
        # scenes.  Round-robin over scenes so even a small development run
        # exercises the same scene-grouped split contract as the full probe.
        by_scene: dict[str, list[str]] = defaultdict(list)
        for key in complete_keys:
            by_scene[str(targets[key]["scene_id"])].append(key)
        selected: list[str] = []
        scene_names = sorted(by_scene)
        while len(selected) < max_states:
            progressed = False
            for scene in scene_names:
                if by_scene[scene] and len(selected) < max_states:
                    selected.append(by_scene[scene].pop(0))
                    progressed = True
            if not progressed:
                break
        complete_keys = selected
    for index, key in enumerate(complete_keys, 1):
        source = sources[key]
        target = targets[key]
        treatments = {
            str(item["treatment_id"]): item
            for item in source["candidate_set"]["treatments"]
        }
        endpoint_rows = {
            str(row["treatment_id"]): row
            for row in continuations[key]
            if bool(row.get("episode_end_authoritative"))
            and isinstance(row.get("episode_end_outcome"), dict)
        }
        baseline_id = str(target["treatment_roles"]["native_mean"])
        if baseline_id not in endpoint_rows or baseline_id not in treatments:
            exclusions["baseline_endpoint_missing"] += 1
            continue
        native_ids = {
            treatment_id
            for treatment_id, treatment in treatments.items()
            if _has_arm(treatment, "native")
        }
        candidate_ids = [baseline_id] + sorted((set(endpoint_rows) & native_ids) - {baseline_id})
        if len(candidate_ids) < 2:
            exclusions["fewer_than_two_endpoint_native_treatments"] += 1
            continue
        baseline_actions = treatments[baseline_id]["spec"]["actions"]
        candidate = np.stack(
            [
                native_candidate_features(
                    treatments[treatment_id],
                    baseline_id=baseline_id,
                    baseline_actions=baseline_actions,
                )
                for treatment_id in candidate_ids
            ]
        )
        specs = tuple(copy.deepcopy(treatments[item]["spec"]) for item in candidate_ids)
        outcomes = tuple(
            copy.deepcopy(endpoint_rows[item]["episode_end_outcome"])
            for item in candidate_ids
        )
        preference = np.zeros(len(candidate_ids), dtype=np.int8)
        preference_weight = np.zeros(len(candidate_ids), dtype=np.float32)
        for candidate_index, outcome in enumerate(outcomes):
            sign, weight, semantic = _endpoint_label(outcome, outcomes[0])
            preference[candidate_index] = sign
            preference_weight[candidate_index] = weight
            if candidate_index:
                label_semantics[semantic] += 1
        system2, metadata, tokens, token_mask, geometry_context = _load_context(
            source, verify_integrity=verify_integrity
        )
        geometry = candidate_heatmap_geometry(
            specs, geometry_context, baseline_index=0
        )
        context_valid = bool(
            np.asarray(geometry_context.get("valid", [False])).reshape(-1)[0]
        )
        active_tokens = int(np.asarray(token_mask, dtype=np.bool_).sum())
        context_coverage["states"] += 1
        context_coverage["heatmap_sample_valid"] += int(context_valid)
        context_coverage["positive_heatmap_tokens"] += int(active_tokens > 0)
        context_coverage["strong_endpoint_label_state"] += int(
            any(
                semantic in {"rescue", "destroy"}
                for semantic in (
                    _endpoint_label(outcome, outcomes[0])[2]
                    for outcome in outcomes[1:]
                )
            )
        )
        context_coverage["strong_label_and_valid_heatmap"] += int(
            context_valid
            and any(
                _endpoint_label(outcome, outcomes[0])[2] in {"rescue", "destroy"}
                for outcome in outcomes[1:]
            )
        )
        active_token_counts.append(active_tokens)
        states.append(
            EndpointState(
                state_key=key,
                scene_id=str(target["scene_id"]),
                episode_id=str(target["episode_id"]),
                step_id=int(target["step_id"]),
                candidate=candidate,
                geometry=geometry,
                candidate_ids=tuple(candidate_ids),
                candidate_specs=specs,
                outcomes=outcomes,
                preference=preference,
                preference_weight=preference_weight,
                baseline_index=0,
                system2_tokens=system2,
                metadata=metadata,
                heatmap_tokens=tokens,
                heatmap_mask=token_mask,
                geometry_context=geometry_context,
            )
        )
        if index % 100 == 0 or index == len(complete_keys):
            print(
                f"[endpoint-probe] loaded endpoint states {index}/{len(complete_keys)}; "
                f"usable={len(states)}",
                flush=True,
            )
    return states, {
        "complete_states_considered": len(complete_keys),
        "usable_native_endpoint_states": len(states),
        "exclusions": dict(sorted(exclusions.items())),
        "baseline_centered_label_semantics": dict(sorted(label_semantics.items())),
        "deployable_heatmap_context_coverage": {
            "states": context_coverage["states"],
            "heatmap_sample_valid": context_coverage["heatmap_sample_valid"],
            "heatmap_sample_valid_rate": _safe_rate(
                context_coverage["heatmap_sample_valid"], context_coverage["states"]
            ),
            "positive_heatmap_tokens": context_coverage["positive_heatmap_tokens"],
            "positive_heatmap_token_rate": _safe_rate(
                context_coverage["positive_heatmap_tokens"], context_coverage["states"]
            ),
            "active_token_count": _numeric_summary(active_token_counts),
            "strong_endpoint_label_states": context_coverage[
                "strong_endpoint_label_state"
            ],
            "strong_label_and_valid_heatmap": context_coverage[
                "strong_label_and_valid_heatmap"
            ],
        },
        "candidate_feature_names": native_candidate_feature_names(),
        "geometry_feature_names": geometry_feature_names(),
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "std": None}
    return {
        "count": len(finite),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(min(finite)),
        "max": float(max(finite)),
    }


def _standardized_mean_difference(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not left or not right:
        return None
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    pooled = math.sqrt((float(left_array.var()) + float(right_array.var())) / 2.0)
    if pooled <= 1e-12:
        return 0.0 if abs(float(left_array.mean() - right_array.mean())) <= 1e-12 else None
    return float((left_array.mean() - right_array.mean()) / pooled)


def _missingness_audit(
    targets: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    continuations: dict[str, list[dict[str, Any]]],
    episode_results: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    complete = {
        key
        for key, rows in continuations.items()
        if rows and all(bool(row.get("episode_end_authoritative")) for row in rows)
    }
    numeric: dict[str, dict[str, list[float]]] = {
        name: {"complete": [], "incomplete": []}
        for name in (
            "step_id",
            "system2_call_index",
            "source_unique_treatments",
            "native_candidate_count",
            "heatmap_only_candidate_count",
            "selected_unique_treatments",
            "baseline_episode_success",
            "baseline_episode_spl",
            "baseline_episode_steps",
            "observed_max_future_cycles",
        )
    }
    categorical: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {"complete": Counter(), "incomplete": Counter()}
    )
    scene_counts = {"complete": Counter(), "incomplete": Counter()}
    for key, target in targets.items():
        group = "complete" if key in complete else "incomplete"
        source = sources[key]
        treatments = list(source["candidate_set"]["treatments"])
        native_count = sum(_has_arm(item, "native") for item in treatments)
        heat_only_count = sum(
            _has_arm(item, "heatmap_control") and not _has_arm(item, "native")
            for item in treatments
        )
        treatment_by_id = {
            str(item["treatment_id"]): item for item in treatments
        }
        numeric["step_id"][group].append(float(target["step_id"]))
        numeric["system2_call_index"][group].append(
            float(target["system2_call_index"])
        )
        numeric["source_unique_treatments"][group].append(float(len(treatments)))
        numeric["native_candidate_count"][group].append(float(native_count))
        numeric["heatmap_only_candidate_count"][group].append(float(heat_only_count))
        numeric["selected_unique_treatments"][group].append(
            float(len(set(target["treatment_roles"].values())))
        )
        result = episode_results.get((str(target["scene_id"]), str(target["episode_id"])))
        if result is not None:
            numeric["baseline_episode_success"][group].append(float(result["success"]))
            numeric["baseline_episode_spl"][group].append(float(result["spl"]))
            numeric["baseline_episode_steps"][group].append(float(result["steps"]))
        cycles = [
            float((row.get("termination") or {}).get("future_system2_cycles", 0))
            for row in continuations.get(key) or []
        ]
        if cycles:
            numeric["observed_max_future_cycles"][group].append(max(cycles))
        categorical["planned_run_to_end"][group][str(bool(target["run_to_episode_end"]))] += 1
        for name, value in sorted((target.get("state_strata") or {}).items()):
            categorical[f"strata/{name}"][group][str(bool(value))] += 1
        diagnostic = target.get("diagnostic_selection") or {}
        categorical["diagnostic/selector_disagreement"][group][
            str(bool(diagnostic.get("selector_disagreement")))
        ] += 1
        categorical["diagnostic/heatmap_adds_local_support"][group][
            str(bool(diagnostic.get("heatmap_adds_local_support")))
        ] += 1
        for role in (
            "native_mean",
            "system2_selector",
            "heatmap_token_selector",
            "union_local_oracle",
        ):
            treatment_id = str(target["treatment_roles"][role])
            treatment = treatment_by_id[treatment_id]
            has_native = _has_arm(treatment, "native")
            has_heatmap = _has_arm(treatment, "heatmap_control")
            source_name = (
                "shared"
                if has_native and has_heatmap
                else "native_only"
                if has_native
                else "heatmap_only"
                if has_heatmap
                else "unknown"
            )
            categorical[f"selected_candidate_source/{role}"][group][source_name] += 1
        scene_counts[group][str(target["scene_id"])] += 1
    numeric_report: dict[str, Any] = {}
    for name, groups in numeric.items():
        numeric_report[name] = {
            "complete": _numeric_summary(groups["complete"]),
            "incomplete": _numeric_summary(groups["incomplete"]),
            "standardized_mean_difference_complete_minus_incomplete": (
                _standardized_mean_difference(groups["complete"], groups["incomplete"])
            ),
        }
    categorical_report: dict[str, Any] = {}
    for name, groups in categorical.items():
        values = sorted(set(groups["complete"]) | set(groups["incomplete"]))
        complete_total = sum(groups["complete"].values())
        incomplete_total = sum(groups["incomplete"].values())
        categorical_report[name] = {
            value: {
                "complete_rate": _safe_rate(groups["complete"][value], complete_total),
                "incomplete_rate": _safe_rate(
                    groups["incomplete"][value], incomplete_total
                ),
                "rate_difference": (
                    _safe_rate(groups["complete"][value], complete_total)
                    - _safe_rate(groups["incomplete"][value], incomplete_total)
                    if complete_total and incomplete_total
                    else None
                ),
            }
            for value in values
        }
    scenes = sorted(set(scene_counts["complete"]) | set(scene_counts["incomplete"]))
    c_total = sum(scene_counts["complete"].values())
    i_total = sum(scene_counts["incomplete"].values())
    total_variation = 0.5 * sum(
        abs(
            scene_counts["complete"][scene] / max(1, c_total)
            - scene_counts["incomplete"][scene] / max(1, i_total)
        )
        for scene in scenes
    )
    return {
        "states": len(targets),
        "complete_states": len(complete),
        "incomplete_states": len(targets) - len(complete),
        "complete_definition": "all observed strategy-selected branches have authoritative episode-end outcomes",
        "missing_at_random_supported": False,
        "reason": (
            "completion is partly assigned by run_to_episode_end and partly caused by "
            "natural termination; inspect standardized differences before training"
        ),
        "numeric": numeric_report,
        "categorical": categorical_report,
        "scene_distribution_total_variation": float(total_variation),
    }


def _compare_outcomes(left: dict[str, Any], right: dict[str, Any]) -> int:
    return int(_endpoint_key(left) > _endpoint_key(right)) - int(
        _endpoint_key(left) < _endpoint_key(right)
    )


def _action_disagreement_audit(
    targets: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    continuations: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    disagreement_endpoint: Counter[str] = Counter()
    for key, target in targets.items():
        roles = target["treatment_roles"]
        baseline_id = str(roles["native_mean"])
        system2_id = str(roles["system2_selector"])
        heatmap_id = str(roles["heatmap_token_selector"])
        source_treatments = {
            str(item["treatment_id"]): item
            for item in sources[key]["candidate_set"]["treatments"]
        }
        counts["states"] += 1
        counts["exact_treatment_agreement"] += int(system2_id == heatmap_id)
        counts["exact_spec_agreement"] += int(
            source_treatments[system2_id]["spec"]
            == source_treatments[heatmap_id]["spec"]
        )
        counts["system2_intervention"] += int(system2_id != baseline_id)
        counts["heatmap_intervention"] += int(heatmap_id != baseline_id)
        if system2_id == heatmap_id:
            continue
        counts["disagreement_states"] += 1
        endpoint = {
            str(row["treatment_id"]): row
            for row in continuations.get(key) or []
            if bool(row.get("episode_end_authoritative"))
            and isinstance(row.get("episode_end_outcome"), dict)
        }
        if not {baseline_id, system2_id, heatmap_id}.issubset(endpoint):
            disagreement_endpoint["missing_endpoint_triplet"] += 1
            continue
        disagreement_endpoint["comparable"] += 1
        hm = endpoint[heatmap_id]["episode_end_outcome"]
        s2 = endpoint[system2_id]["episode_end_outcome"]
        base = endpoint[baseline_id]["episode_end_outcome"]
        sign = _compare_outcomes(hm, s2)
        disagreement_endpoint[{1: "heatmap_better", 0: "equal", -1: "heatmap_worse"}[sign]] += 1
        hm_success = int(float(hm["habitat_success"]) > 0.5)
        s2_success = int(float(s2["habitat_success"]) > 0.5)
        base_success = int(float(base["habitat_success"]) > 0.5)
        disagreement_endpoint["heatmap_rescue_vs_baseline"] += int(
            hm_success and not base_success
        )
        disagreement_endpoint["heatmap_destroy_vs_baseline"] += int(
            base_success and not hm_success
        )
        disagreement_endpoint["system2_rescue_vs_baseline"] += int(
            s2_success and not base_success
        )
        disagreement_endpoint["system2_destroy_vs_baseline"] += int(
            base_success and not s2_success
        )
    states = counts["states"]
    disagreements = counts["disagreement_states"]
    return {
        "states": states,
        "exact_treatment_agreement": counts["exact_treatment_agreement"],
        "exact_treatment_agreement_rate": _safe_rate(
            counts["exact_treatment_agreement"], states
        ),
        "exact_treatment_spec_agreement": counts["exact_spec_agreement"],
        "exact_treatment_spec_agreement_rate": _safe_rate(
            counts["exact_spec_agreement"], states
        ),
        "disagreement_states": disagreements,
        "disagreement_rate": _safe_rate(disagreements, states),
        "system2_intervention_rate": _safe_rate(counts["system2_intervention"], states),
        "heatmap_intervention_rate": _safe_rate(counts["heatmap_intervention"], states),
        "ranking_correlation_exported_by_existing_selectors": False,
        "disagreement_endpoint_subset": dict(sorted(disagreement_endpoint.items())),
    }


def _proposal_oracle_audit(
    targets: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    continuations: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    spl_deltas: list[float] = []
    missing_native = missing_union = 0
    source_native = source_union = observed_native = observed_union = 0
    for key, target in targets.items():
        source_treatments = {
            str(item["treatment_id"]): item
            for item in sources[key]["candidate_set"]["treatments"]
        }
        source_native_ids = {
            treatment_id
            for treatment_id, treatment in source_treatments.items()
            if _has_arm(treatment, "native")
        }
        endpoint = {
            str(row["treatment_id"]): row["episode_end_outcome"]
            for row in continuations.get(key) or []
            if bool(row.get("episode_end_authoritative"))
            and isinstance(row.get("episode_end_outcome"), dict)
        }
        source_union += len(source_treatments)
        source_native += len(source_native_ids)
        observed_union += len(endpoint)
        observed_native += len(set(endpoint) & source_native_ids)
        missing_union += len(set(source_treatments) - set(endpoint))
        missing_native += len(source_native_ids - set(endpoint))
        rows = continuations.get(key) or []
        if not rows or not all(bool(row.get("episode_end_authoritative")) for row in rows):
            continue
        native_outcomes = [endpoint[item] for item in endpoint if item in source_native_ids]
        union_outcomes = list(endpoint.values())
        if not native_outcomes or not union_outcomes:
            continue
        counts["complete_observed_states"] += 1
        best_native = max(native_outcomes, key=_endpoint_key)
        best_union = max(union_outcomes, key=_endpoint_key)
        sign = _compare_outcomes(best_union, best_native)
        counts[{1: "union_better", 0: "equal", -1: "union_worse"}[sign]] += 1
        native_success = int(float(best_native["habitat_success"]) > 0.5)
        union_success = int(float(best_union["habitat_success"]) > 0.5)
        counts["incremental_success_rescue"] += int(union_success > native_success)
        spl_deltas.append(float(best_union["habitat_spl"]) - float(best_native["habitat_spl"]))
    exact = missing_union == 0
    return {
        "estimand": "EndpointOracle(C_native_union) - EndpointOracle(C_native)",
        "exact_full_candidate_oracle_identified": exact,
        "status": ("exact" if exact else "observed_strategy_treatment_lower_bound_only"),
        "source_candidate_counts": {"native": source_native, "union": source_union},
        "observed_endpoint_candidate_counts": {
            "native": observed_native,
            "union": observed_union,
        },
        "additional_endpoint_rollouts_required_for_exact_oracle": {
            "native": missing_native,
            "union": missing_union,
        },
        "observed_lower_bound": {
            "states": counts["complete_observed_states"],
            "union_better": counts["union_better"],
            "equal": counts["equal"],
            "union_worse": counts["union_worse"],
            "incremental_success_rescue": counts["incremental_success_rescue"],
            "mean_spl_delta": _mean(spl_deltas),
        },
    }


def _build_scene_folds(
    states: Sequence[EndpointState], *, folds: int, seed: int
) -> tuple[dict[str, int], dict[str, Any]]:
    scene_counts: Counter[str] = Counter(state.scene_id for state in states)
    if len(scene_counts) < folds:
        raise RuntimeError(
            f"need at least {folds} scenes for grouped folds; got {len(scene_counts)}"
        )
    rng = random.Random(seed)
    scenes = list(scene_counts)
    rng.shuffle(scenes)
    scenes.sort(key=lambda scene: scene_counts[scene], reverse=True)
    loads = [0] * folds
    scene_lists: list[list[str]] = [[] for _ in range(folds)]
    mapping: dict[str, int] = {}
    for scene in scenes:
        fold = min(range(folds), key=lambda item: (loads[item], len(scene_lists[item]), item))
        mapping[scene] = fold
        loads[fold] += scene_counts[scene]
        scene_lists[fold].append(scene)
    return mapping, {
        "folds": folds,
        "seed": seed,
        "state_loads": loads,
        "scene_counts": [len(item) for item in scene_lists],
        "scenes": [sorted(item) for item in scene_lists],
        "scene_disjoint": True,
    }


def _zero_context(states: Sequence[EndpointState]) -> list[EndpointState]:
    return [
        dataclasses.replace(
            state,
            metadata=np.zeros_like(state.metadata),
            heatmap_tokens=np.zeros_like(state.heatmap_tokens),
            heatmap_mask=np.zeros_like(state.heatmap_mask),
            geometry=np.zeros_like(state.geometry),
        )
        for state in states
    ]


def _shuffle_context(
    states: Sequence[EndpointState], *, seed: int
) -> tuple[list[EndpointState], dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        groups[state.scene_id].append(index)
    source_for = list(range(len(states)))
    for indices in groups.values():
        if len(indices) < 2:
            continue
        order = list(indices)
        rng.shuffle(order)
        rotated = order[1:] + order[:1]
        for destination, source in zip(order, rotated):
            source_for[destination] = source
    changed = sum(index != source for index, source in enumerate(source_for))
    result: list[EndpointState] = []
    for index, state in enumerate(states):
        context = states[source_for[index]]
        geometry = candidate_heatmap_geometry(
            state.candidate_specs,
            context.geometry_context,
            baseline_index=state.baseline_index,
        )
        result.append(
            dataclasses.replace(
                state,
                metadata=context.metadata,
                heatmap_tokens=context.heatmap_tokens,
                heatmap_mask=context.heatmap_mask,
                geometry=geometry,
            )
        )
    return result, {
        "policy": "within_scene_cyclic_derangement",
        "states": len(states),
        "changed_states": changed,
        "changed_rate": _safe_rate(changed, len(states)),
    }


def _context_mode(
    states: Sequence[EndpointState], *, mode: str, seed: int
) -> tuple[list[EndpointState], dict[str, Any]]:
    if mode == "matched":
        return list(states), {"policy": "matched", "states": len(states), "changed_rate": 0.0}
    if mode == "zeroed":
        return _zero_context(states), {
            "policy": "all_heatmap_context_zeroed",
            "states": len(states),
            "changed_rate": 1.0,
        }
    if mode == "shuffled":
        return _shuffle_context(states, seed=seed)
    raise ValueError(f"unknown context mode: {mode}")


def _collate(states: Sequence[EndpointState]) -> dict[str, Any]:
    batch_size = len(states)
    max_candidates = max(state.candidate.shape[0] for state in states)
    candidate_width = states[0].candidate.shape[1]
    geometry_width = states[0].geometry.shape[1]
    candidate = np.zeros((batch_size, max_candidates, candidate_width), dtype=np.float32)
    geometry = np.zeros((batch_size, max_candidates, geometry_width), dtype=np.float32)
    candidate_mask = np.zeros((batch_size, max_candidates), dtype=np.bool_)
    preference = np.zeros((batch_size, max_candidates), dtype=np.int8)
    preference_weight = np.zeros((batch_size, max_candidates), dtype=np.float32)
    baseline = np.zeros(batch_size, dtype=np.int64)
    for index, state in enumerate(states):
        count = state.candidate.shape[0]
        candidate[index, :count] = state.candidate
        geometry[index, :count] = state.geometry
        candidate_mask[index, :count] = True
        preference[index, :count] = state.preference
        preference_weight[index, :count] = state.preference_weight
        baseline[index] = state.baseline_index
    heatmap_width = states[0].heatmap_tokens.shape[1]
    max_tokens = max(1, max(state.heatmap_tokens.shape[0] for state in states))
    heatmap_tokens = np.zeros((batch_size, max_tokens, heatmap_width), dtype=np.float32)
    heatmap_mask = np.zeros((batch_size, max_tokens), dtype=np.bool_)
    for index, state in enumerate(states):
        count = state.heatmap_tokens.shape[0]
        if count:
            heatmap_tokens[index, :count] = state.heatmap_tokens
            heatmap_mask[index, :count] = state.heatmap_mask
    return {
        "states": list(states),
        "candidate": torch.from_numpy(candidate),
        "geometry": torch.from_numpy(geometry),
        "candidate_mask": torch.from_numpy(candidate_mask),
        "preference": torch.from_numpy(preference),
        "preference_weight": torch.from_numpy(preference_weight),
        "baseline_index": torch.from_numpy(baseline),
        "system2_tokens": torch.from_numpy(
            np.stack([state.system2_tokens for state in states]).astype(np.float32)
        ),
        "metadata": torch.from_numpy(
            np.stack([state.metadata for state in states]).astype(np.float32)
        ),
        "heatmap_tokens": torch.from_numpy(heatmap_tokens),
        "heatmap_mask": torch.from_numpy(heatmap_mask),
    }


class EndpointRanker(nn.Module):
    def __init__(
        self,
        *,
        variant: str,
        candidate_width: int,
        geometry_width: int,
        system2_width: int,
        metadata_width: int,
        heatmap_width: int,
        hidden_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"unsupported endpoint probe variant: {variant}")
        self.variant = variant
        self.scale = float(hidden_width) ** -0.5
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(candidate_width),
            nn.Linear(candidate_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
        )
        contexts = 0
        if variant != "candidate_only":
            self.system2_norm = nn.LayerNorm(system2_width)
            self.system2_key = nn.Linear(system2_width, hidden_width)
            self.system2_value = nn.Linear(system2_width, hidden_width)
            self.system2_query = nn.Linear(hidden_width, hidden_width)
            contexts += 1
        if variant in {
            "candidate_system2_heatmap_metadata",
            "candidate_system2_heatmap_tokens",
        }:
            self.metadata_encoder = nn.Sequential(
                nn.LayerNorm(metadata_width),
                nn.Linear(metadata_width, hidden_width),
                nn.GELU(),
                nn.Linear(hidden_width, hidden_width),
            )
            contexts += 1
        if variant == "candidate_system2_heatmap_geometry":
            self.geometry_encoder = nn.Sequential(
                nn.LayerNorm(geometry_width),
                nn.Linear(geometry_width, hidden_width),
                nn.GELU(),
                nn.Linear(hidden_width, hidden_width),
            )
            contexts += 1
        if variant == "candidate_system2_heatmap_tokens":
            self.heatmap_norm = nn.LayerNorm(heatmap_width)
            self.heatmap_key = nn.Linear(heatmap_width, hidden_width)
            self.heatmap_value = nn.Linear(heatmap_width, hidden_width)
            self.heatmap_query = nn.Linear(hidden_width, hidden_width)
            contexts += 1
        head_width = hidden_width * (1 + 2 * contexts)
        self.head = nn.Sequential(
            nn.LayerNorm(head_width),
            nn.Linear(head_width, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, 1),
        )

    def _attention(
        self,
        candidate: torch.Tensor,
        tokens: torch.Tensor,
        *,
        query: nn.Linear,
        key: nn.Linear,
        value: nn.Linear,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = torch.einsum("bch,bth->bct", query(candidate), key(tokens))
        logits = logits * self.scale
        valid_state: torch.Tensor | None = None
        if mask is not None:
            valid_state = mask.any(dim=1)
            safe = mask.clone()
            safe[~valid_state, 0] = True
            logits = logits.masked_fill(~safe[:, None, :], -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        result = torch.einsum("bct,bth->bch", weights, value(tokens))
        if valid_state is not None:
            result = result * valid_state[:, None, None].to(result.dtype)
        return result

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        candidate = self.candidate_encoder(batch["candidate"])
        pieces = [candidate]
        if self.variant != "candidate_only":
            system2 = self.system2_norm(batch["system2_tokens"])
            context = self._attention(
                candidate,
                system2,
                query=self.system2_query,
                key=self.system2_key,
                value=self.system2_value,
            )
            pieces.extend((context, candidate * context))
        if self.variant in {
            "candidate_system2_heatmap_metadata",
            "candidate_system2_heatmap_tokens",
        }:
            context = self.metadata_encoder(batch["metadata"])[:, None, :]
            context = context.expand_as(candidate)
            pieces.extend((context, candidate * context))
        if self.variant == "candidate_system2_heatmap_geometry":
            context = self.geometry_encoder(batch["geometry"])
            pieces.extend((context, candidate * context))
        if self.variant == "candidate_system2_heatmap_tokens":
            tokens = self.heatmap_norm(batch["heatmap_tokens"])
            context = self._attention(
                candidate,
                tokens,
                query=self.heatmap_query,
                key=self.heatmap_key,
                value=self.heatmap_value,
                mask=batch["heatmap_mask"],
            )
            pieces.extend((context, candidate * context))
        return self.head(torch.cat(pieces, dim=-1)).squeeze(-1)


def baseline_centered_loss(
    scores: torch.Tensor, batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    baseline = scores.gather(1, batch["baseline_index"][:, None])
    difference = scores - baseline
    label = batch["preference"].to(scores.dtype)
    weight = batch["preference_weight"].to(scores.dtype)
    valid = batch["candidate_mask"] & label.ne(0) & weight.gt(0)
    losses = F.softplus(-label * difference) * weight * valid
    denominator = (weight * valid).sum(dim=1)
    active = denominator.gt(0)
    if not active.any():
        return scores.sum() * 0.0
    per_state = losses.sum(dim=1) / denominator.clamp_min(1e-8)
    return per_state[active].mean()


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _loader(
    states: Sequence[EndpointState], *, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        list(states),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=_collate,
        generator=generator,
        drop_last=False,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _new_model(
    first: EndpointState,
    *,
    variant: str,
    hidden_width: int,
    dropout: float,
) -> EndpointRanker:
    return EndpointRanker(
        variant=variant,
        candidate_width=first.candidate.shape[1],
        geometry_width=first.geometry.shape[1],
        system2_width=first.system2_tokens.shape[1],
        metadata_width=first.metadata.shape[0],
        heatmap_width=first.heatmap_tokens.shape[1],
        hidden_width=hidden_width,
        dropout=dropout,
    )


def _train(
    train_states: Sequence[EndpointState],
    validation_states: Sequence[EndpointState],
    *,
    variant: str,
    seed: int,
    device: torch.device,
    hidden_width: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[EndpointRanker, list[dict[str, float]]]:
    _seed_everything(seed)
    model = _new_model(
        train_states[0], variant=variant, hidden_width=hidden_width, dropout=dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_loader = _loader(
        train_states, batch_size=batch_size, shuffle=True, seed=seed
    )
    validation_loader = _loader(
        validation_states, batch_size=batch_size, shuffle=False, seed=seed
    )
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_total = train_count = 0.0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = baseline_centered_loss(model(batch), batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite endpoint loss: {variant} seed={seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(batch["states"])
            train_total += float(loss.detach().cpu()) * count
            train_count += count
        model.eval()
        validation_total = validation_count = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                batch = _move_batch(batch, device)
                loss = baseline_centered_loss(model(batch), batch)
                count = len(batch["states"])
                validation_total += float(loss.detach().cpu()) * count
                validation_count += count
        row = {
            "epoch": float(epoch),
            "train_loss": train_total / max(1.0, train_count),
            "validation_loss": validation_total / max(1.0, validation_count),
        }
        history.append(row)
        if row["validation_loss"] < best_loss - 1e-6:
            best_loss = row["validation_loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("endpoint ranker produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    return model, history


@torch.no_grad()
def _score(
    model: EndpointRanker,
    states: Sequence[EndpointState],
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> list[np.ndarray]:
    model.eval()
    result: list[np.ndarray] = []
    for batch in _loader(states, batch_size=batch_size, shuffle=False, seed=seed):
        moved = _move_batch(batch, device)
        scores = model(moved).detach().float().cpu().numpy()
        for row, state in zip(scores, batch["states"]):
            result.append(np.asarray(row[: len(state.candidate_ids)], dtype=np.float64))
    if len(result) != len(states):
        raise AssertionError("endpoint score/state count mismatch")
    return result


def _margin_rows(
    states: Sequence[EndpointState], scores: Sequence[np.ndarray]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state, score in zip(states, scores):
        nonbaseline = [
            index for index in range(len(score)) if index != state.baseline_index
        ]
        if not nonbaseline:
            selected = state.baseline_index
            margin = -math.inf
        else:
            selected = max(nonbaseline, key=lambda item: (float(score[item]), -item))
            margin = float(score[selected] - score[state.baseline_index])
        rows.append(
            {"state": state, "selected": selected, "margin": margin}
        )
    return rows


def _policy_summary(
    rows: Sequence[dict[str, Any]], selected_indices: Sequence[int]
) -> dict[str, Any]:
    if len(rows) != len(selected_indices):
        raise ValueError("policy row/selection mismatch")
    counters: Counter[str] = Counter()
    spl_deltas: list[float] = []
    oracle_spl_regret: list[float] = []
    for row, selected in zip(rows, selected_indices):
        state: EndpointState = row["state"]
        baseline = state.outcomes[state.baseline_index]
        outcome = state.outcomes[int(selected)]
        oracle = max(state.outcomes, key=_endpoint_key)
        baseline_success = int(float(baseline["habitat_success"]) > 0.5)
        selected_success = int(float(outcome["habitat_success"]) > 0.5)
        oracle_success = int(float(oracle["habitat_success"]) > 0.5)
        counters["states"] += 1
        counters["interventions"] += int(selected != state.baseline_index)
        counters["baseline_success"] += baseline_success
        counters["selected_success"] += selected_success
        counters["oracle_success"] += oracle_success
        counters["rescue"] += int(selected_success and not baseline_success)
        counters["destroy"] += int(baseline_success and not selected_success)
        counters["baseline_success_retained"] += int(
            baseline_success and selected_success
        )
        counters["oracle_success_regret"] += int(
            oracle_success and not selected_success
        )
        if selected != state.baseline_index:
            sign, _, semantic = _endpoint_label(outcome, baseline)
            counters["positive_intervention"] += int(sign > 0)
            counters["negative_intervention"] += int(sign < 0)
            counters[f"intervention_semantics/{semantic}"] += 1
        spl_delta = float(outcome["habitat_spl"]) - float(baseline["habitat_spl"])
        spl_deltas.append(spl_delta)
        oracle_spl_regret.append(
            float(oracle["habitat_spl"]) - float(outcome["habitat_spl"])
        )
    states = counters["states"]
    interventions = counters["interventions"]
    return {
        "states": states,
        "interventions": interventions,
        "intervention_coverage": _safe_rate(interventions, states),
        "baseline_sr": _safe_rate(counters["baseline_success"], states),
        "selected_sr": _safe_rate(counters["selected_success"], states),
        "delta_sr": _safe_rate(
            counters["selected_success"] - counters["baseline_success"], states
        ),
        "mean_delta_spl": _mean(spl_deltas),
        "rescue": counters["rescue"],
        "destroy": counters["destroy"],
        "rescue_rate": _safe_rate(counters["rescue"], states),
        "destroy_rate": _safe_rate(counters["destroy"], states),
        "positive_interventions": counters["positive_intervention"],
        "negative_interventions": counters["negative_intervention"],
        "positive_precision_given_intervention": _safe_rate(
            counters["positive_intervention"], interventions
        ),
        "baseline_success_retention": _safe_rate(
            counters["baseline_success_retained"], counters["baseline_success"]
        ),
        "oracle_sr": _safe_rate(counters["oracle_success"], states),
        "oracle_success_regret": counters["oracle_success_regret"],
        "mean_oracle_spl_regret": _mean(oracle_spl_regret),
        "intervention_semantics": {
            key.split("/", 1)[1]: value
            for key, value in sorted(counters.items())
            if key.startswith("intervention_semantics/")
        },
    }


def _selections_for_threshold(
    rows: Sequence[dict[str, Any]], threshold: float
) -> list[int]:
    return [
        int(row["selected"])
        if float(row["margin"]) > float(threshold)
        else int(row["state"].baseline_index)
        for row in rows
    ]


def _tune_threshold(
    states: Sequence[EndpointState],
    scores: Sequence[np.ndarray],
    *,
    max_destroy_state_rate: float,
) -> tuple[float, dict[str, Any]]:
    rows = _margin_rows(states, scores)
    margins = sorted(
        {float(row["margin"]) for row in rows if math.isfinite(float(row["margin"]))},
        reverse=True,
    )
    thresholds = [
        (float(np.nextafter(margins[0], math.inf)) if margins else 0.0)
    ] + [float(np.nextafter(value, -math.inf)) for value in margins]
    feasible: list[tuple[tuple[float, ...], float, dict[str, Any]]] = []
    for threshold in thresholds:
        summary = _policy_summary(rows, _selections_for_threshold(rows, threshold))
        if float(summary["destroy_rate"] or 0.0) > max_destroy_state_rate:
            continue
        key = (
            float(summary["rescue"] - summary["destroy"]),
            float(summary["mean_delta_spl"] or 0.0),
            float(summary["rescue"]),
            -float(summary["interventions"]),
        )
        feasible.append((key, threshold, summary))
    if not feasible:
        raise RuntimeError("no feasible endpoint abstention threshold")
    _, threshold, summary = max(feasible, key=lambda item: item[0])
    return threshold, summary


def _coverage_selections(
    rows: Sequence[dict[str, Any]], coverage: float
) -> list[int]:
    count = int(round(float(coverage) * len(rows)))
    ordered = sorted(
        range(len(rows)),
        key=lambda index: (float(rows[index]["margin"]), -index),
        reverse=True,
    )
    selected_states = set(ordered[:count])
    return [
        int(row["selected"])
        if index in selected_states
        else int(row["state"].baseline_index)
        for index, row in enumerate(rows)
    ]


def _coverage_curve(
    states: Sequence[EndpointState],
    scores: Sequence[np.ndarray],
    coverages: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    rows = _margin_rows(states, scores)
    curve: list[dict[str, Any]] = []
    selections: dict[str, list[int]] = {}
    for coverage in coverages:
        selected = _coverage_selections(rows, coverage)
        key = f"{coverage:.4f}"
        selections[key] = selected
        point = _policy_summary(rows, selected)
        point["requested_coverage"] = float(coverage)
        curve.append(point)
    return curve, selections


def _ranking_metrics(
    states: Sequence[EndpointState], scores: Sequence[np.ndarray]
) -> dict[str, Any]:
    correct = total = top_oracle = 0
    for state, score in zip(states, scores):
        baseline_score = float(score[state.baseline_index])
        for index, (label, weight) in enumerate(
            zip(state.preference, state.preference_weight)
        ):
            if int(label) == 0 or float(weight) <= 0.0:
                continue
            prediction = int(float(score[index]) > baseline_score) - int(
                float(score[index]) < baseline_score
            )
            correct += int(prediction == int(label))
            total += 1
        predicted = int(np.argmax(score))
        best = max(_endpoint_key(item) for item in state.outcomes)
        top_oracle += int(_endpoint_key(state.outcomes[predicted]) == best)
    return {
        "baseline_centered_pairwise_accuracy": _safe_rate(correct, total),
        "baseline_centered_pairwise_comparisons": total,
        "top1_endpoint_oracle_hit_rate": _safe_rate(top_oracle, len(states)),
    }


def _rank_vector(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(scores, kind="stable"), kind="stable")
    return order.astype(np.float64)


def _ranking_change(
    states: Sequence[EndpointState],
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
) -> dict[str, Any]:
    changed = 0
    correlations: list[float] = []
    for state, a, b in zip(states, left, right):
        changed += int(int(np.argmax(a)) != int(np.argmax(b)))
        if len(a) < 2:
            continue
        ar = _rank_vector(a)
        br = _rank_vector(b)
        if float(ar.std()) <= 1e-12 or float(br.std()) <= 1e-12:
            correlations.append(1.0 if np.array_equal(ar, br) else 0.0)
        else:
            correlations.append(float(np.corrcoef(ar, br)[0, 1]))
    return {
        "states": len(states),
        "top1_changed": changed,
        "top1_changed_rate": _safe_rate(changed, len(states)),
        "mean_spearman": _mean(correlations),
    }


def _aggregate_policy_summaries(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scalar_names = (
        "intervention_coverage",
        "delta_sr",
        "mean_delta_spl",
        "rescue_rate",
        "destroy_rate",
        "positive_precision_given_intervention",
        "baseline_success_retention",
        "mean_oracle_spl_regret",
    )
    result: dict[str, Any] = {}
    for name in scalar_names:
        values = [float(item[name]) for item in items if item.get(name) is not None]
        result[name] = {"mean": _mean(values), "std": _std(values), "values": values}
    return result


def _cluster_bootstrap_difference(
    rows: Sequence[dict[str, Any]],
    left: Sequence[int],
    right: Sequence[int],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["state"].scene_id].append(index)
    scenes = sorted(groups)
    rng = random.Random(seed)

    def state_value(index: int, selected: int) -> tuple[float, float, float, float]:
        state: EndpointState = rows[index]["state"]
        baseline = state.outcomes[state.baseline_index]
        outcome = state.outcomes[int(selected)]
        bs = float(float(baseline["habitat_success"]) > 0.5)
        ss = float(float(outcome["habitat_success"]) > 0.5)
        return (
            ss - bs,
            float(outcome["habitat_spl"]) - float(baseline["habitat_spl"]),
            float(ss > bs),
            float(ss < bs),
        )

    left_values = [state_value(index, selected) for index, selected in enumerate(left)]
    right_values = [state_value(index, selected) for index, selected in enumerate(right)]
    metrics = ("delta_sr", "delta_spl", "rescue_rate", "destroy_rate")
    samples: dict[str, list[float]] = {name: [] for name in metrics}
    for _ in range(replicates):
        indices: list[int] = []
        for scene in (rng.choice(scenes) for _ in scenes):
            indices.extend(groups[scene])
        for metric_index, name in enumerate(metrics):
            delta = np.mean(
                [
                    left_values[index][metric_index] - right_values[index][metric_index]
                    for index in indices
                ]
            )
            samples[name].append(float(delta))
    result: dict[str, Any] = {}
    for name, values in samples.items():
        result[name] = {
            "mean": _mean(values),
            "ci95": [_quantile(values, 0.025), _quantile(values, 0.975)],
            "probability_positive": _safe_rate(sum(value > 0 for value in values), len(values)),
        }
    return result


def _parse_ints(value: str) -> list[int]:
    try:
        result = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("integer list must be nonempty and unique")
    return result


def _parse_variants(value: str) -> list[str]:
    result = [piece.strip() for piece in value.split(",") if piece.strip()]
    invalid = [item for item in result if item not in MODEL_VARIANTS]
    if not result or invalid:
        raise argparse.ArgumentTypeError(
            f"invalid variants {invalid}; valid={MODEL_VARIANTS}"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-audit-root", type=Path, required=True)
    parser.add_argument("--continuation-root", type=Path, required=True)
    parser.add_argument("--targets-root", type=Path, required=True)
    parser.add_argument("--worker-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260812)
    parser.add_argument("--model-seeds", type=_parse_ints, default=_parse_ints("17,42,73"))
    parser.add_argument("--variants", type=_parse_variants, default=list(MODEL_VARIANTS))
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-validation-destroy-state-rate", type=float, default=0.02)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-integrity-check", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-states", type=int, default=0)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.expected_shards <= 64:
        raise ValueError("expected_shards must be in [1,64]")
    if not 3 <= args.folds <= 10:
        raise ValueError("folds must be in [3,10]")
    if args.hidden_width < 16 or args.batch_size < 1:
        raise ValueError("invalid model dimensions")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    if not 0.0 <= args.max_validation_destroy_state_rate < 1.0:
        raise ValueError("invalid validation destroy constraint")
    if args.bootstrap_replicates < 0:
        raise ValueError("bootstrap replicates must be nonnegative")


def _parameter_count(model: nn.Module) -> int:
    return sum(int(value.numel()) for value in model.parameters())


def main() -> int:
    args = parse_args()
    _validate_args(args)
    source_root = args.source_audit_root.expanduser().resolve()
    continuation_root = args.continuation_root.expanduser().resolve()
    targets_root = args.targets_root.expanduser().resolve()
    worker_root = args.worker_root.expanduser().resolve() if args.worker_root else None
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    verify_integrity = not args.skip_integrity_check

    targets, target_payloads = _load_targets(targets_root, args.expected_shards)
    continuations, continuation_manifests = _load_continuations(
        continuation_root,
        args.expected_shards,
        verify_integrity=verify_integrity,
    )
    if set(continuations) != set(targets):
        raise RuntimeError(
            "target/continuation state closure mismatch: "
            f"targets={len(targets)} continuations={len(continuations)}"
        )
    sources, source_manifests = _load_source_targets(
        source_root,
        set(targets),
        args.expected_shards,
        verify_integrity=verify_integrity,
    )
    episode_results = _load_episode_results(worker_root)

    static_report = {
        "schema": SCHEMA,
        "status": "static_audits_complete",
        "integrity_verified": verify_integrity,
        "inputs": {
            "source_audit_root": str(source_root),
            "continuation_root": str(continuation_root),
            "targets_root": str(targets_root),
            "worker_root": str(worker_root) if worker_root else None,
            "source_manifests": len(source_manifests),
            "continuation_manifests": len(continuation_manifests),
            "target_payloads": len(target_payloads),
            "states": len(targets),
            "continuation_rows": sum(len(rows) for rows in continuations.values()),
        },
        "action_disagreement": _action_disagreement_audit(
            targets, sources, continuations
        ),
        "missingness": _missingness_audit(
            targets, sources, continuations, episode_results
        ),
        "proposal_endpoint_oracle": _proposal_oracle_audit(
            targets, sources, continuations
        ),
    }
    _atomic_json(output_dir / "static_audits.json", static_report)
    print(json.dumps(static_report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.preflight_only or args.static_only:
        return 0

    states, state_build_report = _build_endpoint_states(
        targets,
        sources,
        continuations,
        verify_integrity=verify_integrity,
        max_states=args.max_states,
    )
    if len(states) < args.folds * 3:
        raise RuntimeError(f"too few endpoint states for grouped folds: {len(states)}")
    fold_mapping, fold_report = _build_scene_folds(
        states, folds=args.folds, seed=args.fold_seed
    )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {args.device}, but CUDA is unavailable")
    device = torch.device(args.device)
    first = states[0]
    variants_to_run: list[tuple[str, str]] = []
    for variant in args.variants:
        modes = CONTEXT_MODES if variant in HEATMAP_VARIANTS else ("matched",)
        variants_to_run.extend((variant, mode) for mode in modes)

    report: dict[str, Any] = {
        **static_report,
        "status": "training",
        "endpoint_state_build": state_build_report,
        "fold_assignment": fold_report,
        "training": {
            "variants": args.variants,
            "variant_context_modes": [f"{variant}/{mode}" for variant, mode in variants_to_run],
            "model_seeds": args.model_seeds,
            "hidden_width": args.hidden_width,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": str(device),
            "max_validation_destroy_state_rate": args.max_validation_destroy_state_rate,
            "label": "baseline-centered endpoint rescue/destroy plus both-success delta-SPL",
            "both_fail_policy": "ignored_in_primary_loss",
            "state_weighting": "each state equally weighted after within-state pair normalization",
            "candidate_set": "same observed endpoint-labelled native-only treatments for every variant",
        },
        "models": {},
        "comparisons": {},
        "decision": None,
    }
    _atomic_json(output_dir / "endpoint_identifiability_report.json", report)

    all_predictions: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    all_curves: dict[tuple[str, str, int], dict[str, list[int]]] = {}
    all_policy_rows: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for variant, mode in variants_to_run:
        model_key = f"{variant}/{mode}"
        print(f"[endpoint-probe] model={model_key}", flush=True)
        seed_reports: list[dict[str, Any]] = []
        for model_seed in args.model_seeds:
            oof_scores: dict[str, np.ndarray] = {}
            ablation_oof_scores: dict[str, dict[str, np.ndarray]] = (
                {"zeroed": {}, "shuffled": {}}
                if variant in HEATMAP_VARIANTS and mode == "matched"
                else {}
            )
            selected_by_threshold: dict[str, int] = {}
            fold_reports: list[dict[str, Any]] = []
            sensitivity: list[dict[str, Any]] = []
            parameter_count: int | None = None
            for fold in range(args.folds):
                test = [state for state in states if fold_mapping[state.scene_id] == fold]
                validation_fold = (fold + 1) % args.folds
                validation = [
                    state
                    for state in states
                    if fold_mapping[state.scene_id] == validation_fold
                ]
                train = [
                    state
                    for state in states
                    if fold_mapping[state.scene_id] not in {fold, validation_fold}
                ]
                context_seed = args.fold_seed + 1009 * fold
                train_input, train_context = _context_mode(
                    train, mode=mode, seed=context_seed + 11
                )
                validation_input, validation_context = _context_mode(
                    validation, mode=mode, seed=context_seed + 23
                )
                test_input, test_context = _context_mode(
                    test, mode=mode, seed=context_seed + 37
                )
                seed = model_seed + fold * 100_003
                model, history = _train(
                    train_input,
                    validation_input,
                    variant=variant,
                    seed=seed,
                    device=device,
                    hidden_width=args.hidden_width,
                    dropout=args.dropout,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    patience=args.patience,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                )
                parameter_count = _parameter_count(model)
                validation_scores = _score(
                    model,
                    validation_input,
                    batch_size=args.batch_size,
                    device=device,
                    seed=seed,
                )
                threshold, validation_policy = _tune_threshold(
                    validation_input,
                    validation_scores,
                    max_destroy_state_rate=args.max_validation_destroy_state_rate,
                )
                test_scores = _score(
                    model,
                    test_input,
                    batch_size=args.batch_size,
                    device=device,
                    seed=seed,
                )
                rows = _margin_rows(test_input, test_scores)
                selected = _selections_for_threshold(rows, threshold)
                test_policy = _policy_summary(rows, selected)
                for state, score, choice in zip(test, test_scores, selected):
                    oof_scores[state.state_key] = score
                    selected_by_threshold[state.state_key] = int(choice)
                fold_report: dict[str, Any] = {
                    "fold": fold,
                    "train_states": len(train),
                    "validation_states": len(validation),
                    "test_states": len(test),
                    "epochs_ran": len(history),
                    "best_validation_loss": min(
                        row["validation_loss"] for row in history
                    ),
                    "threshold": threshold,
                    "validation_policy": validation_policy,
                    "test_policy": test_policy,
                    "test_ranking": _ranking_metrics(test_input, test_scores),
                    "contexts": {
                        "train": train_context,
                        "validation": validation_context,
                        "test": test_context,
                    },
                }
                if variant in HEATMAP_VARIANTS and mode == "matched":
                    zero_test = _zero_context(test)
                    shuffled_test, shuffle_info = _shuffle_context(
                        test, seed=context_seed + 37
                    )
                    zero_scores = _score(
                        model,
                        zero_test,
                        batch_size=args.batch_size,
                        device=device,
                        seed=seed,
                    )
                    shuffled_scores = _score(
                        model,
                        shuffled_test,
                        batch_size=args.batch_size,
                        device=device,
                        seed=seed,
                    )
                    for state, zero_score, shuffled_score in zip(
                        test, zero_scores, shuffled_scores
                    ):
                        ablation_oof_scores["zeroed"][state.state_key] = zero_score
                        ablation_oof_scores["shuffled"][
                            state.state_key
                        ] = shuffled_score
                    sensitivity.append(
                        {
                            "fold": fold,
                            "matched_vs_zeroed": _ranking_change(
                                test, test_scores, zero_scores
                            ),
                            "matched_vs_shuffled": _ranking_change(
                                test, test_scores, shuffled_scores
                            ),
                            "shuffle": shuffle_info,
                        }
                    )
                fold_reports.append(fold_report)
                print(
                    f"[endpoint-probe] {model_key} seed={model_seed} "
                    f"fold={fold + 1}/{args.folds} states={len(test)} "
                    f"coverage={test_policy['intervention_coverage']:.3f} "
                    f"rescue={test_policy['rescue']} destroy={test_policy['destroy']} "
                    f"dSR={test_policy['delta_sr']:.4f} "
                    f"dSPL={test_policy['mean_delta_spl']:.4f}",
                    flush=True,
                )
            if set(oof_scores) != {state.state_key for state in states}:
                raise RuntimeError(f"OOF prediction closure failure: {model_key} seed={model_seed}")
            ordered_scores = [oof_scores[state.state_key] for state in states]
            rows = _margin_rows(states, ordered_scores)
            threshold_choices = [selected_by_threshold[state.state_key] for state in states]
            threshold_policy = _policy_summary(rows, threshold_choices)
            curve, curve_selections = _coverage_curve(
                states, ordered_scores, DEFAULT_COVERAGES
            )
            same_model_context_ablation: dict[str, Any] = {}
            for ablation_mode, predictions in ablation_oof_scores.items():
                if set(predictions) != {state.state_key for state in states}:
                    raise RuntimeError(
                        "same-model context-ablation OOF closure failure: "
                        f"{model_key} seed={model_seed} mode={ablation_mode}"
                    )
                ablated_scores = [
                    predictions[state.state_key] for state in states
                ]
                ablated_rows = _margin_rows(states, ablated_scores)
                ablated_curve, ablated_selections = _coverage_curve(
                    states, ablated_scores, DEFAULT_COVERAGES
                )
                fixed_coverage: dict[str, Any] = {}
                for coverage in (0.10, 0.20):
                    coverage_key = f"{coverage:.4f}"
                    matched_selection = curve_selections[coverage_key]
                    ablated_selection = ablated_selections[coverage_key]
                    matched_summary = _policy_summary(rows, matched_selection)
                    ablated_summary = _policy_summary(rows, ablated_selection)
                    same_treatment = sum(
                        int(left == right)
                        for left, right in zip(
                            matched_selection, ablated_selection
                        )
                    )
                    fixed_coverage[coverage_key] = {
                        "matched_policy": matched_summary,
                        "ablated_policy": ablated_summary,
                        "matched_minus_ablation": {
                            "delta_sr": float(matched_summary["delta_sr"])
                            - float(ablated_summary["delta_sr"]),
                            "delta_spl": float(matched_summary["mean_delta_spl"])
                            - float(ablated_summary["mean_delta_spl"]),
                            "rescue": int(matched_summary["rescue"])
                            - int(ablated_summary["rescue"]),
                            "destroy": int(matched_summary["destroy"])
                            - int(ablated_summary["destroy"]),
                        },
                        "selected_treatment_agreement_rate": _safe_rate(
                            same_treatment, len(states)
                        ),
                        "scene_cluster_bootstrap": (
                            _cluster_bootstrap_difference(
                                rows,
                                matched_selection,
                                ablated_selection,
                                seed=(
                                    model_seed
                                    + int(coverage * 10_000)
                                    + len(variant)
                                    + len(ablation_mode)
                                ),
                                replicates=args.bootstrap_replicates,
                            )
                            if args.bootstrap_replicates
                            else None
                        ),
                    }
                same_model_context_ablation[ablation_mode] = {
                    "definition": (
                        "the matched-trained model is held fixed; only held-out "
                        f"heatmap context is replaced by {ablation_mode} context"
                    ),
                    "candidate_ranking_change": _ranking_change(
                        states, ordered_scores, ablated_scores
                    ),
                    "oof_ranking": _ranking_metrics(states, ablated_scores),
                    "coverage_curve": ablated_curve,
                    "fixed_coverage": fixed_coverage,
                }
            seed_report = {
                "seed": model_seed,
                "parameters": parameter_count,
                "folds": fold_reports,
                "oof_ranking": _ranking_metrics(states, ordered_scores),
                "fold_tuned_oof_policy": threshold_policy,
                "coverage_curve": curve,
                "matched_model_context_sensitivity": sensitivity,
                "same_model_context_ablation": same_model_context_ablation,
            }
            seed_reports.append(seed_report)
            all_predictions[(variant, mode, model_seed)] = {
                state.state_key: score for state, score in zip(states, ordered_scores)
            }
            all_curves[(variant, mode, model_seed)] = curve_selections
            all_policy_rows[(variant, mode, model_seed)] = rows
            report["models"][model_key] = {
                "runs": seed_reports,
                "aggregate_fold_tuned_policy": _aggregate_policy_summaries(
                    [item["fold_tuned_oof_policy"] for item in seed_reports]
                ),
            }
            _atomic_json(output_dir / "endpoint_identifiability_report.json", report)

    comparison_coverages = (0.10, 0.20)
    comparison_report: dict[str, Any] = {}
    for variant in sorted(HEATMAP_VARIANTS & set(args.variants)):
        variant_report: dict[str, Any] = {}
        for coverage in comparison_coverages:
            coverage_key = f"{coverage:.4f}"
            seed_items: list[dict[str, Any]] = []
            for seed in args.model_seeds:
                matched_key = (variant, "matched", seed)
                rows = all_policy_rows[matched_key]
                matched_selection = all_curves[matched_key][coverage_key]
                matched_summary = _policy_summary(rows, matched_selection)
                controls: dict[str, Any] = {}
                control_keys: list[tuple[str, tuple[str, str, int]]] = []
                if "candidate_system2" in args.variants:
                    control_keys.append(
                        ("candidate_system2", ("candidate_system2", "matched", seed))
                    )
                control_keys.extend(
                    (
                        ("capacity_matched_zeroed", (variant, "zeroed", seed)),
                        ("capacity_matched_shuffled", (variant, "shuffled", seed)),
                    )
                )
                for name, key in control_keys:
                    control_rows = all_policy_rows[key]
                    control_selection = all_curves[key][coverage_key]
                    control_summary = _policy_summary(control_rows, control_selection)
                    bootstrap = (
                        _cluster_bootstrap_difference(
                            rows,
                            matched_selection,
                            control_selection,
                            seed=seed + int(coverage * 10_000) + len(name),
                            replicates=args.bootstrap_replicates,
                        )
                        if args.bootstrap_replicates
                        else None
                    )
                    controls[name] = {
                        "policy": control_summary,
                        "matched_minus_control": {
                            "delta_sr": float(matched_summary["delta_sr"])
                            - float(control_summary["delta_sr"]),
                            "delta_spl": float(matched_summary["mean_delta_spl"])
                            - float(control_summary["mean_delta_spl"]),
                            "rescue": int(matched_summary["rescue"])
                            - int(control_summary["rescue"]),
                            "destroy": int(matched_summary["destroy"])
                            - int(control_summary["destroy"]),
                        },
                        "scene_cluster_bootstrap": bootstrap,
                    }
                seed_items.append(
                    {
                        "seed": seed,
                        "matched_policy": matched_summary,
                        "controls": controls,
                    }
                )
            variant_report[coverage_key] = seed_items
        comparison_report[variant] = variant_report

    cross_model_ranking: dict[str, Any] = {}
    if "candidate_system2" in args.variants:
        for variant in sorted(HEATMAP_VARIANTS & set(args.variants)):
            items = []
            for seed in args.model_seeds:
                base = [
                    all_predictions[("candidate_system2", "matched", seed)][state.state_key]
                    for state in states
                ]
                matched = [
                    all_predictions[(variant, "matched", seed)][state.state_key]
                    for state in states
                ]
                items.append({"seed": seed, **_ranking_change(states, matched, base)})
            cross_model_ranking[f"{variant}_vs_candidate_system2"] = items
    report["comparisons"] = {
        "fixed_coverage": comparison_report,
        "cross_model_ranking": cross_model_ranking,
    }

    token_variant = "candidate_system2_heatmap_tokens"
    token_evidence: list[dict[str, Any]] = []
    if token_variant in args.variants:
        for coverage in comparison_coverages:
            key = f"{coverage:.4f}"
            for item in comparison_report[token_variant][key]:
                token_evidence.append(
                    {
                        "coverage": coverage,
                        "seed": item["seed"],
                        "vs_no_heatmap": item["controls"].get("candidate_system2", {}).get(
                            "matched_minus_control"
                        ),
                        "vs_shuffled": item["controls"]["capacity_matched_shuffled"][
                            "matched_minus_control"
                        ],
                        "vs_zeroed": item["controls"]["capacity_matched_zeroed"][
                            "matched_minus_control"
                        ],
                    }
                )
    strict_positive = bool(token_evidence) and all(
        evidence["vs_no_heatmap"] is not None
        and float(evidence["vs_no_heatmap"]["delta_sr"]) >= 0.0
        and float(evidence["vs_no_heatmap"]["delta_spl"]) > 0.0
        and int(evidence["vs_no_heatmap"]["destroy"]) <= 0
        and float(evidence["vs_shuffled"]["delta_sr"]) >= 0.0
        and float(evidence["vs_shuffled"]["delta_spl"]) > 0.0
        and int(evidence["vs_shuffled"]["destroy"]) <= 0
        for evidence in token_evidence
    )
    report["decision"] = {
        "heatmap_incremental_information_go": strict_positive,
        "criterion": (
            "matched heatmap tokens must beat no-heatmap and capacity-matched "
            "shuffled controls in every seed at 10% and 20% coverage, with no "
            "additional destroys"
        ),
        "status": (
            "go_expand_endpoint_data_then_train_conservative_critic"
            if strict_positive
            else "no_go_for_high_capacity_heatmap_critic_on_current_evidence"
        ),
        "proposal_generator_decision_deferred": not bool(
            static_report["proposal_endpoint_oracle"][
                "exact_full_candidate_oracle_identified"
            ]
        ),
        "one_intervention_policy_only": True,
    }
    report["status"] = "complete"
    _atomic_json(output_dir / "endpoint_identifiability_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

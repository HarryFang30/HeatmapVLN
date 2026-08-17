#!/usr/bin/env python3
"""Causal-support audit for training-free HCTF policies.

This is deliberately an *audit before deployment*.  It uses deployable inputs
to choose treatments, then evaluates those choices with stored Habitat local
outcomes and, where already available, authoritative one-deviation episode-end
continuations.  Simulator goal/path fields are labels only.

The matched-vs-shuffled comparison keeps the candidate, action history,
odometry, and scene fixed.  It swaps only frozen heatmap predictions within a
scene/history-count stratum.  A pose-only arm is reported separately because
history relative poses already contain substantial geometric information.
"""

from __future__ import annotations

import argparse
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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.hctf import (
    VetoThresholds,
    action_distribution_entropy,
    action_edit_fraction,
    candidate_history_features,
    deployable_loop_signals,
    deployable_recovery_partition,
    heatmap_pose_consistency,
    recovery_anchor_risk,
    select_adaptive_prefix,
    select_directional_veto,
)


SCHEMA = "hctf-training-free-audit-v6"
RISK_MARGIN_AUDIT_VALUES = (0.0001, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.04)
PREDICTED_HEATMAP_FIELDS = (
    "coarse_probabilities",
    "spatial_statistics",
    "view_probabilities",
    "none_probability",
)


@dataclasses.dataclass
class AuditState:
    state_key: str
    scene_id: str
    episode_id: str
    step_id: int
    system2_call_index: int
    scene_split: str
    baseline_id: str
    baseline_execute_len: int
    candidates: tuple[dict[str, Any], ...]
    all_treatments: dict[str, dict[str, Any]]
    local_outcomes: dict[str, dict[str, Any]]
    prefix_ids_by_length: dict[int, str]
    entropy: dict[str, float | int]
    loop_signals: dict[str, bool | float]
    recovery_partition: dict[str, Any]
    context: dict[str, np.ndarray]
    heatmap_valid: bool
    source_strata: dict[str, Any]
    risks: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)
    heatmap_consistency: float | None = None
    shuffled_consistency: float | None = None


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
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


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
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


def _json_lines(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_target_metadata(
    root: Path | None, expected_shards: int
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    if root is None:
        return {}, {}
    targets: dict[str, dict[str, Any]] = {}
    scene_splits: dict[str, str] = {}
    for shard_id in range(expected_shards):
        path = root / f"targets_shard_{shard_id:02d}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("shard_id", -1)) != shard_id:
            raise RuntimeError(f"target shard mismatch: {path}")
        for row in payload.get("targets") or []:
            key = str(row["state_key"])
            if key in targets:
                raise RuntimeError(f"duplicate target state {key}")
            targets[key] = row
            scene = str(row["scene_id"])
            split = str(row["scene_split"])
            previous = scene_splits.setdefault(scene, split)
            if previous != split:
                raise RuntimeError(f"scene leakage for {scene}: {previous}/{split}")
    return targets, scene_splits


def _fallback_scene_splits(scenes: Sequence[str], seed: int) -> dict[str, str]:
    shuffled = sorted(set(map(str, scenes)))
    random.Random(seed).shuffle(shuffled)
    result: dict[str, str] = {}
    for index, scene in enumerate(shuffled):
        fraction = index / max(1, len(shuffled))
        result[scene] = "train" if fraction < 0.70 else "validation" if fraction < 0.85 else "test"
    return result


def _load_endpoint_outcomes(
    root: Path | None,
    expected_shards: int,
    *,
    verify_integrity: bool,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    rows = 0
    authoritative = 0
    if root is None:
        return grouped, {"available": False, "rows": 0, "authoritative_rows": 0}
    for shard_id in range(expected_shards):
        shard = root / f"shard_{shard_id:02d}"
        records_path = shard / "records.jsonl"
        manifest_path = shard / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if verify_integrity and _sha256(records_path) != str(
            manifest.get("records_jsonl_sha256")
        ):
            raise RuntimeError(f"continuation hash mismatch: {records_path}")
        for row in _json_lines(records_path):
            rows += 1
            if not bool(row.get("episode_end_authoritative")):
                continue
            outcome = row.get("episode_end_outcome")
            if not isinstance(outcome, dict):
                continue
            source_key = str(row["source_state_key"])
            treatment_id = str(row["treatment_id"])
            if treatment_id in grouped[source_key]:
                # The same treatment can carry several selector roles but is
                # executed only once by the continuation collector.
                raise RuntimeError(
                    f"duplicate endpoint treatment {source_key}/{treatment_id}"
                )
            grouped[source_key][treatment_id] = outcome
            authoritative += 1
    return grouped, {
        "available": True,
        "rows": rows,
        "authoritative_rows": authoritative,
        "states": len(grouped),
    }


def _empty_context() -> dict[str, np.ndarray]:
    return {
        "fixed_history_mask": np.zeros((0,), dtype=np.bool_),
        "fixed_history_rel_poses": np.zeros((0, 4), dtype=np.float32),
        "fixed_history_age_steps": np.zeros((0,), dtype=np.int32),
        "history_rank": np.zeros((0,), dtype=np.float32),
    }


def _load_context_and_policy_history(
    array_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], bool]:
    with np.load(array_path, allow_pickle=False) as arrays:
        valid = bool(
            np.asarray(arrays["heatmap_sample_valid"]).reshape(-1)[0]
            if "heatmap_sample_valid" in arrays.files
            else False
        )
        if "fixed_history_mask" in arrays.files:
            context = {
                name: np.asarray(arrays[name]).copy()
                for name in (
                    "fixed_history_mask",
                    "fixed_history_rel_poses",
                    "fixed_history_age_steps",
                    "history_rank",
                )
                if name in arrays.files
            }
            if "fixed_history_age_steps" not in context:
                raise RuntimeError(
                    f"fixed history context lacks age steps: {array_path}"
                )
            if "history_rank" not in context:
                context["history_rank"] = np.zeros_like(
                    context["fixed_history_mask"], dtype=np.float32
                )
            if valid:
                missing = [name for name in PREDICTED_HEATMAP_FIELDS if name not in arrays.files]
                if missing:
                    raise RuntimeError(f"valid heatmap archive lacks {missing}: {array_path}")
                for name in PREDICTED_HEATMAP_FIELDS:
                    context[name] = np.asarray(arrays[name]).copy()
        else:
            context = _empty_context()
        history = {
            "executed_actions": np.asarray(
                arrays["client_executed_action_prefix"], dtype=np.int8
            ).copy(),
            "visited_body_poses": np.asarray(
                arrays["client_visited_body_poses"], dtype=np.float32
            ).copy(),
            "current_body_pose": np.asarray(
                arrays["client_current_body_pose"], dtype=np.float32
            ).copy(),
            "local_treatment_ids": [
                value.decode("ascii")
                for value in np.asarray(arrays["local_treatment_id_ascii"]).reshape(-1)
            ],
            "local_endpoint_poses": np.asarray(
                arrays["local_endpoint_poses"], dtype=np.float32
            ).copy(),
            "local_pose_trace_lengths": np.asarray(
                arrays["local_pose_trace_lengths"], dtype=np.int16
            ).copy(),
            "local_pose_traces": np.asarray(
                arrays["local_pose_traces"], dtype=np.float32
            ).copy(),
        }
    return context, history, valid


def _augment_dense_local_labels(
    local_outcomes: dict[str, dict[str, Any]],
    history: Mapping[str, Any],
    *,
    exclude_recent_poses: int = 4,
    revisit_radius_m: float = 0.35,
) -> None:
    """Replace the collector's immediate-neighbour revisit with loop revisit.

    The collector's broad ``revisit`` checks the endpoint against *every*
    visited pose.  A turn-only treatment therefore becomes a revisit merely
    because it remains near the immediately preceding pose.  HCTF instead
    needs a loop label: return to a pose older than a small recency exclusion.
    Stored Habitat endpoints are used strictly as labels here.
    """

    treatment_ids = list(history["local_treatment_ids"])
    endpoints = np.asarray(history["local_endpoint_poses"], dtype=np.float32)
    visited = np.asarray(history["visited_body_poses"], dtype=np.float32)
    if endpoints.shape != (len(treatment_ids), 4, 4):
        raise RuntimeError("local endpoint/treatment array mismatch")
    if set(treatment_ids) != set(local_outcomes):
        raise RuntimeError("local endpoint ids differ from JSON local outcomes")
    older = (
        visited[:-int(exclude_recent_poses), :3, 3]
        if len(visited) > int(exclude_recent_poses)
        else np.empty((0, 3), dtype=np.float32)
    )
    recent = (
        visited[-int(exclude_recent_poses) :, :3, 3]
        if len(visited)
        else np.empty((0, 3), dtype=np.float32)
    )
    for treatment_id, pose in zip(treatment_ids, endpoints):
        endpoint = pose[:3, 3]
        loop_revisit = bool(
            len(older)
            and float(np.linalg.norm(older - endpoint[None, :], axis=1).min())
            <= revisit_radius_m
        )
        recent_revisit = bool(
            len(recent)
            and float(np.linalg.norm(recent - endpoint[None, :], axis=1).min())
            <= revisit_radius_m
        )
        outcome = local_outcomes[treatment_id]
        outcome["collector_any_history_revisit"] = bool(outcome["revisit"])
        outcome["loop_revisit_excluding_recent4"] = loop_revisit
        outcome["recent4_revisit"] = recent_revisit


def _augment_recovery_geometry_labels(
    local_outcomes: dict[str, dict[str, Any]],
    history: Mapping[str, Any],
    partition: Mapping[str, Any],
) -> None:
    """Attach simulator-trace geometry used strictly as recovery audit labels."""

    for outcome in local_outcomes.values():
        outcome["recovery_geometry_available"] = False
    if not bool(partition.get("ready", False)):
        return
    treatment_ids = list(history["local_treatment_ids"])
    traces = np.asarray(history["local_pose_traces"], dtype=np.float32)
    lengths = np.asarray(history["local_pose_trace_lengths"], dtype=np.int64)
    visited = np.asarray(history["visited_body_poses"], dtype=np.float32)
    current = np.asarray(history["current_body_pose"], dtype=np.float32)
    if traces.shape[0] != len(treatment_ids) or lengths.shape != (len(treatment_ids),):
        raise RuntimeError("local trace/treatment array mismatch")
    if set(treatment_ids) != set(local_outcomes):
        raise RuntimeError("local trace ids differ from JSON local outcomes")

    anchor_step = int(partition["anchor_capture_step"])
    if not 0 <= anchor_step <= len(visited):
        return
    anchor_pose = current if anchor_step == len(visited) else visited[anchor_step]
    anchor_xy = anchor_pose[(0, 2), 3]
    capture_steps = np.asarray(partition["capture_steps"], dtype=np.int64)
    loop_mask = np.asarray(partition["loop_history_mask"], dtype=np.bool_)
    loop_steps = capture_steps[loop_mask]
    loop_positions: list[np.ndarray] = []
    for step in loop_steps:
        if 0 <= int(step) <= len(visited):
            pose = current if int(step) == len(visited) else visited[int(step)]
            loop_positions.append(pose[(0, 2), 3])
    if not loop_positions:
        return
    loop_xy = np.stack(loop_positions).astype(np.float32)
    current_xy = current[(0, 2), 3]
    current_anchor_distance = float(np.linalg.norm(current_xy - anchor_xy))
    current_loop_distance = float(
        np.linalg.norm(loop_xy - current_xy[None, :], axis=1).min()
    )
    for treatment_id, raw_trace, raw_length in zip(treatment_ids, traces, lengths):
        length = max(1, min(int(raw_length), len(raw_trace)))
        trace_xy = raw_trace[:length, (0, 2), 3]
        endpoint_xy = trace_xy[-1]
        anchor_distance = float(np.linalg.norm(endpoint_xy - anchor_xy))
        loop_distance = float(
            np.linalg.norm(loop_xy - endpoint_xy[None, :], axis=1).min()
        )
        outcome = local_outcomes[treatment_id]
        outcome["recovery_geometry_available"] = True
        outcome["recovery_anchor_endpoint_distance_m"] = anchor_distance
        outcome["recovery_loop_endpoint_clearance_m"] = loop_distance
        outcome["recovery_anchor_progress_m"] = (
            current_anchor_distance - anchor_distance
        )
        outcome["recovery_loop_escape_m"] = loop_distance - current_loop_distance
        outcome["recovery_geometry_progress_m"] = (
            current_anchor_distance
            - anchor_distance
            + loop_distance
            - current_loop_distance
        )
        # Lower is the privileged geometry target corresponding to the
        # deployable heatmap energy.  It is never consumed by the policy.
        outcome["recovery_geometry_energy_m"] = anchor_distance - loop_distance


def _native_candidate_contract(
    record: dict[str, Any],
) -> tuple[
    str,
    tuple[dict[str, Any], ...],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[int, str],
    dict[str, float | int],
]:
    treatments = {
        str(item["treatment_id"]): item
        for item in record["candidate_set"]["treatments"]
    }
    local = {
        str(item["treatment_id"]): item for item in record["local_outcomes"]
    }
    if set(local) != set(treatments):
        raise RuntimeError(f"local/treatment mismatch at {record['state_key']}")
    entries = record["candidate_set"]["source_entries"]
    sample_entries = sorted(
        (
            item
            for item in entries
            if item.get("arm") == "native" and item.get("aggregation") == "sample"
        ),
        key=lambda item: int(item["sample_index"]),
    )
    sample_ids = [str(item["base_treatment_id"]) for item in sample_entries]
    expected_total = int(record["candidate_set"]["native_sample_total"])
    if len(sample_ids) != expected_total:
        raise RuntimeError(f"native sample count mismatch at {record['state_key']}")
    baseline_id = str(record["candidate_set"]["baselines"]["native_trajectory_mean"])
    base_ids = set(sample_ids)
    base_ids.add(baseline_id)
    candidates = tuple(treatments[item] for item in sorted(base_ids))
    if baseline_id not in {str(item["treatment_id"]) for item in candidates}:
        raise AssertionError("native mean omitted from candidates")

    mean_entries = [
        item
        for item in entries
        if item.get("arm") == "native"
        and item.get("aggregation") == "trajectory_mean"
    ]
    if len(mean_entries) != 1:
        raise RuntimeError(f"expected one native mean entry at {record['state_key']}")
    prefix_ids: dict[int, str] = {}
    for treatment_id in mean_entries[0]["treatment_ids"]:
        spec = treatments[str(treatment_id)]["spec"]
        length = int(spec["execute_len"])
        prefix_ids[length] = str(treatment_id)
    baseline_length = int(treatments[baseline_id]["spec"]["execute_len"])
    prefix_ids[baseline_length] = baseline_id
    return (
        baseline_id,
        candidates,
        treatments,
        local,
        prefix_ids,
        action_distribution_entropy(sample_ids, sample_total=expected_total),
    )


def _load_source_states(
    source_root: Path,
    *,
    expected_shards: int,
    target_keys: set[str] | None,
    target_metadata: Mapping[str, dict[str, Any]],
    scene_splits: Mapping[str, str],
    verify_integrity: bool,
    max_states: int,
) -> tuple[list[AuditState], dict[str, Any]]:
    states: list[AuditState] = []
    source_rows = 0
    selected_rows = 0
    reached_limit = False
    for shard_id in range(expected_shards):
        shard = source_root / f"shard_{shard_id:02d}"
        records_path = shard / "records.jsonl"
        manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        if verify_integrity and _sha256(records_path) != str(
            manifest.get("records_jsonl_sha256")
        ):
            raise RuntimeError(f"source hash mismatch: {records_path}")
        for record in _json_lines(records_path):
            source_rows += 1
            key = str(record["state_key"])
            if target_keys is not None and key not in target_keys:
                continue
            if max_states > 0 and selected_rows >= max_states:
                reached_limit = True
                break
            scene = str(record["scene_id"])
            metadata = target_metadata.get(key) or {}
            split = str(metadata.get("scene_split") or scene_splits.get(scene) or "unknown")
            (
                baseline_id,
                candidates,
                treatments,
                local,
                prefix_ids,
                entropy,
            ) = _native_candidate_contract(record)
            array_path = shard / str(record["array_file"])
            context, history, heatmap_valid = _load_context_and_policy_history(array_path)
            loop_signals = deployable_loop_signals(
                history["executed_actions"],
                history["visited_body_poses"],
                history["current_body_pose"],
            )
            recovery_partition = deployable_recovery_partition(
                fixed_history_mask=context["fixed_history_mask"],
                fixed_history_age_steps=context["fixed_history_age_steps"],
                executed_actions=history["executed_actions"],
                visited_body_poses=history["visited_body_poses"],
                current_body_pose=history["current_body_pose"],
            )
            _augment_dense_local_labels(local, history)
            _augment_recovery_geometry_labels(local, history, recovery_partition)
            baseline_length = int(treatments[baseline_id]["spec"]["execute_len"])
            states.append(
                AuditState(
                    state_key=key,
                    scene_id=scene,
                    episode_id=str(record["episode_id"]),
                    step_id=int(record["step_id"]),
                    system2_call_index=int(record["system2_call_index"]),
                    scene_split=split,
                    baseline_id=baseline_id,
                    baseline_execute_len=baseline_length,
                    candidates=candidates,
                    all_treatments=treatments,
                    local_outcomes=local,
                    prefix_ids_by_length=prefix_ids,
                    entropy=entropy,
                    loop_signals=loop_signals,
                    recovery_partition=recovery_partition,
                    context=context,
                    heatmap_valid=heatmap_valid,
                    source_strata=dict(record.get("state_strata") or {}),
                )
            )
            selected_rows += 1
        print(
            f"[hctf-audit] source shard {shard_id + 1}/{expected_shards}: "
            f"selected={selected_rows}",
            flush=True,
        )
        if reached_limit:
            break
    if not states:
        raise RuntimeError("no source states selected")
    if target_keys is not None and max_states <= 0:
        missing = target_keys - {state.state_key for state in states}
        if missing:
            raise RuntimeError(f"source audit lacks {len(missing)} target states")
    return states, {
        "source_rows_scanned": source_rows,
        "selected_states": len(states),
        "targeted_only": target_keys is not None,
    }


def _history_bucket(state: AuditState) -> int:
    mask = np.asarray(state.context["fixed_history_mask"]).reshape(-1)
    count = int(mask.astype(bool).sum())
    return 0 if count == 0 else 1 if count <= 2 else 2 if count <= 4 else 3


def _shuffle_donors(
    states: Sequence[AuditState], seed: int
) -> tuple[list[int], dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        groups[(state.scene_id, _history_bucket(state))].append(index)
    donors = list(range(len(states)))
    singleton_groups = 0
    for indices in groups.values():
        if len(indices) < 2:
            singleton_groups += 1
            continue
        order = list(indices)
        rng.shuffle(order)
        rotated = order[1:] + order[:1]
        for destination, source in zip(order, rotated):
            donors[destination] = source
    # A singleton history bucket can borrow from the same scene and validity
    # class while still keeping pose/action history fixed.
    by_scene_valid: dict[tuple[str, bool], list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        by_scene_valid[(state.scene_id, state.heatmap_valid)].append(index)
    for index, source in enumerate(donors):
        if source != index or not states[index].heatmap_valid:
            continue
        alternatives = [
            value
            for value in by_scene_valid[(states[index].scene_id, True)]
            if value != index
        ]
        if alternatives:
            donors[index] = alternatives[
                int(hashlib.sha256(f"{seed}:{states[index].state_key}".encode()).hexdigest(), 16)
                % len(alternatives)
            ]
    changed = sum(index != source for index, source in enumerate(donors))
    return donors, {
        "policy": "within_scene_history_bucket_derangement",
        "seed": seed,
        "groups": len(groups),
        "singleton_groups": singleton_groups,
        "changed_states": changed,
        "changed_rate": changed / len(states),
    }


def _shuffled_context(current: AuditState, donor: AuditState) -> dict[str, np.ndarray]:
    result = {name: np.asarray(value) for name, value in current.context.items()}
    if current.heatmap_valid and donor.heatmap_valid:
        for name in PREDICTED_HEATMAP_FIELDS:
            result[name] = np.asarray(donor.context[name])
    return result


def _compute_risks(
    states: Sequence[AuditState], donors: Sequence[int]
) -> dict[str, Any]:
    candidate_count = 0
    valid = 0
    for index, state in enumerate(states):
        donor_context = _shuffled_context(state, states[donors[index]])
        state.heatmap_consistency = heatmap_pose_consistency(state.context)
        state.shuffled_consistency = heatmap_pose_consistency(donor_context)
        state.risks = {
            "pose_only": {},
            "matched_heatmap": {},
            "shuffled_heatmap": {},
            "matched_hybrid": {},
            "shuffled_hybrid": {},
            "pose_recovery_anchor": {},
            "matched_recovery_anchor": {},
            "shuffled_recovery_anchor": {},
        }
        for candidate in state.candidates:
            treatment_id = str(candidate["treatment_id"])
            actions = candidate["spec"]["actions"]
            matched = candidate_history_features(actions, state.context)
            shuffled = candidate_history_features(actions, donor_context)
            state.risks["pose_only"][treatment_id] = matched["pose_only"]
            state.risks["matched_heatmap"][treatment_id] = matched["heatmap_only"]
            state.risks["shuffled_heatmap"][treatment_id] = shuffled["heatmap_only"]
            state.risks["matched_hybrid"][treatment_id] = matched["hybrid"]
            state.risks["shuffled_hybrid"][treatment_id] = shuffled["hybrid"]
            state.risks["pose_recovery_anchor"][treatment_id] = recovery_anchor_risk(
                matched, state.recovery_partition, source="pose"
            )["risk"]
            state.risks["matched_recovery_anchor"][treatment_id] = recovery_anchor_risk(
                matched, state.recovery_partition, source="heatmap"
            )["risk"]
            state.risks["shuffled_recovery_anchor"][treatment_id] = recovery_anchor_risk(
                shuffled, state.recovery_partition, source="heatmap"
            )["risk"]
            candidate_count += 1
        valid += int(state.heatmap_valid)
        if (index + 1) % 100 == 0 or index + 1 == len(states):
            print(
                f"[hctf-audit] geometry {index + 1}/{len(states)}; "
                f"candidates={candidate_count}",
                flush=True,
            )
    return {
        "states": len(states),
        "heatmap_valid_states": valid,
        "heatmap_valid_rate": valid / len(states),
        "native_base_candidates": candidate_count,
    }


def _safe_mean(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    values = np.asarray(scores, dtype=np.float64)
    target = np.asarray(labels, dtype=np.bool_)
    positives = int(target.sum())
    negatives = int((~target).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[target].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _within_state_pairwise_auc(
    states: Sequence[AuditState],
    *,
    mode: str,
    label_name: str,
    loop_only: bool = False,
) -> dict[str, Any]:
    correct = 0.0
    pairs = 0
    informative_states = 0
    for state in states:
        if loop_only and not bool(state.loop_signals["confirmed"]):
            continue
        values: list[tuple[float, bool]] = []
        for candidate in state.candidates:
            treatment_id = str(candidate["treatment_id"])
            outcome = state.local_outcomes[treatment_id]
            if label_name == "revisit":
                label = bool(outcome["loop_revisit_excluding_recent4"])
            elif label_name == "collector_revisit":
                label = bool(outcome["collector_any_history_revisit"])
            elif label_name == "collision":
                label = int(outcome["collision_or_stuck_count"]) > 0
            elif label_name == "enter_then_leave":
                label = bool(outcome["entered_euclidean_success_radius"]) and bool(
                    outcome["left_euclidean_success_radius"]
                )
            elif label_name == "badness":
                label = bool(_local_badness(outcome)[0] > 0)
            elif label_name == "recovery_badness":
                label = bool(_recovery_local_badness(outcome)[0] > 0)
            else:
                raise ValueError(label_name)
            values.append((float(state.risks[mode][treatment_id]), label))
        positives = [score for score, label in values if label]
        negatives = [score for score, label in values if not label]
        if not positives or not negatives:
            continue
        informative_states += 1
        for positive in positives:
            for negative in negatives:
                pairs += 1
                correct += float(positive > negative) + 0.5 * float(positive == negative)
    return {
        "auc": correct / pairs if pairs else None,
        "positive_negative_pairs": pairs,
        "informative_states": informative_states,
    }


def _baseline_label_auc(
    states: Sequence[AuditState], *, mode: str, label_name: str
) -> dict[str, Any]:
    scores: list[float] = []
    labels: list[bool] = []
    for state in states:
        outcome = state.local_outcomes[state.baseline_id]
        if label_name == "revisit":
            label = bool(outcome["loop_revisit_excluding_recent4"])
        elif label_name == "collector_revisit":
            label = bool(outcome["collector_any_history_revisit"])
        elif label_name == "collision":
            label = int(outcome["collision_or_stuck_count"]) > 0
        elif label_name == "enter_then_leave":
            label = bool(outcome["entered_euclidean_success_radius"]) and bool(
                outcome["left_euclidean_success_radius"]
            )
        elif label_name == "badness":
            label = bool(_local_badness(outcome)[0] > 0)
        elif label_name == "recovery_badness":
            label = bool(_recovery_local_badness(outcome)[0] > 0)
        else:
            raise ValueError(label_name)
        scores.append(float(state.risks[mode][state.baseline_id]))
        labels.append(label)
    return {
        "auc": _auc(scores, labels),
        "positive_states": sum(labels),
        "states": len(states),
    }


def _candidate_risk_identifiability(states: Sequence[AuditState]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "validation", "test", "all"):
        selected = list(states) if split == "all" else [s for s in states if s.scene_split == split]
        mode_report: dict[str, Any] = {}
        for mode in (
            "pose_only",
            "matched_heatmap",
            "shuffled_heatmap",
            "matched_hybrid",
            "shuffled_hybrid",
            "pose_recovery_anchor",
            "matched_recovery_anchor",
            "shuffled_recovery_anchor",
        ):
            scores: list[float] = []
            revisit: list[bool] = []
            collector_revisit: list[bool] = []
            collision: list[bool] = []
            leave: list[bool] = []
            for state in selected:
                for candidate in state.candidates:
                    treatment_id = str(candidate["treatment_id"])
                    outcome = state.local_outcomes[treatment_id]
                    scores.append(float(state.risks[mode][treatment_id]))
                    revisit.append(
                        bool(outcome["loop_revisit_excluding_recent4"])
                    )
                    collector_revisit.append(
                        bool(outcome["collector_any_history_revisit"])
                    )
                    collision.append(int(outcome["collision_or_stuck_count"]) > 0)
                    leave.append(
                        bool(outcome["entered_euclidean_success_radius"])
                        and bool(outcome["left_euclidean_success_radius"])
                    )
            mode_report[mode] = {
                "candidate_count": len(scores),
                "revisit_positive": sum(revisit),
                "revisit_auc": _auc(scores, revisit),
                "collector_any_history_revisit_positive": sum(collector_revisit),
                "collector_any_history_revisit_auc": _auc(
                    scores, collector_revisit
                ),
                "collision_positive": sum(collision),
                "collision_auc": _auc(scores, collision),
                "enter_then_leave_positive": sum(leave),
                "enter_then_leave_auc": _auc(scores, leave),
                "within_state": {
                    label: _within_state_pairwise_auc(
                        selected, mode=mode, label_name=label
                    )
                    for label in (
                        "revisit",
                        "collector_revisit",
                        "collision",
                        "enter_then_leave",
                        "badness",
                        "recovery_badness",
                    )
                },
                "within_confirmed_loop_states": {
                    label: _within_state_pairwise_auc(
                        selected,
                        mode=mode,
                        label_name=label,
                        loop_only=True,
                    )
                    for label in (
                        "revisit",
                        "collector_revisit",
                        "collision",
                        "badness",
                        "recovery_badness",
                    )
                },
                "native_mean_only": {
                    label: _baseline_label_auc(
                        selected, mode=mode, label_name=label
                    )
                    for label in (
                        "revisit",
                        "collector_revisit",
                        "collision",
                        "enter_then_leave",
                        "badness",
                        "recovery_badness",
                    )
                },
            }
        result[split] = mode_report
    return result


def _recovery_geometry_fidelity(states: Sequence[AuditState]) -> dict[str, Any]:
    """Candidate ordering against stored local traces, never policy inputs."""

    result: dict[str, Any] = {}
    for split in ("train", "validation", "test", "all"):
        selected = (
            list(states)
            if split == "all"
            else [state for state in states if state.scene_split == split]
        )
        split_report: dict[str, Any] = {}
        for mode in (
            "pose_recovery_anchor",
            "matched_recovery_anchor",
            "shuffled_recovery_anchor",
        ):
            correct = 0.0
            pairs = 0
            informative_states = 0
            eligible_states = 0
            top_agree = 0
            top_improves = 0
            top_worsens = 0
            top_regret: list[float] = []
            native_regret: list[float] = []
            for state in selected:
                if not bool(state.recovery_partition["ready"]):
                    continue
                if mode != "pose_recovery_anchor" and not state.heatmap_valid:
                    continue
                values: list[tuple[str, float, float]] = []
                for candidate in state.candidates:
                    treatment_id = str(candidate["treatment_id"])
                    outcome = state.local_outcomes[treatment_id]
                    if not bool(outcome.get("recovery_geometry_available")):
                        continue
                    values.append(
                        (
                            treatment_id,
                            float(state.risks[mode][treatment_id]),
                            float(outcome["recovery_geometry_energy_m"]),
                        )
                    )
                if len(values) < 2:
                    continue
                eligible_states += 1
                state_pairs = 0
                for left_index, (_, left_risk, left_truth) in enumerate(values):
                    for _, right_risk, right_truth in values[left_index + 1 :]:
                        truth_delta = left_truth - right_truth
                        if abs(truth_delta) <= 1e-6:
                            continue
                        risk_delta = left_risk - right_risk
                        pairs += 1
                        state_pairs += 1
                        correct += float(risk_delta * truth_delta > 0.0)
                        correct += 0.5 * float(abs(risk_delta) <= 1e-12)
                if state_pairs == 0:
                    continue
                informative_states += 1
                predicted = min(values, key=lambda value: (value[1], value[0]))
                oracle = min(values, key=lambda value: (value[2], value[0]))
                baseline = next(
                    value for value in values if value[0] == state.baseline_id
                )
                top_agree += int(predicted[0] == oracle[0])
                regret = max(0.0, predicted[2] - oracle[2])
                top_regret.append(regret)
                native_regret.append(max(0.0, baseline[2] - oracle[2]))
                top_improves += int(predicted[2] < baseline[2] - 1e-6)
                top_worsens += int(predicted[2] > baseline[2] + 1e-6)
            state_count = len(top_regret)
            split_report[mode] = {
                "policy_input_excludes_local_traces": True,
                "heatmap_valid_only": mode != "pose_recovery_anchor",
                "eligible_states": eligible_states,
                "informative_states": informative_states,
                "top1_states": state_count,
                "candidate_pairs": pairs,
                "within_state_ordering_concordance": (
                    correct / pairs if pairs else None
                ),
                "top1_oracle_agreement_rate": (
                    top_agree / state_count if state_count else None
                ),
                "mean_top1_geometry_regret_m": _safe_mean(top_regret),
                "mean_native_geometry_regret_m": _safe_mean(native_regret),
                "top1_improves_over_native_rate": (
                    top_improves / state_count if state_count else None
                ),
                "top1_worsens_vs_native_rate": (
                    top_worsens / state_count if state_count else None
                ),
            }
        result[split] = split_report
    return result


def _recovery_candidate_support(states: Sequence[AuditState]) -> dict[str, Any]:
    """Separate candidate-set support from heatmap ranking support."""

    configurations = {
        "relaxed": (1.0 / 32.0, 0.75),
        "conservative": (2.0 / 32.0, 0.50),
        "strict": (3.0 / 32.0, 0.50),
    }
    modes = (
        "pose_recovery_anchor",
        "matched_recovery_anchor",
        "shuffled_recovery_anchor",
    )
    margin_key = lambda value: str(value).replace(".", "p")
    result: dict[str, Any] = {}
    for split in ("train", "validation", "test", "all"):
        selected_states = (
            list(states)
            if split == "all"
            else [state for state in states if state.scene_split == split]
        )
        split_report: dict[str, Any] = {}
        for config_name, (minimum_mass, maximum_edit) in configurations.items():
            counts: Counter[str] = Counter()
            for state in selected_states:
                if not bool(state.recovery_partition["ready"]):
                    continue
                counts["loop_states"] += 1
                baseline = next(
                    candidate
                    for candidate in state.candidates
                    if str(candidate["treatment_id"]) == state.baseline_id
                )
                baseline_actions = tuple(
                    int(value) for value in baseline["spec"]["actions"]
                )
                baseline_first = baseline_actions[0] if baseline_actions else None
                baseline_bad = bool(
                    _recovery_local_badness(
                        state.local_outcomes[state.baseline_id]
                    )[0]
                    > 0
                )
                counts["native_bad_states"] += int(baseline_bad)
                counts["native_safe_states"] += int(not baseline_bad)
                eligible: list[tuple[str, bool]] = []
                for candidate in state.candidates:
                    treatment_id = str(candidate["treatment_id"])
                    if treatment_id == state.baseline_id:
                        continue
                    actions = tuple(
                        int(value) for value in candidate["spec"]["actions"]
                    )
                    if not actions or actions[0] == baseline_first:
                        continue
                    if float(candidate.get("native_sample_mass", 0.0)) + 1e-12 < minimum_mass:
                        continue
                    if action_edit_fraction(actions, baseline_actions) > maximum_edit + 1e-12:
                        continue
                    bad = bool(
                        _recovery_local_badness(
                            state.local_outcomes[treatment_id]
                        )[0]
                        > 0
                    )
                    eligible.append((treatment_id, bad))
                if not eligible:
                    counts["no_eligible_alternative"] += 1
                    continue
                counts["eligible_alternative_states"] += 1
                safe = [(treatment_id, bad) for treatment_id, bad in eligible if not bad]
                if baseline_bad and safe:
                    counts["oracle_safe_alternative_states"] += 1
                for mode in modes:
                    best_id, best_bad = min(
                        eligible,
                        key=lambda item: (
                            float(state.risks[mode][item[0]]),
                            item[0],
                        ),
                    )
                    baseline_risk = float(state.risks[mode][state.baseline_id])
                    best_risk = float(state.risks[mode][best_id])
                    for margin in RISK_MARGIN_AUDIT_VALUES:
                        key = margin_key(margin)
                        if best_risk <= baseline_risk - margin + 1e-12:
                            counts[f"{mode}/{key}/intervention"] += 1
                            if baseline_bad and not best_bad:
                                counts[f"{mode}/{key}/rescue_support"] += 1
                            if not baseline_bad and best_bad:
                                counts[f"{mode}/{key}/destroy_risk"] += 1
                    if baseline_bad:
                        for margin in (0.0, *RISK_MARGIN_AUDIT_VALUES):
                            if any(
                                not bad
                                and float(state.risks[mode][treatment_id])
                                <= baseline_risk - margin + 1e-12
                                for treatment_id, bad in eligible
                            ):
                                key = margin_key(margin)
                                counts[f"{mode}/safe_support_margin_{key}"] += 1
            oracle = counts["oracle_safe_alternative_states"]
            mode_report: dict[str, Any] = {}
            for mode in modes:
                safe_support_by_margin = {
                    margin_key(margin): counts[
                        f"{mode}/safe_support_margin_{margin_key(margin)}"
                    ]
                    for margin in (0.0, *RISK_MARGIN_AUDIT_VALUES)
                }
                policy_support_by_margin = {
                    margin_key(margin): {
                        "interventions": counts[
                            f"{mode}/{margin_key(margin)}/intervention"
                        ],
                        "rescue_support": counts[
                            f"{mode}/{margin_key(margin)}/rescue_support"
                        ],
                        "destroy_risk": counts[
                            f"{mode}/{margin_key(margin)}/destroy_risk"
                        ],
                    }
                    for margin in RISK_MARGIN_AUDIT_VALUES
                }
                mode_report[mode] = {
                    "safe_support_margin_0": counts[
                        f"{mode}/safe_support_margin_0p0"
                    ],
                    "safe_support_margin_0p02": counts[
                        f"{mode}/safe_support_margin_0p02"
                    ],
                    "safe_support_margin_0p04": counts[
                        f"{mode}/safe_support_margin_0p04"
                    ],
                    "margin_0p02_interventions": counts[
                        f"{mode}/0p02/intervention"
                    ],
                    "margin_0p02_rescue_support": counts[
                        f"{mode}/0p02/rescue_support"
                    ],
                    "margin_0p02_destroy_risk": counts[
                        f"{mode}/0p02/destroy_risk"
                    ],
                    "safe_support_rate_given_oracle": (
                        counts[f"{mode}/safe_support_margin_0p02"] / oracle
                        if oracle
                        else None
                    ),
                    "safe_support_by_margin": safe_support_by_margin,
                    "argmin_policy_support_by_margin": policy_support_by_margin,
                }
            split_report[config_name] = {
                "minimum_native_mass": minimum_mass,
                "maximum_edit_fraction": maximum_edit,
                "loop_states": counts["loop_states"],
                "native_bad_states": counts["native_bad_states"],
                "native_safe_states": counts["native_safe_states"],
                "eligible_alternative_states": counts[
                    "eligible_alternative_states"
                ],
                "no_eligible_alternative": counts["no_eligible_alternative"],
                "oracle_safe_alternative_states": oracle,
                "modes": mode_report,
            }
        result[split] = split_report
    return result


def _local_badness(outcome: Mapping[str, Any]) -> tuple[int, dict[str, int]]:
    leave = int(
        bool(outcome["entered_euclidean_success_radius"])
        and bool(outcome["left_euclidean_success_radius"])
    )
    collision = int(outcome["collision_or_stuck_count"])
    revisit = int(
        bool(
            outcome.get(
                "loop_revisit_excluding_recent4",
                outcome["revisit"],
            )
        )
    )
    return 4 * leave + 2 * min(collision, 2) + revisit, {
        "enter_then_leave": leave,
        "collision_or_stuck": collision,
        "loop_revisit_excluding_recent4": revisit,
    }


def _compare_local(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_badness, candidate_events = _local_badness(candidate)
    baseline_badness, baseline_events = _local_badness(baseline)
    progress_delta = float(candidate["route_progress_delta_m"]) - float(
        baseline["route_progress_delta_m"]
    )
    goal_delta = float(candidate["endpoint_euclidean_goal_distance_m"]) - float(
        baseline["endpoint_euclidean_goal_distance_m"]
    )
    if candidate_badness < baseline_badness:
        semantic = "safety_rescue"
    elif candidate_badness > baseline_badness:
        semantic = "safety_destroy"
    elif progress_delta < -0.25 and goal_delta > 0.25:
        semantic = "navigation_regression"
    elif progress_delta > 0.25 and goal_delta <= 0.25:
        semantic = "navigation_improvement"
    else:
        semantic = "neutral"
    return {
        "semantic": semantic,
        "badness_delta": candidate_badness - baseline_badness,
        "candidate_events": candidate_events,
        "baseline_events": baseline_events,
        "route_progress_delta_difference_m": progress_delta,
        "endpoint_goal_distance_difference_m": goal_delta,
    }


def _recovery_local_badness(
    outcome: Mapping[str, Any],
) -> tuple[int, dict[str, int]]:
    """Recovery safety excludes revisiting the explicitly selected anchor."""

    leave = int(
        bool(outcome["entered_euclidean_success_radius"])
        and bool(outcome["left_euclidean_success_radius"])
    )
    collision = int(outcome["collision_or_stuck_count"])
    return 4 * leave + 2 * min(collision, 2), {
        "enter_then_leave": leave,
        "collision_or_stuck": collision,
    }


def _compare_recovery_local(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    """Fair local comparison for recovery without declaring anchor return bad."""

    candidate_badness, candidate_events = _recovery_local_badness(candidate)
    baseline_badness, baseline_events = _recovery_local_badness(baseline)
    progress_delta = float(candidate["route_progress_delta_m"]) - float(
        baseline["route_progress_delta_m"]
    )
    goal_delta = float(candidate["endpoint_euclidean_goal_distance_m"]) - float(
        baseline["endpoint_euclidean_goal_distance_m"]
    )
    if candidate_badness < baseline_badness:
        semantic = "safety_rescue"
    elif candidate_badness > baseline_badness:
        semantic = "safety_destroy"
    elif progress_delta < -0.25 and goal_delta > 0.25:
        semantic = "navigation_regression"
    elif progress_delta > 0.25 and goal_delta <= 0.25:
        semantic = "navigation_improvement"
    else:
        semantic = "neutral"
    candidate_geometry = candidate.get("recovery_geometry_progress_m")
    baseline_geometry = baseline.get("recovery_geometry_progress_m")
    return {
        "semantic": semantic,
        "badness_delta": candidate_badness - baseline_badness,
        "candidate_events": candidate_events,
        "baseline_events": baseline_events,
        "route_progress_delta_difference_m": progress_delta,
        "endpoint_goal_distance_difference_m": goal_delta,
        "recovery_geometry_progress_difference_m": (
            float(candidate_geometry) - float(baseline_geometry)
            if candidate_geometry is not None and baseline_geometry is not None
            else None
        ),
    }


def _state_veto_decision(
    state: AuditState,
    *,
    mode: str,
    thresholds: VetoThresholds,
) -> dict[str, Any]:
    return select_directional_veto(
        baseline_id=state.baseline_id,
        candidates=state.candidates,
        risks=state.risks[mode],
        loop_confirmed=bool(state.loop_signals["confirmed"]),
        thresholds=thresholds,
    )


def _episode_capped_veto_rows(
    states: Sequence[AuditState],
    *,
    mode: str,
    thresholds: VetoThresholds,
) -> list[dict[str, Any]]:
    by_episode: dict[tuple[str, str], list[AuditState]] = defaultdict(list)
    for state in states:
        by_episode[(state.scene_id, state.episode_id)].append(state)
    rows: list[dict[str, Any]] = []
    for values in by_episode.values():
        intervened = False
        for state in sorted(values, key=lambda item: (item.step_id, item.state_key)):
            if intervened:
                decision = {
                    "treatment_id": state.baseline_id,
                    "intervened": False,
                    "reason": "episode_directional_budget_exhausted",
                }
            else:
                decision = _state_veto_decision(state, mode=mode, thresholds=thresholds)
                intervened = bool(decision["intervened"])
            rows.append({"state": state, "decision": decision})
    return rows


def _summarize_local_decisions(
    rows: Sequence[dict[str, Any]], *, comparison_mode: str = "normal"
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    progress: list[float] = []
    goal: list[float] = []
    recovery_geometry: list[float] = []
    selected_rows: list[dict[str, Any]] = []
    episodes = {(row["state"].scene_id, row["state"].episode_id) for row in rows}
    for row in rows:
        state: AuditState = row["state"]
        decision = row["decision"]
        counts["states"] += 1
        counts[f"reason/{decision['reason']}"] += 1
        if not decision["intervened"]:
            continue
        treatment_id = str(decision["treatment_id"])
        counts["interventions"] += 1
        compare = (
            _compare_recovery_local
            if comparison_mode == "recovery"
            else _compare_local
        )
        comparison = compare(
            state.local_outcomes[treatment_id], state.local_outcomes[state.baseline_id]
        )
        counts[comparison["semantic"]] += 1
        progress.append(float(comparison["route_progress_delta_difference_m"]))
        goal.append(float(comparison["endpoint_goal_distance_difference_m"]))
        if comparison.get("recovery_geometry_progress_difference_m") is not None:
            recovery_geometry.append(
                float(comparison["recovery_geometry_progress_difference_m"])
            )
        selected_rows.append(
            {
                "state_key": state.state_key,
                "scene_split": state.scene_split,
                "treatment_id": treatment_id,
                "baseline_id": state.baseline_id,
                "comparison": comparison,
            }
        )
    interventions = counts["interventions"]
    rescues = counts["safety_rescue"]
    destroys = counts["safety_destroy"] + counts["navigation_regression"]
    return {
        "states": counts["states"],
        "episodes": len(episodes),
        "interventions": interventions,
        "intervention_rate_per_episode": interventions / max(1, len(episodes)),
        "safety_rescue": rescues,
        "safety_destroy": counts["safety_destroy"],
        "navigation_improvement": counts["navigation_improvement"],
        "navigation_regression": counts["navigation_regression"],
        "neutral": counts["neutral"],
        "rescue_destroy_ratio": rescues / destroys if destroys else None,
        "destroy_free_with_rescue": bool(rescues and not destroys),
        "net_conservative_utility": rescues - 2 * destroys,
        "mean_route_progress_delta_difference_m": _safe_mean(progress),
        "mean_endpoint_goal_distance_difference_m": _safe_mean(goal),
        "mean_recovery_geometry_progress_difference_m": _safe_mean(
            recovery_geometry
        ),
        "reason_counts": {
            key.split("/", 1)[1]: value
            for key, value in sorted(counts.items())
            if key.startswith("reason/")
        },
        "selected_rows": selected_rows,
    }


def _threshold_grid() -> Iterable[VetoThresholds]:
    for risk_on in (0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35):
        for margin in (*RISK_MARGIN_AUDIT_VALUES, 0.06, 0.08, 0.12, 0.16):
            if margin >= risk_on + 0.10:
                continue
            for minimum_mass in (1 / 32, 2 / 32, 3 / 32, 4 / 32):
                for maximum_edit in (0.25, 0.50, 0.75):
                    yield VetoThresholds(
                        risk_on=risk_on,
                        risk_margin=margin,
                        minimum_native_mass=minimum_mass,
                        maximum_edit_fraction=maximum_edit,
                    )


def _tune_thresholds(
    states: Sequence[AuditState], *, mode: str, comparison_mode: str = "normal"
) -> tuple[VetoThresholds, dict[str, Any]]:
    train = [state for state in states if state.scene_split == "train"]
    if not train:
        raise RuntimeError("threshold tuning requires train scenes")
    scored: list[tuple[tuple[float, ...], VetoThresholds, dict[str, Any]]] = []
    disabled = VetoThresholds(
        risk_on=1.01,
        risk_margin=0.0,
        minimum_native_mass=1.0,
        maximum_edit_fraction=0.0,
    )
    for thresholds in (disabled, *_threshold_grid()):
        summary = _summarize_local_decisions(
            _episode_capped_veto_rows(train, mode=mode, thresholds=thresholds),
            comparison_mode=comparison_mode,
        )
        interventions = int(summary["interventions"])
        coverage = float(summary["intervention_rate_per_episode"])
        # The route is explicitly conservative.  Above 20% coverage is
        # disfavoured before any endpoint evidence exists.
        coverage_penalty = max(0.0, coverage - 0.20) * max(1, summary["episodes"])
        objective = float(summary["net_conservative_utility"]) - 4.0 * coverage_penalty
        key = (
            objective,
            float(summary["safety_rescue"]),
            -float(summary["safety_destroy"] + summary["navigation_regression"]),
            -coverage,
            float(thresholds.risk_on),
            float(thresholds.risk_margin),
        )
        scored.append((key, thresholds, summary))
    key, selected, summary = max(scored, key=lambda item: item[0])
    active = [item for item in scored if item[2]["interventions"]]
    best_active = max(active, key=lambda item: item[0]) if active else None
    return selected, {
        "status": (
            "disabled_by_train_dense_safety_labels"
            if selected is disabled
            else "selected_on_train_dense_safety_labels"
        ),
        "configs_evaluated": len(scored),
        "objective": key[0],
        "selected": dataclasses.asdict(selected),
        "train_summary": {k: v for k, v in summary.items() if k != "selected_rows"},
        "best_active_alternative": (
            None
            if best_active is None
            else {
                "objective": best_active[0][0],
                "thresholds": dataclasses.asdict(best_active[1]),
                "train_summary": {
                    k: v
                    for k, v in best_active[2].items()
                    if k != "selected_rows"
                },
            }
        ),
        "endpoint_labels_used_for_tuning": False,
    }


def _prefix_rows(
    states: Sequence[AuditState],
    thresholds: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for state in states:
        if thresholds is None:
            decision = {
                "treatment_id": state.baseline_id,
                "intervened": False,
                "reason": "adaptive_prefix_disabled_by_train_audit",
                "normalized_entropy": float(state.entropy["normalized_entropy"]),
                "native_execute_len": state.baseline_execute_len,
                "selected_execute_len": state.baseline_execute_len,
            }
        else:
            high_threshold, medium_threshold = thresholds
            decision = select_adaptive_prefix(
                baseline_id=state.baseline_id,
                baseline_execute_len=state.baseline_execute_len,
                prefix_ids_by_length=state.prefix_ids_by_length,
                normalized_entropy=float(state.entropy["normalized_entropy"]),
                high_threshold=high_threshold,
                medium_threshold=medium_threshold,
            )
        result.append({"state": state, "decision": decision})
    return result


def _tune_prefix_thresholds(
    states: Sequence[AuditState],
) -> tuple[tuple[float, float] | None, dict[str, Any]]:
    train = [state for state in states if state.scene_split == "train"]
    candidates: list[tuple[tuple[float, ...], tuple[float, float] | None, dict[str, Any]]] = []
    # ``None`` is an explicit, valid training-free result: entropy does not
    # justify shortening on the available train scenes.
    for thresholds in (
        None,
        (0.80, 0.60),
        (0.85, 0.70),
        (0.90, 0.75),
        (0.95, 0.80),
        (1.00, 0.85),
        (1.00, 0.90),
        (1.00, 0.95),
    ):
        summary = _summarize_local_decisions(_prefix_rows(train, thresholds))
        destroys = int(summary["safety_destroy"]) + int(summary["navigation_regression"])
        objective = int(summary["safety_rescue"]) - 2 * destroys
        # Prefer fewer interventions and the disabled policy on exact ties.
        key = (
            float(objective),
            -float(destroys),
            float(summary["safety_rescue"]),
            -float(summary["interventions"]),
            float(thresholds is None),
        )
        candidates.append((key, thresholds, summary))
    key, selected, summary = max(candidates, key=lambda item: item[0])
    return selected, {
        "status": (
            "disabled_by_train_dense_safety_labels"
            if selected is None
            else "selected_on_train_dense_safety_labels"
        ),
        "configs_evaluated": len(candidates),
        "objective": key[0],
        "selected": (
            None
            if selected is None
            else {"high_threshold": selected[0], "medium_threshold": selected[1]}
        ),
        "train_summary": {k: v for k, v in summary.items() if k != "selected_rows"},
        "endpoint_labels_used_for_tuning": False,
    }


def _combo_rows(
    states: Sequence[AuditState],
    *,
    mode: str,
    thresholds: VetoThresholds,
    prefix_thresholds: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    veto = {
        row["state"].state_key: row
        for row in _episode_capped_veto_rows(states, mode=mode, thresholds=thresholds)
    }
    prefix = {
        row["state"].state_key: row
        for row in _prefix_rows(states, prefix_thresholds)
    }
    result: list[dict[str, Any]] = []
    for state in states:
        direction = veto[state.state_key]["decision"]
        decision = direction if direction["intervened"] else prefix[state.state_key]["decision"]
        result.append({"state": state, "decision": decision})
    return result


def _endpoint_summary(
    rows: Sequence[dict[str, Any]],
    endpoint: Mapping[str, Mapping[str, dict[str, Any]]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    spl_deltas: list[float] = []
    missing_states: list[str] = []
    for row in rows:
        state: AuditState = row["state"]
        decision = row["decision"]
        if not decision["intervened"]:
            continue
        counts["interventions"] += 1
        treatment_id = str(decision["treatment_id"])
        outcomes = endpoint.get(state.state_key) or {}
        if state.baseline_id not in outcomes or treatment_id not in outcomes:
            counts["missing_exact_pair"] += 1
            if len(missing_states) < 20:
                missing_states.append(state.state_key)
            continue
        counts["exact_pairs"] += 1
        baseline = outcomes[state.baseline_id]
        candidate = outcomes[treatment_id]
        baseline_success = int(float(baseline["habitat_success"]) > 0.5)
        candidate_success = int(float(candidate["habitat_success"]) > 0.5)
        if candidate_success > baseline_success:
            counts["sr_rescue"] += 1
        elif candidate_success < baseline_success:
            counts["sr_destroy"] += 1
        else:
            counts["sr_equal"] += 1
        if candidate_success and baseline_success:
            spl_deltas.append(float(candidate["habitat_spl"]) - float(baseline["habitat_spl"]))
    exact = counts["exact_pairs"]
    return {
        "interventions": counts["interventions"],
        "exact_endpoint_pairs": exact,
        "exact_pair_coverage": exact / counts["interventions"] if counts["interventions"] else None,
        "missing_exact_pairs": counts["missing_exact_pair"],
        "sr_rescue": counts["sr_rescue"],
        "sr_destroy": counts["sr_destroy"],
        "sr_equal": counts["sr_equal"],
        "net_sr_flips": counts["sr_rescue"] - counts["sr_destroy"],
        "mean_spl_delta_when_both_success": _safe_mean(spl_deltas),
        "missingness_warning": (
            "Exact endpoint coverage is selection-dependent; do not extrapolate the observed rate."
            if counts["missing_exact_pair"]
            else None
        ),
        "example_missing_states": missing_states,
    }


def _entropy_overcommit_report(states: Sequence[AuditState]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "validation", "test", "all"):
        selected = list(states) if split == "all" else [s for s in states if s.scene_split == split]
        scores: list[float] = []
        labels: list[bool] = []
        for state in selected:
            baseline_badness, _ = _local_badness(state.local_outcomes[state.baseline_id])
            shorter = [
                state.local_outcomes[treatment_id]
                for length, treatment_id in state.prefix_ids_by_length.items()
                if 0 < length < state.baseline_execute_len
            ]
            overcommit = any(_local_badness(outcome)[0] < baseline_badness for outcome in shorter)
            scores.append(float(state.entropy["normalized_entropy"]))
            labels.append(overcommit)
        result[split] = {
            "states": len(selected),
            "dense_overcommit_positive": sum(labels),
            "action_entropy_auc": _auc(scores, labels),
            "entropy_mean": _safe_mean(scores),
        }
    return result


def _policy_reports(
    states: Sequence[AuditState],
    endpoint: Mapping[str, Mapping[str, dict[str, Any]]],
    *,
    matched_thresholds: VetoThresholds,
    pose_thresholds: VetoThresholds,
    recovery_thresholds: VetoThresholds,
    pose_recovery_thresholds: VetoThresholds,
    prefix_thresholds: tuple[float, float] | None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    report: dict[str, Any] = {}
    decision_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation", "test", "all"):
        selected = list(states) if split == "all" else [s for s in states if s.scene_split == split]
        arms = {
            "adaptive_prefix": _prefix_rows(selected, prefix_thresholds),
            "pose_only_veto": _episode_capped_veto_rows(
                selected, mode="pose_only", thresholds=pose_thresholds
            ),
            "matched_heatmap_veto": _episode_capped_veto_rows(
                selected, mode="matched_heatmap", thresholds=matched_thresholds
            ),
            "shuffled_heatmap_veto": _episode_capped_veto_rows(
                selected, mode="shuffled_heatmap", thresholds=matched_thresholds
            ),
            "pose_recovery_filter": _episode_capped_veto_rows(
                selected,
                mode="pose_recovery_anchor",
                thresholds=pose_recovery_thresholds,
            ),
            "matched_recovery_filter": _episode_capped_veto_rows(
                selected,
                mode="matched_recovery_anchor",
                thresholds=recovery_thresholds,
            ),
            "shuffled_recovery_filter": _episode_capped_veto_rows(
                selected,
                mode="shuffled_recovery_anchor",
                thresholds=recovery_thresholds,
            ),
            "matched_combo": _combo_rows(
                selected,
                mode="matched_heatmap",
                thresholds=matched_thresholds,
                prefix_thresholds=prefix_thresholds,
            ),
            "shuffled_combo": _combo_rows(
                selected,
                mode="shuffled_heatmap",
                thresholds=matched_thresholds,
                prefix_thresholds=prefix_thresholds,
            ),
        }
        split_report: dict[str, Any] = {}
        for name, rows in arms.items():
            local = _summarize_local_decisions(
                rows,
                comparison_mode=(
                    "recovery" if name.endswith("recovery_filter") else "normal"
                ),
            )
            split_report[name] = {
                "local_dense": {k: v for k, v in local.items() if k != "selected_rows"},
                "authoritative_endpoint": _endpoint_summary(rows, endpoint),
            }
            if split == "all":
                decision_rows[name] = rows
        report[split] = split_report
    return report, decision_rows


def _context_summary(states: Sequence[AuditState], shuffle_meta: dict[str, Any]) -> dict[str, Any]:
    matched = [s.heatmap_consistency for s in states if s.heatmap_consistency is not None]
    shuffled = [s.shuffled_consistency for s in states if s.shuffled_consistency is not None]
    loops = Counter()
    recovery = Counter()
    for state in states:
        for name, value in state.loop_signals.items():
            if isinstance(value, (bool, np.bool_)):
                loops[name] += int(value)
        recovery[f"reason/{state.recovery_partition['reason']}"] += 1
        recovery["ready"] += int(bool(state.recovery_partition["ready"]))
    return {
        "shuffle": shuffle_meta,
        "matched_heatmap_pose_consistency_mean": _safe_mean(matched),
        "shuffled_heatmap_pose_consistency_mean": _safe_mean(shuffled),
        "matched_minus_shuffled_consistency": (
            _safe_mean(matched) - _safe_mean(shuffled) if matched and shuffled else None
        ),
        "deployable_loop_signal_counts": dict(sorted(loops.items())),
        "deployable_recovery_partition": {
            "ready_states": recovery["ready"],
            "reason_counts": {
                key.split("/", 1)[1]: value
                for key, value in sorted(recovery.items())
                if key.startswith("reason/")
            },
        },
    }


def _serialize_decisions(
    states: Sequence[AuditState],
    decisions: Mapping[str, Sequence[dict[str, Any]]],
    endpoint: Mapping[str, Mapping[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_arm = {
        arm: {row["state"].state_key: row["decision"] for row in rows}
        for arm, rows in decisions.items()
    }
    result: list[dict[str, Any]] = []
    for state in sorted(states, key=lambda s: (s.scene_split, s.scene_id, s.episode_id, s.step_id)):
        arm_values: dict[str, Any] = {}
        for arm, mapping in by_arm.items():
            decision = dict(mapping[state.state_key])
            treatment_id = str(decision["treatment_id"])
            decision["exact_endpoint_pair_available"] = (
                state.baseline_id in (endpoint.get(state.state_key) or {})
                and treatment_id in (endpoint.get(state.state_key) or {})
            )
            arm_values[arm] = decision
        result.append(
            {
                "schema": "hctf-policy-decision-v6",
                "state_key": state.state_key,
                "scene_id": state.scene_id,
                "episode_id": state.episode_id,
                "step_id": state.step_id,
                "system2_call_index": state.system2_call_index,
                "scene_split": state.scene_split,
                "native_mean_treatment_id": state.baseline_id,
                "heatmap_valid": state.heatmap_valid,
                "loop_signals": state.loop_signals,
                "recovery_partition": {
                    "ready": bool(state.recovery_partition["ready"]),
                    "anchor_index": int(state.recovery_partition["anchor_index"]),
                    "anchor_capture_step": int(
                        state.recovery_partition["anchor_capture_step"]
                    ),
                    "loop_start_step": int(
                        state.recovery_partition["loop_start_step"]
                    ),
                    "loop_history_indices": np.flatnonzero(
                        state.recovery_partition["loop_history_mask"]
                    ).astype(int).tolist(),
                    "reason": str(state.recovery_partition["reason"]),
                },
                "action_entropy": state.entropy,
                "arms": arm_values,
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--targets-root", type=Path)
    parser.add_argument("--continuation-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=8)
    parser.add_argument("--shuffle-seed", type=int, default=20260812)
    parser.add_argument("--max-states", type=int, default=0)
    parser.add_argument(
        "--all-source-states",
        action="store_true",
        help="Use every source state instead of the existing continuation target cohort.",
    )
    parser.add_argument("--verify-integrity", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_shards <= 0 or args.max_states < 0:
        raise ValueError("expected-shards must be positive and max-states non-negative")
    targets, scene_splits = _load_target_metadata(args.targets_root, args.expected_shards)
    if not args.all_source_states and not targets:
        raise ValueError("targeted audit requires --targets-root")
    target_keys = None if args.all_source_states else set(targets)
    states, source_meta = _load_source_states(
        args.source_root,
        expected_shards=args.expected_shards,
        target_keys=target_keys,
        target_metadata=targets,
        scene_splits=scene_splits,
        verify_integrity=args.verify_integrity,
        max_states=args.max_states,
    )
    if not scene_splits:
        fallback = _fallback_scene_splits([state.scene_id for state in states], args.shuffle_seed)
        for state in states:
            state.scene_split = fallback[state.scene_id]
        scene_splits = fallback
    donors, shuffle_meta = _shuffle_donors(states, args.shuffle_seed)
    geometry_meta = _compute_risks(states, donors)
    endpoint, endpoint_meta = _load_endpoint_outcomes(
        args.continuation_root,
        args.expected_shards,
        verify_integrity=args.verify_integrity,
    )

    matched_thresholds, matched_tuning = _tune_thresholds(
        states, mode="matched_heatmap"
    )
    pose_thresholds, pose_tuning = _tune_thresholds(states, mode="pose_only")
    recovery_thresholds, recovery_tuning = _tune_thresholds(
        states, mode="matched_recovery_anchor", comparison_mode="recovery"
    )
    pose_recovery_thresholds, pose_recovery_tuning = _tune_thresholds(
        states, mode="pose_recovery_anchor", comparison_mode="recovery"
    )
    prefix_thresholds, prefix_tuning = _tune_prefix_thresholds(states)
    policies, policy_rows = _policy_reports(
        states,
        endpoint,
        matched_thresholds=matched_thresholds,
        pose_thresholds=pose_thresholds,
        recovery_thresholds=recovery_thresholds,
        pose_recovery_thresholds=pose_recovery_thresholds,
        prefix_thresholds=prefix_thresholds,
    )
    candidate_identifiability = _candidate_risk_identifiability(states)
    recovery_geometry_fidelity = _recovery_geometry_fidelity(states)
    recovery_candidate_support = _recovery_candidate_support(states)
    entropy_report = _entropy_overcommit_report(states)
    context_report = _context_summary(states, shuffle_meta)

    validation_matched = policies.get("validation", {}).get("matched_heatmap_veto", {})
    validation_shuffled = policies.get("validation", {}).get("shuffled_heatmap_veto", {})
    matched_local = validation_matched.get("local_dense") or {}
    shuffled_local = validation_shuffled.get("local_dense") or {}
    veto_incremental_supported = bool(
        int(matched_local.get("net_conservative_utility", 0))
        > int(shuffled_local.get("net_conservative_utility", 0))
        and int(matched_local.get("safety_rescue", 0))
        >= 2
        * max(
            1,
            int(matched_local.get("safety_destroy", 0))
            + int(matched_local.get("navigation_regression", 0)),
        )
    )
    validation_recovery = policies.get("validation", {}).get(
        "matched_recovery_filter", {}
    )
    validation_shuffled_recovery = policies.get("validation", {}).get(
        "shuffled_recovery_filter", {}
    )
    matched_recovery_local = validation_recovery.get("local_dense") or {}
    shuffled_recovery_local = (
        validation_shuffled_recovery.get("local_dense") or {}
    )
    recovery_incremental_supported = bool(
        int(matched_recovery_local.get("net_conservative_utility", 0))
        > int(shuffled_recovery_local.get("net_conservative_utility", 0))
        and int(matched_recovery_local.get("safety_rescue", 0))
        >= 2
        * max(
            1,
            int(matched_recovery_local.get("safety_destroy", 0))
            + int(matched_recovery_local.get("navigation_regression", 0)),
        )
    )
    incremental_supported = bool(
        veto_incremental_supported or recovery_incremental_supported
    )
    validation_safety = candidate_identifiability["validation"]
    test_safety = candidate_identifiability["test"]
    matched_validation_safety_auc = validation_safety[
        "matched_recovery_anchor"
    ]["within_confirmed_loop_states"]["recovery_badness"]["auc"]
    shuffled_validation_safety_auc = validation_safety[
        "shuffled_recovery_anchor"
    ]["within_confirmed_loop_states"]["recovery_badness"]["auc"]
    matched_test_safety_auc = test_safety[
        "matched_recovery_anchor"
    ]["within_confirmed_loop_states"]["recovery_badness"]["auc"]
    shuffled_test_safety_auc = test_safety[
        "shuffled_recovery_anchor"
    ]["within_confirmed_loop_states"]["recovery_badness"]["auc"]
    candidate_specific_safety_signal = bool(
        matched_validation_safety_auc is not None
        and shuffled_validation_safety_auc is not None
        and matched_test_safety_auc is not None
        and shuffled_test_safety_auc is not None
        and matched_validation_safety_auc >= 0.65
        and matched_test_safety_auc >= 0.65
        and matched_validation_safety_auc > shuffled_validation_safety_auc
        and matched_test_safety_auc > shuffled_test_safety_auc
    )
    validation_fidelity = recovery_geometry_fidelity["validation"]
    test_fidelity = recovery_geometry_fidelity["test"]
    matched_validation_fidelity = validation_fidelity[
        "matched_recovery_anchor"
    ]["within_state_ordering_concordance"]
    shuffled_validation_fidelity = validation_fidelity[
        "shuffled_recovery_anchor"
    ]["within_state_ordering_concordance"]
    matched_test_fidelity = test_fidelity[
        "matched_recovery_anchor"
    ]["within_state_ordering_concordance"]
    shuffled_test_fidelity = test_fidelity[
        "shuffled_recovery_anchor"
    ]["within_state_ordering_concordance"]
    recovery_geometry_scene_disjoint_support = bool(
        matched_validation_fidelity is not None
        and shuffled_validation_fidelity is not None
        and matched_test_fidelity is not None
        and shuffled_test_fidelity is not None
        and matched_validation_fidelity > shuffled_validation_fidelity
        and matched_test_fidelity > shuffled_test_fidelity
    )
    relaxed_support = recovery_candidate_support["all"]["relaxed"]
    conservative_support = recovery_candidate_support["all"]["conservative"]
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "inputs": {
            "source_root": str(args.source_root),
            "targets_root": str(args.targets_root) if args.targets_root else None,
            "continuation_root": (
                str(args.continuation_root) if args.continuation_root else None
            ),
            "expected_shards": args.expected_shards,
            "targeted_only": not args.all_source_states,
            "max_states": args.max_states,
            "shuffle_seed": args.shuffle_seed,
        },
        "policy_feature_contract": {
            "used": [
                "native finalized TreatmentSpec actions",
                "native 32-sample action-chunk mass and entropy",
                "executed action history",
                "visited/current body odometry",
                "fixed history relative poses",
                "fixed history age steps for deployable loop-entry anchor selection",
                "frozen heatmap coarse spatial probabilities/statistics",
                "frozen heatmap view/none probabilities",
            ],
            "explicitly_forbidden_from_policy": [
                "client_goal_position",
                "client_reference_path",
                "route progress/offpath",
                "Habitat success/SPL",
                "stored local pose traces",
            ],
            "label_only": [
                "stored local treatment outcomes",
                "stored local pose traces for candidate recovery-geometry fidelity",
                "authoritative one-deviation episode-end outcomes",
            ],
            "pixel_goal_not_reprojected": (
                "The native pixel_goal is a System1 look-down image coordinate, not a "
                "four-view panoramic bearing. Native candidate mass is the deployable "
                "goal-compatibility constraint in this first audit."
            ),
            "recovery_label_contract": (
                "Returning to the selected pre-loop anchor is not counted as a revisit "
                "failure. Recovery safety uses collision/stuck and enter-then-leave; "
                "route progress/goal distance are label-only navigation checks. Stored "
                "local traces are reported separately as geometry fidelity and never "
                "enter policy selection or threshold features."
            ),
        },
        "source": source_meta,
        "geometry": geometry_meta,
        "endpoint": endpoint_meta,
        "scene_splits": {
            split: sorted(scene for scene, value in scene_splits.items() if value == split)
            for split in ("train", "validation", "test")
        },
        "context_control": context_report,
        "candidate_risk_identifiability": candidate_identifiability,
        "recovery_geometry_fidelity": recovery_geometry_fidelity,
        "recovery_candidate_support": recovery_candidate_support,
        "action_entropy_overcommit": entropy_report,
        "threshold_tuning": {
            "matched": matched_tuning,
            "pose_only": pose_tuning,
            "matched_recovery_anchor": recovery_tuning,
            "pose_recovery_anchor": pose_recovery_tuning,
            "adaptive_prefix": prefix_tuning,
            "matched_thresholds_reused_for_shuffled": True,
            "matched_recovery_thresholds_reused_for_shuffled": True,
        },
        "policies": policies,
        "decision": {
            "matched_heatmap_incremental_local_support": incremental_supported,
            "normal_history_veto_support": veto_incremental_supported,
            "mode_aware_recovery_anchor_support": recovery_incremental_supported,
            "candidate_specific_safety_signal": candidate_specific_safety_signal,
            "recovery_geometry_scene_disjoint_support": (
                recovery_geometry_scene_disjoint_support
            ),
            "native_candidate_support": {
                "native_bad_loop_states": relaxed_support["native_bad_states"],
                "relaxed_oracle_safe_alternative_states": relaxed_support[
                    "oracle_safe_alternative_states"
                ],
                "conservative_oracle_safe_alternative_states": conservative_support[
                    "oracle_safe_alternative_states"
                ],
            },
            "criterion": (
                "On scene-disjoint validation: a matched normal-veto or recovery-anchor "
                "arm must have higher net conservative utility than its frozen-policy "
                "shuffled control and at least 2:1 safety rescue to destroy/navigation-"
                "regression. Endpoint evidence remains required."
            ),
            "closed_loop_full_eval_ready": False,
            "reason": (
                "Training-free matched geometry passed at least one local support gate; "
                "inspect exact endpoint coverage and collect missing selected branches "
                "before a closed-loop full evaluation."
                if incremental_supported
                else (
                    "Matched heatmaps contain scene-disjoint candidate-specific safety "
                    "information, but neither the normal veto nor the mode-aware recovery "
                    "policy generalized to validation. Recovery-anchor geometry ordering "
                    "also failed the matched-over-shuffled scene-disjoint gate, and the "
                    "native candidate set has limited oracle recovery support."
                    if candidate_specific_safety_signal
                    else "Neither normal history veto nor mode-aware recovery-anchor "
                    "geometry passed the local incremental-information gate."
                )
            ),
            "next_step": (
                "Do not launch full closed-loop HCTF yet. Collect targeted one-deviation "
                "endpoint branches at flip/recovery states and add a dedicated recovery "
                "proposal when native samples lack a safe alternative; then calibrate a "
                "small safety gate while keeping the generator frozen."
            ),
        },
    }
    report_path = args.output_dir / "hctf_training_free_report.json"
    decisions_path = args.output_dir / "hctf_policy_decisions.jsonl"
    _atomic_json(report_path, report)
    serialized = _serialize_decisions(states, policy_rows, endpoint)
    _atomic_jsonl(decisions_path, serialized)
    print(f"[hctf-audit] report={report_path}", flush=True)
    print(f"[hctf-audit] decisions={decisions_path}", flush=True)
    print(
        "[hctf-audit] decision matched_heatmap_incremental_local_support="
        f"{incremental_supported}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

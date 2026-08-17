#!/usr/bin/env python3
"""Scene-disjoint stage-0.5 probe for candidate identifiability.

This is deliberately not the production reranker.  It asks a narrower
question: can a small model rank locally better treatments using only values
that are available at deployment time?  Simulator goal/path fields are used
only to construct labels and never enter the model inputs.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from scripts.evaluation.summarize_candidate_support_audit import _read_records


SCHEMA = "candidate-identifiability-probe-v1"
VARIANTS = (
    "candidate_only",
    "candidate_system2",
    "candidate_system2_heatmap_metadata",
    "candidate_system2_heatmap_tokens",
)
SPLIT_NAMES = ("train", "validation", "test")
END_REASONS = (
    "early_replan",
    "queue_exhausted_replan",
    "local_stop_replan",
    "anti_deadlock_replan",
    "replan_now",
)
ACTION_NAMES = ("stop", "forward", "left", "right", "pad")
REQUIRED_ARRAYS = (
    "system2_latent_bf16_bits",
    "heatmap_tokens",
    "heatmap_token_mask",
    "heatmap_sample_valid",
)
OPTIONAL_HEATMAP_METADATA_ARRAYS = (
    "visibility_logits",
    "spatial_statistics",
    "view_probabilities",
    "none_probability",
    "normalized_age",
    "history_rank",
    "fixed_history_mask",
    "fixed_history_rel_poses",
    "fixed_history_age_steps",
)
HEATMAP_METADATA_BASE_WIDTH = 360


def _quantize(value: float, resolution_m: float) -> float:
    value = float(value)
    if resolution_m <= 0:
        return value
    return float(round(value / resolution_m))


def local_priority(
    outcome: dict[str, Any], *, resolution_m: float = 0.05
) -> tuple[float, ...]:
    """Robust lexicographic local label; no weighted scalar reward is invented."""

    entered = bool(outcome["entered_euclidean_success_radius"])
    left = bool(outcome["left_euclidean_success_radius"])
    return (
        float(entered and not left),
        float(entered),
        float(not left),
        _quantize(outcome["route_progress_delta_m"], resolution_m),
        -_quantize(outcome["endpoint_offpath_m"], resolution_m),
        -float(outcome["collision_or_stuck_count"]),
        float(not bool(outcome["revisit"])),
        -_quantize(outcome["min_euclidean_goal_distance_m"], resolution_m),
        -_quantize(outcome["endpoint_euclidean_goal_distance_m"], resolution_m),
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


def _parse_ratios(value: str) -> tuple[float, float, float]:
    pieces = [float(piece.strip()) for piece in value.split(",")]
    if len(pieces) != 3 or any(piece <= 0 for piece in pieces):
        raise argparse.ArgumentTypeError("split ratios must be three positive values")
    total = sum(pieces)
    return tuple(piece / total for piece in pieces)  # type: ignore[return-value]


def build_scene_split(
    records: Sequence[dict[str, Any]],
    *,
    seed: int,
    ratios: tuple[float, float, float],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Greedily balance state counts while keeping every scene in one split."""

    scene_counts: dict[str, int] = defaultdict(int)
    for record in records:
        scene = str(record.get("scene_id") or "")
        if not scene:
            raise RuntimeError("audit record has no scene_id")
        scene_counts[scene] += 1
    if len(scene_counts) < 3:
        raise RuntimeError(
            "scene-disjoint train/validation/test needs at least 3 scenes; "
            f"audit contains only {len(scene_counts)}: {sorted(scene_counts)}"
        )

    rng = random.Random(int(seed))
    ordered = list(scene_counts)
    rng.shuffle(ordered)
    ordered.sort(key=lambda scene: scene_counts[scene], reverse=True)
    targets = {
        name: max(1.0, len(records) * ratio)
        for name, ratio in zip(SPLIT_NAMES, ratios)
    }
    assigned_counts = {name: 0 for name in SPLIT_NAMES}
    assigned_scenes: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    mapping: dict[str, str] = {}
    for scene in ordered:
        split = min(
            SPLIT_NAMES,
            key=lambda name: (
                assigned_counts[name] / targets[name],
                len(assigned_scenes[name]),
                SPLIT_NAMES.index(name),
            ),
        )
        mapping[scene] = split
        assigned_scenes[split].append(scene)
        assigned_counts[split] += scene_counts[scene]
    if any(not assigned_scenes[name] for name in SPLIT_NAMES):
        raise RuntimeError(f"failed to create nonempty scene splits: {assigned_scenes}")
    scene_sets = [set(assigned_scenes[name]) for name in SPLIT_NAMES]
    if any(scene_sets[i] & scene_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("scene split leakage")

    summary = {
        "seed": int(seed),
        "ratios": dict(zip(SPLIT_NAMES, ratios)),
        "scene_disjoint": True,
        "splits": {
            name: {
                "states": assigned_counts[name],
                "scenes": sorted(assigned_scenes[name]),
                "scene_count": len(assigned_scenes[name]),
            }
            for name in SPLIT_NAMES
        },
    }
    return mapping, summary


def build_dev_episode_split(
    records: Sequence[dict[str, Any]],
    *,
    seed: int,
    ratios: tuple[float, float, float],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Episode-disjoint fallback used only to exercise the runtime in smoke tests."""

    episode_records: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records:
        key = (str(record.get("scene_id") or ""), str(record.get("episode_id") or ""))
        if not all(key):
            raise RuntimeError("audit record has no scene_id/episode_id")
        episode_records[key].append(str(record["state_key"]))
    if len(episode_records) < 3:
        raise RuntimeError(
            "development episode split needs at least 3 episodes; "
            f"audit contains {len(episode_records)}"
        )
    rng = random.Random(int(seed))
    ordered = list(episode_records)
    rng.shuffle(ordered)
    ordered.sort(key=lambda key: len(episode_records[key]), reverse=True)
    targets = {
        name: max(1.0, len(records) * ratio)
        for name, ratio in zip(SPLIT_NAMES, ratios)
    }
    counts = {name: 0 for name in SPLIT_NAMES}
    episodes: dict[str, list[tuple[str, str]]] = {name: [] for name in SPLIT_NAMES}
    state_mapping: dict[str, str] = {}
    for episode in ordered:
        split = min(
            SPLIT_NAMES,
            key=lambda name: (
                counts[name] / targets[name],
                len(episodes[name]),
                SPLIT_NAMES.index(name),
            ),
        )
        episodes[split].append(episode)
        state_keys = episode_records[episode]
        counts[split] += len(state_keys)
        state_mapping.update({state_key: split for state_key in state_keys})
    if any(not episodes[name] for name in SPLIT_NAMES):
        raise RuntimeError(f"failed to create nonempty development splits: {episodes}")
    return state_mapping, {
        "seed": int(seed),
        "ratios": dict(zip(SPLIT_NAMES, ratios)),
        "scene_disjoint": False,
        "episode_disjoint": True,
        "decision_valid": False,
        "development_override": "nondisjoint_scene_episode_split",
        "splits": {
            name: {
                "states": counts[name],
                "episode_count": len(episodes[name]),
                "episodes": [f"{scene}:{episode}" for scene, episode in episodes[name]],
                "scenes": sorted({scene for scene, _ in episodes[name]}),
                "scene_count": len({scene for scene, _ in episodes[name]}),
            }
            for name in SPLIT_NAMES
        },
    }


def candidate_feature_names() -> list[str]:
    names: list[str] = []
    for position in range(4):
        names.extend(f"action_{position}_{name}" for name in ACTION_NAMES)
    names.extend(f"action_fraction_{name}" for name in ACTION_NAMES[:4])
    names.extend(
        [
            "execute_fraction",
            "replan_after",
            "trigger_anti_deadlock",
            "update_local_stop_counter",
        ]
    )
    names.extend(f"end_reason_{name}" for name in END_REASONS)
    names.extend(
        [
            "native_sample_mass",
            "heatmap_sample_mass",
            "native_sample_fraction",
            "heatmap_sample_fraction",
            "has_native_provenance",
            "has_heatmap_provenance",
            "has_native_mean_provenance",
            "has_heatmap_mean_provenance",
            "is_native_mean_baseline",
            "baseline_common_prefix_fraction",
            "baseline_hamming_fraction",
            "baseline_length_delta_fraction",
        ]
    )
    return names


def candidate_features(
    treatment: dict[str, Any],
    *,
    baseline_treatment_id: str,
    baseline_actions: Sequence[int],
) -> np.ndarray:
    values: list[float] = []
    spec = treatment["spec"]
    actions = [int(action) for action in spec["actions"]]
    if len(actions) > 4 or any(action not in (0, 1, 2, 3) for action in actions):
        raise RuntimeError(f"invalid treatment actions: {actions}")
    padded = actions + [4] * (4 - len(actions))
    for token in padded:
        values.extend(float(token == category) for category in range(5))
    values.extend(float(actions.count(action)) / 4.0 for action in range(4))
    values.extend(
        [
            float(spec["execute_len"]) / 4.0,
            float(bool(spec["replan_after"])),
            float(bool(spec["trigger_anti_deadlock"])),
            float(bool(spec["update_local_stop_counter"])),
        ]
    )
    end_reason = str(spec["end_reason"])
    if end_reason not in END_REASONS:
        raise RuntimeError(f"unsupported treatment end_reason: {end_reason}")
    values.extend(float(end_reason == name) for name in END_REASONS)

    provenances = list(treatment["provenances"])
    native_total = max(1, int(treatment["native_sample_total"]))
    heatmap_total = max(1, int(treatment["heatmap_sample_total"]))
    common_prefix = 0
    for left, right in zip(actions, baseline_actions):
        if left != int(right):
            break
        common_prefix += 1
    baseline_padded = list(map(int, baseline_actions)) + [4] * (
        4 - len(baseline_actions)
    )
    hamming = sum(left != right for left, right in zip(padded, baseline_padded))
    values.extend(
        [
            float(treatment["native_sample_mass"]),
            float(treatment["heatmap_sample_mass"]),
            float(treatment["native_sample_count"]) / native_total,
            float(treatment["heatmap_sample_count"]) / heatmap_total,
            float(any(item["arm"] == "native" for item in provenances)),
            float(any(item["arm"] == "heatmap_control" for item in provenances)),
            float(
                any(
                    item["arm"] == "native"
                    and item["aggregation"] == "trajectory_mean"
                    for item in provenances
                )
            ),
            float(
                any(
                    item["arm"] == "heatmap_control"
                    and item["aggregation"] == "trajectory_mean"
                    for item in provenances
                )
            ),
            float(treatment["treatment_id"] == baseline_treatment_id),
            float(common_prefix) / 4.0,
            float(hamming) / 4.0,
            float(len(actions) - len(baseline_actions)) / 4.0,
        ]
    )
    result = np.asarray(values, dtype=np.float32)
    expected = len(candidate_feature_names())
    if result.shape != (expected,):
        raise AssertionError(f"candidate feature width {result.shape} != {expected}")
    return result


def decode_bfloat16_bits(bits: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(bits, dtype=np.uint16)
    expanded = np.left_shift(value.astype(np.uint32), np.uint32(16))
    return expanded.view(np.float32)


def heatmap_metadata(arrays: Any) -> np.ndarray:
    present = [name in arrays.files for name in OPTIONAL_HEATMAP_METADATA_ARRAYS]
    if any(present) and not all(present):
        missing = [
            name
            for name, available in zip(OPTIONAL_HEATMAP_METADATA_ARRAYS, present)
            if not available
        ]
        raise RuntimeError(f"partial heatmap metadata export; missing={missing}")
    metadata_available = bool(all(present))
    if metadata_available:
        pieces = [
            np.asarray(arrays["visibility_logits"], dtype=np.float32).reshape(-1),
            np.asarray(arrays["spatial_statistics"], dtype=np.float32).reshape(-1),
            np.asarray(arrays["view_probabilities"], dtype=np.float32).reshape(-1),
            np.asarray(arrays["none_probability"], dtype=np.float32).reshape(-1),
            np.asarray(arrays["normalized_age"], dtype=np.float32).reshape(-1),
            np.asarray(arrays["history_rank"], dtype=np.float32).reshape(-1),
            np.asarray(arrays["fixed_history_mask"], dtype=np.float32).reshape(-1),
            np.asarray(arrays["fixed_history_rel_poses"], dtype=np.float32).reshape(-1),
            np.asarray(
                arrays["fixed_history_age_steps"], dtype=np.float32
            ).reshape(-1)
            / 32.0,
        ]
        base = np.concatenate(pieces).astype(np.float32, copy=False)
        if base.shape != (HEATMAP_METADATA_BASE_WIDTH,):
            raise RuntimeError(
                f"unexpected heatmap metadata width {base.shape}; "
                f"expected {(HEATMAP_METADATA_BASE_WIDTH,)}"
            )
    else:
        base = np.zeros(HEATMAP_METADATA_BASE_WIDTH, dtype=np.float32)
    sample_valid_array = np.asarray(arrays["heatmap_sample_valid"], dtype=np.bool_)
    if sample_valid_array.size != 1:
        raise RuntimeError(
            f"heatmap_sample_valid must be scalar-like, got {sample_valid_array.shape}"
        )
    sample_valid = bool(sample_valid_array.reshape(-1)[0])
    if sample_valid and not metadata_available:
        raise RuntimeError("valid heatmap sample has no metadata export")
    result = np.concatenate(
        (
            base,
            np.asarray([metadata_available, sample_valid], dtype=np.float32),
        )
    )
    if not np.all(np.isfinite(result)):
        raise RuntimeError("heatmap metadata contains non-finite values")
    return result


@dataclass
class ProbeState:
    state_key: str
    scene_id: str
    episode_id: str
    candidate: np.ndarray
    priorities: tuple[tuple[float, ...], ...]
    exact_priorities: tuple[tuple[float, ...], ...]
    best_mask: np.ndarray
    baseline_preference: np.ndarray
    baseline_index: int
    system2_tokens: np.ndarray
    metadata: np.ndarray
    heatmap_tokens: np.ndarray
    heatmap_mask: np.ndarray


def state_from_record(
    record: dict[str, Any], *, resolution_m: float
) -> ProbeState:
    candidate_set = record["candidate_set"]
    treatments = list(candidate_set["treatments"])
    outcomes = {
        item["treatment_id"]: item for item in record["local_outcomes"]
    }
    treatment_ids = [item["treatment_id"] for item in treatments]
    if set(treatment_ids) != set(outcomes):
        raise RuntimeError(f"treatment/outcome mismatch: {record['state_key']}")
    baseline_id = str(candidate_set["baselines"]["native_trajectory_mean"])
    try:
        baseline_index = treatment_ids.index(baseline_id)
    except ValueError as exc:
        raise RuntimeError(f"native mean missing: {record['state_key']}") from exc
    baseline_actions = treatments[baseline_index]["spec"]["actions"]
    features = np.stack(
        [
            candidate_features(
                item,
                baseline_treatment_id=baseline_id,
                baseline_actions=baseline_actions,
            )
            for item in treatments
        ]
    )
    priorities = tuple(
        local_priority(outcomes[treatment_id], resolution_m=resolution_m)
        for treatment_id in treatment_ids
    )
    exact = tuple(
        local_priority(outcomes[treatment_id], resolution_m=0.0)
        for treatment_id in treatment_ids
    )
    best = max(priorities)
    best_mask = np.asarray([item == best for item in priorities], dtype=np.bool_)
    baseline = priorities[baseline_index]
    preference = np.asarray(
        [int(item > baseline) - int(item < baseline) for item in priorities],
        dtype=np.int8,
    )

    shard_dir = Path(str(record["__shard_dir"]))
    array_path = shard_dir / str(record["array_file"])
    with np.load(array_path, allow_pickle=False) as arrays:
        missing = [name for name in REQUIRED_ARRAYS if name not in arrays.files]
        if missing:
            raise RuntimeError(f"{array_path} lacks arrays: {missing}")
        system2 = decode_bfloat16_bits(arrays["system2_latent_bf16_bits"])
        system2 = np.asarray(system2[0], dtype=np.float16)
        tokens = np.asarray(arrays["heatmap_tokens"][0], dtype=np.float16)
        token_mask = np.asarray(arrays["heatmap_token_mask"][0], dtype=np.bool_)
        metadata = heatmap_metadata(arrays)
    if system2.ndim != 2 or tokens.ndim != 2 or token_mask.shape != (tokens.shape[0],):
        raise RuntimeError(f"unexpected compact feature shapes: {array_path}")
    if not np.all(np.isfinite(system2)) or not np.all(np.isfinite(tokens)):
        raise RuntimeError(f"non-finite compact feature: {array_path}")
    return ProbeState(
        state_key=str(record["state_key"]),
        scene_id=str(record["scene_id"]),
        episode_id=str(record["episode_id"]),
        candidate=features,
        priorities=priorities,
        exact_priorities=exact,
        best_mask=best_mask,
        baseline_preference=preference,
        baseline_index=baseline_index,
        system2_tokens=system2,
        metadata=metadata,
        heatmap_tokens=tokens,
        heatmap_mask=token_mask,
    )


def read_audit_records(
    root: Path, *, expected_shards: int, verify_integrity: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if verify_integrity:
        records, manifests = _read_records(root, expected_shards)
        by_key = {str(record["state_key"]): record for record in records}
    else:
        by_key = {}
        manifests = []
    for shard_id in range(expected_shards):
        shard = root / f"shard_{shard_id:02d}"
        index_path = shard / "records.jsonl"
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        rows = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not verify_integrity:
            for row in rows:
                key = str(row["state_key"])
                if key in by_key:
                    raise RuntimeError(f"duplicate state key: {key}")
                by_key[key] = row
        for row in rows:
            key = str(row["state_key"])
            if key not in by_key:
                raise RuntimeError(f"integrity/read closure mismatch: {key}")
            by_key[key]["__shard_dir"] = str(shard)
    records = list(by_key.values())
    records.sort(key=lambda row: str(row["state_key"]))
    if not records:
        raise RuntimeError("candidate audit is empty")
    return records, manifests


def _collate(states: Sequence[ProbeState]) -> dict[str, Any]:
    batch_size = len(states)
    max_candidates = max(state.candidate.shape[0] for state in states)
    candidate_width = states[0].candidate.shape[1]
    candidates = np.zeros(
        (batch_size, max_candidates, candidate_width), dtype=np.float32
    )
    candidate_mask = np.zeros((batch_size, max_candidates), dtype=np.bool_)
    best_mask = np.zeros((batch_size, max_candidates), dtype=np.bool_)
    preference = np.zeros((batch_size, max_candidates), dtype=np.int8)
    baseline = np.zeros(batch_size, dtype=np.int64)
    for index, state in enumerate(states):
        count = state.candidate.shape[0]
        candidates[index, :count] = state.candidate
        candidate_mask[index, :count] = True
        best_mask[index, :count] = state.best_mask
        preference[index, :count] = state.baseline_preference
        baseline[index] = state.baseline_index
    heatmap_width = states[0].heatmap_tokens.shape[1]
    max_heatmap_tokens = max(1, max(state.heatmap_tokens.shape[0] for state in states))
    heatmap_tokens = np.zeros(
        (batch_size, max_heatmap_tokens, heatmap_width), dtype=np.float32
    )
    heatmap_mask = np.zeros((batch_size, max_heatmap_tokens), dtype=np.bool_)
    for index, state in enumerate(states):
        token_count = state.heatmap_tokens.shape[0]
        if state.heatmap_tokens.shape[1] != heatmap_width:
            raise RuntimeError("heatmap token widths differ within a batch")
        if token_count:
            heatmap_tokens[index, :token_count] = state.heatmap_tokens
            heatmap_mask[index, :token_count] = state.heatmap_mask
    return {
        "states": list(states),
        "candidate": torch.from_numpy(candidates),
        "candidate_mask": torch.from_numpy(candidate_mask),
        "best_mask": torch.from_numpy(best_mask),
        "baseline_preference": torch.from_numpy(preference),
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


class CandidateRanker(nn.Module):
    def __init__(
        self,
        *,
        variant: str,
        candidate_width: int,
        system2_width: int,
        metadata_width: int,
        heatmap_width: int,
        hidden_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unsupported probe variant: {variant}")
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
        if variant in (
            "candidate_system2_heatmap_metadata",
            "candidate_system2_heatmap_tokens",
        ):
            self.metadata_encoder = nn.Sequential(
                nn.LayerNorm(metadata_width),
                nn.Linear(metadata_width, hidden_width),
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
        if mask is not None:
            valid_state = mask.any(dim=1)
            safe_mask = mask.clone()
            safe_mask[~valid_state, 0] = True
            logits = logits.masked_fill(~safe_mask[:, None, :], -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        result = torch.einsum("bct,bth->bch", weights, value(tokens))
        if mask is not None:
            result = result * valid_state[:, None, None].to(dtype=result.dtype)
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
        if self.variant in (
            "candidate_system2_heatmap_metadata",
            "candidate_system2_heatmap_tokens",
        ):
            context = self.metadata_encoder(batch["metadata"])[:, None, :]
            context = context.expand_as(candidate)
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


def ranking_loss(scores: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    valid = batch["candidate_mask"]
    best = batch["best_mask"] & valid
    masked = scores.masked_fill(~valid, -torch.inf)
    best_scores = scores.masked_fill(~best, -torch.inf)
    listwise = torch.logsumexp(masked, dim=1) - torch.logsumexp(best_scores, dim=1)

    baseline_score = scores.gather(1, batch["baseline_index"][:, None])
    difference = scores - baseline_score
    label = batch["baseline_preference"].to(dtype=scores.dtype)
    pair_mask = valid & label.ne(0)
    pair_loss = F.softplus(-label * difference)
    pair_loss = (pair_loss * pair_mask).sum(dim=1) / pair_mask.sum(dim=1).clamp_min(1)
    return listwise.mean() + pair_loss.mean()


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _loader(
    states: Sequence[ProbeState],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
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


def train_ranker(
    train_states: Sequence[ProbeState],
    validation_states: Sequence[ProbeState],
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
) -> tuple[CandidateRanker, list[dict[str, float]]]:
    _seed_everything(seed)
    first = train_states[0]
    model = CandidateRanker(
        variant=variant,
        candidate_width=first.candidate.shape[1],
        system2_width=first.system2_tokens.shape[1],
        metadata_width=first.metadata.shape[0],
        heatmap_width=first.heatmap_tokens.shape[1],
        hidden_width=hidden_width,
        dropout=dropout,
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
        train_total = 0.0
        train_count = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = ranking_loss(model(batch), batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss: {variant} seed={seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(batch["states"])
            train_total += float(loss.detach().cpu()) * count
            train_count += count

        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.no_grad():
            for batch in validation_loader:
                batch = _move_batch(batch, device)
                loss = ranking_loss(model(batch), batch)
                count = len(batch["states"])
                validation_total += float(loss.detach().cpu()) * count
                validation_count += count
        row = {
            "epoch": float(epoch),
            "train_loss": train_total / max(1, train_count),
            "validation_loss": validation_total / max(1, validation_count),
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
        raise RuntimeError("ranker training produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    return model, history


@torch.no_grad()
def score_states(
    model: CandidateRanker,
    states: Sequence[ProbeState],
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> list[np.ndarray]:
    model.eval()
    result: list[np.ndarray] = []
    for batch in _loader(states, batch_size=batch_size, shuffle=False, seed=seed):
        batch = _move_batch(batch, device)
        scores = model(batch).detach().float().cpu().numpy()
        for row, state in zip(scores, batch["states"]):
            result.append(np.asarray(row[: state.candidate.shape[0]], dtype=np.float64))
    if len(result) != len(states):
        raise AssertionError("score/state count mismatch")
    return result


def _selection_rows(
    states: Sequence[ProbeState], scores: Sequence[np.ndarray]
) -> list[tuple[float, int]]:
    rows: list[tuple[float, int]] = []
    for state, score in zip(states, scores):
        nonbaseline = [
            index for index in range(len(score)) if index != state.baseline_index
        ]
        if not nonbaseline:
            continue
        selected = max(nonbaseline, key=lambda index: (float(score[index]), -index))
        margin = float(score[selected] - score[state.baseline_index])
        baseline = state.priorities[state.baseline_index]
        outcome = state.priorities[selected]
        sign = int(outcome > baseline) - int(outcome < baseline)
        rows.append((margin, sign))
    return rows


def tune_abstention_threshold(
    states: Sequence[ProbeState],
    scores: Sequence[np.ndarray],
    *,
    max_destroy_state_rate: float,
) -> tuple[float, dict[str, Any]]:
    rows = sorted(_selection_rows(states, scores), key=lambda item: item[0], reverse=True)
    total_states = len(states)
    no_intervention_threshold = (
        float(np.nextafter(rows[0][0], math.inf)) if rows else 0.0
    )
    candidates: list[tuple[int, int, int, float]] = [
        (0, 0, 0, no_intervention_threshold)
    ]
    positive = negative = selected = 0
    index = 0
    while index < len(rows):
        margin = rows[index][0]
        while index < len(rows) and rows[index][0] == margin:
            selected += 1
            positive += int(rows[index][1] > 0)
            negative += int(rows[index][1] < 0)
            index += 1
        threshold = float(np.nextafter(margin, -math.inf))
        candidates.append((positive, negative, selected, threshold))
    feasible = [
        item
        for item in candidates
        if float(item[1]) / max(1, total_states) <= max_destroy_state_rate
    ]
    chosen = max(feasible, key=lambda item: (item[0], -item[1], -item[2], item[3]))
    positive, negative, selected, threshold = chosen
    return threshold, {
        "threshold": threshold,
        "constraint_max_destroy_state_rate": max_destroy_state_rate,
        "selected_states": selected,
        "positive_states": positive,
        "destroyed_states": negative,
        "destroy_state_rate": float(negative) / max(1, total_states),
    }


def evaluate_scores(
    states: Sequence[ProbeState],
    scores: Sequence[np.ndarray],
    *,
    threshold: float,
) -> dict[str, Any]:
    pair_correct = pair_total = 0
    baseline_pair_correct = baseline_pair_total = 0
    top_hit = 0
    oracle_positive = 0
    exact_oracle_positive = 0
    raw_counts = defaultdict(int)
    conservative_counts = defaultdict(int)

    def account(bucket: dict[str, int], sign: int, intervened: bool) -> None:
        bucket["states"] += 1
        bucket["interventions"] += int(intervened)
        bucket["positive"] += int(sign > 0)
        bucket["destroy"] += int(sign < 0)
        bucket["tie"] += int(sign == 0)

    for state, score in zip(states, scores):
        count = len(score)
        if count != len(state.priorities):
            raise RuntimeError(f"score width mismatch: {state.state_key}")
        predicted = int(np.argmax(score))
        top_hit += int(state.best_mask[predicted])
        baseline = state.priorities[state.baseline_index]
        exact_baseline = state.exact_priorities[state.baseline_index]
        oracle_positive += int(any(item > baseline for item in state.priorities))
        exact_oracle_positive += int(
            any(item > exact_baseline for item in state.exact_priorities)
        )

        for left in range(count):
            for right in range(left + 1, count):
                truth = int(state.priorities[left] > state.priorities[right]) - int(
                    state.priorities[left] < state.priorities[right]
                )
                if truth == 0:
                    continue
                prediction = int(score[left] > score[right]) - int(
                    score[left] < score[right]
                )
                pair_correct += int(prediction == truth)
                pair_total += 1
        for index, preference in enumerate(state.baseline_preference):
            if int(preference) == 0:
                continue
            prediction = int(score[index] > score[state.baseline_index]) - int(
                score[index] < score[state.baseline_index]
            )
            baseline_pair_correct += int(prediction == int(preference))
            baseline_pair_total += 1

        raw_sign = int(state.priorities[predicted] > baseline) - int(
            state.priorities[predicted] < baseline
        )
        account(raw_counts, raw_sign, predicted != state.baseline_index)

        nonbaseline = [index for index in range(count) if index != state.baseline_index]
        selected = max(nonbaseline, key=lambda index: (float(score[index]), -index))
        margin = float(score[selected] - score[state.baseline_index])
        intervened = margin > threshold
        if not intervened:
            selected = state.baseline_index
        sign = int(state.priorities[selected] > baseline) - int(
            state.priorities[selected] < baseline
        )
        account(conservative_counts, sign, intervened)

    def policy_metrics(bucket: dict[str, int]) -> dict[str, Any]:
        total = max(1, int(bucket["states"]))
        interventions = int(bucket["interventions"])
        positive = int(bucket["positive"])
        destroy = int(bucket["destroy"])
        return {
            "states": int(bucket["states"]),
            "interventions": interventions,
            "intervention_rate": interventions / total,
            "positive_states": positive,
            "positive_state_rate": positive / total,
            "destroyed_states": destroy,
            "destroy_state_rate": destroy / total,
            "tie_states": int(bucket["tie"]),
            "positive_precision_given_intervention": (
                positive / interventions if interventions else None
            ),
            "destroy_rate_given_intervention": (
                destroy / interventions if interventions else None
            ),
            "realizable_positive_recall": (
                positive / oracle_positive if oracle_positive else None
            ),
        }

    total = len(states)
    return {
        "states": total,
        "pairwise_ranking_accuracy": pair_correct / max(1, pair_total),
        "pairwise_comparisons": pair_total,
        "baseline_pairwise_accuracy": baseline_pair_correct
        / max(1, baseline_pair_total),
        "baseline_pairwise_comparisons": baseline_pair_total,
        "top1_local_best_hit_rate": top_hit / max(1, total),
        "oracle_positive_support_rate_robust": oracle_positive / max(1, total),
        "oracle_positive_support_rate_exact": exact_oracle_positive / max(1, total),
        "abstention_threshold": threshold,
        "raw_argmax": policy_metrics(raw_counts),
        "conservative": policy_metrics(conservative_counts),
    }


def shuffle_heatmap_context(
    states: Sequence[ProbeState], *, seed: int
) -> tuple[list[ProbeState], dict[str, Any]]:
    """Shuffle within scene so the control keeps scene-level nuisance cues."""

    rng = random.Random(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        groups[state.scene_id].append(index)
    source_for = list(range(len(states)))
    changed = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        order = list(indices)
        rng.shuffle(order)
        rotated = order[1:] + order[:1]
        for destination, source in zip(order, rotated):
            source_for[destination] = source
            changed += int(destination != source)
    result = [
        dataclasses.replace(
            state,
            metadata=states[source_for[index]].metadata,
            heatmap_tokens=states[source_for[index]].heatmap_tokens,
            heatmap_mask=states[source_for[index]].heatmap_mask,
        )
        for index, state in enumerate(states)
    ]
    return result, {
        "policy": "within_scene_cyclic_derangement",
        "states": len(states),
        "changed_states": changed,
        "changed_rate": changed / max(1, len(states)),
    }


def _aggregate_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "pairwise_ranking_accuracy": ("matched", "pairwise_ranking_accuracy"),
        "baseline_pairwise_accuracy": ("matched", "baseline_pairwise_accuracy"),
        "top1_local_best_hit_rate": ("matched", "top1_local_best_hit_rate"),
        "positive_state_rate": ("matched", "conservative", "positive_state_rate"),
        "destroy_state_rate": ("matched", "conservative", "destroy_state_rate"),
        "intervention_rate": ("matched", "conservative", "intervention_rate"),
        "realizable_positive_recall": (
            "matched",
            "conservative",
            "realizable_positive_recall",
        ),
    }
    result: dict[str, Any] = {}
    for name, path in paths.items():
        values: list[float] = []
        for run in runs:
            value: Any = run
            for key in path:
                value = value[key]
            if value is not None:
                values.append(float(value))
        result[name] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "values": values,
        }
    if runs and all("shuffled_heatmap" in run for run in runs):
        shuffled_paths = {
            "pairwise_ranking_accuracy": ("pairwise_ranking_accuracy",),
            "positive_state_rate": ("conservative", "positive_state_rate"),
            "destroy_state_rate": ("conservative", "destroy_state_rate"),
            "realizable_positive_recall": (
                "conservative",
                "realizable_positive_recall",
            ),
        }
        shuffled_summary: dict[str, Any] = {}
        for name, path in shuffled_paths.items():
            matched_values: list[float] = []
            shuffled_values: list[float] = []
            for run in runs:
                matched: Any = run["matched"]
                shuffled: Any = run["shuffled_heatmap"]
                for key in path:
                    matched = matched[key]
                    shuffled = shuffled[key]
                if matched is not None and shuffled is not None:
                    matched_values.append(float(matched))
                    shuffled_values.append(float(shuffled))
            deltas = [
                matched - shuffled
                for matched, shuffled in zip(matched_values, shuffled_values)
            ]
            shuffled_summary[name] = {
                "shuffled_mean": float(np.mean(shuffled_values))
                if shuffled_values
                else None,
                "matched_minus_shuffled_mean": float(np.mean(deltas))
                if deltas
                else None,
                "matched_minus_shuffled_values": deltas,
            }
        result["matched_vs_within_scene_shuffled_heatmap"] = shuffled_summary
    return result


def _parse_ints(value: str) -> list[int]:
    try:
        result = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("model seeds must be comma-separated ints") from exc
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("model seeds must be nonempty and unique")
    return result


def _parse_variants(value: str) -> list[str]:
    result = [piece.strip() for piece in value.split(",") if piece.strip()]
    invalid = [item for item in result if item not in VARIANTS]
    if not result or invalid:
        raise argparse.ArgumentTypeError(f"invalid variants {invalid}; valid={VARIANTS}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-shards", type=int, default=8)
    parser.add_argument("--scene-split-seed", type=int, default=20260810)
    parser.add_argument("--split-ratios", type=_parse_ratios, default=_parse_ratios("0.7,0.15,0.15"))
    parser.add_argument("--model-seeds", type=_parse_ints, default=_parse_ints("17,42,73"))
    parser.add_argument("--variants", type=_parse_variants, default=list(VARIANTS))
    parser.add_argument("--metric-resolution-m", type=float, default=0.05)
    parser.add_argument("--max-validation-destroy-state-rate", type=float, default=0.02)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-integrity-check", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dev-allow-nondisjoint-scene-split", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.expected_shards <= 64:
        raise ValueError("expected_shards must be in [1,64]")
    if args.metric_resolution_m < 0:
        raise ValueError("metric_resolution_m must be nonnegative")
    if not 0 <= args.max_validation_destroy_state_rate < 1:
        raise ValueError("max validation destroy rate must be in [0,1)")
    if args.hidden_width < 16 or args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        raise ValueError("invalid model/training dimensions")


def main() -> int:
    args = parse_args()
    _validate_args(args)
    audit_root = Path(args.audit_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    records, manifests = read_audit_records(
        audit_root,
        expected_shards=args.expected_shards,
        verify_integrity=not args.skip_integrity_check,
    )
    try:
        scene_mapping, split_summary = build_scene_split(
            records,
            seed=args.scene_split_seed,
            ratios=args.split_ratios,
        )
        state_split_mapping = {
            str(record["state_key"]): scene_mapping[str(record["scene_id"])]
            for record in records
        }
    except RuntimeError:
        if not args.dev_allow_nondisjoint_scene_split:
            raise
        state_split_mapping, split_summary = build_dev_episode_split(
            records,
            seed=args.scene_split_seed,
            ratios=args.split_ratios,
        )
    preflight = {
        "schema": SCHEMA,
        "status": "preflight_passed",
        "audit_root": str(audit_root),
        "records": len(records),
        "audit_manifests": len(manifests),
        "integrity_verified": not args.skip_integrity_check,
        "scene_split": split_summary,
        "decision_valid": bool(split_summary["scene_disjoint"]),
        "label_semantics": {
            "type": "lexicographic_local_diagnostic",
            "metric_resolution_m": args.metric_resolution_m,
            "authoritative_navigation_success": False,
        },
        "input_contract": {
            "deployment_only": True,
            "privileged_simulator_features": [],
            "candidate_features": candidate_feature_names(),
            "compact_arrays": list(REQUIRED_ARRAYS),
            "optional_heatmap_metadata_arrays": list(
                OPTIONAL_HEATMAP_METADATA_ARRAYS
            ),
            "invalid_heatmap_policy": (
                "zero metadata/tokens plus explicit availability and sample-valid masks"
            ),
        },
    }
    _atomic_json(output_dir / "preflight.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
    if args.preflight_only:
        return 0

    states: list[ProbeState] = []
    for index, record in enumerate(records, start=1):
        states.append(
            state_from_record(record, resolution_m=args.metric_resolution_m)
        )
        if index % 500 == 0 or index == len(records):
            print(f"[probe] loaded compact states {index}/{len(records)}", flush=True)
    split_states = {
        name: [
            state for state in states if state_split_mapping[state.state_key] == name
        ]
        for name in SPLIT_NAMES
    }
    if any(not split_states[name] for name in SPLIT_NAMES):
        raise RuntimeError("one or more state splits are empty")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {args.device}, but CUDA is unavailable")
    device = torch.device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        **preflight,
        "status": "completed_local_identifiability_probe",
        "decision_status": "local_labels_only_requires_continuation_validation",
        "training": {
            "variants": args.variants,
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
        },
        "variants": {},
        "required_next": [
            "targeted_one_deviation_continuations_h1_h3_h5_end",
            "short_horizon_sign_stability_against_episode_end",
            "conservative_selector_retest_with_closed_loop_labels",
        ],
    }

    for variant in args.variants:
        variant_runs: list[dict[str, Any]] = []
        for seed in args.model_seeds:
            print(f"[probe] training variant={variant} seed={seed}", flush=True)
            model, history = train_ranker(
                split_states["train"],
                split_states["validation"],
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
            checkpoint_path = output_dir / "checkpoints" / f"{variant}_seed{seed}.pt"
            torch.save(
                {
                    "schema": SCHEMA,
                    "variant": variant,
                    "seed": seed,
                    "model_state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "candidate_feature_names": candidate_feature_names(),
                    "history": history,
                },
                checkpoint_path,
            )
            os.chmod(checkpoint_path, 0o644)
            validation_scores = score_states(
                model,
                split_states["validation"],
                batch_size=args.batch_size,
                device=device,
                seed=seed,
            )
            threshold, threshold_fit = tune_abstention_threshold(
                split_states["validation"],
                validation_scores,
                max_destroy_state_rate=args.max_validation_destroy_state_rate,
            )
            test_scores = score_states(
                model,
                split_states["test"],
                batch_size=args.batch_size,
                device=device,
                seed=seed,
            )
            run: dict[str, Any] = {
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "epochs_ran": len(history),
                "best_validation_loss": min(row["validation_loss"] for row in history),
                "threshold_fit": threshold_fit,
                "matched": evaluate_scores(
                    split_states["test"], test_scores, threshold=threshold
                ),
            }
            if "heatmap" in variant:
                shuffled, shuffle_summary = shuffle_heatmap_context(
                    split_states["test"], seed=seed + 9187
                )
                shuffled_scores = score_states(
                    model,
                    shuffled,
                    batch_size=args.batch_size,
                    device=device,
                    seed=seed,
                )
                run["heatmap_shuffle"] = shuffle_summary
                run["shuffled_heatmap"] = evaluate_scores(
                    shuffled, shuffled_scores, threshold=threshold
                )
            variant_runs.append(run)
            print(
                "[probe] "
                f"variant={variant} seed={seed} "
                f"pair={run['matched']['pairwise_ranking_accuracy']:.4f} "
                f"positive={run['matched']['conservative']['positive_state_rate']:.4f} "
                f"destroy={run['matched']['conservative']['destroy_state_rate']:.4f}",
                flush=True,
            )
        report["variants"][variant] = {
            "runs": variant_runs,
            "aggregate": _aggregate_runs(variant_runs),
        }
        _atomic_json(output_dir / "candidate_identifiability_report.json", report)

    _atomic_json(output_dir / "candidate_identifiability_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

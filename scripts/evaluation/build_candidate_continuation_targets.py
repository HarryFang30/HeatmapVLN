#!/usr/bin/env python3
"""Build a deterministic, scene-disjoint one-deviation continuation plan.

The plan deliberately mixes deployment baselines, two learned local selectors,
and privileged *selection-only* local oracles.  Simulator fields are used to
choose diagnostic branches, never as inputs to the eventual deployed model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluation.probe_candidate_identifiability import (
    CandidateRanker,
    _atomic_json,
    build_scene_split,
    read_audit_records,
    score_states,
    state_from_record,
)


SCHEMA = "candidate-continuation-targets-v1"
SELECTOR_VARIANTS = (
    "candidate_system2",
    "candidate_system2_heatmap_tokens",
)
ROLE_ORDER = (
    "native_mean",
    "system2_selector",
    "heatmap_token_selector",
    "native_local_oracle",
    "union_local_oracle",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_ranker(
    checkpoint_path: Path,
    *,
    variant: str,
    example: Any,
    hidden_width: int,
    dropout: float,
    device: torch.device,
) -> CandidateRanker:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != variant:
        raise RuntimeError(
            f"checkpoint variant mismatch: {checkpoint_path}: "
            f"{checkpoint.get('variant')!r} != {variant!r}"
        )
    model = CandidateRanker(
        variant=variant,
        candidate_width=int(example.candidate.shape[1]),
        system2_width=int(example.system2_tokens.shape[1]),
        metadata_width=int(example.metadata.shape[0]),
        heatmap_width=int(example.heatmap_tokens.shape[1]),
        hidden_width=int(hidden_width),
        dropout=float(dropout),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _normalized_ranks(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("selector scores must be a finite vector")
    if len(scores) <= 1:
        return np.ones_like(scores)
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64) / float(len(scores) - 1)
    return ranks


def _ensemble_choice(
    score_runs: Sequence[np.ndarray], treatment_ids: Sequence[str]
) -> tuple[str, dict[str, Any]]:
    if not score_runs:
        raise ValueError("ensemble requires at least one score run")
    normalized = np.stack([_normalized_ranks(run) for run in score_runs])
    ensemble = normalized.mean(axis=0)
    best_value = float(ensemble.max())
    best_indices = np.flatnonzero(np.isclose(ensemble, best_value, atol=1e-12))
    best_index = min(best_indices.tolist(), key=lambda index: treatment_ids[index])
    per_seed_choices = [
        treatment_ids[
            min(
                np.flatnonzero(np.isclose(run, np.max(run), atol=1e-12)).tolist(),
                key=lambda index: treatment_ids[index],
            )
        ]
        for run in score_runs
    ]
    return treatment_ids[best_index], {
        "rank_ensemble_score": best_value,
        "per_seed_treatment_ids": per_seed_choices,
        "per_seed_agreement": len(set(per_seed_choices)) == 1,
    }


def _best_local_id(
    record: dict[str, Any],
    priorities: Sequence[tuple[float, ...]],
    *,
    arm: str | None,
) -> str:
    treatments = list(record["candidate_set"]["treatments"])
    eligible: list[int] = []
    for index, treatment in enumerate(treatments):
        if arm is None or int(treatment.get(f"{arm}_sample_count", 0)) > 0:
            eligible.append(index)
    if not eligible:
        raise RuntimeError(f"no {arm or 'union'} candidates at {record['state_key']}")
    best_priority = max(priorities[index] for index in eligible)
    tied = [index for index in eligible if priorities[index] == best_priority]
    # Prefer greater arm mass, then fewer actions, then a stable treatment id.
    def tie_key(index: int) -> tuple[float, int, str]:
        treatment = treatments[index]
        mass = (
            float(treatment.get(f"{arm}_sample_mass", 0.0))
            if arm is not None
            else float(treatment.get("native_sample_mass", 0.0))
            + float(treatment.get("heatmap_sample_mass", 0.0))
        )
        action_count = len(treatment["spec"]["actions"])
        return (-mass, action_count, str(treatment["treatment_id"]))

    return str(treatments[min(tied, key=tie_key)]["treatment_id"])


def _candidate_roles(
    record: dict[str, Any],
    state: Any,
    score_runs: dict[str, list[np.ndarray]],
) -> tuple[dict[str, str], dict[str, Any]]:
    treatment_ids = [
        str(treatment["treatment_id"])
        for treatment in record["candidate_set"]["treatments"]
    ]
    system2_id, system2_meta = _ensemble_choice(
        score_runs["candidate_system2"], treatment_ids
    )
    heatmap_id, heatmap_meta = _ensemble_choice(
        score_runs["candidate_system2_heatmap_tokens"], treatment_ids
    )
    roles = {
        "native_mean": str(
            record["candidate_set"]["baselines"]["native_trajectory_mean"]
        ),
        "system2_selector": system2_id,
        "heatmap_token_selector": heatmap_id,
        "native_local_oracle": _best_local_id(
            record, state.priorities, arm="native"
        ),
        "union_local_oracle": _best_local_id(
            record, state.priorities, arm=None
        ),
    }
    heatmap_local = _best_local_id(record, state.priorities, arm="heatmap")
    id_to_index = {value: index for index, value in enumerate(treatment_ids)}
    native_best = state.priorities[id_to_index[roles["native_local_oracle"]]]
    heatmap_best = state.priorities[id_to_index[heatmap_local]]
    baseline = state.priorities[id_to_index[roles["native_mean"]]]
    metadata = {
        "system2_selector": system2_meta,
        "heatmap_token_selector": heatmap_meta,
        "selector_disagreement": system2_id != heatmap_id,
        "heatmap_adds_local_support": heatmap_best > native_best,
        "system2_locally_better_than_native_mean": (
            state.priorities[id_to_index[system2_id]] > baseline
        ),
        "heatmap_selector_locally_better_than_native_mean": (
            state.priorities[id_to_index[heatmap_id]] > baseline
        ),
        "heatmap_local_oracle_treatment_id": heatmap_local,
        "unique_treatment_count": len(set(roles.values())),
    }
    return roles, metadata


def _state_priority(target: dict[str, Any]) -> tuple[int, ...]:
    diagnostic = target["diagnostic_selection"]
    strata = target["state_strata"]
    return (
        int(diagnostic["selector_disagreement"]),
        int(diagnostic["heatmap_adds_local_support"]),
        int(diagnostic["heatmap_selector_locally_better_than_native_mean"]),
        int(diagnostic["system2_locally_better_than_native_mean"]),
        int(strata.get("recovery", False)),
        int(strata.get("near_goal", False)),
        -int(target["system2_call_index"]),
    )


def _select_states(
    candidates: list[dict[str, Any]], *, target_count: int, max_per_episode: int
) -> list[dict[str, Any]]:
    by_episode: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for target in candidates:
        by_episode[(target["scene_id"], int(target["episode_id"]))].append(target)
    chosen: list[dict[str, Any]] = []
    for episode_key in sorted(by_episode):
        values = sorted(
            by_episode[episode_key],
            key=lambda value: (_state_priority(value), value["state_key"]),
            reverse=True,
        )
        if not values:
            continue
        episode_chosen = [values[0]]
        remaining = values[1:]
        while remaining and len(episode_chosen) < max_per_episode:
            existing_calls = [int(value["system2_call_index"]) for value in episode_chosen]
            selected = max(
                remaining,
                key=lambda value: (
                    min(
                        abs(int(value["system2_call_index"]) - call)
                        for call in existing_calls
                    ),
                    _state_priority(value),
                    value["state_key"],
                ),
            )
            episode_chosen.append(selected)
            remaining.remove(selected)
        chosen.extend(episode_chosen)

    chosen.sort(key=lambda value: (_state_priority(value), value["state_key"]), reverse=True)
    if len(chosen) > target_count:
        # Round-robin scenes avoids a few long scenes consuming the truncation.
        scene_queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for value in chosen:
            scene_queues[value["scene_id"]].append(value)
        balanced: list[dict[str, Any]] = []
        scenes = sorted(scene_queues)
        while len(balanced) < target_count:
            progressed = False
            for scene in scenes:
                if scene_queues[scene] and len(balanced) < target_count:
                    balanced.append(scene_queues[scene].pop(0))
                    progressed = True
            if not progressed:
                break
        chosen = balanced
    chosen.sort(key=lambda value: (value["source_shard_id"], value["state_key"]))
    return chosen


def _mark_episode_end_subset(selected: list[dict[str, Any]], count: int) -> None:
    queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_episodes: set[tuple[str, int]] = set()
    for target in sorted(
        selected,
        key=lambda value: (_state_priority(value), value["state_key"]),
        reverse=True,
    ):
        episode = (target["scene_id"], int(target["episode_id"]))
        if episode in seen_episodes:
            continue
        seen_episodes.add(episode)
        queues[target["scene_id"]].append(target)
    picked: list[dict[str, Any]] = []
    scenes = sorted(queues)
    while len(picked) < min(count, sum(map(len, queues.values()))):
        progressed = False
        for scene in scenes:
            if queues[scene] and len(picked) < count:
                picked.append(queues[scene].pop(0))
                progressed = True
        if not progressed:
            break
    picked_keys = {target["state_key"] for target in picked}
    for target in selected:
        target["run_to_episode_end"] = target["state_key"] in picked_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=8)
    parser.add_argument("--target-states", type=int, default=1024)
    parser.add_argument("--episode-end-states", type=int, default=256)
    parser.add_argument("--max-states-per-episode", type=int, default=2)
    parser.add_argument("--scene-split-seed", type=int, default=20260810)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_shards < 1 or args.target_states < 1:
        raise ValueError("invalid shard/target counts")
    if args.episode_end_states < 0 or args.episode_end_states > args.target_states:
        raise ValueError("episode-end state count must be within target count")
    if args.max_states_per_episode < 1:
        raise ValueError("max states per episode must be positive")
    audit_root = args.audit_root.expanduser().resolve()
    report_path = args.probe_report.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not bool(report.get("decision_valid")):
        raise RuntimeError("probe report is not scene-disjoint/decision-valid")
    records, manifests = read_audit_records(
        audit_root, expected_shards=args.expected_shards, verify_integrity=True
    )
    scene_mapping, scene_summary = build_scene_split(
        records,
        seed=args.scene_split_seed,
        ratios=(0.7, 0.15, 0.15),
    )
    expected_split = report.get("scene_split")
    if expected_split and expected_split != scene_summary:
        raise RuntimeError("reconstructed scene split differs from probe report")

    states = []
    for index, record in enumerate(records, start=1):
        states.append(state_from_record(record, resolution_m=0.05))
        if index % 500 == 0 or index == len(records):
            print(f"[targets] loaded {index}/{len(records)} states", flush=True)
    state_index = {state.state_key: index for index, state in enumerate(states)}
    if len(state_index) != len(states):
        raise RuntimeError("duplicate probe states")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    training = report["training"]
    variant_scores: dict[str, list[list[np.ndarray]]] = {}
    variant_seeds: dict[str, list[int]] = {}
    for variant in SELECTOR_VARIANTS:
        runs = report["variants"][variant]["runs"]
        variant_scores[variant] = []
        variant_seeds[variant] = []
        for run in runs:
            checkpoint_path = Path(run["checkpoint"]).expanduser().resolve()
            model = _load_ranker(
                checkpoint_path,
                variant=variant,
                example=states[0],
                hidden_width=int(training["hidden_width"]),
                dropout=float(training["dropout"]),
                device=device,
            )
            scores = score_states(
                model,
                states,
                batch_size=args.batch_size,
                device=device,
                seed=int(run["seed"]),
            )
            variant_scores[variant].append(scores)
            variant_seeds[variant].append(int(run["seed"]))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"[targets] scored variant={variant} seed={run['seed']}", flush=True
            )

    candidates: list[dict[str, Any]] = []
    for record in records:
        index = state_index[str(record["state_key"])]
        state = states[index]
        score_runs = {
            variant: [run[index] for run in variant_scores[variant]]
            for variant in SELECTOR_VARIANTS
        }
        roles, diagnostic = _candidate_roles(record, state, score_runs)
        source_shard_id = int(Path(record["__shard_dir"]).name.split("_")[-1])
        candidates.append(
            {
                "state_key": str(record["state_key"]),
                "source_shard_id": source_shard_id,
                "scene_id": str(record["scene_id"]),
                "episode_id": int(record["episode_id"]),
                "system2_call_index": int(record["system2_call_index"]),
                "step_id": int(record["step_id"]),
                "scene_split": scene_mapping[str(record["scene_id"])],
                "state_strata": dict(record.get("state_strata") or {}),
                "treatment_roles": roles,
                "diagnostic_selection": diagnostic,
            }
        )

    selected = _select_states(
        candidates,
        target_count=args.target_states,
        max_per_episode=args.max_states_per_episode,
    )
    if len(selected) < min(args.target_states, len(candidates)):
        print(
            f"[targets] warning: per-episode cap yielded {len(selected)} "
            f"instead of requested {args.target_states}",
            flush=True,
        )
    _mark_episode_end_subset(selected, args.episode_end_states)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_summaries: list[dict[str, Any]] = []
    for shard_id in range(args.expected_shards):
        shard_targets = [
            target for target in selected if target["source_shard_id"] == shard_id
        ]
        episodes = sorted(
            {
                (target["scene_id"], int(target["episode_id"]))
                for target in shard_targets
            }
        )
        payload = {
            "schema": SCHEMA,
            "source_audit_root": str(audit_root),
            "probe_report": str(report_path),
            "shard_id": shard_id,
            "episodes": [
                {"scene_id": scene, "episode_id": episode}
                for scene, episode in episodes
            ],
            "targets": shard_targets,
        }
        payload["payload_sha256"] = _canonical_sha256(payload)
        path = output_dir / f"targets_shard_{shard_id:02d}.json"
        _atomic_json(path, payload)
        shard_summaries.append(
            {
                "shard_id": shard_id,
                "path": str(path),
                "sha256": _sha256(path),
                "states": len(shard_targets),
                "episodes": len(episodes),
                "unique_branches": sum(
                    len(set(target["treatment_roles"].values()))
                    for target in shard_targets
                ),
                "episode_end_states": sum(
                    bool(target["run_to_episode_end"]) for target in shard_targets
                ),
            }
        )

    split_counts: dict[str, int] = defaultdict(int)
    stratum_counts: dict[str, int] = defaultdict(int)
    for target in selected:
        split_counts[target["scene_split"]] += 1
        for name, value in target["state_strata"].items():
            if bool(value):
                stratum_counts[name] += 1
        for name in (
            "selector_disagreement",
            "heatmap_adds_local_support",
            "system2_locally_better_than_native_mean",
            "heatmap_selector_locally_better_than_native_mean",
        ):
            if bool(target["diagnostic_selection"][name]):
                stratum_counts[name] += 1
    plan = {
        "schema": SCHEMA,
        "status": "ready",
        "source_audit_root": str(audit_root),
        "source_audit_manifests": len(manifests),
        "probe_report": str(report_path),
        "probe_report_sha256": _sha256(report_path),
        "selector_variants": {
            variant: {"seeds": variant_seeds[variant]}
            for variant in SELECTOR_VARIANTS
        },
        "selection": {
            "requested_states": int(args.target_states),
            "selected_states": len(selected),
            "requested_episode_end_states": int(args.episode_end_states),
            "selected_episode_end_states": sum(
                bool(target["run_to_episode_end"]) for target in selected
            ),
            "max_states_per_episode": int(args.max_states_per_episode),
            "roles": list(ROLE_ORDER),
            "unique_episodes": len(
                {
                    (target["scene_id"], int(target["episode_id"]))
                    for target in selected
                }
            ),
            "unique_branches": sum(
                len(set(target["treatment_roles"].values())) for target in selected
            ),
            "split_state_counts": dict(sorted(split_counts.items())),
            "diagnostic_counts": dict(sorted(stratum_counts.items())),
        },
        "scene_split": scene_summary,
        "shards": shard_summaries,
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    _atomic_json(output_dir / "plan.json", plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

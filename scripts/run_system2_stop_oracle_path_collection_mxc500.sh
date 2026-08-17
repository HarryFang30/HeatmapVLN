#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

FJL_ROOT="${FJL_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
COHORT="${SYSTEM2_STOP_ORACLE_PATH_COHORT:?Set SYSTEM2_STOP_ORACLE_PATH_COHORT}"
RUN_STAMP="${SYSTEM2_STOP_ORACLE_PATH_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${SYSTEM2_STOP_ORACLE_PATH_OUT_DIR:-${FJL_ROOT}/model/system2_stop_oracle_path_collection_${RUN_STAMP}}"

test -s "$COHORT" || {
  echo "Missing oracle-path collection cohort: $COHORT" >&2
  exit 1
}
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite oracle-path collection output: $OUTPUT_DIR" >&2
  exit 1
fi

export STAGE3_EVAL_EPISODE_LIST="$COHORT"
export STAGE3_EVAL_OUTPUT_PATH="$OUTPUT_DIR"
export STAGE3_FORCE_CONTINUE_GPU="${SYSTEM2_STOP_ORACLE_PATH_GPU:-7}"
export STAGE3_FORCE_CONTINUE_RPC_PORT="${SYSTEM2_STOP_ORACLE_PATH_RPC_PORT:-50093}"
export STAGE3_EVAL_RPC_PROTOCOL_SEED="${SYSTEM2_STOP_ORACLE_PATH_SEED:-124}"
export STAGE3_EVAL_STOP_COLLECT_ORACLE_RECOVERY_AFTER_NEGATIVE=0
export STAGE3_EVAL_STOP_COLLECT_ORACLE_PATH_FROM_START=1
export STAGE3_EVAL_STOP_ORACLE_RECOVERY_FROM_COHORT_TRIGGERS=0
export STAGE3_EVAL_STOP_ORACLE_RECOVERY_GOAL_PROBES="${SYSTEM2_STOP_ORACLE_PATH_GOAL_PROBES:-8}"
export STAGE3_EVAL_STOP_ORACLE_RECOVERY_ACTIONS_PER_CALL="${SYSTEM2_STOP_ORACLE_PATH_ACTIONS_PER_CALL:-4}"

echo "[$(date '+%F %T')] launching scene-balanced oracle-path STOP collection"
echo "[oracle-path-collection] cohort=$COHORT"
echo "[oracle-path-collection] gpu=$STAGE3_FORCE_CONTINUE_GPU seed=$STAGE3_EVAL_RPC_PROTOCOL_SEED actions_per_call=$STAGE3_EVAL_STOP_ORACLE_RECOVERY_ACTIONS_PER_CALL"
echo "[oracle-path-collection] output=$OUTPUT_DIR"

bash scripts/run_system2_stop_force_continue_train3_smoke_mxc500.sh

PYTHONPATH="$REPO_ROOT" /mnt/afs/lixiaoou/intern/fjl/envs/qwen25/bin/python - \
  "$OUTPUT_DIR" \
  "$COHORT" \
  "$STAGE3_EVAL_STOP_ORACLE_RECOVERY_ACTIONS_PER_CALL" \
  "$STAGE3_EVAL_STOP_ORACLE_RECOVERY_GOAL_PROBES" <<'PY'
import json
import hashlib
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

root = Path(sys.argv[1])
cohort_path = Path(sys.argv[2])
expected_actions_per_call = int(sys.argv[3])
expected_goal_probes = int(sys.argv[4])
expected_protocol_seed = int(os.environ["STAGE3_EVAL_RPC_PROTOCOL_SEED"])
boundary_probe_enabled = os.environ.get(
    "STAGE3_EVAL_STOP_COLLECT_BOUNDARY_PROBE_SWEEP", "0"
) == "1"
expected_boundary_probes = int(
    os.environ.get("STAGE3_EVAL_STOP_BOUNDARY_PROBE_VIEWS", "0")
)

cohort_payload = json.loads(cohort_path.read_text(encoding="utf-8"))
expected = {
    (str(row["scene_id"]), int(row["episode_id"]))
    for row in cohort_payload["episodes"]
}
progress = [
    json.loads(line)
    for line in (root / "progress.json").open(encoding="utf-8")
    if line.strip()
]
observed = {(str(row["scene_id"]), int(row["episode_id"])) for row in progress}
if observed != expected or len(progress) != len(expected):
    raise SystemExit(
        "Oracle-path progress/cohort mismatch: "
        f"expected={len(expected)} observed={len(observed)} "
        f"missing={sorted(expected - observed)[:5]} extra={sorted(observed - expected)[:5]}"
    )

total_calls = 0
total_primitives = 0
for row in progress:
    if row.get("system2_stop_collect_oracle_path_from_start") is not True:
        raise SystemExit(f"Missing oracle path provenance: {row}")
    if row.get("system2_stop_collect_oracle_recovery_after_negative") is not False:
        raise SystemExit(f"Recovery-after-negative was unexpectedly enabled: {row}")
    if int(row.get("system2_stop_oracle_recovery_activations", 0)) != 1:
        raise SystemExit(f"Oracle path did not activate exactly once: {row}")
    if row.get("system2_stop_oracle_recovery_activation_reason") != "oracle_path_from_start":
        raise SystemExit(f"Wrong oracle path activation reason: {row}")
    if int(row.get("system2_stop_oracle_recovery_goal_probes", 0)) != expected_goal_probes:
        raise SystemExit(f"Incomplete goal probes: {row}")
    if int(row.get("system2_stop_oracle_recovery_actions_per_call", 0)) != expected_actions_per_call:
        raise SystemExit(f"Wrong oracle action chunk: {row}")
    total_calls += int(row.get("system2_stop_oracle_recovery_calls", 0))
    total_primitives += int(row.get("system2_stop_oracle_recovery_primitive_actions", 0))
if total_primitives <= total_calls:
    raise SystemExit(
        f"Oracle action chunk was not exercised: calls={total_calls} primitives={total_primitives}"
    )

labels = [
    json.loads(line)
    for line in (root / "system2_stop_rollout_labels.jsonl").open(encoding="utf-8")
    if line.strip()
]
by_episode = defaultdict(Counter)
boundary_probe_indices = defaultdict(set)
goal_probe_indices = defaultdict(set)
expected_namespace = hashlib.sha256(
    str(root.expanduser().resolve()).encode("utf-8")
).hexdigest()[:12]
expected_feature_dir = (root / "system2_stop_features").resolve()
feature_keys = set()
trajectory_feature_rows = 0
fixed_probe_without_trajectory_rows = 0
for row in labels:
    key = (str(row["scene_id"]), int(row["episode_id"]))
    if key not in expected:
        raise SystemExit(f"Unexpected episode in rollout labels: {key}")
    if row.get("dataset_split") != "train":
        raise SystemExit(f"Non-train rollout row: {row}")
    if row.get("oracle_recovery_active") is not True:
        raise SystemExit(f"Missing privileged-recovery provenance: {row}")
    feature_path = Path(str(row.get("path", "")))
    if not feature_path.is_file() or feature_path.stat().st_size == 0:
        raise SystemExit(f"Missing STOP feature tensor: {feature_path}")
    feature_key = str(row.get("key", ""))
    if feature_key in feature_keys:
        raise SystemExit(f"Duplicate STOP feature key in collection: {feature_key}")
    feature_keys.add(feature_key)
    if row.get("collection_namespace") != expected_namespace:
        raise SystemExit(
            f"Wrong STOP feature namespace for {feature_key}: "
            f"{row.get('collection_namespace')} != {expected_namespace}"
        )
    if not feature_key.startswith(f"src{expected_namespace}_"):
        raise SystemExit(f"STOP feature key is not namespaced: {feature_key}")
    if feature_path.resolve().parent != expected_feature_dir:
        raise SystemExit(f"STOP feature escaped collection directory: {feature_path}")
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    if payload.get("key") != feature_key:
        raise SystemExit(f"STOP feature key/payload mismatch: {feature_path}")
    if row.get("trajectory_metrics") is not None:
        if (
            row.get("trajectory_feature_schema")
            != "heatmapvln-system2-stop-trajectory-feature-v1"
            or payload.get("trajectory_feature_schema")
            != "heatmapvln-system2-stop-trajectory-feature-v1"
        ):
            raise SystemExit(f"Missing trajectory feature schema: {feature_path}")
        expected_ranks = {
            "raw_traj_latent": 2,
            "adapted_traj_latent": 2,
            "projected_traj_condition": 2,
            "trajectory": 3,
        }
        for name, expected_rank in expected_ranks.items():
            tensor = payload.get(name)
            if (
                not torch.is_tensor(tensor)
                or tensor.ndim != expected_rank
                or tensor.numel() == 0
                or not bool(torch.isfinite(tensor.float()).all())
            ):
                raise SystemExit(
                    f"Invalid {name} in trajectory feature payload: {feature_path}"
                )
        if payload["raw_traj_latent"].shape != payload["adapted_traj_latent"].shape:
            raise SystemExit(f"Raw/adapted trajectory latent shape mismatch: {feature_path}")
        if payload["projected_traj_condition"].shape[0] != payload["raw_traj_latent"].shape[0]:
            raise SystemExit(f"Projected trajectory query count mismatch: {feature_path}")
        trajectory_feature_rows += 1
    is_boundary_probe = bool(row.get("boundary_probe_sweep"))
    is_goal_probe = bool(row.get("goal_probe_sweep"))
    if is_boundary_probe and is_goal_probe:
        raise SystemExit(f"STOP feature row has two probe kinds: {row}")
    if (is_boundary_probe or is_goal_probe) and row.get("trajectory_metrics") is None:
        # A fixed-view STOP probe may legitimately generate view: stop/turn, in
        # which case System1 is not invoked and no trajectory tensors exist.
        # The exact multimodal prompt and STOP label remain valid training data.
        fixed_probe_without_trajectory_rows += 1
    if is_boundary_probe:
        boundary_probe_indices[key].add(int(row["boundary_probe_index"]))
        expected_sweep_id = f"{key[0]}:{key[1]}:{expected_protocol_seed}:boundary"
        if row.get("boundary_probe_sweep_id") != expected_sweep_id:
            raise SystemExit(f"Boundary probe sweep id mismatch: {row}")
    if is_goal_probe:
        goal_probe_indices[key].add(int(row["goal_probe_index"]))
        expected_sweep_id = f"{key[0]}:{key[1]}:{expected_protocol_seed}:goal"
        if row.get("goal_probe_sweep_id") != expected_sweep_id:
            raise SystemExit(f"Goal probe sweep id mismatch: {row}")
    target = row.get("stop_target")
    if target in (0, 1):
        by_episode[key][int(target)] += 1
for key in expected:
    counts = by_episode[key]
    if counts[1] < expected_goal_probes:
        raise SystemExit(f"Episode {key} has too few positive goal views: {counts}")
    if counts[0] < 1:
        raise SystemExit(f"Episode {key} has no far negative view: {counts}")
    if goal_probe_indices[key] != set(range(expected_goal_probes)):
        raise SystemExit(
            f"Episode {key} has incomplete goal probe indices: "
            f"{sorted(goal_probe_indices[key])}"
        )
    if boundary_probe_enabled:
        if expected_boundary_probes < 1:
            raise SystemExit("Boundary probe collection enabled without a positive view count")
        if boundary_probe_indices[key] != set(range(expected_boundary_probes)):
            raise SystemExit(
                f"Episode {key} has incomplete boundary probe indices: "
                f"{sorted(boundary_probe_indices[key])}"
            )
    elif boundary_probe_indices[key]:
        raise SystemExit(f"Episode {key} unexpectedly contains boundary probes")
if trajectory_feature_rows == 0:
    raise SystemExit("Oracle-path collection produced no augmented trajectory features")

result = json.loads((root / "result.json").read_text(encoding="utf-8"))
if int(result.get("total_episodes", 0)) != len(expected):
    raise SystemExit(f"Result episode count mismatch: {result}")
print(
    "Oracle-path collection postflight passed: "
    f"episodes={len(expected)} rows={len(labels)} "
    f"positives={sum(counts[1] for counts in by_episode.values())} "
    f"negatives={sum(counts[0] for counts in by_episode.values())} "
    f"trajectory_features={trajectory_feature_rows} "
    f"fixed_probes_without_trajectory={fixed_probe_without_trajectory_rows} "
    f"boundary_probes={sum(len(values) for values in boundary_probe_indices.values())} "
    f"goal_probes={sum(len(values) for values in goal_probe_indices.values())} "
    f"calls={total_calls} primitives={total_primitives}"
)
PY

echo "[oracle-path-collection] complete: $OUTPUT_DIR"

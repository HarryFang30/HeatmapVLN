#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

FJL_ROOT="${FJL_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
RUN_STAMP="${SYSTEM2_STOP_ORACLE_PATH_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${SYSTEM2_STOP_ORACLE_PATH_OUT_DIR:-${FJL_ROOT}/model/smoke_system2_stop_oracle_path_train3_${RUN_STAMP}}"

export STAGE3_EVAL_EPISODE_LIST="${STAGE3_EVAL_EPISODE_LIST:-configs/eval_cohorts/system2_stop_force_continue_train3.json}"
export STAGE3_EVAL_OUTPUT_PATH="$OUTPUT_DIR"
export STAGE3_FORCE_CONTINUE_GPU="${SYSTEM2_STOP_ORACLE_PATH_GPU:-7}"
export STAGE3_FORCE_CONTINUE_RPC_PORT="${SYSTEM2_STOP_ORACLE_PATH_RPC_PORT:-50091}"
export STAGE3_EVAL_RPC_PROTOCOL_SEED="${SYSTEM2_STOP_ORACLE_PATH_SEED:-123}"
export STAGE3_EVAL_STOP_COLLECT_ORACLE_RECOVERY_AFTER_NEGATIVE=0
export STAGE3_EVAL_STOP_COLLECT_ORACLE_PATH_FROM_START=1
export STAGE3_EVAL_STOP_ORACLE_RECOVERY_FROM_COHORT_TRIGGERS=0
export STAGE3_EVAL_STOP_ORACLE_RECOVERY_GOAL_PROBES="${SYSTEM2_STOP_ORACLE_PATH_GOAL_PROBES:-8}"
export STAGE3_EVAL_STOP_ORACLE_RECOVERY_ACTIONS_PER_CALL="${SYSTEM2_STOP_ORACLE_PATH_ACTIONS_PER_CALL:-4}"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite oracle-path smoke output: $OUTPUT_DIR" >&2
  exit 1
fi

echo "[$(date '+%F %T')] launching start-to-goal oracle-path STOP collection smoke"
echo "[oracle-path-smoke] gpu=$STAGE3_FORCE_CONTINUE_GPU seed=$STAGE3_EVAL_RPC_PROTOCOL_SEED actions_per_call=$STAGE3_EVAL_STOP_ORACLE_RECOVERY_ACTIONS_PER_CALL"
echo "[oracle-path-smoke] output=$OUTPUT_DIR"

bash scripts/run_system2_stop_force_continue_train3_smoke_mxc500.sh

/mnt/afs/lixiaoou/intern/fjl/envs/qwen25/bin/python - "$OUTPUT_DIR" "$STAGE3_EVAL_STOP_ORACLE_RECOVERY_ACTIONS_PER_CALL" <<'PY'
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1])
expected_actions_per_call = int(sys.argv[2])
progress = [json.loads(line) for line in (root / "progress.json").open() if line.strip()]
if len(progress) != 3:
    raise SystemExit(f"Expected 3 completed oracle-path episodes, found {len(progress)}")
total_recovery_calls = 0
total_primitive_actions = 0
for row in progress:
    if row.get("system2_stop_collect_oracle_path_from_start") is not True:
        raise SystemExit(f"Oracle path-from-start provenance is missing: {row}")
    if row.get("system2_stop_collect_oracle_recovery_after_negative") is not False:
        raise SystemExit("Recovery-after-negative unexpectedly remained enabled")
    if int(row.get("system2_stop_oracle_recovery_activations", 0)) != 1:
        raise SystemExit(f"Oracle path did not activate exactly once: {row}")
    if row.get("system2_stop_oracle_recovery_activation_reason") != "oracle_path_from_start":
        raise SystemExit(f"Wrong oracle-path activation reason: {row}")
    if int(row.get("system2_stop_oracle_recovery_goal_probes", 0)) != 8:
        raise SystemExit(f"Oracle path did not collect all goal probes: {row}")
    if int(row.get("system2_stop_oracle_recovery_actions_per_call", 0)) != expected_actions_per_call:
        raise SystemExit(f"Wrong oracle recovery action chunk: {row}")
    total_recovery_calls += int(row.get("system2_stop_oracle_recovery_calls", 0))
    total_primitive_actions += int(
        row.get("system2_stop_oracle_recovery_primitive_actions", 0)
    )
if total_primitive_actions <= total_recovery_calls:
    raise SystemExit(
        "Oracle action chunk was not exercised: "
        f"calls={total_recovery_calls} primitives={total_primitive_actions}"
    )

labels_path = root / "system2_stop_rollout_labels.jsonl"
labels = [json.loads(line) for line in labels_path.open() if line.strip()]
multimodal_path = root / "system2_stop_multimodal_examples.jsonl"
multimodal = [json.loads(line) for line in multimodal_path.open() if line.strip()]
if len(multimodal) != len(labels):
    raise SystemExit(
        "Feature/multimodal STOP row count mismatch: "
        f"features={len(labels)} multimodal={len(multimodal)}"
    )
seen_keys = set()
image_count = 0
for row in multimodal:
    key = row.get("key")
    if not key or key in seen_keys:
        raise SystemExit(f"Missing or duplicate multimodal STOP key: {key!r}")
    seen_keys.add(key)
    if row.get("dataset_split") != "train":
        raise SystemExit("Multimodal STOP smoke contains a non-train record")
    if row.get("privileged_offline_label") is not True:
        raise SystemExit(f"Multimodal STOP provenance is missing: {key}")
    history = row.get("history_views") or []
    source_indices = row.get("history_source_buffer_indices") or []
    if len(history) != len(source_indices):
        raise SystemExit(f"Multimodal history provenance mismatch: {key}")
    for views in [row.get("current_views"), *history]:
        if not isinstance(views, dict) or set(views) != {"front", "right", "back", "left"}:
            raise SystemExit(f"Incomplete multimodal views: {key}")
        for relative_path in views.values():
            path = root / relative_path
            if not path.is_file() or path.stat().st_size <= 0:
                raise SystemExit(f"Missing multimodal image: {path}")
            image_count += 1
by_episode = defaultdict(Counter)
for row in labels:
    if row.get("dataset_split") != "train":
        raise SystemExit("Oracle-path smoke contains a non-train record")
    if row.get("oracle_recovery_active") is not True:
        raise SystemExit("Oracle-path row was not marked as privileged recovery")
    target = row.get("stop_target")
    if target in (0, 1):
        by_episode[(row["scene_id"], int(row["episode_id"]))][int(target)] += 1
for key, counts in by_episode.items():
    if counts[1] < 8:
        raise SystemExit(f"Episode {key} has too few positive goal views: {counts}")
    if counts[0] < 1:
        raise SystemExit(f"Episode {key} has no far negative views: {counts}")
print(
    "Oracle-path smoke postflight passed: "
    f"episodes={len(progress)} rows={len(labels)} "
    f"multimodal_rows={len(multimodal)} images={image_count} "
    f"positives={sum(counts[1] for counts in by_episode.values())} "
    f"negatives={sum(counts[0] for counts in by_episode.values())}"
)
PY

echo "[oracle-path-smoke] complete: $OUTPUT_DIR"

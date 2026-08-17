#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

FJL_ROOT="${FJL_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
OUTPUT_ROOT="${SYSTEM2_STOP_SEEDED_RECOVERY_ROOT:-${FJL_ROOT}/model/output_system2_stop_oracle_recovery_seeded_near17_20260717}"
GPU_DEVICE="${SYSTEM2_STOP_SEEDED_RECOVERY_GPU:-7}"
BASE_PORT="${SYSTEM2_STOP_SEEDED_RECOVERY_BASE_PORT:-50080}"

SEEDS=(42 118 119 121 122)
COUNTS=(4 1 4 2 3)
COHORTS=(
  configs/eval_cohorts/system2_stop_oracle_recovery_seed42.json
  configs/eval_cohorts/system2_stop_oracle_recovery_seed118.json
  configs/eval_cohorts/system2_stop_oracle_recovery_seed119.json
  configs/eval_cohorts/system2_stop_oracle_recovery_seed121.json
  configs/eval_cohorts/system2_stop_oracle_recovery_seed122.json
)

mkdir -p "$OUTPUT_ROOT"
for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  expected="${COUNTS[$index]}"
  cohort="${COHORTS[$index]}"
  output_dir="$OUTPUT_ROOT/seed_${seed}"
  port="$((BASE_PORT + index))"
  if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite seeded recovery output: $output_dir" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] seeded recovery seed=$seed episodes=$expected cohort=$cohort"
  STAGE3_EVAL_RPC_PROTOCOL_SEED="$seed" \
  STAGE3_EVAL_EPISODE_LIST="$cohort" \
  STAGE3_EVAL_OUTPUT_PATH="$output_dir" \
  STAGE3_FORCE_CONTINUE_GPU="$GPU_DEVICE" \
  STAGE3_FORCE_CONTINUE_RPC_PORT="$port" \
  STAGE3_EVAL_STOP_ORACLE_RECOVERY_FROM_COHORT_TRIGGERS=1 \
  STAGE3_FORCE_CONTINUE_RUN_STAMP="seed${seed}_$(date +%Y%m%d_%H%M%S)" \
    bash scripts/run_system2_stop_force_continue_train3_smoke_mxc500.sh

  /mnt/afs/lixiaoou/intern/fjl/envs/qwen25/bin/python - \
    "$output_dir" "$expected" "$seed" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
seed = int(sys.argv[3])
progress = [json.loads(line) for line in (root / "progress.json").open()]
if len(progress) != expected:
    raise SystemExit(f"seed {seed}: expected {expected} episodes, found {len(progress)}")
for row in progress:
    if int(row.get("rpc_protocol_seed", -1)) != seed:
        raise SystemExit(f"seed {seed}: progress seed mismatch")
    if int(row.get("system2_stop_oracle_recovery_activations", 0)) != 1:
        raise SystemExit(f"seed {seed}: recovery did not activate exactly once: {row}")
    if int(row.get("system2_stop_oracle_recovery_goal_probes", 0)) != 8:
        raise SystemExit(f"seed {seed}: recovery did not collect 8 goal probes: {row}")
    if row.get("system2_stop_oracle_recovery_from_cohort_triggers") is not True:
        raise SystemExit(f"seed {seed}: cohort trigger mode was not recorded")
    reason = row.get("system2_stop_oracle_recovery_activation_reason")
    if reason not in {
        "current_false_stop",
        "current_positive_stop",
        "historical_false_stop_call",
    }:
        raise SystemExit(f"seed {seed}: invalid recovery activation reason: {reason!r}")
    if row.get("system2_stop_historical_trigger_call_index") is None:
        raise SystemExit(f"seed {seed}: historical trigger provenance is missing")
    if reason == "historical_false_stop_call" and not row.get(
        "system2_stop_historical_trigger_reached"
    ):
        raise SystemExit(f"seed {seed}: historical trigger was not reached")
labels_path = root / "system2_stop_rollout_labels.jsonl"
if not labels_path.is_file() or labels_path.stat().st_size == 0:
    raise SystemExit(f"seed {seed}: missing rollout labels")
print(f"seed {seed} recovery postflight passed: episodes={expected}")
PY
done

echo "[seeded-recovery] complete: $OUTPUT_ROOT"

#!/usr/bin/env bash
set -Eeuo pipefail

FJL_ROOT=/mnt/afs/liwenhao/agent/370910109
REPO="$FJL_ROOT/HeatmapVLN"
QWEN_PYTHON="$FJL_ROOT/envs/qwen25/bin/python"
SOURCE_AUDIT="$FJL_ROOT/data/candidate_support_audit_v2/train_balanced_512_native_seed42"
PROBE_REPORT="$FJL_ROOT/model/candidate_identifiability_probe_v2/train_balanced_512_native_seed42/candidate_identifiability_report.json"
TARGETS_DIR="$FJL_ROOT/data/candidate_continuation_targets_v1/train_balanced_1024_seed20260810"
COHORTS_DIR="$FJL_ROOT/data/candidate_support_audit_cohorts_v2/train_balanced_512_seed20260810"
CONTINUATION_ROOT="$FJL_ROOT/data/candidate_continuation_v1/train_balanced_1024_native_pi0_seed42"
OUTPUT_ROOT="$FJL_ROOT/model/candidate_continuation_v1/train_balanced_1024_native_pi0_seed42"
TARGET_BUILDER="$REPO/scripts/evaluation/build_candidate_continuation_targets.py"
CONTINUATION_CLIENT="$REPO/scripts/evaluation/r2r_candidate_continuation_client.py"
CONTINUATION_SUMMARIZER="$REPO/scripts/evaluation/summarize_candidate_continuations.py"
BASE_LAUNCHER="$REPO/scripts/run_candidate_support_audit_8gpu_mxc500.sh"

for path in \
  "$QWEN_PYTHON" \
  "$SOURCE_AUDIT/shard_00/manifest.json" \
  "$PROBE_REPORT" \
  "$COHORTS_DIR/plan.json" \
  "$TARGET_BUILDER" \
  "$CONTINUATION_CLIENT" \
  "$CONTINUATION_SUMMARIZER" \
  "$BASE_LAUNCHER"; do
  [[ -e "$path" ]] || { echo "Missing required continuation input: $path" >&2; exit 1; }
done

cd "$REPO"

if [[ ! -s "$TARGETS_DIR/plan.json" || "${CONTINUATION_REBUILD_TARGETS:-0}" == 1 ]]; then
  echo "[continuation] building deterministic 1024-state target plan"
  "$QWEN_PYTHON" -u "$TARGET_BUILDER" \
    --audit-root "$SOURCE_AUDIT" \
    --probe-report "$PROBE_REPORT" \
    --output-dir "$TARGETS_DIR" \
    --expected-shards 8 \
    --target-states 1024 \
    --episode-end-states 256 \
    --max-states-per-episode 2 \
    --scene-split-seed 20260810 \
    --batch-size 128 \
    --device cuda
fi

for rank in 0 1 2 3 4 5 6 7; do
  target_file="$TARGETS_DIR/targets_shard_$(printf '%02d' "$rank").json"
  [[ -s "$target_file" ]] || { echo "Missing target shard: $target_file" >&2; exit 1; }
done

export CONTINUATION_SOURCE_AUDIT_ROOT="$SOURCE_AUDIT"
export CONTINUATION_TARGETS_DIR="$TARGETS_DIR"
export AUDIT_CONTROL_CLIENT="$CONTINUATION_CLIENT"
export AUDIT_SUMMARIZER_SCRIPT="$CONTINUATION_SUMMARIZER"
export AUDIT_SUMMARY_BASENAME=candidate_continuation_summary.json
export AUDIT_COHORTS_DIR="$COHORTS_DIR"
export AUDIT_COHORT_EPISODES_PER_SHARD=64
export AUDIT_ROOT="$CONTINUATION_ROOT"
export AUDIT_OUTPUT_ROOT="$OUTPUT_ROOT"
export AUDIT_DATASET_SPLIT=train
export AUDIT_DEPLOYMENT_ARM=native
export AUDIT_MAX_EPISODES_PER_SHARD=0
export AUDIT_MAX_GB_TOTAL=40
export AUDIT_MAX_GB_PER_SHARD=5
export EVAL_GPU_DEVICES="${EVAL_GPU_DEVICES:-0,1,2,3,4,5,6,7}"
export EVAL_RPC_PORT_BASE="${EVAL_RPC_PORT_BASE:-51700}"
export EVAL_DISPLAY_BASE="${EVAL_DISPLAY_BASE:-320}"
export CONTINUATION_MAX_FUTURE_CYCLES="${CONTINUATION_MAX_FUTURE_CYCLES:-80}"

exec bash "$BASE_LAUNCHER"

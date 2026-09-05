#!/usr/bin/env bash
# EXP-13 A: cache every DAgger candidate state's decision-layer features.
#
# Eight independent single-GPU workers, one per card, then a CPU merge.  This
# is not a distributed job: each worker owns ``index % 8 == rank`` and writes
# its own shard, so a card that dies costs one eighth of the sweep and can be
# rerun alone.
#
# Website submission form (CLAUDE.md 1.1): every parameter is an environment
# variable and the whole job is one `bash scripts/run_exp13_feature_cache_...`.
#
# Cost: about 1.5 h wall clock for ~30.8k states (measured 0.8 states/s/GPU on
# the EXP-12 D2 probe, same forward).
#
# Ledger: docs/experiments/README.md (EXP-13).

set -uo pipefail

FJL_ROOT="${EXP13_FJL_ROOT:-/mnt/afs/liwenhao/agent/370910109}"
REPO="${EXP13_REPO:-$FJL_ROOT/HeatmapVLN}"
PYTHON="${EXP13_PYTHON:-$FJL_ROOT/envs/qwen25/bin/python}"
OUT_ROOT="${EXP13_CACHE_ROOT:-$FJL_ROOT/model/exp13_decision_features}"
DAGGER_ROOT="${EXP13_DAGGER_ROOT:-$FJL_ROOT/data/heatmap_system1_dagger_v1/round_000/full_train_4way_seed17}"
ORACLE_VIEWS="${EXP13_ORACLE_VIEWS:-$FJL_ROOT/model/exp12_recovery_gate/d1_per_state.jsonl}"
CHECKPOINT="${EXP13_CHECKPOINT:-$FJL_ROOT/model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/best.pth}"
CONFIG="${EXP13_CONFIG:-configs/ppa_action_refine_v2_8gpu.yaml}"
BUCKETS="${EXP13_BUCKETS:-dagger_hard,dagger_normal}"
MAX_STATES="${EXP13_MAX_STATES:-0}"
SHARDS="${EXP13_SHARDS:-8}"

# The probe config carries five $VAR placeholders.  A missing one does not fail
# at startup: it surfaces as HFValidationError after the cold AFS model load,
# twenty minutes in (ledger 5, item 11).  Export them all, then prove it.
export PPA_DATA_ROOT="${PPA_DATA_ROOT:-$FJL_ROOT/r2r_panoramic_data_v2/train}"
export PPA_AMB3R_CACHE_ROOT="${PPA_AMB3R_CACHE_ROOT:-$FJL_ROOT/data/amb3r_endpoint_v3_full_r2r}"
export INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-$FJL_ROOT/InternNav-Model}"
export PPA_ACTION_REFINE_OUTPUT_ROOT="${PPA_ACTION_REFINE_OUTPUT_ROOT:-$OUT_ROOT/_cfg_unused}"
export PPA_TENSORBOARD_ROOT="${PPA_TENSORBOARD_ROOT:-$OUT_ROOT/_cfg_unused/tensorboard}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_HOME}"
export LD_LIBRARY_PATH="$MACA_HOME/lib:$MACA_HOME/ompi/lib:$MACA_HOME/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"

RUNTIME_ROOT="${EXP13_RUNTIME_ROOT:-$FJL_ROOT/model/output_past_plan_action_v1_8gpu_stage2_retry1/_runtime_cache}"
export HF_HOME="$RUNTIME_ROOT/huggingface"
export TORCH_HOME="$RUNTIME_ROOT/torch"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg"
export MPLCONFIGDIR="$RUNTIME_ROOT/matplotlib"
export TRITON_CACHE_DIR="$RUNTIME_ROOT/triton"

die() { printf '[exp13-cache] ERROR: %s\n' "$*" >&2; exit 2; }

[[ -x "$PYTHON" ]] || die "missing python: $PYTHON"
[[ -d "$DAGGER_ROOT" ]] || die "missing DAgger collection: $DAGGER_ROOT"
[[ -f "$ORACLE_VIEWS" ]] || die "missing EXP-12 per-state oracle rows: $ORACLE_VIEWS"
[[ -f "$CHECKPOINT" ]] || die "missing deployed checkpoint: $CHECKPOINT"
for name in PPA_DATA_ROOT PPA_AMB3R_CACHE_ROOT INTERNNAV_MODEL_PATH; do
  path="${!name}"
  [[ -e "$path" ]] || die "\$$name does not exist: $path"
done

IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
[[ "${#gpus[@]}" -eq "$SHARDS" ]] || die "need $SHARDS GPUs, got $CUDA_VISIBLE_DEVICES"

cd "$REPO" || die "cannot cd to $REPO"
mkdir -p "$OUT_ROOT"
echo "[exp13-cache] start $(date -u +%FT%TZ) shards=$SHARDS out=$OUT_ROOT"

pids=()
for index in $(seq 0 $((SHARDS - 1))); do
  gpu="${gpus[$index]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/tools/cache_recovery_decision_features.py \
    --config "$CONFIG" \
    --collection-root "$DAGGER_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --per-state-jsonl "$ORACLE_VIEWS" \
    --buckets "$BUCKETS" \
    --max-states "$MAX_STATES" \
    --shard-index "$index" --shard-count "$SHARDS" \
    --output-npz "$OUT_ROOT/features_shard${index}.npz" \
    > "$OUT_ROOT/cache_shard${index}.log" 2>&1 &
  pids+=("$!")
  echo "[exp13-cache] shard $index -> GPU $gpu (pid ${pids[-1]})"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[exp13-cache] shard $index OK"
  else
    echo "[exp13-cache] shard $index FAILED; see $OUT_ROOT/cache_shard${index}.log" >&2
    failed=$((failed + 1))
  fi
done
[[ "$failed" -eq 0 ]] || die "$failed shard(s) failed"

merge_args=()
for index in $(seq 0 $((SHARDS - 1))); do
  merge_args+=(--merge "$OUT_ROOT/features_shard${index}.npz")
done
"$PYTHON" scripts/tools/cache_recovery_decision_features.py \
  "${merge_args[@]}" --output-npz "$OUT_ROOT/features_merged.npz" \
  > "$OUT_ROOT/merge.log" 2>&1 || die "merge failed; see $OUT_ROOT/merge.log"

"$PYTHON" scripts/tools/fit_recovery_readout.py \
  --features "$OUT_ROOT/features_merged.npz" \
  --output-json "$OUT_ROOT/readout.json" \
  > "$OUT_ROOT/readout.log" 2>&1 || die "readout failed; see $OUT_ROOT/readout.log"

echo "[exp13-cache] done $(date -u +%FT%TZ)"
echo "[exp13-cache] readout: $OUT_ROOT/readout.json"
tail -n 40 "$OUT_ROOT/readout.log"

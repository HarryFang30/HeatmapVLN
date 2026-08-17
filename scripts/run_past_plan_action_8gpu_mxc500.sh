#!/usr/bin/env bash
# Website-submit entry point for the formal 8-GPU PPA two-stage run.
#
# Stage 1: current AMB3R-adapted 79-tensor Past best -> fresh Future/shared-map
#          optimizer (History + Future losses only).
# Stage 2: Stage-1 EMA/deployment best -> fresh optimizer, fresh exact-zero
#          bridge (Action + History + Future + preserve + delta losses).
#
# No checkpoint digest is pinned and no file lock is acquired. Identity is
# enforced by cache/checkpoint schemas, tensor coverage, and run-completion
# records.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PPA_REPO_ROOT="${PPA_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
PPA_ALLOWED_ROOT="${PPA_ALLOWED_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
PPA_QWEN_PYTHON="${PPA_QWEN_PYTHON:-/mnt/afs/lixiaoou/intern/fjl/envs/qwen25/bin/python}"
INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-/mnt/afs/lixiaoou/intern/fjl/InternNav-Model}"
# R2R is physically <corpus>/train/<scene>/clip_*.  This must be the direct
# scene root so VLNSlidingWindowDataset applies its deterministic MD5 scene
# auto-split for the configured logical train/val datasets.
PPA_DATA_ROOT="${PPA_DATA_ROOT:-/mnt/afs/lixiaoou/intern/fjl/r2r_paronamic_data/train}"
PPA_AMB3R_CACHE_ROOT="${PPA_AMB3R_CACHE_ROOT:-}"
PPA_PAST_INIT_CHECKPOINT="${PPA_PAST_INIT_CHECKPOINT:-}"
PPA_OUTPUT_ROOT="${PPA_OUTPUT_ROOT:-}"

PPA_STAGE1_OUTPUT_ROOT="${PPA_STAGE1_OUTPUT_ROOT:-${PPA_OUTPUT_ROOT}/stage1_map_pretrain}"
PPA_STAGE2_OUTPUT_ROOT="${PPA_STAGE2_OUTPUT_ROOT:-${PPA_OUTPUT_ROOT}/stage2_joint}"
PPA_TENSORBOARD_ROOT="${PPA_TENSORBOARD_ROOT:-${PPA_OUTPUT_ROOT}/tensorboard}"
PPA_STAGE1_CONFIG="${PPA_STAGE1_CONFIG:-${PPA_REPO_ROOT}/configs/ppa_stage1_map_pretrain_8gpu.yaml}"
PPA_STAGE2_CONFIG="${PPA_STAGE2_CONFIG:-${PPA_REPO_ROOT}/configs/ppa_stage2_joint_8gpu.yaml}"
PPA_CONTRACT_CHECKER="${PPA_CONTRACT_CHECKER:-${PPA_REPO_ROOT}/scripts/tools/validate_ppa_8gpu_contract.py}"

PPA_GPU_DEVICES="${PPA_GPU_DEVICES:-0,1,2,3,4,5,6,7}"
PPA_MASTER_ADDR="${PPA_MASTER_ADDR:-127.0.0.1}"
PPA_STAGE1_MASTER_PORT="${PPA_STAGE1_MASTER_PORT:-29681}"
PPA_STAGE2_MASTER_PORT="${PPA_STAGE2_MASTER_PORT:-29682}"
PPA_STAGE1_EPOCHS="${PPA_STAGE1_EPOCHS:-4}"
PPA_STAGE2_EPOCHS="${PPA_STAGE2_EPOCHS:-4}"
PPA_NUM_WORKERS="${PPA_NUM_WORKERS:-2}"
PPA_PREFETCH_FACTOR="${PPA_PREFETCH_FACTOR:-2}"
# Warm the AFS-backed Python/module pages once before eight ranks import them.
PPA_PREWARM_IMPORTS="${PPA_PREWARM_IMPORTS:-1}"

# The website job can be submitted before the independent cache phase ends.
# 1 waits fail-closed for the atomically published ready marker; 0 fails now.
PPA_WAIT_FOR_CACHE="${PPA_WAIT_FOR_CACHE:-1}"
PPA_CACHE_POLL_SECONDS="${PPA_CACHE_POLL_SECONDS:-60}"
# Zero means no timeout; scheduler cancellation/signals still stop the job.
PPA_CACHE_WAIT_TIMEOUT_SECONDS="${PPA_CACHE_WAIT_TIMEOUT_SECONDS:-0}"

log() {
  printf '[ppa-8gpu][%s] %s\n' "$(date -Is 2>/dev/null || date)" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 2
}

require_nonempty() {
  local name="$1" value="$2"
  [[ -n "$value" ]] || die "set ${name}"
}

require_positive_integer() {
  local name="$1" value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] && (( 10#$value > 0 )) \
    || die "${name} must be a positive integer, got '$value'"
}

require_nonnegative_integer() {
  local name="$1" value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] \
    || die "${name} must be a non-negative integer, got '$value'"
}

# GNU realpath -m resolves every existing symlink while permitting a missing
# final path. macOS/BSD realpath lacks -m, so the portable fallback recursively
# resolves the longest existing prefix and canonicalizes the missing suffix.
_portable_realpath_m() {
  local path="$1" parent base resolved_parent
  if [[ -e "$path" || -L "$path" ]]; then
    realpath "$path" 2>/dev/null
    return
  fi
  [[ "$path" != "/" ]] || {
    printf '/\n'
    return
  }
  parent="$(dirname -- "$path")"
  base="$(basename -- "$path")"
  resolved_parent="$(_portable_realpath_m "$parent")" || return 1
  case "$base" in
    .)
      printf '%s\n' "$resolved_parent"
      ;;
    ..)
      if [[ "$resolved_parent" == "/" ]]; then
        printf '/\n'
      else
        resolved_parent="${resolved_parent%/*}"
        printf '%s\n' "${resolved_parent:-/}"
      fi
      ;;
    *)
      printf '%s/%s\n' "${resolved_parent%/}" "$base"
      ;;
  esac
}

canonicalize_missing_ok() {
  local raw="$1" absolute
  [[ -n "$raw" ]] || return 1
  if [[ "$raw" == /* ]]; then
    absolute="$raw"
  else
    absolute="$(pwd -P)/$raw"
  fi
  if realpath -m -- / >/dev/null 2>&1; then
    realpath -m -- "$absolute" 2>/dev/null
  else
    _portable_realpath_m "$absolute"
  fi
}

is_strict_descendant() {
  local child="${1%/}" parent="${2%/}"
  [[ "$child" == "$parent/"* ]]
}

paths_overlap() {
  local left="${1%/}" right="${2%/}"
  [[
    "$left" == "$right"
    || "$left" == "$right/"*
    || "$right" == "$left/"*
  ]]
}

resolve_formal_path() {
  local name="$1" raw="$2" resolved
  resolved="$(canonicalize_missing_ok "$raw")" \
    || die "cannot resolve ${name} with realpath semantics: $raw"
  is_strict_descendant "$resolved" "$PPA_ALLOWED_ROOT" \
    || die "${name} escapes PPA_ALLOWED_ROOT: raw=$raw resolved=$resolved allowed=$PPA_ALLOWED_ROOT"
  printf '%s\n' "$resolved"
}

assert_pairwise_nonoverlap() {
  local -a labels=(output data cache init_checkpoint)
  local -a paths=(
    "$PPA_OUTPUT_ROOT"
    "$PPA_DATA_ROOT"
    "$PPA_AMB3R_CACHE_ROOT"
    "$PPA_PAST_INIT_CHECKPOINT"
  )
  local i j
  for ((i = 0; i < ${#paths[@]}; i++)); do
    for ((j = i + 1; j < ${#paths[@]}; j++)); do
      if paths_overlap "${paths[$i]}" "${paths[$j]}"; then
        die "formal path scopes overlap: ${labels[$i]}=${paths[$i]} ${labels[$j]}=${paths[$j]}"
      fi
    done
  done
}

on_signal() {
  local signal_name="$1" exit_code="$2"
  log "received ${signal_name}; stopping"
  exit "$exit_code"
}
trap 'on_signal HUP 129' HUP
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

command -v realpath >/dev/null 2>&1 \
  || die "realpath is required for formal path containment checks"
require_nonempty PPA_ALLOWED_ROOT "$PPA_ALLOWED_ROOT"
require_nonempty PPA_AMB3R_CACHE_ROOT "$PPA_AMB3R_CACHE_ROOT"
require_nonempty PPA_PAST_INIT_CHECKPOINT "$PPA_PAST_INIT_CHECKPOINT"
require_nonempty PPA_OUTPUT_ROOT "$PPA_OUTPUT_ROOT"

PPA_ALLOWED_ROOT="$(canonicalize_missing_ok "$PPA_ALLOWED_ROOT")" \
  || die "cannot resolve PPA_ALLOWED_ROOT: $PPA_ALLOWED_ROOT"
[[ -d "$PPA_ALLOWED_ROOT" ]] \
  || die "PPA_ALLOWED_ROOT must be an existing directory: $PPA_ALLOWED_ROOT"
[[ "$PPA_ALLOWED_ROOT" != "/" ]] \
  || die "PPA_ALLOWED_ROOT cannot be filesystem root"

# Canonicalize every formal path before using any of them. GNU realpath -m
# (or the equivalent fallback above) resolves every existing symlink, so a
# lexical path inside fjl cannot redirect a read or write outside the boundary.
PPA_REPO_ROOT="$(resolve_formal_path PPA_REPO_ROOT "$PPA_REPO_ROOT")"
PPA_QWEN_PYTHON="$(resolve_formal_path PPA_QWEN_PYTHON "$PPA_QWEN_PYTHON")"
INTERNNAV_MODEL_PATH="$(resolve_formal_path INTERNNAV_MODEL_PATH "$INTERNNAV_MODEL_PATH")"
PPA_DATA_ROOT="$(resolve_formal_path PPA_DATA_ROOT "$PPA_DATA_ROOT")"
PPA_AMB3R_CACHE_ROOT="$(resolve_formal_path PPA_AMB3R_CACHE_ROOT "$PPA_AMB3R_CACHE_ROOT")"
PPA_PAST_INIT_CHECKPOINT="$(resolve_formal_path PPA_PAST_INIT_CHECKPOINT "$PPA_PAST_INIT_CHECKPOINT")"
PPA_OUTPUT_ROOT="$(resolve_formal_path PPA_OUTPUT_ROOT "$PPA_OUTPUT_ROOT")"
PPA_STAGE1_OUTPUT_ROOT="$(resolve_formal_path PPA_STAGE1_OUTPUT_ROOT "$PPA_STAGE1_OUTPUT_ROOT")"
PPA_STAGE2_OUTPUT_ROOT="$(resolve_formal_path PPA_STAGE2_OUTPUT_ROOT "$PPA_STAGE2_OUTPUT_ROOT")"
PPA_TENSORBOARD_ROOT="$(resolve_formal_path PPA_TENSORBOARD_ROOT "$PPA_TENSORBOARD_ROOT")"
PPA_STAGE1_CONFIG="$(resolve_formal_path PPA_STAGE1_CONFIG "$PPA_STAGE1_CONFIG")"
PPA_STAGE2_CONFIG="$(resolve_formal_path PPA_STAGE2_CONFIG "$PPA_STAGE2_CONFIG")"
PPA_CONTRACT_CHECKER="$(resolve_formal_path PPA_CONTRACT_CHECKER "$PPA_CONTRACT_CHECKER")"
PPA_CONFIG_SCHEMA="$(resolve_formal_path PPA_CONFIG_SCHEMA "$PPA_REPO_ROOT/src/config_schema.py")"
PPA_TRAIN_ENTRY="$(resolve_formal_path PPA_TRAIN_ENTRY "$PPA_REPO_ROOT/scripts/train.py")"
PPA_RUNTIME_CACHE_ROOT="$(resolve_formal_path PPA_RUNTIME_CACHE_ROOT "$PPA_OUTPUT_ROOT/_runtime_cache")"

[[ -d "$PPA_REPO_ROOT" ]] || die "repository not found: $PPA_REPO_ROOT"
[[ -f "$PPA_QWEN_PYTHON" && -x "$PPA_QWEN_PYTHON" ]] \
  || die "qwen25 Python is not an executable regular file: $PPA_QWEN_PYTHON"
[[ -d "$INTERNNAV_MODEL_PATH" ]] \
  || die "InternNav model directory missing: $INTERNNAV_MODEL_PATH"
[[ -d "$PPA_DATA_ROOT" ]] \
  || die "expert dataset root missing: $PPA_DATA_ROOT"
[[ ! -e "$PPA_AMB3R_CACHE_ROOT" || -d "$PPA_AMB3R_CACHE_ROOT" ]] \
  || die "AMB3R cache root exists but is not a directory: $PPA_AMB3R_CACHE_ROOT"
[[ -f "$PPA_PAST_INIT_CHECKPOINT" && -s "$PPA_PAST_INIT_CHECKPOINT" ]] \
  || die "Past initializer is missing/empty: $PPA_PAST_INIT_CHECKPOINT"
[[ ! -e "$PPA_OUTPUT_ROOT" || -d "$PPA_OUTPUT_ROOT" ]] \
  || die "output root exists but is not a directory: $PPA_OUTPUT_ROOT"
[[ -f "$PPA_STAGE1_CONFIG" && -s "$PPA_STAGE1_CONFIG" ]] \
  || die "Stage-1 config missing/empty: $PPA_STAGE1_CONFIG"
[[ -f "$PPA_STAGE2_CONFIG" && -s "$PPA_STAGE2_CONFIG" ]] \
  || die "Stage-2 config missing/empty: $PPA_STAGE2_CONFIG"
[[ -f "$PPA_CONTRACT_CHECKER" && -s "$PPA_CONTRACT_CHECKER" ]] \
  || die "contract checker missing/empty: $PPA_CONTRACT_CHECKER"
[[ -f "$PPA_CONFIG_SCHEMA" && -s "$PPA_CONFIG_SCHEMA" ]] \
  || die "live config schema missing/empty: $PPA_CONFIG_SCHEMA"
[[ -f "$PPA_TRAIN_ENTRY" && -s "$PPA_TRAIN_ENTRY" ]] \
  || die "training entry missing/empty: $PPA_TRAIN_ENTRY"

assert_pairwise_nonoverlap
is_strict_descendant "$PPA_STAGE1_OUTPUT_ROOT" "$PPA_OUTPUT_ROOT" \
  || die "Stage-1 output must be inside PPA_OUTPUT_ROOT"
is_strict_descendant "$PPA_STAGE2_OUTPUT_ROOT" "$PPA_OUTPUT_ROOT" \
  || die "Stage-2 output must be inside PPA_OUTPUT_ROOT"
is_strict_descendant "$PPA_TENSORBOARD_ROOT" "$PPA_OUTPUT_ROOT" \
  || die "TensorBoard output must be inside PPA_OUTPUT_ROOT"
is_strict_descendant "$PPA_RUNTIME_CACHE_ROOT" "$PPA_OUTPUT_ROOT" \
  || die "runtime cache must be inside PPA_OUTPUT_ROOT"
paths_overlap "$PPA_STAGE1_OUTPUT_ROOT" "$PPA_STAGE2_OUTPUT_ROOT" \
  && die "Stage-1 and Stage-2 output scopes overlap"
paths_overlap "$PPA_STAGE1_OUTPUT_ROOT" "$PPA_TENSORBOARD_ROOT" \
  && die "Stage-1 and TensorBoard output scopes overlap"
paths_overlap "$PPA_STAGE2_OUTPUT_ROOT" "$PPA_TENSORBOARD_ROOT" \
  && die "Stage-2 and TensorBoard output scopes overlap"
paths_overlap "$PPA_RUNTIME_CACHE_ROOT" "$PPA_STAGE1_OUTPUT_ROOT" \
  && die "runtime cache and Stage-1 output scopes overlap"
paths_overlap "$PPA_RUNTIME_CACHE_ROOT" "$PPA_STAGE2_OUTPUT_ROOT" \
  && die "runtime cache and Stage-2 output scopes overlap"
paths_overlap "$PPA_RUNTIME_CACHE_ROOT" "$PPA_TENSORBOARD_ROOT" \
  && die "runtime cache and TensorBoard output scopes overlap"

require_positive_integer PPA_STAGE1_EPOCHS "$PPA_STAGE1_EPOCHS"
require_positive_integer PPA_STAGE2_EPOCHS "$PPA_STAGE2_EPOCHS"
require_nonnegative_integer PPA_NUM_WORKERS "$PPA_NUM_WORKERS"
require_positive_integer PPA_PREFETCH_FACTOR "$PPA_PREFETCH_FACTOR"
require_positive_integer PPA_CACHE_POLL_SECONDS "$PPA_CACHE_POLL_SECONDS"
require_nonnegative_integer PPA_CACHE_WAIT_TIMEOUT_SECONDS "$PPA_CACHE_WAIT_TIMEOUT_SECONDS"
[[ "$PPA_WAIT_FOR_CACHE" == "0" || "$PPA_WAIT_FOR_CACHE" == "1" ]] \
  || die "PPA_WAIT_FOR_CACHE must be 0 or 1"
[[ "$PPA_PREWARM_IMPORTS" == "0" || "$PPA_PREWARM_IMPORTS" == "1" ]] \
  || die "PPA_PREWARM_IMPORTS must be 0 or 1"

IFS=',' read -r -a gpu_ids <<< "$PPA_GPU_DEVICES"
[[ ${#gpu_ids[@]} -eq 8 ]] \
  || die "PPA_GPU_DEVICES must contain exactly eight comma-separated IDs"
for ((i = 0; i < 8; i++)); do
  [[ "${gpu_ids[$i]}" =~ ^[0-9]+$ ]] \
    || die "PPA_GPU_DEVICES contains a non-numeric ID: ${gpu_ids[$i]}"
  for ((j = i + 1; j < 8; j++)); do
    [[ "${gpu_ids[$i]}" != "${gpu_ids[$j]}" ]] \
      || die "PPA_GPU_DEVICES contains duplicate ID ${gpu_ids[$i]}"
  done
done
for port in "$PPA_STAGE1_MASTER_PORT" "$PPA_STAGE2_MASTER_PORT"; do
  require_positive_integer master_port "$port"
  (( 10#$port >= 1024 && 10#$port <= 65535 )) \
    || die "master port outside [1024,65535]: $port"
done
[[ "$PPA_STAGE1_MASTER_PORT" != "$PPA_STAGE2_MASTER_PORT" ]] \
  || die "Stage 1 and Stage 2 must use different rendezvous ports"

[[ ! -e "$PPA_STAGE1_OUTPUT_ROOT" && ! -L "$PPA_STAGE1_OUTPUT_ROOT" ]] \
  || die "refusing to reuse Stage-1 output: $PPA_STAGE1_OUTPUT_ROOT"
[[ ! -e "$PPA_STAGE2_OUTPUT_ROOT" && ! -L "$PPA_STAGE2_OUTPUT_ROOT" ]] \
  || die "refusing to reuse Stage-2 output: $PPA_STAGE2_OUTPUT_ROOT"

export PPA_ALLOWED_ROOT PPA_REPO_ROOT PPA_QWEN_PYTHON INTERNNAV_MODEL_PATH
export PPA_DATA_ROOT PPA_AMB3R_CACHE_ROOT PPA_PAST_INIT_CHECKPOINT
export PPA_OUTPUT_ROOT PPA_STAGE1_OUTPUT_ROOT PPA_STAGE2_OUTPUT_ROOT
export PPA_TENSORBOARD_ROOT PPA_RUNTIME_CACHE_ROOT

# Pin every library cache under the formal output before the first Python
# process starts. This prevents writes to $HOME, ~/.cache, or shared model
# directories and makes cleanup/accounting local to the website job.
export HF_HOME="$PPA_RUNTIME_CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$PPA_RUNTIME_CACHE_ROOT/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="$PPA_RUNTIME_CACHE_ROOT/huggingface/hub"
export HUGGINGFACE_ASSETS_CACHE="$PPA_RUNTIME_CACHE_ROOT/huggingface/assets"
export HF_DATASETS_CACHE="$PPA_RUNTIME_CACHE_ROOT/huggingface/datasets"
export TRANSFORMERS_CACHE="$PPA_RUNTIME_CACHE_ROOT/huggingface/transformers"
export TORCH_HOME="$PPA_RUNTIME_CACHE_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$PPA_RUNTIME_CACHE_ROOT/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$PPA_RUNTIME_CACHE_ROOT/torch_inductor"
export XDG_CACHE_HOME="$PPA_RUNTIME_CACHE_ROOT/xdg"
export MPLCONFIGDIR="$PPA_RUNTIME_CACHE_ROOT/matplotlib"
export TRITON_CACHE_DIR="$PPA_RUNTIME_CACHE_ROOT/triton"
mkdir -p \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$HUGGINGFACE_ASSETS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$MPLCONFIGDIR" \
  "$TRITON_CACHE_DIR"

export CUDA_VISIBLE_DEVICES="$PPA_GPU_DEVICES"
export WORLD_SIZE=8
export MASTER_ADDR="$PPA_MASTER_ADDR"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_HOME}"
export LD_LIBRARY_PATH="${MACA_HOME}/lib:${MACA_HOME}/ompi/lib:${MACA_HOME}/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"

log "validating both YAML files against the live schema"
"$PPA_QWEN_PYTHON" "$PPA_CONTRACT_CHECKER" config \
  --schema "$PPA_CONFIG_SCHEMA" \
  --config "$PPA_STAGE1_CONFIG" \
  --expected-stage stage1_map_pretrain
"$PPA_QWEN_PYTHON" "$PPA_CONTRACT_CHECKER" config \
  --schema "$PPA_CONFIG_SCHEMA" \
  --config "$PPA_STAGE2_CONFIG" \
  --expected-stage stage2_joint

log "validating the exact 79-tensor Past initializer (fresh optimizer semantics)"
"$PPA_QWEN_PYTHON" "$PPA_CONTRACT_CHECKER" checkpoint \
  --path "$PPA_PAST_INIT_CHECKPOINT" \
  --kind past-init

readonly ready_marker="${PPA_AMB3R_CACHE_ROOT}/_control/cache.ready.json"
ready_marker_complete() {
  [[ -e "$ready_marker" || -L "$ready_marker" ]] || return 1
  [[ -f "$ready_marker" && -s "$ready_marker" && ! -L "$ready_marker" ]] \
    || die "ready marker exists but is not a non-empty regular file: $ready_marker"
  "$PPA_QWEN_PYTHON" -c \
    'import json,sys; x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get("schema")=="heatmapvln-amb3r-endpoint-pose-cache-ready-v2" and x.get("complete") is True else 1)' \
    "$ready_marker" >/dev/null 2>&1 \
    || die "published ready marker is malformed or incomplete: $ready_marker"
}

wait_started="$(date +%s)"
while ! ready_marker_complete; do
  if [[ "$PPA_WAIT_FOR_CACHE" == "0" ]]; then
    die "endpoint-v2 cache is not ready: $ready_marker"
  fi
  now="$(date +%s)"
  elapsed="$((now - wait_started))"
  if (( PPA_CACHE_WAIT_TIMEOUT_SECONDS > 0 && elapsed >= PPA_CACHE_WAIT_TIMEOUT_SECONDS )); then
    die "timed out after ${elapsed}s waiting for endpoint-v2 cache: $ready_marker"
  fi
  log "cache phase not complete; waiting ${PPA_CACHE_POLL_SECONDS}s for $ready_marker"
  sleep "$PPA_CACHE_POLL_SECONDS"
done

log "ready marker published; validating exact MD5 auto-split train+val sidecar coverage"
"$PPA_QWEN_PYTHON" "$PPA_CONTRACT_CHECKER" cache \
  --cache-root "$PPA_AMB3R_CACHE_ROOT" \
  --dataset-root "$PPA_DATA_ROOT" \
  --required-splits train val

if [[ "$PPA_PREWARM_IMPORTS" == "1" ]]; then
  prewarm_started="$(date +%s)"
  log "prewarming scripts.train once before eight-rank launch"
  if ! (
    cd "$PPA_REPO_ROOT"
    PYTHONDONTWRITEBYTECODE=1 "$PPA_QWEN_PYTHON" -c 'import scripts.train'
  ); then
    die "single-process scripts.train prewarm failed"
  fi
  prewarm_finished="$(date +%s)"
  log "single-process import prewarm complete in $((prewarm_finished - prewarm_started))s"
else
  log "single-process import prewarm disabled by PPA_PREWARM_IMPORTS=0"
fi

mkdir -p "$PPA_OUTPUT_ROOT/_launcher"
readonly stage1_log="$PPA_OUTPUT_ROOT/_launcher/stage1_map_pretrain.log"
readonly stage2_log="$PPA_OUTPUT_ROOT/_launcher/stage2_joint.log"

run_stage() {
  local label="$1" config="$2" init_checkpoint="$3" epochs="$4" port="$5" log_path="$6"
  local -a command=(
    "$PPA_QWEN_PYTHON" -m torch.distributed.run
    --nproc_per_node=8
    --master_addr="$PPA_MASTER_ADDR"
    --master_port="$port"
    "$PPA_TRAIN_ENTRY"
    --config "$config"
    --load-weights "$init_checkpoint"
    --distributed
    --epochs "$epochs"
    --num-workers "$PPA_NUM_WORKERS"
    --pin-memory
  )
  if (( PPA_NUM_WORKERS > 0 )); then
    command+=(--prefetch-factor "$PPA_PREFETCH_FACTOR")
  fi
  log "starting ${label}: world=8 per_rank_batch=1 accum=1 effective_batch=8"
  log "${label} initializer=${init_checkpoint}"
  if ! "${command[@]}" 2>&1 | tee "$log_path"; then
    die "${label} failed; see $log_path"
  fi
}

cd "$PPA_REPO_ROOT"
run_stage \
  stage1_map_pretrain \
  "$PPA_STAGE1_CONFIG" \
  "$PPA_PAST_INIT_CHECKPOINT" \
  "$PPA_STAGE1_EPOCHS" \
  "$PPA_STAGE1_MASTER_PORT" \
  "$stage1_log"

stage1_best="$(
  "$PPA_QWEN_PYTHON" "$PPA_CONTRACT_CHECKER" run-best \
    --output-root "$PPA_STAGE1_OUTPUT_ROOT" \
    --kind stage1
)"
[[ -n "$stage1_best" ]] || die "Stage-1 best resolver returned an empty path"
stage1_best="$(resolve_formal_path PPA_STAGE1_BEST_CHECKPOINT "$stage1_best")"
[[ -f "$stage1_best" && -s "$stage1_best" ]] \
  || die "validated Stage-1 best is missing/empty: $stage1_best"
is_strict_descendant "$stage1_best" "$PPA_STAGE1_OUTPUT_ROOT" \
  || die "Stage-1 best escapes its Stage-1 output root: $stage1_best"
log "Stage 1 complete; validated EMA/deployment checkpoint: $stage1_best"

# Deliberately use --load-weights, not --resume: only the complete Past Head
# and trained Future Head cross the boundary. The Stage-2 optimizer/scheduler
# are fresh and the new bridge remains at its exact zero initialization.
run_stage \
  stage2_joint \
  "$PPA_STAGE2_CONFIG" \
  "$stage1_best" \
  "$PPA_STAGE2_EPOCHS" \
  "$PPA_STAGE2_MASTER_PORT" \
  "$stage2_log"

stage2_best="$(
  "$PPA_QWEN_PYTHON" "$PPA_CONTRACT_CHECKER" run-best \
    --output-root "$PPA_STAGE2_OUTPUT_ROOT" \
    --kind stage2
)"
[[ -n "$stage2_best" ]] || die "Stage-2 best resolver returned an empty path"
stage2_best="$(resolve_formal_path PPA_STAGE2_BEST_CHECKPOINT "$stage2_best")"
[[ -f "$stage2_best" && -s "$stage2_best" ]] \
  || die "validated Stage-2 best is missing/empty: $stage2_best"
is_strict_descendant "$stage2_best" "$PPA_STAGE2_OUTPUT_ROOT" \
  || die "Stage-2 best escapes its Stage-2 output root: $stage2_best"
log "two-stage training complete"
log "Stage-1 best: $stage1_best"
log "Stage-2 best: $stage2_best"
log "launcher logs: $PPA_OUTPUT_ROOT/_launcher"

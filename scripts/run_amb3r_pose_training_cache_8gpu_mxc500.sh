#!/usr/bin/env bash
# Persistent-model workers for causal map endpoints. No
# flock, lock file, or digest.
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TOOLS_DIR="${REPO_ROOT}/scripts"
ALLOWED_ROOT=${ALLOWED_ROOT:-/mnt/afs/lixiaoou/intern/fjl}
QWEN_PYTHON=${QWEN_PYTHON:-${ALLOWED_ROOT}/envs/qwen25/bin/python}
HEATMAP_REPO=${HEATMAP_REPO:-${REPO_ROOT}}
AMB3R_ROOT=${AMB3R_ROOT:-${ALLOWED_ROOT}/amb3r}
DA3_CHECKPOINT=${DA3_CHECKPOINT:-${AMB3R_ROOT}/checkpoints/DA3NESTED-GIANT-LARGE}
DATASET_ROOT=${DATASET_ROOT:-${ALLOWED_ROOT}/data/heatmap_randomwalk_train_v1}
CACHE_ROOT=${CACHE_ROOT:-${ALLOWED_ROOT}/data/heatmap_randomwalk_amb3r_endpoint_cache_v2}
PLAN_PATH=${PLAN_PATH:-${CACHE_ROOT}/_control/plan.json}
AMB3R_GPU_DEVICES=${AMB3R_GPU_DEVICES:-0,1,2,3,4,5,6,7}
EXPECTED_NUM_GPUS=${EXPECTED_NUM_GPUS:-8}
SPLITS=${SPLITS:-train,val}
NUM_HISTORY=${NUM_HISTORY:-8}
MIN_HISTORY=${MIN_HISTORY:-5}
MAP_INIT_WINDOW=${MAP_INIT_WINDOW:-20}
MAP_EVERY=${MAP_EVERY:-8}
CLIP_RETRIES=${CLIP_RETRIES:-2}
# 0 means unlimited retries. A transient accelerator/AFS failure therefore
# does not discard a hard-won eight-GPU web allocation; valid clips are
# semantically validated and skipped on every retry.
SHARD_MAX_ATTEMPTS=${SHARD_MAX_ATTEMPTS:-0}
SHARD_RETRY_DELAY_SECONDS=${SHARD_RETRY_DELAY_SECONDS:-30}
MAX_CLIPS_PER_SPLIT=${MAX_CLIPS_PER_SPLIT:-0}
AMB3R_PREWARM_IMPORTS=${AMB3R_PREWARM_IMPORTS:-1}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
LOG_ROOT=${LOG_ROOT:-${CACHE_ROOT}/_control/logs/${RUN_TAG}}

# qwen25 is a MetaX build. These must be present before importing torch/triton.
export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"

# Keep all runtime caches inside the user-authorized fjl subtree.
RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT:-${AMB3R_ROOT}/checkpoints/runtime_cache}
export HF_HOME="${HF_HOME:-${RUNTIME_CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${RUNTIME_CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_CACHE_ROOT}/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUNTIME_CACHE_ROOT}/matplotlib}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/triton}"
# qwen25 can import xFormers, but its fused SwiGLU CUDA extension has no MXC
# kernel. DA3's audited PyTorch SwiGLU fallback has the same w12/w3 parameter
# layout and arithmetic, so weights remain exact while execution stays native.
# These two proven MXC compatibility values are deliberately fail-closed:
# inherited web-job values such as 0 must never re-enable the crashing paths.
export DA3_DISABLE_XFORMERS=1
# MXC's PyTorch build has no memory-efficient SDPA kernel. Query-only
# chunking is mathematically exact (all keys/values remain visible) and bounds
# the otherwise quadratic temporary score allocation during DA3 warm-up.
export DA3_SDPA_QUERY_CHUNK_SIZE=256
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" \
  "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"

required_paths=(
  "${QWEN_PYTHON}"
  "${HEATMAP_REPO}/src/vo/online_amb3r.py"
  "${AMB3R_ROOT}/slam/slam_config.yaml"
  "${DA3_CHECKPOINT}/config.json"
  "${DA3_CHECKPOINT}/model.safetensors"
  "${DATASET_ROOT}"
  "${TOOLS_DIR}/amb3r_vo/build_training_cache_plan.py"
  "${TOOLS_DIR}/amb3r_vo/export_training_cache_shard.py"
  "${TOOLS_DIR}/amb3r_vo/validate_training_cache_shard.py"
  "${TOOLS_DIR}/amb3r_vo/validate_training_cache.py"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[amb3r-cache] missing required path: ${path}" >&2
    exit 2
  fi
done

IFS=',' read -r -a gpu_devices <<< "${AMB3R_GPU_DEVICES}"
if [[ ${#gpu_devices[@]} -ne ${EXPECTED_NUM_GPUS} ]]; then
  echo "[amb3r-cache] AMB3R_GPU_DEVICES must contain exactly ${EXPECTED_NUM_GPUS} devices" >&2
  exit 2
fi
if [[ "${NUM_HISTORY}" != "8" || "${MIN_HISTORY}" != "5" ]]; then
  echo "[amb3r-cache] production cache contract requires K=8,min_history=5" >&2
  exit 2
fi
if [[ "${AMB3R_PREWARM_IMPORTS}" != "0" && "${AMB3R_PREWARM_IMPORTS}" != "1" ]]; then
  echo "[amb3r-cache] AMB3R_PREWARM_IMPORTS must be 0 or 1" >&2
  exit 2
fi

mkdir -p "${CACHE_ROOT}/_control" "${LOG_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false

echo "[amb3r-cache] building deterministic frame-balanced plan"
"${QWEN_PYTHON}" "${TOOLS_DIR}/amb3r_vo/build_training_cache_plan.py" \
  --dataset-root "${DATASET_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --plan "${PLAN_PATH}" \
  --splits "${SPLITS}" \
  --num-shards "${EXPECTED_NUM_GPUS}" \
  --num-history "${NUM_HISTORY}" \
  --min-history "${MIN_HISTORY}" \
  --map-init-window "${MAP_INIT_WINDOW}" \
  --map-every "${MAP_EVERY}" \
  --max-clips-per-split "${MAX_CLIPS_PER_SPLIT}" \
  --allowed-root "${ALLOWED_ROOT}" \
  | tee "${LOG_ROOT}/plan.log"
plan_status=${PIPESTATUS[0]}
if [[ ${plan_status} -ne 0 ]]; then
  echo "[amb3r-cache] plan construction failed" >&2
  exit "${plan_status}"
fi

# Fast path: only a published root marker can claim a complete cache. Avoid a
# costly Python/AFS import on a brand-new cache where the marker is absent.
# When a marker exists, full semantic validation remains mandatory and also
# removes a stale marker through --write-ready before any resume attempt.
root_ready="${CACHE_ROOT}/_control/cache.ready.json"
if [[ -f "${root_ready}" ]]; then
  if "${QWEN_PYTHON}" "${TOOLS_DIR}/amb3r_vo/validate_training_cache.py" \
    --plan "${PLAN_PATH}" --workers 16 --write-ready \
    >"${LOG_ROOT}/initial_validation.log" 2>&1; then
    echo "[amb3r-cache] all planned clips were already valid"
    cat "${LOG_ROOT}/initial_validation.log"
    exit 0
  fi
else
  echo "[amb3r-cache] root ready marker absent; skipping redundant full-cache preflight"
fi

declare -a pending_shards=()
for ((shard_id = 0; shard_id < EXPECTED_NUM_GPUS; shard_id++)); do
  pending_shards+=("${shard_id}")
done
attempt=1
imports_prewarmed=0
while [[ ${#pending_shards[@]} -gt 0 && ( ${SHARD_MAX_ATTEMPTS} -eq 0 || ${attempt} -le ${SHARD_MAX_ATTEMPTS} ) ]]; do
  if [[ ${SHARD_MAX_ATTEMPTS} -eq 0 ]]; then
    attempt_label="${attempt}/unlimited"
  else
    attempt_label="${attempt}/${SHARD_MAX_ATTEMPTS}"
  fi
  echo "[amb3r-cache] attempt ${attempt_label}; pending=${pending_shards[*]}"
  declare -a launch_shards=()
  for shard_id in "${pending_shards[@]}"; do
    shard_ready="${CACHE_ROOT}/_control/shard_$(printf '%02d' "${shard_id}").ready.json"
    if [[ -f "${shard_ready}" ]]; then
      if "${QWEN_PYTHON}" "${TOOLS_DIR}/amb3r_vo/validate_training_cache_shard.py" \
        --plan "${PLAN_PATH}" --shard-id "${shard_id}" --workers 12 --write-ready \
        --allowed-root "${ALLOWED_ROOT}" \
        >"${LOG_ROOT}/shard_$(printf '%02d' "${shard_id}")_preflight_attempt_${attempt}.log" 2>&1; then
        echo "[amb3r-cache] shard ${shard_id} already complete; skipping DA3 load"
      else
        launch_shards+=("${shard_id}")
      fi
    else
      echo "[amb3r-cache] shard ${shard_id} ready marker absent; scheduling export"
      launch_shards+=("${shard_id}")
    fi
  done
  if [[ ${#launch_shards[@]} -eq 0 ]]; then
    pending_shards=()
    break
  fi

  # AFS cold-import metadata is much slower when every worker traverses the
  # Python dependency tree concurrently. Import the exact DA3/online backend
  # stack once in a CPU-only process; Linux page cache then serves all workers.
  # This does not construct a model, allocate an accelerator, or write a cache
  # payload. It runs only when at least one shard genuinely needs export.
  if [[ "${AMB3R_PREWARM_IMPORTS}" == "1" && ${imports_prewarmed} -eq 0 ]]; then
    echo "[amb3r-cache] prewarming DA3/online backend imports once"
    if ! PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="${AMB3R_ROOT}:${HEATMAP_REPO}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${QWEN_PYTHON}" -c \
      'from amb3r.model_zoo import load_model; from src.vo.online_amb3r import OnlineAMB3RSession, StatefulAMB3RBackend' \
      >"${LOG_ROOT}/import_prewarm.log" 2>&1; then
      cat "${LOG_ROOT}/import_prewarm.log" >&2
      echo "[amb3r-cache] import prewarm failed" >&2
      exit 2
    fi
    imports_prewarmed=1
    echo "[amb3r-cache] import prewarm complete"
  fi

  declare -a pids=()
  declare -a pid_shards=()
  for shard_id in "${launch_shards[@]}"; do
    gpu=${gpu_devices[${shard_id}]}
    log_path="${LOG_ROOT}/shard_$(printf '%02d' "${shard_id}")_attempt_${attempt}.log"
    echo "[amb3r-cache] start shard=${shard_id} physical_gpu=${gpu} log=${log_path}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${QWEN_PYTHON}" "${TOOLS_DIR}/amb3r_vo/export_training_cache_shard.py" \
      --plan "${PLAN_PATH}" \
      --shard-id "${shard_id}" \
      --repo "${HEATMAP_REPO}" \
      --amb3r-root "${AMB3R_ROOT}" \
      --da3-checkpoint "${DA3_CHECKPOINT}" \
      --device cuda:0 \
      --map-init-window "${MAP_INIT_WINDOW}" \
      --map-every "${MAP_EVERY}" \
      --clip-retries "${CLIP_RETRIES}" \
      --allowed-root "${ALLOWED_ROOT}" \
      >"${log_path}" 2>&1 &
    pids+=("$!")
    pid_shards+=("${shard_id}")
  done

  declare -a next_pending=()
  for index in "${!pids[@]}"; do
    pid=${pids[${index}]}
    shard_id=${pid_shards[${index}]}
    if wait "${pid}"; then
      echo "[amb3r-cache] shard ${shard_id} completed"
    else
      status=$?
      echo "[amb3r-cache] shard ${shard_id} exited ${status}; it will resume" >&2
      next_pending+=("${shard_id}")
    fi
  done
  pending_shards=("${next_pending[@]}")
  if [[ ${#pending_shards[@]} -gt 0 && ( ${SHARD_MAX_ATTEMPTS} -eq 0 || ${attempt} -lt ${SHARD_MAX_ATTEMPTS} ) ]]; then
    echo "[amb3r-cache] retrying failed shards after ${SHARD_RETRY_DELAY_SECONDS}s"
    sleep "${SHARD_RETRY_DELAY_SECONDS}"
  fi
  attempt=$((attempt + 1))
done

echo "[amb3r-cache] validating every endpoint clip and publishing cache.ready.json last"
if "${QWEN_PYTHON}" "${TOOLS_DIR}/amb3r_vo/validate_training_cache.py" \
  --plan "${PLAN_PATH}" --workers 24 --require-shard-ready --write-ready \
  | tee "${LOG_ROOT}/final_validation.log"; then
  echo "[amb3r-cache] READY: ${CACHE_ROOT}/_control/cache.ready.json"
  exit 0
fi

echo "[amb3r-cache] incomplete after ${SHARD_MAX_ATTEMPTS} configured attempts." >&2
echo "[amb3r-cache] rerun this same command; valid clips are skipped semantically." >&2
exit 1

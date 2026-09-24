#!/usr/bin/env bash
# EXP-18: History Head inference dump for one tier (VO + GT pose arms).
#
# Shards the tier's clip list over the GPUs (one dump_history_predictions.py
# process per GPU, --shard-index/--num-shards), retries failed shards a bounded
# number of times (the dump resumes per clip), then merges the shard manifests
# into EXP_ROOT/dumps/<TIER>/manifest.json. Paths come from
# scripts/exp18/common.py (EXP18_ROOT / EXP18_RENDER_ROOT / EXP18_WORKSPACE
# override them). Prerequisites: the tier's clip list and, for C/D/E, its
# AMB3R cache (scripts/exp18/run_amb3r_cache.sh).
#
# Website / dev machine (setsid-friendly: all output goes to log files):
#   cd <staged HeatmapVLN source>
#   export TIER=B
#   bash scripts/exp18/run_dump.sh
# Env:
#   TIER                A | B | C | D | E (required)
#   EXP18_GPUS          comma list of physical GPUs, one shard each (default common.GPU_IDS)
#   EXP18_GIT_SHA       recorded in the manifests (else <source>/.exp18_git_sha)
#   DUMP_NUM_WORKERS    dataloader workers per process (6)
#   DUMP_MAX_ATTEMPTS   2
#   DUMP_EXTRA_ARGS     extra dump CLI args (dev only, e.g. "--max-clips 3")
#   RUN_TAG             default UTC timestamp; logs in EXP_ROOT/logs/dump_<TIER>_<RUN_TAG>/
#   QWEN_PYTHON         default <workspace>/envs/qwen25/bin/python
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE=${EXP18_WORKSPACE:-/mnt/afs/liwenhao/agent/370910109}
QWEN_PYTHON=${QWEN_PYTHON:-${WORKSPACE}/envs/qwen25/bin/python}
DUMP="${REPO_ROOT}/scripts/exp18/dump_history_predictions.py"
TIER=${TIER:?set TIER=A|B|C|D|E}
case "${TIER}" in A|B|C|D|E) ;; *) echo "[exp18-dump] TIER must be one of A..E" >&2; exit 2 ;; esac

COMMON_VARS=$(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${REPO_ROOT}" "${QWEN_PYTHON}" - "${TIER}" <<'PY'
import shlex, sys
from scripts.exp18 import common as c
for k, v in {"EXP_ROOT": c.EXP_ROOT, "CLIP_LIST": c.clip_list_path(sys.argv[1]),
             "DEFAULT_GPUS": ",".join(map(str, c.GPU_IDS))}.items():
    print(f"{k}={shlex.quote(str(v))}")
PY
)
eval "${COMMON_VARS}"

GPUS=${EXP18_GPUS:-${DEFAULT_GPUS}}
MAX_ATTEMPTS=${DUMP_MAX_ATTEMPTS:-2}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
LOG_DIR="${EXP_ROOT}/logs/dump_${TIER}_${RUN_TAG}"
read -r -a EXTRA <<< "${DUMP_EXTRA_ARGS:-}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NUM_SHARDS=${#GPU_LIST[@]}
[[ -f "${CLIP_LIST}" ]] || { echo "[exp18-dump] clip list missing: ${CLIP_LIST}" >&2; exit 2; }
if ! [[ "${MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[exp18-dump] DUMP_MAX_ATTEMPTS must be a positive integer" >&2
  exit 2
fi

# qwen25 is a MetaX build: MACA must be set before torch/triton import.
export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
mkdir -p "${LOG_DIR}"

log() { echo "[exp18-dump] $(date -u +%FT%TZ) $*"; }
log "tier=${TIER} clip_list=${CLIP_LIST} gpus=${GPUS} extra=[${DUMP_EXTRA_ARGS:-}] logs=${LOG_DIR}"
log "repo=${REPO_ROOT} git_sha=${EXP18_GIT_SHA:-$(cat "${REPO_ROOT}/.exp18_git_sha" 2>/dev/null || echo unknown)}"

pending=()
for ((s = 0; s < NUM_SHARDS; s++)); do pending+=("${s}"); done
for ((attempt = 1; attempt <= MAX_ATTEMPTS && ${#pending[@]} > 0; attempt++)); do
  pids=()
  for s in "${pending[@]}"; do
    gpu=${GPU_LIST[${s}]}
    shard_log="${LOG_DIR}/shard_$(printf '%02d' "${s}")_attempt_${attempt}.log"
    log "attempt ${attempt}/${MAX_ATTEMPTS}: shard ${s}/${NUM_SHARDS} on GPU ${gpu} -> ${shard_log}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${QWEN_PYTHON}" "${DUMP}" --tier "${TIER}" --device cuda:0 \
      --shard-index "${s}" --num-shards "${NUM_SHARDS}" --num-workers "${DUMP_NUM_WORKERS:-6}" \
      "${EXTRA[@]+"${EXTRA[@]}"}" > "${shard_log}" 2>&1 < /dev/null &
    pids+=("$!")
  done
  failed=()
  for i in "${!pids[@]}"; do
    if wait "${pids[${i}]}"; then
      log "shard ${pending[${i}]} done"
    else
      log "shard ${pending[${i}]} exited $?; tail of its log:"
      tail -n 5 "${LOG_DIR}/shard_$(printf '%02d' "${pending[${i}]}")_attempt_${attempt}.log" || true
      failed+=("${pending[${i}]}")
    fi
  done
  pending=("${failed[@]+"${failed[@]}"}")
  if [[ ${#pending[@]} -gt 0 && ${attempt} -lt ${MAX_ATTEMPTS} ]]; then
    log "retrying shards ${pending[*]} in 30s (finished clips are skipped)"
    sleep 30
  fi
done

log "merging manifests"
"${QWEN_PYTHON}" "${DUMP}" --tier "${TIER}" --merge-manifests "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 | tee "${LOG_DIR}/merge.log"
if [[ ${#pending[@]} -gt 0 ]]; then
  log "FAILED shards after ${MAX_ATTEMPTS} attempt(s): ${pending[*]}; rerun the same command to resume"
  exit 1
fi
log "done"

#!/usr/bin/env bash
# EXP-18: AMB3R causal endpoint pose cache (the VO arm) for the new tiers C/D/E.
#
# Same builder, contract and settings as the production R2R v2 cache
# (scripts/run_amb3r_pose_training_cache_8gpu_mxc500.sh): plan -> one
# export_training_cache_shard.py per GPU -> per-shard + root validation, with
# NUM_HISTORY=8 MIN_HISTORY=5 MAP_INIT_WINDOW=20 MAP_EVERY=8. Differences:
#   * every python step gets --allowed-root (the stock launcher's validation
#     calls lack it, so any non-default root failed at the very end);
#   * shard retries are bounded (SHARD_MAX_ATTEMPTS, no "0 = forever");
#   * the plan covers exactly the tier's clip list: its dataset root is a
#     symlink view <EXP_ROOT>/amb3r_cache/<TIER>/dataset_view/<scene>/<clip>,
#     so rendered clips the selection rule dropped (e.g. < 20 frames, which
#     would abort the whole plan) never enter it. Cache keys stay
#     <scene>/<clip>, which is what the dump's strict reader resolves against
#     the real tier data root;
#   * runtime caches and logs live under EXP_ROOT, PYTHONDONTWRITEBYTECODE=1
#     (the exporter imports from the shared $W/amb3r tree).
# Paths come from scripts/exp18/common.py (EXP18_ROOT / EXP18_RENDER_ROOT /
# EXP18_WORKSPACE override them). Re-running resumes: valid clips are skipped.
#
# Website / dev machine (setsid-friendly: all output goes to log files):
#   cd <staged HeatmapVLN source>
#   export TIER=C
#   bash scripts/exp18/run_amb3r_cache.sh
# Env:
#   TIER                 C | D | E (required)
#   EXP18_GPUS           comma list of physical GPUs, one shard each (default common.GPU_IDS)
#   CLIP_SOURCE          list (default: common.clip_list_path(TIER)) | root (every clip_* of the
#                        tier data root with >= MAP_INIT_WINDOW frames; shorter ones are reported)
#   AMB3R_ALLOWED_ROOT   path guard for the python steps (default workspace; "/" for /tmp dev roots)
#   SHARD_MAX_ATTEMPTS   2      SHARD_RETRY_DELAY_SECONDS 30     CLIP_RETRIES 2
#   RUN_TAG              default UTC timestamp; logs in EXP_ROOT/logs/amb3r_cache_<TIER>_<RUN_TAG>/
#   QWEN_PYTHON AMB3R_ROOT DA3_CHECKPOINT RUNTIME_CACHE_ROOT  (defaults from common.py / EXP_ROOT)
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TOOLS_DIR="${REPO_ROOT}/scripts/amb3r_vo"
WORKSPACE=${EXP18_WORKSPACE:-/mnt/afs/liwenhao/agent/370910109}
QWEN_PYTHON=${QWEN_PYTHON:-${WORKSPACE}/envs/qwen25/bin/python}
TIER=${TIER:?set TIER=C|D|E}
case "${TIER}" in C|D|E) ;; *) echo "[exp18-cache] TIER must be C, D or E (A/B use the production cache)" >&2; exit 2 ;; esac

# Single source of truth for paths and constants: scripts/exp18/common.py.
COMMON_VARS=$(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${REPO_ROOT}" "${QWEN_PYTHON}" - "${TIER}" <<'PY'
import shlex, sys
from scripts.exp18 import common as c
t = c.TIERS[sys.argv[1]]
for k, v in {
    "EXP_ROOT": c.EXP_ROOT, "DATA_ROOT": t["data_root"], "CACHE_ROOT": t["cache_root"],
    "CLIP_LIST": c.clip_list_path(sys.argv[1]), "DEFAULT_GPUS": ",".join(map(str, c.GPU_IDS)),
    "C_NUM_HISTORY": c.NUM_HISTORY, "C_MIN_HISTORY": c.MIN_HISTORY, "C_MAP_INIT_WINDOW": c.MIN_FRAMES_FOR_AMB3R,
    "C_AMB3R_ROOT": c.AMB3R_ROOT, "C_DA3_CHECKPOINT": c.DA3_CHECKPOINT,
}.items():
    print(f"{k}={shlex.quote(str(v))}")
PY
)
eval "${COMMON_VARS}"

NUM_HISTORY=${C_NUM_HISTORY}
MIN_HISTORY=${C_MIN_HISTORY}
MAP_INIT_WINDOW=${C_MAP_INIT_WINDOW}
MAP_EVERY=8
AMB3R_ROOT=${AMB3R_ROOT:-${C_AMB3R_ROOT}}
DA3_CHECKPOINT=${DA3_CHECKPOINT:-${C_DA3_CHECKPOINT}}
ALLOWED_ROOT=${AMB3R_ALLOWED_ROOT:-${WORKSPACE}}
GPUS=${EXP18_GPUS:-${DEFAULT_GPUS}}
CLIP_SOURCE=${CLIP_SOURCE:-list}
SHARD_MAX_ATTEMPTS=${SHARD_MAX_ATTEMPTS:-2}
SHARD_RETRY_DELAY_SECONDS=${SHARD_RETRY_DELAY_SECONDS:-30}
CLIP_RETRIES=${CLIP_RETRIES:-2}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
WORK_DIR="${EXP_ROOT}/amb3r_cache/${TIER}"
VIEW_ROOT="${WORK_DIR}/dataset_view"
PLAN_PATH="${CACHE_ROOT}/_control/plan.json"
LOG_DIR="${EXP_ROOT}/logs/amb3r_cache_${TIER}_${RUN_TAG}"
RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT:-${EXP_ROOT}/amb3r_cache/runtime_cache}

if ! [[ "${SHARD_MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[exp18-cache] SHARD_MAX_ATTEMPTS must be a positive integer (no unlimited retries)" >&2
  exit 2
fi
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NUM_SHARDS=${#GPU_LIST[@]}
for path in "${QWEN_PYTHON}" "${AMB3R_ROOT}/slam/slam_config.yaml" "${DA3_CHECKPOINT}/model.safetensors" \
  "${REPO_ROOT}/src/vo/online_amb3r.py" "${DATA_ROOT}" "${TOOLS_DIR}/export_training_cache_shard.py"; do
  [[ -e "${path}" ]] || { echo "[exp18-cache] missing required path: ${path}" >&2; exit 2; }
done

# qwen25 is a MetaX build: MACA must be set before torch/triton import.
export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${RUNTIME_CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${RUNTIME_CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${RUNTIME_CACHE_ROOT}/xdg"
export MPLCONFIGDIR="${RUNTIME_CACHE_ROOT}/matplotlib"
export TRITON_CACHE_DIR="${RUNTIME_CACHE_ROOT}/triton"
# Same fail-closed MXC compatibility values as the production cache launcher.
export DA3_DISABLE_XFORMERS=1
export DA3_SDPA_QUERY_CHUNK_SIZE=256
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" \
  "${TRITON_CACHE_DIR}" "${CACHE_ROOT}/_control" "${LOG_DIR}" "${WORK_DIR}"

log() { echo "[exp18-cache] $(date -u +%FT%TZ) $*"; }
log "tier=${TIER} data_root=${DATA_ROOT} cache_root=${CACHE_ROOT} gpus=${GPUS} source=${CLIP_SOURCE} clip_list=${CLIP_LIST}"
log "repo=${REPO_ROOT} git_sha=${EXP18_GIT_SHA:-$(cat "${REPO_ROOT}/.exp18_git_sha" 2>/dev/null || echo unknown)} logs=${LOG_DIR}"

# 1. Symlink view of exactly the clips to cache (rebuilt every run).
if ! PYTHONPATH="${REPO_ROOT}" "${QWEN_PYTHON}" - "${TIER}" "${DATA_ROOT}" "${VIEW_ROOT}" "${CLIP_SOURCE}" \
  "${MAP_INIT_WINDOW}" > "${LOG_DIR}/view.log" 2>&1 <<'PY'
import json, os, shutil, sys
from pathlib import Path
from scripts.exp18 import common
tier, data_root, view, source, min_frames = sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3]), sys.argv[4], int(sys.argv[5])
if source == "list":
    if not common.clip_list_path(tier).is_file():
        sys.exit(f"clip list missing: {common.clip_list_path(tier)} (write it first, or CLIP_SOURCE=root)")
    keys = common.read_clip_list(tier)
elif source == "root":
    keys = sorted(f"{s.name}/{c.name}" for s in data_root.iterdir() if s.is_dir()
                  for c in s.iterdir() if c.is_dir() and c.name.startswith("clip_"))
else:
    sys.exit(f"CLIP_SOURCE must be list or root, got {source!r}")
kept, dropped = [], []
for key in keys:
    scene, clip = key.split("/")
    real = (data_root / key).resolve()
    if not real.is_dir():
        sys.exit(f"{key}: no such clip under {data_root}")
    if not clip.startswith("clip_") or real.parent.parent != data_root:
        sys.exit(f"{key}: not a <scene>/clip_* directory under {data_root} (symlinked clips break the reader)")
    meta = real / "meta.json"
    frames = json.loads(meta.read_text())["num_frames"] if meta.is_file() else -1
    if frames < min_frames:
        if source == "list":
            sys.exit(f"{key}: {frames} frames < {min_frames}; the clip list must only hold cacheable clips")
        dropped.append((key, frames))
        continue
    kept.append((key, real))
if not kept:
    sys.exit("no clips to cache")
if view.exists():
    shutil.rmtree(view)  # holds only symlinks; rmtree never follows them
for key, real in kept:
    (view / key).parent.mkdir(parents=True, exist_ok=True)
    os.symlink(real, view / key)
(view.parent / "view_keys.txt").write_text("".join(f"{k}\n" for k, _ in kept))
print(json.dumps({"view": str(view), "clips": len(kept), "scenes": len({k.split('/')[0] for k, _ in kept}),
                  "dropped_short_or_empty": dropped}))
PY
then
  cat "${LOG_DIR}/view.log" >&2
  log "could not build the clip view"
  exit 1
fi
cat "${LOG_DIR}/view.log"

# 2. Plan (md5 train/val labels; SPLITS=train,val must cover every scene of the flat view).
log "building plan: ${NUM_SHARDS} shard(s)"
if ! "${QWEN_PYTHON}" "${TOOLS_DIR}/build_training_cache_plan.py" \
  --dataset-root "${VIEW_ROOT}" --cache-root "${CACHE_ROOT}" --plan "${PLAN_PATH}" --splits train,val \
  --num-shards "${NUM_SHARDS}" --num-history "${NUM_HISTORY}" --min-history "${MIN_HISTORY}" \
  --map-init-window "${MAP_INIT_WINDOW}" --map-every "${MAP_EVERY}" --max-clips-per-split 0 \
  --allowed-root "${ALLOWED_ROOT}" > "${LOG_DIR}/plan.log" 2>&1; then
  cat "${LOG_DIR}/plan.log" >&2
  log "plan construction failed"
  exit 1
fi
"${QWEN_PYTHON}" - "${PLAN_PATH}" "${WORK_DIR}/view_keys.txt" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
want = sorted(l.strip() for l in open(sys.argv[2]) if l.strip())
got = sorted(e["clip_key"] for s in plan["shards"] for e in s["clips"])
split_sum = sum(v["clips"] for v in plan["by_split"].values())
if got != want or split_sum != plan["clip_count"] or plan["clip_count"] != len(want):
    sys.exit(f"plan does not cover the view: plan={len(got)} view={len(want)} by_split={plan['by_split']} "
             f"missing={sorted(set(want) - set(got))[:5]}")
print(f"[exp18-cache] plan covers all {len(want)} clips; by_split={json.dumps(plan['by_split'])}; "
      f"shard_frames={[s['frame_count'] for s in plan['shards']]}")
PY

validate_root() {  # $1 = log name; extra args passed through
  local name=$1; shift
  "${QWEN_PYTHON}" "${TOOLS_DIR}/validate_training_cache.py" --plan "${PLAN_PATH}" --workers 16 \
    --allowed-root "${ALLOWED_ROOT}" "$@" > "${LOG_DIR}/${name}.log" 2>&1
}

if [[ -f "${CACHE_ROOT}/_control/cache.ready.json" ]] && validate_root initial_validation --write-ready; then
  log "all planned clips already valid: ${CACHE_ROOT}/_control/cache.ready.json"
  exit 0
fi

# 3. Export: one persistent-model process per GPU, bounded retries of failed shards.
pending=()
for ((s = 0; s < NUM_SHARDS; s++)); do pending+=("${s}"); done
prewarmed=0
for ((attempt = 1; attempt <= SHARD_MAX_ATTEMPTS && ${#pending[@]} > 0; attempt++)); do
  launch=()
  for s in "${pending[@]}"; do
    marker="${CACHE_ROOT}/_control/shard_$(printf '%02d' "${s}").ready.json"
    if [[ -f "${marker}" ]] && "${QWEN_PYTHON}" "${TOOLS_DIR}/validate_training_cache_shard.py" \
      --plan "${PLAN_PATH}" --shard-id "${s}" --workers 12 --write-ready --allowed-root "${ALLOWED_ROOT}" \
      > "${LOG_DIR}/shard_$(printf '%02d' "${s}")_preflight_${attempt}.log" 2>&1; then
      log "shard ${s} already complete"
    else
      launch+=("${s}")
    fi
  done
  [[ ${#launch[@]} -gt 0 ]] || { pending=(); break; }
  if [[ ${prewarmed} -eq 0 ]]; then
    # One CPU-only import pass warms the AFS page cache for all workers.
    log "prewarming DA3/online backend imports"
    PYTHONPATH="${AMB3R_ROOT}:${REPO_ROOT}" "${QWEN_PYTHON}" -c \
      'from amb3r.model_zoo import load_model; from src.vo.online_amb3r import OnlineAMB3RSession, StatefulAMB3RBackend' \
      > "${LOG_DIR}/import_prewarm.log" 2>&1 || { cat "${LOG_DIR}/import_prewarm.log" >&2; exit 2; }
    prewarmed=1
  fi
  pids=()
  for s in "${launch[@]}"; do
    gpu=${GPU_LIST[${s}]}
    shard_log="${LOG_DIR}/shard_$(printf '%02d' "${s}")_attempt_${attempt}.log"
    log "attempt ${attempt}/${SHARD_MAX_ATTEMPTS}: shard ${s} on GPU ${gpu} -> ${shard_log}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${QWEN_PYTHON}" "${TOOLS_DIR}/export_training_cache_shard.py" \
      --plan "${PLAN_PATH}" --shard-id "${s}" --repo "${REPO_ROOT}" --amb3r-root "${AMB3R_ROOT}" \
      --da3-checkpoint "${DA3_CHECKPOINT}" --device cuda:0 --map-init-window "${MAP_INIT_WINDOW}" \
      --map-every "${MAP_EVERY}" --clip-retries "${CLIP_RETRIES}" --allowed-root "${ALLOWED_ROOT}" \
      > "${shard_log}" 2>&1 < /dev/null &
    pids+=("$!")
  done
  failed=()
  for i in "${!pids[@]}"; do
    if wait "${pids[${i}]}"; then
      log "shard ${launch[${i}]} completed"
    else
      log "shard ${launch[${i}]} exited $? (see its log)"
      failed+=("${launch[${i}]}")
    fi
  done
  pending=("${failed[@]+"${failed[@]}"}")
  if [[ ${#pending[@]} -gt 0 && ${attempt} -lt ${SHARD_MAX_ATTEMPTS} ]]; then
    log "retrying shards ${pending[*]} in ${SHARD_RETRY_DELAY_SECONDS}s"
    sleep "${SHARD_RETRY_DELAY_SECONDS}"
  fi
done

# 4. Root validation publishes cache.ready.json last.
if validate_root final_validation --require-shard-ready --write-ready; then
  log "READY: ${CACHE_ROOT}/_control/cache.ready.json"
  tail -n 12 "${LOG_DIR}/final_validation.log"
  exit 0
fi
cat "${LOG_DIR}/final_validation.log" >&2
log "incomplete after ${SHARD_MAX_ATTEMPTS} attempt(s); rerun the same command (valid clips are skipped)"
exit 1

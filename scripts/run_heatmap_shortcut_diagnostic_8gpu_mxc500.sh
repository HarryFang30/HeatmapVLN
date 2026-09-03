#!/usr/bin/env bash
# Matched head-only shortcut probes for the History Head (8x MetaX C500).
#
# Four input regimes -- full / vision-only / pose-only / no-input -- are trained
# from one byte-identical fresh head on one frozen backbone, then evaluated on
# one scene-disjoint validation subset.  Comparing them answers whether the
# head can localize history from vision at all, or whether relative pose alone
# already explains its score.
#
# Each (mode, seed) pair owns one GPU.  Four modes x two seeds fills an
# eight-GPU allocation exactly, and the second seed is what separates a real
# gap from probe noise.
#
# Blank-container safe: every parameter arrives as an environment variable,
# the interpreter is called by absolute path, and no conda activation, tmux,
# X server or interactive state is assumed.
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ALLOWED_ROOT=${ALLOWED_ROOT:-/mnt/afs/liwenhao/agent/370910109}
QWEN_PYTHON=${QWEN_PYTHON:-${ALLOWED_ROOT}/envs/qwen25/bin/python}
INTERNNAV_MODEL_PATH=${INTERNNAV_MODEL_PATH:-${ALLOWED_ROOT}/InternNav-Model}

SHORTCUT_ARCHITECTURE=${SHORTCUT_ARCHITECTURE:-internnav_single_view}
SHORTCUT_CONFIG=${SHORTCUT_CONFIG:-${REPO_ROOT}/configs/train_heatmap_internnav_single_view_8gpu.yaml}
SHORTCUT_DATA_ROOT=${SHORTCUT_DATA_ROOT:-${ALLOWED_ROOT}/data/heatmap_randomwalk_train_v1}
SHORTCUT_OUTPUT_ROOT=${SHORTCUT_OUTPUT_ROOT:-${ALLOWED_ROOT}/model/heatmap_shortcut_probe_v1}
# Optional: only legacy_panoramic consumes a frozen Stage1-S2 LoRA checkpoint.
SHORTCUT_CHECKPOINT=${SHORTCUT_CHECKPOINT:-}

SHORTCUT_MODES=${SHORTCUT_MODES:-full,vision-only,pose-only,no-input}
SHORTCUT_SEEDS=${SHORTCUT_SEEDS:-42,1337}
SHORTCUT_NUM_HISTORY=${SHORTCUT_NUM_HISTORY:-8}
# Measured on one C500 at K=8: ~1.4 s per training step and ~1.5 s per
# evaluated sample.  12000 steps plus the seven full-mode conditions over 400
# validation samples is about six hours on the slowest of the eight probes,
# which fits one allocation with room for a cold AFS start.  12000 updates is
# also the same order as the production head's optimizer-step count, so the
# probes are undertrained in samples seen, not in gradient steps taken.
# Steps equal samples so every training occurrence is presented exactly once
# and the regimes cannot differ merely in how well they memorise a small
# repeated subset.  The train pool holds 41416 scene-stratified samples.
SHORTCUT_TRAIN_STEPS=${SHORTCUT_TRAIN_STEPS:-12000}
SHORTCUT_TRAIN_SAMPLES=${SHORTCUT_TRAIN_SAMPLES:-12000}
SHORTCUT_VAL_SAMPLES=${SHORTCUT_VAL_SAMPLES:-400}
SHORTCUT_LEARNING_RATE=${SHORTCUT_LEARNING_RATE:-1e-4}
SHORTCUT_LOG_EVERY=${SHORTCUT_LOG_EVERY:-50}
SHORTCUT_EVAL_LOG_EVERY=${SHORTCUT_EVAL_LOG_EVERY:-50}
SHORTCUT_GPU_DEVICES=${SHORTCUT_GPU_DEVICES:-0,1,2,3,4,5,6,7}
SHORTCUT_PREWARM_IMPORTS=${SHORTCUT_PREWARM_IMPORTS:-1}
# 1 skips any (mode, seed) whose report.json already exists, so resubmitting
# the identical command continues an allocation that was cut short.
SHORTCUT_RESUME=${SHORTCUT_RESUME:-1}

RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
LOG_ROOT=${LOG_ROOT:-${SHORTCUT_OUTPUT_ROOT}/_logs/${RUN_TAG}}

# qwen25 is a MetaX build; these must exist before torch/triton import.
export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"

RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT:-${SHORTCUT_OUTPUT_ROOT}/_runtime_cache}
export HF_HOME="${HF_HOME:-${RUNTIME_CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${RUNTIME_CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_CACHE_ROOT}/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUNTIME_CACHE_ROOT}/matplotlib}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/triton}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export INTERNNAV_MODEL_PATH

die() { echo "[shortcut] ERROR: $*" >&2; exit 2; }

for path in \
  "${QWEN_PYTHON}" \
  "${SHORTCUT_CONFIG}" \
  "${SHORTCUT_DATA_ROOT}" \
  "${INTERNNAV_MODEL_PATH}/config.json" \
  "${REPO_ROOT}/scripts/tools/diagnose_heatmap_shortcuts.py" \
  "${REPO_ROOT}/scripts/tools/summarize_heatmap_shortcuts.py" ; do
  [[ -e "${path}" ]] || die "missing required path: ${path}"
done

case "${SHORTCUT_ARCHITECTURE}" in
  internnav_single_view)
    [[ -z "${SHORTCUT_CHECKPOINT}" ]] \
      || die "internnav_single_view runs on the released ViT; unset SHORTCUT_CHECKPOINT"
    ;;
  legacy_panoramic)
    [[ -n "${SHORTCUT_CHECKPOINT}" ]] \
      || die "legacy_panoramic requires SHORTCUT_CHECKPOINT (frozen Stage1-S2 LoRA)"
    [[ -e "${SHORTCUT_CHECKPOINT}" ]] || die "missing SHORTCUT_CHECKPOINT: ${SHORTCUT_CHECKPOINT}"
    ;;
  *)
    die "SHORTCUT_ARCHITECTURE must be internnav_single_view or legacy_panoramic"
    ;;
esac

IFS=',' read -r -a gpu_devices <<< "${SHORTCUT_GPU_DEVICES}"
IFS=',' read -r -a modes <<< "${SHORTCUT_MODES}"
IFS=',' read -r -a seeds <<< "${SHORTCUT_SEEDS}"
num_jobs=$(( ${#modes[@]} * ${#seeds[@]} ))
if [[ ${num_jobs} -gt ${#gpu_devices[@]} ]]; then
  die "need ${num_jobs} GPUs for ${#modes[@]} modes x ${#seeds[@]} seeds, have ${#gpu_devices[@]}"
fi

mkdir -p "${SHORTCUT_OUTPUT_ROOT}" "${LOG_ROOT}" \
  "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" \
  "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"

echo "[shortcut] repo=${REPO_ROOT}"
echo "[shortcut] architecture=${SHORTCUT_ARCHITECTURE} config=${SHORTCUT_CONFIG}"
echo "[shortcut] data=${SHORTCUT_DATA_ROOT}"
echo "[shortcut] output=${SHORTCUT_OUTPUT_ROOT} logs=${LOG_ROOT}"
echo "[shortcut] modes=${SHORTCUT_MODES} seeds=${SHORTCUT_SEEDS}"
echo "[shortcut] K=${SHORTCUT_NUM_HISTORY} steps=${SHORTCUT_TRAIN_STEPS}" \
     "train_samples=${SHORTCUT_TRAIN_SAMPLES} val_samples=${SHORTCUT_VAL_SAMPLES}"

# A cold AFS walk is far slower when eight processes do it at once.  One
# CPU-only process imports the dependency tree and enumerates every clip's
# chunk metadata; the eight probes then read both from the page cache.  This
# constructs no model and touches no accelerator.
if [[ "${SHORTCUT_PREWARM_IMPORTS}" == "1" ]]; then
  echo "[shortcut] prewarming imports and the clip index once"
  if ! SHORTCUT_CONFIG="${SHORTCUT_CONFIG}" \
     SHORTCUT_DATA_ROOT="${SHORTCUT_DATA_ROOT}" \
     SHORTCUT_ARCHITECTURE="${SHORTCUT_ARCHITECTURE}" \
     SHORTCUT_NUM_HISTORY="${SHORTCUT_NUM_HISTORY}" \
     PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
     "${QWEN_PYTHON}" - <<'PREWARM' >"${LOG_ROOT}/prewarm.log" 2>&1; then
import os
import types

import torch  # noqa: F401  (warms the heaviest import tree)
import transformers  # noqa: F401

from scripts.tools.diagnose_heatmap_shortcuts import build_dataset, load_config

args = types.SimpleNamespace(
    config=os.environ["SHORTCUT_CONFIG"],
    data_root=os.environ["SHORTCUT_DATA_ROOT"],
    device="cpu",
    num_history=int(os.environ["SHORTCUT_NUM_HISTORY"]),
    architecture=os.environ["SHORTCUT_ARCHITECTURE"],
    internnav_model_path=os.environ.get("INTERNNAV_MODEL_PATH"),
)
cfg = load_config(args)
for split in ("train", "val"):
    dataset = build_dataset(cfg, split)
    print(f"prewarm {split}: {len(dataset.clips)} clips, {len(dataset)} samples", flush=True)
PREWARM
    cat "${LOG_ROOT}/prewarm.log" >&2
    die "prewarm failed"
  fi
  tail -n 3 "${LOG_ROOT}/prewarm.log"
fi

declare -a pids=()
declare -a labels=()
gpu_index=0
for seed in "${seeds[@]}"; do
  for mode in "${modes[@]}"; do
    gpu=${gpu_devices[${gpu_index}]}
    gpu_index=$(( gpu_index + 1 ))
    seed_root="${SHORTCUT_OUTPUT_ROOT}/seed_${seed}"
    label="seed${seed}/${mode}"
    if [[ "${SHORTCUT_RESUME}" == "1" && -s "${seed_root}/${mode}/report.json" ]]; then
      echo "[shortcut] skip ${label}: report.json already present"
      continue
    fi
    mkdir -p "${seed_root}"
    log_path="${LOG_ROOT}/seed_${seed}_${mode}.log"
    echo "[shortcut] start ${label} gpu=${gpu} log=${log_path}"

    declare -a extra=()
    [[ -n "${SHORTCUT_CHECKPOINT}" ]] && extra+=(--checkpoint "${SHORTCUT_CHECKPOINT}")

    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${QWEN_PYTHON}" "${REPO_ROOT}/scripts/tools/diagnose_heatmap_shortcuts.py" \
      --mode "${mode}" \
      --architecture "${SHORTCUT_ARCHITECTURE}" \
      --config "${SHORTCUT_CONFIG}" \
      --data-root "${SHORTCUT_DATA_ROOT}" \
      --output-dir "${seed_root}" \
      --device cuda:0 \
      --num-history "${SHORTCUT_NUM_HISTORY}" \
      --train-steps "${SHORTCUT_TRAIN_STEPS}" \
      --train-samples "${SHORTCUT_TRAIN_SAMPLES}" \
      --val-samples "${SHORTCUT_VAL_SAMPLES}" \
      --learning-rate "${SHORTCUT_LEARNING_RATE}" \
      --seed "${seed}" \
      --log-every "${SHORTCUT_LOG_EVERY}" \
      --eval-log-every "${SHORTCUT_EVAL_LOG_EVERY}" \
      --internnav-model-path "${INTERNNAV_MODEL_PATH}" \
      "${extra[@]+"${extra[@]}"}" \
      >"${log_path}" 2>&1 &
    pids+=("$!")
    labels+=("${label}")
  done
done

failures=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    echo "[shortcut] done ${labels[${index}]}"
  else
    status=$?
    echo "[shortcut] FAILED ${labels[${index}]} (exit ${status})" >&2
    failures=$(( failures + 1 ))
  fi
done

summary_failures=0
for seed in "${seeds[@]}"; do
  seed_root="${SHORTCUT_OUTPUT_ROOT}/seed_${seed}"
  echo "[shortcut] summarizing ${seed_root}"
  if ! PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${QWEN_PYTHON}" \
    "${REPO_ROOT}/scripts/tools/summarize_heatmap_shortcuts.py" --root "${seed_root}" \
    >"${LOG_ROOT}/summary_seed_${seed}.log" 2>&1; then
    echo "[shortcut] summary failed for seed ${seed}; see ${LOG_ROOT}/summary_seed_${seed}.log" >&2
    summary_failures=$(( summary_failures + 1 ))
  else
    tail -n 40 "${LOG_ROOT}/summary_seed_${seed}.log"
  fi
done

if [[ ${failures} -gt 0 || ${summary_failures} -gt 0 ]]; then
  echo "[shortcut] ${failures} probe failure(s), ${summary_failures} summary failure(s)." >&2
  echo "[shortcut] Rerun the same command: completed probes are skipped." >&2
  exit 1
fi

echo "[shortcut] COMPLETE. Per-seed tables: ${SHORTCUT_OUTPUT_ROOT}/seed_*/task3_summary.{csv,json}"
exit 0

#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_ROOT="${HEATMAPVLN_PROJECT_ROOT:-/mnt/afs/lixiaoou/intern/fjl/HeatmapVLN}"
CONFIG_PATH="${HEATMAPVLN_PPA_CONFIG:-${PROJECT_ROOT}/configs/train_past_plan_action_stage2_smoke.yaml}"
PAST_CHECKPOINT="${HEATMAPVLN_PAST_CHECKPOINT:-}"

if [ -z "${INTERNNAV_MODEL_PATH:-}" ]; then
  echo "[ppa-smoke] waiting for INTERNNAV_MODEL_PATH" >&2
  exit 2
fi
if [ -z "${HEATMAPVLN_PPA_DATA_ROOT:-}" ]; then
  echo "[ppa-smoke] waiting for HEATMAPVLN_PPA_DATA_ROOT" >&2
  exit 2
fi
if [ -z "${HEATMAPVLN_PPA_OUTPUT_ROOT:-}" ]; then
  echo "[ppa-smoke] waiting for HEATMAPVLN_PPA_OUTPUT_ROOT" >&2
  exit 2
fi
if [ -z "${PAST_CHECKPOINT}" ] || [ ! -f "${PAST_CHECKPOINT}" ]; then
  echo "[ppa-smoke] set HEATMAPVLN_PAST_CHECKPOINT to the current 79-tensor Past best.pth" >&2
  exit 2
fi

cd "${PROJECT_ROOT}" || exit 2

# --dry-run executes one real forward/backward/optimizer step, verifies the
# zero-init native trajectory equality with shared explicit noise, and refuses
# any checkpoint write. It intentionally uses the active qwen25 environment.
python scripts/train.py \
  --config "${CONFIG_PATH}" \
  --load-weights "${PAST_CHECKPOINT}" \
  --dry-run \
  --max-batches 1 \
  --num-workers 0 \
  --no-pin-memory

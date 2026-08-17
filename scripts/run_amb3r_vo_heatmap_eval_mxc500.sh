#!/usr/bin/env bash
set -euo pipefail

FJL_ROOT=${FJL_ROOT:-/mnt/afs/lixiaoou/intern/fjl}
REPO=${REPO:-${FJL_ROOT}/HeatmapVLN}
AMB3R_ROOT=${AMB3R_ROOT:-${FJL_ROOT}/amb3r}
QWEN_ENV=${QWEN_ENV:-${FJL_ROOT}/envs/qwen25}
DA3_CHECKPOINT=${DA3_CHECKPOINT:-${AMB3R_ROOT}/checkpoints/DA3NESTED-GIANT-LARGE}
HEATMAP_RUN=${HEATMAP_RUN:-${FJL_ROOT}/model/output_heatmap_internnav_single_view_v1_4gpu/runs/run_20260803_143402}
HEATMAP_CONFIG=${HEATMAP_CONFIG:-${HEATMAP_RUN}/manifest/config.yaml}
HEATMAP_CHECKPOINT=${HEATMAP_CHECKPOINT:-${HEATMAP_RUN}/checkpoints/best.pth}
AMB3R_CLIP=${AMB3R_CLIP:-${FJL_ROOT}/r2r_paronamic_data/train/17DRP5sb8fy/clip_004345}
AMB3R_GPU_DEVICE=${AMB3R_GPU_DEVICE:-0}
AMB3R_MAX_FRAMES=${AMB3R_MAX_FRAMES:-0}
AMB3R_MAX_HEATMAP_SAMPLES=${AMB3R_MAX_HEATMAP_SAMPLES:-0}
AMB3R_SAMPLE_OFFSET=${AMB3R_SAMPLE_OFFSET:-0}
AMB3R_TRANSLATION_SCALE=${AMB3R_TRANSLATION_SCALE:-1.0}
AMB3R_MODE=${AMB3R_MODE:-backend}
AMB3R_MAP_INIT_WINDOW=${AMB3R_MAP_INIT_WINDOW:-20}
AMB3R_MAP_EVERY=${AMB3R_MAP_EVERY:-8}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT=${OUTPUT_ROOT:-${FJL_ROOT}/model/eval_amb3r_vo_heatmap/${RUN_TAG}}

for required in \
    "${QWEN_ENV}/bin/python" \
    "${REPO}/scripts/amb3r_vo/export_clip_poses.py" \
    "${REPO}/scripts/evaluation/evaluate_amb3r_heatmap_pair.py" \
    "${DA3_CHECKPOINT}/config.json" \
    "${DA3_CHECKPOINT}/model.safetensors" \
    "${HEATMAP_CONFIG}" \
    "${HEATMAP_CHECKPOINT}" \
    "${AMB3R_CLIP}/meta.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "[amb3r-heatmap] missing required path: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}"

export MACA_HOME=${MACA_HOME:-/opt/maca-3.3.0}
export MACA_PATH=${MACA_PATH:-${MACA_HOME}}
export MACA_DIR=${MACA_DIR:-${MACA_PATH}}
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES=${AMB3R_GPU_DEVICE}
export XDG_CACHE_HOME=${FJL_ROOT}/cache/xdg
export XDG_CONFIG_HOME=${FJL_ROOT}/runtime_config/amb3r_vo
export MPLCONFIGDIR=${FJL_ROOT}/runtime_config/amb3r_vo/matplotlib
export HF_HOME=${AMB3R_ROOT}/checkpoints/hf_home
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${FJL_ROOT}/cache/transformers
export TORCH_HOME=${FJL_ROOT}/cache/torch
export DA3_DISABLE_XFORMERS=${DA3_DISABLE_XFORMERS:-1}
export DA3_SDPA_QUERY_CHUNK_SIZE=${DA3_SDPA_QUERY_CHUNK_SIZE:-256}
export PYTHONPATH="${REPO}:${AMB3R_ROOT}:${AMB3R_ROOT}/thirdparty${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${MPLCONFIGDIR}" \
  "${HF_HOME}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}"

POSE_CACHE=${OUTPUT_ROOT}/amb3r_vo_poses.npz

echo "[amb3r-heatmap] phase 1/2: continuous RGB -> AMB3R-VO poses"
"${QWEN_ENV}/bin/python" "${REPO}/scripts/amb3r_vo/export_clip_poses.py" \
  --repo "${REPO}" \
  --amb3r-root "${AMB3R_ROOT}" \
  --da3-checkpoint "${DA3_CHECKPOINT}" \
  --clip "${AMB3R_CLIP}" \
  --output "${POSE_CACHE}" \
  --device cuda:0 \
  --mode "${AMB3R_MODE}" \
  --max-frames "${AMB3R_MAX_FRAMES}" \
  --map-init-window "${AMB3R_MAP_INIT_WINDOW}" \
  --map-every "${AMB3R_MAP_EVERY}"

echo "[amb3r-heatmap] phase 2/2: paired frozen-head GT-pose versus VO-pose audit"
"${QWEN_ENV}/bin/python" "${REPO}/scripts/evaluation/evaluate_amb3r_heatmap_pair.py" \
  --repo "${REPO}" \
  --config "${HEATMAP_CONFIG}" \
  --heatmap-checkpoint "${HEATMAP_CHECKPOINT}" \
  --pose-cache "${POSE_CACHE}" \
  --clip "${AMB3R_CLIP}" \
  --output-dir "${OUTPUT_ROOT}/paired_heatmaps" \
  --device cuda:0 \
  --sample-offset "${AMB3R_SAMPLE_OFFSET}" \
  --max-samples "${AMB3R_MAX_HEATMAP_SAMPLES}" \
  --translation-scale "${AMB3R_TRANSLATION_SCALE}"

echo "[amb3r-heatmap] complete: ${OUTPUT_ROOT}/paired_heatmaps/paired_report.json"

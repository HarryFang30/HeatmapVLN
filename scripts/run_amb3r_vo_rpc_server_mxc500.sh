#!/usr/bin/env bash
set -euo pipefail

FJL_ROOT=${FJL_ROOT:-/mnt/afs/liwenhao/agent/370910109}
REPO=${REPO:-${FJL_ROOT}/HeatmapVLN}
AMB3R_ROOT=${AMB3R_ROOT:-${FJL_ROOT}/amb3r}
RPC_ROOT=${RPC_ROOT:-${FJL_ROOT}/rpc}
QWEN_ENV=${QWEN_ENV:-${FJL_ROOT}/envs/qwen25}
DA3_CHECKPOINT=${DA3_CHECKPOINT:-${AMB3R_ROOT}/checkpoints/DA3NESTED-GIANT-LARGE}
AMB3R_GPU_DEVICE=${AMB3R_GPU_DEVICE:-0}
AMB3R_RPC_HOST=${AMB3R_RPC_HOST:-127.0.0.1}
AMB3R_RPC_PORT=${AMB3R_RPC_PORT:-50081}
AMB3R_MAP_INIT_WINDOW=${AMB3R_MAP_INIT_WINDOW:-20}
AMB3R_MAP_EVERY=${AMB3R_MAP_EVERY:-8}
AMB3R_TRANSLATION_SCALE=${AMB3R_TRANSLATION_SCALE:-1.0}
AMB3R_MAX_FRAMES_LIMIT=${AMB3R_MAX_FRAMES_LIMIT:-4096}

for required in \
    "${QWEN_ENV}/bin/python" \
    "${RPC_ROOT}/src/vla_rpc" \
    "${REPO}/scripts/amb3r_vo/rpc_amb3r_vo_server.py" \
    "${AMB3R_ROOT}/slam/slam_config.yaml" \
    "${DA3_CHECKPOINT}/config.json" \
    "${DA3_CHECKPOINT}/model.safetensors"; do
  if [[ ! -e "${required}" ]]; then
    echo "[amb3r-vo-rpc] missing required path: ${required}" >&2
    exit 1
  fi
done

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
export PYTHONPATH="${RPC_ROOT}/src:${REPO}:${AMB3R_ROOT}:${AMB3R_ROOT}/thirdparty${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${MPLCONFIGDIR}" \
  "${HF_HOME}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}"

echo "[amb3r-vo-rpc] gpu=${AMB3R_GPU_DEVICE} listen=${AMB3R_RPC_HOST}:${AMB3R_RPC_PORT} scale=${AMB3R_TRANSLATION_SCALE}"
exec "${QWEN_ENV}/bin/python" \
  "${REPO}/scripts/amb3r_vo/rpc_amb3r_vo_server.py" \
  --repo "${REPO}" \
  --amb3r-root "${AMB3R_ROOT}" \
  --da3-checkpoint "${DA3_CHECKPOINT}" \
  --device cuda:0 \
  --host "${AMB3R_RPC_HOST}" \
  --port "${AMB3R_RPC_PORT}" \
  --map-init-window "${AMB3R_MAP_INIT_WINDOW}" \
  --map-every "${AMB3R_MAP_EVERY}" \
  --translation-scale "${AMB3R_TRANSLATION_SCALE}" \
  --max-frames-limit "${AMB3R_MAX_FRAMES_LIMIT}"

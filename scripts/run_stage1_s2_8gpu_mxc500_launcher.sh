#!/usr/bin/env bash
# HeatmapVLN Stage1-S2（8×沐曦 C500）启动包装脚本
#
# 用法（单机 8 卡，默认）：
#   bash scripts/run_stage1_s2_8gpu_mxc500_launcher.sh
#
# 多机时：在每台机器上设置相同的 MASTER_ADDR / MASTER_PORT，并设置
#   WORLD_SIZE（机器数）、RANK（本机序号 0..WORLD_SIZE-1），且需自行改用
#   torchrun 的 --nnodes / --node_rank（当前仓库 scripts 仍为单机 torchrun）。
#
# Conda 环境固定为：conda activate /mnt/afs/lixiaoou/intern/fjl/envs/qwen25
# 若非交互 shell 里没有 conda 命令，请先 source conda.sh（见下方 CONDA_INIT_SH）
# 或根据实际情况修改「路径默认值」块中的数据/模型路径。

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 沐曦 MACA / 通信（与朋友 DiffSynth 脚本对齐，可按现场驱动版本改 MACA 路径）
# ---------------------------------------------------------------------------
export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"

export MCCL_IB_HCA="${MCCL_IB_HCA:-mlx5_0:0,mlx5_1:0,mlx5_4:0,mlx5_5:0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
# 使用 IB 时请勿沿用原先 NCCL_IB_DISABLE=1；若需走 socket 可显式 export NCCL_IB_DISABLE=1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# 多机分布式（仅作记录/与其它作业一致；单机可保持默认）
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export MASTER_PORT="${MASTER_PORT:-29617}"
# 单机 torchrun 可不设；多机时需在所有节点 export 相同 MASTER_ADDR
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} WORLD_SIZE=${WORLD_SIZE} RANK=${RANK}"

# ---------------------------------------------------------------------------
# Conda：激活环境固定为 /mnt/afs/lixiaoou/intern/fjl/envs/qwen25
# 仅需初始化 conda 函数（source conda.sh 或 conda shell hook）
# ---------------------------------------------------------------------------
if [[ -n "${CONDA_INIT_SH:-}" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_INIT_SH"
elif [[ -f "/mnt/afs/lixiaoou/intern/fjl/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/mnt/afs/lixiaoou/intern/fjl/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "未找到 conda：请 export CONDA_INIT_SH=/path/to/miniconda3/etc/profile.d/conda.sh 后重试。" >&2
  exit 1
fi
conda activate /mnt/afs/lixiaoou/intern/fjl/envs/qwen25

# ---------------------------------------------------------------------------
# 路径默认值（请按你机器上实际位置修改）
# ---------------------------------------------------------------------------
export INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-/mnt/afs/lixiaoou/intern/fjl/InternNav-Model}"
export INTERNNAV_BACKBONE="${INTERNNAV_BACKBONE:-$INTERNNAV_MODEL_PATH}"
export PANORAMIC_DATA_ROOT="${PANORAMIC_DATA_ROOT:-/mnt/afs/lixiaoou/intern/fjl/r2r_paronamic_data}"

export STAGE1_S2_LOAD_WEIGHTS="${STAGE1_S2_LOAD_WEIGHTS:-/mnt/afs/lixiaoou/intern/fjl/model/output/run_20260519_232017/checkpoints/latest.pth}"
export STAGE1_S2_OUT_DIR="${STAGE1_S2_OUT_DIR:-/mnt/afs/lixiaoou/intern/fjl/model/output_stage1_s2}"
export STAGE1_S2_TB_DIR="${STAGE1_S2_TB_DIR:-/mnt/afs/tensorlog/heatmapvln_stage1_s2_8gpu}"

mkdir -p "$REPO_ROOT/logs" "${STAGE1_S2_OUT_DIR}" "${STAGE1_S2_TB_DIR}"
LOG_FILE="${LOG_FILE:-$REPO_ROOT/logs/stage1_s2_8gpu_mxc500.log}"

# ---------------------------------------------------------------------------
# 训练超参（与原脚本一致，可按显存改 batch）
# ---------------------------------------------------------------------------
export STAGE1_S2_CONFIG="${STAGE1_S2_CONFIG:-configs/train_system2_panoramic_sft_8gpu.yaml}"
export GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT_STAGE1_S2="${MASTER_PORT_STAGE1_S2:-$MASTER_PORT}"
export STAGE1_S2_EPOCHS="${STAGE1_S2_EPOCHS:-4}"
# batch_size / grad_accum: defer to config (deleted overrides that
# were clobbering config values — 12 was the culprit for all OOMs)
export STAGE1_S2_BATCH_SIZE="${STAGE1_S2_BATCH_SIZE:-}"
export STAGE1_S2_GRAD_ACCUM_STEPS="${STAGE1_S2_GRAD_ACCUM_STEPS:-}"
export STAGE1_S2_NUM_WORKERS="${STAGE1_S2_NUM_WORKERS:-8}"
export STAGE1_S2_PREFETCH_FACTOR="${STAGE1_S2_PREFETCH_FACTOR:-2}"

# FlashAttention2：默认强制开启（与 train_system2_panoramic_sft_8gpu.yaml 一致）。
# 若环境无 FA2，运行前可 export STAGE1_S2_REQUIRE_FLASH_ATTN=0 与 STAGE1_S2_LLM_ATTN_IMPLEMENTATION=sdpa
export STAGE1_S2_REQUIRE_FLASH_ATTN="${STAGE1_S2_REQUIRE_FLASH_ATTN:-1}"
export STAGE1_S2_LLM_ATTN_IMPLEMENTATION="${STAGE1_S2_LLM_ATTN_IMPLEMENTATION:-flash_attention_2}"

bash scripts/run_stage1_s2_8gpu.sh 2>&1 | tee "$LOG_FILE"

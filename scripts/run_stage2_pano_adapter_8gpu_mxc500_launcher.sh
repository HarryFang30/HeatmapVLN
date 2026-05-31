#!/usr/bin/env bash
# HeatmapVLN Stage2 Pano Adapter（8×沐曦 C500）启动包装脚本
#
# 用法（单机 8 卡，默认）：
#   bash scripts/run_stage2_pano_adapter_8gpu_mxc500_launcher.sh
#
# 多机时：在每台机器上设置相同的 MASTER_ADDR / MASTER_PORT，并设置
#   WORLD_SIZE（机器数）、RANK（本机序号 0..WORLD_SIZE-1），且需自行改用
#   torchrun 的 --nnodes / --node_rank。
#
# Conda 环境固定为：conda activate /mnt/afs/lixiaoou/intern/fjl/envs/qwen25
# 若非交互 shell 里没有 conda 命令，请先 source conda.sh（见下方 CONDA_INIT_SH）
# 或根据实际情况修改「路径默认值」块中的数据/模型路径。
#
# 训练说明：
#   - Student: Stage1-S2 全景 SFT checkpoint（frozen）
#   - Teacher: InternNav System1（frozen, aligned 模式在线运行）
#   - Adapter: GeometryAwarePanoToNextDiTAdapter（~3M params，唯一可训练部分）
#   - 不需要预先收集 teacher sidecar（aligned 模式）

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 沐曦 MACA / 通信
# ---------------------------------------------------------------------------
export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"

export MCCL_IB_HCA="${MCCL_IB_HCA:-mlx5_0:0,mlx5_1:0,mlx5_4:0,mlx5_5:0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# 多机分布式
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export MASTER_PORT="${MASTER_PORT:-29618}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} WORLD_SIZE=${WORLD_SIZE} RANK=${RANK}"

# ---------------------------------------------------------------------------
# Conda
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

# Stage1-S2 checkpoint (student weights)
export STAGE1_S2_OUT_DIR="${STAGE1_S2_OUT_DIR:-/mnt/afs/lixiaoou/intern/fjl/model/output_stage1_s2}"

# Teacher JSONL (optional for aligned mode — auto-generated from dataset if not set)
# Only needed for sidecar mode or pre-filtered record subsets.
export STAGE2_ADAPTER_TEACHER_JSONL="${STAGE2_ADAPTER_TEACHER_JSONL:-}"

# Output
export STAGE2_ADAPTER_OUT_DIR="${STAGE2_ADAPTER_OUT_DIR:-/mnt/afs/lixiaoou/intern/fjl/model/output_stage2_adapter}"
export STAGE2_ADAPTER_TB_DIR="${STAGE2_ADAPTER_TB_DIR:-/mnt/afs/tensorlog/heatmapvln_stage2_adapter_8gpu}"

mkdir -p "$REPO_ROOT/logs" "${STAGE2_ADAPTER_OUT_DIR}" "${STAGE2_ADAPTER_TB_DIR}"
LOG_FILE="${LOG_FILE:-$REPO_ROOT/logs/stage2_adapter_8gpu_mxc500.log}"

# ---------------------------------------------------------------------------
# 训练超参（空值 = 使用 adapter config YAML 默认值）
# ---------------------------------------------------------------------------
export STAGE2_ADAPTER_STUDENT_CONFIG="${STAGE2_ADAPTER_STUDENT_CONFIG:-configs/train_pano_adapter_stage2_8gpu.yaml}"
export GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT_STAGE2_ADAPTER="${MASTER_PORT_STAGE2_ADAPTER:-$MASTER_PORT}"

# batch_size / lr / epochs: 空 = 使用 adapter config YAML 默认值
# 如需覆盖，export 对应 env var 即可
export STAGE2_ADAPTER_BATCH_SIZE="${STAGE2_ADAPTER_BATCH_SIZE:-}"
export STAGE2_ADAPTER_EPOCHS="${STAGE2_ADAPTER_EPOCHS:-}"
export STAGE2_ADAPTER_LR="${STAGE2_ADAPTER_LR:-}"
export STAGE2_ADAPTER_MAX_SAMPLES="${STAGE2_ADAPTER_MAX_SAMPLES:-0}"

# Teacher: aligned mode (on-the-fly, recommended for MXC500)
export STAGE2_ADAPTER_TEACHER_MODE="${STAGE2_ADAPTER_TEACHER_MODE:-aligned}"
export STAGE2_ADAPTER_TEACHER_DTYPE="${STAGE2_ADAPTER_TEACHER_DTYPE:-bfloat16}"
export STAGE2_ADAPTER_TEACHER_ATTN="${STAGE2_ADAPTER_TEACHER_ATTN:-sdpa}"

# FlashAttention2：默认强制开启（与 MXC500 环境一致）
export STAGE2_ADAPTER_REQUIRE_FLASH_ATTN="${STAGE2_ADAPTER_REQUIRE_FLASH_ATTN:-1}"

# Data — controlled by adapter config YAML. Override via
# STAGE2_ADAPTER_CONFIG=my_config.yaml, not env vars.
export STAGE2_ADAPTER_USE_TRAJ_IMAGES="${STAGE2_ADAPTER_USE_TRAJ_IMAGES:-true}"

bash scripts/run_stage2_pano_adapter_8gpu.sh 2>&1 | tee "$LOG_FILE"

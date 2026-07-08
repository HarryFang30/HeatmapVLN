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
#   - Frozen executor: InternNav cond_projector + NextDiT
#   - Adapter: PanoLatentSpaceAdapter（hidden_dim=256 时约 1.84M params）
#   - 默认先用本机 GPU_DEVICES 并行离线生成 native InternNav teacher sidecar，再训练 adapter
#   - 全程只训练 adapter

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

# 单机 torchrun：平台常注入不可解析的 pod 主机名，会导致 c10d socket 警告
if [[ "${WORLD_SIZE}" == "1" && "${RANK}" == "0" ]]; then
  export MASTER_ADDR="127.0.0.1"
fi

# ---------------------------------------------------------------------------
# Conda：激活环境固定为 /mnt/afs/lixiaoou/intern/fjl/envs/qwen25
# 与 run_stage1_s2_8gpu_mxc500_launcher.sh 一致；集群镜像常见 /opt/conda
# ---------------------------------------------------------------------------
QWEN25_ENV="/mnt/afs/lixiaoou/intern/fjl/envs/qwen25"

activate_qwen25_via_path() {
  if [[ ! -x "${QWEN25_ENV}/bin/python" ]]; then
    return 1
  fi
  export PATH="${QWEN25_ENV}/bin:${PATH}"
  export CONDA_PREFIX="${QWEN25_ENV}"
  export CONDA_DEFAULT_ENV="qwen25"
  hash -r
  echo "[launcher] 已通过 PATH 激活环境: ${QWEN25_ENV} (python=$(command -v python))"
  return 0
}

_CONDA_SH=""
if [[ -n "${CONDA_INIT_SH:-}" && -f "${CONDA_INIT_SH}" ]]; then
  _CONDA_SH="${CONDA_INIT_SH}"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  _CONDA_SH="/opt/conda/etc/profile.d/conda.sh"
elif [[ -f "/mnt/afs/lixiaoou/intern/fjl/miniconda3/etc/profile.d/conda.sh" ]]; then
  _CONDA_SH="/mnt/afs/lixiaoou/intern/fjl/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  _CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

if [[ -n "${_CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${_CONDA_SH}"
  conda activate "${QWEN25_ENV}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${QWEN25_ENV}"
elif activate_qwen25_via_path; then
  :
else
  echo "未找到 conda 且 ${QWEN25_ENV}/bin/python 不可用。" >&2
  echo "请 export CONDA_INIT_SH=/path/to/conda.sh，或确认 qwen25 环境路径正确。" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 路径默认值（请按你机器上实际位置修改）
# ---------------------------------------------------------------------------
export INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-/mnt/afs/lixiaoou/intern/fjl/InternNav-Model}"
export INTERNNAV_REPO="${INTERNNAV_REPO:-/mnt/afs/lixiaoou/intern/fjl/InternNav}"
export INTERNNAV_BACKBONE="${INTERNNAV_BACKBONE:-$INTERNNAV_MODEL_PATH}"
export PANORAMIC_DATA_ROOT="${PANORAMIC_DATA_ROOT:-/mnt/afs/lixiaoou/intern/fjl/r2r_paronamic_data}"

# Stage1-S2 checkpoint (student weights)
export STAGE2_ADAPTER_LOAD_WEIGHTS="${STAGE2_ADAPTER_LOAD_WEIGHTS:-/mnt/afs/lixiaoou/intern/fjl/model/output_stage1_s2_full_11000_rank32_alllayer_from_heatmap/run_20260701_212615/checkpoints/epoch_005.pth}"
export STAGE1_S2_OUT_DIR="${STAGE1_S2_OUT_DIR:-/mnt/afs/lixiaoou/intern/fjl/model/output_stage1_s2_full_11000_rank32_alllayer_from_heatmap}"

# Output
export STAGE2_ADAPTER_OUT_DIR="${STAGE2_ADAPTER_OUT_DIR:-/mnt/afs/lixiaoou/intern/fjl/model/output_stage2_adapter_full_11000_alllora}"

mkdir -p "$REPO_ROOT/logs" "${STAGE2_ADAPTER_OUT_DIR}"
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
export STAGE2_ADAPTER_DATASET_MAX_CLIPS="${STAGE2_ADAPTER_DATASET_MAX_CLIPS:-0}"
export STAGE2_ADAPTER_PREFETCH_BATCHES="${STAGE2_ADAPTER_PREFETCH_BATCHES:-}"
export STAGE2_ADAPTER_PREFETCH_WORKERS="${STAGE2_ADAPTER_PREFETCH_WORKERS:-}"
export STAGE2_ADAPTER_TEACHER_CACHE_MODE="${STAGE2_ADAPTER_TEACHER_CACHE_MODE:-}"
export STAGE2_ADAPTER_TEACHER_CACHE_MAX_ITEMS="${STAGE2_ADAPTER_TEACHER_CACHE_MAX_ITEMS:-}"
export STAGE2_ADAPTER_TEACHER_PRELOAD_CACHE="${STAGE2_ADAPTER_TEACHER_PRELOAD_CACHE:-}"
export STAGE2_ADAPTER_TEACHER_PRELOAD_WORKERS="${STAGE2_ADAPTER_TEACHER_PRELOAD_WORKERS:-}"
export STAGE2_ADAPTER_CHECK_TEACHER_TENSOR_FILES="${STAGE2_ADAPTER_CHECK_TEACHER_TENSOR_FILES:-}"
export STAGE2_ADAPTER_TRUST_NATIVE_PANO_LABELS="${STAGE2_ADAPTER_TRUST_NATIVE_PANO_LABELS:-}"

# Teacher targets: native InternNav sidecar by default; teacher model is not
# loaded during adapter training.
export STAGE2_ADAPTER_TEACHER_MODE="${STAGE2_ADAPTER_TEACHER_MODE:-native_sidecar}"
export STAGE2_ADAPTER_COMPUTE_TEACHER_MSE="${STAGE2_ADAPTER_COMPUTE_TEACHER_MSE:-0}"
export STAGE2_ADAPTER_TEACHER_DTYPE="${STAGE2_ADAPTER_TEACHER_DTYPE:-bfloat16}"
export STAGE2_ADAPTER_TEACHER_ATTN="${STAGE2_ADAPTER_TEACHER_ATTN:-sdpa}"
export STAGE2_ADAPTER_RAW_WEIGHT="${STAGE2_ADAPTER_RAW_WEIGHT:-0.1}"
export STAGE2_ADAPTER_COND_WEIGHT="${STAGE2_ADAPTER_COND_WEIGHT:-1.0}"
export STAGE2_ADAPTER_GT_WEIGHT="${STAGE2_ADAPTER_GT_WEIGHT:-0.2}"

# Native teacher sidecar collection.  This is not a new Habitat dataset; it is
# a cache of InternNav native front/lookdown teacher latents for the existing
# trajectory dataset, aligned later by (clip_idx,current_t).
export STAGE2_TEACHER_SIDECAR_DIR="${STAGE2_TEACHER_SIDECAR_DIR:-/mnt/afs/lixiaoou/intern/fjl/teacher_sidecars/stage2_native_dataset}"
export STAGE2_TEACHER_TENSOR_DIR="${STAGE2_TEACHER_TENSOR_DIR:-${STAGE2_TEACHER_SIDECAR_DIR}/tensors}"
export STAGE2_ADAPTER_TEACHER_JSONL="${STAGE2_ADAPTER_TEACHER_JSONL:-${STAGE2_TEACHER_SIDECAR_DIR}/train_native_teacher.jsonl}"
export STAGE2_TEACHER_COLLECT_CONFIG="${STAGE2_TEACHER_COLLECT_CONFIG:-configs/train_pano_adapter_stage2_8gpu.yaml}"
export STAGE2_TEACHER_COLLECT_SPLIT="${STAGE2_TEACHER_COLLECT_SPLIT:-train}"
export STAGE2_TEACHER_COLLECT_GPU="${STAGE2_TEACHER_COLLECT_GPU:-${GPU_DEVICES%%,*}}"
export STAGE2_TEACHER_COLLECT_GPU_DEVICES="${STAGE2_TEACHER_COLLECT_GPU_DEVICES:-$GPU_DEVICES}"
export STAGE2_TEACHER_COLLECT_NPROC="${STAGE2_TEACHER_COLLECT_NPROC:-}"
export STAGE2_TEACHER_COLLECT_NUM_SAMPLES="${STAGE2_TEACHER_COLLECT_NUM_SAMPLES:-0}"
export STAGE2_TEACHER_COLLECT_SAMPLE_STRIDE="${STAGE2_TEACHER_COLLECT_SAMPLE_STRIDE:-1}"
export STAGE2_TEACHER_COLLECT_CLIP_LEVEL_SAMPLING="${STAGE2_TEACHER_COLLECT_CLIP_LEVEL_SAMPLING:-0}"
export STAGE2_TEACHER_COLLECT_SAMPLES_PER_CLIP="${STAGE2_TEACHER_COLLECT_SAMPLES_PER_CLIP:-0}"
export STAGE2_TEACHER_COLLECT_PIXEL_GOAL_DIRECTION="${STAGE2_TEACHER_COLLECT_PIXEL_GOAL_DIRECTION:-front_down}"
export STAGE2_TEACHER_COLLECT_PROGRESS_INTERVAL="${STAGE2_TEACHER_COLLECT_PROGRESS_INTERVAL:-100}"
export STAGE2_TEACHER_COLLECT_PROGRESS_STYLE="${STAGE2_TEACHER_COLLECT_PROGRESS_STYLE:-tqdm}"
export STAGE2_TEACHER_COLLECT_TQDM_MININTERVAL="${STAGE2_TEACHER_COLLECT_TQDM_MININTERVAL:-5.0}"
export STAGE2_TEACHER_COLLECT_NUM_SAMPLE_TRAJS="${STAGE2_TEACHER_COLLECT_NUM_SAMPLE_TRAJS:-32}"
export STAGE2_TEACHER_COLLECT_NUM_INFERENCE_STEPS="${STAGE2_TEACHER_COLLECT_NUM_INFERENCE_STEPS:-10}"
export STAGE2_TEACHER_COLLECT_GUIDANCE_SCALE="${STAGE2_TEACHER_COLLECT_GUIDANCE_SCALE:-1.0}"
export STAGE2_TEACHER_COLLECT_TRAJ_IMAGE_SIZE="${STAGE2_TEACHER_COLLECT_TRAJ_IMAGE_SIZE:-224}"
export STAGE2_TEACHER_COLLECT_ENABLE="${STAGE2_TEACHER_COLLECT_ENABLE:-1}"
export STAGE2_TEACHER_FORCE_RECOLLECT="${STAGE2_TEACHER_FORCE_RECOLLECT:-0}"
export STAGE2_TEACHER_INCREMENTAL_COLLECT="${STAGE2_TEACHER_INCREMENTAL_COLLECT:-1}"
export STAGE2_TEACHER_TENSOR_PATH_MODE="${STAGE2_TEACHER_TENSOR_PATH_MODE:-stable_key}"
export STAGE2_TEACHER_COLLECT_LOG_FILE="${STAGE2_TEACHER_COLLECT_LOG_FILE:-$REPO_ROOT/logs/stage2_native_teacher_sidecar_collect.log}"
export STAGE2_TEACHER_COLLECT_SHARD_DIR="${STAGE2_TEACHER_COLLECT_SHARD_DIR:-${STAGE2_ADAPTER_TEACHER_JSONL}.shards}"
export STAGE2_TEACHER_WAIT_TIMEOUT_S="${STAGE2_TEACHER_WAIT_TIMEOUT_S:-172800}"

# FlashAttention2：默认强制开启（与 MXC500 环境一致）
export STAGE2_ADAPTER_REQUIRE_FLASH_ATTN="${STAGE2_ADAPTER_REQUIRE_FLASH_ATTN:-1}"

# Data — controlled by adapter config YAML. Override via
# STAGE2_ADAPTER_CONFIG=my_config.yaml, not env vars.

is_truthy_launcher() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

dataset_signature() {
  python - "$PANORAMIC_DATA_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser()
parts = []
stats_path = root / "collection_stats.json"
if stats_path.is_file():
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    for key in ("successful", "failed", "total_frames"):
        if key in stats:
            parts.append(f"{key}={stats[key]}")
try:
    parts.append(f"meta_count={sum(1 for _ in root.rglob('meta.json'))}")
except Exception:
    pass
print(",".join(parts) if parts else "unknown")
PY
}

sidecar_signature() {
  printf 'root=%s|split=%s|config=%s|model=%s|repo=%s|dataset=%s|coord_source=dataset|sample_mode=pixel|num_samples=%s|sample_stride=%s|clip_level_sampling=%s|samples_per_clip=%s|pixel_goal_direction=%s|num_sample_trajs=%s|num_inference_steps=%s|guidance=%s|traj_image_size=%s|tensor_path_mode=%s\n' \
    "$PANORAMIC_DATA_ROOT" \
    "$STAGE2_TEACHER_COLLECT_SPLIT" \
    "$STAGE2_TEACHER_COLLECT_CONFIG" \
    "$INTERNNAV_MODEL_PATH" \
    "$INTERNNAV_REPO" \
    "$(dataset_signature)" \
    "$STAGE2_TEACHER_COLLECT_NUM_SAMPLES" \
    "$STAGE2_TEACHER_COLLECT_SAMPLE_STRIDE" \
    "$STAGE2_TEACHER_COLLECT_CLIP_LEVEL_SAMPLING" \
    "$STAGE2_TEACHER_COLLECT_SAMPLES_PER_CLIP" \
    "$STAGE2_TEACHER_COLLECT_PIXEL_GOAL_DIRECTION" \
    "$STAGE2_TEACHER_COLLECT_NUM_SAMPLE_TRAJS" \
    "$STAGE2_TEACHER_COLLECT_NUM_INFERENCE_STEPS" \
    "$STAGE2_TEACHER_COLLECT_GUIDANCE_SCALE" \
    "$STAGE2_TEACHER_COLLECT_TRAJ_IMAGE_SIZE" \
    "$STAGE2_TEACHER_TENSOR_PATH_MODE"
}

parse_teacher_collect_gpus() {
  local raw="${1:-}"
  local -a split_gpus=()
  local gpu
  IFS=',' read -r -a split_gpus <<< "$raw"
  for gpu in "${split_gpus[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    if [[ -n "$gpu" ]]; then
      printf '%s\n' "$gpu"
    fi
  done
}

prime_incremental_shards_from_master() {
  local collect_nproc="$1"
  if ! is_truthy_launcher "$STAGE2_TEACHER_INCREMENTAL_COLLECT"; then
    return 0
  fi
  if [[ ! -s "$STAGE2_ADAPTER_TEACHER_JSONL" ]]; then
    return 0
  fi
  if [[ "$collect_nproc" -lt 1 ]]; then
    return 0
  fi

  local existing_shard=""
  if [[ -d "$STAGE2_TEACHER_COLLECT_SHARD_DIR" ]]; then
    existing_shard="$(find "$STAGE2_TEACHER_COLLECT_SHARD_DIR" -maxdepth 1 -name 'shard_*.jsonl' -size +0c -print -quit 2>/dev/null || true)"
  fi
  if [[ -n "$existing_shard" ]]; then
    return 0
  fi

  echo "[launcher] 未找到旧 shard JSONL，从 master sidecar 拆分以便增量复用"
  mkdir -p "$STAGE2_TEACHER_COLLECT_SHARD_DIR"
  python - "$STAGE2_ADAPTER_TEACHER_JSONL" "$STAGE2_TEACHER_COLLECT_SHARD_DIR" "$collect_nproc" <<'PY'
import json
import sys
from pathlib import Path

master = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
nproc = int(sys.argv[3])
handles = {}
count = 0
try:
    with master.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                idx = rec.get("dataset_index")
                if idx is None:
                    continue
                shard = int(idx) % nproc
            except Exception:
                continue
            if shard not in handles:
                path = shard_dir / f"shard_{shard:02d}.jsonl"
                handles[shard] = path.open("a", encoding="utf-8")
            handles[shard].write(line)
            count += 1
finally:
    for handle in handles.values():
        handle.close()
print(f"[launcher] primed {count} records into {len(handles)} shard files", flush=True)
PY
}

ensure_native_teacher_sidecar() {
  if [[ "$STAGE2_ADAPTER_TEACHER_MODE" != "native_sidecar" ]]; then
    return 0
  fi

  local marker="${STAGE2_ADAPTER_TEACHER_JSONL}.done"
  local meta="${STAGE2_ADAPTER_TEACHER_JSONL}.meta"
  local collecting_meta="${STAGE2_ADAPTER_TEACHER_JSONL}.collecting.meta"

  if ! is_truthy_launcher "$STAGE2_TEACHER_COLLECT_ENABLE"; then
    if [[ ! -s "$STAGE2_ADAPTER_TEACHER_JSONL" ]]; then
      echo "STAGE2_TEACHER_COLLECT_ENABLE=0 but sidecar JSONL is missing or empty: $STAGE2_ADAPTER_TEACHER_JSONL" >&2
      exit 1
    fi
    echo "[launcher] 跳过 teacher sidecar 采集，使用已有: $STAGE2_ADAPTER_TEACHER_JSONL"
    return 0
  fi

  local current_sig
  current_sig="$(sidecar_signature)"

  if [[ -s "$STAGE2_ADAPTER_TEACHER_JSONL" && -f "$marker" && -f "$meta" ]] \
    && [[ "$(cat "$meta")" == "$current_sig" ]] \
    && ! is_truthy_launcher "$STAGE2_TEACHER_FORCE_RECOLLECT"; then
    echo "[launcher] native teacher sidecar 已存在，跳过采集: $STAGE2_ADAPTER_TEACHER_JSONL"
    return 0
  fi

  if [[ ! -f "$STAGE2_TEACHER_COLLECT_CONFIG" ]]; then
    echo "Teacher collect config not found: $STAGE2_TEACHER_COLLECT_CONFIG" >&2
    exit 1
  fi
  if [[ ! -d "$PANORAMIC_DATA_ROOT" ]]; then
    echo "PANORAMIC_DATA_ROOT not found: $PANORAMIC_DATA_ROOT" >&2
    exit 1
  fi
  if [[ ! -d "$INTERNNAV_MODEL_PATH" ]]; then
    echo "INTERNNAV_MODEL_PATH not found: $INTERNNAV_MODEL_PATH" >&2
    exit 1
  fi
  if [[ ! -d "$INTERNNAV_REPO" ]]; then
    echo "INTERNNAV_REPO not found: $INTERNNAV_REPO" >&2
    exit 1
  fi

  mkdir -p "$STAGE2_TEACHER_SIDECAR_DIR" "$STAGE2_TEACHER_TENSOR_DIR" "$(dirname "$STAGE2_TEACHER_COLLECT_LOG_FILE")"

  if is_truthy_launcher "$STAGE2_TEACHER_FORCE_RECOLLECT"; then
    echo "[launcher] STAGE2_TEACHER_FORCE_RECOLLECT=1，清理旧 sidecar/shard JSONL 后重新采集"
    rm -f "$STAGE2_ADAPTER_TEACHER_JSONL" "$marker" "$meta" "$collecting_meta"
    rm -rf "$STAGE2_TEACHER_COLLECT_SHARD_DIR"
  elif [[ -s "$STAGE2_ADAPTER_TEACHER_JSONL" && ( ! -f "$marker" || ! -f "$meta" ) ]]; then
    if is_truthy_launcher "$STAGE2_TEACHER_INCREMENTAL_COLLECT"; then
      echo "[launcher] sidecar 缺少 .done/.meta，保留旧记录并增量恢复采集"
      rm -f "$marker" "$collecting_meta"
    else
      echo "[launcher] sidecar 缺少 .done/.meta，清理旧 sidecar/shard JSONL 后重新采集"
      rm -f "$STAGE2_ADAPTER_TEACHER_JSONL" "$marker" "$meta" "$collecting_meta"
      rm -rf "$STAGE2_TEACHER_COLLECT_SHARD_DIR"
    fi
  elif [[ -f "$meta" && "$(cat "$meta")" != "$current_sig" ]]; then
    if is_truthy_launcher "$STAGE2_TEACHER_INCREMENTAL_COLLECT"; then
      echo "[launcher] sidecar meta 与当前参数/数据集不一致，保留旧记录并增量补齐"
      rm -f "$marker" "$collecting_meta"
    else
      echo "[launcher] sidecar meta 与当前参数不一致，清理旧 sidecar/shard JSONL 后重新采集"
      rm -f "$STAGE2_ADAPTER_TEACHER_JSONL" "$marker" "$meta" "$collecting_meta"
      rm -rf "$STAGE2_TEACHER_COLLECT_SHARD_DIR"
    fi
  elif [[ -f "$collecting_meta" && "$(cat "$collecting_meta")" != "$current_sig" ]]; then
    if is_truthy_launcher "$STAGE2_TEACHER_INCREMENTAL_COLLECT"; then
      echo "[launcher] 未完成采集的参数与当前参数不一致，保留旧记录并增量恢复"
      rm -f "$marker" "$collecting_meta"
    else
      echo "[launcher] 未完成采集的参数与当前参数不一致，清理旧 shard JSONL 后重新采集"
      rm -f "$STAGE2_ADAPTER_TEACHER_JSONL" "$marker" "$collecting_meta"
      rm -rf "$STAGE2_TEACHER_COLLECT_SHARD_DIR"
    fi
  fi

  local -a collect_gpus=()
  mapfile -t collect_gpus < <(parse_teacher_collect_gpus "$STAGE2_TEACHER_COLLECT_GPU_DEVICES")
  if [[ "${#collect_gpus[@]}" -eq 0 ]]; then
    collect_gpus=("$STAGE2_TEACHER_COLLECT_GPU")
  fi

  local collect_nproc
  if [[ -n "$STAGE2_TEACHER_COLLECT_NPROC" ]]; then
    collect_nproc="$STAGE2_TEACHER_COLLECT_NPROC"
  else
    collect_nproc="${#collect_gpus[@]}"
  fi
  if [[ "$collect_nproc" -lt 1 ]]; then
    echo "STAGE2_TEACHER_COLLECT_NPROC must be >= 1, got: $collect_nproc" >&2
    exit 1
  fi
  if [[ "$collect_nproc" -gt "${#collect_gpus[@]}" ]]; then
    echo "STAGE2_TEACHER_COLLECT_NPROC=$collect_nproc exceeds GPU count in STAGE2_TEACHER_COLLECT_GPU_DEVICES=${STAGE2_TEACHER_COLLECT_GPU_DEVICES}" >&2
    exit 1
  fi

  prime_incremental_shards_from_master "$collect_nproc"
  mkdir -p "$STAGE2_TEACHER_COLLECT_SHARD_DIR"
  printf '%s' "$current_sig" > "$collecting_meta"

  echo "[launcher] 开始/恢复 native InternNav teacher sidecar 采集"
  echo "[launcher]   data_root=$PANORAMIC_DATA_ROOT"
  echo "[launcher]   output=$STAGE2_ADAPTER_TEACHER_JSONL"
  echo "[launcher]   tensors=$STAGE2_TEACHER_TENSOR_DIR"
  echo "[launcher]   shard_dir=$STAGE2_TEACHER_COLLECT_SHARD_DIR"
  echo "[launcher]   collect_config=$STAGE2_TEACHER_COLLECT_CONFIG"
  echo "[launcher]   sample_stride=$STAGE2_TEACHER_COLLECT_SAMPLE_STRIDE clip_level_sampling=$STAGE2_TEACHER_COLLECT_CLIP_LEVEL_SAMPLING samples_per_clip=$STAGE2_TEACHER_COLLECT_SAMPLES_PER_CLIP pixel_goal_direction=$STAGE2_TEACHER_COLLECT_PIXEL_GOAL_DIRECTION"
  echo "[launcher]   incremental=$STAGE2_TEACHER_INCREMENTAL_COLLECT tensor_path_mode=$STAGE2_TEACHER_TENSOR_PATH_MODE"
  echo "[launcher]   gpus=${collect_gpus[*]:0:$collect_nproc} nproc=$collect_nproc"

  local shard_num_samples="$STAGE2_TEACHER_COLLECT_NUM_SAMPLES"
  if [[ "$STAGE2_TEACHER_COLLECT_NUM_SAMPLES" -gt 0 && "$collect_nproc" -gt 1 ]]; then
    shard_num_samples=$(( (STAGE2_TEACHER_COLLECT_NUM_SAMPLES + collect_nproc - 1) / collect_nproc ))
    echo "[launcher]   num_samples=$STAGE2_TEACHER_COLLECT_NUM_SAMPLES total target; per-shard cap=$shard_num_samples"
  fi

  local -a pids=()
  local -a shard_outputs=()
  local shard_idx gpu shard_output shard_log
  for shard_idx in $(seq 0 $((collect_nproc - 1))); do
    gpu="${collect_gpus[$shard_idx]}"
    shard_output="${STAGE2_TEACHER_COLLECT_SHARD_DIR}/shard_$(printf '%02d' "$shard_idx").jsonl"
    shard_log="${STAGE2_TEACHER_COLLECT_SHARD_DIR}/shard_$(printf '%02d' "$shard_idx").log"
    shard_outputs+=("$shard_output")
    echo "[launcher]   shard $shard_idx/$collect_nproc -> GPU $gpu output=$shard_output log=$shard_log"

    local -a collect_args=(
      --config "$STAGE2_TEACHER_COLLECT_CONFIG"
      --root "$PANORAMIC_DATA_ROOT"
      --split "$STAGE2_TEACHER_COLLECT_SPLIT"
      --output "$shard_output"
      --tensor-output-dir "$STAGE2_TEACHER_TENSOR_DIR"
      --tensor-path-mode "$STAGE2_TEACHER_TENSOR_PATH_MODE"
      --internnav-repo "$INTERNNAV_REPO"
      --model-path "$INTERNNAV_MODEL_PATH"
      --device cuda:0
      --coord-source dataset
      --sample-mode pixel
      --num-samples "$shard_num_samples"
      --num-shards "$collect_nproc"
      --shard-index "$shard_idx"
      --sample-stride "$STAGE2_TEACHER_COLLECT_SAMPLE_STRIDE"
      --pixel-goal-direction "$STAGE2_TEACHER_COLLECT_PIXEL_GOAL_DIRECTION"
      --traj-image-size "$STAGE2_TEACHER_COLLECT_TRAJ_IMAGE_SIZE"
      --num-sample-trajs "$STAGE2_TEACHER_COLLECT_NUM_SAMPLE_TRAJS"
      --num-inference-steps "$STAGE2_TEACHER_COLLECT_NUM_INFERENCE_STEPS"
      --guidance-scale "$STAGE2_TEACHER_COLLECT_GUIDANCE_SCALE"
      --save-traj-latents
      --save-traj-latents-768
      --save-dp-actions
      --progress-interval "$STAGE2_TEACHER_COLLECT_PROGRESS_INTERVAL"
      --progress-style "$STAGE2_TEACHER_COLLECT_PROGRESS_STYLE"
      --tqdm-mininterval "$STAGE2_TEACHER_COLLECT_TQDM_MININTERVAL"
    )
    if is_truthy_launcher "$STAGE2_TEACHER_INCREMENTAL_COLLECT" && [[ -s "$STAGE2_ADAPTER_TEACHER_JSONL" ]]; then
      collect_args+=(--resume-from-jsonl "$STAGE2_ADAPTER_TEACHER_JSONL")
    fi
    if is_truthy_launcher "$STAGE2_TEACHER_COLLECT_CLIP_LEVEL_SAMPLING"; then
      collect_args+=(--clip-level-sampling)
    else
      collect_args+=(--no-clip-level-sampling)
    fi
    if [[ "$STAGE2_TEACHER_COLLECT_SAMPLES_PER_CLIP" -gt 0 ]]; then
      collect_args+=(--samples-per-clip "$STAGE2_TEACHER_COLLECT_SAMPLES_PER_CLIP")
    fi

    CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/evaluation/collect_internnav_teacher_sidecar.py \
      "${collect_args[@]}" \
      > "$shard_log" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for shard_idx in "${!pids[@]}"; do
    if wait "${pids[$shard_idx]}"; then
      echo "[launcher] shard $shard_idx finished"
    else
      echo "[launcher] shard $shard_idx failed; see ${STAGE2_TEACHER_COLLECT_SHARD_DIR}/shard_$(printf '%02d' "$shard_idx").log" >&2
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "Teacher sidecar collection failed. Fix the failed shard logs above, then rerun; completed shards will resume." >&2
    exit 1
  fi

  local tmp_output="${STAGE2_ADAPTER_TEACHER_JSONL}.tmp.$$"
  : > "$tmp_output"
  for shard_output in "${shard_outputs[@]}"; do
    if [[ -f "$shard_output" ]]; then
      cat "$shard_output" >> "$tmp_output"
    fi
  done
  mv "$tmp_output" "$STAGE2_ADAPTER_TEACHER_JSONL"

  if [[ ! -s "$STAGE2_ADAPTER_TEACHER_JSONL" ]]; then
    echo "Teacher sidecar collection finished but JSONL is missing or empty: $STAGE2_ADAPTER_TEACHER_JSONL" >&2
    exit 1
  fi
  printf '%s' "$current_sig" > "$meta"
  rm -f "$collecting_meta"
  touch "$marker"
  echo "[launcher] native teacher sidecar 已就绪: $STAGE2_ADAPTER_TEACHER_JSONL"
}

if [[ "$RANK" == "0" ]]; then
  ensure_native_teacher_sidecar
else
  echo "[launcher] RANK=$RANK 等待 rank0 生成 native teacher sidecar: $STAGE2_ADAPTER_TEACHER_JSONL"
  _wait_iters=$(( (STAGE2_TEACHER_WAIT_TIMEOUT_S + 29) / 30 ))
  for _ in $(seq 1 "$_wait_iters"); do
    if [[ -s "$STAGE2_ADAPTER_TEACHER_JSONL" && -f "${STAGE2_ADAPTER_TEACHER_JSONL}.done" ]]; then
      break
    fi
    sleep 30
  done
  if [[ ! -s "$STAGE2_ADAPTER_TEACHER_JSONL" ]]; then
    echo "Timed out waiting for teacher sidecar: $STAGE2_ADAPTER_TEACHER_JSONL" >&2
    exit 1
  fi
fi

bash scripts/run_stage2_pano_adapter_8gpu.sh 2>&1 | tee "$LOG_FILE"

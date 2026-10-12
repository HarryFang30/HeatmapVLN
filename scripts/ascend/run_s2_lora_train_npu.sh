#!/usr/bin/env bash
# EXP-23 component C on the 910B: System2 LoRA SFT, the G0 parity check, or the
# offline gates, one torchrun process per card over HCCL (exp23-s2-dagger-design.md
# §2 C and §3).  Everything comes from the environment (CLAUDE.md §1.1):
#
#   S2_LORA_RECORD_ROOTS=/abs/r1:/abs/r2 S2_LORA_LABEL_ROOTS=/abs/l1:/abs/l2 \
#   S2_LORA_OUTPUT_DIR=$HOME/work/zhr/zhr_1/exp23/round1/lora \
#   S2_LORA_NPU_DEVICES=0,1,2,3,4,5,6,7 bash scripts/ascend/run_s2_lora_train_npu.sh
#
#   S2_LORA_STREAM_DIR  train from the shard buffer the 4090 pushes into (design §4)
#                       instead of records; then no record/label roots are needed
#   S2_LORA_MODE        train (default) | parity | eval
#   S2_LORA_PARITY_N    calls the G0 parity check replays (parity; default 2000)
#   S2_LORA_ADAPTER     adapter to gate (eval; default $S2_LORA_OUTPUT_DIR/adapter)
#   S2_LORA_EXTRA_ARGS  passed to the python entry point verbatim, e.g.
#                       "--max-steps 20 --limit-samples 64 --save-every 0"
#
# This box is shared and the user's rule is that nothing is written outside
# $S2_LORA_ROOT (default ~/work/zhr/zhr_1): the output directory must be inside it,
# and TMPDIR is a fresh short directory under $S2_LORA_ROOT/tmp, removed on exit.

set -Eeuo pipefail

ROOT="${S2_LORA_ROOT:-$HOME/work/zhr/zhr_1}"
REPO="${S2_LORA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RPC_ROOT="${S2_LORA_RPC_ROOT:-$ROOT/rpc}"
PYTHON="${S2_LORA_PYTHON:-$ROOT/envs/ppa/bin/python}"
INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-$ROOT/InternNav_Model}"
ASCEND_ENV="${S2_LORA_ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
NPU_HBM_AWK="$REPO/scripts/ascend/npu_hbm_used_mib.awk"

MODE="${S2_LORA_MODE:-train}"
NPU_CSV="${S2_LORA_NPU_DEVICES:-0,1,2,3,4,5,6,7}"
# A card counts as free below this many MiB of HBM in use: an idle 910B3 already
# shows about 3.4 GB (deploy_ascend_910b.md §1).
MAX_USED_MIB="${S2_LORA_MAX_USED_MIB:-4096}"
RECORD_ROOTS="${S2_LORA_RECORD_ROOTS:-}"
LABEL_ROOTS="${S2_LORA_LABEL_ROOTS:-}"
STREAM_DIR="${S2_LORA_STREAM_DIR:-}"
OUTPUT_DIR="${S2_LORA_OUTPUT_DIR:-}"
PARITY_N="${S2_LORA_PARITY_N:-2000}"
ADAPTER="${S2_LORA_ADAPTER:-}"
read -r -a EXTRA_ARGS <<< "${S2_LORA_EXTRA_ARGS:-}"

TMP_DIR=""
die() { printf '[s2-lora] ERROR: %s\n' "$*" >&2; exit 2; }
cleanup() {
  local status=$?
  trap - EXIT
  [[ -n "$TMP_DIR" && -d "$TMP_DIR" ]] && rm -rf -- "$TMP_DIR"
  printf '[s2-lora] exit status=%s\n' "$status"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

inside_root() {
  # Resolved path of $1, refused unless it is $ROOT or below it.
  local resolved
  resolved="$(realpath -m -- "$1")"
  [[ "$resolved" == "$ROOT_REAL" || "$resolved" == "$ROOT_REAL"/* ]] \
    || die "$1 resolves to $resolved, outside $ROOT_REAL; this box only takes writes under it"
  printf '%s' "$resolved"
}

[[ -d "$ROOT" ]] || die "S2_LORA_ROOT does not exist: $ROOT"
ROOT_REAL="$(realpath -- "$ROOT")"
[[ "$MODE" =~ ^(train|parity|eval)$ ]] || die "S2_LORA_MODE must be train, parity or eval"
if [[ -n "$STREAM_DIR" ]]; then
  [[ "$MODE" == "train" ]] || die "S2_LORA_STREAM_DIR is for training; parity and eval read records"
  STREAM_DIR="$(inside_root "$STREAM_DIR")"
  [[ -s "$STREAM_DIR/stream_manifest.json" ]] || die "no stream_manifest.json in $STREAM_DIR (push it first)"
else
  [[ -n "$RECORD_ROOTS" ]] || die "S2_LORA_RECORD_ROOTS is empty (colon-separated record roots)"
fi
[[ -n "$OUTPUT_DIR" ]] || die "S2_LORA_OUTPUT_DIR is empty"
OUTPUT_DIR="$(inside_root "$OUTPUT_DIR")"
[[ "$PARITY_N" =~ ^[1-9][0-9]*$ ]] || die "S2_LORA_PARITY_N must be a positive integer"
[[ "$MAX_USED_MIB" =~ ^(0|[1-9][0-9]*)$ ]] || die "S2_LORA_MAX_USED_MIB must be a plain non-negative integer"
[[ -x "$PYTHON" ]] || die "missing python: $PYTHON"
[[ -d "$INTERNNAV_MODEL_PATH" ]] || die "missing InternNav model directory: $INTERNNAV_MODEL_PATH"
[[ -d "$RPC_ROOT/src/vla_rpc" ]] || die "missing vla_rpc under $RPC_ROOT/src (the server's JPEG decoder)"
for file in "$REPO/scripts/train_system2_lora.py" "$REPO/scripts/tools/eval_system2_lora_offline.py" "$NPU_HBM_AWK"; do
  [[ -s "$file" ]] || die "missing file: $file"
done

declare -a DATA_ARGS=()
if [[ -n "$STREAM_DIR" ]]; then
  DATA_ARGS+=(--stream-dir "$STREAM_DIR")
  RECORD_ROOTS=""
fi
IFS=':' read -r -a RECORD_LIST <<< "$RECORD_ROOTS"
for root in "${RECORD_LIST[@]}"; do
  [[ -d "$root" ]] || die "record root does not exist: $root"
  DATA_ARGS+=(--record-root "$root")
done
if [[ -n "$LABEL_ROOTS" ]]; then
  IFS=':' read -r -a LABEL_LIST <<< "$LABEL_ROOTS"
  (( ${#LABEL_LIST[@]} == 1 || ${#LABEL_LIST[@]} == ${#RECORD_LIST[@]} )) \
    || die "give one label root per record root, or a single one"
  for root in "${LABEL_LIST[@]}"; do
    [[ -d "$root" ]] || die "label root does not exist: $root"
    DATA_ARGS+=(--label-root "$root")
  done
elif [[ "$MODE" != "parity" && -z "$STREAM_DIR" ]]; then
  die "S2_LORA_LABEL_ROOTS is empty; only the parity check runs without labels"
fi

# A non-interactive shell does not necessarily load CANN; without it torch_npu sees
# no device at all (deploy_ascend_910b.md §4).
[[ -r "$ASCEND_ENV" ]] || die "CANN env script not readable: $ASCEND_ENV (set S2_LORA_ASCEND_ENV)"
# shellcheck disable=SC1090
source "$ASCEND_ENV"
[[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]] || die "ASCEND_RT_VISIBLE_DEVICES is already set; unset it"
command -v npu-smi >/dev/null || die "npu-smi not found"

IFS=',' read -r -a NPUS <<< "$NPU_CSV"
NPROC="${#NPUS[@]}"
(( NPROC >= 1 && NPROC <= 8 )) || die "S2_LORA_NPU_DEVICES must hold 1-8 ids"
for npu in "${NPUS[@]}"; do
  [[ "$npu" =~ ^(0|[1-9][0-9]?)$ ]] || die "invalid NPU id: '$npu' (plain decimal, no leading zeros)"
done
[[ "$(printf '%s\n' "${NPUS[@]}" | sort -u | wc -l | tr -d ' ')" -eq "$NPROC" ]] || die "NPU ids must be unique"

# The cards are shared: only take free ones.  The awk parser prints nothing rather
# than a wrong number, and nothing is a refusal here.
NPU_SMI_TABLE="$(npu-smi info 2>&1)" || die "npu-smi info failed: $NPU_SMI_TABLE"
for npu in "${NPUS[@]}"; do
  used="$(printf '%s\n' "$NPU_SMI_TABLE" | awk -v id="$npu" -f "$NPU_HBM_AWK")"
  if [[ -z "$used" ]]; then
    printf '%s\n' "$NPU_SMI_TABLE" >&2
    die "could not read HBM use of NPU $npu from the npu-smi table above"
  fi
  (( used <= MAX_USED_MIB )) || die "NPU $npu already has ${used} MiB in use (limit $MAX_USED_MIB); pick another card"
done

mkdir -p "$OUTPUT_DIR/logs" "$ROOT_REAL/tmp"
# Short and unique: CANN binds an AF_UNIX socket under TMPDIR (108-byte limit,
# deploy_ascend_910b.md §7 ①), and two runs must never share one.
TMP_DIR="$(mktemp -d "$ROOT_REAL/tmp/s2l.XXXXXX")" || die "could not create a TMPDIR under $ROOT_REAL/tmp"
(( ${#TMP_DIR} + 45 <= 108 )) || die "TMPDIR $TMP_DIR is too long for an AF_UNIX socket"
mkdir -p "$TMP_DIR"/{xdg,hf,mpl}

export TMPDIR="$TMP_DIR" XDG_CACHE_HOME="$TMP_DIR/xdg" HF_HOME="$TMP_DIR/hf" MPLCONFIGDIR="$TMP_DIR/mpl"
export PYTHONPATH="$RPC_ROOT/src:$REPO"
export ASCEND_RT_VISIBLE_DEVICES="$NPU_CSV"
unset CUDA_VISIBLE_DEVICES http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export USE_TF=0 TRANSFORMERS_NO_TF=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Keeps the caching allocator from fragmenting the card so CANN can still get its
# own workspace (deploy_ascend_910b.md §4, §8).
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
# Rank 0 indexes every record on the NFS share while the others wait at a barrier.
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
# Pin the tree: the trainer refuses to start if the code moved after this line.
HEATMAPVLN_SOURCE_FINGERPRINT="$("$PYTHON" "$REPO/scripts/tools/source_fingerprint.py" "$REPO")"
export HEATMAPVLN_SOURCE_FINGERPRINT

REPO_COMMIT="$(git -c safe.directory="$REPO" -C "$REPO" rev-parse HEAD 2>/dev/null || echo none)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$OUTPUT_DIR/logs/${MODE}_${STAMP}.log"

declare -a ENTRY
case "$MODE" in
  train)
    ENTRY=("$REPO/scripts/train_system2_lora.py" "${DATA_ARGS[@]}" --output-dir "$OUTPUT_DIR")
    ;;
  parity)
    ENTRY=("$REPO/scripts/train_system2_lora.py" "${DATA_ARGS[@]}" --output-dir "$OUTPUT_DIR"
           --parity-check "$PARITY_N")
    if [[ -n "$ADAPTER" ]]; then
      ENTRY+=(--adapter "$ADAPTER")
    fi
    ;;
  eval)
    ADAPTER="${ADAPTER:-$OUTPUT_DIR/adapter}"
    [[ -s "$ADAPTER/adapter_model.safetensors" ]] || die "no adapter at $ADAPTER"
    ENTRY=("$REPO/scripts/tools/eval_system2_lora_offline.py" "${DATA_ARGS[@]}"
           --adapter "$ADAPTER" --output "$OUTPUT_DIR/gates_${STAMP}.json")
    ;;
esac

{
  echo "[s2-lora] mode=$MODE npus=$NPU_CSV nproc=$NPROC repo=$REPO commit=${REPO_COMMIT:0:12}"
  echo "[s2-lora] fingerprint=$HEATMAPVLN_SOURCE_FINGERPRINT model=$INTERNNAV_MODEL_PATH"
  echo "[s2-lora] output=$OUTPUT_DIR tmp=$TMP_DIR alloc=$PYTORCH_NPU_ALLOC_CONF"
} | tee -a "$LOG"

# --standalone: a private rendezvous on a free local port, so two runs on this box
# cannot join each other.
"$PYTHON" -m torch.distributed.run --standalone --nnodes 1 --nproc_per_node "$NPROC" \
  "${ENTRY[@]}" --device npu --model-path "$INTERNNAV_MODEL_PATH" "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "$LOG"

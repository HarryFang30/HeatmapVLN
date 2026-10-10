#!/usr/bin/env bash
# Start the PPA model and AMB3R-VO RPC servers on Ascend NPUs and keep them running.
#
# This is the server half of the split deployment: the Habitat clients, Xvfb and the
# locked evaluation plan stay on the x86 CUDA box and reach these servers over an SSH
# tunnel, so nothing in the client path changes (docs/ops/deploy_ascend_910b.md).
# There is no Habitat, no dataset and no client here.
#
# Both servers bind 127.0.0.1 only.  The gRPC channels have no authentication at
# either end, so they must never be reachable from outside this host; the tunnel is
# what carries them to the client box.
#
# One slot = one (model server, VO server) pair.  They share a card by default, and
# that is the measured ceiling rather than a comfortable fit: under load the pair
# reserves about 62.2 of the card's 65.5 GB (model peak_reserved 41.8 GB, VO 17.9 GB,
# about 3.3 GB resident per card), so nothing else fits beside it.  The idle reading
# of 26.5 GB is not the deployment footprint; see docs/ops/deploy_ascend_910b.md
# section 8.  PLACEMENT_BUDGET below refuses the layouts that cannot fit.
#
#   PPA_NPU_DEVICES=0,1 bash scripts/ascend/run_ppa_servers_npu.sh
#
# Stop it with SIGINT/SIGTERM; the trap stops every server it started.

set -Eeuo pipefail

ROOT="${PPA_EVAL_ROOT:-$HOME/work/zhr/zhr_1}"
REPO="${PPA_EVAL_REPO:-$ROOT/HeatmapVLN}"
RPC_ROOT="${PPA_EVAL_RPC_ROOT:-$ROOT/rpc}"
INTERNNAV_MODEL_PATH="${INTERNNAV_MODEL_PATH:-$ROOT/InternNav_Model}"
AMB3R_ROOT="${PPA_EVAL_AMB3R_ROOT:-$ROOT/amb3r}"
DA3_CHECKPOINT="${PPA_EVAL_DA3_CHECKPOINT:-$AMB3R_ROOT/checkpoints/DA3NESTED-GIANT-LARGE}"
PYTHON="${PPA_EVAL_PYTHON:-$ROOT/envs/ppa/bin/python}"

PPA_CHECKPOINT="${PPA_EVAL_CHECKPOINT:-$ROOT/weights/ppa_refine_v2_best.pth}"
PPA_CONFIG="${PPA_EVAL_CONFIG:-$REPO/configs/ppa_action_refine_v2_8gpu.yaml}"

MODEL_SERVER="$REPO/scripts/evaluation/rpc_model_server.py"
VO_SERVER="$REPO/scripts/amb3r_vo/rpc_amb3r_vo_server.py"
NPU_HBM_AWK="$REPO/scripts/ascend/npu_hbm_used_mib.awk"

# Logical NPU ids, one per slot.  Each server process sees only its own card, as
# device npu:0, so a wrong id cannot silently co-locate two slots on one card.
NPU_CSV="${PPA_NPU_DEVICES:-0}"
VO_NPU_CSV="${PPA_NPU_VO_DEVICES:-$NPU_CSV}"
# A card counts as free below this many MiB of HBM in use.  The cards are shared.
MAX_USED_MIB="${PPA_NPU_MAX_USED_MIB:-4096}"
THREADS_PER_SERVER="${PPA_NPU_THREADS_PER_SERVER:-8}"
ASCEND_ENV="${PPA_NPU_ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

# Same names as the CUDA launcher, so one set of exports drives both halves.
TIMING="${PPA_EVAL_TIMING:-0}"
BRIDGE_OFF="${PPA_EVAL_BRIDGE_OFF:-0}"
NUM_SAMPLE_TRAJS="${PPA_EVAL_NUM_SAMPLE_TRAJS:-}"
NUM_INFERENCE_STEPS="${PPA_EVAL_NUM_INFERENCE_STEPS:-}"
# Re-seed the VO device RNG at every episode.  Required for run-to-run reproducible
# poses on Ascend, where the default device generator seed is not a constant.
VO_RNG_SEED="${PPA_NPU_VO_RNG_SEED:-0}"

MODEL_PORT_BASE="${PPA_EVAL_MODEL_PORT_BASE:-52400}"
VO_PORT_BASE="${PPA_EVAL_VO_PORT_BASE:-52500}"
SERVER_START_TIMEOUT_S="${PPA_EVAL_SERVER_START_TIMEOUT_S:-3600}"
SERVER_STAGGER_S="${PPA_EVAL_SERVER_STAGGER_S:-20}"

RUN_STAMP="${PPA_EVAL_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_$$}"
RUNTIME_DIR="${PPA_NPU_RUNTIME_DIR:-$ROOT/servers/$RUN_STAMP}"

# A token this run's servers report through GetServerInfo, recorded in servers.json.
# The client checks it live, which is the only way it can tell these servers from a
# local CUDA pair: the remote slots use 52400+k / 52500+k, the same ports the 4090's
# own certified launcher defaults to, and model_version is built from a path
# component, so two boxes with the same layout are otherwise indistinguishable.
SERVER_INSTANCE="${PPA_NPU_SERVER_INSTANCE:-$RUN_STAMP-$(od -An -N8 -tx1 /dev/urandom | tr -d ' \n')}"

declare -a NPUS VO_NPUS MODEL_PIDS VO_PIDS TMP_DIRS RETIRED

die() { printf '[ppa-npu] ERROR: %s\n' "$*" >&2; exit 2; }
require_file() { [[ -s "$1" ]] || die "missing file: $1"; }
require_dir() { [[ -d "$1" ]] || die "missing directory: $1"; }
tcp_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

stop_pid() {
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  # Orphaned servers would hold their cards and their ports, so always stop them.
  for pid in "${MODEL_PIDS[@]:-}"; do stop_pid "$pid"; done
  for pid in "${VO_PIDS[@]:-}"; do stop_pid "$pid"; done
  # Only the directories this run created, and only after its servers are gone: the
  # CANN kernel bank and a Manager socket live in there while they serve.
  for directory in "${TMP_DIRS[@]:-}"; do
    [[ -n "$directory" ]] && rm -rf -- "$directory" 2>/dev/null || true
  done
  printf '[ppa-npu] stopped (status=%s)\n' "$status"
  [[ -d "$RUNTIME_DIR" ]] && date +%Y-%m-%dT%H:%M:%S%z > "$RUNTIME_DIR/STOPPED"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# A non-interactive shell does not necessarily load CANN; without it torch_npu
# cannot see a device at all (the MACA equivalent is in CLAUDE.md section 3.1).
[[ -r "$ASCEND_ENV" ]] || die "CANN env script not readable: $ASCEND_ENV (set PPA_NPU_ASCEND_ENV)"
# shellcheck disable=SC1090
source "$ASCEND_ENV"
[[ -n "${ASCEND_TOOLKIT_HOME:-}${ASCEND_HOME_PATH:-}" ]] || die "sourcing $ASCEND_ENV left ASCEND_TOOLKIT_HOME empty"

# An inherited ASCEND_RT_VISIBLE_DEVICES would remap the per-slot ids below.
[[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]] || die "ASCEND_RT_VISIBLE_DEVICES is already set; unset it"
unset CUDA_VISIBLE_DEVICES
# The eval is offline; a proxy left in the shell only breaks local connections.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

for file in "$PPA_CHECKPOINT" "$PPA_CONFIG" "$MODEL_SERVER" "$VO_SERVER" \
  "$DA3_CHECKPOINT/model.safetensors" "$AMB3R_ROOT/slam/slam_config.yaml" \
  "$NPU_HBM_AWK" "$REPO/scripts/ascend/check_amb3r_patch.sh" "$REPO/scripts/ascend/amb3r_npu.patch"; do
  require_file "$file"
done
# The locked plan is a client-side input: it holds the cohorts, the dataset shards
# and the merge tool, none of which either server imports.  It stays on the client box.
for directory in "$REPO" "$RPC_ROOT/src/vla_rpc" "$INTERNNAV_MODEL_PATH"; do
  require_dir "$directory"
done
[[ -x "$PYTHON" ]] || die "missing executable: $PYTHON"
command -v npu-smi >/dev/null || die "npu-smi not found"

IFS=',' read -r -a NPUS <<< "$NPU_CSV"
IFS=',' read -r -a VO_NPUS <<< "$VO_NPU_CSV"
NUM_SLOTS="${#NPUS[@]}"
(( NUM_SLOTS >= 1 && NUM_SLOTS <= 8 )) || die "PPA_NPU_DEVICES must hold 1-8 ids"
[[ "${#VO_NPUS[@]}" -eq "$NUM_SLOTS" ]] || die "PPA_NPU_VO_DEVICES must have one id per slot"
# Canonical digits before anything compares or counts ids: "0" and "00" are the same
# card, but different strings, so sort -u would accept them as two and every later
# check -- uniqueness, the per-card budget, the free-card read -- would be about a
# card that is not the one the server lands on.
for npu in "${NPUS[@]}" "${VO_NPUS[@]}"; do
  [[ "$npu" =~ ^(0|[1-9][0-9]?)$ ]] || die "invalid NPU id: '$npu' (plain decimal, no leading zeros)"
done
[[ "$(printf '%s\n' "${NPUS[@]}" | sort -u | wc -l | tr -d ' ')" -eq "$NUM_SLOTS" ]] || die "NPU ids must be unique"
[[ "$MAX_USED_MIB" =~ ^(0|[1-9][0-9]*)$ ]] \
  || die "PPA_NPU_MAX_USED_MIB must be a non-negative integer without leading zeros (bash reads 0800 as octal)"
[[ "$TIMING" =~ ^[01]$ ]] || die "PPA_EVAL_TIMING must be 0 or 1"
[[ "$BRIDGE_OFF" =~ ^[01]$ ]] || die "PPA_EVAL_BRIDGE_OFF must be 0 or 1"
[[ -z "$NUM_SAMPLE_TRAJS" || "$NUM_SAMPLE_TRAJS" =~ ^[1-9][0-9]*$ ]] || die "PPA_EVAL_NUM_SAMPLE_TRAJS must be a positive integer"
[[ -z "$NUM_INFERENCE_STEPS" || "$NUM_INFERENCE_STEPS" =~ ^[1-9][0-9]*$ ]] || die "PPA_EVAL_NUM_INFERENCE_STEPS must be a positive integer"
[[ "$VO_RNG_SEED" =~ ^[0-9]+$ ]] || die "PPA_NPU_VO_RNG_SEED must be a non-negative integer"

# What will actually sit on each card, checked before anything launches.  The
# free-card read below cannot see this: it runs before this run's own servers exist,
# so it only ever notices other people's memory.  A duplicated VO id used to pass
# every gate and then OOM hours into a run.  Budget from section 8 of the deploy doc:
# about 3.3 GB resident per card, model peak_reserved 41.8 GB, VO 17.9 GB, 65.5 GB total.
for card in $(printf '%s\n' "${NPUS[@]}" "${VO_NPUS[@]}" | sort -nu); do
  models=0 vos=0
  for npu in "${NPUS[@]}"; do [[ "$npu" == "$card" ]] && models=$((models + 1)); done
  for npu in "${VO_NPUS[@]}"; do [[ "$npu" == "$card" ]] && vos=$((vos + 1)); done
  if (( models >= 1 && vos >= 2 )); then
    die "card $card would hold a model server and $vos VO servers: about $((42 + vos * 18 + 3)) GB against 65.5 GB (deploy_ascend_910b.md section 8)"
  fi
  if (( models == 0 && vos >= 4 )); then
    die "card $card would hold $vos VO servers: about $((vos * 18 + 3)) GB against 65.5 GB (deploy_ascend_910b.md section 8)"
  fi
  if (( models == 0 && vos >= 2 )); then
    printf '[ppa-npu] WARNING: card %s takes %s VO servers (about %s GB); that layout has never been measured\n' \
      "$card" "$vos" "$((vos * 18 + 3))" >&2
  fi
done

# Latency work only: an NPU profile of a few requests, written here, while the
# server keeps serving.  Never set for a run whose numbers are reported -- profiling
# perturbs exactly what it measures -- so it is deliberately not part of MODEL_EXTRA's
# comparability contract and the launcher says so out loud.
PROFILE_DIR="${PPA_NPU_PROFILE_DIR:-}"
PROFILE_SKIP="${PPA_NPU_PROFILE_SKIP:-1}"
PROFILE_CALLS="${PPA_NPU_PROFILE_CALLS:-2}"
declare -a MODEL_PROFILE=()
if [[ -n "$PROFILE_DIR" ]]; then
  [[ "$PROFILE_SKIP" =~ ^(0|[1-9][0-9]*)$ ]] || die "PPA_NPU_PROFILE_SKIP must be a non-negative integer"
  [[ "$PROFILE_CALLS" =~ ^[1-9][0-9]*$ ]] || die "PPA_NPU_PROFILE_CALLS must be a positive integer"
  MODEL_PROFILE=(--profile_dir "$PROFILE_DIR" --profile_skip "$PROFILE_SKIP" --profile_calls "$PROFILE_CALLS")
  printf '[ppa-npu] WARNING: profiling the model server into %s (skip %s, calls %s). This perturbs latency: do not report numbers from this run.\n' \
    "$PROFILE_DIR" "$PROFILE_SKIP" "$PROFILE_CALLS" >&2
fi

declare -a MODEL_EXTRA=()
[[ "$BRIDGE_OFF" -eq 1 ]] && MODEL_EXTRA=(--ppa_bridge_off)
[[ -n "$NUM_SAMPLE_TRAJS" ]] && MODEL_EXTRA+=(--nextdit_num_sample_trajs "$NUM_SAMPLE_TRAJS")
[[ -n "$NUM_INFERENCE_STEPS" ]] && MODEL_EXTRA+=(--nextdit_num_inference_steps "$NUM_INFERENCE_STEPS")

# Placeholders the train config expands on load; the eval never reads them.
PLACEHOLDER_DIR="$RUNTIME_DIR/config_placeholders"
export PPA_DATA_ROOT="$PLACEHOLDER_DIR" PPA_AMB3R_CACHE_ROOT="$PLACEHOLDER_DIR"
export PPA_STAGE2_OUTPUT_ROOT="$PLACEHOLDER_DIR" PPA_TENSORBOARD_ROOT="$PLACEHOLDER_DIR"
export PPA_ACTION_REFINE_OUTPUT_ROOT="$PLACEHOLDER_DIR"
export INTERNNAV_MODEL_PATH HEATMAPVLN_INTERNNAV_MODEL_PATH="$INTERNNAV_MODEL_PATH"
export HEATMAPVLN_FJL_ROOT="$ROOT"
export USE_TF=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Keep the DA3 code path the certified numbers came from: the reference SwiGLU and
# the query-chunked SDPA.  Both must be set here too, or the forward silently takes
# a different path (and unchunked attention materialises a ~20 GiB score matrix).
export DA3_DISABLE_XFORMERS=1
export DA3_SDPA_QUERY_CHUNK_SIZE=256
export PYTHONDONTWRITEBYTECODE=1
export HEATMAPVLN_TIMING="$TIMING"
# The vision-tower pass reuse in generate_latents.  The counter and the window-mask
# patch install themselves on NPU, but the reuse is opt-in in
# src/models/qwen2_5_vl_vision_count.py (``_on`` wants exactly "1"), so without this
# line nothing turns it on and the -2749 ms on the heaviest call is simply not there.
# EXP-22's 设置 records it as on by default and refusable with =0, which is what this
# makes true, and setting it here also makes it symmetric across passes: the value is
# in this log and the same for every slot, rather than in whoever's shell started the
# servers.  =0 still refuses it.
export HEATMAPVLN_QWEN_VISION_REUSE="${HEATMAPVLN_QWEN_VISION_REUSE:-1}"
echo "[ppa-npu] qwen_vision_reuse=$HEATMAPVLN_QWEN_VISION_REUSE mask_patch=${HEATMAPVLN_QWEN_VISION_MASK_PATCH:-<on for npu>} pass_count=${HEATMAPVLN_QWEN_VISION_COUNT:-<on for npu>}"
# torch_npu's async dispatch queue.  Deliberately not set here: the platform default
# (on) measured 7.0-7.7% FASTER per plan call than TASK_QUEUE_ENABLE=0, in every call
# class, over two passes each on two separately started servers, with all four
# pairwise comparisons identical field by field.  The 2026-10-09 canary carried =0 to
# work around the AICPU timeout; it is a non-default, uncertified switch and it costs
# speed, so a run that still sets it should be visible in this log.
if [[ "${TASK_QUEUE_ENABLE:-}" == "0" ]]; then
  printf '[ppa-npu] WARN TASK_QUEUE_ENABLE=0 is set: about 7%% slower per plan call and not a certified setting (ascend_910b_open_problems.md)\n' >&2
fi
echo "[ppa-npu] task_queue_enable=${TASK_QUEUE_ENABLE:-<platform default>}"

RPC_PYTHONPATH="$RPC_ROOT/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RUNTIME_DIR/logs" "$PLACEHOLDER_DIR"

# TMPDIR has to stay short, must not live under the timestamped runtime directory,
# and must be unique per run.  CANN initialises its kernel bank through
# multiprocessing.Manager, whose AF_UNIX socket path is TMPDIR plus about 32 bytes
# of "/pymp-XXXXXXXX/listener-XXXXXXXX".  AF_UNIX allows 108 bytes in total, and a
# path under $RUNTIME_DIR/slot_N/model/tmp crosses it: the model server then died
# with "AF_UNIX path too long" surfacing as an unrelated-looking ACL_PRECISION_MODE /
# GEInitialize failure.
#
# Local /tmp rather than the share: it is short, it is per instance, and mktemp makes
# each run's directory unique without anyone reasoning about who else is alive.  The
# previous naming was $ROOT/tmp/m<slot>, which ignored the run, so two "slot 0" runs
# shared one TMPDIR on NFS -- they collided (Errno 39 on exit) and leaked a dozen
# pymp-* directories that nothing ever cleaned up.  Point PPA_NPU_TMP_ROOT at a short
# path on the share if /tmp on some future instance is too small for the kernel bank.
TMP_ROOT="${PPA_NPU_TMP_ROOT:-/tmp}"
mkdir -p "$TMP_ROOT"
# Absolute, or the AF_UNIX length check below would measure a relative string while
# the socket is bound under the full path.
TMP_ROOT="$(cd "$TMP_ROOT" && pwd)" || die "PPA_NPU_TMP_ROOT is not usable: $TMP_ROOT"
AF_UNIX_MAX=108
AF_UNIX_RESERVE=45

npu_used_mib() {
  # HBM-Usage(MB) in use on one chip, parsed by scripts/ascend/npu_hbm_used_mib.awk
  # (which prints nothing rather than a wrong number; see the header there).  Every
  # card shows a few GB in use even when idle -- about 3.4 GB on this driver -- so
  # the threshold has to sit above that, not at zero.
  local id="$1" used
  used="$(printf '%s\n' "$NPU_SMI_TABLE" | awk -v id="$id" -f "$NPU_HBM_AWK")"
  printf '%s' "${used:-unknown}"
}

echo "[ppa-npu] slots=$NUM_SLOTS npus=$NPU_CSV vo_npus=$VO_NPU_CSV timing=$TIMING bridge_off=$BRIDGE_OFF"
echo "[ppa-npu] num_sample_trajs=${NUM_SAMPLE_TRAJS:-config} num_inference_steps=${NUM_INFERENCE_STEPS:-config} vo_rng_seed=$VO_RNG_SEED"
echo "[ppa-npu] checkpoint=$PPA_CHECKPOINT config=$PPA_CONFIG"
echo "[ppa-npu] runtime=$RUNTIME_DIR instance=$SERVER_INSTANCE"

# Which build is about to serve.  Read here, before anything starts: the client
# refuses a server set whose commit it cannot identify, and finding that out after
# two 16 GB model loads is an expensive way to learn it.  A tar-extracted copy of
# the repo has no git metadata, so it is refused here rather than silently recorded
# as "unknown" -- an evaluation whose serving code cannot be named is not one anyone
# can reproduce.
REPO_COMMIT="$(git -c safe.directory="$REPO" -C "$REPO" rev-parse HEAD 2>/dev/null)" \
  || die "could not read the repo commit of $REPO (not a git checkout?); the client refuses a server set it cannot identify, so serve from a clone"
REPO_DIRTY=0
git -c safe.directory="$REPO" -C "$REPO" diff --quiet HEAD -- || REPO_DIRTY=1
echo "[ppa-npu] repo=$REPO commit=${REPO_COMMIT:0:12} dirty=$REPO_DIRTY"

# AMB3R is third party and its two NPU fixes are applied by hand, so a clean clone
# and a patched tree are indistinguishable to every other check here.  Both of them
# fail silently (fp32 mapping, fp16 DA3), so this runs before anything is started.
bash "$REPO/scripts/ascend/check_amb3r_patch.sh" "$AMB3R_ROOT" \
  "$REPO/scripts/ascend/amb3r_npu.patch" || die "AMB3R tree verification failed"

# The cards are shared with other users, like the 4090 box: only take free ones.
# One table for every id, so the refusal can show what it read.
NPU_SMI_TABLE="$(npu-smi info 2>&1)" || die "npu-smi info failed: $NPU_SMI_TABLE"
for npu in "${NPUS[@]}" "${VO_NPUS[@]}"; do
  used="$(npu_used_mib "$npu")"
  if [[ "$used" == "unknown" ]]; then
    printf '%s\n' "$NPU_SMI_TABLE" >&2
    die "could not read HBM use of NPU $npu from the npu-smi table above (layout changed, or that id is not present)"
  fi
  (( used <= MAX_USED_MIB )) || die "NPU $npu already has ${used} MiB in use (limit $MAX_USED_MIB); pick another card"
  echo "[ppa-npu] npu=$npu used=${used}MiB free enough"
done

# Platform preflight: everything the servers silently depend on, checked once, loudly.
PYTHONPATH="$RPC_PYTHONPATH" "$PYTHON" - <<'PY' || die "platform preflight failed"
import json
import sys

import numpy
import torch
import torch_npu  # noqa: F401
import transformers

problems = []
if transformers.__version__ != "4.51.0":
    problems.append(f"transformers {transformers.__version__} != 4.51.0 (runtime_compat gate)")
# numpy 2 broke the ABI of the numpy-1 extensions in this environment, including
# cv2 and the ones CANN's own op-compile toolchain imports.  The symptom was not an
# import error at startup but a CANN kernel-bank failure on the first request, so
# check the version and that cv2 really loads.
if not numpy.__version__.startswith("1."):
    problems.append(f"numpy {numpy.__version__} is not 1.x; the certified stack is 1.26.4")
try:
    import cv2
except Exception as exc:  # noqa: BLE001 - any failure here is fatal and worth printing
    problems.append(f"cv2 does not import ({type(exc).__name__}: {exc})")
    cv2 = None
if not torch.npu.is_available():
    problems.append("torch.npu.is_available() is False")
elif not torch.npu.is_bf16_supported():
    problems.append("torch.npu.is_bf16_supported() is False; the reference dtype is bf16")
print(json.dumps({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "transformers": transformers.__version__,
    "numpy": numpy.__version__,
    "cv2": getattr(cv2, "__version__", None),
    "npu_count": torch.npu.device_count() if torch.npu.is_available() else 0,
}, sort_keys=True))
if problems:
    print("PREFLIGHT: " + "; ".join(problems), file=sys.stderr)
    raise SystemExit(1)
PY

for slot in $(seq 0 $((NUM_SLOTS - 1))); do
  model_port=$((MODEL_PORT_BASE + slot))
  vo_port=$((VO_PORT_BASE + slot))
  # A live port here means a stale server or another run; starting anyway would
  # leave the client talking to whichever process won the bind.
  tcp_open "$model_port" && die "port $model_port is already in use"
  tcp_open "$vo_port" && die "port $vo_port is already in use"

  runtime="$RUNTIME_DIR/slot_${slot}"
  mkdir -p "$runtime"/model/{xdg,hf,matplotlib} "$runtime"/vo/{xdg,hf}
  # Recorded one at a time: if the second mktemp fails we still have to remove the
  # first, and the trap only removes what this array holds.
  model_tmp="$(mktemp -d "$TMP_ROOT/ppa-m${slot}.XXXXXX")" || die "could not create a TMPDIR under $TMP_ROOT"
  TMP_DIRS+=("$model_tmp")
  vo_tmp="$(mktemp -d "$TMP_ROOT/ppa-v${slot}.XXXXXX")" || die "could not create a TMPDIR under $TMP_ROOT"
  TMP_DIRS+=("$vo_tmp")
  for directory in "$model_tmp" "$vo_tmp"; do
    (( ${#directory} + AF_UNIX_RESERVE <= AF_UNIX_MAX )) \
      || die "TMPDIR $directory is too long for an AF_UNIX socket (${#directory} + $AF_UNIX_RESERVE > $AF_UNIX_MAX); use a shorter PPA_NPU_TMP_ROOT"
  done
  # Each server sees exactly one card, as npu:0.
  env PYTHONPATH="$RPC_PYTHONPATH" ASCEND_RT_VISIBLE_DEVICES="${NPUS[$slot]}" \
    OMP_NUM_THREADS="$THREADS_PER_SERVER" MKL_NUM_THREADS="$THREADS_PER_SERVER" \
    TMPDIR="$model_tmp" XDG_CACHE_HOME="$runtime/model/xdg" HF_HOME="$runtime/model/hf" \
    MPLCONFIGDIR="$runtime/model/matplotlib" HEATMAPVLN_FORCE_FLASH_ATTN_STUB=0 \
    "$PYTHON" -u "$MODEL_SERVER" \
      --config "$PPA_CONFIG" --checkpoint "$PPA_CHECKPOINT" \
      --internnav_model_path "$INTERNNAV_MODEL_PATH" \
      --device npu --gpu_id 0 --host 127.0.0.1 --port "$model_port" --workers 1 \
      --server_instance "$SERVER_INSTANCE/slot$slot" \
      --require_deterministic_sampling --require_ppa_online_amb3r \
      "${MODEL_EXTRA[@]}" "${MODEL_PROFILE[@]}" \
      --log_level INFO >"$RUNTIME_DIR/logs/model_${slot}.log" 2>&1 &
  MODEL_PIDS[$slot]="$!"
  env PYTHONPATH="$AMB3R_ROOT:$AMB3R_ROOT/thirdparty:$RPC_PYTHONPATH" \
    ASCEND_RT_VISIBLE_DEVICES="${VO_NPUS[$slot]}" \
    OMP_NUM_THREADS="$THREADS_PER_SERVER" MKL_NUM_THREADS="$THREADS_PER_SERVER" \
    TMPDIR="$vo_tmp" XDG_CACHE_HOME="$runtime/vo/xdg" HF_HOME="$runtime/vo/hf" \
    "$PYTHON" -u "$VO_SERVER" \
      --repo "$REPO" --amb3r-root "$AMB3R_ROOT" \
      --da3-checkpoint "$DA3_CHECKPOINT" --device npu:0 \
      --host 127.0.0.1 --port "$vo_port" \
      --map-init-window 20 --map-every 8 --max-history 8 \
      --resolution 518 392 --translation-scale 1.0 \
      --max-frames-limit 4096 --max-message-mb 32 \
      --rng-seed "$VO_RNG_SEED" \
      --server-instance "$SERVER_INSTANCE/slot$slot" \
      --log-level INFO >"$RUNTIME_DIR/logs/vo_${slot}.log" 2>&1 &
  VO_PIDS[$slot]="$!"
  sleep "$SERVER_STAGGER_S"
done

# Both servers answer, are healthy, speak their protocol, and carry THIS run's
# instance token.  Prints the model server's model_version, which goes into
# servers.json so the client can check the same string against the live servers.
rpc_ready() {
  PYTHONPATH="$RPC_PYTHONPATH" "$PYTHON" - "$1" "$2" "$3" <<'PY'
import sys
from vla_rpc.client import VLAClient

model_address, vo_address, token = sys.argv[1:4]
model_version = None
for address, expected in ((model_address, "ppa-online-amb3r-v1"), (vo_address, "json+jpeg")):
    client = VLAClient(server_addr=address, timeout_ms=5000)
    try:
        client.connect()
        info = client.get_server_info()
        if not client.health_check() or info is None:
            raise SystemExit(1)
        formats = set(info.supported_formats)
        if expected not in formats:
            raise SystemExit(2)
        if f"heatmapvln-instance:{token}" not in formats:
            raise SystemExit(3)
        if address == model_address:
            model_version = info.model_version
    finally:
        client.close()
print(model_version or "")
PY
}

echo "[ppa-npu] waiting for $((2 * NUM_SLOTS)) RPC servers"
deadline=$(( $(date +%s) + SERVER_START_TIMEOUT_S ))
declare -a MODEL_VERSIONS=()
for slot in $(seq 0 $((NUM_SLOTS - 1))); do
  model_addr="127.0.0.1:$((MODEL_PORT_BASE + slot))"
  vo_addr="127.0.0.1:$((VO_PORT_BASE + slot))"
  while true; do
    kill -0 "${MODEL_PIDS[$slot]}" 2>/dev/null || { tail -120 "$RUNTIME_DIR/logs/model_${slot}.log" >&2; die "model server slot $slot exited"; }
    kill -0 "${VO_PIDS[$slot]}" 2>/dev/null || { tail -120 "$RUNTIME_DIR/logs/vo_${slot}.log" >&2; die "VO server slot $slot exited"; }
    if model_version="$(rpc_ready "$model_addr" "$vo_addr" "$SERVER_INSTANCE/slot$slot" 2>/dev/null)"; then
      MODEL_VERSIONS[$slot]="$model_version"
      break
    fi
    (( $(date +%s) < deadline )) || die "RPC startup timeout at slot $slot"
    sleep 10
  done
  # Same preflight evidence the CUDA launcher greps for, plus the device lines that
  # only this platform can get wrong.
  grep -F "Formal PPA online AMB3R runtime enabled" "$RUNTIME_DIR/logs/model_${slot}.log" >/dev/null \
    || die "model slot $slot lacks PPA preflight evidence"
  grep -F "Model server device: npu:0" "$RUNTIME_DIR/logs/model_${slot}.log" >/dev/null \
    || die "model slot $slot did not report an NPU device"
  # Which of the three vision settings this process actually came up with, in its own
  # words.  The counter prints this after it has wrapped the tower's forward, so the
  # line proves the wrapper went in (it refuses a transformers version or a forward
  # signature it was not written against) and reports the reuse state it read from the
  # environment.  It does not prove a reuse ever hit; the served responses'
  # vision_tower counts do that.
  grep -F "vision tower passes counted per request; reuse $(
    [[ "$HEATMAPVLN_QWEN_VISION_REUSE" == "1" ]] && echo on || echo off
  ) (HEATMAPVLN_QWEN_VISION_REUSE)" "$RUNTIME_DIR/logs/model_${slot}.log" >/dev/null \
    || die "model slot $slot did not report the vision-tower pass counter with reuse=$HEATMAPVLN_QWEN_VISION_REUSE"
  # That line is printed when the counter wraps the tower, which is before the model
  # is built; refuse_reuse_on_adapted_tower can still turn the reuse off afterwards,
  # when the built tower turns out to carry adapters that the pixels do not key.  Then
  # the line above says "reuse on" and the server serves with it off.  Running without
  # the reuse is safe -- it is the original computation -- but it is not the
  # configuration EXP-22 names, so say so instead of serving a third thing.
  if [[ "$HEATMAPVLN_QWEN_VISION_REUSE" == "1" ]]; then
    grep -F "vision tower reuse disabled" "$RUNTIME_DIR/logs/model_${slot}.log" >/dev/null \
      && die "model slot $slot turned the vision-tower reuse off at model load (the tower carries adapters); set HEATMAPVLN_QWEN_VISION_REUSE=0 to run without it on purpose"
  fi
  grep -F "VO server device: npu:0" "$RUNTIME_DIR/logs/vo_${slot}.log" >/dev/null \
    || die "VO slot $slot did not report an NPU device"
  # What DA3's own parser makes of the chunk size, read out of the module that
  # decides whether to chunk.  The line this replaced grepped for
  # "DA3_SDPA_QUERY_CHUNK_SIZE=256", which the VO server echoed straight back from
  # the environment this script had just exported: it could not fail, whatever DA3
  # did with the variable, and the deploy doc quoted it as proof that the certified
  # chunked path was taken.  This one can fail -- it comes from DA3 -- but note what
  # it still does not say: that any particular forward chunked.  That depends on the
  # query length, and the dtype of a forward is covered by the patch check instead.
  # One grep over the whole line, so a "memory_bounded=True" occurring anywhere else
  # in the log cannot stand in for the chunk size DA3 reported.
  grep -F "DA3 attention: query_chunk=$DA3_SDPA_QUERY_CHUNK_SIZE (parsed by DA3), memory_bounded=True" \
    "$RUNTIME_DIR/logs/vo_${slot}.log" >/dev/null \
    || die "VO slot $slot: DA3 does not report the configured query chunk with memory-bounded attention bound (unpatched or unexpected AMB3R tree?)"
  if [[ "$BRIDGE_OFF" -eq 1 ]]; then
    grep -F "PPA bridge off (EXP-20 A1)" "$RUNTIME_DIR/logs/model_${slot}.log" >/dev/null \
      || die "model slot $slot lacks bridge-off evidence"
  fi
  for key in num_sample_trajs num_inference_steps; do
    if { [[ "$key" == num_sample_trajs && -n "$NUM_SAMPLE_TRAJS" ]] || [[ "$key" == num_inference_steps && -n "$NUM_INFERENCE_STEPS" ]]; }; then
      grep -F "Sensitivity override (EXP-21): nextdit.$key" "$RUNTIME_DIR/logs/model_${slot}.log" >/dev/null \
        || die "model slot $slot lacks the $key override evidence"
    fi
  done
  # Every slot must be serving the same build, or the shards are not one arm.
  if [[ "${MODEL_VERSIONS[$slot]}" != "${MODEL_VERSIONS[0]}" ]]; then
    die "slot $slot reports model_version '${MODEL_VERSIONS[$slot]}' but slot 0 reports '${MODEL_VERSIONS[0]}'"
  fi
  echo "[ppa-npu] slot=$slot npu=${NPUS[$slot]} vo_npu=${VO_NPUS[$slot]} model=$model_addr vo=$vo_addr ready"
done

# What the client box needs in order to point its tunnel at these servers, to check
# that the servers it reaches are these ones, and to record which build served its
# episodes.  The ports alone cannot identify them: 52400+k / 52500+k are also the
# defaults of the 4090's own certified launcher, so a local CUDA pair left listening
# would answer every check the client had before the instance token.
"$PYTHON" - "$RUNTIME_DIR/servers.json" "$NUM_SLOTS" "$MODEL_PORT_BASE" "$VO_PORT_BASE" \
  "$NPU_CSV" "$VO_NPU_CSV" "$REPO_COMMIT" \
  "$VO_RNG_SEED" "$TIMING" "$BRIDGE_OFF" "$SERVER_INSTANCE" "${MODEL_VERSIONS[0]}" \
  "$REPO_DIRTY" "$NUM_SAMPLE_TRAJS" "$NUM_INFERENCE_STEPS" <<'PY'
import json
import sys

(path, slots, model_base, vo_base, npus, vo_npus, commit, rng_seed, timing, bridge_off,
 instance, model_version, dirty, num_sample_trajs, num_inference_steps) = sys.argv[1:16]
record = {
    "schema": "heatmapvln-npu-servers-v2",
    "slots": int(slots),
    "model_ports": [int(model_base) + i for i in range(int(slots))],
    "vo_ports": [int(vo_base) + i for i in range(int(slots))],
    "npus": npus,
    "vo_npus": vo_npus,
    "repo_commit": commit,
    "repo_dirty": int(dirty),
    "vo_rng_seed": int(rng_seed),
    "timing": int(timing),
    "bridge_off": int(bridge_off),
    # Per slot the servers advertise "<server_instance>/slot<k>" in supported_formats.
    "server_instance": instance,
    "device": "npu",
    "model_version": model_version,
    "num_sample_trajs": num_sample_trajs,
    "num_inference_steps": num_inference_steps,
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
print(json.dumps(record, sort_keys=True))
PY

echo "[ppa-npu] all servers ready; servers.json=$RUNTIME_DIR/servers.json"
echo "[ppa-npu] logs=$RUNTIME_DIR/logs"
echo "[ppa-npu] holding; SIGINT/SIGTERM stops every server"

# Hold the servers.  Two ways a slot stops being usable, and they look nothing alike:
#
#   - the process dies.  Then its port closes and the client notices.
#   - the process lives and its device does not.  An Ascend device error does not
#     kill the process: a VO server that took an AICPU timeout (ACL 507017) was still
#     LISTENing an hour later, and a probe of it answered HealthCheck, GetServerInfo,
#     reset_episode and 19 ingests before failing on the first call that touched the
#     device.  Nothing the client could see from outside said it was broken.
#
# The servers now log a marker when they find their device unusable (and report
# NOT_SERVING from then on), so the second case is detected here, where the log is.
# Only that slot's pair is stopped -- which frees its card and closes its port, so
# its client fails fast -- and the other slots keep serving their own shards.
POISON_MARKER="NPU device unusable after a failed request"
while true; do
  live=0
  for slot in $(seq 0 $((NUM_SLOTS - 1))); do
    [[ "${RETIRED[$slot]:-0}" -eq 1 ]] && continue
    for role in model vo; do
      if grep -qF "$POISON_MARKER" "$RUNTIME_DIR/logs/${role}_${slot}.log" 2>/dev/null; then
        tail -40 "$RUNTIME_DIR/logs/${role}_${slot}.log" >&2
        printf '[ppa-npu] slot %s: the %s server reports its NPU unusable; retiring the slot\n' "$slot" "$role" >&2
        stop_pid "${MODEL_PIDS[$slot]}"
        stop_pid "${VO_PIDS[$slot]}"
        # Forget the pids once they are reaped: a long hold can outlive a pid's reuse,
        # and the exit trap would then signal whatever process inherited the number.
        MODEL_PIDS[$slot]=""
        VO_PIDS[$slot]=""
        RETIRED[$slot]=1
        printf 'slot=%s role=%s %s\n' "$slot" "$role" "$(date +%Y-%m-%dT%H:%M:%S%z)" >> "$RUNTIME_DIR/RETIRED"
        break
      fi
    done
    [[ "${RETIRED[$slot]:-0}" -eq 1 ]] && continue
    kill -0 "${MODEL_PIDS[$slot]}" 2>/dev/null || { tail -60 "$RUNTIME_DIR/logs/model_${slot}.log" >&2; die "model server slot $slot died"; }
    kill -0 "${VO_PIDS[$slot]}" 2>/dev/null || { tail -60 "$RUNTIME_DIR/logs/vo_${slot}.log" >&2; die "VO server slot $slot died"; }
    live=$((live + 1))
  done
  (( live > 0 )) || die "every slot has been retired (see $RUNTIME_DIR/RETIRED)"
  sleep 30
done

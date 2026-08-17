#!/usr/bin/bash -p
# Single-GPU native InternNav System2+System1 RPC launcher for train DAgger.
#
# This launcher accepts the existing STAGE3_EVAL_* data/control variables only
# for wrapper compatibility. It never reads or loads a Stage1, Stage3, LoRA,
# heatmap, panoramic-adapter, or CorrectNav checkpoint.

set -Eeuo pipefail

unset BASH_ENV ENV
readonly FIXED_SYSTEM_PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/maca-3.3.0/bin:/opt/maca-3.3.0/ompi/bin:/opt/mxdriver/bin"
export PATH="$FIXED_SYSTEM_PATH"

readonly ALLOWED_ROOT="/mnt/afs/lixiaoou/intern/fjl"
readonly EXPECTED_MODEL_MANIFEST_SHA256="f37a6df2e0703e38c34ccdba89c861bb8490ad3a36201bc1ec24a7509bf56581"
readonly EXPECTED_RUNTIME_MANIFEST_SHA256="99844c9592b40c6756a7b1fcf124e3fe2d0db15236abdf965b4f4588cb3d1eef"
readonly EXPECTED_PLAN_MANIFEST_SHA256="db1821a78cac9f5df77d6b0d19a1ad49c2beaf82a1e5936e7373c1394a9e9fcd"
readonly EXPECTED_QWEN_PYTHON_SHA256="dd01ccf9241044e50c511632c030fd52e97ecaeb68259441876ec76851d5de8f"
readonly EXPECTED_VLNCE_PYTHON_SHA256="95c92ef178223498301f5319a2397d0eb20c88f30640ec9c4631d2d927111098"
readonly EXPECTED_INTERNNAV_RUNTIME_ROWS=21
readonly EXPECTED_RPC_RUNTIME_ROWS=21
readonly NATIVE_PROTOCOL="internnav-native-joint-front-history-lookdown-v1"

die() {
  echo "[internnav-native-dagger] ERROR: $*" >&2
  exit 1
}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

canonical_under_allowed_root() {
  local candidate="$1"
  local label="$2"
  local resolved
  case "${candidate}/" in
    "${ALLOWED_ROOT}/"*) ;;
    *) die "${label} must stay under ${ALLOWED_ROOT}: ${candidate}" ;;
  esac
  resolved="$(readlink -m -- "$candidate")"
  case "${resolved}/" in
    "${ALLOWED_ROOT}/"*) ;;
    *) die "${label} resolves outside ${ALLOWED_ROOT}: ${candidate} -> ${resolved}" ;;
  esac
  printf '%s\n' "$resolved"
}

require_file() {
  [[ -s "$1" ]] || die "missing required non-empty file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "missing required directory: $1"
}

sha256_of() {
  sha256sum -- "$1" | awk '{print $1}'
}

require_sha256() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(sha256_of "$path")"
  [[ "$actual" == "$expected" ]] || die "${label} SHA256 mismatch: expected=${expected} actual=${actual} path=${path}"
}

verify_runtime_subset() {
  local prefix="$1"
  local expected_rows="$2"
  local label="$3"
  local actual_rows
  actual_rows="$(awk -v prefix="$prefix" 'index($2, prefix) == 1 {count++} END {print count + 0}' "$RUNTIME_MANIFEST")"
  [[ "$actual_rows" == "$expected_rows" ]] || die "${label} manifest row count mismatch: expected=${expected_rows} actual=${actual_rows}"
  if ! awk -v prefix="$prefix" 'index($2, prefix) == 1' "$RUNTIME_MANIFEST" | sha256sum -c - >/dev/null; then
    die "${label} source closure mismatch"
  fi
}

stop_pid() {
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

FJL_ROOT="$ALLOWED_ROOT"
REPO="$(canonical_under_allowed_root "${FJL_ROOT}/HeatmapVLN" repository)"
if [[ -n "${HEATMAPVLN_REPO_ROOT:-}" ]]; then
  requested_repo="$(canonical_under_allowed_root "$HEATMAPVLN_REPO_ROOT" "requested repository")"
  [[ "$requested_repo" == "$REPO" ]] || die "HEATMAPVLN_REPO_ROOT override is forbidden: $requested_repo"
fi
INTERNNAV_REPO="$(canonical_under_allowed_root "${FJL_ROOT}/InternNav" "InternNav repository")"
MODEL_DIR="$(canonical_under_allowed_root "${FJL_ROOT}/InternNav-Model" "native model")"
PLAN="$(canonical_under_allowed_root "${FJL_ROOT}/evaluation_plans/internnav_native_r2r_val_unseen_4gpu_20260802" "native plan")"
RPC_ROOT="$(canonical_under_allowed_root "${FJL_ROOT}/rpc" "RPC repository")"
CALLER_VLNCE_PYTHON="${VLNCE_PYTHON:-}"
QWEN_PYTHON="$(canonical_under_allowed_root "${FJL_ROOT}/envs/qwen25/bin/python" "qwen Python")"
VLNCE_PYTHON="$(canonical_under_allowed_root "${FJL_ROOT}/envs/vlnce/bin/python" "VLN-CE Python")"
if [[ -n "${QWEN25_PYTHON:-}" ]]; then
  requested_qwen_python="$(canonical_under_allowed_root "$QWEN25_PYTHON" "requested qwen Python")"
  [[ "$requested_qwen_python" == "$QWEN_PYTHON" ]] || die "QWEN25_PYTHON override is forbidden: $requested_qwen_python"
fi
if [[ -n "$CALLER_VLNCE_PYTHON" ]]; then
  requested_vlnce_python="$(canonical_under_allowed_root "$CALLER_VLNCE_PYTHON" "requested VLN-CE Python")"
  [[ "$requested_vlnce_python" == "$VLNCE_PYTHON" ]] || die "VLNCE_PYTHON override is forbidden: $requested_vlnce_python"
fi
MODEL_MANIFEST="$PLAN/manifests/internnav_model.sha256"
RUNTIME_MANIFEST="$PLAN/manifests/runtime_code.sha256"
PLAN_MANIFEST="$PLAN/manifests/plan_code.sha256"
ORIGINAL_SERVER="$PLAN/tools/rpc_internnav_native_server.py"
SERVER_FACADE="$REPO/scripts/evaluation/rpc_internnav_native_dagger_server.py"
EVALUATOR="$REPO/scripts/evaluation/r2r_val_unseen.py"
COLLECTOR="$REPO/scripts/evaluation/trajectory_dagger.py"
RPC_PROTOCOL_SOURCE="$REPO/scripts/evaluation/rpc_protocol.py"
CONFIG="$PLAN/configs/internnav_native_eval.yaml"
SCENES_DIR="$(canonical_under_allowed_root "${STAGE3_EVAL_SCENES_DIR:-${FJL_ROOT}/habitat/VLN-CE/data/scene_datasets}" "scene directory")"
DATA_PATH="$(canonical_under_allowed_root "${STAGE3_EVAL_DATA_PATH:-${FJL_ROOT}/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz}" dataset)"
DATASET_SPLIT="${STAGE3_EVAL_DATASET_SPLIT:-train}"
OUTPUT_PATH="$(canonical_under_allowed_root "${STAGE3_EVAL_OUTPUT_PATH:?STAGE3_EVAL_OUTPUT_PATH is required}" "control output")"
LOG_DIR="$(canonical_under_allowed_root "${STAGE3_EVAL_LOG_DIR:-${OUTPUT_PATH}/logs}" "log directory")"
EPISODE_LIST="${STAGE3_EVAL_EPISODE_LIST:-}"
if [[ -n "$EPISODE_LIST" ]]; then
  EPISODE_LIST="$(canonical_under_allowed_root "$EPISODE_LIST" "episode list")"
fi

COLLECT_ROOT="$(canonical_under_allowed_root "${STAGE3_EVAL_TRAJECTORY_DAGGER_ROOT:?STAGE3_EVAL_TRAJECTORY_DAGGER_ROOT is required}" "collection root")"
ROUND_ID="${STAGE3_EVAL_TRAJECTORY_DAGGER_ROUND:-0}"
MAX_GB="${STAGE3_EVAL_TRAJECTORY_DAGGER_MAX_GB:-300.0}"
NORMAL_QUOTA="${STAGE3_EVAL_TRAJECTORY_DAGGER_NORMAL_QUOTA:-1}"
HARD_QUOTA="${STAGE3_EVAL_TRAJECTORY_DAGGER_HARD_QUOTA:-2}"
JPEG_QUALITY="${STAGE3_EVAL_TRAJECTORY_DAGGER_JPEG_QUALITY:-75}"
HARD_OFFPATH_M="${STAGE3_EVAL_TRAJECTORY_DAGGER_HARD_OFFPATH_M:-0.75}"
MAX_ORACLE_ACTIONS="${STAGE3_EVAL_TRAJECTORY_DAGGER_MAX_ORACLE_ACTIONS:-128}"
MIN_HISTORY="${STAGE3_EVAL_TRAJECTORY_DAGGER_MIN_HISTORY:-2}"
MAX_STEPS="${STAGE3_EVAL_MAX_STEPS:-120}"
MAX_SYSTEM2_CALLS="${STAGE3_EVAL_MAX_SYSTEM2_CALLS:-64}"
NUM_HISTORY="${STAGE3_EVAL_NUM_HISTORY:-8}"
MAX_EPISODES="${STAGE3_EVAL_MAX_EPISODES:-}"
RESUME="${STAGE3_EVAL_RESUME:-0}"
OVERWRITE="${STAGE3_EVAL_OVERWRITE:-0}"
GPU_ID="${INTERNNAV_NATIVE_GPU_ID:-${STAGE3_EVAL_GPU_ID:-0}}"
RPC_PORT="${INTERNNAV_NATIVE_RPC_PORT:-52500}"
DISPLAY_NUM="${INTERNNAV_NATIVE_DISPLAY_NUM:-331}"
RPC_TIMEOUT_MS="${STAGE3_EVAL_RPC_TIMEOUT_MS:-600000}"
SERVER_START_TIMEOUT_S="${INTERNNAV_NATIVE_SERVER_START_TIMEOUT_S:-2400}"
LP_THREADS="${INTERNNAV_NATIVE_LP_NUM_THREADS:-8}"
SERVER_CPU_THREADS="${INTERNNAV_NATIVE_SERVER_CPU_THREADS:-4}"
CLIENT_CPU_THREADS="${INTERNNAV_NATIVE_CLIENT_CPU_THREADS:-1}"

[[ "$DATASET_SPLIT" == train ]] || die "native DAgger launcher is train-only"
is_true "${STAGE3_EVAL_COLLECT_TRAJECTORY_DAGGER:-0}" || die "trajectory DAgger collection must be enabled"
[[ "${STAGE3_EVAL_AUTO_STOP_DISTANCE:-0.0}" == "0.0" || "${STAGE3_EVAL_AUTO_STOP_DISTANCE:-0}" == "0" ]] || die "privileged auto-stop is forbidden"
! is_true "${STAGE3_EVAL_ORACLE_SYSTEM2:-0}" || die "oracle System2 is forbidden"
! is_true "${STAGE3_EVAL_PANO_RECENTER_BEFORE_SYSTEM1:-0}" || die "pano recenter is not native InternNav"
[[ "$NUM_HISTORY" =~ ^[0-9]+$ ]] && (( NUM_HISTORY >= 1 && NUM_HISTORY <= 8 )) || die "num_history must be in [1,8]"
[[ "$MAX_STEPS" =~ ^[0-9]+$ ]] && (( MAX_STEPS >= 1 && MAX_STEPS <= 500 )) || die "max steps must be in [1,500]"
[[ "$MAX_SYSTEM2_CALLS" =~ ^[0-9]+$ ]] && (( MAX_SYSTEM2_CALLS >= 1 && MAX_SYSTEM2_CALLS <= 500 )) || die "max System2 calls must be in [1,500]"
[[ "$ROUND_ID" =~ ^[0-9]+$ ]] || die "round id must be non-negative"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || die "GPU id must be non-negative"
[[ "$RPC_PORT" =~ ^[0-9]+$ ]] && (( RPC_PORT >= 1024 && RPC_PORT <= 65535 )) || die "invalid RPC port"
[[ "$DISPLAY_NUM" =~ ^[0-9]+$ ]] || die "display number must be non-negative"
[[ "$LP_THREADS" =~ ^([1-9]|1[0-6])$ ]] || die "llvmpipe threads must be an integer in [1,16]"
[[ "$SERVER_CPU_THREADS" =~ ^([1-9]|1[0-6])$ ]] || die "server CPU threads must be an integer in [1,16]"
[[ "$CLIENT_CPU_THREADS" =~ ^([1-9]|1[0-6])$ ]] || die "client CPU threads must be an integer in [1,16]"

for path in "$QWEN_PYTHON" "$VLNCE_PYTHON"; do
  [[ -x "$path" ]] || die "missing Python executable: $path"
done
require_sha256 "$QWEN_PYTHON" "$EXPECTED_QWEN_PYTHON_SHA256" "qwen Python"
require_sha256 "$VLNCE_PYTHON" "$EXPECTED_VLNCE_PYTHON_SHA256" "VLN-CE Python"
for path in "$MODEL_MANIFEST" "$RUNTIME_MANIFEST" "$PLAN_MANIFEST" "$ORIGINAL_SERVER" "$SERVER_FACADE" "$EVALUATOR" "$COLLECTOR" "$RPC_PROTOCOL_SOURCE" "$CONFIG" "$DATA_PATH"; do
  require_file "$path"
done
for path in "$REPO" "$INTERNNAV_REPO" "$MODEL_DIR" "$RPC_ROOT/src/vla_rpc" "$SCENES_DIR"; do
  require_dir "$path"
done
if [[ -n "$EPISODE_LIST" ]]; then
  require_file "$EPISODE_LIST"
fi
gzip -t "$DATA_PATH"

X11_BUNDLE="$FJL_ROOT/tools/x11_headless_bundle_ubuntu22_20260801_v4"
X11_MANIFEST="$X11_BUNDLE/manifest.sha256"
XVFB_BIN="$X11_BUNDLE/bin/Xvfb"
XDPYINFO_BIN="$X11_BUNDLE/bin/xdpyinfo"
GLXINFO_BIN="$X11_BUNDLE/bin/glxinfo"
X11_DRI_PATH="$X11_BUNDLE/dri"
X11_FONT_PATH="$X11_BUNDLE/share/fonts/misc"
X11_XKB_PATH="$X11_BUNDLE/share/X11/xkb"
for path in "$X11_BUNDLE" "$X11_DRI_PATH" "$X11_FONT_PATH" "$X11_XKB_PATH"; do require_dir "$path"; done
for path in "$X11_MANIFEST" "$X11_BUNDLE/manifest.json" "$X11_DRI_PATH/swrast_dri.so"; do require_file "$path"; done
for path in "$XVFB_BIN" "$XDPYINFO_BIN" "$GLXINFO_BIN"; do [[ -x "$path" ]] || die "missing X11 executable: $path"; done
sha256sum -c "$X11_MANIFEST" >/dev/null

# Do not let a caller inject alternate Python packages into the audited policy.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

readonly MACA_RUNTIME_ROOT="/opt/maca-3.3.0"
readonly SERVER_LD_LIBRARY_PATH="${MACA_RUNTIME_ROOT}/lib:${MACA_RUNTIME_ROOT}/ompi/lib:${MACA_RUNTIME_ROOT}/ucx/lib:/opt/mxdriver/lib"
export MACA_HOME="$MACA_RUNTIME_ROOT"
export MACA_PATH="$MACA_RUNTIME_ROOT"
export MACA_DIR="$MACA_RUNTIME_ROOT"
export LD_LIBRARY_PATH="$SERVER_LD_LIBRARY_PATH"
export INTERNNAV_MODEL_PATH="$MODEL_DIR"
export INTERNNAV_BACKBONE="$MODEL_DIR"
export INTERNNAV_REPO
export HEATMAPVLN_REPO="$REPO"
export HEATMAPVLN_FJL_ROOT="$FJL_ROOT"
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export HEATMAPVLN_REQUIRE_FLASH_ATTN=1

require_sha256 "$MODEL_MANIFEST" "$EXPECTED_MODEL_MANIFEST_SHA256" "native model manifest"
require_sha256 "$RUNTIME_MANIFEST" "$EXPECTED_RUNTIME_MANIFEST_SHA256" "runtime source manifest"
require_sha256 "$PLAN_MANIFEST" "$EXPECTED_PLAN_MANIFEST_SHA256" "native plan manifest"
echo "[internnav-native-dagger] verifying official InternNav model closure (4 shards, 1338 tensors)"
sha256sum -c "$MODEL_MANIFEST" >/dev/null
echo "[internnav-native-dagger] native model closure verified"

runtime_manifest_expected="$(awk -v path="$RUNTIME_MANIFEST" '$2 == path {print $1}' "$PLAN_MANIFEST")"
[[ "$runtime_manifest_expected" == "$EXPECTED_RUNTIME_MANIFEST_SHA256" ]] || die "trusted plan does not pin the expected runtime manifest"
config_expected="$(awk -v path="$CONFIG" '$2 == path {print $1}' "$PLAN_MANIFEST")"
[[ -n "$config_expected" ]] || die "native config is absent from locked plan manifest"
config_actual="$(sha256_of "$CONFIG")"
[[ "$config_actual" == "$config_expected" ]] || die "locked native config hash mismatch"
original_server_expected="$(awk -v path="$ORIGINAL_SERVER" '$2 == path {print $1}' "$PLAN_MANIFEST")"
[[ -n "$original_server_expected" ]] || die "native server is absent from locked plan manifest"
original_server_actual="$(sha256_of "$ORIGINAL_SERVER")"
[[ "$original_server_actual" == "$original_server_expected" ]] || die "locked native server source hash mismatch"
verify_runtime_subset "$FJL_ROOT/InternNav/" "$EXPECTED_INTERNNAV_RUNTIME_ROWS" "InternNav runtime"
verify_runtime_subset "$FJL_ROOT/rpc/src/vla_rpc/" "$EXPECTED_RPC_RUNTIME_ROWS" "RPC runtime"
echo "[internnav-native-dagger] locked InternNav and RPC runtime source closures verified"

mkdir -p "$LOG_DIR"
RUN_STAMP="${INTERNNAV_NATIVE_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_job$$}"
[[ "$RUN_STAMP" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || die "invalid INTERNNAV_NATIVE_RUN_STAMP: $RUN_STAMP"
RUNTIME_BASE_RAW="$OUTPUT_PATH/native_runtime"
[[ ! -L "$RUNTIME_BASE_RAW" ]] || die "refusing symlinked native runtime base: $RUNTIME_BASE_RAW"
RUNTIME_BASE="$(canonical_under_allowed_root "$RUNTIME_BASE_RAW" "native runtime base")"
[[ "$RUNTIME_BASE" == "$RUNTIME_BASE_RAW" ]] || die "native runtime base changed during canonicalization: $RUNTIME_BASE_RAW -> $RUNTIME_BASE"
RUNTIME_DIR="$(canonical_under_allowed_root "$RUNTIME_BASE/$RUN_STAMP" "native runtime directory")"
case "${RUNTIME_DIR}/" in
  "${RUNTIME_BASE}/"*) ;;
  *) die "native runtime directory escaped its control root: $RUNTIME_DIR" ;;
esac
[[ ! -e "$RUNTIME_DIR" && ! -L "$RUNTIME_DIR" ]] || die "native runtime directory already exists: $RUNTIME_DIR"
mkdir -p "$RUNTIME_DIR/logs" "$RUNTIME_DIR/server" "$RUNTIME_DIR/client" "$RUNTIME_DIR/xvfb"
POLICY_CLOSURE="$OUTPUT_PATH/native_policy_closure.json"

server_facade_sha="$(sha256sum "$SERVER_FACADE" | awk '{print $1}')"
evaluator_sha="$(sha256sum "$EVALUATOR" | awk '{print $1}')"
collector_sha="$(sha256sum "$COLLECTOR" | awk '{print $1}')"
rpc_protocol_sha="$(sha256sum "$RPC_PROTOCOL_SOURCE" | awk '{print $1}')"
launcher_sha="$(sha256sum "$0" | awk '{print $1}')"
verify_policy_source_snapshot() {
  require_sha256 "$MODEL_MANIFEST" "$EXPECTED_MODEL_MANIFEST_SHA256" "native model manifest"
  require_sha256 "$RUNTIME_MANIFEST" "$EXPECTED_RUNTIME_MANIFEST_SHA256" "runtime source manifest"
  require_sha256 "$PLAN_MANIFEST" "$EXPECTED_PLAN_MANIFEST_SHA256" "native plan manifest"
  require_sha256 "$QWEN_PYTHON" "$EXPECTED_QWEN_PYTHON_SHA256" "qwen Python"
  require_sha256 "$VLNCE_PYTHON" "$EXPECTED_VLNCE_PYTHON_SHA256" "VLN-CE Python"
  require_sha256 "$CONFIG" "$config_actual" "native config"
  require_sha256 "$ORIGINAL_SERVER" "$original_server_actual" "native server"
  require_sha256 "$SERVER_FACADE" "$server_facade_sha" "native DAgger facade"
  require_sha256 "$EVALUATOR" "$evaluator_sha" "native DAgger evaluator"
  require_sha256 "$COLLECTOR" "$collector_sha" "trajectory DAgger collector"
  require_sha256 "$RPC_PROTOCOL_SOURCE" "$rpc_protocol_sha" "RPC protocol"
  require_sha256 "$0" "$launcher_sha" "native DAgger launcher"
  verify_runtime_subset "$FJL_ROOT/InternNav/" "$EXPECTED_INTERNNAV_RUNTIME_ROWS" "InternNav runtime"
  verify_runtime_subset "$FJL_ROOT/rpc/src/vla_rpc/" "$EXPECTED_RPC_RUNTIME_ROWS" "RPC runtime"
}

policy_fingerprint="$("$QWEN_PYTHON" - "$POLICY_CLOSURE" "$MODEL_MANIFEST" "$RUNTIME_MANIFEST" "$ORIGINAL_SERVER" "$original_server_actual" "$SERVER_FACADE" "$server_facade_sha" "$EVALUATOR" "$evaluator_sha" "$COLLECTOR" "$collector_sha" "$RPC_PROTOCOL_SOURCE" "$rpc_protocol_sha" "$0" "$launcher_sha" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

(
    output,
    model_manifest,
    runtime_manifest,
    original_server,
    original_server_sha,
    facade,
    facade_sha,
    evaluator,
    evaluator_sha,
    collector,
    collector_sha,
    rpc_protocol,
    rpc_protocol_sha,
    launcher,
    launcher_sha,
) = sys.argv[1:]

def manifest_rows(path, prefix=None):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, file_path = line.split(None, 1)
        file_path = file_path.strip()
        if prefix is None or file_path.startswith(prefix):
            rows.append({"path": file_path, "sha256": digest})
    return rows

closure = {
    "schema": "internnav-native-policy-closure-v1",
    "policy_backend": "internnav_native",
    "native_protocol": "internnav-native-joint-front-history-lookdown-v1",
    "model_manifest": {
        "path": model_manifest,
        "sha256": hashlib.sha256(Path(model_manifest).read_bytes()).hexdigest(),
        "entries": manifest_rows(model_manifest),
    },
    "internnav_runtime_sources": manifest_rows(
        runtime_manifest,
        "/mnt/afs/lixiaoou/intern/fjl/InternNav/",
    ),
    "rpc_runtime_sources": manifest_rows(
        runtime_manifest,
        "/mnt/afs/lixiaoou/intern/fjl/rpc/src/vla_rpc/",
    ),
    "harness_sources": [
        {"path": original_server, "sha256": original_server_sha},
        {"path": facade, "sha256": facade_sha},
        {"path": evaluator, "sha256": evaluator_sha},
        {"path": collector, "sha256": collector_sha},
        {"path": rpc_protocol, "sha256": rpc_protocol_sha},
        {"path": str(Path(launcher).resolve()), "sha256": launcher_sha},
    ],
    "model_contract": {
        "class": "InternVLAN1ForCausalLM",
        "system2": "internnav_native_qwen",
        "system1": "internnav_native_nextdit_async",
        "tensor_count": 1338,
        "shard_count": 4,
        "lora": False,
        "adapter": False,
        "external_checkpoint": False,
    },
}
canonical = json.dumps(
    closure,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
fingerprint = "internnav-native-v1:" + hashlib.sha256(canonical).hexdigest()
document = {**closure, "policy_fingerprint": fingerprint}
path = Path(output)
payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
print(fingerprint)
PY
)"
[[ "$policy_fingerprint" =~ ^internnav-native-v1:[0-9a-f]{64}$ ]] || die "invalid generated policy fingerprint"
verify_policy_source_snapshot
echo "[internnav-native-dagger] post-fingerprint source snapshot verified"
if [[ -n "${STAGE3_EVAL_TRAJECTORY_DAGGER_POLICY_FINGERPRINT:-}" && "$STAGE3_EVAL_TRAJECTORY_DAGGER_POLICY_FINGERPRINT" != "$policy_fingerprint" ]]; then
  die "caller-supplied policy fingerprint does not match verified native closure"
fi
export STAGE3_EVAL_TRAJECTORY_DAGGER_POLICY_FINGERPRINT="$policy_fingerprint"
export INTERNNAV_NATIVE_POLICY_FINGERPRINT="$policy_fingerprint"
echo "[internnav-native-dagger] policy_fingerprint=$policy_fingerprint"
echo "[internnav-native-dagger] policy_closure=$POLICY_CLOSURE"
echo "[internnav-native-dagger] external_checkpoint=false lora=false adapter=false"

RPC_PYTHONPATH="$PLAN/tools:$RPC_ROOT/src:$REPO:$INTERNNAV_REPO"
X11_TOOL_LD_LIBRARY_PATH="$X11_BUNDLE/lib:$LD_LIBRARY_PATH"
X11_CLIENT_LD_LIBRARY_PATH="$X11_BUNDLE/mesa_lib:$LD_LIBRARY_PATH"
DISPLAY_ADDR="127.0.0.1:${DISPLAY_NUM}.0"
SERVER_LOG="$RUNTIME_DIR/logs/native_server.log"
CLIENT_LOG="$RUNTIME_DIR/logs/native_client.log"
XVFB_LOG="$RUNTIME_DIR/logs/xvfb.log"

DIST_ENV_UNSET_ARGS=(
  -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE
  -u GROUP_RANK -u ROLE_RANK -u ROLE_WORLD_SIZE -u NODE_RANK
  -u MASTER_ADDR -u MASTER_PORT
  -u OMPI_COMM_WORLD_RANK -u OMPI_COMM_WORLD_SIZE
  -u PMI_RANK -u PMI_SIZE -u PMIX_RANK
  -u SLURM_PROCID -u SLURM_LOCALID -u SLURM_NTASKS -u SLURM_NPROCS
)
GL_ENV_UNSET_ARGS=(-u __GLX_VENDOR_LIBRARY_NAME -u __EGL_VENDOR_LIBRARY_FILENAMES)
SERVER_CLEAN_ENV_UNSET_ARGS=(
  -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY
  -u __GLX_VENDOR_LIBRARY_NAME -u __EGL_VENDOR_LIBRARY_FILENAMES
  -u EGL_PLATFORM -u EGL_LOG_LEVEL -u GLX_VENDOR_LIBRARY_NAME
  -u LIBGL_DRIVERS_PATH -u LIBGL_ALWAYS_SOFTWARE -u LIBGL_ALWAYS_INDIRECT -u LIBGL_DEBUG -u LIBGL_DRI3_DISABLE
  -u GALLIUM_DRIVER
  -u MESA_LOADER_DRIVER_OVERRIDE -u MESA_SHADER_CACHE_DIR -u MESA_SHADER_CACHE_DISABLE
  -u MESA_GL_VERSION_OVERRIDE -u MESA_GLSL_VERSION_OVERRIDE -u MESA_EXTENSION_OVERRIDE
  -u MESA_NO_ERROR -u MESA_DEBUG
  -u DRI_PRIME -u GBM_BACKEND
  -u PYOPENGL_PLATFORM -u MUJOCO_GL
  -u VK_ICD_FILENAMES -u __NV_PRIME_RENDER_OFFLOAD
  -u LP_NUM_THREADS
  -u HABITAT_GL_GPU_ID -u HABITAT_SIM_LOG
  -u HEATMAPVLN_ALLOW_NVIDIA_GLX -u HEATMAPVLN_PREINIT_GL
  -u HEATMAPVLN_PREINIT_EMPTY_GL -u HEATMAPVLN_PREINIT_SCENE
  -u MAGNUM_LOG -u MAGNUM_GPU_VALIDATION
  -u LD_PRELOAD
)
# Also remove any less-common inherited variables in the same graphics/runtime namespaces.
while IFS='=' read -r -d '' inherited_name _; do
  case "$inherited_name" in
    __GLX_*|__EGL_*|EGL_*|GLX_*|LIBGL_*|GALLIUM_*|MESA_*|HABITAT_*|MAGNUM_*|GBM_*)
      SERVER_CLEAN_ENV_UNSET_ARGS+=(-u "$inherited_name")
      ;;
  esac
done < <(env -0)
CHECKPOINT_ENV_UNSET_ARGS=(
  -u STAGE3_EVAL_BASE_CKPT
  -u STAGE3_EVAL_CHECKPOINT
  -u STAGE3_EVAL_PANO_LATENT_ADAPTER_CHECKPOINT
)

SERVER_PID=""
CLIENT_PID=""
XVFB_PID=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_pid "$CLIENT_PID"
  stop_pid "$SERVER_PID"
  stop_pid "$XVFB_PID"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

verify_xvfb_renderer() {
  local phase="$1"
  local renderer
  kill -0 "$XVFB_PID" 2>/dev/null || { tail -100 "$XVFB_LOG" >&2 || true; die "Xvfb died during ${phase}"; }
  if ! env "${GL_ENV_UNSET_ARGS[@]}" LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH" DISPLAY="$DISPLAY_ADDR" timeout 5 "$XDPYINFO_BIN" >/dev/null 2>&1; then
    tail -100 "$XVFB_LOG" >&2 || true
    die "DISPLAY=$DISPLAY_ADDR failed xdpyinfo during ${phase}"
  fi
  renderer="$(env "${GL_ENV_UNSET_ARGS[@]}" LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH" LIBGL_DRIVERS_PATH="$X11_DRI_PATH" DISPLAY="$DISPLAY_ADDR" LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=swrast timeout 120 "$GLXINFO_BIN" -B 2>/dev/null | grep -F 'OpenGL renderer string:' | head -1 || true)"
  [[ "${renderer,,}" == *llvmpipe* ]] || die "DISPLAY=$DISPLAY_ADDR is not using llvmpipe during ${phase}: ${renderer:-missing}"
  echo "[internnav-native-dagger] phase=$phase DISPLAY=$DISPLAY_ADDR $renderer"
}

"$QWEN_PYTHON" - "$RPC_PORT" <<'PY'
import socket
import sys
port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", port))
finally:
    sock.close()
print(f"RPC port available: {port}")
PY

if env "${GL_ENV_UNSET_ARGS[@]}" LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH" DISPLAY="$DISPLAY_ADDR" timeout 5 "$XDPYINFO_BIN" >/dev/null 2>&1; then
  die "DISPLAY=$DISPLAY_ADDR is already active; choose INTERNNAV_NATIVE_DISPLAY_NUM"
fi
mkdir -p "$RUNTIME_DIR/xvfb/.xkb-cache"
(
  cd "$RUNTIME_DIR/xvfb"
  exec 9<"$RUNTIME_DIR/xvfb/.xkb-cache"
  exec env "${GL_ENV_UNSET_ARGS[@]}"     PATH="$X11_BUNDLE/bin:$PATH"     LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH"     LIBGL_DRIVERS_PATH="$X11_DRI_PATH"     LIBGL_ALWAYS_SOFTWARE=1     GALLIUM_DRIVER=llvmpipe     MESA_LOADER_DRIVER_OVERRIDE=swrast     LP_NUM_THREADS="$LP_THREADS"     "$XVFB_BIN" ":$DISPLAY_NUM"     -screen 0 1024x768x24 -nolock -nolisten unix -listen tcp +iglx -ac     -fp "$X11_FONT_PATH" -xkbdir "$X11_XKB_PATH"
) >"$XVFB_LOG" 2>&1 &
XVFB_PID="$!"
xvfb_ready=0
for _ in $(seq 1 60); do
  kill -0 "$XVFB_PID" 2>/dev/null || { tail -100 "$XVFB_LOG" >&2 || true; die "Xvfb exited"; }
  if env "${GL_ENV_UNSET_ARGS[@]}" LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH" DISPLAY="$DISPLAY_ADDR" timeout 5 "$XDPYINFO_BIN" >/dev/null 2>&1; then
    xvfb_ready=1
    break
  fi
  sleep 1
done
[[ "$xvfb_ready" == 1 ]] || die "Xvfb did not become ready"
verify_xvfb_renderer "initial-preflight"

IMPORT_PROBE_ROOT="$RUNTIME_DIR/client_import_probe"
mkdir -p \
  "$IMPORT_PROBE_ROOT/tmp" \
  "$IMPORT_PROBE_ROOT/home" \
  "$IMPORT_PROBE_ROOT/xdg_cache" \
  "$IMPORT_PROBE_ROOT/xdg_runtime" \
  "$IMPORT_PROBE_ROOT/hf_home" \
  "$IMPORT_PROBE_ROOT/matplotlib" \
  "$IMPORT_PROBE_ROOT/mesa_shader_cache" \
  "$IMPORT_PROBE_ROOT/numba_cache" \
  "$IMPORT_PROBE_ROOT/pycache"
chmod 700 "$IMPORT_PROBE_ROOT/xdg_runtime"
env "${DIST_ENV_UNSET_ARGS[@]}" "${GL_ENV_UNSET_ARGS[@]}" "${CHECKPOINT_ENV_UNSET_ARGS[@]}" \
  PYTHONPATH="$RPC_PYTHONPATH" \
  LD_LIBRARY_PATH="$X11_CLIENT_LD_LIBRARY_PATH" \
  LIBGL_DRIVERS_PATH="$X11_DRI_PATH" \
  DISPLAY="$DISPLAY_ADDR" \
  CUDA_VISIBLE_DEVICES="" \
  HOME="$IMPORT_PROBE_ROOT/home" \
  TMPDIR="$IMPORT_PROBE_ROOT/tmp" \
  XDG_CACHE_HOME="$IMPORT_PROBE_ROOT/xdg_cache" \
  XDG_RUNTIME_DIR="$IMPORT_PROBE_ROOT/xdg_runtime" \
  HF_HOME="$IMPORT_PROBE_ROOT/hf_home" \
  MPLCONFIGDIR="$IMPORT_PROBE_ROOT/matplotlib" \
  MESA_SHADER_CACHE_DIR="$IMPORT_PROBE_ROOT/mesa_shader_cache" \
  NUMBA_CACHE_DIR="$IMPORT_PROBE_ROOT/numba_cache" \
  PYTHONPYCACHEPREFIX="$IMPORT_PROBE_ROOT/pycache" \
  PYTHONDONTWRITEBYTECODE=1 \
  HABITAT_GL_GPU_ID=0 \
  HEATMAPVLN_PREINIT_GL=0 \
  HEATMAPVLN_PREINIT_EMPTY_GL=1 \
  HEATMAPVLN_ALLOW_NVIDIA_GLX=0 \
  LIBGL_ALWAYS_SOFTWARE=1 \
  GALLIUM_DRIVER=llvmpipe \
  MESA_LOADER_DRIVER_OVERRIDE=swrast \
  LP_NUM_THREADS="$LP_THREADS" \
  OMP_NUM_THREADS="$CLIENT_CPU_THREADS" \
  MKL_NUM_THREADS="$CLIENT_CPU_THREADS" \
  OPENBLAS_NUM_THREADS="$CLIENT_CPU_THREADS" \
  NUMBA_NUM_THREADS="$CLIENT_CPU_THREADS" \
  USE_TF=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3 \
  "$VLNCE_PYTHON" -c 'import magnum, habitat_sim; print("magnum/habitat_sim import OK")'

if is_true "${STAGE3_EVAL_PREFLIGHT_ONLY:-0}"; then
  echo "[internnav-native-dagger] preflight-only passed; no model or evaluator launched"
  exit 0
fi

verify_policy_source_snapshot
echo "[internnav-native-dagger] pre-server source snapshot verified"
mkdir -p "$RUNTIME_DIR/server/tmp" "$RUNTIME_DIR/server/home" "$RUNTIME_DIR/server/xdg_cache" "$RUNTIME_DIR/server/xdg_runtime" "$RUNTIME_DIR/server/hf_home" "$RUNTIME_DIR/server/torch_extensions" "$RUNTIME_DIR/server/triton_cache" "$RUNTIME_DIR/server/matplotlib" "$RUNTIME_DIR/server/pycache"
chmod 700 "$RUNTIME_DIR/server/xdg_runtime"
env "${DIST_ENV_UNSET_ARGS[@]}" "${CHECKPOINT_ENV_UNSET_ARGS[@]}" "${SERVER_CLEAN_ENV_UNSET_ARGS[@]}" \
  LD_LIBRARY_PATH="$SERVER_LD_LIBRARY_PATH" \
  PYTHONPATH="$RPC_PYTHONPATH" \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  INTERNNAV_NATIVE_POLICY_FINGERPRINT="$policy_fingerprint" \
  HOME="$RUNTIME_DIR/server/home" \
  TMPDIR="$RUNTIME_DIR/server/tmp" \
  XDG_CACHE_HOME="$RUNTIME_DIR/server/xdg_cache" \
  XDG_RUNTIME_DIR="$RUNTIME_DIR/server/xdg_runtime" \
  HF_HOME="$RUNTIME_DIR/server/hf_home" \
  TORCH_EXTENSIONS_DIR="$RUNTIME_DIR/server/torch_extensions" \
  TRITON_CACHE_DIR="$RUNTIME_DIR/server/triton_cache" \
  MPLCONFIGDIR="$RUNTIME_DIR/server/matplotlib" \
  PYTHONPYCACHEPREFIX="$RUNTIME_DIR/server/pycache" \
  PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS="$SERVER_CPU_THREADS" \
  MKL_NUM_THREADS="$SERVER_CPU_THREADS" \
  OPENBLAS_NUM_THREADS="$SERVER_CPU_THREADS" \
  USE_TF=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3 \
  HEATMAPVLN_REQUIRE_FLASH_ATTN=1 \
  "$QWEN_PYTHON" -u "$SERVER_FACADE" \
    --model_path "$MODEL_DIR" \
    --gpu_id 0 \
    --host 127.0.0.1 \
    --port "$RPC_PORT" \
    --workers 1 \
    --require_deterministic_sampling \
    --log_level INFO \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"
echo "[internnav-native-dagger] server gpu=$GPU_ID port=$RPC_PORT pid=$SERVER_PID log=$SERVER_LOG"

start_time="$(date +%s)"
server_ready=0
while (( $(date +%s) - start_time < SERVER_START_TIMEOUT_S )); do
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -240 "$SERVER_LOG" >&2 || true; die "native RPC server exited during startup"; }
  if env "${DIST_ENV_UNSET_ARGS[@]}" PYTHONPATH="$RPC_PYTHONPATH" "$VLNCE_PYTHON" - "127.0.0.1:$RPC_PORT" "$policy_fingerprint" <<'PY' >/dev/null 2>&1
import sys
from vla_rpc.client import VLAClient
address, fingerprint = sys.argv[1:]
client = VLAClient(server_addr=address, timeout_ms=5000)
try:
    client.connect()
    info = client.get_server_info()
    if not client.health_check() or info is None:
        raise SystemExit(1)
    if info.model_version != f"internnav-native-r2r:{fingerprint}":
        raise SystemExit(1)
    if "internnav-native-joint-front-history-lookdown-v1" not in info.supported_formats:
        raise SystemExit(1)
finally:
    client.close()
PY
  then
    server_ready=1
    break
  fi
  sleep 10
done
[[ "$server_ready" == 1 ]] || { tail -240 "$SERVER_LOG" >&2 || true; die "native RPC server startup timeout"; }

for required in   "[internnav-native-dagger-facade] method=plan_native_internnav protocol=$NATIVE_PROTOCOL fingerprint=$policy_fingerprint"   "Native InternNav checkpoint index verified: tensors=1338 shards=4 lora=0 adapter=0"   "Native InternNav strict load verified: missing=0 unexpected=0 mismatched=0 errors=0"   "Native InternNav architecture verified: class=InternVLAN1ForCausalLM system1=nextdit_async n_query=4 state_tensors=1338"   "Native InternNav evaluation mode: front_only_prompt=True adapter=False lora=False external_checkpoint=False vlm_image_size=384 lookdown_vlm_size=640x480 traj_image_size=224"   "require_deterministic_sampling=True"   "Native InternNav RPC server listening on 127.0.0.1:$RPC_PORT"; do
  grep -Fq "$required" "$SERVER_LOG" || { tail -240 "$SERVER_LOG" >&2 || true; die "missing server assertion: $required"; }
done
echo "[internnav-native-dagger] full native System2+System1 load guards passed"
verify_xvfb_renderer "post-server-pre-client"

client_args=(
  --config "$CONFIG"
  --rpc_server "127.0.0.1:$RPC_PORT"
  --rpc_policy_mode internnav_native
  --rpc_policy_fingerprint "$policy_fingerprint"
  --rpc_timeout_ms "$RPC_TIMEOUT_MS"
  --rpc_jpeg_quality 90
  --rpc_protocol_seed 42
  --rpc_require_deterministic_sampling
  --scenes_dir "$SCENES_DIR"
  --data_path "$DATA_PATH"
  --dataset_split train
  --output_path "$OUTPUT_PATH"
  --sim_gpu_id 0
  --resize_w 384
  --resize_h 384
  --num_history "$NUM_HISTORY"
  --max_steps_per_episode "$MAX_STEPS"
  --auto_stop_distance 0.0
  --max_system2_calls_per_episode "$MAX_SYSTEM2_CALLS"
  --trajectory_selection mean
  --trajectory_x_sign 1
  --trajectory_heading_alignment none
  --system1_coord_order generated
  --no-pano_recenter_before_system1
  --no-debug_input_trace
  --debug_save_input_images 0
  --collect_trajectory_dagger
  --trajectory_dagger_root "$COLLECT_ROOT"
  --trajectory_dagger_round "$ROUND_ID"
  --trajectory_dagger_max_gb "$MAX_GB"
  --trajectory_dagger_normal_quota "$NORMAL_QUOTA"
  --trajectory_dagger_hard_quota "$HARD_QUOTA"
  --trajectory_dagger_jpeg_quality "$JPEG_QUALITY"
  --trajectory_dagger_hard_offpath_m "$HARD_OFFPATH_M"
  --trajectory_dagger_max_oracle_actions "$MAX_ORACLE_ACTIONS"
  --trajectory_dagger_min_history "$MIN_HISTORY"
  --trajectory_dagger_policy_fingerprint "$policy_fingerprint"
)
if [[ -n "$EPISODE_LIST" ]]; then client_args+=(--episode_list "$EPISODE_LIST"); fi
if [[ -n "$MAX_EPISODES" ]]; then client_args+=(--max_episodes "$MAX_EPISODES"); fi
if is_true "$RESUME"; then
  client_args+=(--resume)
elif is_true "$OVERWRITE"; then
  client_args+=(--overwrite_output)
fi

verify_policy_source_snapshot
echo "[internnav-native-dagger] pre-client source snapshot verified"
mkdir -p "$RUNTIME_DIR/client/tmp" "$RUNTIME_DIR/client/xdg_cache" "$RUNTIME_DIR/client/xdg_runtime" "$RUNTIME_DIR/client/hf_home" "$RUNTIME_DIR/client/matplotlib" "$RUNTIME_DIR/client/mesa_shader_cache" "$RUNTIME_DIR/client/numba_cache"
chmod 700 "$RUNTIME_DIR/client/xdg_runtime"
env "${DIST_ENV_UNSET_ARGS[@]}" "${GL_ENV_UNSET_ARGS[@]}" "${CHECKPOINT_ENV_UNSET_ARGS[@]}"   PYTHONPATH="$RPC_PYTHONPATH"   LD_LIBRARY_PATH="$X11_CLIENT_LD_LIBRARY_PATH"   LIBGL_DRIVERS_PATH="$X11_DRI_PATH"   DISPLAY="$DISPLAY_ADDR"   CUDA_VISIBLE_DEVICES="$GPU_ID"   TMPDIR="$RUNTIME_DIR/client/tmp"   XDG_CACHE_HOME="$RUNTIME_DIR/client/xdg_cache"   XDG_RUNTIME_DIR="$RUNTIME_DIR/client/xdg_runtime"   HF_HOME="$RUNTIME_DIR/client/hf_home"   MPLCONFIGDIR="$RUNTIME_DIR/client/matplotlib"   MESA_SHADER_CACHE_DIR="$RUNTIME_DIR/client/mesa_shader_cache"   NUMBA_CACHE_DIR="$RUNTIME_DIR/client/numba_cache"   HABITAT_GL_GPU_ID=0   HEATMAPVLN_PREINIT_GL=0   HEATMAPVLN_PREINIT_EMPTY_GL=1   HEATMAPVLN_ALLOW_NVIDIA_GLX=0   LIBGL_ALWAYS_SOFTWARE=1   GALLIUM_DRIVER=llvmpipe   MESA_LOADER_DRIVER_OVERRIDE=swrast   LP_NUM_THREADS="$LP_THREADS"   OMP_NUM_THREADS="$CLIENT_CPU_THREADS"   MKL_NUM_THREADS="$CLIENT_CPU_THREADS"   OPENBLAS_NUM_THREADS="$CLIENT_CPU_THREADS"   NUMBA_NUM_THREADS="$CLIENT_CPU_THREADS"   USE_TF=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3   "$VLNCE_PYTHON" -u "$EVALUATOR" "${client_args[@]}"   >"$CLIENT_LOG" 2>&1 &
CLIENT_PID="$!"
echo "[internnav-native-dagger] client pid=$CLIENT_PID log=$CLIENT_LOG"
set +e
wait "$CLIENT_PID"
client_status="$?"
set -e
CLIENT_PID=""
kill -0 "$SERVER_PID" 2>/dev/null || { tail -240 "$SERVER_LOG" >&2 || true; die "native server died during collection"; }
if (( client_status != 0 )); then
  tail -240 "$CLIENT_LOG" >&2 || true
  exit "$client_status"
fi
verify_policy_source_snapshot
echo "[internnav-native-dagger] post-collection source snapshot verified"

"$QWEN_PYTHON" - "$OUTPUT_PATH/progress.json" "$policy_fingerprint" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
fingerprint = sys.argv[2]
if not path.is_file():
    raise SystemExit(f"missing progress file: {path}")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    raise SystemExit("native collection produced no completed episode rows")
for index, row in enumerate(rows):
    expected = {
        "rpc_policy_mode": "internnav_native",
        "rpc_policy_fingerprint": fingerprint,
        "native_protocol": "internnav-native-joint-front-history-lookdown-v1",
        "rpc_model_version": f"internnav-native-r2r:{fingerprint}",
    }
    mismatches = {
        key: {"expected": value, "actual": row.get(key)}
        for key, value in expected.items()
        if row.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"progress row {index} policy provenance mismatch: {mismatches}")
print(f"verified native policy provenance in {len(rows)} progress rows")
PY

echo "[internnav-native-dagger] collection client completed with verified native policy provenance"
exit 0

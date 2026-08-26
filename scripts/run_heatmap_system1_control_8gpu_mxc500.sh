#!/usr/bin/env bash
#SBATCH --job-name=hm_s1_control
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --open-mode=append
#SBATCH --output=/mnt/afs/liwenhao/agent/370910109/data/heatmap_system1_training_v1/cluster_logs/%x_%j.out
#SBATCH --error=/mnt/afs/liwenhao/agent/370910109/data/heatmap_system1_training_v1/cluster_logs/%x_%j.err

# Train the structured heatmap residual control on the released InternNav path.
# Data collection is deliberately out of scope: this launcher consumes only a
# sealed four-root training_roots.json produced by the collection finalizer.
# Collection shard count is a data-layout property and is deliberately
# independent from the eight DDP ranks used for training.

set -Eeuo pipefail
umask 027

readonly ALLOWED_ROOT="/mnt/afs/liwenhao/agent/370910109"
readonly REPO_ROOT="${ALLOWED_ROOT}/HeatmapVLN"
readonly NATIVE_INTERNNAV_MODEL="${ALLOWED_ROOT}/InternNav-Model"
readonly NATIVE_MODEL_MANIFEST="${ALLOWED_ROOT}/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802/manifests/internnav_model.sha256"
readonly EXPECTED_NATIVE_MODEL_MANIFEST_SHA256="f37a6df2e0703e38c34ccdba89c861bb8490ad3a36201bc1ec24a7509bf56581"
readonly EXPECTED_NATIVE_MODEL_FILE_COUNT=14
readonly NATIVE_DEPENDENCY_SCHEMA="native-internnav-checkpoint-v1"
readonly QWEN25_ENV="${ALLOWED_ROOT}/envs/qwen25"
readonly PYTHON="${QWEN25_ENV}/bin/python"
readonly TORCHRUN="${QWEN25_ENV}/bin/torchrun"
readonly NUM_GPUS=8
readonly NUM_DAGGER_ROOTS=4
readonly MAX_COLLECTION_BYTES=300000000000
readonly EXPECTED_MANIFEST_SCHEMA="heatmapvln-trajectory-dagger-training-roots-v1"
readonly EXPECTED_NATIVE_PROTOCOL="internnav-native-joint-front-history-lookdown-v1"
readonly CONTROL_EVAL_LAUNCHER="${ALLOWED_ROOT}/evaluation_plans/heatmap_control_r2r_val_unseen_8gpu_20260804/scripts/run_8gpu_heatmap_control_rpc_eval.sh"
readonly CONTROL_EVAL_SERVER="${ALLOWED_ROOT}/evaluation_plans/heatmap_control_r2r_val_unseen_8gpu_20260804/tools/rpc_heatmap_control_server.py"

die() {
  echo "[heatmap-control] ERROR: $*" >&2
  exit 1
}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

usage() {
  cat >&2 <<'USAGE'
Usage: run_heatmap_system1_control_8gpu_mxc500.sh [--dry-run]

Required environment:
  HEATMAP_CONTROL_CKPT          Frozen single-view heatmap checkpoint.
  HEATMAP_CONTROL_CKPT_SHA256   Exact 64-character lowercase SHA-256.

Optional environment:
  DAGGER_TRAINING_ROOTS_MANIFEST  Sealed 4-shard training_roots.json.
  R2R_EXPERT_ROOT                 Expert root (default: r2r_paronamic_data).
  HEATMAP_CONTROL_EXPERIMENT_ROOT  Output/TensorBoard/launcher-log parent.
  HEATMAP_CONTROL_CONFIG          Alternate schema-compatible YAML.
  HEATMAP_CONTROL_EPOCH_SIZE      Formal global samples/epoch lock: 72000.
                                  Any other value is rejected.
  GPU_DEVICES                     Exactly eight unique GPU ids.
  HEATMAP_CONTROL_AUTO_RESUME=1   Resume the latest run under output root.
  HEATMAP_CONTROL_RESUME          Resume an explicit checkpoint file.
  HEATMAP_CONTROL_AUTO_EVAL       Run full 8-GPU R2R val_unseen after epoch 3
                                  (default: 1; set 0 to disable explicitly).
  EVAL_GPU_DEVICES                Eight eval GPU ids (default: GPU_DEVICES).
  EVAL_RPC_PORT_BASE              Eval RPC base port (default: 51400).
  EVAL_DISPLAY_BASE               Eval X11 display base (default: 280).
  EVAL_OUTPUT_ROOT                Unique eval output directory override.
  HEATMAP_CONTROL_DRY_RUN=1       Same as --dry-run.
USAGE
}

DRY_RUN="${HEATMAP_CONTROL_DRY_RUN:-0}"
while (( $# > 0 )); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      die "unknown argument: $1"
      ;;
  esac
done
if is_true "$DRY_RUN"; then
  DRY_RUN=1
else
  DRY_RUN=0
fi

canonical_under_root() {
  local candidate="$1"
  local resolved
  resolved="$(readlink -m -- "$candidate")"
  [[ "$resolved" != "$ALLOWED_ROOT" ]] || die "broad FJL root is not a valid target"
  case "${resolved}/" in
    "${ALLOWED_ROOT}/"*) ;;
    *) die "path escapes allowed root: $candidate -> $resolved" ;;
  esac
  printf '%s\n' "$resolved"
}

require_regular_file() {
  [[ -f "$1" && -s "$1" && ! -L "$1" ]] || die "missing, empty, or symlinked file: $1"
}

require_directory() {
  [[ -d "$1" && ! -L "$1" ]] || die "missing or symlinked directory: $1"
}

require_executable() {
  local raw="$1"
  local resolved
  resolved="$(readlink -f -- "$raw")"
  canonical_under_root "$resolved" >/dev/null
  [[ -f "$resolved" && -x "$resolved" ]] || die "missing executable: $raw"
}

CONFIG="${HEATMAP_CONTROL_CONFIG:-configs/heatmap_system1_control_8gpu.yaml}"
if [[ "$CONFIG" != /* ]]; then
  CONFIG="${REPO_ROOT}/${CONFIG}"
fi
CONFIG="$(canonical_under_root "$CONFIG")"

TRAINING_ROOTS_MANIFEST="${DAGGER_TRAINING_ROOTS_MANIFEST:-${ALLOWED_ROOT}/data/heatmap_system1_training_v1/rollout_control/round_000/full_train_4way_seed17/training_roots.json}"
TRAINING_ROOTS_MANIFEST="$(canonical_under_root "$TRAINING_ROOTS_MANIFEST")"

R2R_EXPERT_ROOT="${R2R_EXPERT_ROOT:-${ALLOWED_ROOT}/r2r_paronamic_data}"
R2R_EXPERT_ROOT="$(canonical_under_root "$R2R_EXPERT_ROOT")"

EXPERIMENT_ROOT="${HEATMAP_CONTROL_EXPERIMENT_ROOT:-${ALLOWED_ROOT}/model/output_heatmap_system1_control_v1}"
EXPERIMENT_ROOT="$(canonical_under_root "$EXPERIMENT_ROOT")"
HEATMAP_CONTROL_OUT_DIR="${HEATMAP_CONTROL_OUT_DIR:-${EXPERIMENT_ROOT}/runs}"
HEATMAP_CONTROL_OUT_DIR="$(canonical_under_root "$HEATMAP_CONTROL_OUT_DIR")"
HEATMAP_CONTROL_TB_DIR="${HEATMAP_CONTROL_TB_DIR:-${EXPERIMENT_ROOT}/tensorboard}"
HEATMAP_CONTROL_TB_DIR="$(canonical_under_root "$HEATMAP_CONTROL_TB_DIR")"
LAUNCHER_LOG_DIR="$(canonical_under_root "${EXPERIMENT_ROOT}/launcher_logs")"

HEATMAP_CONTROL_CKPT="${HEATMAP_CONTROL_CKPT:-}"
HEATMAP_CONTROL_CKPT_SHA256="${HEATMAP_CONTROL_CKPT_SHA256:-}"
[[ -n "$HEATMAP_CONTROL_CKPT" ]] || die "HEATMAP_CONTROL_CKPT is required"
[[ "$HEATMAP_CONTROL_CKPT_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "HEATMAP_CONTROL_CKPT_SHA256 must be 64 lowercase hex characters"
HEATMAP_CONTROL_CKPT="$(canonical_under_root "$HEATMAP_CONTROL_CKPT")"

GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29641}"
HEATMAP_CONTROL_EPOCH_SIZE="${HEATMAP_CONTROL_EPOCH_SIZE:-72000}"
HEATMAP_CONTROL_AUTO_RESUME="${HEATMAP_CONTROL_AUTO_RESUME:-0}"
HEATMAP_CONTROL_RESUME="${HEATMAP_CONTROL_RESUME:-}"
HEATMAP_CONTROL_AUTO_EVAL="${HEATMAP_CONTROL_AUTO_EVAL:-1}"
[[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || die "MASTER_PORT must be an integer"
MASTER_PORT=$((10#$MASTER_PORT))
(( MASTER_PORT >= 1024 && MASTER_PORT <= 65535 )) || die "MASTER_PORT is outside [1024,65535]"
[[ "$HEATMAP_CONTROL_EPOCH_SIZE" =~ ^[0-9]+$ ]] || die "HEATMAP_CONTROL_EPOCH_SIZE must be a positive integer"
HEATMAP_CONTROL_EPOCH_SIZE=$((10#$HEATMAP_CONTROL_EPOCH_SIZE))
(( HEATMAP_CONTROL_EPOCH_SIZE > 0 )) || die "HEATMAP_CONTROL_EPOCH_SIZE must be positive"
(( HEATMAP_CONTROL_EPOCH_SIZE == 72000 )) || die "formal heatmap-control recipe requires HEATMAP_CONTROL_EPOCH_SIZE=72000"
# 160=lcm(8 ranks * batch 1 * accumulation 4, denominator 10 of 50/20/30).
# This preserves exact source counts and full optimizer accumulation windows.
(( HEATMAP_CONTROL_EPOCH_SIZE % 160 == 0 )) || die "HEATMAP_CONTROL_EPOCH_SIZE must be a multiple of 160"
if is_true "$HEATMAP_CONTROL_AUTO_RESUME"; then
  HEATMAP_CONTROL_AUTO_RESUME=1
else
  HEATMAP_CONTROL_AUTO_RESUME=0
fi
case "${HEATMAP_CONTROL_AUTO_EVAL,,}" in
  1|true|yes|y|on) HEATMAP_CONTROL_AUTO_EVAL=1 ;;
  0|false|no|n|off) HEATMAP_CONTROL_AUTO_EVAL=0 ;;
  *) die "HEATMAP_CONTROL_AUTO_EVAL must be boolean-like" ;;
esac
if (( DRY_RUN == 1 && HEATMAP_CONTROL_AUTO_RESUME == 1 )); then
  die "dry-run cannot be combined with HEATMAP_CONTROL_AUTO_RESUME=1"
fi
if [[ -n "$HEATMAP_CONTROL_RESUME" ]]; then
  (( DRY_RUN == 0 )) || die "dry-run cannot be combined with HEATMAP_CONTROL_RESUME"
  (( HEATMAP_CONTROL_AUTO_RESUME == 0 )) || die "set only one of HEATMAP_CONTROL_AUTO_RESUME and HEATMAP_CONTROL_RESUME"
  HEATMAP_CONTROL_RESUME="$(canonical_under_root "$HEATMAP_CONTROL_RESUME")"
fi

require_directory "$REPO_ROOT"
require_directory "$NATIVE_INTERNNAV_MODEL"
require_directory "$R2R_EXPERT_ROOT"
require_executable "$PYTHON"
require_executable "$TORCHRUN"
require_regular_file "$NATIVE_MODEL_MANIFEST"
require_regular_file "$CONFIG"
require_regular_file "$TRAINING_ROOTS_MANIFEST"
require_regular_file "$HEATMAP_CONTROL_CKPT"
if [[ -n "$HEATMAP_CONTROL_RESUME" ]]; then
  require_regular_file "$HEATMAP_CONTROL_RESUME"
fi
if (( DRY_RUN == 0 && HEATMAP_CONTROL_AUTO_EVAL == 1 )); then
  require_regular_file "$CONTROL_EVAL_LAUNCHER"
  require_regular_file "$CONTROL_EVAL_SERVER"
fi

# Pin the released InternNav bytes once in the launcher, before torchrun forks
# eight ranks.  train.py receives only the verified scalar closure and must not
# re-hash the four large safetensor shards on every rank.
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required for native-model verification"
read -r ACTUAL_NATIVE_MODEL_MANIFEST_SHA256 _ < <(sha256sum -- "$NATIVE_MODEL_MANIFEST")
[[ "$ACTUAL_NATIVE_MODEL_MANIFEST_SHA256" == "$EXPECTED_NATIVE_MODEL_MANIFEST_SHA256" ]] || \
  die "native InternNav manifest SHA mismatch: expected=$EXPECTED_NATIVE_MODEL_MANIFEST_SHA256 actual=$ACTUAL_NATIVE_MODEL_MANIFEST_SHA256"
"$PYTHON" - "$NATIVE_MODEL_MANIFEST" "$NATIVE_INTERNNAV_MODEL" "$EXPECTED_NATIVE_MODEL_FILE_COUNT" <<'PY'
import re
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
model_root = Path(sys.argv[2])
expected_count = int(sys.argv[3])
lines = manifest.read_text(encoding="utf-8").splitlines()
if len(lines) != expected_count:
    raise SystemExit(
        f"native model manifest must contain {expected_count} rows, got {len(lines)}"
    )
paths = []
for index, line in enumerate(lines):
    match = re.fullmatch(r"[0-9a-f]{64}  (/.+)", line)
    if match is None:
        raise SystemExit(f"malformed native model manifest row {index + 1}")
    path = Path(match.group(1))
    if path.parent != model_root:
        raise SystemExit(f"native model manifest row escapes locked model root: {path}")
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"native model dependency is missing or symlinked: {path}")
    paths.append(path)
if len(set(paths)) != expected_count:
    raise SystemExit("native model manifest contains duplicate file paths")
PY
sha256sum -c "$NATIVE_MODEL_MANIFEST" >/dev/null

export HEATMAPVLN_NATIVE_DEPENDENCY_SCHEMA="$NATIVE_DEPENDENCY_SCHEMA"
export HEATMAPVLN_NATIVE_MODEL_PATH="$NATIVE_INTERNNAV_MODEL"
export HEATMAPVLN_NATIVE_MODEL_MANIFEST_PATH="$NATIVE_MODEL_MANIFEST"
export HEATMAPVLN_NATIVE_MODEL_MANIFEST_SHA256="$EXPECTED_NATIVE_MODEL_MANIFEST_SHA256"
export HEATMAPVLN_NATIVE_MODEL_FILE_COUNT="$EXPECTED_NATIVE_MODEL_FILE_COUNT"
export HEATMAPVLN_NATIVE_MODEL_VERIFIED=1

IFS=',' read -r -a GPU_LIST <<< "$GPU_DEVICES"
[[ "${#GPU_LIST[@]}" -eq "$NUM_GPUS" ]] || die "GPU_DEVICES must contain exactly 8 ids"
declare -A GPU_SEEN=()
for gpu in "${GPU_LIST[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU id: $gpu"
  [[ -z "${GPU_SEEN[$gpu]:-}" ]] || die "duplicate GPU id: $gpu"
  GPU_SEEN[$gpu]=1
done

EVAL_GPU_DEVICES="${EVAL_GPU_DEVICES:-$GPU_DEVICES}"
if (( DRY_RUN == 0 && HEATMAP_CONTROL_AUTO_EVAL == 1 )); then
  [[ "$EVAL_GPU_DEVICES" == "$GPU_DEVICES" ]] || die "automatic evaluation must use the exact training GPU_DEVICES in the same order"
  IFS=',' read -r -a EVAL_GPU_LIST <<< "$EVAL_GPU_DEVICES"
  [[ "${#EVAL_GPU_LIST[@]}" -eq "$NUM_GPUS" ]] || die "EVAL_GPU_DEVICES must contain exactly 8 ids"
  declare -A EVAL_GPU_SEEN=()
  for gpu in "${EVAL_GPU_LIST[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid eval GPU id: $gpu"
    [[ -z "${EVAL_GPU_SEEN[$gpu]:-}" ]] || die "duplicate eval GPU id: $gpu"
    EVAL_GPU_SEEN[$gpu]=1
  done
fi

# Validate the aggregate authority before exporting anything into the YAML.
if ! MANIFEST_OUTPUT="$("$PYTHON" - "$TRAINING_ROOTS_MANIFEST" "$ALLOWED_ROOT" "$MAX_COLLECTION_BYTES" "$EXPECTED_MANIFEST_SCHEMA" "$EXPECTED_NATIVE_PROTOCOL" "$NUM_DAGGER_ROOTS" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
allowed_root = Path(sys.argv[2]).resolve(strict=True)
max_bytes = int(sys.argv[3])
expected_schema = sys.argv[4]
expected_protocol = sys.argv[5]
expected_root_count = int(sys.argv[6])

def fail(message):
    raise SystemExit(message)

def require(condition, message):
    if not condition:
        fail(message)

payload = json.loads(manifest_path.read_text(encoding="utf-8"))
require(isinstance(payload, dict), "training roots payload is not a mapping")
require(payload.get("schema") == expected_schema, "wrong training roots schema")
require(payload.get("ready") is True, "training roots manifest is not ready")

partition = payload.get("partition")
require(isinstance(partition, dict), "partition contract is missing")
require(
    partition.get("num_shards") == expected_root_count,
    f"partition must contain exactly {expected_root_count} shards",
)
require(partition.get("unit") == "canonical_route", "partition unit is not canonical_route")

dataset = payload.get("dataset")
require(isinstance(dataset, dict), "dataset provenance is missing")
require(dataset.get("episodes") == 10819, "manifest is not the full R2R train cohort")

policy = payload.get("policy")
require(isinstance(policy, dict), "native policy contract is missing")
fingerprint = policy.get("fingerprint")
require(policy.get("mode") == "internnav_native", "policy is not internnav_native")
require(
    isinstance(fingerprint, str)
    and re.fullmatch(r"internnav-native-v1:[0-9a-f]{64}", fingerprint),
    "native policy fingerprint is malformed",
)
require(policy.get("protocol") == expected_protocol, "native protocol mismatch")
require(policy.get("system2") == "internnav_native_qwen", "System2 is not native Qwen")
require(
    policy.get("system1") == "internnav_native_nextdit_async",
    "System1 is not native NextDiT",
)
require(policy.get("external_checkpoint") is False, "DAgger policy used an external checkpoint")
require(policy.get("lora") is False, "DAgger policy used LoRA")
require(policy.get("adapter") is False, "DAgger policy used an adapter")
contract_sha = policy.get("collection_contract_invariant_sha256")
require(
    isinstance(contract_sha, str) and re.fullmatch(r"[0-9a-f]{64}", contract_sha),
    "collection contract invariant SHA is malformed",
)

capacity = payload.get("global_capacity")
require(isinstance(capacity, dict), "global capacity contract is missing")
limit = capacity.get("limit_bytes")
actual = capacity.get("actual_bytes")
require(type(limit) is int and 0 < limit <= max_bytes, "invalid collection byte limit")
require(type(actual) is int and 0 < actual < max_bytes, "collection is not below 300GB")
require(actual <= limit, "collection exceeds its declared byte limit")

mixture = payload.get("training_mixture")
require(
    isinstance(mixture, dict)
    and mixture.get("expert") == 0.5
    and mixture.get("dagger_normal") == 0.2
    and mixture.get("dagger_hard") == 0.3
    and mixture.get("basis") == "per_training_sample",
    "manifest mixture is not expert/normal/hard = 50/20/30",
)
storage = payload.get("storage_policy")
require(
    isinstance(storage, dict)
    and storage.get("copy_existing_images") is False
    and storage.get("persist_predicted_heatmaps") is False
    and storage.get("online_heatmap_generation") is True,
    "manifest does not require online, non-persisted heatmaps",
)

raw_roots = payload.get("collection_roots")
require(
    isinstance(raw_roots, list) and len(raw_roots) == expected_root_count,
    f"expected {expected_root_count} collection roots",
)
resolved_roots = []
for index, raw in enumerate(raw_roots):
    require(isinstance(raw, str) and raw, f"root {index} is invalid")
    original = Path(raw)
    require(original.is_absolute(), f"root {index} is not absolute")
    require(not original.is_symlink(), f"root {index} may not be a symlink")
    resolved = original.resolve(strict=True)
    require(resolved.is_dir(), f"root {index} is not a directory")
    try:
        resolved.relative_to(allowed_root)
    except ValueError:
        fail(f"root {index} escapes the allowed FJL root")
    require(resolved.name == f"shard_{index:02d}", f"root {index} has a non-canonical shard name")
    resolved_roots.append(resolved)
require(
    len(set(resolved_roots)) == expected_root_count,
    "collection roots contain duplicates",
)
collection_base = resolved_roots[0].parent
require(
    all(root.parent == collection_base for root in resolved_roots),
    "collection roots do not share one collection base",
)

shards = payload.get("shards")
require(
    isinstance(shards, list) and len(shards) == expected_root_count,
    "shard audit list is incomplete",
)
for index, (entry, root) in enumerate(zip(shards, resolved_roots, strict=True)):
    require(isinstance(entry, dict), f"shard audit {index} is invalid")
    require(entry.get("index") == index, f"shard audit index mismatch at {index}")
    require(entry.get("collection_root") == str(root), f"shard root mismatch at {index}")
require(
    sum(entry.get("actual_bytes", -1) for entry in shards) == actual,
    "per-shard bytes do not sum to global actual_bytes",
)

source_counts = payload.get("source_counts")
require(
    isinstance(source_counts, dict)
    and type(source_counts.get("dagger_normal")) is int
    and source_counts["dagger_normal"] > 0
    and type(source_counts.get("dagger_hard")) is int
    and source_counts["dagger_hard"] > 0,
    "both normal and hard DAgger sources must be non-empty",
)
sample_count = payload.get("sample_count")
require(type(sample_count) is int and sample_count > 0, "sample_count is empty")
require(
    sample_count
    == source_counts["dagger_normal"] + source_counts["dagger_hard"],
    "global sample_count does not equal global source_counts",
)
require(
    sum(entry.get("samples", -1) for entry in shards) == sample_count,
    "per-shard sample counts do not sum to global sample_count",
)
for source in ("dagger_normal", "dagger_hard"):
    require(
        sum(
            entry.get("source_counts", {}).get(source, -1)
            for entry in shards
        )
        == source_counts[source],
        f"per-shard {source} counts do not sum to the global count",
    )
for index, entry in enumerate(shards):
    shard_source_counts = entry.get("source_counts")
    require(
        isinstance(shard_source_counts, dict)
        and type(shard_source_counts.get("dagger_normal")) is int
        and type(shard_source_counts.get("dagger_hard")) is int
        and entry.get("samples")
        == shard_source_counts["dagger_normal"]
        + shard_source_counts["dagger_hard"],
        f"shard {index} sample/source counts disagree",
    )
require(
    sum(entry.get("cohort", {}).get("episodes", -1) for entry in shards)
    == dataset["episodes"],
    "per-shard cohort episodes do not cover the full dataset",
)

print(collection_base)
print(fingerprint)
for root in resolved_roots:
    print(root)
PY
)"; then
  die "training roots manifest validation failed"
fi
mapfile -t MANIFEST_FIELDS <<< "$MANIFEST_OUTPUT"
[[ "${#MANIFEST_FIELDS[@]}" -eq $((NUM_DAGGER_ROOTS + 2)) ]] || die "manifest validator returned an invalid export record"
DAGGER_COLLECTION_BASE="${MANIFEST_FIELDS[0]}"
DAGGER_POLICY_FINGERPRINT="${MANIFEST_FIELDS[1]}"
export DAGGER_COLLECTION_BASE DAGGER_POLICY_FINGERPRINT
export HEATMAP_SYSTEM1_COLLECTION_BASE="$DAGGER_COLLECTION_BASE"
export HEATMAP_SYSTEM1_POLICY_FINGERPRINT="$DAGGER_POLICY_FINGERPRINT"
for index in $(seq 0 $((NUM_DAGGER_ROOTS - 1))); do
  printf -v variable 'DAGGER_ROOT_%02d' "$index"
  printf -v "$variable" '%s' "${MANIFEST_FIELDS[$((index + 2))]}"
  export "$variable"
done

# Hash before deserialization, then enforce one weights-only heatmap_vln state.
"$PYTHON" - "$HEATMAP_CONTROL_CKPT" "$HEATMAP_CONTROL_CKPT_SHA256" <<'PY'
import hashlib
import re
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

path = Path(sys.argv[1])
expected = sys.argv[2]
digest = hashlib.sha256()
with path.open("rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
actual = digest.hexdigest()
if actual != expected:
    raise SystemExit(f"heatmap checkpoint SHA mismatch: expected={expected}, actual={actual}")

try:
    payload = torch.load(path, map_location="cpu", weights_only=True)
except TypeError as exc:
    raise SystemExit("runtime lacks mandatory torch.load(weights_only=True)") from exc
if not isinstance(payload, Mapping):
    raise SystemExit("heatmap checkpoint payload is not a mapping")
state = payload.get("trainable_state_dict")
if not isinstance(state, Mapping) or not state:
    raise SystemExit("checkpoint requires a non-empty trainable_state_dict")

seen = set()
for raw_name, value in state.items():
    if not isinstance(raw_name, str) or not raw_name:
        raise SystemExit("checkpoint contains an invalid state key")
    name = raw_name
    while name.startswith("module."):
        name = name[len("module."):]
    name = name.replace(".module.", ".")
    if name in seen:
        raise SystemExit(f"duplicate normalized checkpoint key: {name}")
    seen.add(name)
    if not name.startswith("heatmap_vln."):
        raise SystemExit(f"non-heatmap trainable tensor is forbidden: {name}")
    lowered = name.lower()
    if any(marker in lowered for marker in (
        "lora", "qwen", "system1", "system2", "nextdit",
        "adapter", "tokenizer", "control",
    )):
        raise SystemExit(f"forbidden dependency tensor: {name}")
    if not torch.is_tensor(value) or value.layout != torch.strided:
        raise SystemExit(f"checkpoint value is not a dense tensor: {name}")
    if not value.is_floating_point():
        raise SystemExit(f"checkpoint parameter is not floating point: {name}")
    if not torch.isfinite(value).all().item():
        raise SystemExit(f"checkpoint parameter is non-finite: {name}")
print(f"validated frozen heatmap checkpoint: sha256={actual} tensors={len(state)}")
PY

export INTERNNAV_MODEL_PATH="$NATIVE_INTERNNAV_MODEL"
export INTERNNAV_BACKBONE="$NATIVE_INTERNNAV_MODEL"
export HEATMAPVLN_INTERNNAV_MODEL_PATH="$NATIVE_INTERNNAV_MODEL"
export R2R_EXPERT_ROOT
export HEATMAP_CONTROL_OUT_DIR HEATMAP_CONTROL_TB_DIR
export HEATMAP_CONTROL_CKPT HEATMAP_CONTROL_CKPT_SHA256
export HEATMAP_CONTROL_EPOCH_SIZE

mkdir -p -- "$HEATMAP_CONTROL_OUT_DIR" "$HEATMAP_CONTROL_TB_DIR" "$LAUNCHER_LOG_DIR"
command -v flock >/dev/null 2>&1 || die "flock is required for exclusive train/eval ownership"
TRAIN_EVAL_LOCK="$(canonical_under_root "${HEATMAP_CONTROL_OUT_DIR}/.heatmap_system1_control_train_eval.lock")"
exec 9>"$TRAIN_EVAL_LOCK"
flock -n 9 || die "another heatmap-control train/eval job owns $HEATMAP_CONTROL_OUT_DIR"

# Expand the YAML and assert the immutable native path plus the complete
# heatmap-control/data/training contract before constructing eight models.
"$PYTHON" - "$CONFIG" "$NATIVE_INTERNNAV_MODEL" "$R2R_EXPERT_ROOT" "$HEATMAP_CONTROL_CKPT" "$HEATMAP_CONTROL_CKPT_SHA256" "$DAGGER_POLICY_FINGERPRINT" "$NUM_DAGGER_ROOTS" "$HEATMAP_CONTROL_EPOCH_SIZE" <<'PY'
import os
import sys

from src.config_schema import load_and_validate_config
from scripts.training.formal_heatmap_control_contract import (
    assert_formal_heatmap_control_no_training_eval,
)

(
    config_path,
    native_path,
    expert_root,
    checkpoint,
    checkpoint_sha,
    fingerprint,
    expected_root_count_raw,
    expected_epoch_size_raw,
) = sys.argv[1:]
expected_root_count = int(expected_root_count_raw)
expected_epoch_size = int(expected_epoch_size_raw)
assert expected_epoch_size == 72000
cfg = load_and_validate_config(config_path)
formal_eval_contract = assert_formal_heatmap_control_no_training_eval(
    cfg,
    require_formal_recipe=True,
)
assert formal_eval_contract["per_epoch_validation"] is False
assert formal_eval_contract["best_checkpoint_selection"] is False
assert formal_eval_contract["external_eval_checkpoint"] == "epoch_003.pth"
model = cfg["model"]
llm = model["llm"]
nextdit = model["action_head"]["nextdit"]
control = nextdit["heatmap_control"]
data = cfg["data"]
stage = cfg["training"]["stages"][0]
gpu = cfg["gpu"]

assert llm["model_path"] == native_path
assert nextdit["internnav_model_path"] == native_path
assert llm["use_lora"] is False
assert llm["gradient_checkpointing"] is False
assert nextdit.get("internnav_system1_path", "") == ""
assert nextdit.get("pretrained_system1_path") in (None, "")
assert nextdit.get("dav2_ckpt_path", "") == ""
assert nextdit.get("pano_latent_adapter") in (None, {})
assert control["enabled"] is True
assert control["heatmap_checkpoint_path"] == checkpoint
assert control["heatmap_checkpoint_sha256"] == checkpoint_sha
assert (
    control["token_dim"],
    control["control_dim"],
    control["num_heads"],
    control["coarse_size"],
    control["temporal_layers"],
    control["temporal_heads"],
    control["temporal_ffn_dim"],
) == (128, 128, 4, 8, 1, 4, 512)
assert model["heatmap"]["input_mode"] == "internnav_single_view"
assert data["root"] == expert_root
assert data["image_size"] == [384, 384]
assert data["dataset_type"] == "expert_dagger_mixture"
assert data["mixture"]["profile"] == "expert50_normal20_hard30"
assert data["mixture"]["epoch_size"] == expected_epoch_size
assert data["mixture"]["seed"] == 42
assert data["in_order"] is True
assert data["trajectory_dagger"]["expected_policy_mode"] == "internnav_native"
assert data["trajectory_dagger"]["expected_policy_fingerprint"] == fingerprint
expected_roots = [
    os.environ[f"DAGGER_ROOT_{index:02d}"]
    for index in range(expected_root_count)
]
assert data["trajectory_dagger"]["collection_roots"] == expected_roots
assert data["trajectory"]["panoramic_vlm_input"] is False
assert data["trajectory"]["load_single_view_history_frames"] is True
assert data["trajectory"]["load_traj_images"] is True
assert data["trajectory"]["load_history_heatmap"] is False
assert data["trajectory"]["trajectory_target_convention"] == "internnav_habitat"
assert data["trajectory"]["pixel_goal_direction"] == "front_down"
assert stage["trainable_modules"] == ["heatmap_tokenizer", "heatmap_control"]
assert stage["name"] == "heatmap_system1_control"
assert stage["epochs"] == 3
assert stage["strict_trainable_modules"] is True
assert stage["train_action"] is True
assert stage["train_heatmap"] is False
assert stage["train_lm"] is False
assert cfg["optim"]["batch_size"] == 1
assert cfg["optim"]["grad_accum_steps"] == 4
assert gpu["devices"] == list(range(8))
assert gpu["multi_gpu"]["enabled"] is True
assert cfg["loss"]["trajectory_weight"] == 1.0
assert cfg["loss"]["heatmap_weight"] == 0.0
assert cfg["loss"]["lm_weight"] == 0.0
print("validated heatmap -> native InternNav System1 control config:", config_path)
print("validated train-side no-eval/best-selection contract:", formal_eval_contract)
PY

export MACA_HOME="${MACA_HOME:-/opt/maca-3.3.0}"
export MACA_PATH="${MACA_PATH:-$MACA_HOME}"
export MACA_DIR="${MACA_DIR:-$MACA_PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export MCCL_IB_HCA="${MCCL_IB_HCA:-mlx5_0:0,mlx5_1:0,mlx5_4:0,mlx5_5:0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PATH="${QWEN25_ENV}/bin:${PATH}"
export CONDA_PREFIX="$QWEN25_ENV"
export CONDA_DEFAULT_ENV=qwen25
hash -r

RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"
if (( DRY_RUN == 1 )); then
  RUN_MODE="dry_run"
else
  RUN_MODE="train"
fi
LOG_FILE="${LOG_FILE:-${LAUNCHER_LOG_DIR}/${RUN_MODE}_${RUN_STAMP}.log}"

echo "[heatmap-control] config=$CONFIG"
echo "[heatmap-control] native_internnav=$NATIVE_INTERNNAV_MODEL"
echo "[heatmap-control] expert_root=$R2R_EXPERT_ROOT"
echo "[heatmap-control] training_roots_manifest=$TRAINING_ROOTS_MANIFEST"
echo "[heatmap-control] collection_base=$DAGGER_COLLECTION_BASE"
echo "[heatmap-control] policy_fingerprint=$DAGGER_POLICY_FINGERPRINT"
echo "[heatmap-control] frozen_heatmap=$HEATMAP_CONTROL_CKPT"
echo "[heatmap-control] frozen_heatmap_sha256=$HEATMAP_CONTROL_CKPT_SHA256"
echo "[heatmap-control] mixture_epoch_size=$HEATMAP_CONTROL_EPOCH_SIZE source_counts=expert:$((HEATMAP_CONTROL_EPOCH_SIZE * 5 / 10)),normal:$((HEATMAP_CONTROL_EPOCH_SIZE * 2 / 10)),hard:$((HEATMAP_CONTROL_EPOCH_SIZE * 3 / 10))"
echo "[heatmap-control] output=$HEATMAP_CONTROL_OUT_DIR"
echo "[heatmap-control] tensorboard=$HEATMAP_CONTROL_TB_DIR"
echo "[heatmap-control] mode=$RUN_MODE"
echo "[heatmap-control] gpu_devices=$GPU_DEVICES nproc=8 master=$MASTER_ADDR:$MASTER_PORT"
echo "[heatmap-control] effective_batch=32 (1/rank x 8 ranks x accum 4)"

TRAIN_ARGS=(
  scripts/train.py
  --config "$CONFIG"
  --distributed
)
if (( DRY_RUN == 1 )); then
  TRAIN_ARGS+=(--dry-run)
fi
if (( HEATMAP_CONTROL_AUTO_RESUME == 1 )); then
  TRAIN_ARGS+=(--auto-resume)
elif [[ -n "$HEATMAP_CONTROL_RESUME" ]]; then
  TRAIN_ARGS+=(--resume "$HEATMAP_CONTROL_RESUME")
fi

cd "$REPO_ROOT"
set -o pipefail
CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$TORCHRUN" \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

if (( DRY_RUN == 0 && HEATMAP_CONTROL_AUTO_EVAL == 1 )); then
  FINAL_CHECKPOINT="$(canonical_under_root "${HEATMAP_CONTROL_OUT_DIR}/latest/checkpoints/epoch_003.pth")"
  require_regular_file "$FINAL_CHECKPOINT"

  # Accept only the frozen-dependency-matched EMA control deployment from the
  # third complete epoch. No alternate checkpoint filename is substituted.
  FINAL_CHECKPOINT_SHA256="$("$PYTHON" - "$FINAL_CHECKPOINT" "$HEATMAP_CONTROL_CKPT_SHA256" "$DAGGER_POLICY_FINGERPRINT" <<'PY'
import os
import sys

from scripts.training.heatmap_control_deployment import (
    validate_heatmap_control_deployment_checkpoint,
)

report = validate_heatmap_control_deployment_checkpoint(
    sys.argv[1],
    expected_heatmap_sha256=sys.argv[2],
    expected_policy_fingerprint=sys.argv[3],
    expected_collection_roots=[
        os.environ[f"DAGGER_ROOT_{index:02d}"] for index in range(4)
    ],
    expected_epoch=3,
)
print(report["checkpoint_sha256"])
PY
)"
  [[ "$FINAL_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "final checkpoint validator returned an invalid SHA-256"

  CONTROL_EVAL_SERVER_SHA256="$(sha256sum -- "$CONTROL_EVAL_SERVER" | awk '{print $1}')"
  [[ "$CONTROL_EVAL_SERVER_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "control eval server SHA-256 was malformed"
  EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${EXPERIMENT_ROOT}/evaluation/r2r_val_unseen_epoch003_${FINAL_CHECKPOINT_SHA256:0:12}_plan${CONTROL_EVAL_SERVER_SHA256:0:12}}"
  EVAL_OUTPUT_ROOT="$(canonical_under_root "$EVAL_OUTPUT_ROOT")"
  EVAL_RPC_PORT_BASE="${EVAL_RPC_PORT_BASE:-51400}"
  EVAL_DISPLAY_BASE="${EVAL_DISPLAY_BASE:-280}"
  EVAL_CONTROL_MODE="${EVAL_CONTROL_MODE:-on}"
  [[ "$EVAL_CONTROL_MODE" == "on" ]] || die "automatic post-training evaluation requires EVAL_CONTROL_MODE=on"
  EVAL_X11_MODE=bundle

  export EVAL_GPU_DEVICES EVAL_RPC_PORT_BASE EVAL_DISPLAY_BASE EVAL_OUTPUT_ROOT
  export EVAL_CONTROL_MODE EVAL_X11_MODE
  export EVAL_HEATMAP_CHECKPOINT="$HEATMAP_CONTROL_CKPT"
  export EVAL_HEATMAP_SHA256="$HEATMAP_CONTROL_CKPT_SHA256"
  export EVAL_CONTROL_CHECKPOINT="$FINAL_CHECKPOINT"
  export EVAL_CONTROL_SHA256="$FINAL_CHECKPOINT_SHA256"
  unset EVAL_PREFLIGHT_ONLY EVAL_SMOKE_ONLY EVAL_SKIP_SMOKE EVAL_REUSE_XVFB

  echo "[heatmap-control] training completed; handing off full 8-GPU val_unseen evaluation"
  echo "[heatmap-control] eval_control_checkpoint=$FINAL_CHECKPOINT"
  echo "[heatmap-control] eval_control_sha256=$FINAL_CHECKPOINT_SHA256"
  echo "[heatmap-control] eval_output=$EVAL_OUTPUT_ROOT"
  bash "$CONTROL_EVAL_LAUNCHER"
fi

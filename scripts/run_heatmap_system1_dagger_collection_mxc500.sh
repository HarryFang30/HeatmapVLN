#!/usr/bin/bash -p
# Guarded R2R-train trajectory-DAgger collection wrapper for one MXC500 node.
#
# Round-0 learner rollout uses native InternNav System2/System1 models from the
# unified InternNav-Model checkpoint through the deterministic DAgger harness.
# The evaluator generates oracle-relabeled
# normal/hard samples online and commits atomic, deduplicated episode tar files.
# Heatmaps are generated later at training time; this wrapper never copies
# expert images or persists predicted heatmaps.

set -Eeuo pipefail

unset BASH_ENV ENV
readonly FIXED_SYSTEM_PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/maca-3.3.0/bin:/opt/maca-3.3.0/ompi/bin:/opt/mxdriver/bin"
export PATH="$FIXED_SYSTEM_PATH"

readonly ALLOWED_ROOT="/mnt/afs/liwenhao/agent/370910109"
readonly ABSOLUTE_HARD_LIMIT_BYTES=300000000000
readonly CAPACITY_RESERVE_BYTES=5000000000
readonly EXPECTED_REPO_ROOT="${ALLOWED_ROOT}/HeatmapVLN"
readonly EXPECTED_BASE_LAUNCHER="${EXPECTED_REPO_ROOT}/scripts/run_internnav_native_dagger_rpc_mxc500.sh"
readonly EXPECTED_QWEN25_PYTHON="${ALLOWED_ROOT}/envs/qwen25/bin/python"
readonly NATIVE_RPC_POLICY_MODE="internnav_native"
readonly NATIVE_PROTOCOL="internnav-native-joint-front-history-lookdown-v1"
readonly WRAPPER_SCHEMA="heatmap-system1-trajectory-dagger-wrapper-v4"

die() {
  echo "[heatmap-system1-dagger] ERROR: $*" >&2
  exit 1
}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

is_uint() {
  [[ "$1" =~ ^[0-9]+$ ]]
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

is_same_or_nested() {
  local candidate="$1"
  local parent="$2"
  [[ "$candidate" == "$parent" || "$candidate" == "${parent}/"* ]]
}

collection_bytes() {
  if [[ ! -e "$COLLECTION_ROOT" ]]; then
    echo 0
    return 0
  fi
  du -sb -- "$COLLECTION_ROOT" | awk '{print $1}'
}

export FJL_ROOT="$ALLOWED_ROOT"

ROUND_ID="${HEATMAP_SYSTEM1_DAGGER_ROUND:-0}"
is_uint "$ROUND_ID" || die "HEATMAP_SYSTEM1_DAGGER_ROUND must be a non-negative integer"
ROUND_NUMBER=$((10#$ROUND_ID))
printf -v ROUND_TAG 'round_%03d' "$ROUND_NUMBER"

canonical_expected_repo="$(canonical_under_allowed_root "$EXPECTED_REPO_ROOT" "fixed repository")"
REPO_ROOT="$(canonical_under_allowed_root "${HEATMAPVLN_REPO_ROOT:-$EXPECTED_REPO_ROOT}" "repository")"
[[ "$REPO_ROOT" == "$canonical_expected_repo" ]] || die \
  "HEATMAPVLN_REPO_ROOT is fixed to $canonical_expected_repo; got $REPO_ROOT"
canonical_expected_launcher="$(canonical_under_allowed_root "$EXPECTED_BASE_LAUNCHER" "fixed native InternNav launcher")"
BASE_LAUNCHER="$(canonical_under_allowed_root "${REPO_ROOT}/scripts/run_internnav_native_dagger_rpc_mxc500.sh" "native InternNav launcher")"
[[ "$BASE_LAUNCHER" == "$canonical_expected_launcher" ]] || die \
  "native base launcher is fixed to $canonical_expected_launcher; got $BASE_LAUNCHER"
if [[ -n "${HEATMAP_SYSTEM1_BASE_RPC_LAUNCHER:-}" ]]; then
  requested_launcher="$(canonical_under_allowed_root "$HEATMAP_SYSTEM1_BASE_RPC_LAUNCHER" "requested launcher")"
  [[ "$requested_launcher" == "$canonical_expected_launcher" ]] || die \
    "round-0 policy is fixed to $canonical_expected_launcher; custom base launcher is forbidden"
fi
canonical_expected_qwen25_python="$(canonical_under_allowed_root "$EXPECTED_QWEN25_PYTHON" "fixed qwen25 Python")"
if [[ -n "${QWEN25_PYTHON:-}" ]]; then
  requested_qwen25_python="$(canonical_under_allowed_root "$QWEN25_PYTHON" "requested qwen25 Python")"
  [[ "$requested_qwen25_python" == "$canonical_expected_qwen25_python" ]] || die \
    "QWEN25_PYTHON is fixed to $canonical_expected_qwen25_python; got $requested_qwen25_python"
fi
QWEN25_PYTHON="$canonical_expected_qwen25_python"
readonly QWEN25_PYTHON
VALIDATOR="$(canonical_under_allowed_root "${REPO_ROOT}/scripts/tools/validate_trajectory_dagger_collection.py" "collection validator")"
DATASET_ROOT="$(canonical_under_allowed_root "${HEATMAP_SYSTEM1_DATASET_ROOT:-${ALLOWED_ROOT}/data/heatmap_system1_training_v1}" "dataset root")"
COLLECTION_ROOT="$(canonical_under_allowed_root "${HEATMAP_SYSTEM1_COLLECTION_ROOT:-${ALLOWED_ROOT}/data/heatmap_system1_dagger_v1}" "collection root")"
CONTROL_ROOT="$(canonical_under_allowed_root "${HEATMAP_SYSTEM1_CONTROL_ROOT:-${DATASET_ROOT}/rollout_control/${ROUND_TAG}}" "control root")"
TRAIN_DATA_PATH="$(canonical_under_allowed_root "${HEATMAP_SYSTEM1_TRAIN_DATA_PATH:-${ALLOWED_ROOT}/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz}" "train data")"

EPISODE_LIST="${HEATMAP_SYSTEM1_EPISODE_LIST:-${STAGE3_EVAL_EPISODE_LIST:-}}"
if [[ -n "$EPISODE_LIST" ]]; then
  EPISODE_LIST="$(canonical_under_allowed_root "$EPISODE_LIST" "episode list")"
fi
MAX_EPISODES="${HEATMAP_SYSTEM1_MAX_EPISODES:-}"
if [[ -n "$MAX_EPISODES" ]]; then
  is_uint "$MAX_EPISODES" || die "HEATMAP_SYSTEM1_MAX_EPISODES must be a positive integer"
  (( MAX_EPISODES > 0 )) || die "HEATMAP_SYSTEM1_MAX_EPISODES must be > 0"
fi

ROLLOUT_MAX_STEPS="${HEATMAP_SYSTEM1_MAX_STEPS:-120}"
is_uint "$ROLLOUT_MAX_STEPS" || die "HEATMAP_SYSTEM1_MAX_STEPS must be a positive integer"
(( ROLLOUT_MAX_STEPS > 0 && ROLLOUT_MAX_STEPS <= 500 )) || die "HEATMAP_SYSTEM1_MAX_STEPS must be in [1, 500]"
MAX_SYSTEM2_CALLS="${HEATMAP_SYSTEM1_MAX_SYSTEM2_CALLS:-64}"
is_uint "$MAX_SYSTEM2_CALLS" || die "HEATMAP_SYSTEM1_MAX_SYSTEM2_CALLS must be a positive integer"
(( MAX_SYSTEM2_CALLS > 0 && MAX_SYSTEM2_CALLS <= 500 )) || die "HEATMAP_SYSTEM1_MAX_SYSTEM2_CALLS must be in [1, 500]"

MAX_BYTES="${HEATMAP_SYSTEM1_MAX_BYTES:-300000000000}"
CHECK_INTERVAL_S="${HEATMAP_SYSTEM1_SIZE_CHECK_INTERVAL_S:-30}"
is_uint "$MAX_BYTES" || die "HEATMAP_SYSTEM1_MAX_BYTES must be an integer byte count"
is_uint "$CHECK_INTERVAL_S" || die "HEATMAP_SYSTEM1_SIZE_CHECK_INTERVAL_S must be a positive integer"
(( MAX_BYTES > CAPACITY_RESERVE_BYTES )) || die "hard capacity must exceed ${CAPACITY_RESERVE_BYTES} bytes"
(( MAX_BYTES <= ABSOLUTE_HARD_LIMIT_BYTES )) || die "hard capacity must be <= ${ABSOLUTE_HARD_LIMIT_BYTES} bytes"
(( CHECK_INTERVAL_S > 0 )) || die "size check interval must be > 0"
MODULE_COMMIT_CEILING_BYTES=$((MAX_BYTES - CAPACITY_RESERVE_BYTES))
SOFT_STOP_BYTES="${HEATMAP_SYSTEM1_SOFT_STOP_BYTES:-$MODULE_COMMIT_CEILING_BYTES}"
is_uint "$SOFT_STOP_BYTES" || die "HEATMAP_SYSTEM1_SOFT_STOP_BYTES must be an integer byte count"
(( SOFT_STOP_BYTES > 0 )) || die "soft stop must be > 0"
(( SOFT_STOP_BYTES <= MODULE_COMMIT_CEILING_BYTES )) || die "soft stop must be <= evaluator commit ceiling ${MODULE_COMMIT_CEILING_BYTES}"

[[ -d "$REPO_ROOT" ]] || die "missing repository: $REPO_ROOT"
[[ -f "$BASE_LAUNCHER" ]] || die "missing base RPC launcher: $BASE_LAUNCHER"
[[ ! -L "$BASE_LAUNCHER" ]] || die "fixed native base launcher may not be a symlink: $BASE_LAUNCHER"
[[ "$BASE_LAUNCHER" == "$canonical_expected_launcher" ]] || die "native base launcher path changed after validation"
[[ -x "$QWEN25_PYTHON" ]] || die "missing qwen25 Python: $QWEN25_PYTHON"
[[ -f "$VALIDATOR" ]] || die "missing collection validator: $VALIDATOR"
[[ -s "$TRAIN_DATA_PATH" ]] || die "missing canonical R2R train dataset: $TRAIN_DATA_PATH"
if [[ -n "$EPISODE_LIST" ]]; then
  [[ -s "$EPISODE_LIST" ]] || die "missing or empty episode list: $EPISODE_LIST"
fi
if is_same_or_nested "$CONTROL_ROOT" "$COLLECTION_ROOT" || is_same_or_nested "$COLLECTION_ROOT" "$CONTROL_ROOT"; then
  die "collection and control roots must be disjoint: collection=$COLLECTION_ROOT control=$CONTROL_ROOT"
fi

COLLECTION_MANIFEST="$COLLECTION_ROOT/collection_manifest.json"
CONTROL_PROGRESS="$CONTROL_ROOT/progress.json"
WRAPPER_MANIFEST="$CONTROL_ROOT/collection_wrapper_manifest.json"
for protected_path in "$COLLECTION_MANIFEST" "$CONTROL_PROGRESS" "$WRAPPER_MANIFEST"; do
  [[ ! -L "$protected_path" ]] || die "refusing symlinked state file: $protected_path"
done
native_policy_closure_fingerprint() {
  local closure_path="$CONTROL_ROOT/native_policy_closure.json"
  [[ -f "$closure_path" && ! -L "$closure_path" ]] || die \
    "missing or symlinked native policy closure: $closure_path"
  "$QWEN25_PYTHON" - "$closure_path" "$NATIVE_RPC_POLICY_MODE" "$NATIVE_PROTOCOL" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_mode, expected_protocol = sys.argv[2:]
document = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "schema": "internnav-native-policy-closure-v1",
    "policy_backend": expected_mode,
    "native_protocol": expected_protocol,
}
mismatches = {
    key: {"expected": value, "actual": document.get(key)}
    for key, value in expected.items()
    if document.get(key) != value
}
fingerprint = document.get("policy_fingerprint")
if not isinstance(fingerprint, str) or not re.fullmatch(
    r"internnav-native-v1:[0-9a-f]{64}", fingerprint
):
    mismatches["policy_fingerprint"] = {
        "expected": "internnav-native-v1:<64 lowercase hex>",
        "actual": fingerprint,
    }
model_contract = document.get("model_contract")
if not isinstance(model_contract, dict):
    mismatches["model_contract"] = {"expected": "object", "actual": model_contract}
else:
    model_expected = {
        "system2": "internnav_native_qwen",
        "system1": "internnav_native_nextdit_async",
        "external_checkpoint": False,
        "lora": False,
        "adapter": False,
    }
    model_mismatches = {
        key: {"expected": value, "actual": model_contract.get(key)}
        for key, value in model_expected.items()
        if model_contract.get(key) != value
    }
    if model_mismatches:
        mismatches["model_contract"] = model_mismatches
if mismatches:
    raise SystemExit(f"native policy closure mismatch: {mismatches}")
print(fingerprint)
PY
}

verify_native_collection_provenance() {
  local require_ready="${1:-0}"
  local closure_path="$CONTROL_ROOT/native_policy_closure.json"
  [[ -f "$COLLECTION_MANIFEST" && ! -L "$COLLECTION_MANIFEST" ]] || die \
    "missing or symlinked collection manifest: $COLLECTION_MANIFEST"
  [[ -f "$closure_path" && ! -L "$closure_path" ]] || die \
    "missing or symlinked native policy closure: $closure_path"
  "$QWEN25_PYTHON" - \
    "$COLLECTION_MANIFEST" "$closure_path" "$NATIVE_RPC_POLICY_MODE" \
    "$NATIVE_PROTOCOL" "$require_ready" <<'PY'
import json
import re
import sys
from pathlib import Path

manifest_path, closure_path, expected_mode, expected_protocol, require_ready_raw = sys.argv[1:]
manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
closure = json.loads(Path(closure_path).read_text(encoding="utf-8"))
ready = manifest.get("ready")
if not isinstance(ready, bool):
    raise SystemExit("collection manifest ready flag must be boolean")
if require_ready_raw == "1" and not ready:
    raise SystemExit("sealed fast path requires collection manifest ready=true")
contract = manifest.get("contract")
if not isinstance(contract, dict):
    raise SystemExit("collection manifest contract must be an object")
fingerprint = contract.get("rpc_policy_fingerprint")
expected = {
    "rpc_policy_mode": expected_mode,
    "native_protocol": expected_protocol,
    "policy_fingerprint": fingerprint,
    "rpc_model_version": (
        f"internnav-native-r2r:{fingerprint}" if isinstance(fingerprint, str) else None
    ),
}
mismatches = {
    key: {"expected": value, "actual": contract.get(key)}
    for key, value in expected.items()
    if contract.get(key) != value
}
if not isinstance(fingerprint, str) or not re.fullmatch(
    r"internnav-native-v1:[0-9a-f]{64}", fingerprint
):
    mismatches["rpc_policy_fingerprint"] = {
        "expected": "internnav-native-v1:<64 lowercase hex>",
        "actual": fingerprint,
    }
closure_expected = {
    "schema": "internnav-native-policy-closure-v1",
    "policy_backend": expected_mode,
    "native_protocol": expected_protocol,
    "policy_fingerprint": fingerprint,
}
closure_mismatches = {
    key: {"expected": value, "actual": closure.get(key)}
    for key, value in closure_expected.items()
    if closure.get(key) != value
}
if closure_mismatches:
    mismatches["native_policy_closure"] = closure_mismatches
if mismatches:
    raise SystemExit(f"native collection provenance mismatch: {mismatches}")
print(fingerprint)
PY
}

initialize_wrapper_manifest() {
  "$QWEN25_PYTHON" - \
    "$WRAPPER_MANIFEST" "$WRAPPER_SCHEMA" "$REPO_ROOT" "$BASE_LAUNCHER" \
    "$NATIVE_RPC_POLICY_MODE" "$NATIVE_PROTOCOL" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
schema, repo_root, base_launcher, policy_mode, native_protocol = sys.argv[2:]
identity = {
    "split": os.environ["STAGE3_EVAL_DATASET_SPLIT"],
    "expected_dataset_episodes": int(os.environ["STAGE3_EVAL_EXPECTED_EPISODES"]),
    "train_data_path": os.environ["STAGE3_EVAL_DATA_PATH"],
    "episode_list": os.environ["STAGE3_EVAL_EPISODE_LIST"] or None,
    "max_episodes": (
        int(os.environ["STAGE3_EVAL_MAX_EPISODES"])
        if os.environ["STAGE3_EVAL_MAX_EPISODES"]
        else None
    ),
    "max_steps_per_episode": int(os.environ["STAGE3_EVAL_MAX_STEPS"]),
    "max_system2_calls_per_episode": int(os.environ["STAGE3_EVAL_MAX_SYSTEM2_CALLS"]),
    "collection_root": os.environ["STAGE3_EVAL_TRAJECTORY_DAGGER_ROOT"],
    "control_root": os.environ["STAGE3_EVAL_OUTPUT_PATH"],
    "round_id": int(os.environ["STAGE3_EVAL_TRAJECTORY_DAGGER_ROUND"]),
    "deterministic_rpc_sampling": True,
    "requested_policy": {
        "rpc_policy_mode": policy_mode,
        "native_protocol": native_protocol,
        "repo_root": repo_root,
        "base_launcher": base_launcher,
        "model_root": "/mnt/afs/liwenhao/agent/370910109/InternNav-Model",
        "system2": "internnav_native_qwen",
        "system1": "internnav_native_nextdit_async",
        "external_checkpoint": False,
        "lora": False,
        "adapter": False,
    },
    "candidate_quotas": {"normal": 1, "hard": 2},
    "jpeg_quality": 75,
    "hard_offpath_m": 0.75,
    "max_oracle_actions": 128,
    "min_history": 2,
    "hard_limit_bytes": int(os.environ["HEATMAP_SYSTEM1_MAX_BYTES"]),
    "soft_stop_bytes": int(os.environ["HEATMAP_SYSTEM1_SOFT_STOP_BYTES"]),
    "module_commit_reserve_bytes": 5_000_000_000,
    "persist_predicted_heatmaps": False,
    "copy_existing_expert_images": False,
    "save_duplicate_trajectory_trace": False,
    "privileged_policy_inputs": False,
}
if path.exists():
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != schema:
        raise SystemExit(
            f"wrapper manifest schema mismatch: {document.get('schema')!r} != {schema!r}"
        )
    if document.get("identity") != identity:
        raise SystemExit("wrapper manifest identity changed; refusing resume overwrite")
    status = document.get("verification_status")
    if status not in {
        "pending_native_verification",
        "native_preflight_passed",
        "native_runtime_verified",
        "sealed_native_verified",
    }:
        raise SystemExit(f"invalid wrapper verification_status: {status!r}")
    print(f"reusing wrapper manifest without overwrite: status={status}")
    raise SystemExit(0)

document = {
    "schema": schema,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "verification_status": "pending_native_verification",
    "identity": identity,
    "verified_policy": None,
}
payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
print("created pending wrapper manifest; native policy is not yet marked verified")
PY
}

mark_wrapper_manifest() {
  local next_status="$1"
  local fingerprint="${2:-}"
  "$QWEN25_PYTHON" - \
    "$WRAPPER_MANIFEST" "$WRAPPER_SCHEMA" "$next_status" "$fingerprint" \
    "$NATIVE_RPC_POLICY_MODE" "$NATIVE_PROTOCOL" \
    "$CONTROL_ROOT/native_policy_closure.json" <<'PY'
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
schema, next_status, fingerprint, policy_mode, native_protocol, closure_path = sys.argv[2:]
document = json.loads(path.read_text(encoding="utf-8"))
if document.get("schema") != schema:
    raise SystemExit("cannot update wrapper manifest with an unexpected schema")
allowed = {
    "pending_native_verification": {
        "native_preflight_passed",
        "native_runtime_verified",
        "sealed_native_verified",
    },
    "native_preflight_passed": {
        "native_preflight_passed",
        "native_runtime_verified",
        "sealed_native_verified",
    },
    "native_runtime_verified": {
        "native_runtime_verified",
        "sealed_native_verified",
    },
    "sealed_native_verified": {"sealed_native_verified"},
}
current = document.get("verification_status")
if next_status not in allowed.get(current, set()):
    raise SystemExit(f"invalid wrapper status transition: {current!r} -> {next_status!r}")
if not re.fullmatch(r"internnav-native-v1:[0-9a-f]{64}", fingerprint):
    raise SystemExit(f"invalid verified native fingerprint: {fingerprint!r}")
verified = {
    "rpc_policy_mode": policy_mode,
    "native_protocol": native_protocol,
    "policy_fingerprint": fingerprint,
    "policy_closure": closure_path,
}
existing_verified = document.get("verified_policy")
if existing_verified not in (None, verified):
    raise SystemExit("verified wrapper policy changed; refusing overwrite")
if current == next_status and existing_verified == verified:
    print(f"wrapper manifest already has status={next_status}")
    raise SystemExit(0)
document["verification_status"] = next_status
document["verified_policy"] = verified
document[f"{next_status}_at"] = datetime.now(timezone.utc).isoformat()
payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
print(f"wrapper manifest status={next_status}")
PY
}

if [[ -L "$COLLECTION_ROOT/.staging" ]]; then
  die "refusing symlinked collection staging directory: $COLLECTION_ROOT/.staging"
fi
stale_staging=""
if [[ -d "$COLLECTION_ROOT/.staging" ]]; then
  stale_staging="$(find "$COLLECTION_ROOT/.staging" -mindepth 1 -maxdepth 1 -printf '%f\n' 2>/dev/null | head -n 5 || true)"
fi
if [[ -n "$stale_staging" ]]; then
  {
    echo "[heatmap-system1-dagger] ERROR: stale incomplete staging entries found:"
    echo "$stale_staging"
    echo "[heatmap-system1-dagger] Recovery: stop every collector using this root; inspect each"
    echo "partial entry against episodes/<episode_key> and collection_progress.jsonl; then"
    echo "move confirmed-uncommitted entries to a quarantine directory under $ALLOWED_ROOT."
    echo "Nothing was deleted or overwritten. Re-run only after .staging is empty."
  } >&2
  exit 1
fi

collection_manifest_present=0
control_progress_present=0
[[ -f "$COLLECTION_MANIFEST" ]] && collection_manifest_present=1
[[ -f "$CONTROL_PROGRESS" ]] && control_progress_present=1
if (( collection_manifest_present == 1 )); then
  # The collection manifest is committed before the first episode finishes,
  # so progress.json may legitimately be absent after an early interruption.
  # The evaluator treats that as an empty progress ledger and strictly checks
  # the collection fingerprint when resuming.
  RESUME_MODE=1
elif (( collection_manifest_present == 0 && control_progress_present == 0 )); then
  RESUME_MODE=0
else
  die "control progress exists without collection_manifest.json. Restore the verified collection manifest or select new disjoint collection/control roots; this wrapper will not guess or delete state."
fi

manifest_ready=0
if (( collection_manifest_present == 1 )); then
  manifest_ready="$("$QWEN25_PYTHON" - "$COLLECTION_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ready = manifest.get("ready")
if not isinstance(ready, bool):
    raise SystemExit("collection manifest ready flag is not boolean")
print("1" if ready else "0")
PY
)"
fi

if (( RESUME_MODE == 0 )) && [[ -d "$COLLECTION_ROOT" ]]; then
  unexpected_fresh_entry="$(find "$COLLECTION_ROOT" -mindepth 1 -maxdepth 1 ! -name '.capacity.lock' -printf '%f\n' 2>/dev/null | head -n 1 || true)"
  [[ -z "$unexpected_fresh_entry" ]] || die "fresh collection root is non-empty without collection_manifest.json (first entry: $unexpected_fresh_entry); quarantine/recover it manually"
fi

current_bytes="$(collection_bytes)"
is_uint "$current_bytes" || die "could not determine collection size"
if [[ "$manifest_ready" != "1" ]]; then
  (( current_bytes < SOFT_STOP_BYTES )) || die "collection already uses ${current_bytes} bytes; soft stop is ${SOFT_STOP_BYTES}"
fi

MAX_GB_WHOLE=$((MAX_BYTES / 1000000000))
MAX_GB_FRACTION=$((MAX_BYTES % 1000000000))
printf -v MAX_GB '%d.%09d' "$MAX_GB_WHOLE" "$MAX_GB_FRACTION"

mkdir -p "$CONTROL_ROOT/logs"

export HEATMAPVLN_REPO_ROOT="$REPO_ROOT"
export QWEN25_PYTHON
export STAGE3_EVAL_DATA_PATH="$TRAIN_DATA_PATH"
export STAGE3_EVAL_DATASET_SPLIT=train
export STAGE3_EVAL_EXPECTED_EPISODES=10819
export STAGE3_EVAL_OUTPUT_PATH="$CONTROL_ROOT"
export STAGE3_EVAL_LOG_DIR="$CONTROL_ROOT/logs"
export STAGE3_EVAL_EPISODE_LIST="$EPISODE_LIST"
export STAGE3_EVAL_MAX_EPISODES="$MAX_EPISODES"
export STAGE3_EVAL_MAX_STEPS="$ROLLOUT_MAX_STEPS"
export STAGE3_EVAL_MAX_SYSTEM2_CALLS="$MAX_SYSTEM2_CALLS"
export STAGE3_EVAL_NUM_HISTORY=8
export STAGE3_EVAL_AUTO_STOP_DISTANCE=0.0
export STAGE3_EVAL_ALLOW_PRIVILEGED=0
export STAGE3_EVAL_ORACLE_SYSTEM2=0
export STAGE3_EVAL_PANO_RECENTER_BEFORE_SYSTEM1=0
export STAGE3_EVAL_RPC_REQUIRE_DETERMINISTIC_SAMPLING=1
export STAGE3_EVAL_RESUME="$RESUME_MODE"
export STAGE3_EVAL_OVERWRITE=0
export STAGE3_EVAL_SAVE_TRAJECTORY_STEPS=0
export STAGE3_EVAL_PREFLIGHT_ONLY="${STAGE3_EVAL_PREFLIGHT_ONLY:-0}"
export STAGE3_EVAL_COLLECT_TRAJECTORY_DAGGER=1
export STAGE3_EVAL_TRAJECTORY_DAGGER_ROOT="$COLLECTION_ROOT"
export STAGE3_EVAL_TRAJECTORY_DAGGER_ROUND="$ROUND_NUMBER"
export STAGE3_EVAL_TRAJECTORY_DAGGER_MAX_GB="$MAX_GB"
export STAGE3_EVAL_TRAJECTORY_DAGGER_NORMAL_QUOTA=1
export STAGE3_EVAL_TRAJECTORY_DAGGER_HARD_QUOTA=2
export STAGE3_EVAL_TRAJECTORY_DAGGER_JPEG_QUALITY=75
export STAGE3_EVAL_TRAJECTORY_DAGGER_HARD_OFFPATH_M=0.75
export STAGE3_EVAL_TRAJECTORY_DAGGER_MAX_ORACLE_ACTIONS=128
export STAGE3_EVAL_TRAJECTORY_DAGGER_MIN_HISTORY=2
export STAGE3_EVAL_TRAJECTORY_DAGGER_POLICY_FINGERPRINT="${HEATMAP_SYSTEM1_POLICY_FINGERPRINT:-}"
export HEATMAP_SYSTEM1_PERSIST_HEATMAPS=0
export HEATMAP_SYSTEM1_COPY_EXPERT_IMAGES=0
export HEATMAP_SYSTEM1_MAX_BYTES="$MAX_BYTES"
export HEATMAP_SYSTEM1_SOFT_STOP_BYTES="$SOFT_STOP_BYTES"

initialize_wrapper_manifest

if [[ "$manifest_ready" == "1" ]]; then
  [[ -d "$CONTROL_ROOT" ]] || die "sealed collection is missing its control root: $CONTROL_ROOT"
  closure_fingerprint="$(native_policy_closure_fingerprint)"
  sealed_fingerprint="$(verify_native_collection_provenance 1)"
  [[ "$closure_fingerprint" == "$sealed_fingerprint" ]] || die \
    "native policy closure fingerprint disagrees with sealed collection: $closure_fingerprint != $sealed_fingerprint"
  "$QWEN25_PYTHON" "$VALIDATOR" \
    --collection-root "$COLLECTION_ROOT" \
    --control-root "$CONTROL_ROOT" \
    --max-bytes "$MAX_BYTES"
  mark_wrapper_manifest sealed_native_verified "$sealed_fingerprint"
  echo "[heatmap-system1-dagger] sealed native collection provenance and contents verified; nothing to launch"
  exit 0
fi


echo "[heatmap-system1-dagger] split=train expected_episodes=10819"
echo "[heatmap-system1-dagger] train_data=$TRAIN_DATA_PATH"
echo "[heatmap-system1-dagger] episode_list=${EPISODE_LIST:-<all-train-episodes>}"
echo "[heatmap-system1-dagger] collection_root=$COLLECTION_ROOT"
echo "[heatmap-system1-dagger] control_root=$CONTROL_ROOT"
echo "[heatmap-system1-dagger] resume=$RESUME_MODE round=$ROUND_NUMBER"
echo "[heatmap-system1-dagger] max_steps=$ROLLOUT_MAX_STEPS max_system2_calls=$MAX_SYSTEM2_CALLS"
echo "[heatmap-system1-dagger] current_bytes=$current_bytes soft_stop=$SOFT_STOP_BYTES hard_limit=$MAX_BYTES"
echo "[heatmap-system1-dagger] policy=native InternNav System2 + native InternNav System1"
echo "[heatmap-system1-dagger] external_checkpoint=false lora=false adapter=false"
echo "[heatmap-system1-dagger] base_launcher=$BASE_LAUNCHER"

launcher_pid=""
monitor_failed=0
capacity_reached=0

cleanup() {
  if [[ -n "$launcher_pid" ]] && kill -0 "$launcher_pid" 2>/dev/null; then
    kill -TERM "$launcher_pid" 2>/dev/null || true
    wait "$launcher_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

/usr/bin/bash -p "$BASE_LAUNCHER" &
launcher_pid="$!"

while kill -0 "$launcher_pid" 2>/dev/null; do
  used_bytes="$(collection_bytes || true)"
  if ! is_uint "$used_bytes"; then
    echo "[heatmap-system1-dagger] collection size check failed; stopping fail-closed" >&2
    monitor_failed=1
    kill -TERM "$launcher_pid" 2>/dev/null || true
    break
  fi
  if (( used_bytes >= SOFT_STOP_BYTES )); then
    echo "[heatmap-system1-dagger] soft storage stop reached: ${used_bytes} >= ${SOFT_STOP_BYTES}" >&2
    HEATMAP_SYSTEM1_USED_BYTES="$used_bytes" "$QWEN25_PYTHON" - "$CONTROL_ROOT/storage_cap_reached.json" <<'PY' || true
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = json.dumps({
    "stopped_at": datetime.now(timezone.utc).isoformat(),
    "used_bytes": int(os.environ["HEATMAP_SYSTEM1_USED_BYTES"]),
    "soft_stop_bytes": int(os.environ["HEATMAP_SYSTEM1_SOFT_STOP_BYTES"]),
    "hard_limit_bytes": int(os.environ["HEATMAP_SYSTEM1_MAX_BYTES"]),
}, indent=2, sort_keys=True) + "\n"
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
temporary.write_text(payload, encoding="utf-8")
os.replace(temporary, path)
PY
    capacity_reached=1
    kill -TERM "$launcher_pid" 2>/dev/null || true
    break
  fi
  sleep "$CHECK_INTERVAL_S"
done

set +e
wait "$launcher_pid"
launcher_status="$?"
set -e
launcher_pid=""

final_bytes="$(collection_bytes)"
is_uint "$final_bytes" || die "could not determine final collection size"
echo "[heatmap-system1-dagger] final_bytes=$final_bytes launcher_status=$launcher_status"
(( final_bytes <= MAX_BYTES )) || die "absolute hard limit breached: ${final_bytes} > ${MAX_BYTES}"

if (( monitor_failed == 1 )); then
  exit 74
fi
if (( capacity_reached == 1 )); then
  echo "[heatmap-system1-dagger] stopped safely at the configured soft capacity ceiling" >&2
  exit 75
fi
if (( launcher_status != 0 )); then
  exit "$launcher_status"
fi

if is_true "$STAGE3_EVAL_PREFLIGHT_ONLY"; then
  preflight_fingerprint="$(native_policy_closure_fingerprint)"
  mark_wrapper_manifest native_preflight_passed "$preflight_fingerprint"
  echo "[heatmap-system1-dagger] native base preflight passed; collection was not launched or sealed"
  exit 0
fi

runtime_closure_fingerprint="$(native_policy_closure_fingerprint)"
runtime_fingerprint="$(verify_native_collection_provenance 0)"
[[ "$runtime_closure_fingerprint" == "$runtime_fingerprint" ]] || die \
  "native policy closure fingerprint disagrees with runtime collection: $runtime_closure_fingerprint != $runtime_fingerprint"
mark_wrapper_manifest native_runtime_verified "$runtime_fingerprint"
echo "[heatmap-system1-dagger] native runtime provenance verified; validating and sealing exact cohort"
"$QWEN25_PYTHON" "$VALIDATOR" \
  --collection-root "$COLLECTION_ROOT" \
  --control-root "$CONTROL_ROOT" \
  --max-bytes "$MAX_BYTES" \
  --seal
sealed_fingerprint="$(verify_native_collection_provenance 1)"
[[ "$sealed_fingerprint" == "$runtime_fingerprint" ]] || die \
  "native policy fingerprint changed while sealing: $runtime_fingerprint -> $sealed_fingerprint"
mark_wrapper_manifest sealed_native_verified "$sealed_fingerprint"
echo "[heatmap-system1-dagger] exact cohort sealed with verified native policy provenance"
exit 0

#!/usr/bin/env bash
set -Eeuo pipefail

readonly TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly STAGE_ROOT="$(cd "${TEST_DIR}/.." && pwd)"
readonly LAUNCHER="${STAGE_ROOT}/scripts/run_past_plan_action_8gpu_mxc500.sh"
readonly WEBSITE="${STAGE_ROOT}/PPA_8GPU_WEBSITE_SUBMISSION.md"
readonly TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ppa-launcher-test.XXXXXX")"
readonly ALLOWED_ROOT="${TEST_ROOT}/allowed"
readonly REPO_ROOT="${ALLOWED_ROOT}/repo"
readonly FAKE_LOG="${ALLOWED_ROOT}/fake-python.log"

cleanup() {
  rm -r "$TEST_ROOT"
}
trap cleanup EXIT

mkdir -p \
  "$REPO_ROOT/configs" \
  "$REPO_ROOT/scripts/tools" \
  "$REPO_ROOT/src" \
  "$ALLOWED_ROOT/env/bin" \
  "$ALLOWED_ROOT/native-model" \
  "$ALLOWED_ROOT/expert-data/train_scene/clip_000001" \
  "$ALLOWED_ROOT/expert-data/scene_002/clip_000002" \
  "$ALLOWED_ROOT/amb3r-cache/_control" \
  "$ALLOWED_ROOT/model"

printf 'stage1\n' > "$REPO_ROOT/configs/ppa_stage1.yaml"
printf 'stage2\n' > "$REPO_ROOT/configs/ppa_stage2.yaml"
printf 'schema\n' > "$REPO_ROOT/src/config_schema.py"
printf 'train\n' > "$REPO_ROOT/scripts/train.py"
printf 'checker\n' > "$REPO_ROOT/scripts/tools/checker.py"
printf 'checkpoint\n' > "$ALLOWED_ROOT/model/past-best.pth"
printf 'stage1 checkpoint\n' > "$ALLOWED_ROOT/model/stage1-best.pth"
printf '%s\n' \
  '{"schema":"heatmapvln-amb3r-endpoint-pose-cache-ready-v2","complete":true}' \
  > "$ALLOWED_ROOT/amb3r-cache/_control/cache.ready.json"

cat > "$ALLOWED_ROOT/env/bin/python" <<'FAKE_PYTHON'
#!/usr/bin/env bash
set -Eeuo pipefail
printf 'CACHE|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
  "$HF_HOME" "$HF_HUB_CACHE" "$HUGGINGFACE_HUB_CACHE" \
  "$HUGGINGFACE_ASSETS_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" \
  "$PPA_RUNTIME_CACHE_ROOT" \
  >> "$PPA_FAKE_LOG"
printf 'ARGS|PDBC=%s' "${PYTHONDONTWRITEBYTECODE:-}" >> "$PPA_FAKE_LOG"
printf '|%s' "$@" >> "$PPA_FAKE_LOG"
printf '\n' >> "$PPA_FAKE_LOG"

if [[ "|$*|" == *"|run-best|"* || " $* " == *" run-best "* ]]; then
  kind=""
  while (($#)); do
    if [[ "$1" == "--kind" ]]; then
      shift
      kind="$1"
      break
    fi
    shift
  done
  if [[ "$kind" == "stage1" ]]; then
    best="$PPA_STAGE1_OUTPUT_ROOT/run_fake/checkpoints/best.pth"
  elif [[ "$kind" == "stage2" ]]; then
    best="$PPA_STAGE2_OUTPUT_ROOT/run_fake/checkpoints/best.pth"
  else
    exit 91
  fi
  mkdir -p "$(dirname "$best")"
  printf 'fake checkpoint\n' > "$best"
  printf '%s\n' "$best"
fi
FAKE_PYTHON
chmod +x "$ALLOWED_ROOT/env/bin/python"

common_env=(
  PPA_ALLOWED_ROOT="$ALLOWED_ROOT"
  PPA_REPO_ROOT="$REPO_ROOT"
  PPA_QWEN_PYTHON="$ALLOWED_ROOT/env/bin/python"
  INTERNNAV_MODEL_PATH="$ALLOWED_ROOT/native-model"
  PPA_DATA_ROOT="$ALLOWED_ROOT/expert-data"
  PPA_AMB3R_CACHE_ROOT="$ALLOWED_ROOT/amb3r-cache"
  PPA_PAST_INIT_CHECKPOINT="$ALLOWED_ROOT/model/past-best.pth"
  PPA_STAGE1_CONFIG="$REPO_ROOT/configs/ppa_stage1.yaml"
  PPA_STAGE2_CONFIG="$REPO_ROOT/configs/ppa_stage2.yaml"
  PPA_CONTRACT_CHECKER="$REPO_ROOT/scripts/tools/checker.py"
  PPA_FAKE_LOG="$FAKE_LOG"
  PPA_WAIT_FOR_CACHE=0
  PPA_PREWARM_IMPORTS=1
  PPA_GPU_DEVICES=0,1,2,3,4,5,6,7
)

# Keep the copy/paste website contracts tied to the formal flat R2R scene root,
# full two-split cache coverage, the eight-GPU wrapper, and a deliberately
# runtime-selected Past initializer. These assertions are exact enough to catch a
# parent-root or smoke-subset regression in documentation.
grep -F \
  'export PPA_DATA_ROOT=/mnt/afs/lixiaoou/intern/fjl/r2r_paronamic_data/train' \
  "$WEBSITE" >/dev/null
grep -F 'ALLOWED_ROOT="$PPA_ALLOWED_ROOT" \' "$WEBSITE" >/dev/null
grep -F 'DATASET_ROOT="$PPA_DATA_ROOT" \' "$WEBSITE" >/dev/null
grep -F 'CACHE_ROOT="$PPA_AMB3R_CACHE_ROOT" \' "$WEBSITE" >/dev/null
grep -F 'SPLITS=train,val \' "$WEBSITE" >/dev/null
grep -F 'MAX_CLIPS_PER_SPLIT=0 \' "$WEBSITE" >/dev/null
grep -F 'bash scripts/run_amb3r_pose_training_cache_8gpu_mxc500.sh' \
  "$WEBSITE" >/dev/null
grep -F \
  'export PPA_PAST_INIT_CHECKPOINT=/mnt/afs/lixiaoou/intern/fjl/model/REPLACE_WITH_FINAL_AMB3R_HISTORY_ADAPT_BEST/checkpoints/best.pth' \
  "$WEBSITE" >/dev/null
grep -F \
  'PPA_DATA_ROOT="${PPA_DATA_ROOT:-/mnt/afs/lixiaoou/intern/fjl/r2r_paronamic_data/train}"' \
  "$LAUNCHER" >/dev/null

run_success_case() {
  local workers="$1" prefetch="$2" output_name="$3"
  local output_root="$ALLOWED_ROOT/$output_name"
  local canonical_output_root="$(realpath "$ALLOWED_ROOT")/$output_name"
  : > "$FAKE_LOG"
  env \
    "${common_env[@]}" \
    PPA_OUTPUT_ROOT="$output_root" \
    PPA_NUM_WORKERS="$workers" \
    PPA_PREFETCH_FACTOR="$prefetch" \
    bash "$LAUNCHER" > "$TEST_ROOT/${output_name}.out" 2>&1

  local expected_runtime="$canonical_output_root/_runtime_cache"
  local first_line
  first_line="$(sed -n '1p' "$FAKE_LOG")"
  [[ "$first_line" == \
    "CACHE|$expected_runtime/huggingface|$expected_runtime/huggingface/hub|$expected_runtime/huggingface/hub|$expected_runtime/huggingface/assets|$expected_runtime/huggingface/datasets|$expected_runtime/huggingface/transformers|$expected_runtime/torch|$expected_runtime/torch_extensions|$expected_runtime/torch_inductor|$expected_runtime/xdg|$expected_runtime/matplotlib|$expected_runtime/triton|$expected_runtime" \
  ]] || {
    printf 'unexpected first-Python cache environment: %s\n' "$first_line" >&2
    return 1
  }

  grep -F 'torch.distributed.run' "$FAKE_LOG" > "$TEST_ROOT/${output_name}.training"
  [[ "$(wc -l < "$TEST_ROOT/${output_name}.training" | tr -d ' ')" == "2" ]]
  [[ "$(grep -Fc -- '--nproc_per_node=8' "$TEST_ROOT/${output_name}.training")" == "2" ]]
  grep -F 'world=8 per_rank_batch=1 accum=1 effective_batch=8' \
    "$TEST_ROOT/${output_name}.out" >/dev/null
  local prewarm_pattern='ARGS|PDBC=1|-c|import scripts.train'
  [[ "$(grep -Fc "$prewarm_pattern" "$FAKE_LOG")" == "1" ]] || {
    printf 'scripts.train prewarm did not occur exactly once\n' >&2
    return 1
  }
  local cache_contract_line prewarm_line first_distributed_line
  cache_contract_line="$(
    grep -nF '|cache|--cache-root|' "$FAKE_LOG" | head -n 1 | cut -d: -f1
  )"
  prewarm_line="$(
    grep -nF "$prewarm_pattern" "$FAKE_LOG" | cut -d: -f1
  )"
  first_distributed_line="$(
    grep -nF 'torch.distributed.run' "$FAKE_LOG" | head -n 1 | cut -d: -f1
  )"
  (( cache_contract_line < prewarm_line && prewarm_line < first_distributed_line )) \
    || {
      printf 'prewarm order invalid: cache=%s prewarm=%s distributed=%s\n' \
        "$cache_contract_line" "$prewarm_line" "$first_distributed_line" >&2
      return 1
    }
  if (( workers == 0 )); then
    if grep -F -- '--prefetch-factor' "$TEST_ROOT/${output_name}.training" >/dev/null; then
      printf 'num_workers=0 unexpectedly passed --prefetch-factor\n' >&2
      return 1
    fi
  else
    grep -F -- "--prefetch-factor|$prefetch" \
      "$TEST_ROOT/${output_name}.training" >/dev/null
  fi
}

run_success_case 0 7 output-workers-zero
run_success_case 2 7 output-workers-two

# A failed Stage 2 can restart from a completed Stage-1 deployment checkpoint
# without rerunning Stage 1 or restoring its optimizer/scheduler.
: > "$FAKE_LOG"
env \
  "${common_env[@]}" \
  PPA_RUN_MODE=stage2_only \
  PPA_STAGE1_BEST_CHECKPOINT="$ALLOWED_ROOT/model/stage1-best.pth" \
  PPA_OUTPUT_ROOT="$ALLOWED_ROOT/output-stage2-only" \
  PPA_NUM_WORKERS=0 \
  bash "$LAUNCHER" > "$TEST_ROOT/stage2-only.out" 2>&1
grep -F 'torch.distributed.run' "$FAKE_LOG" > "$TEST_ROOT/stage2-only.training"
[[ "$(wc -l < "$TEST_ROOT/stage2-only.training" | tr -d ' ')" == "1" ]]
grep -F '|--config|' "$TEST_ROOT/stage2-only.training" | grep -F 'ppa_stage2.yaml' >/dev/null
grep -F '|checkpoint|--path|' "$FAKE_LOG" | grep -F '|--kind|stage1' >/dev/null
if grep -F '|run-best|--output-root|' "$FAKE_LOG" | grep -F '|--kind|stage1' >/dev/null; then
  printf 'stage2-only mode unexpectedly resolved or ran Stage 1\n' >&2
  exit 1
fi
grep -F 'Stage-2-only retry: using validated Stage-1 best' \
  "$TEST_ROOT/stage2-only.out" >/dev/null

# Exactly eight unique numeric devices are required before any Python starts.
: > "$FAKE_LOG"
if env \
  "${common_env[@]}" \
  PPA_GPU_DEVICES=0,1,2,3,4,5,6 \
  PPA_OUTPUT_ROOT="$ALLOWED_ROOT/output-seven-gpus" \
  bash "$LAUNCHER" > "$TEST_ROOT/seven-gpus.out" 2>&1; then
  printf 'seven-GPU device list unexpectedly passed\n' >&2
  exit 1
fi
grep -F 'PPA_GPU_DEVICES must contain exactly eight comma-separated IDs' \
  "$TEST_ROOT/seven-gpus.out" >/dev/null
[[ ! -s "$FAKE_LOG" ]] || {
  printf 'GPU-count rejection occurred after Python started\n' >&2
  exit 1
}

# The prewarm switch is strict boolean state, rejected before any Python call.
: > "$FAKE_LOG"
if env \
  "${common_env[@]}" \
  PPA_PREWARM_IMPORTS=2 \
  PPA_OUTPUT_ROOT="$ALLOWED_ROOT/output-invalid-prewarm" \
  bash "$LAUNCHER" > "$TEST_ROOT/invalid-prewarm.out" 2>&1; then
  printf 'invalid PPA_PREWARM_IMPORTS unexpectedly passed\n' >&2
  exit 1
fi
grep -F 'PPA_PREWARM_IMPORTS must be 0 or 1' \
  "$TEST_ROOT/invalid-prewarm.out" >/dev/null
[[ ! -s "$FAKE_LOG" ]] || {
  printf 'invalid prewarm rejection occurred after Python started\n' >&2
  exit 1
}

# An existing lexical symlink below the allowed root may not redirect any
# formal path outside it. This must fail before the fake Python is called.
mkdir -p "$TEST_ROOT/outside-model"
ln -s "$TEST_ROOT/outside-model" "$ALLOWED_ROOT/escaped-model-link"
: > "$FAKE_LOG"
if env \
  "${common_env[@]}" \
  INTERNNAV_MODEL_PATH="$ALLOWED_ROOT/escaped-model-link" \
  PPA_OUTPUT_ROOT="$ALLOWED_ROOT/output-escape" \
  bash "$LAUNCHER" > "$TEST_ROOT/escape.out" 2>&1; then
  printf 'outside-root symlink unexpectedly passed\n' >&2
  exit 1
fi
grep -F 'INTERNNAV_MODEL_PATH escapes PPA_ALLOWED_ROOT' \
  "$TEST_ROOT/escape.out" >/dev/null
[[ ! -s "$FAKE_LOG" ]] || {
  printf 'path-escape rejection occurred after Python started\n' >&2
  exit 1
}

# Output/data/cache/init are pairwise isolated scopes.
mkdir -p "$ALLOWED_ROOT/expert-data/nested-cache"
: > "$FAKE_LOG"
if env \
  "${common_env[@]}" \
  PPA_AMB3R_CACHE_ROOT="$ALLOWED_ROOT/expert-data/nested-cache" \
  PPA_OUTPUT_ROOT="$ALLOWED_ROOT/output-overlap" \
  bash "$LAUNCHER" > "$TEST_ROOT/overlap.out" 2>&1; then
  printf 'overlapping data/cache scopes unexpectedly passed\n' >&2
  exit 1
fi
grep -F 'formal path scopes overlap' "$TEST_ROOT/overlap.out" >/dev/null
[[ ! -s "$FAKE_LOG" ]] || {
  printf 'overlap rejection occurred after Python started\n' >&2
  exit 1
}

printf 'launcher hardening self-test: passed\n'

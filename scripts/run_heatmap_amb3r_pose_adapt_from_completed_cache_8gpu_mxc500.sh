#!/usr/bin/env bash
# Resume after the completed four-shard AMB3R endpoint export and launch the
# formal eight-GPU pose-domain adaptation. This script deliberately performs
# no smoke and never reruns AMB3R.
set -Eeuo pipefail

REPO_ROOT="/mnt/afs/lixiaoou/intern/fjl/HeatmapVLN"
ALLOWED_ROOT="/mnt/afs/lixiaoou/intern/fjl"
QWEN25_PYTHON="${ALLOWED_ROOT}/envs/qwen25/bin/python"

DATASET_ROOT="${ALLOWED_ROOT}/data/heatmap_randomwalk_train_v1"
# The shard count records how the cache was produced; it does not constrain
# DDP world size. All 6000 clips are present, so eight-rank training can consume
# this completed four-shard cache without recomputing any pose.
CACHE_ROOT="${ALLOWED_ROOT}/data/heatmap_randomwalk_amb3r_endpoint_cache_v2_4gpu"
PLAN_PATH="${CACHE_ROOT}/_control/plan.json"
CACHE_READY="${CACHE_ROOT}/_control/cache.ready.json"
CONFIG="configs/train_heatmap_amb3r_pose_adapt_8gpu.yaml"
INIT_CKPT="${ALLOWED_ROOT}/model/output_heatmap_internnav_single_view_v1_4gpu/runs/run_20260803_143402/checkpoints/best.pth"
EXPERIMENT_ROOT="${ALLOWED_ROOT}/model/output_heatmap_amb3r_pose_adapt_endpoint_v2_8gpu"
GPU_DEVICES="0,1,2,3,4,5,6,7"
MASTER_PORT="${MASTER_PORT:-29643}"
MAP_INIT_WINDOW=20
MAP_EVERY=8

cd "$REPO_ROOT"

require_file() {
  [[ -s "$1" ]] || {
    echo "[pose-adapt-resume-8gpu] missing non-empty file: $1" >&2
    exit 2
  }
}
require_dir() {
  [[ -d "$1" ]] || {
    echo "[pose-adapt-resume-8gpu] missing directory: $1" >&2
    exit 2
  }
}
require_under_allowed_root() {
  local resolved
  resolved="$(realpath -m -- "$1")"
  case "$resolved" in
    "$ALLOWED_ROOT"/*) ;;
    *)
      echo "[pose-adapt-resume-8gpu] path escapes $ALLOWED_ROOT: $resolved" >&2
      exit 2
      ;;
  esac
}

require_file "$QWEN25_PYTHON"
require_file "$PLAN_PATH"
require_file "$CONFIG"
require_file "$INIT_CKPT"
require_dir "$DATASET_ROOT"
require_dir "$CACHE_ROOT"
for shard_id in 00 01 02 03; do
  require_file "${CACHE_ROOT}/_control/shard_${shard_id}.ready.json"
done
for path in   "$DATASET_ROOT"   "$CACHE_ROOT"   "$INIT_CKPT"   "$EXPERIMENT_ROOT"; do
  require_under_allowed_root "$path"
done

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
VALIDATION_LOG_ROOT="${CACHE_ROOT}/_control/logs/resume_8gpu_${RUN_STAMP}"
mkdir -p "$VALIDATION_LOG_ROOT"

echo "[pose-adapt-resume-8gpu] validating all 6000 endpoint caches"
"$QWEN25_PYTHON" scripts/amb3r_vo/validate_training_cache.py   --plan "$PLAN_PATH"   --workers 24   --require-shard-ready   --write-ready   --allowed-root "$ALLOWED_ROOT"   2>&1 | tee "$VALIDATION_LOG_ROOT/final_validation.log"

require_file "$CACHE_READY"

# Fail closed before loading InternNav: this is the exact completed formal
# cache, not a partial/smoke cache and not a cache generated from another root.
"$QWEN25_PYTHON" - "$PLAN_PATH" "$CACHE_READY" "$DATASET_ROOT" "$CACHE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ready = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
dataset_root = str(Path(sys.argv[3]).resolve())
cache_root = str(Path(sys.argv[4]).resolve())

assert plan["schema"] == "heatmapvln-amb3r-endpoint-pose-cache-plan-v2"
assert ready["schema"] == "heatmapvln-amb3r-endpoint-pose-cache-ready-v2"
assert ready["complete"] is True
assert plan["dataset_root"] == dataset_root == ready["dataset_root"]
assert plan["cache_root"] == cache_root == ready["cache_root"]
assert plan["clip_count"] == ready["clips_total"] == 6000
assert plan["frame_count"] == ready["frames_total"] == 759461
assert plan["query_rows"] == ready["query_rows_total"] == 88548
assert len(plan["shards"]) == 4
assert plan["splits"] == ready["splits"] == ["train", "val"]
for payload in (plan, ready):
    assert payload["endpoint_only"] is True
    assert payload["row_policy"] == "official_map_update_endpoints_plus_final"
    assert payload["query_only_at_map_endpoints"] is True
    assert payload["query_every_frame"] is False
    assert payload["future_pose_revisions_used"] is False
    assert payload["map_init_window"] == 20
    assert payload["map_every"] == 8
print("Completed four-shard cache is valid for eight-rank training")
PY

# Do not inherit smoke, dry-run, partial-batch, or resume controls from the web
# job environment. This is a fresh five-epoch initialization from runtime
# best.pth; no checkpoint digest is pinned.
unset POSE_ADAPT_RESUME
unset POSE_ADAPT_MAX_BATCHES
unset POSE_ADAPT_8GPU_SMOKE_AUDIT
unset POSE_ADAPT_SMOKE_WORLD_SIZE
unset LOG_FILE
unset HF_HOME
unset HUGGINGFACE_HUB_CACHE
unset TORCH_HOME
unset XDG_CACHE_HOME
unset MPLCONFIGDIR
unset TRITON_CACHE_DIR

echo "[pose-adapt-resume-8gpu] launching formal eight-rank training"
INTERNNAV_MODEL_PATH="$ALLOWED_ROOT/InternNav-Model" RUNTIME_CACHE_ROOT="$ALLOWED_ROOT/model/.runtime_cache" HEATMAP_DATA_ROOT="$DATASET_ROOT" HEATMAP_AMB3R_POSE_CACHE_ROOT="$CACHE_ROOT" POSE_ADAPT_CONFIG="$CONFIG" POSE_ADAPT_INIT_CKPT="$INIT_CKPT" POSE_ADAPT_EXPERIMENT_ROOT="$EXPERIMENT_ROOT" SINGLE_VIEW_HM_OUT_DIR="$EXPERIMENT_ROOT/runs" SINGLE_VIEW_HM_TB_DIR="$EXPERIMENT_ROOT/tensorboard" LOG_FILE="$EXPERIMENT_ROOT/launcher_logs/train_${RUN_STAMP}.log" POSE_ADAPT_DRY_RUN=0 POSE_ADAPT_RESUME= POSE_ADAPT_MAX_BATCHES= POSE_ADAPT_8GPU_SMOKE_AUDIT=0 MAP_INIT_WINDOW="$MAP_INIT_WINDOW" MAP_EVERY="$MAP_EVERY" GPU_DEVICES="$GPU_DEVICES" NPROC_PER_NODE=8 EXPECTED_NUM_GPUS=8 MASTER_ADDR=127.0.0.1 MASTER_PORT="$MASTER_PORT" bash scripts/run_heatmap_amb3r_pose_adapt_8gpu_mxc500.sh

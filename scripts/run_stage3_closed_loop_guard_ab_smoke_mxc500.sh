#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

export FJL_ROOT="${FJL_ROOT:-/mnt/afs/lixiaoou/intern/fjl}"
export STAGE3_EVAL_MODEL_GPU="${STAGE3_EVAL_MODEL_GPU:-7}"
export STAGE3_EVAL_RPC_PORT="${STAGE3_EVAL_RPC_PORT:-50067}"
export STAGE3_EVAL_DISPLAY="${STAGE3_EVAL_DISPLAY:-localhost:200.0}"
export STAGE3_EVAL_CHECKPOINT="${STAGE3_EVAL_CHECKPOINT:-${FJL_ROOT}/model/output_stage3_pano_system1_full_11000_alllora_h1024_internnavcoords_priorfix/latest/checkpoints/epoch_002.pth}"
if [[ -z "${STAGE3_EVAL_TRAIN_OUT_DIR:-}" ]]; then
  export STAGE3_EVAL_TRAIN_OUT_DIR
  STAGE3_EVAL_TRAIN_OUT_DIR="$(dirname "$(dirname "$(dirname "$STAGE3_EVAL_CHECKPOINT")")")"
fi
export STAGE3_EVAL_EPISODE_LIST="${STAGE3_EVAL_EPISODE_LIST:-${REPO_ROOT}/configs/eval_cohorts/stage3_closed_loop_smoke_15.json}"
export STAGE3_EVAL_MAX_STEPS="${STAGE3_EVAL_MAX_STEPS:-200}"
export STAGE3_EVAL_REQUIRE_DETERMINISTIC_SAMPLING=1
export STAGE3_EVAL_RESUME=0
export STAGE3_EVAL_OVERWRITE=1
export STAGE3_EVAL_SAVE_TRAJECTORY_STEPS="${STAGE3_EVAL_SAVE_TRAJECTORY_STEPS:-0}"
export STAGE3_EVAL_CHECKPOINT_SETTLE_S=0

RUN_STAMP="${STAGE3_GUARD_SMOKE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${STAGE3_GUARD_SMOKE_OUT_ROOT:-${FJL_ROOT}/model/smoke_stage3_closed_loop_guard_${RUN_STAMP}}"

run_arm() {
  local tag="$1"
  local action_chunk="$2"
  local stop_confirmations="$3"
  local loop_guard="$4"

  export STAGE3_EVAL_ACTION_CHUNK_SIZE="$action_chunk"
  export STAGE3_EVAL_STOP_CONFIRMATIONS="$stop_confirmations"
  export STAGE3_EVAL_CLOSED_LOOP_GUARD="$loop_guard"
  export STAGE3_EVAL_OUTPUT_PATH="${OUT_ROOT}/${tag}"
  export STAGE3_EVAL_SERVER_LOG="${REPO_ROOT}/logs/stage3_guard_smoke_${RUN_STAMP}_${tag}_server.log"
  export STAGE3_EVAL_CLIENT_LOG="${REPO_ROOT}/logs/stage3_guard_smoke_${RUN_STAMP}_${tag}_client.log"

  echo "[$(date '+%F %T')] START arm=$tag action_chunk=$action_chunk stop_confirmations=$stop_confirmations loop_guard=$loop_guard"
  bash scripts/run_stage3_r2r_val_unseen_rpc_mxc500.sh
  echo "[$(date '+%F %T')] PASSED arm=$tag result=${STAGE3_EVAL_OUTPUT_PATH}/result.json"
}

run_arm baseline 4 1 0
run_arm guarded 2 2 1

QWEN25_PYTHON="${QWEN25_PYTHON:-${FJL_ROOT}/envs/qwen25/bin/python}"
"$QWEN25_PYTHON" - "$OUT_ROOT/baseline/progress.json" "$OUT_ROOT/guarded/progress.json" <<'PY'
import json
import sys


def load(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return {(str(row["scene_id"]), int(row["episode_id"])): row for row in rows}


baseline = load(sys.argv[1])
guarded = load(sys.argv[2])
if baseline.keys() != guarded.keys():
    raise SystemExit("A/B episode sets do not match")


def metrics(rows):
    values = list(rows.values())
    return {
        "n": len(values),
        "SR": sum(float(row["success"]) for row in values) / len(values),
        "SPL": sum(float(row["spl"]) for row in values) / len(values),
        "OS": sum(float(row["os"]) for row in values) / len(values),
        "NE": sum(float(row["ne"]) for row in values) / len(values),
        "max_steps": sum(int(row["steps"]) >= 200 for row in values),
    }


print("baseline", json.dumps(metrics(baseline), sort_keys=True))
print("guarded ", json.dumps(metrics(guarded), sort_keys=True))
for key in baseline:
    old, new = baseline[key], guarded[key]
    print(
        key,
        f"success {int(old['success'])}->{int(new['success'])}",
        f"ne {float(old['ne']):.2f}->{float(new['ne']):.2f}",
        f"steps {int(old['steps'])}->{int(new['steps'])}",
        f"probes={int(new.get('closed_loop_stop_probes', 0))}",
        f"recoveries={new.get('closed_loop_recoveries', [])}",
    )
PY

echo "[stage3-guard-smoke] COMPLETE out=$OUT_ROOT"

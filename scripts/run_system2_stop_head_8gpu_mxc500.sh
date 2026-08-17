#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${HEATMAPVLN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

export STAGE1_S2_CONFIG="${SYSTEM2_STOP_HEAD_CONFIG:-configs/train_system2_panoramic_stop_head_8gpu.yaml}"
export STAGE1_S2_LOAD_WEIGHTS="${SYSTEM2_STOP_HEAD_BASE_CKPT:-/mnt/afs/lixiaoou/intern/fjl/model/output_stage1_s2_full_11000_rank32_alllayer_from_heatmap/run_20260701_212615/checkpoints/epoch_005.pth}"
export STAGE1_S2_OUT_DIR="${SYSTEM2_STOP_HEAD_OUT_DIR:-/mnt/afs/lixiaoou/intern/fjl/model/output_system2_stop_head_full_11000_alllora_h1024}"
export STAGE1_S2_TB_DIR="${SYSTEM2_STOP_HEAD_TB_DIR:-/mnt/afs/lixiaoou/intern/fjl/tensorlog/heatmapvln_system2_stop_head_full_11000_alllora_h1024}"
export STAGE1_S2_EPOCHS="${SYSTEM2_STOP_HEAD_EPOCHS:-1}"
export LOG_FILE="${LOG_FILE:-${REPO_ROOT}/logs/system2_stop_head_full_11000_alllora_h1024_8gpu_mxc500.log}"

if [[ ! -s "$STAGE1_S2_CONFIG" ]]; then
  echo "Missing STOP-head config: $STAGE1_S2_CONFIG" >&2
  exit 1
fi
if [[ ! -s "$STAGE1_S2_LOAD_WEIGHTS" ]]; then
  echo "Missing complete Stage1-S2 checkpoint: $STAGE1_S2_LOAD_WEIGHTS" >&2
  exit 1
fi

QWEN25_PYTHON="${QWEN25_PYTHON:-/mnt/afs/lixiaoou/intern/fjl/envs/qwen25/bin/python}"
"$QWEN25_PYTHON" - "$STAGE1_S2_CONFIG" "$STAGE1_S2_LOAD_WEIGHTS" <<'PY'
import sys

import torch
import yaml

config_path, path = sys.argv[1:]
with open(config_path, encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
trajectory = config["data"]["trajectory"]
llm = config["model"]["llm"]
stop_head = config["model"]["stop_head"]
stage = config["training"]["stages"][0]
validation = config["validation"]
assert trajectory["system2_stop_oversample"] == 1
assert trajectory["system2_stop_path_radius_m"] == 3.0
assert trajectory["system2_near_stop_hard_negative_oversample"] == 2
assert trajectory["system2_near_stop_hard_negative_min_goal_distance_m"] == 4.0
assert trajectory["system2_near_stop_hard_negative_max_goal_distance_m"] == 18.0
assert not trajectory.get("system2_near_stop_hard_negative_min_path_m", 0.0)
assert not trajectory.get("system2_near_stop_hard_negative_max_path_m", 0.0)
assert trajectory["sft_include_turns"] is True
assert llm["lora_rank"] == 32
assert llm["lora_layer_indices"] == list(range(28))
assert stop_head["enabled"] and stop_head["pos_weight"] == 1.0
assert stop_head["bce_mix"] == 1.0
assert stop_head["add_stop_threshold"] > stop_head["veto_stop_threshold"]
assert stage["train_system2_stop_head"] is True
assert stage["sft_include_turns"] is True
assert stage["trainable_modules"] == ["stop_head"]
assert validation["enabled"] is True
assert 0.0 < validation["holdout_clip_fraction"] < 1.0
assert 0.0 <= validation["stop_add_max_false_positive_rate"] < 1.0
assert 0.0 < validation["stop_veto_min_recall"] <= 1.0
assert validation["stop_add_min_threshold"] == 0.9
assert validation["stop_veto_max_threshold"] == 0.5

checkpoint = torch.load(path, map_location="cpu", weights_only=False)
state = checkpoint.get("trainable_state_dict") or {}
lora = {key: value for key, value in state.items() if ".lora_" in key}
if len(lora) != 224:
    raise SystemExit(
        f"Expected complete rank-32 all-layer LoRA checkpoint with 224 tensors, "
        f"found {len(lora)} in {path}"
    )
if not all(torch.isfinite(value).all() for value in lora.values()):
    raise SystemExit(f"Non-finite LoRA tensor in {path}")
print(
    "Verified System2 STOP-head contract: "
    f"frozen_lora_tensors={len(lora)} trainable=stop_head "
    "stop_path_radius_m=3.0 hard_negative_goal_distance_m=[4.0,18.0] "
    "clip_disjoint_validation=enabled"
)
PY

bash scripts/run_stage1_s2_8gpu_mxc500_launcher.sh

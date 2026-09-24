#!/usr/bin/env bash
# EXP-18 GPU pipeline (dev machine, GPUs 7,6,5): dumps A/B, then VO caches + dumps C/D/E.
set -euo pipefail
W=/mnt/afs/liwenhao/agent/370910109
EXP=$W/model/exp18_first_person_viz
SRC=${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$SRC"
export PYTHONDONTWRITEBYTECODE=1
QWEN=$W/envs/qwen25/bin/python
echo "[gpu] $(date -u +%FT%TZ) select A/B"
PYTHONPATH=$SRC $QWEN scripts/exp18/select_clips_ab.py
for T in B A; do echo "[gpu] $(date -u +%FT%TZ) dump $T"; TIER=$T bash scripts/exp18/run_dump.sh; done
for T in C D E; do
  while [[ ! -f $EXP/clip_lists/$T.txt ]]; do sleep 60; done
  echo "[gpu] $(date -u +%FT%TZ) amb3r cache $T"; TIER=$T bash scripts/exp18/run_amb3r_cache.sh
  echo "[gpu] $(date -u +%FT%TZ) dump $T"; TIER=$T bash scripts/exp18/run_dump.sh
done
echo "[gpu] $(date -u +%FT%TZ) DONE"

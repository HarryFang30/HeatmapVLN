#!/usr/bin/env bash
# EXP-18 CPU pipeline: select + render + finalize C/D/E, then top-down maps for all tiers.
set -euo pipefail
W=/mnt/afs/liwenhao/agent/370910109
EXP=$W/model/exp18_first_person_viz
SRC=${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$SRC"
export PYTHONDONTWRITEBYTECODE=1
VLNCE=$W/envs/vlnce/bin/python
echo "[cpu] $(date -u +%FT%TZ) select episodes"
env -u DISPLAY $VLNCE -m scripts.exp18.render.select_episodes
for T in C D E; do
  echo "[cpu] $(date -u +%FT%TZ) render $T"; TIER=$T NUM_WORKERS=8 bash scripts/exp18/render/run_render.sh
  echo "[cpu] $(date -u +%FT%TZ) finalize $T"; $VLNCE -m scripts.exp18.render.finalize_clip_lists --tier $T
done
for T in C D B A; do echo "[cpu] $(date -u +%FT%TZ) topdown $T"; TIER=$T NUM_PROCS=4 bash scripts/exp18/topdown/run_topdown.sh; done
echo "[cpu] $(date -u +%FT%TZ) DONE"

#!/usr/bin/env bash
# EXP-31 R0 on the 910B (writes only under ~/work/zhr/zhr_1): serve the base model twice from src_x31 (the commit with
# the sampling top_k fix) until 10-21 07:00 UTC: tag r0s = System2 sampling (T 0.7, top_p 0.95, top_k 0) on NPUs 0-3,
# ports 52410-13 / 52510-13; tag r0g = greedy on NPUs 4-6, ports 52414-16 / 52514-16.  Status in $D/STATUS.
R=~/work/zhr/zhr_1; D=$R/exp31; SRC=$R/exp23/src_x31
mkdir -p "$D"
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/r0.log"; echo "$(date -u +%FT%TZ) $*" > "$D/STATUS"; }
END=$(date -d "2026-10-21T07:00:00Z" +%s)
left=$(( END - $(date +%s) ))
grep -q "top_k=0" "$SRC/scripts/evaluation/rpc_model_server.py" || { log "STOPPED: $SRC lacks the top_k fix"; exit 2; }
ss -Hltn "sport = :52417" | grep -q . || { log "STOPPED: nothing serves the export port 52417"; exit 2; }
cd "$SRC" || exit 2
EXP28_TAG=r0s EXP28_SRC="$SRC" EXP28_NPUS=0,1,2,3 EXP28_MODEL_PORT_BASE=52410 EXP28_VO_PORT_BASE=52510 EXP28_SERVE_S=$left \
  PPA_EVAL_SYSTEM2_SAMPLE_TEMPERATURE=0.7 PPA_EVAL_SYSTEM2_SAMPLE_TOP_P=0.95 \
  nohup setsid bash scripts/exp28/serve_910b.sh > "$D/serve_r0s.out" 2>&1 < /dev/null &
sleep 20
EXP28_TAG=r0g EXP28_SRC="$SRC" EXP28_NPUS=4,5,6 EXP28_MODEL_PORT_BASE=52414 EXP28_VO_PORT_BASE=52514 EXP28_SERVE_S=$left \
  nohup setsid bash scripts/exp28/serve_910b.sh > "$D/serve_r0g.out" 2>&1 < /dev/null &
log "starting r0s (sampled, 0-3) and r0g (greedy, 4-6) from $(git -C "$SRC" rev-parse --short HEAD) until 10-21 07:00 UTC"

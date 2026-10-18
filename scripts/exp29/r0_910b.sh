#!/usr/bin/env bash
# EXP-29 R0 on the 910B (report only): once V1 is exported (exp29_910b.sh DONE), serve the base twice until
# 10-19 01:00 UTC: tag r0s = System2 sampling (T 0.7, top_p 0.95) on NPUs 0-3 / ports 52410-13 + 52510-13, tag r0g =
# greedy on NPUs 4-6 / ports 52414-16 + 52514-16.  The 4090's r0_4090.sh runs the same 200 failed episodes on both.
R=~/work/zhr/zhr_1; D=$R/exp29; SRC=$R/exp23/src_x28
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/r0.log"; echo "$(date -u +%FT%TZ) $*" > "$D/R0_STATUS"; }
END=$(date -d "2026-10-19T01:00:00Z" +%s)
log "waiting for V1's export"
until grep -q "^.* DONE: exported" "$D/STATUS" 2>/dev/null; do
  grep -q "STOPPED" "$D/STATUS" 2>/dev/null && { log "STOPPED: V1 training stopped, R0 not started"; exit 2; }
  (( $(date +%s) < END )) || { log "STOPPED: no V1 export before the end"; exit 2; }
  sleep 120
done
sleep 30
left=$(( END - $(date +%s) )); (( left > 3600 )) || { log "STOPPED: less than 1 h left"; exit 2; }
cd "$SRC" || exit 2
EXP28_TAG=r0s EXP28_NPUS=0,1,2,3 EXP28_MODEL_PORT_BASE=52410 EXP28_VO_PORT_BASE=52510 EXP28_SERVE_S=$left \
  PPA_EVAL_SYSTEM2_SAMPLE_TEMPERATURE=0.7 PPA_EVAL_SYSTEM2_SAMPLE_TOP_P=0.95 \
  nohup setsid bash scripts/exp28/serve_910b.sh > "$D/serve_r0s.out" 2>&1 < /dev/null &
sleep 20
EXP28_TAG=r0g EXP28_NPUS=4,5,6 EXP28_MODEL_PORT_BASE=52414 EXP28_VO_PORT_BASE=52514 EXP28_SERVE_S=$left \
  nohup setsid bash scripts/exp28/serve_910b.sh > "$D/serve_r0g.out" 2>&1 < /dev/null &
log "R0 servers starting (sampled r0s on 0-3, greedy r0g on 4-6) until 10-19 01:00 UTC"

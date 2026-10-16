#!/usr/bin/env bash
# EXP-28: incremental line-B labelling of the base collection every 2 h (CPU, 8 workers; the labeller skips
# records whose labels are current).  Ends after one more pass once both launchers report DONE/STOPPED,
# or when $LR/STOP exists.
WT=/workspace/exp23/wt_exp28
OUT=/workspace/exp28/collect_base
LR=/workspace/exp28/labels_base
mkdir -p "$LR"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LR/label_loop.log"; }
pass() {
  log "pass start: $(find "$OUT/records" -name '*.s2rec.tar.sha256' 2>/dev/null | wc -l) records"
  ( cd "$WT" && PYTHONPATH="$WT" /opt/conda/bin/python scripts/tools/label_route_corrections.py \
      --record_root "$OUT/records" --label_root "$LR" --scenes_dir /workspace/exp28/scene_datasets/hm3d \
      --episodes /workspace/exp28/episodes/pool.json.gz --workers 8 >> "$LR/label_runs.log" 2>&1 )
  log "pass exit $?: $(find "$LR" -name '*.s2labels.jsonl' | wc -l) label files"
  chown -R 1015:1015 "$LR"
}
while true; do
  [[ -e "$LR/STOP" ]] && { log "STOP file"; exit 0; }
  finished=$(cat "$OUT"/STATUS_local "$OUT"/STATUS_external 2>/dev/null | grep -c -E 'DONE|STOPPED')
  pass
  [[ "$finished" -ge 2 ]] && { log "both launchers finished; last pass done"; exit 0; }
  sleep 7200
done

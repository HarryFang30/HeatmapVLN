#!/usr/bin/env bash
# EXP-28 Phase-0 on the 4090 (CPU): probe_metrics over the committed episodes on the HM3D navmeshes, then the report
# on the 400 held-out (910B) and the first 400 local training episodes (shards in $1, launcher order).
WT=/workspace/exp23/wt_exp28
C=/workspace/exp28/collect_base
OUT=/workspace/exp28/phase0
PY=/opt/conda/bin/python
LOCAL_SHARDS="$1"
mkdir -p "$OUT"
cd "$WT" && export PYTHONPATH="$WT"
echo "$(date -u +%FT%TZ) probe_metrics" > "$OUT/STATUS"
$PY scripts/exp25/probe_metrics.py --name phase0 --scenes /workspace/exp28/scene_datasets/hm3d \
  --datasets /workspace/exp28/episodes/cohorts/dataset_shard_*.json.gz \
  --progress "$C/workers/shard_*/progress.json" --logs "$C/runtime/*/logs/client_shard_*.log" \
  --out "$OUT" --workers 16 > "$OUT/probe_metrics.log" 2>&1 || { echo "$(date -u +%FT%TZ) STOPPED: probe_metrics" > "$OUT/STATUS"; exit 2; }
D=$(ls "$OUT"/phase0*.decomp.json | head -1)
echo "$(date -u +%FT%TZ) report ($D)" > "$OUT/STATUS"
$PY scripts/exp28/phase0_report.py --decomp heldout="$D" train="$D" --cohort /workspace/exp28/episodes/cohorts \
  --shards heldout=0-19 train="$LOCAL_SHARDS" --limit train=400 --expect heldout=400 train=400 \
  --progress heldout="$C/workers/shard_*/progress.json" train="$C/workers/shard_*/progress.json" \
  --overlap-keys /workspace/exp28/episodes/overlap_keys.json --out "$OUT" > "$OUT/report.log" 2>&1
rc=$?
chown -R 1015:1015 "$OUT"
echo "$(date -u +%FT%TZ) DONE (report exit $rc)" > "$OUT/STATUS"

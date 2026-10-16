#!/usr/bin/env bash
# EXP-28's offline gates (ledger EXP-28 D, written 2026-10-15 19:35 UTC before any HM3D result) for one
# System2 LoRA, on the 4090 inside the fjl-habitat container.  One run is one gate pass ("attempt"):
#   O1  teacher-forced decision-class agreement on B3's ScaleVLN scenes (700 samples each, seed 42),
#       candidate >= base - 1.0 pt on the same samples (scripts/tools/teacher_forced_agreement.py);
#   O2  the same on r2r's held-out 17DRP5sb8fy and 5LpN3gDmAk7 (300 each);
#   O3  EXP-23's held-out records decoded with the adapter on and off (eval_system2_lora_offline.py),
#       read by stratified_gates.py --gate exp28-o3: (b)-(e) and the new (f);
#   O4  the base's HM3D held-out records with line-B labels (labelled here; idempotent), decoded with the
#       adapter on and off over all cards and judged by line B (scripts/exp28/hm3d_route_gate.py);
#   O5  the same corrections under another episode's instruction of the same scene (--shuffle-instructions);
# then hm3d_route_gate.py verdict: every gate's PASS/FAIL and the pre-registered next step (val,
# retrain_2_epochs, scale_0.5, kill; invalid when something could not be measured) in $OUT/verdict.json,
# its first line in $OUT/STATUS.
#
#   EXP28_WT=/workspace/exp23/wt_<commit> EXP28_ADAPTER=/workspace/exp28/c1/adapter EXP28_NAME=c1 \
#     EXP28_HOLDOUT_SCENES_FILE=/workspace/exp28/episodes/splits/heldout_scenes.txt \
#     EXP28_O4_RECORDS=/workspace/exp28/collect_base/records \
#     nohup setsid bash scripts/exp28/run_offline_gates.sh > /dev/null 2>&1 &
#
# The one retry ("重试只有一次，类型预先定") is a second run with EXP28_ATTEMPT=2: after retrain_2_epochs with
# the new adapter; after scale_0.5 with the same adapter and EXP28_LORA_SCALE=0.5, which gates a copy of the
# adapter with every lora_B tensor halved ($OUT/adapter_scale0.5, the directory val then serves).
#
# GPU jobs (teacher-forced O1/O2 with and without the adapter, the O3 decode, the O4/O5 decode shards) are
# claimed one at a time by one worker per card in EXP28_GPUS, so a card never holds two models.  The base
# references of O1/O2 are decoded once into $EXP28_GATE_ROOT/base (one file per scene root, scenes and
# sample count) and reused by every candidate (the verdict checks they cover the very same samples);
# EXP28_O1_BASE / EXP28_O2_BASE instead name an existing teacher_forced_agreement.py summary of the base
# (its .calls.jsonl beside it, read only, e.g. EXP-25 B3's /workspace/exp23/b3/scalevln_b3.json).  Do not
# run two gate passes at once: they would share the base decode.  A rerun skips finished jobs
# ($OUT/jobs/*.ok); the O1/O2 and O4 decodes also resume inside a job.  $OUT belongs to the adapter it was
# first run with ($OUT/ADAPTER_SHA256); another adapter under the same name and attempt is refused.  A smoke
# run with EXP28_O4_MAX_EPISODES set ends "invalid", never "val".
# Every path below is an environment variable with a default.

set -uo pipefail

WT="${EXP28_WT:?set EXP28_WT to the pinned code worktree, e.g. /workspace/exp23/wt_<commit>}"
ADAPTER="${EXP28_ADAPTER:?set EXP28_ADAPTER to the PEFT directory of the candidate}"
NAME="${EXP28_NAME:?set EXP28_NAME to the candidate name, e.g. c1}"
ATTEMPT="${EXP28_ATTEMPT:-1}"
LORA_SCALE="${EXP28_LORA_SCALE:-1}"
PYTHON="${EXP28_PYTHON:-/opt/conda/bin/python}"
MODEL="${EXP28_MODEL_PATH:-/workspace/InternNav_Model}"
RPC_SRC="${EXP28_RPC_SRC:-/workspace/rpc/src}"
GPUS="${EXP28_GPUS:-5,4,0}"
GATE_ROOT="${EXP28_GATE_ROOT:-/workspace/exp28/gates}"
BASE_DIR="${EXP28_GATE_BASE_DIR:-$GATE_ROOT/base}"
# O1 / O2: the same tool, scenes and sample counts as EXP-25 B3 and the r2r reference (0.859, 0.975).
O1_ROOT="${EXP28_O1_SCENE_ROOT:-/workspace/exp23/data/interndata_n1/scalevln}"
O1_SCENES="${EXP28_O1_SCENES:-00481-5jLhtVmWd5F,00393-dD37vuCa4FE,00396-wuKkTq5GJbi,00560-gjhYih4upQ9,00770-NBg5UqG3di3}"
O1_PER_SCENE="${EXP28_O1_PER_SCENE:-700}"
O2_ROOT="${EXP28_O2_SCENE_ROOT:-/workspace/exp23/data/interndata_n1/r2r}"
O2_SCENES="${EXP28_O2_SCENES:-17DRP5sb8fy,5LpN3gDmAk7}"
O2_PER_SCENE="${EXP28_O2_PER_SCENE:-300}"
O1_BASE="${EXP28_O1_BASE:-}"  # empty: decode the base once into $BASE_DIR
O2_BASE="${EXP28_O2_BASE:-}"
# O3: EXP-23 round 1's records and labels, its six held-out scenes (s2_sft_dataset.holdout_scenes()), and
# the seeded 2,000 calls EXP-26's O3 decoded (0 = every held-out call, several hours a card).
O3_RECORDS="${EXP28_O3_RECORDS:-/workspace/exp23/round1/records}"
O3_LABELS="${EXP28_O3_LABELS:-/workspace/exp23/round1/labels}"
O3_HOLDOUT="${EXP28_O3_HOLDOUT_SCENES:-17DRP5sb8fy,5LpN3gDmAk7,VzqfbhrpDEA,gZ6f7yhEvPG,kEZ7cmS4wCh,r47D5H71a5s}"
O3_MAX_CALLS="${EXP28_O3_MAX_CALLS:-2000}"
# O4 / O5: the base's closed-loop records of the 20 held-out HM3D scenes (Phase-0, collected on the 910B).
O4_RECORDS="${EXP28_O4_RECORDS:-/workspace/exp28/collect_base/records}"
# The 20 held-out scenes, one per line (build_hm3d_episodes.py's splits/heldout_scenes.txt).
HOLDOUT_FILE="${EXP28_HOLDOUT_SCENES_FILE:-/workspace/exp28/episodes/splits/heldout_scenes.txt}"
SCENES_DIR="${EXP28_HM3D_SCENES_DIR:-/workspace/exp28/scene_datasets/hm3d}"
O4_EPISODES="${EXP28_O4_EPISODES:-}"  # optional R2R-CE-format json.gz with the episodes' paths and goals
O4_DIR="${EXP28_O4_DIR:-$GATE_ROOT/hm3d_heldout}"
O4_MAX_EPISODES="${EXP28_O4_MAX_EPISODES:-0}"
LABEL_WORKERS="${EXP28_LABEL_WORKERS:-12}"

[[ "$ATTEMPT" =~ ^[12]$ ]] || { echo "EXP28_ATTEMPT must be 1 or 2" >&2; exit 2; }
SUFFIX="_a$ATTEMPT"
[[ "$LORA_SCALE" == 1 ]] || SUFFIX="${SUFFIX}_scale$LORA_SCALE"
OUT="${EXP28_OUT:-$GATE_ROOT/$NAME$SUFFIX}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
O4_SHARDS="${EXP28_O4_SHARDS:-$(( 2 * ${#GPU_LIST[@]} ))}"
export PYTHONPATH="$WT:$RPC_SRC"

mkdir -p "$OUT/logs" "$OUT/jobs" "$BASE_DIR" "$O4_DIR/records"
LOG="$OUT/logs/gates.log"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
status() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" > "$OUT/STATUS"; log "STATUS $*"; }
die() { status "STOPPED: $*"; chown -R 1015:1015 "$OUT" 2>/dev/null; exit 2; }
git -c safe.directory='*' -C "$WT" diff --quiet HEAD -- || die "$WT has uncommitted changes"
[[ -s "$ADAPTER/adapter_model.safetensors" ]] || die "$ADAPTER has no adapter_model.safetensors"
[[ -s "$HOLDOUT_FILE" ]] || die "no held-out scene list at $HOLDOUT_FILE"
for base in "$O1_BASE" "$O2_BASE"; do
  [[ -z "$base" || -s "${base%.json}.calls.jsonl" ]] || die "no base rows ${base%.json}.calls.jsonl"
done
# The base decode is named by its sample settings, so a smoke run with fewer samples never stands in for
# the full one (the verdict would refuse the mismatch as invalid, but only after the whole pass).
settings_tag() { printf '%s|%s|%s' "$1" "$2" "$3" | sha256sum | cut -c1-10; }
O1_BASE="${O1_BASE:-$BASE_DIR/o1_scalevln_$(settings_tag "$O1_ROOT" "$O1_SCENES" "$O1_PER_SCENE").json}"
O2_BASE="${O2_BASE:-$BASE_DIR/o2_r2r_$(settings_tag "$O2_ROOT" "$O2_SCENES" "$O2_PER_SCENE").json}"
log "name=$NAME attempt=$ATTEMPT lora_scale=$LORA_SCALE wt=$WT commit=$(git -c safe.directory='*' -C "$WT" rev-parse HEAD)"
SOURCE_SHA="$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)"
log "adapter=$ADAPTER sha256=${SOURCE_SHA:0:16} gpus=$GPUS out=$OUT"
# $OUT belongs to one adapter: a rerun skips finished jobs, resumes their rows and reuses the scaled copy,
# all of which would silently be another adapter's after a retrain under the same name and attempt.
if [[ -s "$OUT/ADAPTER_SHA256" ]]; then
  [[ "$(cat "$OUT/ADAPTER_SHA256")" == "$SOURCE_SHA" ]] \
    || die "$OUT was gated with adapter $(cat "$OUT/ADAPTER_SHA256"), not $SOURCE_SHA: set EXP28_NAME or EXP28_OUT anew"
else
  echo "$SOURCE_SHA" > "$OUT/ADAPTER_SHA256"
fi

# 1 -------------------------------------------------------------------------------
# The scale retry gates (and val later serves) a copy whose lora_B tensors are multiplied by the scale:
# the LoRA update B.A.(alpha/r) scales with it exactly (0.5 is exact in bf16).
if [[ "$LORA_SCALE" != 1 ]]; then
  SCALED="$OUT/adapter_scale$LORA_SCALE"
  if [[ ! -s "$SCALED/adapter_model.safetensors" ]]; then
    status "scaling the adapter by $LORA_SCALE"
    "$PYTHON" - "$ADAPTER" "$SCALED" "$LORA_SCALE" >> "$LOG" 2>&1 <<'PY' || die "could not scale the adapter"
import json
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

src, dst, scale = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
tmp = dst.with_name(dst.name + ".part")
shutil.rmtree(tmp, ignore_errors=True)
shutil.copytree(src, tmp)
weights = load_file(str(src / "adapter_model.safetensors"))
scaled = sorted(k for k in weights if ".lora_B." in k)
if not scaled:
    raise SystemExit(f"{src}: no lora_B tensors")
for key in scaled:
    weights[key] = (weights[key].float() * scale).to(weights[key].dtype)
save_file(weights, str(tmp / "adapter_model.safetensors"), metadata={"format": "pt"})
(tmp / "lora_scale.json").write_text(json.dumps(
    {"source": str(src.resolve()), "scale": scale, "lora_B_tensors": len(scaled)}, indent=1) + "\n")
tmp.rename(dst)
print(f"scaled {len(scaled)} lora_B tensors by {scale}: {dst}")
PY
  fi
  ADAPTER="$SCALED"
  log "gating $ADAPTER sha256=$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -c1-16)"
fi

# 2 -------------------------------------------------------------------------------
# O4's records: the held-out scenes only (a link per scene directory), so neither the labeller nor the
# gate ever reads a training scene.  Labels are line B's defaults, as the training stream's.
status "linking and labelling the HM3D held-out records"
linked=0
while read -r scene; do
  scene="${scene%%#*}"; scene="${scene//[[:space:]]/}"
  [[ -n "$scene" ]] || continue
  [[ "$scene" =~ ^[0-9]+-(.+)$ ]] && scene="${BASH_REMATCH[1]}"  # 00481-5jLhtVmWd5F -> 5jLhtVmWd5F
  if [[ -d "$O4_RECORDS/$scene" ]]; then
    ln -sfn "$O4_RECORDS/$scene" "$O4_DIR/records/$scene"
    linked=$((linked + 1))
  else
    log "no records for held-out scene $scene under $O4_RECORDS"
  fi
done < "$HOLDOUT_FILE"
(( linked > 0 )) || die "none of the held-out scenes has records under $O4_RECORDS"
label_args=(--record_root "$O4_DIR/records" --label_root "$O4_DIR/labels" --scenes_dir "$SCENES_DIR"
            --workers "$LABEL_WORKERS")
[[ -n "$O4_EPISODES" ]] && label_args+=(--episodes "$O4_EPISODES")
"$PYTHON" "$WT/scripts/tools/label_route_corrections.py" "${label_args[@]}" >> "$OUT/logs/label_heldout.log" 2>&1 \
  || die "labelling the held-out records failed (logs/label_heldout.log)"
log "held-out scenes linked: $linked; labels: $(tail -n 1 "$OUT/logs/label_heldout.log")"

O4_SELECT=(--record-root "$O4_DIR/records" --label-root "$O4_DIR/labels" --holdout-scenes-file "$HOLDOUT_FILE"
           --output "$OUT/o4/hm3d.json" --max-episodes "$O4_MAX_EPISODES")
mkdir -p "$OUT/o1" "$OUT/o2" "$OUT/o3" "$OUT/o4"
"$PYTHON" "$WT/scripts/exp28/hm3d_route_gate.py" decode "${O4_SELECT[@]}" --adapter "$ADAPTER" --model-path "$MODEL" \
  --shuffle-instructions --dry-run >> "$LOG" 2>&1 || die "O4 state selection failed (see the log)"

# 3 -------------------------------------------------------------------------------
JOBS=(o3_eval o1)
[[ -n "${EXP28_O1_BASE:-}" ]] || JOBS+=(o1_base)
for ((k = 0; k < O4_SHARDS; k++)); do JOBS+=("o4_$k"); done
JOBS+=(o2)
[[ -n "${EXP28_O2_BASE:-}" ]] || JOBS+=(o2_base)

marker() {
  case "$1" in
    o1_base) echo "${O1_BASE%.json}.ok" ;;
    o2_base) echo "${O2_BASE%.json}.ok" ;;
    *) echo "$OUT/jobs/$1.ok" ;;
  esac
}

teacher_forced() {  # scene_root scenes per_scene adapter output
  local args=(--scene-root "$1" --max-samples-per-scene "$3" --model-path "$MODEL" --output "$5" --device cuda)
  [[ -n "$2" ]] && args+=(--scenes "$2")
  [[ -n "$4" ]] && args+=(--adapter "$4")
  "$PYTHON" "$WT/scripts/tools/teacher_forced_agreement.py" "${args[@]}"
}

run_job() {
  local job="$1"
  case "$job" in
    o1) teacher_forced "$O1_ROOT" "$O1_SCENES" "$O1_PER_SCENE" "$ADAPTER" "$OUT/o1/scalevln.json" ;;
    o1_base) teacher_forced "$O1_ROOT" "$O1_SCENES" "$O1_PER_SCENE" "" "$O1_BASE" ;;
    o2) teacher_forced "$O2_ROOT" "$O2_SCENES" "$O2_PER_SCENE" "$ADAPTER" "$OUT/o2/r2r.json" ;;
    o2_base) teacher_forced "$O2_ROOT" "$O2_SCENES" "$O2_PER_SCENE" "" "$O2_BASE" ;;
    o3_eval)
      "$PYTHON" "$WT/scripts/tools/eval_system2_lora_offline.py" --record-root "$O3_RECORDS" --label-root "$O3_LABELS" \
        --adapter "$ADAPTER" --model-path "$MODEL" --output "$OUT/o3/gates.json" --holdout-scenes "$O3_HOLDOUT" \
        --max-calls "$O3_MAX_CALLS" --device cuda ;;
    o4_*)
      "$PYTHON" "$WT/scripts/exp28/hm3d_route_gate.py" decode "${O4_SELECT[@]}" --adapter "$ADAPTER" \
        --model-path "$MODEL" --shard "${job#o4_}/$O4_SHARDS" --shuffle-instructions --device cuda ;;
    *) echo "unknown job $job" >&2; return 2 ;;
  esac
}

worker() {  # one per card; claims the next unfinished job (mkdir is atomic)
  local gpu="$1" job
  for job in "${JOBS[@]}"; do
    [[ -e "$(marker "$job")" ]] && continue
    mkdir "$OUT/jobs/claims/$job" 2>/dev/null || continue
    log "gpu $gpu: $job"
    if CUDA_VISIBLE_DEVICES="$gpu" run_job "$job" > "$OUT/logs/$job.log" 2>&1; then
      touch "$(marker "$job")"
      log "gpu $gpu: $job done"
    else
      touch "$OUT/jobs/$job.failed"
      log "gpu $gpu: $job FAILED (logs/$job.log)"
    fi
  done
}

rm -rf "$OUT/jobs/claims" "$OUT/jobs/"*.failed
mkdir -p "$OUT/jobs/claims"
status "decoding: ${#JOBS[@]} jobs on cards $GPUS"
for gpu in "${GPU_LIST[@]}"; do worker "$gpu" & done
wait
failed=$(cd "$OUT/jobs" && ls -- *.failed 2>/dev/null | sed 's/\.failed$//' | tr '\n' ' ')
[[ -z "$failed" ]] || die "jobs failed: $failed"

# 4 -------------------------------------------------------------------------------
status "scoring"
rm -f "$OUT/o3/gates.stratified.json" "$OUT/o4/hm3d.json" "$OUT/verdict.json"
"$PYTHON" "$WT/scripts/tools/stratified_gates.py" --calls "$OUT/o3/gates.calls.jsonl" --gate exp28-o3 >> "$LOG" 2>&1
[[ -s "$OUT/o3/gates.stratified.json" ]] || die "stratified_gates.py wrote nothing (see the log)"
score_args=("${O4_SELECT[@]}" --scenes-dir "$SCENES_DIR")
[[ -n "$O4_EPISODES" ]] && score_args+=(--episodes "$O4_EPISODES")
"$PYTHON" "$WT/scripts/exp28/hm3d_route_gate.py" score "${score_args[@]}" >> "$LOG" 2>&1
[[ -s "$OUT/o4/hm3d.json" ]] || die "hm3d_route_gate.py score wrote nothing (see the log)"
"$PYTHON" "$WT/scripts/exp28/hm3d_route_gate.py" verdict --o1 "$OUT/o1/scalevln.json" \
  --o1-base "$O1_BASE" --o2 "$OUT/o2/r2r.json" --o2-base "$O2_BASE" \
  --o3 "$OUT/o3/gates.stratified.json" --o4 "$OUT/o4/hm3d.json" --attempt "$ATTEMPT" --output "$OUT/verdict.json" \
  > "$OUT/logs/verdict.txt" 2>&1
[[ -s "$OUT/verdict.json" ]] || die "the verdict could not be written (logs/verdict.txt)"
cat "$OUT/logs/verdict.txt" >> "$LOG"
chown -R 1015:1015 "$OUT" "$BASE_DIR" "$O4_DIR" 2>/dev/null
status "DONE: $(head -n 1 "$OUT/logs/verdict.txt") (adapter $ADAPTER)"

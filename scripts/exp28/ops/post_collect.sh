#!/usr/bin/env bash
# EXP-28 on the 4090 after the base collection (fjl-habitat).  Unattended; status in $R1/POST_STATUS, log $R1/post.log.
#  1. wait for the external (910B) launcher to finish;
#  2. P's validity rerun: held-out shards 0-6 on the 910B's still-running base servers until >= 10 records (compared
#     later); then wait for the local launcher (or the cutoff 10-17 14:00 UTC: TERM the launchers);
#  3. final labelling pass, AUD, DAgger stream, merge with the replay stream;
#  4. push 150 npu-recorded training records (G0-npu) and the stream to the 910B (retrying until its receivers are up);
#     meanwhile G0-cuda (2,000 cuda calls, adapters off) on GPUs 5,4,0;
#  5. pull adapter_rX28C1 from the export port, run the offline gates (run_offline_gates.sh);
#  6. hand over to exp28_auto.sh (retries, val, P, seeds).
WT=/workspace/exp23/wt_exp28; C=/workspace/exp28/collect_base; LR=/workspace/exp28/labels_base
R1=/workspace/exp28/round1; X=/workspace/exp28/c1
H=/workspace/exp28/episodes/splits/heldout_scenes.txt; T=/workspace/exp28/episodes/splits/train_scenes.txt
PY=/opt/conda/bin/python
CUTOFF=$(date -d "2026-10-17T14:00:00Z" +%s)
mkdir -p "$R1/logs" "$X"
st() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$R1/post.log" > "$R1/POST_STATUS"; }
die() { st "STOPPED: $*"; chown -R 1015:1015 "$R1" "$X" 2>/dev/null; exit 2; }
term_launchers() {  # $1 = PPA_COLLECT_OUTPUT_ROOT of the launchers to stop
  for p in $(pgrep -f "^bash scripts/run_s2_dagger_collect_cuda[.]sh$"); do
    ps -o args= -p "$(ps -o ppid= -p "$p" | tr -d ' ')" | grep -q run_s2_dagger_collect_cuda && continue
    tr '\0' '\n' < "/proc/$p/environ" | grep -q "^PPA_COLLECT_OUTPUT_ROOT=$1$" && kill -TERM "$p" && st "TERM launcher $p ($1)"
  done
}
cd "$WT" || exit 2
export PYTHONPATH="$WT:/workspace/rpc/src"

# 1 ----------------------------------------------------------------------------
# The P rerun goes as soon as the 910B half is done, while its base servers still run (they stop after 30 min idle).
st "waiting for the external (910B) launcher"
until grep -q -E "DONE|STOPPED" "$C/STATUS_external" 2>/dev/null || (( $(date +%s) >= CUTOFF )); do sleep 120; done

# 2 ----------------------------------------------------------------------------
if (( $(date +%s) < CUTOFF )); then
  st "P validity rerun: held-out shards 0-6 on the base servers"
  ( EXP28_MODE=external EXP28_TAG=base EXP28_WT="$WT" EXP28_COHORT_DIR=/workspace/exp28/episodes/cohorts \
      EXP28_OUT_DIR=/workspace/exp28/p_repro EXP28_SHARDS=0-6 EXP28_DISPLAY_BASE=741 \
      setsid bash scripts/exp28/collect_4090.sh > /dev/null 2>&1 < /dev/null & )
  for _ in $(seq 90); do
    [[ "$(find /workspace/exp28/p_repro/records -name '*.s2rec.tar.sha256' 2>/dev/null | wc -l)" -ge 10 ]] && break
    sleep 60
  done
  st "P rerun: $(find /workspace/exp28/p_repro/records -name '*.s2rec.tar.sha256' 2>/dev/null | wc -l) records; stopping it"
  term_launchers /workspace/exp28/p_repro
fi
st "waiting for the local launcher"
while ! grep -q -E "DONE|STOPPED" "$C/STATUS_local" 2>/dev/null; do
  if (( $(date +%s) >= CUTOFF )); then st "cutoff: stopping the collection"; term_launchers "$C"; sleep 180; break; fi
  sleep 300
done
st "collection over: local [$(cat "$C/STATUS_local")] external [$(cat "$C/STATUS_external")]"

# 3 ----------------------------------------------------------------------------
touch "$LR/STOP"
st "final labelling pass"
$PY scripts/tools/label_route_corrections.py --record_root "$C/records" --label_root "$LR" \
  --scenes_dir /workspace/exp28/scene_datasets/hm3d --episodes /workspace/exp28/episodes/pool.json.gz --workers 16 \
  > "$R1/logs/label_final.log" 2>&1 || die "final labelling (logs/label_final.log)"
st "AUD"
$PY /workspace/exp28/aud.py --label-root "$LR" --train-scenes-file "$T" --out "$R1/aud.json" > "$R1/logs/aud.log" 2>&1 \
  || die "AUD: $(tail -1 "$R1/logs/aud.log")"
st "DAgger stream ($(tail -1 "$R1/logs/aud.log" | cut -c1-120))"
$PY scripts/tools/build_route_stream.py --record-root "$C/records" --label-root "$LR" \
  --holdout-scenes-file "$H" --train-scenes-file "$T" --first-calls 48 --min-distance 3 \
  --stop-correction-min-distance 5 --max-stop-corrections-per-episode 1 --drop-reversals --no-pad \
  --out-dir "$R1/stream_dagger" > "$R1/logs/stream_dagger.log" 2>&1 || die "DAgger stream (logs/stream_dagger.log)"
grep -q "^EXIT 0" "$R1/stream_replay.log" || die "the replay stream is not built (stream_replay.log)"
st "merging"
$PY scripts/exp28/merge_streams.py --dagger-stream "$R1/stream_dagger" --replay-stream "$R1/stream_replay" \
  --holdout-scenes-file "$H" --out-dir "$R1/stream" > "$R1/logs/merge.log" 2>&1 || die "merge (logs/merge.log)"
chown -R 1015:1015 "$R1"

# 4 ----------------------------------------------------------------------------
st "pushing (G0 records, stream) and G0-cuda"
$PY - "$C" > "$R1/g0_files.txt" <<'PY' || die "could not list the G0 records"
import glob, json, os, re, sys, tarfile
root = sys.argv[1]; out = []
for sha in sorted(glob.glob(os.path.join(root, "records", "*", "*.s2rec.tar.sha256"))):
    tar = sha[:-7]
    with tarfile.open(tar) as t:
        ep = json.load(t.extractfile("episode.json"))
    shard = int(re.search(r"shard_(\d+)\.json", ep["episode_list"]).group(1))
    if shard >= 20 and not ep["server"]["address"].startswith("127.0.0.1:5247"):
        out += [os.path.relpath(tar, root), os.path.relpath(sha, root)]
    if len(out) >= 300:
        break
print("\n".join(out))
PY
echo G0_PUSHED > "$R1/g0_marker.txt"; date -u +%FT%TZ > "$C/G0_PUSHED"
( $PY scripts/tools/stream_transfer.py push --src "$C" --ports 52413 --files "$R1/g0_files.txt" --retries 1000000 \
    > "$R1/logs/push_g0.log" 2>&1 \
  && $PY scripts/tools/stream_transfer.py push --src "$C" --ports 52413 --files "$R1/g0_marker.txt" --retries 1000000 \
    >> "$R1/logs/push_g0.log" 2>&1 ) &
( $PY scripts/tools/stream_transfer.py push --src "$R1/stream" --ports 52414,52415,52416 --retries 1000000 --busy-sleep 15 \
    > "$R1/logs/push_stream.log" 2>&1 ) &
for _ in $(seq 60); do  # the local collection servers must be gone
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '($1==0 || $1==4 || $1==5) && $2>15000' | wc -l)
  [[ "$busy" -eq 0 ]] && break; sleep 20
done
CUDA_VISIBLE_DEVICES=5,4,0 $PY -m torch.distributed.run --standalone --nproc_per_node 3 scripts/train_system2_lora.py \
  --model-path /workspace/InternNav_Model --record-root "$C/records" --output-dir "$R1/g0_cuda" \
  --parity-check 2000 --parity-split all --device cuda > "$R1/logs/g0_cuda.log" 2>&1
g0=$($PY -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["exact_rate"], d["g0_pass"])' "$R1/g0_cuda/parity_g0.json" 2>/dev/null)
st "G0-cuda: ${g0:-no report}"
[[ "$g0" == *True ]] || die "G0-cuda did not pass (${g0:-no report}; logs/g0_cuda.log)"
wait  # both pushes
st "pushed: $(tail -1 "$R1/logs/push_stream.log" | cut -c1-120)"

# 5 ----------------------------------------------------------------------------
st "waiting for adapter_rX28C1 on 52417"
A="$X/adapter_pull/adapter_rX28C1"
until [[ -e "$A/.pulled" ]]; do
  $PY scripts/exp23/pull_dirs.py "$X/adapter_pull" 52417 adapter_rX28C1 >> "$X/pull_adapter.log" 2>&1
  [[ -e "$A/.pulled" ]] || sleep 300
done
[[ -s "$A/adapter_model.safetensors" ]] || die "the pulled adapter has no weights"
st "offline gates on $(sha256sum "$A/adapter_model.safetensors" | cut -c1-16)"
EXP28_WT="$WT" EXP28_ADAPTER="$A" EXP28_NAME=c1 EXP28_HOLDOUT_SCENES_FILE="$H" EXP28_O4_RECORDS="$C/records" \
  EXP28_O1_BASE=/workspace/exp23/b3/scalevln_b3.json EXP28_O2_BASE=/workspace/exp23/exp26/base_r2r_holdout.json \
  bash scripts/exp28/run_offline_gates.sh > "$X/gates_run.log" 2>&1
if [[ ! -s /workspace/exp28/gates/c1_a1/verdict.json ]]; then  # one more pass (it resumes its finished jobs)
  st "gates wrote no verdict; running them once more"
  EXP28_WT="$WT" EXP28_ADAPTER="$A" EXP28_NAME=c1 EXP28_HOLDOUT_SCENES_FILE="$H" EXP28_O4_RECORDS="$C/records" \
    EXP28_O1_BASE=/workspace/exp23/b3/scalevln_b3.json EXP28_O2_BASE=/workspace/exp23/exp26/base_r2r_holdout.json \
    bash scripts/exp28/run_offline_gates.sh > "$X/gates_run2.log" 2>&1
fi
[[ -s /workspace/exp28/gates/c1_a1/verdict.json ]] || die "the offline gates wrote no verdict ($(cat /workspace/exp28/gates/c1_a1/STATUS 2>/dev/null))"
V=/workspace/exp28/gates/c1_a1/verdict.json
next=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1])).get("next", "?"))' "$V" 2>/dev/null)
st "gates: next=${next:-none} ($V)"

# 6 ----------------------------------------------------------------------------
# The rest of the decision tree (retries, val, P, seeds) is exp28_auto.sh's: it reads gates/c1_a1/verdict.json.
st "DONE: gates next=${next:-none}; exp28_auto.sh takes over"
chown -R 1015:1015 "$R1" "$X"

#!/usr/bin/env bash
# EXP-37 on the 4090 (ledger 运行记录 2): X37's val with local servers, sdpa, seed 1337, step budget on, paired with
# A0 seed 1337 + step budget (scripts/exp37/build_a0_budget_control.py).  The criteria and the nine parts are
# val_4090.sh's; the servers are this box's instead of the 910B's, so the parts run one after another on the slots
# in $X/gpus.env, read again before each try (X37_GPU_DEVICES, X37_VO_GPU_DEVICES: one entry per slot), so a slot on
# a card that filled up can be dropped there.  A part's tries count only tries without a new episode.
# Starts only after equiv.json and the gate's verdict pass.  Interim futility, one look: the first time the
# completed parts pair >= 900 episodes, at dSR <= -2.0 pt the remaining parts are not run and the verdict is
# 否定（中途无效性）.  Markers in $V make a rerun pick up where it stopped: a live launcher is waited for, the
# launcher's --resume finishes a part, and futility.done / paired.done end the run for good.
set -uo pipefail
X=/workspace/exp37/run4090; V=$X/val9; SRC=/workspace/exp37/src_x37; PY=/opt/conda/bin/python
WT=/workspace/exp23/wt_x32; PLAN=/workspace/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802
ADAPTER=/workspace/exp37/x37_s400/adapter
CONTROL=/workspace/eval_runs/exp37_A0budget_seed1337_4090/merged/progress.jsonl
RUN=/workspace/eval_runs/exp37_X37_seed1337_4090_stepbudget   # _p1 .. _p9, _merged
VS_PY=/workspace/exp35/auto/val_split3.py; VS_SHA=c119326418d24513ea628871f696b41dda3a8966d66e142903e44af039d4f46a
FUTILITY_N=900
RETRIES=20
PARTS=(p1 p2 p3 p4 p5 p6 p7 p8 p9)
mkdir -p "$V"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$V/val.log"; }
status() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" > "$V/STATUS"; log "STATUS $*"; }
die() { status "STOPPED: $*"; exit 2; }
VS() { "$PY" "$VS_PY" "$@"; }
old=$(cat "$V/controller.pid" 2>/dev/null)
if [[ -n $old && $old != "$$" ]] && kill -0 "$old" 2>/dev/null \
   && tr '\0' ' ' < "/proc/$old/cmdline" 2>/dev/null | grep -q "val_4090_local.sh"; then
  log "refused: val_4090_local.sh $old runs"; exit 1
fi
echo $$ > "$V/controller.pid"
[[ -e $V/futility.done ]] && { log "already stopped by the interim futility rule ($V/interim.final)"; exit 0; }
[[ -e $V/paired.done ]] && { log "already paired ($V/verdict_adjusted.txt)"; exit 0; }

# --- preconditions
[[ "$(sha256sum "$VS_PY" | cut -d' ' -f1)" == "$VS_SHA" ]] || die "$VS_PY is not the val_split3.py EXP-35 ran"
SHA=$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)
[[ $SHA == "$X37_SHA" ]] || die "the adapter is $SHA, not $X37_SHA"
python3 -c "import json,sys;sys.exit(0 if json.load(open('$X/equiv.json')).get('pass') is True else 1)" 2>/dev/null \
  || die "the equivalence check has not passed ($X/equiv.json)"
python3 - "$X/gate/verdict.json" "$SHA" <<'PY' || die "the offline gate has not passed for this adapter ($X/gate/verdict.json)"
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get("all_pass") is True and d.get("next") == "val" and (d.get("adapter_sha256") or {}).get("candidate") == sys.argv[2] else 1)
PY
[[ "$(wc -l < "$CONTROL")" -eq 1839 ]] || die "the control has not 1,839 rows ($CONTROL)"

[[ -d $V/lists_p1 ]] || VS split "$PLAN/cohorts" $(for p in "${PARTS[@]}"; do printf '%s ' "$V/lists_$p"; done) 2>> "$V/val.log" || die "could not split"

row_check() {  # run dirs: "<reading> <n paired> <dSR> <SE> <lower> <dSPL> <rows not base-stop>" against the control, as run
  python3 - "$CONTROL" "$SHA" "$FUTILITY_N" "$@" <<'PY'
import glob, json, math, sys
control, sha, need, runs = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
key = lambda r: (str(r["scene_id"]), str(r["episode_id"]))
t, bad = {}, 0
for run in runs:
    for path in sorted(glob.glob(f"{run}/workers/shard_0*/progress.json")):
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if key(r) in t:
                continue
            t[key(r)] = r
            v = str(r.get("rpc_model_version") or "")
            bad += not (v.endswith("+base-stop-v1") and sha in v)
c = {key(r): r for r in map(json.loads, open(control))}
ids = sorted(set(t) & set(c))
n = len(ids)
if n < 2:
    print("badmode" if bad else "short", n, "- - - -", bad); sys.exit(0)
s = sum(int(round(float(t[i]["success"]) - float(c[i]["success"]))) for i in ids)
d = [100.0 * (float(t[i]["success"]) - float(c[i]["success"])) for i in ids]
m = sum(d) / n
se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
dspl = 100.0 * sum(float(t[i]["spl"]) - float(c[i]["spl"]) for i in ids) / n
reading = "badmode" if bad else "short" if n < need else "futility" if 50 * s <= -n else "continue"
print(reading, n, f"{m:+.2f}", f"{se:.2f}", f"{m - 1.96 * se:+.2f}", f"{dspl:+.2f}", bad)
PY
}

# --- the parts
for p in "${PARTS[@]}"; do
  [[ -e $V/$p.done ]] && continue
  ok=0; stuck=0; last=-1
  old=$(cat "$V/launcher.pid" 2>/dev/null)
  if [[ -n $old ]] && kill -0 "$old" 2>/dev/null && tr '\0' ' ' < "/proc/$old/cmdline" 2>/dev/null | grep -q "run_ppa_r2r_val_unseen_cuda"; then
    status "part $p: waiting for the running launcher $old"
    while kill -0 "$old" 2>/dev/null; do sleep 60; done
  fi
  while (( stuck < RETRIES )); do
    read -r got listed < <(VS count "${RUN}_$p" "$V/lists_$p")
    [[ $got == "$listed" && $listed -gt 0 ]] && { ok=1; break; }
    source "$X/gpus.env"
    status "part $p on GPUs $X37_GPU_DEVICES (VO $X37_VO_GPU_DEVICES), $got of $listed episodes"
    ( cd "$SRC" && exec setsid env PPA_EVAL_ROOT=/workspace PPA_EVAL_REPO="$SRC" PPA_EVAL_CONFIG=$WT/configs/ppa_action_refine_v2_8gpu.yaml \
        PPA_EVAL_ALLOW_COMMIT_MISMATCH=1 PPA_EVAL_ATTN_IMPL=sdpa PPA_EVAL_GPU_DEVICES="$X37_GPU_DEVICES" PPA_EVAL_VO_GPU_DEVICES="$X37_VO_GPU_DEVICES" \
        PPA_EVAL_PROTOCOL_SEED=1337 PPA_EVAL_STOP_AT_STEP_BUDGET=1 PPA_EVAL_EPISODE_LISTS_DIR="$V/lists_$p" \
        PPA_EVAL_OUTPUT_ROOT="${RUN}_$p" PPA_EVAL_DISPLAY_BASE=470 PPA_EVAL_SHARD_RETRIES=3 \
        PPA_EVAL_SYSTEM2_LORA="$ADAPTER" PPA_EVAL_SYSTEM2_BASE_STOP=1 \
        bash scripts/run_ppa_r2r_val_unseen_cuda.sh ) >> "$V/$p.log" 2>&1 &
    launcher=$!; echo "$launcher" > "$V/launcher.pid"
    wait "$launcher"; rc=$?
    read -r got listed < <(VS count "${RUN}_$p" "$V/lists_$p")
    log "part $p: launcher exit $rc, $got of $listed episodes"
    [[ $got == "$listed" && $listed -gt 0 ]] && { ok=1; break; }
    if (( got > last )); then stuck=0; else stuck=$((stuck + 1)); fi
    last=$got
    sleep 300
  done
  (( ok )) || die "part $p: no new episode in $RETRIES tries"
  touch "$V/$p.done"
  done_runs=(); for q in "${PARTS[@]}"; do [[ -e $V/$q.done ]] && done_runs+=("${RUN}_$q"); done
  if [[ -e $V/interim.final ]]; then
    read -r reading _ _ _ _ _ bad < <(row_check "${done_runs[@]}")
    [[ $reading != badmode && $bad == 0 ]] || die "rows of part $p are not in base-stop mode with adapter $SHA"
    continue
  fi
  read -r reading n m se lower dspl bad < <(row_check "${done_runs[@]}")
  [[ $reading != badmode && $bad == 0 ]] || die "rows of part $p are not in base-stop mode with adapter $SHA"
  case $reading in
    short) log "interim: $n paired so far, under $FUTILITY_N, outcomes not read" ;;
    futility)
      echo "$reading $n $m $se $lower $dspl" > "$V/interim.final"
      touch "$V/futility.done"
      status "X37 val stopped by the interim futility rule: $n paired, dSR $m (SE $se), dSPL $dspl; verdict 否定（中途无效性）"
      exit 0 ;;
    continue)
      echo "$reading $n" > "$V/interim.final"
      log "interim: $n paired, the rule is spent; the val continues" ;;
    *) die "unknown reading $reading" ;;
  esac
done

# --- merge with the locked tool, pair, verdict
CM=${RUN}_merged
rm -rf "$CM"; mkdir -p "$CM/merged"
VS combine "$CM" $(for p in "${PARTS[@]}"; do printf '%s ' "${RUN}_$p"; done) > "$V/merge.txt" 2>&1 \
  && ( cd "$WT" && PYTHONPATH=$PLAN/tools:/workspace/rpc/src:$WT "$PY" "$PLAN/tools/merge_shards.py" \
    --dataset /workspace/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz --cohorts-dir "$PLAN/cohorts" \
    --workers-dir "$CM/workers" --output-dir "$CM/merged" --num-shards 8 --expected-episodes 1839 \
    --protocol heatmapvln-r2r-json-v3 --sampling-protocol heatmapvln-nextdit-sha256-v1 --protocol-seed 1337 \
    --evaluation-arm ppa_refine_v2_online_amb3r_cuda >> "$V/merge.txt" 2>&1 ) \
  || die "merging failed (see $V/merge.txt)"
read -r reading n _ _ _ _ bad < <(FUTILITY_N=1 row_check "$CM")
[[ $reading != badmode && $bad == 0 && $n == 1839 ]] || die "merged rows: ${bad:-?} not in base-stop mode, ${n:-?} paired (not 1,839)"
for mode in asrun standard; do
  extra=(); [[ $mode == standard ]] && extra=(--standard-protocol)
  out=$V/paired_vs_a0budget_seed1337_$mode.json
  ( cd "$WT" && "$PY" scripts/tools/paired_closed_loop_bootstrap.py --treatment "$CM/merged/progress.jsonl" --control "$CONTROL" \
      --bootstrap-draws 20000 --label "x37_vs_a0budget_seed1337_$mode" "${extra[@]}" --output-json "$out" > "${out%.json}.txt" 2>&1 ) \
    || die "pairing ($mode) failed (see ${out%.json}.txt)"
done
PYTHONPATH=$WT/scripts/tools "$PY" "$VS_PY" verdict "$CM/merged/progress.jsonl" "$CONTROL" "X37 vs A0+budget" main > "$V/verdict.txt" 2>&1 \
  || die "the verdict failed (see $V/verdict.txt)"
python3 - "$CM/merged/progress.jsonl" "$CONTROL" > "$V/mechanism.txt" 2>&1 <<'PY'
import json, sys
def counts(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    far = [r for r in rows if not float(r["os"]) > 0]
    budget = sum(1 for r in far if r.get("step_budget_stop") or float(r.get("steps", 0)) >= 500)  # A0's capped rows carry no flag
    rbf = sum(1 for r in rows if float(r["os"]) > 0 and not float(r["success"]) > 0)
    return len(far) - budget, budget, rbf
(xs, xb, xr), (bs, bb, br) = counts(sys.argv[1]), counts(sys.argv[2])
print(f"mechanism (report only), X37 vs A0+budget: never within 3 m {xs + xb} vs {bs + bb} (own STOP {xs} vs {bs}, step budget {xb} vs {bb}); "
      f"within 3 m but failed {xr} vs {br}")
PY
# per call, from the servers' logs; a restarted client repeats the calls of its in-flight episode
grep -h "System2 stop source" "$RUN"_p*/runtime/*/logs/model_*.log 2>/dev/null | sed 's/.*System2 stop source: //' | sort | uniq -c > "$V/stop_source.txt"
# the deciding line (ledger 运行记录 2): the control is A0's draw, 0.46 pt under seed 1337's four-arm mean of 61.20, so
# dSR is read after taking that off; verdict.txt above is the unadjusted line, reported beside it
python3 - "$CM/merged/progress.jsonl" "$CONTROL" > "$V/verdict_adjusted.txt" 2>&1 <<'PY' || die "the adjusted verdict failed"
import json, math, sys
OFFSET = 0.46
load = lambda path: {(str(r["scene_id"]), str(r["episode_id"])): r for r in map(json.loads, open(path))}
t, c = load(sys.argv[1]), load(sys.argv[2])
ids = sorted(set(t) & set(c)); n = len(ids)
d = [100.0 * (float(t[i]["success"]) - float(c[i]["success"])) for i in ids]
m = sum(d) / n
se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
dspl = 100.0 * sum(float(t[i]["spl"]) - float(c[i]["spl"]) for i in ids) / n
adj = m - OFFSET
reading = "support" if adj - 1.96 * se > 0 and dspl >= -1.0 else "refuted" if adj <= 0 else "not measured"
print(f"X37 vs A0+budget, adjusted by {OFFSET}: n {n}, dSR {m:+.2f} -> {adj:+.2f} (SE {se:.2f}, lower {adj - 1.96 * se:+.2f}), dSPL {dspl:+.2f} -> {reading}")
PY
touch "$V/paired.done"
status "X37 val done: $(cat "$V/verdict_adjusted.txt")"

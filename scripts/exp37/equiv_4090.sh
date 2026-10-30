#!/usr/bin/env bash
# EXP-37 on the 4090 (ledger 运行记录 2): the zero-adapter equivalence check against A0 seed 1337.  A zero adapter
# (lora_B = 0) in base-stop-v1 mode, sdpa, seed 1337, on B0's episodes 1/25/41 must answer call for call as A0's
# client logs do.  The no-adapter run of this code already reads 56/56 identical ($X/canary_vs_a0.json); this
# one adds what the mode adds.  Writes $X/equiv.json; exit 0 only when it passes.
set -uo pipefail
X=/workspace/exp37/run4090; SRC=/workspace/exp37/src_x37; PY=/opt/conda/bin/python
ZERO=$X/zero_adapter; RUN=$X/run_zero; LISTS=/workspace/exp37/equiv/lists
A0_LOGS="/workspace/eval_runs/exp20_a0_seed1337_4090/runtime/*/logs/client_shard_*.log"
GPU=${X37_EQUIV_GPU:-0}
mkdir -p "$X/logs"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$X/logs/equiv.log"; }
: > "$X/equiv.json"

if [[ ! -s $ZERO/adapter_model.safetensors ]]; then
  rm -rf "$ZERO"
  log "building the zero adapter on the CPU"
  ( cd "$SRC" && CUDA_VISIBLE_DEVICES= PYTHONPATH="$SRC:/workspace/rpc/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      "$PY" scripts/exp37/make_zero_adapter.py --model-path /workspace/InternNav_Model \
      --reference-config /workspace/exp37/x37_s400/adapter/adapter_config.json --output "$ZERO" ) >> "$X/logs/zero_adapter.log" 2>&1 \
    || { log "STOPPED: make_zero_adapter failed (see $X/logs/zero_adapter.log)"; exit 2; }
fi
log "zero adapter $(sha256sum "$ZERO/adapter_model.safetensors" | cut -c1-16); running 3 episodes on GPU $GPU"
rm -rf "$RUN"
( cd "$SRC" && PPA_EVAL_ROOT=/workspace PPA_EVAL_REPO="$SRC" PPA_EVAL_CONFIG=/workspace/exp23/wt_x32/configs/ppa_action_refine_v2_8gpu.yaml \
    PPA_EVAL_ALLOW_COMMIT_MISMATCH=1 PPA_EVAL_ATTN_IMPL=sdpa PPA_EVAL_GPU_DEVICES="$GPU" PPA_EVAL_SHARDS=0 PPA_EVAL_PROTOCOL_SEED=1337 \
    PPA_EVAL_STOP_AT_STEP_BUDGET=1 PPA_EVAL_EPISODE_LISTS_DIR="$LISTS" PPA_EVAL_OUTPUT_ROOT="$RUN" PPA_EVAL_DISPLAY_BASE=470 \
    PPA_EVAL_SHARD_RETRIES=0 PPA_EVAL_SYSTEM2_LORA="$ZERO" PPA_EVAL_SYSTEM2_BASE_STOP=1 \
    bash scripts/run_ppa_r2r_val_unseen_cuda.sh ) > "$X/logs/run_zero.log" 2>&1 &
launcher=$!
# what a slot holds, for the slot layout of the val: model and VO server separately, at the run's busiest
for _ in $(seq 40); do
  sleep 30; kill -0 "$launcher" 2>/dev/null || break
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> "$X/logs/run_zero.mem" 2>/dev/null
done
wait "$launcher"; rc=$?
log "launcher exit $rc"
(( rc == 0 )) || { log "STOPPED: the zero-adapter run failed (see $X/logs/run_zero.log)"; exit 2; }
"$PY" "$SRC/scripts/tools/compare_client_logs.py" --a "$A0_LOGS" --b "$RUN/runtime/*/logs/client_shard_*.log" \
  --out "$X/equiv_compare.json" --require-identical >> "$X/logs/equiv.log" 2>&1
rc=$?
python3 - "$RUN" "$ZERO/adapter_model.safetensors" "$X/equiv_compare.json" "$rc" "$X/equiv.json" <<'PY' | tee -a "$X/logs/equiv.log"
import glob, hashlib, json, sys
run, adapter, compare, rc, out = sys.argv[1:]
sha = hashlib.sha256(open(adapter, "rb").read()).hexdigest()
rows = [json.loads(l) for p in glob.glob(f"{run}/workers/shard_0*/progress.json") for l in open(p) if l.strip()]
versions = {str(r.get("rpc_model_version") or "") for r in rows}
mode_ok = bool(rows) and all(v.endswith("+base-stop-v1") and sha in v for v in versions)
cmp = json.load(open(compare))
ok = rc == "0" and mode_ok and cmp.get("shared_episodes") == 3 and cmp.get("identical") == 3
doc = {"pass": ok, "zero_adapter_sha256": sha, "mode_ok": mode_ok, "episodes": len(rows), "compare": cmp, "reference": "exp20_a0_seed1337_4090"}
json.dump(doc, open(out, "w"), indent=1)
print("EQUIV", {k: doc[k] for k in ("pass", "mode_ok", "episodes")}, {k: cmp.get(k) for k in ("shared_episodes", "identical", "calls_compared")})
sys.exit(0 if ok else 1)
PY

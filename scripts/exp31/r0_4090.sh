#!/usr/bin/env bash
# EXP-31 R0 on the 4090: once EXP-24's targeted rerun has finished (its GPUs), run the 200 base-greedy-failed HM3D
# episodes (exp29/r0/cohort) against the 910B's r0s (sampled System2, 4 slots) and r0g (greedy, 3 slots), System1 seed
# 1337, code wt_x31 (= the servers' commit).  Stop at 10-21 06:00 UTC at the latest; then the paired rescue rates with a
# scene bootstrap and the pre-registered GO / NO-GO in /workspace/exp31/r0_report.json.
WT=/workspace/exp23/wt_x31; PY=/opt/conda/bin/python; X=/workspace/exp31; COH=/workspace/exp29/r0/cohort
END=$(date -d "2026-10-21T06:00:00Z" +%s)
mkdir -p "$X"
st() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$X/r0.log" > "$X/STATUS"; }
st "waiting for EXP-24's rerun to finish"
until grep -q -E " (DONE|STOPPED):" /workspace/exp24/run_1020/STATUS 2>/dev/null; do
  (( $(date +%s) < END )) || { st "STOPPED: EXP-24 never finished"; exit 2; }
  sleep 120
done
sleep 60
st "starting both arms (they wait for the 910B's server lists)"
for arm in "r0s 5,4,0,5 52410 52510 781" "r0g 4,0,5 52414 52514 791"; do set -- $arm
  ( PPA_EVAL_PROTOCOL_SEED=1337 EXP28_MODE=external EXP28_TAG=$1 EXP28_WT="$WT" EXP28_COHORT_DIR="$COH" \
      EXP28_OUT_DIR="$X/$1" EXP28_SHARDS=0-19 EXP28_CLIENT_GPUS=$2 EXP28_MODEL_PORT_BASE=$3 EXP28_VO_PORT_BASE=$4 \
      EXP28_DISPLAY_BASE=$5 setsid bash "$WT/scripts/exp28/collect_4090.sh" > /dev/null 2>&1 < /dev/null & )
done
while :; do
  n=$(cat "$X/r0s/STATUS_external" "$X/r0g/STATUS_external" 2>/dev/null | grep -c -E "DONE|STOPPED")
  [[ "$n" -ge 2 ]] && break
  if (( $(date +%s) >= END )); then
    st "end time: stopping both arms"
    for p in $(pgrep -f "^bash scripts/run_s2_dagger_collect_cuda[.]sh$"); do
      ps -o args= -p "$(ps -o ppid= -p "$p" | tr -d ' ')" | grep -q run_s2_dagger_collect_cuda && continue
      tr '\0' '\n' < "/proc/$p/environ" | grep -q -E "^PPA_COLLECT_OUTPUT_ROOT=$X/r0[sg]$" && kill -TERM "$p"
    done
    sleep 180; break
  fi
  sleep 600
done
"$PY" - "$X" > "$X/r0_report.json" 2> "$X/r0_report.err" <<'PY'
import glob, json, os, random, sys, tarfile
root = sys.argv[1]
def arm(name):
    out = {}
    for sha in glob.glob(os.path.join(root, name, "records", "*", "*.s2rec.tar.sha256")):
        with tarfile.open(sha[:-7]) as t:
            ep = json.load(t.extractfile("episode.json"))
        o = ep["outcome"]
        out[(ep["scene_id"], int(ep["episode_id"]))] = {"success": float(bool(o.get("success"))), "seed": ep.get("protocol_seed"),
                                                        "steps": o.get("steps"), "os": o.get("oracle_success")}
    return out
s, g = arm("r0s"), arm("r0g")
keys = sorted(set(s) & set(g))
d = [s[k]["success"] - g[k]["success"] for k in keys]
scenes = sorted({k[0] for k in keys}); by = {sc: [i for i, k in enumerate(keys) if k[0] == sc] for sc in scenes}
rng = random.Random(0); boots = []
for _ in range(20000):
    idx = [i for sc in (rng.choice(scenes) for _ in scenes) for i in by[sc]]
    boots.append(sum(d[i] for i in idx) / len(idx) if idx else 0.0)
boots.sort()
mean = lambda xs: sum(xs) / len(xs) if xs else None
diff = mean(d); lo, hi = (boots[500], boots[19499]) if keys else (None, None)
if len(keys) < 100:
    verdict = "REPORT ONLY (fewer than 100 paired episodes)"
elif diff >= 0.10 and lo > 0:
    verdict = "GO: sampled System2 rescues clearly more than a System1 reseed"
elif diff <= 0.03:
    verdict = "NO-GO: sampling's diversity does not turn into closed-loop successes"
else:
    verdict = "NOT DETECTED"
print(json.dumps({
    "schema": "heatmapvln-exp31-r0-v1", "verdict": verdict, "paired_episodes": len(keys),
    "episodes_sampled": len(s), "episodes_greedy": len(g),
    "rescue_sampled": mean([s[k]["success"] for k in keys]), "rescue_greedy": mean([g[k]["success"] for k in keys]),
    "diff_sampled_minus_greedy": diff, "ci95_scene_bootstrap": [lo, hi],
    "os_sampled": mean([float(bool(s[k]["os"])) for k in keys]), "os_greedy": mean([float(bool(g[k]["os"])) for k in keys]),
    "ran_out_of_steps_sampled": mean([float((s[k]["steps"] or 0) >= 500) for k in keys]),
    "ran_out_of_steps_greedy": mean([float((g[k]["steps"] or 0) >= 500) for k in keys]),
    "protocol_seeds": sorted({str(v["seed"]) for v in list(s.values()) + list(g.values())})}, indent=1))
PY
st "DONE: $(tr -d '\n' < "$X/r0_report.json" | cut -c1-400)"
chown -R 1015:1015 "$X"

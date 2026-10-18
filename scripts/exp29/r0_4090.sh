#!/usr/bin/env bash
# EXP-29 R0 on the 4090 (report only): the 200 base-greedy-failed HM3D training episodes (r0/cohort) against the
# 910B's r0s (sampled System2, 4 slots) and r0g (greedy, 3 slots), both with System1 seed 1337; stop at 10-19 00:00
# UTC at the latest; then the per-arm rescue rate on the episodes both arms finished, with a scene bootstrap.
WT=/workspace/exp23/wt_exp28; PY=/opt/conda/bin/python; R0=/workspace/exp29/r0
END=$(date -d "2026-10-19T00:00:00Z" +%s)
st() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$R0/r0.log" > "$R0/STATUS"; }
st "starting both arms (they wait for the 910B's server lists)"
for arm in "r0s 5,4,0,5 52410 52510 781" "r0g 4,0,5 52414 52514 791"; do set -- $arm
  ( PPA_EVAL_PROTOCOL_SEED=1337 EXP28_MODE=external EXP28_TAG=$1 EXP28_WT="$WT" EXP28_COHORT_DIR="$R0/cohort" \
      EXP28_OUT_DIR="$R0/$1" EXP28_SHARDS=0-19 EXP28_CLIENT_GPUS=$2 EXP28_MODEL_PORT_BASE=$3 EXP28_VO_PORT_BASE=$4 \
      EXP28_DISPLAY_BASE=$5 setsid bash "$WT/scripts/exp28/collect_4090.sh" > /dev/null 2>&1 < /dev/null & )
done
while :; do
  n=$(cat "$R0/r0s/STATUS_external" "$R0/r0g/STATUS_external" 2>/dev/null | grep -c -E "DONE|STOPPED")
  [[ "$n" -ge 2 ]] && break
  if (( $(date +%s) >= END )); then
    st "end time: stopping both arms"
    for p in $(pgrep -f "^bash scripts/run_s2_dagger_collect_cuda[.]sh$"); do
      ps -o args= -p "$(ps -o ppid= -p "$p" | tr -d ' ')" | grep -q run_s2_dagger_collect_cuda && continue
      tr '\0' '\n' < "/proc/$p/environ" | grep -q -E "^PPA_COLLECT_OUTPUT_ROOT=$R0/r0[sg]$" && kill -TERM "$p"
    done
    sleep 180; break
  fi
  sleep 600
done
"$PY" - "$R0" > "$R0/r0_report.json" 2> "$R0/r0_report.err" <<'PY'
import glob, json, os, random, sys, tarfile
root = sys.argv[1]
def arm(name):
    out = {}
    for sha in glob.glob(os.path.join(root, name, "records", "*", "*.s2rec.tar.sha256")):
        with tarfile.open(sha[:-7]) as t:
            ep = json.load(t.extractfile("episode.json"))
        out[(ep["scene_id"], int(ep["episode_id"]))] = float(bool(ep["outcome"].get("success"))), ep.get("protocol_seed")
    return out
s, g = arm("r0s"), arm("r0g")
keys = sorted(set(s) & set(g))
d = [s[k][0] - g[k][0] for k in keys]
scenes = sorted({k[0] for k in keys}); by = {sc: [i for i, k in enumerate(keys) if k[0] == sc] for sc in scenes}
rng = random.Random(0); boots = []
for _ in range(20000):
    idx = [i for sc in (rng.choice(scenes) for _ in scenes) for i in by[sc]]
    boots.append(sum(d[i] for i in idx) / len(idx) if idx else 0.0)
boots.sort()
mean = lambda xs: sum(xs) / len(xs) if xs else None
print(json.dumps({
    "schema": "heatmapvln-exp29-r0-v1", "paired_episodes": len(keys),
    "episodes_sampled": len(s), "episodes_greedy": len(g),
    "rescue_sampled": mean([s[k][0] for k in keys]), "rescue_greedy": mean([g[k][0] for k in keys]),
    "diff_sampled_minus_greedy": mean(d), "ci95_scene_bootstrap": [boots[500], boots[19499]] if keys else None,
    "protocol_seeds": sorted({str(v[1]) for v in list(s.values()) + list(g.values())}),
    "note": "report only (ledger EXP-29 R0): System1 seed 1337 on both arms; base-greedy outcome on these episodes was failure"}, indent=1))
PY
st "DONE: $(tr -d '\n' < "$R0/r0_report.json" | cut -c1-300)"
chown -R 1015:1015 "$R0"

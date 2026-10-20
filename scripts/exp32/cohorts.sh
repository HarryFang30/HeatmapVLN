#!/usr/bin/env bash
# EXP-32 cohorts (4090, CPU), ledger EXP-32 part 2 B1-B3.  Built as scripts/exp29/r0_cohort.sh builds R0's: the wanted
# episodes' committed records are linked into OUT/links, then remaining_cohort.py --keep committed over EXP-28's cohort
# writes an ordinary cohort to OUT/cohort (collect_4090.sh takes it as EXP28_COHORT_DIR).
#   cohorts.sh pass2   OUT   B1: P1's failures minus R0-S's successes (about 620), 20 shards.  After part 2's commit
#                            and P1's end.
#   cohorts.sh heldout OUT   B3: the 112 base-failed held-out episodes (collect_base shards 0-19, success 0), 8 shards.
#   cohorts.sh pass3   OUT   B2: the pass-2 cohort's episodes that pass 2 did not rescue either (no successful
#                            trajectory in P1, R0-S or pass 2; pass-2 episodes the 7 h cutoff left uncollected stay in),
#                            20 shards.  Only when B2's dose rule triggers it, after pass 2's end.
# Writes only OUT/{links,cohort,selection.json,cohorts.log} and refuses an OUT that already holds a cohort.
# Refuses (exit 2) when: a collection it reads has a STATUS_* that is not DONE/STOPPED (EXP32_ALLOW_UNFINISHED=1 skips
# this); the base's failed training episodes are not 1,017 or its failed held-out ones not 112; a P1, R0-S or pass-2
# record is not one of the episodes its cohort could hold; a picked scene is on the wrong side of the scene split; the
# cohort written does not hold exactly the picked episodes.  selection.json says what was picked from what.
# Inputs (env; the defaults are the ledger's paths): EXP32_P1, EXP32_R0S, EXP32_P2, EXP32_BASE, EXP32_COHORT,
# EXP32_HOLDOUT_SCENES, EXP32_TRAIN_SCENES, EXP32_NUM_SHARDS, EXP32_WT (default: this checkout), EXP32_PYTHON.
set -euo pipefail
MODE="${1:-}"; OUT="${2:-}"
case "$MODE" in pass2|pass3) SHARDS="${EXP32_NUM_SHARDS:-20}" ;; heldout) SHARDS="${EXP32_NUM_SHARDS:-8}" ;;
  *) echo "usage: cohorts.sh pass2|heldout|pass3 OUT" >&2; exit 2 ;; esac
[[ -n "$OUT" ]] || { echo "usage: cohorts.sh $MODE OUT" >&2; exit 2; }
WT="${EXP32_WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; PY="${EXP32_PYTHON:-/opt/conda/bin/python}"
export EXP32_P1="${EXP32_P1:-/workspace/exp32/collect_a}" EXP32_R0S="${EXP32_R0S:-/workspace/exp31/r0s}" \
  EXP32_P2="${EXP32_P2:-/workspace/exp32/p2}" EXP32_BASE="${EXP32_BASE:-/workspace/exp28/collect_base}" \
  EXP32_COHORT="${EXP32_COHORT:-/workspace/exp28/episodes/cohorts}" \
  EXP32_HOLDOUT_SCENES="${EXP32_HOLDOUT_SCENES:-/workspace/exp28/episodes/splits/heldout_scenes.txt}" \
  EXP32_TRAIN_SCENES="${EXP32_TRAIN_SCENES:-/workspace/exp28/episodes/splits/train_scenes.txt}"
export PYTHONDONTWRITEBYTECODE=1  # no __pycache__ in the checkout: only OUT is written
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/cohorts.log"; }
die() { log "cohorts.sh $MODE: STOPPED: $*" >&2; exit 2; }
[[ -e "$OUT/cohort" ]] && die "$OUT/cohort exists; a cohort is built once (remove it to rebuild)"
finished() {  # $1: a collect_4090.sh output dir -- it has a STATUS_* and every one says DONE or STOPPED
  local f n=0
  for f in "$1"/STATUS_*; do
    [[ -f "$f" ]] || continue
    n=$((n + 1)); grep -q -E '^[^ ]+ (DONE|STOPPED):' "$f" || return 1
  done
  [[ "$n" -gt 0 ]]
}
reads=(); [[ "$MODE" == heldout ]] || reads=("$EXP32_P1" "$EXP32_R0S"); [[ "$MODE" == pass3 ]] && reads+=("$EXP32_P2")
for d in ${reads[@]+"${reads[@]}"}; do
  finished "$d" && continue
  [[ "${EXP32_ALLOW_UNFINISHED:-0}" == 1 ]] || die "$d is still collecting (STATUS_* not DONE/STOPPED)"
  log "EXP32_ALLOW_UNFINISHED=1: $d is not finished, reading what it has committed"
done
rm -rf "$OUT/links" "$OUT/selection.json"
log "mode=$MODE out=$OUT wt=$WT shards<=$SHARDS"
PYTHONPATH="$WT" "$PY" - "$MODE" "$OUT" <<'PY' | tee -a "$OUT/cohorts.log"
import json, os, re, sys, tarfile, time
from collections import Counter
from pathlib import Path
from scripts.tools.build_route_stream import read_scene_list, scene_key  # merge_streams.py's leak check uses these
from src.data.s2_records import iter_record_paths
mode, out = sys.argv[1], Path(sys.argv[2]); E = os.environ
BASE_FAILED_TRAIN, BASE_FAILED_HELDOUT = 1017, 112  # ledger EXP-32 part 1 and B3
def die(msg):  # as the shell's die: to stderr and the log, exit 2
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} cohorts.sh {mode}: STOPPED: {msg}"
    with open(out / "cohorts.log", "a") as log:
        log.write(line + "\n")
    print(line, file=sys.stderr, flush=True); sys.exit(2)
def scan(name, root):  # {(scene ID, episode id): (record path, episode.json)} of root's committed records
    recs = {}
    for tar in iter_record_paths(Path(root) / "records"):
        with tarfile.open(tar, "r:") as t:
            first = t.next()  # EpisodeRecordWriter.commit writes episode.json first
            ep = json.load(t.extractfile(first if first is not None and first.name == "episode.json" else "episode.json"))
        recs[(scene_key(ep["scene_id"]), int(ep["episode_id"]))] = (Path(os.path.abspath(tar)), ep)
    if not recs:
        die(f"{name}: no committed record under {root}/records")
    return recs
def shard(ep):
    m = re.search(r"shard_(\d+)\.json", str(ep.get("episode_list", "")))
    return int(m.group(1)) if m else die(f"collect_base episode_list {ep.get('episode_list')!r} names no shard")
won = lambda ep: bool(ep["outcome"].get("success"))
seeds = lambda recs: dict(sorted(Counter(str(ep.get("protocol_seed")) for _, ep in recs.values()).items()))
held = {scene_key(s) for s in read_scene_list(E["EXP32_HOLDOUT_SCENES"])}
train = {scene_key(s) for s in read_scene_list(E["EXP32_TRAIN_SCENES"])} - held
base = scan("collect_base", E["EXP32_BASE"])
report = {"schema": "heatmapvln-exp32-cohort-v1", "mode": mode, "base": E["EXP32_BASE"]}
if mode == "heldout":
    held_base = {k: v for k, v in base.items() if shard(v[1]) < 20}  # EXP-28 cohort part heldout = shards 0-19
    picked = {k: v[0] for k, v in held_base.items() if not won(v[1])}
    if len(picked) != BASE_FAILED_HELDOUT:
        die(f"collect_base shards 0-19 hold {len(picked)} failed episodes, not {BASE_FAILED_HELDOUT}")
    report.update(base_heldout_records=len(held_base), protocol_seeds={"base": seeds(held_base)})
else:
    universe = {k for k, (_, ep) in base.items() if shard(ep) >= 20 and not won(ep)}
    if len(universe) != BASE_FAILED_TRAIN:
        die(f"collect_base has {len(universe)} failed training episodes, not {BASE_FAILED_TRAIN}")
    src = {"p1": scan("P1", E["EXP32_P1"]), "r0s": scan("R0-S", E["EXP32_R0S"])}
    for name, recs in src.items():
        if set(recs) - universe:
            die(f"{name}: {len(set(recs) - universe)} records are not base-failed training episodes, "
                f"e.g. {sorted(set(recs) - universe)[:3]}")
    p1_failed = {k: v[0] for k, v in src["p1"].items() if not won(v[1])}
    r0s_won = {k for k, v in src["r0s"].items() if won(v[1])}
    picked = {k: tar for k, tar in p1_failed.items() if k not in r0s_won}
    report.update(p1=E["EXP32_P1"], r0s=E["EXP32_R0S"], base_failed_training=len(universe),
                  p1_records=len(src["p1"]), p1_uncommitted=len(universe - set(src["p1"])), p1_failed=len(p1_failed),
                  r0s_records=len(src["r0s"]), r0s_success=len(r0s_won),
                  r0s_success_removed=len(set(p1_failed) & r0s_won), pass2_cohort=len(picked))
    if mode == "pass3":
        src["p2"] = scan("pass 2", E["EXP32_P2"])
        if set(src["p2"]) - set(picked):
            die(f"pass 2: {len(set(src['p2']) - set(picked))} records are not in the pass-2 cohort, "
                f"e.g. {sorted(set(src['p2']) - set(picked))[:3]}")
        p2_won = {k for k, v in src["p2"].items() if won(v[1])}
        report.update(p2=E["EXP32_P2"], p2_records=len(src["p2"]), p2_uncommitted=len(set(picked) - set(src["p2"])),
                      p2_success=len(p2_won))
        picked = {k: tar for k, tar in picked.items() if k not in p2_won}
    report["protocol_seeds"] = {name: seeds(recs) for name, recs in src.items()}
scenes = {k[0] for k in picked}
wrong = sorted(scenes - held) if mode == "heldout" else sorted(scenes - train)
if wrong:
    die(f"{len(wrong)} picked scenes are on the wrong side of the scene split, e.g. {wrong[:3]}")
if not picked:
    die("no episode is left to pick")
for _, tar in sorted(picked.items()):
    d = out / "links" / tar.parent.name
    d.mkdir(parents=True, exist_ok=True)
    for f in (tar, tar.with_name(tar.name + ".sha256")):
        os.symlink(f, d / f.name)
report.update(picked=len(picked), scenes=len(scenes))
(out / "selection.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
print(json.dumps(report, sort_keys=True))
PY
picked=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["picked"])' "$OUT/selection.json")
(( picked < SHARDS )) && SHARDS=$picked
( cd "$WT" && PYTHONPATH="$WT" "$PY" scripts/tools/remaining_cohort.py --cohort "$EXP32_COHORT" \
    --records "$OUT/links" --keep committed --num-shards "$SHARDS" --out "$OUT/cohort" ) >> "$OUT/cohorts.log" 2>&1 \
  || die "remaining_cohort.py refused (see $OUT/cohorts.log)"
total=$("$PY" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["total_episodes"])' "$OUT/cohort/manifest.json")
[[ "$total" == "$picked" ]] || die "the cohort holds $total episodes, $picked were picked"
log "DONE: $MODE cohort of $total episodes over $SHARDS shards in $OUT/cohort"
chown -h 1015:1015 "$OUT" 2>/dev/null || true
chown -R -h 1015:1015 "$OUT/links" "$OUT/cohort" "$OUT/selection.json" "$OUT/cohorts.log" 2>/dev/null || true

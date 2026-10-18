#!/usr/bin/env bash
# EXP-29 R0 cohort (4090, CPU): 200 base-greedy-failed HM3D training episodes (shards >= 20, outcome.success 0),
# drawn with seed 29, as an ordinary cohort over 20 shards (remaining_cohort.py --keep committed on a links dir).
set -euo pipefail
WT=/workspace/exp23/wt_exp28; PY=/opt/conda/bin/python; R0=/workspace/exp29/r0
mkdir -p "$R0"; rm -rf "$R0/failed_links" "$R0/cohort"
"$PY" - "$R0/failed_links" <<'PY'
import glob, json, os, random, re, sys, tarfile
out = sys.argv[1]; failed = []
for sha in sorted(glob.glob("/workspace/exp28/collect_base/records/*/*.s2rec.tar.sha256")):
    tar = sha[:-7]
    with tarfile.open(tar) as t:
        ep = json.load(t.extractfile("episode.json"))
    shard = int(re.search(r"shard_(\d+)\.json", ep["episode_list"]).group(1))
    if shard >= 20 and not ep["outcome"].get("success"):
        failed.append((tar, sha))
pick = sorted(random.Random(29).sample(failed, 200))
for tar, sha in pick:
    d = os.path.join(out, os.path.basename(os.path.dirname(tar))); os.makedirs(d, exist_ok=True)
    os.symlink(tar, os.path.join(d, os.path.basename(tar))); os.symlink(sha, os.path.join(d, os.path.basename(sha)))
print(json.dumps({"failed_training_episodes": len(failed), "picked": len(pick)}))
PY
cd "$WT" && PYTHONPATH="$WT" "$PY" scripts/tools/remaining_cohort.py --cohort /workspace/exp28/episodes/cohorts \
  --records "$R0/failed_links" --keep committed --num-shards 20 --out "$R0/cohort"
"$PY" -c 'import json,sys; m=json.load(open(sys.argv[1])); print({k: m[k] for k in ("num_shards","total_episodes")})' "$R0/cohort/manifest.json"
chown -R 1015:1015 "$R0"

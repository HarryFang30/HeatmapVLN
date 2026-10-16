"""EXP-28: which platform collected each shard; the local training shards that make up the first 400 local episodes."""
import glob, json, re, tarfile, collections
m = json.load(open("/workspace/exp28/episodes/cohorts/manifest.json"))
size = {s["index"]: s["episodes"] for s in m["shards"]}
plat = collections.defaultdict(set); cnt = collections.Counter()
for sha in glob.glob("/workspace/exp28/collect_base/records/*/*.s2rec.tar.sha256"):
    with tarfile.open(sha[:-7]) as t:
        ep = json.load(t.extractfile("episode.json"))
    s = int(re.search(r"shard_(\d+)\.json", ep["episode_list"]).group(1))
    plat[s].add("910b" if not ep["server"]["address"].startswith("127.0.0.1:5247") else "4090"); cnt[s] += 1
done = {int(re.search(r"shard_(\d+)", p).group(1)) for p in glob.glob("/workspace/exp28/collect_base/workers/shard_*/DONE")}
loc = sorted(s for s in plat if s >= 20 and plat[s] == {"4090"})
print("mixed", [s for s in plat if len(plat[s]) > 1])
print("local shards", len(loc), "complete", sum(cnt[s] == size[s] and s in done for s in loc))
acc, spec = 0, []
for s in loc:
    if not (cnt[s] == size[s] and s in done):
        print(" incomplete", s, cnt[s], size[s]); continue
    spec.append(s); acc += size[s]
    if acc >= 400: break
print("phase0 local: n_shards", len(spec), "episodes", acc, "spec", ",".join(map(str, spec)))
print("ext train shards", sorted(s for s in plat if s >= 20 and plat[s] == {"910b"})[:12])
print("heldout done", sum(1 for s in range(20) if s in done), "records", sum(cnt[s] for s in range(20)))

"""EXP-28 retry "retrain_2_epochs": a stream that holds every shard of a merged stream twice (two passes).

The trainer reads a stream once (steps = total_samples / global_batch), so two epochs are two copies of the
shards: pass 1 is the stream in its own order, pass 2 the same shards in a seeded permutation.  Shards are hard
links (the trainer's reader deletes a shard only on the 910B side, after the push), the manifest is the input's
with num_shards / total_samples doubled, epochs 2.0 and the shard list rewritten.
"""
import argparse, json, os, random, shutil, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--seed", type=int, default=2)
a = ap.parse_args()
src, dst = Path(a.src), Path(a.dst)
man = json.loads((src / "stream_manifest.json").read_text())
shards = man["shards"]
n = len(shards)
assert n == int(man["num_shards"]), "manifest shard count mismatch"
order = list(range(n)) + random.Random(a.seed).sample(range(n), n)
tmp = dst.with_name(dst.name + ".part")
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir(parents=True)
new = []
for i, j in enumerate(order):
    s = shards[j]
    name = f"stream-{i:06d}.tar"
    os.link(src / s["name"], tmp / name)
    e = dict(s)
    e["name"] = name
    new.append(e)
man.update({"num_shards": 2 * n, "total_samples": 2 * int(man["total_samples"]), "epochs": 2.0, "shards": new,
            "doubled_from": {"dir": str(src.resolve()), "num_shards": n, "pass2_seed": a.seed}})
(tmp / "stream_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
if dst.exists():
    shutil.rmtree(dst)
tmp.rename(dst)
print(json.dumps({"num_shards": 2 * n, "total_samples": man["total_samples"]}))

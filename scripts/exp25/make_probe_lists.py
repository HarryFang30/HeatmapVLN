#!/usr/bin/env python3
"""EXP-25 navigation probes B1/B2: a fixed, scene-stratified episode sample of one R2R-CE split.

Rule (fixed before any probe run, docs/experiments/README.md EXP-25 B1/B2):
- population: every episode of ``--split-file`` (e.g. val_seen.json.gz, envdrop.json.gz);
- strata: scenes; each scene gets ``n * scene_episodes / all_episodes`` episodes, rounded by largest
  remainder (ties broken by scene name), so the sample keeps the split's scene mix;
- within a scene, ``random.Random(seed)`` draws without replacement from the episode ids in ascending
  order, scenes visited in sorted order with one generator for the whole draw;
- the sample, sorted by (scene, episode_id), is dealt round-robin into ``--shards`` shards.

Only episode properties decide; no result is read.  Writes a plan directory that
scripts/run_ppa_r2r_val_unseen_cuda.sh accepts through PPA_EVAL_LOCKED_PLAN / PPA_EVAL_EPISODE_LISTS_DIR:

  <out>/cohorts/shard_0N.json             episode list (cohort schema)
  <out>/cohorts/dataset_shard_0N.json.gz  that shard's episodes only, the split's other top-level keys kept
  <out>/lists/shard_0N.json               the same lists again: a lists directory other than the cohorts
                                          one makes the launcher treat the run as a subset (no 1839 merge)
  <out>/tools -> <locked plan>/tools      merge tool and client-side modules on the RPC PYTHONPATH
  <out>/manifest.json                     rule, seed, per-scene allocation, sha256 of inputs and outputs

  python scripts/exp25/make_probe_lists.py --split-file .../val_seen/val_seen.json.gz --n 300 --seed 25 \
      --shards 3 --tools <locked plan>/tools --out /workspace/exp23/probes/val_seen/plan
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_name(scene_id: str) -> str:
    """"mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb" or "zsNo4HB9uLZ" -> "zsNo4HB9uLZ"."""
    return Path(str(scene_id)).stem


def allocate(counts: Dict[str, int], n: int) -> Dict[str, int]:
    """Largest-remainder proportional allocation of ``n`` over scenes; never more than a scene holds."""
    total = sum(counts.values())
    if n > total:
        raise ValueError(f"asked for {n} episodes, the split has {total}")
    quota = {s: n * c / total for s, c in counts.items()}
    alloc = {s: min(int(q), counts[s]) for s, q in quota.items()}
    order = sorted(counts, key=lambda s: (-(quota[s] - int(quota[s])), s))
    i = 0
    while sum(alloc.values()) < n:
        s = order[i % len(order)]
        if alloc[s] < counts[s]:
            alloc[s] += 1
        i += 1
    return alloc


def draw(episodes: List[dict], n: int, seed: int) -> Tuple[List[Tuple[str, int]], Dict[str, dict]]:
    by_scene: Dict[str, List[int]] = defaultdict(list)
    for ep in episodes:
        by_scene[scene_name(ep["scene_id"])].append(int(ep["episode_id"]))
    alloc = allocate({s: len(v) for s, v in by_scene.items()}, n)
    rng = random.Random(seed)
    chosen: List[Tuple[str, int]] = []
    per_scene = {}
    for scene in sorted(by_scene):
        ids = sorted(by_scene[scene])
        picked = sorted(rng.sample(ids, alloc[scene])) if alloc[scene] else []
        chosen += [(scene, e) for e in picked]
        per_scene[scene] = {"episodes": len(ids), "chosen": len(picked)}
    return sorted(chosen), per_scene


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split-file", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--tools", type=Path, help="locked plan tools directory to link as <out>/tools")
    args = ap.parse_args(argv)
    if not 1 <= args.shards <= 8:
        raise SystemExit("--shards must be 1-8 (the launcher takes shard ids 0-7)")

    with gzip.open(args.split_file, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    episodes = data["episodes"]
    chosen, per_scene = draw(episodes, args.n, args.seed)
    by_key = {(scene_name(e["scene_id"]), int(e["episode_id"])): e for e in episodes}

    cohorts, lists = args.out / "cohorts", args.out / "lists"
    cohorts.mkdir(parents=True, exist_ok=True)
    lists.mkdir(parents=True, exist_ok=True)
    rest = {k: v for k, v in data.items() if k != "episodes"}
    shards = []
    for shard in range(args.shards):
        keys = chosen[shard :: args.shards]
        listing = {
            "cohort_name": f"exp25_{args.split_file.name.split('.')[0]}_n{args.n}_seed{args.seed}_{shard:02d}_of_{args.shards:02d}",
            "dataset_sha256": sha256_file(args.split_file),
            "episodes": [{"episode_id": e, "scene_id": s} for s, e in keys],
        }
        text = json.dumps(listing, indent=2) + "\n"
        (cohorts / f"shard_0{shard}.json").write_text(text, encoding="utf-8")
        (lists / f"shard_0{shard}.json").write_text(text, encoding="utf-8")
        subset = dict(rest, episodes=[by_key[k] for k in keys])
        with gzip.open(cohorts / f"dataset_shard_0{shard}.json.gz", "wt", encoding="utf-8") as fh:
            json.dump(subset, fh)
        shards.append({"shard": shard, "episodes": len(keys),
                       "list_sha256": sha256_file(cohorts / f"shard_0{shard}.json"),
                       "dataset_sha256": sha256_file(cohorts / f"dataset_shard_0{shard}.json.gz")})
    if args.tools is not None:
        link = args.out / "tools"
        if link.is_symlink() or link.exists():
            if not (link.is_symlink() and Path(os.readlink(link)) == args.tools):
                raise SystemExit(f"{link} exists and is not a link to {args.tools}")
        else:
            link.symlink_to(args.tools)
    manifest = {
        "schema": "exp25-probe-sample-v1",
        "split_file": str(args.split_file),
        "split_sha256": sha256_file(args.split_file),
        "population": len(episodes),
        "n": len(chosen),
        "seed": args.seed,
        "rule": "scenes as strata, largest-remainder proportional allocation (ties by scene name); "
                "random.Random(seed).sample over ascending episode ids, scenes in sorted order, one generator; "
                "sample sorted by (scene, episode_id) and dealt round-robin into shards",
        "scenes": len(per_scene),
        "scenes_sampled": sum(1 for v in per_scene.values() if v["chosen"]),
        "per_scene": per_scene,
        "shards": shards,
        "episodes": [{"scene_id": s, "episode_id": e} for s, e in chosen],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"n": len(chosen), "per_shard": [s["episodes"] for s in shards],
                      "scenes": len(per_scene), "scenes_sampled": manifest["scenes_sampled"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

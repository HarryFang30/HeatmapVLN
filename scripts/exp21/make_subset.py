#!/usr/bin/env python3
"""EXP-21 sensitivity subset: a fixed, stratified subset of the locked R2R val_unseen cohorts.

Rule (fixed before any sensitivity run, docs/experiments/README.md EXP-21):
- strata: quartiles of ``info.geodesic_distance`` over all episodes of the 8 cohort shards;
- the same number per stratum (``n / 4``);
- within a stratum, episodes ordered by ``sha1("<scene>:<episode_id>")`` ascending, the first ones taken;
- each shard's list keeps the cohort's episode order, in the cohort file's schema, so the client reads it with
  ``--episode_list`` against the unchanged ``dataset_shard_0N.json.gz``.

Only episode properties decide; no result is read.  Writes ``shard_0N.json`` and ``manifest.json`` (rule, quartile
edges, per-stratum and per-shard counts, sha256 of inputs and outputs) to ``--out``.

  python scripts/exp21/make_subset.py --cohorts <plan>/cohorts --out /workspace/exp21/subset500 [--n 500]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

NUM_SHARDS = 8
STRATA = 4


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode_key(scene: str, episode_id: int) -> str:
    return hashlib.sha1(f"{scene}:{int(episode_id)}".encode()).hexdigest()


def scene_name(scene_id: str) -> str:
    """"mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb" or "zsNo4HB9uLZ" -> "zsNo4HB9uLZ"."""
    return Path(str(scene_id)).stem


def load_cohorts(cohorts: Path) -> Tuple[Dict[int, dict], Dict[Tuple[str, int], float]]:
    """(shard index -> cohort list file content, (scene, episode_id) -> geodesic distance)."""
    lists, dist = {}, {}
    for shard in range(NUM_SHARDS):
        lists[shard] = json.loads((cohorts / f"shard_0{shard}.json").read_text(encoding="utf-8"))
        with gzip.open(cohorts / f"dataset_shard_0{shard}.json.gz", "rt", encoding="utf-8") as fh:
            for ep in json.load(fh)["episodes"]:
                dist[(scene_name(ep["scene_id"]), int(ep["episode_id"]))] = float(ep["info"]["geodesic_distance"])
    return lists, dist


def select(lists: Dict[int, dict], dist: Dict[Tuple[str, int], float], n: int) -> Tuple[set, dict]:
    if n % STRATA:
        raise ValueError(f"n must be a multiple of {STRATA}")
    eps = [(e["scene_id"], int(e["episode_id"])) for s in sorted(lists) for e in lists[s]["episodes"]]
    missing = [e for e in eps if e not in dist]
    if missing:
        raise ValueError(f"{len(missing)} cohort episodes have no geodesic distance, e.g. {missing[:3]}")
    d = np.array([dist[e] for e in eps])
    edges = np.quantile(d, np.linspace(0, 1, STRATA + 1))
    stratum = np.clip(np.searchsorted(edges[1:-1], d, side="right"), 0, STRATA - 1)
    per = n // STRATA
    chosen, counts = set(), []
    for k in range(STRATA):
        members = sorted((episode_key(*e), e) for e, s in zip(eps, stratum) if s == k)
        if len(members) < per:
            raise ValueError(f"stratum {k} has {len(members)} episodes < {per}")
        chosen.update(e for _, e in members[:per])
        counts.append({"stratum": k, "geodesic_m": [round(float(edges[k]), 3), round(float(edges[k + 1]), 3)],
                       "episodes": len(members), "chosen": per})
    return chosen, {"quartile_edges_m": [round(float(x), 4) for x in edges], "strata": counts,
                    "total_episodes": len(eps)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohorts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args(argv)
    lists, dist = load_cohorts(args.cohorts)
    chosen, info = select(lists, dist, args.n)
    args.out.mkdir(parents=True, exist_ok=True)
    shards = []
    for shard, cohort in sorted(lists.items()):
        kept = [e for e in cohort["episodes"] if (e["scene_id"], int(e["episode_id"])) in chosen]
        out = {**cohort, "cohort_name": f"exp21_subset{args.n}_{shard:02d}_of_08", "episodes": kept,
               "parent_cohort": cohort.get("cohort_name")}
        path = args.out / f"shard_0{shard}.json"
        path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        shards.append({"shard": shard, "episodes": len(kept), "sha256": sha256_file(path)})
    manifest = {
        "schema": "exp21-sensitivity-subset-v1", "n": args.n, **info, "shards": shards,
        "rule": "quartiles of geodesic distance over the 1839 cohort episodes; n/4 per quartile, lowest "
                "sha1('<scene>:<episode_id>') first; cohort order kept per shard",
        "inputs": {f"shard_0{s}.json": sha256_file(args.cohorts / f"shard_0{s}.json") for s in range(NUM_SHARDS)},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"n": sum(s["episodes"] for s in shards), "per_shard": [s["episodes"] for s in shards],
                      "edges_m": info["quartile_edges_m"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

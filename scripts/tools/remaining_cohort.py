#!/usr/bin/env python3
"""The cohort a later DAgger round collects: an earlier cohort minus what it committed.

EXP-23 round 1 stopped after about 4,000 of its cohort's 7,206 episodes (user,
2026-10-13), and round 2 collects the rest with round 1's adapter.  The output is an
ordinary ``build_r2r_train_cohorts.py`` cohort, so ``run_s2_dagger_collect_cuda.sh``
takes it as is: every source shard that still holds an uncollected episode, renumbered
0..K-1 in source order, listing only those episodes, with its dataset shard filtered to
the same episodes (every other top-level key unchanged).  An episode counts as
collected when its record is committed -- the tar and its ``.sha256`` marker
(``src/data/s2_records.py``); an episode in flight when the round stopped is not.

    remaining_cohort.py --cohort round1/cohorts --records round1/records --out round2/cohorts

The manifest keeps the source's fields, recounts the totals, and records where it came
from (``derived_from``).  The output directory is created whole or not at all.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.build_r2r_train_cohorts import (  # noqa: E402
    SCHEMA,
    CohortError,
    deterministic_gzip_json,
    scene_name,
    sha256_file,
)
from src.data.s2_records import record_path  # noqa: E402


def _committed(records: Path, scene: str, episode_id: int) -> bool:
    path = record_path(records, scene, episode_id)
    return path.is_file() and path.with_name(path.name + ".sha256").is_file()


def remaining(cohort: Path, records: Path, out: Path) -> dict:
    manifest_path = cohort / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("split") != "train":
        raise CohortError(f"{manifest_path} is not a {SCHEMA} train manifest")
    if out.exists() and any(out.iterdir()):
        raise CohortError(f"refusing to write into a non-empty {out}")

    kept = []  # (source record, shard document, kept episodes)
    collected = 0
    for record in sorted(manifest["shards"], key=lambda r: int(r["index"])):
        document = json.loads((cohort / record["path"]).read_text(encoding="utf-8"))
        left = [e for e in document["episodes"] if not _committed(records, str(e["scene_id"]), int(e["episode_id"]))]
        collected += len(document["episodes"]) - len(left)
        if left:
            kept.append((record, document, left))
    if not kept:
        raise CohortError(f"every episode of {cohort} is already committed under {records}")

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        shard_records, scenes, routes, total = [], set(), set(), 0
        for index, (source, document, left) in enumerate(kept):
            want = {(str(e["scene_id"]), int(e["episode_id"])) for e in left}
            with gzip.open(cohort / source["dataset_path"], "rt", encoding="utf-8") as handle:
                dataset = json.load(handle)
            episodes = [e for e in dataset["episodes"] if (scene_name(e["scene_id"]), int(e["episode_id"])) in want]
            if len(episodes) != len(want):
                raise CohortError(f"{source['dataset_path']} lacks {len(want) - len(episodes)} of its listed episodes")
            name, dataset_name = f"shard_{index:03d}.json", f"dataset_shard_{index:03d}.json.gz"
            shard = dict(document)
            shard.update({
                "cohort_name": f"r2r_train_s2_collect_{index:03d}_of_{len(kept):03d}",
                "num_shards": len(kept),
                "shard_index": index,
                "source_shard": int(source["index"]),
                "episodes": left,
            })
            (staging / name).write_text(json.dumps(shard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (staging / dataset_name).write_bytes(deterministic_gzip_json(dict(dataset, episodes=episodes)))
            shard_routes = {(str(e["scene_id"]), int(e["trajectory_id"])) for e in left}
            shard_records.append({
                "index": index,
                "path": name,
                "sha256": sha256_file(staging / name),
                "dataset_path": dataset_name,
                "dataset_sha256": sha256_file(staging / dataset_name),
                "episodes": len(left),
                "routes": len(shard_routes),
                "scenes": sorted({str(e["scene_id"]) for e in left}),
                "source_shard": int(source["index"]),
            })
            scenes.update(str(e["scene_id"]) for e in left)
            routes.update(shard_routes)
            total += len(left)
        derived = dict(manifest)
        derived.update({
            "num_shards": len(kept),
            "total_episodes": total,
            "total_routes": len(routes),
            "scenes": sorted(scenes),
            "shards": shard_records,
            "derived_from": {
                "cohort": str(cohort),
                "manifest_sha256": sha256_file(manifest_path),
                "records": str(records),
                "committed_episodes_dropped": collected,
                "source_episodes": int(manifest.get("total_episodes", 0)),
            },
        })
        (staging / "manifest.json").write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if out.exists():
            out.rmdir()  # empty, checked above
        os.replace(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return derived


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort", required=True, help="the earlier round's cohort directory")
    parser.add_argument("--records", required=True, help="the earlier round's record root")
    parser.add_argument("--out", required=True, help="the new cohort directory (created)")
    args = parser.parse_args(argv)
    try:
        derived = remaining(Path(args.cohort).resolve(), Path(args.records).resolve(), Path(args.out).resolve())
    except CohortError as exc:
        print(f"remaining_cohort: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"num_shards": derived["num_shards"], "total_episodes": derived["total_episodes"],
                      "dropped": derived["derived_from"]["committed_episodes_dropped"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

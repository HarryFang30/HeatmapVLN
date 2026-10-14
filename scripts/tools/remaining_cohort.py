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

EXP-25's closed-loop canary needs the opposite subset: the episodes of the held-out scenes
that round 1 *did* commit, so an adapter runs exactly the episodes whose base-model outcome
is on record (392).  ``--keep committed`` keeps those instead, ``--scenes`` (or
``--holdout``: the six scenes of ``holdout_scenes()``) restricts the scenes, and
``--num-shards K`` deals the kept routes afresh over K shards (whole routes, least-loaded
first, ``build_r2r_train_cohorts.deal_shards``): the kept episodes are a few per source
shard, and every shard costs the launcher a client start.

    remaining_cohort.py --cohort round1/cohorts --records round1/records --keep committed \\
        --holdout --num-shards 28 --out roundA/canary_cohort
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
    deal_shards,
    deterministic_gzip_json,
    route_key,
    scene_name,
    sha256_file,
)
from src.data.s2_records import record_path  # noqa: E402


def _committed(records: Path, scene: str, episode_id: int) -> bool:
    path = record_path(records, scene, episode_id)
    return path.is_file() and path.with_name(path.name + ".sha256").is_file()


def remaining(cohort: Path, records: Path, out: Path) -> dict:
    """The uncommitted rest of ``cohort`` (round N's collection)."""
    return subset(cohort, records, out, keep="uncommitted")


def subset(
    cohort: Path,
    records: Path,
    out: Path,
    *,
    keep: str = "uncommitted",
    scenes: set[str] | None = None,
    num_shards: int | None = None,
) -> dict:
    """The episodes of ``cohort`` that are not yet (``keep="uncommitted"``) or are
    (``"committed"``) committed under ``records``, optionally only those in ``scenes``, in
    their source shards or dealt over ``num_shards`` new ones."""
    if keep not in ("uncommitted", "committed"):
        raise CohortError(f"keep must be uncommitted or committed, not {keep!r}")
    manifest_path = cohort / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("split") != "train":
        raise CohortError(f"{manifest_path} is not a {SCHEMA} train manifest")
    if out.exists() and any(out.iterdir()):
        raise CohortError(f"refusing to write into a non-empty {out}")

    def wanted(e: dict) -> bool:
        if scenes is not None and str(e["scene_id"]) not in scenes:
            return False
        return _committed(records, str(e["scene_id"]), int(e["episode_id"])) == (keep == "committed")

    kept = []  # (source record, shard document, kept list entries, kept dataset episodes, dataset)
    dropped = 0
    for record in sorted(manifest["shards"], key=lambda r: int(r["index"])):
        document = json.loads((cohort / record["path"]).read_text(encoding="utf-8"))
        left = [e for e in document["episodes"] if wanted(e)]
        dropped += len(document["episodes"]) - len(left)
        if not left:
            continue
        want = {(str(e["scene_id"]), int(e["episode_id"])) for e in left}
        with gzip.open(cohort / record["dataset_path"], "rt", encoding="utf-8") as handle:
            dataset = json.load(handle)
        episodes = [e for e in dataset["episodes"] if (scene_name(e["scene_id"]), int(e["episode_id"])) in want]
        if len(episodes) != len(want):
            raise CohortError(f"{record['dataset_path']} lacks {len(want) - len(episodes)} of its listed episodes")
        kept.append((record, document, left, episodes, dataset))
    if not kept:
        raise CohortError(f"no episode of {cohort} is left to keep ({keep} under {records}, scenes {scenes})")

    # (shard document template, list entries, dataset episodes, dataset template, source shards)
    groups = []
    if num_shards is None:
        for record, document, left, episodes, dataset in kept:
            groups.append((document, left, episodes, dataset, [int(record["index"])]))
    else:
        source_of: dict[tuple[str, int], int] = {}
        routes: dict[tuple[str, int], list[dict]] = {}
        for record, _, _, episodes, _ in kept:
            for e in episodes:
                routes.setdefault(route_key(e), []).append(e)
                source_of[(scene_name(e["scene_id"]), int(e["episode_id"]))] = int(record["index"])
        dealt = deal_shards(list(routes.items()), num_shards=int(num_shards), shard_size=None)
        template = {k: v for k, v in kept[0][1].items() if k != "source_shard"}
        for episodes in dealt:
            entries = [{"scene_id": scene_name(e["scene_id"]), "episode_id": int(e["episode_id"]),
                        "trajectory_id": int(e["trajectory_id"])} for e in episodes]
            sources = sorted({source_of[(x["scene_id"], x["episode_id"])] for x in entries})
            groups.append((template, entries, episodes, kept[0][4], sources))

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        shard_records, all_scenes, routes_seen, total = [], set(), set(), 0
        for index, (document, left, episodes, dataset, sources) in enumerate(groups):
            name, dataset_name = f"shard_{index:03d}.json", f"dataset_shard_{index:03d}.json.gz"
            source_field = {"source_shard": sources[0]} if num_shards is None else {"source_shards": sources}
            shard = dict(document)
            shard.update({
                "cohort_name": f"r2r_train_s2_collect_{index:03d}_of_{len(groups):03d}",
                "num_shards": len(groups),
                "shard_index": index,
                "episodes": left,
                **source_field,
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
                **source_field,
            })
            all_scenes.update(str(e["scene_id"]) for e in left)
            routes_seen.update(shard_routes)
            total += len(left)
        derived = dict(manifest)
        derived.update({
            "num_shards": len(groups),
            "total_episodes": total,
            "total_routes": len(routes_seen),
            "scenes": sorted(all_scenes),
            "shards": shard_records,
            "derived_from": {
                "cohort": str(cohort),
                "manifest_sha256": sha256_file(manifest_path),
                "records": str(records),
                "keep": keep,
                "scenes": sorted(scenes) if scenes is not None else None,
                "regrouped_num_shards": num_shards,
                # what was left out: the committed episodes for "uncommitted", every other one for "committed"
                ("committed_episodes_dropped" if keep == "uncommitted" else "episodes_dropped"): dropped,
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
    parser.add_argument("--keep", choices=("uncommitted", "committed"), default="uncommitted")
    parser.add_argument("--scenes", default="", help="comma-separated scene names to keep (default: all)")
    parser.add_argument("--holdout", action="store_true", help="keep only the six held-out scenes (holdout_scenes())")
    parser.add_argument("--num-shards", type=int, default=0, help="deal the kept routes over this many shards (0: source shards)")
    args = parser.parse_args(argv)
    scenes = {s for s in args.scenes.split(",") if s} or None
    if args.holdout:
        if scenes is not None:
            print("remaining_cohort: give --scenes or --holdout, not both", file=sys.stderr)
            return 2
        from src.data.s2_sft_dataset import holdout_scenes

        scenes = set(holdout_scenes())
    try:
        derived = subset(Path(args.cohort).resolve(), Path(args.records).resolve(), Path(args.out).resolve(),
                         keep=args.keep, scenes=scenes, num_shards=args.num_shards or None)
    except CohortError as exc:
        print(f"remaining_cohort: {exc}", file=sys.stderr)
        return 2
    source = derived["derived_from"]
    print(json.dumps({"num_shards": derived["num_shards"], "total_episodes": derived["total_episodes"],
                      "total_routes": derived["total_routes"], "scenes": derived["scenes"],
                      "dropped": source.get("committed_episodes_dropped", source.get("episodes_dropped"))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

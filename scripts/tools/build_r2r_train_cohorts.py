#!/usr/bin/env python3
"""Route-disjoint R2R-CE train shards for the EXP-23 System2 collection.

scripts/run_s2_dagger_collect_cuda.sh runs one client per shard, the same way the
val_unseen evaluation runs the locked cohorts, so a shard here has the locked
cohorts' two files (evaluation_plans/.../tools/build_shards.py):

* ``shard_{i:03d}.json`` -- the ``--episode_list`` the client pins its episodes with;
* ``dataset_shard_{i:03d}.json.gz`` -- the ``--data_path``: train.json.gz with only
  those episodes and every other top-level key (``instruction_vocab``) unchanged.

Selection (docs/experiments/exp23-s2-dagger-design.md §2 A):

* a route is ``(scene, trajectory_id)``; train has 3603 of them, nearly all with
  three instructions.  ``--episodes-per-route K`` keeps K of each route's
  instructions (0 keeps all), chosen by a seeded hash, so a smaller round is a
  prefix-free subset rather than whatever came first in the file;
* whole routes go to one shard, so no two shards (and no train/held-out split made
  later by scene) share a route's paraphrases;
* routes are shuffled by a seeded hash and dealt to the least-loaded shard; inside a
  shard episodes are grouped by scene, which is the order Habitat's episode
  iterator visits them in anyway;
* ``--exclude-scenes`` drops scenes by name (an unknown name is an error, not a
  silent no-op); ``--max-routes`` takes the first routes of the shuffle for a pilot.

Nothing is random but the seed, so the same arguments rebuild byte-identical files,
and the output directory is created whole or not at all.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "heatmapvln-r2r-train-cohorts-v1"


class CohortError(RuntimeError):
    """An input that would make the shards wrong; nothing is written."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_name(raw_scene_id: str) -> str:
    """``mp3d/<scene>/<scene>.glb`` -> ``<scene>``, as the client keys episodes."""
    scene = PurePosixPath(str(raw_scene_id)).parent.name
    if not scene:
        raise CohortError(f"cannot derive a scene name from {raw_scene_id!r}")
    return scene


def _digest(seed: int, *parts: object) -> str:
    return hashlib.sha256("\x00".join([str(seed), *map(str, parts)]).encode("utf-8")).hexdigest()


def deterministic_gzip_json(value: object) -> bytes:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as stream:
        stream.write(payload)
    return buffer.getvalue()


def load_dataset(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, list) or not episodes:
        raise CohortError(f"{path} has no 'episodes' array")
    seen: set[tuple[str, int]] = set()
    for index, episode in enumerate(episodes):
        try:
            key = (scene_name(episode["scene_id"]), int(episode["episode_id"]))
            int(episode["trajectory_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CohortError(f"episode[{index}] lacks scene_id/episode_id/trajectory_id: {exc}") from exc
        if key in seen:
            raise CohortError(f"duplicate episode key {key}")
        seen.add(key)
    return payload


def select_routes(
    episodes: list[dict[str, Any]],
    *,
    seed: int,
    episodes_per_route: int,
    exclude_scenes: set[str],
    max_routes: int | None,
) -> list[tuple[tuple[str, int], list[dict[str, Any]]]]:
    """Routes in shuffled order, each with the episodes kept from it."""
    scenes = {scene_name(episode["scene_id"]) for episode in episodes}
    unknown = sorted(exclude_scenes - scenes)
    if unknown:
        raise CohortError(f"--exclude-scenes names scenes the dataset does not have: {unknown}")
    if episodes_per_route < 0:
        raise CohortError("--episodes-per-route must be >= 0 (0 keeps every instruction)")
    routes: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        scene = scene_name(episode["scene_id"])
        if scene not in exclude_scenes:
            routes[(scene, int(episode["trajectory_id"]))].append(episode)
    ordered = sorted(routes, key=lambda route: (_digest(seed, "route", *route), route))
    if max_routes is not None:
        if max_routes <= 0:
            raise CohortError("--max-routes must be positive")
        ordered = ordered[:max_routes]
    selected = []
    for route in ordered:
        kept = sorted(
            routes[route],
            key=lambda episode: (_digest(seed, "instruction", *route, int(episode["episode_id"])), int(episode["episode_id"])),
        )
        if episodes_per_route:
            kept = kept[:episodes_per_route]
        selected.append((route, kept))
    if not selected:
        raise CohortError("no route is left to shard")
    return selected


def deal_shards(
    routes: list[tuple[tuple[str, int], list[dict[str, Any]]]],
    *,
    num_shards: int | None,
    shard_size: int | None,
) -> list[list[dict[str, Any]]]:
    """Whole routes to the least-loaded shard (ties: lowest index), in shuffle order."""
    total = sum(len(kept) for _, kept in routes)
    if (num_shards is None) == (shard_size is None):
        raise CohortError("give exactly one of --num-shards and --shard-size")
    if shard_size is not None:
        if shard_size <= 0:
            raise CohortError("--shard-size must be positive")
        num_shards = -(-total // shard_size)
    if num_shards <= 0 or num_shards > len(routes):
        raise CohortError(f"{num_shards} shards for {len(routes)} routes: every shard needs a route")
    if num_shards > 1000:
        raise CohortError("at most 1000 shards (three-digit names)")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
    for _, kept in routes:
        target = min(range(num_shards), key=lambda index: (len(shards[index]), index))
        shards[target].extend(kept)
    for shard in shards:
        # Scene by scene, as Habitat's iterator groups them; the route shuffle inside.
        order = {id(episode): position for position, episode in enumerate(shard)}
        shard.sort(key=lambda episode: (scene_name(episode["scene_id"]), order[id(episode)]))
    return shards


def episode_key(episode: dict[str, Any]) -> tuple[str, int]:
    return scene_name(episode["scene_id"]), int(episode["episode_id"])


def route_key(episode: dict[str, Any]) -> tuple[str, int]:
    return scene_name(episode["scene_id"]), int(episode["trajectory_id"])


def build(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise CohortError(f"refusing to write into a non-empty {output_dir}")
    raw = load_dataset(dataset_path)
    dataset_sha = sha256_file(dataset_path)
    exclude = {scene for item in args.exclude_scenes for scene in item.split(",") if scene}
    routes = select_routes(
        raw["episodes"],
        seed=args.seed,
        episodes_per_route=args.episodes_per_route,
        exclude_scenes=exclude,
        max_routes=args.max_routes,
    )
    shards = deal_shards(routes, num_shards=args.num_shards, shard_size=args.shard_size)

    # Route-disjoint and exact, checked on what is about to be written.
    selected = [episode_key(episode) for _, kept in routes for episode in kept]
    written = [episode_key(episode) for shard in shards for episode in shard]
    if sorted(selected) != sorted(written) or len(set(written)) != len(written):
        raise CohortError("internal error: the shards do not cover the selection exactly once")
    owner: dict[tuple[str, int], int] = {}
    for index, shard in enumerate(shards):
        for episode in shard:
            if owner.setdefault(route_key(episode), index) != index:
                raise CohortError(f"internal error: route {route_key(episode)} is in two shards")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        records = []
        for index, shard in enumerate(shards):
            name = f"shard_{index:03d}.json"
            dataset_name = f"dataset_shard_{index:03d}.json.gz"
            document = {
                "cohort_name": f"r2r_train_s2_collect_{index:03d}_of_{len(shards):03d}",
                "split": "train",
                "dataset_sha256": dataset_sha,
                "num_shards": len(shards),
                "shard_index": index,
                "episodes": [
                    {
                        "scene_id": scene_name(episode["scene_id"]),
                        "episode_id": int(episode["episode_id"]),
                        "trajectory_id": int(episode["trajectory_id"]),
                    }
                    for episode in shard
                ],
            }
            (staging / name).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            dataset_document = dict(raw)
            dataset_document["episodes"] = shard
            (staging / dataset_name).write_bytes(deterministic_gzip_json(dataset_document))
            records.append(
                {
                    "index": index,
                    "path": name,
                    "sha256": sha256_file(staging / name),
                    "dataset_path": dataset_name,
                    "dataset_sha256": sha256_file(staging / dataset_name),
                    "episodes": len(shard),
                    "routes": len({route_key(episode) for episode in shard}),
                    "scenes": sorted({scene_name(episode["scene_id"]) for episode in shard}),
                }
            )
        manifest = {
            "schema": SCHEMA,
            "split": "train",
            "dataset": str(dataset_path),
            "dataset_sha256": dataset_sha,
            "seed": int(args.seed),
            "episodes_per_route": int(args.episodes_per_route),
            "max_routes": args.max_routes,
            "exclude_scenes": sorted(exclude),
            "num_shards": len(shards),
            "total_episodes": len(written),
            "total_routes": len(routes),
            "scenes": sorted({scene for scene, _ in owner}),
            "route_disjoint": True,
            "shards": records,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if output_dir.exists():
            output_dir.rmdir()  # empty, checked above
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="R2R-CE train.json.gz")
    parser.add_argument("--output-dir", required=True, help="new (or empty) directory for the shards")
    parser.add_argument("--seed", type=int, default=0, help="seed of the route shuffle and instruction choice")
    parser.add_argument(
        "--episodes-per-route", type=int, default=0, help="instructions kept per route; 0 keeps all (default)"
    )
    shards = parser.add_mutually_exclusive_group(required=True)
    shards.add_argument("--num-shards", type=int, default=None)
    shards.add_argument(
        "--shard-size", type=int, default=None, help="target episodes per shard: ceil(total / size) shards"
    )
    parser.add_argument(
        "--exclude-scenes",
        action="append",
        default=[],
        help="comma-separated scene names to leave out; repeatable",
    )
    parser.add_argument("--max-routes", type=int, default=None, help="keep only the first N shuffled routes")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build(args)
    except CohortError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    summary = {key: manifest[key] for key in ("num_shards", "total_episodes", "total_routes", "seed")}
    summary["scenes"] = len(manifest["scenes"])
    summary["output_dir"] = str(Path(args.output_dir).resolve())
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

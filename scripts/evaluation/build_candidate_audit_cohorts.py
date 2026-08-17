#!/usr/bin/env python3
"""Build and verify scene-balanced, route-diverse R2R audit shards."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "candidate-audit-balanced-cohort-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_int(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _scene_id(episode: dict[str, Any]) -> str:
    parts = str(episode["scene_id"]).rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"invalid R2R scene_id: {episode['scene_id']!r}")
    return parts[-2]


def _episode_key(episode: dict[str, Any]) -> tuple[str, int]:
    return _scene_id(episode), int(episode["episode_id"])


def _route_key(episode: dict[str, Any]) -> tuple[str, str]:
    return _scene_id(episode), str(episode["trajectory_id"])


def _instruction_text(episode: dict[str, Any]) -> str:
    instruction = episode.get("instruction")
    if isinstance(instruction, dict):
        return str(instruction.get("instruction_text") or "")
    return str(instruction or "")


def _json_bytes(value: Any, *, compact: bool = False) -> bytes:
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if compact:
        options["separators"] = (",", ":")
    else:
        options["indent"] = 2
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(temp, 0o644)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _read_dataset(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else path.open
    if path.suffix == ".gz":
        with opener(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        with opener("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), list):
        raise ValueError(f"invalid R2R dataset: {path}")
    return payload


def _select_episodes(
    episodes: list[dict[str, Any]],
    *,
    total: int,
    seed: int,
    excluded_scenes: set[str],
) -> list[dict[str, Any]]:
    routes: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for episode in episodes:
        scene = _scene_id(episode)
        if scene in excluded_scenes:
            continue
        routes[scene][str(episode["trajectory_id"])].append(episode)
    if not routes:
        raise ValueError("no eligible scenes remain after exclusions")

    ordered_routes: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {}
    for scene, scene_routes in routes.items():
        values = list(scene_routes.items())
        values.sort(key=lambda item: (_stable_int(seed, "route", scene, item[0]), item[0]))
        ordered_routes[scene] = values

    capacity = sum(len(values) for values in ordered_routes.values())
    if total > capacity:
        raise ValueError(f"requested {total} unique routes but only {capacity} exist")
    scene_order = sorted(
        ordered_routes,
        key=lambda scene: (_stable_int(seed, "scene", scene), scene),
    )
    cursors = Counter()
    selected: list[dict[str, Any]] = []
    while len(selected) < total:
        progressed = False
        for scene in scene_order:
            if len(selected) >= total:
                break
            cursor = int(cursors[scene])
            values = ordered_routes[scene]
            if cursor >= len(values):
                continue
            route, variants = values[cursor]
            variants = sorted(variants, key=lambda episode: int(episode["episode_id"]))
            variant_index = _stable_int(seed, "instruction", scene, route) % len(variants)
            selected.append(variants[variant_index])
            cursors[scene] += 1
            progressed = True
        if not progressed:
            raise RuntimeError("route-balanced selection exhausted unexpectedly")
    if len({_route_key(episode) for episode in selected}) != len(selected):
        raise AssertionError("route selection produced duplicates")
    return selected


def _assign_shards(
    episodes: list[dict[str, Any]],
    *,
    num_shards: int,
    episodes_per_shard: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    if len(episodes) != num_shards * episodes_per_shard:
        raise ValueError("selected episode count does not match exact shard capacity")
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_scene[_scene_id(episode)].append(episode)
    scene_order = sorted(
        by_scene,
        key=lambda scene: (_stable_int(seed, "assign-scene", scene), scene),
    )
    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
    scene_loads = [Counter() for _ in range(num_shards)]
    for scene in scene_order:
        values = sorted(
            by_scene[scene],
            key=lambda episode: (
                _stable_int(seed, "assign-route", scene, episode["trajectory_id"]),
                int(episode["episode_id"]),
            ),
        )
        for episode in values:
            candidates = [
                index
                for index, shard in enumerate(shards)
                if len(shard) < episodes_per_shard
            ]
            if not candidates:
                raise RuntimeError("all shards filled before assignment completed")
            index = min(
                candidates,
                key=lambda shard: (
                    int(scene_loads[shard][scene]),
                    len(shards[shard]),
                    _stable_int(seed, "assign-tie", scene, episode["episode_id"], shard),
                    shard,
                ),
            )
            shards[index].append(episode)
            scene_loads[index][scene] += 1

    for index, shard in enumerate(shards):
        if len(shard) != episodes_per_shard:
            raise RuntimeError(
                f"shard {index} has {len(shard)} episodes, expected {episodes_per_shard}"
            )
        # Habitat iterates dataset order; grouping by scene minimizes GL scene reloads.
        shard.sort(
            key=lambda episode: (
                _scene_id(episode),
                _stable_int(seed, "within-scene", episode["trajectory_id"]),
                int(episode["episode_id"]),
            )
        )
    return shards


def build(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset).expanduser().resolve(strict=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _read_dataset(dataset_path)
    episodes = payload["episodes"]
    excluded = set(args.exclude_scene)
    total = int(args.num_shards) * int(args.episodes_per_shard)
    selected = _select_episodes(
        episodes,
        total=total,
        seed=int(args.seed),
        excluded_scenes=excluded,
    )
    shards = _assign_shards(
        selected,
        num_shards=int(args.num_shards),
        episodes_per_shard=int(args.episodes_per_shard),
        seed=int(args.seed),
    )

    entries: list[dict[str, Any]] = []
    for index, shard in enumerate(shards):
        cohort_path = output_dir / f"shard_{index:02d}.json"
        dataset_shard_path = output_dir / f"dataset_shard_{index:02d}.json.gz"
        cohort = {
            "schema": SCHEMA,
            "shard_id": index,
            "seed": int(args.seed),
            "count": len(shard),
            "scene_count": len({_scene_id(episode) for episode in shard}),
            "route_count": len({_route_key(episode) for episode in shard}),
            "episodes": [
                {
                    "scene_id": _scene_id(episode),
                    "episode_id": int(episode["episode_id"]),
                    "trajectory_id": str(episode["trajectory_id"]),
                    "instruction": _instruction_text(episode),
                }
                for episode in shard
            ],
        }
        shard_dataset = {key: value for key, value in payload.items() if key != "episodes"}
        shard_dataset["episodes"] = shard
        _atomic_write(cohort_path, _json_bytes(cohort))
        _atomic_write(
            dataset_shard_path,
            gzip.compress(_json_bytes(shard_dataset, compact=True), compresslevel=6, mtime=0),
        )
        entries.append(
            {
                "index": index,
                "cohort_file": cohort_path.name,
                "cohort_sha256": _sha256(cohort_path),
                "dataset_file": dataset_shard_path.name,
                "dataset_sha256": _sha256(dataset_shard_path),
                "episode_count": len(shard),
                "scene_count": cohort["scene_count"],
                "route_count": cohort["route_count"],
            }
        )

    selected_scene_counts = Counter(_scene_id(episode) for episode in selected)
    plan = {
        "schema": SCHEMA,
        "seed": int(args.seed),
        "selection": "scene_waterfill_unique_route_one_instruction_v1",
        "assignment": "scene_spread_exact_load_v1",
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256(dataset_path),
            "episode_count": len(episodes),
            "scene_count": len({_scene_id(episode) for episode in episodes}),
        },
        "excluded_scenes": sorted(excluded),
        "num_shards": int(args.num_shards),
        "episodes_per_shard": int(args.episodes_per_shard),
        "selected_episode_count": len(selected),
        "selected_route_count": len({_route_key(episode) for episode in selected}),
        "selected_scene_count": len(selected_scene_counts),
        "selected_scene_episode_counts": dict(sorted(selected_scene_counts.items())),
        "shards": entries,
    }
    _atomic_write(output_dir / "plan.json", _json_bytes(plan))
    return verify(args)


def verify(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset).expanduser().resolve(strict=True)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=True)
    plan_path = output_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != SCHEMA:
        raise RuntimeError("balanced cohort plan schema mismatch")
    if plan.get("dataset", {}).get("sha256") != _sha256(dataset_path):
        raise RuntimeError("balanced cohort source dataset SHA256 mismatch")
    expected_shards = int(args.num_shards)
    expected_per_shard = int(args.episodes_per_shard)
    if int(plan.get("num_shards", -1)) != expected_shards:
        raise RuntimeError("balanced cohort shard count mismatch")
    if int(plan.get("episodes_per_shard", -1)) != expected_per_shard:
        raise RuntimeError("balanced cohort per-shard episode count mismatch")
    entries = plan.get("shards")
    if not isinstance(entries, list) or len(entries) != expected_shards:
        raise RuntimeError("balanced cohort entries are incomplete")

    all_episode_keys: set[tuple[str, int]] = set()
    all_route_keys: set[tuple[str, str]] = set()
    all_scenes: set[str] = set()
    shard_scene_counts: list[int] = []
    for index, entry in enumerate(entries):
        cohort_path = output_dir / f"shard_{index:02d}.json"
        dataset_shard_path = output_dir / f"dataset_shard_{index:02d}.json.gz"
        if entry.get("index") != index:
            raise RuntimeError(f"balanced cohort entry index mismatch: {index}")
        if entry.get("cohort_file") != cohort_path.name:
            raise RuntimeError(f"balanced cohort filename mismatch: {index}")
        if entry.get("dataset_file") != dataset_shard_path.name:
            raise RuntimeError(f"balanced dataset filename mismatch: {index}")
        if entry.get("cohort_sha256") != _sha256(cohort_path):
            raise RuntimeError(f"balanced cohort SHA256 mismatch: {index}")
        if entry.get("dataset_sha256") != _sha256(dataset_shard_path):
            raise RuntimeError(f"balanced dataset shard SHA256 mismatch: {index}")
        cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
        shard_payload = _read_dataset(dataset_shard_path)
        cohort_episodes = cohort.get("episodes")
        shard_episodes = shard_payload.get("episodes")
        if not isinstance(cohort_episodes, list) or len(cohort_episodes) != expected_per_shard:
            raise RuntimeError(f"balanced cohort size mismatch: {index}")
        if not isinstance(shard_episodes, list) or len(shard_episodes) != expected_per_shard:
            raise RuntimeError(f"balanced dataset shard size mismatch: {index}")
        cohort_keys = {
            (str(episode["scene_id"]), int(episode["episode_id"]))
            for episode in cohort_episodes
        }
        dataset_keys = {_episode_key(episode) for episode in shard_episodes}
        if cohort_keys != dataset_keys or len(cohort_keys) != expected_per_shard:
            raise RuntimeError(f"balanced cohort/dataset closure mismatch: {index}")
        if all_episode_keys.intersection(cohort_keys):
            raise RuntimeError(f"episode duplicated across balanced shards: {index}")
        all_episode_keys.update(cohort_keys)
        shard_routes = {_route_key(episode) for episode in shard_episodes}
        if len(shard_routes) != expected_per_shard:
            raise RuntimeError(f"route duplicated within balanced shard: {index}")
        if all_route_keys.intersection(shard_routes):
            raise RuntimeError(f"route duplicated across balanced shards: {index}")
        all_route_keys.update(shard_routes)
        scenes = {_scene_id(episode) for episode in shard_episodes}
        all_scenes.update(scenes)
        shard_scene_counts.append(len(scenes))

    expected_total = expected_shards * expected_per_shard
    if len(all_episode_keys) != expected_total or len(all_route_keys) != expected_total:
        raise RuntimeError("balanced cohort global episode/route count mismatch")
    excluded = set(plan.get("excluded_scenes", []))
    if all_scenes.intersection(excluded):
        raise RuntimeError("balanced cohort contains an excluded scene")
    if int(plan.get("selected_scene_count", -1)) != len(all_scenes):
        raise RuntimeError("balanced cohort selected scene count mismatch")
    return {
        "status": "verified",
        "schema": SCHEMA,
        "episodes": expected_total,
        "routes": len(all_route_keys),
        "scenes": len(all_scenes),
        "shard_scene_counts": shard_scene_counts,
        "output_dir": str(output_dir),
        "plan_sha256": _sha256(plan_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--episodes-per-shard", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--exclude-scene", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.num_shards <= 64:
        parser.error("--num-shards must be in [1,64]")
    if args.episodes_per_shard < 1:
        parser.error("--episodes-per-shard must be positive")
    return args


def main() -> int:
    args = parse_args()
    result = verify(args) if args.verify_only else build(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

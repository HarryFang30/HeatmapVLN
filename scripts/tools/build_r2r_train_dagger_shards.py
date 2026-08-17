#!/usr/bin/env python3
"""Build deterministic, route-disjoint R2R-train DAgger cohort shards.

The complete episode cohort is selected once, then every canonical
(scene_id, trajectory_id) route, including all instruction paraphrases,
is assigned to exactly one shard. Partial episode cohorts are rejected
because truncating them can split a route group. Assignment is
scene-stratified and load-balanced by episode count. Outputs are
deterministic and are created as one atomic directory below FJL_ROOT.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from . import build_r2r_train_dagger_cohort as cohort
else:
    import build_r2r_train_dagger_cohort as cohort


FJL_ROOT = Path("/mnt/afs/lixiaoou/intern/fjl")
DEFAULT_DATASET = (
    FJL_ROOT
    / "habitat/VLN-CE/data/datasets/"
    "R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
)
PLAN_SCHEMA = "r2r-dagger-shard-plan-v1"
SHARD_SCHEMA = "r2r-dagger-shard-selection-v1"
PARTITION_STRATEGY = "scene_stratified_route_grouped_least_load_v1"


class ShardError(RuntimeError):
    """Raised when shard construction or validation must fail closed."""


def _resolved_root() -> Path:
    try:
        root = FJL_ROOT.resolve(strict=True)
    except OSError as exc:
        raise ShardError(f"FJL root is unavailable: {FJL_ROOT}: {exc}") from exc
    if not root.is_dir():
        raise ShardError(f"FJL root is not a directory: {root}")
    return root


def _resolve_input(raw_path: str, root: Path) -> Path:
    try:
        return cohort._resolve_input(raw_path, root)
    except cohort.CohortError as exc:
        raise ShardError(str(exc)) from exc


def _resolve_output_dir(raw_path: str, root: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.name in {"", ".", ".."}:
        raise ShardError(f"invalid output directory: {candidate}")
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise ShardError(
            f"output parent must already exist: {candidate.parent}: {exc}"
        ) from exc
    resolved = parent / candidate.name
    if not cohort._is_within(resolved, root):
        raise ShardError(f"output escapes FJL root: {resolved}")
    if resolved.is_symlink():
        raise ShardError(f"output directory may not be a symlink: {resolved}")
    if resolved.exists() and not resolved.is_dir():
        raise ShardError(f"output exists but is not a directory: {resolved}")
    return resolved


def _stable_digest(seed: int, namespace: str, *parts: object) -> bytes:
    return cohort._stable_digest(seed, namespace, *parts)


def _episode_key(episode: dict[str, Any]) -> tuple[str, int]:
    return cohort._episode_key(episode)


def _route_key(episode: dict[str, Any]) -> tuple[str, int]:
    return cohort._route_key(episode)


def _key_digest(keys: set[tuple[str, int]]) -> str:
    payload = json.dumps(
        [[scene, value] for scene, value in sorted(keys)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _route_grouped_shards(
    episodes: list[dict[str, Any]],
    num_shards: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    if num_shards <= 0:
        raise ShardError("num_shards must be positive")
    if not episodes:
        raise ShardError("cannot shard an empty episode cohort")

    route_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        route_groups[_route_key(episode)].append(episode)
    if num_shards > len(route_groups):
        raise ShardError(
            f"num_shards {num_shards} exceeds route count {len(route_groups)}"
        )

    routes_by_scene: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for route in route_groups:
        routes_by_scene[route[0]].append(route)

    assigned_routes: list[list[tuple[str, int]]] = [
        [] for _ in range(num_shards)
    ]
    episode_loads = [0 for _ in range(num_shards)]
    route_loads = [0 for _ in range(num_shards)]
    per_scene_route_loads: dict[str, list[int]] = {
        scene: [0 for _ in range(num_shards)] for scene in routes_by_scene
    }

    scene_order = sorted(
        routes_by_scene,
        key=lambda scene: (
            -sum(len(route_groups[route]) for route in routes_by_scene[scene]),
            _stable_digest(seed, "shard-scene", scene),
            scene,
        ),
    )
    for scene in scene_order:
        routes = sorted(
            routes_by_scene[scene],
            key=lambda route: (
                _stable_digest(seed, "shard-route", route[0], route[1]),
                route[1],
            ),
        )
        for route in routes:
            shard_index = min(
                range(num_shards),
                key=lambda index: (
                    per_scene_route_loads[scene][index],
                    episode_loads[index],
                    route_loads[index],
                    _stable_digest(
                        seed,
                        "shard-tie",
                        route[0],
                        route[1],
                        index,
                    ),
                    index,
                ),
            )
            assigned_routes[shard_index].append(route)
            route_size = len(route_groups[route])
            episode_loads[shard_index] += route_size
            route_loads[shard_index] += 1
            per_scene_route_loads[scene][shard_index] += 1

    shards: list[list[dict[str, Any]]] = []
    for shard_index, routes in enumerate(assigned_routes):
        ordered_routes = sorted(
            routes,
            key=lambda route: (
                _stable_digest(seed, "output-scene", route[0]),
                route[0],
                _stable_digest(seed, "output-route", route[0], route[1]),
                route[1],
            ),
        )
        shard: list[dict[str, Any]] = []
        for route in ordered_routes:
            route_episodes = sorted(
                route_groups[route],
                key=lambda episode: (
                    _stable_digest(
                        seed,
                        "output-episode",
                        route[0],
                        route[1],
                        episode["episode_id"],
                    ),
                    episode["episode_id"],
                ),
            )
            shard.extend(route_episodes)
        if not shard:
            raise ShardError(f"internal error: shard {shard_index} is empty")
        shards.append(shard)

    expected_episode_keys = {_episode_key(episode) for episode in episodes}
    actual_episode_keys: set[tuple[str, int]] = set()
    actual_route_owners: dict[tuple[str, int], int] = {}
    for shard_index, shard in enumerate(shards):
        for episode in shard:
            episode_key = _episode_key(episode)
            if episode_key in actual_episode_keys:
                raise ShardError(
                    f"internal error: duplicate episode key {episode_key}"
                )
            actual_episode_keys.add(episode_key)
            route = _route_key(episode)
            owner = actual_route_owners.setdefault(route, shard_index)
            if owner != shard_index:
                raise ShardError(
                    f"internal error: route {route} crosses shards {owner}/{shard_index}"
                )
    if actual_episode_keys != expected_episode_keys:
        raise ShardError("internal error: shard episode union is incomplete")

    for scene, loads in per_scene_route_loads.items():
        if max(loads) - min(loads) > 1:
            raise ShardError(
                f"internal error: scene route imbalance for {scene}: {loads}"
            )
    return shards


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _expected_files(
    dataset_path: Path,
    selected: list[dict[str, Any]],
    shards: list[list[dict[str, Any]]],
    seed: int,
) -> dict[str, bytes]:
    dataset_sha256 = cohort._sha256_file(dataset_path)
    all_episodes = cohort._load_episodes(dataset_path)
    all_episode_keys = {_episode_key(episode) for episode in selected}
    all_route_keys = {_route_key(episode) for episode in selected}
    dataset_info = {
        "path": str(dataset_path),
        "sha256": dataset_sha256,
        "episode_count": len(all_episodes),
        "route_count": len({_route_key(episode) for episode in all_episodes}),
        "scene_count": len({episode["scene_id"] for episode in all_episodes}),
    }
    files: dict[str, bytes] = {}
    manifest_shards: list[dict[str, Any]] = []
    for shard_index, shard in enumerate(shards):
        shard_episode_keys = {_episode_key(episode) for episode in shard}
        shard_route_keys = {_route_key(episode) for episode in shard}
        scene_ids = sorted({episode["scene_id"] for episode in shard})
        filename = f"shard_{shard_index:02d}.json"
        shard_payload = {
            "split": "train",
            "count": len(shard),
            "episodes": [cohort._evaluator_episode(item) for item in shard],
            "selection": {
                "schema": SHARD_SCHEMA,
                "strategy": PARTITION_STRATEGY,
                "seed": seed,
                "shard_index": shard_index,
                "num_shards": len(shards),
                "route_grouped": True,
                "dataset": dataset_info,
                "global_selected_episode_count": len(selected),
                "global_selected_route_count": len(all_route_keys),
                "global_episode_key_sha256": _key_digest(all_episode_keys),
                "global_route_key_sha256": _key_digest(all_route_keys),
                "shard_route_count": len(shard_route_keys),
                "shard_episode_key_sha256": _key_digest(shard_episode_keys),
                "shard_route_key_sha256": _key_digest(shard_route_keys),
                "scenes": scene_ids,
            },
        }
        shard_bytes = _json_bytes(shard_payload)
        files[filename] = shard_bytes
        manifest_shards.append(
            {
                "index": shard_index,
                "file": filename,
                "sha256": hashlib.sha256(shard_bytes).hexdigest(),
                "episode_count": len(shard),
                "route_count": len(shard_route_keys),
                "scene_count": len(scene_ids),
                "episode_key_sha256": _key_digest(shard_episode_keys),
                "route_key_sha256": _key_digest(shard_route_keys),
            }
        )

    plan = {
        "schema": PLAN_SCHEMA,
        "split": "train",
        "seed": seed,
        "num_shards": len(shards),
        "partition_strategy": PARTITION_STRATEGY,
        "route_grouped": True,
        "dataset": dataset_info,
        "selected_episode_count": len(selected),
        "selected_route_count": len(all_route_keys),
        "episode_key_sha256": _key_digest(all_episode_keys),
        "route_key_sha256": _key_digest(all_route_keys),
        "shards": manifest_shards,
    }
    files["plan.json"] = _json_bytes(plan)
    return files


def _validate_existing(output_dir: Path, expected: dict[str, bytes]) -> None:
    actual_names = {path.name for path in output_dir.iterdir()}
    if actual_names != set(expected):
        raise ShardError(
            "existing shard directory has unexpected contents: "
            f"expected={sorted(expected)}, actual={sorted(actual_names)}"
        )
    for name, expected_bytes in expected.items():
        path = output_dir / name
        if path.is_symlink() or not path.is_file():
            raise ShardError(f"existing shard artifact is not a regular file: {path}")
        if path.read_bytes() != expected_bytes:
            raise ShardError(f"existing shard artifact disagrees with plan: {path}")


def _atomic_create_directory(
    output_dir: Path,
    expected: dict[str, bytes],
) -> bool:
    if output_dir.exists():
        _validate_existing(output_dir, expected)
        return False

    lock_path = output_dir.parent / f".{output_dir.name}.create.lock"
    try:
        lock_fd = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise ShardError(f"shard creation lock already exists: {lock_path}") from exc

    temporary_dir: Path | None = None
    try:
        os.close(lock_fd)
        temporary_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.",
                suffix=".tmp",
                dir=output_dir.parent,
            )
        )
        for name, payload in expected.items():
            path = temporary_dir / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        directory_fd = os.open(temporary_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output_dir.exists() or output_dir.is_symlink():
            raise ShardError(f"refusing to replace output directory: {output_dir}")
        os.rename(temporary_dir, output_dir)
        temporary_dir = None
        parent_fd = os.open(output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except OSError as exc:
        raise ShardError(f"failed to create shard directory {output_dir}: {exc}") from exc
    finally:
        if temporary_dir is not None:
            shutil.rmtree(temporary_dir, ignore_errors=True)
        with contextlib.suppress(OSError):
            lock_path.unlink()
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument(
        "--count",
        type=int,
        default=10819,
        help=(
            "complete dataset episode count; partial counts are rejected "
            "(10819 for the default R2R train dataset)"
        ),
    )
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        root = _resolved_root()
        dataset_path = _resolve_input(args.dataset, root)
        output_dir = _resolve_output_dir(args.output_dir, root)
        episodes = cohort._load_episodes(dataset_path)
        if args.count != len(episodes):
            raise ShardError(
                "--count must equal the complete dataset episode count; "
                "partial cohorts can split route groups: "
                f"requested={args.count}, dataset={len(episodes)}"
            )
        selected = cohort._build_round_robin(
            episodes,
            count=args.count,
            seed=args.seed,
        )
        shards = _route_grouped_shards(
            selected,
            num_shards=args.num_shards,
            seed=args.seed,
        )
        expected = _expected_files(
            dataset_path,
            selected,
            shards,
            seed=args.seed,
        )
        created = _atomic_create_directory(output_dir, expected)
    except (ShardError, cohort.CohortError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    plan = json.loads(expected["plan.json"])
    print(
        json.dumps(
            {
                "status": "created" if created else "verified_existing",
                "output_dir": str(output_dir),
                "plan_sha256": hashlib.sha256(expected["plan.json"]).hexdigest(),
                "episodes": plan["selected_episode_count"],
                "routes": plan["selected_route_count"],
                "num_shards": plan["num_shards"],
                "episode_loads": [
                    item["episode_count"] for item in plan["shards"]
                ],
                "route_loads": [
                    item["route_count"] for item in plan["shards"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

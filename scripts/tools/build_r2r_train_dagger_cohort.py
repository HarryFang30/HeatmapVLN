#!/usr/bin/env python3
"""Build a deterministic, scene-balanced cohort from R2R train episodes.

The output is directly accepted by scripts/evaluation/r2r_val_unseen.py via
--episode_list. Only annotation metadata is read; no image data is copied.
All filesystem paths are required to remain under FJL_ROOT. Optional route
deduplication treats ``(scene_id, trajectory_id)`` as the canonical route and
can exclude every instruction associated with routes present in an earlier
cohort.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

FJL_ROOT = Path("/mnt/afs/lixiaoou/intern/fjl")
DEFAULT_DATASET = FJL_ROOT / "habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"


class CohortError(RuntimeError):
    """Raised when an input would make cohort construction unsafe or invalid."""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolved_root() -> Path:
    try:
        root = FJL_ROOT.resolve(strict=True)
    except OSError as exc:
        raise CohortError(f"FJL root is unavailable: {FJL_ROOT}: {exc}") from exc
    if not root.is_dir():
        raise CohortError(f"FJL root is not a directory: {root}")
    return root


def _resolve_input(raw_path: str, root: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise CohortError(f"input does not resolve to an existing file: {candidate}: {exc}") from exc
    if not _is_within(resolved, root):
        raise CohortError(f"input escapes FJL root: {resolved}")
    if not resolved.is_file():
        raise CohortError(f"input is not a regular file: {resolved}")
    return resolved


def _resolve_new_output(raw_path: str, root: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.name in {"", ".", ".."}:
        raise CohortError(f"invalid output filename: {candidate}")
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise CohortError(f"output parent must already exist: {candidate.parent}: {exc}") from exc
    resolved = parent / candidate.name
    if not _is_within(resolved, root):
        raise CohortError(f"output escapes FJL root: {resolved}")
    if resolved.suffix.lower() != ".json":
        raise CohortError(f"output must end in .json: {resolved}")
    if resolved.exists() or resolved.is_symlink():
        raise CohortError(f"refusing to overwrite existing output: {resolved}")
    return resolved


def _scene_name(raw_scene_id: Any) -> str:
    if not isinstance(raw_scene_id, str) or not raw_scene_id.strip():
        raise CohortError(f"invalid scene_id: {raw_scene_id!r}")
    parts = [part for part in raw_scene_id.replace("\\", "/").split("/") if part]
    if not parts:
        raise CohortError(f"invalid scene_id: {raw_scene_id!r}")
    if parts[-1].lower().endswith(".glb") and len(parts) >= 2:
        scene = parts[-2]
    else:
        scene = Path(parts[-1]).stem
    if not scene or scene in {".", ".."} or "/" in scene or "\\" in scene:
        raise CohortError(f"could not normalize scene_id: {raw_scene_id!r}")
    return scene


def _episode_id(raw_episode_id: Any) -> int:
    if isinstance(raw_episode_id, bool):
        raise CohortError(f"invalid episode_id: {raw_episode_id!r}")
    try:
        episode_id = int(raw_episode_id)
    except (TypeError, ValueError) as exc:
        raise CohortError(f"invalid episode_id: {raw_episode_id!r}") from exc
    if episode_id < 0 or str(episode_id) != str(raw_episode_id).strip():
        raise CohortError(f"episode_id must be a canonical non-negative integer: {raw_episode_id!r}")
    return episode_id


def _instruction(raw_episode: dict[str, Any]) -> str:
    raw_instruction = raw_episode.get("instruction")
    if isinstance(raw_instruction, dict):
        raw_instruction = raw_instruction.get("instruction_text")
    if not isinstance(raw_instruction, str) or not raw_instruction.strip():
        raise CohortError("episode has no non-empty instruction text")
    return raw_instruction.strip()


def _trajectory_id(raw_trajectory_id: Any) -> int:
    if isinstance(raw_trajectory_id, bool):
        raise CohortError(f"invalid trajectory_id: {raw_trajectory_id!r}")
    try:
        trajectory_id = int(raw_trajectory_id)
    except (TypeError, ValueError) as exc:
        raise CohortError(f"invalid trajectory_id: {raw_trajectory_id!r}") from exc
    if trajectory_id < 0 or str(trajectory_id) != str(raw_trajectory_id).strip():
        raise CohortError(f"trajectory_id must be a canonical non-negative integer: {raw_trajectory_id!r}")
    return trajectory_id


def _load_episodes(dataset_path: Path) -> list[dict[str, Any]]:
    if dataset_path.name != "train.json.gz":
        raise CohortError(f"refusing a non-train annotation file; expected train.json.gz, got {dataset_path.name}")
    try:
        with gzip.open(dataset_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CohortError(f"cannot read R2R train annotations: {dataset_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), list):
        raise CohortError("R2R train annotations must contain an 'episodes' array")
    if not payload["episodes"]:
        raise CohortError("R2R train annotations contain no episodes")

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for index, raw_episode in enumerate(payload["episodes"]):
        if not isinstance(raw_episode, dict):
            raise CohortError(f"episode[{index}] is not an object")
        try:
            scene_id = _scene_name(raw_episode.get("scene_id"))
            episode_id = _episode_id(raw_episode.get("episode_id"))
            trajectory_id = _trajectory_id(raw_episode.get("trajectory_id"))
            instruction = _instruction(raw_episode)
        except CohortError as exc:
            raise CohortError(f"invalid episode[{index}]: {exc}") from exc
        key = (scene_id, episode_id)
        if key in seen:
            raise CohortError(f"duplicate (scene_id, episode_id): {key}")
        seen.add(key)
        normalized.append(
            {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "trajectory_id": trajectory_id,
                "instruction": instruction,
            }
        )
    return normalized


def _stable_digest(seed: int, namespace: str, *parts: object) -> bytes:
    encoded = "\x00".join([str(seed), namespace, *(str(part) for part in parts)]).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def _episode_key(episode: dict[str, Any]) -> tuple[str, int]:
    return str(episode["scene_id"]), int(episode["episode_id"])


def _route_key(episode: dict[str, Any]) -> tuple[str, int]:
    return str(episode["scene_id"]), int(episode["trajectory_id"])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_excluded_routes(
    cohort_path: Path,
    episodes: list[dict[str, Any]],
) -> tuple[set[tuple[str, int]], dict[str, Any]]:
    """Resolve an evaluator cohort to canonical train routes, failing closed."""
    if cohort_path.suffix.lower() != ".json":
        raise CohortError(f"--exclude-cohort must end in .json: {cohort_path}")
    try:
        payload = json.loads(cohort_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CohortError(f"cannot read exclude cohort {cohort_path}: {exc}") from exc

    if isinstance(payload, dict):
        if payload.get("split", "train") != "train":
            raise CohortError(f"exclude cohort must declare split='train': {cohort_path}")
        raw_entries = payload.get("episodes")
        declared_count = payload.get("count")
    else:
        raw_entries = payload
        declared_count = None
    if not isinstance(raw_entries, list) or not raw_entries:
        raise CohortError(f"exclude cohort must contain a non-empty 'episodes' array: {cohort_path}")
    if declared_count is not None:
        try:
            normalized_count = _episode_id(declared_count)
        except CohortError as exc:
            raise CohortError(f"invalid exclude cohort count: {exc}") from exc
        if normalized_count != len(raw_entries):
            raise CohortError(
                "exclude cohort count does not match its episodes array: "
                f"declared {normalized_count}, found {len(raw_entries)}"
            )

    episode_lookup = {_episode_key(episode): episode for episode in episodes}
    seen_episode_keys: set[tuple[str, int]] = set()
    excluded_routes: set[tuple[str, int]] = set()
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise CohortError(f"exclude cohort episode[{index}] is not an object")
        try:
            key = (
                _scene_name(raw_entry.get("scene_id")),
                _episode_id(raw_entry.get("episode_id")),
            )
        except CohortError as exc:
            raise CohortError(f"invalid exclude cohort episode[{index}]: {exc}") from exc
        if key in seen_episode_keys:
            raise CohortError(f"duplicate exclude cohort episode key: {key}")
        seen_episode_keys.add(key)
        canonical = episode_lookup.get(key)
        if canonical is None:
            raise CohortError(f"exclude cohort episode does not map to canonical R2R train data: {key}")

        if "trajectory_id" in raw_entry:
            try:
                declared_trajectory = _trajectory_id(raw_entry["trajectory_id"])
            except CohortError as exc:
                raise CohortError(f"invalid exclude cohort episode[{index}]: {exc}") from exc
            if declared_trajectory != canonical["trajectory_id"]:
                raise CohortError(
                    "exclude cohort trajectory_id disagrees with canonical R2R train data: "
                    f"{key}: declared {declared_trajectory}, canonical "
                    f"{canonical['trajectory_id']}"
                )
        if "instruction" in raw_entry:
            try:
                declared_instruction = _instruction(raw_entry)
            except CohortError as exc:
                raise CohortError(f"invalid exclude cohort episode[{index}]: {exc}") from exc
            if declared_instruction != canonical["instruction"]:
                raise CohortError(f"exclude cohort instruction disagrees with canonical R2R train data: {key}")
        excluded_routes.add(_route_key(canonical))

    provenance = {
        "path": str(cohort_path),
        "sha256": _sha256_file(cohort_path),
        "episode_count": len(seen_episode_keys),
        "route_count": len(excluded_routes),
    }
    return excluded_routes, provenance


def _choose_unique_trajectory_representatives(episodes: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Choose one deterministic instruction episode for every canonical route."""
    by_route: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_route[_route_key(episode)].append(episode)

    representatives: list[dict[str, Any]] = []
    for (scene_id, trajectory_id), route_episodes in by_route.items():
        representatives.append(
            min(
                route_episodes,
                key=lambda episode: (
                    _stable_digest(
                        seed,
                        "trajectory-instruction",
                        scene_id,
                        trajectory_id,
                        episode["episode_id"],
                    ),
                    episode["episode_id"],
                ),
            )
        )
    return representatives


def _evaluator_episode(episode: dict[str, Any]) -> dict[str, Any]:
    """Strip builder-only route identity while retaining evaluator compatibility."""
    return {
        "scene_id": episode["scene_id"],
        "episode_id": episode["episode_id"],
        "instruction": episode["instruction"],
    }


def _build_round_robin(episodes: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise CohortError(f"--count must be positive, got {count}")
    if count > len(episodes):
        raise CohortError(f"--count {count} exceeds available episodes {len(episodes)}")

    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_scene[episode["scene_id"]].append(episode)

    scene_order = sorted(
        by_scene,
        key=lambda scene: (_stable_digest(seed, "scene", scene), scene),
    )
    for scene, scene_episodes in by_scene.items():
        scene_episodes.sort(
            key=lambda episode: (
                _stable_digest(seed, "episode", scene, episode["episode_id"]),
                episode["episode_id"],
            )
        )

    offsets = {scene: 0 for scene in scene_order}
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        made_progress = False
        for scene in scene_order:
            offset = offsets[scene]
            if offset >= len(by_scene[scene]):
                continue
            selected.append(by_scene[scene][offset])
            offsets[scene] = offset + 1
            made_progress = True
            if len(selected) == count:
                break
        if not made_progress:
            raise CohortError("internal error: round-robin selection exhausted early")
    return selected


def _atomic_create_json(output_path: Path, payload: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-linking provides atomic no-clobber semantics; os.replace would
        # silently overwrite a file created after our initial path check.
        os.link(temporary_path, output_path)
    except FileExistsError as exc:
        raise CohortError(f"refusing to overwrite existing output: {output_path}") from exc
    except OSError as exc:
        raise CohortError(f"failed to create output {output_path}: {exc}") from exc
    finally:
        if temporary_path is not None:
            with contextlib.suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="R2R train.json.gz annotation file (must remain under FJL_ROOT)",
    )
    parser.add_argument("--count", type=int, required=True, help="number of episodes to select")
    parser.add_argument("--seed", type=int, default=0, help="deterministic ordering seed")
    parser.add_argument(
        "--exclude-cohort",
        help=(
            "evaluator-compatible train cohort whose canonical routes are excluded in full (must remain under FJL_ROOT)"
        ),
    )
    parser.add_argument(
        "--unique-trajectories",
        action="store_true",
        help=("select at most one instruction episode per canonical (scene_id, trajectory_id) route"),
    )
    parser.add_argument("--output", required=True, help="new evaluator-compatible JSON file under FJL_ROOT")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        root = _resolved_root()
        dataset_path = _resolve_input(args.dataset, root)
        output_path = _resolve_new_output(args.output, root)
        all_episodes = _load_episodes(dataset_path)
        excluded_routes: set[tuple[str, int]] = set()
        exclude_provenance: dict[str, Any] | None = None
        if args.exclude_cohort:
            exclude_path = _resolve_input(args.exclude_cohort, root)
            excluded_routes, exclude_provenance = _load_excluded_routes(exclude_path, all_episodes)

        eligible_episodes = [episode for episode in all_episodes if _route_key(episode) not in excluded_routes]
        available_route_count = len({_route_key(episode) for episode in eligible_episodes})
        candidate_episodes = eligible_episodes
        if args.unique_trajectories:
            candidate_episodes = _choose_unique_trajectory_representatives(eligible_episodes, args.seed)
        selected_internal = _build_round_robin(candidate_episodes, args.count, args.seed)
        selected = [_evaluator_episode(episode) for episode in selected_internal]
        selected_route_count = len({_route_key(episode) for episode in selected_internal})
        payload = {
            "split": "train",
            "count": len(selected),
            "episodes": selected,
            "selection": {
                "schema": "r2r-dagger-cohort-selection-v1",
                "strategy": "scene_balanced_round_robin",
                "seed": int(args.seed),
                "canonical_route": ["scene_id", "trajectory_id"],
                "unique_trajectories": bool(args.unique_trajectories),
                "dataset": {
                    "path": str(dataset_path),
                    "sha256": _sha256_file(dataset_path),
                    "episode_count": len(all_episodes),
                    "route_count": len({_route_key(episode) for episode in all_episodes}),
                },
                "exclude_cohort": exclude_provenance,
                "excluded_route_count": len(excluded_routes),
                "eligible_episode_count": len(eligible_episodes),
                "eligible_route_count": available_route_count,
                "candidate_count": len(candidate_episodes),
                "selected_route_count": selected_route_count,
            },
        }
        _atomic_create_json(output_path, payload)
    except CohortError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    scene_counts: dict[str, int] = defaultdict(int)
    for episode in selected:
        scene_counts[episode["scene_id"]] += 1
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "output": str(output_path),
                "count": len(selected),
                "unique_trajectories": bool(args.unique_trajectories),
                "selected_routes": selected_route_count,
                "excluded_routes": len(excluded_routes),
                "scenes": len(scene_counts),
                "min_per_selected_scene": min(scene_counts.values()),
                "max_per_selected_scene": max(scene_counts.values()),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

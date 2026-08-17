"""Utilities for evaluating an exact Habitat episode cohort efficiently."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Sequence, Tuple


EpisodeKey = Tuple[str, int]


def load_episode_cohort(
    path: str | Path,
) -> tuple[list[EpisodeKey], dict[EpisodeKey, dict[str, Any]]]:
    """Load ordered episode keys and retain per-episode diagnostic metadata."""
    cohort_path = Path(path)
    data = json.loads(cohort_path.read_text(encoding="utf-8"))
    episodes = data.get("episodes", data) if isinstance(data, dict) else data
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(
            "episode cohort must contain a non-empty 'episodes' array: "
            f"{cohort_path}"
        )

    keys: list[EpisodeKey] = []
    metadata: dict[EpisodeKey, dict[str, Any]] = {}
    for index, item in enumerate(episodes):
        if not isinstance(item, dict):
            raise ValueError(
                f"episode cohort entry {index} must be an object: {cohort_path}"
            )
        try:
            key = (str(item["scene_id"]), int(item["episode_id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"episode cohort entry {index} has an invalid scene/episode key"
            ) from exc
        if key in metadata:
            raise ValueError(f"episode cohort contains duplicate key: {key}")
        keys.append(key)
        metadata[key] = dict(item)
    return keys, metadata


def habitat_episode_key(episode: Any) -> EpisodeKey:
    """Return the repository's canonical ``(scene_id, episode_id)`` key."""
    scene_path = Path(str(episode.scene_id))
    scene_id = scene_path.parent.name if scene_path.suffix else scene_path.name
    return scene_id, int(episode.episode_id)


def restrict_habitat_env_to_episode_keys(
    env: Any,
    requested_keys: Sequence[EpisodeKey],
) -> list[Any]:
    """Replace Habitat's iterator with the requested episodes in exact order.

    Filtering only after ``env.reset()`` still loads every skipped scene. This
    function resolves the cohort in memory before the first reset, which keeps
    fixed-cohort and DAgger diagnostics proportional to the requested size.
    """
    requested = [(str(scene_id), int(episode_id)) for scene_id, episode_id in requested_keys]
    if not requested:
        raise ValueError("Episode cohort must not be empty")
    if len(requested) != len(set(requested)):
        raise ValueError("Episode cohort contains duplicate scene/episode keys")

    by_key: dict[EpisodeKey, Any] = {}
    duplicate_dataset_keys: list[EpisodeKey] = []
    for episode in env.episodes:
        key = habitat_episode_key(episode)
        if key in by_key:
            duplicate_dataset_keys.append(key)
        else:
            by_key[key] = episode
    if duplicate_dataset_keys:
        raise RuntimeError(
            "Habitat dataset contains duplicate episode keys: "
            f"{duplicate_dataset_keys[:10]}"
        )

    missing = [key for key in requested if key not in by_key]
    if missing:
        raise ValueError(f"Episode cohort is missing from Habitat dataset: {missing[:10]}")
    selected = [by_key[key] for key in requested]

    if not hasattr(env, "_episodes") or not hasattr(env, "_episode_iterator"):
        raise RuntimeError("Unsupported Habitat Env: episode iterator internals are unavailable")
    env._episodes = selected
    dataset = getattr(env, "_dataset", None)
    if dataset is not None:
        dataset.episodes = selected
    env._episode_iterator = iter(selected)
    if hasattr(env, "number_of_episodes"):
        env.number_of_episodes = len(selected)
    return selected

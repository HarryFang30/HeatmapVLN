#!/usr/bin/env python3
"""Build a deterministic scene-balanced R2R episode cohort."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def _score(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-scene", type=int, default=5)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help=(
            "Optional total episode cap applied after round-robin scene "
            "interleaving. Zero keeps every selected episode."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.episodes_per_scene < 1
        or args.max_scenes < 0
        or args.max_episodes < 0
    ):
        raise ValueError(
            "episodes-per-scene must be >= 1 and max-scenes/max-episodes "
            "must be >= 0"
        )
    with gzip.open(args.data_path, "rt", encoding="utf-8") as handle:
        episodes = json.load(handle).get("episodes", [])
    grouped: dict[str, list[dict]] = defaultdict(list)
    for episode in episodes:
        scene_path = str(episode["scene_id"])
        parts = Path(scene_path).parts
        scene_id = parts[-2] if len(parts) >= 2 else parts[-1]
        grouped[scene_id].append(episode)
    scenes = sorted(grouped, key=lambda scene: _score(args.seed, scene))
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]

    selected_by_scene: dict[str, list[dict]] = {}
    for scene in scenes:
        selected_by_scene[scene] = sorted(
            grouped[scene],
            key=lambda episode: _score(
                args.seed,
                f"{scene}:{int(episode['episode_id'])}",
            ),
        )[: args.episodes_per_scene]
    selected = []
    for offset in range(args.episodes_per_scene):
        for scene in scenes:
            scene_episodes = selected_by_scene[scene]
            if offset < len(scene_episodes):
                selected.append(
                    {
                        "scene_id": scene,
                        "episode_id": int(scene_episodes[offset]["episode_id"]),
                    }
                )
    if args.max_episodes:
        selected = selected[: args.max_episodes]
    payload = {
        "description": "Deterministic scene-balanced R2R rollout cohort.",
        "source_data_path": str(args.data_path),
        "seed": int(args.seed),
        "episodes_per_scene": int(args.episodes_per_scene),
        "max_episodes": int(args.max_episodes),
        "scene_count": len(scenes),
        "episodes": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(selected)} episodes across {len(scenes)} scenes to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

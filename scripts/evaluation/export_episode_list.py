#!/usr/bin/env python3
"""Export the first N Habitat val_unseen episodes in iterator order (SHUFFLE=False)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import habitat
from scripts.evaluation.r2r_val_unseen import (
    build_habitat_config,
    ensure_vln_measures_registered,
)


def export_episodes(scenes_dir: str, data_path: str, count: int) -> list[dict]:
    from types import SimpleNamespace

    ensure_vln_measures_registered()
    args = SimpleNamespace(scenes_dir=scenes_dir, data_path=data_path, sim_gpu_id=0)
    env = habitat.Env(config=build_habitat_config(args))
    episodes: list[dict] = []
    seen: set[tuple[str, int]] = set()

    while len(episodes) < count:
        env.reset()
        ep = env.current_episode
        scene_id = ep.scene_id.split("/")[-2]
        episode_id = int(ep.episode_id)
        key = (scene_id, episode_id)
        if key in seen:
            break
        seen.add(key)
        episodes.append(
            {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "instruction": ep.instruction.instruction_text[:120],
            }
        )

    if len(episodes) < count:
        raise RuntimeError(f"Only collected {len(episodes)} episodes, requested {count}")
    return episodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes_dir", type=str, default="/dataset/mp3d")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/workspace/InternNav/data/vln_ce/raw_data/r2r/{split}/{split}.json.gz",
    )
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    episodes = export_episodes(args.scenes_dir, args.data_path, args.count)
    out = {
        "split": "val_unseen",
        "scenes_dir": args.scenes_dir,
        "data_path": args.data_path,
        "count": len(episodes),
        "episodes": episodes,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {len(episodes)} episodes to {out_path}")
    for i, ep in enumerate(episodes, 1):
        print(f"  {i:2d}. {ep['scene_id']}_{ep['episode_id']:04d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest
from scripts.tools import build_r2r_train_dagger_cohort as cohort
from scripts.tools import build_r2r_train_dagger_shards as shards


def _raw_episodes() -> list[dict]:
    episodes: list[dict] = []
    episode_id = 0
    for scene_index in range(4):
        scene = f"scene_{scene_index}"
        for route_offset in range(6):
            trajectory_id = scene_index * 100 + route_offset
            paraphrases = 4 if (scene_index, route_offset) == (0, 0) else 3
            for paraphrase in range(paraphrases):
                episodes.append(
                    {
                        "scene_id": f"mp3d/{scene}/{scene}.glb",
                        "episode_id": episode_id,
                        "trajectory_id": trajectory_id,
                        "instruction": {
                            "instruction_text": (
                                f"Route {trajectory_id}, wording {paraphrase}."
                            )
                        },
                    }
                )
                episode_id += 1
    return episodes


def _write_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "train.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"episodes": _raw_episodes()}, handle)
    return path


def _owners(parts: list[list[dict]]) -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}
    for shard_index, part in enumerate(parts):
        for episode in part:
            route = cohort._route_key(episode)
            previous = result.setdefault(route, shard_index)
            assert previous == shard_index
    return result


def test_route_grouped_shards_are_deterministic_complete_and_disjoint(
    tmp_path: Path,
) -> None:
    episodes = cohort._load_episodes(_write_dataset(tmp_path))

    parts = shards._route_grouped_shards(episodes, num_shards=4, seed=17)
    reversed_parts = shards._route_grouped_shards(
        list(reversed(episodes)),
        num_shards=4,
        seed=17,
    )

    keys = [
        [cohort._episode_key(episode) for episode in part]
        for part in parts
    ]
    reversed_keys = [
        [cohort._episode_key(episode) for episode in part]
        for part in reversed_parts
    ]
    assert keys == reversed_keys
    assert sum(len(part) for part in parts) == len(episodes)
    assert len({key for part in keys for key in part}) == len(episodes)

    owners = _owners(parts)
    assert len(owners) == len({cohort._route_key(item) for item in episodes})
    four_instruction_route = ("scene_0", 0)
    assert sum(
        cohort._route_key(episode) == four_instruction_route
        for part in parts
        for episode in part
    ) == 4
    assert four_instruction_route in owners


def test_each_scene_route_load_differs_by_at_most_one(tmp_path: Path) -> None:
    episodes = cohort._load_episodes(_write_dataset(tmp_path))
    parts = shards._route_grouped_shards(episodes, num_shards=4, seed=17)

    route_sets = [
        {cohort._route_key(episode) for episode in part}
        for part in parts
    ]
    scenes = sorted({episode["scene_id"] for episode in episodes})
    for scene in scenes:
        loads = [
            sum(route[0] == scene for route in route_set)
            for route_set in route_sets
        ]
        assert max(loads) - min(loads) <= 1
    episode_loads = [len(part) for part in parts]
    assert max(episode_loads) - min(episode_loads) <= 4


def test_cli_atomically_creates_then_verifies_exact_existing_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = _write_dataset(tmp_path)
    output_dir = tmp_path / "cohorts"
    monkeypatch.setattr(shards, "FJL_ROOT", tmp_path)
    argv = [
        "build_r2r_train_dagger_shards.py",
        "--dataset",
        str(dataset_path),
        "--count",
        str(len(_raw_episodes())),
        "--num-shards",
        "4",
        "--seed",
        "17",
        "--output-dir",
        str(output_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert shards.main() == 0
    expected_names = {"plan.json", *(f"shard_{index:02d}.json" for index in range(4))}
    assert {path.name for path in output_dir.iterdir()} == expected_names

    plan = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
    assert plan["schema"] == shards.PLAN_SCHEMA
    assert plan["selected_episode_count"] == len(_raw_episodes())
    assert plan["selected_route_count"] == 24
    assert plan["route_grouped"] is True
    assert sum(item["episode_count"] for item in plan["shards"]) == len(_raw_episodes())
    assert sum(item["route_count"] for item in plan["shards"]) == 24

    original = {
        path.name: path.read_bytes()
        for path in output_dir.iterdir()
    }
    assert shards.main() == 0
    assert {
        path.name: path.read_bytes()
        for path in output_dir.iterdir()
    } == original

    shard_path = output_dir / "shard_00.json"
    shard_path.write_bytes(shard_path.read_bytes() + b" ")
    assert shards.main() == 2


@pytest.mark.parametrize("count_delta", [-1, 1])
def test_cli_rejects_non_full_count_before_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    count_delta: int,
) -> None:
    dataset_path = _write_dataset(tmp_path)
    output_dir = tmp_path / f"partial_{count_delta}"
    monkeypatch.setattr(shards, "FJL_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_r2r_train_dagger_shards.py",
            "--dataset",
            str(dataset_path),
            "--count",
            str(len(_raw_episodes()) + count_delta),
            "--num-shards",
            "4",
            "--seed",
            "17",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert shards.main() == 2
    assert not output_dir.exists()
    assert (
        "--count must equal the complete dataset episode count"
        in capsys.readouterr().err
    )


def test_num_shards_cannot_exceed_route_count(tmp_path: Path) -> None:
    episodes = cohort._load_episodes(_write_dataset(tmp_path))
    with pytest.raises(shards.ShardError, match="exceeds route count"):
        shards._route_grouped_shards(
            episodes[:3],
            num_shards=4,
            seed=17,
        )

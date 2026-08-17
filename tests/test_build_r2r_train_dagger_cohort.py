from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import pytest
from scripts.tools import build_r2r_train_dagger_cohort as cohort

SCENES = ("scene_a", "scene_b", "scene_c")


def _raw_episodes() -> list[dict]:
    episodes: list[dict] = []
    for scene_index, scene_id in enumerate(SCENES):
        for route_offset in range(4):
            trajectory_id = scene_index * 100 + route_offset
            for paraphrase in range(2):
                episode_id = route_offset * 2 + paraphrase
                episodes.append(
                    {
                        "scene_id": f"mp3d/{scene_id}/{scene_id}.glb",
                        "episode_id": episode_id,
                        "trajectory_id": trajectory_id,
                        "instruction": {"instruction_text": (f"Take route {trajectory_id}, wording {paraphrase}.")},
                    }
                )
    return episodes


def _write_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "train.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"episodes": _raw_episodes()}, handle)
    return path


def _pilot_entries(episodes: list[dict]) -> list[dict]:
    by_scene = {episode["scene_id"]: episode for episode in episodes if episode["episode_id"] == 0}
    return [cohort._evaluator_episode(by_scene[scene_id]) for scene_id in SCENES]


def _write_exclude(path: Path, entries: list[dict]) -> Path:
    path.write_text(
        json.dumps({"split": "train", "count": len(entries), "episodes": entries}),
        encoding="utf-8",
    )
    return path


def _selected_unique(
    episodes: list[dict], excluded_routes: set[tuple[str, int]], count: int, seed: int = 17
) -> list[dict]:
    eligible = [episode for episode in episodes if cohort._route_key(episode) not in excluded_routes]
    representatives = cohort._choose_unique_trajectory_representatives(eligible, seed)
    return cohort._build_round_robin(representatives, count, seed)


def test_unique_trajectory_selection_is_deterministic_across_input_order(tmp_path: Path) -> None:
    episodes = cohort._load_episodes(_write_dataset(tmp_path))

    selected = _selected_unique(episodes, set(), count=9)
    selected_reversed = _selected_unique(list(reversed(episodes)), set(), count=9)

    assert [cohort._episode_key(item) for item in selected] == [cohort._episode_key(item) for item in selected_reversed]
    assert len({cohort._route_key(item) for item in selected}) == len(selected)


def test_exclude_cohort_removes_whole_routes_without_episode_or_route_overlap(
    tmp_path: Path,
) -> None:
    episodes = cohort._load_episodes(_write_dataset(tmp_path))
    pilot_entries = _pilot_entries(episodes)
    exclude_path = _write_exclude(tmp_path / "pilot.json", pilot_entries)

    excluded_routes, provenance = cohort._load_excluded_routes(exclude_path, episodes)
    selected = _selected_unique(episodes, excluded_routes, count=6)

    selected_episode_keys = {cohort._episode_key(item) for item in selected}
    pilot_episode_keys = {(str(item["scene_id"]), int(item["episode_id"])) for item in pilot_entries}
    assert selected_episode_keys.isdisjoint(pilot_episode_keys)
    assert {cohort._route_key(item) for item in selected}.isdisjoint(excluded_routes)
    assert len({cohort._route_key(item) for item in selected}) == 6
    assert provenance["episode_count"] == 3
    assert provenance["route_count"] == 3

    # Each excluded route has two instruction episodes; neither may survive.
    eligible = [item for item in episodes if cohort._route_key(item) not in excluded_routes]
    assert not ({cohort._route_key(item) for item in eligible} & excluded_routes)


@pytest.mark.parametrize(
    "entries, message",
    [
        ([{"scene_id": "unknown", "episode_id": 0}], "does not map"),
        (
            [
                {"scene_id": "scene_a", "episode_id": 0},
                {"scene_id": "scene_a", "episode_id": 0},
            ],
            "duplicate exclude cohort episode key",
        ),
        (
            [{"scene_id": "scene_a", "episode_id": 0, "trajectory_id": 999}],
            "trajectory_id disagrees",
        ),
    ],
)
def test_exclude_cohort_fails_closed_on_unknown_duplicate_or_mismatched_entries(
    tmp_path: Path, entries: list[dict], message: str
) -> None:
    episodes = cohort._load_episodes(_write_dataset(tmp_path))
    exclude_path = _write_exclude(tmp_path / "bad.json", entries)

    with pytest.raises(cohort.CohortError, match=message):
        cohort._load_excluded_routes(exclude_path, episodes)


def test_count_cannot_exceed_remaining_unique_trajectories(tmp_path: Path) -> None:
    episodes = cohort._load_episodes(_write_dataset(tmp_path))
    pilot_entries = _pilot_entries(episodes)
    exclude_path = _write_exclude(tmp_path / "pilot.json", pilot_entries)
    excluded_routes, _ = cohort._load_excluded_routes(exclude_path, episodes)
    eligible = [item for item in episodes if cohort._route_key(item) not in excluded_routes]
    representatives = cohort._choose_unique_trajectory_representatives(eligible, seed=17)

    assert len(representatives) == 9
    with pytest.raises(cohort.CohortError, match="exceeds available episodes 9"):
        cohort._build_round_robin(representatives, count=10, seed=17)


def test_unique_route_round_robin_is_scene_balanced_after_exclusion(tmp_path: Path) -> None:
    episodes = cohort._load_episodes(_write_dataset(tmp_path))
    pilot_entries = _pilot_entries(episodes)
    exclude_path = _write_exclude(tmp_path / "pilot.json", pilot_entries)
    excluded_routes, _ = cohort._load_excluded_routes(exclude_path, episodes)

    selected = _selected_unique(episodes, excluded_routes, count=6)
    scene_counts = Counter(item["scene_id"] for item in selected)

    assert scene_counts == {scene_id: 2 for scene_id in SCENES}


def test_cli_output_has_provenance_is_evaluator_compatible_and_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = _write_dataset(tmp_path)
    episodes = cohort._load_episodes(dataset_path)
    exclude_path = _write_exclude(tmp_path / "pilot.json", _pilot_entries(episodes))
    output_path = tmp_path / "tail.json"
    monkeypatch.setattr(cohort, "FJL_ROOT", tmp_path)
    argv = [
        "build_r2r_train_dagger_cohort.py",
        "--dataset",
        str(dataset_path),
        "--count",
        "6",
        "--seed",
        "17",
        "--exclude-cohort",
        str(exclude_path),
        "--unique-trajectories",
        "--output",
        str(output_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert cohort.main() == 0
    original = output_path.read_bytes()
    payload = json.loads(original)
    assert payload["split"] == "train"
    assert payload["count"] == 6
    assert all(set(item) == {"scene_id", "episode_id", "instruction"} for item in payload["episodes"])
    assert payload["selection"]["canonical_route"] == ["scene_id", "trajectory_id"]
    assert payload["selection"]["unique_trajectories"] is True
    assert payload["selection"]["exclude_cohort"]["sha256"] == cohort._sha256_file(exclude_path)
    assert payload["selection"]["selected_route_count"] == 6

    assert cohort.main() == 2
    assert output_path.read_bytes() == original

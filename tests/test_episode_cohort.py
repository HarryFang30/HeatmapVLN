from types import SimpleNamespace

import pytest

from scripts.evaluation.episode_cohort import (
    habitat_episode_key,
    load_episode_cohort,
    restrict_habitat_env_to_episode_keys,
)


def _episode(scene_id: str, episode_id: int):
    return SimpleNamespace(scene_id=f"/data/mp3d/{scene_id}/{scene_id}.glb", episode_id=episode_id)


def _env(episodes):
    dataset = SimpleNamespace(episodes=list(episodes))
    return SimpleNamespace(
        episodes=dataset.episodes,
        _episodes=dataset.episodes,
        _episode_iterator=iter(dataset.episodes),
        _dataset=dataset,
        number_of_episodes=len(dataset.episodes),
    )


def test_restrict_habitat_env_uses_exact_requested_order():
    first = _episode("scene-a", 1)
    second = _episode("scene-b", 2)
    skipped = _episode("scene-c", 3)
    env = _env([first, skipped, second])

    selected = restrict_habitat_env_to_episode_keys(
        env,
        [("scene-b", 2), ("scene-a", 1)],
    )

    assert selected == [second, first]
    assert env._episodes == [second, first]
    assert env._dataset.episodes == [second, first]
    assert list(env._episode_iterator) == [second, first]
    assert env.number_of_episodes == 2
    assert habitat_episode_key(first) == ("scene-a", 1)


def test_restrict_habitat_env_rejects_missing_and_duplicate_keys():
    env = _env([_episode("scene-a", 1)])

    with pytest.raises(ValueError, match="missing from Habitat dataset"):
        restrict_habitat_env_to_episode_keys(env, [("scene-b", 2)])
    with pytest.raises(ValueError, match="duplicate"):
        restrict_habitat_env_to_episode_keys(
            env,
            [("scene-a", 1), ("scene-a", 1)],
        )


def test_load_episode_cohort_retains_diagnostic_metadata(tmp_path):
    path = tmp_path / "cohort.json"
    path.write_text(
        '{"episodes": [{"scene_id": "scene-a", "episode_id": 7, '
        '"historical_false_stop_system2_call_index": 12}]}',
        encoding="utf-8",
    )

    keys, metadata = load_episode_cohort(path)

    assert keys == [("scene-a", 7)]
    assert metadata[("scene-a", 7)][
        "historical_false_stop_system2_call_index"
    ] == 12


def test_load_episode_cohort_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "cohort.json"
    path.write_text(
        '{"episodes": [{"scene_id": "scene-a", "episode_id": 7}, '
        '{"scene_id": "scene-a", "episode_id": 7}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key"):
        load_episode_cohort(path)

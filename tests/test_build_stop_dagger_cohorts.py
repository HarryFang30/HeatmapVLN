import json

from scripts.evaluation.build_stop_dagger_cohorts import (
    _rollout_report_label_paths,
    build_cohorts,
    shard_cohort,
)


def _episode(scene: str, episode_id: int, distance: float):
    return {
        "scene_id": f"mp3d/{scene}/{scene}.glb",
        "episode_id": episode_id,
        "info": {"geodesic_distance": distance},
    }


def test_builds_disjoint_scene_balanced_longest_cohorts():
    episodes = [
        _episode(scene, episode_id, float(episode_id))
        for scene in ("scene_a", "scene_b")
        for episode_id in range(1, 6)
    ]
    cohorts = build_cohorts(
        episodes,
        excluded={("scene_a", 5)},
        num_cohorts=2,
        per_scene_per_cohort=1,
        seed=7,
    )

    assert [[row["episode_id"] for row in cohort] for cohort in cohorts] == [
        [4, 5],
        [3, 4],
    ]
    keys = [
        (row["scene_id"], row["episode_id"])
        for cohort in cohorts
        for row in cohort
    ]
    assert len(keys) == len(set(keys)) == 4


def test_can_explicitly_skip_scenes_without_enough_unused_episodes():
    episodes = [
        _episode("complete", episode_id, float(episode_id))
        for episode_id in range(1, 5)
    ] + [_episode("short", 1, 1.0)]

    cohorts = build_cohorts(
        episodes,
        excluded=set(),
        num_cohorts=2,
        per_scene_per_cohort=1,
        seed=7,
        allow_incomplete_scenes=True,
    )

    assert [[row["scene_id"] for row in cohort] for cohort in cohorts] == [
        ["complete"],
        ["complete"],
    ]


def test_q25_selects_lower_quartile_without_changing_scene_balance():
    episodes = [
        _episode(scene, episode_id, float(episode_id))
        for scene in ("scene_a", "scene_b")
        for episode_id in range(1, 10)
    ]

    cohorts = build_cohorts(
        episodes,
        excluded=set(),
        num_cohorts=1,
        per_scene_per_cohort=1,
        seed=7,
        selection="q25",
    )

    assert [[row["episode_id"] for row in cohort] for cohort in cohorts] == [
        [3, 3]
    ]


def test_shards_are_complete_disjoint_and_balanced():
    cohort = [
        {"scene_id": f"scene_{index:02d}", "episode_id": index}
        for index in range(10)
    ]

    shards = shard_cohort(cohort, num_shards=4)

    assert [len(shard) for shard in shards] == [3, 3, 2, 2]
    assert {
        (row["scene_id"], row["episode_id"])
        for shard in shards
        for row in shard
    } == {(row["scene_id"], row["episode_id"]) for row in cohort}


def test_rollout_report_expands_to_validated_multimodal_labels(tmp_path):
    root = tmp_path / "rollout"
    root.mkdir()
    labels = root / "system2_stop_multimodal_examples.jsonl"
    labels.write_text("{}\n", encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "status": "passed",
                "root_count": 1,
                "roots": [{"root": str(root)}],
            }
        ),
        encoding="utf-8",
    )

    assert _rollout_report_label_paths([report]) == [labels]

import gzip
import json
import sys

from scripts.evaluation import build_r2r_scene_balanced_cohort as cohort


def test_max_episodes_preserves_round_robin_scene_coverage(tmp_path, monkeypatch):
    data_path = tmp_path / "val_unseen.json.gz"
    output_path = tmp_path / "cohort.json"
    episodes = []
    for scene_index in range(3):
        scene = f"scene_{scene_index}"
        for episode_index in range(5):
            episodes.append(
                {
                    "scene_id": f"data/scene_datasets/{scene}/{scene}.glb",
                    "episode_id": scene_index * 100 + episode_index,
                }
            )
    with gzip.open(data_path, "wt", encoding="utf-8") as handle:
        json.dump({"episodes": episodes}, handle)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_r2r_scene_balanced_cohort.py",
            "--data-path",
            str(data_path),
            "--output",
            str(output_path),
            "--episodes-per-scene",
            "5",
            "--max-episodes",
            "7",
            "--seed",
            "42",
        ],
    )

    assert cohort.main() == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    selected_scenes = [row["scene_id"] for row in payload["episodes"]]

    assert len(selected_scenes) == 7
    assert set(selected_scenes[:3]) == {"scene_0", "scene_1", "scene_2"}
    assert selected_scenes[:3] == selected_scenes[3:6]
    assert payload["max_episodes"] == 7
    assert payload["scene_count"] == 3

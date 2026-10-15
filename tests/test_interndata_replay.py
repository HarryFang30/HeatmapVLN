import io
import json

import numpy as np
import pytest
from PIL import Image

from src.data.interndata_replay import episode_samples, sample_jpegs, scene_samples


# Episode 0 of InternData-N1 r2r scene gZ6f7yhEvPG (31 frames), goal.125cm_30deg per frame.
ACTIONS = [-1, 1, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1]
GOALS = {
    0: (12, [495, 345]), 1: (9, [466, 398]), 3: (7, [418, 431]), 7: (18, [513, 208]), 8: (17, [528, 222]),
    9: (17, [408, 192]), 10: (15, [430, 218]), 11: (14, [336, 211]), 12: (13, [338, 222]), 13: (12, [246, 225]),
    14: (11, [240, 243]), 15: (11, [210, 250]), 16: (9, [223, 293]), 17: (12, [131, 268]), 18: (11, [111, 296]),
    20: (9, [185, 298]), 21: (8, [168, 333]), 22: (7, [253, 313]), 23: (6, [243, 352]), 24: (5, [233, 403]),
    25: (4, [301, 392]),
}
REL = [GOALS.get(t, (-1, None))[0] for t in range(31)]
GOAL = [GOALS.get(t, (None, [-1, -1]))[1] for t in range(31)]


def _samples(**kw):
    return episode_samples(
        scene_id="gZ6f7yhEvPG", episode_index=0, instruction="Walk forward.",
        parquet_actions=ACTIONS, goals=GOAL, relative_goal_frame_ids=REL, **kw,
    )


def test_internnav_selection_on_a_real_episode():
    got = [(s.frame_id, s.kind, s.turn1, s.turn2) for s in _samples()]
    assert got == [
        (0, "pixel", "↓", "495 345"),
        (4, "turn", "→", None),
        (8, "pixel", "↓", "528 222"),
        (12, "pixel", "↓", "338 222"),
        (16, "pixel", "↓", "223 293"),
        (20, "pixel", "↓", "185 298"),
        (24, "pixel", "↓", "233 403"),
        (28, "turn", "←", None),
        (30, "stop", "STOP", None),
    ]
    by_frame = {s.frame_id: s for s in _samples()}
    assert by_frame[0].history_frame_ids == ()
    assert by_frame[4].history_frame_ids == (0, 1, 2, 3)
    assert by_frame[30].history_frame_ids == (0, 4, 8, 12, 16, 20, 24, 29)


def test_off_grid_phase_shifts_the_frames_and_drops_the_shared_stop():
    got = _samples(phase=2)
    assert got and all(s.frame_id % 4 == 2 for s in got)
    assert all(s.kind != "stop" for s in got)
    assert max(s.frame_id for s in got) < 30
    with pytest.raises(ValueError, match="phase"):
        _samples(phase=4)


def test_near_goals_are_skipped():
    rel = list(REL)
    rel[24] = 2
    frames = {s.frame_id for s in episode_samples(
        scene_id="s", episode_index=0, instruction="x", parquet_actions=ACTIONS, goals=GOAL,
        relative_goal_frame_ids=rel)}
    assert 24 not in frames and 20 in frames


def _jpeg(size, value):
    buf = io.BytesIO()
    Image.new("RGB", size, (value, 0, 0)).save(buf, format="JPEG")
    return buf.getvalue()


def test_scene_reader_and_client_image_path(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    scene = tmp_path / "gZ6f7yhEvPG"
    (scene / "meta").mkdir(parents=True)
    (scene / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["Walk forward. <INSTRUCTION_SEP>Go ahead. "], "length": 31}) + "\n"
    )
    (scene / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.table({
            "action": ACTIONS,
            "goal.125cm_30deg": GOAL,
            "relative_goal_frame_id.125cm_30deg": REL,
        }),
        scene / "data" / "chunk-000" / "episode_000000.parquet",
    )
    for setting in ("125cm_0deg", "125cm_30deg"):
        d = scene / "videos" / "chunk-000" / f"observation.images.rgb.{setting}"
        d.mkdir(parents=True)
        for t in range(31):
            (d / f"episode_000000_{t}.jpg").write_bytes(_jpeg((640, 480), t))

    samples = list(scene_samples(scene))
    assert len(samples) == 18
    assert {s.instruction for s in samples} == {"Walk forward. ", "Go ahead. "}  # sent as is, never stripped

    seen = []

    def encode(arr, quality):
        seen.append((arr.shape, quality))
        return b"jpeg"

    pixel = next(s for s in samples if s.frame_id == 4 + 4)
    history, current, lookdown = sample_jpegs(scene, pixel, encode=encode)
    assert len(history) == len(pixel.history_frame_ids) and current == b"jpeg" and lookdown == b"jpeg"
    assert seen[:-1] == [((384, 384, 3), 90)] * (len(history) + 1)
    assert seen[-1] == ((480, 640, 3), 90)  # lookdown keeps 640x480
    turn = next(s for s in samples if s.kind == "turn")
    assert sample_jpegs(scene, turn, encode=encode)[2] == b"jpeg"  # captured on every call


def test_replay_instructions_follow_the_clients_normalisation():
    import ast
    from pathlib import Path

    from src.data.interndata_replay import client_instruction

    # The client's own function, read from its source so the two cannot drift apart.
    source = Path(__file__).resolve().parents[1] / "scripts" / "evaluation" / "r2r_val_unseen.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_normalize_instruction")
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    for raw in ("Walk forward. ", "Walk forward.", "Stop!", "Where?", "no punctuation", "  padded  "):
        assert client_instruction(raw) == namespace["_normalize_instruction"](raw)

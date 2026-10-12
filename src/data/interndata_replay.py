"""Expert replay samples from InternData-N1 (LeRobot v2.1, ``vln_ce/traj_data/r2r``).

EXP-23 mixes these into the System2 DAgger fine-tune so the adapter keeps the
released model's expert behaviour while it learns the corrections.  Sample
selection and targets follow InternNav's own ``NavPixelGoalDataset``
(``internnav/dataset/internvla_n1_lerobot_dataset.py`` at 0f82252), because
that is the distribution the released System2 was trained on:

* one sample every ``sample_step`` frames; frame ``t``'s action is the parquet
  action of frame ``t + 1`` (the parquet stores the action that *led to* each
  frame);
* a visible goal (``goal.<setting> != [-1, -1]``) gives ``"↓"`` then
  ``"u v"``, printed as stored -- column first, the order the released model
  emits (its trajectory outputs reach 601 in the first number and never pass
  455 in the second);  goals fewer than 3 frames ahead are skipped;
* no visible goal and a forward next action is skipped; otherwise the turn
  actions up to the next forward, at most ``num_future_steps`` of them;
* the last frame of every episode is a STOP sample;
* history is ``linspace(0, t - 1, num_history)`` unique.

The *inputs*, by contrast, follow deployment rather than InternNav training:
the fixed ``"you can see "`` conjunction (via ``build_native_messages`` in the
trainer, not here) and the client's image path -- front views resized to the
VLM size and JPEG-encoded at the RPC quality, the 640x480 lookdown encoded
unresized -- so a replay sample and a DAgger record reach the model through
the same decode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
from PIL import Image


IDX2ACTION = {0: "STOP", 1: "↑", 2: "←", 3: "→", 5: "↓"}
FRONT_SETTING = "125cm_0deg"
LOOKDOWN_SETTING = "125cm_30deg"
INSTRUCTION_SEP = "<INSTRUCTION_SEP>"


@dataclass(frozen=True)
class ReplaySample:
    scene_id: str
    episode_index: int
    instruction: str
    frame_id: int
    history_frame_ids: tuple[int, ...]
    kind: str  # "pixel" | "turn" | "stop"
    turn1: str
    turn2: str | None


def _history_ids(frame_id: int, num_history: int) -> tuple[int, ...]:
    if frame_id <= 0:
        return ()
    return tuple(
        int(i) for i in np.unique(np.linspace(0, frame_id - 1, num_history, dtype=np.int32))
    )


def episode_samples(
    *,
    scene_id: str,
    episode_index: int,
    instruction: str,
    parquet_actions: Sequence[int],
    goals: Sequence[Sequence[int]],
    relative_goal_frame_ids: Sequence[int],
    sample_step: int = 4,
    num_history: int = 8,
    num_future_steps: int = 4,
) -> list[ReplaySample]:
    """InternNav's sample selection for one (episode, instruction) pair."""
    actions = [int(a) for a in list(parquet_actions)[1:]] + [0]
    length = len(actions)
    if length < 4:
        return []
    out: list[ReplaySample] = []

    def make(frame_id: int, kind: str, turn1: str, turn2: str | None) -> ReplaySample:
        return ReplaySample(
            scene_id=scene_id,
            episode_index=int(episode_index),
            instruction=instruction,
            frame_id=int(frame_id),
            history_frame_ids=_history_ids(frame_id, num_history),
            kind=kind,
            turn1=turn1,
            turn2=turn2,
        )

    for n in range(length // sample_step + 1):
        t = n * sample_step
        if t == length or t == length - 1:
            continue
        goal = [int(v) for v in goals[t]]
        if goal[0] == -1:
            if actions[t] == 1:
                continue
            turns = []
            for i in range(t, min(length, t + num_future_steps)):
                if actions[i] == 1:
                    break
                turns.append(actions[i])
            out.append(make(t, "turn", "".join(IDX2ACTION[a] for a in turns), None))
        else:
            if int(relative_goal_frame_ids[t]) < 3:
                continue
            out.append(make(t, "pixel", IDX2ACTION[5], f"{goal[0]} {goal[1]}"))
    out.append(make(length - 1, "stop", IDX2ACTION[0], None))
    return out


def scene_samples(scene_dir: str | Path, **kwargs) -> Iterator[ReplaySample]:
    """All replay samples of one extracted scene directory (``<scene>/meta``, ``data``)."""
    import pyarrow.parquet as pq

    scene_dir = Path(scene_dir)
    scene_id = scene_dir.name
    with open(scene_dir / "meta" / "episodes.jsonl", encoding="utf-8") as handle:
        episodes = [json.loads(line) for line in handle if line.strip()]
    for ep in episodes:
        index = int(ep["episode_index"])
        table = pq.read_table(
            scene_dir / "data" / f"chunk-{index // 1000:03d}" / f"episode_{index:06d}.parquet",
            columns=["action", f"goal.{LOOKDOWN_SETTING}", f"relative_goal_frame_id.{LOOKDOWN_SETTING}"],
        ).to_pydict()
        if len(table["action"]) != int(ep["length"]):
            raise ValueError(f"{scene_id}/{index}: action length != episode length")
        for instruction in str(ep["tasks"][0]).split(INSTRUCTION_SEP):
            yield from episode_samples(
                scene_id=scene_id,
                episode_index=index,
                instruction=instruction.strip(),
                parquet_actions=table["action"],
                goals=table[f"goal.{LOOKDOWN_SETTING}"],
                relative_goal_frame_ids=table[f"relative_goal_frame_id.{LOOKDOWN_SETTING}"],
                **kwargs,
            )


def _frame_path(scene_dir: Path, setting: str, episode_index: int, frame_id: int) -> Path:
    return (
        scene_dir
        / "videos"
        / f"chunk-{episode_index // 1000:03d}"
        / f"observation.images.rgb.{setting}"
        / f"episode_{episode_index:06d}_{frame_id}.jpg"
    )


def _client_jpeg(
    path: Path,
    size: tuple[int, int] | None,
    quality: int,
    encode: Callable[[np.ndarray, int], bytes],
) -> bytes:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != tuple(size):
        image = image.resize(tuple(size))  # the client's _rgb_array_to_pil resize
    return encode(np.asarray(image, dtype=np.uint8), quality)


def _default_encode(arr: np.ndarray, quality: int) -> bytes:
    from vla_rpc.core.image import encode_rgb_to_jpeg

    return encode_rgb_to_jpeg(arr, quality=quality)


def sample_jpegs(
    scene_dir: str | Path,
    sample: ReplaySample,
    *,
    vlm_image_size: tuple[int, int] = (384, 384),
    jpeg_quality: int = 90,
    encode: Callable[[np.ndarray, int], bytes] | None = None,
) -> tuple[list[bytes], bytes, bytes | None]:
    """(history fronts, current front, lookdown or None) as the client would send them."""
    scene_dir = Path(scene_dir)
    encode = encode or _default_encode
    front = [
        _client_jpeg(_frame_path(scene_dir, FRONT_SETTING, sample.episode_index, f), vlm_image_size, jpeg_quality, encode)
        for f in (*sample.history_frame_ids, sample.frame_id)
    ]
    lookdown = None
    if sample.turn1 == IDX2ACTION[5]:
        lookdown = _client_jpeg(
            _frame_path(scene_dir, LOOKDOWN_SETTING, sample.episode_index, sample.frame_id),
            None,
            jpeg_quality,
            encode,
        )
    return front[:-1], front[-1], lookdown


__all__ = [
    "FRONT_SETTING",
    "IDX2ACTION",
    "LOOKDOWN_SETTING",
    "ReplaySample",
    "episode_samples",
    "sample_jpegs",
    "scene_samples",
]

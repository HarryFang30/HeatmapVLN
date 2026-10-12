"""System2 LoRA SFT samples from EXP-23 on-policy records (design §2 C).

EXP-14's LoRA kept only 58% of the model's own pixel answers, and at least one
cause is known: it trained on ``construct_input_stage2``, which is six tokens away
from the prompt the server builds (ledger pitfall #1).  So nothing here builds a
prompt or prepares an image by itself:

* prompts come from ``build_native_messages`` / ``append_native_lookdown_turn``,
  the two functions ``rpc_model_server._plan_panoramic_native`` calls;
* images take the server's path (``_pil_from_blob``): the exact JPEG bytes the
  client sent (``s2_records``) -> ``vla_rpc`` ``decode_jpeg_to_rgb`` -> PIL ->
  ``resize(vlm_image_size)`` for the history and current fronts, while the 640x480
  lookdown is not resized;
* tokens come from ``processor.apply_chat_template``.

Each sample is checked when it is built.  The tokens in front of every assistant
turn must equal ``apply_chat_template(..., add_generation_prompt=True)`` of the
messages the server would have generated that turn from.  Only the assistant
content and its ``<|im_end|>`` are supervised.

Targets (design §2 B4).  Where the labeller judged the model's own answer good
enough (``agree``), the target is that answer verbatim, both turns if it looked
down, so most of training is self-distillation that anchors the policy.
Otherwise the target is the oracle answer in the native protocol: ``STOP``; ``↓``
then ``"u v"`` (column first); or a run of ``←``/``→``.

Selection.  Six of the 61 R2R train scenes are held out by a salted hash: they
never enter training, and the offline gates run on them.  An episode contributes
at most 48 calls, because an agent stuck in a loop would otherwise fill the set.
Calls the model got wrong are drawn three times as often as calls it got right.

Label rows (``heatmapvln-s2-labels-v1``, one per call, written by the labeller
next to the record).  These are the fields read here:

    call_index        int, the join key into calls.jsonl
    skip              bool, optional; the state gets no label
    agree             bool or null (null: unjudgeable, no target), design §2 B4
    oracle_kind       "stop" | "pixel" | "turn" | "skip"
    oracle_turn1/2    the oracle's answer as served turns (turn2 null for one turn);
                      a nested {"oracle": {"kind", "turns" | "text"}} is also read
    target_turn1/2    optional explicit target (or ``target_turns``); when ``agree``
                      it must equal the recorded answer
    distance_to_goal  geodesic d at the call; used by the gates, else calls.jsonl's
    oracle_path       [[x, y, z], ...], optional world route; only gate G4's pixel
                      test reads it (without it pixel corrections stay unscored)
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Iterator, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from src.data.s2_records import EpisodeRecord, iter_record_paths, label_path_for, read_labels
from src.models.heatmap.native_internnav_exact import (
    NATIVE_ACTION_FORWARD,
    NATIVE_ACTION_LEFT,
    NATIVE_ACTION_LOOKDOWN,
    NATIVE_ACTION_RIGHT,
    NATIVE_ACTION_STOP,
    NATIVE_LOOKDOWN_SIZE,
    append_native_lookdown_turn,
    build_native_messages,
    native_requests_lookdown,
    parse_native_actions,
)

SFT_SCHEMA = "heatmapvln-s2-sft-v1"
IGNORE_INDEX = -100
IM_END = "<|im_end|>"
LOOKDOWN_TEXT = "↓"
STOP_TEXT = "STOP"

# configs/ppa_action_refine_v2_8gpu.yaml data.image_size, the server's fallback when
# a request carries no vlm_image_size.  Records normally carry their own.
DEFAULT_VLM_IMAGE_SIZE = (384, 384)
DEFAULT_MAX_CALLS_PER_EPISODE = 48
# An agent spinning inside the goal zone yields one STOP correction per call (32 of the
# first smoke episode's 60 calls were such near-duplicates); the earliest few carry the
# lesson "stop when you get here".
DEFAULT_MAX_STOP_CORRECTIONS_PER_EPISODE = 4
DEFAULT_AGREE_WEIGHT = 1.0
DEFAULT_DISAGREE_WEIGHT = 3.0
DEFAULT_HOLDOUT_COUNT = 6
HOLDOUT_SALT = "exp23-holdout-v1"

# The 61 scenes of R2R_VLNCE_v1-3_preprocessed/train/train.json.gz (10,819 episodes),
# read on the 4090 on 2026-10-12.  The held-out set is drawn from this fixed list,
# not from whichever scenes a round happened to collect, so it cannot move between
# rounds.
R2R_TRAIN_SCENES = (
    "17DRP5sb8fy", "1LXtFkjw3qL", "1pXnuDYAj8r", "29hnd4uzFmX", "2n8kARJN3HM",
    "5LpN3gDmAk7", "5q7pvUzZiYa", "759xd9YjKW5", "7y3sRwLe3Va", "82sE5b5pLXE",
    "8WUmhLawc2A", "B6ByNegPMKs", "D7G3Y4RVNrH", "D7N2EKCX4Sj", "E9uDoFAP3SH",
    "EDJbREhghzL", "GdvgFV5R1Z5", "HxpKQynjfin", "JF19kD82Mey", "JeFG25nYj2p",
    "JmbYfDe2QKZ", "PX4nDJXEHrG", "Pm6F8kyY3z2", "PuKPg4mmafe", "S9hNv5qa7GM",
    "SN83YJsR3w2", "ULsKaCPVFJR", "Uxmj2M2itWa", "V2XKFyX4ASd", "VFuaQ6m2Qom",
    "VLzqgDo317F", "VVfe2KiqLaN", "Vvot9Ly1tCj", "VzqfbhrpDEA", "XcA2TqTSSAj",
    "YmJkqBEsHnH", "ZMojNkEp431", "aayBHfsNo7d", "ac26ZMwG7aT", "b8cTxDM8gDG",
    "cV4RVeZvu5T", "dhjEzFoUFzH", "e9zR4mvMWw7", "gTV8FGcVJC9", "gZ6f7yhEvPG",
    "i5noydFURQK", "jh4fc5c5qoQ", "kEZ7cmS4wCh", "mJXqzFtmKg4", "p5wJjkQkbXX",
    "pRbA3pwrgk9", "qoiz87JEwZ2", "r1Q1Z4BcV1o", "r47D5H71a5s", "rPc6DW4iMge",
    "s8pcmisQ38h", "sKLMLpTHeUy", "sT4fr6TAbpF", "uNb9QFRL6hY", "ur6pFq6Qu1A",
    "vyrNrziPKCB",
)

ORACLE_KINDS = ("stop", "pixel", "turn")
_DIGIT = re.compile(r"\d")
_INTEGER = re.compile(r"\d+")


class S2SFTError(ValueError):
    """A record, label or target that cannot be turned into a faithful sample."""


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------


def holdout_scenes(
    count: int = DEFAULT_HOLDOUT_COUNT,
    *,
    salt: str = HOLDOUT_SALT,
    scenes: Sequence[str] = R2R_TRAIN_SCENES,
) -> tuple[str, ...]:
    """The ``count`` scenes with the smallest ``sha256(salt:scene)``, sorted by name."""
    if not 0 <= int(count) <= len(scenes):
        raise ValueError(f"holdout count {count} outside [0, {len(scenes)}]")
    ranked = sorted(scenes, key=lambda s: hashlib.sha256(f"{salt}:{s}".encode("utf-8")).hexdigest())
    return tuple(sorted(ranked[: int(count)]))


# ---------------------------------------------------------------------------
# Decisions and targets
# ---------------------------------------------------------------------------


def decision_class(text: str) -> str:
    """What the server does with a final System2 answer (``_plan_panoramic_native``).

    ``pixel`` (digits: a pixel goal), ``stop`` (no digits and no arrow -- the
    server's ``fallback_stop`` -- or a leading ``STOP``), ``turn_left`` /
    ``turn_right`` / ``forward`` / ``lookdown`` (the leading primitive of an arrow
    chunk), or ``invalid`` (digits but fewer than two integers, which the server
    refuses).
    """
    text = text or ""
    if _DIGIT.search(text):
        return "pixel" if len(_INTEGER.findall(text)) >= 2 else "invalid"
    actions = parse_native_actions(text)
    if not actions or actions[0] == NATIVE_ACTION_STOP:
        return "stop"
    return {
        NATIVE_ACTION_LEFT: "turn_left",
        NATIVE_ACTION_RIGHT: "turn_right",
        NATIVE_ACTION_FORWARD: "forward",
        NATIVE_ACTION_LOOKDOWN: "lookdown",
    }[actions[0]]


def pixel_uv(text: str) -> tuple[int, int] | None:
    """(u, v) -- column, row -- of a native pixel answer, which is printed column first.

    The released model's first number reaches 601 on the 640x480 lookdown and its
    second never passes 455; InternData-N1 prints ``goal = [u, v]`` the same way.
    (The server's ``_parse_internnav_pixel_goal`` calls it ``"row col"``; it only
    re-encodes the same text for System1, so that naming never reaches an action.)
    """
    if not _DIGIT.search(text or ""):
        return None
    values = [int(v) for v in _INTEGER.findall(text)]
    if len(values) < 2:
        return None
    return values[0], values[1]


def validate_turns(turns: Sequence[str], *, what: str) -> tuple[str, ...]:
    """A target the deployed two-turn protocol can produce, or an error.

    Two turns exist only after a first turn the server reads as a lookdown request;
    a single turn must not be one, or the server would ask for a second.  An empty
    turn is legal: the model can end its answer at once (the server's fallback_stop).
    """
    turns = tuple(str(t) for t in turns)
    if len(turns) not in (1, 2):
        raise S2SFTError(f"{what}: expected one or two turns, got {turns!r}")
    if len(turns) == 2 and not native_requests_lookdown(turns[0]):
        raise S2SFTError(f"{what}: a second turn follows {turns[0]!r}, which is not a lookdown request")
    if len(turns) == 1 and native_requests_lookdown(turns[0]):
        raise S2SFTError(f"{what}: {turns[0]!r} requests a lookdown but no second turn follows")
    return turns


def recorded_turns(call: Mapping[str, Any]) -> tuple[str, ...]:
    """The assistant turns the deployed model produced for this call."""
    final = str(call["llm_output"])
    looked_down = int(call.get("native_lookdown_turns") or 0)
    if looked_down == 0:
        return validate_turns([final], what=f"call {call.get('call_index')}")
    if looked_down != 1 or call.get("native_first_output") is None:
        raise S2SFTError(
            f"call {call.get('call_index')}: native_lookdown_turns={looked_down} "
            f"with native_first_output={call.get('native_first_output')!r}"
        )
    return validate_turns(
        [str(call["native_first_output"]), final], what=f"call {call.get('call_index')}"
    )


def _oracle_turns_from_text(kind: str, text: str) -> list[str]:
    text = str(text).strip()
    if kind == "stop":
        return [STOP_TEXT]
    if kind == "pixel":
        if text.startswith(LOOKDOWN_TEXT):
            text = text[len(LOOKDOWN_TEXT):].strip()
        if pixel_uv(text) is None:
            raise S2SFTError(f"oracle pixel text {text!r} has no 'u v'")
        return [LOOKDOWN_TEXT, text]
    return [text]


@dataclass(frozen=True)
class CallLabel:
    """One labeller row, normalised."""

    call_index: int
    skip: bool
    agree: bool | None
    oracle_kind: str | None
    oracle_turns: tuple[str, ...] | None
    target_turns: tuple[str, ...] | None
    distance_to_goal: float | None
    oracle_path: tuple[tuple[float, float, float], ...] | None


def _turn_pair(first: Any, second: Any) -> list[str] | None:
    """The labeller's ``*_turn1`` / ``*_turn2`` columns as a turn list (None: no answer)."""
    if first is None:
        return None
    return [str(first)] + ([str(second)] if second is not None else [])


def normalize_label(row: Mapping[str, Any]) -> CallLabel:
    """One labeller row, in either the nested or the flat (``oracle_turn1``...) shape.

    A row is trainable when it names a target, when the model's own answer was
    accepted (``agree`` true; that can happen with an oracle of kind ``skip``),
    or when it was rejected and the oracle has an answer.  Everything else
    (``agree`` null, an explicit ``skip``) is kept as a skipped label: no target,
    but still a call the offline gates can count.
    """
    index = int(row["call_index"])
    what = f"label of call {index}"
    oracle = row.get("oracle") if isinstance(row.get("oracle"), Mapping) else {}
    kind = oracle.get("kind", row.get("oracle_kind", row.get("label")))
    kind = str(kind).lower() if kind is not None else None
    if kind is not None and kind not in ORACLE_KINDS + ("skip",):
        raise S2SFTError(f"{what}: oracle kind {kind!r} not in {ORACLE_KINDS + ('skip',)}")
    distance = row.get("distance_to_goal", row.get("d"))
    distance = float(distance) if distance is not None else None
    path = row.get("oracle_path")
    path = tuple(tuple(float(v) for v in point) for point in path) if path else None
    if row.get("skip"):
        return CallLabel(index, True, None, kind, None, None, distance, path)
    if "agree" not in row:
        raise S2SFTError(f"{what}: no 'agree' field")
    agree = None if row["agree"] is None else bool(row["agree"])

    oracle_turns = None
    if kind in ORACLE_KINDS:
        turns = oracle.get("turns", row.get("oracle_turns"))
        if turns is None:
            turns = _turn_pair(row.get("oracle_turn1"), row.get("oracle_turn2"))
        if turns is None:
            text = oracle.get("text", row.get("oracle_text"))
            if text is None:
                raise S2SFTError(f"{what}: oracle has neither turns nor text")
            turns = _oracle_turns_from_text(kind, text)
        oracle_turns = validate_turns(turns, what=f"{what} (oracle)")
        expected = {"stop": {"stop"}, "pixel": {"pixel"}, "turn": {"turn_left", "turn_right"}}[kind]
        if decision_class(oracle_turns[-1]) not in expected:
            raise S2SFTError(f"{what}: oracle turns {oracle_turns!r} are not a {kind}")
    target = row.get("target_turns")
    if target is None:
        target = _turn_pair(row.get("target_turn1"), row.get("target_turn2"))
    target_turns = validate_turns(target, what=f"{what} (target)") if target is not None else None
    trainable = target_turns is not None or agree is True or (agree is False and oracle_turns is not None)
    return CallLabel(index, not trainable, agree, kind, oracle_turns, target_turns, distance, path)


def training_turns(recorded: Sequence[str], label: CallLabel) -> tuple[str, ...]:
    """The supervised answer: the model's own (``recorded_turns``) when it agrees, else the oracle's."""
    if label.skip:
        raise S2SFTError(f"call {label.call_index} is skipped by the labeller")
    if label.target_turns is not None:
        return label.target_turns
    if label.agree:
        return validate_turns(recorded, what=f"call {label.call_index} (recorded)")
    assert label.oracle_turns is not None
    return label.oracle_turns


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallRef:
    """One System2 call of one record, with its label joined (when there is one)."""

    record_path: str
    scene_id: str
    episode_id: int
    call_index: int
    recorded: tuple[str, ...]
    distance_to_goal: float | None
    label: CallLabel | None = None
    turns: tuple[str, ...] | None = None
    weight: float = 1.0

    @property
    def key(self) -> str:
        return f"{self.scene_id}_{self.episode_id:06d}#{self.call_index}"

    @property
    def uid(self) -> str:
        """Unique across roots: two rounds can record the same episode."""
        return f"{self.record_path}#{self.call_index}"


def _pair_label_roots(
    record_roots: Sequence[str | Path], label_roots: Sequence[str | Path] | None
) -> list[tuple[Path, Path | None]]:
    record_roots = [Path(r) for r in record_roots]
    if not label_roots:
        return [(r, None) for r in record_roots]
    label_roots = [Path(r) for r in label_roots]
    if len(label_roots) == 1:
        label_roots = label_roots * len(record_roots)
    if len(label_roots) != len(record_roots):
        raise ValueError(
            f"{len(record_roots)} record roots but {len(label_roots)} label roots; give one label "
            "root per record root, or a single one for all of them"
        )
    return list(zip(record_roots, label_roots))


def index_calls(
    record_roots: Sequence[str | Path],
    label_roots: Sequence[str | Path] | None,
    *,
    split: str,
    holdout: Collection[str],
    require_labels: bool = True,
    verify_records: bool = False,
) -> tuple[list[CallRef], Counter]:
    """Every call of every committed record in ``split`` ("train", "holdout" or "all").

    A call whose label is missing (or whose labels file is missing) is dropped and
    counted when ``require_labels``; otherwise it is kept with ``label=None``.
    Nothing is defaulted: a malformed label or recorded answer raises.
    """
    if split not in ("train", "holdout", "all"):
        raise ValueError(f"split must be train, holdout or all, not {split!r}")
    holdout = set(holdout)
    refs: list[CallRef] = []
    stats: Counter = Counter()
    for record_root, label_root in _pair_label_roots(record_roots, label_roots):
        for path in iter_record_paths(record_root):
            scene = path.parent.name
            in_holdout = scene in holdout
            if (split == "train" and in_holdout) or (split == "holdout" and not in_holdout):
                stats["episodes_other_split"] += 1
                continue
            record = EpisodeRecord(path, verify=verify_records)
            labels: dict[int, CallLabel] = {}
            label_file = label_path_for(path, label_root) if label_root is not None else None
            if label_file is not None and label_file.is_file():
                for row in read_labels(label_file):
                    label = normalize_label(row)
                    if label.call_index in labels:
                        raise S2SFTError(f"{label_file}: call {label.call_index} labelled twice")
                    labels[label.call_index] = label
            elif require_labels:
                stats["episodes_without_labels"] += 1
                stats["calls_without_label"] += len(record.calls)
                continue
            stats["episodes"] += 1
            for call in record.calls:
                index = int(call["call_index"])
                label = labels.get(index)
                if label is None and require_labels:
                    stats["calls_without_label"] += 1
                    continue
                # The labeller's geodesic is the d its R_stop rule used; the gates use the same.
                distance = label.distance_to_goal if label is not None else None
                if distance is None:
                    distance = call.get("distance_to_goal")
                refs.append(
                    CallRef(
                        record_path=str(path),
                        scene_id=record.scene_id,
                        episode_id=record.episode_id,
                        call_index=index,
                        recorded=recorded_turns(call),
                        distance_to_goal=float(distance) if distance is not None else None,
                        label=label,
                    )
                )
                stats["calls"] += 1
    return refs, stats


def _stable_rank(*parts: Any) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def select_training_refs(
    refs: Sequence[CallRef],
    *,
    max_calls_per_episode: int | None = DEFAULT_MAX_CALLS_PER_EPISODE,
    agree_weight: float = DEFAULT_AGREE_WEIGHT,
    disagree_weight: float = DEFAULT_DISAGREE_WEIGHT,
    seed: int = 0,
    max_stop_corrections_per_episode: int | None = DEFAULT_MAX_STOP_CORRECTIONS_PER_EPISODE,
) -> tuple[list[CallRef], Counter]:
    """Labelled, unskipped calls with targets and weights; at most N per episode.

    Only the earliest ``max_stop_corrections_per_episode`` calls an episode got wrong
    by not stopping are kept (the rest are the same spin at the goal).  Over-long
    episodes then keep a seeded uniform subset (not, say, their disagreements first):
    a loop episode is mostly disagreements about the same place, which is exactly
    what the cap is for.
    """
    stats: Counter = Counter()
    by_episode: "OrderedDict[str, list[CallRef]]" = OrderedDict()
    for ref in refs:
        if ref.label is None:
            stats["unlabelled"] += 1
            continue
        if ref.label.skip:
            stats["skipped_by_label"] += 1
            continue
        by_episode.setdefault(ref.record_path, []).append(ref)
    selected: list[CallRef] = []
    for episode, calls in by_episode.items():
        if max_stop_corrections_per_episode is not None:
            kept, stops = [], 0
            for ref in sorted(calls, key=lambda r: r.call_index):
                corrected_to_stop = ref.label.agree is False and training_turns(ref.recorded, ref.label) == (STOP_TEXT,)
                if corrected_to_stop:
                    stops += 1
                    if stops > max_stop_corrections_per_episode:
                        stats["dropped_stop_corrections"] += 1
                        continue
                kept.append(ref)
            calls = kept
        if max_calls_per_episode is not None and len(calls) > max_calls_per_episode:
            stats["capped_episodes"] += 1
            stats["dropped_by_cap"] += len(calls) - max_calls_per_episode
            calls = sorted(calls, key=lambda r: _stable_rank(seed, episode, r.call_index))
            calls = sorted(calls[:max_calls_per_episode], key=lambda r: r.call_index)
        for ref in calls:
            if ref.label.agree and ref.label.target_turns not in (None, ref.recorded):
                # The labeller and this reader disagree about what the model said.
                raise S2SFTError(
                    f"{ref.key}: accepted target {ref.label.target_turns!r} is not the recorded "
                    f"answer {ref.recorded!r}"
                )
            call_turns = training_turns(ref.recorded, ref.label)
            agree = bool(ref.label.agree)
            weight = float(agree_weight if agree else disagree_weight)
            selected.append(dataclasses.replace(ref, turns=call_turns, weight=weight))
            stats["agree" if agree else "disagree"] += 1
            stats[f"target_{decision_class(call_turns[-1])}"] += 1
    stats["selected"] = len(selected)
    stats["episodes"] = len(by_episode)
    return selected, stats


# ---------------------------------------------------------------------------
# Inputs, exactly as served
# ---------------------------------------------------------------------------


def pil_from_jpeg(data: bytes, image_size: tuple[int, int] | None = None) -> Image.Image:
    """``rpc_model_server._pil_from_blob`` on raw bytes: same decoder, same resize."""
    # The server's decoder, not PIL's: a different JPEG decoder is a different image.
    from vla_rpc.core.image import decode_jpeg_to_rgb

    arr = decode_jpeg_to_rgb(data)
    image = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    if image_size is not None and image.size != image_size:
        image = image.resize(image_size)
    return image


def episode_vlm_image_size(
    record: EpisodeRecord, default: tuple[int, int] | None = DEFAULT_VLM_IMAGE_SIZE
) -> tuple[int, int]:
    size = record.episode.get("vlm_image_size") or default
    if size is None:
        raise S2SFTError(f"{record.path}: episode.json has no vlm_image_size")
    size = tuple(int(v) for v in size)
    if len(size) != 2:
        raise S2SFTError(f"{record.path}: vlm_image_size {size!r} is not (w, h)")
    return size  # type: ignore[return-value]


def inference_messages(
    record: EpisodeRecord,
    call: Mapping[str, Any],
    *,
    default_vlm_image_size: tuple[int, int] | None = DEFAULT_VLM_IMAGE_SIZE,
) -> tuple[list[dict], list[Image.Image], Image.Image]:
    """The first-turn messages and images the server built for this call, plus the lookdown."""
    instruction = record.episode.get("instruction")
    if not isinstance(instruction, str):
        raise S2SFTError(f"{record.path}: episode.json instruction must be the string sent, got {type(instruction).__name__}")
    size = episode_vlm_image_size(record, default_vlm_image_size)
    history = [pil_from_jpeg(record.front_jpeg(step), size) for step in call["history_step_ids"]]
    current = pil_from_jpeg(record.front_jpeg(call["step_id"]), size)
    lookdown = pil_from_jpeg(record.lookdown_jpeg(call["call_index"]))
    if lookdown.size != NATIVE_LOOKDOWN_SIZE:
        # The server refuses any other size for the native two-turn protocol.
        raise S2SFTError(
            f"{record.path} call {call['call_index']}: lookdown is {lookdown.size}, "
            f"the native protocol requires {NATIVE_LOOKDOWN_SIZE}"
        )
    messages, images = build_native_messages(instruction, history, current)
    return messages, images, lookdown


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def conversation_turns(
    messages: list[dict],
    images: list[Image.Image],
    lookdown: Image.Image,
    turns: Sequence[str],
) -> tuple[list[dict], list[list[dict]]]:
    """The full training conversation and, per assistant turn, the messages served before it."""
    turns = validate_turns(turns, what="target")
    prompts = [copy.deepcopy(messages)]
    if len(turns) == 2:
        second, _ = append_native_lookdown_turn(messages, images, turns[0], lookdown)
        prompts.append(second)
    full = copy.deepcopy(prompts[-1]) + [_assistant(turns[-1])]
    return full, prompts


class PromptPrefixMismatch(S2SFTError):
    """The training tokens in front of an assistant turn are not what the server sends."""


def encode_conversation(
    processor: Any,
    messages: list[dict],
    images: list[Image.Image],
    lookdown: Image.Image,
    turns: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Tokenise one conversation; labels on assistant content and ``<|im_end|>`` only.

    The served prompt of turn ``i`` is tokenised on its own and must be a prefix
    of the conversation, token for token; turn ``i``'s span starts where it ends
    and runs to the first ``<|im_end|>`` after it, and must decode to the target.
    """
    full, prompts = conversation_turns(messages, images, lookdown, turns)
    encoded = processor.apply_chat_template(
        full, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt"
    )
    input_ids = encoded["input_ids"][0]
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    im_end = processor.tokenizer.convert_tokens_to_ids(IM_END)
    turns = tuple(turns)
    for index, prompt in enumerate(prompts):
        served = processor.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )["input_ids"][0]
        start = int(served.shape[0])
        if start >= input_ids.shape[0] or not torch.equal(input_ids[:start], served):
            raise PromptPrefixMismatch(
                f"turn {index}: the {start} served prompt tokens are not a prefix of the "
                f"training conversation ({input_ids.shape[0]} tokens)"
            )
        ends = (input_ids[start:] == im_end).nonzero()
        if ends.numel() == 0:
            raise S2SFTError(f"turn {index}: no {IM_END} after the prompt")
        stop = start + int(ends[0, 0]) + 1
        content = processor.tokenizer.decode(input_ids[start : stop - 1], skip_special_tokens=True)
        if content != turns[index]:
            raise S2SFTError(f"turn {index}: supervised span decodes to {content!r}, not {turns[index]!r}")
        labels[start:stop] = input_ids[start:stop]
    sample = {
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"][0],
        "labels": labels,
    }
    for name in ("pixel_values", "image_grid_thw"):
        if name in encoded:
            sample[name] = encoded[name]
    return sample


# ---------------------------------------------------------------------------
# Dataset, sampler, collate
# ---------------------------------------------------------------------------


class RecordCache:
    """A few open records per process; opening one reads its whole tar index."""

    def __init__(self, size: int = 16):
        self.size = max(1, int(size))
        self._open: "OrderedDict[str, tuple[EpisodeRecord, dict[int, dict]]]" = OrderedDict()

    def get(self, path: str) -> tuple[EpisodeRecord, dict[int, dict]]:
        hit = self._open.get(path)
        if hit is not None:
            self._open.move_to_end(path)
            return hit
        record = EpisodeRecord(path)
        hit = (record, {int(c["call_index"]): c for c in record.calls})
        self._open[path] = hit
        while len(self._open) > self.size:
            self._open.popitem(last=False)
        return hit


class S2SFTDataset(Dataset):
    """Tokenised training samples for selected calls (``select_training_refs``)."""

    def __init__(
        self,
        refs: Sequence[CallRef],
        processor: Any,
        *,
        default_vlm_image_size: tuple[int, int] | None = DEFAULT_VLM_IMAGE_SIZE,
        record_cache_size: int = 16,
    ):
        missing = [ref.key for ref in refs if ref.turns is None]
        if missing:
            raise S2SFTError(f"{len(missing)} refs have no training target, e.g. {missing[:3]}")
        self.refs = list(refs)
        self.processor = processor
        self.default_vlm_image_size = default_vlm_image_size
        self._records = RecordCache(record_cache_size)

    def __len__(self) -> int:
        return len(self.refs)

    @property
    def weights(self) -> torch.Tensor:
        return torch.tensor([ref.weight for ref in self.refs], dtype=torch.double)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[int(index)]
        record, calls = self._records.get(ref.record_path)
        messages, images, lookdown = inference_messages(
            record, calls[ref.call_index], default_vlm_image_size=self.default_vlm_image_size
        )
        sample = encode_conversation(self.processor, messages, images, lookdown, ref.turns)
        sample["key"] = ref.key
        return sample


def collate_sft(batch: Sequence[Mapping[str, Any]], *, pad_token_id: int) -> dict[str, Any]:
    """Right-pad token rows; images of all rows are concatenated in row order."""
    length = max(int(item["input_ids"].shape[0]) for item in batch)
    input_ids = torch.full((len(batch), length), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), length), dtype=torch.long)
    labels = torch.full((len(batch), length), IGNORE_INDEX, dtype=torch.long)
    for row, item in enumerate(batch):
        n = int(item["input_ids"].shape[0])
        input_ids[row, :n] = item["input_ids"]
        attention_mask[row, :n] = item["attention_mask"]
        labels[row, :n] = item["labels"]
    out: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "keys": [item.get("key") for item in batch],
    }
    if any("pixel_values" in item for item in batch):
        out["pixel_values"] = torch.cat([item["pixel_values"] for item in batch], dim=0)
        out["image_grid_thw"] = torch.cat([item["image_grid_thw"] for item in batch], dim=0)
    return out


class WeightedDistributedSampler(Sampler[int]):
    """Weighted draws with replacement, one global stream per epoch, sharded by rank.

    Every rank draws the same ``num_samples`` indices from a generator seeded by
    (seed, epoch) and keeps positions ``rank, rank + world, ...``, so the ranks
    together consume the global stream in order and a resume can skip an exact
    number of per-rank draws (``start``).
    """

    def __init__(
        self,
        weights: Sequence[float] | torch.Tensor,
        num_samples: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if self.weights.numel() == 0 or bool((self.weights < 0).any()) or float(self.weights.sum()) <= 0:
            raise ValueError("weights must be non-negative with a positive sum")
        if num_samples % world_size:
            raise ValueError(f"num_samples {num_samples} is not a multiple of world size {world_size}")
        self.num_samples = int(num_samples)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0
        self.start = 0

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        self.epoch = int(epoch)
        self.start = int(start)

    def global_indices(self) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
        return torch.multinomial(self.weights, self.num_samples, replacement=True, generator=generator)

    def __iter__(self) -> Iterator[int]:
        mine = self.global_indices()[self.rank :: self.world_size].tolist()
        return iter(mine[self.start :])

    def __len__(self) -> int:
        return self.num_samples // self.world_size - self.start


def summarize_refs(refs: Sequence[CallRef]) -> dict[str, Any]:
    scenes = Counter(ref.scene_id for ref in refs)
    return {
        "calls": len(refs),
        "episodes": len({ref.record_path for ref in refs}),
        "scenes": len(scenes),
        "agree": sum(1 for ref in refs if ref.label is not None and ref.label.agree),
        "disagree": sum(1 for ref in refs if ref.label is not None and ref.label.agree is False),
        "two_turn_targets": sum(1 for ref in refs if ref.turns is not None and len(ref.turns) == 2),
        "weight_sum": float(sum(ref.weight for ref in refs)),
    }


__all__ = [
    "CallLabel",
    "CallRef",
    "DEFAULT_AGREE_WEIGHT",
    "DEFAULT_DISAGREE_WEIGHT",
    "DEFAULT_HOLDOUT_COUNT",
    "DEFAULT_MAX_CALLS_PER_EPISODE",
    "DEFAULT_VLM_IMAGE_SIZE",
    "HOLDOUT_SALT",
    "IGNORE_INDEX",
    "IM_END",
    "PromptPrefixMismatch",
    "R2R_TRAIN_SCENES",
    "RecordCache",
    "S2SFTDataset",
    "S2SFTError",
    "SFT_SCHEMA",
    "WeightedDistributedSampler",
    "collate_sft",
    "conversation_turns",
    "decision_class",
    "encode_conversation",
    "episode_vlm_image_size",
    "holdout_scenes",
    "index_calls",
    "inference_messages",
    "normalize_label",
    "pil_from_jpeg",
    "pixel_uv",
    "recorded_turns",
    "select_training_refs",
    "summarize_refs",
    "training_turns",
    "validate_turns",
]

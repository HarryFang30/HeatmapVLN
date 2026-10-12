import io
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.s2_records import LABEL_SCHEMA, EpisodeRecord, EpisodeRecordWriter, iter_record_paths, label_path_for
from src.data.s2_sft_dataset import (
    IGNORE_INDEX,
    R2R_TRAIN_SCENES,
    PromptPrefixMismatch,
    S2SFTDataset,
    S2SFTError,
    WeightedDistributedSampler,
    collate_sft,
    decision_class,
    encode_conversation,
    holdout_scenes,
    index_calls,
    inference_messages,
    normalize_label,
    recorded_turns,
    select_training_refs,
    validate_turns,
)
from src.models.heatmap.native_internnav_exact import build_native_messages

INSTRUCTION = "Walk past the sofa and stop at the door."


# ---------------------------------------------------------------------------
# A processor with the Qwen2.5-VL chat template, tokenised one character per id
# ---------------------------------------------------------------------------


class _QwenLikeTokenizer:
    SPECIAL = {"<|im_start|>": 1, "<|im_end|>": 2, "<|vision_start|>": 3, "<|vision_end|>": 4, "<|image_pad|>": 5}
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return self.SPECIAL[token]

    def encode(self, text):
        ids, i = [], 0
        while i < len(text):
            for token, token_id in self.SPECIAL.items():
                if text.startswith(token, i):
                    ids.append(token_id)
                    i += len(token)
                    break
            else:
                ids.append(100 + ord(text[i]))
                i += 1
        return ids

    def decode(self, ids, skip_special_tokens=True):
        inverse = {v: k for k, v in self.SPECIAL.items()}
        out = []
        for i in (int(x) for x in ids):
            if i in inverse:
                if not skip_special_tokens:
                    out.append(inverse[i])
            else:
                out.append(chr(i - 100))
        return "".join(out)


class _QwenLikeProcessor:
    """The released chat template (system turn, <|im_end|>\\n after every turn); 1 pad per 64x64 px."""

    def __init__(self, generation_prompt="<|im_start|>assistant\n"):
        self.tokenizer = _QwenLikeTokenizer()
        self.generation_prompt = generation_prompt
        self.seen_images = []

    def render(self, messages, add_generation_prompt):
        text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        images = []
        for message in messages:
            text += f"<|im_start|>{message['role']}\n"
            for item in message["content"]:
                if item["type"] == "image":
                    image = item["image"]
                    images.append(image)
                    pads = max(1, (image.size[0] * image.size[1]) // (64 * 64))
                    text += "<|vision_start|>" + "<|image_pad|>" * pads + "<|vision_end|>"
                else:
                    text += item["text"]
            text += "<|im_end|>\n"
        if add_generation_prompt:
            text += self.generation_prompt
        return text, images

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt"):
        assert tokenize and return_dict and return_tensors == "pt"
        text, images = self.render(messages, add_generation_prompt)
        self.seen_images.append(images)
        ids = torch.tensor([self.tokenizer.encode(text)], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.tensor(
                [[float(np.asarray(im).mean())] * 3 for im in images], dtype=torch.float32
            ),
            "image_grid_thw": torch.tensor([[1, im.size[1] // 14, im.size[0] // 14] for im in images]),
        }


# ---------------------------------------------------------------------------
# Synthetic records
# ---------------------------------------------------------------------------


def _jpeg(seed, size=(640, 480)):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size[1] // 16, size[0] // 16, 3), dtype=np.uint8)
    image = Image.fromarray(arr).resize(size, Image.BILINEAR)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _write_record(root, scene, episode, calls, *, vlm_image_size=(384, 384)):
    writer = EpisodeRecordWriter(
        root=root,
        scene_id=scene,
        episode_id=episode,
        episode={"instruction": INSTRUCTION, "split": "train", "vlm_image_size": list(vlm_image_size)},
    )
    steps = sorted({s for c in calls for s in c["history_step_ids"] + [c["step_id"]]})
    for step in steps:
        writer.add_step(step, [0.0, 0.0, -0.25 * step], [1.0, 0.0, 0.0, 0.0])
        writer.add_front(step, _jpeg(1000 * episode + step))
    for call in calls:
        writer.add_call(
            {"kind": "trajectory", "distance_to_goal": 5.0, **call},
            lookdown_jpeg=_jpeg(77 + call["call_index"]),
            lookdown_depth_m=np.full((480, 640), 1.5, dtype=np.float32),
        )
    return writer.commit()


def _write_labels(record_path, label_root, rows):
    path = label_path_for(record_path, label_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"schema": LABEL_SCHEMA, **r}) + "\n" for r in rows), encoding="utf-8")
    return path


def _calls(n, *, start_step=0):
    calls = []
    for i in range(n):
        step = start_step + 2 * i + 2
        history = [s for s in range(start_step, step, 2)][-3:]
        if i % 2 == 0:
            calls.append({"call_index": i, "step_id": step, "history_step_ids": history,
                          "llm_output": "245 320", "native_first_output": "↓", "native_lookdown_turns": 1})
        else:
            calls.append({"call_index": i, "step_id": step, "history_step_ids": history,
                          "llm_output": "←←", "native_first_output": "←←", "native_lookdown_turns": 0})
    return calls


# ---------------------------------------------------------------------------
# Scenes, decisions, labels
# ---------------------------------------------------------------------------


def test_holdout_is_six_stable_train_scenes_and_moves_with_the_salt():
    scenes = holdout_scenes()
    assert len(scenes) == 6 and list(scenes) == sorted(scenes)
    assert set(scenes) <= set(R2R_TRAIN_SCENES) and len(R2R_TRAIN_SCENES) == 61
    assert holdout_scenes() == scenes
    assert holdout_scenes(salt="another") != scenes
    with pytest.raises(ValueError):
        holdout_scenes(62)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("STOP", "stop"),
        ("", "stop"),  # the server's fallback_stop
        ("no arrows here", "stop"),
        ("←←", "turn_left"),
        ("→", "turn_right"),
        ("↑↑", "forward"),
        ("↓", "lookdown"),
        ("245 320", "pixel"),
        ("12", "invalid"),
    ],
)
def test_decision_class_reads_answers_like_the_server(text, expected):
    assert decision_class(text) == expected


def test_turns_must_be_a_shape_the_two_turn_protocol_produces():
    assert recorded_turns({"call_index": 0, "llm_output": "STOP", "native_lookdown_turns": 0}) == ("STOP",)
    assert recorded_turns(
        {"call_index": 0, "llm_output": "1 2", "native_first_output": "↓", "native_lookdown_turns": 1}
    ) == ("↓", "1 2")
    with pytest.raises(S2SFTError, match="no second turn"):
        validate_turns(["↓"], what="t")
    with pytest.raises(S2SFTError, match="not a lookdown request"):
        validate_turns(["←", "1 2"], what="t")
    with pytest.raises(S2SFTError):
        recorded_turns({"call_index": 3, "llm_output": "1 2", "native_lookdown_turns": 1})


def test_label_rows_normalise_to_oracle_turns():
    pixel = normalize_label({"call_index": 1, "agree": False, "oracle": {"kind": "pixel", "text": "200 300"}})
    assert pixel.oracle_turns == ("↓", "200 300") and pixel.agree is False
    flat = normalize_label({"call_index": 2, "agree": True, "oracle_kind": "turn", "oracle_text": "→→",
                            "oracle_path": [[0, 0, 0], [1, 0, 0]]})
    assert flat.oracle_turns == ("→→",) and flat.oracle_path == ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    stop = normalize_label({"call_index": 3, "agree": False, "oracle": {"kind": "STOP", "turns": ["STOP"]}})
    assert stop.oracle_kind == "stop" and stop.oracle_turns == ("STOP",)
    skip = normalize_label({"call_index": 4, "skip": True})
    assert skip.skip and skip.oracle_turns is None
    with pytest.raises(S2SFTError, match="not a turn"):
        normalize_label({"call_index": 5, "agree": False, "oracle": {"kind": "turn", "turns": ["STOP"]}})
    with pytest.raises(S2SFTError, match="agree"):
        normalize_label({"call_index": 6, "oracle": {"kind": "stop", "turns": ["STOP"]}})


def test_labeller_flat_rows_are_read():
    # The shape scripts/tools/label_s2_records.py writes: *_turn1/*_turn2 columns, agree may be null.
    accepted_without_oracle = normalize_label({
        "call_index": 0, "agree": True, "oracle_kind": "skip", "oracle_turn1": None, "oracle_turn2": None,
        "target_turn1": "STOP", "target_turn2": None, "distance_to_goal": 1.2,
    })
    assert not accepted_without_oracle.skip and accepted_without_oracle.target_turns == ("STOP",)
    assert accepted_without_oracle.oracle_turns is None and accepted_without_oracle.distance_to_goal == 1.2
    unjudgeable = normalize_label({"call_index": 1, "agree": None, "oracle_kind": "turn", "oracle_turn1": "→",
                                   "oracle_turn2": None, "target_turn1": None, "target_turn2": None})
    assert unjudgeable.skip and unjudgeable.oracle_turns == ("→",)
    corrected = normalize_label({"call_index": 2, "agree": False, "oracle_kind": "pixel", "oracle_turn1": "↓",
                                 "oracle_turn2": "200 300", "target_turn1": "↓", "target_turn2": "200 300",
                                 "distance_to_goal": float("inf")})
    assert corrected.oracle_turns == corrected.target_turns == ("↓", "200 300")
    assert corrected.distance_to_goal == float("inf")
    rejected_without_oracle = normalize_label({"call_index": 3, "agree": False, "oracle_kind": "skip",
                                               "target_turn1": None})
    assert rejected_without_oracle.skip


def test_an_accepted_target_must_be_the_recorded_answer(tmp_path):
    scene = R2R_TRAIN_SCENES[2]
    path = _write_record(tmp_path / "r", scene, 1, _calls(2))
    _write_labels(path, tmp_path / "l", [
        {"call_index": 0, "agree": True, "oracle_kind": "skip", "target_turn1": "↓", "target_turn2": "245 320"},
        {"call_index": 1, "agree": True, "oracle_kind": "skip", "target_turn1": "→", "target_turn2": None},
    ])
    refs, _ = index_calls([tmp_path / "r"], [tmp_path / "l"], split="all", holdout=())
    with pytest.raises(S2SFTError, match="not the recorded answer"):
        select_training_refs(refs)
    selected, _ = select_training_refs(refs[:1])
    assert selected[0].turns == ("↓", "245 320")


# ---------------------------------------------------------------------------
# Index and selection
# ---------------------------------------------------------------------------


def test_index_splits_by_scene_and_targets_follow_the_agreement_rule(tmp_path):
    held = holdout_scenes()[0]
    trained = next(s for s in R2R_TRAIN_SCENES if s not in holdout_scenes())
    records, labels = tmp_path / "records", tmp_path / "labels"
    a = _write_record(records, trained, 1, _calls(4))
    _write_labels(a, labels, [
        {"call_index": 0, "agree": True, "oracle": {"kind": "pixel", "text": "100 100"}},
        {"call_index": 1, "agree": False, "oracle": {"kind": "turn", "turns": ["→→"]}},
        {"call_index": 2, "skip": True},
        # call 3 has no label row
    ])
    b = _write_record(records, held, 2, _calls(2))
    _write_labels(b, labels, [{"call_index": 0, "agree": True, "oracle": {"kind": "stop", "turns": ["STOP"]}},
                              {"call_index": 1, "agree": True, "oracle": {"kind": "stop", "turns": ["STOP"]}}])
    _write_record(records, trained, 3, _calls(1))  # no labels file at all

    train, stats = index_calls([records], [labels], split="train", holdout=holdout_scenes())
    assert {r.scene_id for r in train} == {trained}
    assert stats["calls_without_label"] == 2 and stats["episodes_without_labels"] == 1
    assert stats["episodes_other_split"] == 1
    selected, sel = select_training_refs(train, seed=0)
    by_call = {r.call_index: r for r in selected}
    assert set(by_call) == {0, 1} and sel["skipped_by_label"] == 1
    assert by_call[0].turns == ("↓", "245 320") and by_call[0].weight == 1.0  # the model's own answer
    assert by_call[1].turns == ("→→",) and by_call[1].weight == 3.0  # the oracle's

    held_refs, _ = index_calls([records], [labels], split="holdout", holdout=holdout_scenes())
    assert {r.scene_id for r in held_refs} == {held} and len(held_refs) == 2
    unlabelled, _ = index_calls([records], None, split="all", holdout=(), require_labels=False)
    assert len(unlabelled) == 4 + 2 + 1 and all(r.label is None for r in unlabelled)


def test_episode_cap_keeps_a_seeded_subset(tmp_path):
    scene = R2R_TRAIN_SCENES[0]
    path = _write_record(tmp_path / "r", scene, 1, _calls(10))
    _write_labels(path, tmp_path / "l", [
        {"call_index": i, "agree": True, "oracle": {"kind": "turn", "turns": ["←"]}} for i in range(10)
    ])
    refs, _ = index_calls([tmp_path / "r"], [tmp_path / "l"], split="all", holdout=())
    first, stats = select_training_refs(refs, max_calls_per_episode=4, seed=1)
    again, _ = select_training_refs(refs, max_calls_per_episode=4, seed=1)
    other, _ = select_training_refs(refs, max_calls_per_episode=4, seed=2)
    assert len(first) == 4 and stats["dropped_by_cap"] == 6 and stats["capped_episodes"] == 1
    assert [r.call_index for r in first] == [r.call_index for r in again] == sorted(r.call_index for r in first)
    assert [r.call_index for r in first] != [r.call_index for r in other]


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _messages():
    history = [Image.new("RGB", (384, 384), (i, i, i)) for i in (10, 20)]
    current = Image.new("RGB", (384, 384), (30, 30, 30))
    messages, images = build_native_messages(INSTRUCTION, history, current)
    return messages, images, Image.new("RGB", (640, 480), (40, 40, 40))


def _supervised_text(processor, sample):
    labels = sample["labels"]
    spans, current = [], []
    for token, label in zip(sample["input_ids"].tolist(), labels.tolist()):
        if label != IGNORE_INDEX:
            assert label == token
            current.append(token)
        elif current:
            spans.append(current)
            current = []
    if current:
        spans.append(current)
    return [processor.tokenizer.decode(span, skip_special_tokens=False) for span in spans]


@pytest.mark.parametrize("turns", [("STOP",), ("←←",), ("↓", "245 320"), ("",)])
def test_only_assistant_content_and_im_end_are_supervised(turns):
    processor = _QwenLikeProcessor()
    messages, images, lookdown = _messages()
    sample = encode_conversation(processor, messages, images, lookdown, turns)
    assert _supervised_text(processor, sample) == [t + "<|im_end|>" for t in turns]
    # The served prompt of the last turn is exactly the unsupervised prefix before it.
    served = processor.render(messages if len(turns) == 1 else _turn2_messages(messages, images, lookdown, turns), True)[0]
    assert processor.tokenizer.decode(sample["input_ids"][: len(processor.tokenizer.encode(served))], False) == served
    assert sample["image_grid_thw"].shape[0] == len(images) + (len(turns) == 2)


def _turn2_messages(messages, images, lookdown, turns):
    from src.models.heatmap.native_internnav_exact import append_native_lookdown_turn

    return append_native_lookdown_turn(messages, images, turns[0], lookdown)[0]


def test_a_training_prompt_that_is_not_the_served_prompt_is_refused():
    # EXP-14's failure mode: the training side renders something the server never sends.
    processor = _QwenLikeProcessor(generation_prompt="<|im_start|>assistant\n\n")
    messages, images, lookdown = _messages()
    with pytest.raises(PromptPrefixMismatch, match="turn 0"):
        encode_conversation(processor, messages, images, lookdown, ("STOP",))
    with pytest.raises(PromptPrefixMismatch, match="turn 1"):
        encode_conversation(_SecondTurnDrift(), messages, images, lookdown, ("↓", "245 320"))


class _SecondTurnDrift(_QwenLikeProcessor):
    """Renders the lookdown turn's text differently only when it is the served prompt."""

    def render(self, messages, add_generation_prompt):
        text, images = super().render(messages, add_generation_prompt)
        if add_generation_prompt and len(messages) > 1:
            text = text.replace("you can see", "you can see ")
        return text, images


def test_dataset_feeds_the_server_image_path_and_prompt(tmp_path):
    pytest.importorskip("vla_rpc.core.image")
    server = pytest.importorskip("scripts.evaluation.rpc_model_server")
    scene = R2R_TRAIN_SCENES[1]
    path = _write_record(tmp_path / "r", scene, 4, _calls(2))
    _write_labels(path, tmp_path / "l", [
        {"call_index": 0, "agree": True, "oracle": {"kind": "stop", "turns": ["STOP"]}},
        {"call_index": 1, "agree": False, "oracle": {"kind": "stop", "turns": ["STOP"]}},
    ])
    refs, _ = index_calls([tmp_path / "r"], [tmp_path / "l"], split="all", holdout=())
    selected, _ = select_training_refs(refs)
    processor = _QwenLikeProcessor()
    dataset = S2SFTDataset(selected, processor)
    sample = dataset[0]
    assert sample["key"] == f"{scene}_000004#0"
    images = processor.seen_images[0]  # the full conversation of call 0 (two turns)
    record = EpisodeRecord(path)
    call = record.calls[0]
    expected = [server._pil_from_blob(SimpleNamespace(data=record.front_jpeg(s)), (384, 384))
                for s in call["history_step_ids"] + [call["step_id"]]]
    expected.append(server._pil_from_blob(SimpleNamespace(data=record.lookdown_jpeg(0))))
    assert [im.size for im in images] == [(384, 384)] * (len(expected) - 1) + [(640, 480)]
    for got, want in zip(images, expected):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    messages, _, _ = inference_messages(record, call)
    assert messages[0]["content"][0]["text"].startswith("You are an autonomous navigation assistant. Your task is to Walk past")
    assert dataset[1]["labels"].ne(IGNORE_INDEX).sum() == len("STOP") + 1


def test_lookdown_of_another_size_is_refused(tmp_path):
    pytest.importorskip("vla_rpc.core.image")
    writer = EpisodeRecordWriter(root=tmp_path, scene_id="s", episode_id=1,
                                 episode={"instruction": INSTRUCTION, "vlm_image_size": [384, 384]})
    writer.add_step(0, [0, 0, 0], [1, 0, 0, 0])
    writer.add_front(0, _jpeg(1))
    writer.add_call({"call_index": 0, "step_id": 0, "history_step_ids": [], "llm_output": "STOP", "kind": "stop"},
                    lookdown_jpeg=_jpeg(2, (384, 384)), lookdown_depth_m=None)
    record = EpisodeRecord(writer.commit())
    with pytest.raises(S2SFTError, match="lookdown"):
        inference_messages(record, record.calls[0])


def test_collate_right_pads_and_keeps_image_order():
    a = {"input_ids": torch.tensor([5, 6, 7]), "attention_mask": torch.ones(3, dtype=torch.long),
         "labels": torch.tensor([-100, 6, 7]), "pixel_values": torch.zeros(2, 3), "image_grid_thw": torch.tensor([[1, 1, 2]]), "key": "a"}
    b = {"input_ids": torch.tensor([8]), "attention_mask": torch.ones(1, dtype=torch.long),
         "labels": torch.tensor([8]), "pixel_values": torch.ones(4, 3), "image_grid_thw": torch.tensor([[1, 2, 2]]), "key": "b"}
    out = collate_sft([a, b], pad_token_id=0)
    assert out["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert out["labels"].tolist() == [[-100, 6, 7], [8, -100, -100]]
    assert out["pixel_values"].shape == (6, 3) and out["image_grid_thw"].tolist() == [[1, 1, 2], [1, 2, 2]]
    assert out["keys"] == ["a", "b"]


def test_weighted_sampler_shards_one_stream_and_resumes_exactly():
    weights = [1.0] * 50 + [3.0] * 50
    ranks = [WeightedDistributedSampler(weights, 4000, rank=r, world_size=2, seed=7) for r in range(2)]
    for sampler in ranks:
        sampler.set_epoch(3)
    stream = ranks[0].global_indices().tolist()
    assert list(ranks[0]) == stream[0::2] and list(ranks[1]) == stream[1::2]
    heavy = sum(1 for i in stream if i >= 50) / len(stream)
    assert 0.70 < heavy < 0.80  # 3:1 odds
    ranks[0].set_epoch(3, start=10)
    assert list(ranks[0]) == stream[0::2][10:] and len(ranks[0]) == 2000 - 10
    ranks[0].set_epoch(4)
    assert ranks[0].global_indices().tolist() != stream


# ---------------------------------------------------------------------------
# The released processor (needs the model directory; runs on the 910B)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("HEATMAPVLN_INTERNNAV_MODEL_PATH"),
    reason="set HEATMAPVLN_INTERNNAV_MODEL_PATH to the InternNav_Model directory",
)
@pytest.mark.parametrize("turns", [("STOP",), ("→→",), ("↓", "245 320")])
def test_released_processor_prefix_and_labels(turns):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(os.environ["HEATMAPVLN_INTERNNAV_MODEL_PATH"], trust_remote_code=True)
    messages, images, lookdown = _messages()
    sample = encode_conversation(processor, messages, images, lookdown, turns)
    supervised = sample["input_ids"][sample["labels"] != IGNORE_INDEX]
    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert int((supervised == im_end).sum()) == len(turns)
    text = processor.tokenizer.decode(supervised, skip_special_tokens=False)
    assert text == "".join(t + "<|im_end|>" for t in turns)
    assert sample["image_grid_thw"].shape[0] == len(images) + (len(turns) == 2)


def test_only_the_earliest_stop_corrections_of_an_episode_are_kept(tmp_path):
    # An agent spinning at the goal is corrected to STOP on every call; keep the first few.
    calls = _calls(10)
    path = _write_record(tmp_path / "rec", "1LXtFkjw3qL", 0, calls)
    rows = [{"call_index": c["call_index"], "agree": False, "oracle_kind": "stop", "oracle_turn1": "STOP",
             "oracle_turn2": None, "distance_to_goal": 1.0} for c in calls[:8]]
    rows += [{"call_index": 8, "agree": True, "oracle_kind": "pixel", "oracle_turn1": "↓", "oracle_turn2": "245 320",
              "distance_to_goal": 5.0},
             {"call_index": 9, "agree": False, "oracle_kind": "turn", "oracle_turn1": "→", "oracle_turn2": None,
              "distance_to_goal": 5.0}]
    _write_labels(path, tmp_path / "lab", rows)
    refs, _ = index_calls([tmp_path / "rec"], [tmp_path / "lab"], split="train", holdout=())
    selected, stats = select_training_refs(refs, max_stop_corrections_per_episode=3)
    assert [r.call_index for r in selected] == [0, 1, 2, 8, 9]
    assert stats["dropped_stop_corrections"] == 5
    everything, _ = select_training_refs(refs, max_stop_corrections_per_episode=None)
    assert len(everything) == 10

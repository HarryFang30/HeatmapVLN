import torch

from src.data.panoramic_tokenized_collator import (
    IGNORE_INDEX,
    PanoramicTokenizedCollator,
)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "left"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) + 3 for ch in text]


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(
        self,
        messages_batch,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        truncation=False,
        max_length=None,
    ):
        del tokenize, return_dict, return_tensors, padding, truncation, max_length
        rows = []
        for messages in messages_batch:
            row = []
            for message in messages:
                text = ""
                text += f"<{message['role']}>"
                for item in message["content"]:
                    if item["type"] == "text":
                        text += item["text"]
                    else:
                        text += f"<{item['type']}>"
                row.extend(self.tokenizer.encode(text, add_special_tokens=False))
                if message["role"] == "assistant":
                    row.append(self.tokenizer.eos_token_id)
            if add_generation_prompt:
                row.extend(self.tokenizer.encode("<assistant>", add_special_tokens=False))
            rows.append(row)

        max_len = max(len(row) for row in rows)
        input_ids = []
        attention_mask = []
        for row in rows:
            pad_len = max_len - len(row)
            input_ids.append([self.tokenizer.pad_token_id] * pad_len + row)
            attention_mask.append([0] * pad_len + [1] * len(row))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def _sample(pixel_goal=None, discrete_action=1, is_stop=0.0, pano_view_id=None, pano_pixel_goal=None, pano_sample_kind=None):
    sample = {
        "history_frames": torch.zeros(1, 3, 2, 2),
        "current_frame": torch.zeros(3, 2, 2),
        "heatmap": torch.zeros(1, 2, 2),
        "action": torch.zeros(2),
        "action_valid": 1.0,
        "discrete_action": discrete_action,
        "is_stop": is_stop,
        "text": "go forward",
        "current_views": torch.zeros(4, 3, 2, 2),
        "history_panoramas": torch.zeros(0, 4, 3, 2, 2),
        "lookdown_frame": torch.zeros(3, 2, 2),
    }
    if pixel_goal is not None:
        sample["pixel_goal"] = pixel_goal
    if pano_view_id is not None:
        sample["pano_view_id"] = pano_view_id
    if pano_pixel_goal is not None:
        sample["pano_pixel_goal"] = pano_pixel_goal
    if pano_sample_kind is not None:
        sample["pano_sample_kind"] = pano_sample_kind
    return sample


def test_panoramic_sft_collator_builds_assistant_labels():
    collator = PanoramicTokenizedCollator(_FakeProcessor(), sft_mode=True)

    batch = [
        _sample(pixel_goal=[12, 34]),
        _sample(discrete_action=0, is_stop=1.0),
    ]
    out = collator(batch)

    labels = out["pano_inputs"]["labels"]
    assert out["sft_target_text"] == [["12 34"], ["STOP"]]
    assert labels.shape == out["pano_inputs"]["input_ids"].shape
    assert torch.any(labels[0] != IGNORE_INDEX)
    assert torch.any(labels[1] != IGNORE_INDEX)

    tokenizer = collator.processor.tokenizer
    row0_targets = labels[0][labels[0] != IGNORE_INDEX].tolist()
    row1_targets = labels[1][labels[1] != IGNORE_INDEX].tolist()
    assert row0_targets == [
        *tokenizer.encode("12 34", add_special_tokens=False),
        tokenizer.eos_token_id,
    ]
    assert row1_targets == [
        *tokenizer.encode("STOP", add_special_tokens=False),
        tokenizer.eos_token_id,
    ]


def test_panoramic_sft_collator_labels_turns_and_skips_forward_by_default():
    collator = PanoramicTokenizedCollator(_FakeProcessor(), sft_mode=True)
    out = collator([
        _sample(discrete_action=2),
        _sample(discrete_action=3),
        _sample(discrete_action=5),
    ])
    assert out["sft_target_text"] == [["←"], ["→"], ["↓"]]
    assert torch.any(out["pano_inputs"]["labels"] != IGNORE_INDEX)


def test_panoramic_sft_collator_can_label_forward_when_enabled():
    collator = PanoramicTokenizedCollator(
        _FakeProcessor(),
        sft_mode=True,
        sft_include_forward=True,
    )
    out = collator([_sample(discrete_action=1)])
    assert out["sft_target_text"] == [["↑"]]
    assert torch.any(out["pano_inputs"]["labels"] != IGNORE_INDEX)


def test_panoramic_sft_collator_internnav_protocol_labels_down_then_coord():
    collator = PanoramicTokenizedCollator(
        _FakeProcessor(),
        sft_mode=True,
        sft_protocol="internnav",
    )

    out = collator([_sample(pixel_goal=[12, 34])])
    assert out["sft_target_text"] == [["↓", "12 34"]]

    tokenizer = collator.processor.tokenizer
    targets = out["pano_inputs"]["labels"][0]
    targets = targets[targets != IGNORE_INDEX].tolist()
    assert targets == (
        [
            *tokenizer.encode("↓", add_special_tokens=False),
            tokenizer.eos_token_id,
            *tokenizer.encode("12 34", add_special_tokens=False),
            tokenizer.eos_token_id,
        ]
    )


def test_panoramic_sft_collator_structured_pano_pixel_goal():
    collator = PanoramicTokenizedCollator(_FakeProcessor(), sft_mode=True)
    sample = _sample(
        pixel_goal=[128, 192],
        pano_view_id="front",
        pano_pixel_goal=[128, 192],
        pano_sample_kind="pixel",
    )
    out = collator([sample])
    assert out["sft_target_text"] == [["view: front\npixel: 128 192"]]

    tokenizer = collator.processor.tokenizer
    targets = out["pano_inputs"]["labels"][0]
    targets = targets[targets != IGNORE_INDEX].tolist()
    assert targets == [
        *tokenizer.encode("view: front\npixel: 128 192", add_special_tokens=False),
        tokenizer.eos_token_id,
    ]


def test_panoramic_sft_collator_structured_side_goal_without_legacy_pixel_goal():
    collator = PanoramicTokenizedCollator(_FakeProcessor(), sft_mode=True)
    sample = _sample(
        pixel_goal=None,
        pano_view_id="right",
        pano_pixel_goal=[211, 128],
        pano_sample_kind="pixel",
    )
    out = collator([sample])
    assert out["sft_target_text"] == [["view: right\npixel: 211 128"]]


def test_panoramic_sft_collator_structured_pano_stop_and_turn():
    collator = PanoramicTokenizedCollator(_FakeProcessor(), sft_mode=True)
    out = collator([
        _sample(discrete_action=0, is_stop=1.0, pano_view_id="view_stop", pano_sample_kind="stop"),
        _sample(discrete_action=2, pano_view_id="view_turn", pano_sample_kind="turn"),
    ])
    assert out["sft_target_text"] == [["view: stop"], ["view: turn"]]

import torch

from src.data.panoramic_tokenized_collator import (
    IGNORE_INDEX,
    PanoramicTokenizedCollator,
)


class _FakeTokenizer:
    pad_token_id = 0
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
    ):
        del tokenize, return_dict, return_tensors, padding
        rows = []
        for messages in messages_batch:
            text = ""
            for message in messages:
                text += f"<{message['role']}>"
                for item in message["content"]:
                    if item["type"] == "text":
                        text += item["text"]
                    else:
                        text += f"<{item['type']}>"
            if add_generation_prompt:
                text += "<assistant>"
            rows.append(self.tokenizer.encode(text, add_special_tokens=False))

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


def _sample(pixel_goal=None, discrete_action=1, is_stop=0.0):
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
    }
    if pixel_goal is not None:
        sample["pixel_goal"] = pixel_goal
    return sample


def test_panoramic_sft_collator_builds_assistant_labels():
    collator = PanoramicTokenizedCollator(_FakeProcessor(), sft_mode=True)

    batch = [
        _sample(pixel_goal=[12, 34]),
        _sample(discrete_action=0, is_stop=1.0),
    ]
    out = collator(batch)

    labels = out["pano_inputs"]["labels"]
    assert out["sft_target_text"] == ["12 34", "STOP"]
    assert labels.shape == out["pano_inputs"]["input_ids"].shape
    assert torch.any(labels[0] != IGNORE_INDEX)
    assert torch.any(labels[1] != IGNORE_INDEX)

    tokenizer = collator.processor.tokenizer
    row0_targets = labels[0][labels[0] != IGNORE_INDEX].tolist()
    row1_targets = labels[1][labels[1] != IGNORE_INDEX].tolist()
    assert row0_targets == tokenizer.encode("12 34", add_special_tokens=False)
    assert row1_targets == tokenizer.encode("STOP", add_special_tokens=False)


def test_panoramic_sft_collator_labels_turn_and_forward_fallbacks():
    collator = PanoramicTokenizedCollator(_FakeProcessor(), sft_mode=True)
    out = collator([
        _sample(discrete_action=2),
        _sample(discrete_action=3),
        _sample(discrete_action=1),
    ])
    assert out["sft_target_text"] == ["←", "→", "↑"]
    assert torch.any(out["pano_inputs"]["labels"] != IGNORE_INDEX)

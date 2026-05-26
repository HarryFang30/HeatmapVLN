#!/usr/bin/env python3
"""Verify Stage1-S2 structured pano SFT label format before full training.

Modes:
  synthetic  - instant collator smoke (no dataset / no GPU)
  dataset    - sample real training items via config (slow init, no GPU)

Example:
  python scripts/evaluation/verify_structured_pano_sft_format.py --mode synthetic

  python scripts/evaluation/verify_structured_pano_sft_format.py \\
    --mode dataset \\
    --config configs/train_config_internnav_8gpu_stage2_wider.yaml \\
    --num-samples 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.data.panoramic_tokenized_collator import PanoramicTokenizedCollator
from src.models.heatmap.input_constructor import (
    STRUCTURED_PANO_OUTPUT_SUFFIX,
    construct_input,
    format_structured_pano_assistant_text,
)


class _PrintProcessor:
    """Minimal processor that prints assistant targets instead of tokenizing images."""

    class _Tok:
        pad_token_id = 0
        eos_token_id = 2
        padding_side = "left"

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [ord(c) for c in text]

    def __init__(self):
        self.tokenizer = self._Tok()

    def apply_chat_template(self, messages_batch, **kwargs):
        del kwargs
        rows = []
        for messages in messages_batch:
            row = []
            for message in messages:
                for item in message["content"]:
                    if item["type"] == "text":
                        row.extend(self.tokenizer.encode(item["text"]))
            rows.append(row)
        max_len = max(len(r) for r in rows)
        input_ids = []
        attention_mask = []
        for row in rows:
            pad = max_len - len(row)
            input_ids.append([0] * pad + row)
            attention_mask.append([0] * pad + [1] * len(row))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def _base_sample(**overrides):
    sample = {
        "history_frames": torch.zeros(2, 3, 8, 8),
        "current_frame": torch.zeros(3, 8, 8),
        "heatmap": torch.zeros(2, 4, 4, 4),
        "action": torch.zeros(2),
        "action_valid": 1.0,
        "discrete_action": 1,
        "is_stop": 0.0,
        "text": "Walk to the kitchen.",
        "current_views": torch.zeros(4, 3, 8, 8),
        "history_panoramas": torch.zeros(1, 4, 3, 8, 8),
        "lookdown_frame": torch.zeros(3, 8, 8),
    }
    sample.update(overrides)
    return sample


def _print_case(title: str, sample: dict, collator: PanoramicTokenizedCollator) -> dict:
    out = collator([sample])
    target = out["sft_target_text"][0]
    labels = out["pano_inputs"]["labels"][0]
    n_labeled = int((labels != -100).sum().item())

    print(f"\n=== {title} ===")
    print(f"  pano_view_id      : {sample.get('pano_view_id')}")
    print(f"  pano_pixel_goal   : {sample.get('pano_pixel_goal')}")
    print(f"  pano_sample_kind  : {sample.get('pano_sample_kind')}")
    print(f"  legacy pixel_goal : {sample.get('pixel_goal')}")
    print(f"  sft_target_text   : {target!r}")
    print(f"  labeled tokens    : {n_labeled}")

    # Also show prompt tail + assistant from construct_input directly.
    views = {n: sample["current_views"][i] for i, n in enumerate(("front", "right", "back", "left"))}
    hist = [{
        n: sample["history_panoramas"][0, i]
        for i, n in enumerate(("front", "right", "back", "left"))
    }]
    pg = sample.get("pano_pixel_goal") or sample.get("pixel_goal")
    use_structured = sample.get("pano_view_id") is not None or sample.get("pano_sample_kind") is not None
    assistant = target[-1] if target else None
    messages = construct_input(
        current_views=views,
        history_panoramas=hist,
        instruction=sample.get("text"),
        pixel_goal=pg,
        assistant_text=assistant,
        lookdown_frame=sample.get("lookdown_frame"),
        internnav_protocol=collator.sft_protocol == "internnav",
        structured_pano_output=use_structured,
    )
    user_tail = ""
    for item in messages[0]["content"]:
        if item["type"] == "text" and "view:" in item["text"] or "Output the next waypoint" in item["text"]:
            user_tail = item["text"]
    assistant_msg = messages[-1]["content"][0]["text"] if len(messages) > 1 else None
    print(f"  user suffix used  : {'STRUCTURED' if use_structured else 'LEGACY'}")
    if user_tail:
        print(f"  user suffix text  : {user_tail[:120]}...")
    print(f"  assistant in prompt: {assistant_msg!r}")
    print(f"  num chat turns    : {len(messages)}")

    return {
        "title": title,
        "sft_target_text": target,
        "assistant_in_prompt": assistant_msg,
        "structured": use_structured,
        "num_turns": len(messages),
    }


def run_synthetic(protocol: str) -> None:
    collator = PanoramicTokenizedCollator(
        _PrintProcessor(),
        sft_mode=True,
        sft_protocol=protocol,
        structured_pano_output=True,
    )
    print(f"protocol={protocol}, structured_pano_output=True")
    print(f"expected suffix snippet: {STRUCTURED_PANO_OUTPUT_SUFFIX[:80]}...")

    cases = [
        ("pixel/front", _base_sample(
            pixel_goal=[128, 192],
            pano_view_id="front",
            pano_pixel_goal=[128, 192],
            pano_sample_kind="pixel",
        )),
        ("pixel/right", _base_sample(
            pixel_goal=None,
            pano_view_id="right",
            pano_pixel_goal=[211, 128],
            pano_sample_kind="pixel",
        )),
        ("stop", _base_sample(
            discrete_action=0,
            is_stop=1.0,
            pano_view_id="view_stop",
            pano_sample_kind="stop",
        )),
        ("turn", _base_sample(
            discrete_action=2,
            pano_view_id="view_turn",
            pano_sample_kind="turn",
        )),
    ]
    results = [_print_case(name, sample, collator) for name, sample in cases]

    print("\n=== synthetic summary ===")
    for rec in results:
        ok = (
            rec["structured"]
            and rec["num_turns"] == 2
            and rec["sft_target_text"]
            and rec["assistant_in_prompt"] == rec["sft_target_text"][0]
        )
        print(f"  {rec['title']:14s} turns={rec['num_turns']} target={rec['sft_target_text']!r} OK={ok}")


def run_dataset(config_path: str, num_samples: int, seed: int) -> None:
    from scripts.training.utils import load_config
    from src.data.factory import build_dataset
    from transformers import AutoProcessor

    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    traj_cfg = data_cfg.get("trajectory", data_cfg.get("sliding_window", {}))
    stage_cfg = cfg.get("training", {}).get("stage2", cfg.get("stage2", {}))

    print("=== config check ===")
    print(f"  panoramic_vlm_input       : {traj_cfg.get('panoramic_vlm_input')}")
    print(f"  compute_pixel_goal        : {traj_cfg.get('compute_pixel_goal')}")
    print(f"  compute_pano_view_pixel_goal: {traj_cfg.get('compute_pano_view_pixel_goal', 'auto')}")
    print(f"  system2_sft_protocol      : {traj_cfg.get('system2_sft_protocol')}")
    print(f"  train_system2_sft        : {stage_cfg.get('train_system2_sft')}")

    llm_path = cfg["model"]["llm"]["model_path"]
    processor = AutoProcessor.from_pretrained(llm_path, trust_remote_code=True)
    protocol = str(
        stage_cfg.get("system2_sft_protocol", traj_cfg.get("system2_sft_protocol", "direct"))
    ).lower()
    collator = PanoramicTokenizedCollator(
        processor,
        n_traj_query=0,
        sft_mode=True,
        sft_protocol=protocol,
        structured_pano_output=True,
    )

    print("\n=== building train dataset (may take a few minutes) ===")
    dataset = build_dataset(cfg, split="train")
    print(f"  dataset len = {len(dataset)}")
    print(f"  compute_pano_view_pixel_goal = {getattr(dataset, 'compute_pano_view_pixel_goal', 'N/A')}")

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[:num_samples]

    kind_counter: Counter[str] = Counter()
    view_counter: Counter[str] = Counter()

    for rank, idx in enumerate(indices):
        sample = dataset[idx]
        batch = collator([sample])
        target = batch["sft_target_text"][0]
        kind = sample.get("pano_sample_kind", "?")
        view = sample.get("pano_view_id", "?")
        kind_counter[kind] += 1
        view_counter[str(view)] += 1

        print(f"\n--- dataset sample #{rank} (idx={idx}) ---")
        print(f"  instruction       : {str(sample.get('text', ''))[:80]}")
        print(f"  pano_sample_kind  : {kind}")
        print(f"  pano_view_id      : {view}")
        print(f"  pano_pixel_goal   : {sample.get('pano_pixel_goal')}")
        print(f"  legacy pixel_goal : {sample.get('pixel_goal')}")
        print(f"  sft_target_text   : {target!r}")

    summary = {
        "num_samples": num_samples,
        "pano_sample_kind_counts": dict(kind_counter),
        "pano_view_id_counts": dict(view_counter),
    }
    print("\n=== dataset summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser(description="Verify structured pano Stage1-S2 SFT format")
    p.add_argument("--mode", choices=["synthetic", "dataset"], default="synthetic")
    p.add_argument("--config", default="configs/train_config_internnav_8gpu_stage2_wider.yaml")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--protocol", default="internnav", choices=["direct", "internnav"])
    args = p.parse_args()

    if args.mode == "synthetic":
        run_synthetic(args.protocol)
    else:
        run_dataset(args.config, args.num_samples, args.seed)


if __name__ == "__main__":
    main()

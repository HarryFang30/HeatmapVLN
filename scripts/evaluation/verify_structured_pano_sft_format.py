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
import os
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
    meta = {
        "pano_view_id": sample.get("pano_view_id"),
        "pano_pixel_goal": sample.get("pano_pixel_goal"),
        "pano_sample_kind": sample.get("pano_sample_kind"),
        "pixel_goal": sample.get("pixel_goal"),
        "text": sample.get("text"),
        "lookdown_frame": sample.get("lookdown_frame"),
    }
    views = {
        n: sample["current_views"][i]
        for i, n in enumerate(("front", "right", "back", "left"))
    }
    hist = [{
        n: sample["history_panoramas"][0, i]
        for i, n in enumerate(("front", "right", "back", "left"))
    }]
    pg = meta["pano_pixel_goal"] or meta["pixel_goal"]
    use_structured = (
        meta["pano_view_id"] is not None or meta["pano_sample_kind"] is not None
    )

    out = collator([sample])
    target = out["sft_target_text"][0]
    labels = out["pano_inputs"]["labels"][0]
    n_labeled = int((labels != -100).sum().item())

    print(f"\n=== {title} ===")
    print(f"  pano_view_id      : {meta['pano_view_id']}")
    print(f"  pano_pixel_goal   : {meta['pano_pixel_goal']}")
    print(f"  pano_sample_kind  : {meta['pano_sample_kind']}")
    print(f"  legacy pixel_goal : {meta['pixel_goal']}")
    print(f"  sft_target_text   : {target!r}")
    print(f"  labeled tokens    : {n_labeled}")

    assistant = target[-1] if target else None
    messages = construct_input(
        current_views=views,
        history_panoramas=hist,
        instruction=meta["text"],
        pixel_goal=pg,
        assistant_text=assistant,
        lookdown_frame=meta["lookdown_frame"],
        internnav_protocol=collator.sft_protocol == "internnav",
        structured_pano_output=use_structured,
    )
    user_tail = ""
    for item in messages[0]["content"]:
        if item["type"] == "text" and (
            "view:" in item["text"] or "Output the next waypoint" in item["text"]
        ):
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


def _resolve_model_path(cfg: dict) -> str:
    raw = (
        os.environ.get("INTERNNAV_MODEL_PATH")
        or os.environ.get("INTERNNAV_BACKBONE")
        or cfg.get("paths", {}).get("internnav_model_path", "")
        or cfg.get("model", {})
        .get("action_head", {})
        .get("nextdit", {})
        .get("internnav_model_path", "")
        or cfg.get("model", {}).get("llm", {}).get("model_path", "")
    )
    return os.path.expandvars(os.path.expanduser(str(raw or "").strip()))


def _apply_data_root_override(cfg: dict, data_root: str | None) -> None:
    if not data_root:
        return
    cfg.setdefault("data", {})["root"] = os.path.expanduser(data_root)
    cfg.setdefault("paths", {})["dataset_root"] = os.path.expanduser(data_root)


def run_dataset(
    config_path: str,
    num_samples: int,
    seed: int,
    *,
    data_root: str | None,
    use_real_processor: bool,
    processor_path: str | None,
) -> None:
    from scripts.training.utils import load_config
    from src.data.factory import build_dataset

    cfg = load_config(config_path)
    _apply_data_root_override(cfg, data_root)
    data_cfg = cfg["data"]
    traj_cfg = data_cfg.get("trajectory", data_cfg.get("sliding_window", {}))
    stages = cfg.get("training", {}).get("stages", [])
    stage_cfg = stages[0] if stages else cfg.get("training", {}).get("stage2", cfg.get("stage2", {}))

    print("=== config check ===")
    print(f"  data.root                 : {data_cfg.get('root')}")
    print(f"  panoramic_vlm_input       : {traj_cfg.get('panoramic_vlm_input')}")
    print(f"  compute_pixel_goal        : {traj_cfg.get('compute_pixel_goal')}")
    print(f"  compute_pano_view_pixel_goal: {traj_cfg.get('compute_pano_view_pixel_goal', 'auto')}")
    print(f"  system2_sft_protocol      : {traj_cfg.get('system2_sft_protocol')}")
    print(f"  stage.train_lm            : {stage_cfg.get('train_lm')}")
    print(f"  stage.train_system2_sft   : {stage_cfg.get('train_system2_sft')}")

    protocol = str(
        stage_cfg.get("system2_sft_protocol", traj_cfg.get("system2_sft_protocol", "direct"))
    ).lower()

    if use_real_processor:
        from transformers import AutoProcessor

        llm_path = processor_path or _resolve_model_path(cfg)
        if not llm_path or llm_path.startswith("$"):
            raise RuntimeError(
                "Cannot load real Qwen processor: set INTERNNAV_MODEL_PATH or pass "
                "--processor-path /path/to/InternNav_Model. "
                "For format-only checks, omit --use-real-processor (default)."
            )
        if not Path(llm_path).is_dir():
            raise FileNotFoundError(f"Processor path does not exist: {llm_path}")
        print(f"  processor.path            : {llm_path}")
        processor = AutoProcessor.from_pretrained(llm_path, trust_remote_code=True)
    else:
        print("  processor                 : lightweight fake (format check only)")
        processor = _PrintProcessor()

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
        meta = {
            "text": sample.get("text"),
            "pano_sample_kind": sample.get("pano_sample_kind", "?"),
            "pano_view_id": sample.get("pano_view_id", "?"),
            "pano_pixel_goal": sample.get("pano_pixel_goal"),
            "pixel_goal": sample.get("pixel_goal"),
        }
        batch = collator([sample])
        target = batch["sft_target_text"][0]
        kind_counter[str(meta["pano_sample_kind"])] += 1
        view_counter[str(meta["pano_view_id"])] += 1

        print(f"\n--- dataset sample #{rank} (idx={idx}) ---")
        print(f"  instruction       : {str(meta['text'] or '')[:80]}")
        print(f"  pano_sample_kind  : {meta['pano_sample_kind']}")
        print(f"  pano_view_id      : {meta['pano_view_id']}")
        print(f"  pano_pixel_goal   : {meta['pano_pixel_goal']}")
        print(f"  legacy pixel_goal : {meta['pixel_goal']}")
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
    p.add_argument(
        "--data-root",
        default=os.environ.get("DATASET_ROOT", "/home/intern/zhr/fjl/r2r_paronamic_data"),
        help="Override cfg paths.dataset_root / data.root",
    )
    p.add_argument(
        "--use-real-processor",
        action="store_true",
        help="Load real Qwen AutoProcessor (needs INTERNNAV_MODEL_PATH). Default uses fake processor.",
    )
    p.add_argument(
        "--processor-path",
        default="",
        help="Optional explicit HF model dir when --use-real-processor is set",
    )
    args = p.parse_args()

    if args.mode == "synthetic":
        run_synthetic(args.protocol)
    else:
        run_dataset(
            args.config,
            args.num_samples,
            args.seed,
            data_root=args.data_root,
            use_real_processor=args.use_real_processor,
            processor_path=args.processor_path or None,
        )


if __name__ == "__main__":
    main()

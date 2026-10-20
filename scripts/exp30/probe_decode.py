#!/usr/bin/env python3
"""EXP-30 probe 1, step 2 (910B or CUDA): decode each state's first System2 call with the base model.

Per state (``probe_select.py``): greedy with its own instruction, greedy with the donor instruction (same scene, other
episode), and ``--samples`` sampled answers with its own instruction (temperature / top_p; when the first turn asks
for the lookdown, the second turn is sampled too).  Inputs are built exactly as the server builds them
(``inference_messages``), answers decoded as ``greedy_native_decision`` decodes them; only the sampling knobs differ.
One process per card: ``--shard i/n`` takes the states whose index is i mod n.  Resumable: rows already in the output
are skipped.  Writes ``<out>/decode.shardIIofNN.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class WithInstruction:
    """A record that reads as another instruction (frames, lookdown and image size stay the record's own)."""

    def __init__(self, record, instruction: str):
        self._record = record
        self.episode = {**record.episode, "instruction": instruction}

    def __getattr__(self, name):
        return getattr(self._record, name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--states", required=True)
    ap.add_argument("--record-parent", required=True, help="directory the states' record paths are relative to")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=30)
    args = ap.parse_args()

    import torch

    import scripts.train_system2_lora as T
    from src.data.s2_sft_dataset import DEFAULT_VLM_IMAGE_SIZE, RecordCache, inference_messages

    index, count = (int(x) for x in args.shard.split("/"))
    states = [json.loads(line) for line in Path(args.states).read_text(encoding="utf-8").splitlines() if line.strip()]
    mine = [s for i, s in enumerate(states) if i % count == index]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"decode.shard{index:02d}of{count:02d}.jsonl"
    done = set()
    if path.is_file():
        done = {json.loads(line)["key"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [s for s in mine if s["key"] not in done]
    print(json.dumps({"shard": args.shard, "states": len(mine), "todo": len(todo)}), flush=True)
    if not todo:
        return 0

    rt = T.init_runtime(args.device)
    T.seed_everything(args.seed, rt.device)
    model, processor = T.load_system2(args.model_path, rt.device)
    model.eval()
    sample_kwargs = {**T.NATIVE_GENERATE_KWARGS, "do_sample": True, "temperature": args.temperature, "top_p": args.top_p}

    def decide(messages, images, lookdown, kwargs):
        inputs = T._served_inputs(processor, messages, rt.device)
        n = inputs["input_ids"].shape[1]
        first = processor.tokenizer.decode(model.generate(**inputs, **kwargs).sequences[0][n:], skip_special_tokens=True)
        if not T.native_requests_lookdown(first):
            return [first]
        messages, _ = T.append_native_lookdown_turn(messages, images, first, lookdown)
        inputs = T._served_inputs(processor, messages, rt.device)
        n = inputs["input_ids"].shape[1]
        return [first, processor.tokenizer.decode(model.generate(**inputs, **kwargs).sequences[0][n:], skip_special_tokens=True)]

    records = RecordCache(8)
    size = tuple(DEFAULT_VLM_IMAGE_SIZE)
    started = time.time()
    with open(path, "a", encoding="utf-8") as handle:
        for k, state in enumerate(todo, 1):
            t0 = time.time()
            record, calls = records.get(str(Path(args.record_parent) / state["record"]))
            call = calls[int(state["call_index"])]
            row = {"key": state["key"], "group": state["group"], "device": rt.device.type}
            with torch.no_grad():
                messages, images, lookdown = inference_messages(record, call, default_vlm_image_size=size)
                row["greedy_own"] = decide(messages, images, lookdown, T.NATIVE_GENERATE_KWARGS)
                if state.get("donor_instruction"):
                    m2, i2, l2 = inference_messages(WithInstruction(record, state["donor_instruction"]), call,
                                                    default_vlm_image_size=size)
                    row["greedy_shuf"] = decide(m2, i2, l2, T.NATIVE_GENERATE_KWARGS)
                else:
                    row["greedy_shuf"] = None
                samples = []
                base_seed = int.from_bytes(state["key"].encode()[:8].ljust(8, b"\0"), "little") % (2 ** 31)
                for j in range(args.samples):
                    torch.manual_seed(base_seed + j)
                    samples.append(decide(messages, images, lookdown, sample_kwargs))
                row["samples"] = samples
            row["seconds"] = round(time.time() - t0, 2)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if k % 10 == 0 or k == len(todo):
                print(json.dumps({"shard": args.shard, "done": k, "of": len(todo),
                                  "s_per_state": round((time.time() - started) / k, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

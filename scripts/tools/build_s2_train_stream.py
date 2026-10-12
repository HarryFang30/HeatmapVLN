#!/usr/bin/env python3
"""Pack the EXP-23 training stream on the 4090 (design §4).

Selection is the trainer's own (``index_calls`` + ``select_training_refs``:
held-out scenes out, at most N calls per episode, disagreements weighted up);
InternData-N1 replay samples of the non-held-out scenes are added with one
shared weight chosen so they make up ``--replay-fraction`` of the draws.  The
whole training order is drawn once (``draw_order``) and written as shards of
``--shard-size`` samples, each sample carrying its own image bytes, plus a
manifest the trainer reads.  Nothing here tokenises: the 910B trainer still
runs the prompt-prefix check on every sample it reads.

    python scripts/tools/build_s2_train_stream.py \\
        --record-root /workspace/exp23/round1/records --label-root /workspace/exp23/round1/labels \\
        --replay-root /workspace/exp23/data/interndata_n1/r2r --out-dir /workspace/exp23/round1/stream
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.interndata_replay import ReplaySample, sample_jpegs, scene_samples  # noqa: E402
from src.data.s2_sft_dataset import (  # noqa: E402
    DEFAULT_AGREE_WEIGHT,
    DEFAULT_DISAGREE_WEIGHT,
    DEFAULT_HOLDOUT_COUNT,
    DEFAULT_MAX_CALLS_PER_EPISODE,
    DEFAULT_MAX_STOP_CORRECTIONS_PER_EPISODE,
    DEFAULT_VLM_IMAGE_SIZE,
    HOLDOUT_SALT,
    RecordCache,
    decision_class,
    episode_vlm_image_size,
    holdout_scenes,
    index_calls,
    select_training_refs,
)
from src.data.s2_stream import (  # noqa: E402
    MANIFEST_NAME,
    STREAM_SCHEMA,
    PackedSample,
    draw_order,
    shard_name,
    write_shard,
)


def _dagger_item(ref) -> dict:
    return {"source": "dagger", "record_path": ref.record_path, "call_index": ref.call_index,
            "key": ref.key, "turns": list(ref.turns)}


def _replay_item(scene_dir: Path, sample: ReplaySample) -> dict:
    turns = [sample.turn1] + ([sample.turn2] if sample.turn2 is not None else [])
    return {"source": "replay", "scene_dir": str(scene_dir), "sample": sample.__dict__ | {"history_frame_ids": list(sample.history_frame_ids)},
            "key": f"replay:{sample.scene_id}_{sample.episode_index:06d}@{sample.frame_id}:{hashlib.sha1(sample.instruction.encode()).hexdigest()[:8]}",
            "turns": turns}


_CACHE: RecordCache | None = None


def _pack(item: dict, vlm_image_size: tuple[int, int], jpeg_quality: int) -> PackedSample:
    global _CACHE
    if item["source"] == "dagger":
        if _CACHE is None:
            _CACHE = RecordCache(32)
        record, calls = _CACHE.get(item["record_path"])
        call = calls[int(item["call_index"])]
        return PackedSample(
            key=item["key"],
            source="dagger",
            instruction=str(record.episode["instruction"]),
            turns=tuple(item["turns"]),
            vlm_image_size=episode_vlm_image_size(record, DEFAULT_VLM_IMAGE_SIZE),
            history_jpegs=tuple(record.front_jpeg(s) for s in call["history_step_ids"]),
            current_jpeg=record.front_jpeg(call["step_id"]),
            lookdown_jpeg=record.lookdown_jpeg(call["call_index"]),
        )
    raw = dict(item["sample"])
    raw["history_frame_ids"] = tuple(raw["history_frame_ids"])
    sample = ReplaySample(**raw)
    history, current, lookdown = sample_jpegs(
        item["scene_dir"], sample, vlm_image_size=vlm_image_size, jpeg_quality=jpeg_quality
    )
    return PackedSample(
        key=item["key"],
        source="replay",
        instruction=sample.instruction,
        turns=tuple(item["turns"]),
        vlm_image_size=tuple(vlm_image_size),
        history_jpegs=tuple(history),
        current_jpeg=current,
        lookdown_jpeg=lookdown,
    )


def _write(job: tuple) -> tuple[int, str, int]:
    index, items, out_dir, vlm_image_size, jpeg_quality = job
    path = Path(out_dir) / shard_name(index)
    samples = [_pack(item, vlm_image_size, jpeg_quality) for item in items]
    return index, write_shard(path, samples), path.stat().st_size


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record-root", action="append", required=True)
    parser.add_argument("--label-root", action="append", required=True)
    parser.add_argument("--replay-root", default="", help="InternData-N1 r2r scene dirs (empty: no replay)")
    parser.add_argument("--replay-fraction", type=float, default=0.3)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=float, default=1.0, help="expected passes over the selected DAgger calls")
    parser.add_argument("--total-samples", type=int, default=0,
                        help="draw exactly this many samples (rounded up to whole shards) instead of --epochs")
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--max-calls-per-episode", type=int, default=DEFAULT_MAX_CALLS_PER_EPISODE)
    parser.add_argument("--max-stop-corrections-per-episode", type=int, default=DEFAULT_MAX_STOP_CORRECTIONS_PER_EPISODE)
    parser.add_argument("--agree-weight", type=float, default=DEFAULT_AGREE_WEIGHT)
    parser.add_argument("--disagree-weight", type=float, default=DEFAULT_DISAGREE_WEIGHT)
    parser.add_argument("--holdout-count", type=int, default=DEFAULT_HOLDOUT_COUNT)
    parser.add_argument("--holdout-salt", default=HOLDOUT_SALT)
    parser.add_argument("--vlm-image-size", type=int, nargs=2, default=list(DEFAULT_VLM_IMAGE_SIZE))
    parser.add_argument("--jpeg-quality", type=int, default=90, help="the client's --rpc_jpeg_quality, for replay")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)

    if args.shard_size % args.global_batch or args.global_batch % args.world_size:
        raise SystemExit("shard size must be a multiple of the global batch, and that of the world size")
    if not 0.0 <= args.replay_fraction < 1.0:
        raise SystemExit("--replay-fraction must be in [0, 1)")
    out_dir = Path(args.out_dir)
    if (out_dir / MANIFEST_NAME).exists():
        raise SystemExit(f"{out_dir} already holds a stream")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    holdout = holdout_scenes(count=args.holdout_count, salt=args.holdout_salt)
    refs, index_stats = index_calls(args.record_root, args.label_root, split="train", holdout=holdout)
    selected, select_stats = select_training_refs(
        refs,
        max_calls_per_episode=args.max_calls_per_episode or None,
        agree_weight=args.agree_weight,
        disagree_weight=args.disagree_weight,
        seed=args.seed,
        max_stop_corrections_per_episode=args.max_stop_corrections_per_episode,
    )
    if not selected:
        raise SystemExit(f"no DAgger calls selected (index {dict(index_stats)}, selection {dict(select_stats)})")
    items = [_dagger_item(ref) for ref in selected]
    weights = [float(ref.weight) for ref in selected]
    dagger_weight = sum(weights)

    replay_counts: Counter = Counter()
    if args.replay_root and args.replay_fraction > 0:
        replay = []
        for scene_dir in sorted(p for p in Path(args.replay_root).iterdir() if p.is_dir() and not p.name.startswith(".")):
            if scene_dir.name in holdout:
                replay_counts["holdout_scenes_skipped"] += 1
                continue
            for sample in scene_samples(scene_dir):
                replay.append(_replay_item(scene_dir, sample))
                replay_counts[f"kind_{sample.kind}"] += 1
        if replay:
            # One shared weight: replay is drawn uniformly among itself and makes up
            # replay_fraction of all draws in expectation.
            w = args.replay_fraction / (1.0 - args.replay_fraction) * dagger_weight / len(replay)
            items += replay
            weights += [w] * len(replay)
            replay_counts["samples"] = len(replay)
            replay_counts["scenes"] = len({i["sample"]["scene_id"] for i in replay})

    if args.total_samples:
        total = int(args.total_samples)
    else:
        total = math.ceil(args.epochs * len(selected) / (1.0 - (args.replay_fraction if replay_counts["samples"] else 0.0)))
    total = math.ceil(total / args.shard_size) * args.shard_size
    order = draw_order(weights, total, args.seed)
    drawn = Counter()
    for i in order:
        item = items[int(i)]
        drawn[item["source"]] += 1
        drawn[f"{item['source']}_target_{decision_class(item['turns'][-1])}"] += 1

    num_shards = total // args.shard_size
    jobs = [
        (s, [items[int(i)] for i in order[s * args.shard_size:(s + 1) * args.shard_size]],
         str(out_dir), tuple(args.vlm_image_size), args.jpeg_quality)
        for s in range(num_shards)
    ]
    shards: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (index, digest, size) in enumerate(pool.map(_write, jobs, chunksize=1), 1):
            shards[index] = {"name": shard_name(index), "sha256": digest, "bytes": size}
            if done % 50 == 0 or done == num_shards:
                print(f"[stream] {done}/{num_shards} shards", flush=True)

    manifest = {
        "schema": STREAM_SCHEMA,
        "created_unix": time.time(),
        "seed": args.seed,
        "shard_size": args.shard_size,
        "num_shards": num_shards,
        "total_samples": total,
        "global_batch": args.global_batch,
        "world_size": args.world_size,
        "epochs": args.epochs,
        "total_samples_requested": args.total_samples,
        "replay_fraction": args.replay_fraction,
        "holdout": {"scenes": list(holdout), "salt": args.holdout_salt, "count": len(holdout)},
        "record_roots": [str(Path(r).resolve()) for r in args.record_root],
        "label_roots": [str(Path(r).resolve()) for r in args.label_root],
        "replay_root": str(Path(args.replay_root).resolve()) if args.replay_root else "",
        "selection": {
            "index": dict(index_stats),
            "select": dict(select_stats),
            "dagger_selected": len(selected),
            "selected_keys_sha256": hashlib.sha256(
                "\n".join(r.key + "|" + "\x1f".join(r.turns) for r in selected).encode()
            ).hexdigest(),
            "replay": dict(replay_counts),
        },
        "drawn": dict(drawn),
        "shards": [shards[s] for s in range(num_shards)],
        "build_s": round(time.time() - started, 1),
    }
    tmp = out_dir / f".{MANIFEST_NAME}.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out_dir / MANIFEST_NAME)
    print(json.dumps({k: manifest[k] for k in ("num_shards", "total_samples", "drawn", "build_s")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

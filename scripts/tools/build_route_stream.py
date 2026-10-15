#!/usr/bin/env python3
"""Pack EXP-25's line-B training stream: route corrections beyond 3 m, anchors, replay (+ line A).

Line B's labels (``label_route_corrections.py``) give every call beyond 3 m of the goal a row
``kind``: ``correction`` (the model's answer made no progress; target = the oracle's pixel
goal or turn), ``anchor`` (it did; target = its own answer) or ``none``.  Round 1 let its
corrections dominate (weight x3, 46% of them on answers that were fine), so here nothing is
weighted up and the composition is set explicitly, without replacement:

=====================  =================================================================
part                   what is taken
=====================  =================================================================
B corrections          per episode at most ``--max-corrections-per-episode`` (the
                       earliest, or a seeded uniform subset), then ``--correction-weight``
                       copies (1.0: each once; < 1 subsamples; > 1 repeats)
B anchors              per episode at most ``--max-anchors-per-episode`` (seeded uniform),
                       then ``--anchors-per-correction`` x the corrections taken, uniform
                       over episodes' anchors (capped by the pool)
A (optional)           ``--merge-a-manifest``: line A's (SpinBreak) DAgger samples exactly
                       as ``build_spinbreak_stream.py`` drew them for that stream (spin
                       STOPs, the model's own STOPs, A's anchors), re-derived from the
                       records and checked against the manifest's ``selected_keys_sha256``;
                       ``--merge-a-fraction`` < 1 subsamples them
replay                 InternData-N1 r2r samples of the training scenes, so that they make
                       up ``--replay-fraction`` of the stream (then the stream is padded with
                       replay to whole shards)
=====================  =================================================================

Line B never targets STOP (a STOP target in the B part is an error); A and B act on disjoint
states (A: within 3 m; B: beyond).  The six held-out scenes stay out of both.  The order is
one seeded shuffle and the shards and manifest are the ones ``build_s2_train_stream.py``
writes (its packing is reused), so the trainer reads this stream as it read round 1's:
shard 256, global batch 64, world size 8, one pass.

    python scripts/tools/build_route_stream.py \\
        --record-root /workspace/exp23/envdrop1/records --label-root /workspace/exp23/envdrop1/labels_routeB \\
        --replay-root /workspace/exp23/data/interndata_n1/r2r \\
        --merge-a-manifest /workspace/exp23/roundA/stream/stream_manifest.json \\
        --out-dir /workspace/exp23/envdrop1/stream_B

``--dry-run`` stops after the selection and prints the composition.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools import build_spinbreak_stream as sb  # noqa: E402
from scripts.tools.build_s2_train_stream import _dagger_item, _replay_item, _write  # noqa: E402
from src.data.interndata_replay import scene_samples  # noqa: E402
from src.data.s2_sft_dataset import (  # noqa: E402
    DEFAULT_HOLDOUT_COUNT,
    DEFAULT_VLM_IMAGE_SIZE,
    HOLDOUT_SALT,
    STOP_TEXT,
    CallRef,
    S2SFTError,
    _stable_rank,
    decision_class,
    holdout_scenes,
    index_calls,
    training_turns,
)
from src.data.s2_stream import MANIFEST_NAME, STREAM_SCHEMA, shard_name  # noqa: E402

BUILDER = "scripts/tools/build_route_stream.py (EXP-25 B)"
DISTANCE_BINS = ((3.0, 5.0), (5.0, 7.0), (7.0, 10.0), (10.0, math.inf))


def _dbin(d: float | None) -> str:
    if d is None:
        return "unknown"
    for lo, hi in DISTANCE_BINS:
        if lo < d <= hi:
            return f"{lo:g}-{hi:g}" if math.isfinite(hi) else f">{lo:g}"
    return f"<={DISTANCE_BINS[0][0]:g}"


# --------------------------------------------------------------------------- #
# Line B selection
# --------------------------------------------------------------------------- #
def route_pools(
    refs: Sequence[CallRef],
    *,
    max_corrections_per_episode: int | None,
    max_anchors_per_episode: int | None,
    correction_pick: str = "earliest",
    seed: int = 0,
) -> tuple[list[CallRef], list[CallRef], Counter]:
    """(corrections, anchors) with targets, after the per-episode caps, plus counts."""
    if correction_pick not in ("earliest", "uniform"):
        raise ValueError(f"correction_pick must be earliest or uniform, not {correction_pick!r}")
    stats: Counter = Counter()
    by_episode: dict[str, list[CallRef]] = defaultdict(list)
    for ref in refs:
        if ref.label is None:
            stats["unlabelled"] += 1
        elif ref.label.skip:
            stats["no_target"] += 1
        else:
            by_episode[ref.record_path].append(ref)
    corrections: list[CallRef] = []
    anchors: list[CallRef] = []
    for episode, calls in by_episode.items():
        mine_c, mine_a = [], []
        for ref in sorted(calls, key=lambda r: r.call_index):
            turns = training_turns(ref.recorded, ref.label)
            if turns == (STOP_TEXT,) or decision_class(turns[-1]) == "stop":
                raise S2SFTError(f"{ref.key}: line B never targets STOP, got {turns!r}")
            if ref.label.agree:
                if ref.label.target_turns not in (None, ref.recorded):
                    raise S2SFTError(f"{ref.key}: accepted target {ref.label.target_turns!r} is not the "
                                     f"recorded answer {ref.recorded!r}")
                mine_a.append(dataclasses.replace(ref, turns=turns))
            else:
                mine_c.append(dataclasses.replace(ref, turns=turns))
        stats["episodes"] += 1
        stats["corrections_before_cap"] += len(mine_c)
        stats["anchors_before_cap"] += len(mine_a)
        if max_corrections_per_episode is not None and len(mine_c) > max_corrections_per_episode:
            stats["episodes_corrections_capped"] += 1
            if correction_pick == "uniform":
                mine_c = sorted(mine_c, key=lambda r: _stable_rank(seed, "c", episode, r.call_index))
            mine_c = sorted(mine_c[:max_corrections_per_episode], key=lambda r: r.call_index)
        if max_anchors_per_episode is not None and len(mine_a) > max_anchors_per_episode:
            stats["episodes_anchors_capped"] += 1
            mine_a = sorted(mine_a, key=lambda r: _stable_rank(seed, "a", episode, r.call_index))
            mine_a = sorted(mine_a[:max_anchors_per_episode], key=lambda r: r.call_index)
        stats["episodes_with_corrections"] += bool(mine_c)
        corrections += mine_c
        anchors += mine_a
    stats["corrections"] = len(corrections)
    stats["anchors"] = len(anchors)
    return corrections, anchors, stats


def take(items: Sequence, factor: float, rng: np.random.Generator) -> list:
    """``round(factor * len(items))`` items: whole copies, then a subsample without replacement."""
    if factor < 0:
        raise ValueError("factor must be >= 0")
    n = int(round(factor * len(items)))
    whole, rest = divmod(n, len(items)) if items else (0, 0)
    out = list(items) * whole
    if rest:
        out += [items[int(i)] for i in sorted(rng.choice(len(items), size=rest, replace=False))]
    return out


# --------------------------------------------------------------------------- #
# Line A, exactly as its stream drew it
# --------------------------------------------------------------------------- #
def replay_items(replay_root: str, holdout: Sequence[str]) -> tuple[list[dict], Counter]:
    items: list[dict] = []
    counts: Counter = Counter()
    for scene_dir in sorted(p for p in Path(replay_root).iterdir() if p.is_dir() and not p.name.startswith(".")):
        if scene_dir.name in holdout:
            counts["holdout_scenes_skipped"] += 1
            continue
        for sample in scene_samples(scene_dir):
            items.append(_replay_item(scene_dir, sample))
            counts[f"pool_kind_{sample.kind}"] += 1
    counts["pool"] = len(items)
    counts["scenes"] = len({i["sample"]["scene_id"] for i in items})
    return items, counts


def line_a_items(manifest_path: str | os.PathLike, replay_cache: dict[str, list[dict]] | None = None) -> tuple[list[tuple[dict, str]], dict]:
    """Line A's DAgger (item, category) list as its stream drew it, verified by key hash."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    selection = manifest["selection"]
    holdout = tuple(manifest["holdout"]["scenes"])
    refs, _ = index_calls(manifest["record_roots"], manifest["label_roots"], split="train", holdout=holdout)
    success = sb.episode_success(sorted({r.record_path for r in refs}))
    selected, _ = sb.categorize(refs, success, sb.SpinBreakConfig(**{
        k: tuple(v) if isinstance(v, list) else v for k, v in selection["config"].items()}))
    pools: dict[str, list] = defaultdict(list)
    for ref, category in selected:
        pools[category].append(_dagger_item(ref))
    replay: list[dict] = []
    if manifest.get("replay_root") and int(selection.get("replay_samples") or 0) > 0:
        key = str(manifest["replay_root"])
        if replay_cache is not None and key in replay_cache:
            replay = replay_cache[key]
        else:
            replay, _ = replay_items(key, holdout)
            if replay_cache is not None:
                replay_cache[key] = replay
    chosen, _ = sb.draw(
        pools, replay, total=int(manifest["total_samples"]),
        replay_samples=int(selection.get("replay_samples") or 0) if replay else 0,
        own_stop_max=selection.get("own_stop_max"), spin_stop_max=selection.get("spin_stop_max"),
        seed=int(manifest["seed"]),
    )
    order = np.random.default_rng(int(manifest["seed"]) + 1).permutation(len(chosen))
    items = [chosen[int(i)] for i in order]
    dagger = [(item, category) for item, category in items if item["source"] == "dagger"]
    digest = hashlib.sha256("\n".join(i["key"] + "|" + "\x1f".join(i["turns"]) for i, _ in dagger).encode()).hexdigest()
    if digest != selection["selected_keys_sha256"]:
        raise SystemExit(f"line A re-derived from {manifest_path} does not match its selected_keys_sha256 "
                         f"({digest} vs {selection['selected_keys_sha256']}): records or labels changed")
    info = {"manifest": str(Path(manifest_path).resolve()), "selected_keys_sha256": digest, "dagger": len(dagger),
            "holdout": list(holdout), "record_roots": manifest["record_roots"], "label_roots": manifest["label_roots"]}
    return dagger, info


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record-root", action="append", required=True)
    parser.add_argument("--label-root", action="append", required=True)
    parser.add_argument("--replay-root", default="", help="InternData-N1 r2r scene dirs (empty: no replay)")
    parser.add_argument("--replay-fraction", type=float, default=0.3)
    parser.add_argument("--max-corrections-per-episode", type=int, default=8, help="-1: no cap")
    parser.add_argument("--correction-pick", choices=("earliest", "uniform"), default="earliest")
    parser.add_argument("--max-anchors-per-episode", type=int, default=12, help="-1: no cap")
    parser.add_argument("--correction-weight", type=float, default=1.0, help="copies per correction (no x3)")
    parser.add_argument("--anchors-per-correction", type=float, default=2.0)
    parser.add_argument("--merge-a-manifest", default="", help="line A's stream manifest: merge its DAgger samples")
    parser.add_argument("--merge-a-fraction", type=float, default=1.0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--holdout-count", type=int, default=DEFAULT_HOLDOUT_COUNT)
    parser.add_argument("--holdout-salt", default=HOLDOUT_SALT)
    parser.add_argument("--vlm-image-size", type=int, nargs=2, default=list(DEFAULT_VLM_IMAGE_SIZE))
    parser.add_argument("--jpeg-quality", type=int, default=90, help="the client's --rpc_jpeg_quality, for replay")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true", help="select and print the composition, write nothing")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.shard_size % args.global_batch or args.global_batch % args.world_size:
        raise SystemExit("shard size must be a multiple of the global batch, and that of the world size")
    if not 0.0 <= args.replay_fraction < 1.0:
        raise SystemExit("--replay-fraction must be in [0, 1)")
    out_dir = Path(args.out_dir)
    if not args.dry_run and (out_dir / MANIFEST_NAME).exists():
        raise SystemExit(f"{out_dir} already holds a stream")
    started = time.time()
    rng = np.random.default_rng(args.seed)
    holdout = holdout_scenes(count=args.holdout_count, salt=args.holdout_salt)

    # --- line B
    refs, index_stats = index_calls(args.record_root, args.label_root, split="train", holdout=holdout)
    corrections, anchors, pool_stats = route_pools(
        refs,
        max_corrections_per_episode=None if args.max_corrections_per_episode < 0 else args.max_corrections_per_episode,
        max_anchors_per_episode=None if args.max_anchors_per_episode < 0 else args.max_anchors_per_episode,
        correction_pick=args.correction_pick,
        seed=args.seed,
    )
    if not corrections:
        raise SystemExit(f"no line-B corrections (index {dict(index_stats)}, pools {dict(pool_stats)})")
    taken_c = take(corrections, args.correction_weight, rng)
    n_anchor = int(round(args.anchors_per_correction * len(taken_c)))
    anchors_capped = n_anchor > len(anchors)
    taken_a = take(anchors, min(n_anchor, len(anchors)) / len(anchors), rng) if anchors and n_anchor else []
    b_parts = [(ref, f"B_correction_{decision_class(ref.turns[-1]).split('_')[0]}") for ref in taken_c]
    b_parts += [(ref, "B_anchor") for ref in taken_a]
    chosen: list[tuple[dict, str]] = [(_dagger_item(ref), category) for ref, category in b_parts]
    b_distance = Counter(f"{category}@{_dbin(ref.distance_to_goal)}" for ref, category in b_parts)

    # --- replay pool (shared with line A's re-derivation)
    replay_cache: dict[str, list[dict]] = {}
    replay: list[dict] = []
    replay_counts: Counter = Counter()
    if args.replay_root and args.replay_fraction > 0:
        replay, replay_counts = replay_items(args.replay_root, holdout)
        replay_cache[str(args.replay_root)] = replay

    # --- line A
    merge_info: dict[str, Any] = {}
    if args.merge_a_manifest:
        a_items, merge_info = line_a_items(args.merge_a_manifest, replay_cache)
        if tuple(merge_info["holdout"]) != tuple(holdout):
            raise SystemExit(f"line A's held-out scenes {merge_info['holdout']} are not this stream's {list(holdout)}")
        if args.merge_a_fraction < 1.0:
            a_items = take(a_items, args.merge_a_fraction, rng)
        chosen += [(item, f"A_{category}") for item, category in a_items]
        merge_info["taken"] = len(a_items)

    # --- replay share, then whole shards
    dagger_n = len(chosen)
    n_replay = int(math.ceil(args.replay_fraction / (1.0 - args.replay_fraction) * dagger_n)) if replay else 0
    total = int(math.ceil((dagger_n + n_replay) / args.shard_size) * args.shard_size)
    if replay:
        n_replay = total - dagger_n
        if n_replay > len(replay):
            raise SystemExit(f"{n_replay} replay samples needed, the pool holds {len(replay)}")
        chosen += [(replay[int(i)], "replay") for i in sorted(rng.choice(len(replay), size=n_replay, replace=False))]
    elif total > dagger_n:
        # no replay: pad with more anchors (without replacement while they last)
        used = {r.uid for r in taken_a}
        spare = [r for r in anchors if r.uid not in used]
        need = total - dagger_n
        if need > len(spare):
            raise SystemExit(f"{need} padding samples needed but only {len(spare)} spare anchors; give --replay-root")
        chosen += [(_dagger_item(spare[int(i)]), "B_anchor") for i in sorted(rng.choice(len(spare), size=need, replace=False))]
    order = np.random.default_rng(args.seed + 1).permutation(len(chosen))
    items = [chosen[int(i)] for i in order]

    drawn: Counter = Counter()
    for item, category in items:
        drawn[item["source"]] += 1
        drawn[f"{item['source']}_target_{decision_class(item['turns'][-1])}"] += 1
        drawn[f"category_{category}"] += 1
    stop_targets = sum(v for k, v in drawn.items() if k.endswith("_target_stop"))
    b_total = sum(v for k, v in drawn.items() if k.startswith("category_B_"))
    composition = {
        "total_samples": total,
        "num_shards": total // args.shard_size,
        "steps_at_global_batch": total // args.global_batch,
        "by_category": {k[len("category_"):]: v for k, v in sorted(drawn.items()) if k.startswith("category_")},
        "shares": {
            "B_corrections": round(len(taken_c) / total, 4),
            "B_anchors": round(len([1 for _, c in items if c == "B_anchor"]) / total, 4),
            "A": round(sum(v for k, v in drawn.items() if k.startswith("category_A_")) / total, 4),
            "replay": round(drawn["replay"] / total, 4),
            "stop_targets": round(stop_targets / total, 4),
            "corrections_of_B": round(len(taken_c) / b_total, 4) if b_total else None,
        },
        "B_by_distance": dict(sorted(b_distance.items())),
        "anchors_capped_by_pool": anchors_capped,
    }
    print(json.dumps(composition, indent=1, sort_keys=True), flush=True)
    if args.dry_run:
        print(json.dumps({"index": dict(index_stats), "pools": dict(pool_stats), "replay": dict(replay_counts),
                          "merge_a": merge_info}, indent=1, sort_keys=True))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    num_shards = total // args.shard_size
    jobs = [
        (s, [item for item, _ in items[s * args.shard_size:(s + 1) * args.shard_size]],
         str(out_dir), tuple(args.vlm_image_size), args.jpeg_quality)
        for s in range(num_shards)
    ]
    shards: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (index, digest, size) in enumerate(pool.map(_write, jobs, chunksize=1), 1):
            shards[index] = {"name": shard_name(index), "sha256": digest, "bytes": size}
            if done % 10 == 0 or done == num_shards:
                print(f"[stream] {done}/{num_shards} shards", flush=True)

    dagger_items = [item for item, _ in items if item["source"] == "dagger"]
    manifest = {
        "schema": STREAM_SCHEMA,
        "created_unix": time.time(),
        "builder": BUILDER,
        "seed": args.seed,
        "shard_size": args.shard_size,
        "num_shards": num_shards,
        "total_samples": total,
        "global_batch": args.global_batch,
        "world_size": args.world_size,
        "epochs": 1.0,
        "total_samples_requested": 0,
        "replay_fraction": round(drawn["replay"] / total, 6),
        "holdout": {"scenes": list(holdout), "salt": args.holdout_salt, "count": len(holdout)},
        "record_roots": [str(Path(r).resolve()) for r in args.record_root],
        "label_roots": [str(Path(r).resolve()) for r in args.label_root],
        "replay_root": str(Path(args.replay_root).resolve()) if args.replay_root else "",
        "selection": {
            "rule": "EXP-25 B: route corrections beyond 3 m (no STOP), the model's own progressing answers as "
                    "anchors, InternData replay; optionally line A's SpinBreak DAgger samples",
            "knobs": {
                "max_corrections_per_episode": args.max_corrections_per_episode,
                "correction_pick": args.correction_pick,
                "max_anchors_per_episode": args.max_anchors_per_episode,
                "correction_weight": args.correction_weight,
                "anchors_per_correction": args.anchors_per_correction,
                "replay_fraction": args.replay_fraction,
                "merge_a_manifest": args.merge_a_manifest,
                "merge_a_fraction": args.merge_a_fraction,
            },
            "index": dict(index_stats),
            "pools": dict(pool_stats),
            "merge_a": merge_info,
            "replay": dict(replay_counts),
            "composition": composition,
            "dagger_selected": len(dagger_items),
            "selected_keys_sha256": hashlib.sha256(
                "\n".join(i["key"] + "|" + "\x1f".join(i["turns"]) for i in dagger_items).encode()
            ).hexdigest(),
            "stop_share": composition["shares"]["stop_targets"],
        },
        "drawn": dict(drawn),
        "shards": [shards[s] for s in range(num_shards)],
        "build_s": round(time.time() - started, 1),
    }
    tmp = out_dir / f".{MANIFEST_NAME}.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out_dir / MANIFEST_NAME)
    print(json.dumps({k: manifest[k] for k in ("num_shards", "total_samples", "build_s")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

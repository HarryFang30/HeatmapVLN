#!/usr/bin/env python3
"""Pack EXP-25's SpinBreak (A) training stream from round 1's records: STOP only in the spin trap.

EXP-23 round 1 trained on oracle labels and lost 13.2 pt: STOP on the approach at 1-2 m,
turn labels that flip direction (closed-loop reversals 0.35% -> 45%), pixel targets beyond
the released model's convention.  The base model's own failure that the training set
does reproduce is the spin trap: inside 1 m of the goal it turns and never says STOP
right after a turn.  So this stream supervises STOP there and nowhere else, and every
other target is the model's own recorded answer -- no oracle pixel or turn target at all
(ledger EXP-25 "设置"; categories and boundaries below are that text):

=================  ========================================================  ===========
category           call (previous call = the one before it in the episode)   episodes
=================  ========================================================  ===========
stop_spin          d <= 1 m, previous call a turn, >= 1 earlier call at      all
                   d <= 1 m, the model did not stop; target ``STOP``; the
                   earliest 3 per episode
anchor_own_stop    the model's own STOP at d <= 1 m                          successful
anchor_post_pixel  previous call a pixel goal, 3 < d <= 6 m                  successful
anchor_post_turn   previous call a turn, d > 3 m (its own turn direction)    successful
anchor_keep_going  1.25 <= d <= 3 m, previous call not a turn, the model     successful
                   kept going (pixel goal 1.5-3 m lands here)
(none)             previous call a turn at 1.25-3 m; a STOP beyond 1 m;     --
                   everything else
=================  ========================================================  ===========

Anchors (all but ``stop_spin``) take the recorded answer verbatim, both turns when it
looked down.  The six held-out scenes stay out (``holdout_scenes``).  Distances are the
labeller's geodesic (``index_calls``); success is the record's ``outcome.success``.

Every ``stop_spin`` call (up to ``--spin-stop-max``) and every own STOP (up to
``--own-stop-max``; these two set the STOP share, 12-15% in EXP-25) is taken once;
``--replay-samples`` InternData-N1 r2r samples of the training scenes are drawn without
replacement (``interndata_replay``, the same items ``build_s2_train_stream.py`` packs);
anchors are drawn without replacement, uniformly over the three pools, to fill
``--total-samples`` (rounded up to whole shards).  The order is one seeded shuffle, and the
shards and manifest are the ones ``build_s2_train_stream.py`` writes (its packing is reused),
so the trainer reads this stream as it read round 1's: one pass.

    python scripts/tools/build_spinbreak_stream.py \\
        --record-root /workspace/exp23/round1/records --label-root /workspace/exp23/round1/labels \\
        --replay-root /workspace/exp23/data/interndata_n1/r2r --out-dir /workspace/exp23/roundA/stream

``--dry-run`` stops after the selection and prints the pools.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.build_s2_train_stream import _dagger_item, _replay_item, _write  # noqa: E402
from src.data.interndata_replay import scene_samples  # noqa: E402
from src.data.s2_records import EpisodeRecord  # noqa: E402
from src.data.s2_sft_dataset import (  # noqa: E402
    DEFAULT_HOLDOUT_COUNT,
    DEFAULT_VLM_IMAGE_SIZE,
    HOLDOUT_SALT,
    STOP_TEXT,
    CallRef,
    decision_class,
    holdout_scenes,
    index_calls,
)
from src.data.s2_stream import MANIFEST_NAME, STREAM_SCHEMA, shard_name  # noqa: E402

STOP_CATEGORY = "stop_spin"
ANCHOR_CATEGORIES = ("anchor_post_pixel", "anchor_post_turn", "anchor_keep_going")
OWN_STOP_CATEGORY = "anchor_own_stop"
CATEGORIES = (STOP_CATEGORY, OWN_STOP_CATEGORY) + ANCHOR_CATEGORIES


@dataclass(frozen=True)
class SpinBreakConfig:
    spin_m: float = 1.0  # stop_spin: d <= this, and an earlier call at d <= this
    min_earlier_within_spin: int = 1
    max_spin_stops_per_episode: int = 3  # the earliest ones
    own_stop_m: float = 1.0  # anchor_own_stop: the model's own STOP at d <= this
    post_pixel_m: tuple[float, float] = (3.0, 6.0)  # (low, high]; 1.5-3 m is keep_going
    post_turn_gt_m: float = 3.0
    keep_going_m: tuple[float, float] = (1.25, 3.0)  # [low, high]


def _kind(cls: str | None) -> str:
    if cls is None:
        return "none"
    if cls == "pixel":
        return "pixel"
    if cls in ("turn_left", "turn_right"):
        return "turn"
    return "other"


def categorize(
    refs: Sequence[CallRef], success: Mapping[str, bool], config: SpinBreakConfig = SpinBreakConfig()
) -> tuple[list[tuple[CallRef, str]], Counter]:
    """(ref with its target turns, category) for every selected call, plus why the rest were not.

    ``refs`` are ``index_calls`` refs (labelled; the label's geodesic is ``distance_to_goal``);
    ``success`` maps a record path to its episode's outcome.  Nothing is drawn here.
    """
    stats: Counter = Counter()
    by_episode: dict[str, list[CallRef]] = defaultdict(list)
    for ref in refs:
        by_episode[ref.record_path].append(ref)
    selected: list[tuple[CallRef, str]] = []
    for path, calls in by_episode.items():
        calls = sorted(calls, key=lambda r: r.call_index)
        by_index = {r.call_index: r for r in calls}
        ok = bool(success[path])
        stats["episodes_successful" if ok else "episodes_failed"] += 1
        within, spin_stops = 0, 0
        for ref in calls:
            d = ref.distance_to_goal
            cls = decision_class(ref.recorded[-1])
            if ref.call_index == 0:
                prev = "none"
            elif ref.call_index - 1 in by_index:
                prev = _kind(decision_class(by_index[ref.call_index - 1].recorded[-1]))
            else:
                prev = "unknown"  # the previous call has no label
            earlier = within
            if d is not None and d <= config.spin_m:
                within += 1
            if d is None:
                stats["skip_no_distance"] += 1
                continue
            if prev == "unknown":
                stats["skip_previous_call_unlabelled"] += 1
                continue
            if d <= config.spin_m and prev == "turn" and earlier >= config.min_earlier_within_spin and cls != "stop":
                spin_stops += 1
                if spin_stops > config.max_spin_stops_per_episode:
                    stats["skip_spin_over_episode_cap"] += 1
                    continue
                selected.append((dataclasses.replace(ref, turns=(STOP_TEXT,)), STOP_CATEGORY))
                stats[f"{STOP_CATEGORY}_{'successful' if ok else 'failed'}_episode"] += 1
                continue
            if not ok:
                stats["skip_failed_episode"] += 1
                continue
            if cls == "stop":
                if d <= config.own_stop_m:
                    selected.append((dataclasses.replace(ref, turns=tuple(ref.recorded)), OWN_STOP_CATEGORY))
                else:
                    stats["skip_own_stop_beyond_own_stop_m"] += 1
                continue
            if cls == "invalid":
                stats["skip_invalid_answer"] += 1
                continue
            category = None
            if prev == "turn":
                if d > config.post_turn_gt_m:
                    category = "anchor_post_turn"
                elif config.keep_going_m[0] <= d <= config.keep_going_m[1]:
                    stats["skip_post_turn_1.25_3m_unlabelled"] += 1
                    continue
            elif config.keep_going_m[0] <= d <= config.keep_going_m[1]:
                category = "anchor_keep_going"
            elif prev == "pixel" and config.post_pixel_m[0] < d <= config.post_pixel_m[1]:
                category = "anchor_post_pixel"
            if category is None:
                stats[f"skip_other_prev_{prev}"] += 1
                continue
            selected.append((dataclasses.replace(ref, turns=tuple(ref.recorded)), category))
    for _, category in selected:
        stats[f"pool_{category}"] += 1
    return selected, stats


def episode_success(record_paths: Sequence[str]) -> dict[str, bool]:
    """``episode.json`` ``outcome.success`` of each record (a record without one is an error)."""
    out = {}
    for path in record_paths:
        outcome = EpisodeRecord(path).episode.get("outcome")
        if not isinstance(outcome, Mapping) or "success" not in outcome:
            raise SystemExit(f"{path}: episode.json has no outcome.success")
        out[path] = float(outcome["success"]) > 0.5
    return out


def draw(
    pools: Mapping[str, Sequence], replay: Sequence, *, total: int, replay_samples: int,
    own_stop_max: int | None, seed: int, spin_stop_max: int | None = None,
) -> tuple[list, Counter]:
    """The training list (unshuffled) of (item, category): spin STOPs and own STOPs (each up
    to a cap, None: all), ``replay_samples`` replay items and enough anchors to make
    ``total``, all without replacement."""
    rng = np.random.default_rng(seed)

    def pick(items: Sequence, k: int) -> list:
        if k > len(items):
            raise SystemExit(f"asked for {k} of {len(items)} items")
        return [items[int(i)] for i in sorted(rng.choice(len(items), size=k, replace=False))]

    chosen: list = []
    counts: Counter = Counter()
    for category, cap in ((STOP_CATEGORY, spin_stop_max), (OWN_STOP_CATEGORY, own_stop_max)):
        items = list(pools.get(category, []))
        if cap is not None and len(items) > cap:
            items = pick(items, cap)
        chosen += [(x, category) for x in items]
    chosen += [(x, "replay") for x in pick(list(replay), replay_samples)]
    anchors = [(x, c) for c in ANCHOR_CATEGORIES for x in pools.get(c, [])]
    need = total - len(chosen)
    if need < 0:
        raise SystemExit(f"STOPs and replay alone are {len(chosen)} samples, more than the total {total}")
    if need > len(anchors):
        raise SystemExit(f"{need} anchors needed but the pools hold {len(anchors)}; lower --total-samples")
    chosen += pick(anchors, need)
    for _, category in chosen:
        counts[category] += 1
    return chosen, counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record-root", action="append", required=True)
    parser.add_argument("--label-root", action="append", required=True)
    parser.add_argument("--replay-root", default="", help="InternData-N1 r2r scene dirs (empty: no replay)")
    parser.add_argument("--replay-samples", type=int, default=10000)
    parser.add_argument("--total-samples", type=int, default=28160, help="rounded up to whole shards")
    parser.add_argument("--own-stop-max", type=int, default=-1, help="cap on the model's own STOPs (-1: all)")
    parser.add_argument("--spin-stop-max", type=int, default=-1, help="cap on the spin STOP targets (-1: all)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--holdout-count", type=int, default=DEFAULT_HOLDOUT_COUNT)
    parser.add_argument("--holdout-salt", default=HOLDOUT_SALT)
    parser.add_argument("--vlm-image-size", type=int, nargs=2, default=list(DEFAULT_VLM_IMAGE_SIZE))
    parser.add_argument("--jpeg-quality", type=int, default=90, help="the client's --rpc_jpeg_quality, for replay")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="select and print the pools, write nothing")
    args = parser.parse_args(argv)

    if args.shard_size % args.global_batch or args.global_batch % args.world_size:
        raise SystemExit("shard size must be a multiple of the global batch, and that of the world size")
    out_dir = Path(args.out_dir)
    if not args.dry_run and (out_dir / MANIFEST_NAME).exists():
        raise SystemExit(f"{out_dir} already holds a stream")
    started = time.time()
    config = SpinBreakConfig()

    holdout = holdout_scenes(count=args.holdout_count, salt=args.holdout_salt)
    refs, index_stats = index_calls(args.record_root, args.label_root, split="train", holdout=holdout)
    success = episode_success(sorted({r.record_path for r in refs}))
    selected, select_stats = categorize(refs, success, config)
    pools: dict[str, list] = defaultdict(list)
    for ref, category in selected:
        pools[category].append(ref)

    replay: list[dict] = []
    replay_counts: Counter = Counter()
    if args.replay_root and args.replay_samples > 0:
        for scene_dir in sorted(p for p in Path(args.replay_root).iterdir() if p.is_dir() and not p.name.startswith(".")):
            if scene_dir.name in holdout:
                replay_counts["holdout_scenes_skipped"] += 1
                continue
            for sample in scene_samples(scene_dir):
                replay.append(_replay_item(scene_dir, sample))
                replay_counts[f"pool_kind_{sample.kind}"] += 1
        replay_counts["pool"] = len(replay)
        replay_counts["scenes"] = len({i["sample"]["scene_id"] for i in replay})

    total = math.ceil(int(args.total_samples) / args.shard_size) * args.shard_size
    own_stop_max = None if args.own_stop_max < 0 else int(args.own_stop_max)
    spin_stop_max = None if args.spin_stop_max < 0 else int(args.spin_stop_max)
    chosen, counts = draw(
        {c: [_dagger_item(r) for r in refs_] for c, refs_ in pools.items()}, replay,
        total=total, replay_samples=args.replay_samples if replay else 0, own_stop_max=own_stop_max,
        spin_stop_max=spin_stop_max, seed=args.seed,
    )
    order = np.random.default_rng(args.seed + 1).permutation(len(chosen))
    items = [chosen[int(i)] for i in order]
    drawn: Counter = Counter()
    for item, category in items:
        drawn[item["source"]] += 1
        drawn[f"{item['source']}_target_{decision_class(item['turns'][-1])}"] += 1
        drawn[f"category_{category}"] += 1
    stops = sum(v for k, v in drawn.items() if k.endswith("_target_stop"))
    stop_share = round(stops / total, 4)
    summary = {
        "pools": {c: len(pools.get(c, [])) for c in CATEGORIES} | {"replay": len(replay)},
        "drawn_by_category": {k[len("category_"):]: v for k, v in sorted(drawn.items()) if k.startswith("category_")},
        "stop_targets": stops, "stop_share": stop_share, "total_samples": total, "num_shards": total // args.shard_size,
        "episodes": {"successful": select_stats["episodes_successful"], "failed": select_stats["episodes_failed"]},
    }
    print(json.dumps(summary, indent=1, sort_keys=True), flush=True)
    if args.dry_run:
        print(json.dumps({"index": dict(index_stats), "select": dict(select_stats), "replay": dict(replay_counts)},
                         indent=1, sort_keys=True))
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

    dagger_items = [item for item, category in items if item["source"] == "dagger"]
    manifest = {
        "schema": STREAM_SCHEMA,
        "created_unix": time.time(),
        "builder": "scripts/tools/build_spinbreak_stream.py (EXP-25 A)",
        "seed": args.seed,
        "shard_size": args.shard_size,
        "num_shards": num_shards,
        "total_samples": total,
        "global_batch": args.global_batch,
        "world_size": args.world_size,
        "epochs": 1.0,
        "total_samples_requested": args.total_samples,
        "replay_fraction": round(drawn["replay"] / total, 6),
        "holdout": {"scenes": list(holdout), "salt": args.holdout_salt, "count": len(holdout)},
        "record_roots": [str(Path(r).resolve()) for r in args.record_root],
        "label_roots": [str(Path(r).resolve()) for r in args.label_root],
        "replay_root": str(Path(args.replay_root).resolve()) if args.replay_root else "",
        "selection": {
            "rule": "EXP-25 SpinBreak: STOP only in the spin trap, every other target the model's own answer",
            "config": dataclasses.asdict(config),
            "own_stop_max": own_stop_max,
            "spin_stop_max": spin_stop_max,
            "replay_samples": args.replay_samples,
            "anchor_draw": "uniform without replacement over the anchor pools, to fill total_samples",
            "index": dict(index_stats),
            "select": dict(select_stats),
            "pools": summary["pools"],
            "dagger_selected": len(dagger_items),
            "selected_keys_sha256": hashlib.sha256(
                "\n".join(i["key"] + "|" + "\x1f".join(i["turns"]) for i in dagger_items).encode()
            ).hexdigest(),
            "replay": dict(replay_counts),
            "stop_share": stop_share,
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

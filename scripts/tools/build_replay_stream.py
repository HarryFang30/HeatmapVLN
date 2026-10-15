#!/usr/bin/env python3
"""Pack an expert-data training stream from InternData-N1 scene directories (EXP-26).

EXP-25's probes put most of val_unseen's navigation failures on unseen scenes (a bad first
pixel goal on 6.3% of val_seen episodes, 15.4% of val_unseen), and InternData-N1's ScaleVLN
part (794 HM3D scenes, the base model's own label format) was left out of the released
System2 training.  This packs supervised fine-tuning data from such scene directories --
no DAgger records, no hand-written labels: every target is the dataset's expert label,
built by ``src/data/interndata_replay.py`` exactly as round 1's replay was.

Each ``--root NAME=DIR`` is a directory of scene dirs; ``--samples NAME=K`` draws K samples
from it, after capping every scene at ``--max-per-scene NAME=K`` (a seeded subset), so the
draw spreads over scenes instead of following the largest ones.  ``--stop-upsample F``
repeats every drawn STOP sample F times in all, as InternNav's System2 data loader does
(``stop_list * 5``), keeping the base model's STOP calibration.  ``--exclude-scenes`` and
``--exclude-holdout`` (EXP-23's six held-out R2R scenes) keep scenes out.  The list is
shuffled once, trimmed to whole shards, and written in the format the trainer reads
(``build_s2_train_stream.py``'s packing, reused).

    python scripts/tools/build_replay_stream.py \\
        --root scalevln=/workspace/exp23/data/interndata_n1/scalevln_train --samples scalevln=40000 \\
        --max-per-scene scalevln=400 \\
        --root r2r=/workspace/exp23/data/interndata_n1/r2r --samples r2r=40000 --exclude-holdout \\
        --stop-upsample 5 --out-dir /workspace/exp23/roundS/stream
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.build_s2_train_stream import _replay_item, _write  # noqa: E402
from src.data.interndata_replay import scene_samples  # noqa: E402
from src.data.s2_sft_dataset import DEFAULT_VLM_IMAGE_SIZE, holdout_scenes  # noqa: E402
from src.data.s2_stream import MANIFEST_NAME, STREAM_SCHEMA, shard_name  # noqa: E402


def _pairs(values: Sequence[str], flag: str) -> dict[str, str]:
    out = {}
    for value in values:
        name, sep, rest = value.partition("=")
        if not sep or not name or not rest:
            raise SystemExit(f"{flag} takes NAME=VALUE, got {value!r}")
        out[name] = rest
    return out


def select(
    pools: Mapping[str, Mapping[str, Sequence]], *, samples: Mapping[str, int],
    max_per_scene: Mapping[str, int], stop_upsample: int, seed: int,
) -> tuple[list[tuple[str, object]], Counter]:
    """(root name, sample) pairs before shuffling: per root, every scene capped at its
    seeded subset, then ``samples[root]`` drawn without replacement, then each STOP sample
    repeated to ``stop_upsample`` copies.  ``pools[root][scene]`` lists the scene's samples
    (anything with a ``kind``)."""
    if stop_upsample < 1:
        raise SystemExit("--stop-upsample must be >= 1")
    rng = np.random.default_rng(seed)
    chosen: list[tuple[str, object]] = []
    counts: Counter = Counter()
    for root in sorted(pools):
        cap = max_per_scene.get(root)
        pool: list = []
        for scene in sorted(pools[root]):
            items = list(pools[root][scene])
            if cap is not None and len(items) > cap:
                items = [items[int(i)] for i in sorted(rng.choice(len(items), size=cap, replace=False))]
            pool += items
        want = int(samples.get(root, 0))
        if want > len(pool):
            raise SystemExit(f"{root}: asked for {want} samples, the capped pool holds {len(pool)}")
        picked = [pool[int(i)] for i in sorted(rng.choice(len(pool), size=want, replace=False))]
        counts[f"{root}_pool"] = len(pool)
        counts[f"{root}_scenes"] = len(pools[root])
        for sample in picked:
            copies = stop_upsample if sample.kind == "stop" else 1
            chosen += [(root, sample)] * copies
            counts[f"{root}_target_{sample.kind}"] += copies
            counts[root] += copies
    return chosen, counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", action="append", required=True, help="NAME=DIR of scene dirs")
    parser.add_argument("--samples", action="append", default=[], help="NAME=K samples to draw from that root")
    parser.add_argument("--max-per-scene", action="append", default=[], help="NAME=K cap per scene before the draw")
    parser.add_argument("--exclude-scenes", default="", help="comma-separated scene dir names to leave out")
    parser.add_argument("--exclude-holdout", action="store_true", help="leave out EXP-23's six held-out R2R scenes")
    parser.add_argument("--stop-upsample", type=int, default=1, help="copies of every drawn STOP sample (InternNav: 5)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--vlm-image-size", type=int, nargs=2, default=list(DEFAULT_VLM_IMAGE_SIZE))
    parser.add_argument("--jpeg-quality", type=int, default=90, help="the client's --rpc_jpeg_quality")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="select and print the counts, write nothing")
    args = parser.parse_args(argv)

    if args.shard_size % args.global_batch or args.global_batch % args.world_size:
        raise SystemExit("shard size must be a multiple of the global batch, and that of the world size")
    out_dir = Path(args.out_dir)
    if not args.dry_run and (out_dir / MANIFEST_NAME).exists():
        raise SystemExit(f"{out_dir} already holds a stream")
    started = time.time()
    roots = {k: Path(v) for k, v in _pairs(args.root, "--root").items()}
    samples = {k: int(v) for k, v in _pairs(args.samples, "--samples").items()}
    caps = {k: int(v) for k, v in _pairs(args.max_per_scene, "--max-per-scene").items()}
    unknown = (set(samples) | set(caps)) - set(roots)
    if unknown:
        raise SystemExit(f"--samples/--max-per-scene name roots that --root does not: {sorted(unknown)}")
    excluded = {s for s in args.exclude_scenes.split(",") if s}
    if args.exclude_holdout:
        excluded |= set(holdout_scenes())

    pools: dict[str, dict[str, list]] = defaultdict(dict)
    skipped: Counter = Counter()
    for name, root in roots.items():
        for scene_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            if scene_dir.name in excluded:
                skipped[f"{name}_excluded_scenes"] += 1
                continue
            if not (scene_dir / "meta").is_dir():
                skipped[f"{name}_not_a_scene"] += 1
                continue
            pools[name][scene_dir.name] = [(scene_dir, s) for s in scene_samples(scene_dir)]

    class _Kinded:  # select() only needs .kind; carry (scene_dir, sample) through it
        __slots__ = ("kind", "value")

        def __init__(self, value):
            self.value, self.kind = value, value[1].kind

    wrapped = {r: {s: [_Kinded(x) for x in items] for s, items in scenes.items()} for r, scenes in pools.items()}
    chosen, counts = select(wrapped, samples=samples, max_per_scene=caps, stop_upsample=args.stop_upsample, seed=args.seed)
    order = np.random.default_rng(args.seed + 1).permutation(len(chosen))
    total = (len(chosen) // args.shard_size) * args.shard_size
    if total == 0:
        raise SystemExit(f"{len(chosen)} samples do not fill one shard of {args.shard_size}")
    items = []
    drawn: Counter = Counter()
    for i in order[:total]:
        root, k = chosen[int(i)]
        scene_dir, sample = k.value
        item = _replay_item(scene_dir, sample)  # source stays "replay": _pack reads expert samples by it
        item["root"] = root
        items.append(item)
        drawn[root] += 1
        drawn[f"{root}_target_{sample.kind}"] += 1
    stops = sum(v for key, v in drawn.items() if key.endswith("_target_stop"))
    summary = {"drawn": dict(drawn), "selected_before_trim": len(chosen), "total_samples": total,
               "num_shards": total // args.shard_size, "stop_share": round(stops / total, 4),
               "pools": {k: v for k, v in counts.items() if k.endswith(("_pool", "_scenes"))}, "skipped": dict(skipped)}
    print(json.dumps(summary, indent=1, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    num_shards = total // args.shard_size
    jobs = [(s, items[s * args.shard_size:(s + 1) * args.shard_size], str(out_dir), tuple(args.vlm_image_size),
             args.jpeg_quality) for s in range(num_shards)]
    shards: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (index, digest, size) in enumerate(pool.map(_write, jobs, chunksize=1), 1):
            shards[index] = {"name": shard_name(index), "sha256": digest, "bytes": size}
            if done % 20 == 0 or done == num_shards:
                print(f"[stream] {done}/{num_shards} shards", flush=True)
    manifest = {
        "schema": STREAM_SCHEMA,
        "created_unix": time.time(),
        "builder": "scripts/tools/build_replay_stream.py (EXP-26)",
        "seed": args.seed,
        "shard_size": args.shard_size,
        "num_shards": num_shards,
        "total_samples": total,
        "global_batch": args.global_batch,
        "world_size": args.world_size,
        "epochs": 1.0,
        "replay_fraction": 1.0,
        "roots": {k: str(v.resolve()) for k, v in roots.items()},
        "holdout": {"scenes": sorted(excluded), "count": len(excluded)},
        "selection": {
            "rule": "EXP-26: expert labels only (interndata_replay), per-scene cap, seeded draw, STOP upsampled",
            "samples": samples, "max_per_scene": caps, "stop_upsample": args.stop_upsample,
            "counts": dict(counts), "skipped": dict(skipped), "stop_share": summary["stop_share"],
            "keys_sha256": hashlib.sha256("\n".join(i["key"] for i in items).encode()).hexdigest(),
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

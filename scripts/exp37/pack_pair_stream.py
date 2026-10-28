#!/usr/bin/env python3
"""Pack EXP-37's DPO pair stream from the pair sidecar and EXP-35's stream shards: same samples, one more field.

Each pair of ``build_pair_sidecar.py``'s ``pairs.jsonl`` names a sample of EXP-35's stream (shard and
index).  That sample's members are copied byte for byte -- history, current and lookdown JPEGs as the
server saw them -- and its json keeps every field (``turns`` is the chosen answer, ``kl`` false) and gains
two: ``rejected_turns`` (the base's own recorded answer, the format of ``turns``) and ``pair`` (source,
learner, d, call_index, classes, pixel distance, episode).  The sample's key and turns must equal the
sidecar's, and the stream must be the one the sidecar was built from: the digest of its manifest's shard
sha256s and the manifest's own sha256 are compared with the sidecar's ``summary.json``, and every shard
read is checked against its manifest sha256.

Order: sha256("exp37-order:" + key).  The trainer's reader takes whole shards only, so the stream keeps
the largest whole number of ``--shard-size`` shards (a multiple of the global batch) and drops the tail
of that order; how many, and their keys' sha256, are in the manifest.  The manifest is the trainer's
stream manifest (every key EXP-35's has; ``holdout`` copied from it) with ``"pairs": true`` and the schema
``heatmapvln-s2-pair-stream-v1``; ``keys.tsv`` lists the samples in stream order.

On the 910B (CPU only)::

    R=/mnt/sfs_turbo/zhr/zhr_1
    PYTHONPATH=$SRC:$R/rpc/src $R/envs/ppa/bin/python scripts/exp37/pack_pair_stream.py \\
        --pairs $R/exp37/pairs/pairs.jsonl --stream-dir $R/exp35/stream/main --out-dir $R/exp37/stream/main

``--limit N`` keeps the first N pairs of the order (a smoke subset; recorded in the manifest).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.exp28 import merge_streams as ms  # noqa: E402
from scripts.exp34 import build_exp34_stream as x34  # noqa: E402
from scripts.exp35 import build_exp35_stream as x35  # noqa: E402
from scripts.exp37 import build_pair_sidecar as sc  # noqa: E402
from src.data.s2_sft_dataset import decision_class, validate_turns  # noqa: E402
from src.data.s2_stream import MANIFEST_NAME, shard_digest_ok, shard_name  # noqa: E402

PAIR_SCHEMA = "heatmapvln-s2-pair-stream-v1"
BUILDER = "scripts/exp37/pack_pair_stream.py (EXP-37)"
ORDER_RANK = "exp37-order:"
SEED = 37  # recorded only (the trainer copies a stream's seed into its run key); the order is by key
KL_BY_PART = {"COR": False, "COR2": False}


def order_rank(key: str) -> str:
    return hashlib.sha256(f"{ORDER_RANK}{key}".encode("utf-8")).hexdigest()


def ordered(pairs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    keys = [p["key"] for p in pairs]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{len(keys) - len(set(keys))} pair keys repeat")
    return sorted(pairs, key=lambda p: (order_rank(p["key"]), p["key"]))


def with_pair(members: Sequence[tuple[str, bytes]], rejected: Sequence[str], pair: Mapping[str, Any], *,
              chosen: Sequence[str], key: str) -> list[tuple[str, bytes]]:
    """A sample's members with ``rejected_turns`` and ``pair`` added to its json (the first member);
    every other member and every other json field as they were."""
    suffix, raw = members[0]
    if suffix != "json":
        raise SystemExit(f"{key}: first member {suffix}, not the json")
    meta = json.loads(raw)
    if meta["key"] != key or [str(t) for t in meta["turns"]] != list(chosen):
        raise SystemExit(f"{key}: the sample is {meta['key']} {meta['turns']!r}, the pair says {key} {list(chosen)!r}")
    if meta.get("kl") is not False:
        raise SystemExit(f"{key}: kl is {meta.get('kl')!r}; a pair sample is kl false")
    if "rejected_turns" in meta or "pair" in meta:
        raise SystemExit(f"{key}: the sample is already a pair")
    validate_turns(rejected, what=f"{key} rejected")
    meta["rejected_turns"] = [str(t) for t in rejected]
    meta["pair"] = dict(pair)
    return [("json", json.dumps(meta, ensure_ascii=False).encode("utf-8")), *members[1:]]


def _index_job(job: tuple[str, str | None]) -> tuple[str, list[ms.Entry]]:
    path, expected = job
    if expected is not None and not shard_digest_ok(Path(path), expected):
        raise SystemExit(f"{path}: sha256 differs from the manifest")
    return Path(path).name, ms.index_shard(path)


def _write_job(job: tuple) -> tuple[int, str, int]:
    index, slots, out_dir = job
    samples = [with_pair(x35._members_of_entry(entry), p["rejected_turns"], p["pair"], chosen=p["chosen_turns"],
                         key=p["key"]) for entry, p in slots]
    path = Path(out_dir) / shard_name(index)
    return index, x35.write_shard(path, samples), path.stat().st_size


def drawn_of(pairs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """``drawn`` as every stream builder writes it (the trainer logs it)."""
    drawn: Counter = Counter()
    for p in pairs:
        drawn["dagger"] += 1
        drawn[f"dagger_target_{decision_class(p['chosen_turns'][-1])}"] += 1
        drawn[f"category_{p['part']}"] += 1
    return dict(drawn)


def keys_sha256(pairs: Sequence[Mapping[str, Any]]) -> str:
    """sha256 of key|chosen|rejected (turns joined by \\x1f) of the samples, in stream order."""
    return hashlib.sha256("\n".join(p["key"] + "|" + "\x1f".join(p["chosen_turns"]) + "|"
                                    + "\x1f".join(p["rejected_turns"]) for p in pairs).encode()).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", required=True, help="build_pair_sidecar.py's pairs.jsonl (summary.json beside it)")
    parser.add_argument("--stream-dir", required=True, help="EXP-35's stream (the shards the sidecar names)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--world-size", type=int, default=16, help="recorded only (EXP-37: 2 nodes x 8)")
    parser.add_argument("--limit", type=int, default=0, help="keep the first N pairs of the order (a smoke subset)")
    parser.add_argument("--no-verify-shards", action="store_true", help="skip the sha256 of the shards read")
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.shard_size % args.global_batch or args.global_batch % args.world_size:
        raise SystemExit("shard size must be a multiple of the global batch, and that of the world size")
    out_dir = Path(args.out_dir)
    if (out_dir / MANIFEST_NAME).exists():
        raise SystemExit(f"{out_dir} already holds a stream")
    started = time.time()
    pairs_path = Path(args.pairs)
    pairs_text = pairs_path.read_text(encoding="utf-8")
    summary_raw = (pairs_path.parent / sc.SUMMARY_NAME).read_bytes()
    summary = json.loads(summary_raw)
    if summary.get("schema") != sc.SIDECAR_SCHEMA:
        raise SystemExit(f"{pairs_path.parent}: sidecar schema {summary.get('schema')!r}")
    if hashlib.sha256(pairs_text.encode("utf-8")).hexdigest() != summary["pairs_sha256"]:
        raise SystemExit(f"{pairs_path} does not hash to its summary's pairs_sha256")
    manifest, identity = sc.stream_identity(args.stream_dir)
    for field in ("manifest_sha256", "shards_sha256", "total_samples", "shard_size"):
        if identity[field] != summary["stream"][field]:
            raise SystemExit(f"{args.stream_dir}: {field} {identity[field]} is not the sidecar's "
                             f"{summary['stream'][field]} (another stream)")
    pairs = [json.loads(line) for line in pairs_text.splitlines() if line]
    for p in pairs:
        if p["pair"]["learner"] != "base" or p["part"] not in KL_BY_PART:
            raise SystemExit(f"{p['key']}: learner {p['pair']['learner']!r}, part {p['part']!r}")
    chosen = ordered(pairs)
    if args.limit:
        chosen = chosen[: args.limit]
    total = len(chosen) // args.shard_size * args.shard_size
    tail = chosen[total:]
    chosen = chosen[:total]
    if not chosen:
        raise SystemExit(f"{len(tail)} pairs do not fill one shard of {args.shard_size}")

    # --- index the source shards the pairs sit in (each checked against EXP-35's manifest)
    sha_of = {s["name"]: s["sha256"] for s in manifest["shards"]}
    needed = sorted({p["shard"] for p in chosen})
    jobs = [(str(Path(args.stream_dir) / name), None if args.no_verify_shards else sha_of[name]) for name in needed]
    entries: dict[str, list[ms.Entry]] = dict(x34._map(_index_job, jobs, args.workers))
    print(f"[exp37] {len(pairs)} pairs in the sidecar, {len(chosen)} packed ({len(tail)} dropped as the tail), "
          f"{len(needed)} source shards indexed", flush=True)

    # --- write
    out_dir.mkdir(parents=True, exist_ok=True)
    num_shards = total // args.shard_size
    write_jobs = []
    for s in range(num_shards):
        slots = []
        for p in chosen[s * args.shard_size:(s + 1) * args.shard_size]:
            entry = entries[p["shard"]][int(p["index"])]
            if entry.key != p["key"]:
                raise SystemExit(f"{p['shard']}#{p['index']} is {entry.key}, the sidecar says {p['key']}")
            slots.append((entry, p))
        write_jobs.append((s, slots, str(out_dir)))
    shards: dict[int, dict] = {}
    for index, digest, size in x34._map(_write_job, write_jobs, args.workers):
        shards[index] = {"name": shard_name(index), "sha256": digest, "bytes": size}
    keys_text = "".join(f"{p['key']}\t{p['part']}\tfalse\t{p['shard']}#{p['index']}\t{p['record']}\t"
                        f"{p['pair']['d']:.4f}\n" for p in chosen)
    (out_dir / x34.KEYS_TSV).write_text(keys_text, encoding="utf-8")

    sources = sorted({p["pair"]["source"] for p in chosen})
    roots = summary["sources"]
    manifest_out = {
        "schema": PAIR_SCHEMA,
        "pairs": True,
        "created_unix": time.time(),
        "builder": BUILDER,
        "seed": SEED,
        "shard_size": args.shard_size,
        "num_shards": num_shards,
        "total_samples": total,
        "global_batch": args.global_batch,
        "world_size": args.world_size,
        "epochs": 1.0,
        "total_samples_requested": 0,
        "replay_fraction": 0.0,
        "holdout": manifest["holdout"],
        "record_roots": [roots[s]["record_root"] for s in sources],
        "label_roots": [roots[s]["label_root"] for s in sources],
        "replay_root": "",
        "kl": {"field": "kl", "where": "each sample's {i:06d}.json in the shard (also 'part'; keys.tsv column 3)",
               "by_part": KL_BY_PART},
        "pair_fields": {"chosen": "turns", "rejected": "rejected_turns", "meta": "pair"},
        "source_stream": {**identity, "keys_tsv_sha256": manifest["selection"]["keys_tsv_sha256"],
                          "shards_verified": not args.no_verify_shards},
        "sidecar": {"pairs": str(pairs_path.resolve()), "pairs_sha256": summary["pairs_sha256"],
                    "summary_sha256": hashlib.sha256(summary_raw).hexdigest(), "pairs_in_sidecar": len(pairs),
                    "settings": summary["settings"], "counts": summary["counts"]},
        "selection": {
            "rule": "EXP-37: the sidecar's pairs (EXP-35 COR/COR2 rows whose learner is the base; d > 3 m, "
                    "call < 48, neither side STOP, pixel pairs >= 25 px), each EXP-35 sample copied byte for byte "
                    "with rejected_turns and pair added to its json; order sha256('exp37-order:' + key); whole "
                    "shards only, the tail of the order dropped",
            "order_rank": f"sha256('{ORDER_RANK}' + key)",
            "limit": args.limit,
            "dropped_tail": len(tail),
            "dropped_tail_keys_sha256": hashlib.sha256("\n".join(p["key"] for p in tail).encode()).hexdigest()
            if tail else None,
            "stats": sc.pair_stats(chosen),
            "keys_sha256": keys_sha256(chosen),
            "keys_tsv_sha256": hashlib.sha256(keys_text.encode("utf-8")).hexdigest(),
        },
        "drawn": drawn_of(chosen),
        "shards": [shards[s] for s in range(num_shards)],
        "build_s": round(time.time() - started, 1),
    }
    tmp = out_dir / f".{MANIFEST_NAME}.tmp"
    tmp.write_text(json.dumps(manifest_out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out_dir / MANIFEST_NAME)
    print(json.dumps({k: manifest_out[k] for k in ("num_shards", "total_samples", "build_s")}
                     | {"dropped_tail": len(tail)}, sort_keys=True))
    print("trainer: --holdout-scenes " + ",".join(manifest["holdout"]["scenes"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

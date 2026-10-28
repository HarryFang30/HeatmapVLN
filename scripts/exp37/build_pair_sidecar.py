#!/usr/bin/env python3
"""EXP-37's preference pairs, as a sidecar to EXP-35's stream: chosen = the stream's correction target,
rejected = the base's own recorded greedy answer on the same state.

The ledger's EXP-37 (written 2026-10-28 before any EXP-37 training, gate or val reading).  A pair is one
correction row of EXP-35's training stream -- EXP-34 D1's ``COR`` (line B on P_old: ``envdrop``, ``r2r``,
``hm3d``) or D2's ``COR2`` (line C) -- whose learner is the base model (no adapter served the record:
P_old and ``envdrop2``; ``envdrop3``'s learner was X34M, so its answer is not the reference model's own
and its rows are left out), and:

* ``d > 3 m`` (the label's geodesic, the d every builder and gate uses) and ``call_index < 48``;
* neither side is a STOP (``decision_class`` of the final turn, the server's reading, fallback included;
  a turn string holding a STOP counts as one);
* a pixel -> pixel pair whose two goals are under 25 px apart is dropped.

``chosen`` is the stream sample's ``turns`` (read from the shard) and must equal the label row's
``target_turn1/2``; ``rejected`` is the label row's ``model_turn1/2`` and must equal ``recorded_turns`` of
the record's call.  Both must pass ``validate_turns``, the two must differ, and every record of a base
source is checked to have been served without an adapter (``hm3d_route_gate.served_adapter``, as the
builders do).  EXP-35's D1 rows name only ``d1:stream-NNNNNN.tar#i``; they are joined through EXP-34's
``keys.tsv`` (same key required).

Output, ``--out-dir``:

* ``pairs.jsonl`` -- one row per pair in stream order: the EXP-35 shard and index of the sample (the
  910B's copy of the stream holds the same shards: ``summary.json`` records the manifest's sha256 and the
  digest of its shard sha256s, and ``pack_pair_stream.py`` refuses a stream that differs), ``key``,
  ``chosen_turns``, ``rejected_turns`` and the pair meta of the contract (``pair``);
* ``summary.json`` -- counts by source, band, pair type (rejected -> chosen class), drops by reason.

On the 4090 (CPU only)::

    PYTHONPATH=$WT:/workspace/rpc/src /opt/conda/bin/python scripts/exp37/build_pair_sidecar.py \\
        --stream /workspace/exp35/stream --d1-stream /workspace/exp34/stream/main \\
        --out-dir /workspace/exp37/pairs --workers 24
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.exp28.hm3d_route_gate import served_adapter  # noqa: E402
from scripts.exp34 import build_exp34_stream as x34  # noqa: E402
from scripts.exp35 import build_exp35_stream as x35  # noqa: E402
from src.data.s2_records import EpisodeRecord, episode_key, label_path_for, read_labels  # noqa: E402
from src.data.s2_sft_dataset import decision_class, pixel_uv, record_device, recorded_turns, validate_turns  # noqa: E402
from src.data.s2_stream import MANIFEST_NAME, shard_digest_ok  # noqa: E402

SIDECAR_SCHEMA = "heatmapvln-exp37-pair-sidecar-v1"
BUILDER = "scripts/exp37/build_pair_sidecar.py (EXP-37)"
PAIRS_NAME = "pairs.jsonl"
SUMMARY_NAME = "summary.json"
FAR_M = 3.0
FIRST_CALLS = 48
MIN_PIXEL_PX = 25.0
CORRECTION_PARTS = ("COR", "COR2")
BANDS = (("3-6m", 3.0, 6.0), ("6-10m", 6.0, 10.0), (">10m", 10.0, math.inf))
PIXEL_BINS = (25.0, 50.0, 100.0, 200.0, math.inf)


# --------------------------------------------------------------------------- #
# One pair
# --------------------------------------------------------------------------- #
def coarse(cls: str) -> str:
    """``pixel`` / ``turn`` / ``stop`` / ... of a ``decision_class``."""
    return "turn" if cls in ("turn_left", "turn_right") else cls


def band(d: float) -> str:
    for name, lo, hi in BANDS:
        if lo < d <= hi:
            return name
    return "<=3m"


def pixel_distance(rejected: Sequence[str], chosen: Sequence[str]) -> float | None:
    """Euclidean distance in lookdown pixels of two pixel answers; None unless both are pixels."""
    if decision_class(rejected[-1]) != "pixel" or decision_class(chosen[-1]) != "pixel":
        return None
    (ru, rv), (cu, cv) = pixel_uv(rejected[-1]), pixel_uv(chosen[-1])
    return round(math.hypot(ru - cu, rv - cv), 3)


def judge_pair(
    rejected: Sequence[str],
    chosen: Sequence[str],
    *,
    d: float | None,
    call_index: int,
    far_m: float = FAR_M,
    first_calls: int = FIRST_CALLS,
    min_pixel_px: float = MIN_PIXEL_PX,
) -> tuple[str | None, dict[str, Any]]:
    """(why the pair is dropped or None, its classes and pixel distance)."""
    meta = {"chosen_class": decision_class(chosen[-1]), "rejected_class": decision_class(rejected[-1]),
            "pixel_dist": pixel_distance(rejected, chosen)}
    if int(call_index) >= first_calls:
        return "beyond_first_calls", meta
    if d is None or not math.isfinite(float(d)) or not float(d) > far_m:
        return "within_far_radius", meta
    if meta["rejected_class"] == "stop":
        return "rejected_stop", meta
    if meta["chosen_class"] == "stop":
        return "chosen_stop", meta
    if x34.turn_with_stop(rejected):
        return "rejected_turn_with_stop", meta
    if x34.turn_with_stop(chosen):
        return "chosen_turn_with_stop", meta
    if tuple(rejected) == tuple(chosen):
        return "rejected_equals_chosen", meta
    if meta["pixel_dist"] is not None and meta["pixel_dist"] < min_pixel_px:
        return "pixel_pair_within_min_px", meta
    return None, meta


def pair_meta(*, source: str, d: float, call_index: int, judged: Mapping[str, Any], episode: str) -> dict[str, Any]:
    """The ``pair`` field of a pair sample (EXP-37's fixed contract)."""
    return {"source": source, "learner": "base", "d": round(float(d), 4), "call_index": int(call_index),
            "chosen_class": judged["chosen_class"], "rejected_class": judged["rejected_class"],
            "pixel_dist": judged["pixel_dist"], "episode": episode}


def pair_type(pair: Mapping[str, Any]) -> str:
    return f"{coarse(pair['rejected_class'])}->{coarse(pair['chosen_class'])}"


def label_turns(first: Any, second: Any) -> tuple[str, ...] | None:
    if first is None:
        return None
    return (str(first),) + ((str(second),) if second is not None else ())


def key_source(key: str) -> tuple[str, int]:
    """(source, call index) of a DAgger key ``{scene}_{episode:06d}#{source}.{call}``."""
    source, call = key.rsplit("#", 1)[1].rsplit(".", 1)
    return source, int(call)


# --------------------------------------------------------------------------- #
# The streams
# --------------------------------------------------------------------------- #
def stream_identity(stream_dir: str | Path) -> tuple[dict, dict[str, Any]]:
    """(manifest, what identifies it: manifest sha256, digest of its shard sha256s -- the trainer's
    ``stream_sha256`` -- size and ``keys.tsv``)."""
    raw = (Path(stream_dir) / MANIFEST_NAME).read_bytes()
    manifest = json.loads(raw)
    return manifest, {
        "dir": str(Path(stream_dir).resolve()),
        "builder": manifest.get("builder"),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "shards_sha256": hashlib.sha256("".join(s["sha256"] for s in manifest["shards"]).encode()).hexdigest(),
        "total_samples": int(manifest["total_samples"]),
        "shard_size": int(manifest["shard_size"]),
        "num_shards": int(manifest["num_shards"]),
    }


def read_keys_tsv(stream_dir: str | Path, manifest: Mapping[str, Any]) -> list[list[str]]:
    text = (Path(stream_dir) / x34.KEYS_TSV).read_text(encoding="utf-8")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != manifest["selection"]["keys_tsv_sha256"]:
        raise SystemExit(f"{stream_dir}/{x34.KEYS_TSV} does not hash to the manifest's keys_tsv_sha256")
    rows = [line.split("\t") for line in text.splitlines()]
    if len(rows) != int(manifest["total_samples"]):
        raise SystemExit(f"{stream_dir}/{x34.KEYS_TSV}: {len(rows)} rows, the manifest says {manifest['total_samples']}")
    return rows


def _shard_metas(job: tuple[str, str | None]) -> tuple[str, list[dict]]:
    """Every sample's json of one shard, in order; the shard's sha256 checked when one is given."""
    path, expected = job
    if expected is not None and not shard_digest_ok(Path(path), expected):
        raise SystemExit(f"{path}: sha256 differs from the manifest")
    metas: dict[int, dict] = {}
    with tarfile.open(path, mode="r:") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".json"):
                metas[int(member.name.split(".", 1)[0])] = json.loads(tar.extractfile(member).read())
    if sorted(metas) != list(range(len(metas))):
        raise SystemExit(f"{path}: sample numbering has a gap")
    return path, [metas[i] for i in range(len(metas))]


def stream_metas(stream_dir: str | Path, manifest: Mapping[str, Any], *, verify: bool, workers: int) -> list[dict]:
    """Every sample's json of a stream, in stream order (every shard's sha256 checked with ``verify``)."""
    jobs = [(str(Path(stream_dir) / s["name"]), s["sha256"] if verify else None) for s in manifest["shards"]]
    out: list[dict] = []
    for path, metas in x34._map(_shard_metas, jobs, workers):
        if len(metas) != int(manifest["shard_size"]):
            raise SystemExit(f"{path}: {len(metas)} samples, the shard size is {manifest['shard_size']}")
        out += metas
    return out


def source_table(d1_manifest: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per source name: record root, label root, the stream that took it, and its learner (``base`` or
    the adapter's sha256).  EXP-34's builder refuses adapter-served records, so its sources are the base's."""
    table: dict[str, dict[str, Any]] = {}
    for name, s in d1_manifest["selection"]["sources"].items():
        table[name] = {"record_root": s["record_root"], "label_root": s["label_root"], "adapter": "base", "stream": "EXP-34"}
    for name, s in manifest["selection"]["sources"].items():
        if name in table:
            raise SystemExit(f"source {name} is in both streams")
        table[name] = {"record_root": s["record_root"], "label_root": s["label_root"], "adapter": s["adapter"],
                       "stream": "EXP-35"}
    return table


def candidates_of(
    rows: Sequence[Sequence[str]],
    metas: Sequence[Mapping[str, Any]],
    d1_rows: Sequence[Sequence[str]],
    *,
    shard_size: int,
    d1_shard_size: int,
    sources: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Counter]:
    """The stream's correction rows whose learner is the base, with their record, call and chosen answer."""
    out: list[dict[str, Any]] = []
    counts: Counter = Counter()
    for position, (row, meta) in enumerate(zip(rows, metas, strict=True)):
        key, part = row[0], row[1]
        if meta["key"] != key or meta.get("part") != part:
            raise SystemExit(f"stream sample {position}: {meta['key']} {meta.get('part')} in the shard, "
                             f"{key} {part} in keys.tsv")
        if part not in CORRECTION_PARTS:
            continue
        source, call = key_source(key)
        counts[f"rows_{part}"] += 1
        counts[f"rows_{part}:{source}"] += 1
        if meta.get("kl") is not False:
            raise SystemExit(f"{key}: a {part} sample with kl {meta.get('kl')!r}")
        where, d_tsv = row[3], None
        if where.startswith("d1:"):
            shard, index = where[len("d1:"):].rsplit("#", 1)
            joined = d1_rows[int(shard[len("stream-"):-len(".tar")]) * d1_shard_size + int(index)]
            if joined[0] != key or joined[1] != part:
                raise SystemExit(f"{key}: {where} is {joined[0]} {joined[1]} in EXP-34's keys.tsv")
            where, d_tsv = joined[3], float(joined[5])
        record_path, call_index = where.rsplit("#", 1)
        if int(call_index) != call:
            raise SystemExit(f"{key}: keys.tsv names call {call_index}")
        if source not in sources:
            raise SystemExit(f"{key}: source {source} is in neither manifest")
        if sources[source]["adapter"] != "base":
            counts["dropped:learner_not_base"] += 1
            counts[f"dropped:learner_not_base:{source}"] += 1
            continue
        if Path(record_path).parent.parent != Path(sources[source]["record_root"]):
            raise SystemExit(f"{key}: {record_path} is not under {source}'s record root")
        out.append({"position": position, "shard": f"stream-{position // shard_size:06d}.tar",
                    "index": position % shard_size, "key": key, "part": part, "source": source,
                    "record_path": record_path, "call_index": call, "chosen": tuple(str(t) for t in meta["turns"]),
                    "d_tsv": d_tsv, "label_root": sources[source]["label_root"]})
    return out, counts


# --------------------------------------------------------------------------- #
# Records and labels
# --------------------------------------------------------------------------- #
def _judge_record(job: tuple[str, list[dict[str, Any]], dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    """Pairs (or why not) of one record's candidates; raises on anything the builders should have refused."""
    record_path, cands, settings = job
    record = EpisodeRecord(record_path)
    adapter = served_adapter(record.episode)
    if adapter is not None:
        raise SystemExit(f"{record_path}: served with System2 LoRA {adapter}, its source names the base model")
    calls = {int(c["call_index"]): c for c in record.calls}
    rows = {int(r["call_index"]): r for r in read_labels(label_path_for(record_path, cands[0]["label_root"]))}
    episode = episode_key(record.scene_id, record.episode_id)
    out: list[dict[str, Any]] = []
    counts: Counter = Counter()
    for c in cands:
        what = f"{c['key']} ({record_path}#{c['call_index']})"
        row = rows[c["call_index"]]
        if row.get("kind") != "correction" or row.get("agree") is not False:
            raise SystemExit(f"{what}: a {c['part']} sample whose label row is {row.get('kind')} agree={row.get('agree')}")
        chosen = validate_turns(c["chosen"], what=f"{what} chosen")
        if label_turns(row.get("target_turn1"), row.get("target_turn2")) != chosen:
            raise SystemExit(f"{what}: stream turns {chosen!r}, label target "
                             f"{label_turns(row.get('target_turn1'), row.get('target_turn2'))!r}")
        rejected = label_turns(row.get("model_turn1"), row.get("model_turn2"))
        if rejected is None:
            raise SystemExit(f"{what}: the label row has no model_turn1")
        rejected = validate_turns(rejected, what=f"{what} rejected")
        if recorded_turns(calls[c["call_index"]]) != rejected:
            raise SystemExit(f"{what}: label model turns {rejected!r}, record {recorded_turns(calls[c['call_index']])!r}")
        d = row.get("distance_to_goal")
        d = float(d) if d is not None else None
        if c["d_tsv"] is not None and (d is None or abs(d - c["d_tsv"]) > 1e-4):
            raise SystemExit(f"{what}: label d {d}, EXP-34 keys.tsv d {c['d_tsv']}")
        why, judged = judge_pair(rejected, chosen, d=d, call_index=c["call_index"], **settings)
        if why is not None:
            counts[f"dropped:{why}"] += 1
            counts[f"dropped:{why}:{c['source']}"] += 1
            if why == "pixel_pair_within_min_px":
                counts["dropped_pixel_pair_px_lt_10"] += judged["pixel_dist"] < 10.0
            continue
        out.append({
            "position": c["position"], "shard": c["shard"], "index": c["index"], "key": c["key"], "part": c["part"],
            "record": f"{record_path}#{c['call_index']}",
            "device": record_device(record),
            "chosen_turns": list(chosen), "rejected_turns": list(rejected),
            "pair": pair_meta(source=c["source"], d=d, call_index=c["call_index"], judged=judged, episode=episode),
        })
    return out, counts


def pair_stats(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts of kept pairs: by source, band, type, part, device; pixel distances of pixel -> pixel pairs."""
    pixel = Counter()
    for p in pairs:
        dist = p["pair"]["pixel_dist"]
        if dist is not None:
            hi = next(b for b in PIXEL_BINS if dist < b)
            pixel[f"<{hi:g}px" if math.isfinite(hi) else f">={PIXEL_BINS[-2]:g}px"] += 1
    return {
        "pairs": len(pairs),
        "by_source": dict(sorted(Counter(p["pair"]["source"] for p in pairs).items())),
        "by_part": dict(sorted(Counter(p["part"] for p in pairs).items())),
        "by_band": dict(sorted(Counter(band(p["pair"]["d"]) for p in pairs).items())),
        "by_type": dict(sorted(Counter(pair_type(p["pair"]) for p in pairs).items())),
        "by_source_type": dict(sorted(Counter(f"{p['pair']['source']}/{pair_type(p['pair'])}" for p in pairs).items())),
        "by_source_band": dict(sorted(Counter(f"{p['pair']['source']}/{band(p['pair']['d'])}" for p in pairs).items())),
        "by_device": dict(sorted(Counter(p.get("device", "?") for p in pairs).items())),
        "pixel_pair_distance": dict(pixel),
        "rejected_mixed_turn": sum(1 for p in pairs if coarse(p["pair"]["rejected_class"]) == "turn"
                                   and "←" in p["rejected_turns"][-1] and "→" in p["rejected_turns"][-1]),
        "episodes": len({p["pair"]["episode"] for p in pairs}),
        "max_pairs_per_episode": max(Counter(p["pair"]["episode"] for p in pairs).values(), default=0),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stream", required=True, help="EXP-35's stream directory (build_exp35_stream.py, keys.tsv)")
    parser.add_argument("--d1-stream", required=True, help="EXP-34's stream directory (its keys.tsv joins D1 rows)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--far-m", type=float, default=FAR_M)
    parser.add_argument("--first-calls", type=int, default=FIRST_CALLS)
    parser.add_argument("--min-pixel-px", type=float, default=MIN_PIXEL_PX)
    parser.add_argument("--no-verify-shards", action="store_true", help="skip the sha256 of every shard read")
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    if (out_dir / SUMMARY_NAME).exists():
        raise SystemExit(f"{out_dir} already holds a sidecar")
    started = time.time()
    manifest, identity = stream_identity(args.stream)
    d1_manifest, d1_identity = stream_identity(args.d1_stream)
    if manifest.get("builder") != x35.BUILDER or d1_manifest.get("builder") != x34.BUILDER:
        raise SystemExit(f"streams built by {manifest.get('builder')!r} / {d1_manifest.get('builder')!r}, "
                         f"not {x35.BUILDER!r} / {x34.BUILDER!r}")
    if manifest["d1"]["keys_sha256"] != d1_manifest["selection"]["keys_sha256"]:
        raise SystemExit(f"{args.stream} was built from another D1 than {args.d1_stream}")
    rows = read_keys_tsv(args.stream, manifest)
    d1_rows = read_keys_tsv(args.d1_stream, d1_manifest)
    sources = source_table(d1_manifest, manifest)
    metas = stream_metas(args.stream, manifest, verify=not args.no_verify_shards, workers=args.workers)
    print(f"[exp37] {args.stream}: {len(metas)} samples read; sources "
          f"{ {k: v['adapter'][:12] for k, v in sources.items()} }", flush=True)
    cands, counts = candidates_of(rows, metas, d1_rows, shard_size=identity["shard_size"],
                                  d1_shard_size=d1_identity["shard_size"], sources=sources)
    settings = {"far_m": args.far_m, "first_calls": args.first_calls, "min_pixel_px": args.min_pixel_px}
    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cands:
        by_record[c["record_path"]].append(c)
    pairs: list[dict[str, Any]] = []
    jobs = [(path, by_record[path], settings) for path in sorted(by_record)]
    for found, c in x34._map(_judge_record, jobs, args.workers, chunksize=8):
        pairs += found
        counts.update(c)
    pairs.sort(key=lambda p: p["position"])
    counts["candidates_base_learner"] = len(cands)
    counts["records_checked"] = len(jobs)

    out_dir.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n" for p in pairs)
    (out_dir / PAIRS_NAME).write_text(text, encoding="utf-8")
    stats = pair_stats(pairs)
    summary = {
        "schema": SIDECAR_SCHEMA,
        "builder": BUILDER,
        "created_unix": time.time(),
        "stream": {**identity, "keys_tsv_sha256": manifest["selection"]["keys_tsv_sha256"],
                   "holdout": manifest["holdout"]},
        "d1_stream": {**d1_identity, "keys_tsv_sha256": d1_manifest["selection"]["keys_tsv_sha256"]},
        "shards_verified": not args.no_verify_shards,
        "sources": sources,
        "settings": settings,
        "rule": "EXP-37: COR (EXP-34 D1) and COR2 (EXP-35 D2) rows of EXP-35's stream whose record the base served; "
                "chosen = the stream's turns (= label target), rejected = label model_turn1/2 (= recorded_turns); "
                "label d > far_m, call_index < first_calls, neither side STOP (a turn string holding a STOP "
                "included), pixel -> pixel pairs under min_pixel_px dropped",
        "counts": dict(sorted(counts.items())),
        "stats": stats,
        "pairs_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "keys_sha256": hashlib.sha256("\n".join(p["key"] for p in pairs).encode("utf-8")).hexdigest(),
        "build_s": round(time.time() - started, 1),
    }
    tmp = out_dir / f".{SUMMARY_NAME}.tmp"
    tmp.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out_dir / SUMMARY_NAME)
    print(f"[exp37] pairs {stats['pairs']}  by source {stats['by_source']}\n[exp37] by type {stats['by_type']}  "
          f"by band {stats['by_band']}\n[exp37] drops "
          f"{ {k: v for k, v in sorted(counts.items()) if k.startswith('dropped:') and k.count(':') == 1} }", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

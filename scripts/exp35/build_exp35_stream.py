#!/usr/bin/env python3
"""Pack EXP-35's training stream: EXP-34's stream (D1) as packed, plus D2 from line-C labels.

D1 is read from its directory, every shard checked against its manifest, and every sample is
copied byte for byte, its ``kl`` and ``part`` included.  D2 comes from line-C labels
(``label_route_corrections.py --terminal_stop_radius_m 1.0``) of the round-2 records, every
sample taken once, weight 1:

====  =====  ==========================================================================
part  kl     what is taken
====  =====  ==========================================================================
COR2  false  every line-C ``correction`` row (the oracle's pixel goal, turn, or STOP
             within the STOP radius): at most 24 per episode, the earliest
AGR2  false  line-C ``anchor`` rows (the learner's own answer, verbatim): per episode the
             6 lowest of sha256("exp34-agr:" + key), then the lowest len(COR2) of those
             (``build_exp34_stream.draw_dagger``).  kl false: the anchor is the round-2
             learner's answer the expert agreed with, not the base's
REP2  true   InternData r2r expert samples not already in D1, every STOP 5 times, whole
             units in a seeded order until they make up 20% of D2, rounded up to whole
             shards (``build_exp34_stream.compose``, no STOP band)
====  =====  ==========================================================================

As in EXP-34: only ``call_index < 48``; a target turning net against the turn the episode's
previous call executed, and a turn string with a STOP in it, are dropped; held-out scenes and
records of scenes that are not training scenes are refused; every drop is counted.  Refused too:
a row that is not line C or not this labeller's (``RouteLabelConfig(terminal_stop_radius_m=R)``),
a STOP target farther than R or a STOP anchor beyond 3 m, a record whose serving adapter is not the
one its ``--source`` names (``base`` or a sha256), and a D2 source name D1 already used.
``--holdout-cohort-shards`` (required, and it must match some record) keeps the offline gate's
episodes out (EXP-35: shards 000-003 of ``collect_envdrop3``).  A STOP anchor whose text is more
than a plain ``STOP`` is dropped, as EXP-34's TERM.

The two parts go into one order, the sha256 of ("exp35-order:" + key + "#" + copy number), and
are re-sharded; only D2's new samples are encoded (``build_s2_train_stream._pack``, as EXP-34).

On the 4090 (CPU only)::

    PYTHONPATH=$WT:/workspace/rpc/src /opt/conda/bin/python scripts/exp35/build_exp35_stream.py \\
        --d1-stream /workspace/exp34/stream/main \\
        --source envdrop3 /workspace/exp34/collect_envdrop3/records /workspace/exp35/labels/envdrop3 <X34M sha256> \\
        --source envdrop2 /workspace/exp34/collect_envdrop2/records /workspace/exp35/labels/envdrop2 base \\
        --holdout-cohort-shards /workspace/exp34/collect_envdrop3/cohorts 000,001,002,003 \\
        --replay-root /workspace/exp23/data/interndata_n1/r2r --workers 24 --out-dir /workspace/exp35/stream

``--dry-run`` stops after the selection and prints the page of statistics.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import math
import os
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.exp28 import merge_streams as ms  # noqa: E402
from scripts.exp34 import build_exp34_stream as x34  # noqa: E402
from scripts.tools import build_route_stream as brs  # noqa: E402
from scripts.tools import build_s2_train_stream as bst  # noqa: E402
from src.data.s2_records import label_path_for, read_labels  # noqa: E402
from src.data.s2_route_labels import TERMINAL_LABEL_LINE, RouteLabelConfig  # noqa: E402
from src.data.s2_sft_dataset import (  # noqa: E402
    DEFAULT_VLM_IMAGE_SIZE,
    STOP_TEXT,
    CallRef,
    decision_class,
    index_calls,
    training_turns,
)
from src.data.s2_stream import MANIFEST_NAME, STREAM_SCHEMA, PackedSample, shard_name  # noqa: E402

BUILDER = "scripts/exp35/build_exp35_stream.py (EXP-35)"
PARTS = ("COR2", "AGR2", "REP2")
KL = {"COR2": False, "AGR2": False, "REP2": True}
D2_OF = {"COR": "COR2", "AGR": "AGR2", "REP": "REP2"}
ORDER_RANK = "exp35-order:"
STOP_RADIUS_M = 1.0
SUCCESS_RADIUS_M = 3.0
EPS = 1e-6


# --------------------------------------------------------------------------- #
# Candidates of one record
# --------------------------------------------------------------------------- #
def classify_record(
    refs: Sequence[CallRef],
    rows: Mapping[int, Mapping[str, Any]],
    *,
    source: str,
    stop_radius_m: float,
    first_calls: int = x34.FIRST_CALLS,
) -> tuple[list[x34.Candidate], Counter, list[tuple[float, bool]]]:
    """(candidates, counts, (d, learner answered STOP) of every call with a distance) of one record;
    candidates are ``build_exp34_stream.Candidate`` with part COR or AGR."""
    refs = sorted(refs, key=lambda r: r.call_index)
    directions = {r.call_index: brs.executed_direction(r.recorded) for r in refs}
    if refs and sorted(directions) != list(range(len(directions))):
        directions = {**brs.record_directions(refs[0].record_path), **directions}
    candidates: list[x34.Candidate] = []
    counts: Counter = Counter()
    reference: list[tuple[float, bool]] = []
    for ref in refs:
        row = rows.get(ref.call_index)
        if row is None or ref.label is None:
            counts["unlabelled"] += 1
            continue
        if row.get("line") != TERMINAL_LABEL_LINE:
            raise SystemExit(f"{ref.record_path}#{ref.call_index}: not a line-{TERMINAL_LABEL_LINE} row ({row.get('line')!r})")
        d = ref.distance_to_goal
        known = d is not None and math.isfinite(float(d))
        if known:
            reference.append((float(d), x34.is_stop(ref.recorded)))
        if ref.call_index >= first_calls:
            counts["beyond_first_calls"] += 1
            continue
        if not known:
            counts["no_distance"] += 1
            continue
        kind, reason = row.get("kind"), str(row.get("reason") or "")
        if kind == "correction":
            if ref.label.skip or ref.label.agree is not False:
                raise SystemExit(f"{ref.record_path}#{ref.call_index}: a correction row the dataset reads as "
                                 f"agree={ref.label.agree}, skip={ref.label.skip}")
            part, turns = "COR", tuple(training_turns(ref.recorded, ref.label))
            if x34.is_stop(turns) and float(d) > stop_radius_m + EPS:
                raise SystemExit(f"{ref.record_path}#{ref.call_index}: a STOP target {float(d):.3f} m out, "
                                 f"beyond the {stop_radius_m} m STOP radius")
        elif kind == "anchor":
            if ref.label.agree is not True or ref.label.target_turns not in (None, ref.recorded):
                raise SystemExit(f"{ref.record_path}#{ref.call_index}: anchor target {ref.label.target_turns!r} is "
                                 f"not the recorded answer {ref.recorded!r}")
            part, turns = "AGR", tuple(ref.recorded)
            if x34.is_stop(turns) and float(d) > SUCCESS_RADIUS_M + EPS:
                raise SystemExit(f"{ref.record_path}#{ref.call_index}: a STOP anchor {float(d):.3f} m out, "
                                 f"beyond the {SUCCESS_RADIUS_M} m success radius")
            decision = decision_class(turns[-1])
            if decision == "invalid" or (decision == "stop" and turns[-1].strip() != STOP_TEXT):
                counts["AGR_dropped_unreadable"] += 1  # as EXP-34's TERM: only a plain STOP is kept
                continue
        elif kind == "none":
            counts["none"] += 1
            counts[f"none:{reason.split('|')[0].split(':')[0]}"] += 1
            continue
        else:
            raise SystemExit(f"{ref.record_path}#{ref.call_index}: line-C row kind {kind!r}")
        counts[f"{part}_labelled"] += 1
        if x34.turn_with_stop(turns):
            counts[f"{part}_dropped_turn_with_stop"] += 1
            continue
        previous = x34.previous_direction(directions, ref.call_index)
        if previous and brs.executed_direction(turns) == -previous:
            counts[f"{part}_dropped_reversal"] += 1
            continue
        candidates.append(x34.Candidate(part, source, dataclasses.replace(ref, turns=turns),
                                        x34.sample_key(ref, source), reason.split("|")[0].split(":")[0], previous))
    return candidates, counts, reference


def cohort_episodes(cohort_dir: str, shards: Sequence[str]) -> set[tuple[str, int]]:
    """(scene, episode_id) of the named shards of a ``build_r2r_train_cohorts.py`` cohort."""
    out: set[tuple[str, int]] = set()
    for shard in shards:
        path = Path(cohort_dir) / f"shard_{int(shard):03d}.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        episodes = doc.get("episodes", doc) if isinstance(doc, dict) else doc
        if not episodes:
            raise SystemExit(f"{path} lists no episodes")
        out |= {(brs.scene_key(str(e["scene_id"])), int(e["episode_id"])) for e in episodes}
    return out


# --------------------------------------------------------------------------- #
# D1
# --------------------------------------------------------------------------- #
def entry_meta(entry: ms.Entry) -> dict[str, Any]:
    with open(entry.shard, "rb") as handle:
        name, offset, size = next(m for m in entry.members if m[0].endswith(".json"))
        handle.seek(offset)
        return json.loads(handle.read(size))


def load_d1(stream_dir: str) -> tuple[dict, list[ms.Entry], str, Counter]:
    """(manifest, entries in stream order, manifest sha256, parts) of EXP-34's stream, every shard checked."""
    manifest, entries, digest = ms.load_stream(stream_dir, schemas=[STREAM_SCHEMA], verify=True)
    if manifest.get("builder") != x34.BUILDER:
        raise SystemExit(f"{stream_dir}: built by {manifest.get('builder')!r}, not {x34.BUILDER!r}")
    keys_sha = hashlib.sha256("\n".join(e.key + "|" + "\x1f".join(e.turns) for e in entries).encode()).hexdigest()
    if keys_sha != manifest["selection"]["keys_sha256"]:
        raise SystemExit(f"{stream_dir}: the shards' keys and turns do not hash to the manifest's keys_sha256")
    parts = Counter()
    for e in entries:
        meta = entry_meta(e)
        if meta.get("part") not in x34.PARTS or meta.get("kl") is not x34.KL[meta["part"]]:
            raise SystemExit(f"{e.shard}#{e.index}: part {meta.get('part')!r} with kl {meta.get('kl')!r}")
        parts[meta["part"]] += 1
    return manifest, entries, digest, parts


# --------------------------------------------------------------------------- #
# D2 items, the order, packing
# --------------------------------------------------------------------------- #
def make_d2_items(parts: Mapping[str, Sequence]) -> list[dict]:
    items: list[dict] = []
    for part in ("COR", "AGR"):
        for c in parts[part]:
            item = bst._dagger_item(c.ref)
            item.update(key=c.key, part=D2_OF[part], kl=KL[D2_OF[part]], d=c.d, device=c.ref.record_device,
                        origin_source=c.source, reason=c.reason, previous=c.previous)
            items.append(item)
    items += [{**rep, "part": "REP2", "kl": KL["REP2"]} for rep in parts["REP"]]
    return items


def order(keys: Sequence[str]) -> list[int]:
    """Indices of ``keys`` in sha256("exp35-order:" + key + "#" + copy number) order (a STOP's five
    replay copies share a key, so the copy number spreads them)."""
    seen: Counter = Counter()
    ranks = []
    for i, key in enumerate(keys):
        ranks.append((hashlib.sha256(f"{ORDER_RANK}{key}#{seen[key]}".encode("utf-8")).hexdigest(), i))
        seen[key] += 1
    return [i for _, i in sorted(ranks)]


def _members_of_packed(sample: PackedSample, kl: bool, part: str) -> list[tuple[str, bytes]]:
    meta = {"key": sample.key, "source": sample.source, "instruction": sample.instruction, "turns": list(sample.turns),
            "vlm_image_size": list(sample.vlm_image_size), "num_history": len(sample.history_jpegs),
            "kl": bool(kl), "part": str(part)}
    out = [("json", json.dumps(meta, ensure_ascii=False).encode("utf-8"))]
    out += [(f"h{k:02d}.jpg", data) for k, data in enumerate(sample.history_jpegs)]
    return out + [("cur.jpg", sample.current_jpeg), ("look.jpg", sample.lookdown_jpeg)]


def _members_of_entry(entry: ms.Entry) -> list[tuple[str, bytes]]:
    """D1's members byte for byte, in ``write_kl_shard``'s order (json, history, current, lookdown)."""
    stem = f"{entry.index:06d}."
    blobs: dict[str, bytes] = {}
    with open(entry.shard, "rb") as handle:
        for name, offset, size in entry.members:
            handle.seek(offset)
            blobs[name[len(stem):]] = handle.read(size)
    meta = json.loads(blobs["json"])
    names = ["json"] + [f"h{k:02d}.jpg" for k in range(int(meta["num_history"]))] + ["cur.jpg", "look.jpg"]
    if sorted(names) != sorted(blobs):
        raise SystemExit(f"{entry.shard}#{entry.index}: members {sorted(blobs)}")
    return [(n, blobs[n]) for n in names]


def write_shard(path: Path, samples: Sequence[list[tuple[str, bytes]]]) -> str:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as raw:
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for i, members in enumerate(samples):
                for suffix, data in members:
                    info = tarfile.TarInfo(f"{i:06d}.{suffix}")
                    info.size = len(data)
                    info.mtime = 0
                    tar.addfile(info, io.BytesIO(data))
        raw.flush()
        os.fsync(raw.fileno())
    digest = hashlib.sha256()
    with open(tmp, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    os.replace(tmp, path)
    return digest.hexdigest()


def _write_job(job: tuple) -> tuple[int, str, int]:
    index, slots, out_dir, vlm_image_size, jpeg_quality = job
    samples = []
    for kind, value in slots:
        if kind == "d1":
            samples.append(_members_of_entry(value))
        else:
            packed = bst._pack(value, vlm_image_size, jpeg_quality)
            if packed.key != value["key"] or tuple(packed.turns) != tuple(value["turns"]):
                raise RuntimeError(f"packed {packed.key} {packed.turns!r} for item {value['key']} {value['turns']!r}")
            samples.append(_members_of_packed(packed, value["kl"], value["part"]))
    path = Path(out_dir) / shard_name(index)
    return index, write_shard(path, samples), path.stat().st_size


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def d2_stats(items: Sequence[Mapping[str, Any]], reference: Sequence[tuple[float, bool]]) -> dict[str, Any]:
    """D2's parts, its STOP shares by distance against the learner's own per-call shares, reversals."""
    by_part = Counter(i["part"] for i in items)
    stops = Counter(i["part"] for i in items if x34.is_stop(i["turns"]))
    bins: dict[str, Counter] = defaultdict(Counter)
    for item in items:
        if item["part"] != "REP2":
            b = bins[f"{item['part']}/{x34.stop_bin(float(item['d']))}"]
            b["n"] += 1
            b["stop"] += x34.is_stop(item["turns"])
    ref: dict[str, Counter] = defaultdict(Counter)
    for d, stop in reference:
        ref[x34.stop_bin(d)]["n"] += 1
        ref[x34.stop_bin(d)]["stop"] += stop
    tokens = Counter()
    for item in items:
        n = x34.supervised_tokens(item["turns"])
        tokens["all"] += n
        tokens["kl"] += n if item["kl"] else 0
    cor = [i for i in items if i["part"] == "COR2"]
    return {
        "total": len(items),
        "by_part": {p: by_part[p] for p in PARTS},
        "stop_by_part": {p: stops[p] for p in PARTS},
        "stop_share": x34._rate(sum(stops.values()), len(items)),
        "stop_by_part_and_distance": {k: {"samples": v["n"], "stop": v["stop"], "share": x34._rate(v["stop"], v["n"])}
                                      for k, v in sorted(bins.items())},
        "learner_stop_by_distance": {k: {"calls": v["n"], "stop": v["stop"], "share": x34._rate(v["stop"], v["n"])}
                                     for k, v in sorted(ref.items())},
        "reversals": sum(1 for i in items if i["part"] != "REP2" and i["previous"]
                         and brs.executed_direction(i["turns"]) == -int(i["previous"])),
        "turn_with_stop": sum(x34.turn_with_stop(i["turns"]) for i in items),
        "pixel_outside_window": sum(1 for i in items if x34.outside_window(i["turns"])),
        "cor_by_source_reason": dict(sorted(Counter(f"{i['origin_source']}/{i['reason']}" for i in cor).items())),
        "cor_by_target": dict(sorted(Counter(decision_class(i["turns"][-1]).split("_")[0] for i in cor).items())),
        "kl_token_share": x34._rate(tokens["kl"], tokens["all"]),
    }


def print_page(d1_parts: Mapping[str, int], stats: Mapping[str, Any], counts: Mapping[str, Any], total: int) -> None:
    p = stats["by_part"]
    pct = lambda x: "n/a" if x is None else f"{100 * x:.2f}%"  # noqa: E731
    lines = [
        f"D1         {dict(d1_parts)} = {sum(d1_parts.values())}",
        f"D2         COR2 {p['COR2']}  AGR2 {p['AGR2']}  REP2 {p['REP2']}  = {stats['total']}",
        f"stream     {total}",
        f"COR2       by source/reason: {stats['cor_by_source_reason']}",
        f"           by target: {stats['cor_by_target']}",
        f"STOP       D2 {pct(stats['stop_share'])}; by part {stats['stop_by_part']}",
    ]
    for name, row in stats["stop_by_part_and_distance"].items():
        lines.append(f"           {name:>10}: {row['stop']}/{row['samples']} = {pct(row['share'])}")
    for name, row in stats["learner_stop_by_distance"].items():
        lines.append(f"           learner {name:>5}: {row['stop']}/{row['calls']} = {pct(row['share'])} of its calls")
    lines += [
        f"reversals  {stats['reversals']}  turn+STOP {stats['turn_with_stop']}  pixels outside the window "
        f"{stats['pixel_outside_window']}  kl tokens {pct(stats['kl_token_share'])}",
        f"not taken  {dict(sorted((k, v) for k, v in counts.items() if not k.endswith('_labelled')))}",
    ]
    print("\n".join(f"[exp35] {line}" for line in lines), flush=True)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--d1-stream", required=True, help="EXP-34's stream directory (build_exp34_stream.py)")
    parser.add_argument("--source", nargs=4, action="append", required=True,
                        metavar=("NAME", "RECORD_ROOT", "LABEL_ROOT", "ADAPTER"),
                        help="round-2 records, their line-C labels, and who served them: 'base' or the adapter's "
                             "sha256; repeatable")
    parser.add_argument("--replay-root", required=True, help="InternData-N1 r2r scene dirs")
    parser.add_argument("--train-scenes-file", default="", help="HM3D training scenes, if a source has HM3D records")
    parser.add_argument("--holdout-scenes-file", default="", help="more held-out scenes, added to EXP-34's 26")
    parser.add_argument("--holdout-cohort-shards", nargs=2, required=True, metavar=("COHORT_DIR", "SHARDS"),
                        help="episodes of these cohort shards (comma-separated) stay out: the offline gate's")
    parser.add_argument("--stop-radius", type=float, default=STOP_RADIUS_M, help="line C's terminal_stop_radius_m")
    parser.add_argument("--first-calls", type=int, default=x34.FIRST_CALLS)
    parser.add_argument("--max-cor-per-episode", type=int, default=x34.MAX_COR_PER_EPISODE)
    parser.add_argument("--max-agr-per-episode", type=int, default=x34.MAX_AGR_PER_EPISODE)
    parser.add_argument("--replay-fraction", type=float, default=x34.REPLAY_FRACTION)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--vlm-image-size", type=int, nargs=2, default=list(DEFAULT_VLM_IMAGE_SIZE))
    parser.add_argument("--jpeg-quality", type=int, default=90, help="the client's --rpc_jpeg_quality, for replay")
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="select and print the statistics, write nothing")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.shard_size % args.global_batch or args.global_batch % args.world_size:
        raise SystemExit("shard size must be a multiple of the global batch, and that of the world size")
    out_dir = Path(args.out_dir)
    if not args.dry_run and (out_dir / MANIFEST_NAME).exists():
        raise SystemExit(f"{out_dir} already holds a stream")
    started = time.time()
    labeler = RouteLabelConfig(terminal_stop_radius_m=args.stop_radius).digest()

    # --- D1
    d1_manifest, d1_entries, d1_digest, d1_parts = load_d1(args.d1_stream)
    d1_sources = set(d1_manifest["selection"]["sources"])
    d1_replay = {e.key for e in d1_entries if e.key.startswith("replay:")}
    held = x34.held_out_scenes(args.holdout_scenes_file)
    if sorted(held) != sorted(d1_manifest["holdout"]["scenes"]):
        raise SystemExit(f"held-out scenes differ from D1's: {sorted(held ^ set(d1_manifest['holdout']['scenes']))}")
    train = x34.training_scenes(args.train_scenes_file, held)
    cohort_dir, shard_list = args.holdout_cohort_shards
    gate_episodes = cohort_episodes(cohort_dir, shard_list.split(","))
    print(f"[exp35] D1 {args.d1_stream}: {len(d1_entries)} samples {dict(d1_parts)}; "
          f"{len(gate_episodes)} gate episodes held out", flush=True)

    # --- D2: every source
    candidates: list[x34.Candidate] = []
    counts: Counter = Counter()
    reference: list[tuple[float, bool]] = []
    sources: dict[str, Any] = {}
    names = [s[0] for s in args.source]
    if len(set(names)) != len(names) or not all(x34._SOURCE_NAME.match(n) for n in names):
        raise SystemExit(f"source names must be distinct and [A-Za-z0-9-]: {names}")
    if set(names) & d1_sources:
        raise SystemExit(f"source names {sorted(set(names) & d1_sources)} are D1's: their sample keys would collide")
    for name, record_root, label_root, adapter in args.source:
        want = None if adapter == "base" else adapter
        if want is not None and not (len(want) == 64 and all(c in "0123456789abcdef" for c in want)):
            raise SystemExit(f"source {name}: ADAPTER must be 'base' or a 64-hex sha256, got {adapter!r}")
        record_root, label_root = str(Path(record_root).resolve()), str(Path(label_root).resolve())
        refs, index_stats = index_calls([record_root], [label_root], split="train", holdout=brs.holdout_match_set(held))
        refs, held_by_id = brs.drop_held_out(refs, held)
        n_gate = sum(1 for r in refs if (brs.scene_key(r.scene_id), int(r.episode_id)) in gate_episodes)
        refs = [r for r in refs if (brs.scene_key(r.scene_id), int(r.episode_id)) not in gate_episodes]
        outside = brs.outside_scenes(refs, train)
        if outside:
            raise SystemExit(f"source {name}: records of scenes neither held out nor training scenes (calls per scene): "
                             f"{dict(sorted(outside.items())[:20])} ({len(outside)} scenes)")
        paths = sorted({r.record_path for r in refs})
        labels = brs.label_provenance([record_root], [label_root], paths)
        if labels["stale_label_files"]:
            raise SystemExit(f"source {name}: {labels['stale_label_files']} label files label another version of their "
                             f"record, e.g. {labels['stale_examples']}: relabel them")
        if paths and (set(labels["labelers"]) != {labeler} or set(labels["lines"]) != {TERMINAL_LABEL_LINE}):
            raise SystemExit(f"source {name}: labels by {labels['labelers']} (lines {labels['lines']}), not line C's "
                             f"RouteLabelConfig(terminal_stop_radius_m={args.stop_radius}) {labeler}")
        served = Counter(x34._map(x34._record_adapter, paths, args.workers, chunksize=32))
        if set(served) - {want}:
            raise SystemExit(f"source {name}: records served by {dict(served)}, the source names {adapter}")
        by_record: dict[str, list[CallRef]] = defaultdict(list)
        for ref in refs:
            by_record[ref.record_path].append(ref)
        mine: Counter = Counter()
        for path in paths:
            rows = {int(r["call_index"]): r for r in read_labels(label_path_for(path, label_root))}
            found, c, ref_calls = classify_record(by_record[path], rows, source=name, stop_radius_m=args.stop_radius,
                                                  first_calls=args.first_calls)
            candidates += found
            mine.update(c)
            reference += ref_calls
        counts.update(mine)
        sources[name] = {"record_root": record_root, "label_root": label_root, "adapter": adapter,
                         "index": dict(index_stats), "calls_held_out_by_scene_id": held_by_id,
                         "calls_held_out_for_the_gate": n_gate, "records": len(paths), "labels": labels,
                         "counts": dict(mine), "candidates": dict(Counter(c.part for c in candidates if c.source == name)),
                         "devices": dict(Counter(r.record_device for r in refs))}
        print(f"[exp35] source {name}: {len(paths)} records, {len(refs)} calls, "
              f"candidates {sources[name]['candidates']}", flush=True)
    if not any(s["calls_held_out_for_the_gate"] for s in sources.values()):
        raise SystemExit(f"--holdout-cohort-shards {cohort_dir} {shard_list} matched no record of any source")
    dagger, dagger_info = x34.draw_dagger(candidates, max_cor_per_episode=args.max_cor_per_episode,
                                          max_agr_per_episode=args.max_agr_per_episode)
    if not dagger["COR"]:
        raise SystemExit(f"no COR2 samples (counts {dict(counts)})")

    # --- REP2 (InternData not already in D1), then one order over D1 and D2
    pool, replay_counts = x34.load_replay_pool(args.replay_root, held)
    fresh = [item for item in pool if item["key"] not in d1_replay]
    replay_counts["pool_in_d1"] = len(pool) - len(fresh)
    parts, band = x34.compose(dagger, x34.replay_units(fresh), replay_fraction=args.replay_fraction,
                              shard_size=args.shard_size, seed=args.seed, band=None)
    d2 = make_d2_items(parts)
    leaks = Counter(brs.scene_key(ms.sample_scene(i["key"])) for i in d2 if brs.scene_key(ms.sample_scene(i["key"])) in held)
    if leaks:
        raise SystemExit(f"held-out scenes in D2 (samples per scene): {dict(leaks)}")
    stats = d2_stats(d2, reference)
    slots: list[tuple[str, Any]] = [("d1", e) for e in d1_entries] + [("d2", i) for i in d2]
    keys = [e.key for e in d1_entries] + [i["key"] for i in d2]
    if set(keys[: len(d1_entries)]) & {i["key"] for i in d2}:
        raise SystemExit("a D2 key is also a D1 key")
    slots = [slots[i] for i in order(keys)]
    total = len(slots)
    print_page(d1_parts, stats, counts, total)
    if total % args.shard_size:
        raise SystemExit(f"{total} samples are not whole shards of {args.shard_size}")
    if args.dry_run:
        print(json.dumps({"dagger": dagger_info, "replay": dict(replay_counts), "band": band}, sort_keys=True))
        return 0

    # --- write
    out_dir.mkdir(parents=True, exist_ok=True)
    num_shards = total // args.shard_size
    jobs = [(s, slots[s * args.shard_size:(s + 1) * args.shard_size], str(out_dir), tuple(args.vlm_image_size),
             args.jpeg_quality) for s in range(num_shards)]
    shards: dict[int, dict] = {}
    for index, digest, size in x34._map(_write_job, jobs, args.workers):
        shards[index] = {"name": shard_name(index), "sha256": digest, "bytes": size}
    print(f"[exp35] {num_shards} shards written", flush=True)

    def row(kind: str, value: Any) -> str:
        if kind == "d1":
            meta = entry_meta(value)
            return f"{value.key}\t{meta['part']}\t{str(meta['kl']).lower()}\td1:{Path(value.shard).name}#{value.index}"
        where = f"{value['record_path']}#{value['call_index']}" if value["part"] != "REP2" else \
            f"{value['scene_dir']}@{value['sample']['frame_id']}"
        return f"{value['key']}\t{value['part']}\t{str(value['kl']).lower()}\t{where}"

    keys_text = "".join(row(kind, value) + "\n" for kind, value in slots)
    (out_dir / x34.KEYS_TSV).write_text(keys_text, encoding="utf-8")
    turns_of = lambda kind, v: v.turns if kind == "d1" else tuple(v["turns"])  # noqa: E731
    key_of = lambda kind, v: v.key if kind == "d1" else v["key"]  # noqa: E731
    holdout = sorted(held)
    drawn: Counter = Counter()  # train_system2_lora.py logs it, as every stream builder writes it
    for kind, value in slots:
        source, part = (value.source, entry_meta(value)["part"]) if kind == "d1" else (value["source"], value["part"])
        drawn[source] += 1
        drawn[f"{source}_target_{decision_class(turns_of(kind, value)[-1])}"] += 1
        drawn[f"category_{part}"] += 1
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
        "replay_fraction": round((d1_parts["REP"] + len(parts["REP"])) / total, 6),
        "holdout": {**d1_manifest["holdout"], "gate_cohort_shards": args.holdout_cohort_shards,
                    "gate_episodes": len(gate_episodes)},
        "record_roots": list(d1_manifest["record_roots"]) + [s["record_root"] for s in sources.values()],
        "label_roots": list(d1_manifest["label_roots"]) + [s["label_root"] for s in sources.values()],
        "replay_root": str(Path(args.replay_root).resolve()),
        "kl": {"field": "kl", "where": "each sample's {i:06d}.json in the shard (also 'part'; keys.tsv column 3)",
               "by_part": {**x34.KL, **KL}},
        "d1": {"stream_dir": str(Path(args.d1_stream).resolve()), "manifest_sha256": d1_digest,
               "keys_sha256": d1_manifest["selection"]["keys_sha256"], "samples": len(d1_entries),
               "parts": dict(d1_parts)},
        "selection": {
            "rule": "EXP-35: D1 = EXP-34's stream byte for byte; D2 on line-C labels of the round-2 records: COR2 every "
                    "line-C correction (kl false, STOP within the STOP radius), AGR2 line-C anchors 1:1 (<=6 per "
                    "episode, kl false), REP2 InternData r2r not in D1 at 20% of D2 with STOP x5 (kl true); "
                    "call_index < 48, <=24 COR2 per episode, reversals and turn strings with STOP dropped; one order "
                    "by sha256('exp35-order:' + key + '#' + copy)",
            "knobs": {"stop_radius_m": args.stop_radius, "labeler": labeler, "first_calls": args.first_calls,
                      "max_cor_per_episode": args.max_cor_per_episode, "max_agr_per_episode": args.max_agr_per_episode,
                      "replay_fraction": args.replay_fraction, "stop_upsample": x34.STOP_UPSAMPLE,
                      "agr_rank": f"sha256('{x34.AGR_RANK}' + key)", "order_rank": f"sha256('{ORDER_RANK}' + key + '#' + copy)"},
            "sources": sources,
            "counts": dict(counts),
            "dagger": dagger_info,
            "replay": dict(replay_counts),
            "d2_parts": stats["by_part"],
            "stats": stats,
            "keys_sha256": hashlib.sha256("\n".join(key_of(k, v) + "|" + "\x1f".join(turns_of(k, v))
                                                    for k, v in slots).encode()).hexdigest(),
            "keys_tsv_sha256": hashlib.sha256(keys_text.encode("utf-8")).hexdigest(),
        },
        "drawn": dict(drawn),
        "shards": [shards[s] for s in range(num_shards)],
        "build_s": round(time.time() - started, 1),
    }
    tmp = out_dir / f".{MANIFEST_NAME}.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out_dir / MANIFEST_NAME)
    print(json.dumps({k: manifest[k] for k in ("num_shards", "total_samples", "replay_fraction", "build_s")},
                     sort_keys=True))
    print("trainer: --holdout-scenes " + ",".join(holdout), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

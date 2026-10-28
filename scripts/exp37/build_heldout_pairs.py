#!/usr/bin/env python3
"""EXP-37 offline gate G1's held-out preference pairs, packed in the pair stream's sample format.

The ledger's EXP-37 G1: pairs on scenes EXP-34/35's streams held out -- EXP-23's six R2R H scenes
(round-1 base records, line-B labels) and EXP-28's 20 HM3D scenes (the base's collect records, line-B
labels) -- chosen by the training pairs' rules (``build_pair_sidecar.judge_pair``): a line-B
``correction`` the base made (records an adapter served are refused), label ``d > 3 m``,
``call_index < 48``, neither side STOP.  The training pairs' 25 px drop of pixel -> pixel pairs is not
applied (``--min-pixel-px`` defaults to 0): the ledger lists it under the training pairs only, and G1's
pre-registered set (about 1,475 pairs) is the one without it.  No stream-only filter (reversals, the
24-per-episode cap) applies either; what they would have removed is reported.

The states are ``hm3d_route_gate.select_states``' (the O4 gate's: every committed record of the scenes,
labels of line B's default labeller on the same record version, ``call_index < 48``, ``d > 3 m``), so
rejected is the label row's ``model_turn1/2`` and chosen its ``target_turn1/2``; rejected must equal
``recorded_turns`` of the record's call.  Each sample is packed from its record exactly as
``build_exp34_stream.py`` packs a call (``build_s2_train_stream._pack``: the record's front, history and
lookdown JPEG bytes as the server saw them) with ``turns`` = chosen, ``kl`` false, ``part`` ``HELD``, and
the pair fields of the training stream (``rejected_turns``, ``pair``; ``pack_pair_stream.with_pair``).

Every scene must be held out by the training stream (``--stream-manifest``'s ``holdout``).  Output
``--out-dir``: shards of ``--shard-size`` in sha256("exp37-order:" + key) order (the last one may be short:
this set is scored, never trained on), ``stream_manifest.json`` (``"pairs": true``, ``"heldout": true``)
and ``keys.tsv``.

On the 4090 (CPU only)::

    PYTHONPATH=$WT:/workspace/rpc/src /opt/conda/bin/python scripts/exp37/build_heldout_pairs.py \\
        --source r2r-H /workspace/exp23/round1/records /workspace/exp34/labels/r2r r2r \\
        --source hm3d-H /workspace/exp28/collect_base/records /workspace/exp28/labels_base \\
            /workspace/exp28/episodes/splits/heldout_scenes.txt \\
        --stream-manifest /workspace/exp35/stream/stream_manifest.json --out-dir /workspace/exp37/heldout_pairs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.exp28 import hm3d_route_gate as hg  # noqa: E402
from scripts.exp34 import build_exp34_stream as x34  # noqa: E402
from scripts.exp35 import build_exp35_stream as x35  # noqa: E402
from scripts.exp37 import build_pair_sidecar as sc  # noqa: E402
from scripts.exp37 import pack_pair_stream as pk  # noqa: E402
from scripts.tools import build_route_stream as brs  # noqa: E402
from scripts.tools import build_s2_train_stream as bst  # noqa: E402
from src.data.s2_records import EpisodeRecord, episode_key  # noqa: E402
from src.data.s2_route_labels import RouteLabelConfig  # noqa: E402
from src.data.s2_sft_dataset import DEFAULT_VLM_IMAGE_SIZE, record_device, recorded_turns, validate_turns  # noqa: E402
from src.data.s2_stream import MANIFEST_NAME, shard_name  # noqa: E402

BUILDER = "scripts/exp37/build_heldout_pairs.py (EXP-37)"
PART = "HELD"
MIN_PIXEL_PX = 0.0  # G1 keeps near pixel -> pixel pairs; the 25 px drop (sc.MIN_PIXEL_PX) is the training pairs' only
SCENE_LISTS = {"r2r": x34.R2R_HOLDOUT, "hm3d": x34.HM3D_HOLDOUT}  # build_exp34_stream's held-out constants


def scene_list(spec: str) -> tuple[str, ...]:
    """``r2r`` / ``hm3d`` (EXP-34's held-out constants) or a scene-list file."""
    return tuple(SCENE_LISTS[spec]) if spec in SCENE_LISTS else brs.read_scene_list(spec)


def source_pairs(name: str, record_root: str, label_root: str, scenes: Sequence[str], *, settings: dict[str, Any],
                 labeler: str) -> tuple[list[dict[str, Any]], Counter, dict[str, Any]]:
    """The pairs of one held-out source (not packed yet), the drops, and the gate selection's meta."""
    states, meta = hg.select_states([record_root], label_root, scenes=brs.holdout_match_set(scenes),
                                    config_digest=labeler, max_call_index=settings["first_calls"],
                                    far_m=settings["far_m"])
    counts: Counter = Counter()
    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        counts[f"states_{state['label_kind']}"] += 1
        if state["label_kind"] == "correction":
            by_record[state["record_path"]].append(state)
    pairs: list[dict[str, Any]] = []
    for path in sorted(by_record):
        record = EpisodeRecord(path)
        calls = {int(c["call_index"]): c for c in record.calls}
        rows = sorted(by_record[path], key=lambda s: s["call_index"])
        directions = brs.record_directions(path)
        for state in rows:
            what = f"{state['key']} ({path})"
            if state["label_agree"] is not False or state["label_target"] is None:
                raise SystemExit(f"{what}: a correction with agree={state['label_agree']} target={state['label_target']}")
            rejected = validate_turns(state["recorded"], what=f"{what} rejected")
            chosen = validate_turns(state["label_target"], what=f"{what} chosen")
            if recorded_turns(calls[state["call_index"]]) != rejected:
                raise SystemExit(f"{what}: label model turns {rejected!r}, record "
                                 f"{recorded_turns(calls[state['call_index']])!r}")
            why, judged = sc.judge_pair(rejected, chosen, d=state["distance_to_goal"], call_index=state["call_index"],
                                        **settings)
            if why is not None:
                counts[f"dropped:{why}"] += 1
                continue
            previous = x34.previous_direction(directions, state["call_index"])
            if previous and brs.executed_direction(chosen) == -previous:
                counts["report_only:stream_would_drop_reversal"] += 1
            pairs.append({
                "key": f"{record.scene_id}_{record.episode_id:06d}#{name}.{state['call_index']}",
                "part": PART, "record_path": path, "call_index": state["call_index"],
                "record": f"{path}#{state['call_index']}", "device": record_device(record),
                "chosen_turns": list(chosen), "rejected_turns": list(rejected),
                "pair": sc.pair_meta(source=name, d=state["distance_to_goal"], call_index=state["call_index"],
                                     judged=judged, episode=episode_key(record.scene_id, record.episode_id)),
            })
    per_episode = Counter(p["pair"]["episode"] for p in pairs)
    counts["report_only:stream_would_drop_episode_cap"] = sum(max(0, n - x34.MAX_COR_PER_EPISODE)
                                                              for n in per_episode.values())
    return pairs, counts, meta


def pack(pair: dict[str, Any]) -> list[tuple[str, bytes]]:
    """The pair as a stream sample: ``build_exp34_stream``'s packing of the call, plus the pair fields."""
    item = {"source": "dagger", "record_path": pair["record_path"], "call_index": pair["call_index"],
            "key": pair["key"], "turns": list(pair["chosen_turns"])}
    packed = bst._pack(item, DEFAULT_VLM_IMAGE_SIZE, 90)
    if packed.key != pair["key"] or list(packed.turns) != pair["chosen_turns"]:
        raise RuntimeError(f"packed {packed.key} {packed.turns!r} for {pair['key']} {pair['chosen_turns']!r}")
    return pk.with_pair(x35._members_of_packed(packed, False, PART), pair["rejected_turns"], pair["pair"],
                        chosen=pair["chosen_turns"], key=pair["key"])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", nargs=4, action="append", required=True,
                        metavar=("NAME", "RECORD_ROOT", "LABEL_ROOT", "SCENES"),
                        help="base records of held-out scenes, their line-B labels, and the scenes: 'r2r' / 'hm3d' "
                             "(build_exp34_stream's held-out lists) or a scene-list file; repeatable")
    parser.add_argument("--stream-manifest", required=True,
                        help="the training stream's manifest: every scene here must be in its holdout")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--far-m", type=float, default=sc.FAR_M)
    parser.add_argument("--first-calls", type=int, default=sc.FIRST_CALLS)
    parser.add_argument("--min-pixel-px", type=float, default=MIN_PIXEL_PX,
                        help="drop pixel -> pixel pairs closer than this; 0 (default) keeps them, as G1 is defined")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    if (out_dir / MANIFEST_NAME).exists():
        raise SystemExit(f"{out_dir} already holds a stream")
    started = time.time()
    names = [s[0] for s in args.source]
    if len(set(names)) != len(names) or not all(x34._SOURCE_NAME.match(n) for n in names):
        raise SystemExit(f"source names must be distinct and [A-Za-z0-9-]: {names}")
    stream_raw = Path(args.stream_manifest).read_bytes()
    held = {brs.scene_key(s) for s in json.loads(stream_raw)["holdout"]["scenes"]}
    settings = {"far_m": args.far_m, "first_calls": args.first_calls, "min_pixel_px": args.min_pixel_px}
    labeler = RouteLabelConfig().digest()

    pairs: list[dict[str, Any]] = []
    counts: Counter = Counter()
    sources: dict[str, Any] = {}
    for name, record_root, label_root, spec in args.source:
        scenes = scene_list(spec)
        outside = sorted({brs.scene_key(s) for s in scenes} - held)
        if outside:
            raise SystemExit(f"source {name}: scenes the training stream did not hold out: {outside}")
        record_root, label_root = str(Path(record_root).resolve()), str(Path(label_root).resolve())
        found, c, meta = source_pairs(name, record_root, label_root, scenes, settings=settings, labeler=labeler)
        if not meta["records_in_pool"]:
            raise SystemExit(f"source {name}: no record of {len(scenes)} scenes under {record_root}")
        pairs += found
        counts.update({f"{k}:{name}": v for k, v in c.items()})
        sources[name] = {"record_root": record_root, "label_root": label_root, "scenes_spec": spec,
                         "scenes": sorted(scenes), "gate_selection": meta, "counts": dict(sorted(c.items())),
                         "pairs": len(found)}
        print(f"[exp37] held-out {name}: {meta['records_in_pool']} records, {len(found)} pairs, "
              f"{dict(sorted(c.items()))}", flush=True)
    if not pairs:
        raise SystemExit("no held-out pairs")
    chosen = pk.ordered(pairs)

    out_dir.mkdir(parents=True, exist_ok=True)
    num_shards = -(-len(chosen) // args.shard_size)
    shards = []
    for s in range(num_shards):
        samples = [pack(p) for p in chosen[s * args.shard_size:(s + 1) * args.shard_size]]
        path = out_dir / shard_name(s)
        shards.append({"name": shard_name(s), "sha256": x35.write_shard(path, samples), "bytes": path.stat().st_size,
                       "samples": len(samples)})
    keys_text = "".join(f"{p['key']}\t{PART}\tfalse\t{p['record']}\t{p['device']}\t{p['pair']['d']:.4f}\n"
                        for p in chosen)
    (out_dir / x34.KEYS_TSV).write_text(keys_text, encoding="utf-8")
    manifest = {
        "schema": pk.PAIR_SCHEMA,
        "pairs": True,
        "heldout": True,
        "created_unix": time.time(),
        "builder": BUILDER,
        "shard_size": args.shard_size,
        "num_shards": num_shards,
        "total_samples": len(chosen),
        "epochs": 1.0,
        "kl": {"field": "kl", "where": "each sample's {i:06d}.json in the shard", "by_part": {PART: False}},
        "pair_fields": {"chosen": "turns", "rejected": "rejected_turns", "meta": "pair"},
        "training_stream_manifest": {"path": str(Path(args.stream_manifest).resolve()),
                                     "sha256": hashlib.sha256(stream_raw).hexdigest()},
        "record_roots": [s["record_root"] for s in sources.values()],
        "label_roots": [s["label_root"] for s in sources.values()],
        "selection": {
            "rule": "EXP-37 G1: line-B corrections of the base on the held-out scenes (hm3d_route_gate.select_states: "
                    "call < first_calls, label d > far_m), neither side STOP, pixel pairs >= min_pixel_px (0: all "
                    "kept); chosen = label target, rejected = label model turns (= recorded_turns); packed as "
                    "build_exp34_stream packs a call, with rejected_turns and pair; order sha256('exp37-order:' + key)",
            "settings": settings,
            "labeler": labeler,
            "sources": sources,
            "counts": dict(sorted(counts.items())),
            "stats": sc.pair_stats(chosen),
            "order_rank": f"sha256('{pk.ORDER_RANK}' + key)",
            "keys_sha256": pk.keys_sha256(chosen),
            "keys_tsv_sha256": hashlib.sha256(keys_text.encode("utf-8")).hexdigest(),
        },
        "drawn": pk.drawn_of(chosen),
        "shards": shards,
        "build_s": round(time.time() - started, 1),
    }
    tmp = out_dir / f".{MANIFEST_NAME}.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out_dir / MANIFEST_NAME)
    stats = manifest["selection"]["stats"]
    print(f"[exp37] held-out pairs {stats['pairs']} by source {stats['by_source']}\n[exp37] by type {stats['by_type']}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

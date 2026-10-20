#!/usr/bin/env python3
"""Mix EXP-28's DAgger stream with an expert-replay stream into one training stream (R7, R8).

Inputs, both packed on the 4090:

* ``--dagger-stream``: ``scripts/tools/build_route_stream.py --no-pad`` -- line B's
  corrections and 2x anchors, no STOP target, no replay;
* ``--replay-stream``: ``scripts/tools/build_replay_stream.py --stop-upsample 5
  --exclude-holdout --exclude-scenes <held-out HM3D scene dirs>`` over InternData r2r and
  ScaleVLN-train.

EXP-32 passes ``scripts/exp32/build_rft_stream.py --no-pad`` (RF1-RF7: the model's own sampled
answers on successful trajectories plus 1:1 base anchors) as ``--dagger-stream``, with one r2r
replay root at ``--replay-fraction 0.2`` (RF8, RF9).  Its knobs are not EXP-28's, so it merges
only with ``--allow-other-rules``; the checks refused always below hold for it unchanged.

Every DAgger sample is taken once.  Replay is added until it makes up ``--replay-fraction``
(0.3) of the stream, rounded up to whole shards, and split over the replay stream's roots by
``--replay-shares`` (default ``equal``: half r2r, half ScaleVLN).  Replay is drawn by key, so
a STOP sample comes with all of its ``--stop-upsample`` copies or not at all (a STOP whose
copies the replay builder's trim to whole shards cut short is left out).  STOP targets thus
appear only in replay, at InternData's expert end frames.  No sample of a held-out scene --
``--holdout-scenes-file`` (``00xxx-ID`` and bare IDs both match), EXP-23's six R2R scenes,
and whatever either input already held out -- may appear in either input.  The samples are
re-sharded in one seeded shuffle; their bytes are copied as packed, nothing is re-encoded.

Refused unless ``--allow-other-rules`` (then the manifest lists what differed): a DAgger stream
not built with EXP-28's knobs (R1, R2, R6, R7: ``EXP28_DAGGER_KNOBS``), without a training-scene
list, from labels not made by the default ``RouteLabelConfig`` (line B), or holding fewer
anchors than 2 x its corrections; and a draw that is not R8 (replay fraction 0.3, STOP x5, the
replay half InternData r2r and half ScaleVLN by the samples' scenes).  Refused always: a DAgger
sample of a scene outside the DAgger stream's training scenes, a held-out scene anywhere, a
STOP target in the DAgger part, inputs whose bytes or keys do not match their manifests.

The manifest is the one ``train_system2_lora.py --stream-dir`` reads: ``holdout.scenes`` is
the bare IDs of every excluded scene (the trainer needs the same list as ``--holdout-scenes``;
it is printed at the end), ``record_roots`` / ``label_roots`` come from the DAgger stream and
``replay_root`` from the replay stream.

The whole round on the 4090 (``H`` the held-out file; the replay stream only has to be larger
than the share drawn from it, about ``0.3 / 0.7 x DAgger / 2`` per root plus one shard --
``--dry-run`` here prints the quotas, and a pool too small is refused)::

    H=/workspace/exp28/episodes/splits/heldout_scenes.txt
    T=/workspace/exp28/episodes/splits/train_scenes.txt
    /opt/conda/bin/python scripts/tools/build_route_stream.py \\
        --record-root /workspace/exp28/collect_base/records --label-root /workspace/exp28/collect_base/labels_routeB \\
        --holdout-scenes-file $H --train-scenes-file $T --first-calls 48 --min-distance 3 \\
        --stop-correction-min-distance 5 --max-stop-corrections-per-episode 1 --drop-reversals --no-pad \\
        --out-dir /workspace/exp28/round1/stream_dagger
    /opt/conda/bin/python scripts/tools/build_replay_stream.py \\
        --root r2r=/workspace/exp23/data/interndata_n1/r2r --samples r2r=8000 \\
        --root scalevln=/workspace/exp23/data/interndata_n1/scalevln_train --samples scalevln=8000 \\
        --exclude-holdout --exclude-scenes "$(grep -v '^#' $H | paste -sd, -)" \\
        --stop-upsample 5 --out-dir /workspace/exp28/round1/stream_replay
    /opt/conda/bin/python scripts/exp28/merge_streams.py \\
        --dagger-stream /workspace/exp28/round1/stream_dagger \\
        --replay-stream /workspace/exp28/round1/stream_replay \\
        --holdout-scenes-file $H --out-dir /workspace/exp28/round1/stream

``--dry-run`` stops after the draw and prints the composition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tarfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.build_route_stream import UNPADDED_SCHEMA, read_scene_list, scene_key  # noqa: E402
from src.data.s2_route_labels import LABEL_LINE, RouteLabelConfig  # noqa: E402
from src.data.s2_sft_dataset import R2R_TRAIN_SCENES, decision_class, holdout_scenes  # noqa: E402
from src.data.s2_stream import (  # noqa: E402
    MANIFEST_NAME,
    STREAM_SCHEMA,
    PackedSample,
    shard_digest_ok,
    shard_name,
    write_shard,
)

BUILDER = "scripts/exp28/merge_streams.py (EXP-28)"
DAGGER_BUILDER = "scripts/tools/build_route_stream.py"
# EXP-32's RFT stream (RF1-RF7) stands in for the DAgger part; it is not built with EXP-28's knobs,
# so it merges only with --allow-other-rules, and every always-refused check below still applies.
RFT_BUILDER = "scripts/exp32/build_rft_stream.py"
REPLAY_BUILDER = "scripts/tools/build_replay_stream.py"
# The ledger's rules R1, R2, R6, R7 as build_route_stream.py's knobs; a DAgger stream built
# otherwise is refused unless --allow-other-rules (and then the manifest says so).
EXP28_DAGGER_KNOBS: dict[str, Any] = {
    "first_calls": 48,
    "min_distance": 3.0,
    "stop_correction_min_distance": 5.0,
    "max_stop_corrections_per_episode": 1,
    "drop_reversals": True,
    "max_corrections_per_episode": 8,
    "correction_pick": "earliest",
    "max_anchors_per_episode": 12,
    "anchors_per_correction": 2.0,
    "correction_weight": 1.0,
    "no_pad": True,  # padding would add anchors beyond 2 x corrections
}
# R8 as this script's knobs.
EXP28_REPLAY_FRACTION = 0.3
EXP28_STOP_UPSAMPLE = 5


def rule_mismatches(knobs: Mapping[str, Any]) -> dict[str, Any]:
    """The DAgger stream's knobs that are not EXP-28's (name -> its value; absent: None)."""
    return {k: knobs.get(k) for k, v in EXP28_DAGGER_KNOBS.items() if knobs.get(k) != v}


def dagger_rule_mismatches(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Everything about the DAgger stream that is not EXP-28's: its knobs, no training-scene list
    (R1), labels by another labeller than the default ``RouteLabelConfig`` (the pre-registered
    rules) or line, and fewer anchors than 2 x corrections (R7)."""
    selection = manifest.get("selection") or {}
    knobs = selection.get("knobs") or {}
    out = rule_mismatches(knobs)
    if not knobs.get("train_scenes_file"):
        out["train_scenes_file"] = knobs.get("train_scenes_file")
    labels = selection.get("labels") or {}
    if set(labels.get("labelers") or {}) != {RouteLabelConfig().digest()}:
        out["labelers"] = labels.get("labelers")
    if set(labels.get("lines") or {}) != {LABEL_LINE}:
        out["label_lines"] = labels.get("lines")
    capped = (selection.get("composition") or {}).get("anchors_capped_by_pool")
    if capped is not False:
        out["anchors_capped_by_pool"] = capped
    return out


def replay_dataset(key: str) -> str:
    """``hm3d`` (an HM3D scene dir, ScaleVLN), ``r2r`` (an R2R train scene) or ``other``."""
    scene = sample_scene(key)
    if scene_key(scene) != scene:
        return "hm3d"
    return "r2r" if scene in R2R_TRAIN_SCENES else "other"


def merge_rule_mismatches(replay_fraction: float, stop_upsample: int, by_dataset: Mapping[str, int]) -> dict[str, Any]:
    """R8 for the draw: 30% replay, STOP x5, half InternData r2r and half ScaleVLN (HM3D)."""
    out: dict[str, Any] = {}
    if replay_fraction != EXP28_REPLAY_FRACTION:
        out["replay_fraction"] = replay_fraction
    if stop_upsample != EXP28_STOP_UPSAMPLE:
        out["stop_upsample"] = stop_upsample
    counts = {k: int(v) for k, v in by_dataset.items() if v}
    if set(counts) != {"r2r", "hm3d"} or abs(counts["r2r"] - counts["hm3d"]) > 1:
        out["replay_by_dataset"] = counts
    return out


# --------------------------------------------------------------------------- #
# Reading packed samples without loading whole shards
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Entry:
    """One packed sample of an input shard: its metadata and where its members sit in the tar."""

    shard: str
    index: int
    key: str
    source: str
    turns: tuple[str, ...]
    members: tuple[tuple[str, int, int], ...]  # (member name, data offset, size)


def index_shard(path: str | os.PathLike) -> list[Entry]:
    """Every sample of one shard (``s2_stream.write_shard``'s layout), headers and json only."""
    path = str(path)
    members: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    metas: dict[str, dict] = {}
    with tarfile.open(path, mode="r:") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            stem = member.name.split(".", 1)[0]
            members[stem].append((member.name, member.offset_data, member.size))
            if member.name.endswith(".json"):
                metas[stem] = json.loads(tar.extractfile(member).read())
    entries = []
    i = 0
    while f"{i:06d}" in metas:
        meta = metas[f"{i:06d}"]
        entries.append(Entry(path, i, str(meta["key"]), str(meta["source"]), tuple(str(t) for t in meta["turns"]),
                             tuple(sorted(members[f"{i:06d}"]))))
        i += 1
    if len(entries) != len(metas):
        raise SystemExit(f"{path}: sample numbering has a gap ({len(metas)} json members, {len(entries)} in sequence)")
    return entries


def read_entry(entry: Entry) -> PackedSample:
    """The packed sample, byte for byte as ``read_shard`` would give it."""
    blobs: dict[str, bytes] = {}
    with open(entry.shard, "rb") as handle:
        for name, offset, size in entry.members:
            handle.seek(offset)
            blobs[name] = handle.read(size)
    stem = f"{entry.index:06d}"
    meta = json.loads(blobs[f"{stem}.json"])
    return PackedSample(
        key=str(meta["key"]),
        source=str(meta["source"]),
        instruction=str(meta["instruction"]),
        turns=tuple(str(t) for t in meta["turns"]),
        vlm_image_size=tuple(int(v) for v in meta["vlm_image_size"]),  # type: ignore[arg-type]
        history_jpegs=tuple(blobs[f"{stem}.h{k:02d}.jpg"] for k in range(int(meta["num_history"]))),
        current_jpeg=blobs[f"{stem}.cur.jpg"],
        lookdown_jpeg=blobs[f"{stem}.look.jpg"],
    )


def load_stream(stream_dir: str | os.PathLike, *, schemas: Sequence[str], verify: bool) -> tuple[dict, list[Entry], str]:
    """(manifest, entries in stream order, manifest sha256) of a stream directory."""
    stream_dir = Path(stream_dir)
    raw = (stream_dir / MANIFEST_NAME).read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") not in schemas:
        raise SystemExit(f"{stream_dir}: schema {manifest.get('schema')!r} is not one of {list(schemas)}")
    entries: list[Entry] = []
    for shard in manifest["shards"]:
        path = stream_dir / shard["name"]
        if not path.is_file():
            raise SystemExit(f"{path} is missing (pushed and consumed already?)")
        if verify and not shard_digest_ok(path, shard["sha256"]):
            raise SystemExit(f"{path}: sha256 differs from the manifest")
        entries += index_shard(path)
    if len(entries) != int(manifest["total_samples"]):
        raise SystemExit(f"{stream_dir}: {len(entries)} samples in the shards, the manifest says {manifest['total_samples']}")
    return manifest, entries, hashlib.sha256(raw).hexdigest()


def sample_scene(key: str) -> str:
    """Scene of a packed sample's key: ``{scene}_{episode:06d}#{call}`` (DAgger) or
    ``replay:{scene}_{episode:06d}@{frame}:{hash}`` (replay)."""
    if key.startswith("replay:"):
        return key[len("replay:"):].rsplit("@", 1)[0].rsplit("_", 1)[0]
    return key.rsplit("#", 1)[0].rsplit("_", 1)[0]


def dagger_keys_sha256(entries: Iterable[Entry]) -> str:
    """``build_route_stream``'s ``selected_keys_sha256`` (key|turns of the DAgger samples, in order)."""
    return hashlib.sha256("\n".join(e.key + "|" + "\x1f".join(e.turns) for e in entries).encode()).hexdigest()


def replay_keys_sha256(entries: Iterable[Entry]) -> str:
    """``build_replay_stream``'s ``keys_sha256`` (keys in order)."""
    return hashlib.sha256("\n".join(e.key for e in entries).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# The draw
# --------------------------------------------------------------------------- #
def stream_size(n_dagger: int, replay_fraction: float, shard_size: int) -> tuple[int, int]:
    """(total, replay): replay at least ``replay_fraction`` of the total, the total whole shards."""
    if not 0.0 <= replay_fraction < 1.0:
        raise SystemExit("--replay-fraction must be in [0, 1)")
    n_replay = math.ceil(round(replay_fraction / (1.0 - replay_fraction) * n_dagger, 9))
    total = math.ceil((n_dagger + n_replay) / shard_size) * shard_size
    return total, total - n_dagger


def apportion(total: int, shares: Mapping[str, float]) -> dict[str, int]:
    """Split ``total`` over ``shares`` (largest remainders; ties go to the earlier name)."""
    names = sorted(shares)
    weight = sum(float(shares[n]) for n in names)
    if weight <= 0 or any(float(shares[n]) < 0 for n in names):
        raise SystemExit(f"replay shares must be >= 0 with a positive sum: {dict(shares)}")
    exact = {n: total * float(shares[n]) / weight for n in names}
    out = {n: int(math.floor(exact[n])) for n in names}
    for n in sorted(names, key=lambda n: (-(exact[n] - out[n]), names.index(n)))[: total - sum(out.values())]:
        out[n] += 1
    return out


def parse_shares(text: str, roots: Sequence[str]) -> dict[str, float]:
    """``equal`` (every root alike), ``pooled`` (one pool, no split) or ``NAME=W,NAME=W``."""
    if text == "equal":
        return {r: 1.0 for r in roots}
    if text == "pooled":
        return {"all": 1.0}
    shares = {}
    for part in text.split(","):
        name, sep, value = part.partition("=")
        if not sep:
            raise SystemExit(f"--replay-shares takes equal, pooled or NAME=W,...; got {text!r}")
        shares[name.strip()] = float(value)
    if set(shares) != set(roots):
        raise SystemExit(f"--replay-shares names {sorted(shares)}, the replay stream's roots are {sorted(roots)}")
    return shares


def scene_roots(roots: Mapping[str, str]) -> dict[str, str]:
    """bare scene ID -> replay root name, from the root directories' scene dirs."""
    out: dict[str, str] = {}
    for name, root in sorted(roots.items()):
        root = Path(root)
        if not root.is_dir():
            raise SystemExit(f"replay root {name}={root} is not a directory here; pass --replay-shares pooled to skip the split")
        for scene_dir in root.iterdir():
            if scene_dir.is_dir() and not scene_dir.name.startswith("."):
                key = scene_key(scene_dir.name)
                if out.setdefault(key, name) != name:
                    raise SystemExit(f"scene {key} is in replay roots {out[key]} and {name}")
    return out


def replay_units(
    entries: Sequence[Entry], *, root_of: Mapping[str, str] | None, stop_upsample: int,
) -> tuple[dict[str, list[list[Entry]]], Counter]:
    """Replay samples grouped by key into units (a STOP and its copies), per root.

    ``root_of`` maps a bare scene ID to its root (None: one pool, ``all``).  A STOP unit
    whose copies are not a multiple of ``stop_upsample`` was cut by the replay builder's
    trim and is left out.
    """
    by_key: dict[str, list[Entry]] = defaultdict(list)
    for entry in entries:
        by_key[entry.key].append(entry)
    units: dict[str, list[list[Entry]]] = defaultdict(list)
    counts: Counter = Counter()
    for key in sorted(by_key):
        copies = by_key[key]
        if len({c.turns for c in copies}) != 1:
            raise SystemExit(f"replay key {key} carries different targets")
        stop = decision_class(copies[0].turns[-1]) == "stop"
        if stop and len(copies) % stop_upsample:
            counts["stop_units_cut_short"] += 1
            counts["stop_samples_cut_short"] += len(copies)
            continue
        if not stop and len(copies) > 1:
            counts["non_stop_keys_repeated"] += 1  # the same instruction twice in one InternData episode
        if root_of is None:
            root = "all"
        else:
            root = root_of.get(scene_key(sample_scene(key)))
            if root is None:
                raise SystemExit(f"replay sample {key}: its scene is in none of the replay stream's roots")
        units[root].append(copies)
        counts[f"{root}_units"] += 1
        counts[f"{root}_samples"] += len(copies)
        if stop:
            counts[f"{root}_stop_units"] += 1
    return units, counts


def draw_replay(
    units: Mapping[str, Sequence[Sequence[Entry]]], need: int, shares: Mapping[str, float], rng: np.random.Generator,
) -> tuple[list[Entry], Counter]:
    """``need`` replay samples split over roots by ``shares``: whole units in a seeded order,
    skipping a unit that would overshoot its root's quota."""
    quotas = apportion(need, shares)
    chosen: list[Entry] = []
    counts: Counter = Counter()
    for root in sorted(quotas):
        quota, pool, taken = quotas[root], list(units.get(root, ())), 0
        for i in rng.permutation(len(pool)) if quota else ():
            unit = pool[int(i)]
            if taken + len(unit) > quota:
                continue
            chosen += unit
            taken += len(unit)
            counts[f"{root}_stop" if decision_class(unit[0].turns[-1]) == "stop" else f"{root}_other"] += len(unit)
            if taken == quota:
                break
        if taken < quota:
            raise SystemExit(f"replay root {root}: {quota} samples needed, its pool holds "
                             f"{sum(len(u) for u in pool)}; build a larger replay stream")
        counts[root] = taken
    return chosen, counts


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _write_job(job: tuple) -> tuple[int, str, int]:
    index, entries, out_dir = job
    path = Path(out_dir) / shard_name(index)
    return index, write_shard(path, [read_entry(e) for e in entries]), path.stat().st_size


def _input_info(stream_dir: str, manifest: Mapping[str, Any], manifest_sha256: str) -> dict[str, Any]:
    return {
        "dir": str(Path(stream_dir).resolve()),
        "manifest_sha256": manifest_sha256,
        "schema": manifest.get("schema"),
        "builder": manifest.get("builder"),
        "seed": manifest.get("seed"),
        "total_samples": manifest.get("total_samples"),
        "holdout": manifest.get("holdout"),
        "stream_sha256": hashlib.sha256("".join(s["sha256"] for s in manifest["shards"]).encode()).hexdigest(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dagger-stream", required=True, help="build_route_stream.py --no-pad output")
    parser.add_argument("--replay-stream", required=True, help="build_replay_stream.py output")
    parser.add_argument("--holdout-scenes-file", required=True, help="the held-out scenes (EXP-28: 20 HM3D scenes)")
    parser.add_argument("--replay-fraction", type=float, default=0.3)
    parser.add_argument("--replay-shares", default="equal", help="equal | pooled | NAME=W,NAME=W over the replay roots")
    parser.add_argument("--stop-upsample", type=int, default=5, help="the replay stream's --stop-upsample, checked")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--no-verify", action="store_true", help="skip the input shards' sha256 check")
    parser.add_argument("--allow-other-rules", action="store_true",
                        help="merge a DAgger stream off EXP-28's R1/R2/R6/R7 or draw off R8 anyway (the manifest records what differed)")
    parser.add_argument("--dry-run", action="store_true", help="draw and print the composition, write nothing")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.shard_size % args.global_batch or args.global_batch % args.world_size:
        raise SystemExit("shard size must be a multiple of the global batch, and that of the world size")
    if args.stop_upsample < 1:
        raise SystemExit("--stop-upsample must be >= 1")
    out_dir = Path(args.out_dir)
    if not args.dry_run and (out_dir / MANIFEST_NAME).exists():
        raise SystemExit(f"{out_dir} already holds a stream")
    started = time.time()
    verify = not args.no_verify

    # --- inputs, checked
    dagger_manifest, dagger, dagger_sha = load_stream(
        args.dagger_stream, schemas=(UNPADDED_SCHEMA, STREAM_SCHEMA), verify=verify)
    replay_manifest, replay, replay_sha = load_stream(args.replay_stream, schemas=(STREAM_SCHEMA,), verify=verify)
    if not str(dagger_manifest.get("builder", "")).startswith((DAGGER_BUILDER, RFT_BUILDER)):
        raise SystemExit(f"{args.dagger_stream} was built by {dagger_manifest.get('builder')!r}, "
                         f"not {DAGGER_BUILDER} or {RFT_BUILDER}")
    if not str(replay_manifest.get("builder", "")).startswith(REPLAY_BUILDER):
        raise SystemExit(f"{args.replay_stream} was built by {replay_manifest.get('builder')!r}, not {REPLAY_BUILDER}")
    if any(e.source != "dagger" for e in dagger):
        raise SystemExit(f"{args.dagger_stream} holds non-DAgger samples: build it without --replay-root / --merge-a-manifest")
    if any(e.source != "replay" for e in replay):
        raise SystemExit(f"{args.replay_stream} holds non-replay samples")
    if dagger_keys_sha256(dagger) != dagger_manifest["selection"]["selected_keys_sha256"]:
        raise SystemExit(f"{args.dagger_stream}: the samples read back do not match its selected_keys_sha256")
    if replay_keys_sha256(replay) != replay_manifest["selection"]["keys_sha256"]:
        raise SystemExit(f"{args.replay_stream}: the samples read back do not match its keys_sha256")
    stop_dagger = [e.key for e in dagger if decision_class(e.turns[-1]) == "stop"]
    if stop_dagger:
        raise SystemExit(f"{len(stop_dagger)} DAgger samples target STOP (R2 allows none), e.g. {stop_dagger[:3]}")
    off_rules = dagger_rule_mismatches(dagger_manifest)
    if off_rules and not args.allow_other_rules:
        raise SystemExit(f"{args.dagger_stream} was not built with EXP-28's rules: {off_rules} "
                         f"(expected {EXP28_DAGGER_KNOBS}, a --train-scenes-file, labels of the default "
                         f"RouteLabelConfig {RouteLabelConfig().digest()}, anchors = 2 x corrections); "
                         f"--allow-other-rules merges it anyway")
    train = {scene_key(s) for s in ((dagger_manifest.get("selection") or {}).get("train_scenes") or {}).get("scenes", ())}
    if train:
        outside = Counter(scene_key(sample_scene(e.key)) for e in dagger if scene_key(sample_scene(e.key)) not in train)
        if outside:
            raise SystemExit(f"DAgger samples of scenes outside the stream's training scenes (per scene): {dict(outside)}")
    upsample = int(replay_manifest["selection"].get("stop_upsample") or 1)
    if upsample != args.stop_upsample:
        raise SystemExit(f"the replay stream repeats each STOP {upsample}x, --stop-upsample says {args.stop_upsample}")

    holdout_file = read_scene_list(args.holdout_scenes_file)
    exp23 = holdout_scenes()
    held = {scene_key(s) for s in (*holdout_file, *exp23,
                                   *(dagger_manifest.get("holdout") or {}).get("scenes", ()),
                                   *(replay_manifest.get("holdout") or {}).get("scenes", ()))}
    leaks = Counter(scene_key(sample_scene(e.key)) for e in (*dagger, *replay) if scene_key(sample_scene(e.key)) in held)
    if leaks:
        raise SystemExit(f"held-out scenes in the inputs (samples per scene): {dict(leaks)}")

    # --- the draw
    roots = sorted((replay_manifest.get("roots") or {}).keys())
    shares = parse_shares(args.replay_shares, roots)
    root_of = None if args.replay_shares == "pooled" else scene_roots(replay_manifest["roots"])
    units, unit_counts = replay_units(replay, root_of=root_of, stop_upsample=args.stop_upsample)
    total, n_replay = stream_size(len(dagger), args.replay_fraction, args.shard_size)
    rng = np.random.default_rng(args.seed)
    replay_chosen, draw_counts = draw_replay(units, n_replay, shares, rng)
    chosen = list(dagger) + replay_chosen
    order = np.random.default_rng(args.seed + 1).permutation(len(chosen))
    items = [chosen[int(i)] for i in order]

    root_by_entry = {id(e): root for root, us in units.items() for u in us for e in u}
    drawn: Counter = Counter()
    by_dataset: Counter = Counter()
    for entry in items:
        drawn[entry.source] += 1
        drawn[f"{entry.source}_target_{decision_class(entry.turns[-1])}"] += 1
        if entry.source == "replay":
            drawn[f"replay_root_{root_by_entry[id(entry)]}"] += 1
            by_dataset[replay_dataset(entry.key)] += 1
    off_merge = merge_rule_mismatches(args.replay_fraction, args.stop_upsample, by_dataset)
    for key, value in (dagger_manifest.get("drawn") or {}).items():
        if key.startswith("category_"):
            drawn[key] = int(value)  # line B's categories, every DAgger sample taken once
    drawn["category_replay"] = n_replay
    stops = sum(v for k, v in drawn.items() if k.endswith("_target_stop"))
    composition = {
        "total_samples": total,
        "num_shards": total // args.shard_size,
        "steps_at_global_batch": total // args.global_batch,
        "dagger": len(dagger),
        "replay": n_replay,
        "shares": {
            "dagger": round(len(dagger) / total, 4),
            "replay": round(n_replay / total, 4),
            "stop_targets": round(stops / total, 4),
            **{f"replay_{r}": round(draw_counts[r] / total, 4) for r in sorted(shares)},
        },
        "replay_quota": apportion(n_replay, shares),
        "replay_units": dict(unit_counts),
        "replay_draw": dict(draw_counts),
        "replay_by_dataset": dict(by_dataset),
        "dagger_reversals": ((dagger_manifest.get("selection") or {}).get("composition") or {}).get("reversals"),
    }
    print(json.dumps(composition, indent=1, sort_keys=True), flush=True)
    if off_merge and not args.allow_other_rules:
        raise SystemExit(f"this merge is not EXP-28's R8: {off_merge} (expected replay fraction "
                         f"{EXP28_REPLAY_FRACTION}, STOP x{EXP28_STOP_UPSAMPLE}, replay half InternData r2r and half "
                         f"ScaleVLN: one replay root of each, --replay-shares equal); --allow-other-rules merges anyway")
    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    num_shards = total // args.shard_size
    jobs = [(s, items[s * args.shard_size:(s + 1) * args.shard_size], str(out_dir)) for s in range(num_shards)]
    shards: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (index, digest, size) in enumerate(pool.map(_write_job, jobs, chunksize=1), 1):
            shards[index] = {"name": shard_name(index), "sha256": digest, "bytes": size}
            if done % 10 == 0 or done == num_shards:
                print(f"[merge] {done}/{num_shards} shards", flush=True)

    holdout = sorted(held)
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
        "replay_fraction": round(n_replay / total, 6),
        "holdout": {
            "scenes": holdout,
            "count": len(holdout),
            "file": str(Path(args.holdout_scenes_file).resolve()),
            "file_sha256": hashlib.sha256(Path(args.holdout_scenes_file).read_bytes()).hexdigest(),
            "file_scenes": list(holdout_file),
            "exp23_scenes": list(exp23),
        },
        # train_system2_lora.py copies these three into the adapter manifest
        "record_roots": list(dagger_manifest.get("record_roots") or []),
        "label_roots": list(dagger_manifest.get("label_roots") or []),
        "replay_root": replay_manifest.get("replay_root", ""),
        "inputs": {
            "dagger": _input_info(args.dagger_stream, dagger_manifest, dagger_sha),
            "replay": _input_info(args.replay_stream, replay_manifest, replay_sha),
        },
        "selection": {
            "rule": "EXP-28 R7/R8: every line-B DAgger sample once (corrections, 2x anchors, no STOP target) plus "
                    "InternData expert replay to the replay fraction, split over its roots, STOP copies kept whole",
            "knobs": {"replay_fraction": args.replay_fraction, "replay_shares": args.replay_shares,
                      "stop_upsample": args.stop_upsample, "allow_other_rules": args.allow_other_rules},
            "dagger_rules": {"expected": EXP28_DAGGER_KNOBS, "mismatches": off_rules},
            "merge_rules": {"expected": {"replay_fraction": EXP28_REPLAY_FRACTION, "stop_upsample": EXP28_STOP_UPSAMPLE,
                                         "replay_by_dataset": "r2r and hm3d within one sample"},
                            "mismatches": off_merge},
            "replay_shares": shares,
            "composition": composition,
            "dagger": dagger_manifest.get("selection"),
            "replay": replay_manifest.get("selection"),
            "dagger_selected": len(dagger),
            "selected_keys_sha256": dagger_keys_sha256(dagger),
            "keys_sha256": hashlib.sha256(
                "\n".join(e.key + "|" + "\x1f".join(e.turns) for e in items).encode()).hexdigest(),
            "stop_share": composition["shares"]["stop_targets"],
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

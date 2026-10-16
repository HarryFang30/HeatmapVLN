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

EXP-28 (ledger rules R1, R2, R6, R7) adds filters that run before the per-episode caps;
each is off unless asked for, so EXP-25's stream rebuilds as it was:

==========================================  =================================================
``--holdout-scenes-file F``                 the held-out scenes (one per line, ``#``
                                            comments) instead of EXP-23's salted six; HM3D's
                                            ``00xxx-ID`` directory names and the bare
                                            11-character ID match each other (records live
                                            under the ID)
``--train-scenes-file F``                   the training scenes (R1): a record of a scene that
                                            is neither held out nor listed here is an error
``--first-calls 48``                        only the first 48 calls of an episode (R1)
``--min-distance 3``                        only calls beyond 3 m of the goal (R1); the
                                            labeller already skips them, this counts any it
                                            did not
``--stop-correction-min-distance 5``        a correction of the model's own STOP (its target
                                            is the oracle's pixel or turn, never STOP) only
                                            beyond 5 m ... (R2)
``--max-stop-corrections-per-episode 1``    ... and only the earliest one per episode (R2)
``--drop-reversals``                        drop a sample whose target turns net against the
                                            turn the server executed on the episode's
                                            previous call, corrections and anchors alike,
                                            and count it (R6)
``--no-pad``                                no padding to whole shards: an input for
                                            ``scripts/exp28/merge_streams.py`` (which mixes
                                            in the replay and re-shards); its manifest schema
                                            is not the trainer's, so the trainer refuses it
==========================================  =================================================

The defaults already give R7: at most 8 corrections per episode (the earliest), at most 12
anchors (a seeded uniform subset), anchors = 2 x corrections, every weight 1.

Every stream also records which labeller made its labels (``selection.labels``: the rows'
``labeler`` digest and ``line`` per record) and refuses label files whose ``record_sha256``
is not the record's committed sidecar (labels of another version of the record).

    /opt/conda/bin/python scripts/tools/build_route_stream.py \\
        --record-root /workspace/exp28/collect_base/records --label-root /workspace/exp28/collect_base/labels_routeB \\
        --holdout-scenes-file /workspace/exp28/episodes/splits/heldout_scenes.txt \\
        --train-scenes-file /workspace/exp28/episodes/splits/train_scenes.txt --first-calls 48 --min-distance 3 \\
        --stop-correction-min-distance 5 --max-stop-corrections-per-episode 1 --drop-reversals \\
        --no-pad --out-dir /workspace/exp28/round1/stream_dagger

``--dry-run`` stops after the selection and prints the composition.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools import build_spinbreak_stream as sb  # noqa: E402
from scripts.tools.build_s2_train_stream import _dagger_item, _replay_item, _write  # noqa: E402
from src.data.interndata_replay import scene_samples  # noqa: E402
from src.data.s2_oracle_labels import parse_model_answer  # noqa: E402
from src.data.s2_records import EpisodeRecord, iter_record_paths, label_path_for  # noqa: E402
from src.data.s2_route_labels import (  # noqa: E402
    LABEL_LINE,
    ROUTE_RULES_VERSION,
    RouteLabelConfig,
    answer_direction,
    executed_turn_direction,
)
from src.data.s2_sft_dataset import (  # noqa: E402
    DEFAULT_HOLDOUT_COUNT,
    DEFAULT_VLM_IMAGE_SIZE,
    HOLDOUT_SALT,
    STOP_TEXT,
    CallRef,
    S2SFTError,
    _pair_label_roots,
    _stable_rank,
    decision_class,
    holdout_scenes,
    index_calls,
    training_turns,
)
from src.data.s2_stream import MANIFEST_NAME, STREAM_SCHEMA, shard_name  # noqa: E402

BUILDER = "scripts/tools/build_route_stream.py (EXP-25 B)"
# A --no-pad stream: same shards, last one short.  Only merge_streams.py reads it; the
# trainer's read_manifest refuses the schema, so it cannot be trained on by mistake.
UNPADDED_SCHEMA = f"{STREAM_SCHEMA}+unpadded"
DISTANCE_BINS = ((3.0, 5.0), (5.0, 7.0), (7.0, 10.0), (10.0, math.inf))
_HM3D_SCENE_DIR = re.compile(r"^\d{5}-(?P<id>[A-Za-z0-9]{11})$")


# --------------------------------------------------------------------------- #
# Scenes
# --------------------------------------------------------------------------- #
def scene_key(name: str) -> str:
    """The scene's bare ID: HM3D's ``00481-5jLhtVmWd5F``, ``5jLhtVmWd5F`` and the ``5jLhtVmWd5F.basis``
    of an un-linked ``.basis.glb`` stem all meet here (MP3D names are unchanged)."""
    name = str(name).strip()
    name = name[: -len(".basis")] if name.endswith(".basis") else name
    match = _HM3D_SCENE_DIR.match(name)
    return match.group("id") if match else name


def read_scene_list(path: str | os.PathLike) -> tuple[str, ...]:
    """Scene names from a file: a JSON list (or ``{"scenes": [...]}``), else text with one
    or more names per line (comma or blank separated) and ``#`` comments.  Sorted, unique."""
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith(("[", "{")):
        data = json.loads(text)
        names = data.get("scenes") if isinstance(data, Mapping) else data
        if not isinstance(names, list):
            raise SystemExit(f"{path}: a JSON scene list is a list or {{\"scenes\": [...]}}")
    else:
        names = [tok for line in text.splitlines() for tok in re.split(r"[,\s]+", line.split("#", 1)[0]) if tok]
    names = sorted({str(n).strip() for n in names if str(n).strip()})
    if not names:
        raise SystemExit(f"{path} names no scene")
    return tuple(names)


def holdout_match_set(holdout: Iterable[str]) -> set[str]:
    """Names to hand ``index_calls``: each held-out name as given and as its bare ID."""
    holdout = list(holdout)
    return set(holdout) | {scene_key(s) for s in holdout}


def drop_held_out(refs: Sequence[CallRef], holdout: Iterable[str]) -> tuple[list[CallRef], int]:
    """Refs whose scene (record dir or episode scene_id) is not held out, by bare ID."""
    keys = {scene_key(s) for s in holdout}
    kept = [r for r in refs if scene_key(Path(r.record_path).parent.name) not in keys and scene_key(r.scene_id) not in keys]
    return kept, len(refs) - len(kept)


def outside_scenes(refs: Sequence[CallRef], scenes: Iterable[str]) -> Counter:
    """Calls per scene (bare ID) of the refs whose record dir or episode scene is not in ``scenes``."""
    keys = {scene_key(s) for s in scenes}
    return Counter(scene_key(Path(r.record_path).parent.name) for r in refs
                   if scene_key(Path(r.record_path).parent.name) not in keys or scene_key(r.scene_id) not in keys)


# --------------------------------------------------------------------------- #
# Which labeller labelled what
# --------------------------------------------------------------------------- #
def label_provenance(
    record_roots: Sequence[str], label_roots: Sequence[str], record_paths: Iterable[str]
) -> dict[str, Any]:
    """Per labelled record (the first row of its label file; the labeller writes one digest per
    file): the ``labeler`` config digest, the label ``line``, and whether ``record_sha256`` is the
    record's committed sidecar.  A stale file labels another version of the record, so its
    call indices may not be this record's."""
    label_of: dict[str, Path] = {}
    for record_root, label_root in _pair_label_roots(record_roots, label_roots):
        for path in iter_record_paths(record_root):
            label_of.setdefault(str(path), label_path_for(path, label_root))
    labelers: Counter = Counter()
    lines: Counter = Counter()
    stale: list[str] = []
    for record in sorted(set(record_paths)):
        with open(label_of[record], encoding="utf-8") as handle:
            first = json.loads(handle.readline())
        labelers[str(first.get("labeler"))] += 1
        lines[str(first.get("line"))] += 1
        sidecar = Path(record).with_name(Path(record).name + ".sha256").read_text(encoding="utf-8").split()[0]
        if first.get("record_sha256") is not None and first["record_sha256"] != sidecar:
            stale.append(record)
    return {
        "records": sum(labelers.values()),
        "labelers": dict(labelers),
        "lines": dict(lines),
        "default_labeler": RouteLabelConfig().digest(),
        "route_rules_version": ROUTE_RULES_VERSION,
        "stale_label_files": len(stale),
        "stale_examples": stale[:5],
    }


# --------------------------------------------------------------------------- #
# Turn reversals (R6)
# --------------------------------------------------------------------------- #
def executed_direction(turns: Sequence[str] | None) -> int:
    """Net turn the server executes for an answer: +1 left, -1 right, 0 otherwise (pixel,
    STOP, mixed or cancelling arrows).  Same reading as the labeller's ``previous_turn_direction``."""
    if not turns:
        return 0
    return int(answer_direction(turns[0], turns[1] if len(turns) > 1 else None))


@functools.lru_cache(maxsize=256)
def record_directions(record_path: str) -> dict[int, int]:
    """call_index -> net turn the server executed, for every call of the record, read as the
    labeller reads its previous call (``parse_model_answer``; a malformed answer is 0)."""
    record = EpisodeRecord(record_path)
    return {int(c["call_index"]): executed_turn_direction(parse_model_answer(c)) for c in record.calls}


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
    first_calls: int | None = None,
    min_distance_m: float | None = None,
    stop_correction_min_distance_m: float | None = None,
    max_stop_corrections_per_episode: int | None = None,
    drop_reversals: bool = False,
    directions_from_record: Callable[[str], Mapping[int, int]] = record_directions,
) -> tuple[list[CallRef], list[CallRef], Counter]:
    """(corrections, anchors) with targets, after the filters and the per-episode caps, plus counts.

    Per episode, in call order: calls at or past ``first_calls`` go, and so do calls not beyond
    ``min_distance_m`` of the goal (or of unknown distance); a correction of the
    model's own STOP (what the server executed: ``decision_class`` stop, which includes its
    fallback STOP for an answer without an action) stays only beyond
    ``stop_correction_min_distance_m``; with
    ``drop_reversals`` a sample whose target nets against the turn executed on the previous
    call goes.  Then the caps: the earliest ``max_stop_corrections_per_episode`` STOP
    corrections, the earliest (or a seeded uniform subset of) ``max_corrections_per_episode``
    corrections, a seeded uniform ``max_anchors_per_episode`` anchors.  The previous call is
    the episode's call just before (as the labeller takes it); its answer comes from
    ``refs`` (the labeller writes a row for every call) or, when that call has no label,
    from the record (``directions_from_record``, read once per record).
    """
    if correction_pick not in ("earliest", "uniform"):
        raise ValueError(f"correction_pick must be earliest or uniform, not {correction_pick!r}")
    stats: Counter = Counter()
    by_episode: dict[str, list[CallRef]] = defaultdict(list)
    directions: dict[str, dict[int, int]] = defaultdict(dict)  # record -> call_index -> executed net turn
    read_records: set[str] = set()
    for ref in refs:
        directions[ref.record_path][ref.call_index] = executed_direction(ref.recorded)
        if ref.label is None:
            stats["unlabelled"] += 1
        elif ref.label.skip:
            stats["no_target"] += 1
        else:
            by_episode[ref.record_path].append(ref)

    def previous_direction(ref: CallRef) -> int:
        if ref.call_index <= 0:
            return 0  # calls count from 0: the episode's first call has none before it
        known = directions[ref.record_path]
        if ref.call_index - 1 in known:
            return known[ref.call_index - 1]
        if ref.record_path not in read_records:
            read_records.add(ref.record_path)
            stats["previous_answer_read_from_record"] += 1
            for index, direction in directions_from_record(ref.record_path).items():
                known.setdefault(index, direction)
        earlier = [i for i in known if i < ref.call_index]
        return known[max(earlier)] if earlier else 0

    corrections: list[CallRef] = []
    anchors: list[CallRef] = []
    for episode, calls in by_episode.items():
        mine_c, mine_a, stop_c = [], [], []
        for ref in sorted(calls, key=lambda r: r.call_index):
            if first_calls is not None and ref.call_index >= first_calls:
                stats["beyond_first_calls"] += 1
                continue
            if min_distance_m is not None and not (ref.distance_to_goal is not None
                                                   and ref.distance_to_goal > min_distance_m):
                stats["within_min_distance"] += 1  # the default labeller skips these: > 0 means another config
                continue
            turns = training_turns(ref.recorded, ref.label)
            if turns == (STOP_TEXT,) or decision_class(turns[-1]) == "stop":
                raise S2SFTError(f"{ref.key}: line B never targets STOP, got {turns!r}")
            agree = bool(ref.label.agree)
            if agree and ref.label.target_turns not in (None, ref.recorded):
                raise S2SFTError(f"{ref.key}: accepted target {ref.label.target_turns!r} is not the "
                                 f"recorded answer {ref.recorded!r}")
            part = "anchor" if agree else "correction"
            stats[f"{part}s_labelled"] += 1
            stop_answer = not agree and decision_class(ref.recorded[-1]) == "stop"
            if stop_answer:
                stats["stop_answer_corrections_labelled"] += 1
                d = ref.distance_to_goal
                if stop_correction_min_distance_m is not None and not (d is not None and d > stop_correction_min_distance_m):
                    stats["stop_answer_corrections_dropped_near"] += 1
                    continue
            previous = previous_direction(ref)
            if previous and executed_direction(turns) == -previous:
                if drop_reversals:
                    stats[f"reversal_dropped_{part}s"] += 1
                    continue
                stats[f"reversed_{part}s_kept"] += 1
            ref = dataclasses.replace(ref, turns=turns)
            if agree:
                mine_a.append(ref)
            else:
                mine_c.append(ref)
                if stop_answer:
                    stop_c.append(ref)
        if max_stop_corrections_per_episode is not None and len(stop_c) > max_stop_corrections_per_episode:
            dropped = {r.uid for r in stop_c[max_stop_corrections_per_episode:]}  # stop_c is in call order
            stats["stop_answer_corrections_dropped_by_cap"] += len(dropped)
            mine_c = [r for r in mine_c if r.uid not in dropped]
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
    stats["stop_answer_corrections"] = sum(decision_class(r.recorded[-1]) == "stop" for r in corrections)
    return corrections, anchors, stats


def reversal_report(stats: Mapping[str, int]) -> dict[str, Any]:
    """R6's numbers: reversal drops, and their share of the disagreeing (correction) samples."""
    labelled = int(stats.get("corrections_labelled", 0))
    dropped_c = int(stats.get("reversal_dropped_corrections", 0))
    dropped_a = int(stats.get("reversal_dropped_anchors", 0))
    return {
        "dropped_corrections": dropped_c,
        "dropped_anchors": dropped_a,
        # the denominator: every disagreeing row with a target in the first calls (R2's near STOPs included)
        "corrections_labelled": labelled,
        "share_of_corrections": round(dropped_c / labelled, 4) if labelled else None,
        "reversed_targets_kept": int(stats.get("reversed_corrections_kept", 0)) + int(stats.get("reversed_anchors_kept", 0)),
    }


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
    held = {scene_key(s) for s in holdout}
    for scene_dir in sorted(p for p in Path(replay_root).iterdir() if p.is_dir() and not p.name.startswith(".")):
        if scene_key(scene_dir.name) in held:
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
    parser.add_argument("--first-calls", type=int, default=-1,
                        help="only calls with call_index below this (EXP-28 R1: 48; -1: all)")
    parser.add_argument("--min-distance", type=float, default=None,
                        help="only calls beyond this many metres of the goal (EXP-28 R1: 3; the labeller already skips them)")
    parser.add_argument("--stop-correction-min-distance", type=float, default=None,
                        help="keep a correction of the model's own STOP only beyond this many metres (EXP-28 R2: 5)")
    parser.add_argument("--max-stop-corrections-per-episode", type=int, default=-1,
                        help="the earliest N corrections of the model's own STOP per episode (EXP-28 R2: 1; -1: no cap)")
    parser.add_argument("--drop-reversals", action="store_true",
                        help="drop samples whose target turns net against the previous call's executed turn (EXP-28 R6)")
    parser.add_argument("--no-pad", action="store_true",
                        help="do not pad to whole shards: an input for scripts/exp28/merge_streams.py, not for the trainer")
    parser.add_argument("--merge-a-manifest", default="", help="line A's stream manifest: merge its DAgger samples")
    parser.add_argument("--merge-a-fraction", type=float, default=1.0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--holdout-scenes-file", default="",
                        help="held-out scenes, one per line (EXP-28: the 20 HM3D scenes); overrides --holdout-count/--holdout-salt")
    parser.add_argument("--train-scenes-file", default="",
                        help="training scenes, one per line (EXP-28 R1: the 285); a record of any other scene not held out is an error")
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
    if args.no_pad and (args.replay_root or args.merge_a_manifest):
        raise SystemExit("--no-pad builds the DAgger part alone for merge_streams.py: no --replay-root, no --merge-a-manifest")
    out_dir = Path(args.out_dir)
    if not args.dry_run and (out_dir / MANIFEST_NAME).exists():
        raise SystemExit(f"{out_dir} already holds a stream")
    started = time.time()
    rng = np.random.default_rng(args.seed)
    if args.holdout_scenes_file:
        holdout = read_scene_list(args.holdout_scenes_file)
        holdout_info: dict[str, Any] = {
            "scenes": list(holdout), "count": len(holdout), "file": str(Path(args.holdout_scenes_file).resolve()),
            "file_sha256": hashlib.sha256(Path(args.holdout_scenes_file).read_bytes()).hexdigest()}
    else:
        holdout = holdout_scenes(count=args.holdout_count, salt=args.holdout_salt)
        holdout_info = {"scenes": list(holdout), "salt": args.holdout_salt, "count": len(holdout)}

    # --- line B
    refs, index_stats = index_calls(args.record_root, args.label_root, split="train", holdout=holdout_match_set(holdout))
    refs, held = drop_held_out(refs, holdout)  # a record dir named otherwise than the list (00xxx-ID vs ID)
    if held:
        index_stats["calls_held_out_by_scene_id"] = held
    train_info: dict[str, Any] = {}
    if args.train_scenes_file:
        train = read_scene_list(args.train_scenes_file)
        clash = sorted({scene_key(s) for s in train} & {scene_key(s) for s in holdout})
        if clash:
            raise SystemExit(f"{args.train_scenes_file} lists held-out scenes: {clash}")
        outside = outside_scenes(refs, train)
        if outside:
            raise SystemExit(f"records of scenes neither held out nor in {args.train_scenes_file} "
                             f"(calls per scene): {dict(sorted(outside.items())[:20])} ({len(outside)} scenes)")
        train_info = {"scenes": sorted({scene_key(s) for s in train}), "count": len(train),
                      "file": str(Path(args.train_scenes_file).resolve()),
                      "file_sha256": hashlib.sha256(Path(args.train_scenes_file).read_bytes()).hexdigest()}
    labels_info = label_provenance(args.record_root, args.label_root, {r.record_path for r in refs})
    if labels_info["stale_label_files"]:
        raise SystemExit(f"{labels_info['stale_label_files']} label files were made for another version of their "
                         f"record (record_sha256 is not the sidecar), e.g. {labels_info['stale_examples']}: relabel them")
    if set(labels_info["labelers"]) != {labels_info["default_labeler"]} or set(labels_info["lines"]) != {LABEL_LINE}:
        print(f"[stream] note: labels by {labels_info['labelers']} (line {labels_info['lines']}); the default "
              f"RouteLabelConfig is {labels_info['default_labeler']}", file=sys.stderr, flush=True)
    corrections, anchors, pool_stats = route_pools(
        refs,
        max_corrections_per_episode=None if args.max_corrections_per_episode < 0 else args.max_corrections_per_episode,
        max_anchors_per_episode=None if args.max_anchors_per_episode < 0 else args.max_anchors_per_episode,
        correction_pick=args.correction_pick,
        seed=args.seed,
        first_calls=None if args.first_calls < 0 else args.first_calls,
        min_distance_m=args.min_distance,
        stop_correction_min_distance_m=args.stop_correction_min_distance,
        max_stop_corrections_per_episode=(None if args.max_stop_corrections_per_episode < 0
                                          else args.max_stop_corrections_per_episode),
        drop_reversals=args.drop_reversals,
    )
    if not corrections:
        raise SystemExit(f"no line-B corrections (index {dict(index_stats)}, pools {dict(pool_stats)})")
    if pool_stats["within_min_distance"]:
        print(f"[stream] note: {pool_stats['within_min_distance']} labelled calls within {args.min_distance} m dropped; "
              f"the default labeller labels none there", file=sys.stderr, flush=True)
    reversals = reversal_report(pool_stats)
    if reversals["share_of_corrections"] is not None and reversals["share_of_corrections"] > 0.4:
        # R6: report it, change nothing
        print(f"[stream] note: reversal drops are {reversals['share_of_corrections']:.1%} of the corrections (> 40%)",
              file=sys.stderr, flush=True)
    taken_c = take(corrections, args.correction_weight, rng)
    n_anchor = int(round(args.anchors_per_correction * len(taken_c)))
    anchors_capped = n_anchor > len(anchors)
    if anchors_capped:
        print(f"[stream] warning: {n_anchor} anchors wanted, the capped pool holds {len(anchors)}",
              file=sys.stderr, flush=True)
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
    if args.no_pad:
        total = dagger_n  # merge_streams.py adds the replay and pads
    elif replay:
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
        "num_shards": int(math.ceil(total / args.shard_size)),
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
        "B_stop_answer_corrections": sum(decision_class(ref.recorded[-1]) == "stop" for ref in taken_c),
        "anchors_capped_by_pool": anchors_capped,
        "reversals": reversals,
        "padded": not args.no_pad,
    }
    print(json.dumps(composition, indent=1, sort_keys=True), flush=True)
    if args.dry_run:
        print(json.dumps({"index": dict(index_stats), "pools": dict(pool_stats), "replay": dict(replay_counts),
                          "merge_a": merge_info, "labels": labels_info,
                          "train_scenes": {k: v for k, v in train_info.items() if k != "scenes"}},
                         indent=1, sort_keys=True))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    num_shards = composition["num_shards"]  # with --no-pad the last shard is short
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
        "schema": UNPADDED_SCHEMA if args.no_pad else STREAM_SCHEMA,
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
        "holdout": holdout_info,
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
                "first_calls": args.first_calls,
                "min_distance": args.min_distance,
                "stop_correction_min_distance": args.stop_correction_min_distance,
                "max_stop_corrections_per_episode": args.max_stop_corrections_per_episode,
                "drop_reversals": args.drop_reversals,
                "no_pad": args.no_pad,
                "holdout_scenes_file": args.holdout_scenes_file,
                "train_scenes_file": args.train_scenes_file,
            },
            "index": dict(index_stats),
            "pools": dict(pool_stats),
            "labels": labels_info,
            "train_scenes": train_info,
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

#!/usr/bin/env python3
"""Pack EXP-32's RFT stream: the model's own sampled answers on its successful trajectories, plus 1:1 base anchors.

The ledger's EXP-32 第二部分 C (pre-registered before any part-1 reading) fixes the rules; this
script is RF1-RF7, and ``scripts/exp28/merge_streams.py --allow-other-rules`` adds RF8/RF9 (the
r2r replay at 20%, one seeded shuffle, shards of 256):

=====  =====================================================================================
RF1    trajectories of ``--rft-source TAG=ROOT:SEED`` (p1 = pass 1, r0s = EXP-31 R0-S, p2, p3):
       committed records (tar + ``.sha256``) with ``outcome.success == 1``, of a scene in
       ``--train-scenes-file`` (a held-out scene -- ``--holdout-scenes-file`` or EXP-23's six --
       or any other scene in a source is refused), served by the base model (``served_adapter``
       None).  Per episode at most ``--max-trajectories-per-episode`` (2), in the order p1, r0s,
       p2, p3; a trajectory whose recorded answers equal an earlier one's call for call (R0-S
       rerunning pass 1's episode the same way) is left out.
RF2    calls: ``call_index < --first-calls`` (48), the record's ``distance_to_goal`` (calls.jsonl,
       as ``index_calls(require_labels=False)`` reads it) beyond ``--min-distance`` (3 m), final
       decision class pixel, turn_left or turn_right.  A call without a distance, STOP, forward,
       invalid or a lookdown gives no sample; each is counted.
RF3    a call whose answer turns net against the turn executed on the record's previous call is
       left out (EXP-28 R6: ``build_route_stream.executed_direction`` / ``record_directions``).
RF4    target: the recorded answer verbatim (``recorded_turns``; both turns after a lookdown).
       No labels, no oracle, no further filter.
RF5    anchors: ``--anchor-root`` (``collect_base``: the base model, greedy, System1 seed 42),
       episodes of cohort shard >= 20 (``episode_list``'s ``shard_NN.json``) with success 1, the
       calls of RF2/RF3, ranked by sha256("exp32-anchor:<--anchor-seed>:" + uid) (uid = the
       CallRef's ``record_path#call_index``, the path under the resolved ``--anchor-root`` the
       manifest names), at most ``--max-anchors-per-episode`` (6) per
       episode, ``--anchor-ratio`` (1.0) x the RFT samples in all (a pool too small is taken whole
       and reported: ``anchors_capped_by_pool``).
RF6    every sample once, weight 1.
RF7    keys ``{scene}_{episode:06d}#{tag}.{call}`` (tag p1, r0s, p2, p3 or g for an anchor), which
       ``merge_streams.sample_scene`` reads.
=====  =====================================================================================

A4 (System1 seed): every call's ``sampling.protocol_seed`` must be the source's (p1 and r0s
1337, p2 42, p3 2024, anchors 42; a declaration off these is refused).  A record that differs is
left out and counted; more than 1% of a source's records differing stops the build.

``rft_calls(record)`` is RF2/RF3 on one record, for ``aud.py`` and ``rft_gate.py``.

Output (``--no-pad`` is required: the stream is only an input for ``merge_streams.py``, whose
manifest schema the trainer refuses): unpadded shards packed by ``build_s2_train_stream._write``
in one seeded shuffle, ``stream_manifest.json``, ``rft_records.txt`` (``record_path<TAB>
record_device<TAB>tag``, one line per RFT trajectory taken) and ``keys.tsv`` (``key<TAB>
record_path#call_index<TAB>category<TAB>record_device``, one line per sample, stream order).

    /opt/conda/bin/python scripts/exp32/build_rft_stream.py \\
        --rft-source p1=$P1/records:1337 --rft-source r0s=/workspace/exp31/r0s/records:1337 \\
        --rft-source p2=/workspace/exp32/p2/records:42 \\
        --anchor-root /workspace/exp28/collect_base/records \\
        --train-scenes-file /workspace/exp28/episodes/splits/train_scenes.txt \\
        --holdout-scenes-file /workspace/exp28/episodes/splits/heldout_scenes.txt \\
        --first-calls 48 --min-distance 3 --max-anchors-per-episode 6 --anchor-ratio 1.0 --anchor-seed 32 \\
        --max-trajectories-per-episode 2 --no-pad --out-dir /workspace/exp32/stream_rft

``--dry-run`` stops after the selection and prints the composition (``drawn``'s
``category_rft_*`` sum is B2's N_rft); it writes nothing.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.exp28.hm3d_route_gate import served_adapter  # noqa: E402
from scripts.tools.build_route_stream import (  # noqa: E402
    UNPADDED_SCHEMA,
    executed_direction,
    holdout_match_set,
    read_scene_list,
    record_directions,
    scene_key,
)
from scripts.tools.build_s2_train_stream import _write  # noqa: E402
from src.data.s2_records import EpisodeRecord, iter_record_paths  # noqa: E402
from src.data.s2_sft_dataset import (  # noqa: E402
    DEFAULT_VLM_IMAGE_SIZE,
    CallRef,
    decision_class,
    holdout_scenes,
    index_calls,
    record_device,
    recorded_turns,
)
from src.data.s2_stream import MANIFEST_NAME, shard_name  # noqa: E402

BUILDER = "scripts/exp32/build_rft_stream.py (EXP-32)"
RFT_TAGS = ("p1", "r0s", "p2", "p3")  # RF1's sources, in the order an episode keeps them
ANCHOR_TAG = "g"
# A4: the System1 protocol seed each source was collected with (ledger EXP-32 第二部分 A4)
DECLARED_SEEDS = {"p1": 1337, "r0s": 1337, "p2": 42, "p3": 2024}
ANCHOR_PROTOCOL_SEED = 42
MAX_SEED_MISMATCH_SHARE = Fraction(1, 100)  # A4: more than 1% of a source's records -> stop
ANCHOR_MIN_SHARD = 20  # RF5: collect_base's shards 0-19 are the 400 held-out episodes (proxy P's base arm)
RFT_CLASSES = ("pixel", "turn_left", "turn_right")  # RF2
DEFAULT_FIRST_CALLS = 48
DEFAULT_MIN_DISTANCE = 3.0
RECORDS_LIST = "rft_records.txt"
KEYS_TSV = "keys.tsv"
_SHARD = re.compile(r"shard_(\d+)\.json")


# --------------------------------------------------------------------------- #
# RF2 / RF3 on calls (shared with aud.py and rft_gate.py through rft_calls)
# --------------------------------------------------------------------------- #
def previous_direction(directions: Mapping[int, int], call_index: int) -> int:
    """Net turn executed on the record's call just before ``call_index`` (0: none before it)."""
    if call_index - 1 in directions:
        return int(directions[call_index - 1])
    earlier = [i for i in directions if i < call_index]
    return int(directions[max(earlier)]) if earlier else 0


def call_verdict(
    call_index: int,
    turns: Sequence[str],
    distance_to_goal: float | None,
    previous: int,
    *,
    first_calls: int = DEFAULT_FIRST_CALLS,
    min_distance: float = DEFAULT_MIN_DISTANCE,
) -> str | None:
    """None when the call is an RFT sample (RF2, RF3), else why not: ``beyond_first_calls``,
    ``no_distance``, ``within_min_distance``, ``class_<decision>`` (stop, forward, invalid, lookdown)
    or ``reversal`` (the answer nets against ``previous``, the turn executed on the call before)."""
    if call_index >= first_calls:
        return "beyond_first_calls"
    if distance_to_goal is None:
        return "no_distance"
    if not float(distance_to_goal) > min_distance:
        return "within_min_distance"
    decision = decision_class(turns[-1])
    if decision not in RFT_CLASSES:
        return f"class_{decision}"
    if previous and executed_direction(turns) == -previous:
        return "reversal"
    return None


@dataclass(frozen=True)
class RftCall:
    """One call of a record that RF2/RF3 keep: its index, the recorded answer (the RFT target), d."""

    call_index: int
    turns: tuple[str, ...]
    distance_to_goal: float
    decision: str


def rft_calls(
    record: EpisodeRecord | str | os.PathLike,
    *,
    first_calls: int = DEFAULT_FIRST_CALLS,
    min_distance: float = DEFAULT_MIN_DISTANCE,
) -> tuple[list[RftCall], Counter]:
    """RF2/RF3 over every call of one record: (the calls kept, in call order; counts of ``kept`` and of
    each drop reason of ``call_verdict``).  Call-level only: RF1's record rules (success, seed, adapter,
    scene) are the caller's.  Reads the answers as ``index_calls`` does (``recorded_turns``, the call's
    ``distance_to_goal``) and the previous call's executed turn as ``record_directions`` does, so it keeps
    exactly the calls this builder packs."""
    if not isinstance(record, EpisodeRecord):
        record = EpisodeRecord(record)
    directions = record_directions(str(record.path))
    kept: list[RftCall] = []
    counts: Counter = Counter()
    for call in sorted(record.calls, key=lambda c: int(c["call_index"])):
        index = int(call["call_index"])
        turns = recorded_turns(call)
        distance = call.get("distance_to_goal")
        distance = float(distance) if distance is not None else None
        why = call_verdict(index, turns, distance, previous_direction(directions, index),
                           first_calls=first_calls, min_distance=min_distance)
        if why is not None:
            counts[why] += 1
            continue
        counts["kept"] += 1
        kept.append(RftCall(index, turns, float(distance), decision_class(turns[-1])))
    return kept, counts


def filter_refs(
    refs: Sequence[CallRef], *, first_calls: int, min_distance: float,
) -> tuple[list[CallRef], Counter]:
    """``rft_calls`` on one record's CallRefs (every call of the record, from ``index_calls``):
    the refs kept, with ``turns`` = the recorded answer (RF4), and the counts."""
    directions = record_directions(refs[0].record_path) if refs else {}
    kept: list[CallRef] = []
    counts: Counter = Counter()
    for ref in sorted(refs, key=lambda r: r.call_index):
        why = call_verdict(ref.call_index, ref.recorded, ref.distance_to_goal,
                           previous_direction(directions, ref.call_index),
                           first_calls=first_calls, min_distance=min_distance)
        if why is not None:
            counts[why] += 1
            continue
        counts["kept"] += 1
        kept.append(dataclasses.replace(ref, turns=tuple(ref.recorded)))
    return kept, counts


# --------------------------------------------------------------------------- #
# RF1: records
# --------------------------------------------------------------------------- #
def parse_source(text: str) -> tuple[str, str, int]:
    """``TAG=ROOT:SEED`` -> (tag, root, seed); the seed must be A4's for the tag."""
    tag, sep, rest = str(text).partition("=")
    root, colon, seed = rest.rpartition(":")
    if not sep or not colon or not root or tag not in RFT_TAGS:
        raise SystemExit(f"--rft-source takes TAG=ROOT:SEED with TAG in {list(RFT_TAGS)}; got {text!r}")
    try:
        seed_value = int(seed)
    except ValueError:
        raise SystemExit(f"--rft-source {text!r}: the seed {seed!r} is not an integer") from None
    if seed_value != DECLARED_SEEDS[tag]:
        raise SystemExit(f"--rft-source {text!r}: source {tag} was collected with System1 seed "
                         f"{DECLARED_SEEDS[tag]} (ledger EXP-32 A4), not {seed_value}")
    return tag, root, seed_value


def episode_shard(episode: Mapping[str, Any]) -> int | None:
    """The cohort shard of a record (``episode_list``'s ``shard_NN.json``), None when it names none."""
    match = _SHARD.search(str(episode.get("episode_list") or ""))
    return int(match.group(1)) if match else None


def episode_success(episode: Mapping[str, Any]) -> bool:
    """``outcome.success == 1`` (Habitat: STOP called within 3 m of the goal)."""
    value = (episode.get("outcome") or {}).get("success")
    return value is not None and float(value) == 1.0


def call_seeds(calls: Iterable[Mapping[str, Any]]) -> set:
    """Every ``sampling.protocol_seed`` a record's calls carry (None: a call without one)."""
    out = set()
    for call in calls:
        seed = (call.get("sampling") or {}).get("protocol_seed")
        out.add(None if seed is None else int(seed))
    return out


def seed_share_ok(mismatched: int, records: int) -> bool:
    """A4: at most 1% of a source's records may carry another System1 seed (they are left out)."""
    return records == 0 or Fraction(mismatched, records) <= MAX_SEED_MISMATCH_SHARE


@dataclass
class Trajectory:
    """One record RF1 accepts, with every call of it (``index_calls``' CallRefs, call order)."""

    tag: str
    record_path: str
    scene: str  # bare scene ID
    episode_id: int
    device: str
    refs: list[CallRef]

    @property
    def episode(self) -> tuple[str, int]:
        return self.scene, self.episode_id

    def answers(self) -> tuple[tuple[int, tuple[str, ...]], ...]:
        return tuple((r.call_index, tuple(r.recorded)) for r in self.refs)


def scan_records(
    tag: str,
    root: str,
    seed: int,
    *,
    held: set[str],
    train: set[str],
    anchor: bool,
) -> tuple[list[Trajectory], Counter, dict[str, Any]]:
    """RF1 (or, ``anchor``, RF5's record rules) over one root: the trajectories accepted, counts per reason.

    An RFT source holding a record of a held-out scene or of a scene outside the training list is
    refused; the anchor root skips its held-out records (``collect_base``'s shards 0-19) and refuses
    any other scene outside the list.  Records whose calls carry another protocol seed than ``seed``
    are left out; more than 1% of them stops the build (A4).  The root is resolved first, so record paths
    (CallRef ``uid``, hence the anchors' rank; ``keys.tsv``) start with the manifest's root."""
    what = "anchor root" if anchor else f"source {tag}"
    root = str(Path(root).resolve())
    if not Path(root).is_dir():
        raise SystemExit(f"{what} {root} is not a directory")
    refs, index_stats = index_calls([root], None, split="train" if anchor else "all",
                                    holdout=holdout_match_set(held) if anchor else (), require_labels=False)
    by_record: dict[str, list[CallRef]] = defaultdict(list)
    for ref in refs:
        by_record[ref.record_path].append(ref)
    counts: Counter = Counter()
    devices: Counter = Counter()
    held_out: Counter = Counter()
    outside: Counter = Counter()
    seeds_seen: Counter = Counter()
    accepted: list[Trajectory] = []
    for path in iter_record_paths(root):
        dir_scene = scene_key(path.parent.name)
        if anchor and dir_scene in held:
            counts["held_out_scene_skipped"] += 1
            continue
        record = EpisodeRecord(path)
        scene = scene_key(record.scene_id)
        if dir_scene in held or scene in held:
            if anchor:
                counts["held_out_scene_skipped"] += 1
                continue
            held_out[scene] += 1
            continue
        if scene not in train or dir_scene not in train:
            outside[scene] += 1
            continue
        counts["records"] += 1
        seeds = call_seeds(record.calls)
        for value in seeds:
            seeds_seen[str(value)] += 1
        if not record.calls:
            counts["records_without_calls"] += 1
        if seeds - {seed}:
            counts["seed_mismatch"] += 1
            continue
        if served_adapter(record.episode) is not None:
            counts["served_adapter"] += 1
            continue
        if anchor:
            shard = episode_shard(record.episode)
            if shard is None:
                counts["no_cohort_shard"] += 1
                continue
            if shard < ANCHOR_MIN_SHARD:
                counts[f"shard_below_{ANCHOR_MIN_SHARD}"] += 1
                continue
        if not episode_success(record.episode):
            counts["not_success"] += 1
            continue
        mine = sorted(by_record.get(str(path), []), key=lambda r: r.call_index)
        if len(mine) != len(record.calls):
            raise SystemExit(f"{path}: index_calls gave {len(mine)} calls, the record holds {len(record.calls)}")
        counts["success"] += 1
        device = record_device(record)
        devices[device] += 1
        accepted.append(Trajectory(tag, str(path), scene, record.episode_id, device, mine))
    if held_out:
        raise SystemExit(f"{what} {root} holds records of held-out scenes (records per scene): {dict(held_out)}")
    if outside:
        raise SystemExit(f"{what} {root} holds records of scenes outside the training list (records per scene): "
                         f"{dict(sorted(outside.items())[:20])} ({len(outside)} scenes)")
    if not counts["records"]:
        raise SystemExit(f"{what} {root} holds no committed record of a training scene")
    if not seed_share_ok(counts["seed_mismatch"], counts["records"]):
        raise SystemExit(f"{what} {root}: {counts['seed_mismatch']} of {counts['records']} records carry another "
                         f"System1 seed than {seed} (more than 1%, ledger EXP-32 A4; seeds seen {dict(seeds_seen)}): "
                         f"check the collection before training on it")
    info = {"root": root, "protocol_seed": seed, "index": dict(index_stats),
            "devices": dict(devices), "seeds_seen": dict(seeds_seen)}
    return accepted, counts, info


def keep_per_episode(
    trajectories: Sequence[Trajectory], max_per_episode: int,
) -> tuple[list[Trajectory], dict[str, Counter]]:
    """RF1's count: per episode the trajectories in source order (p1, r0s, p2, p3), one equal to an
    earlier kept one call for call left out, then at most ``max_per_episode``."""
    rank = {tag: i for i, tag in enumerate(RFT_TAGS)}
    by_episode: dict[tuple[str, int], list[Trajectory]] = defaultdict(list)
    for traj in trajectories:
        by_episode[traj.episode].append(traj)
    counts: dict[str, Counter] = defaultdict(Counter)
    kept: list[Trajectory] = []
    for episode in sorted(by_episode):
        mine: list[Trajectory] = []
        for traj in sorted(by_episode[episode], key=lambda t: (rank[t.tag], t.record_path)):
            same = next((k for k in mine if k.answers() == traj.answers()), None)
            if same is not None:
                counts[traj.tag][f"duplicate_of_{same.tag}"] += 1
                continue
            if len(mine) >= max_per_episode:
                counts[traj.tag]["over_episode_cap"] += 1
                continue
            mine.append(traj)
        kept += mine
    return kept, counts


# --------------------------------------------------------------------------- #
# RF5: anchors
# --------------------------------------------------------------------------- #
def anchor_rank(anchor_seed: int, uid: str) -> str:
    return hashlib.sha256(f"exp32-anchor:{anchor_seed}:{uid}".encode("utf-8")).hexdigest()


def draw_anchors(
    pool: Mapping[tuple[str, int], Sequence[CallRef]], n: int, *, per_episode: int, anchor_seed: int,
) -> tuple[list[CallRef], dict[str, Any]]:
    """RF5: per episode the ``per_episode`` lowest-ranked calls, then the ``n`` lowest-ranked of those
    (all of them when fewer), in (record, call) order."""
    capped: list[tuple[str, CallRef]] = []
    episodes_capped = 0
    for episode in sorted(pool):
        ranked = sorted(((anchor_rank(anchor_seed, r.uid), r) for r in pool[episode]), key=lambda x: x[0])
        episodes_capped += len(ranked) > per_episode
        capped += ranked[:per_episode]
    capped.sort(key=lambda x: x[0])
    chosen = sorted((r for _, r in capped[:n]), key=lambda r: (r.record_path, r.call_index))
    info = {"wanted": n, "pool_calls": sum(len(v) for v in pool.values()), "pool_after_episode_cap": len(capped),
            "pool_episodes": sum(bool(v) for v in pool.values()), "episodes_capped": episodes_capped,
            "taken": len(chosen), "anchors_capped_by_pool": len(capped) < n,
            "episodes_taken": len({(r.scene_id, r.episode_id) for r in chosen})}
    return chosen, info


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def sample_key(scene: str, episode_id: int, tag: str, call_index: int) -> str:
    """RF7: ``{scene}_{episode:06d}#{tag}.{call}``."""
    return f"{scene}_{int(episode_id):06d}#{tag}.{int(call_index)}"


def _item(ref: CallRef, tag: str) -> dict:
    return {"source": "dagger", "record_path": ref.record_path, "call_index": ref.call_index,
            "key": sample_key(ref.scene_id, ref.episode_id, tag, ref.call_index), "turns": list(ref.turns)}


def keys_sha256(items: Iterable[Mapping[str, Any]]) -> str:
    """``merge_streams.dagger_keys_sha256`` of the items (key|turns, in stream order)."""
    return hashlib.sha256("\n".join(i["key"] + "|" + "\x1f".join(i["turns"]) for i in items).encode()).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rft-source", action="append", required=True, metavar="TAG=ROOT:SEED",
                        help=f"a record root of sampled trajectories, TAG in {list(RFT_TAGS)} (repeatable)")
    parser.add_argument("--anchor-root", required=True, help="collect_base's records (base, greedy, seed 42)")
    parser.add_argument("--train-scenes-file", required=True, help="the 279 HM3D training scenes")
    parser.add_argument("--holdout-scenes-file", required=True, help="the 20 held-out HM3D scenes")
    parser.add_argument("--first-calls", type=int, default=DEFAULT_FIRST_CALLS)
    parser.add_argument("--min-distance", type=float, default=DEFAULT_MIN_DISTANCE)
    parser.add_argument("--max-anchors-per-episode", type=int, default=6)
    parser.add_argument("--anchor-ratio", type=float, default=1.0, help="anchors per RFT sample")
    parser.add_argument("--anchor-seed", type=int, default=32, help="the anchors' rank: sha256('exp32-anchor:SEED:'+uid)")
    parser.add_argument("--max-trajectories-per-episode", type=int, default=2)
    parser.add_argument("--no-pad", action="store_true", help="required: an input for scripts/exp28/merge_streams.py")
    parser.add_argument("--seed", type=int, default=42, help="the shuffle (seed + 1, as build_route_stream.py)")
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true", help="select and print the composition, write nothing")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.no_pad:
        raise SystemExit("--no-pad is required: the RFT stream is an input for merge_streams.py, which adds the "
                         "replay and pads to whole shards")
    if args.shard_size % args.global_batch or args.global_batch % args.world_size:
        raise SystemExit("shard size must be a multiple of the global batch, and that of the world size")
    if args.max_trajectories_per_episode < 1 or args.max_anchors_per_episode < 0 or args.anchor_ratio < 0:
        raise SystemExit("--max-trajectories-per-episode >= 1, --max-anchors-per-episode and --anchor-ratio >= 0")
    out_dir = Path(args.out_dir)
    if not args.dry_run and (out_dir / MANIFEST_NAME).exists():
        raise SystemExit(f"{out_dir} already holds a stream")
    started = time.time()

    sources = [parse_source(s) for s in args.rft_source]
    tags = [t for t, _, _ in sources]
    if len(set(tags)) != len(tags):
        raise SystemExit(f"--rft-source names a tag twice: {tags}")
    holdout_file = read_scene_list(args.holdout_scenes_file)
    exp23 = holdout_scenes()
    held = {scene_key(s) for s in (*holdout_file, *exp23)}
    train_list = read_scene_list(args.train_scenes_file)
    train = {scene_key(s) for s in train_list}
    clash = sorted(train & held)
    if clash:
        raise SystemExit(f"{args.train_scenes_file} lists held-out scenes: {clash}")

    # --- RF1: trajectories
    report: dict[str, Any] = {}
    trajectories: list[Trajectory] = []
    for tag, root, seed in sorted(sources, key=lambda s: RFT_TAGS.index(s[0])):
        accepted, counts, info = scan_records(tag, root, seed, held=held, train=train, anchor=False)
        trajectories += accepted
        report[tag] = {**info, "records": dict(counts)}
    kept, episode_counts = keep_per_episode(trajectories, args.max_trajectories_per_episode)

    # --- RF2-RF4: calls of the kept trajectories
    rft: list[tuple[CallRef, str]] = []
    rft_records: list[Trajectory] = []
    for tag in report:
        report[tag]["records"].update(episode_counts.get(tag, {}))
        report[tag]["calls"] = Counter()
        report[tag]["trajectories"] = 0
        report[tag]["trajectory_devices"] = Counter()
    for traj in kept:
        refs, counts = filter_refs(traj.refs, first_calls=args.first_calls, min_distance=args.min_distance)
        report[traj.tag]["calls"].update(counts)
        report[traj.tag]["trajectories"] += 1
        report[traj.tag]["trajectory_devices"][traj.device] += 1
        rft_records.append(traj)
        rft += [(ref, traj.tag) for ref in refs]
    for tag in report:
        report[tag]["calls"] = dict(report[tag]["calls"])
        report[tag]["trajectory_devices"] = dict(report[tag]["trajectory_devices"])
        report[tag]["episodes"] = len({t.episode for t in kept if t.tag == tag})
    n_rft = len(rft)
    if not n_rft:
        raise SystemExit(f"no RFT samples (sources {json.dumps(report, sort_keys=True, default=dict)})")

    # --- RF5: anchors
    anchor_trajs, anchor_counts, anchor_info = scan_records(
        ANCHOR_TAG, args.anchor_root, ANCHOR_PROTOCOL_SEED, held=held, train=train, anchor=True)
    pool: dict[tuple[str, int], list[CallRef]] = {}
    anchor_calls: Counter = Counter()
    for traj in anchor_trajs:
        refs, counts = filter_refs(traj.refs, first_calls=args.first_calls, min_distance=args.min_distance)
        anchor_calls.update(counts)
        if traj.episode in pool:
            raise SystemExit(f"anchor episode {traj.episode} is recorded twice under {args.anchor_root}")
        pool[traj.episode] = refs
    n_anchor = int(round(args.anchor_ratio * n_rft))
    anchors, draw_info = draw_anchors(pool, n_anchor, per_episode=args.max_anchors_per_episode,
                                      anchor_seed=args.anchor_seed)
    if draw_info["anchors_capped_by_pool"]:
        print(f"[rft] warning: {n_anchor} anchors wanted, the capped pool holds {draw_info['pool_after_episode_cap']}",
              file=sys.stderr, flush=True)
    anchor_report = {**anchor_info, "records": dict(anchor_counts), "calls": dict(anchor_calls), "draw": draw_info}

    # --- items, one seeded shuffle
    chosen: list[tuple[dict, str, str]] = [(_item(ref, tag), f"rft_{tag}", ref.record_device) for ref, tag in rft]
    chosen += [(_item(ref, ANCHOR_TAG), "anchor", ref.record_device) for ref in anchors]
    keys = [item["key"] for item, _, _ in chosen]
    if len(set(keys)) != len(keys):
        raise SystemExit("two samples share a key")
    order = np.random.default_rng(args.seed + 1).permutation(len(chosen))
    items = [chosen[int(i)] for i in order]

    drawn: Counter = Counter({f"category_rft_{tag}": 0 for tag in RFT_TAGS})
    for item, category, _ in items:
        drawn[item["source"]] += 1
        drawn[f"{item['source']}_target_{decision_class(item['turns'][-1])}"] += 1
        drawn[f"category_{category}"] += 1
    total = len(items)
    reversals = {"dropped_rft": sum(int(r["calls"].get("reversal", 0)) for r in report.values()),
                 "dropped_anchors": int(anchor_calls.get("reversal", 0)), "reversed_targets_kept": 0}
    composition = {
        "total_samples": total,
        "num_shards": int(math.ceil(total / args.shard_size)),
        "steps_at_global_batch": total // args.global_batch,
        "n_rft": n_rft,
        "n_anchor": len(anchors),
        "by_category": {k[len("category_"):]: v for k, v in sorted(drawn.items()) if k.startswith("category_")},
        "shares": {"rft": round(n_rft / total, 4), "anchors": round(len(anchors) / total, 4),
                   "stop_targets": round(drawn["dagger_target_stop"] / total, 4)},
        "rft_trajectories": len(rft_records),
        "rft_episodes": len({t.episode for t in rft_records}),
        "rft_by_device": dict(Counter(device for _, c, device in chosen if c != "anchor")),
        "anchors_by_device": dict(Counter(device for _, c, device in chosen if c == "anchor")),
        "anchors_capped_by_pool": draw_info["anchors_capped_by_pool"],
        "reversals": reversals,
        "padded": False,
    }
    print(json.dumps({"composition": composition, "drawn": dict(sorted(drawn.items()))}, indent=1, sort_keys=True),
          flush=True)
    if args.dry_run:
        print(json.dumps({"sources": report, "anchors": anchor_report}, indent=1, sort_keys=True, default=dict))
        return 0

    # --- write
    out_dir.mkdir(parents=True, exist_ok=True)
    num_shards = composition["num_shards"]  # unpadded: the last shard is short
    jobs = [(s, [item for item, _, _ in items[s * args.shard_size:(s + 1) * args.shard_size]],
             str(out_dir), tuple(DEFAULT_VLM_IMAGE_SIZE), 90) for s in range(num_shards)]
    shards: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool_exec:
        for done, (index, digest, size) in enumerate(pool_exec.map(_write, jobs, chunksize=1), 1):
            shards[index] = {"name": shard_name(index), "sha256": digest, "bytes": size}
            if done % 10 == 0 or done == num_shards:
                print(f"[rft] {done}/{num_shards} shards", flush=True)
    records_text = "".join(f"{t.record_path}\t{t.device}\t{t.tag}\n" for t in sorted(rft_records, key=lambda t: t.record_path))
    keys_text = "".join(f"{item['key']}\t{item['record_path']}#{item['call_index']}\t{category}\t{device}\n"
                        for item, category, device in items)
    (out_dir / RECORDS_LIST).write_text(records_text, encoding="utf-8")
    (out_dir / KEYS_TSV).write_text(keys_text, encoding="utf-8")

    roots = [report[tag]["root"] for tag in RFT_TAGS if tag in report]
    manifest = {
        "schema": UNPADDED_SCHEMA,
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
        "replay_fraction": 0.0,
        "holdout": {"scenes": list(holdout_file), "count": len(holdout_file),
                    "file": str(Path(args.holdout_scenes_file).resolve()),
                    "file_sha256": hashlib.sha256(Path(args.holdout_scenes_file).read_bytes()).hexdigest(),
                    "exp23_scenes": list(exp23)},
        "record_roots": roots + [anchor_info["root"]],
        "label_roots": [],
        "replay_root": "",
        "selection": {
            "rule": "EXP-32 RF1-RF7: the model's own sampled answers (verbatim) on its successful HM3D training "
                    "trajectories beyond 3 m in the first 48 calls (pixel or turn, no reversal), at most 2 "
                    "trajectories per episode, plus as many base-greedy anchors of collect_base's successful "
                    "training episodes (at most 6 per episode, sha256 rank); every sample once, no STOP target",
            "knobs": {
                "rft_sources": {tag: {"root": report[tag]["root"], "protocol_seed": report[tag]["protocol_seed"]}
                                for tag in report},
                "anchor_root": anchor_info["root"],
                "anchor_protocol_seed": ANCHOR_PROTOCOL_SEED,
                "anchor_min_shard": ANCHOR_MIN_SHARD,
                "first_calls": args.first_calls,
                "min_distance": args.min_distance,
                "rft_classes": list(RFT_CLASSES),
                "drop_reversals": True,
                "max_anchors_per_episode": args.max_anchors_per_episode,
                "anchor_ratio": args.anchor_ratio,
                "anchor_seed": args.anchor_seed,
                "anchor_rank": f"sha256('exp32-anchor:{args.anchor_seed}:' + record_path#call_index)",
                "max_trajectories_per_episode": args.max_trajectories_per_episode,
                "weight": 1.0,
                "no_pad": True,
                "holdout_scenes_file": args.holdout_scenes_file,
                "train_scenes_file": args.train_scenes_file,
            },
            "train_scenes": {"scenes": sorted(train), "count": len(train_list),
                             "file": str(Path(args.train_scenes_file).resolve()),
                             "file_sha256": hashlib.sha256(Path(args.train_scenes_file).read_bytes()).hexdigest()},
            "sources": report,
            "anchors": anchor_report,
            "composition": composition,
            "dagger_selected": total,
            "selected_keys_sha256": keys_sha256(item for item, _, _ in items),
            "rft_records_sha256": hashlib.sha256(records_text.encode("utf-8")).hexdigest(),
            "keys_tsv_sha256": hashlib.sha256(keys_text.encode("utf-8")).hexdigest(),
            "stop_share": composition["shares"]["stop_targets"],
        },
        "drawn": dict(drawn),
        "shards": [shards[s] for s in range(num_shards)],
        "build_s": round(time.time() - started, 1),
    }
    tmp = out_dir / f".{MANIFEST_NAME}.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True, default=dict) + "\n", encoding="utf-8")
    os.replace(tmp, out_dir / MANIFEST_NAME)
    print(json.dumps({k: manifest[k] for k in ("num_shards", "total_samples", "build_s")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

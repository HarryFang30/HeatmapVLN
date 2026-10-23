#!/usr/bin/env python3
"""Pack EXP-34's training stream from line-B labels on the base model's own records: COR, AGR, TERM, REP with ``kl``.

The ledger's EXP-34 D1, run on line-B labels (``scripts/tools/label_route_corrections.py``,
the labeller EXP-25/28 used) of the existing base records (P_old); no takeover data.  Every
sample is taken once, weight 1:

====  =====  ==========================================================================
part  kl     what is taken
====  =====  ==========================================================================
COR   false  every line-B ``correction`` row (target: the oracle's pixel goal or turn;
             line B never targets STOP)
AGR   true   line-B ``anchor`` rows (target: the base's own answer, verbatim): per
             episode the 6 lowest of sha256("exp34-agr:" + key), then the lowest
             len(COR) of those (1:1)
TERM  true   the base's executed calls within 3 m of the goal, which line B leaves
             unlabelled (``within_line_a_radius``; target: the recorded answer, STOP
             included): the lowest min(candidates, ceil(COR / 2)) of
             sha256("exp34-term:" + key) over all candidates, no per-episode cap
REP   true   InternData r2r expert samples (``build_route_stream.replay_items``: the
             ``interndata_replay`` samples, held-out scenes left out), every STOP 5
             times, whole units in a seeded order (``merge_streams.draw_replay``) until
             they make up 20% of the stream, rounded up to whole shards (EXP-32 RF8)
====  =====  ==========================================================================

On COR and AGR only ``call_index < 48`` (ledger B6) and at most 24 COR per episode (the
earliest).  TERM takes every executed call within 3 m whatever its index: C7 compares its STOP
share per distance bin with the base's over *all* its executed calls there, and with the 48-call
cut TERM's <=1 m share came out 1.33x that (r2r + EnvDrop P_old: the calls past 48 are an
episode circling near the goal, rarely a STOP).  On all three (C3, C7): a sample whose target
turns net against the turn executed on the episode's previous call is dropped (R6/RF3:
``build_route_stream.executed_direction``); so is a turn string with a STOP in it; a TERM answer
the server only reads as STOP by fallback, or cannot read (``invalid``), is dropped.  ``d`` is line B's geodesic.  Line-B ``none`` rows
beyond 3 m do not enter; every drop is counted.  A key is ``{scene}_{episode:06d}#{source}.{call}``
(``merge_streams.sample_scene`` reads it), unique across sources.

Held out, always: EXP-23's six R2R scenes and EXP-28's 20 HM3D scenes (constants below; any
``--holdout-scenes-file`` adds to them).  A record of a scene that is neither held out nor a
training scene (the 61 R2R train scenes, ``--train-scenes-file``'s HM3D ones) is refused, as
are records an adapter served and label files of another record version or labeller.

Encoding: items are ``build_s2_train_stream._dagger_item`` / ``_replay_item`` and every sample
is packed by ``build_s2_train_stream._pack``, the path every earlier stream took (EXP-28's
passed G0 at 0.9955); the shard is ``s2_stream.write_shard``'s, with ``kl`` and ``part`` added
to each sample's ``{i:06d}.json`` (``read_shard`` ignores them).  The manifest is the trainer's
(``STREAM_SCHEMA``); it carries the parts' counts, predicted and measured STOP shares, the
supervised-token share of ``kl`` samples, sources and seed.  ``keys.tsv`` lists every sample
in stream order.  D2's only STOP-band adjustment is applied when the share leaves [3%, 10%]:
TERM's STOP (or non-STOP) samples dropped in reverse sha256 order.

On the 4090 (CPU only; the label roots from label_route_corrections.py; replay packing needs
``vla_rpc`` on the path)::

    PYTHONPATH=$WT:/workspace/rpc/src /opt/conda/bin/python scripts/exp34/build_exp34_stream.py \\
        --source r2r /workspace/exp23/round1/records /workspace/exp34/labels/r2r \\
        --source envdrop /workspace/exp23/envdrop1/records /workspace/exp34/labels/envdrop \\
        --source hm3d /workspace/exp28/collect_base/records /workspace/exp34/labels/hm3d \\
        --train-scenes-file /workspace/exp28/episodes/splits/train_scenes.txt \\
        --replay-root /workspace/exp23/data/interndata_n1/r2r --workers 24 --out-dir /workspace/exp34/stream

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
import re
import sys
import tarfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.exp28 import merge_streams as ms  # noqa: E402
from scripts.exp28.hm3d_route_gate import served_adapter  # noqa: E402
from scripts.exp32.build_rft_stream import previous_direction  # noqa: E402
from scripts.tools import build_route_stream as brs  # noqa: E402
from scripts.tools import build_s2_train_stream as bst  # noqa: E402
from src.data.s2_records import EpisodeRecord, label_path_for, read_labels  # noqa: E402
from src.data.s2_route_labels import LABEL_LINE, RouteLabelConfig  # noqa: E402
from src.data.s2_sft_dataset import (  # noqa: E402
    DEFAULT_VLM_IMAGE_SIZE,
    R2R_TRAIN_SCENES,
    STOP_TEXT,
    CallRef,
    decision_class,
    holdout_scenes,
    index_calls,
    pixel_uv,
    training_turns,
)
from src.data.s2_stream import MANIFEST_NAME, STREAM_SCHEMA, PackedSample, shard_name  # noqa: E402
from src.models.heatmap.native_internnav_exact import NATIVE_ACTION_STOP, parse_native_actions  # noqa: E402

BUILDER = "scripts/exp34/build_exp34_stream.py (EXP-34)"
# EXP-23's six held-out R2R scenes (= holdout_scenes(), checked at start) and EXP-28's 20 held-out
# HM3D scene dirs (scripts/exp28/build_hm3d_episodes.py's splits/heldout_scenes.txt).
R2R_HOLDOUT = ("17DRP5sb8fy", "5LpN3gDmAk7", "VzqfbhrpDEA", "gZ6f7yhEvPG", "kEZ7cmS4wCh", "r47D5H71a5s")
HM3D_HOLDOUT = (
    "00026-tzaZQQmUVXZ", "00083-16tymPtM7uS", "00360-yA3RqPqMrGE", "00380-fdfRpE6Cfsr", "00393-dD37vuCa4FE",
    "00396-wuKkTq5GJbi", "00401-H8rQCnvBgo6", "00437-S3BfyR31Wc9", "00438-XvJjCZv6SYp", "00481-5jLhtVmWd5F",
    "00560-gjhYih4upQ9", "00566-qmvPLqLAgvC", "00573-1zDbEdygBeW", "00684-D5dEbkUphhr", "00716-VaEwVD182FS",
    "00743-31DHHWieDMS", "00753-hTTnuAeSN6d", "00756-1K7P6ZQS4VM", "00770-NBg5UqG3di3", "00784-pYGGNqSbHp1",
)
PARTS = ("COR", "AGR", "TERM", "REP")
KL = {"COR": False, "AGR": True, "TERM": True, "REP": True}
AGR_RANK = "exp34-agr:"
TERM_RANK = "exp34-term:"
TERM_REASON = "within_line_a_radius"  # line B's own skip of the states within 3 m
FIRST_CALLS = 48
MAX_COR_PER_EPISODE = 24
MAX_AGR_PER_EPISODE = 6
REPLAY_FRACTION = 0.2
STOP_UPSAMPLE = 5
STOP_BAND = (0.03, 0.10)
PIXEL_WINDOW = (100, 540, 32, 455)  # u_min, u_max, v_min, v_max
STOP_BINS = (("<=1m", 0.0, 1.0), ("1-3m", 1.0, 3.0), (">3m", 3.0, math.inf))
KEYS_TSV = "keys.tsv"
_SOURCE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")


# --------------------------------------------------------------------------- #
# Small readings
# --------------------------------------------------------------------------- #
def rank(prefix: str, key: str) -> str:
    return hashlib.sha256(f"{prefix}{key}".encode("utf-8")).hexdigest()


def sample_key(ref: CallRef, source: str) -> str:
    return f"{ref.scene_id}_{int(ref.episode_id):06d}#{source}.{int(ref.call_index)}"


def is_stop(turns: Sequence[str]) -> bool:
    return decision_class(turns[-1]) == "stop"


def turn_with_stop(turns: Sequence[str]) -> bool:
    """An arrow string the server executes as a turn or a step that also holds a STOP."""
    final = turns[-1]
    return decision_class(final) in ("turn_left", "turn_right", "forward") and NATIVE_ACTION_STOP in parse_native_actions(final)


def outside_window(turns: Sequence[str]) -> bool:
    uv = pixel_uv(turns[-1]) if decision_class(turns[-1]) == "pixel" else None
    if uv is None:
        return False
    u_min, u_max, v_min, v_max = PIXEL_WINDOW
    return not (u_min <= uv[0] <= u_max and v_min <= uv[1] <= v_max)


def supervised_tokens(turns: Sequence[str]) -> int:
    """Supervised tokens of a target: each turn's content and its ``<|im_end|>``.  On the
    InternNav model's tokenizer an arrow, a digit, a space and ``STOP`` are one token each."""
    return sum(len(str(t).replace(STOP_TEXT, "S")) + 1 for t in turns)


def stop_bin(d: float) -> str:
    for name, lo, hi in STOP_BINS:
        if (lo < d <= hi) or (lo == 0.0 and d <= hi):
            return name
    raise ValueError(d)


# --------------------------------------------------------------------------- #
# Scenes
# --------------------------------------------------------------------------- #
def held_out_scenes(extra_file: str = "") -> set[str]:
    """Bare IDs of every scene kept out: the R2R six, the HM3D 20, and ``extra_file``'s."""
    if tuple(holdout_scenes()) != R2R_HOLDOUT:
        raise SystemExit(f"holdout_scenes() is {holdout_scenes()}, EXP-34 holds out {R2R_HOLDOUT}")
    names = [*R2R_HOLDOUT, *HM3D_HOLDOUT, *(brs.read_scene_list(extra_file) if extra_file else ())]
    return {brs.scene_key(s) for s in names}


def training_scenes(train_file: str, held: set[str]) -> set[str]:
    """Bare IDs a record may come from: the R2R train scenes and ``train_file``'s (HM3D), none held out."""
    listed = {brs.scene_key(s) for s in brs.read_scene_list(train_file)} if train_file else set()
    clash = sorted(listed & held)
    if clash:
        raise SystemExit(f"{train_file} lists held-out scenes: {clash}")
    return (set(R2R_TRAIN_SCENES) | listed) - held


def _map(fn: Callable, values: Sequence, workers: int, chunksize: int = 1) -> list:
    if workers <= 1:
        return [fn(v) for v in values]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, values, chunksize=chunksize))


# --------------------------------------------------------------------------- #
# Candidates of one record
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class Candidate:
    part: str  # COR | AGR | TERM
    source: str
    ref: CallRef  # ``turns`` is the target
    key: str
    reason: str  # line B's row reason (for COR: why the base's answer disagreed)
    previous: int  # net turn executed on the episode's previous call

    @property
    def episode(self) -> tuple[str, str]:
        return self.source, self.ref.record_path

    @property
    def d(self) -> float:
        return float(self.ref.distance_to_goal)


def classify_record(
    refs: Sequence[CallRef],
    rows: Mapping[int, Mapping[str, Any]],
    *,
    source: str,
    first_calls: int = FIRST_CALLS,
    directions_from_record: Callable[[str], Mapping[int, int]] = brs.record_directions,
) -> tuple[list[Candidate], Counter, list[tuple[float, bool]]]:
    """(candidates, counts, (d, base answered STOP) of every call with a distance) of one record.

    ``refs`` are ``index_calls``' CallRefs of the record (labels joined), ``rows`` its raw line-B
    rows by call index.  The previous call's executed turn is read from ``refs`` when they cover
    every call, else from the record (``directions_from_record``)."""
    refs = sorted(refs, key=lambda r: r.call_index)
    directions = {r.call_index: brs.executed_direction(r.recorded) for r in refs}
    if refs and sorted(directions) != list(range(len(directions))):
        directions = {**directions_from_record(refs[0].record_path), **directions}
    candidates: list[Candidate] = []
    counts: Counter = Counter()
    reference: list[tuple[float, bool]] = []
    for ref in refs:
        row = rows.get(ref.call_index)
        if row is None or ref.label is None:
            counts["unlabelled"] += 1
            continue
        d = ref.distance_to_goal
        known = d is not None and math.isfinite(float(d))
        if known:
            reference.append((float(d), is_stop(ref.recorded)))
        kind, reason = row.get("kind"), str(row.get("reason") or "")
        if row.get("line") != LABEL_LINE:
            raise SystemExit(f"{ref.record_path}#{ref.call_index}: not a line-{LABEL_LINE} row ({row.get('line')!r})")
        term = kind == "none" and reason == TERM_REASON
        if ref.call_index >= first_calls:
            if not term:
                counts["beyond_first_calls"] += 1
                continue
            counts["TERM_beyond_first_calls_kept"] += 1
        if not known:
            counts["no_distance"] += 1
            continue
        if kind == "correction":
            if ref.label.skip or ref.label.agree is not False:
                raise SystemExit(f"{ref.record_path}#{ref.call_index}: a correction row the dataset reads as "
                                 f"agree={ref.label.agree}, skip={ref.label.skip}")
            part, turns = "COR", training_turns(ref.recorded, ref.label)
            if is_stop(turns):
                raise SystemExit(f"{ref.record_path}#{ref.call_index}: line B never targets STOP, got {turns!r}")
        elif kind == "anchor":
            if ref.label.agree is not True or ref.label.target_turns not in (None, ref.recorded):
                raise SystemExit(f"{ref.record_path}#{ref.call_index}: anchor target {ref.label.target_turns!r} is "
                                 f"not the recorded answer {ref.recorded!r}")
            part, turns = "AGR", tuple(ref.recorded)
        elif term:
            part, turns = "TERM", tuple(ref.recorded)
            decision = decision_class(turns[-1])
            if decision == "invalid" or (decision == "stop" and turns[-1].strip() != STOP_TEXT):
                counts["TERM_dropped_unreadable"] += 1
                continue
        elif kind == "none":
            counts["none_beyond_radius"] += 1
            counts[f"none_beyond_radius:{reason.split('|')[0].split(':')[0]}"] += 1
            continue
        else:
            raise SystemExit(f"{ref.record_path}#{ref.call_index}: line-B row kind {kind!r}")
        counts[f"{part}_labelled"] += 1
        if turn_with_stop(turns):
            counts[f"{part}_dropped_turn_with_stop"] += 1
            continue
        previous = previous_direction(directions, ref.call_index)
        if previous and brs.executed_direction(turns) == -previous:
            counts[f"{part}_dropped_reversal"] += 1
            continue
        candidates.append(Candidate(part, source, dataclasses.replace(ref, turns=tuple(turns)),
                                    sample_key(ref, source), reason.split("|")[0].split(":")[0], previous))
    return candidates, counts, reference


# --------------------------------------------------------------------------- #
# D1: the draw
# --------------------------------------------------------------------------- #
def draw_dagger(
    candidates: Sequence[Candidate],
    *,
    max_cor_per_episode: int = MAX_COR_PER_EPISODE,
    max_agr_per_episode: int = MAX_AGR_PER_EPISODE,
) -> tuple[dict[str, list[Candidate]], dict[str, Any]]:
    """COR (all, at most ``max_cor_per_episode`` per episode, the earliest), AGR (1:1) and TERM
    (ceil(COR / 2)); AGR and TERM in their sha256 rank order."""
    keys = [c.key for c in candidates]
    if len(set(keys)) != len(keys):
        dup = [k for k, n in Counter(keys).items() if n > 1]
        raise SystemExit(f"{len(dup)} keys repeat (the same record under two sources?), e.g. {dup[:3]}")
    by_episode: dict[tuple[str, str], dict[str, list[Candidate]]] = defaultdict(lambda: defaultdict(list))
    for c in candidates:
        by_episode[c.episode][c.part].append(c)
    cor: list[Candidate] = []
    agr_pool: list[Candidate] = []
    term_pool: list[Candidate] = []
    info: Counter = Counter()
    for episode in sorted(by_episode):
        parts = by_episode[episode]
        mine = sorted(parts["COR"], key=lambda c: c.ref.call_index)
        if len(mine) > max_cor_per_episode:
            info["episodes_cor_capped"] += 1
            info["cor_dropped_by_episode_cap"] += len(mine) - max_cor_per_episode
        cor += mine[:max_cor_per_episode]
        ranked = sorted(parts["AGR"], key=lambda c: (rank(AGR_RANK, c.key), c.key))
        if len(ranked) > max_agr_per_episode:
            info["episodes_agr_capped"] += 1
        agr_pool += ranked[:max_agr_per_episode]
        term_pool += parts["TERM"]
    agr = sorted(agr_pool, key=lambda c: (rank(AGR_RANK, c.key), c.key))[: len(cor)]
    n_term = min(len(term_pool), math.ceil(len(cor) / 2))
    term = sorted(term_pool, key=lambda c: (rank(TERM_RANK, c.key), c.key))[:n_term]
    out = {**info, "cor": len(cor), "agr_pool_after_episode_cap": len(agr_pool), "agr": len(agr),
           "term_pool": len(term_pool), "term": len(term), "agr_capped_by_pool": len(agr) < len(cor),
           "term_pool_stop": sum(is_stop(c.ref.turns) for c in term_pool)}
    return {"COR": cor, "AGR": agr, "TERM": term}, out


class _Rep:
    """A replay item as ``merge_streams.draw_replay`` reads its units' entries (``.turns``)."""

    __slots__ = ("item", "turns")

    def __init__(self, item: Mapping[str, Any]):
        self.item, self.turns = item, tuple(item["turns"])


def load_replay_pool(replay_root: str, held: set[str]) -> tuple[list[dict], Counter]:
    """InternData r2r samples of the scenes not held out (``build_route_stream.replay_items``)."""
    return brs.replay_items(replay_root, sorted(held))


def replay_units(pool: Sequence[Mapping[str, Any]], stop_upsample: int = STOP_UPSAMPLE) -> list[list[_Rep]]:
    """One unit per sample; a STOP's unit holds its ``stop_upsample`` copies."""
    units = []
    for item in pool:
        rep = _Rep(item)
        units.append([rep] * (stop_upsample if is_stop(rep.turns) else 1))
    return units


def compose(
    dagger: Mapping[str, Sequence[Candidate]],
    units: Sequence[Sequence[_Rep]],
    *,
    replay_fraction: float = REPLAY_FRACTION,
    shard_size: int = 256,
    seed: int = 42,
    band: tuple[float, float] | None = STOP_BAND,
) -> tuple[dict[str, list], dict[str, Any]]:
    """The four parts: REP sized and drawn for the DAgger parts (``merge_streams.stream_size`` /
    ``draw_replay``), then, while the whole stream's STOP share is outside ``band``, TERM's STOP
    (share too high) or non-STOP (too low) samples dropped from the end of its rank order and
    REP redrawn (ledger D2 item 3, the only adjustment allowed)."""
    term = list(dagger["TERM"])
    dropped: list[Candidate] = []
    for _ in range(1000):
        parts = {"COR": list(dagger["COR"]), "AGR": list(dagger["AGR"]), "TERM": term}
        n_dagger = sum(len(v) for v in parts.values())
        total, n_rep = ms.stream_size(n_dagger, replay_fraction, shard_size)
        rep, _ = ms.draw_replay({"r2r": units}, n_rep, {"r2r": 1.0}, np.random.default_rng(seed))
        s_dagger = sum(is_stop(c.ref.turns) for v in parts.values() for c in v)
        s_rep = sum(is_stop(r.turns) for r in rep)
        share = (s_dagger + s_rep) / total
        if band is None or band[0] <= share <= band[1]:
            break
        too_high = share > band[1]
        victims = [i for i, c in enumerate(term) if is_stop(c.ref.turns) == too_high]
        if not victims:
            raise SystemExit(f"STOP share {share:.4f} is outside {band} and TERM has no "
                             f"{'STOP' if too_high else 'non-STOP'} sample left to drop")
        rho, r = s_rep / max(n_rep, 1), n_rep / max(n_dagger, 1)
        if too_high:
            need = (s_dagger + rho * r * n_dagger - band[1] * (1 + r) * n_dagger) / (1 + rho * r - band[1] * (1 + r))
        else:
            denominator = band[0] * (1 + r) - rho * r
            need = n_dagger - s_dagger / denominator if denominator > 0 else 1
        k = max(1, min(len(victims), int(math.ceil(need))))
        gone = set(victims[-k:])
        dropped += [term[i] for i in sorted(gone)]
        term = [c for i, c in enumerate(term) if i not in gone]
    else:
        raise SystemExit(f"the STOP share did not come into {band}")
    parts["REP"] = [r.item for r in rep]
    return parts, {"stop_share": round(share, 6), "band": list(band) if band else None,
                   "term_dropped_for_band": len(dropped),
                   "term_dropped_stop": sum(is_stop(c.ref.turns) for c in dropped),
                   "term_dropped_keys_sha256": hashlib.sha256("\n".join(c.key for c in dropped).encode()).hexdigest()
                   if dropped else None,
                   "total": total, "rep": n_rep}


# --------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------- #
def make_items(parts: Mapping[str, Sequence], seed: int) -> list[dict]:
    """Items as ``build_s2_train_stream`` makes them, plus ``part`` / ``kl`` / ``d``, in one seeded shuffle."""
    chosen: list[dict] = []
    for part in ("COR", "AGR", "TERM"):
        for c in parts[part]:
            item = bst._dagger_item(c.ref)
            item.update(key=c.key, part=part, kl=KL[part], d=c.d, device=c.ref.record_device, origin_source=c.source,
                        reason=c.reason, previous=c.previous)
            chosen.append(item)
    for rep in parts["REP"]:
        chosen.append({**rep, "part": "REP", "kl": KL["REP"]})
    order = np.random.default_rng(seed + 1).permutation(len(chosen))
    return [chosen[int(i)] for i in order]


def write_kl_shard(path: str | os.PathLike, samples: Sequence[PackedSample], items: Sequence[Mapping[str, Any]]) -> str:
    """``s2_stream.write_shard`` with each sample's ``kl`` and ``part`` added to its json; returns the sha256."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as raw:
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for i, (sample, item) in enumerate(zip(samples, items, strict=True)):
                meta = {
                    "key": sample.key,
                    "source": sample.source,
                    "instruction": sample.instruction,
                    "turns": list(sample.turns),
                    "vlm_image_size": list(sample.vlm_image_size),
                    "num_history": len(sample.history_jpegs),
                    "kl": bool(item["kl"]),
                    "part": str(item["part"]),
                }
                members = [(f"{i:06d}.json", json.dumps(meta, ensure_ascii=False).encode("utf-8"))]
                members += [(f"{i:06d}.h{k:02d}.jpg", data) for k, data in enumerate(sample.history_jpegs)]
                members += [(f"{i:06d}.cur.jpg", sample.current_jpeg), (f"{i:06d}.look.jpg", sample.lookdown_jpeg)]
                for name, data in members:
                    info = tarfile.TarInfo(name)
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
    index, items, out_dir, vlm_image_size, jpeg_quality = job
    path = Path(out_dir) / shard_name(index)
    samples = [bst._pack(item, vlm_image_size, jpeg_quality) for item in items]
    for sample, item in zip(samples, items, strict=True):
        if sample.key != item["key"] or tuple(sample.turns) != tuple(item["turns"]):
            raise RuntimeError(f"packed {sample.key} {sample.turns!r} for item {item['key']} {item['turns']!r}")
    return index, write_kl_shard(path, samples, items), path.stat().st_size


def _record_adapter(path: str) -> str | None:
    return served_adapter(EpisodeRecord(path).episode)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _rate(k: float, n: float) -> float | None:
    return round(k / n, 4) if n else None


def stream_stats(items: Sequence[Mapping[str, Any]], reference: Sequence[tuple[float, bool]]) -> dict[str, Any]:
    """D1/C7 numbers of the final stream: parts, STOP shares by distance against the base's own
    per-call shares (``reference``: every call of the indexed episodes), reversals, pixels
    outside the window, ``kl`` sample and token shares."""
    by_part = Counter(i["part"] for i in items)
    stops = Counter(i["part"] for i in items if is_stop(i["turns"]))
    total = len(items)
    bins: dict[str, Counter] = {name: Counter() for name, _, _ in STOP_BINS}
    for item in items:
        if item["part"] != "REP":
            b = bins[stop_bin(float(item["d"]))]
            b["n"] += 1
            b["stop"] += is_stop(item["turns"])
    ref_bins: dict[str, Counter] = {name: Counter() for name, _, _ in STOP_BINS}
    for d, stop in reference:
        b = ref_bins[stop_bin(d)]
        b["n"] += 1
        b["stop"] += stop
    by_distance = {}
    for name, _, _ in STOP_BINS:
        share, ref = _rate(bins[name]["stop"], bins[name]["n"]), _rate(ref_bins[name]["stop"], ref_bins[name]["n"])
        by_distance[name] = {"samples": bins[name]["n"], "stop": bins[name]["stop"], "share": share,
                             "reference_calls": ref_bins[name]["n"], "reference_stop": ref_bins[name]["stop"],
                             "reference_share": ref, "ratio": round(share / ref, 4) if share is not None and ref else None}
    reversals = sum(1 for i in items if i["part"] != "REP" and i["previous"]
                    and brs.executed_direction(i["turns"]) == -int(i["previous"]))
    tokens = Counter()
    for item in items:
        n = supervised_tokens(item["turns"])
        tokens["all"] += n
        tokens["kl"] += n if item["kl"] else 0
        tokens[item["part"]] += n
    cor = [i for i in items if i["part"] == "COR"]
    return {
        "total": total,
        "by_part": {p: by_part[p] for p in PARTS},
        "stop_by_part": {p: stops[p] for p in PARTS},
        "stop_share": _rate(sum(stops.values()), total),
        "stop_share_by_part": {p: _rate(stops[p], by_part[p]) for p in PARTS},
        "stop_by_distance": by_distance,
        "reversals": reversals,
        "turn_with_stop": sum(turn_with_stop(i["turns"]) for i in items),
        "pixel_outside_window": {p: sum(1 for i in items if i["part"] == p and outside_window(i["turns"])) for p in PARTS},
        "pixel_samples": {p: sum(1 for i in items if i["part"] == p and decision_class(i["turns"][-1]) == "pixel")
                          for p in PARTS},
        "cor_by_source_reason": dict(sorted(Counter(f"{i['origin_source']}/{i['reason']}" for i in cor).items())),
        "cor_by_source_target": dict(sorted(Counter(f"{i['origin_source']}/{decision_class(i['turns'][-1]).split('_')[0]}"
                                                    for i in cor).items())),
        "by_part_device": dict(sorted(Counter(f"{i['part']}/{i['device']}" for i in items if i["part"] != "REP").items())),
        "kl_sample_share": _rate(sum(1 for i in items if i["kl"]), total),
        "kl_token_share": _rate(tokens["kl"], tokens["all"]),
        "supervised_tokens": dict(tokens),
    }


def predicted_stop(dagger_info: Mapping[str, Any], parts: Mapping[str, Sequence], pool: Sequence[Mapping[str, Any]],
                   total: int) -> dict[str, Any]:
    """D1's predicted STOP shares: TERM at its candidates' share, REP at the pool's under STOP x5,
    COR and AGR as selected; and the stream's, from those."""
    term_pred = _rate(dagger_info["term_pool_stop"], dagger_info["term_pool"]) or 0.0
    s = sum(is_stop(i["turns"]) for i in pool)
    rep_pred = _rate(STOP_UPSAMPLE * s, STOP_UPSAMPLE * s + len(pool) - s) or 0.0
    cor = sum(is_stop(c.ref.turns) for c in parts["COR"])
    agr = sum(is_stop(c.ref.turns) for c in parts["AGR"])
    expected = cor + agr + term_pred * len(parts["TERM"]) + rep_pred * len(parts["REP"])
    return {"COR": _rate(cor, len(parts["COR"])), "AGR": _rate(agr, len(parts["AGR"])), "TERM": term_pred,
            "REP": rep_pred, "stream": _rate(expected, total)}


def print_page(stats: Mapping[str, Any], pred: Mapping[str, Any], counts: Mapping[str, Any], band: Mapping[str, Any]) -> None:
    p = stats["by_part"]
    pct = lambda x: "n/a" if x is None else f"{100 * x:.2f}%"  # noqa: E731
    lines = [
        f"parts      COR {p['COR']}  AGR {p['AGR']}  TERM {p['TERM']}  REP {p['REP']}  total {stats['total']}"
        f"  (REP {pct(_rate(p['REP'], stats['total']))})",
        f"COR        by source/reason: {stats['cor_by_source_reason']}",
        f"           by source/target: {stats['cor_by_source_target']}",
        f"           by part/device:   {stats['by_part_device']}",
        f"STOP       stream {pct(stats['stop_share'])} (predicted {pct(pred['stream'])}; by part "
        f"{ {k: pct(v) for k, v in stats['stop_share_by_part'].items()} })",
    ]
    for name, row in stats["stop_by_distance"].items():
        lines.append(f"           {name:>5}: {row['stop']}/{row['samples']} = {pct(row['share'])}  base per call "
                     f"{row['reference_stop']}/{row['reference_calls']} = {pct(row['reference_share'])}  ratio {row['ratio']}")
    lines += [
        f"           band {band['band']}: TERM dropped for it {band['term_dropped_for_band']}",
        f"reversals  in the stream {stats['reversals']} (dropped before: "
        f"{ {k: v for k, v in counts.items() if k.endswith('_dropped_reversal')} })",
        f"turn+STOP  in the stream {stats['turn_with_stop']} (dropped before: "
        f"{ {k: v for k, v in counts.items() if k.endswith('_dropped_turn_with_stop')} })",
        f"pixels     outside u[100,540] v[32,455]: {stats['pixel_outside_window']} of {stats['pixel_samples']}",
        f"kl         samples {pct(stats['kl_sample_share'])}, supervised tokens {pct(stats['kl_token_share'])}",
        f"not taken  {dict(sorted((k, v) for k, v in counts.items() if not k.endswith(('_labelled', '_kept'))))}",
        f"TERM       candidates past call 48 kept: {counts.get('TERM_beyond_first_calls_kept', 0)}",
    ]
    if p["COR"] < 15000:
        lines.append(f"dose       COR {p['COR']} < 15,000 (D1's minimum)")
    print("\n".join(f"[exp34] {line}" for line in lines), flush=True)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", nargs=3, action="append", required=True, metavar=("NAME", "RECORD_ROOT", "LABEL_ROOT"),
                        help="base records and their line-B labels (label_route_corrections.py); repeatable")
    parser.add_argument("--replay-root", required=True, help="InternData-N1 r2r scene dirs")
    parser.add_argument("--train-scenes-file", default="", help="HM3D training scenes (EXP-28's train_scenes.txt)")
    parser.add_argument("--holdout-scenes-file", default="", help="more held-out scenes, added to the R2R six and HM3D 20")
    parser.add_argument("--first-calls", type=int, default=FIRST_CALLS)
    parser.add_argument("--max-cor-per-episode", type=int, default=MAX_COR_PER_EPISODE)
    parser.add_argument("--max-agr-per-episode", type=int, default=MAX_AGR_PER_EPISODE)
    parser.add_argument("--replay-fraction", type=float, default=REPLAY_FRACTION)
    parser.add_argument("--stop-band", type=float, nargs=2, default=list(STOP_BAND), metavar=("LO", "HI"))
    parser.add_argument("--allow-other-labeler", action="store_true",
                        help="accept labels of another RouteLabelConfig than the default (recorded in the manifest)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--vlm-image-size", type=int, nargs=2, default=list(DEFAULT_VLM_IMAGE_SIZE))
    parser.add_argument("--jpeg-quality", type=int, default=90, help="the client's --rpc_jpeg_quality, for replay")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="select and print the statistics, write nothing")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.shard_size % args.global_batch or args.global_batch % args.world_size:
        raise SystemExit("shard size must be a multiple of the global batch, and that of the world size")
    names = [s[0] for s in args.source]
    if len(set(names)) != len(names) or not all(_SOURCE_NAME.match(n) for n in names):
        raise SystemExit(f"source names must be distinct and [A-Za-z0-9-]: {names}")
    out_dir = Path(args.out_dir)
    if not args.dry_run and (out_dir / MANIFEST_NAME).exists():
        raise SystemExit(f"{out_dir} already holds a stream")
    started = time.time()
    held = held_out_scenes(args.holdout_scenes_file)
    train = training_scenes(args.train_scenes_file, held)
    default_labeler = RouteLabelConfig().digest()

    # --- every source: index, check, classify
    candidates: list[Candidate] = []
    counts: Counter = Counter()
    reference: list[tuple[float, bool]] = []
    sources: dict[str, Any] = {}
    for name, record_root, label_root in args.source:
        record_root, label_root = str(Path(record_root).resolve()), str(Path(label_root).resolve())
        refs, index_stats = index_calls([record_root], [label_root], split="train", holdout=brs.holdout_match_set(held))
        refs, held_by_id = brs.drop_held_out(refs, held)
        outside = brs.outside_scenes(refs, train)
        if outside:
            raise SystemExit(f"source {name}: records of scenes neither held out nor training scenes (calls per scene): "
                             f"{dict(sorted(outside.items())[:20])} ({len(outside)} scenes)")
        paths = sorted({r.record_path for r in refs})
        labels = brs.label_provenance([record_root], [label_root], paths)
        if labels["stale_label_files"]:
            raise SystemExit(f"source {name}: {labels['stale_label_files']} label files label another version of their "
                             f"record, e.g. {labels['stale_examples']}: relabel them")
        if (set(labels["labelers"]) != {default_labeler} or set(labels["lines"]) != {LABEL_LINE}) and paths:
            if not args.allow_other_labeler:
                raise SystemExit(f"source {name}: labels by {labels['labelers']} (lines {labels['lines']}), not line B's "
                                 f"default RouteLabelConfig {default_labeler}; --allow-other-labeler takes them")
        adapters = Counter(a for a in _map(_record_adapter, paths, args.workers, chunksize=32) if a is not None)
        if adapters:
            raise SystemExit(f"source {name}: records served by an adapter, not the base model: {dict(adapters)}")
        by_record: dict[str, list[CallRef]] = defaultdict(list)
        for ref in refs:
            by_record[ref.record_path].append(ref)
        mine: Counter = Counter()
        for path in paths:
            rows = {int(r["call_index"]): r for r in read_labels(label_path_for(path, label_root))}
            found, c, ref_calls = classify_record(by_record[path], rows, source=name, first_calls=args.first_calls)
            candidates += found
            mine.update(c)
            reference += ref_calls
        counts.update(mine)
        sources[name] = {"record_root": record_root, "label_root": label_root, "index": dict(index_stats),
                         "calls_held_out_by_scene_id": held_by_id, "records": len(paths), "labels": labels,
                         "counts": dict(mine), "candidates": dict(Counter(c.part for c in candidates if c.source == name)),
                         "devices": dict(Counter(r.record_device for r in refs))}
        print(f"[exp34] source {name}: {len(paths)} records, {len(refs)} calls, "
              f"candidates {sources[name]['candidates']}", flush=True)
    dagger, dagger_info = draw_dagger(candidates, max_cor_per_episode=args.max_cor_per_episode,
                                      max_agr_per_episode=args.max_agr_per_episode)
    if not dagger["COR"]:
        raise SystemExit(f"no COR samples (counts {dict(counts)})")
    if dagger_info["agr_capped_by_pool"]:
        print(f"[exp34] warning: AGR {dagger_info['agr']} < COR {dagger_info['cor']}: the capped anchor pool is short",
              file=sys.stderr, flush=True)

    # --- REP, the STOP band, the order
    pool, replay_counts = load_replay_pool(args.replay_root, held)
    parts, band = compose(dagger, replay_units(pool), replay_fraction=args.replay_fraction, shard_size=args.shard_size,
                          seed=args.seed, band=tuple(args.stop_band))
    items = make_items(parts, args.seed)
    leaks = Counter(brs.scene_key(ms.sample_scene(i["key"])) for i in items
                    if brs.scene_key(ms.sample_scene(i["key"])) in held)
    if leaks:
        raise SystemExit(f"held-out scenes in the stream (samples per scene): {dict(leaks)}")
    stats = stream_stats(items, reference)
    pred = predicted_stop(dagger_info, parts, pool, len(items))
    print_page(stats, pred, counts, band)
    total = len(items)
    if total % args.shard_size:
        raise SystemExit(f"{total} samples are not whole shards of {args.shard_size}")
    if args.dry_run:
        print(json.dumps({"dagger": dagger_info, "replay": dict(replay_counts), "band": band}, sort_keys=True))
        return 0

    # --- write
    out_dir.mkdir(parents=True, exist_ok=True)
    num_shards = total // args.shard_size
    jobs = [(s, items[s * args.shard_size:(s + 1) * args.shard_size], str(out_dir), tuple(args.vlm_image_size),
             args.jpeg_quality) for s in range(num_shards)]
    shards: dict[int, dict] = {}
    for index, digest, size in _map(_write_job, jobs, args.workers):
        shards[index] = {"name": shard_name(index), "sha256": digest, "bytes": size}
    print(f"[exp34] {num_shards} shards written", flush=True)
    keys_text = "".join(
        f"{i['key']}\t{i['part']}\t{str(i['kl']).lower()}\t"
        + (f"{i['record_path']}#{i['call_index']}\t{i['device']}\t{i['d']:.4f}" if i["part"] != "REP"
           else f"{i['scene_dir']}@{i['sample']['frame_id']}\treplay\t") + "\n"
        for i in items)
    (out_dir / KEYS_TSV).write_text(keys_text, encoding="utf-8")

    drawn: Counter = Counter()
    for item in items:
        drawn[item["source"]] += 1
        drawn[f"{item['source']}_target_{decision_class(item['turns'][-1])}"] += 1
        drawn[f"category_{item['part']}"] += 1
    holdout = sorted(held)
    dagger_items = [i for i in items if i["source"] == "dagger"]
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
        "replay_fraction": round(len(parts["REP"]) / total, 6),
        "holdout": {"scenes": holdout, "count": len(holdout), "r2r_scenes": list(R2R_HOLDOUT),
                    "hm3d_scenes": list(HM3D_HOLDOUT), "file": args.holdout_scenes_file},
        # train_system2_lora.py copies these three into the adapter manifest
        "record_roots": [s["record_root"] for s in sources.values()],
        "label_roots": [s["label_root"] for s in sources.values()],
        "replay_root": str(Path(args.replay_root).resolve()),
        "kl": {"field": "kl", "where": "each sample's {i:06d}.json in the shard (also 'part'; keys.tsv column 3)",
               "by_part": KL},
        "selection": {
            "rule": "EXP-34 D1 on line-B labels of the base's own records (P_old): COR every line-B correction "
                    "(kl false), AGR line-B anchors 1:1 (<=6 per episode, sha256 'exp34-agr:'+key), TERM the base's "
                    "executed calls within 3 m (ceil(COR/2), sha256 'exp34-term:'+key), REP InternData r2r at 20% "
                    "with STOP x5; COR/AGR call_index < 48 (TERM any index), <=24 COR per episode, reversals and turn "
                    "strings with STOP dropped",
            "knobs": {"first_calls": args.first_calls, "max_cor_per_episode": args.max_cor_per_episode,
                      "max_agr_per_episode": args.max_agr_per_episode, "replay_fraction": args.replay_fraction,
                      "stop_upsample": STOP_UPSAMPLE, "stop_band": list(args.stop_band),
                      "allow_other_labeler": args.allow_other_labeler, "train_scenes_file": args.train_scenes_file,
                      "agr_rank": f"sha256('{AGR_RANK}' + key)", "term_rank": f"sha256('{TERM_RANK}' + key)"},
            "sources": sources,
            "counts": dict(counts),
            "dagger": dagger_info,
            "replay": dict(replay_counts),
            "stop_band": band,
            "parts": stats["by_part"],
            "stop_share_predicted": pred,
            "stop_share": stats["stop_share"],
            "stats": stats,
            "kl_token_share_estimate": stats["kl_token_share"],
            "dagger_selected": len(dagger_items),
            "selected_keys_sha256": hashlib.sha256(
                "\n".join(i["key"] + "|" + "\x1f".join(i["turns"]) for i in dagger_items).encode()).hexdigest(),
            "keys_sha256": hashlib.sha256(
                "\n".join(i["key"] + "|" + "\x1f".join(i["turns"]) for i in items).encode()).hexdigest(),
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

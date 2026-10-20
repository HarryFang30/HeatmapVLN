#!/usr/bin/env python3
"""EXP-32's checks before training: data validity (A1-A4), the stream audit (D1) and the sampled dose (D3).

The ledger's EXP-32 第二部分 (pre-registered before any part-1 reading) writes every line; this script
only reads them.  Three subcommands, all on the 4090's CPU:

``validity`` -- A, per collection directory ``TAG=DIR:SEED`` (DIR holds ``records/``, ``npu_servers/``,
``runtime/`` and ``logs/``; ``DIR/records`` is accepted too), against ``--anchor-dir`` (``collect_base``):

=====  ===================================================================================================
A1     Every model-server log of a sampled source matches ``SAMPLES_LOG`` -- the 910B's
       (``npu_servers/*/logs/model_*.log``, pulled with its ``servers.json``) and the 4090's local slots'
       (``runtime/*/logs/model_*.log``) -- and every commit those servers ran has ``top_k=0`` in its
       ``scripts/evaluation/rpc_model_server.py`` (``git show`` in ``--git-repo``; the commit is
       ``servers.json``'s ``repo_commit``, or for a local slot the ``commit=`` that ``collect_4090.sh``
       logged in ``logs/collect_local.log`` before ``collecting (<runtime stamp>)``).  No model-server
       log of the anchor directory may match.
A2     Pass 1 only: the share of its records whose call 0 (``recorded_turns``) equals the base's call 0
       on the same episode (the anchor directory's record) is below 0.90.  Reported for every source.
A3     Pass 1 only: its success rate is in [20%, 50%].  Outside the band with A1 met (the logs and the
       server commit) the source is used and the report says so (``a3.note``: write it in the run
       record).  Reported for every source.
A4     Every call's ``sampling.protocol_seed`` is the source's: p1 and r0s 1337, p2 42, p3 2024,
       ``heldout_sN`` N, the anchors 42.  A record that differs is counted; more than 1% voids the source.
=====  ===================================================================================================

A source that fails is void (``valid`` false); the others stand.

``stream`` -- D1 on the merged training stream (``merge_streams.py`` output) and the RFT stream it was
merged from (``inputs.dagger.dir``, whose ``keys.tsv`` maps each sample to its record and call):

1. no sample of a held-out scene (``--holdout-scenes-file``, EXP-23's six); every RFT or anchor sample of
   a training scene (``--train-scenes-file``), every replay sample InternData r2r (RF8);
2. RF1-RF7: every RFT record ``success == 1``, base-served, of a training scene, its calls all of the
   source's seed, its source valid in a ``validity`` report (A1, A4); every anchor record under
   ``--anchor-root``, cohort shard >= 20, ``success == 1``, seed 42; no STOP, forward or invalid target;
   every sample's call one that ``build_rft_stream.rft_calls`` keeps (no reversal, nothing within 3 m
   or beyond call 47) with the recorded answer as its target; keys ``{scene}_{episode:06d}#{tag}.{call}``;
   every sample once; at most 2 trajectories and 6 anchors per episode, anchors 1:1 and each episode's
   lowest sha256 ranks; the builder's knobs the ledger's;
3. 500 RFT samples drawn with seed 32 (``merge_streams.read_entry``): turns = ``recorded_turns``,
   instruction and the history, current and lookdown JPEGs byte for byte the record's -- 500/500;
4. the merged stream's STOP share in [3%, 12%];
5. B2's dose rule: N_rft (``category_rft_*``) after pass 2 >= 4,000 and no pass 3, or pass 3 run and
   N_rft >= 2,000 after it;
6. ``git diff e515ec2 <code commit> -- src scripts/evaluation scripts/train_system2_lora.py
   scripts/run_*.sh scripts/ascend`` empty.

``dose`` -- D3 on ``--parity-check 2000 --parity-split all --parity-device npu`` (adapter off) over
150 npu RFT records: G-samp = 1 - exact >= 0.10.  Reported: the ``same_answer`` divergence
(``scripts/exp30/probe_judge.py``), over all rows and over the rows that are RFT samples of the stream,
and the effective dose N_rft x the latter.  The D3 records are copies; each is matched to the RFT
trajectory it copies (``rft_records.txt``) by its ``<scene dir>/<file>`` and, when two sources share an
episode, by a path component named for the source's tag (stage them as ``<dir>/<tag>/<scene dir>/<file>``)
or by ``--record-map`` (``staged path<TAB>original path``); every row's recorded answer is then checked
against the original.  ``--m-states`` gets M's states: the stream's RFT calls among the rows whose npu
base replay does not ``same_answer`` the recorded answer.

Each prints its verdict line first and writes ``--out``.  Exit status: 0 = every line met, 3 = a line
failed (the report says which), 1 = refused (an input that cannot be judged)::

    P=/opt/conda/bin/python; A=scripts/exp32/aud.py; H=/workspace/exp28/episodes/splits
    $P $A validity --source p1=/workspace/exp32/collect_a:1337 --source r0s=/workspace/exp31/r0s:1337 \\
        --source p2=/workspace/exp32/p2:42 --anchor-dir /workspace/exp28/collect_base \\
        --out /workspace/exp32/aud/validity.json
    $P $A stream --stream /workspace/exp32/stream --validity /workspace/exp32/aud/validity.json \\
        --anchor-root /workspace/exp28/collect_base/records \\
        --train-scenes-file $H/train_scenes.txt --holdout-scenes-file $H/heldout_scenes.txt \\
        --out /workspace/exp32/aud/stream.json
    $P $A dose --parity-dir <D3 output> --rft-stream /workspace/exp32/stream_rft \\
        --m-states /workspace/exp32/aud/m_states.jsonl --out /workspace/exp32/aud/dose.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.exp28 import merge_streams as ms  # noqa: E402
from scripts.exp30.probe_judge import same_answer  # noqa: E402
from scripts.exp32 import build_rft_stream as brs  # noqa: E402
from scripts.tools.build_route_stream import read_scene_list, record_directions, scene_key  # noqa: E402
from src.data.s2_records import RECORD_SUFFIX, EpisodeRecord, iter_record_paths  # noqa: E402
from src.data.s2_sft_dataset import (  # noqa: E402
    S2SFTError,
    decision_class,
    holdout_scenes,
    record_device,
    recorded_turns,
)
from src.data.s2_stream import MANIFEST_NAME, STREAM_SCHEMA  # noqa: E402

VALIDITY_SCHEMA = "heatmapvln-exp32-aud-validity-v1"
STREAM_AUDIT_SCHEMA = "heatmapvln-exp32-aud-stream-v1"
DOSE_SCHEMA = "heatmapvln-exp32-aud-dose-v1"
PARITY_SCHEMA = "heatmapvln-s2-parity-v1"  # train_system2_lora.PARITY_SCHEMA (not imported: it pulls in torch)
MERGE_BUILDER = "scripts/exp28/merge_streams.py"
PASS, FAIL = 0, 3

# A1 (the 910B's serve_910b.sh:45 and rpc_model_server.py:1077-1078 write the two forms)
SAMPLES_LOG = re.compile(r"System2 SAMPLES \((T=0\.7 top_p=0\.95|\{'temperature': 0\.7, 'top_p': 0\.95\})\)")
ANY_SAMPLES_LOG = "System2 SAMPLES"
MODEL_LOG_GLOBS = ("npu_servers/*/logs/model_*.log", "runtime/*/logs/model_*.log")
SERVER_FILE = "scripts/evaluation/rpc_model_server.py"
TOP_K_FIX = "top_k=0"
HELDOUT_TAG = re.compile(r"heldout_s(\d+)")
_COMMIT = re.compile(r"\bcommit=([0-9a-f]{7,40})\b")
_COLLECTING = re.compile(r"collecting \(([^)\s]+)\)")
# A2, A3 (pass 1 only)
A2_BELOW = Fraction(9, 10)
A3_BAND = (Fraction(1, 5), Fraction(1, 2))
GATED_A2_A3 = ("p1",)
# D1
BYTE_SAMPLES, BYTE_SEED = 500, 32
STOP_SHARE_BAND = (Fraction(3, 100), Fraction(12, 100))
DOSE_PASS3_BELOW, DOSE_TRAIN_MIN = 4000, 2000  # B2
GIT_BASE = "e515ec2"
GIT_PATHS = ("src", "scripts/evaluation", "scripts/train_system2_lora.py", "scripts/run_*.sh", "scripts/ascend")
RF_KNOBS: dict[str, Any] = {  # the ledger's RF1-RF6 as build_rft_stream.py's knobs
    "first_calls": brs.DEFAULT_FIRST_CALLS, "min_distance": brs.DEFAULT_MIN_DISTANCE,
    "rft_classes": list(brs.RFT_CLASSES), "drop_reversals": True, "max_trajectories_per_episode": 2,
    "max_anchors_per_episode": 6, "anchor_ratio": 1.0, "anchor_seed": 32,
    "anchor_protocol_seed": brs.ANCHOR_PROTOCOL_SEED, "anchor_min_shard": brs.ANCHOR_MIN_SHARD,
    "weight": 1.0, "no_pad": True,
}
REQUIRED_SOURCES = ("p1", "r0s", "p2")  # RF1 (B1: pass 2 always runs)
# D3
G_SAMP_MIN = Fraction(1, 10)


def _pmap(fn: Callable, items: Sequence, workers: int) -> list:
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items, chunksize=8))


def _rate(k: int, n: int) -> float | None:
    return round(k / n, 6) if n else None


def _write_json(path: str | os.PathLike, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, Fraction):
        return float(value)
    return str(value)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(repo: str | os.PathLike, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "safe.directory=*", "-C", str(repo), *args], capture_output=True, text=True)


def episode_of(path: str | os.PathLike) -> tuple[str, int]:
    """(bare scene, episode) of a record path (``s2_records.record_path``: ``{scene}_{episode:06d}.s2rec.tar``)."""
    name = Path(path).name
    stem = name[: -len(RECORD_SUFFIX)] if name.endswith(RECORD_SUFFIX) else name
    scene, _, episode = stem.rpartition("_")
    return scene_key(scene), int(episode)


# --------------------------------------------------------------------------- #
# A: validity
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RecordSummary:
    """What A2-A4 read of one committed record."""

    path: str
    scene: str  # bare scene ID
    episode_id: int
    success: bool
    seeds: frozenset  # every call's sampling.protocol_seed (None: a call without one)
    call0: tuple[str, ...] | None  # recorded_turns of call 0 (None: no readable call 0)
    device: str


def summarize_record(path: str) -> RecordSummary:
    record = EpisodeRecord(path)
    call0 = None
    for call in record.calls:
        if int(call["call_index"]) == 0:
            try:
                call0 = recorded_turns(call)
            except S2SFTError:
                call0 = None
    return RecordSummary(str(path), scene_key(record.scene_id), record.episode_id,
                         brs.episode_success(record.episode), frozenset(brs.call_seeds(record.calls)), call0,
                         record_device(record))


def parse_collection(text: str) -> tuple[str, Path, int]:
    """``TAG=DIR:SEED`` -> (tag, collection directory, declared System1 seed), the seed A4's for the tag."""
    tag, sep, rest = str(text).partition("=")
    root, colon, seed = rest.rpartition(":")
    if not sep or not colon or not root:
        raise SystemExit(f"--source takes TAG=DIR:SEED; got {text!r}")
    try:
        seed_value = int(seed)
    except ValueError:
        raise SystemExit(f"--source {text!r}: the seed {seed!r} is not an integer") from None
    held = HELDOUT_TAG.fullmatch(tag)
    if tag in brs.DECLARED_SEEDS:
        if seed_value != brs.DECLARED_SEEDS[tag]:
            raise SystemExit(f"--source {text!r}: source {tag} was collected with System1 seed "
                             f"{brs.DECLARED_SEEDS[tag]} (ledger EXP-32 A4), not {seed_value}")
    elif held:
        if int(held.group(1)) != seed_value:
            raise SystemExit(f"--source {text!r}: {tag} names System1 seed {held.group(1)}, not {seed_value}")
    else:
        raise SystemExit(f"--source {text!r}: TAG is one of {list(brs.RFT_TAGS)} or heldout_s<SEED>")
    return tag, collection_dir(root), seed_value


def collection_dir(path: str | os.PathLike) -> Path:
    """A collection directory (``collect_4090.sh``'s EXP28_OUT_DIR); its ``records`` subdirectory names it too."""
    path = Path(path)
    return path.parent if path.name == "records" else path


def records_root(directory: Path) -> str:
    """The resolved ``records/`` of a collection, the form ``build_rft_stream`` writes into its manifest."""
    root = (directory / "records").resolve()
    if not root.is_dir():
        raise SystemExit(f"{directory} has no records/ directory")
    return str(root)


def model_logs(directory: Path) -> list[Path]:
    return sorted({p for pattern in MODEL_LOG_GLOBS for p in directory.glob(pattern) if p.is_file()})


def log_lines(path: Path) -> dict[str, int]:
    """Lines matching A1's regex, and lines naming System2 sampling in any form (reported)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return {"matches": len(SAMPLES_LOG.findall(text)), "any_samples_line": text.count(ANY_SAMPLES_LOG)}


def launch_commits(log_dir: Path) -> dict[str, str]:
    """runtime stamp -> the commit ``collect_4090.sh`` logged before ``collecting (<stamp>)``."""
    out: dict[str, str] = {}
    for log in sorted(log_dir.glob("collect_*.log")):
        commit = None
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            found = _COMMIT.search(line)
            if found and " mode=" in line:
                commit = found.group(1)
            stamp = _COLLECTING.search(line)
            if stamp and commit:
                out[stamp.group(1)] = commit
    return out


def serving_commits(directory: Path) -> tuple[dict[str, list[str]], list[str]]:
    """commit -> the server sets (with model-server logs) that ran it; problems (a set whose commit is unknown
    or whose checkout was dirty, so the commit does not say what ran)."""
    commits: dict[str, list[str]] = defaultdict(list)
    problems: list[str] = []
    for server_set in sorted(p for p in directory.glob("npu_servers/*") if p.is_dir()):
        if not list(server_set.glob("logs/model_*.log")):
            continue
        rel = str(server_set.relative_to(directory))
        servers = server_set / "servers.json"
        if not servers.is_file():
            problems.append(f"{rel} has model-server logs but no servers.json")
            continue
        data = json.loads(servers.read_text(encoding="utf-8"))
        commit = str(data.get("repo_commit") or "")
        if not commit:
            problems.append(f"{rel}/servers.json names no repo_commit")
            continue
        if int(data.get("repo_dirty") or 0):
            problems.append(f"{rel}/servers.json: the servers' checkout had uncommitted changes")
        commits[commit].append(rel)
    stamps = launch_commits(directory / "logs")
    for runtime in sorted(p for p in directory.glob("runtime/*") if p.is_dir()):
        if not list(runtime.glob("logs/model_*.log")):
            continue  # an external run: its servers are the npu_servers set's
        rel = str(runtime.relative_to(directory))
        commit = stamps.get(runtime.name)
        if not commit:
            problems.append(f"{rel}: no logs/collect_*.log names the commit its local servers ran")
            continue
        commits[commit].append(rel)
    return dict(commits), problems


def commit_has_top_k_fix(git_repo: str | os.PathLike, commit: str) -> tuple[bool, str]:
    shown = _git(git_repo, "show", f"{commit}:{SERVER_FILE}")
    if shown.returncode:
        return False, f"git show {commit}:{SERVER_FILE} failed in {git_repo}: {shown.stderr.strip()[:200]}"
    if TOP_K_FIX not in shown.stdout:
        return False, f"commit {commit}: {SERVER_FILE} has no {TOP_K_FIX} (sampling would be greedy)"
    return True, ""


def a1_check(directory: Path, *, sampled: bool, git_repo: str | os.PathLike) -> dict[str, Any]:
    """A1 for one collection directory: a sampled source's logs all match and its servers' commits carry the
    top_k fix; the anchor's logs match none."""
    logs = {str(p.relative_to(directory)): log_lines(p) for p in model_logs(directory)}
    reasons: list[str] = []
    out: dict[str, Any] = {"logs": logs, "n_logs": len(logs),
                           "n_matching": sum(bool(v["matches"]) for v in logs.values())}
    if sampled:
        if not logs:
            reasons.append("no model-server log")
        unmatched = sorted(k for k, v in logs.items() if not v["matches"])
        if unmatched:
            reasons.append(f"{len(unmatched)} of {len(logs)} model-server logs lack the sampling line: {unmatched[:5]}")
        commits, problems = serving_commits(directory)
        reasons += problems
        if logs and not commits and not problems:
            reasons.append("no server set names its commit")
        out["serving_commits"] = {}
        for commit in sorted(commits):
            ok, why = commit_has_top_k_fix(git_repo, commit)
            out["serving_commits"][commit] = {"top_k_fix": ok, "server_sets": commits[commit]}
            if not ok:
                reasons.append(why)
    else:
        matched = sorted(k for k, v in logs.items() if v["matches"])
        if matched:
            reasons.append(f"model-server logs of the greedy base show the sampling line: {matched[:5]}")
    out["pass"] = not reasons
    out["reasons"] = reasons
    return out


def call0_identity(source: Iterable[RecordSummary], base: Mapping[tuple[str, int], RecordSummary]) -> dict[str, Any]:
    """A2: the share of the source's records whose call 0 is the base record's call 0, verbatim."""
    pairs = identical = unpaired = 0
    same_device: Counter = Counter()
    for summary in source:
        other = base.get((summary.scene, summary.episode_id))
        if other is None or summary.call0 is None or other.call0 is None:
            unpaired += 1
            continue
        same = summary.call0 == other.call0
        pairs += 1
        identical += same
        if summary.device == other.device:
            same_device["pairs"] += 1
            same_device["identical"] += same
    rate = Fraction(identical, pairs) if pairs else None
    return {"pairs": pairs, "identical": identical, "rate": None if rate is None else round(float(rate), 6),
            "line": f"< {float(A2_BELOW)}", "pass": rate is not None and rate < A2_BELOW, "unpaired": unpaired,
            "same_device": {**same_device, "rate": _rate(same_device["identical"], same_device["pairs"])}}


def success_band(source: Sequence[RecordSummary]) -> dict[str, Any]:
    """A3: the source's success rate against [20%, 50%]."""
    n, k = len(source), sum(s.success for s in source)
    rate = Fraction(k, n) if n else None
    return {"records": n, "successes": k, "rate": None if rate is None else round(float(rate), 6),
            "band": [float(A3_BAND[0]), float(A3_BAND[1])],
            "in_band": rate is not None and A3_BAND[0] <= rate <= A3_BAND[1]}


def seed_share(source: Sequence[RecordSummary], seed: int) -> dict[str, Any]:
    """A4: records whose calls carry another System1 seed (``build_rft_stream``'s reading: a record without
    calls has none to disagree); more than 1% fails."""
    mismatched = [s for s in source if s.seeds - {seed}]
    seen: Counter = Counter(str(v) for s in source for v in s.seeds)
    return {"records": len(source), "mismatched": len(mismatched), "declared_seed": seed,
            "share": _rate(len(mismatched), len(source)), "line": "<= 0.01", "seeds_seen": dict(seen),
            "examples": [s.path for s in mismatched[:5]],
            "pass": bool(source) and brs.seed_share_ok(len(mismatched), len(source))}


def summarize_root(root: str, workers: int) -> list[RecordSummary]:
    paths = [str(p) for p in iter_record_paths(root)]
    if not paths:
        raise SystemExit(f"{root} holds no committed record")
    return _pmap(summarize_record, paths, workers)


def validity(args: argparse.Namespace) -> int:
    sources = [parse_collection(text) for text in args.source]
    tags = [tag for tag, _, _ in sources]
    if len(set(tags)) != len(tags):
        raise SystemExit(f"--source names a tag twice: {tags}")
    anchor_dir = collection_dir(args.anchor_dir)
    anchor_root = records_root(anchor_dir)
    anchor = summarize_root(anchor_root, args.workers)
    base: dict[tuple[str, int], RecordSummary] = {}
    for summary in anchor:
        if base.setdefault((summary.scene, summary.episode_id), summary) is not summary:
            raise SystemExit(f"episode {summary.scene} {summary.episode_id} is recorded twice under {anchor_root}")
    anchor_a1 = a1_check(anchor_dir, sampled=False, git_repo=args.git_repo)
    anchor_a4 = seed_share(anchor, brs.ANCHOR_PROTOCOL_SEED)
    report: dict[str, Any] = {
        "schema": VALIDITY_SCHEMA,
        "created_unix": time.time(),
        "lines": {"a1_regex": SAMPLES_LOG.pattern, "a1_top_k": TOP_K_FIX, "a2_below": float(A2_BELOW),
                  "a3_band": [float(A3_BAND[0]), float(A3_BAND[1])], "a4_max_share": float(brs.MAX_SEED_MISMATCH_SHARE),
                  "a2_a3_gated_for": list(GATED_A2_A3), "git_repo": str(args.git_repo)},
        "anchor": {"dir": str(anchor_dir.resolve()), "records_root": anchor_root,
                   "declared_seed": brs.ANCHOR_PROTOCOL_SEED, "a1": anchor_a1, "a4": anchor_a4,
                   "valid": anchor_a1["pass"] and anchor_a4["pass"],
                   "void_reasons": [f"A1: {r}" for r in anchor_a1["reasons"]] + (
                       [] if anchor_a4["pass"] else [f"A4: {anchor_a4['mismatched']} records off seed 42"])},
        "sources": {},
    }
    for tag, directory, seed in sources:
        root = records_root(directory)
        summaries = summarize_root(root, args.workers)
        a1 = a1_check(directory, sampled=True, git_repo=args.git_repo)
        a2 = {**call0_identity(summaries, base), "gated": tag in GATED_A2_A3}
        a3 = {**success_band(summaries), "gated": tag in GATED_A2_A3}
        a4 = seed_share(summaries, seed)
        void = [f"A1: {r}" for r in a1["reasons"]]
        if not a4["pass"]:
            void.append(f"A4: {a4['mismatched']} of {a4['records']} records off System1 seed {seed}")
        if a2["gated"] and not a2["pass"]:
            void.append(f"A2: call-0 identity {a2['rate']} over {a2['pairs']} pairs is not below {float(A2_BELOW)}")
        if a3["gated"] and not a3["in_band"]:
            if a1["pass"]:
                a3["note"] = (f"success rate {a3['rate']} outside [0.2, 0.5] with A1 and the server commit met: "
                              f"used; write it in the run record (ledger EXP-32 A3)")
            else:
                void.append(f"A3: success rate {a3['rate']} outside [0.2, 0.5] and A1 not met: resample")
        report["sources"][tag] = {"tag": tag, "dir": str(directory.resolve()), "records_root": root,
                                  "declared_seed": seed, "a1": a1, "a2": a2, "a3": a3, "a4": a4,
                                  "valid": not void, "void_reasons": void}
    valid = report["anchor"]["valid"] and all(s["valid"] for s in report["sources"].values())
    report["verdict"] = "PASS" if valid else "FAIL"
    parts = [f"{tag} {'valid' if s['valid'] else 'VOID'}" for tag, s in report["sources"].items()]
    parts.append(f"anchor {'valid' if report['anchor']['valid'] else 'VOID'}")
    print(f"VALIDITY {report['verdict']}: " + ", ".join(parts), flush=True)
    for name, entry in [*report["sources"].items(), ("anchor", report["anchor"])]:
        for reason in entry["void_reasons"]:
            print(f"  {name}: {reason}", flush=True)
        note = (entry.get("a3") or {}).get("note")
        if note:
            print(f"  {name}: A3 note: {note}", flush=True)
    _write_json(args.out, report)
    return PASS if valid else FAIL


# --------------------------------------------------------------------------- #
# D1: the stream
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KeyRow:
    """One line of the RFT stream's ``keys.tsv``."""

    key: str
    record_path: str
    call_index: int
    category: str  # rft_p1, rft_r0s, rft_p2, rft_p3 or anchor
    device: str


def read_keys_tsv(text: str) -> tuple[dict[str, KeyRow], list[str]]:
    """key -> its row; keys listed twice (RF6)."""
    rows: dict[str, KeyRow] = {}
    twice: list[str] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 4 or "#" not in parts[1]:
            raise SystemExit(f"{brs.KEYS_TSV} line {number}: expected key<TAB>record#call<TAB>category<TAB>device")
        key, uid, category, device = parts
        path, _, call = uid.rpartition("#")
        if key in rows:
            twice.append(key)
            continue
        rows[key] = KeyRow(key, path, int(call), category, device)
    return rows, twice


def read_rft_records(text: str) -> dict[str, tuple[str, str]]:
    """record path -> (record_device, tag), from ``rft_records.txt``."""
    out: dict[str, tuple[str, str]] = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise SystemExit(f"{brs.RECORDS_LIST} line {number}: expected record_path<TAB>device<TAB>tag")
        if parts[0] in out:
            raise SystemExit(f"{brs.RECORDS_LIST} lists {parts[0]} twice")
        out[parts[0]] = (parts[1], parts[2])
    return out


def category_tag(category: str) -> str | None:
    """keys.tsv category -> the key's tag (RF7), None for an unknown category."""
    if category == "anchor":
        return brs.ANCHOR_TAG
    tag = category[len("rft_"):] if category.startswith("rft_") else ""
    return tag if tag in brs.RFT_TAGS else None


def leak_check(entries: Iterable[ms.Entry], *, train: set[str], held: set[str]) -> dict[str, Any]:
    """D1 item 1 over the merged stream's samples, by their keys' scenes: none held out; RFT and anchor samples
    (``dagger``) of a training scene; replay samples InternData r2r's (RF8)."""
    held_out: Counter = Counter()
    outside: Counter = Counter()
    not_r2r: Counter = Counter()
    other: Counter = Counter()
    for entry in entries:
        scene = scene_key(ms.sample_scene(entry.key))
        if scene in held:
            held_out[scene] += 1
        if entry.source == "dagger":
            if scene not in train:
                outside[scene] += 1
        elif entry.source == "replay":
            if ms.replay_dataset(entry.key) != "r2r":
                not_r2r[scene] += 1
        else:
            other[entry.source] += 1
    reasons = [f"{what} (samples per scene): {dict(found)}" for what, found in (
        ("held-out scenes", held_out), ("RFT or anchor samples outside the training scenes", outside),
        ("replay samples not of InternData r2r", not_r2r), ("samples of another source", other)) if found]
    return {"held_out": dict(held_out), "dagger_outside_training_scenes": dict(outside),
            "replay_not_r2r": dict(not_r2r), "other_sources": dict(other), "pass": not reasons, "reasons": reasons}


def stop_share(stops: int, total: int) -> dict[str, Any]:
    """D1 item 4: the merged stream's STOP targets within [3%, 12%]."""
    share = Fraction(stops, total) if total else None
    ok = share is not None and STOP_SHARE_BAND[0] <= share <= STOP_SHARE_BAND[1]
    band = [float(STOP_SHARE_BAND[0]), float(STOP_SHARE_BAND[1])]
    return {"stops": stops, "total": total, "share": None if share is None else round(float(share), 6), "band": band,
            "pass": ok, "reasons": [] if ok else [f"{stops} STOP targets of {total} samples, outside {band}"]}


def dose_rule(drawn: Mapping[str, Any], tags: Iterable[str]) -> dict[str, Any]:
    """D1 item 5, B2: N_rft (``category_rft_*``) after pass 2 >= 4,000 and no pass 3; or below it, pass 3 run
    and N_rft >= 2,000 after it.  Pass 3 only adds episodes no earlier pass succeeded on, so the counts of p1,
    r0s and p2 are what the dry run after pass 2 gave."""
    tags = set(tags)
    counts = {tag: int(drawn.get(f"category_rft_{tag}", 0) or 0) for tag in brs.RFT_TAGS}
    after_pass2 = sum(counts[t] for t in ("p1", "r0s", "p2"))
    total = after_pass2 + counts["p3"]
    reasons = []
    missing = [t for t in REQUIRED_SOURCES if t not in tags]
    if missing:
        reasons.append(f"sources {missing} are not in the RFT stream (RF1; B1 runs pass 2 always)")
    if after_pass2 < DOSE_PASS3_BELOW:
        if "p3" not in tags:
            reasons.append(f"N_rft after pass 2 is {after_pass2} < {DOSE_PASS3_BELOW}: pass 3 runs first (B2)")
        elif total < DOSE_TRAIN_MIN:
            reasons.append(f"N_rft after pass 3 is {total} < {DOSE_TRAIN_MIN}: no training, EXP-32 stops (B2)")
    elif "p3" in tags:
        reasons.append(f"N_rft after pass 2 is {after_pass2} >= {DOSE_PASS3_BELOW}: pass 3 was not to run (B2)")
    return {"n_rft": total, "after_pass2": after_pass2, "by_source": counts, "pass3_run": "p3" in tags,
            "lines": {"pass3_below": DOSE_PASS3_BELOW, "train_min": DOSE_TRAIN_MIN},
            "pass": not reasons, "reasons": reasons}


def git_unchanged(repo: str | os.PathLike, base: str, commit: str, paths: Sequence[str] = GIT_PATHS) -> dict[str, Any]:
    """D1 item 6: ``git diff <base> <commit> -- <paths>`` is empty."""
    shas = {}
    for name, ref in (("base", base), ("commit", commit)):
        parsed = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
        if parsed.returncode:
            return {"pass": False, "repo": str(repo), "paths": list(paths), "changed": None,
                    "reasons": [f"git cannot resolve {ref} in {repo}: {parsed.stderr.strip()[:200]}"]}
        shas[name] = parsed.stdout.strip()
    diff = _git(repo, "diff", "--name-only", shas["base"], shas["commit"], "--", *paths)
    if diff.returncode:
        return {"pass": False, "repo": str(repo), **shas, "paths": list(paths), "changed": None,
                "reasons": [f"git diff failed: {diff.stderr.strip()[:200]}"]}
    changed = [line for line in diff.stdout.splitlines() if line.strip()]
    return {"pass": not changed, "repo": str(repo), **shas, "paths": list(paths), "changed": changed,
            "reasons": [f"{len(changed)} files changed between {base} and {commit}: {changed[:10]}"] if changed else []}


def _byte_mismatches(packed: Any, record: EpisodeRecord, call: Mapping[str, Any], key: str) -> list[str]:
    """The packed sample's fields that are not the record's, byte for byte (D1 item 3)."""
    fields = []
    try:
        if packed.key != key:
            fields.append("key")
        if packed.turns != recorded_turns(call):
            fields.append("turns")
        if packed.instruction != str(record.episode["instruction"]):
            fields.append("instruction")
        if packed.history_jpegs != tuple(record.front_jpeg(s) for s in call["history_step_ids"]):
            fields.append("history_jpegs")
        if packed.current_jpeg != record.front_jpeg(call["step_id"]):
            fields.append("current_jpeg")
        if packed.lookdown_jpeg != record.lookdown_jpeg(int(call["call_index"])):
            fields.append("lookdown_jpeg")
    except KeyError as err:
        fields.append(f"missing {err}")
    return fields


def audit_record(job: tuple) -> dict[str, Any]:
    """One record of the stream's RFT or anchor part: its RF1/RF5 facts, and every one of its samples checked
    against ``rft_calls`` (RF2-RF4, RF7) and, for the drawn ones, byte for byte (item 3).

    ``job`` = (record path, tag, [(key, call_index, packed turns, keys.tsv device, Entry or None)], anchor seed).
    """
    path, tag, samples, anchor_seed = job
    record = EpisodeRecord(path)
    calls = {int(c["call_index"]): c for c in record.calls}
    kept = {c.call_index: c for c in brs.rft_calls(record)[0]}
    directions = record_directions(str(record.path))
    device = record_device(record)
    problems: Counter = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    mismatched: list[dict[str, Any]] = []
    checked = 0

    def flag(reason: str, key: str) -> None:
        problems[reason] += 1
        if len(examples[reason]) < 5:
            examples[reason].append(key)

    for key, call_index, turns, listed_device, entry in samples:
        if key != brs.sample_key(record.scene_id, record.episode_id, tag, call_index):
            flag("key", key)
        if listed_device != device:
            flag("device", key)
        call = calls.get(int(call_index))
        if call is None:
            flag("call_missing", key)
            continue
        recorded = recorded_turns(call)
        if tuple(turns) != recorded:
            flag("target_not_recorded", key)
        if call_index not in kept:
            distance = call.get("distance_to_goal")
            why = brs.call_verdict(call_index, recorded, None if distance is None else float(distance),
                                   brs.previous_direction(directions, call_index))
            flag(why or "not_kept", key)
        if entry is not None:
            checked += 1
            fields = _byte_mismatches(ms.read_entry(entry), record, call, key)
            if fields:
                mismatched.append({"key": key, "fields": fields})
    info: dict[str, Any] = {
        "path": str(path), "tag": tag, "scene": scene_key(record.scene_id),
        "dir_scene": scene_key(Path(path).parent.name), "episode": (scene_key(record.scene_id), record.episode_id),
        "success": brs.episode_success(record.episode), "seeds": frozenset(brs.call_seeds(record.calls)),
        "adapter": brs.served_adapter(record.episode), "shard": brs.episode_shard(record.episode), "device": device,
        "samples": len(samples), "problems": dict(problems), "examples": dict(examples),
        "bytes_checked": checked, "bytes_mismatched": mismatched,
    }
    if tag == brs.ANCHOR_TAG:
        ranked = sorted(kept, key=lambda i: brs.anchor_rank(anchor_seed, f"{path}#{i}"))
        info["anchor_lowest_ranks"] = sorted(s[1] for s in samples) == sorted(ranked[: len(samples)])
    else:
        answers = [(i, list(recorded_turns(calls[i]))) for i in sorted(calls)]
        info["answers_sha256"] = _sha256(json.dumps(answers, ensure_ascii=False))
    return info


def record_problems(info: Mapping[str, Any], *, root: str, seed: int, train: set[str], held: set[str]) -> list[str]:
    """RF1 (an RFT record) or RF5 (an anchor record, tag ``g``) on one audited record."""
    out = []
    if not str(info["path"]).startswith(root.rstrip(os.sep) + os.sep):
        out.append("outside_its_root")
    if not info["success"]:
        out.append("not_success")
    if info["seeds"] - {seed}:
        out.append("seed_mismatch")
    if info["adapter"] is not None:
        out.append("served_adapter")
    if info["scene"] in held or info["dir_scene"] in held:
        out.append("held_out_scene")
    elif info["scene"] not in train or info["dir_scene"] not in train:
        out.append("outside_training_scenes")
    if info["tag"] == brs.ANCHOR_TAG:
        if info["shard"] is None:
            out.append("no_cohort_shard")
        elif info["shard"] < brs.ANCHOR_MIN_SHARD:
            out.append(f"shard_below_{brs.ANCHOR_MIN_SHARD}")
    return out


def load_validity(paths: Sequence[str]) -> tuple[dict[str, dict], list[dict]]:
    """The sources (tag -> entry) and anchor entries of one or more ``validity`` reports."""
    sources: dict[str, dict] = {}
    anchors: list[dict] = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema") != VALIDITY_SCHEMA:
            raise SystemExit(f"{path} is not an aud.py validity report")
        for tag, entry in (data.get("sources") or {}).items():
            if tag in sources:
                raise SystemExit(f"source {tag} is in two validity reports")
            sources[tag] = entry
        if data.get("anchor"):
            anchors.append(data["anchor"])
    return sources, anchors


def _validity_problems(entry: Mapping[str, Any] | None, name: str, root: str) -> list[str]:
    if entry is None:
        return [f"{name}: no validity report covers it"]
    if entry.get("records_root") != root:
        return [f"{name}: the validity report checked {entry.get('records_root')}, the stream read {root}"]
    out = []
    if not (entry.get("a1") or {}).get("pass"):
        out.append(f"{name}: A1 not met")
    if not (entry.get("a4") or {}).get("pass"):
        out.append(f"{name}: A4 not met")
    if not entry.get("valid"):
        out.append(f"{name}: void ({'; '.join(entry.get('void_reasons') or [])})")
    return out


def stream(args: argparse.Namespace) -> int:
    holdout_file = read_scene_list(args.holdout_scenes_file)
    held = {scene_key(s) for s in (*holdout_file, *holdout_scenes())}
    train = {scene_key(s) for s in read_scene_list(args.train_scenes_file)}
    if train & held:
        raise SystemExit(f"{args.train_scenes_file} lists held-out scenes: {sorted(train & held)}")
    merged, entries, merged_sha = ms.load_stream(args.stream, schemas=(STREAM_SCHEMA,), verify=True)
    if not str(merged.get("builder", "")).startswith(MERGE_BUILDER):
        raise SystemExit(f"{args.stream} was built by {merged.get('builder')!r}, not {MERGE_BUILDER}")
    dagger_input = (merged.get("inputs") or {}).get("dagger") or {}
    rft_dir = Path(args.rft_stream or dagger_input.get("dir") or "")
    if not (rft_dir / MANIFEST_NAME).is_file():
        raise SystemExit(f"no RFT stream manifest at {rft_dir} (pass --rft-stream)")
    rft_raw = (rft_dir / MANIFEST_NAME).read_bytes()
    rft = json.loads(rft_raw)
    if not str(rft.get("builder", "")).startswith(ms.RFT_BUILDER):
        raise SystemExit(f"{rft_dir} was built by {rft.get('builder')!r}, not {ms.RFT_BUILDER}")
    if hashlib.sha256(rft_raw).hexdigest() != dagger_input.get("manifest_sha256"):
        raise SystemExit(f"{args.stream} was not merged from {rft_dir} (its inputs.dagger.manifest_sha256 differs)")
    selection = rft.get("selection") or {}
    knobs = selection.get("knobs") or {}
    keys_text = (rft_dir / brs.KEYS_TSV).read_text(encoding="utf-8")
    records_text = (rft_dir / brs.RECORDS_LIST).read_text(encoding="utf-8")
    if (_sha256(keys_text) != selection.get("keys_tsv_sha256")
            or _sha256(records_text) != selection.get("rft_records_sha256")):
        raise SystemExit(f"{rft_dir}: {brs.KEYS_TSV} or {brs.RECORDS_LIST} does not match its manifest")
    rows, keys_twice = read_keys_tsv(keys_text)
    trajectories = read_rft_records(records_text)
    validity_sources, validity_anchors = load_validity(args.validity)
    started = time.time()

    # --- 1: leaks
    leaks = leak_check(entries, train=train, held=held)
    leaks["merged_holdout_complete"] = held <= {scene_key(s) for s in (merged.get("holdout") or {}).get("scenes", ())}

    # --- 2: RF1-RF7
    reasons: list[str] = []
    off_knobs = {k: knobs.get(k) for k, v in RF_KNOBS.items() if knobs.get(k) != v}
    if off_knobs:
        reasons.append(f"the RFT stream was built with other knobs than RF1-RF6: {off_knobs}")
    rft_roots = {tag: str(info.get("root")) for tag, info in (knobs.get("rft_sources") or {}).items()}
    for tag, info in (knobs.get("rft_sources") or {}).items():
        if tag not in brs.DECLARED_SEEDS or int(info.get("protocol_seed", -1)) != brs.DECLARED_SEEDS[tag]:
            reasons.append(f"RFT source {tag} declared seed {info.get('protocol_seed')} "
                           f"(A4: {brs.DECLARED_SEEDS.get(tag)})")
        reasons += _validity_problems(validity_sources.get(tag), f"source {tag}", rft_roots[tag])
    anchor_root = str(Path(args.anchor_root).resolve())
    if knobs.get("anchor_root") != anchor_root:
        reasons.append(f"the anchors were drawn from {knobs.get('anchor_root')}, not {anchor_root}")
    matching_anchor = [a for a in validity_anchors if a.get("records_root") == anchor_root]
    reasons += _validity_problems(matching_anchor[0] if matching_anchor else None, "anchor root", anchor_root)
    reasons += [p for a in matching_anchor[1:] for p in _validity_problems(a, "anchor root", anchor_root)]

    dagger = [e for e in entries if e.source == "dagger"]
    dagger_keys = Counter(e.key for e in dagger)
    unmapped = sorted(k for k in dagger_keys if k not in rows)
    missing = sorted(k for k in rows if k not in dagger_keys)
    twice_in_stream = sorted(k for k, n in dagger_keys.items() if n > 1)
    uids = Counter((row.record_path, row.call_index) for row in rows.values())
    calls_twice = sorted(f"{p}#{c}" for (p, c), n in uids.items() if n > 1)
    bad_categories = sorted({row.category for row in rows.values() if category_tag(row.category) is None})
    rft_keys = sorted(k for k in dagger_keys if k in rows and rows[k].category != "anchor")
    order = np.random.default_rng(args.byte_seed).permutation(len(rft_keys))
    drawn_keys = {rft_keys[int(i)] for i in order[: args.byte_samples]}
    wanted_bytes = min(args.byte_samples, len(rft_keys))

    by_record: dict[str, list[tuple]] = defaultdict(list)
    tags_of: dict[str, set[str]] = defaultdict(set)
    for entry in dagger:
        row = rows.get(entry.key)
        if row is None or category_tag(row.category) is None:
            continue
        tags_of[row.record_path].add(category_tag(row.category))
        by_record[row.record_path].append((entry.key, row.call_index, entry.turns, row.device,
                                           entry if entry.key in drawn_keys else None))
    for path, (_, tag) in trajectories.items():
        tags_of[path].add(tag)
        by_record.setdefault(path, [])
    two_roles = sorted(p for p, t in tags_of.items() if len(t) > 1)
    jobs = [(path, next(iter(tags_of[path])), by_record[path], int(knobs.get("anchor_seed", RF_KNOBS["anchor_seed"])))
            for path in sorted(by_record) if path not in two_roles]
    infos = _pmap(audit_record, jobs, args.workers)

    record_issues: Counter = Counter()
    record_examples: dict[str, list[str]] = defaultdict(list)
    sample_issues: Counter = Counter()
    sample_examples: dict[str, list[str]] = defaultdict(list)
    per_episode_rft: dict[tuple, list[dict]] = defaultdict(list)
    per_episode_anchor: Counter = Counter()
    prefix_off: list[str] = []
    for info in infos:
        tag = info["tag"]
        if tag == brs.ANCHOR_TAG:
            root, seed = anchor_root, brs.ANCHOR_PROTOCOL_SEED
            per_episode_anchor[info["episode"]] += info["samples"]
            if not info["anchor_lowest_ranks"]:
                prefix_off.append(info["path"])
        else:
            root, seed = rft_roots.get(tag, ""), brs.DECLARED_SEEDS.get(tag, -1)
            if tag not in brs.DECLARED_SEEDS:
                record_issues["unknown_tag"] += 1
                record_examples["unknown_tag"].append(info["path"])
            per_episode_rft[info["episode"]].append(info)
            listed = trajectories.get(info["path"])
            if listed is None:
                record_issues["not_in_rft_records"] += 1
                record_examples["not_in_rft_records"].append(info["path"])
            elif listed != (info["device"], tag):
                record_issues["rft_records_line_differs"] += 1
                record_examples["rft_records_line_differs"].append(info["path"])
        if not root:
            record_issues["source_not_in_manifest"] += 1
        for reason in record_problems(info, root=root or "/nonexistent", seed=seed, train=train, held=held):
            record_issues[reason] += 1
            if len(record_examples[reason]) < 5:
                record_examples[reason].append(info["path"])
        sample_issues.update(info["problems"])
        for reason, keys in info["examples"].items():
            sample_examples[reason] = (sample_examples[reason] + keys)[:5]
    over_cap = sorted(f"{s}_{e:06d}" for (s, e), v in per_episode_rft.items()
                      if len(v) > RF_KNOBS["max_trajectories_per_episode"])
    duplicates = sorted(f"{s}_{e:06d}" for (s, e), v in per_episode_rft.items()
                        if len({i["answers_sha256"] for i in v}) < len(v))
    anchors_over = sorted(f"{s}_{e:06d}" for (s, e), n in per_episode_anchor.items()
                          if n > RF_KNOBS["max_anchors_per_episode"])
    anchor_records_per_episode = Counter(i["episode"] for i in infos if i["tag"] == brs.ANCHOR_TAG)
    anchor_episode_twice = sorted(f"{s}_{e:06d}" for (s, e), n in anchor_records_per_episode.items() if n > 1)
    n_rft = sum(1 for k in dagger_keys if k in rows and rows[k].category != "anchor")
    n_anchor = sum(1 for k in dagger_keys if k in rows and rows[k].category == "anchor")
    draw = (selection.get("anchors") or {}).get("draw") or {}
    capped = bool((selection.get("composition") or {}).get("anchors_capped_by_pool"))
    one_to_one = n_anchor == n_rft or (capped and n_anchor < n_rft and n_anchor == draw.get("pool_after_episode_cap"))
    targets = Counter(decision_class(e.turns[-1]) for e in dagger)
    bad_targets = {c: n for c, n in targets.items() if c not in brs.RFT_CLASSES}
    for condition, reason in (
        (record_issues, f"records: {dict(record_issues)} (e.g. {dict(record_examples)})"),
        (sample_issues, f"samples: {dict(sample_issues)} (e.g. {dict(sample_examples)})"),
        (bad_targets, f"targets other than pixel or turn in the RFT/anchor part: {bad_targets}"),
        (unmapped, f"{len(unmapped)} samples not in {brs.KEYS_TSV}: {unmapped[:5]}"),
        (missing, f"{len(missing)} {brs.KEYS_TSV} keys not in the merged stream: {missing[:5]}"),
        (twice_in_stream or keys_twice, f"keys twice (RF6): {(twice_in_stream + keys_twice)[:5]}"),
        (calls_twice, f"calls packed twice (RF6): {calls_twice[:5]}"),
        (bad_categories, f"unknown categories in {brs.KEYS_TSV}: {bad_categories}"),
        (two_roles, f"records under two tags: {two_roles[:5]}"),
        (over_cap, f"episodes with more than {RF_KNOBS['max_trajectories_per_episode']} trajectories: {over_cap[:5]}"),
        (duplicates, f"episodes with two trajectories equal call for call: {duplicates[:5]}"),
        (anchors_over, f"episodes with more than {RF_KNOBS['max_anchors_per_episode']} anchors: {anchors_over[:5]}"),
        (anchor_episode_twice, f"anchor episodes from two records: {anchor_episode_twice[:5]}"),
        (prefix_off, f"anchor records whose samples are not their lowest sha256 ranks: {prefix_off[:5]}"),
        (not one_to_one, f"{n_anchor} anchors for {n_rft} RFT samples (RF5: 1:1, fewer only when the pool is short)"),
    ):
        if condition:
            reasons.append(reason)
    rules = {
        "pass": not reasons, "reasons": reasons, "knobs_off": off_knobs, "rft_sources": rft_roots,
        "anchor_root": anchor_root,
        "records": {"rft": sum(1 for i in infos if i["tag"] != brs.ANCHOR_TAG),
                    "anchor": sum(1 for i in infos if i["tag"] == brs.ANCHOR_TAG),
                    "problems": dict(record_issues), "examples": dict(record_examples)},
        "samples": {"rft": n_rft, "anchor": n_anchor, "problems": dict(sample_issues),
                    "examples": dict(sample_examples)},
        "targets": dict(targets),
        "stop_forward_invalid_targets": sum(targets.get(c, 0) for c in ("stop", "forward", "invalid")),
        "reversals_left": int(sample_issues.get("reversal", 0)),
        "within_3m_or_beyond_call_47": sum(int(sample_issues.get(r, 0))
                                           for r in ("within_min_distance", "no_distance", "beyond_first_calls")),
        "trajectories": {"listed": len(trajectories), "episodes": len(per_episode_rft),
                         "max_per_episode": max((len(v) for v in per_episode_rft.values()), default=0)},
        "anchors": {"wanted": draw.get("wanted"), "taken": n_anchor, "capped_by_pool": capped,
                    "max_per_episode": max(per_episode_anchor.values(), default=0),
                    "lowest_ranks_off": len(prefix_off)},
    }

    # --- 3: bytes of 500 RFT samples
    checked = sum(i["bytes_checked"] for i in infos)
    byte_mismatches = [m for i in infos for m in i["bytes_mismatched"]]
    byte_check = {"seed": args.byte_seed, "wanted": wanted_bytes, "checked": checked,
                  "mismatched": len(byte_mismatches), "examples": byte_mismatches[:10], "keys": sorted(drawn_keys),
                  "pass": checked == wanted_bytes and wanted_bytes > 0 and not byte_mismatches}
    if not byte_check["pass"]:
        byte_check["reasons"] = [f"{checked - len(byte_mismatches)}/{wanted_bytes} drawn RFT samples are their "
                                 f"records' byte for byte ({len(byte_mismatches)} differ, {checked} checked)"]

    # --- 4: STOP share of the merged stream
    stops = stop_share(sum(decision_class(e.turns[-1]) == "stop" for e in entries), len(entries))
    stops["manifest_stop_share"] = (merged.get("selection") or {}).get("stop_share")

    # --- 5: B2
    dose = dose_rule(rft.get("drawn") or {}, rft_roots)
    dose["merged_categories_agree"] = all(int((merged.get("drawn") or {}).get(f"category_rft_{t}", 0) or 0) ==
                                          dose["by_source"][t] for t in brs.RFT_TAGS)
    if not dose["merged_categories_agree"]:
        dose["pass"] = False
        dose["reasons"].append("the merged manifest's category_rft_* differ from the RFT stream's")

    # --- 6: the deployed code
    git = git_unchanged(args.git_repo, args.base_commit, args.code_commit)

    items = {"1_leaks": leaks, "2_rules": rules, "3_bytes": byte_check, "4_stop_share": stops, "5_dose": dose,
             "6_code_unchanged": git}
    failed = [name for name, item in items.items() if not item["pass"]]
    report = {
        "schema": STREAM_AUDIT_SCHEMA, "created_unix": time.time(), "verdict": "FAIL" if failed else "PASS",
        "failed": failed, "stream": str(Path(args.stream).resolve()), "stream_manifest_sha256": merged_sha,
        "rft_stream": str(rft_dir.resolve()), "rft_manifest_sha256": hashlib.sha256(rft_raw).hexdigest(),
        "total_samples": len(entries), "n_rft": dose["n_rft"], "items": items,
        "audit_s": round(time.time() - started, 1),
    }
    print(f"AUD {report['verdict']}" + (f": {', '.join(failed)}" if failed else "") + " " + json.dumps({
        "total": len(entries), "n_rft": n_rft, "n_anchor": n_anchor,
        "bytes": f"{checked - len(byte_mismatches)}/{wanted_bytes}",
        "stop_share": stops["share"], "dose_after_pass2": dose["after_pass2"], "code_changed": git.get("changed")}),
        flush=True)
    for name in failed:
        for reason in items[name].get("reasons") or []:
            print(f"  {name}: {reason}", flush=True)
    _write_json(args.out, report)
    return FAIL if failed else PASS


# --------------------------------------------------------------------------- #
# D3: the sampled dose
# --------------------------------------------------------------------------- #
def read_record_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    out = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise SystemExit(f"{path} line {number}: expected staged_path<TAB>original_path")
        out[parts[0]] = parts[1]
    return out


def map_records(paths: Iterable[str], trajectories: Mapping[str, tuple[str, str]],
                explicit: Mapping[str, str]) -> dict[str, str]:
    """D3 record path -> the RFT trajectory (``rft_records.txt``) it is a copy of: ``explicit``, the path itself,
    else the one with its ``<scene dir>/<file>``, or of those the one whose tag names a directory on the path."""
    by_tail: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for original in trajectories:
        by_tail[Path(original).parts[-2:]].append(original)
    out: dict[str, str] = {}
    problems = []
    for path in sorted(set(paths)):
        if path in explicit:
            if explicit[path] not in trajectories:
                problems.append(f"--record-map maps {path} to {explicit[path]}, which is no RFT trajectory")
            out[path] = explicit[path]
            continue
        if path in trajectories:
            out[path] = path
            continue
        candidates = by_tail.get(Path(path).parts[-2:], [])
        if len(candidates) > 1:
            dirs = set(Path(path).parts[:-2])
            candidates = [c for c in candidates if trajectories[c][1] in dirs]
        if len(candidates) == 1:
            out[path] = candidates[0]
        else:
            problems.append(f"{path}: {len(candidates)} RFT trajectories fit")
    if problems:
        raise SystemExit(f"cannot tell which RFT trajectory {len(problems)} D3 records copy (stage them as "
                         f"<dir>/<tag>/<scene dir>/<file> or pass --record-map): {problems[:5]}")
    return out


def dose(args: argparse.Namespace) -> int:
    parity_dir = Path(args.parity_dir)
    report_path, rows_path = parity_dir / "parity_g0.json", parity_dir / "parity_g0.calls.jsonl"
    if not report_path.is_file() or not rows_path.is_file():
        raise SystemExit(f"{parity_dir} lacks parity_g0.json or parity_g0.calls.jsonl")
    parity = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if parity.get("schema") != PARITY_SCHEMA:
        raise SystemExit(f"{report_path}: schema {parity.get('schema')!r} is not {PARITY_SCHEMA}")
    if parity.get("split") != "all" or parity.get("record_device") != "npu":
        raise SystemExit(f"{report_path}: D3 is --parity-split all --parity-device npu, this ran split "
                         f"{parity.get('split')!r} on {parity.get('record_device')!r} records")
    if not rows or len(rows) != int(parity.get("n", -1)):
        raise SystemExit(f"{rows_path} holds {len(rows)} rows, the report says {parity.get('n')}")
    rft_dir = Path(args.rft_stream)
    rft = json.loads((rft_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    keys, _ = read_keys_tsv((rft_dir / brs.KEYS_TSV).read_text(encoding="utf-8"))
    trajectories = read_rft_records((rft_dir / brs.RECORDS_LIST).read_text(encoding="utf-8"))
    in_stream = {(r.record_path, r.call_index): r for r in keys.values() if r.category != "anchor"}
    n_rft = sum(int(v or 0) for k, v in (rft.get("drawn") or {}).items() if k.startswith("category_rft_"))

    split_uid = [(row["uid"].rpartition("#")[0], int(row["uid"].rpartition("#")[2])) for row in rows]
    mapping = map_records((p for p, _ in split_uid), trajectories, read_record_map(args.record_map))
    originals: dict[str, dict[int, Mapping[str, Any]]] = {}
    for original in sorted(set(mapping.values())):
        originals[original] = {int(c["call_index"]): c for c in EpisodeRecord(original).calls}
    wrong = []
    for row, (path, call_index) in zip(rows, split_uid):
        call = originals[mapping[path]].get(call_index)
        if call is None or recorded_turns(call) != tuple(row["recorded"]):
            wrong.append(row["uid"])
    if wrong:
        raise SystemExit(f"{len(wrong)} D3 rows do not carry their RFT record's recorded answer (wrong mapping?): "
                         f"{wrong[:5]}")

    n = len(rows)
    exact = sum(bool(row["exact"]) for row in rows)
    g_samp = Fraction(n - exact, n)
    passed = g_samp >= G_SAMP_MIN
    diverged = [not same_answer(row["replay"], row["recorded"]) for row in rows]
    states = []
    hits = 0
    for row, (path, call_index), apart in zip(rows, split_uid, diverged):
        original = mapping[path]
        key_row = in_stream.get((original, call_index))
        if key_row is None:
            continue
        hits += 1
        if apart:
            scene, episode = episode_of(original)
            states.append({"key": key_row.key, "uid": f"{original}#{call_index}", "record_path": original,
                           "call_index": call_index, "tag": category_tag(key_row.category),
                           "record_device": key_row.device, "scene_id": scene, "episode_id": episode,
                           "recorded": list(row["recorded"]), "base_replay_npu": list(row["replay"]),
                           "d3_uid": row["uid"]})
    states.sort(key=lambda s: s["key"])
    in_diverged = len(states)
    rate_in = Fraction(in_diverged, hits) if hits else None
    devices = Counter(trajectories[o][0] for o in set(mapping.values()))
    report = {
        "schema": DOSE_SCHEMA, "created_unix": time.time(), "verdict": "PASS" if passed else "FAIL",
        "parity_dir": str(parity_dir.resolve()), "rft_stream": str(rft_dir.resolve()),
        "n": n, "exact": exact, "exact_rate": round(exact / n, 6), "report_exact_rate": parity.get("exact_rate"),
        "g_samp": round(float(g_samp), 6), "line": f">= {float(G_SAMP_MIN)}", "pass": passed,
        "records": {"d3": len(mapping), "rft_trajectories": len(set(mapping.values())), "by_device": dict(devices),
                    "by_tag": dict(Counter(trajectories[o][1] for o in set(mapping.values())))},
        "divergence_same_answer": {
            "all": {"diverged": sum(diverged), "n": n, "rate": _rate(sum(diverged), n)},
            "stream_rft_rows": {"diverged": in_diverged, "n": hits, "rate": _rate(in_diverged, hits)}},
        "n_rft": n_rft,
        "effective_dose": None if rate_in is None else round(float(n_rft * rate_in), 1),
        "m_states": {"n": len(states), "file": str(Path(args.m_states).resolve())},
        "parity": {k: parity.get(k)
                   for k in ("split", "record_device", "adapter", "device", "world_size", "provenance")},
    }
    Path(args.m_states).parent.mkdir(parents=True, exist_ok=True)
    with open(args.m_states, "w", encoding="utf-8") as handle:
        for state in states:
            handle.write(json.dumps(state, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"D3 {report['verdict']}: G-samp {report['g_samp']} (1 - exact, line >= 0.10) " + json.dumps({
        "n": n, "exact": exact, "divergence_all": report["divergence_same_answer"]["all"]["rate"],
        "divergence_stream_rows": report["divergence_same_answer"]["stream_rft_rows"]["rate"],
        "n_rft": n_rft, "effective_dose": report["effective_dose"], "m_states": len(states)}), flush=True)
    _write_json(args.out, report)
    return PASS if passed else FAIL


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    val = sub.add_parser("validity", help="A1-A4 per collection directory")
    val.add_argument("--source", action="append", required=True, metavar="TAG=DIR:SEED",
                     help=f"a sampled collection, TAG in {list(brs.RFT_TAGS)} or heldout_s<SEED> (repeatable)")
    val.add_argument("--anchor-dir", required=True, help="collect_base: the base model, greedy, System1 seed 42")
    val.add_argument("--git-repo", default=str(REPO_ROOT), help="a checkout holding the servers' commits (A1's top_k)")
    st = sub.add_parser("stream", help="D1 on the merged stream")
    st.add_argument("--stream", required=True, help="merge_streams.py's output")
    st.add_argument("--rft-stream", default="", help="build_rft_stream.py's output (default: the merge's input)")
    st.add_argument("--validity", action="append", required=True,
                    help="validity reports covering every source (repeatable)")
    st.add_argument("--anchor-root", required=True, help="collect_base's records")
    st.add_argument("--train-scenes-file", required=True, help="the 279 HM3D training scenes")
    st.add_argument("--holdout-scenes-file", required=True, help="the 20 held-out HM3D scenes")
    st.add_argument("--byte-samples", type=int, default=BYTE_SAMPLES)
    st.add_argument("--byte-seed", type=int, default=BYTE_SEED)
    st.add_argument("--git-repo", default=str(REPO_ROOT))
    st.add_argument("--base-commit", default=GIT_BASE)
    st.add_argument("--code-commit", default="HEAD", help="x32, the deployed implementation commit")
    do = sub.add_parser("dose", help="D3: G-samp, the same_answer divergence, the effective dose and M's states")
    do.add_argument("--parity-dir", required=True, help="train_system2_lora.py --parity-check output (parity_g0.*)")
    do.add_argument("--rft-stream", required=True, help="build_rft_stream.py's output (keys.tsv, rft_records.txt)")
    do.add_argument("--record-map", default="", help="staged_path<TAB>original_path, for D3 copies not staged by tag")
    do.add_argument("--m-states", required=True, help="jsonl of M's states to write")
    for p in (val, st):
        p.add_argument("--workers", type=int, default=12)
    for p in (val, st, do):
        p.add_argument("--out", required=True, help="JSON report")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return {"validity": validity, "stream": stream, "dose": dose}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

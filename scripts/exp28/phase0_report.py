#!/usr/bin/env python3
"""EXP-28 Phase-0: the base model's GO/NO-GO on HM3D, from probe_metrics' per-episode decompositions (CPU, no Habitat).

Phase-0 (ledger EXP-28 B, fixed before any HM3D closed-loop result; definitions as EXP-25 B1):
- sample: the 400 held-out episodes (910B servers) and the first 400 training-cohort episodes (4090 local slots),
  800 in all; greedy, System1 seed 42, 500 steps, no step-budget STOP.
- SR, B and E7 are counted exactly as ``scripts/exp25/probe_metrics.py`` ``summarize`` counts them (that script run
  with ``--scenes`` on the rebaked HM3D navmeshes): SR = success > 0 over all episodes; B = first pixel goal "bad"
  over the episodes that have one; E7 = failed, never within 3 m, ended with STOP, >= 7 m, over all episodes.
- NO-GO: B <= 5.5% or E7 < 5.0% or SR >= 85%  -> EXP-28 stops, collection too.  Checked first, so SR = 85% is NO-GO.
- GO:    B >= 8.0% and E7 >= 7.0% and SR <= 85%.
- middle (neither): E7 alone decides, E7 >= 7.0% -> GO, else NO-GO.
The verdict is taken on the pooled point estimates, compared as exact fractions (no rounding).  Reported only:
scene-cluster bootstrap intervals (scenes resampled with replacement within each platform half, so a half keeps its
scene count; 20,000 draws), each platform half on its own, OS, and the share of the episodes that InternData also
has (``--overlap-keys``).

Replay gate (B is the only one of the three read off geometry): probe_metrics replays each episode's logged actions
on the navmesh, and its last call's ``next_g`` is the replayed end distance, which must be the client's own ``ne``.
On MP3D (the research decompositions of A0 val_unseen, seeds 42 and 1337, 1,839 episodes each) the two agree to
0.00 m on every episode.  More than 2% of the episodes off by > 0.25 m, or with a non-finite geodesic, means the
replay is not the walk the client made (coordinates, heading or navmesh) and B is not read: exit 2.
``--no-replay-gate`` only after finding why; the report then says the gate was skipped.

Inputs are probe_metrics ``<name>.decomp.json`` files, labelled by platform half (``LABEL=PATH``, repeatable; one
file may serve both halves when ``--shards`` splits it):

    probe_metrics.py --name phase0 --scenes /workspace/exp28/scene_datasets/hm3d \\
        --datasets <cohort>/dataset_shard_*.json.gz --progress '<collect>/workers/shard_*/progress.json' \\
        --logs '<collect>/runtime/*/logs/client_shard_*.log' --out <dir>
    phase0_report.py --decomp 910b=<dir>/phase0.decomp.json 4090=<dir>/phase0.decomp.json \\
        --cohort <cohort> --shards 910b=0-19 4090=<the local launcher's EXP28_SHARDS> --limit 4090=400 \\
        --expect 910b=400 4090=400 --progress '910b=<collect>/workers/shard_*/progress.json' '4090=...' \\
        --overlap-keys <episodes>/overlap_keys.json --out <dir>

``--shards`` (PPA_COLLECT_SHARDS syntax, needs ``--cohort``, the cohort directory with ``manifest.json``) makes a
half exactly the episodes of those shards, in the order the launcher takes them (shards as listed, episodes in shard
file order); ``--limit N`` (needs ``--shards``) cuts that to the first N.  Every one of them must have a
decomposition -- an episode not yet run or without a log is an input error, never replaced by a later one -- and rows
of other episodes are left out and counted.  Without ``--shards`` a half is every row of its files.  ``--expect``
requires a half's final count.  ``--progress`` checks the protocol on the client's own rows: every episode has a
row, none ran with the step-budget STOP, every row's protocol seed is ``--seed`` (System1's), and no row was served
with a System2 LoRA (``rpc_model_version`` ``...+system2-lora:<sha>``; Phase-0 is the base).  Greedy decoding is not
in the rows (the client drops ``system2_sampling``).  ``--overlap-keys`` is a JSON list
of the episodes that InternData also has, each ``[scene, episode_id]`` or ``{"scene_id": ..., "episode_id": ...}``;
a scene may be written as the HM3D ID, ``00xxx-<ID>`` or a scene path.  A scene in two halves, or an episode twice,
is an input error.

Writes ``<out>/<name>.phase0.json`` and ``.md``; prints the verdict first.  Exit 0 on GO, 1 on NO-GO, 2 when the
inputs are wrong, the replay gate fails, B is undefined, or anything else fails (never 1, which reads as NO-GO).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import traceback
from collections import Counter
from fractions import Fraction
from pathlib import Path, PurePosixPath

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.exp25 import probe_metrics as pm
from scripts.tools.build_r2r_train_cohorts import SCHEMA as COHORT_SCHEMA

SCHEMA = "heatmapvln-exp28-phase0-report-v1"
# ledger EXP-28 B, percent
NOGO_B, GO_B = Fraction(11, 2), Fraction(8)
NOGO_E7, GO_E7 = Fraction(5), Fraction(7)
SR_CAP = Fraction(85)
CRITERIA = {
    "NO-GO": "B <= 5.5% or E7 < 5.0% or SR >= 85% (checked first)",
    "GO": "B >= 8.0% and E7 >= 7.0% and SR <= 85%",
    "middle": "E7 >= 7.0% -> GO, else NO-GO",
}
DRAWS = 20_000
# Replayed end distance vs the client's ne: exact on MP3D (0/1,839 episodes off, A0 s42 and s1337).
REPLAY_TOL_M = 0.25
REPLAY_MAX_OFF = Fraction(2, 100)
LORA_TAG = "+system2-lora:"  # rpc_model_server.py appends it to model_version when it serves an adapter
COUNTS =("episodes", "success", "os", "e7", "with_pixel", "bad")
RATES = {"sr": ("success", "episodes"), "os": ("os", "episodes"), "B": ("bad", "with_pixel"), "E7": ("e7", "episodes")}
_HM3D_DIR = re.compile(r"^\d{5}-(.+)$")


class InputError(Exception):
    """Inputs that cannot give a Phase-0 reading (main exits 2)."""


def scene_key(raw) -> str:
    """``hm3d/<ID>/<ID>.glb``, ``train/00xxx-<ID>/<ID>.basis.glb``, ``00xxx-<ID>``, ``<ID>.basis.glb`` -> ``<ID>``."""
    s = str(raw).strip()
    if "/" in s:
        path = PurePosixPath(s)
        s = path.parent.name if path.suffix else path.name
    s = s.split(".", 1)[0]
    m = _HM3D_DIR.match(s)
    return m.group(1) if m else s


def episode_key(row) -> tuple[str, int]:
    return scene_key(row["scene"]), int(row["episode_id"])


def labelled(values, what: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for value in values or []:
        label, sep, rest = str(value).partition("=")
        if not sep or not label or not rest:
            raise InputError(f"{what} wants LABEL=VALUE, got {value!r}")
        out.setdefault(label, []).append(rest)
    return out


def labelled_int(values, what: str) -> dict[str, int]:
    out = {}
    for label, items in labelled(values, what).items():
        if len(items) != 1 or not items[0].isdigit():
            raise InputError(f"{what} {label}: wants one non-negative integer, got {items}")
        out[label] = int(items[0])
    return out


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_decomp(path: str) -> list[dict]:
    with open(path) as fh:
        rows = json.load(fh)
    if not isinstance(rows, list) or not all(isinstance(r, dict) and "calls" in r for r in rows):
        raise InputError(f"{path}: not a probe_metrics decomp.json (a list of decomposed episodes)")
    return rows


def parse_shards(spec: str) -> list[int]:
    """PPA_COLLECT_SHARDS syntax, in the given order: ``"0-19"``, ``"20-120,130"``."""
    out: list[int] = []
    for part in str(spec).split(","):
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part.strip())
        if not m or (m.group(2) is not None and int(m.group(2)) < int(m.group(1))):
            raise InputError(f"--shards wants indices and ranges like 0,3,5-9, got {spec!r}")
        out.extend(range(int(m.group(1)), int(m.group(2) or m.group(1)) + 1))
    if len(set(out)) != len(out):
        raise InputError(f"--shards {spec!r} names a shard twice")
    return out


def load_cohort(cohort_dir: str) -> dict[int, dict]:
    """Shard index -> {"part", "keys"} of a ``heatmapvln-r2r-train-cohorts-v1`` cohort, files checked by sha256."""
    root = Path(cohort_dir)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputError(f"--cohort {cohort_dir}: cannot read manifest.json ({exc})") from exc
    if manifest.get("schema") != COHORT_SCHEMA:
        raise InputError(f"--cohort {cohort_dir}: schema {manifest.get('schema')!r}, not {COHORT_SCHEMA}")
    shards = {}
    for record in manifest["shards"]:
        path = root / record["path"]
        if sha256(str(path)) != record["sha256"]:
            raise InputError(f"--cohort {cohort_dir}: {path} does not match the manifest's sha256")
        episodes = json.loads(path.read_text(encoding="utf-8"))["episodes"]
        shards[int(record["index"])] = {"part": record.get("part"),
                                        "keys": [(scene_key(e["scene_id"]), int(e["episode_id"])) for e in episodes]}
    return shards


def target_keys(cohort: dict[int, dict], spec: str, limit: int | None) -> tuple[list[tuple[str, int]], list]:
    """The episodes of ``spec``'s shards in launcher order, cut to the first ``limit``; and the cohort parts used."""
    pairs = []
    for index in parse_shards(spec):
        if index not in cohort:
            raise InputError(f"--shards {spec}: shard {index} is not in the cohort ({len(cohort)} shards)")
        pairs.extend((key, cohort[index]["part"]) for key in cohort[index]["keys"])
    if limit is not None:
        if limit > len(pairs):
            raise InputError(f"--limit {limit} is more than the {len(pairs)} episodes of --shards {spec}")
        pairs = pairs[:limit]
    return [key for key, _ in pairs], sorted({str(part) for _, part in pairs})


def select(rows, target: list[tuple[str, int]] | None) -> list[dict]:
    """A half: exactly the ``target`` episodes in that order (each must have a row), else every row by key."""
    if target is None:
        return sorted(rows, key=episode_key)
    by_key = {episode_key(r): r for r in rows}
    missing = [k for k in target if k not in by_key]
    if missing:
        raise InputError(f"{len(missing)} of the {len(target)} episodes the shards name have no decomposition (not "
                         f"run yet, failed, or no client log), e.g. {missing[:3]}")
    return [by_key[k] for k in target]


def episode_counts(row) -> list[int]:
    """One episode's contribution to COUNTS, by probe_metrics' own definitions."""
    cls = pm.pixel_class(pm.first_pixel(row))
    return [1, int(row["success"] > 0), int(row["os"] > 0), int(pm.category(row) == "E7"),
            int(cls is not None), int(cls == "bad")]


def scene_matrix(rows) -> tuple[list[str], np.ndarray]:
    per: dict[str, np.ndarray] = {}
    for r in rows:
        scene = scene_key(r["scene"])
        if scene not in per:
            per[scene] = np.zeros(len(COUNTS), dtype=np.int64)
        per[scene] += np.asarray(episode_counts(r), dtype=np.int64)
    scenes = sorted(per)
    matrix = np.stack([per[s] for s in scenes]) if scenes else np.zeros((0, len(COUNTS)), dtype=np.int64)
    return scenes, matrix


def rates(totals) -> dict[str, np.ndarray]:
    """Percent per RATES from count totals (last axis COUNTS); NaN where the denominator is 0."""
    t = np.asarray(totals, dtype=np.float64)
    out = {}
    for name, (num, den) in RATES.items():
        n, d = t[..., COUNTS.index(num)], t[..., COUNTS.index(den)]
        out[name] = np.where(d > 0, 100.0 * n / np.where(d > 0, d, 1.0), np.nan)
    return out


def cluster_bootstrap(strata, draws: int, seed: int) -> dict:
    """Percentile 95% intervals and SEs (pt) with scenes resampled with replacement within each stratum."""
    rng = np.random.default_rng(seed)
    totals = np.zeros((draws, len(COUNTS)), dtype=np.float64)
    for counts in strata:
        m = len(counts)
        if m == 0:
            continue
        weights = rng.multinomial(m, np.full(m, 1.0 / m), size=draws)
        totals += weights @ np.asarray(counts, dtype=np.float64)
    out = {}
    for name, values in rates(totals).items():
        ok = values[np.isfinite(values)]
        if len(ok) == 0:
            out[name] = {"ci95": [None, None], "se": None, "undefined_draws": draws}
            continue
        lo, hi = np.percentile(ok, [2.5, 97.5])
        out[name] = {"ci95": [round(float(lo), 2), round(float(hi), 2)],
                     "se": round(float(ok.std(ddof=1)), 2) if len(ok) > 1 else 0.0,
                     "undefined_draws": int(draws - len(ok))}
    return out


def exact_pct(share: dict) -> Fraction | None:
    return None if share["n"] == 0 else Fraction(100 * share["k"], share["n"])


def decide(sr: Fraction | None, b: Fraction | None, e7: Fraction | None) -> dict:
    """Ledger EXP-28 B: NO-GO first (SR = 85% is NO-GO), then GO, then the middle tier on E7 alone."""
    if sr is None or e7 is None:
        return {"tier": "undecidable", "verdict": None, "reasons": ["no episodes"]}
    if b is None:
        return {"tier": "undecidable", "verdict": None, "reasons": ["no episode has a first pixel goal, B undefined"]}
    nogo = [text for text, hit in (("B <= 5.5%", b <= NOGO_B), ("E7 < 5.0%", e7 < NOGO_E7),
                                   ("SR >= 85%", sr >= SR_CAP)) if hit]
    if nogo:
        return {"tier": "NO-GO", "verdict": "NO-GO", "reasons": nogo}
    if b >= GO_B and e7 >= GO_E7 and sr <= SR_CAP:
        return {"tier": "GO", "verdict": "GO", "reasons": ["B >= 8.0%", "E7 >= 7.0%", "SR <= 85%"]}
    why = (["5.5% < B < 8.0%"] if b < GO_B else []) + (["5.0% <= E7 < 7.0%"] if e7 < GO_E7 else [])
    if e7 >= GO_E7:
        return {"tier": "middle", "verdict": "GO", "reasons": [*why, "middle: E7 >= 7.0% -> GO"]}
    return {"tier": "middle", "verdict": "NO-GO", "reasons": [*why, "middle: E7 < 7.0% -> NO-GO"]}


def headline(rows) -> dict:
    s = pm.summarize(rows)
    return {k: s[k] for k in ("episodes", "sr", "os", "B", "E7")}


def _spread(values) -> dict:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return {"n": 0, "median": None, "p90": None, "max": None, "nonfinite": len(values)}
    return {"n": len(finite), "median": round(statistics.median(finite), 3),
            "p90": round(finite[min(len(finite) - 1, int(0.9 * len(finite)))], 3), "max": round(finite[-1], 3),
            "nonfinite": len(values) - len(finite)}


def replay_check(rows) -> dict:
    """probe_metrics' replay against the client: the last call's ``next_g`` (replayed end distance) vs ``ne``.

    An episode is off when the two differ by more than REPLAY_TOL_M or any call's geodesic is not finite (a
    replayed position off the goal's navmesh island; such a first pixel would class as good or mid, never bad).
    Also reported: the first call's g (the start, before any action) vs ``info.geodesic_distance``, which for HM3D
    is the annotation's own and may differ by the episode filter's max(1 m, 15%)."""
    ends, starts, off, nonfinite, without_calls = [], [], [], 0, 0
    for r in rows:
        calls = r.get("calls") or []
        if not calls:
            without_calls += 1
            continue
        bad_g = any(not math.isfinite(float(c["g"])) for c in calls)
        nonfinite += int(bad_g)
        end = calls[-1].get("next_g")
        err = math.inf if end is None or not math.isfinite(float(end)) else abs(float(end) - float(r["ne"]))
        ends.append(err)
        if bad_g or not err <= REPLAY_TOL_M:
            off.append(list(episode_key(r)))
        if r.get("geo0") is not None:
            starts.append(abs(float(calls[0]["g"]) - float(r["geo0"])))
    return {"episodes_checked": len(ends), "episodes_without_calls": without_calls,
            "off": pm.share(len(off), len(ends)), "off_examples": off[:5], "nonfinite_g_episodes": nonfinite,
            "end_replayed_vs_ne_m": _spread(ends), "start_g_vs_info_geodesic_m": _spread(starts)}


def block(rows, strata, draws: int, seed: int) -> dict:
    per_scene = Counter(scene_key(r["scene"]) for r in rows)
    sizes = sorted(per_scene.values())
    return {
        "episodes": len(rows),
        "scenes": len(per_scene),
        "episodes_per_scene": (
            {"min": sizes[0], "median": statistics.median(sizes), "max": sizes[-1]} if sizes else None),
        "point": pm.summarize(rows),
        "cluster_bootstrap": cluster_bootstrap(strata, draws, seed),
        "replay": replay_check(rows),
    }


def load_overlap_keys(path: str) -> set[tuple[str, int]]:
    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise InputError(f"--overlap-keys {path}: wants a JSON list of [scene, episode_id] or objects")
    keys = set()
    for item in data:
        if isinstance(item, dict):
            scene = item.get("scene_id", item.get("scene"))
            eid = item.get("episode_id")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            scene, eid = item
        else:
            scene = eid = None
        if scene is None or eid is None:
            raise InputError(f"--overlap-keys {path}: cannot read {item!r}")
        keys.add((scene_key(scene), int(eid)))
    return keys


def overlap_report(halves: dict[str, list[dict]], keys: set) -> dict:
    out = {}
    pooled = [r for rows in halves.values() for r in rows]
    for label, rows in [("pooled", pooled), *halves.items()]:
        inside = [r for r in rows if episode_key(r) in keys]
        outside = [r for r in rows if episode_key(r) not in keys]
        out[label] = {"share": pm.share(len(inside), len(rows)),
                      "overlapping": headline(inside), "not_overlapping": headline(outside)}
    return out


def check_progress(label: str, rows, patterns, seed: int) -> dict:
    """The client's own progress rows: one per episode, no step-budget STOP, protocol seed ``seed``."""
    prog = {(scene_key(k[0]), int(k[1])): v for k, v in pm.load_progress(patterns).items()}
    if not prog:
        raise InputError(f"--progress {label}: no rows in {patterns}")
    missing = [episode_key(r) for r in rows if episode_key(r) not in prog]
    if missing:
        raise InputError(f"--progress {label}: {len(missing)} episodes have no progress row, e.g. {missing[:3]}")
    used = [prog[episode_key(r)] for r in rows]
    budget = [episode_key(r) for r, p in zip(rows, used) if p.get("stop_at_step_budget") or p.get("step_budget_stop")]
    if budget:
        raise InputError(f"--progress {label}: {len(budget)} episodes ran with the step-budget STOP, which Phase-0 "
                         f"runs without, e.g. {budget[:3]}")
    seeds = sorted({p.get("rpc_protocol_seed") for p in used}, key=str)
    if seeds != [seed]:
        raise InputError(f"--progress {label}: protocol seed(s) {seeds}, Phase-0 runs seed {seed}")
    adapters = sorted({str(p.get("rpc_model_version")) for p in used if LORA_TAG in str(p.get("rpc_model_version"))})
    if adapters:
        raise InputError(f"--progress {label}: rows served with a System2 LoRA, Phase-0 is the base: {adapters[:2]}")
    return {"rows_checked": len(used), "protocol_seed": seed, "step_budget_stop": False,
            "rpc_policy_fingerprints": sorted({str(p.get("rpc_policy_fingerprint")) for p in used}),
            "rpc_model_versions": sorted({str(p.get("rpc_model_version")) for p in used}),
            "rpc_servers": len({str(p.get("rpc_server")) for p in used})}


def build_report(args) -> dict:
    decomp = labelled(args.decomp, "--decomp")
    expect = labelled_int(args.expect, "--expect")
    limit = labelled_int(args.limit, "--limit")
    shards = {}
    for label, items in labelled(args.shards, "--shards").items():
        if len(items) != 1:
            raise InputError(f"--shards {label}: given {len(items)} times, write one spec like 20-120,130")
        shards[label] = items[0]
    progress = labelled(args.progress, "--progress")
    for what, given in (("--expect", expect), ("--limit", limit), ("--shards", shards), ("--progress", progress)):
        unknown = sorted(set(given) - set(decomp))
        if unknown:
            raise InputError(f"{what}: labels {unknown} are not --decomp labels {sorted(decomp)}")
    unordered = sorted(set(limit) - set(shards))
    if unordered:
        raise InputError(f"--limit needs --shards for the same half (the launcher's order): {unordered}")
    if shards and args.cohort is None:
        raise InputError("--shards needs --cohort")
    cohort = load_cohort(args.cohort) if shards else None

    halves: dict[str, list[dict]] = {}
    inputs = {}
    seen: dict[tuple[str, int], str] = {}
    for label, paths in decomp.items():
        raw = []
        for path in paths:
            raw.extend(load_decomp(path))
        keys = [episode_key(r) for r in raw]
        dup = [k for k, c in Counter(keys).items() if c > 1]
        if dup:
            raise InputError(f"--decomp {label}: {len(dup)} episodes appear twice, e.g. {dup[:3]}")
        target, parts = target_keys(cohort, shards[label], limit.get(label)) if label in shards else (None, None)
        try:
            rows = select(raw, target)
        except InputError as exc:
            raise InputError(f"--decomp {label}: {exc}") from None
        if label in expect and len(rows) != expect[label]:
            raise InputError(f"--decomp {label}: {len(rows)} episodes, --expect {expect[label]}")
        for r in rows:
            k = episode_key(r)
            if k in seen:
                raise InputError(f"episode {k} is in both {seen[k]} and {label}")
            seen[k] = label
        halves[label] = rows
        inputs[label] = {"decomp": [{"path": p, "sha256": sha256(p)} for p in paths], "rows": len(raw),
                         "kept": len(rows), "rows_left_out": len(raw) - len(rows), "shards": shards.get(label),
                         "cohort_parts": parts, "limit": limit.get(label), "expect": expect.get(label)}
    scene_halves: dict[str, set] = {}
    for label, rows in halves.items():
        for r in rows:
            scene_halves.setdefault(scene_key(r["scene"]), set()).add(label)
    shared = sorted(s for s, labels in scene_halves.items() if len(labels) > 1)
    if shared:
        raise InputError(f"scenes in more than one half (held-out and training scenes are disjoint): {shared[:5]}")

    matrices = {label: scene_matrix(rows)[1] for label, rows in halves.items()}
    pooled_rows = [r for rows in halves.values() for r in rows]
    pooled = block(pooled_rows, list(matrices.values()), args.bootstrap_draws, args.bootstrap_seed)
    replay = pooled["replay"]
    off = replay["off"]
    replay_gate = "skipped (--no-replay-gate)" if args.no_replay_gate else "passed"
    if not args.no_replay_gate and off["n"] and Fraction(off["k"], off["n"]) > REPLAY_MAX_OFF:
        end = replay["end_replayed_vs_ne_m"]
        raise InputError(
            f"replay gate: {off['k']} of {off['n']} episodes' replayed end distance is off the client's ne by more "
            f"than {REPLAY_TOL_M} m or has a non-finite geodesic (gate {float(REPLAY_MAX_OFF):.0%}; MP3D 0 of 1,839): "
            f"median {end['median']} m, max {end['max']} m, e.g. {replay['off_examples'][:3]}.  probe_metrics did not "
            "replay the walk the client made (navmesh, start pose or heading), so B is not read")
    point = pooled["point"]
    sr, b, e7 = exact_pct(point["sr"]), exact_pct(point["B"]), exact_pct(point["E7"])
    decision = decide(sr, b, e7)
    decision["on"] = {name: None if v is None else round(float(v), 4) for name, v in (("sr", sr), ("B", b), ("E7", e7))}
    decision["counts"] = {name: [point[name]["k"], point[name]["n"]] for name in ("sr", "B", "E7")}
    return {
        "schema": SCHEMA,
        "criteria": CRITERIA,
        "decision": decision,
        "inputs": inputs,
        "cohort": None if cohort is None else {"dir": args.cohort,
                                               "manifest_sha256": sha256(os.path.join(args.cohort, "manifest.json"))},
        "bootstrap": {"unit": "scene", "strata": list(halves), "draws": args.bootstrap_draws,
                      "seed": args.bootstrap_seed, "interval": "percentile 95%"},
        "replay_gate": {"gate": replay_gate, "tolerance_m": REPLAY_TOL_M, "max_off_share": float(REPLAY_MAX_OFF),
                        "rule": "pooled episodes off by > tolerance or with a non-finite geodesic <= max_off_share"},
        "pooled": pooled,
        "halves": {label: block(rows, [matrices[label]], args.bootstrap_draws, args.bootstrap_seed)
                   for label, rows in halves.items()},
        "protocol": {label: check_progress(label, halves[label], progress[label], args.seed) for label in progress}
        or None,
        "overlap": None if args.overlap_keys is None else {
            "keys_file": args.overlap_keys, "sha256": sha256(args.overlap_keys),
            "by_set": overlap_report(halves, load_overlap_keys(args.overlap_keys))},
    }


def _cell(blk: dict, name: str) -> str:
    share = blk["point"][name]
    if share["pct"] is None:
        return "n/a"
    lo, hi = blk["cluster_bootstrap"][name]["ci95"]
    interval = "n/a" if lo is None else f"{lo:.2f}, {hi:.2f}"
    return f"{share['pct']:.2f} [{interval}] ({share['k']}/{share['n']})"


def _pct(share: dict) -> str:
    pct = "n/a" if share["pct"] is None else f"{share['pct']}%"
    return f"{pct} ({share['k']}/{share['n']})"


def markdown(report: dict) -> str:
    d = report["decision"]
    on = d["on"]
    replay, gate = report["pooled"]["replay"], report["replay_gate"]
    end, start = replay["end_replayed_vs_ne_m"], replay["start_g_vs_info_geodesic_m"]
    lines = [f"# EXP-28 Phase-0: {d['verdict'] or 'UNDECIDABLE'} (tier {d['tier']})", "",
             f"Pooled SR {on['sr']}%, B {on['B']}%, E7 {on['E7']}% -> {'; '.join(d['reasons'])}.", "",
             f"Replay gate {gate['gate']}: {_pct(replay['off'])} of the episodes' replayed end distance is off the "
             f"client's ne by > {gate['tolerance_m']} m or has a non-finite geodesic (gate "
             f"{100 * gate['max_off_share']:g}%, MP3D 0/1839); |end - ne| median {end['median']} m, p90 {end['p90']} "
             f"m, max {end['max']} m; start g vs info.geodesic_distance median {start['median']} m, max {start['max']}"
             " m.", "",
             "| set | episodes | scenes | SR | OS | B | E7 |", "|---|---|---|---|---|---|---|"]
    for label, blk in [("pooled", report["pooled"]), *report["halves"].items()]:
        lines.append(f"| {label} | {blk['episodes']} | {blk['scenes']} | "
                     + " | ".join(_cell(blk, n) for n in ("sr", "os", "B", "E7")) + " |")
    picked = [f"{label} shards {entry['shards']}" + (f" first {entry['limit']}" if entry["limit"] is not None else "")
              + f" ({', '.join(entry['cohort_parts'])})"
              for label, entry in report["inputs"].items() if entry["shards"] is not None]
    if picked:
        lines += ["", "Halves from the cohort: " + "; ".join(picked) + "."]
    by_type = report["pooled"]["point"]["by_instruction_type"]
    lines += ["", "Pooled B by instruction start (reported only): " + ", ".join(
        f"{t} {_pct(by_type[t]['B'])}" for t in ("turn", "other")) + "."]
    left_out = {label: entry["rows_left_out"] for label, entry in report["inputs"].items() if entry["rows_left_out"]}
    if left_out:
        lines += ["", "Decomposed episodes outside the halves' shards or limits, left out: "
                  + ", ".join(f"{label} {n}" for label, n in left_out.items()) + "."]
    bs = report["bootstrap"]
    lines += ["", f"Intervals: scene-cluster bootstrap, {bs['interval']}, {bs['draws']} draws, seed {bs['seed']}, "
              f"scenes resampled within each half ({', '.join(bs['strata'])}); reported only, the verdict is on the "
              "pooled point estimates."]
    if report["overlap"] is not None:
        parts = [f"{label} {_pct(e['share'])}" for label, e in report["overlap"]["by_set"].items()]
        lines += ["", "Episodes InternData also has: " + ", ".join(parts) + "."]
    checked = report["protocol"] or {}
    if checked:
        lines += ["", "Protocol checked on progress rows: " + ", ".join(
            f"{label} {entry['rows_checked']} rows, seed {entry['protocol_seed']}, no step-budget STOP, no LoRA"
            for label, entry in checked.items()) + "."]
    unchecked = [label for label in report["halves"] if label not in checked]
    if unchecked:
        lines += ["", f"Protocol NOT checked (no --progress): {', '.join(unchecked)}."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decomp", nargs="+", required=True, metavar="LABEL=PATH",
                    help="probe_metrics <name>.decomp.json per platform half; a label may repeat")
    ap.add_argument("--expect", nargs="+", default=[], metavar="LABEL=N", help="required episode count of a half")
    ap.add_argument("--cohort", default=None, help="cohort directory (manifest.json, shard_NNN.json) for --shards")
    ap.add_argument("--shards", nargs="+", default=[], metavar="LABEL=SPEC",
                    help="a half is exactly these shards' episodes (PPA_COLLECT_SHARDS syntax, launcher order)")
    ap.add_argument("--limit", nargs="+", default=[], metavar="LABEL=N",
                    help="keep the first N episodes of the half's --shards")
    ap.add_argument("--progress", nargs="+", default=[], metavar="LABEL=GLOB",
                    help="progress.json glob(s) of a half, for the protocol check")
    ap.add_argument("--seed", type=int, default=42, help="protocol seed the progress rows must carry")
    ap.add_argument("--overlap-keys", default=None, help="JSON list of the episodes InternData also has")
    ap.add_argument("--bootstrap-draws", type=int, default=DRAWS)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--no-replay-gate", action="store_true",
                    help="report even when probe_metrics' replay does not reproduce the client (only after finding why)")
    ap.add_argument("--name", default="phase0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        report = build_report(args)
        text = markdown(report)
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"{args.name}.phase0.json"), "w") as fh:
            json.dump(report, fh, indent=1)
        with open(os.path.join(args.out, f"{args.name}.phase0.md"), "w") as fh:
            fh.write(text)
    except InputError as exc:
        print(f"phase0_report: {exc}", file=sys.stderr)
        return 2
    except Exception:  # any failure must not exit 1, which reads as NO-GO
        traceback.print_exc()
        return 2
    print(text, end="")
    verdict = report["decision"]["verdict"]
    return 0 if verdict == "GO" else 1 if verdict == "NO-GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())

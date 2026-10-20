#!/usr/bin/env python3
"""EXP-32 offline gates R (does the RFT target transfer to held-out scenes) and M (was it learnt).

The ledger's EXP-32 第二部分 F (pre-registered before any part-1 reading) fixes both.  Every state is one
recorded System2 call; its prompt is the one the server built for it (``inference_messages``: the recorded
JPEG bytes, the recorded history), decoded greedily the way the server decodes (``greedy_native_decision``)
on one card, once with the candidate adapter on and once inside ``adapter_off`` (the base model).  "Agrees"
is ``scripts/exp30/probe_judge.same_answer`` with the recorded (sampled) answer: same decision class, a pixel
goal within 50 px on both axes, a turn the same way.

======  ======================================================================================  ============
R       states: the calls RF2/RF3 keep (``build_rft_stream.rft_calls``: call < 48, recorded d >    D_R >= 0.05
        3 m, pixel or turn, no reversal) of the *successful* trajectories of B3's held-out         and the 95%
        sampled collection (``heldout_s{42,1337}``, plus ``s2024`` when added).  Divergent: the    lower bound
        base's answer does not agree with the recorded one.  D_R = share of divergent states on    > 0
        which the candidate agrees (the base is 0 there by construction); episode-cluster
        bootstrap, 20,000 draws.  Fewer than 150 divergent states: B3 adds a seed-2024 pass once (score
        is invalid until a source of seed 2024 is in R); still fewer with it: R is report-only, M gates.
R_fail  report-only: the same reading on the *failed* trajectories' states (``--failed``, seeded
        (32) down to as many states as R has); also D_R - R_fail.
M       states: the D3 parity rows (910B, npu, base, adapters off) that are RFT samples of the    D_M >= 0.15
        training stream and whose npu replay does not agree with the recorded answer.  D_M =       (gates only
        agreeing share of the candidate minus that of the base, both decoded here.                 when R is
                                                                                                   report-only)
======  ======================================================================================  ============

Shares are compared as whole-count fractions, so a value exactly on its line passes.  M always decides the
retry type after an R failure (ledger F's table: D_M < 0.15 -> train the stream twice; >= 0.15 -> kill).

Subcommands::

    select-r  CPU.  ``--source NAME=ROOT:SEED`` (repeatable; SEED is the System1 protocol seed the root was
              collected with, one of B3's 42 / 1337 / 2024).  Every record must be of a scene in
              ``--holdout-scenes-file`` and served by the base model (``served_adapter`` None), or the
              selection is refused; a record whose calls carry another protocol seed is left out and counted,
              more than 1% of a root's records stops it (A4).  Writes ``--output`` (states, JSON lines) and
              ``<output stem>.selection.json``.
    select-m  CPU.  ``--d3-calls`` = D3's ``parity_g0.calls.jsonl`` (``parity_g0.json`` beside it must say
              npu for both the records and the device), ``--rft-stream`` = the RFT stream directory
              (``build_rft_stream.py``: ``keys.tsv`` and ``rft_records.txt``, both checked against its
              manifest's sha256).  Each D3 record is the RFT trajectory it copies as ``aud.py dose`` maps it
              (``aud.map_records``: the same path; the one trajectory with its ``<scene dir>/<file>``; of
              several, the one whose tag (p1, r0s, p2, p3) names a directory on the path -- stage D3's records
              as ``<tag>/<scene dir>/<file>``; or ``--record-map``), so M's states are the rows ``dose`` counts
              as diverged RFT rows.  A record no trajectory fits, a row whose recorded answer is not its
              original's, or an RFT call the stream records as answered by cuda stops the selection.
    decode    GPU.  Decodes every state of ``--r-states`` / ``--m-states`` (deduplicated by uid) with
              ``--adapter`` on and off.  ``--shard i/n`` keeps the states whose uid hashes to i (one process
              per card); ``<output stem>.decode.shardIIofNN.jsonl`` is the resume log, and rows of another
              adapter in it are refused.  ``--dry-run`` counts the states without a model.
    score     CPU.  Joins the states with every decode shard beside ``--output``, writes ``--output``
              (``rft_gate.json``) and ``<output stem>.calls.jsonl``.  Exit 0: the gate passes; 1: it fails;
              2: the measurement is invalid (states not decoded, rows of several adapters or not of
              ``--adapter``, select-r's selection.json missing, R under 150 divergent states without the
              seed-2024 pass (``gate.needs`` = "heldout_s2024"), R report-only without M states).

``rft_gate.json`` for ``hm3d_route_gate.py verdict --rules exp32``: ``valid``; ``gate`` = {``uses``: "R", "M"
or None, ``pass``} (the R group of the verdict); ``R`` = {``pass`` (None under 150 divergent states),
``report_only``, ``needs_seed_2024``, ``value`` = D_R, ``divergent``, ``ci95_episode_bootstrap``}; ``M`` =
{``measured``, ``learnt`` (D_M >= 0.15, None when not measured), ``value`` = D_M}; ``R_fail`` and
``delta_R_minus_R_fail`` are report-only.

    G=/workspace/exp28/gates/x32r1_a1/rgate
    python scripts/exp32/rft_gate.py select-r --source s42=/workspace/exp32/heldout_s42/records:42 \\
        --source s1337=/workspace/exp32/heldout_s1337/records:1337 \\
        --holdout-scenes-file /workspace/exp28/episodes/splits/heldout_scenes.txt --failed --output $G/r_states.jsonl
    python scripts/exp32/rft_gate.py select-m --d3-calls <D3 dir>/parity_g0.calls.jsonl \\
        --rft-stream /workspace/exp32/stream_rft --output $G/m_states.jsonl
    python scripts/exp32/rft_gate.py decode --r-states $G/r_states.jsonl --m-states $G/m_states.jsonl \\
        --adapter <adapter dir> --model-path /workspace/InternNav_Model --output $G/rft_gate.json --shard 0/1 \\
        --device cuda
    python scripts/exp32/rft_gate.py score --r-states $G/r_states.jsonl --m-states $G/m_states.jsonl \\
        --adapter <adapter dir> --output $G/rft_gate.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.exp28.hm3d_route_gate import (  # noqa: E402
    GateInvalid,
    _arm,
    _num,
    _rate,
    decode_path,
    episode_bootstrap,
    parse_shard,
    read_decode_rows,
    served_adapter,
    shard_of,
)
from scripts.exp30.probe_judge import same_answer  # noqa: E402
from scripts.exp32 import aud  # noqa: E402
from scripts.exp32 import build_rft_stream as brs  # noqa: E402
from scripts.tools.build_route_stream import read_scene_list, scene_key  # noqa: E402
from src.data.s2_records import EpisodeRecord, episode_key, iter_record_paths  # noqa: E402
from src.data.s2_sft_dataset import decision_class, record_device, recorded_turns  # noqa: E402
from src.data.s2_stream import MANIFEST_NAME  # noqa: E402

LOGGER = logging.getLogger("rft_gate")

SCHEMA = "heatmapvln-exp32-rft-gate-v1"
STATE_SCHEMA = "heatmapvln-exp32-rft-gate-state-v1"
DECODE_SCHEMA = "heatmapvln-exp32-rft-gate-decode-v1"

# Ledger EXP-32 第二部分 F and B3, written before any part-1 reading.
R_MIN_DELTA = Fraction(5, 100)
R_MIN_DIVERGENT = 150  # fewer: B3 adds seed 2024 once; still fewer: R is report-only and M gates
R_EXTRA_SEED = 2024
M_MIN_DELTA = Fraction(15, 100)
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 0  # episode_bootstrap's own default
HELDOUT_SEEDS = (42, 1337, R_EXTRA_SEED)  # B3's System1 seeds
FAIL_SEED = 32  # R_fail: seeded down to R's size
D3_DEVICE = "npu"  # D3 replays npu-answered calls on the 910B
GROUPS_OF = {"r": ("R", "R_fail"), "m": ("M",)}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state(group: str, source: str, record: EpisodeRecord, call_index: int, turns: Sequence[str],
           distance: float | None, device: str, **extra: Any) -> dict[str, Any]:
    path = str(record.path)
    return {"schema": STATE_SCHEMA, "group": group, "source": source, "uid": f"{path}#{int(call_index)}",
            "record_path": path, "record_key": episode_key(record.scene_id, record.episode_id),
            "scene_id": record.scene_id, "episode_id": record.episode_id, "call_index": int(call_index),
            "distance_to_goal": distance, "recorded": [str(t) for t in turns],
            "decision": decision_class(str(turns[-1])), "record_device": device, **extra}


def _write_states(output: Path, states: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.write_text("".join(json.dumps(s, sort_keys=True, ensure_ascii=False) + "\n" for s in states), encoding="utf-8")
    tmp.replace(output)
    output.with_suffix(".selection.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# select-r: B3's held-out sampled trajectories
# --------------------------------------------------------------------------- #
def parse_heldout_source(text: str) -> tuple[str, str, int]:
    """``NAME=ROOT:SEED`` -> (name, root, seed); the seed must be one of B3's."""
    name, sep, rest = str(text).partition("=")
    root, colon, seed = rest.rpartition(":")
    if not sep or not colon or not root or not name or not name.replace("_", "").isalnum():
        raise SystemExit(f"--source takes NAME=ROOT:SEED; got {text!r}")
    try:
        value = int(seed)
    except ValueError:
        raise SystemExit(f"--source {text!r}: the seed {seed!r} is not an integer") from None
    if value not in HELDOUT_SEEDS:
        raise SystemExit(f"--source {text!r}: B3's held-out sampling runs System1 seeds {list(HELDOUT_SEEDS)}")
    return name, root, value


def fail_rank(seed: int, uid: str) -> str:
    return hashlib.sha256(f"exp32-rfail:{seed}:{uid}".encode("utf-8")).hexdigest()


def select_r(
    sources: Sequence[tuple[str, str, int]],
    held: set[str],
    *,
    first_calls: int = brs.DEFAULT_FIRST_CALLS,
    min_distance: float = brs.DEFAULT_MIN_DISTANCE,
    failed: bool = False,
    fail_seed: int = FAIL_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """R's states (RF2/RF3 calls of the successful trajectories) and, with ``failed``, R_fail's."""
    names = [name for name, _, _ in sources]
    if len(set(names)) != len(names):
        raise SystemExit(f"--source names a source twice: {names}")
    if not held:
        raise SystemExit("no held-out scenes: R reads B3's held-out sampled trajectories only")
    success: list[dict[str, Any]] = []
    fail_pool: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for name, root, seed in sources:
        root = str(Path(root).resolve())
        if not Path(root).is_dir():
            raise SystemExit(f"source {name} {root} is not a directory")
        counts: Counter = Counter()
        calls = {"success": Counter(), "failed": Counter()}
        seeds_seen: Counter = Counter()
        outside: Counter = Counter()
        devices: Counter = Counter()
        for path in iter_record_paths(root):
            record = EpisodeRecord(path)
            scene = scene_key(record.scene_id)
            if scene not in held or scene_key(path.parent.name) not in held:
                outside[scene] += 1
                continue
            adapter = served_adapter(record.episode)
            if adapter is not None:
                raise SystemExit(f"{path} was served with System2 LoRA {adapter}: R reads the base model's samples")
            counts["records"] += 1
            seeds = brs.call_seeds(record.calls)
            for value in seeds:
                seeds_seen[str(value)] += 1
            if seeds - {seed}:
                counts["seed_mismatch"] += 1
                continue
            won = brs.episode_success(record.episode)
            counts["success" if won else "failed"] += 1
            kept, why = brs.rft_calls(record, first_calls=first_calls, min_distance=min_distance)
            calls["success" if won else "failed"].update(why)
            device = record_device(record)
            devices[device] += 1
            for call in kept:
                state = _state("R" if won else "R_fail", name, record, call.call_index, call.turns,
                               call.distance_to_goal, device)
                (success if won else fail_pool).append(state)
        if outside:
            raise SystemExit(f"source {name} {root} holds records of scenes that are not held out (records per scene): "
                             f"{dict(sorted(outside.items())[:20])} ({len(outside)} scenes)")
        if not counts["records"]:
            raise SystemExit(f"source {name} {root} holds no committed record of a held-out scene")
        if not brs.seed_share_ok(counts["seed_mismatch"], counts["records"]):
            raise SystemExit(f"source {name} {root}: {counts['seed_mismatch']} of {counts['records']} records carry "
                             f"another System1 seed than {seed} (more than 1%, ledger EXP-32 A4; seeds seen "
                             f"{dict(seeds_seen)})")
        report[name] = {"root": root, "protocol_seed": seed, "records": dict(counts),
                        "calls_success": dict(calls["success"]), "calls_failed": dict(calls["failed"]),
                        "devices": dict(devices), "seeds_seen": dict(seeds_seen)}
    uids = [s["uid"] for s in success + fail_pool]
    if len(set(uids)) != len(uids):
        raise SystemExit("two sources hold the same record (a root given twice?)")
    success.sort(key=lambda s: s["uid"])
    chosen_fail: list[dict[str, Any]] = []
    if failed:
        ranked = sorted(fail_pool, key=lambda s: fail_rank(fail_seed, s["uid"]))
        chosen_fail = sorted(ranked[: len(success)], key=lambda s: s["uid"])
    meta = {
        "schema": STATE_SCHEMA,
        "selection": "select-r",
        "rule": "ledger EXP-32 第二部分 F, R: RF2/RF3 calls (build_rft_stream.rft_calls) of B3's held-out sampled "
                "trajectories; R = successful ones, R_fail = failed ones seeded down to R's size",
        "sources": report,
        "holdout_scenes": sorted(held),
        "knobs": {"first_calls": first_calls, "min_distance": min_distance, "failed": failed, "fail_seed": fail_seed,
                  "fail_rank": f"sha256('exp32-rfail:{fail_seed}:' + uid)"},
        "states": {"R": len(success), "R_fail": len(chosen_fail), "R_fail_pool": len(fail_pool)},
        "episodes": {"R": len({s["record_key"] for s in success}),
                     "R_fail": len({s["record_key"] for s in chosen_fail})},
    }
    return success + chosen_fail, meta


# --------------------------------------------------------------------------- #
# select-m: D3's divergent rows that are RFT samples of the stream
# --------------------------------------------------------------------------- #
def read_rft_stream(stream_dir: str | Path) -> tuple[str, str, dict[str, Any]]:
    """The texts of a ``build_rft_stream.py`` stream's ``keys.tsv`` and ``rft_records.txt``, checked against its
    manifest."""
    stream_dir = Path(stream_dir)
    manifest = json.loads((stream_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("builder") != brs.BUILDER:
        raise SystemExit(f"{stream_dir} was built by {manifest.get('builder')!r}, not {brs.BUILDER!r}")
    selection = manifest.get("selection") or {}
    texts = {}
    for name, field in ((brs.KEYS_TSV, "keys_tsv_sha256"), (brs.RECORDS_LIST, "rft_records_sha256")):
        data = (stream_dir / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != selection.get(field):
            raise SystemExit(f"{stream_dir / name} is not the file the stream's manifest names ({field})")
        texts[name] = data.decode("utf-8")
    info = {"dir": str(stream_dir.resolve()), "keys_tsv_sha256": selection.get("keys_tsv_sha256"),
            "rft_records_sha256": selection.get("rft_records_sha256"),
            "selected_keys_sha256": selection.get("selected_keys_sha256"),
            "n_rft": (selection.get("composition") or {}).get("n_rft")}
    return texts[brs.KEYS_TSV], texts[brs.RECORDS_LIST], info


def select_m(
    d3_rows: Sequence[Mapping[str, Any]],
    keys_text: str,
    records_text: str,
    record_map: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """M's states: D3 rows that are RFT samples of the stream and whose npu replay does not agree with the record.

    The rows are read as ``aud.py dose`` reads them (its ``map_records``: each D3 record is the RFT trajectory it
    copies -- the same path, the one ``<scene dir>/<file>``, or of several the one whose tag names a directory on
    the path, or ``record_map``'s; every row's recorded answer must be its original's), so M's states are the
    rows ``dose`` counts as diverged RFT rows."""
    keys, twice = aud.read_keys_tsv(keys_text)
    if twice:
        raise SystemExit(f"{brs.KEYS_TSV} lists keys twice: {twice[:5]}")
    trajectories = aud.read_rft_records(records_text)
    in_stream = {(r.record_path, r.call_index): r for r in keys.values() if r.category.startswith("rft_")}
    split = []
    for row in d3_rows:
        path, _, call = str(row["uid"]).rpartition("#")
        _, episode = aud.episode_of(path)
        if str(row["key"]).rpartition("#")[2] != call or not str(row["key"]).endswith(f"_{episode:06d}#{call}"):
            raise SystemExit(f"D3 row {row['key']!r} does not name its record {row['uid']!r}")
        split.append((path, int(call)))
    mapping = aud.map_records((path for path, _ in split), trajectories, record_map or {})

    counts: Counter = Counter()
    states: dict[str, dict[str, Any]] = {}
    records: dict[str, tuple[EpisodeRecord, dict[int, Mapping[str, Any]]]] = {}
    for row, (path, call_index) in zip(d3_rows, split):
        counts["d3_rows"] += 1
        original = mapping[path]
        if original not in records:
            record = EpisodeRecord(original)
            records[original] = (record, {int(c["call_index"]): c for c in record.calls})
        record, calls = records[original]
        call = calls.get(call_index)
        recorded = list(recorded_turns(call)) if call is not None else None
        if recorded is None or [str(t) for t in row["recorded"]] != recorded:
            raise SystemExit(f"D3's recorded answer of {row['uid']} is {row['recorded']!r}, {original} call "
                             f"{call_index} holds {recorded!r}: D3's records are not the stream's (wrong mapping?)")
        sample = in_stream.get((original, call_index))
        if sample is None:
            counts["call_not_an_rft_sample"] += 1  # RF2/RF3 left it out
            continue
        if sample.device != D3_DEVICE:
            raise SystemExit(f"D3 replayed {row['uid']}, which the stream ({sample.key}) records as answered by "
                             f"{sample.device}: D3's records are not the stream's")
        counts["rft_samples"] += 1
        replay = [str(t) for t in row["replay"]]
        if same_answer(replay, recorded):
            counts["replay_agrees"] += 1
            continue
        distance = call.get("distance_to_goal")
        state = _state("M", aud.category_tag(sample.category), record, call_index, recorded,
                       float(distance) if distance is not None else None, record_device(record),
                       rft_key=sample.key, d3_uid=str(row["uid"]), npu_replay=replay,
                       npu_replay_exact=bool(row.get("exact")))
        if state["uid"] in states:
            raise SystemExit(f"two D3 rows are copies of {state['uid']}")
        states[state["uid"]] = state
        counts["divergent"] += 1
    chosen = [states[uid] for uid in sorted(states)]
    meta = {
        "schema": STATE_SCHEMA,
        "selection": "select-m",
        "rule": "ledger EXP-32 第二部分 F, M: D3's rows (npu base replay) that are RFT samples of the training stream "
                "and whose replay does not same_answer the recorded answer (rows mapped as aud.py dose maps them)",
        "counts": dict(counts),
        "records": {"d3": len(mapping), "rft_trajectories": len(set(mapping.values())),
                    "by_tag": dict(Counter(trajectories[o][1] for o in set(mapping.values())))},
        "states": {"M": len(chosen)},
        "episodes": {"M": len({s["record_key"] for s in chosen})},
        "divergence_rate_of_rft_rows": _rate(counts["divergent"], counts["rft_samples"]),
    }
    return chosen, meta


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# States, decode
# --------------------------------------------------------------------------- #
def read_states(r_states: str | None, m_states: str | None) -> list[dict[str, Any]]:
    """Both files' states; each file holds only its own groups, and a (group, uid) once (ValueError if not)."""
    states: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for kind, path in (("r", r_states), ("m", m_states)):
        if not path:
            continue
        for state in _read_jsonl(path):
            if state.get("schema") != STATE_SCHEMA or state.get("group") not in GROUPS_OF[kind]:
                raise ValueError(f"{path}: a row of schema {state.get('schema')!r} group {state.get('group')!r} is "
                                 f"not a {'select-r' if kind == 'r' else 'select-m'} state")
            ident = (state["group"], state["uid"])
            if ident in seen:
                raise ValueError(f"{path}: state {ident} is listed twice")
            seen.add(ident)
            states.append(state)
    return states


def decode(args: argparse.Namespace) -> int:
    started = time.time()
    try:
        states = read_states(args.r_states, args.m_states)
    except ValueError as err:
        raise SystemExit(str(err)) from None
    unique: dict[str, dict[str, Any]] = {}
    for state in states:
        unique.setdefault(state["uid"], state)
    index, count = parse_shard(args.shard)
    mine = [unique[uid] for uid in sorted(unique) if shard_of(uid, count) == index]
    output = Path(args.output)
    rows_path = decode_path(output, index, count)
    groups = dict(Counter(s["group"] for s in states))
    LOGGER.info("%d states (%s), %d calls, %d in shard %d/%d", len(states), groups, len(unique), len(mine), index,
                count)
    if args.dry_run:
        print(json.dumps({"states": groups, "calls": len(unique), "shard": len(mine)}, sort_keys=True), flush=True)
        return 0

    import torch

    from scripts.train_system2_lora import (
        adapter_off,
        greedy_native_decision,
        init_runtime,
        load_adapter,
        load_system2,
        seed_everything,
        sha256_file,
    )
    from src.data.s2_sft_dataset import RecordCache, inference_messages

    output.parent.mkdir(parents=True, exist_ok=True)
    adapter_sha = sha256_file(Path(args.adapter) / "adapter_model.safetensors")
    done = read_decode_rows([rows_path]) if rows_path.is_file() else {}
    others = {r.get("adapter_sha256") for r in done.values()} - {adapter_sha}
    if others:
        raise SystemExit(f"{rows_path} holds rows of another adapter {sorted(map(str, others))}; use another --output")
    todo = [s for s in mine if not ("on" in done.get(s["uid"], {}) and "off" in done.get(s["uid"], {}))]
    LOGGER.info("%d already decoded, %d to go", len(mine) - len(todo), len(todo))
    if todo:
        rt = init_runtime(args.device)
        seed_everything(args.seed, rt.device)
        model, processor = load_system2(args.model_path, rt.device)
        model = load_adapter(model, args.adapter)
        model.eval()
        records = RecordCache(8)
        size = tuple(args.default_vlm_image_size)
        with open(rows_path, "a", encoding="utf-8") as handle:
            for i, state in enumerate(todo, 1):
                t0 = time.time()
                record, calls = records.get(state["record_path"])
                call = calls[int(state["call_index"])]
                if list(recorded_turns(call)) != list(state["recorded"]):
                    raise SystemExit(f"{state['uid']}: the record answers {list(recorded_turns(call))!r}, the state "
                                     f"says {state['recorded']!r}; select again")
                messages, images, lookdown = inference_messages(record, call, default_vlm_image_size=size)
                with torch.no_grad():
                    on = greedy_native_decision(model, processor, messages, images, lookdown, rt.device)
                    with adapter_off(model):
                        off = greedy_native_decision(model, processor, messages, images, lookdown, rt.device)
                row = {"schema": DECODE_SCHEMA, "key": state["uid"], "uid": state["uid"], "adapter_sha256": adapter_sha,
                       "device": rt.device.type, "record_device": record_device(record), "on": _arm(on),
                       "off": _arm(off), "seconds": round(time.time() - t0, 3)}
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                handle.flush()
                if i % args.log_every == 0 or i == len(todo):
                    LOGGER.info("%d/%d decoded (%.1f s/state)", i, len(todo), (time.time() - started) / i)
    print(json.dumps({"shard": f"{index}/{count}", "states": len(mine), "decoded_now": len(todo),
                      "rows": str(rows_path), "seconds": round(time.time() - started, 1)}), flush=True)
    return 0


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def join_rows(states: Sequence[Mapping[str, Any]], decoded: Mapping[str, Mapping[str, Any]]
              ) -> tuple[list[dict[str, Any]], str]:
    """States with their decoded answers and whether each agrees with the recorded answer; the adapter."""
    missing = [s["uid"] for s in states if not {"on", "off"} <= set(decoded.get(s["uid"], {}))]
    if missing:
        raise GateInvalid(f"{len(missing)} of {len(states)} states were not decoded (first: {missing[:3]})")
    adapters = {decoded[s["uid"]].get("adapter_sha256") for s in states}
    if len(adapters) != 1:
        raise GateInvalid(f"the decoded rows come from {len(adapters)} adapters: {sorted(map(str, adapters))}")
    rows = []
    for state in states:
        got = decoded[state["uid"]]
        row = dict(state)
        row.update({"on": dict(got["on"]), "off": dict(got["off"]), "device": got.get("device"),
                    "on_agree": same_answer(got["on"]["turns"], state["recorded"]),
                    "off_agree": same_answer(got["off"]["turns"], state["recorded"])})
        rows.append(row)
    return rows, str(next(iter(adapters)))


def _ci(rows: Sequence[Mapping[str, Any]], diff) -> list[float] | None:
    return episode_bootstrap(rows, diff, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)


def divergent_reading(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[Mapping[str, Any]], Fraction | None]:
    """R's reading on any rows: the divergent ones (the base disagrees with the recorded answer) and the
    candidate's agreeing share among them, as (counts, divergent rows, share)."""
    divergent = [r for r in rows if not r["off_agree"]]
    d = len(divergent)
    k = sum(1 for r in divergent if r["on_agree"])
    share = Fraction(k, d) if d else None
    return {"states": len(rows), "divergent": d, "candidate_agree": k, "value": _num(share)}, divergent, share


def r_summary(rows: Sequence[Mapping[str, Any]], *, seeds: Collection[int]) -> dict[str, Any]:
    """R: D_R >= 0.05 and the episode-cluster 95% lower bound > 0.  Under 150 divergent states R gates
    nothing: it is report-only once B3's seed-2024 pass is among ``seeds`` (the R sources' System1 seeds),
    and before that it needs that pass (``needs_seed_2024``)."""
    reading, divergent, delta = divergent_reading(rows)
    ci = _ci(divergent, lambda r: float(bool(r["on_agree"])))
    too_few = reading["divergent"] < R_MIN_DIVERGENT
    with_2024 = R_EXTRA_SEED in set(seeds)
    ok = (not too_few) and delta is not None and delta >= R_MIN_DELTA and ci is not None and ci[0] > 0
    by_source = {source: divergent_reading([r for r in rows if r["source"] == source])[0]
                 for source in sorted({r["source"] for r in rows})}
    return {
        **reading,
        "threshold": _num(R_MIN_DELTA),
        "direction": ">=",
        "report_only": too_few and with_2024,
        "needs_seed_2024": too_few and not with_2024,
        "min_divergent": R_MIN_DIVERGENT,
        "seeds": sorted(set(seeds)),
        "pass": None if too_few else bool(ok),
        "base_agree_on_divergent": 0,  # by construction
        "base_agree_share": _rate(sum(1 for r in rows if r["off_agree"]), len(rows)),
        "ci95_episode_bootstrap": ci,
        "ci_lower_gt_0": bool(ci is not None and ci[0] > 0),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "cluster": "episode (record_key)"},
        "episodes": len({r["record_key"] for r in rows}),
        "divergent_episodes": len({r["record_key"] for r in divergent}),
        "by_source": by_source,
        "divergent_by_recorded_class": dict(sorted(Counter(r["decision"] for r in divergent).items())),
        "divergent_on_class": dict(sorted(Counter(r["on"]["class"] for r in divergent).items())),
        "divergent_base_class": dict(sorted(Counter(r["off"]["class"] for r in divergent).items())),
    }


def m_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """M: D_M = candidate's agreeing share - the base's on the same states; learnt when >= 0.15."""
    n = len(rows)
    on = sum(1 for r in rows if r["on_agree"])
    off = sum(1 for r in rows if r["off_agree"])
    delta = Fraction(on - off, n) if n else None
    return {
        "measured": bool(n),
        "states": n,
        "candidate_agree": on,
        "base_agree": off,
        "candidate_rate": _rate(on, n),
        "base_rate": _rate(off, n),
        "value": _num(delta),
        "threshold": _num(M_MIN_DELTA),
        "direction": ">=",
        "learnt": None if delta is None else bool(delta >= M_MIN_DELTA),
        "ci95_episode_bootstrap": _ci(rows, lambda r: float(bool(r["on_agree"])) - float(bool(r["off_agree"]))),
        "episodes": len({r["record_key"] for r in rows}),
        "by_source": {s: {"states": sum(1 for r in rows if r["source"] == s),
                          "candidate_agree": sum(1 for r in rows if r["source"] == s and r["on_agree"]),
                          "base_agree": sum(1 for r in rows if r["source"] == s and r["off_agree"])}
                      for s in sorted({r["source"] for r in rows})},
    }


def summarize(rows: Sequence[Mapping[str, Any]], *, r_seeds: Collection[int]) -> dict[str, Any]:
    """R, R_fail, M and the gate they give: R; M when R is report-only; nothing while R needs seed 2024."""
    groups = {g: [r for r in rows if r["group"] == g] for g in ("R", "R_fail", "M")}
    r = r_summary(groups["R"], seeds=r_seeds)
    fail = divergent_reading(groups["R_fail"])[0]
    fail["measured"] = fail["value"] is not None
    m = m_summary(groups["M"])
    notes = []
    if r["pass"] is not None:
        gate = {"uses": "R", "pass": r["pass"], "why": f"{r['divergent']} divergent states (>= {R_MIN_DIVERGENT})"}
    elif r["report_only"]:
        gate = {"uses": "M", "pass": m["learnt"],
                "why": f"R is report-only: {r['divergent']} divergent states (< {R_MIN_DIVERGENT}) with B3's seed-2024 "
                       "pass"}
    else:
        gate = {"uses": None, "pass": None, "needs": "heldout_s2024",
                "why": f"R has {r['divergent']} divergent states (< {R_MIN_DIVERGENT}) without B3's seed-2024 pass: "
                       f"collect it once (System1 seed {R_EXTRA_SEED}), select-r again with it, decode and score again"}
    if not m["measured"]:
        notes.append("M not measured: the retry type after an R failure cannot be read")
    if not fail["measured"]:
        notes.append("R_fail not measured (select-r without --failed, or no divergent failed state)")
    return {
        "gate": gate,
        "R": r,
        "R_fail": fail,
        "delta_R_minus_R_fail": (round(r["value"] - fail["value"], 6)
                                 if r["value"] is not None and fail["value"] is not None else None),
        "M": m,
        "report_only": {
            "on_off_identical": _rate(sum(1 for x in rows if x["on"]["turns"] == x["off"]["turns"]), len(rows)),
            "states": dict(Counter(x["group"] for x in rows)),
        },
        "notes": notes,
    }


def measurement_problems(summary: Mapping[str, Any]) -> list[str]:
    """Why the gate cannot be read (empty: it can)."""
    if summary["gate"]["uses"] is None:
        return [summary["gate"]["why"]]
    if summary["gate"]["uses"] == "M" and not summary["M"]["measured"]:
        return ["R is report-only and M has no state: neither can gate"]
    return []


def r_source_seeds(r_states: str) -> set[int]:
    """The System1 seeds of the sources select-r read (its ``<stem>.selection.json``)."""
    meta = _selection_meta(r_states)
    if not meta or meta.get("selection") != "select-r":
        raise GateInvalid(f"{Path(r_states).with_suffix('.selection.json')} (select-r's record of its sources) "
                          "is missing")
    return {int(source["protocol_seed"]) for source in meta["sources"].values()}


def _selection_meta(path: str | None) -> Any:
    if not path:
        return None
    side = Path(path).with_suffix(".selection.json")
    return json.loads(side.read_text(encoding="utf-8")) if side.is_file() else None


def score(args: argparse.Namespace) -> int:
    started = time.time()
    output = Path(args.output)
    paths = sorted(output.parent.glob(f"{output.stem}.decode.shard*of*.jsonl"))
    report: dict[str, Any] = {"schema": SCHEMA, "valid": False}
    rows: list[dict[str, Any]] = []
    try:
        states = read_states(args.r_states, args.m_states)
        rows, adapter_sha = join_rows(states, read_decode_rows(paths))
        if args.adapter:
            expected = _sha256(Path(args.adapter) / "adapter_model.safetensors")
            if adapter_sha != expected:
                raise GateInvalid(f"the decoded rows are of adapter {adapter_sha}, --adapter is {expected}")
        report["adapter_sha256"] = adapter_sha
        summary = summarize(rows, r_seeds=r_source_seeds(args.r_states))
        report.update(summary)
        problems = measurement_problems(summary)
        if problems:
            report["invalid"] = "GateInvalid: " + "; ".join(problems)
        else:
            report["valid"] = True
    except (RuntimeError, ValueError, KeyError, OSError) as err:
        report["invalid"] = f"{type(err).__name__}: {err}"
        rows = []
    report.update({
        "selection": {"R": _selection_meta(args.r_states), "M": _selection_meta(args.m_states)},
        "inputs": {"r_states": args.r_states, "m_states": args.m_states, "adapter": args.adapter,
                   "decode_files": [str(p) for p in paths]},
        "rules": "ledger EXP-32 第二部分 F (R, M), written before any part-1 reading",
        "seconds": round(time.time() - started, 1),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output.with_suffix(".calls.jsonl"), "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.write_text(json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(output)
    if not report["valid"]:
        print(f"INVALID R/M: {report['invalid']}", flush=True)
        return 2
    r, m, gate = report["R"], report["M"], report["gate"]
    line = (f"{'PASS' if gate['pass'] else 'FAIL'} EXP-32 gate R (uses {gate['uses']}): "
            f"D_R={r['value']} ci95={r['ci95_episode_bootstrap']} divergent={r['divergent']}"
            f"{' report-only' if r['report_only'] else ''}; D_M={m['value']} learnt={m['learnt']}; "
            f"R_fail={report['R_fail']['value']}")
    print(line, flush=True)
    for note in report["notes"]:
        print(f"[rft_gate] {note}", file=sys.stderr)
    return 0 if gate["pass"] else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run_select_r(args: argparse.Namespace) -> int:
    held = {scene_key(s) for s in read_scene_list(args.holdout_scenes_file)}
    sources = [parse_heldout_source(s) for s in args.source]
    states, meta = select_r(sources, held, first_calls=args.first_calls, min_distance=args.min_distance,
                            failed=args.failed, fail_seed=args.fail_seed)
    meta["holdout_scenes_file"] = str(Path(args.holdout_scenes_file).resolve())
    _write_states(Path(args.output), states, meta)
    print(json.dumps({"states": meta["states"], "episodes": meta["episodes"], "output": args.output}, sort_keys=True),
          flush=True)
    return 0


def run_select_m(args: argparse.Namespace) -> int:
    report_path = Path(args.d3_report) if args.d3_report else Path(args.d3_calls).with_name("parity_g0.json")
    d3 = json.loads(report_path.read_text(encoding="utf-8"))
    if d3.get("record_device") != D3_DEVICE or d3.get("device") != D3_DEVICE:
        raise SystemExit(f"{report_path}: D3 replays npu calls on the 910B; this report says records "
                         f"{d3.get('record_device')!r}, device {d3.get('device')!r}")
    keys_text, records_text, stream = read_rft_stream(args.rft_stream)
    states, meta = select_m(_read_jsonl(args.d3_calls), keys_text, records_text, aud.read_record_map(args.record_map))
    meta.update({"d3_calls": str(Path(args.d3_calls).resolve()), "d3_calls_sha256": _sha256(args.d3_calls),
                 "d3_report": str(report_path.resolve()), "d3_exact_rate": d3.get("exact_rate"), "d3_n": d3.get("n"),
                 "record_map": args.record_map or None, "rft_stream": stream})
    _write_states(Path(args.output), states, meta)
    print(json.dumps({"states": meta["states"], "counts": meta["counts"], "records": meta["records"],
                      "output": args.output}, sort_keys=True), flush=True)
    return 0


def _state_files(parser: argparse.ArgumentParser, *, r_required: bool) -> None:
    parser.add_argument("--r-states", required=r_required, default=None, help="select-r's output")
    parser.add_argument("--m-states", default=None, help="select-m's output")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sel = sub.add_parser("select-r", help="R's (and R_fail's) states from B3's held-out sampled records")
    sel.add_argument("--source", action="append", required=True, metavar="NAME=ROOT:SEED",
                     help="held-out sampled records and their System1 seed (repeatable)")
    sel.add_argument("--holdout-scenes-file", required=True, help="the 20 held-out HM3D scenes")
    sel.add_argument("--first-calls", type=int, default=brs.DEFAULT_FIRST_CALLS)
    sel.add_argument("--min-distance", type=float, default=brs.DEFAULT_MIN_DISTANCE)
    sel.add_argument("--failed", action="store_true", help="also R_fail: the failed trajectories, seeded to R's size")
    sel.add_argument("--fail-seed", type=int, default=FAIL_SEED)
    sel.add_argument("--output", required=True, help="states (JSON lines); <stem>.selection.json beside it")

    sem = sub.add_parser("select-m", help="M's states from D3's parity rows and the RFT stream")
    sem.add_argument("--d3-calls", required=True, help="D3's parity_g0.calls.jsonl")
    sem.add_argument("--d3-report", default="", help="D3's parity_g0.json (default: beside --d3-calls)")
    sem.add_argument("--rft-stream", required=True, help="build_rft_stream.py's output directory")
    sem.add_argument("--record-map", default="", help="staged path<TAB>original path, as aud.py dose takes it")
    sem.add_argument("--output", required=True, help="states (JSON lines); <stem>.selection.json beside it")

    dec = sub.add_parser("decode", help="decode the states with the adapter on and off (GPU)")
    _state_files(dec, r_required=False)
    dec.add_argument("--adapter", required=True,
                     help="PEFT directory (adapter_config.json + adapter_model.safetensors)")
    dec.add_argument("--model-path", required=True, help="the deployed InternNav_Model directory")
    dec.add_argument("--output", required=True, help="rft_gate.json; the decode rows go beside it")
    dec.add_argument("--shard", default="0/1", help="i/n: decode the states whose uid hashes to i")
    dec.add_argument("--device", choices=("cuda", "npu"), default="cuda")
    dec.add_argument("--default-vlm-image-size", type=int, nargs=2, default=[384, 384])
    dec.add_argument("--seed", type=int, default=42)
    dec.add_argument("--log-every", type=int, default=100)
    dec.add_argument("--dry-run", action="store_true", help="count the states only; no model")

    sco = sub.add_parser("score", help="R, R_fail, M and the gate (CPU)")
    _state_files(sco, r_required=True)
    sco.add_argument("--adapter", default="", help="when given, the decoded rows must be of this adapter")
    sco.add_argument("--output", required=True, help="rft_gate.json; <stem>.calls.jsonl beside it")
    args = parser.parse_args(argv)
    if args.command == "decode" and not (args.r_states or args.m_states):
        parser.error("decode needs --r-states and/or --m-states")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    return {"select-r": run_select_r, "select-m": run_select_m, "decode": decode, "score": score}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

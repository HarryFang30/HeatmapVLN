#!/usr/bin/env python3
"""EXP-35's one offline gate: the candidate's STOP rate by distance on the round-2 learner's held-out states.

The ledger's EXP-35 (written 2026-10-24 before any EXP-35 training data was read): the records of
``collect_envdrop3``'s cohort shards 000-003 (X34M's own rollouts, kept out of the stream by
``build_exp35_stream.py --holdout-cohort-shards``) are decoded teacher-forced, every call, once with the
candidate adapter on and once with it off -- the base model, on the same card, in the same process --
and the STOP rate is counted per distance band (the call's recorded geodesic to the goal):

====  ==============================  ============================================
band  rate                            line
====  ==============================  ============================================
>3m   STOP answers / calls, on, off    on <= off + 1.0 pt (EXP-29 died of far stops)
1-3m  the same                         report only
<=1m  the same                         report only
====  ==============================  ============================================

A STOP is what the server would execute as one: ``decision_class`` of the final turn is ``stop``
(``STOP`` first, or no action token at all, the server's fallback).  The learner's own recorded rates
are reported beside them.  The line is compared in whole counts: ``100 * (stop_on - stop_off) <= calls``.

Decoding is ``hm3d_route_gate.py decode``'s: the prompt the server built for the recorded call
(``inference_messages``), ``greedy_native_decision`` with the adapter on and in ``adapter_off``.

Subcommands::

    decode  NPU / GPU.  --shard i/n keeps the states whose key hashes to i (one process per card);
            <output stem>.decode.shardIIofNN.jsonl is the resume log.  --dry-run counts the states.
    score   No model.  Reads every shard's rows and writes --output.  Exit 0: the gate passes; 1: it
            fails; 2: invalid (a gate episode not recorded, a record another adapter served, states
            not decoded, rows of another adapter than --adapter, shards the candidate's stream
            (--stream-manifest) did not hold out, an unreadable input).

    python scripts/exp35/stop_rate_gate.py decode --record-root /workspace/exp34/collect_envdrop3/records \\
        --cohort-dir /workspace/exp34/collect_envdrop3/cohorts --shards 000,001,002,003 \\
        --learner-adapter <X34M sha256> --adapter <candidate adapter dir> \\
        --model-path /workspace/InternNav_Model --output <dir>/stop_gate.json --shard 0/8 --device npu
    python scripts/exp35/stop_rate_gate.py score <same selection flags> --adapter <dir> \\
        --stream-manifest /workspace/exp35/stream/stream_manifest.json --output <dir>/stop_gate.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.exp28 import hm3d_route_gate as hg  # noqa: E402
from scripts.exp35.build_exp35_stream import cohort_episodes  # noqa: E402
from scripts.tools.stratified_gates import decision_class  # noqa: E402

REC, ORL = hg.REC, hg.ORL
LOGGER = logging.getLogger("stop_rate_gate")
DECODE_SCHEMA = "heatmapvln-exp35-stop-rate-gate-decode-v1"
SCHEMA = "heatmapvln-exp35-stop-rate-gate-v1"
BANDS = (("<=1m", 0.0, 1.0), ("1-3m", 1.0, 3.0), (">3m", 3.0, math.inf))
GATED_BAND = ">3m"
MARGIN_PCT = 1  # on <= off + 1.0 pt


def band(d: float) -> str:
    for name, lo, hi in BANDS:
        if (lo < d <= hi) or (lo == 0.0 and d <= hi):
            return name
    raise ValueError(d)


def is_stop(turns: Sequence[str]) -> bool:
    return decision_class(str(turns[-1])) == "stop"


def recorded_turns(call: Mapping[str, Any]) -> list[str]:
    answer = ORL.parse_model_answer(call)
    return [answer.turn1] + ([answer.turn2] if answer.turn2 is not None else [])


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def select_states(record_roots: Sequence[str], cohort_dir: str, shards: Sequence[str], *,
                  learner_adapter: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every call of every gate episode's record, with its recorded distance and answer."""
    wanted = cohort_episodes(cohort_dir, shards)
    found: dict[tuple[str, int], str] = {}
    states: list[dict[str, Any]] = []
    stats: Counter = Counter()
    for root in record_roots:
        for path in REC.iter_record_paths(root):
            record = REC.EpisodeRecord(path)
            episode = (hg.scene_key(record.scene_id), int(record.episode_id))
            if episode not in wanted:
                continue
            if episode in found:
                raise SystemExit(f"{episode} is recorded twice: {found[episode]} and {path}")
            served = hg.served_adapter(record.episode)
            if served != learner_adapter:
                raise SystemExit(f"{path} was served by {served!r}, not the learner {learner_adapter}")
            found[episode] = str(path)
            for call in sorted(record.calls, key=lambda c: int(c["call_index"])):
                d = call.get("distance_to_goal")
                if d is None or not math.isfinite(float(d)):
                    stats["no_distance"] += 1
                    continue
                key = f"{REC.episode_key(record.scene_id, record.episode_id)}#{int(call['call_index'])}"
                states.append({"key": key, "record_path": str(path), "call_index": int(call["call_index"]),
                               "d": float(d), "band": band(float(d)), "recorded": recorded_turns(call)})
                stats[f"calls_{band(float(d))}"] += 1
    missing = sorted(wanted - set(found))
    meta = {"cohort_dir": str(cohort_dir), "shards": list(shards), "episodes_wanted": len(wanted),
            "episodes_found": len(found), "episodes_missing": len(missing), "missing_examples": missing[:5],
            "states": len(states), "stats": dict(stats), "learner_adapter": learner_adapter}
    return sorted(states, key=lambda s: s["key"]), meta


def _selection(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return select_states(args.record_root, args.cohort_dir, args.shards.split(","), learner_adapter=args.learner_adapter)


# --------------------------------------------------------------------------- #
# decode
# --------------------------------------------------------------------------- #
def decode(args: argparse.Namespace) -> int:
    started = time.time()
    states, meta = _selection(args)
    index, count = hg.parse_shard(args.shard)
    mine = [s for s in states if hg.shard_of(s["key"], count) == index]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_path = hg.decode_path(output, index, count)
    LOGGER.info("%d states (%s), %d in shard %d/%d", len(states), meta["stats"], len(mine), index, count)
    if args.dry_run:
        print(json.dumps({"states": len(states), "shard": len(mine), "selection": meta}, ensure_ascii=False), flush=True)
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
    from src.data.s2_sft_dataset import RecordCache, inference_messages, record_device

    adapter_sha = sha256_file(Path(args.adapter) / "adapter_model.safetensors")
    done = hg.read_decode_rows([rows_path]) if rows_path.is_file() else {}
    others = {r.get("adapter_sha256") for r in done.values()} - {adapter_sha}
    if others:
        raise SystemExit(f"{rows_path} holds rows of another adapter {sorted(map(str, others))}; use another --output")
    todo = [s for s in mine if s["key"] not in done]
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
                messages, images, lookdown = inference_messages(record, calls[state["call_index"]],
                                                                default_vlm_image_size=size)
                with torch.no_grad():
                    on = greedy_native_decision(model, processor, messages, images, lookdown, rt.device)
                    with adapter_off(model):
                        off = greedy_native_decision(model, processor, messages, images, lookdown, rt.device)
                row = {"schema": DECODE_SCHEMA, "key": state["key"], "adapter_sha256": adapter_sha,
                       "device": rt.device.type, "record_device": record_device(record),
                       "on": hg._arm(on), "off": hg._arm(off), "seconds": round(time.time() - t0, 3)}
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
def summarize(states: Sequence[Mapping[str, Any]], rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Per band: calls, STOP counts on / off / recorded, the 2x2 of on against off, and the gate."""
    out: dict[str, Any] = {}
    for name, _, _ in BANDS:
        mine = [s for s in states if s["band"] == name]
        on = [is_stop(rows[s["key"]]["on"]["turns"]) for s in mine]
        off = [is_stop(rows[s["key"]]["off"]["turns"]) for s in mine]
        rec = [is_stop(s["recorded"]) for s in mine]
        n = len(mine)
        rate = lambda k: round(k / n, 4) if n else None  # noqa: E731
        out[name] = {
            "calls": n, "episodes": len({s["key"].rsplit("#", 1)[0] for s in mine}),
            "stop_on": sum(on), "stop_off": sum(off), "stop_recorded": sum(rec),
            "rate_on": rate(sum(on)), "rate_off": rate(sum(off)), "rate_recorded": rate(sum(rec)),
            "on_only": sum(a and not b for a, b in zip(on, off)), "off_only": sum(b and not a for a, b in zip(on, off)),
        }
    gated = out[GATED_BAND]
    passed = 100 * (gated["stop_on"] - gated["stop_off"]) <= MARGIN_PCT * gated["calls"]
    return {"bands": out, "gate": {"band": GATED_BAND, "line": f"rate_on <= rate_off + {MARGIN_PCT}.0 pt",
                                   "delta_pt": round(100 * (gated["stop_on"] - gated["stop_off"]) / gated["calls"], 3)
                                   if gated["calls"] else None, "pass": bool(passed and gated["calls"] > 0)}}


def stream_problems(manifest_path: str, cohort_dir: str, shards: Sequence[str]) -> list[str]:
    """The gate's shards must be the ones ``build_exp35_stream.py`` held out of the candidate's stream."""
    held = (json.loads(Path(manifest_path).read_text(encoding="utf-8")).get("holdout") or {}).get("gate_cohort_shards")
    if not held:
        return [f"{manifest_path} names no gate cohort shards"]
    held_dir, held_shards = held
    problems = []
    if Path(held_dir).resolve() != Path(cohort_dir).resolve():
        problems.append(f"the stream held out shards of {held_dir}, the gate reads {cohort_dir}")
    if sorted(int(x) for x in str(held_shards).split(",")) != sorted(int(x) for x in shards):
        problems.append(f"the stream held out shards {held_shards}, the gate reads {','.join(shards)}")
    return problems


def score(args: argparse.Namespace) -> int:
    from scripts.train_system2_lora import sha256_file

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"schema": SCHEMA}
    problems: list[str] = []
    try:
        problems += stream_problems(args.stream_manifest, args.cohort_dir, args.shards.split(","))
        states, meta = _selection(args)
        rows = hg.read_decode_rows(sorted(output.parent.glob(f"{output.stem}.decode.shard*.jsonl")))
        if meta["episodes_missing"]:
            problems.append(f"{meta['episodes_missing']} gate episodes have no record, e.g. {meta['missing_examples']}")
        if not states:
            problems.append("no states")
        missing = [s["key"] for s in states
                   if s["key"] not in rows or "on" not in rows[s["key"]] or "off" not in rows[s["key"]]]
        if missing:
            problems.append(f"{len(missing)} states not decoded, e.g. {missing[:3]}")
        keys = {s["key"] for s in states}
        adapters = {r.get("adapter_sha256") for k, r in rows.items() if k in keys}
        want = sha256_file(Path(args.adapter) / "adapter_model.safetensors")
        if adapters - {want}:
            problems.append(f"rows of adapters {sorted(map(str, adapters - {want}))}, not the candidate's {want}")
        result.update(selection=meta, adapter_sha256=want,
                      devices=dict(Counter(str(rows[s["key"]].get("device")) for s in states if s["key"] in rows)))
        if not problems:
            result.update(summarize(states, rows))
    except (SystemExit, RuntimeError, ValueError, KeyError, OSError) as err:
        # a record twice or another adapter's, a missing cohort shard, a torn decode row: no pass or fail
        # can be read from it (exit 2, never the 1 of a failed gate, which means no val)
        problems.append(f"{type(err).__name__}: {err}")
    if problems:
        result.update(status="invalid", problems=problems)
        code = 2
    else:
        result["status"] = "pass" if result["gate"]["pass"] else "fail"
        code = 0 if result["gate"]["pass"] else 1
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.write_text(json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(output)
    print(json.dumps({"status": result["status"], "gate": result.get("gate"), "problems": result.get("problems")},
                     ensure_ascii=False), flush=True)
    return code


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--record-root", action="append", required=True)
    parser.add_argument("--cohort-dir", required=True)
    parser.add_argument("--shards", default="000,001,002,003", help="the gate's cohort shards, comma-separated")
    parser.add_argument("--learner-adapter", required=True, help="sha256 of the adapter that served the records")
    parser.add_argument("--output", required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    dec = sub.add_parser("decode", help="decode every state with the adapter on and off (NPU / GPU)")
    _selection_args(dec)
    dec.add_argument("--adapter", required=True)
    dec.add_argument("--model-path", required=True, help="the deployed InternNav_Model directory")
    dec.add_argument("--shard", default="0/1")
    dec.add_argument("--device", choices=("auto", "cuda", "npu", "cpu"), default="auto")
    dec.add_argument("--seed", type=int, default=42)
    dec.add_argument("--default-vlm-image-size", type=int, nargs=2, default=[384, 384])
    dec.add_argument("--log-every", type=int, default=100)
    dec.add_argument("--dry-run", action="store_true")
    sco = sub.add_parser("score", help="STOP rates by distance and the gate (no model)")
    _selection_args(sco)
    sco.add_argument("--adapter", required=True, help="the candidate's adapter dir: every row must be its")
    sco.add_argument("--stream-manifest", required=True,
                     help="the candidate's training stream manifest: it must have held out these cohort shards")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args(argv)
    return {"decode": decode, "score": score}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

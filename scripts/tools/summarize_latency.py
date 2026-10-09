#!/usr/bin/env python3
"""Summarise the opt-in latency logs of the RPC evaluation (HEATMAPVLN_TIMING=1).

Input: the client's timing JSONL (schema ``heatmapvln-latency-v1``, one line per
plan call, written by ``PlanCallTimingLog`` in scripts/evaluation/r2r_val_unseen.py),
as files or as directories searched recursively for ``*.jsonl``.  A plan call
seen twice (an episode rerun after a crash) keeps its last line in file order.

Output: printed, and written to ``--output-dir`` (default: the first input
directory, or the first file's directory) as ``latency_summary.json`` and
``latency_summary.md``.  Every row is n / mean / median / p90 in milliseconds,
separately for each path a call took through the model server, because the
paths run different stages (grouped by the response's kind and ppa_applied):
  ppa             kind trajectory, ppa_applied: System 2, History Head, bridge,
                  System 1, Future Head (the AMB3R map is ready)
  native_system1  kind trajectory without PPA: System 2 and native System 1
                  (in deployment, the AMB3R warm-up before the map exists)
  system2_only    any other kind (native_actions, stop, fallback_stop): System 2's
                  arrows or stop were executed as they are, or its answer was
                  unusable; no System 1 ran
  all             every call, for the per-action cost of the whole loop
PPA's extra cost over native is the ppa group's PPA-only stages
(ppa_history_memory, ppa_bridge) plus the VO time, not ppa minus native_system1:
the warm-up calls come early in an episode, with fewer history frames in System
2's prompt.

Sections
  per plan call    one value per call and the chunk it executed:
                   model      client JPEG encoding + model RPC round trip
                   vo         VO pose query + every VO frame ingest of the window
                   simulator  panorama and lookdown captures + env.step (Habitat
                              rendering; never part of model latency)
                   timed_total, cycle_wall (measured), untimed (the difference),
                   plan_latency (replanning decision -> actions available:
                   plan-side stages only), model_server (handler_total) and
                   model_rpc_overhead (round trip - handler_total)
  per action       the same per-call sums divided by the actions the call's
                   chunk executed (calls that executed none are skipped), with
                   pooled = sum over calls / sum of actions
  model server     the model server's timing_ms stages
  VO server        the VO server's timing_ms per method, plus the round trip
                   the client measured
  GPU memory       per server, n / median / max in MiB over requests:
                   peak_allocated and peak_reserved (that process, peak within
                   a request) and device_used (the whole card after a request,
                   every process on it: model + VO when they share a card)
  client           client plan stages (per call) and step stages (per run)

HEATMAPVLN_TIMING takes effect per process, so a client timed against servers
that were not leaves every server-side table empty while the rest of the
summary looks complete.  ``server_timing_missing`` counts the model-server
calls and VO RPCs whose response carried no timing_ms; when either is nonzero
the Markdown opens with a WARNING and the exit code is 3 (both files are still
written).  Exit code 1: no timing records found.

Usage:
  python scripts/tools/summarize_latency.py <eval output dir or timing files> [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA = "heatmapvln-latency-v1"
SUMMARY_SCHEMA = "heatmapvln-latency-summary-v1"
GROUPS = (
    ("ppa", "PPA calls (trajectory, ppa_applied)"),
    ("native_system1", "Native System 1 calls (trajectory without PPA: the AMB3R warm-up)"),
    ("system2_only", "System 2 only (native_actions / stop / fallback_stop: no System 1)"),
    ("all", "All calls"),
)
MODEL_STAGES = (
    "request_decode",
    "system2_turn1_prep",
    "system2_turn1_generate",
    "system2_turn2_prep",
    "system2_turn2_generate",
    "ppa_history_memory",
    "system1_condition_latents",
    "ppa_bridge",
    "system1_nextdit_sampling",
    "trajectory_to_actions",
    "future_heatmap_diagnostics",
    "handler_total",
)
VO_STAGES = (
    "jpeg_decode",
    "ingest",
    "ingest_map_init",
    "ingest_map_update",
    "query",
    "query_map_update",
    "total",
    "round_trip",
)
PLAN_STAGES = ("pano_capture", "vo_query", "lookdown_capture", "model_encode", "model_rpc")
STEP_STAGES = ("env_step", "pano_capture", "vo_ingest")
PER_CALL = (
    "model",
    "vo",
    "simulator",
    "timed_total",
    "cycle_wall",
    "untimed",
    "plan_latency",
    "model_server",
    "model_rpc_overhead",
)
PER_ACTION = ("model", "vo", "simulator", "timed_total", "cycle_wall")
MEMORY = ("peak_allocated", "peak_reserved", "device_used")


def percentile(values: list[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default)."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 3),
        "median": round(percentile(values, 50), 3),
        "p90": round(percentile(values, 90), 3),
    }


def memory_stats(values: list[float]) -> dict[str, float]:
    return {"n": len(values), "median": round(percentile(values, 50), 1), "max": max(values)}


def _ordered(keys: Iterable[str], preferred: tuple[str, ...]) -> list[str]:
    keys = set(keys)
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def timing_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    return files


def load_records(files: list[Path]) -> list[dict[str, Any]]:
    by_call: dict[tuple[str, int], dict[str, Any]] = {}
    for path in files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema") != SCHEMA:
                continue
            by_call[(record["episode"], int(record["call_index"]))] = record
    return list(by_call.values())


def _group(record: dict[str, Any]) -> str | None:
    """The model server path the call took (see the module docstring)."""
    kind = record.get("kind")
    if kind == "trajectory":
        return "ppa" if record.get("ppa_applied") is True else "native_system1"
    return None if kind is None else "system2_only"


def per_call_values(record: dict[str, Any]) -> dict[str, float]:
    plan = record.get("plan_ms") or {}
    steps = record.get("step_ms") or {}
    model = plan.get("model_encode", 0.0) + plan.get("model_rpc", 0.0)
    vo = plan.get("vo_query", 0.0) + sum(steps.get("vo_ingest", []))
    simulator = (
        plan.get("pano_capture", 0.0)
        + plan.get("lookdown_capture", 0.0)
        + sum(steps.get("env_step", []))
        + sum(steps.get("pano_capture", []))
    )
    timed = model + vo + simulator
    values = {
        "model": model,
        "vo": vo,
        "simulator": simulator,
        "timed_total": timed,
        "plan_latency": sum(plan.values()),
    }
    if "cycle_wall_ms" in record:
        values["cycle_wall"] = record["cycle_wall_ms"]
        values["untimed"] = record["cycle_wall_ms"] - timed
    server = record.get("model_server_ms") or {}
    if "handler_total" in server:
        values["model_server"] = server["handler_total"]
        if "model_rpc" in plan:
            values["model_rpc_overhead"] = plan["model_rpc"] - server["handler_total"]
    return values


def summarise_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_call: dict[str, list[float]] = {}
    per_action: dict[str, list[float]] = {}
    pooled: dict[str, float] = {}
    model: dict[str, list[float]] = {}
    memory: dict[str, list[float]] = {}
    vo: dict[str, list[float]] = {}
    client: dict[str, list[float]] = {}
    actions = 0
    for record in records:
        executed = int(record.get("actions_executed", 0))
        actions += executed
        values = per_call_values(record)
        for name, value in values.items():
            per_call.setdefault(name, []).append(value)
        for name in PER_ACTION:
            if name not in values:
                continue
            pooled[name] = pooled.get(name, 0.0) + values[name]
            if executed > 0:
                per_action.setdefault(name, []).append(values[name] / executed)
        for stage, value in (record.get("model_server_ms") or {}).items():
            model.setdefault(stage, []).append(value)
        for name, value in (record.get("model_cuda_mib") or {}).items():
            memory.setdefault(f"model.{name}", []).append(value)
        for entry in record.get("vo_rpc") or []:
            method = entry.get("method", "?")
            vo.setdefault(f"{method}.round_trip", []).append(entry["rpc_ms"])
            for stage, value in (entry.get("server_ms") or {}).items():
                vo.setdefault(f"{method}.{stage}", []).append(value)
            for name, value in (entry.get("server_cuda_mib") or {}).items():
                memory.setdefault(f"vo.{name}", []).append(value)
        for stage, value in (record.get("plan_ms") or {}).items():
            client.setdefault(f"plan.{stage}", []).append(value)
        for stage, values_ms in (record.get("step_ms") or {}).items():
            client.setdefault(f"step.{stage}", []).extend(values_ms)

    vo_order = [
        f"{method}.{stage}"
        for method in ("ingest_frame", "query_relative_poses")
        for stage in VO_STAGES
    ]
    client_order = [f"plan.{stage}" for stage in PLAN_STAGES] + [f"step.{stage}" for stage in STEP_STAGES]
    memory_order = [f"{server}.{name}" for server in ("model", "vo") for name in MEMORY]
    per_action_stats = {}
    for name in _ordered(per_action, PER_ACTION):
        per_action_stats[name] = stats(per_action[name])
        per_action_stats[name]["pooled"] = round(pooled[name] / actions, 3) if actions else None
    return {
        "calls": len(records),
        "actions_executed": actions,
        "per_call": {name: stats(per_call[name]) for name in _ordered(per_call, PER_CALL)},
        "per_action": per_action_stats,
        "model_server": {name: stats(model[name]) for name in _ordered(model, MODEL_STAGES)},
        "cuda_memory_mib": {name: memory_stats(memory[name]) for name in _ordered(memory, tuple(memory_order))},
        "vo_server": {name: stats(vo[name]) for name in _ordered(vo, tuple(vo_order))},
        "client": {name: stats(client[name]) for name in _ordered(client, tuple(client_order))},
    }


def server_timing_missing(records: list[dict[str, Any]]) -> dict[str, int]:
    """Server calls that came back without the server's own stage breakdown.

    A server with timing on puts timing_ms in every response, so an answered
    call without it means that server process ran untimed.  ``model_server_ms``
    exists in a record only once the model server answered (end_plan), and
    every ``vo_rpc`` entry is an answered VO RPC.
    """
    model = [r["model_server_ms"] for r in records if "model_server_ms" in r]
    vo = [entry.get("server_ms") for r in records for entry in r.get("vo_rpc") or []]
    return {
        "model_server_calls": sum(1 for stages in model if not stages),
        "model_server_calls_seen": len(model),
        "vo_rpcs": sum(1 for stages in vo if not stages),
        "vo_rpcs_seen": len(vo),
    }


def server_timing_warning(missing: dict[str, int]) -> str | None:
    if not (missing["model_server_calls"] or missing["vo_rpcs"]):
        return None
    return (
        f"WARNING: {missing['model_server_calls']} of {missing['model_server_calls_seen']} model-server calls "
        f"and {missing['vo_rpcs']} of {missing['vo_rpcs_seen']} VO RPCs carry no server-side stages: "
        "the server processes behind them were not running with HEATMAPVLN_TIMING=1, so this summary "
        "covers client-side stages only for them and no end-to-end latency may be quoted from it."
    )


def summarise(records: list[dict[str, Any]], sources: list[str]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for key, _title in GROUPS:
        members = records if key == "all" else [r for r in records if _group(r) == key]
        if members:
            groups[key] = summarise_group(members)
    return {
        "schema": SUMMARY_SCHEMA,
        "sources": sources,
        "calls": len(records),
        "actions_executed": sum(int(r.get("actions_executed", 0)) for r in records),
        "server_timing_missing": server_timing_missing(records),
        "groups": groups,
    }


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _table(rows: dict[str, dict[str, Any]], *, pooled: bool = False) -> list[str]:
    head = "| stage | n | mean | median | p90 |" + (" pooled |" if pooled else "")
    rule = "|---|---:|---:|---:|---:|" + ("---:|" if pooled else "")
    lines = [head, rule]
    for name, row in rows.items():
        cells = [name, str(row["n"]), _fmt(row["mean"]), _fmt(row["median"]), _fmt(row["p90"])]
        if pooled:
            cells.append(_fmt(row["pooled"]))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _memory_table(rows: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["| server.memory | n | median | max |", "|---|---:|---:|---:|"]
    for name, row in rows.items():
        lines.append(f"| {name} | {row['n']} | {_fmt(row['median'])} | {_fmt(row['max'])} |")
    return lines


def to_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Latency summary", ""]
    warning = server_timing_warning(summary["server_timing_missing"])
    if warning is not None:
        lines += [f"**{warning}**", ""]
    lines += [
        f"{summary['calls']} plan calls, {summary['actions_executed']} executed actions, "
        f"from {len(summary['sources'])} file(s). Milliseconds; n / mean / median / p90.",
        "Simulator time (Habitat rendering, env.step) is reported apart from model and VO time.",
    ]
    for key, title in GROUPS:
        group = summary["groups"].get(key)
        if group is None:
            continue
        lines += ["", f"## {title}: {group['calls']} calls, {group['actions_executed']} actions"]
        lines += ["", "### Per plan call", "", *_table(group["per_call"])]
        lines += ["", "### Per executed action (pooled = sum / actions)", ""]
        lines += _table(group["per_action"], pooled=True)
        if group["model_server"]:
            lines += ["", "### Model server stages", "", *_table(group["model_server"])]
        if group["vo_server"]:
            lines += ["", "### VO server stages (per RPC)", "", *_table(group["vo_server"])]
        if group["cuda_memory_mib"]:
            lines += ["", "### GPU memory per request (MiB; device_used counts every process on the card)", ""]
            lines += _memory_table(group["cuda_memory_mib"])
        if group["client"]:
            lines += ["", "### Client stages (plan: per call; step: per run)", "", *_table(group["client"])]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="timing JSONL files or directories holding them")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    files = timing_files(args.paths)
    records = load_records(files)
    if not records:
        print(f"no {SCHEMA} records under {[str(p) for p in args.paths]}", file=sys.stderr)
        return 1
    summary = summarise(records, [str(path) for path in files])
    markdown = to_markdown(summary)
    output_dir = args.output_dir
    if output_dir is None:
        first = args.paths[0]
        output_dir = first if first.is_dir() else first.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latency_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "latency_summary.md").write_text(markdown)
    print(markdown)
    print(f"written: {output_dir / 'latency_summary.json'} and latency_summary.md")
    warning = server_timing_warning(summary["server_timing_missing"])
    if warning is not None:
        print(warning, file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

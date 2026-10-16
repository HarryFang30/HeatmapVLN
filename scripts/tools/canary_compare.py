#!/usr/bin/env python3
"""EXP-25's closed-loop canary (C-A): an adapter's episodes against the base model's round-1 outcomes.

The canary runs the held-out scenes' 392 round-1 episodes (``remaining_cohort.py --keep
committed --holdout``) with the adapter served, recording every call as round 1 did; the
base model's records of the same episodes are round 1's.  Both sides are read from the
records (``episode.json`` outcome, ``calls.jsonl``), episode by episode, on the episodes
both have.  An episode is (scene, episode id); a key read twice is an error.

    canary_compare.py --canary-records roundA/canary/records --base-records round1/records \\
        [--cohort roundA/canary_cohort] [--output canary_compare.json]

How an episode ended is read from its last call: a STOP answer ends it by the model's
STOP; otherwise it ran out of steps (500, or a step-budget STOP the client issued under
``--stop_at_step_budget``, which the model did not say).  Success under the standard
protocol is a model STOP within 3 m, so a step-budget STOP never counts (EXP-23 run record
8); the record's own ``success`` is reported beside it.

C-A (ledger EXP-25, fixed before any result):

* episodes ended by a STOP at >= 3 m:  canary <= base + 4
* episodes that ran out of steps:      canary <= base + 4
* turn reversals (a turn answer right after a turn the other way), share of turn calls:
  canary <= 0.02
* SR (standard protocol):              canary >= base - 1.0 pt

With ``--cohort`` every cohort episode must be on both sides, or the canary is incomplete
and does not pass.  Prints PASS/FAIL first; exit 0 on PASS.

``--criteria exp28-p`` judges EXP-28's HM3D held-out proxy P instead (ledger EXP-28 E,
written 2026-10-15 before any HM3D result; a one-sided harm filter, any line crossed
withdraws the candidate's val run):

* paired standard-protocol ΔSR:                                    > -3.0 pt
* episodes that stopped but never came within 3 m (OS = 0), share: canary <= base + 3 pt
* episodes that ran out of steps, share:                           canary <= base + 3 pt
* turn reversals, share of turn calls:                             canary <= 0.02
* both arms served from the 910B (``heatmapvln-device:npu``; ``--require-device``)

with ΔSR's interval from a scene-cluster bootstrap of 20,000 draws (``--bootstrap-*``;
report only).  The bootstrap and the platform check can be added to C-A too; without them
C-A's output is unchanged.

P reads how an episode ended from what Habitat executed, not from the last answer: SR is
the record's own ``success`` (Habitat's: a STOP called within 3 m), and an episode stopped
when its last primitive step was STOP.  Phase-0's SR is the same ``success``, so the base
arm's SR here is Phase-0's on these episodes.  A STOP the client issued for the step
budget (``--stop_at_step_budget``, phase ``step_budget_stop``) is no STOP under the standard
protocol: such an episode ran out of steps and scores 0.  C-A's answer-based ending
(``ended_by``, ``success_standard``) is reported beside it, with the episodes where the
two disagree (a client STOP after an empty or unparsable answer).

P's verdict, printed first, is ``kill`` when any line is crossed, ``pass`` when none is and
every cohort episode is on both sides, else ``incomplete`` (not a kill: rerun the missing
ones); ``invalid`` when the platform check fails (no pair at all).  Exit 0 / 1 / 2 / 3.

A cohort whose manifest has several ``parts`` (EXP-28's: ``heldout``, ``train_pass1``, ...)
needs ``--cohort-part``; P's is ``--cohort-part heldout``.

``--rerun-records R --reference-records O`` is the per-call determinism check instead
(EXP-28 E validity: the base server rerun on the same commit must reproduce >= 9 of 10
Phase-0 held-out episodes call by call, ``--min-identical-episodes 9``).  A call is the
same when its step and every decision field the server returned are equal
(``DECISION_FIELDS``: the ledger's step / kind / System2 text / action chunk, plus the
first native turn and the pixel goal); an episode is the same when every call and its
outcome are.  Whether the prompt bytes were the same is reported beside it.  A rerun on
another platform than the reference is not a determinism test and does not pass.
"""

from __future__ import annotations

import argparse
import json
import operator
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.paired_closed_loop_bootstrap import bootstrap_ci, cluster_bootstrap_ci  # noqa: E402
from scripts.tools.stratified_gates import decision_class, turn_direction  # noqa: E402
from src.data.s2_records import EpisodeRecord, episode_key, iter_record_paths, record_path  # noqa: E402

SCHEMA = "heatmapvln-exp25-canary-compare-v1"
P_SCHEMA = "heatmapvln-exp28-proxy-compare-v1"
IDENTITY_SCHEMA = "heatmapvln-exp28-call-identity-v1"
FAR_STOP_M = 3.0
SPIN_M = 1.0
# C-A
STOP_FAR_MARGIN = 4
BUDGET_MARGIN = 4
REVERSAL_MAX = 0.02
SR_DROP_MAX_PT = 1.0
# EXP-28 E, proxy P
P_SR_KILL_PT = -3.0
P_STOP_OUTSIDE_MARGIN_PT = 3.0
P_OUT_OF_STEPS_MARGIN_PT = 3.0
P_REVERSAL_MAX = 0.02
P_BOOTSTRAP_DRAWS = 20_000
P_DEVICE = "npu"
CRITERIA_SETS = ("exp25-ca", "exp28-p")
STOP_ACTION = 0  # ActionCode.STOP
BUDGET_STOP_PHASE = "step_budget_stop"
# Per-call identity: what the server decided for a call (``_S2_RECORD_RESPONSE_FIELDS`` of
# r2r_val_unseen.py), the prompt bytes it saw, and how the episode ended.
DECISION_FIELDS = ("step_id", "kind", "native_first_output", "native_lookdown_turns", "llm_output",
                   "pixel_goal", "actions", "terminal")
INPUT_FIELDS = ("front_sha256", "lookdown_sha256")
OUTCOME_FIELDS = ("success", "spl", "distance_to_goal", "oracle_success", "steps", "system2_calls",
                  "trajectory_calls", "episode_over")
META_FIELDS = ("client_code_sha256", "config_sha256", "protocol_seed", "rpc_policy_fingerprint")

_COMPARE = {"<=": operator.le, ">=": operator.ge, ">": operator.gt, "==": operator.eq}


def record_device(record: EpisodeRecord) -> str:
    """``src.data.s2_sft_dataset.record_device`` without its torch import (tests pin them equal).

    The Ascend servers advertise ``heatmapvln-device:npu``; the 4090's local servers carry no tag.
    """
    formats = (record.episode.get("server") or {}).get("supported_formats") or []
    for item in formats:
        if str(item).startswith("heatmapvln-device:"):
            return str(item).split(":", 1)[1]
    return "cuda"


def episode_summary(record: EpisodeRecord) -> dict[str, Any]:
    calls = sorted(record.calls, key=lambda c: int(c["call_index"]))
    classes = [decision_class(str(c.get("llm_output") or "")) for c in calls]
    outcome = record.episode.get("outcome") or {}
    distance = float(outcome["distance_to_goal"])
    stopped = bool(classes) and classes[-1] == "stop"
    turns = sum(1 for c in classes if turn_direction(c))
    reversals = sum(1 for a, b in zip(classes, classes[1:])
                    if turn_direction(a) and turn_direction(b) and turn_direction(a) != turn_direction(b))
    spin, spin_stop = 0, 0
    for previous, call, cls in zip(classes, calls[1:], classes[1:]):
        d = call.get("distance_to_goal")
        if turn_direction(previous) and d is not None and float(d) <= SPIN_M:
            spin += 1
            spin_stop += cls == "stop"
    if outcome.get("oracle_success") is not None:
        entered = float(outcome["oracle_success"]) > 0.5
    else:  # Habitat's OS: distance < 3 m at some step
        seen = [float(row["distance_to_goal"]) for row in list(record.steps) + calls
                if row.get("distance_to_goal") is not None]
        entered = min(seen + [distance]) < FAR_STOP_M
    # What Habitat executed last (the collector records every primitive step with its action
    # and phase); None when the steps carry no action, which P refuses.
    last = record.steps[-1] if record.steps else {}
    stop_called = None if "action" not in last else (last["action"] is not None
                                                     and int(last["action"]) == STOP_ACTION)
    budget_stop = bool(stop_called) and last.get("phase") == BUDGET_STOP_PHASE
    success_recorded = float(outcome.get("success", 0.0)) > 0.5
    return {
        "ended_by": "stop" if stopped else "steps",
        "distance_to_goal": distance,
        "success_standard": bool(stopped and distance <= FAR_STOP_M),
        "success_recorded": success_recorded,
        "steps": int(outcome.get("steps", 0)),
        "calls": len(classes),
        "turn_calls": turns,
        "reversals": reversals,
        "spin_states": spin,
        "spin_stops": spin_stop,
        "oracle_success": entered,
        "oracle_success_recorded": outcome.get("oracle_success") is not None,
        # P: Habitat's own ending under the standard protocol (a step-budget STOP is no STOP)
        "stop_called": stop_called,
        "stop_phase": last.get("phase") if stop_called else None,
        "stopped_standard": bool(stop_called) and not budget_stop,
        "success_habitat_standard": success_recorded and bool(stop_called) and not budget_stop,
        "device": record_device(record),
        "client_code_sha256": record.episode.get("client_code_sha256"),
    }


def _read_records(root: Path, summarise=None) -> dict[tuple[str, int], Any]:
    """Committed records under ``root`` by (scene, episode id): their summary, or their path.

    One episode recorded twice (a record outside its scene's directory) is an error.
    """
    where: dict[tuple[str, int], Path] = {}
    out: dict[tuple[str, int], Any] = {}
    for path in iter_record_paths(root):
        record = EpisodeRecord(path)
        key = (record.scene_id, record.episode_id)
        if key in where:
            raise ValueError(f"{path} and {where[key]} are both scene {key[0]} episode {key[1]}")
        where[key] = path
        out[key] = path if summarise is None else summarise(record)
    return out


def load(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    return _read_records(root, episode_summary)


def side_totals(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(episodes)
    turns = sum(e["turn_calls"] for e in episodes)
    reversals = sum(e["reversals"] for e in episodes)
    spin = sum(e["spin_states"] for e in episodes)
    spin_stop = sum(e["spin_stops"] for e in episodes)
    return {
        "episodes": n,
        "sr_pct": round(100.0 * sum(e["success_standard"] for e in episodes) / n, 3) if n else None,
        "sr_recorded_pct": round(100.0 * sum(e["success_recorded"] for e in episodes) / n, 3) if n else None,
        "stop_far": sum(e["ended_by"] == "stop" and e["distance_to_goal"] >= FAR_STOP_M for e in episodes),
        "out_of_steps": sum(e["ended_by"] == "steps" for e in episodes),
        "out_of_steps_min_steps": min((e["steps"] for e in episodes if e["ended_by"] == "steps"), default=None),
        "turn_calls": turns,
        "reversals": reversals,
        "reversal_share": round(reversals / turns, 6) if turns else None,
        "spin_states": spin,
        "spin_stop_rate": round(spin_stop / spin, 6) if spin else None,
        "mean_calls": round(sum(e["calls"] for e in episodes) / n, 2) if n else None,
    }


def criterion(value, threshold, direction, **extra) -> dict[str, Any]:
    ok = value is not None and threshold is not None and _COMPARE[direction](value, threshold)
    return {"value": value, "threshold": threshold, "direction": direction, "pass": bool(ok), **extra}


def _pct(count: int, n: int) -> float | None:
    return round(100.0 * count / n, 3) if n else None


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else round(a - b, 3)


def p_side_totals(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """P's counts on one arm, from what Habitat executed (``stopped_standard``, ``success_habitat_standard``)."""
    unknown = [e for e in episodes if e["stop_called"] is None]
    if unknown:
        raise ValueError(f"{len(unknown)} records have steps without actions, so P cannot tell how they ended")
    n = len(episodes)
    stopped = [e for e in episodes if e["stopped_standard"]]
    return {
        "sr_standard_pct": _pct(sum(e["success_habitat_standard"] for e in episodes), n),
        "stopped_never_within_3m_pct": _pct(sum(not e["oracle_success"] for e in stopped), n),
        "out_of_steps_pct": _pct(n - len(stopped), n),
        "os_pct": _pct(sum(e["oracle_success"] for e in episodes), n),
        "step_budget_stops": sum(e["stop_phase"] == BUDGET_STOP_PHASE for e in episodes),
        # STOPs the client issued (an empty or unparsable answer), not after a STOP answer
        "client_stops": sum(e["stopped_standard"] and e["ended_by"] != "stop" for e in episodes),
        # episodes whose answer-based ending (C-A's) is not the executed one
        "ending_disagrees_with_answer": sum((e["ended_by"] == "stop") != e["stopped_standard"] for e in episodes),
        "success_disagrees_with_answer": sum(e["success_standard"] != e["success_habitat_standard"]
                                             for e in episodes),
        "os_from_distances": sum(not e["oracle_success_recorded"] for e in episodes),
    }


def p_criteria(c: Mapping, b: Mapping) -> dict:
    """EXP-28 E's kill lines, each as the line the candidate must stay on to pass (``c``/``b``: P's side totals)."""
    return {
        "sr_delta_pt": criterion(_delta(c["sr_standard_pct"], b["sr_standard_pct"]), P_SR_KILL_PT, ">",
                                 base=b["sr_standard_pct"], canary=c["sr_standard_pct"]),
        "stop_never_within_3m_delta_pt": criterion(
            _delta(c["stopped_never_within_3m_pct"], b["stopped_never_within_3m_pct"]), P_STOP_OUTSIDE_MARGIN_PT,
            "<=", base=b["stopped_never_within_3m_pct"], canary=c["stopped_never_within_3m_pct"]),
        "out_of_steps_delta_pt": criterion(_delta(c["out_of_steps_pct"], b["out_of_steps_pct"]),
                                           P_OUT_OF_STEPS_MARGIN_PT, "<=",
                                           base=b["out_of_steps_pct"], canary=c["out_of_steps_pct"]),
        # no turn call at all is no reversal (C-A, written earlier, fails it instead)
        "turn_reversal_share": criterion(c["reversal_share"] if c["turn_calls"] else 0.0, P_REVERSAL_MAX, "<=",
                                         base=b["reversal_share"]),
    }


def platform_criterion(canary_eps: Sequence[Mapping], base_eps: Sequence[Mapping], device: str) -> dict:
    tally = {side: dict(sorted(Counter(e["device"] for e in eps).items()))
             for side, eps in (("canary", canary_eps), ("base", base_eps))}
    seen = sorted(set(tally["canary"]) | set(tally["base"]))
    return criterion(seen, [device], "==", base=tally["base"], canary=tally["canary"])


def sr_bootstrap(canary: Mapping, base: Mapping, keys: Sequence, *, unit: str, draws: int, seed: int,
                 field: str = "success_standard") -> dict:
    """Interval of the paired standard-protocol ΔSR, resampling scenes (clusters) or episodes."""
    diffs = [float(canary[k][field]) - float(base[k][field]) for k in keys]
    rng = random.Random(seed)
    if unit == "scene":
        lo, hi = cluster_bootstrap_ci(diffs, [str(k[0]) for k in keys], rng, draws)
    else:
        lo, hi = bootstrap_ci(diffs, rng, draws)
    return {
        "metric": field,
        "unit": unit,
        "draws": draws,
        "seed": seed,
        "episodes": len(keys),
        "clusters": len({k[0] for k in keys}),
        "diff_pt": round(100.0 * sum(diffs) / len(diffs), 3),
        "ci95_pt": [round(100.0 * lo, 3), round(100.0 * hi, 3)],
    }


def compare(canary: Mapping, base: Mapping, expected: set | None, *, criteria: str = "exp25-ca",
            require_device: str | None = None, bootstrap: Mapping | None = None) -> dict[str, Any]:
    """C-A (or P) on the episodes in scope (the cohort's, else the canary's) that both sides have.

    ``bootstrap``: {"unit", "draws", "seed"} adds the ΔSR interval; ``require_device`` adds the
    platform criterion.  Without either, C-A's result is the one it always was.
    """
    if criteria not in CRITERIA_SETS:
        raise ValueError(f"criteria {criteria!r} is not one of {CRITERIA_SETS}")
    scope = set(expected) if expected is not None else set(canary)
    missing_canary = sorted(scope - set(canary))
    missing_base = sorted(scope - set(base))
    keys = sorted(scope & set(canary) & set(base))
    canary_eps, base_eps = [canary[k] for k in keys], [base[k] for k in keys]
    c = side_totals(canary_eps)
    b = side_totals(base_eps)

    crit = criterion
    if criteria == "exp25-ca":
        sr_field = "success_standard"
        criteria_out = {
            "stop_far_episodes": crit(c["stop_far"], b["stop_far"] + STOP_FAR_MARGIN, "<=", base=b["stop_far"]),
            "out_of_steps_episodes": crit(c["out_of_steps"], b["out_of_steps"] + BUDGET_MARGIN, "<=", base=b["out_of_steps"]),
            "turn_reversal_share": crit(c["reversal_share"], REVERSAL_MAX, "<=", base=b["reversal_share"]),
            "sr_pct": crit(c["sr_pct"], None if b["sr_pct"] is None else round(b["sr_pct"] - SR_DROP_MAX_PT, 3), ">=",
                           base=b["sr_pct"]),
        }
    else:
        # P's own counts (Habitat's executed ending) sit beside C-A's answer-based ones
        sr_field = "success_habitat_standard"
        c, b = {**c, **p_side_totals(canary_eps)}, {**b, **p_side_totals(base_eps)}
        criteria_out = p_criteria(c, b)
    if require_device:
        criteria_out["platform"] = platform_criterion(canary_eps, base_eps, require_device)
    complete = not missing_canary and not missing_base
    flips = {
        "base_success_canary_fail": sum(base[k][sr_field] and not canary[k][sr_field] for k in keys),
        "base_fail_canary_success": sum(canary[k][sr_field] and not base[k][sr_field] for k in keys),
    }
    result = {
        "schema": SCHEMA if criteria == "exp25-ca" else P_SCHEMA,
        "matched_episodes": len(keys),
        "expected_episodes": len(expected) if expected is not None else None,
        "complete": complete,
        "missing_in_canary": [f"{s}_{e:06d}" for s, e in missing_canary][:50],
        "missing_in_base": [f"{s}_{e:06d}" for s, e in missing_base][:50],
        "criteria": criteria_out,
        "all_pass": complete and all(v["pass"] for v in criteria_out.values()),
        "canary": c,
        "base": b,
        "report_only": {"paired_flips": flips},
    }
    if criteria != "exp25-ca":
        result["criteria_set"] = criteria
        lines = {name: v for name, v in criteria_out.items() if name != "platform"}
        crossed = [name for name, v in lines.items() if v["value"] is not None and not v["pass"]]
        result["kill_lines_crossed"] = crossed
        if not keys:  # nothing paired (a wrong root or cohort): no verdict on the candidate
            result["verdict"] = "incomplete"
        elif "platform" in criteria_out and not criteria_out["platform"]["pass"]:
            result["verdict"] = "invalid"  # arms served on different platforms are no pair
        elif crossed:
            result["verdict"] = "kill"
        elif complete and all(v["value"] is not None for v in lines.values()):
            result["verdict"] = "pass"
        else:
            result["verdict"] = "incomplete"
        code = {side: sorted({str(e["client_code_sha256"]) for e in eps})
                for side, eps in (("canary", canary_eps), ("base", base_eps))}
        result["report_only"].update({
            "os_pct": {"canary": c["os_pct"], "base": b["os_pct"]},
            "os_delta_pt": _delta(c["os_pct"], b["os_pct"]),
            "client_code_sha256": code,
            # the ledger freezes the client from Phase-0 on: one client on both arms
            "same_client_code": len(code["canary"]) == 1 and code["canary"] == code["base"],
        })
    if bootstrap and keys:
        result["report_only"]["sr_bootstrap"] = sr_bootstrap(canary, base, keys, field=sr_field, **bootstrap)
    return result


def cohort_parts(cohort: Path) -> list[str]:
    """The ``part`` names of a cohort's shards (EXP-28's: heldout, train_pass1, ...); [] for EXP-25's."""
    manifest = json.loads((cohort / "manifest.json").read_text(encoding="utf-8"))
    return sorted({str(shard["part"]) for shard in manifest["shards"] if shard.get("part") is not None})


def cohort_episodes(cohort: Path, part: str | None = None) -> set[tuple[str, int]]:
    """The cohort's (scene, episode id), every shard's or only part ``part``'s."""
    manifest = json.loads((cohort / "manifest.json").read_text(encoding="utf-8"))
    shards = manifest["shards"]
    if part is not None:
        shards = [shard for shard in shards if shard.get("part") == part]
        if not shards:
            raise ValueError(f"{cohort}: no shard of part {part!r} (parts: {cohort_parts(cohort) or 'none'})")
    keys: set[tuple[str, int]] = set()
    for shard in shards:
        document = json.loads((cohort / shard["path"]).read_text(encoding="utf-8"))
        if part is not None and document.get("part") != part:
            raise ValueError(f"{cohort / shard['path']}: part {document.get('part')!r}, the manifest says {part!r}")
        for e in document["episodes"]:
            key = (str(e["scene_id"]), int(e["episode_id"]))
            if key in keys:
                raise ValueError(f"{cohort}: scene {key[0]} episode {key[1]} is listed twice")
            keys.add(key)
    return keys


# ---------------------------------------------------------------------------- per-call identity
def call_identity(rerun: EpisodeRecord, reference: EpisodeRecord) -> dict[str, Any]:
    """Call-by-call and outcome equality of one episode's rerun against its reference record."""
    a = {int(c["call_index"]): c for c in rerun.calls}
    b = {int(c["call_index"]): c for c in reference.calls}
    same = same_inputs = 0
    first, first_fields = None, []
    for index in range(max(list(a) + list(b), default=-1) + 1):
        x, y = a.get(index), b.get(index)
        both = x is not None and y is not None
        if both and all(x.get(f) == y.get(f) for f in DECISION_FIELDS):
            same += 1
        elif first is None:
            first = index
            first_fields = [f for f in DECISION_FIELDS if x.get(f) != y.get(f)] if both else ["missing"]
        same_inputs += both and all(x.get(f) == y.get(f) for f in INPUT_FIELDS)
    oa, ob = rerun.episode.get("outcome") or {}, reference.episode.get("outcome") or {}
    outcome_differs = [f for f in OUTCOME_FIELDS if oa.get(f) != ob.get(f)]
    meta = [f for f in META_FIELDS if rerun.episode.get(f) != reference.episode.get(f)]
    if (rerun.episode.get("server") or {}).get("model_version") != (reference.episode.get("server") or {}).get("model_version"):
        meta.append("server.model_version")
    return {
        "episode": episode_key(rerun.scene_id, rerun.episode_id),
        "calls": len(a),
        "reference_calls": len(b),
        "identical_calls": same,
        "identical_input_calls": int(same_inputs),
        "first_divergent_call": first,
        "first_divergent_fields": first_fields,
        "outcome_differs": outcome_differs,
        "identical": first is None and not outcome_differs,
        "device": record_device(rerun),
        "reference_device": record_device(reference),
        "meta_differs": meta,
    }


def identity_compare(rerun_root: Path, reference_root: Path, expected: set | None,
                     min_identical: int | None = None) -> dict[str, Any]:
    """How many episodes in scope the rerun reproduced call by call; a missing one is not reproduced."""
    rerun = _read_records(rerun_root)
    scope = set(expected) if expected is not None else set(rerun)
    missing_rerun = sorted(scope - set(rerun))
    missing_reference, episodes = [], []
    for scene, episode in sorted(scope & set(rerun)):
        path = record_path(reference_root, scene, episode)
        if not path.with_name(path.name + ".sha256").is_file():
            missing_reference.append((scene, episode))
            continue
        episodes.append(call_identity(EpisodeRecord(rerun[(scene, episode)]), EpisodeRecord(path)))
    need = len(scope) if min_identical is None else int(min_identical)
    identical = sum(e["identical"] for e in episodes)
    calls = sum(max(e["calls"], e["reference_calls"]) for e in episodes)
    same_calls = sum(e["identical_calls"] for e in episodes)
    devices = sorted({e["device"] for e in episodes} | {e["reference_device"] for e in episodes})
    same_platform = len(devices) == 1
    return {
        "schema": IDENTITY_SCHEMA,
        "scope_episodes": len(scope),
        "compared_episodes": len(episodes),
        "identical_episodes": identical,
        "min_identical_episodes": need,
        "identical_share": round(identical / len(scope), 6) if scope else None,
        "calls": calls,
        "identical_calls": same_calls,
        "identical_call_share": round(same_calls / calls, 6) if calls else None,
        "identical_input_calls": sum(e["identical_input_calls"] for e in episodes),
        "complete": not missing_rerun and not missing_reference,
        "missing_in_rerun": [episode_key(s, e) for s, e in missing_rerun][:50],
        "missing_in_reference": [episode_key(s, e) for s, e in missing_reference][:50],
        "devices": devices,
        "same_platform": same_platform,
        "meta_differs": dict(sorted(Counter(f for e in episodes for f in e["meta_differs"]).items())),
        "pass": bool(scope) and same_platform and identical >= need,
        "episodes": episodes,
    }


def _write(result: Mapping, output: str) -> None:
    if output:
        Path(output).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(result, indent=1, sort_keys=True))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--canary-records", default="", help="the candidate's records (with --base-records)")
    parser.add_argument("--base-records", default="", help="the base model's records of the same episodes")
    parser.add_argument("--rerun-records", default="",
                        help="per-call identity: records of a rerun with the same server code (with --reference-records)")
    parser.add_argument("--reference-records", default="", help="per-call identity: the original records")
    parser.add_argument("--cohort", default="", help="the cohort: every episode must be on both sides")
    parser.add_argument("--cohort-part", default=None,
                        help="only this part of the cohort (shard 'part'; P: heldout); needed when it has several")
    parser.add_argument("--output", default="", help="json (default: print only)")
    parser.add_argument("--criteria", choices=CRITERIA_SETS, default="exp25-ca",
                        help="exp25-ca (default) or exp28-p, the HM3D held-out proxy's kill lines")
    parser.add_argument("--bootstrap-draws", type=int, default=None,
                        help=f"paired ΔSR interval (report only); default 0, or {P_BOOTSTRAP_DRAWS} with exp28-p")
    parser.add_argument("--bootstrap-unit", choices=("scene", "episode"), default="scene",
                        help="resample whole scenes (default) or episodes")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--require-device", default=None,
                        help=f"every matched record on both sides served by this device (npu, cuda); "
                             f"default none, or {P_DEVICE} with exp28-p; 'none' turns it off")
    parser.add_argument("--min-identical-episodes", type=int, default=None,
                        help="per-call identity passes with this many episodes reproduced (default: all in scope)")
    args = parser.parse_args(argv)
    expected = None
    if args.cohort:
        parts = cohort_parts(Path(args.cohort))
        if args.cohort_part is None and len(parts) > 1:
            parser.error(f"{args.cohort} has parts {parts}: name one with --cohort-part (P: heldout)")
        expected = cohort_episodes(Path(args.cohort), args.cohort_part)
    elif args.cohort_part is not None:
        parser.error("--cohort-part needs --cohort")
    cohort_input = {"cohort": args.cohort or None}
    if args.cohort_part is not None:
        cohort_input["cohort_part"] = args.cohort_part

    if args.rerun_records or args.reference_records:
        if not (args.rerun_records and args.reference_records):
            parser.error("--rerun-records and --reference-records go together")
        if args.canary_records or args.base_records:
            parser.error("a per-call identity check takes no --canary-records / --base-records")
        if args.min_identical_episodes is not None and args.min_identical_episodes < 1:
            parser.error("--min-identical-episodes must be at least 1")
        result = identity_compare(Path(args.rerun_records), Path(args.reference_records), expected,
                                  args.min_identical_episodes)
        result["inputs"] = {"rerun_records": args.rerun_records, "reference_records": args.reference_records,
                            **cohort_input}
        print(("PASS" if result["pass"] else "FAIL")
              + f" per-call identity {result['identical_episodes']}/{result['scope_episodes']} episodes"
              f" (need {result['min_identical_episodes']}) calls {result['identical_calls']}/{result['calls']}"
              f" prompt bytes {result['identical_input_calls']}/{result['calls']} devices={result['devices']}"
              f" complete={result['complete']} meta_differs={result['meta_differs']}")
        _write(result, args.output)
        return 0 if result["pass"] else 1

    if not (args.canary_records and args.base_records):
        parser.error("--canary-records and --base-records are required")
    if args.min_identical_episodes is not None:
        parser.error("--min-identical-episodes belongs to the per-call identity check (--rerun-records)")
    p = args.criteria == "exp28-p"
    draws = args.bootstrap_draws if args.bootstrap_draws is not None else (P_BOOTSTRAP_DRAWS if p else 0)
    device = args.require_device if args.require_device is not None else (P_DEVICE if p else None)
    if device == "none":
        device = None
    canary = load(Path(args.canary_records))
    base = {}
    for scene, episode in sorted(expected if expected is not None else canary):
        path = record_path(args.base_records, scene, episode)
        if path.with_name(path.name + ".sha256").is_file():
            base[(scene, episode)] = episode_summary(EpisodeRecord(path))
    bootstrap = {"unit": args.bootstrap_unit, "draws": draws, "seed": args.bootstrap_seed} if draws > 0 else None
    result = compare(canary, base, expected, criteria=args.criteria, require_device=device, bootstrap=bootstrap)
    result["inputs"] = {"canary_records": args.canary_records, "base_records": args.base_records, **cohort_input}
    if p:  # kill / pass / incomplete / invalid, not C-A's PASS/FAIL
        line = f"{result['verdict'].upper()} P matched={result['matched_episodes']}"
    else:
        line = ("PASS" if result["all_pass"] else "FAIL") + f" C-A matched={result['matched_episodes']}"
    line += f" complete={result['complete']} " + " ".join(
        f"{k}={v['value']}(base {v['base']}, line {v['direction']}{v['threshold']}){'' if v['pass'] else ' FAIL'}"
        for k, v in result["criteria"].items())
    interval = result["report_only"].get("sr_bootstrap")
    if interval:
        line += (f" | ΔSR {interval['diff_pt']:+.2f} pt [{interval['ci95_pt'][0]:+.2f}, {interval['ci95_pt'][1]:+.2f}]"
                 f" ({interval['unit']} bootstrap, {interval['clusters']} scenes, {interval['draws']} draws)")
    print(line)
    if p and not result["report_only"]["same_client_code"]:
        print(f"warning: the arms ran other client code {result['report_only']['client_code_sha256']}; "
              "EXP-28 E freezes the client from Phase-0 on", file=sys.stderr)
    if args.output:
        _write(result, args.output)
    else:
        print(json.dumps(result, indent=1, sort_keys=True))
    if p:
        return {"pass": 0, "kill": 1, "incomplete": 2, "invalid": 3}[result["verdict"]]
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

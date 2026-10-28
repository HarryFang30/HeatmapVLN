#!/usr/bin/env python3
"""EXP-37's offline gate: G1 and G2 over the X37 adapter, and the pre-registered next step.

The ledger's EXP-37 判据 (written 2026-10-28, before any X37 training, gate or val reading).  After training
and before val, two items gate the candidate; everything else here is report-only:

====  ===========================================================================  ==========================
G1    held-out preference pairs (scripts/exp37/eval_pairs.py summarize): implicit  accuracy >= 0.60 and the
      reward accuracy, the share with [s_pol(c) - s_ref(c)] - [s_pol(r) - s_ref(r)]  episode-cluster bootstrap
      > 0                                                                          lower bound > 0.50
G2    HM3D-20 held-out anchor states (base agrees with line B, d > 3 m; 3,529),    class kept >= 0.92
      decoded the way X37 is deployed (``hm3d_route_gate.py decode --on-mode
      base-stop``): the on answer keeps the base's decision class (EXP-28 O4
      anchor, ``hm3d_route_gate.summarize``)
====  ===========================================================================  ==========================

Report-only: O4 correction (on agree - off agree on the corrections, line B, with its episode bootstrap CI),
O5 (d_shuf / d_corr under shuffled instructions), L (instruction sensitivity, on vs off, EXP-32's reading), the
far STOP counts and who decided the on answers' STOPs, and G1's policy-only and reference-only accuracies.

Next step ("重试只有一次"): both pass -> ``val`` (after the zero-adapter equivalence check of 设置); on the first
attempt, G1 fails and the training pairs' reward accuracy is under 0.70 (underfit) -> ``retry_lr_5e-5`` (retrain
from the base at lr 5e-5, gate G1 and G2 again as attempt 2); any other failure, or any failure of attempt 2 ->
``no_val``.  The retry's condition reads G1 and the training accuracy only, as the ledger writes it: G2 failing
beside an underfit G1 still gets the retry (attempt 2 is gated on G2 again); G2 failing with G1 passed is
"其他任何不过", ``no_val``.  G1's accuracy and G2's keep are compared in whole counts (G1's count is recovered
from eval_pairs.py's accuracy and pair count), so a value on its line passes ">=" and fails ">"; G1's lower bound
is the bootstrap quantile eval_pairs.py wrote.  The training pairs' reward accuracy is ``eval_pairs.py`` on the
candidate's own training pair stream (``--train-pairs``; the verdict records its pair count as
``train_reward_accuracy_pairs``, since run_gate_910b.sh scores only the first X37_TRAIN_PAIRS_MAX training pairs),
or a number read elsewhere (``--train-reward-accuracy``).

Exit codes as ``scripts/exp35/stop_rate_gate.py``: 0 every gate passed; 1 a gate failed (``retry_lr_5e-5`` or
``no_val``); 2 invalid -- an input is unreadable, of another adapter, or not the deployed decode (``hm3d.json``
not valid, not ``--on-mode base-stop``, a smoke subset), or G1 failed on attempt 1 without the training reward
accuracy that picks the retry.  An invalid reading is never a retry: fix the inputs and run it again.

Inputs: G2 and the report-only items from ``hm3d_route_gate.py decode --on-mode base-stop --shuffle-instructions``
(910B, 8 cards) and ``score`` (4090 CPU), see that module's doc; G1 from ``eval_pairs.py score`` / ``summarize``
on the held-out pair stream.  No model here::

    python scripts/exp37/gate_verdict.py --hrg $G/o4/hm3d.json --pairs $G/g1/heldout_summary.json \\
        --train-pairs $G/g1/train_summary.json --adapter $R/exp37/train/X37/adapter --attempt 1 \\
        --output $G/verdict.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.exp28 import hm3d_route_gate as hg  # noqa: E402

SCHEMA = "heatmapvln-exp37-offline-gate-verdict-v1"
PAIR_SCHEMA = "heatmapvln-exp37-g1-v1"  # scripts/exp37/eval_pairs.py summarize
RULES = "ledger EXP-37 判据, written 2026-10-28 before any X37 training, gate or val reading"
ON_MODE = "base-stop"  # the deployed decode (hm3d_route_gate.py decode --on-mode base-stop)
G1_MIN_ACCURACY = Fraction(60, 100)
G1_LOWER_ABOVE = 0.50
G2_KEEP_MIN = hg.O4_ANCHOR_KEEP_MIN  # 0.92, EXP-28 O4 anchor's line
RETRY_BELOW_TRAIN_ACCURACY = Fraction(70, 100)
LEDGER_SIZES = {"G1": 1475, "G2": 3529}  # 判据: "约 1,475 对", "3,529 个"; a different size is a note
GROUPS = ("G1", "G2")
NEXT_RULES = {
    "val": "G1 and G2 passed: the zero-adapter equivalence check (设置 等价检查), then val",
    "retry_lr_5e-5": "attempt 1: G1 failed and the training pairs' reward accuracy is under 0.70 (underfit), "
                     "whatever G2 read: retrain once from the base at lr 5e-5 and gate G1 and G2 again as attempt 2",
    "no_val": "a gate failed and the one retry does not apply (G2 failed with G1 passed; G1 failed although the "
              "training pairs were fitted, reward accuracy >= 0.70; or attempt 2 failed): the candidate does not "
              "go to val",
    "invalid": "a gate could not be read (see notes): fix the inputs and run this verdict again; it is not a retry",
}


class VerdictInvalid(ValueError):
    """An input from which no pass or fail may be read."""


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def pair_reading(summary: Mapping[str, Any], what: str) -> dict[str, Any]:
    """The numbers G1 (or the retry rule) reads from ``eval_pairs.py summarize``'s JSON (``PAIR_SCHEMA``).

    The count of pairs with a positive margin is recovered from ``reward_accuracy`` (rounded to 6 places) and
    ``pairs``, so G1's accuracy line is compared in whole counts; a summary whose accuracy is no count over its
    pairs, or whose own ``g1.pass`` disagrees with the reading here, is refused.
    """
    if summary.get("schema") != PAIR_SCHEMA:
        raise VerdictInvalid(f"{what} is not eval_pairs.py summarize's output (schema {summary.get('schema')!r})")
    if not summary.get("valid"):
        raise VerdictInvalid(f"{what} is not valid: {summary.get('error')}")
    n, accuracy, ci = summary.get("pairs"), summary.get("reward_accuracy"), summary.get("reward_accuracy_ci95")
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise VerdictInvalid(f"{what}: no pair count (pairs = {n!r})")
    if not _number(accuracy) or not 0.0 <= accuracy <= 1.0:
        raise VerdictInvalid(f"{what}: no reward accuracy ({accuracy!r})")
    if not isinstance(ci, (list, tuple)) or len(ci) != 2 or not all(_number(v) for v in ci):
        raise VerdictInvalid(f"{what}: no episode-cluster bootstrap interval ({ci!r})")
    correct = round(float(accuracy) * n)
    if abs(correct / n - float(accuracy)) > 1e-6:
        raise VerdictInvalid(f"{what}: reward accuracy {accuracy} is no count over {n} pairs")
    return {
        "n": n,
        "correct": correct,
        "accuracy": Fraction(correct, n),
        "lower": float(ci[0]),
        "policy_accuracy": summary.get("policy_accuracy"),
        "reference_accuracy": summary.get("reference_accuracy"),
        "episodes": summary.get("episodes"),
        "adapter_sha256": summary.get("adapter_sha256"),
        "own_pass": (summary.get("g1") or {}).get("pass"),
    }


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
def gate_g1(pairs: Mapping[str, Any]) -> dict[str, Any]:
    """G1 on ``pair_reading``: accuracy >= 0.60 and the episode-cluster lower bound > 0.50."""
    accuracy = pairs["accuracy"]
    ok_accuracy = accuracy >= G1_MIN_ACCURACY
    ok_lower = pairs["lower"] > G1_LOWER_ABOVE
    if pairs["own_pass"] is not None and bool(pairs["own_pass"]) != bool(ok_accuracy and ok_lower):
        raise VerdictInvalid(f"G1: eval_pairs.py read pass={pairs['own_pass']}, this reading "
                             f"{bool(ok_accuracy and ok_lower)} (accuracy {pairs['correct']}/{pairs['n']}, "
                             f"lower bound {pairs['lower']})")
    return {
        "value": round(float(accuracy), 6),
        "threshold": float(G1_MIN_ACCURACY),
        "direction": ">=",
        "lower_bound": pairs["lower"],
        "lower_bound_above": G1_LOWER_ABOVE,
        "pass_accuracy": bool(ok_accuracy),
        "pass_lower_bound": bool(ok_lower),
        "pass": bool(ok_accuracy and ok_lower),
        "n": pairs["n"],
        "correct": pairs["correct"],
        "episodes": pairs["episodes"],
        # report-only beside G1: the margin's two halves read alone
        "policy_accuracy": pairs["policy_accuracy"],
        "reference_accuracy": pairs["reference_accuracy"],
    }


def gate_g2(hrg: Mapping[str, Any]) -> dict[str, Any]:
    """G2: O4's anchor class keep (``hm3d_route_gate.summarize``) of the base-stop decode, >= 0.92 in counts."""
    anchor = ((hrg.get("O4") or {}).get("criteria") or {}).get("anchor_class_keep") or {}
    n, kept = anchor.get("n"), anchor.get("kept")
    if not isinstance(n, int) or not isinstance(kept, int) or n <= 0 or not 0 <= kept <= n:
        raise VerdictInvalid(f"G2: hm3d.json has no anchor class keep to read ({dict(anchor)})")
    keep = Fraction(kept, n)
    return {"value": round(float(keep), 6), "threshold": float(G2_KEEP_MIN), "direction": ">=",
            "pass": bool(keep >= G2_KEEP_MIN), "n": n, "kept": kept, "on_mode": ON_MODE}


def hrg_problems(hrg: Mapping[str, Any]) -> list[str]:
    """Why ``hm3d.json`` is not G2's measurement (empty: it is)."""
    problems = []
    if hrg.get("schema") != hg.SCHEMA:
        problems.append(f"--hrg is not hm3d_route_gate.py score's output (schema {hrg.get('schema')!r})")
    if not hrg.get("valid"):
        problems.append(f"hm3d.json is not valid: {hrg.get('invalid')}")
    mode = (hrg.get("join") or {}).get("on_mode")
    if mode != ON_MODE:
        problems.append(f"hm3d.json scored on-mode {mode!r}: G2 reads the deployed decode (decode --on-mode {ON_MODE})")
    subset = ((hrg.get("selection") or {}).get("settings") or {}).get("max_episodes")
    if subset:
        problems.append(f"hm3d.json ran on a seeded subset of {subset} episodes (--max-episodes): a smoke run")
    return problems


def report_only(hrg: Mapping[str, Any], rows: Sequence[Mapping[str, Any]] | None) -> tuple[dict[str, Any], list[str]]:
    """O4 correction, O5, L, the far STOP item and the on answers' STOP sources, with notes on what is missing."""
    notes: list[str] = []
    criteria = (hrg.get("O4") or {}).get("criteria") or {}
    corr = criteria.get("correction_agree_gain") or {}
    o5 = ((hrg.get("O5") or {}).get("criteria") or {}).get("shuffle_vs_own_instruction") or {}
    out: dict[str, Any] = {
        "O4_correction": {key: corr.get(key) for key in ("value", "n", "on_rate", "base_rate", "on_agree",
                                                         "base_agree", "on_unjudged", "base_unjudged",
                                                         "ci95_episode_bootstrap")},
        "O5": {"ratio": o5.get("ratio"), "delta_shuf": o5.get("value"), "delta_corr": o5.get("delta_corr"),
               "ratio_max": o5.get("ratio_max"), "exp28_line_pass": o5.get("pass"), "n": o5.get("n"),
               "delta_corr_nonpositive": o5.get("delta_corr_nonpositive"),
               "ci95_delta_shuf": o5.get("ci95_delta_shuf")},
        "far_stop": criteria.get("far_stop_vs_base"),
        "on_stop_source": (hrg.get("report_only") or {}).get("on_stop_source"),
        "L": None,
    }
    corr_n = corr.get("n")
    if o5.get("delta_corr_nonpositive"):
        notes.append("O5 (report-only) compares against delta_corr <= 0: its ratio says nothing about language")
    if rows is None:
        notes.append("L not read: no rows file beside hm3d.json")
    else:
        lang = hg.language_gate(rows)
        out["L"] = {key: lang.get(key) for key in ("value", "base_rate", "delta", "threshold", "pass", "n",
                                                   "on_different", "base_different", "on_different_by",
                                                   "base_different_by")}
        out["L"]["exp32_line_pass"] = out["L"].pop("pass")
        notes += [f"report-only {p}" for p in hg._l_problems(lang, hrg, rows)]
    if corr_n is not None and not corr_n:
        notes.append("O4 correction: no correction state")
    return out, notes


def next_step(groups: Mapping[str, bool], attempt: int, *, train_accuracy: Fraction | float | None) -> str:
    """The pre-registered consequence of one gate reading (ledger EXP-37 "重试只有一次")."""
    if all(groups[g] for g in GROUPS):
        return "val"
    if attempt >= 2 or groups["G1"]:
        return "no_val"
    # G1 failed on the first attempt (G2 either way): underfit (training reward accuracy < 0.70) gets the one retry
    if train_accuracy is None:
        return "invalid"
    if not isinstance(train_accuracy, Fraction):  # a number as written: 0.70 is on the line, not under it
        train_accuracy = Fraction(str(float(train_accuracy)))
    return "retry_lr_5e-5" if train_accuracy < RETRY_BELOW_TRAIN_ACCURACY else "no_val"


def verdict(*, hrg: Mapping[str, Any], pairs_summary: Mapping[str, Any], attempt: int,
            rows: Sequence[Mapping[str, Any]] | None = None, train_summary: Mapping[str, Any] | None = None,
            train_accuracy: float | None = None, adapter_sha256: str | None = None) -> dict[str, Any]:
    """The verdict JSON.  ``hrg``: ``hm3d_route_gate.py score``'s hm3d.json; ``rows``: its calls.jsonl rows (L);
    ``pairs_summary``: ``eval_pairs.py summarize`` on the held-out pairs (G1); ``train_summary`` or
    ``train_accuracy``: the training pairs' reward accuracy (the retry rule); ``adapter_sha256``: the candidate."""
    problems: list[str] = []
    notes: list[str] = []
    g1 = g2 = None
    pairs = train = None
    try:
        pairs = pair_reading(pairs_summary, "--pairs")
        g1 = gate_g1(pairs)
    except VerdictInvalid as err:
        problems.append(str(err))
    hrg_bad = hrg_problems(hrg)
    problems += hrg_bad
    try:
        g2 = gate_g2(hrg)
    except VerdictInvalid as err:
        if not hrg_bad:
            problems.append(str(err))
    if train_summary is not None:
        try:
            train = pair_reading(train_summary, "--train-pairs")
            train_accuracy = train["accuracy"]
        except VerdictInvalid as err:
            problems.append(str(err))
    adapters = {"hrg": (hrg.get("join") or {}).get("adapter_sha256"),
                "pairs": pairs["adapter_sha256"] if pairs else None,
                "train_pairs": train["adapter_sha256"] if train else None, "candidate": adapter_sha256}
    named = {k: v for k, v in adapters.items() if v}
    if len(set(named.values())) > 1:
        problems.append(f"the inputs name different adapters: {named}")
    for name, read in (("hrg", not hrg_bad), ("pairs", pairs is not None)):
        if read and not adapters[name]:
            notes.append(f"{name}: the input names no adapter sha256, so it is not checked against the others")

    groups = {"G1": bool(g1 and g1["pass"]), "G2": bool(g2 and g2["pass"])}
    step = "invalid" if problems else next_step(groups, attempt, train_accuracy=train_accuracy)
    if step == "invalid" and not problems:
        problems.append("G1 failed on attempt 1 and the training pairs' reward accuracy was not given "
                        "(--train-pairs or --train-reward-accuracy): the retry rule cannot be read")
    measured = not problems
    extra, extra_notes = report_only(hrg, rows) if not hrg_bad else ({}, [])
    notes = problems + notes + extra_notes
    for name, gate in (("G1", g1), ("G2", g2)):
        if gate and gate["n"] != LEDGER_SIZES[name]:
            notes.append(f"{name}: {gate['n']} {'pairs' if name == 'G1' else 'anchor states'}, the ledger's "
                         f"about {LEDGER_SIZES[name]}")
    return {
        "schema": SCHEMA,
        "rules": RULES,
        "attempt": attempt,
        "groups": groups,
        "all_pass": measured and all(groups.values()),
        "measured": measured,
        "next": step,
        "next_rule": NEXT_RULES[step],
        "G1": g1,
        "G2": g2,
        "train_reward_accuracy": None if train_accuracy is None else round(float(train_accuracy), 6),
        "train_reward_accuracy_pairs": train["n"] if train else None,  # None: given by hand or not read
        "retry_below_train_accuracy": float(RETRY_BELOW_TRAIN_ACCURACY),
        "report_only": extra,
        "adapter_sha256": adapters,
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hrg", required=True,
                        help="hm3d_route_gate.py score's hm3d.json (decoded with --on-mode base-stop)")
    parser.add_argument("--hrg-calls", default="", help="its rows, for L (default: <hrg stem>.calls.jsonl)")
    parser.add_argument("--pairs", required=True, help="eval_pairs.py summarize's JSON on the held-out pairs (G1)")
    train = parser.add_mutually_exclusive_group()
    train.add_argument("--train-pairs", default="",
                       help="eval_pairs.py summarize's JSON on training pairs (the retry rule)")
    train.add_argument("--train-reward-accuracy", type=float, default=None,
                       help="the training pairs' reward accuracy, read elsewhere (retry rule)")
    parser.add_argument("--adapter", default="", help="the candidate's adapter directory: every input must be of it")
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1, help="2: the one retry (lr 5e-5)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if not args.hrg_calls:
        args.hrg_calls = str(Path(args.hrg).with_suffix(".calls.jsonl"))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    def load(path: str) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    try:
        hrg = load(args.hrg)
        calls = Path(args.hrg_calls)
        rows = hg.read_rows(calls) if calls.is_file() else None
        result = verdict(
            hrg=hrg, pairs_summary=load(args.pairs), attempt=args.attempt, rows=rows,
            train_summary=load(args.train_pairs) if args.train_pairs else None,
            train_accuracy=args.train_reward_accuracy,
            adapter_sha256=_sha256(Path(args.adapter) / "adapter_model.safetensors") if args.adapter else None,
        )
    except (ValueError, KeyError, TypeError, OSError) as err:  # json errors are ValueErrors
        # A missing or malformed input: nothing was judged (exit 2, never the FAIL of exit 1).
        print(f"INVALID EXP-37 offline gate attempt {args.attempt}: {type(err).__name__}: {err}", flush=True)
        return 2
    result["inputs"] = {"hrg": str(Path(args.hrg).resolve()), "hrg_calls": str(Path(args.hrg_calls).resolve()),
                        "pairs": str(Path(args.pairs).resolve()),
                        "train_pairs": str(Path(args.train_pairs).resolve()) if args.train_pairs else None,
                        "adapter": str(Path(args.adapter).resolve()) if args.adapter else None}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.write_text(json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(output)
    groups = " ".join(f"{k}={'pass' if v else 'FAIL'}" for k, v in result["groups"].items())
    head = "PASS" if result["all_pass"] else ("FAIL" if result["measured"] else "INVALID")
    print(f"{head} EXP-37 offline gate attempt {args.attempt}: {groups} -> next: {result['next']}", flush=True)
    for note in result["notes"]:
        print(f"[verdict] {note}", file=sys.stderr)
    return 0 if result["all_pass"] else (1 if result["measured"] else 2)


if __name__ == "__main__":
    raise SystemExit(main())

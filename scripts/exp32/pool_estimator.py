#!/usr/bin/env python3
"""EXP-32 J: the val_unseen reference pool, the estimator Ê_k and its pre-registered decisions (4090, CPU).

The reference pool is EXP-20's eight sdpa full runs, arms A0-A3 x System1 seeds 42 and 1337
(``--ref-template``).  Every number the ledger wrote about the pool -- the eight runs' SR, the seed means
62.02 / 61.20, the pool mean 61.61 and its SD 0.80 -- is recomputed from the files under the standard
protocol and must be within 0.01 of the ledger, or nothing is written (停查).  Every run, the pool's and
the candidate's, must hold exactly 1,839 episodes, the same ones, every row naming the protocol seed the
run is declared under.

``pool-ref --seed 42|1337 --out pool_ref_seed42.jsonl`` writes the per-episode mean of that seed's four
arms (``success``, ``spl``, ``os``, ``ne``) as a progress.jsonl that the unchanged
``paired_progress_monitor.py --reference`` reads for the 600-episode rule.  The rows carry no ``steps``,
so the monitor's report-only ``sr_step_budget`` compares a switched-on run against the pool without the
budget estimate.

``estimate --candidate 42=<run> [--candidate 1337=<run> ...]`` takes the candidate's finished seeds in
the ledger's order 42, 1337, 2024, 3407 (a run directory or its merged/progress.jsonl), scores them under
the standard protocol (``paired_progress_monitor.standard``: a step-budget STOP is no STOP) and applies
the rules stage by stage, k = 1 .. the number of seeds (SR in points):

* Ê_k = (1/k) Σ_s (SR_s − ref_s) with the ledger's ref = 62.02 / 61.20 / 61.61 / 61.61;
* SE_k = √(max(0.80, s_c)²/k + 0.41²), s_c the sample SD of the candidate's k SRs (k ≥ 2);
* k=1: Ê_1 ≤ −0.5 → stop, not supported (futility);
* k=2: Ê_2 ≥ max(1.0, 3·SE_2) and the pooled paired ΔSR of seeds 42+1337 against A0 > 0 → supported
  (two seeds, early); else Ê_2 < 0.3 → stop, not supported if Ê_2 ≤ 0, otherwise not detected; else
  run 2024 and 3407;
* k=3: no rule, run 3407;
* k=4: Ê_4 ≥ 1.0 and Ê_4 − 1.96·SE_4 > 0 and the paired ΔSR > 0 → H-32 supported; Ê_4 ≤ 0 → refuted;
  otherwise not detected.  There is no fifth seed.

The first stage that ends the experiment is the decision; seeds run past it are reported, not judged.
The paired comparison with A0 is ``paired_closed_loop_bootstrap.py --standard-protocol``'s (its functions,
draw order and seed, so the CLI on the same files prints the same numbers), per seed and for 42+1337
pooled, SR / SPL / OS / NE; only the pooled ΔSR's sign enters a decision.  Report only as well: the
switched-on SR, the gap to A0's two-seed mean, and an episode-level bootstrap of the candidate against the
per-episode pool (the seed's four arms for 42 and 1337, all eight runs for 2024 and 3407; it leaves out the
pool's own seed noise, which SE_k carries).

Prints the decision line first and writes ``--out``.  Exit status: 0 = run the next seed, 3 = a terminal
decision, 1 = refused::

    P=/opt/conda/bin/python; E=scripts/exp32/pool_estimator.py
    $P $E pool-ref --seed 42 --out /workspace/exp32/pool_ref_seed42.jsonl
    $P $E estimate --candidate 42=/workspace/eval_runs/exp32_X32R1_seed42_4090_stepbudget \\
        --candidate 1337=/workspace/eval_runs/exp32_X32R1_seed1337_4090_stepbudget --out <dir>/estimate_k2.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.tools import paired_closed_loop_bootstrap as pb  # noqa: E402

EPISODES = 1839
ARMS = ("a0", "a1", "a2", "a3")
POOL_SEEDS = (42, 1337)
SEED_ORDER = (42, 1337, 2024, 3407)
REF_TEMPLATE = "/workspace/eval_runs/exp20_{arm}_seed{seed}_4090/merged/progress.jsonl"
# The ledger's numbers (EXP-32 J, 参照池), standard-protocol SR in points.
LEDGER_RUNS = {("a0", 42): 62.70, ("a1", 42): 61.45, ("a2", 42): 60.96, ("a3", 42): 62.97,
               ("a0", 1337): 60.74, ("a1", 1337): 61.34, ("a2", 1337): 61.50, ("a3", 1337): 61.23}
LEDGER_SEED_MEANS = {42: 62.02, 1337: 61.20}
LEDGER_POOL_MEAN, LEDGER_POOL_SD = 61.61, 0.80
TOLERANCE = 0.01
REF = {42: 62.02, 1337: 61.20, 2024: 61.61, 3407: 61.61}
SIGMA_FLOOR, POOL_SE = 0.80, 0.41
FUTILITY_K1 = -0.5
STOP_K2 = 0.3
EARLY_MIN, EARLY_SE = 1.0, 3.0
SUPPORT_MIN, SUPPORT_Z = 1.0, 1.96
VERDICTS = {
    "continue": "继续",
    "not_supported_futility": "未支持（无效性）",
    "supported_early": "支持（两种子提前）",
    "not_supported": "未支持",
    "not_detected": "没测出来",
    "supported": "H-32 支持",
    "refuted": "否定",
}


def run_path(path: str | Path) -> Path:
    path = Path(path)
    return path / "merged" / "progress.jsonl" if path.is_dir() else path


def load_run(path: Path, seed: int, label: str) -> dict:
    """A finished run keyed as ``paired_closed_loop_bootstrap`` keys it; refused unless 1,839 rows of ``seed``."""
    rows = pb.load_progress(path)
    if len(rows) != EPISODES:
        raise SystemExit(f"{label} ({path}): {len(rows)} episodes; a finished val_unseen run has {EPISODES}")
    seeds = {row.get("rpc_protocol_seed") for row in rows.values()}
    if seeds != {seed}:
        raise SystemExit(f"{label} ({path}): rows name protocol seed(s) {sorted(map(str, seeds))}, declared {seed}")
    return rows


def sr(rows: dict) -> float:
    """Standard-protocol SR in points."""
    return 100.0 * sum(float(pb.standard_row(row)["success"]) for row in rows.values()) / len(rows)


def sample_sd(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def load_pool(template: str) -> dict:
    pool = {(arm, seed): load_run(run_path(template.format(arm=arm, seed=seed)), seed, f"pool run {arm} seed {seed}")
            for seed in POOL_SEEDS for arm in ARMS}
    keys = set(pool[(ARMS[0], POOL_SEEDS[0])])
    for (arm, seed), rows in pool.items():
        if set(rows) != keys:
            raise SystemExit(f"pool run {arm} seed {seed} holds other episodes than {ARMS[0]} seed {POOL_SEEDS[0]}")
    return pool


def check_pool(pool: dict, template: str = "") -> dict:
    """Recompute what the ledger wrote about the pool; refuse (停查) on any difference > 0.01."""
    runs = {name: sr(rows) for name, rows in pool.items()}
    seed_means = {seed: sum(runs[(arm, seed)] for arm in ARMS) / len(ARMS) for seed in POOL_SEEDS}
    mean = sum(runs.values()) / len(runs)
    sd = sample_sd(list(runs.values()))
    checks = [(f"run {arm} seed {seed}", runs[(arm, seed)], LEDGER_RUNS[(arm, seed)]) for arm, seed in runs]
    checks += [(f"seed {seed} mean", seed_means[seed], LEDGER_SEED_MEANS[seed]) for seed in POOL_SEEDS]
    checks += [("pool mean", mean, LEDGER_POOL_MEAN), ("pool SD", sd, LEDGER_POOL_SD)]
    off = [f"{name} {got:.4f} vs ledger {want:.2f}" for name, got, want in checks if abs(got - want) > TOLERANCE]
    if off:
        raise SystemExit("the reference pool does not reproduce the ledger's numbers (停查): " + "; ".join(off))
    return {"template": template, "tolerance_pt": TOLERANCE,
            "runs_pt": {f"{arm}_seed{seed}": round(v, 4) for (arm, seed), v in runs.items()},
            "seed_means_pt": {str(s): round(v, 4) for s, v in seed_means.items()},
            "mean_pt": round(mean, 4), "sd_pt": round(sd, 4),
            "ledger": {"seed_means_pt": {str(s): v for s, v in LEDGER_SEED_MEANS.items()},
                       "mean_pt": LEDGER_POOL_MEAN, "sd_pt": LEDGER_POOL_SD}}


def pool_rows(pool: dict, seeds: Sequence[int]) -> dict:
    """Per-episode mean of the runs of ``seeds`` (standard protocol), as progress rows in episode order."""
    runs = [pool[(arm, seed)] for seed in seeds for arm in ARMS]
    out = {}
    for key in sorted(runs[0], key=lambda k: (int(k[1]), k[0])):
        rows = [pb.standard_row(run[key]) for run in runs]
        row = {"scene_id": runs[0][key]["scene_id"], "episode_id": runs[0][key]["episode_id"]}
        row.update({metric: sum(float(r[metric]) for r in rows) / len(rows) for metric in pb.METRICS})
        if len(seeds) == 1:
            row["rpc_protocol_seed"] = seeds[0]
        row["pool_runs"] = [f"{arm}_seed{seed}" for seed in seeds for arm in ARMS]
        out[key] = row
    return out


def standard_error(k: int, s_c: float | None) -> float:
    sigma = SIGMA_FLOOR if s_c is None else max(SIGMA_FLOOR, s_c)
    return math.sqrt(sigma ** 2 / k + POOL_SE ** 2)


def decide(k: int, e_hat: float, s_c: float | None = None, paired_delta: float | None = None) -> dict:
    """The ledger's rule at stage ``k`` (points); ``paired_delta`` is the pooled ΔSR of seeds 42+1337 vs A0."""
    se = standard_error(k, s_c) if k >= 2 else None
    lines: dict = {}
    paired_up = paired_delta is not None and paired_delta > 0

    def stage(verdict: str, rule: str, next_seeds: Sequence[int] = ()) -> dict:
        return {"k": k, "e_hat": e_hat, "s_c": s_c, "se": se, **lines, "paired_delta_sr": paired_delta,
                "terminal": verdict != "continue", "verdict": verdict, "ledger": VERDICTS[verdict], "rule": rule,
                "next_seeds": list(next_seeds)}

    if k == 1:
        if e_hat <= FUTILITY_K1:
            return stage("not_supported_futility", "Ê_1 = SR_42 − 62.02 ≤ −0.5")
        return stage("continue", "Ê_1 > −0.5", (1337,))
    if k == 2:
        lines["early_line"] = max(EARLY_MIN, EARLY_SE * se)
        if e_hat >= lines["early_line"] and paired_up:
            return stage("supported_early", "Ê_2 ≥ max(1.0, 3·SE_2) and pooled paired ΔSR vs A0 > 0")
        if e_hat < STOP_K2:
            return stage("not_supported" if e_hat <= 0 else "not_detected", "Ê_2 < 0.3 (≤ 0: not supported)")
        return stage("continue", "0.3 ≤ Ê_2 and not the early line", (2024, 3407))
    if k == 3:
        return stage("continue", "no rule at k=3", (3407,))
    if k == 4:
        lines["lower_95"] = e_hat - SUPPORT_Z * se
        if e_hat >= SUPPORT_MIN and lines["lower_95"] > 0 and paired_up:
            return stage("supported", "Ê_4 ≥ 1.0, Ê_4 − 1.96·SE_4 > 0, pooled paired ΔSR vs A0 > 0")
        if e_hat <= 0:
            return stage("refuted", "Ê_4 ≤ 0")
        return stage("not_detected", "neither supported nor Ê_4 ≤ 0")
    raise ValueError(f"k = {k}: the ledger runs at most four seeds")


def _points(entry: dict, metric: str) -> dict:
    scale = 1.0 if metric == "ne" else 100.0
    return {"candidate": round(entry["treatment"] * scale, 4), "a0": round(entry["control"] * scale, 4),
            "delta": round(entry["diff"] * scale, 4), "ci95": [round(v * scale, 4) for v in entry["ci95"]]}


def paired_with_a0(candidates: dict, pool: dict, seeds: Sequence[int], draws: int, seed: int) -> tuple[dict, float]:
    """``paired_closed_loop_bootstrap.py --standard-protocol`` over (candidate seed s, A0 seed s) for ``seeds``."""
    treatment, control, ids = {}, {}, []
    for index, s in enumerate(seeds):
        t_rows, c_rows, pair_ids = pb.pair_rows(candidates[s], pool[("a0", s)], Path(f"candidate_seed{s}"), Path(f"a0_seed{s}"))
        for scene, episode in pair_ids:
            key = (index, scene, episode)
            treatment[key] = pb.standard_row(t_rows[(scene, episode)])
            control[key] = pb.standard_row(c_rows[(scene, episode)])
            ids.append(key)
    raw = pb.compare(treatment, control, ids, random.Random(seed), draws)
    report = {"seeds": list(seeds), "n": raw["n"], **{m: _points(raw[m], m) for m in pb.METRICS}}
    return report, 100.0 * raw["success"]["diff"]


def pool_bootstrap(candidates: dict, pool: dict, seeds: Sequence[int], draws: int, seed: int) -> dict:
    """Report only: candidate minus the per-episode pool, averaged over ``seeds``, episodes resampled."""
    refs = {s: pool_rows(pool, (s,) if s in POOL_SEEDS else POOL_SEEDS) for s in seeds}
    keys = list(refs[seeds[0]])
    diffs = [sum(float(pb.standard_row(candidates[s][key])["success"]) - refs[s][key]["success"] for s in seeds) / len(seeds)
             for key in keys]
    lo, hi = pb.bootstrap_ci(diffs, random.Random(seed), draws)
    return {"seeds": list(seeds), "episodes": len(keys), "delta_pt": round(100.0 * pb.mean(diffs), 4),
            "ci95_pt": [round(100.0 * lo, 4), round(100.0 * hi, 4)],
            "reference": "per-episode mean of the seed's four arms (2024, 3407: all eight runs)"}


def parse_candidates(specs: Sequence[str]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for spec in specs:
        seed, sep, path = spec.partition("=")
        if not sep or not seed.strip().isdigit() or not path:
            raise SystemExit(f"--candidate {spec!r}: expected SEED=RUN (run directory or merged/progress.jsonl)")
        if int(seed) in out:
            raise SystemExit(f"--candidate: seed {seed} given twice")
        out[int(seed)] = run_path(path)
    seeds = tuple(sorted(out, key=lambda s: SEED_ORDER.index(s) if s in SEED_ORDER else len(SEED_ORDER)))
    if seeds != SEED_ORDER[:len(seeds)]:
        raise SystemExit(f"candidate seeds {sorted(out)}: the ledger runs them in the order {SEED_ORDER}, "
                         f"so k seeds are the first k of it")
    return {s: out[s] for s in seeds}


def pool_ref(args: argparse.Namespace) -> int:
    pool = load_pool(args.ref_template)
    check_pool(pool, args.ref_template)
    rows = pool_rows(pool, (args.seed,))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows.values()), encoding="utf-8")
    summary = {"seed": args.seed, "episodes": len(rows), "out": str(out), "ledger_ref_pt": REF[args.seed],
               "sr_pt": round(100.0 * sum(row["success"] for row in rows.values()) / len(rows), 4)}
    print(json.dumps(summary, sort_keys=True))
    return 0


def _rounded(stage: dict) -> dict:
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in stage.items()}


def estimate(args: argparse.Namespace) -> int:
    paths = parse_candidates(args.candidate)
    seeds = list(paths)
    pool = load_pool(args.ref_template)
    reference = check_pool(pool, args.ref_template)
    keys = set(pool[(ARMS[0], POOL_SEEDS[0])])
    candidates = {}
    for s, path in paths.items():
        rows = load_run(path, s, f"candidate seed {s}")
        if set(rows) != keys:
            raise SystemExit(f"candidate seed {s} ({path}) holds other episodes than the reference pool")
        candidates[s] = rows
    srs = {s: sr(rows) for s, rows in candidates.items()}

    paired: dict = {}
    for s in seeds:
        if s in POOL_SEEDS:
            paired[f"seed{s}"], _ = paired_with_a0(candidates, pool, (s,), args.bootstrap_draws, args.bootstrap_seed)
    pooled_delta = None
    if all(s in candidates for s in POOL_SEEDS):
        paired["seeds42+1337"], pooled_delta = paired_with_a0(candidates, pool, POOL_SEEDS, args.bootstrap_draws,
                                                               args.bootstrap_seed)

    stages = []
    for k in range(1, len(seeds) + 1):
        used = seeds[:k]
        e_hat = sum(srs[s] - REF[s] for s in used) / k
        s_c = sample_sd([srs[s] for s in used]) if k >= 2 else None
        stages.append(decide(k, e_hat, s_c, pooled_delta if k >= 2 else None))
    decision = next((st for st in stages if st["terminal"]), stages[-1])

    report = {
        "schema": "heatmapvln-exp32-pool-estimate-v1",
        "seeds": seeds,
        "decision": {**_rounded(decision), "decided_at_k": decision["k"], "seeds_run": len(seeds),
                     "seeds_past_decision": [s for s in seeds[decision["k"]:]] if decision["terminal"] else []},
        "stages": [_rounded(st) for st in stages],
        "candidate": {str(s): {"path": str(paths[s]), "sr_pt": round(srs[s], 4), "ref_pt": REF[s],
                               "sr_minus_ref_pt": round(srs[s] - REF[s], 4),
                               "sr_switched_on_pt": round(100.0 * sum(float(r["success"]) for r in candidates[s].values())
                                                          / EPISODES, 4),
                               "step_budget_stops": sum(bool(r.get("step_budget_stop")) for r in candidates[s].values())}
                      for s in seeds},
        "reference_pool": reference,
        "report_only": {
            "paired_vs_a0": paired,
            "pool_episode_bootstrap": pool_bootstrap(candidates, pool, seeds, args.bootstrap_draws, args.bootstrap_seed),
        },
        "bootstrap": {"draws": args.bootstrap_draws, "seed": args.bootstrap_seed},
        "protocol": "standard: step-budget STOP scored as no STOP (success = spl = 0)",
    }
    if pooled_delta is not None:
        a0_mean = sum(sr(pool[("a0", s)]) for s in POOL_SEEDS) / len(POOL_SEEDS)
        report["report_only"]["vs_a0_two_seed_mean_pt"] = round(sum(srs[s] for s in POOL_SEEDS) / len(POOL_SEEDS) - a0_mean, 4)
    d = report["decision"]
    print(f"{d['verdict'].upper()} ({d['ledger']}, k={d['k']} of {len(seeds)}) " + json.dumps(
        {"e_hat": d["e_hat"], "s_c": d["s_c"], "se": d["se"], "paired_delta_sr": d["paired_delta_sr"],
         "next_seeds": d["next_seeds"], "sr": {str(s): round(v, 2) for s, v in srs.items()}}, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 3 if decision["terminal"] else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("pool-ref", help="per-episode mean of one seed's four arms, for paired_progress_monitor.py")
    ref.add_argument("--seed", type=int, choices=POOL_SEEDS, required=True)
    ref.add_argument("--out", required=True, help="progress.jsonl to write (e.g. pool_ref_seed42.jsonl)")
    est = sub.add_parser("estimate", help="Ê_k, SE_k and the ledger's decision over the candidate's finished seeds")
    est.add_argument("--candidate", action="append", required=True, metavar="SEED=RUN",
                     help="a finished seed: run directory or its merged/progress.jsonl; repeat in the order 42, 1337, 2024, 3407")
    est.add_argument("--bootstrap-draws", type=int, default=20000)
    est.add_argument("--bootstrap-seed", type=int, default=0)
    est.add_argument("--out", default="", help="JSON report")
    for p in (ref, est):
        p.add_argument("--ref-template", default=REF_TEMPLATE, help="the pool's runs, with {arm} (a0-a3) and {seed}")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return {"pool-ref": pool_ref, "estimate": estimate}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

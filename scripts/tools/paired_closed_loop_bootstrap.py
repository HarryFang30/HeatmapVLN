#!/usr/bin/env python3
"""Paired bootstrap of two closed-loop R2R evaluations on the same episodes.

Both inputs are ``merged/progress.jsonl`` files (one row per episode with
``scene_id``, ``episode_id``, ``success``, ``spl``, ``os``, ``ne``).  Episodes are
paired by (scene, episode id), so the per-episode difference cancels episode
difficulty; a key that appears twice in one file is an error, not "the last row
wins".  Rows without ``scene_id`` (older runs) are paired by episode id alone,
which must then be unique in each file.  On R2R val_unseen every episode id is
unique, so the two keys give the same pairs in the same order and the same
numbers.

The bootstrap resamples episodes with replacement (the default), or with
``--bootstrap-unit scene`` whole scenes (EXP-28's HM3D proxy: episodes of one
scene share its geometry, so they are not independent draws).  A scene keeps
its episodes from every pooled pair together.  Optional strata come from the
episode geodesic distance in the Habitat dataset json.gz (``--dataset``) and are
declared in advance via ``--geodesic-min``.

``--standard-protocol`` scores a run made with ``--stop_at_step_budget`` (EXP-24)
under the standard protocol, as ``paired_progress_monitor.py`` does: an episode
the budget's STOP ended (``step_budget_stop``) would have run out of steps
without it, which Habitat scores 0 (EXP-23 run record 8).  Without the option, a run with
rows that say ``stop_at_step_budget`` is labelled "switched on" in the table and the json,
and with it the json counts the rows that ran with the switch (none on the treatment side:
nothing was rescored, which a run whose rows lost those fields would also show).  Rows that
name their ``rpc_protocol_seed`` are expected to pair seed with seed; other seeds are
reported (``protocol_seed_mismatch``), not refused.

The ledger cites these numbers, so the output is written as JSON next to a
markdown table and the script is deterministic (``--bootstrap-seed``).  Without
the new options the output is the one the ledger's existing numbers came from.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path, PurePosixPath

METRICS = ("success", "spl", "os", "ne")

# (scene or None, episode id as str); None when the row has no scene_id.
Key = tuple


def scene_name(raw_scene_id: str) -> str:
    """``mp3d/<scene>/<scene>.glb`` -> ``<scene>``, as the client keys episodes.

    ``scripts.tools.build_r2r_train_cohorts.scene_name`` without its imports (tests pin them equal);
    ``hm3d/<ID>/<ID>.glb`` gives ``<ID>`` the same way.
    """
    scene = PurePosixPath(str(raw_scene_id)).parent.name
    if not scene:
        raise SystemExit(f"cannot derive a scene name from {raw_scene_id!r}")
    return scene


def standard_row(row: dict) -> dict:
    """``paired_progress_monitor.standard`` for one row: a step-budget STOP is no STOP."""
    if row.get("step_budget_stop"):
        return dict(row, success=0.0, spl=0.0)
    return row


def load_progress(path: Path) -> dict[Key, dict]:
    rows: dict[Key, dict] = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        scene = row.get("scene_id")
        key = (None if scene is None else str(scene), str(row["episode_id"]))
        if key in rows:
            raise SystemExit(
                f"{path}:{number}: scene {key[0]!r} episode {key[1]} appears twice; "
                "a progress file has one row per episode"
            )
        rows[key] = row
    return rows


def _by_episode_id(rows: dict[Key, dict], path: Path) -> dict[Key, dict]:
    out: dict[Key, dict] = {}
    for (_, episode), row in rows.items():
        if (None, episode) in out:
            raise SystemExit(
                f"{path}: episode {episode} appears in two scenes and some rows have no scene_id, "
                "so the episodes cannot be paired"
            )
        out[(None, episode)] = row
    return out


def pair_rows(
    t_rows: dict[Key, dict], c_rows: dict[Key, dict], t_path: Path, c_path: Path
) -> tuple[dict[Key, dict], dict[Key, dict], list[Key]]:
    """Both arms keyed alike, and their common keys in episode-id order (the order the bootstrap draws in)."""
    if any(scene is None for scene, _ in list(t_rows) + list(c_rows)):
        print(
            f"warning: {t_path} or {c_path} has rows without scene_id; pairing by episode id alone",
            file=sys.stderr,
        )
        t_rows, c_rows = _by_episode_id(t_rows, t_path), _by_episode_id(c_rows, c_path)
    else:
        # One episode id under two scenes across the arms means two different datasets,
        # unless ids repeat across scenes within an arm (then they name nothing alone).
        t_scene, c_scene = {}, {}
        for rows, scenes in ((t_rows, t_scene), (c_rows, c_scene)):
            for scene, episode in rows:
                scenes.setdefault(episode, set()).add(scene)
        if all(len(s) == 1 for s in t_scene.values()) and all(len(s) == 1 for s in c_scene.values()):
            clash = sorted(
                (e for e in set(t_scene) & set(c_scene) if t_scene[e] != c_scene[e]), key=int
            )
            if clash:
                e = clash[0]
                raise SystemExit(
                    f"{len(clash)} episode ids are in different scenes in {t_path} and {c_path} "
                    f"(e.g. {e}: {sorted(t_scene[e])} vs {sorted(c_scene[e])}); the arms ran different datasets"
                )
    pair = sorted(set(t_rows) & set(c_rows), key=lambda k: (int(k[1]), k[0] or ""))
    return t_rows, c_rows, pair


def load_geodesic(path: Path) -> dict[Key, float]:
    with gzip.open(path) as handle:
        dataset = json.load(handle)
    out: dict[Key, float] = {}
    for episode in dataset["episodes"]:
        key = (scene_name(episode["scene_id"]), str(episode["episode_id"]))
        if key in out:
            raise SystemExit(f"{path}: scene {key[0]} episode {key[1]} appears twice")
        out[key] = float(episode["info"]["geodesic_distance"])
    return out


def geodesic_of(geodesic: dict[Key, float], scene: str | None, episode: str) -> float:
    if scene is not None:
        if (scene, episode) not in geodesic:
            raise SystemExit(f"scene {scene} episode {episode} is not in --dataset (keyed by its scene directory)")
        return geodesic[(scene, episode)]
    found = [value for (_, e), value in geodesic.items() if e == episode]
    if len(found) != 1:
        raise SystemExit(f"episode {episode} has no scene_id and {len(found)} dataset entries")
    return found[0]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def bootstrap_ci(diffs: list[float], rng: random.Random, draws: int) -> tuple[float, float]:
    n = len(diffs)
    means = sorted(mean([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def cluster_bootstrap_ci(
    diffs: list[float], clusters: list[str], rng: random.Random, draws: int
) -> tuple[float, float]:
    """Resample whole clusters with replacement; a draw's statistic is the mean over the episodes it holds."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for diff, cluster in zip(diffs, clusters, strict=True):
        sums[cluster] = sums.get(cluster, 0.0) + diff
        counts[cluster] = counts.get(cluster, 0) + 1
    names = sorted(sums)
    total = [sums[name] for name in names]
    size = [counts[name] for name in names]
    k = len(names)
    means = []
    for _ in range(draws):
        picks = [rng.randrange(k) for _ in range(k)]
        means.append(sum(total[i] for i in picks) / sum(size[i] for i in picks))
    means.sort()
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def compare(
    treatment: dict,
    control: dict,
    ids: list,
    rng: random.Random,
    draws: int,
    unit: str = "episode",
) -> dict[str, dict]:
    """``ids`` are (pair index, scene, episode id); ``unit`` "scene" resamples scenes across pairs."""
    out: dict[str, dict] = {"n": len(ids)}
    if unit == "scene":
        out["clusters"] = len({i[1] for i in ids})
    for metric in METRICS:
        a = [float(treatment[i][metric]) for i in ids]
        b = [float(control[i][metric]) for i in ids]
        diffs = [x - y for x, y in zip(a, b, strict=True)]
        if unit == "scene":
            lo, hi = cluster_bootstrap_ci(diffs, [i[1] for i in ids], rng, draws)
        else:
            lo, hi = bootstrap_ci(diffs, rng, draws)
        out[metric] = {
            "treatment": mean(a),
            "control": mean(b),
            "diff": mean(diffs),
            "ci95": [lo, hi],
        }
    return out


def fmt(entry: dict, metric: str) -> str:
    scale = 1.0 if metric == "ne" else 100.0
    unit = " m" if metric == "ne" else ""
    lo, hi = entry["ci95"]
    return (
        f"{entry['treatment'] * scale:.2f}{unit} vs {entry['control'] * scale:.2f}{unit} "
        f"→ {entry['diff'] * scale:+.2f} [{lo * scale:+.2f}, {hi * scale:+.2f}]"
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", type=Path, required=True, action="append", help="merged/progress.jsonl of the treatment arm; repeat to pool several matched pairs (e.g. two protocol seeds)")
    parser.add_argument("--control", type=Path, required=True, action="append", help="merged/progress.jsonl of the control arm; repeat once per --treatment, in the same order")
    parser.add_argument("--dataset", type=Path, default=None, help="val_unseen.json.gz for geodesic strata")
    parser.add_argument("--geodesic-min", type=float, action="append", default=[], help="pre-declared stratum: geodesic distance >= this many metres")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-unit", choices=("episode", "scene"), default="episode", help="resample episodes (default) or whole scenes (cluster bootstrap; needs scene_id in every row)")
    parser.add_argument("--standard-protocol", action="store_true", help="score step-budget STOPs (step_budget_stop) as no STOP: success = spl = 0")
    parser.add_argument("--label", default="treatment_vs_control")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)

    if len(args.treatment) != len(args.control):
        raise SystemExit("--treatment and --control must be repeated the same number of times")
    # Pooling several pairs keeps every episode's own control: each pair is
    # matched on its own, then the per-episode differences are pooled, so a
    # bootstrap over the pool resamples matched differences, never raw scores.
    treatment: dict = {}
    control: dict = {}
    ids: list = []
    missing_total = 0
    budget_stops = {"treatment": 0, "control": 0}
    # paired episodes whose row says the run had --stop_at_step_budget on, per arm
    budget_rows = {"treatment": 0, "control": 0}
    # (pair, treatment seed, control seed) -> paired episodes, where both rows name a protocol seed
    seed_pairs: dict[tuple, int] = {}
    for index, (t_path, c_path) in enumerate(zip(args.treatment, args.control, strict=True)):
        t_rows, c_rows, pair_ids = pair_rows(load_progress(t_path), load_progress(c_path), t_path, c_path)
        if not pair_ids:
            raise SystemExit(f"no paired episodes between {t_path} and {c_path}")
        missing_total += len((set(t_rows) | set(c_rows)) - set(pair_ids))
        for scene, episode in pair_ids:
            key = (index, scene, episode)
            t_row, c_row = t_rows[(scene, episode)], c_rows[(scene, episode)]
            budget_rows["treatment"] += bool(t_row.get("stop_at_step_budget"))
            budget_rows["control"] += bool(c_row.get("stop_at_step_budget"))
            if t_row.get("rpc_protocol_seed") is not None and c_row.get("rpc_protocol_seed") is not None:
                seeds = (index, int(t_row["rpc_protocol_seed"]), int(c_row["rpc_protocol_seed"]))
                seed_pairs[seeds] = seed_pairs.get(seeds, 0) + 1
            if args.standard_protocol:
                budget_stops["treatment"] += bool(t_row.get("step_budget_stop"))
                budget_stops["control"] += bool(c_row.get("step_budget_stop"))
                t_row, c_row = standard_row(t_row), standard_row(c_row)
            treatment[key] = t_row
            control[key] = c_row
            ids.append(key)
    # A pair is matched on its protocol seed too (A0 seed 42 against the candidate's seed 42):
    # other seeds pair two draws of System1's noise, not one episode under two models.
    seed_mismatch = {f"pair {i}: treatment seed {t} vs control seed {c}": n
                     for (i, t, c), n in sorted(seed_pairs.items()) if t != c}
    if seed_mismatch:
        print(f"warning: paired episodes ran with other protocol seeds: {seed_mismatch}", file=sys.stderr)
    switched_on = any(budget_rows.values()) and not args.standard_protocol
    if switched_on:
        print(
            f"warning: paired episodes ran with --stop_at_step_budget {budget_rows}; these numbers count "
            "its STOPs (pass --standard-protocol for the standard protocol)",
            file=sys.stderr,
        )
    if args.standard_protocol and not budget_rows["treatment"]:
        # Nothing to rescore: right for a run without the switch, wrong if its rows lost the fields.
        print(
            "warning: --standard-protocol, but no treatment row says it ran with --stop_at_step_budget, so "
            "nothing is rescored; if the run had the switch on, read progress rows that keep "
            "stop_at_step_budget / step_budget_stop",
            file=sys.stderr,
        )
    if args.bootstrap_unit == "scene" and any(i[1] is None for i in ids):
        raise SystemExit("--bootstrap-unit scene needs scene_id in every progress row")
    rng = random.Random(args.bootstrap_seed)

    def stratum(subset: list) -> dict:
        return compare(treatment, control, subset, rng, args.bootstrap_draws, args.bootstrap_unit)

    report = {
        "label": args.label,
        "treatment": [str(path) for path in args.treatment],
        "control": [str(path) for path in args.control],
        "pairs_pooled": len(args.treatment),
        "paired_episodes": len(ids),
        "unpaired_episodes": missing_total,
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
    }
    # Only the new options, or a step-budget run, add keys, so a default run on runs without
    # the switch (A0, the MP3D pairings) writes what the ledger's numbers came from.
    if args.bootstrap_unit != "episode":
        report["bootstrap_unit"] = args.bootstrap_unit
    if args.standard_protocol:
        report["protocol"] = "standard: step-budget STOP scored as no STOP (success = spl = 0)"
        report["step_budget_stops_rescored"] = budget_stops
        report["step_budget_rows"] = budget_rows
    elif switched_on:
        report["protocol"] = "switched on: step-budget STOPs counted as STOPs (not the standard protocol)"
        report["step_budget_rows"] = budget_rows
    if seed_mismatch:
        report["protocol_seed_mismatch"] = seed_mismatch
    report["strata"] = {"all": stratum(ids)}
    if args.geodesic_min:
        if args.dataset is None:
            raise SystemExit("--geodesic-min requires --dataset")
        geodesic = load_geodesic(args.dataset)
        for threshold in args.geodesic_min:
            subset = [i for i in ids if geodesic_of(geodesic, i[1], i[2]) >= threshold]
            report["strata"][f"geodesic_ge_{threshold:g}m"] = stratum(subset)

    both = sum(1 for i in ids if treatment[i]["success"] and control[i]["success"])
    only_t = sum(1 for i in ids if treatment[i]["success"] and not control[i]["success"])
    only_c = sum(1 for i in ids if control[i]["success"] and not treatment[i]["success"])
    report["success_overlap"] = {"both": both, "treatment_only": only_t, "control_only": only_c}

    print(f"# {args.label}: {len(ids)} paired episodes from {len(args.treatment)} pair(s) ({missing_total} unpaired dropped)")
    if args.bootstrap_unit == "scene":
        print(f"bootstrap over whole scenes ({report['strata']['all']['clusters']} clusters)")
    if args.standard_protocol:
        print(f"standard protocol: step-budget STOPs rescored as failures: {budget_stops} "
              f"(rows that ran with the switch: {budget_rows})")
    elif switched_on:
        print(f"SWITCHED-ON protocol: step-budget STOPs count as STOPs (rows that ran with the switch: {budget_rows})")
    if seed_mismatch:
        print(f"OTHER PROTOCOL SEEDS paired: {seed_mismatch}")
    print("| stratum | n | SR | SPL | OS | NE |")
    print("|---|---|---|---|---|---|")
    for name, entry in report["strata"].items():
        cells = " | ".join(fmt(entry[m], m) for m in METRICS)
        print(f"| {name} | {entry['n']} | {cells} |")
    print(f"success overlap: both={both} treatment_only={only_t} control_only={only_c}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Report what the EXP-13 relabelling would supervise, before any GPU is used.

The System2 fine-tune only earns its cost if a meaningful number of states
actually get corrected, and if the corrections land where EXP-12 said the target
is (``wrong_branch`` / ``off_route``) rather than being spread over states where
native was already right.  This is CPU-only and opens no JPEG, so the answer is
available in minutes and can be checked against the pre-registered criteria
before the training arm is submitted.

It also prints the exact assistant strings for a handful of states, because a
supervision bug that swaps "turn left" for "turn right" is invisible in
aggregate counts and catastrophic in closed loop.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dagger_system2_sft import (  # noqa: E402
    load_oracle_views,
    plan_for_sample,
)

VIEWS = ("front", "right", "back", "left")
RECOVERY_TAGS = ("wrong_branch", "off_route")
SCHEMA = "heatmapvln-exp13-relabel-audit-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--oracle-views", type=Path, required=True, help="EXP-12 d1_per_state.jsonl")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    from src.data.trajectory_dagger_dataset import TrajectoryDaggerDataset

    shard_paths = sorted(args.collection_root.glob("shard_*"))
    if not shard_paths:
        raise SystemExit(f"no shard_* directories under {args.collection_root}")
    fingerprints = {
        json.loads((shard / "collection_manifest.json").read_text(encoding="utf-8"))["contract"][
            "policy_fingerprint"
        ]
        for shard in shard_paths
    }
    if len(fingerprints) != 1:
        raise SystemExit(f"shards disagree on the collecting policy: {sorted(fingerprints)}")

    dataset = TrajectoryDaggerDataset(
        collection_roots=[str(shard) for shard in shard_paths],
        source_types=["dagger_hard", "dagger_normal"],
        num_history=8,
        image_size=(384, 384),
        require_lookdown=True,
        expected_policy_mode="internnav_native",
        expected_policy_fingerprint=fingerprints.pop(),
    )
    oracle_views = load_oracle_views(args.oracle_views)
    print(f"dataset: {len(dataset)} states | oracle rows: {len(oracle_views)}", flush=True)

    kinds: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = {}
    by_tag: dict[str, Counter[str]] = {}
    corrected_turns: Counter[int] = Counter()
    corrected_views: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for index in range(len(dataset)):
        sample = dataset.sample_metadata(index)
        key = str(sample.get("key") or "")
        plan = plan_for_sample(sample, oracle_views.get(key), max_turns=args.max_turns)
        source = str(sample.get("source_type") or "")
        tags = list(sample.get("failure_tags") or [])
        kind = plan.get("kind")
        if kind is None:
            dropped[str(plan.get("reason") or "unlabelled")] += 1
            continue
        kinds[kind] += 1
        by_source.setdefault(source, Counter())[kind] += 1
        for tag in tags:
            by_tag.setdefault(str(tag), Counter())[kind] += 1
        if kind == "correct_turn":
            corrected_turns[int(plan["emitted_turns"])] += 1
            corrected_views[VIEWS[int(plan["oracle_view"])]] += 1
            if len(examples) < args.examples:
                examples.append(
                    {
                        "sample_key": key,
                        "source_type": source,
                        "failure_tags": tags,
                        "oracle_view": VIEWS[int(plan["oracle_view"])],
                        "native_view": VIEWS[int(plan["native_view"])],
                        "oracle_actions_head": [
                            int(value) for value in (sample.get("oracle") or {}).get("actions", [])[:8]
                        ],
                        "native_llm_output": str((sample.get("native") or {}).get("llm_output") or ""),
                        "target_texts": plan["target_texts"],
                    }
                )

    labelled = sum(kinds.values())
    recovery_totals = Counter()
    for tag in RECOVERY_TAGS:
        recovery_totals.update(by_tag.get(tag, Counter()))
    recovery_labelled = sum(recovery_totals.values())

    report = {
        "schema": SCHEMA,
        "collection_root": str(args.collection_root),
        "oracle_views": str(args.oracle_views),
        "max_turns": args.max_turns,
        "states_seen": len(dataset),
        "states_labelled": labelled,
        "kinds": dict(kinds),
        "corrected_fraction": kinds["correct_turn"] / labelled if labelled else None,
        "corrected_turn_lengths": {str(k): v for k, v in sorted(corrected_turns.items())},
        "corrected_oracle_views": dict(corrected_views),
        "by_source_type": {name: dict(counter) for name, counter in sorted(by_source.items())},
        "by_failure_tag": {name: dict(counter) for name, counter in sorted(by_tag.items())},
        "recovery_slice": {
            "tags": list(RECOVERY_TAGS),
            "states_labelled": recovery_labelled,
            "kinds": dict(recovery_totals),
            "corrected_fraction": (
                recovery_totals["correct_turn"] / recovery_labelled if recovery_labelled else None
            ),
        },
        "dropped": dict(dropped),
        "examples": examples,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, ensure_ascii=False, indent=2))
    for example in examples:
        print(json.dumps(example, ensure_ascii=False), flush=True)
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()

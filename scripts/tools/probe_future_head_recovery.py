#!/usr/bin/env python3
"""EXP-12 D2: does the future head already point the right way in recovery states?

The decision layer proposed after EXP-05/07/09 would turn the future cognition
head into a recovery proposer: when the history head flags a revisit, the future
head says which direction to go and that overrides System2's pixel goal.  D2
measures the cheapest version of that -- the **zero-shot** head, no DAgger
fine-tuning -- against the System2 proposal it would replace.

Both directions are scored with the *same* oracle-view definition D1 used: the
probe never recomputes geometry, it reads ``--per-state-jsonl`` produced by
``summarize_recovery_state_geometry.py`` and joins on ``sample_key``.  If the two
tools ever disagreed the comparison would be meaningless, so there is exactly one
implementation and D2 is a consumer of it.

Ground-truth Habitat poses reach the history head here (the DAgger reader has no
AMB3R cache), which EXP-12 boundary 3 declares: a positive D2 result is an upper
bound on deployment behaviour.  ``src/config_schema.py`` fail-closes on exactly
this mismatch -- a PPA config may not set ``dataset_type: trajectory_dagger`` --
so the probe deliberately does not route the DAgger rows through the config.  It
loads the unmodified training config for the *model* and constructs the DAgger
dataset explicitly, leaving the training guard intact for everyone else.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

VIEWS = ("front", "right", "back", "left")


def load_oracle_index(path: Path, bucket: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source_type") != bucket:
                continue
            key = row.get("sample_key")
            if key:
                index[str(key)] = row
    return index


def merge(paths: list[Path], output: Path) -> None:
    shards = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    states: list[dict[str, Any]] = []
    for shard in shards:
        states.extend(shard["states"])
    report = summarise(states)
    report["schema"] = "heatmapvln-exp12-future-head-probe-v1"
    report["merged_from"] = [str(p) for p in paths]
    report["inputs"] = shards[0].get("inputs", {})
    report["inputs"]["shards"] = len(shards)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in report.items() if k != "states"}, ensure_ascii=False, indent=2))
    print(f"wrote {output}")


def summarise(states: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [s for s in states if s["oracle_view"] is not None]
    total = len(scored)
    if total == 0:
        return {"scored_states": 0, "states": states}
    head_correct = sum(1 for s in scored if s["future_head_view"] == s["oracle_view"])
    with_native = [s for s in scored if s["native_view"] is not None]
    native_correct = sum(1 for s in with_native if s["native_view"] == s["oracle_view"])
    head_on_native_subset = sum(
        1 for s in with_native if s["future_head_view"] == s["oracle_view"]
    )
    outside_front = [s for s in scored if s["oracle_view"] != 0]
    return {
        "states_seen": len(states),
        "scored_states": total,
        "states_dropped_oracle_invisible": len(states) - total,
        "future_head_top1_acc": head_correct / total,
        "future_head_pred_distribution": {
            VIEWS[i]: sum(1 for s in scored if s["future_head_view"] == i) / total
            for i in range(4)
        },
        "oracle_view_distribution": {
            VIEWS[i]: sum(1 for s in scored if s["oracle_view"] == i) / total
            for i in range(4)
        },
        "paired_subset": {
            "states": len(with_native),
            "system2_top1_acc": native_correct / len(with_native) if with_native else None,
            "future_head_top1_acc": head_on_native_subset / len(with_native)
            if with_native
            else None,
            "delta_pt": (
                (head_on_native_subset - native_correct) / len(with_native) * 100.0
                if with_native
                else None
            ),
        },
        "oracle_outside_front_subset": {
            "states": len(outside_front),
            "future_head_top1_acc": (
                sum(1 for s in outside_front if s["future_head_view"] == s["oracle_view"])
                / len(outside_front)
                if outside_front
                else None
            ),
            "system2_top1_acc": (
                sum(
                    1
                    for s in outside_front
                    if s["native_view"] is not None and s["native_view"] == s["oracle_view"]
                )
                / sum(1 for s in outside_front if s["native_view"] is not None)
                if any(s["native_view"] is not None for s in outside_front)
                else None
            ),
        },
        "tag_counts": dict(Counter(tag for s in scored for tag in s.get("tags", []))),
        "states": states,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge", type=Path, action="append", default=[])
    parser.add_argument("--config", type=Path, default=None, help="unmodified PPA training config; supplies the model only")
    parser.add_argument("--collection-root", type=Path, default=None, help="DAgger collection root holding shard_*/")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--per-state-jsonl", type=Path, default=None)
    parser.add_argument("--bucket", default="dagger_hard")
    parser.add_argument("--max-states", type=int, default=4000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    if args.merge:
        merge(args.merge, args.output_json)
        return

    for name in ("config", "checkpoint", "per_state_jsonl", "collection_root"):
        if getattr(args, name) is None:
            raise SystemExit(f"--{name.replace('_', '-')} is required unless --merge is used")

    import torch
    from transformers import AutoProcessor

    from scripts.training.model_builder import build_model
    from scripts.training.pose_adaptation import load_past_plan_action_initialization
    from src.config_schema import load_and_validate_config
    from src.data.internnav_heatmap_control_collator import InternNavHeatmapControlCollator
    from src.data.trajectory_dagger_dataset import TrajectoryDaggerDataset

    oracle_index = load_oracle_index(args.per_state_jsonl, args.bucket)
    print(f"oracle index: {len(oracle_index)} {args.bucket} states", flush=True)

    cfg = load_and_validate_config(args.config)
    if cfg["data"]["dataset_type"] != "trajectory":
        raise SystemExit(
            "--config must be the unmodified PPA training config; the DAgger rows "
            "are supplied by --collection-root, not by the config"
        )
    shard_paths = sorted(args.collection_root.glob("shard_*"))
    if not shard_paths:
        raise SystemExit(f"no shard_* directories under {args.collection_root}")
    # The reader fail-closes unless the caller states which policy collected the
    # rows.  Read it from the sealed manifests rather than hard-coding it, and
    # require every shard to agree.
    fingerprints = {
        json.loads((shard / "collection_manifest.json").read_text(encoding="utf-8"))[
            "contract"
        ]["policy_fingerprint"]
        for shard in shard_paths
    }
    if len(fingerprints) != 1:
        raise SystemExit(f"shards disagree on the collecting policy: {sorted(fingerprints)}")
    policy_fingerprint = fingerprints.pop()
    dataset = TrajectoryDaggerDataset(
        collection_roots=[str(shard) for shard in shard_paths],
        source_types=[args.bucket],
        num_history=8,
        image_size=tuple(cfg["data"]["image_size"]),
        require_lookdown=True,
        expected_policy_mode="internnav_native",
        expected_policy_fingerprint=policy_fingerprint,
    )
    print(
        f"dataset: {len(dataset)} {args.bucket} samples from {len(shard_paths)} shards "
        f"(policy {policy_fingerprint})",
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(
        cfg["model"]["llm"]["model_path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    collator = InternNavHeatmapControlCollator(
        processor,
        n_traj_query=4,
        max_seq_length=8192,
        teacher_force_system2_answer=True,
        include_future_trajectory_targets=False,
        required_history_pose_provider=None,
    )

    device = torch.device("cuda:0")
    torch.manual_seed(42)
    model = build_model(cfg, verbose=True, device="cuda:0", enable_action_head=True)
    if model.vlm_backbone.model is None:
        model.vlm_backbone._load_model()
    model._ensure_heatmap_vln()
    report = load_past_plan_action_initialization(
        model,
        str(args.checkpoint),
        stage="stage2_joint",
        # The future head reads Z = Z0 + bridge(M), so the deployed bridge has
        # to come along; the default stage-transition behaviour deliberately
        # leaves it at exact zero and would probe a different model.
        load_trained_bridge=True,
    )
    print(f"checkpoint init: {report}", flush=True)
    model.eval()

    indices = [i for i in range(len(dataset)) if i % args.shard_count == args.shard_index]
    states: list[dict[str, Any]] = []
    started = time.perf_counter()
    per_shard_budget = max(1, args.max_states // max(1, args.shard_count))

    with torch.no_grad():
        for count, index in enumerate(indices):
            if len(states) >= per_shard_budget:
                break
            sample = dataset[index]
            key = str(sample.get("sample_key") or "")
            truth = oracle_index.get(key)
            if truth is None:
                continue
            batch = collator([sample])
            output = model(
                video_frames=None,
                instruction_text=[sample["text"]],
                current_observation=batch["current_frame"].to(device)
                if "current_frame" in batch
                else None,
                panoramic_inputs=batch.get("pano_inputs"),
                panoramic_num_histories=batch.get("pano_num_histories"),
                panoramic_text_anchor_positions=batch.get("pano_text_anchor_positions"),
                heatmap_single_view_inputs=batch.get("heatmap_single_view_inputs"),
                heatmap_single_view_num_histories=batch.get(
                    "heatmap_single_view_num_histories"
                ),
                heatmap_control_history_mask=batch.get("heatmap_control_history_mask"),
                history_valid_mask=batch.get("history_valid_mask"),
                history_age_steps=batch.get("history_age_steps"),
                history_rel_poses=batch["history_rel_poses"].to(device),
                traj_images=batch["traj_images"].to(device)
                if "traj_images" in batch
                else None,
                sample_trajectory=False,
                return_heatmaps=True,
                return_heatmap_logits=True,
                return_future_heatmaps=True,
                return_actions=False,
                return_lm_loss=False,
            )
            future = output.get("future_visibility")
            if future is None:
                raise RuntimeError(
                    "forward returned no future_visibility; keys="
                    + ",".join(sorted(output.keys()))
                )
            # First future time bin: which of the four canonical views does the
            # head put the next segment in.
            predicted_view = int(future[0, 0].float().argmax().item())
            states.append(
                {
                    "sample_key": key,
                    "future_head_view": predicted_view,
                    "oracle_view": truth["oracle_view"],
                    "native_view": truth["native_view"],
                    "tags": truth.get("tags", []),
                }
            )
            if len(states) % 50 == 0:
                rate = len(states) / (time.perf_counter() - started)
                print(
                    f"  {len(states)}/{per_shard_budget} scored "
                    f"({count + 1} visited, {rate:.2f} states/s)",
                    flush=True,
                )

    result = summarise(states)
    result["schema"] = "heatmapvln-exp12-future-head-probe-v1"
    result["inputs"] = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "per_state_jsonl": str(args.per_state_jsonl),
        "bucket": args.bucket,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "per_shard_budget": per_shard_budget,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in result.items() if k != "states"}, ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()

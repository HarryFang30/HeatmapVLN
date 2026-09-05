#!/usr/bin/env python3
"""EXP-13 B: score a trained System2 arm's recovery decisions on held-out scenes.

Two numbers decide the arm, and both are properties of a single forward pass:

``recovery_turn_accuracy``
    on val states the relabeller marked ``correct_turn``, does the policy start
    its answer with a turn in the oracle's direction?  The first token of a
    greedy decode conditions on the prompt alone, so the argmax at the first
    supervised position *is* that token -- no generation loop is needed, and
    none is used, so the number cannot drift from what a decoder would emit.

``normal_preservation``
    on val ``dagger_normal`` states the relabeller left alone, does the policy
    still reproduce the frozen model's own answer token for token?  A fine-tune
    that wins on recovery by forgetting how to navigate is not a win, and this
    is the number that catches it before eight GPU-days of closed loop do.

Both are reported per arm and per relabel kind, together with the distribution
of first tokens.  EXP-12 spent a whole probe comparing two constant predictors
because nobody printed that distribution first.

EXP-14 adds the stop decision on the same forward pass:

``stop_recall``
    on val states the relabeller marked ``correct_stop`` (the oracle's route
    ends at the goal within the horizon, native kept walking), is the first
    token ``STOP``?

``stop_false_alarm``
    on every other val state, is the first token ``STOP``?  Native never stops
    on these states by construction -- the collector kept only states where it
    emitted a pixel goal -- so this rate is exactly the early stops the
    fine-tune *introduced*, and the criterion caps it.

``STOP`` is one token in the released tokenizer (checked at start-up), so the
first-token argmax is the whole answer.
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

SCHEMA = "heatmapvln-exp13-system2-decisions-v2"
ARROW_LEFT = "←"
ARROW_RIGHT = "→"
STOP_TEXT = "STOP"
STOP_TARGET_KINDS = ("correct_stop", "keep_stop")


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return sum(1 for s in rows if s[key]) / len(rows) if rows else None


def merge(paths: list[Path], output: Path) -> None:
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    states: list[dict[str, Any]] = []
    for shard in shards:
        states.extend(shard["states"])
    report = summarise(states)
    report["schema"] = SCHEMA
    report["merged_from"] = [str(path) for path in paths]
    report["inputs"] = shards[0].get("inputs", {})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in report.items() if k != "states"}, ensure_ascii=False, indent=2))
    print(f"wrote {output}")


def _direction_of(text: str) -> str | None:
    has_left = ARROW_LEFT in text
    has_right = ARROW_RIGHT in text
    if has_left == has_right:
        return None
    return "left" if has_left else "right"


def summarise(states: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for state in states:
        by_kind.setdefault(str(state["relabel_kind"]), []).append(state)

    corrections = by_kind.get("correct_turn", [])
    scored = [s for s in corrections if s["target_direction"] is not None]
    turn_accuracy = (
        sum(1 for s in scored if s["predicted_direction"] == s["target_direction"]) / len(scored)
        if scored
        else None
    )

    preserved = [
        s
        for s in by_kind.get("keep_pixel", [])
        if s["source_type"] == "dagger_normal"
    ]
    preservation = (
        sum(1 for s in preserved if s["all_supervised_tokens_match"]) / len(preserved)
        if preserved
        else None
    )

    # Stop metrics (EXP-14).  Rows written by the v1 schema carry no
    # predicted_is_stop and are excluded, never defaulted to "did not stop".
    stop_scored = [
        s for s in by_kind.get("correct_stop", []) if s.get("predicted_is_stop") is not None
    ]
    non_stop = [
        s
        for kind, rows in by_kind.items()
        if kind not in STOP_TARGET_KINDS
        for s in rows
        if s.get("predicted_is_stop") is not None
    ]
    false_alarm_by_source: dict[str, float | None] = {}
    for source in sorted({str(s["source_type"]) for s in non_stop}):
        false_alarm_by_source[source] = _rate(
            [s for s in non_stop if str(s["source_type"]) == source], "predicted_is_stop"
        )

    return {
        "state_count": len(states),
        "recovery_turn_accuracy": turn_accuracy,
        "recovery_turn_states": len(scored),
        "normal_preservation": preservation,
        "normal_preservation_states": len(preserved),
        "stop_recall": _rate(stop_scored, "predicted_is_stop"),
        "stop_recall_states": len(stop_scored),
        "stop_false_alarm": _rate(non_stop, "predicted_is_stop"),
        "stop_false_alarm_states": len(non_stop),
        "stop_false_alarm_by_source": false_alarm_by_source,
        "by_kind": {
            kind: {
                "states": len(rows),
                "first_token_exact": (
                    sum(1 for s in rows if s["first_token_match"]) / len(rows) if rows else None
                ),
                "all_tokens_exact": (
                    sum(1 for s in rows if s["all_supervised_tokens_match"]) / len(rows)
                    if rows
                    else None
                ),
                "predicted_stop": _rate(
                    [s for s in rows if s.get("predicted_is_stop") is not None],
                    "predicted_is_stop",
                ),
                "first_token_texts": dict(
                    Counter(str(s["predicted_first_text"]) for s in rows).most_common(8)
                ),
            }
            for kind, rows in sorted(by_kind.items())
        },
        "predicted_direction_distribution": dict(
            Counter(str(s["predicted_direction"]) for s in corrections)
        ),
        # Kept last and excluded from every printout: the per-state rows are
        # what a merge concatenates and what a disputed number is checked in.
        "states": states,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge", type=Path, action="append", default=[])
    parser.add_argument("--config", type=Path, default=None, help="the arm's training config")
    parser.add_argument("--checkpoint", type=Path, default=None, help="the arm's trained best.pth")
    parser.add_argument(
        "--parent-checkpoint",
        type=Path,
        default=None,
        help="deployed v2 best.pth holding the frozen 79-tensor Past Head",
    )
    parser.add_argument("--collection-root", type=Path, default=None)
    parser.add_argument("--oracle-views", type=Path, default=None)
    parser.add_argument("--max-states", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    if args.merge:
        merge(args.merge, args.output_json)
        return
    for name in ("config", "checkpoint", "parent_checkpoint", "collection_root", "oracle_views"):
        if getattr(args, name) is None:
            raise SystemExit(f"--{name.replace('_', '-')} is required unless --merge is used")

    import torch
    from transformers import AutoProcessor

    from scripts.training.model_builder import build_model
    from scripts.training.pose_adaptation import load_pose_adaptation_initialization
    from scripts.training.utils import _load_normalized_state_dict, safe_torch_load
    from src.config_schema import load_and_validate_config
    from src.data.dagger_system2_sft import DaggerSystem2SFTDataset
    from src.data.internnav_heatmap_control_collator import InternNavHeatmapControlCollator
    from src.data.trajectory_dagger_dataset import TrajectoryDaggerDataset

    import logging

    logging.basicConfig(level=logging.INFO)

    cfg = load_and_validate_config(args.config)
    memory_cfg = cfg["model"].get("system2_memory") or {}
    if not memory_cfg.get("enabled", False):
        raise SystemExit("--config must be one of the EXP-13 System2 memory arms")
    sft_cfg = cfg["data"].get("dagger_system2_sft") or {}

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

    reader = TrajectoryDaggerDataset(
        collection_roots=[str(shard) for shard in shard_paths],
        source_types=list(cfg["data"]["trajectory_dagger"]["source_types"]),
        num_history=int(cfg["data"]["trajectory_dagger"]["num_history"]),
        image_size=tuple(cfg["data"]["image_size"]),
        require_lookdown=True,
        expected_policy_mode="internnav_native",
        expected_policy_fingerprint=fingerprints.pop(),
    )
    dataset = DaggerSystem2SFTDataset(
        reader,
        oracle_views=args.oracle_views,
        max_turns=int(sft_cfg.get("max_turns", 4)),
        # Held-out scenes only: the arms trained on the complement of this.
        scene_split="val",
        val_scene_pct=int(sft_cfg.get("val_scene_pct", 25)),
        # Same relabelling the arm trained on; the val slice is never oversampled.
        stop_supervision=bool(sft_cfg.get("stop_supervision", False)),
        stop_horizon_m=float(sft_cfg.get("stop_horizon_m", 1.0)),
        stop_oversample=1,
    )
    print(f"val states: {len(dataset)} | {json.dumps(dataset.summary(), ensure_ascii=False)}", flush=True)

    processor = AutoProcessor.from_pretrained(
        cfg["model"]["llm"]["model_path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    tokenizer = processor.tokenizer
    stop_ids = [int(v) for v in tokenizer.encode(STOP_TEXT, add_special_tokens=False)]
    if len(stop_ids) != 1:
        raise SystemExit(
            f"{STOP_TEXT!r} is not a single token in this tokenizer ({stop_ids}); the "
            "first-token stop metrics would be reading a fragment"
        )
    stop_token_id = stop_ids[0]
    print(f"STOP token id: {stop_token_id}", flush=True)
    collator = InternNavHeatmapControlCollator(
        processor,
        n_traj_query=0,
        max_seq_length=int(cfg["model"]["llm"].get("max_seq_length", 8192)),
        teacher_force_system2_answer=True,
        include_future_trajectory_targets=False,
        required_history_pose_provider=None,
        build_sft_labels=True,
        memory_token_count=int(memory_cfg.get("num_tokens", 8)),
        internnav_conjunction="you can see ",
    )

    device = torch.device("cuda:0")
    torch.manual_seed(42)
    model = build_model(cfg, verbose=True, device="cuda:0")
    if model.vlm_backbone.model is None:
        model.vlm_backbone._load_model()
    model._ensure_heatmap_vln()

    head_report = load_pose_adaptation_initialization(model, str(args.parent_checkpoint))
    print(f"frozen Past Head: {head_report['loaded_tensor_count']} tensors", flush=True)

    payload = safe_torch_load(str(args.checkpoint))
    trained = payload.get("trainable_state_dict") or {}
    if not trained:
        raise SystemExit(f"{args.checkpoint} carries no trainable_state_dict")
    _missing, unexpected, loaded = _load_normalized_state_dict(model, trained)
    if unexpected:
        raise SystemExit(f"arm checkpoint has tensors this model does not own: {unexpected[:8]}")
    print(f"trained arm: {loaded}/{len(trained)} tensors loaded", flush=True)
    # Nothing is trained here.  Freezing everything also puts the Past Head on
    # its no-grad path, so M_t is produced exactly as it is at deployment.
    model.requires_grad_(False)
    model.eval()

    indices = [i for i in range(len(dataset)) if i % args.shard_count == args.shard_index]
    budget = len(indices)
    if args.max_states > 0:
        budget = min(budget, max(1, args.max_states // max(1, args.shard_count)))

    states: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for visited, index in enumerate(indices):
            if len(states) >= budget:
                break
            sample = dataset[index]
            plan = dataset.plans[index]
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
                sample_trajectory=False,
                return_heatmaps=False,
                return_actions=False,
                return_future_heatmaps=False,
                inject_system2_memory=True,
                return_lm_correct_logprobs=True,
            )
            alignment = output["lm_correct_label_alignment"]
            predicted = output["lm_predicted_token_ids"][0]
            correct = alignment["sample_correct_token_ids"][0]
            if not correct:
                continue
            predicted_first_text = tokenizer.decode([int(predicted[0])])
            target_first_text = tokenizer.decode([int(correct[0])])
            states.append(
                {
                    "sample_key": plan["sample_key"],
                    "scene_id": plan["scene_id"],
                    "source_type": plan["source_type"],
                    "relabel_kind": plan["kind"],
                    "failure_tags": plan["failure_tags"],
                    "target_texts": plan["target_texts"],
                    "predicted_first_text": predicted_first_text,
                    "target_first_text": target_first_text,
                    "predicted_direction": _direction_of(predicted_first_text),
                    "target_direction": _direction_of(target_first_text),
                    "predicted_is_stop": int(predicted[0]) == stop_token_id,
                    "target_is_stop": plan["kind"] in STOP_TARGET_KINDS,
                    "first_token_match": int(predicted[0]) == int(correct[0]),
                    "all_supervised_tokens_match": [int(v) for v in predicted]
                    == [int(v) for v in correct],
                    "supervised_tokens": len(correct),
                }
            )
            if len(states) % max(1, args.progress_every) == 0:
                rate = len(states) / (time.perf_counter() - started)
                print(
                    f"  {len(states)}/{budget} scored ({visited + 1} visited, {rate:.2f} states/s)",
                    flush=True,
                )

    report = summarise(states)
    report["schema"] = SCHEMA
    report["inputs"] = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "parent_checkpoint": str(args.parent_checkpoint),
        "collection_root": str(args.collection_root),
        "oracle_views": str(args.oracle_views),
        "system2_memory_mode": memory_cfg.get("mode"),
        "stop_supervision": bool(sft_cfg.get("stop_supervision", False)),
        "stop_horizon_m": float(sft_cfg.get("stop_horizon_m", 1.0)),
        "stop_token_id": stop_token_id,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in report.items() if k != "states"}, ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""EXP-13 A: cache the decision-layer features of every DAgger candidate state.

EXP-12 established that neither the zero-shot future head nor native System2
emits anything but "front" in recovery states, so the cheap implementation
(gate an existing head) is dead.  The open question this tool feeds is one step
earlier and much more basic:

    does the history memory ``M_t`` carry decision-relevant information that
    System2's own summary does not already contain?

The tool answers nothing by itself.  It runs the deployed checkpoint forward
once per DAgger state and writes the four feature families a readout has to
choose between, plus the oracle label:

* ``traj_hidden``  [4, 3584]  System2's own summary of its reasoning (the
  richest System2-side representation; the primary "System2 already knows"
  arm, deliberately chosen to be maximally favourable to the null).
* ``plan_z0``      [4, 768]   the frozen projection actually handed to System1.
* ``history_memory`` [K, 256] ``M_t``, the Past Head bottleneck the bridge reads.
* ``history_rel_poses`` [K, 4] + ``visibility`` [K, 4]  the geometry/occlusion
  control.  EXP-02/EXP-04 showed ``M_t`` is close to a function of pose, so a
  readout that reaches the same accuracy from pose alone would mean the useful
  content is geometry, not learned memory.  Reporting that honestly is the
  point of caching it.

``future_visibility`` (the future head's first time bin) is stored too, so the
EXP-12 D2 "constant front predictor" reading can be re-checked on all 30k
states rather than the 4k subsample, at no extra cost.

Labels are never recomputed here.  ``--per-state-jsonl`` is EXP-12's
``d1_per_state.jsonl``; ``oracle_view``/``native_view`` come from that one
implementation and are joined on ``sample_key``, exactly as the D2 probe does.

Poses are the Habitat ground truth stored in the DAgger tars, not deployment
AMB3R VO (EXP-12 boundary 3): a *positive* readout result from this cache is an
upper bound.  ``src/config_schema.py`` fail-closes on "DAgger + PPA" training
for that reason, so the probe constructs the dataset explicitly and leaves the
training guard intact, exactly like ``probe_future_head_recovery.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

VIEWS = ("front", "right", "back", "left")
SCHEMA = "heatmapvln-exp13-decision-features-v1"


def load_oracle_index(path: Path) -> dict[str, dict[str, Any]]:
    """Read every EXP-12 per-state row, both buckets, keyed by sample_key."""
    index: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("sample_key")
            if key:
                index[str(key)] = row
    return index


def merge(paths: list[Path], output: Path) -> None:
    """Concatenate shard caches in shard order and re-verify the join."""
    shards = [np.load(path, allow_pickle=False) for path in paths]
    metas = [json.loads(Path(str(path) + ".json").read_text(encoding="utf-8")) for path in paths]
    array_keys = [key for key in shards[0].files if key != "sample_key"]
    merged = {key: np.concatenate([shard[key] for shard in shards], axis=0) for key in array_keys}
    merged["sample_key"] = np.concatenate([shard["sample_key"] for shard in shards], axis=0)
    counts = {key: int(value.shape[0]) for key, value in merged.items()}
    if len(set(counts.values())) != 1:
        raise SystemExit(f"merged arrays disagree on state count: {counts}")
    if len(set(merged["sample_key"].tolist())) != counts["sample_key"]:
        raise SystemExit("merged cache contains duplicate sample keys")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    meta = {
        "schema": SCHEMA,
        "states": counts["sample_key"],
        "merged_from": [str(path) for path in paths],
        "shard_meta": metas,
        "oracle_view_counts": _view_counts(merged["oracle_view"]),
        "source_type_counts": {
            name: int(count)
            for name, count in Counter(merged["source_type"].tolist()).items()
        },
    }
    Path(str(output) + ".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in meta.items() if k != "shard_meta"}, ensure_ascii=False, indent=2))
    print(f"wrote {output}")


def _view_counts(views: np.ndarray) -> dict[str, int]:
    counts = Counter(int(value) for value in views.tolist())
    return {
        (VIEWS[index] if 0 <= index < len(VIEWS) else "undefined"): int(count)
        for index, count in sorted(counts.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge", type=Path, action="append", default=[])
    parser.add_argument("--config", type=Path, default=None, help="unmodified PPA training config; supplies the model only")
    parser.add_argument("--collection-root", type=Path, default=None, help="DAgger collection root holding shard_*/")
    parser.add_argument("--checkpoint", type=Path, default=None, help="deployed v2 best.pth")
    parser.add_argument("--per-state-jsonl", type=Path, default=None, help="EXP-12 d1_per_state.jsonl")
    parser.add_argument(
        "--buckets",
        default="dagger_hard,dagger_normal",
        help="comma-separated DAgger source types to cache",
    )
    parser.add_argument("--max-states", type=int, default=0, help="0 means every state (total across shards)")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    if args.merge:
        merge(args.merge, args.output_npz)
        return

    for name in ("config", "checkpoint", "per_state_jsonl", "collection_root"):
        if getattr(args, name) is None:
            raise SystemExit(f"--{name.replace('_', '-')} is required unless --merge is used")
    buckets = [value.strip() for value in str(args.buckets).split(",") if value.strip()]
    if not buckets:
        raise SystemExit("--buckets cannot be empty")

    import torch
    from transformers import AutoProcessor

    from scripts.training.model_builder import build_model
    from scripts.training.pose_adaptation import load_past_plan_action_initialization
    from src.config_schema import load_and_validate_config
    from src.data.internnav_heatmap_control_collator import InternNavHeatmapControlCollator
    from src.data.trajectory_dagger_dataset import TrajectoryDaggerDataset

    oracle_index = load_oracle_index(args.per_state_jsonl)
    print(f"oracle index: {len(oracle_index)} states (all buckets)", flush=True)

    cfg = load_and_validate_config(args.config)
    if cfg["data"]["dataset_type"] != "trajectory":
        raise SystemExit(
            "--config must be the unmodified PPA training config; the DAgger rows "
            "are supplied by --collection-root, not by the config"
        )
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
    policy_fingerprint = fingerprints.pop()

    dataset = TrajectoryDaggerDataset(
        collection_roots=[str(shard) for shard in shard_paths],
        source_types=buckets,
        num_history=8,
        image_size=tuple(cfg["data"]["image_size"]),
        require_lookdown=True,
        expected_policy_mode="internnav_native",
        expected_policy_fingerprint=policy_fingerprint,
    )
    print(
        f"dataset: {len(dataset)} states from {len(shard_paths)} shards "
        f"(buckets={buckets}, policy {policy_fingerprint})",
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
        # The deployed bridge belongs to the deployed model: plan_z is
        # Z0 + bridge(M).  Probing the reset bridge would probe a different
        # model than the one every other EXP-1x number describes.
        load_trained_bridge=True,
    )
    print(f"checkpoint init: {report}", flush=True)
    model.eval()

    indices = [i for i in range(len(dataset)) if i % args.shard_count == args.shard_index]
    budget = len(indices)
    if args.max_states > 0:
        budget = min(budget, max(1, args.max_states // max(1, args.shard_count)))

    columns: dict[str, list[Any]] = {
        name: []
        for name in (
            "sample_key",
            "scene_id",
            "episode_key",
            "source_type",
            "tags",
            "oracle_view",
            "native_view",
            "traj_hidden",
            "plan_z0",
            "history_memory",
            "history_memory_mask",
            "history_rel_poses",
            "history_visibility",
            "history_age_steps",
            "future_visibility",
        )
    }
    skipped_unjoined = 0
    started = time.perf_counter()

    with torch.no_grad():
        for visited, index in enumerate(indices):
            if len(columns["sample_key"]) >= budget:
                break
            sample = dataset[index]
            key = str(sample.get("sample_key") or "")
            truth = oracle_index.get(key)
            if truth is None:
                skipped_unjoined += 1
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
                traj_images=batch["traj_images"].to(device) if "traj_images" in batch else None,
                sample_trajectory=False,
                return_heatmaps=True,
                return_heatmap_logits=False,
                return_future_heatmaps=True,
                return_history_memory=True,
                return_actions=False,
                return_lm_loss=False,
            )
            missing = sorted(
                {
                    "traj_hidden_states",
                    "plan_z0",
                    "history_memory",
                    "history_memory_mask",
                    "visibility",
                    "future_visibility",
                }
                - set(output)
            )
            if missing:
                raise RuntimeError(f"forward is missing {missing}; keys={sorted(output)}")

            def take(name: str) -> np.ndarray:
                return output[name][0].detach().float().cpu().numpy().astype(np.float16)

            columns["sample_key"].append(key)
            columns["scene_id"].append(str(truth.get("scene_id") or ""))
            columns["episode_key"].append(str(truth.get("episode_key") or ""))
            columns["source_type"].append(str(truth.get("source_type") or ""))
            columns["tags"].append("|".join(str(tag) for tag in (truth.get("tags") or [])))
            columns["oracle_view"].append(
                -1 if truth.get("oracle_view") is None else int(truth["oracle_view"])
            )
            columns["native_view"].append(
                -1 if truth.get("native_view") is None else int(truth["native_view"])
            )
            columns["traj_hidden"].append(take("traj_hidden_states"))
            columns["plan_z0"].append(take("plan_z0"))
            columns["history_memory"].append(take("history_memory"))
            columns["history_memory_mask"].append(
                output["history_memory_mask"][0].detach().cpu().numpy().astype(np.uint8)
            )
            columns["history_rel_poses"].append(
                batch["history_rel_poses"][0].detach().float().cpu().numpy().astype(np.float16)
            )
            columns["history_visibility"].append(take("visibility"))
            columns["history_age_steps"].append(
                batch["history_age_steps"][0].detach().cpu().numpy().astype(np.int16)
            )
            columns["future_visibility"].append(take("future_visibility"))

            done = len(columns["sample_key"])
            if done % max(1, args.progress_every) == 0:
                rate = done / (time.perf_counter() - started)
                print(
                    f"  {done}/{budget} cached ({visited + 1} visited, {rate:.2f} states/s)",
                    flush=True,
                )

    if not columns["sample_key"]:
        raise SystemExit("cached no states; check --per-state-jsonl and --buckets")

    arrays = {
        name: (
            np.asarray(values, dtype=np.str_)
            if name in ("sample_key", "scene_id", "episode_key", "source_type", "tags")
            else np.stack(values, axis=0)
            if isinstance(values[0], np.ndarray)
            else np.asarray(values, dtype=np.int32)
        )
        for name, values in columns.items()
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)

    meta = {
        "schema": SCHEMA,
        "states": int(arrays["sample_key"].shape[0]),
        "skipped_unjoined": skipped_unjoined,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "buckets": buckets,
        "elapsed_seconds": time.perf_counter() - started,
        "inputs": {
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "per_state_jsonl": str(args.per_state_jsonl),
            "collection_root": str(args.collection_root),
            "policy_fingerprint": policy_fingerprint,
        },
        "checkpoint_init": report,
        "feature_shapes": {
            name: list(arrays[name].shape[1:])
            for name in (
                "traj_hidden",
                "plan_z0",
                "history_memory",
                "history_rel_poses",
                "history_visibility",
                "future_visibility",
            )
        },
        "oracle_view_counts": _view_counts(arrays["oracle_view"]),
        "source_type_counts": {
            name: int(count)
            for name, count in Counter(arrays["source_type"].tolist()).items()
        },
    }
    Path(str(args.output_npz) + ".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"wrote {args.output_npz}")


if __name__ == "__main__":
    main()

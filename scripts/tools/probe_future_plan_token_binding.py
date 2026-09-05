#!/usr/bin/env python3
"""EXP-10 H2: is Z's i-th vector bound to the i-th future segment?

The method section says the future head decodes the i-th plan vector into the
i-th future segment.  H2 asks whether that binding is real structure or just a
description of how the tensor happens to be laid out.

The probe follows the pre-registered setup literally: wrap
``PastPlanAction.decode_future``, capture ``plan_z`` / ``past_output`` /
``past_head`` from the ordinary forward, then decode a second time from the
*same* captured state with ``plan_z`` permuted along the token axis.  Both
decodes therefore share one forward pass, one batch and one set of ground-truth
tensors -- the permutation is the only thing that differs.

Metrics are not reimplemented here.  ``future_tube_sufficient_statistics`` is
the same function ``scripts/training/validate.py`` accumulates, so an arm's
numbers are directly comparable with the bridge-on arm recorded in the ledger.

Arms
----
``identity``            no permutation; the internal reference every other arm
                        is scored against, and a wiring check -- it must land on
                        the bridge-on validation numbers.
``reverse``             [3,2,1,0]; a derangement (no vector keeps its slot).
``roll1`` / ``roll2``   [3,0,1,2] and [2,3,0,1]; the other two rotations.
``random_derangement``  a fresh per-sample derangement, seeded.  This is the
                        headline arm: a single fixed permutation could be
                        unluckily easy or hard, and averaging over derangements
                        removes that.
``cross_sample``        ``plan_z`` from a *different* batch entirely.  Not a
                        permutation at all -- it is the "totally wrong Z" upper
                        bound on damage, which says how much of the scale of a
                        drop is attributable to ordering rather than to Z's
                        content.  Without it, "shuffling costs X%" has nothing
                        to be large or small relative to.

Criteria live in the ledger (pre-registered 2026-09-04) and are read off the
relative drop of ``soft_iou`` / ``topk_support_recall`` against ``identity``.
This tool prints the drops; it does not judge them.

Per-batch statistics are written next to the summary as ``.npz`` so the arms can
be re-scored on any subset and so a *paired* bootstrap over batches is possible:
every arm sees the same resampled batch indices, which is the only way to put an
interval on a difference between arms that share one forward pass.

``--shard-index`` / ``--shard-count`` split the val set into **contiguous**
blocks.  Contiguous rather than strided on purpose: ``cross_sample`` reads the
previous batch's Z, and striding would silently change what "the previous batch"
means between a sharded and an unsharded run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from scripts.training.model_builder import build_model  # noqa: E402
from scripts.training.pose_adaptation import (  # noqa: E402
    load_past_plan_action_initialization,
)
from src.config_schema import load_and_validate_config  # noqa: E402
from src.data.factory import build_dataset  # noqa: E402
from src.data.internnav_heatmap_control_collator import (  # noqa: E402
    InternNavHeatmapControlCollator,
)
from src.models.future_trajectory_objective import (  # noqa: E402
    future_tube_metrics_from_statistics,
    future_tube_sufficient_statistics,
)

SCHEMA = "heatmapvln-exp10-h2-token-binding-v1"
FIXED_ARMS: dict[str, tuple[int, ...]] = {
    "identity": (0, 1, 2, 3),
    "reverse": (3, 2, 1, 0),
    "roll1": (3, 0, 1, 2),
    "roll2": (2, 3, 0, 1),
}


def random_derangement(n: int, generator: torch.Generator) -> torch.Tensor:
    """A permutation of ``range(n)`` with no fixed point (rejection sampling)."""
    while True:
        candidate = torch.randperm(n, generator=generator)
        if not bool((candidate == torch.arange(n)).any()):
            return candidate


def permute_plan_z(plan_z: torch.Tensor, arm: str, generator: torch.Generator) -> torch.Tensor:
    """Return ``plan_z`` with its token axis permuted according to ``arm``."""
    if arm in FIXED_ARMS:
        order = torch.tensor(FIXED_ARMS[arm], device=plan_z.device)
        return plan_z.index_select(1, order)
    if arm == "random_derangement":
        out = torch.empty_like(plan_z)
        for row in range(plan_z.shape[0]):
            order = random_derangement(plan_z.shape[1], generator).to(plan_z.device)
            out[row] = plan_z[row].index_select(0, order)
        return out
    raise ValueError(f"unknown arm {arm}")


ARMS = tuple(FIXED_ARMS) + ("random_derangement", "cross_sample")


def metrics_of(summed: "np.ndarray") -> Any:
    return future_tube_metrics_from_statistics(torch.from_numpy(summed))


def merge_and_report(args: argparse.Namespace) -> None:
    """Pool per-batch statistics from shards and put a paired interval on them.

    The bootstrap resamples *batches*, using the same resampled indices for
    every arm.  Arms share one forward pass per batch, so a drop between two
    arms is a paired quantity; resampling them independently would inflate the
    interval with variance that the comparison does not actually carry.
    """
    stacks: dict[str, Any] = {}
    for path in args.merge:
        with np.load(path) as handle:
            for arm in ARMS:
                stacks.setdefault(arm, []).append(handle[arm])
    per_arm = {arm: np.concatenate(stacks[arm], axis=0) for arm in ARMS}
    n_batches = per_arm["identity"].shape[0]
    if any(per_arm[arm].shape[0] != n_batches for arm in ARMS):
        raise SystemExit("arms disagree on batch count; the shards are not paired")

    point = {arm: metrics_of(per_arm[arm].sum(axis=0)) for arm in ARMS}
    base = point["identity"]

    rng = np.random.default_rng(0)
    draws: dict[str, dict[str, list[float]]] = {
        arm: {"soft_iou": [], "topk": []} for arm in ARMS
    }
    for _ in range(args.bootstrap):
        picks = rng.integers(0, n_batches, size=n_batches)
        resampled = {arm: per_arm[arm][picks].sum(axis=0) for arm in ARMS}
        ref = metrics_of(resampled["identity"])
        for arm in ARMS:
            m = metrics_of(resampled[arm])
            if ref.soft_iou:
                draws[arm]["soft_iou"].append(
                    (ref.soft_iou - m.soft_iou) / ref.soft_iou * 100.0
                )
            if ref.topk_support_recall:
                draws[arm]["topk"].append(
                    (ref.topk_support_recall - m.topk_support_recall)
                    / ref.topk_support_recall
                    * 100.0
                )

    def interval(values: list[float]) -> dict[str, float]:
        array = np.asarray(values)
        return {
            "lo95": float(np.percentile(array, 2.5)),
            "hi95": float(np.percentile(array, 97.5)),
            "median": float(np.median(array)),
        }

    result: dict[str, Any] = {
        "schema": SCHEMA + "-merged",
        "batches_scored": int(n_batches),
        "supported_view_bins": base.supported_view_bins,
        "bootstrap_resamples": args.bootstrap,
        "merged_from": [str(path) for path in args.merge],
        "arms": {},
    }
    for arm in ARMS:
        m = point[arm]
        result["arms"][arm] = {
            "soft_iou": m.soft_iou,
            "topk_support_recall": m.topk_support_recall,
            "supported_view_bins": m.supported_view_bins,
            "soft_iou_relative_drop_pct": (
                None if not base.soft_iou
                else (base.soft_iou - m.soft_iou) / base.soft_iou * 100.0
            ),
            "topk_relative_drop_pct": (
                None if not base.topk_support_recall
                else (base.topk_support_recall - m.topk_support_recall)
                / base.topk_support_recall * 100.0
            ),
            "soft_iou_drop_ci": interval(draws[arm]["soft_iou"]),
            "topk_drop_ci": interval(draws[arm]["topk"]),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote {args.output_json}")


def to_device(value: Any, device: torch.device) -> Any:
    """Move tensors (including those nested one dict deep) onto ``device``."""
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def build_forward_kwargs(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Assemble the forward call for ``InternNavHeatmapControlCollator`` batches.

    The batch carries many keys ``VLNPipeline.forward`` does not accept, so the
    arguments are listed explicitly rather than splatted.  This mirrors the call
    in ``scripts/tools/probe_future_head_recovery.py``, which drives the same
    collator; that collator tokenizes into ``pano_inputs`` rather than the
    worker-tokenized ``pixel_values`` path used by ``scripts/train.py``.
    """
    text = batch.get("text")
    return {
        "video_frames": None,
        "instruction_text": list(text) if text else None,
        "current_observation": to_device(batch.get("current_frame"), device),
        "panoramic_inputs": to_device(batch.get("pano_inputs"), device),
        "panoramic_num_histories": batch.get("pano_num_histories"),
        "panoramic_text_anchor_positions": batch.get("pano_text_anchor_positions"),
        "heatmap_single_view_inputs": to_device(
            batch.get("heatmap_single_view_inputs"), device
        ),
        "heatmap_single_view_num_histories": batch.get(
            "heatmap_single_view_num_histories"
        ),
        "heatmap_control_history_mask": to_device(
            batch.get("heatmap_control_history_mask"), device
        ),
        "history_valid_mask": to_device(batch.get("history_valid_mask"), device),
        "history_age_steps": to_device(batch.get("history_age_steps"), device),
        "history_rel_poses": to_device(batch.get("history_rel_poses"), device),
        "traj_images": to_device(batch.get("traj_images"), device),
        "sample_trajectory": False,
        "return_heatmaps": True,
        "return_heatmap_logits": True,
        "return_future_heatmaps": True,
        "return_actions": False,
        "return_lm_loss": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--merge",
        type=Path,
        action="append",
        default=[],
        help="per-batch .npz files to pool; with this the probe only reports",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    if args.merge:
        merge_and_report(args)
        return

    cfg = load_and_validate_config(str(args.config))
    torch.manual_seed(args.seed)

    trajectory_cfg = cfg["data"].get("trajectory", {}) or {}
    val_dataset = build_dataset(
        cfg,
        split=cfg["data"].get("val_split", "val"),
        root=cfg["data"].get("val_root") or cfg["data"]["dataset_root"],
        samples_per_clip=trajectory_cfg.get("val_samples_per_clip", 1),
        random_subsequence=False,
        enable_trajectory_augmentation=False,
    )
    total_samples = len(val_dataset)
    if args.shard_count > 1:
        block = -(-total_samples // args.shard_count)
        start = args.shard_index * block
        stop = min(start + block, total_samples)
        val_dataset = Subset(val_dataset, list(range(start, stop)))
        print(
            f"val dataset: {total_samples} samples, contiguous shard "
            f"{args.shard_index}/{args.shard_count} = [{start}, {stop})",
            flush=True,
        )
    else:
        print(f"val dataset: {total_samples} samples", flush=True)

    processor = AutoProcessor.from_pretrained(
        cfg["model"]["llm"]["model_path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    collator = InternNavHeatmapControlCollator(
        processor,
        n_traj_query=4,
        max_seq_length=int(cfg["model"]["llm"].get("max_seq_length", 8192)),
        teacher_force_system2_answer=True,
        # H2 scores the future tube, so the future ground truth has to be in
        # the batch; this is the one collator flag that differs from EXP-12.
        include_future_trajectory_targets=True,
        required_history_pose_provider=None,
    )
    loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["optim"]["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collator,
        persistent_workers=False,
    )

    device = torch.device(args.device)
    model = build_model(cfg, verbose=True, device=args.device, enable_action_head=True)
    if model.vlm_backbone.model is None:
        model.vlm_backbone._load_model()
    model._ensure_heatmap_vln()
    report = load_past_plan_action_initialization(
        model,
        str(args.checkpoint),
        stage="stage2_joint",
        # H2 probes the deployed model, so the trained bridge must come along;
        # the stage-transition default would leave it at exact zero.
        load_trained_bridge=True,
    )
    print(f"checkpoint init: {report}", flush=True)
    model.eval()

    # Wrap decode_future so the ordinary forward hands us the state a second
    # decode needs.  Nothing about the first decode changes.
    chain = model.past_plan_action
    original_decode = chain.decode_future
    captured: dict[str, Any] = {}

    def capturing_decode(plan_z, *, past_output, past_head, time_mask=None):
        captured["plan_z"] = plan_z
        captured["past_output"] = past_output
        captured["past_head"] = past_head
        captured["time_mask"] = time_mask
        return original_decode(
            plan_z, past_output=past_output, past_head=past_head, time_mask=time_mask
        )

    chain.decode_future = capturing_decode  # type: ignore[method-assign]

    arms = list(FIXED_ARMS) + ["random_derangement", "cross_sample"]
    stats: dict[str, torch.Tensor] = {
        arm: torch.zeros(17, dtype=torch.float64, device=device) for arm in arms
    }
    # One 17-vector per batch per arm.  These are additive, so any subset can be
    # re-scored exactly and a paired bootstrap becomes possible.
    per_batch: dict[str, list[Any]] = {arm: [] for arm in arms}
    generator = torch.Generator().manual_seed(args.seed)
    previous_plan_z: torch.Tensor | None = None
    batches_used = 0
    cross_sample_batches = 0
    started = time.perf_counter()

    with torch.no_grad():
        for index, batch in enumerate(loader):
            if batches_used >= args.max_batches:
                break
            required = (
                "future_trajectory_visibility",
                "future_trajectory_heatmap",
                "future_trajectory_time_mask",
            )
            if any(batch.get(key) is None for key in required):
                continue

            captured.clear()
            output = model(**build_forward_kwargs(batch, device))
            if "plan_z" not in captured:
                raise SystemExit(
                    "decode_future was never called; the forward did not take "
                    "the future-heatmap path"
                )

            gt_visibility = to_device(batch["future_trajectory_visibility"], device)
            gt_heatmaps = to_device(batch["future_trajectory_heatmap"], device)
            time_mask = to_device(batch["future_trajectory_time_mask"], device)

            def score(arm: str, decoded: dict[str, torch.Tensor]) -> None:
                batch_stats = future_tube_sufficient_statistics(
                    pred_visibility_logits=decoded["future_visibility"],
                    pred_heatmaps=decoded["future_heatmaps"],
                    gt_visibility=gt_visibility,
                    gt_heatmaps=gt_heatmaps,
                    future_time_mask=time_mask,
                )
                stats[arm] += batch_stats
                per_batch[arm].append(batch_stats.detach().cpu().numpy().copy())

            plan_z = captured["plan_z"]
            # ``cross_sample`` needs a Z from an earlier batch, so the very
            # first batch can only seed the buffer.  Nothing is scored on it --
            # scoring identity there while cross_sample sat it out would leave
            # the arms on different sample sets and make the relative drops
            # incomparable, which is the whole quantity H2 is judged on.
            if previous_plan_z is None or previous_plan_z.shape != plan_z.shape:
                previous_plan_z = plan_z.clone()
                continue

            score("identity", output)
            for arm in ("reverse", "roll1", "roll2", "random_derangement"):
                score(
                    arm,
                    original_decode(
                        permute_plan_z(plan_z, arm, generator),
                        past_output=captured["past_output"],
                        past_head=captured["past_head"],
                        time_mask=captured["time_mask"],
                    ),
                )
            score(
                "cross_sample",
                original_decode(
                    previous_plan_z,
                    past_output=captured["past_output"],
                    past_head=captured["past_head"],
                    time_mask=captured["time_mask"],
                ),
            )
            cross_sample_batches += 1
            previous_plan_z = plan_z.clone()

            batches_used += 1
            if batches_used % 25 == 0:
                rate = batches_used / (time.perf_counter() - started)
                print(
                    f"  {batches_used}/{args.max_batches} batches "
                    f"({index + 1} seen, {rate:.2f} batch/s)",
                    flush=True,
                )

    if batches_used == 0:
        raise SystemExit("no batch carried future supervision; nothing was scored")

    metrics = {arm: future_tube_metrics_from_statistics(stats[arm]) for arm in arms}
    reference = metrics["identity"]

    def relative_drop(value: float, base: float) -> float | None:
        return None if base == 0 else (base - value) / base * 100.0

    # Every arm must have seen exactly the same supported bins; if they have
    # not, the relative drops are comparing different sample sets.
    aligned = len({metrics[arm].supported_view_bins for arm in arms}) == 1

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "batches_scored": batches_used,
        "cross_sample_batches": cross_sample_batches,
        "arms_share_one_sample_set": aligned,
        "arms": {},
        "inputs": {
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "max_batches": args.max_batches,
            "seed": args.seed,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    for arm in arms:
        m = metrics[arm]
        result["arms"][arm] = {
            "soft_iou": m.soft_iou,
            "topk_support_recall": m.topk_support_recall,
            "visibility_f1": m.visibility_f1,
            "valid_time_bins": m.valid_time_bins,
            "supported_view_bins": m.supported_view_bins,
            "per_view_soft_iou": list(m.per_view_soft_iou),
            "per_view_support": list(m.per_view_support),
            "soft_iou_relative_drop_pct": relative_drop(m.soft_iou, reference.soft_iou),
            "topk_relative_drop_pct": relative_drop(
                m.topk_support_recall, reference.topk_support_recall
            ),
        }

    stats_path = args.output_json.with_suffix(".per_batch.npz")
    np.savez_compressed(
        stats_path, **{arm: np.stack(per_batch[arm]) for arm in arms}
    )
    result["per_batch_stats"] = str(stats_path)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()

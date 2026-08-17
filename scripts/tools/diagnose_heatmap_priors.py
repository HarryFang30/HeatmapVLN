#!/usr/bin/env python3
"""Fit and evaluate a train-only empirical heatmap prior.

This is the non-parametric half of the Task-3.5 shortcut diagnostic.  It uses
the exact deterministic, scene-stratified sample selection from Task 3, fits
one visibility probability and one mean heatmap for every
``(history_slot, panorama_view)`` pair on the training subset, and evaluates
the fixed templates on the scene-disjoint validation subset.

No image, pose, checkpoint, or validation target is used to fit the prior.
The saved selection manifests and compact per-sample predictions make that
contract auditable and support paired downstream comparisons without storing
full 64x64 prediction tensors for every validation sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools.diagnose_heatmap_shortcuts import (
    build_dataset,
    compute_metrics,
    load_config,
    make_loss,
    scene_stratified_indices,
    set_seed,
)

LOGGER = logging.getLogger("heatmap_empirical_prior")
MODE = "empirical-prior"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-history", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument(
        "--max-clip-id",
        type=int,
        default=0,
        help="Restrict an append-only collection to clip ids <= this value.",
    )
    parser.add_argument("--visibility-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Loss-evaluation device only; this diagnostic never builds the VLM.",
    )
    return parser.parse_args()


def _stable_hash(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def sample_identity(dataset: Any, sample_idx: int) -> tuple[str, str]:
    """Return a stable ``(relative_clip:frame, scene)`` sample identity."""
    clip_idx, frame_idx = dataset.sample_index[sample_idx]
    clip_path = Path(dataset.clips[clip_idx])
    root = Path(dataset.root)
    try:
        relative_clip = clip_path.relative_to(root)
    except ValueError:
        relative_clip = clip_path
    scene = clip_path.parent.name
    return f"{relative_clip.as_posix()}:frame={int(frame_idx)}", scene


def selection_manifest(dataset: Any, indices: list[int]) -> dict[str, Any]:
    identities_and_scenes = [sample_identity(dataset, index) for index in indices]
    sample_ids = [item[0] for item in identities_and_scenes]
    scenes = sorted({item[1] for item in identities_and_scenes})
    return {
        "samples": len(sample_ids),
        "sample_ids": sample_ids,
        "sample_identity_hash": _stable_hash(sample_ids),
        "sample_identity_sha256": _stable_hash(sample_ids),
        "scenes": scenes,
        "scene_hash": _stable_hash(scenes),
    }


def assert_scene_disjoint(
    train_manifest: dict[str, Any],
    val_manifest: dict[str, Any],
) -> None:
    overlap = sorted(set(train_manifest["scenes"]) & set(val_manifest["scenes"]))
    if overlap:
        raise RuntimeError(
            "Task-3.5 train/validation scenes overlap: " + ", ".join(overlap)
        )


def fit_empirical_prior(
    dataset: Any,
    indices: list[int],
    *,
    visibility_alpha: float = 0.5,
) -> dict[str, torch.Tensor | float | int]:
    """Fit a slot-by-view prior from training targets only.

    Visibility uses a symmetric Beta/Laplace-style smoothed estimate
    ``(positive + alpha) / (n + 2*alpha)``.  The heatmap prior is the raw
    training-target mean, including the all-zero maps of invisible views.  It
    is therefore a fixed expected heatmap, not an oracle conditioned on a
    validation sample's visibility.
    """
    if not indices:
        raise ValueError("Cannot fit an empirical prior from zero samples")
    if visibility_alpha <= 0:
        raise ValueError("visibility_alpha must be positive")

    visibility_sum: torch.Tensor | None = None
    heatmap_sum: torch.Tensor | None = None
    expected_visibility_shape: tuple[int, ...] | None = None
    expected_heatmap_shape: tuple[int, ...] | None = None

    for sample_idx in indices:
        sample = dataset[sample_idx]
        visibility = sample["gt_visibility"].detach().float().cpu()
        heatmap = sample["heatmap"].detach().float().cpu()
        if heatmap.shape[:2] != visibility.shape:
            raise ValueError(
                "GT visibility/heatmap shape mismatch while fitting prior: "
                f"visibility={tuple(visibility.shape)} heatmap={tuple(heatmap.shape)}"
            )
        if expected_visibility_shape is None:
            expected_visibility_shape = tuple(visibility.shape)
            expected_heatmap_shape = tuple(heatmap.shape)
            visibility_sum = torch.zeros_like(visibility, dtype=torch.float64)
            heatmap_sum = torch.zeros_like(heatmap, dtype=torch.float64)
        elif (
            tuple(visibility.shape) != expected_visibility_shape
            or tuple(heatmap.shape) != expected_heatmap_shape
        ):
            raise ValueError(
                "Task-3.5 requires fixed slot/view target shapes; got "
                f"visibility={tuple(visibility.shape)} heatmap={tuple(heatmap.shape)}, "
                f"expected visibility={expected_visibility_shape} heatmap={expected_heatmap_shape}"
            )
        assert visibility_sum is not None and heatmap_sum is not None
        visibility_sum.add_(visibility.to(torch.float64))
        heatmap_sum.add_(heatmap.to(torch.float64))

    assert visibility_sum is not None and heatmap_sum is not None
    sample_count = len(indices)
    visibility_probability = (
        visibility_sum + float(visibility_alpha)
    ) / (sample_count + 2.0 * float(visibility_alpha))
    visibility_logits = torch.logit(visibility_probability)
    mean_heatmap = heatmap_sum / sample_count
    return {
        "sample_count": sample_count,
        "visibility_alpha": float(visibility_alpha),
        "visibility_positive_count": visibility_sum.float(),
        "visibility_probability": visibility_probability.float(),
        # compute_metrics expects logits and applies sigmoid internally.
        "visibility_logits": visibility_logits.float(),
        "mean_heatmap": mean_heatmap.float(),
    }


def _peak_xy(heatmaps: torch.Tensor) -> list[list[list[int]]]:
    width = int(heatmaps.shape[-1])
    flat_indices = heatmaps.reshape(*heatmaps.shape[:-2], -1).argmax(dim=-1)
    output: list[list[list[int]]] = []
    for history_row in flat_indices.tolist():
        output.append(
            [[int(flat_index % width), int(flat_index // width)] for flat_index in history_row]
        )
    return output


def compact_prediction_record(
    *,
    sample_id: str,
    visibility_logits: torch.Tensor,
    mean_heatmap: torch.Tensor,
    sample: dict[str, Any],
) -> dict[str, Any]:
    gt_visibility = sample["gt_visibility"].detach().float().cpu()
    gt_heatmap = sample["heatmap"].detach().float().cpu()
    return {
        "sample_id": sample_id,
        "visibility_logits": visibility_logits.tolist(),
        "visibility_probability": visibility_logits.sigmoid().tolist(),
        "gt_visibility": gt_visibility.tolist(),
        "pred_xy": _peak_xy(mean_heatmap),
        "gt_xy": _peak_xy(gt_heatmap),
    }


@torch.no_grad()
def evaluate_empirical_prior(
    dataset: Any,
    indices: list[int],
    prior: dict[str, torch.Tensor | float | int],
    *,
    criterion: torch.nn.Module | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate fixed train priors with the exact Task-3 metric function."""
    visibility_logits = prior["visibility_logits"]
    mean_heatmap = prior["mean_heatmap"]
    if not torch.is_tensor(visibility_logits) or not torch.is_tensor(mean_heatmap):
        raise TypeError("Prior visibility_logits and mean_heatmap must be tensors")

    metric_records: list[dict[str, torch.Tensor]] = []
    compact_records: list[dict[str, Any]] = []
    losses: list[float] = []
    for sample_idx in indices:
        sample = dataset[sample_idx]
        sample_id, _scene = sample_identity(dataset, sample_idx)
        gt_visibility = sample["gt_visibility"].detach().float().cpu()
        gt_heatmap = sample["heatmap"].detach().float().cpu()
        if (
            tuple(gt_visibility.shape) != tuple(visibility_logits.shape)
            or tuple(gt_heatmap.shape) != tuple(mean_heatmap.shape)
        ):
            raise ValueError(
                f"Validation target shape mismatch for {sample_id}: "
                f"visibility={tuple(gt_visibility.shape)} heatmap={tuple(gt_heatmap.shape)}"
            )
        metric_records.append(
            {
                "visibility": visibility_logits.unsqueeze(0),
                "heatmaps": mean_heatmap.unsqueeze(0),
                "gt_visibility": gt_visibility,
                "gt_heatmaps": gt_heatmap,
            }
        )
        compact_records.append(
            compact_prediction_record(
                sample_id=sample_id,
                visibility_logits=visibility_logits,
                mean_heatmap=mean_heatmap,
                sample=sample,
            )
        )
        if criterion is not None:
            history_mask = torch.ones(
                1,
                gt_heatmap.shape[0],
                dtype=torch.bool,
                device=device,
            )
            loss_dict = criterion(
                visibility_logits.unsqueeze(0).to(device),
                mean_heatmap.unsqueeze(0).to(device),
                gt_vis=gt_visibility.unsqueeze(0).to(device),
                gt_heatmaps=gt_heatmap.unsqueeze(0).to(device),
                history_mask=history_mask,
            )
            losses.append(float(loss_dict["total"].detach().float().item()))

    metrics = compute_metrics(metric_records)
    if losses:
        metrics["loss"] = float(sum(losses) / len(losses))
    metrics["samples"] = len(indices)
    return metrics, compact_records


def _prior_summary(prior: dict[str, torch.Tensor | float | int]) -> dict[str, Any]:
    visibility_count = prior["visibility_positive_count"]
    visibility_probability = prior["visibility_probability"]
    visibility_logits = prior["visibility_logits"]
    mean_heatmap = prior["mean_heatmap"]
    assert all(
        torch.is_tensor(value)
        for value in (
            visibility_count,
            visibility_probability,
            visibility_logits,
            mean_heatmap,
        )
    )
    return {
        "sample_count": int(prior["sample_count"]),
        "visibility_alpha": float(prior["visibility_alpha"]),
        "visibility_positive_count": visibility_count.tolist(),
        "visibility_probability": visibility_probability.tolist(),
        "visibility_logits": visibility_logits.tolist(),
        "mean_heatmap_shape": list(mean_heatmap.shape),
        "mean_heatmap_peak_xy": _peak_xy(mean_heatmap),
        "mean_heatmap_sha256": hashlib.sha256(
            mean_heatmap.contiguous().numpy().tobytes()
        ).hexdigest(),
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.train_samples != 128 or args.val_samples != 64 or args.num_history != 2:
        LOGGER.warning(
            "Non-canonical Task-3.5 contract requested: train=%d val=%d history=%d",
            args.train_samples,
            args.val_samples,
            args.num_history,
        )
    set_seed(args.seed)
    cfg = load_config(args)
    train_dataset = build_dataset(cfg, "train", max_clip_id=args.max_clip_id)
    val_dataset = build_dataset(cfg, "val", max_clip_id=args.max_clip_id)
    train_indices = scene_stratified_indices(train_dataset, args.train_samples)
    val_indices = scene_stratified_indices(val_dataset, args.val_samples)

    train_manifest = selection_manifest(train_dataset, train_indices)
    val_manifest = selection_manifest(val_dataset, val_indices)
    assert_scene_disjoint(train_manifest, val_manifest)

    # This is the only fitting call.  Its API receives the training dataset
    # and training indices, making validation-target leakage explicit and
    # mechanically testable.
    prior = fit_empirical_prior(
        train_dataset,
        train_indices,
        visibility_alpha=args.visibility_alpha,
    )
    device = torch.device(args.device)
    criterion = make_loss(cfg, device)
    metrics, predictions = evaluate_empirical_prior(
        val_dataset,
        val_indices,
        prior,
        criterion=criterion,
        device=device,
    )

    output_dir = Path(args.output_dir) / MODE
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_path = output_dir / "prior.pth"
    torch.save(prior, prior_path)
    selections_path = output_dir / "selection_manifest.json"
    with selections_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"train": train_manifest, "val": val_manifest},
            handle,
            indent=2,
            ensure_ascii=False,
        )
    predictions_path = output_dir / "compact_predictions.json"
    with predictions_path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2, ensure_ascii=False, allow_nan=False)

    report = {
        "mode": MODE,
        "seed": args.seed,
        "config": str(Path(args.config).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "fit_split": "train",
        "evaluation_split": "val",
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "num_history": args.num_history,
        "max_clip_id": args.max_clip_id,
        "lambda_coord": 0.0,
        "scene_disjoint": True,
        "selection": {
            "train_sample_identity_hash": train_manifest["sample_identity_hash"],
            "train_scene_hash": train_manifest["scene_hash"],
            "train_scenes": train_manifest["scenes"],
            "val_sample_identity_hash": val_manifest["sample_identity_hash"],
            "val_scene_hash": val_manifest["scene_hash"],
            "val_scenes": val_manifest["scenes"],
        },
        "prior": _prior_summary(prior),
        "evaluations": {"standard": metrics},
        "artifacts": {
            "prior": str(prior_path),
            "selection_manifest": str(selections_path),
            "compact_predictions": str(predictions_path),
        },
    }
    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())

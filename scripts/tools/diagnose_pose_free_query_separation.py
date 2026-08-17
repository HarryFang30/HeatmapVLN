#!/usr/bin/env python3
"""Read-only Task-3.6 query-separation diagnostic.

The tool loads an existing pose-free pilot checkpoint and runs only strict
validation samples.  A forward pre-hook observes the actual inputs to the
shared ``PoseFreeHistoryMatcher``.  Because the pilot expands K=4 histories
into four independent one-history Qwen chains, the tool captures four strict
B=1 matcher calls and regroups them by history.  No history slot/frame/pose is
supplied to the model.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools.train_pose_free_multihistory_pilot import (
    assert_blank_chain_input_identity,
    build_explicit_dataset,
    exact_sample,
    flatten_isolated_pair_chains,
    forward_loss,
    json_dump,
    load_manifest_contract,
    load_pilot_checkpoint_strict,
    load_pilot_config,
    make_criterion,
    materialize_model,
    pose_free_config_contract,
    set_seed,
    transform_sample,
)

LOGGER = logging.getLogger("pose_free_query_separation")
REPORT_SCHEMA = "task36d_pose_free_query_separation_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("head-only", "heatmap-lora"), required=True)
    parser.add_argument("--eval-lora", choices=("trained", "off"), default="trained")
    parser.add_argument(
        "--intervention",
        choices=("standard", "blank-images"),
        default="standard",
    )
    parser.add_argument("--pilot-checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Stage1-S2 checkpoint.")
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--source-inventory-sha256", default=None)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")


def assert_blank_pair_chains_identical(transformed: dict[str, Any]) -> dict[str, Any]:
    """Require all four model-side chains to be bitwise identical when blank."""
    chains = flatten_isolated_pair_chains(transformed)
    return assert_blank_chain_input_identity(chains)


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError(f"Expected a square pairwise matrix with K>=2, got {tuple(matrix.shape)}")
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def pairwise_metrics(vectors: torch.Tensor) -> dict[str, Any]:
    """Return complete KxK matrices and off-diagonal separation summaries."""
    values = vectors.detach().float().reshape(vectors.shape[0], -1)
    if values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError(f"Expected [K,D] with K>=2 and D>0, got {tuple(values.shape)}")
    normalized = F.normalize(values, dim=-1)
    cosine = normalized @ normalized.T
    euclidean = torch.cdist(values, values, p=2)
    mean_l1 = torch.cdist(values, values, p=1) / values.shape[1]
    cosine_off = _off_diagonal(cosine)
    euclidean_off = _off_diagonal(euclidean)
    l1_off = _off_diagonal(mean_l1)
    return {
        "vector_shape": list(values.shape),
        "pairwise_cosine": cosine.cpu().tolist(),
        "pairwise_euclidean": euclidean.cpu().tolist(),
        "pairwise_mean_l1": mean_l1.cpu().tolist(),
        "off_diagonal": {
            "cosine_mean": float(cosine_off.mean().item()),
            "cosine_min": float(cosine_off.min().item()),
            "cosine_max": float(cosine_off.max().item()),
            "euclidean_mean": float(euclidean_off.mean().item()),
            "euclidean_min": float(euclidean_off.min().item()),
            "euclidean_max": float(euclidean_off.max().item()),
            "mean_l1": float(l1_off.mean().item()),
        },
    }


def _peak_xy(heatmap: torch.Tensor) -> tuple[int, int]:
    index = int(heatmap.reshape(-1).argmax().item())
    width = int(heatmap.shape[-1])
    return index % width, index // width


def output_peak_diversity(
    visibility: torch.Tensor,
    heatmaps: torch.Tensor,
) -> dict[str, Any]:
    """Describe selected panoramic peaks for K independent history outputs."""
    if visibility.ndim != 2 or heatmaps.ndim != 4:
        raise ValueError(
            f"Expected visibility [K,4], heatmaps [K,4,H,W], got {tuple(visibility.shape)}, {tuple(heatmaps.shape)}"
        )
    if tuple(visibility.shape) != tuple(heatmaps.shape[:2]) or visibility.shape[1] != 4:
        raise ValueError("Visibility/heatmap history-view dimensions do not match")
    width = int(heatmaps.shape[-1])
    panorama_width = 4 * width
    peaks = []
    for history in range(int(visibility.shape[0])):
        view = int(visibility[history].argmax().item())
        x, y = _peak_xy(heatmaps[history, view])
        peaks.append({"view": view, "x": x, "y": y, "panorama_x": view * width + x})
    distances = []
    for left_index, left in enumerate(peaks):
        for right in peaks[left_index + 1 :]:
            dx = abs(int(left["panorama_x"]) - int(right["panorama_x"]))
            dx = min(dx, panorama_width - dx)
            distances.append(math.hypot(dx, int(left["y"]) - int(right["y"])))
    unique = {(peak["view"], peak["x"], peak["y"]) for peak in peaks}
    return {
        "peaks": peaks,
        "unique_peak_count": len(unique),
        "all_histories_have_distinct_peaks": len(unique) == len(peaks),
        "unique_selected_view_count": len({peak["view"] for peak in peaks}),
        "pairwise_peak_distances": distances,
        "minimum_pairwise_peak_distance": min(distances) if distances else None,
        "mean_pairwise_peak_distance": float(np.mean(distances)) if distances else None,
    }


@contextmanager
def capture_matcher_inputs(matcher: torch.nn.Module) -> Iterator[list[dict[str, Any]]]:
    """Capture the real matcher inputs from one sample's four B=1 forwards."""
    captures: list[dict[str, Any]] = []

    def pre_hook(
        module: torch.nn.Module,
        positional: tuple[Any, ...],
        keyword: dict[str, Any],
    ) -> None:
        current = keyword.get("current_patches", positional[0] if positional else None)
        queries = keyword.get("history_queries", positional[1] if len(positional) > 1 else None)
        if not torch.is_tensor(current) or not torch.is_tensor(queries):
            raise RuntimeError("Matcher hook did not receive tensor current_patches/history_queries")
        if current.ndim != 5 or queries.ndim != 3:
            raise RuntimeError(
                f"Matcher hook rank mismatch: current={tuple(current.shape)} queries={tuple(queries.shape)}"
            )
        if current.shape[0] != 1 or queries.shape[:2] != (1, 1):
            raise RuntimeError(
                "Query-separation requires a strict B=1 one-history matcher call; "
                f"current={tuple(current.shape)} queries={tuple(queries.shape)}"
            )
        raw = queries[:, 0].detach()
        with torch.no_grad():
            projected = module.query_projection(module.query_norm(raw))
            pooled_current = current.detach().float().mean(dim=(1, 2, 3))
        captures.append(
            {
                "current_patches_shape": list(current.shape),
                "history_queries_shape": list(queries.shape),
                "current_patches": current.detach().float().cpu(),
                "raw_queries": raw.float().cpu(),
                "projected_queries": projected.detach().float().cpu(),
                "pooled_current_patches": pooled_current.cpu(),
            }
        )

    handle = matcher.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield captures
    finally:
        handle.remove()


def regroup_matcher_captures(captures: list[dict[str, Any]]) -> dict[str, Any]:
    """Regroup four independently captured B=1 matcher calls by history."""
    if len(captures) != 4:
        raise RuntimeError(f"Expected four strict B=1 matcher captures, got {len(captures)}")
    for index, capture in enumerate(captures):
        if capture["current_patches_shape"][0] != 1 or capture["history_queries_shape"][:2] != [1, 1]:
            raise RuntimeError(f"Matcher capture {index} was not a B=1, one-history call")
    current = torch.cat([capture["current_patches"] for capture in captures], dim=0)
    raw = torch.cat([capture["raw_queries"] for capture in captures], dim=0)
    projected = torch.cat([capture["projected_queries"] for capture in captures], dim=0)
    pooled = torch.cat([capture["pooled_current_patches"] for capture in captures], dim=0)
    differences = (current - current[0:1]).abs().flatten(1)
    return {
        "current_patches_shape": list(current.shape),
        "history_queries_shape": [4, 1, int(raw.shape[-1])],
        "per_call_current_patches_shapes": [capture["current_patches_shape"] for capture in captures],
        "per_call_history_queries_shapes": [capture["history_queries_shape"] for capture in captures],
        "raw_queries": raw,
        "projected_queries": projected,
        "pooled_current_patches": pooled,
        "current_chain_max_abs_difference_from_chain0": differences.max(dim=1).values.tolist(),
        "current_chain_mean_abs_difference_from_chain0": differences.mean(dim=1).tolist(),
    }


def analyze_sample(
    capture: dict[str, Any],
    prediction: dict[str, torch.Tensor],
    *,
    sample_id: str,
) -> dict[str, Any]:
    visibility = prediction["visibility"].squeeze(0)
    heatmaps = prediction["heatmaps"].squeeze(0)
    if visibility.shape[:2] != (4, 4) or heatmaps.shape[:2] != (4, 4):
        raise RuntimeError(
            f"Regrouped output is not K=4: visibility={tuple(visibility.shape)} heatmaps={tuple(heatmaps.shape)}"
        )
    return {
        "sample_id": sample_id,
        "captured_input_contract": {
            "current_patches_shape": capture["current_patches_shape"],
            "history_queries_shape": capture["history_queries_shape"],
            "isolated_pair_chains": True,
            "histories_per_chain": 1,
            "qwen_forward_batch_size": 1,
            "qwen_forwards_per_sample": 4,
            "pose_slot_frame_model_inputs": False,
        },
        "current_patch_chain_consistency": {
            "max_abs_difference_from_chain0": capture["current_chain_max_abs_difference_from_chain0"],
            "mean_abs_difference_from_chain0": capture["current_chain_mean_abs_difference_from_chain0"],
            "pooled_feature_pairwise": pairwise_metrics(capture["pooled_current_patches"]),
        },
        "raw_history_query_separation": pairwise_metrics(capture["raw_queries"]),
        "projected_history_query_separation": pairwise_metrics(capture["projected_queries"]),
        "output_heatmap_similarity": pairwise_metrics(heatmaps),
        "output_visibility_similarity": pairwise_metrics(visibility),
        "output_peak_diversity": output_peak_diversity(visibility, heatmaps),
    }


def aggregate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot aggregate an empty query-separation report")

    def mean_path(*keys: str) -> float:
        values = []
        for sample in samples:
            current: Any = sample
            for key in keys:
                current = current[key]
            values.append(float(current))
        return float(np.mean(values))

    current_max = max(
        max(sample["current_patch_chain_consistency"]["max_abs_difference_from_chain0"])
        for sample in samples
    )
    unique_peaks = [sample["output_peak_diversity"]["unique_peak_count"] for sample in samples]
    return {
        "samples": len(samples),
        "raw_query_off_diagonal_cosine_mean": mean_path(
            "raw_history_query_separation", "off_diagonal", "cosine_mean"
        ),
        "raw_query_off_diagonal_euclidean_mean": mean_path(
            "raw_history_query_separation", "off_diagonal", "euclidean_mean"
        ),
        "projected_query_off_diagonal_cosine_mean": mean_path(
            "projected_history_query_separation", "off_diagonal", "cosine_mean"
        ),
        "projected_query_off_diagonal_euclidean_mean": mean_path(
            "projected_history_query_separation", "off_diagonal", "euclidean_mean"
        ),
        "output_heatmap_off_diagonal_cosine_mean": mean_path(
            "output_heatmap_similarity", "off_diagonal", "cosine_mean"
        ),
        "output_heatmap_off_diagonal_mean_l1": mean_path(
            "output_heatmap_similarity", "off_diagonal", "mean_l1"
        ),
        "mean_unique_peak_count": float(np.mean(unique_peaks)),
        "fraction_with_four_distinct_peaks": float(np.mean(np.asarray(unique_peaks) == 4)),
        "maximum_current_patch_cross_chain_abs_difference": float(current_max),
    }


def run(args: argparse.Namespace) -> int:
    started = time.time()
    cfg = load_pilot_config(args)
    config_contract = pose_free_config_contract(cfg)
    records, manifest_contract = load_manifest_contract(args)
    split_inventory = manifest_contract["split_source_inventories"]["val"]
    dataset = build_explicit_dataset(
        cfg,
        "val",
        records["val"],
        seed=args.seed + 3600,
        reshuffle_slots_each_epoch=False,
        max_clip_id=manifest_contract["max_clip_id"],
        expected_inventory_sha256=split_inventory["inventory_sha256"],
        expected_inventory_clips=split_inventory["clips"],
    )
    model, stage1_contract, runtime_contract = materialize_model(args, cfg)
    _payload, pilot_contract = load_pilot_checkpoint_strict(
        model,
        args.pilot_checkpoint,
        branch=args.branch,
        stage1_contract=stage1_contract,
        manifest_contract=manifest_contract,
        eval_lora=args.eval_lora,
        runtime_contract=runtime_contract,
        config_contract=config_contract,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    model.heatmap_vln.feat_extractor.detach_features = True
    matcher = model.heatmap_vln.pose_free_matcher
    if matcher is None:
        raise RuntimeError("PoseFreeHistoryMatcher was not materialized")
    device = torch.device(args.device)
    criterion = make_criterion(cfg, device)

    sample_reports = []
    sample_ids = []
    sample_count = min(args.max_samples, len(dataset))
    for index in range(sample_count):
        sample = exact_sample(dataset, index)
        transformed = transform_sample(sample, intervention=args.intervention)
        blank_input_contract = None
        if args.intervention == "blank-images":
            blank_input_contract = assert_blank_pair_chains_identical(transformed)
        with capture_matcher_inputs(matcher) as captures, torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            _loss, prediction = forward_loss(
                model,
                criterion,
                transformed,
                device,
                history_rel_poses=None,
            )
        capture = regroup_matcher_captures(captures)
        sample_id = transformed["sample_id"]
        sample_ids.append(sample_id)
        sample_report = analyze_sample(capture, prediction, sample_id=sample_id)
        if blank_input_contract is not None:
            sample_report["blank_transformed_chain_identity"] = blank_input_contract
            blank_output_gate = prediction.get("blank_output_identity_gate")
            if not isinstance(blank_output_gate, dict) or not blank_output_gate.get(
                "four_blank_chain_outputs_bitwise_identical"
            ):
                raise RuntimeError("Blank query diagnostic did not pass/record output identity gate")
            sample_report["blank_output_identity_gate"] = blank_output_gate
        sample_reports.append(sample_report)
        LOGGER.info("captured query separation sample %d/%d", index + 1, sample_count)

    aggregate = aggregate_samples(sample_reports)
    if args.intervention == "blank-images":
        aggregate["blank_output_identity_gate"] = {
            "passed": True,
            "bitwise_exact": True,
            "samples": sample_count,
        }
    report = {
        "schema": REPORT_SCHEMA,
        "read_only": True,
        "branch": args.branch,
        "eval_lora": args.eval_lora,
        "intervention": args.intervention,
        "duration_seconds": time.time() - started,
        "requested_max_samples": args.max_samples,
        "evaluated_samples": sample_count,
        "sample_identity_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
        "model_input_contract": {
            "isolated_pair_chains": True,
            "histories_per_qwen_chain": 1,
            "history_anchor_number_per_chain": 1,
            "qwen_forward_batch_size": 1,
            "qwen_forwards_per_sample": 4,
            "exact_relative_pose": None,
            "slot_id": None,
            "frame_index": None,
            "intervention": args.intervention,
            "blank_chain_tensor_identity_asserted": args.intervention == "blank-images",
        },
        "stage1_s2_contract": stage1_contract,
        "manifest_contract": manifest_contract,
        "pose_free_config_contract": config_contract,
        "runtime_contract": runtime_contract,
        "pilot_checkpoint": pilot_contract,
        "aggregate": aggregate,
        "samples": sample_reports,
    }
    intervention_label = args.intervention.replace("-", "_")
    output_path = (
        Path(args.output_dir)
        / f"query_separation_{args.branch}_{args.eval_lora}_{intervention_label}.json"
    )
    json_dump(output_path, report)
    LOGGER.info("Wrote %s", output_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

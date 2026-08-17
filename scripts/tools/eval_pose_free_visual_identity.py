#!/usr/bin/env python3
"""Strict evaluator for the v3 visual-history identity pilot.

The three evaluation cells share one bitwise-identical warmup matcher head:

* ``warmup-original`` keeps the freshly loaded Stage1-S2 LoRA;
* ``identity-trained`` loads only the LoRA from ``lora-identity``;
* ``heatmap-control-trained`` loads only the LoRA from
  ``lora-heatmap-control``.

Every cell is materialized from Stage1-S2 in a fresh process and evaluated as
four physically separate B=1 Qwen calls per K=4 sample.  Legacy text-anchor,
B=4-row, joint-head/LoRA, and legacy checkpoints are rejected before model state
is changed.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.tools.train_pose_free_multihistory_pilot as base_pilot
import scripts.tools.train_pose_free_visual_identity_pilot as visual_pilot

from src.models.heatmap import (
    extract_primary_panorama_targets,
    target_grounded_panorama_losses,
    target_grounded_score_matrix,
)

LOGGER = logging.getLogger("pose_free_visual_identity_eval")

REPORT_SCHEMA = "pose_free_visual_identity_eval_report_v3"
PROTOCOL = "strict_b1_visual_identity_eval_v3"
CELLS = (
    "warmup-original",
    "identity-trained",
    "heatmap-control-trained",
)
CELL_TRAIN_MODE = {
    "warmup-original": "head-warmup",
    "identity-trained": "lora-identity",
    "heatmap-control-trained": "lora-heatmap-control",
}
# Standard must run first because three causal gates are paired to it.
EVAL_INTERVENTIONS = (
    "standard",
    "history-shuffle",
    "current-shuffle",
    "single-anchor-swap",
    "blank-images",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=CELLS, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Pinned original Stage1-S2 checkpoint with all 224 LoRA tensors.",
    )
    parser.add_argument(
        "--warmup-checkpoint",
        required=True,
        help="Strict v3 head-warmup checkpoint shared by all three cells.",
    )
    parser.add_argument(
        "--trained-checkpoint",
        default=None,
        help="Required for a trained cell; forbidden for warmup-original.",
    )
    parser.add_argument(
        "--paired-checkpoint",
        default=None,
        help=(
            "Counterpart trained checkpoint used to prove the identity/control causal pair. "
            "Required for both trained cells; forbidden for warmup-original."
        ),
    )
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--source-inventory-sha256", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--standard-only",
        action="store_true",
        help="Run only the standard cell for train/validation generalization diagnosis.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.cell not in CELLS:
        raise ValueError(f"Unknown visual-identity evaluation cell: {args.cell}")
    if not args.warmup_checkpoint:
        raise ValueError("Every cell requires --warmup-checkpoint")
    if args.selection_split == "train" and not args.standard_only:
        raise ValueError("Train-split evaluation is diagnostic-only and requires --standard-only")
    if args.cell == "warmup-original":
        if args.trained_checkpoint is not None or args.paired_checkpoint is not None:
            raise ValueError("warmup-original forbids trained/paired checkpoints")
    elif not args.trained_checkpoint or not args.paired_checkpoint:
        raise ValueError(f"{args.cell} requires --trained-checkpoint and --paired-checkpoint")
    if args.trained_checkpoint is not None:
        if Path(args.trained_checkpoint).resolve() == Path(args.warmup_checkpoint).resolve():
            raise ValueError("The trained and warmup checkpoint paths must differ")
    if args.paired_checkpoint is not None:
        resolved = Path(args.paired_checkpoint).resolve()
        if resolved == Path(args.warmup_checkpoint).resolve():
            raise ValueError("The paired and warmup checkpoint paths must differ")
        if args.trained_checkpoint is not None and resolved == Path(args.trained_checkpoint).resolve():
            raise ValueError("The selected and paired checkpoint paths must differ")


def _evaluation_model_args(args: argparse.Namespace) -> argparse.Namespace:
    """Adapt evaluation to the audited v3 construction helpers.

    ``head-warmup`` is deliberate here: it disables gradient checkpointing and
    makes the legacy adapter choose head-only feature capture.  Neither field
    changes the v3 config/runtime checkpoint contract, and evaluation freezes
    every parameter after loading the selected cell.
    """

    adapted = copy.copy(args)
    adapted.train_mode = "head-warmup"
    return adapted


def load_eval_config(args: argparse.Namespace) -> dict[str, Any]:
    return visual_pilot.load_visual_identity_config(_evaluation_model_args(args))


def _require_equal_fields(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    fields: tuple[str, ...],
    *,
    label: str,
) -> None:
    for field in fields:
        if actual.get(field) != expected.get(field):
            raise RuntimeError(f"{label} mismatch: {field}")


def _strict_head_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    matcher = model.heatmap_vln.pose_free_matcher
    if matcher is None:
        raise RuntimeError("Visual-identity evaluation requires PoseFreeHistoryMatcher")
    tensors: dict[str, torch.Tensor] = dict(matcher.named_parameters())
    tensors.update(dict(matcher.named_buffers()))
    return tensors


def _validate_cell_loss_contract(payload: Mapping[str, Any], cell: str) -> dict[str, Any]:
    loss = payload.get("loss_contract")
    if not isinstance(loss, dict):
        raise RuntimeError("Selected v3 checkpoint has no loss contract")
    expected = visual_pilot.expected_loss_contract(CELL_TRAIN_MODE[cell])
    if loss != expected:
        raise RuntimeError(f"Selected checkpoint loss contract does not match cell {cell}")
    return expected


def load_eval_cell_strict(
    model: torch.nn.Module,
    *,
    cell: str,
    warmup_checkpoint: str | Path,
    trained_checkpoint: str | Path | None,
    paired_checkpoint: str | Path | None = None,
    stage1_contract: dict[str, Any],
    manifest_contract: dict[str, Any],
    config_contract: dict[str, Any],
    runtime_contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one cell after proving fresh Stage1 and exact warmup pairing."""

    if cell not in CELLS:
        raise ValueError(f"Unknown visual-identity evaluation cell: {cell}")
    if cell == "warmup-original":
        if trained_checkpoint is not None or paired_checkpoint is not None:
            raise ValueError("warmup-original accepts no trained checkpoint pair")
    elif trained_checkpoint is None or paired_checkpoint is None:
        raise ValueError("A trained cell requires both selected and counterpart checkpoints")

    current_lora = base_pilot.lora_state_dict(model)
    fresh_lora_hash = base_pilot.tensor_state_sha256(current_lora)
    if fresh_lora_hash != stage1_contract.get("loaded_lora_sha256"):
        raise RuntimeError("Evaluation model was not freshly initialized from pinned Stage1-S2 LoRA")

    warmup_payload, warmup_file_hash = visual_pilot.validate_visual_identity_checkpoint_payload_strict(
        warmup_checkpoint,
        expected_train_mode="head-warmup",
        stage1_contract=stage1_contract,
        manifest_contract=manifest_contract,
        config_contract=config_contract,
        runtime_contract=runtime_contract,
    )
    actual_warmup_contract = {
        "schema": visual_pilot.CHECKPOINT_SCHEMA,
        "protocol": visual_pilot.PROTOCOL,
        "file_sha256": warmup_file_hash,
        "head_state_sha256": warmup_payload["head_state_sha256"],
        "lora_state_sha256": warmup_payload["lora_state_sha256"],
        "step": int(warmup_payload["step"]),
        "training_sample_schedule_sha256": warmup_payload["training_sample_schedule_sha256"],
        "optimization_contract": warmup_payload["optimization_contract"],
    }

    selected_payload = warmup_payload
    selected_file_hash = warmup_file_hash
    expected_train_mode = CELL_TRAIN_MODE[cell]
    selected_path = Path(warmup_checkpoint)
    loss_contract = _validate_cell_loss_contract(warmup_payload, cell) if cell == "warmup-original" else None
    pair_gate = None
    if trained_checkpoint is not None:
        selected_payload, selected_file_hash = visual_pilot.validate_visual_identity_checkpoint_payload_strict(
            trained_checkpoint,
            expected_train_mode=expected_train_mode,
            stage1_contract=stage1_contract,
            manifest_contract=manifest_contract,
            config_contract=config_contract,
            runtime_contract=runtime_contract,
        )
        selected_path = Path(trained_checkpoint)
        paired_warmup = selected_payload.get("warmup_checkpoint_contract")
        if not isinstance(paired_warmup, dict):
            raise RuntimeError("Trained cell has no warmup pairing contract")
        _require_equal_fields(
            paired_warmup,
            actual_warmup_contract,
            (
                "schema",
                "protocol",
                "file_sha256",
                "head_state_sha256",
                "lora_state_sha256",
                "step",
                "training_sample_schedule_sha256",
                "optimization_contract",
            ),
            label="Selected LoRA checkpoint versus supplied warmup checkpoint",
        )
        if selected_payload["head_state_sha256"] != warmup_payload["head_state_sha256"]:
            raise RuntimeError("Selected LoRA checkpoint does not preserve the supplied warmup head")
        if selected_payload["lora_state_sha256"] == stage1_contract["loaded_lora_sha256"]:
            raise RuntimeError("Selected trained cell did not change the original Stage1-S2 LoRA")
        loss_contract = _validate_cell_loss_contract(selected_payload, cell)
        if cell == "identity-trained":
            pair_gate = visual_pilot.validate_identity_control_checkpoint_pair(
                trained_checkpoint,
                paired_checkpoint,
            )
        else:
            pair_gate = visual_pilot.validate_identity_control_checkpoint_pair(
                paired_checkpoint,
                trained_checkpoint,
            )
        selected_pair_key = "identity_checkpoint" if cell == "identity-trained" else "control_checkpoint"
        if pair_gate[selected_pair_key]["file_sha256"] != selected_file_hash:
            raise RuntimeError("Identity/control pair validator did not bind the selected checkpoint file")

    # The head always comes from the independently validated warmup file, never
    # from a trained-LoRA checkpoint merely claiming to contain that head.
    base_pilot.strict_load_named_state(
        _strict_head_tensors(model),
        warmup_payload["head_state_dict"],
        label="shared v2 warmup head",
    )
    if trained_checkpoint is not None:
        base_pilot.strict_load_named_state(
            base_pilot.normalized_lora_parameters(model),
            selected_payload["lora_state_dict"],
            label=f"{cell} v2 LoRA",
        )

    active_head_hash = base_pilot.tensor_state_sha256(base_pilot.pose_free_head_state_dict(model))
    active_lora_hash = base_pilot.tensor_state_sha256(base_pilot.lora_state_dict(model))
    expected_lora_hash = (
        stage1_contract["loaded_lora_sha256"] if cell == "warmup-original" else selected_payload["lora_state_sha256"]
    )
    if active_head_hash != warmup_payload["head_state_sha256"]:
        raise RuntimeError("Shared warmup head did not load bitwise exactly")
    if active_lora_hash != expected_lora_hash:
        raise RuntimeError(f"The {cell} LoRA source did not load bitwise exactly")

    checkpoint_sources = {
        "head": {
            "source": "shared-head-warmup",
            "path": str(Path(warmup_checkpoint).resolve()),
            "file_sha256": warmup_file_hash,
            "head_state_sha256": active_head_hash,
        },
        "lora": {
            "source": "stage1-s2" if cell == "warmup-original" else expected_train_mode,
            "path": (stage1_contract["path"] if cell == "warmup-original" else str(selected_path.resolve())),
            "file_sha256": (stage1_contract["file_sha256"] if cell == "warmup-original" else selected_file_hash),
            "lora_state_sha256": active_lora_hash,
        },
    }
    selected_contract = {
        "cell": cell,
        "expected_train_mode": expected_train_mode,
        "selected_checkpoint_path": str(selected_path.resolve()),
        "selected_checkpoint_file_sha256": selected_file_hash,
        "training_pid": selected_payload.get("training_pid"),
        "step": int(selected_payload["step"]),
        "fresh_stage1_lora_loaded_before_cell_state": True,
        "fresh_stage1_lora_sha256": fresh_lora_hash,
        "shared_warmup_contract": actual_warmup_contract,
        "active_head_sha256": active_head_hash,
        "active_lora_sha256": active_lora_hash,
        "loss_contract": loss_contract,
        "identity_control_pair_gate": pair_gate,
    }
    return selected_contract, checkpoint_sources


def assert_single_swap_untargeted_invariance(
    standard_records: list[dict[str, Any]],
    swap_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless swapping history i leaves every output j!=i exact."""

    standard_by_id = {record["sample_id"]: record for record in standard_records}
    if len(standard_by_id) != len(standard_records):
        raise RuntimeError("Single-swap gate requires unique standard sample IDs")
    tensor_comparisons = 0
    output_slots = 0
    seen_pairs: set[tuple[str, int]] = set()
    for swapped in swap_records:
        sample_id = swapped.get("sample_id")
        target_slot = swapped.get("target_slot")
        standard = standard_by_id.get(sample_id)
        if standard is None or not isinstance(target_slot, int) or target_slot not in range(4):
            raise RuntimeError("Single-swap gate received an invalid paired record")
        pair = (sample_id, target_slot)
        if pair in seen_pairs:
            raise RuntimeError(f"Single-swap gate received duplicate pair: {pair}")
        seen_pairs.add(pair)
        for output_slot in range(4):
            if output_slot == target_slot:
                continue
            output_slots += 1
            for key in ("visibility", "heatmaps", "heatmap_logits"):
                if key not in standard or key not in swapped:
                    raise RuntimeError(f"Single-swap gate requires paired raw-logit records; missing {key}")
                expected = standard[key][:, output_slot]
                actual = swapped[key][:, output_slot]
                if not torch.equal(actual, expected):
                    maximum = float((actual - expected).abs().max().item())
                    raise RuntimeError(
                        "Single-anchor swap changed an untargeted output: "
                        f"sample={sample_id} target={target_slot} output={output_slot} "
                        f"tensor={key} max_abs_difference={maximum:.6e}"
                    )
                tensor_comparisons += 1
    expected_pairs = {(sample_id, slot) for sample_id in standard_by_id for slot in range(4)}
    if seen_pairs != expected_pairs:
        raise RuntimeError("Single-swap gate does not contain exactly four target slots per sample")
    return {
        "passed": True,
        "bitwise_exact": True,
        "source_samples": len(standard_records),
        "swap_pairs": len(seen_pairs),
        "untargeted_output_slots": output_slots,
        "tensor_comparisons": tensor_comparisons,
        "maximum_abs_difference": 0.0,
        "contract": "replace history i; every output j!=i remains bitwise identical",
    }


def assert_paired_sample_schedule(
    standard_records: list[dict[str, Any]],
    intervention_records: list[dict[str, Any]],
    *,
    intervention: str,
) -> dict[str, Any]:
    standard_ids = [record["sample_id"] for record in standard_records]
    intervention_ids = [record["sample_id"] for record in intervention_records]
    if len(set(standard_ids)) != len(standard_ids) or intervention_ids != standard_ids:
        raise RuntimeError(f"{intervention} did not preserve the paired validation schedule")
    return {
        "passed": True,
        "paired_source_samples": len(standard_ids),
        "sample_order_exact": True,
    }


def compact_visual_identity_record(record: dict[str, Any]) -> dict[str, Any]:
    """Add the training objective's target-grounded 4x4 score matrix.

    The score is computed directly from the explicitly requested raw logits.
    Inverting BF16 sigmoid probabilities is invalid because moderately large
    logits already round to probability 1.0 and destroy score ordering.
    """

    compact = base_pilot.compact_record(record)
    probabilities = record.get("heatmaps")
    raw_logits = record.get("heatmap_logits")
    gt_visibility = record.get("gt_visibility")
    gt_heatmaps = record.get("gt_heatmaps")
    if not torch.is_tensor(probabilities) or probabilities.ndim != 5:
        raise RuntimeError("Visual-identity compact record requires heatmaps [1,K,4,H,W]")
    if tuple(probabilities.shape[:3]) != (1, 4, 4):
        raise RuntimeError(
            f"Visual-identity compact record requires strict [1,4,4,H,W] heatmaps, got {tuple(probabilities.shape)}"
        )
    if not torch.is_tensor(raw_logits) or tuple(raw_logits.shape) != tuple(probabilities.shape):
        raise RuntimeError("Visual-identity compact record requires raw heatmap_logits matching heatmaps")
    if not torch.is_tensor(gt_visibility) or tuple(gt_visibility.shape) != (4, 4):
        raise RuntimeError("Visual-identity compact record requires gt_visibility [4,4]")
    if not torch.is_tensor(gt_heatmaps) or tuple(gt_heatmaps.shape) != tuple(probabilities.shape[1:]):
        raise RuntimeError("Visual-identity compact record prediction/GT heatmap shapes differ")
    probabilities = probabilities.detach().float()
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("Visual-identity compact record heatmap probabilities must be finite")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise RuntimeError("Visual-identity compact record heatmaps are not sigmoid probabilities")
    raw_logits = raw_logits.detach().float()
    if not torch.isfinite(raw_logits).all():
        raise RuntimeError("Visual-identity compact record raw heatmap logits must be finite")
    targets = extract_primary_panorama_targets(
        gt_visibility.detach().float().unsqueeze(0),
        gt_heatmaps.detach().float().unsqueeze(0),
        expected_num_targets=4,
    )
    score_matrix = target_grounded_score_matrix(raw_logits, targets)
    if tuple(score_matrix.shape) != (1, 4, 4) or not torch.isfinite(score_matrix).all():
        raise RuntimeError("Target-grounded raw-logit scoring did not produce finite [1,4,4]")
    compact["target_score_matrix"] = score_matrix[0].detach().float().cpu().tolist()
    panorama = target_grounded_panorama_losses(
        raw_logits,
        gt_visibility.detach().float().unsqueeze(0),
        gt_heatmaps.detach().float().unsqueeze(0),
    )
    view_logits = panorama["view_logits"]
    if tuple(view_logits.shape) != (1, 4, 4) or not torch.isfinite(view_logits).all():
        raise RuntimeError("Raw-logit panorama marginal did not produce finite [1,4,4] view logits")
    compact["matcher_visibility_logits"] = compact["visibility_logits"]
    compact["visibility_logits"] = view_logits[0].detach().float().cpu().tolist()
    compact["probability_reconstructed_pred_xy"] = compact["pred_xy"]
    raw_pred_xy: list[list[list[int]]] = []
    global_pred: list[list[int]] = []
    height, width = (int(value) for value in raw_logits.shape[-2:])
    for history_slot in range(4):
        per_view: list[list[int]] = []
        for view_index in range(4):
            flat_index = int(raw_logits[0, history_slot, view_index].reshape(-1).argmax().item())
            per_view.append([flat_index % width, flat_index // width])
        raw_pred_xy.append(per_view)
        global_index = int(raw_logits[0, history_slot].reshape(-1).argmax().item())
        global_view = global_index // (height * width)
        within_view = global_index % (height * width)
        global_pred.append([global_view, within_view % width, within_view // width])
    compact["pred_xy"] = raw_pred_xy
    compact["global_pred_view_xy"] = global_pred
    compact["visibility_reconstruction"] = {
        "source": "explicit_raw_heatmap_logits",
        "operation": "per_view_spatial_logsumexp",
        "semantics": "categorical_panorama_view_marginal",
        "learned_readout_used": False,
    }
    compact["peak_reconstruction"] = {
        "source": "explicit_raw_heatmap_logits",
        "per_view_operation": "argmax_xy",
        "global_operation": "argmax_over_4hw",
        "bf16_sigmoid_probability_used": False,
    }
    compact["score_reconstruction"] = {
        "source": "explicit_raw_heatmap_logits",
        "inverse": None,
        "raw_logits_opt_in": "return_heatmap_logits=True",
        "normalization": "per_view_spatial_log_softmax",
        "target_extraction": "primary_visible_gt_heatmap_peak",
        "target_sampling": "circular_panorama_bilinear_grid_sample_align_corners_false",
        "matrix_axes": ["history_query", "ground_truth_target"],
        "matrix_shape": [4, 4],
    }
    return compact


def _raw_logit_metric_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace legacy readouts with raw-logit marginals for metric computation."""

    converted: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        raw_logits = record.get("heatmap_logits")
        heatmaps = record.get("heatmaps")
        if not torch.is_tensor(raw_logits) or not torch.is_tensor(heatmaps):
            raise RuntimeError(f"Raw-logit metric record {index} lacks heatmap tensors")
        if tuple(raw_logits.shape) != tuple(heatmaps.shape) or raw_logits.ndim != 5:
            raise RuntimeError(f"Raw-logit metric record {index} has invalid shapes")
        view_logits = torch.logsumexp(raw_logits.detach().float().flatten(-2), dim=-1)
        item = dict(record)
        item["legacy_visibility"] = record["visibility"]
        item["visibility"] = view_logits
        # compute_metrics only consumes heatmap argmax.  Raw logits avoid BF16
        # sigmoid saturation while preserving the model's exact spatial order.
        item["heatmaps"] = raw_logits.detach().float()
        converted.append(item)
    return converted


def _global_map_metrics(
    records: list[dict[str, Any]],
    *,
    dynamic_slots: str | None = None,
) -> dict[str, Any]:
    if dynamic_slots not in (None, "targeted", "untargeted"):
        raise ValueError(f"Unknown global-map dynamic slot mode: {dynamic_slots}")
    errors: list[float] = []
    for record in records:
        logits = record["heatmap_logits"].detach().float().squeeze(0)
        gt_visibility = record["gt_visibility"]
        gt_heatmaps = record["gt_heatmaps"]
        if dynamic_slots is None:
            slots = range(int(logits.shape[0]))
        else:
            target_slot = record.get("target_slot")
            if target_slot is None:
                raise RuntimeError("Global-map targeted metrics require target_slot")
            slots = (
                [int(target_slot)]
                if dynamic_slots == "targeted"
                else [slot for slot in range(int(logits.shape[0])) if slot != int(target_slot)]
            )
        height, width = (int(value) for value in logits.shape[-2:])
        for history_slot in slots:
            global_index = int(logits[history_slot].reshape(-1).argmax().item())
            pred_view = global_index // (height * width)
            within_view = global_index % (height * width)
            pred_x, pred_y = within_view % width, within_view // width
            visible_mask = gt_visibility[history_slot] > 0.5
            if int(visible_mask.sum().item()) != 1:
                raise RuntimeError("Global-map metric requires exactly one visible GT view per query")
            visible_gt = (
                gt_heatmaps[history_slot]
                .detach()
                .float()
                .masked_fill(
                    ~visible_mask[..., None, None],
                    -torch.inf,
                )
            )
            if not torch.isfinite(visible_gt[visible_mask]).any() or float(visible_gt[visible_mask].max()) <= 0:
                raise RuntimeError("Global-map metric requires positive visible GT heatmap mass")
            gt_index = int(visible_gt.reshape(-1).argmax().item())
            gt_view = gt_index // (height * width)
            gt_within_view = gt_index % (height * width)
            gt_x, gt_y = gt_within_view % width, gt_within_view // width
            pred_panorama_x = pred_view * width + pred_x
            gt_panorama_x = gt_view * width + gt_x
            dx = abs(pred_panorama_x - gt_panorama_x)
            dx = min(dx, 4 * width - dx)
            errors.append(math.hypot(dx, pred_y - gt_y))
    return {
        "global_map_joint_pck4": sum(error <= 4.0 for error in errors) / max(len(errors), 1),
        "global_map_joint_pck8": sum(error <= 8.0 for error in errors) / max(len(errors), 1),
        "global_map_count": len(errors),
    }


def _replace_legacy_metrics_with_raw_logits(
    legacy_metrics: dict[str, Any],
    raw_records: list[dict[str, Any]],
    *,
    intervention: str,
) -> dict[str, Any]:
    converted = _raw_logit_metric_records(raw_records)
    metrics = base_pilot.compute_metrics(converted)
    metrics["per_slot"] = {str(slot): base_pilot.compute_metrics(converted, slot=slot) for slot in range(4)}
    metrics["loss"] = legacy_metrics["loss"]
    metrics["samples"] = legacy_metrics["samples"]
    metrics["metric_source_contract"] = {
        "view": "raw_heatmap_spatial_logsumexp_marginal",
        "per_view_peak": "raw_heatmap_argmax_xy",
        "legacy_visibility_readout_used": False,
    }
    metrics.update(_global_map_metrics(raw_records))
    metrics["legacy_visibility_readout_metrics"] = {
        key: legacy_metrics.get(key)
        for key in (
            "visibility_auroc",
            "visibility_auprc",
            "visibility_f1",
            "visible_view_accuracy",
            "joint_pck4",
            "joint_pck8",
            "anchor_identity",
        )
    }
    metrics["legacy_evaluation"] = legacy_metrics
    if intervention == "blank-images":
        metrics["blank_input_identity_gate"] = legacy_metrics["blank_input_identity_gate"]
        metrics["blank_output_identity_gate"] = legacy_metrics["blank_output_identity_gate"]
    if intervention == "single-anchor-swap":
        metrics["source_samples"] = legacy_metrics["source_samples"]
        metrics["swap_evaluations_per_sample"] = legacy_metrics["swap_evaluations_per_sample"]
        metrics["targeted_slot_metrics"] = base_pilot.compute_metrics(converted, dynamic_slots="targeted")
        metrics["targeted_slot_metrics"].update(_global_map_metrics(raw_records, dynamic_slots="targeted"))
        metrics["untargeted_slot_metrics"] = base_pilot.compute_metrics(converted, dynamic_slots="untargeted")
        metrics["untargeted_slot_metrics"].update(_global_map_metrics(raw_records, dynamic_slots="untargeted"))
    return metrics


def evaluate_all_interventions(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataset: Any,
    device: torch.device,
    *,
    interventions: tuple[str, ...] = EVAL_INTERVENTIONS,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if interventions not in (EVAL_INTERVENTIONS, ("standard",)):
        raise ValueError("Visual-identity evaluation permits only the full protocol or standard-only diagnosis")
    if len(dataset) <= 1:
        raise RuntimeError("Causal current/single-history swaps require at least two validation samples")
    evaluations: dict[str, Any] = {}
    prediction_records: dict[str, list[dict[str, Any]]] = {}
    intervention_gates: dict[str, Any] = {}
    standard_records: list[dict[str, Any]] | None = None

    for intervention in interventions:
        LOGGER.info("Evaluating strict visual-identity intervention: %s", intervention)
        legacy_metrics, raw_records = base_pilot.evaluate_intervention(
            model,
            criterion,
            dataset,
            intervention,
            device,
            return_heatmap_logits=True,
        )
        metrics = _replace_legacy_metrics_with_raw_logits(
            legacy_metrics,
            raw_records,
            intervention=intervention,
        )
        if intervention == "standard":
            if len(raw_records) != len(dataset):
                raise RuntimeError("Standard evaluation did not produce one record per source sample")
            standard_records = raw_records
            if len({record["sample_id"] for record in raw_records}) != len(raw_records):
                raise RuntimeError("Standard evaluation produced duplicate sample IDs")
            intervention_gates[intervention] = {
                "passed": True,
                "source_samples": len(raw_records),
                "unique_sample_ids": True,
            }
        elif standard_records is None:
            raise RuntimeError(f"{intervention} requires standard predictions first")
        elif intervention == "history-shuffle":
            gate = base_pilot.assert_history_permutation_equivariance(
                standard_records,
                raw_records,
            )
            metrics["permutation_equivariance_gate"] = gate
            intervention_gates[intervention] = gate
        elif intervention == "current-shuffle":
            gate = assert_paired_sample_schedule(
                standard_records,
                raw_records,
                intervention=intervention,
            )
            metrics["paired_schedule_gate"] = gate
            intervention_gates[intervention] = gate
        elif intervention == "single-anchor-swap":
            locality_gate = assert_single_swap_untargeted_invariance(
                standard_records,
                raw_records,
            )
            change = base_pilot.paired_single_swap_output_change(
                standard_records,
                raw_records,
            )
            metrics["untargeted_invariance_gate"] = locality_gate
            metrics["paired_output_change_vs_standard"] = change
            intervention_gates[intervention] = locality_gate
        elif intervention == "blank-images":
            input_gate = metrics.get("blank_input_identity_gate")
            output_gate = metrics.get("blank_output_identity_gate")
            if not isinstance(input_gate, dict) or input_gate.get("passed") is not True:
                raise RuntimeError("Blank input bitwise identity gate is absent or failed")
            if not isinstance(output_gate, dict) or output_gate.get("passed") is not True:
                raise RuntimeError("Blank output bitwise identity gate is absent or failed")
            intervention_gates[intervention] = {
                "passed": True,
                "input": input_gate,
                "output": output_gate,
            }
        evaluations[intervention] = metrics
        prediction_records[intervention] = [compact_visual_identity_record(record) for record in raw_records]
    return evaluations, prediction_records, intervention_gates


def run_eval(args: argparse.Namespace) -> int:
    started = time.time()
    adapted = _evaluation_model_args(args)
    cfg = load_eval_config(args)
    config_contract = visual_pilot.visual_identity_config_contract(cfg)
    records, manifest_contract = visual_pilot.load_visual_identity_manifest_contract(adapted)
    selection_split = args.selection_split
    split_inventory = manifest_contract["split_source_inventories"][selection_split]
    dataset = base_pilot.build_explicit_dataset(
        cfg,
        selection_split,
        records[selection_split],
        seed=args.seed + (3700 if selection_split == "train" else 3800),
        reshuffle_slots_each_epoch=False,
        max_clip_id=manifest_contract["max_clip_id"],
        expected_inventory_sha256=split_inventory["inventory_sha256"],
        expected_inventory_clips=split_inventory["clips"],
    )
    model, stage1_contract, runtime_contract = visual_pilot.materialize_visual_identity_model(
        adapted,
        cfg,
    )
    if runtime_contract.get("history_query_source") != visual_pilot.HISTORY_QUERY_SOURCE:
        raise RuntimeError("Evaluator materialized the legacy text-anchor query source")
    if runtime_contract.get("qwen_forward_batch_size") != 1 or runtime_contract.get("qwen_forwards_per_sample") != 4:
        raise RuntimeError("Evaluator did not materialize the strict 4xB=1 runtime")

    selected_contract, checkpoint_sources = load_eval_cell_strict(
        model,
        cell=args.cell,
        warmup_checkpoint=args.warmup_checkpoint,
        trained_checkpoint=args.trained_checkpoint,
        paired_checkpoint=args.paired_checkpoint,
        stage1_contract=stage1_contract,
        manifest_contract=manifest_contract,
        config_contract=config_contract,
        runtime_contract=runtime_contract,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    model.heatmap_vln.feat_extractor.detach_features = True
    device = torch.device(args.device)
    criterion = base_pilot.make_criterion(cfg, device)
    interventions = ("standard",) if args.standard_only else EVAL_INTERVENTIONS
    evaluations, prediction_records, intervention_gates = evaluate_all_interventions(
        model,
        criterion,
        dataset,
        device,
        interventions=interventions,
    )

    report = {
        "schema": REPORT_SCHEMA,
        "protocol": PROTOCOL,
        "phase": "eval",
        "cell": args.cell,
        "evaluation_scope": {
            "selection_split": selection_split,
            "standard_only": args.standard_only,
            "source_samples": len(dataset),
        },
        "duration_seconds": time.time() - started,
        "evaluation_pid": os.getpid(),
        "fresh_process_contract": {
            "training_pid": selected_contract["training_pid"],
            "evaluation_pid": os.getpid(),
            "fresh_stage1_loaded_before_cell_state": selected_contract["fresh_stage1_lora_loaded_before_cell_state"],
        },
        "stage1_s2_contract": stage1_contract,
        "manifest_contract": manifest_contract,
        "pose_free_config_contract": config_contract,
        "runtime_contract": runtime_contract,
        "selected_cell_contract": selected_contract,
        "checkpoint_sources": checkpoint_sources,
        "state_and_input_contract": {
            "explicit_pose_inputs_removed": True,
            "history_query_source": visual_pilot.HISTORY_QUERY_SOURCE,
            "history_query_layer": 20,
            "history_visual_views_per_query": 4,
            "shared_head_across_cells": True,
            "view_metric_source": "raw_heatmap_spatial_logsumexp_marginal",
            "learned_visibility_readout_used_for_view_metric": False,
            "qwen_forward_batch_size": 1,
            "qwen_forwards_per_sample": 4,
        },
        "interventions": list(interventions),
        "intervention_gates": intervention_gates,
        "evaluations": evaluations,
        "prediction_records": prediction_records,
    }
    base_pilot.json_dump(Path(args.output_dir) / args.cell / "report.json", report)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    base_pilot.set_seed(args.seed)
    return run_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())

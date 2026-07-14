#!/usr/bin/env python3
"""Fail-closed summary for the strict visual-history identity pilot.

The three cells share one frozen warmup head and differ only in the active
LoRA state: original Stage1-S2, target-grounded identity training, or the
heatmap-only causal control.  All estimates are reconstructed from compact
per-source prediction records.  Bootstrap replicates resample complete source
samples, keeping cells, interventions, four history outputs, and the four
single-anchor-swap replicas paired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

REPORT_SCHEMA = "pose_free_visual_identity_eval_report_v3"
REPORT_PROTOCOL = "strict_b1_visual_identity_eval_v3"
TRAIN_PROTOCOL = "strict_b1_visual_identity_two_stage_v3"
CHECKPOINT_SCHEMA = "pose_free_visual_identity_checkpoint_v3"
SUMMARY_SCHEMA = "pose_free_visual_identity_summary_v3"
HISTORY_QUERY_SOURCE = "history_visual_equal_view_mean_v1"

CELL_NAMES = (
    "warmup-original",
    "identity-trained",
    "heatmap-control-trained",
)
CELL_TRAIN_MODES = {
    "warmup-original": "head-warmup",
    "identity-trained": "lora-identity",
    "heatmap-control-trained": "lora-heatmap-control",
}
CELL_LORA_SOURCES = {
    "warmup-original": "stage1-s2",
    "identity-trained": "lora-identity",
    "heatmap-control-trained": "lora-heatmap-control",
}
INTERVENTIONS = (
    "standard",
    "history-shuffle",
    "current-shuffle",
    "single-anchor-swap",
    "blank-images",
)
CAUSAL_INTERVENTIONS = (
    "history-shuffle",
    "current-shuffle",
    "single-anchor-swap",
)
METRICS = (
    "score_nearest_identity_accuracy",
    "peak_nearest_identity_accuracy",
    "visible_view_accuracy",
    "conditional_pck8",
    "true_joint_pck8",
    "global_map_joint_pck8",
)
COMPARISONS = (
    ("identity-minus-control", "identity-trained", "heatmap-control-trained"),
    ("identity-minus-warmup", "identity-trained", "warmup-original"),
    ("control-minus-warmup", "heatmap-control-trained", "warmup-original"),
)

SCORE_IDENTITY_MINIMUM = 0.45
SCORE_IDENTITY_VS_CONTROL_MINIMUM = 0.10
PEAK_IDENTITY_MINIMUM = 0.35
PEAK_IDENTITY_VS_CONTROL_MINIMUM = 0.05
CAUSAL_SCORE_IDENTITY_DROP_MINIMUM = 0.05
ALL_SAME_PEAK_FRACTION_MAXIMUM = 0.40
TRUE_JOINT_PCK8_VS_CONTROL_MINIMUM = -0.03
GLOBAL_MAP_JOINT_PCK8_VS_CONTROL_MINIMUM = -0.03
PANORAMA_CONTROL_VIEW_ACCURACY_MINIMUM = 0.40
PANORAMA_CONTROL_VIEW_GAIN_MINIMUM = 0.10
PANORAMA_CONTROL_GLOBAL_MAP_PCK8_MINIMUM = 0.15
PANORAMA_CONTROL_GLOBAL_MAP_GAIN_MINIMUM = 0.05
PANORAMA_CONTROL_CONDITIONAL_PCK8_GAIN_MINIMUM = 0.03
REQUIRED_TARGETS = 4
MINIMUM_TARGET_SEPARATION = 12.0

PAIRED_SWAP_CONTRACT = "replace history i; compare output i against all output j!=i on the same current"
PAIRED_SWAP_METRICS = (
    "mean_heatmap_l1",
    "mean_visibility_l1",
    "mean_peak_displacement",
)
SCORE_RECONSTRUCTION_CONTRACT = {
    "source": "explicit_raw_heatmap_logits",
    "inverse": None,
    "raw_logits_opt_in": "return_heatmap_logits=True",
    "normalization": "per_view_spatial_log_softmax",
    "target_extraction": "primary_visible_gt_heatmap_peak",
    "target_sampling": "circular_panorama_bilinear_grid_sample_align_corners_false",
    "matrix_axes": ["history_query", "ground_truth_target"],
    "matrix_shape": [4, 4],
}
VISIBILITY_RECONSTRUCTION_CONTRACT = {
    "source": "explicit_raw_heatmap_logits",
    "operation": "per_view_spatial_logsumexp",
    "semantics": "categorical_panorama_view_marginal",
    "learned_readout_used": False,
}
PEAK_RECONSTRUCTION_CONTRACT = {
    "source": "explicit_raw_heatmap_logits",
    "per_view_operation": "argmax_xy",
    "global_operation": "argmax_over_4hw",
    "bf16_sigmoid_probability_used": False,
}
EXPECTED_PAIR_MATCHED_CONTRACTS = {
    "stage1_s2_contract",
    "stage1_actual_file_sha256",
    "manifest_contract",
    "pose_free_config_contract",
    "runtime_contract",
    "warmup_actual_file_sha256",
    "warmup_head_sha256",
    "frozen_active_head_sha256",
    "initial_head_sha256",
    "initial_lora_sha256",
    "training_sample_schedule_sha256",
    "optimization_contract",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-report", required=True)
    parser.add_argument("--identity-report", required=True)
    parser.add_argument("--control-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heatmap-width", type=int, default=64)
    args = parser.parse_args(argv)
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if args.heatmap_width <= 0:
        parser.error("--heatmap-width must be positive")
    return args


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _same(reports: Mapping[str, Mapping[str, Any]], key: str) -> bool:
    values = [reports[cell].get(key) for cell in CELL_NAMES]
    return all(value == values[0] for value in values[1:])


def _validate_hash_source(source: Any, *, cell: str, component: str) -> dict[str, Any]:
    _require(isinstance(source, dict), f"Cell {cell} has no {component} checkpoint source")
    _require(isinstance(source.get("path"), str) and bool(source["path"]), f"Cell {cell} {component} path missing")
    _require(_valid_sha256(source.get("file_sha256")), f"Cell {cell} {component} file hash invalid")
    state_key = "head_state_sha256" if component == "head" else "lora_state_sha256"
    _require(_valid_sha256(source.get(state_key)), f"Cell {cell} {component} state hash invalid")
    return source


def _validate_exact_gate(gate: Any, *, context: str, maximum_required: bool = True) -> dict[str, Any]:
    _require(isinstance(gate, dict), f"{context} gate missing")
    _require(gate.get("passed") is True, f"{context} gate did not pass")
    _require(gate.get("bitwise_exact") is True, f"{context} gate is not bitwise exact")
    if maximum_required:
        _require(gate.get("maximum_abs_difference") == 0.0, f"{context} gate maximum difference is not zero")
    return gate


def _validate_swap_routing(evaluation: Any, *, cell: str, source_samples: int) -> None:
    _require(isinstance(evaluation, dict), f"Cell {cell} single-anchor-swap evaluation missing")
    paired = evaluation.get("paired_output_change_vs_standard")
    _require(isinstance(paired, dict), f"Cell {cell} paired single-swap metrics missing")
    _require(paired.get("contract") == PAIRED_SWAP_CONTRACT, f"Cell {cell} paired swap contract mismatch")
    expected = {"targeted": source_samples * 4, "untargeted": source_samples * 12}
    for route, comparisons in expected.items():
        values = paired.get(route)
        _require(isinstance(values, dict), f"Cell {cell} paired swap {route} route missing")
        _require(values.get("comparisons") == comparisons, f"Cell {cell} paired swap {route} count mismatch")
        for metric in PAIRED_SWAP_METRICS:
            _require(_finite_number(values.get(metric)), f"Cell {cell} paired swap {route} {metric} is non-finite")
            _require(float(values[metric]) >= 0.0, f"Cell {cell} paired swap {route} {metric} is negative")
    for metric in PAIRED_SWAP_METRICS:
        _require(
            float(paired["untargeted"][metric]) == 0.0,
            f"Cell {cell} single-swap changed an untargeted {metric}",
        )
    _require(
        paired.get("targeted_to_untargeted_heatmap_l1_ratio") is None,
        f"Cell {cell} paired swap ratio must be null for exact-zero untargeted L1",
    )


def _record_key(record: Mapping[str, Any]) -> tuple[str, int | None]:
    return str(record.get("sample_id")), record.get("target_slot")


def _labels(record: Mapping[str, Any]) -> tuple[Any, Any]:
    return record.get("gt_visibility"), record.get("gt_xy")


def _validate_record(record: Any, *, width: int, context: str) -> None:
    _require(isinstance(record, dict), f"{context}: compact record is not an object")
    visibility = np.asarray(record.get("visibility_logits"), dtype=np.float64)
    gt_visibility = np.asarray(record.get("gt_visibility"), dtype=np.float64)
    pred_xy = np.asarray(record.get("pred_xy"), dtype=np.int64)
    global_pred = np.asarray(record.get("global_pred_view_xy"), dtype=np.int64)
    gt_xy = np.asarray(record.get("gt_xy"), dtype=np.int64)
    score = np.asarray(record.get("target_score_matrix"), dtype=np.float64)
    _require(visibility.shape == (4, 4), f"{context}: visibility_logits must be [4,4]")
    _require(gt_visibility.shape == (4, 4), f"{context}: gt_visibility must be [4,4]")
    _require(pred_xy.shape == (4, 4, 2), f"{context}: pred_xy must be [4,4,2]")
    _require(global_pred.shape == (4, 3), f"{context}: global_pred_view_xy must be [4,3]")
    _require(gt_xy.shape == (4, 4, 2), f"{context}: gt_xy must be [4,4,2]")
    _require(score.shape == (4, 4), f"{context}: target_score_matrix must be [4,4]")
    _require(np.isfinite(visibility).all(), f"{context}: visibility logits are non-finite")
    _require(np.isfinite(score).all(), f"{context}: target score matrix is non-finite")
    _require(
        record.get("score_reconstruction") == SCORE_RECONSTRUCTION_CONTRACT,
        f"{context}: score reconstruction contract mismatch",
    )
    _require(
        record.get("visibility_reconstruction") == VISIBILITY_RECONSTRUCTION_CONTRACT,
        f"{context}: visibility reconstruction contract mismatch",
    )
    _require(
        record.get("peak_reconstruction") == PEAK_RECONSTRUCTION_CONTRACT,
        f"{context}: peak reconstruction contract mismatch",
    )
    positive_views = (gt_visibility > 0.5).sum(axis=1)
    _require(
        bool(np.all(positive_views == 1)),
        f"{context}: every target must have exactly one visible GT view; got {positive_views.tolist()}",
    )
    for name, xy in (("pred_xy", pred_xy), ("gt_xy", gt_xy)):
        _require(bool(((xy[..., 0] >= 0) & (xy[..., 0] < width)).all()), f"{context}: {name} x outside width")
        _require(bool((xy[..., 1] >= 0).all()), f"{context}: {name} y is negative")
    _require(bool(((global_pred[:, 0] >= 0) & (global_pred[:, 0] < 4)).all()), f"{context}: global view invalid")
    _require(bool(((global_pred[:, 1] >= 0) & (global_pred[:, 1] < width)).all()), f"{context}: global x invalid")
    _require(bool((global_pred[:, 2] >= 0).all()), f"{context}: global y is negative")


def _validate_prediction_records(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    width: int,
    source_samples: int,
) -> tuple[list[str], str]:
    canonical_ids: list[str] | None = None
    canonical_labels: dict[str, tuple[Any, Any]] | None = None
    canonical_keys: dict[str, list[tuple[str, int | None]]] = {}
    for cell in CELL_NAMES:
        records_by_intervention = reports[cell].get("prediction_records")
        _require(isinstance(records_by_intervention, dict), f"Cell {cell} prediction records missing")
        _require(
            set(records_by_intervention) == set(INTERVENTIONS), f"Cell {cell} prediction intervention set mismatch"
        )
        standard = records_by_intervention["standard"]
        _require(isinstance(standard, list), f"Cell {cell} standard records are not a list")
        ids = [str(record.get("sample_id")) for record in standard]
        _require(
            len(ids) == source_samples,
            f"Cell {cell} must contain exactly {source_samples} source IDs",
        )
        _require(len(ids) == len(set(ids)), f"Cell {cell} standard source IDs are not unique")
        labels: dict[str, tuple[Any, Any]] = {}
        for position, record in enumerate(standard):
            _validate_record(record, width=width, context=f"cell={cell} standard[{position}]")
            _require(record.get("target_slot") is None, f"Cell {cell} standard target_slot must be null")
            labels[ids[position]] = _labels(record)
        if canonical_ids is None:
            canonical_ids = ids
            canonical_labels = labels
        else:
            _require(ids == canonical_ids, f"Cell {cell} source ID order differs across cells")
            _require(labels == canonical_labels, f"Cell {cell} GT labels differ across cells")

        for intervention in INTERVENTIONS:
            records = records_by_intervention[intervention]
            _require(isinstance(records, list), f"Cell {cell} {intervention} records are not a list")
            expected_count = source_samples * (4 if intervention == "single-anchor-swap" else 1)
            _require(len(records) == expected_count, f"Cell {cell} {intervention} record count mismatch")
            keys = [_record_key(record) for record in records]
            expected_keys = (
                [(sample_id, slot) for sample_id in ids for slot in range(4)]
                if intervention == "single-anchor-swap"
                else [(sample_id, None) for sample_id in ids]
            )
            _require(keys == expected_keys, f"Cell {cell} {intervention} source/slot order mismatch")
            for position, record in enumerate(records):
                _validate_record(record, width=width, context=f"cell={cell} {intervention}[{position}]")
                _require(
                    _labels(record) == labels[str(record["sample_id"])],
                    f"Cell {cell} {intervention} changes GT labels for {record['sample_id']}",
                )
            if intervention in canonical_keys:
                _require(keys == canonical_keys[intervention], f"Cell {cell} {intervention} keys differ across cells")
            else:
                canonical_keys[intervention] = keys
    assert canonical_ids is not None
    source_hash = hashlib.sha256("\n".join(canonical_ids).encode("utf-8")).hexdigest()
    return canonical_ids, source_hash


def _validate_report_gates(
    report: Mapping[str, Any],
    *,
    cell: str,
    source_samples: int,
) -> None:
    evaluations = report.get("evaluations")
    gates = report.get("intervention_gates")
    _require(isinstance(evaluations, dict), f"Cell {cell} evaluations missing")
    _require(isinstance(gates, dict), f"Cell {cell} intervention gates missing")
    _require(set(evaluations) == set(INTERVENTIONS), f"Cell {cell} evaluation intervention set mismatch")
    _require(set(gates) == set(INTERVENTIONS), f"Cell {cell} gate intervention set mismatch")
    for intervention in INTERVENTIONS:
        evaluation = evaluations[intervention]
        _require(isinstance(evaluation, dict), f"Cell {cell} {intervention} evaluation is invalid")
        expected = source_samples * (4 if intervention == "single-anchor-swap" else 1)
        _require(evaluation.get("samples") == expected, f"Cell {cell} {intervention} sample count mismatch")

    standard = gates["standard"]
    _require(
        isinstance(standard, dict)
        and standard.get("passed") is True
        and standard.get("source_samples") == source_samples
        and standard.get("unique_sample_ids") is True,
        f"Cell {cell} standard source-ID gate failed",
    )
    history = _validate_exact_gate(gates["history-shuffle"], context=f"Cell {cell} history permutation")
    _require(
        evaluations["history-shuffle"].get("permutation_equivariance_gate") == history,
        f"Cell {cell} duplicated history gate differs",
    )
    current = gates["current-shuffle"]
    _require(
        isinstance(current, dict)
        and current.get("passed") is True
        and current.get("paired_source_samples") == source_samples
        and current.get("sample_order_exact") is True,
        f"Cell {cell} current-shuffle pairing gate failed",
    )
    _require(
        evaluations["current-shuffle"].get("paired_schedule_gate") == current,
        f"Cell {cell} duplicated current-shuffle gate differs",
    )
    swap = _validate_exact_gate(gates["single-anchor-swap"], context=f"Cell {cell} single-swap locality")
    _require(
        swap.get("source_samples") == source_samples
        and swap.get("swap_pairs") == source_samples * 4
        and swap.get("untargeted_output_slots") == source_samples * 12,
        f"Cell {cell} single-swap locality counts mismatch",
    )
    _require(
        evaluations["single-anchor-swap"].get("untargeted_invariance_gate") == swap,
        f"Cell {cell} duplicated single-swap gate differs",
    )
    blank = gates["blank-images"]
    _require(isinstance(blank, dict) and blank.get("passed") is True, f"Cell {cell} blank gate failed")
    blank_input = _validate_exact_gate(blank.get("input"), context=f"Cell {cell} blank input", maximum_required=False)
    blank_output = _validate_exact_gate(blank.get("output"), context=f"Cell {cell} blank output")
    _require(
        evaluations["blank-images"].get("blank_input_identity_gate") == blank_input
        and evaluations["blank-images"].get("blank_output_identity_gate") == blank_output,
        f"Cell {cell} duplicated blank gates differ",
    )
    _validate_swap_routing(
        evaluations["single-anchor-swap"],
        cell=cell,
        source_samples=source_samples,
    )


def _validate_selected_cell_contracts(reports: Mapping[str, Mapping[str, Any]]) -> None:
    stage1 = reports["warmup-original"]["stage1_s2_contract"]
    _require(_valid_sha256(stage1.get("file_sha256")), "Stage1-S2 file hash invalid")
    _require(_valid_sha256(stage1.get("loaded_lora_sha256")), "Stage1-S2 LoRA hash invalid")
    heads: dict[str, dict[str, Any]] = {}
    loras: dict[str, dict[str, Any]] = {}
    selected: dict[str, dict[str, Any]] = {}
    for cell in CELL_NAMES:
        sources = reports[cell].get("checkpoint_sources")
        _require(isinstance(sources, dict), f"Cell {cell} checkpoint sources missing")
        heads[cell] = _validate_hash_source(sources.get("head"), cell=cell, component="head")
        loras[cell] = _validate_hash_source(sources.get("lora"), cell=cell, component="lora")
        _require(heads[cell].get("source") == "shared-head-warmup", f"Cell {cell} does not use shared warmup head")
        _require(loras[cell].get("source") == CELL_LORA_SOURCES[cell], f"Cell {cell} LoRA source mismatch")
        contract = reports[cell].get("selected_cell_contract")
        _require(isinstance(contract, dict), f"Cell {cell} selected-cell contract missing")
        selected[cell] = contract
        _require(contract.get("cell") == cell, f"Cell {cell} selected-cell label mismatch")
        _require(contract.get("expected_train_mode") == CELL_TRAIN_MODES[cell], f"Cell {cell} train mode mismatch")
        _require(contract.get("fresh_stage1_lora_loaded_before_cell_state") is True, f"Cell {cell} was not fresh")
        _require(
            contract.get("fresh_stage1_lora_sha256") == stage1["loaded_lora_sha256"],
            f"Cell {cell} fresh Stage1 LoRA hash mismatch",
        )
        _require(
            contract.get("active_head_sha256") == heads[cell]["head_state_sha256"], f"Cell {cell} active head mismatch"
        )
        _require(
            contract.get("active_lora_sha256") == loras[cell]["lora_state_sha256"], f"Cell {cell} active LoRA mismatch"
        )
        expected_loss = {
            "base_weight": 1.0,
            "identity_weight": 2.0 if cell == "identity-trained" else 0.0,
            "panorama_weight": 0.0 if cell == "warmup-original" else 1.0,
            "panorama_objective": "global_raw_heatmap_pixel_ce",
            "view_readout": "raw_heatmap_spatial_logsumexp_marginal",
            "control_differs_only_by_identity_term": True,
        }
        _require(contract.get("loss_contract") == expected_loss, f"Cell {cell} loss contract mismatch")

    _require(
        all(heads[cell] == heads[CELL_NAMES[0]] for cell in CELL_NAMES[1:]),
        "Three cells do not share one exact warmup head",
    )
    warmup_contract = selected["warmup-original"].get("shared_warmup_contract")
    _require(isinstance(warmup_contract, dict), "Shared warmup contract missing")
    _require(
        all(selected[cell].get("shared_warmup_contract") == warmup_contract for cell in CELL_NAMES),
        "Three cells do not share the exact warmup checkpoint contract",
    )
    _require(warmup_contract.get("schema") == CHECKPOINT_SCHEMA, "Warmup checkpoint schema mismatch")
    _require(warmup_contract.get("protocol") == TRAIN_PROTOCOL, "Warmup checkpoint protocol mismatch")
    _require(_valid_sha256(warmup_contract.get("file_sha256")), "Warmup checkpoint file hash invalid")
    _require(_valid_sha256(warmup_contract.get("head_state_sha256")), "Warmup head hash invalid")
    _require(_valid_sha256(warmup_contract.get("lora_state_sha256")), "Warmup LoRA hash invalid")
    _require(
        (heads["warmup-original"]["file_sha256"], heads["warmup-original"]["head_state_sha256"])
        == (warmup_contract["file_sha256"], warmup_contract["head_state_sha256"]),
        "Active shared head is not bound to the supplied warmup checkpoint",
    )
    _require(
        selected["warmup-original"].get("selected_checkpoint_path") == heads["warmup-original"]["path"]
        and selected["warmup-original"].get("selected_checkpoint_file_sha256")
        == heads["warmup-original"]["file_sha256"],
        "warmup-original selected checkpoint is not the shared-head source",
    )
    _require(
        loras["warmup-original"]["path"] == stage1.get("path")
        and loras["warmup-original"]["file_sha256"] == stage1["file_sha256"]
        and loras["warmup-original"]["lora_state_sha256"] == stage1["loaded_lora_sha256"],
        "warmup-original LoRA is not the pinned Stage1-S2 source",
    )
    for cell in ("identity-trained", "heatmap-control-trained"):
        _require(
            selected[cell].get("selected_checkpoint_path") == loras[cell]["path"]
            and selected[cell].get("selected_checkpoint_file_sha256") == loras[cell]["file_sha256"],
            f"Cell {cell} selected checkpoint is not its active LoRA source",
        )
        _require(
            loras[cell]["lora_state_sha256"] != stage1["loaded_lora_sha256"],
            f"Cell {cell} LoRA did not change from Stage1-S2",
        )
    _require(
        loras["identity-trained"]["file_sha256"] != loras["heatmap-control-trained"]["file_sha256"],
        "Identity and control point to the same checkpoint file",
    )
    _require(
        loras["identity-trained"]["lora_state_sha256"] != loras["heatmap-control-trained"]["lora_state_sha256"],
        "Identity and control active LoRA states are bitwise identical",
    )

    _require(
        selected["warmup-original"].get("identity_control_pair_gate") is None,
        "Warmup cell unexpectedly has a causal-pair gate",
    )
    identity_pair = selected["identity-trained"].get("identity_control_pair_gate")
    control_pair = selected["heatmap-control-trained"].get("identity_control_pair_gate")
    _require(isinstance(identity_pair, dict) and identity_pair == control_pair, "Trained cells disagree on causal pair")
    _require(identity_pair.get("passed") is True, "Identity/control causal-pair gate failed")
    _require(
        set(identity_pair.get("matched_contracts", ())) == EXPECTED_PAIR_MATCHED_CONTRACTS,
        "Identity/control causal pair did not match every registered contract",
    )
    _require(
        identity_pair.get("only_registered_difference") == {"identity_weight": [2.0, 0.0]},
        "Identity/control pair differs outside the registered auxiliary loss",
    )
    for pair_key, cell in (
        ("identity_checkpoint", "identity-trained"),
        ("control_checkpoint", "heatmap-control-trained"),
    ):
        pair_source = identity_pair.get(pair_key)
        _require(isinstance(pair_source, dict), f"Causal pair has no {pair_key}")
        _require(
            pair_source.get("path") == loras[cell]["path"]
            and pair_source.get("file_sha256") == loras[cell]["file_sha256"],
            f"Causal pair {pair_key} does not bind the active cell checkpoint",
        )


def _validated_source_samples(reports: Mapping[str, Mapping[str, Any]]) -> int:
    """Bind the full-validation sample count across all three eval reports."""

    canonical: int | None = None
    for cell in CELL_NAMES:
        manifest = reports[cell].get("manifest_contract")
        _require(isinstance(manifest, dict), f"Cell {cell} manifest contract missing")
        value = manifest.get("val_samples")
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"Cell {cell} manifest val_samples must be a positive integer",
        )
        scope = reports[cell].get("evaluation_scope")
        _require(
            scope
            == {
                "selection_split": "val",
                "standard_only": False,
                "source_samples": value,
            },
            f"Cell {cell} evaluation scope is not the complete manifest validation split",
        )
        if canonical is None:
            canonical = value
        else:
            _require(
                value == canonical,
                "Three eval reports do not agree on manifest val_samples",
            )
    assert canonical is not None
    return canonical


def validate_contracts(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    width: int,
) -> dict[str, Any]:
    _require(set(reports) == set(CELL_NAMES), "Exactly the three registered visual-identity cells are required")
    source_samples = _validated_source_samples(reports)
    for cell in CELL_NAMES:
        report = reports[cell]
        _require(report.get("schema") == REPORT_SCHEMA, f"Cell {cell} report schema mismatch")
        _require(report.get("protocol") == REPORT_PROTOCOL, f"Cell {cell} report protocol mismatch")
        _require(report.get("phase") == "eval", f"Cell {cell} is not an eval report")
        _require(report.get("cell") == cell, f"Cell {cell} report cell label mismatch")
        _require(tuple(report.get("interventions", ())) == INTERVENTIONS, f"Cell {cell} intervention order mismatch")
        manifest = report.get("manifest_contract")
        _require(isinstance(manifest, dict), f"Cell {cell} manifest contract missing")
        _require(
            manifest.get("val_samples") == source_samples,
            f"Cell {cell} validation sample count differs from the shared contract",
        )
        _require(manifest.get("scene_disjoint") is True, f"Cell {cell} train/validation scenes are not disjoint")
        _require(manifest.get("identity_targets_per_sample") == REQUIRED_TARGETS, f"Cell {cell} does not use K=4")
        _require(
            _finite_number(manifest.get("minimum_target_separation_pixels"))
            and float(manifest["minimum_target_separation_pixels"]) >= MINIMUM_TARGET_SEPARATION,
            f"Cell {cell} target-separation contract is below 12 pixels",
        )
        state = report.get("state_and_input_contract")
        _require(isinstance(state, dict), f"Cell {cell} state/input contract missing")
        expected_state = {
            "explicit_pose_inputs_removed": True,
            "history_query_source": HISTORY_QUERY_SOURCE,
            "history_query_layer": 20,
            "history_visual_views_per_query": 4,
            "shared_head_across_cells": True,
            "qwen_forward_batch_size": 1,
            "qwen_forwards_per_sample": 4,
            "view_metric_source": "raw_heatmap_spatial_logsumexp_marginal",
            "learned_visibility_readout_used_for_view_metric": False,
        }
        _require(state == expected_state, f"Cell {cell} state/input contract mismatch")
        fresh = report.get("fresh_process_contract")
        _require(isinstance(fresh, dict), f"Cell {cell} fresh-process contract missing")
        _require(fresh.get("fresh_stage1_loaded_before_cell_state") is True, f"Cell {cell} did not load fresh Stage1")
        _require(
            isinstance(fresh.get("training_pid"), int)
            and isinstance(fresh.get("evaluation_pid"), int)
            and fresh["training_pid"] != fresh["evaluation_pid"],
            f"Cell {cell} was not evaluated in a fresh process",
        )
        _validate_report_gates(
            report,
            cell=cell,
            source_samples=source_samples,
        )

    for key in ("stage1_s2_contract", "manifest_contract", "pose_free_config_contract", "runtime_contract"):
        _require(_same(reports, key), f"Three cells do not share the same {key}")
    config = reports["warmup-original"]["pose_free_config_contract"]
    runtime = reports["warmup-original"]["runtime_contract"]
    for name, contract in (("config", config), ("runtime", runtime)):
        _require(contract.get("history_query_source") == HISTORY_QUERY_SOURCE, f"{name} uses the wrong query source")
        _require(contract.get("qwen_forward_batch_size") == 1, f"{name} Qwen forward batch is not B=1")
        _require(contract.get("qwen_forwards_per_sample") == 4, f"{name} does not use four Qwen forwards")
    _require(config.get("history_query_layer") == 20, "Config history query layer is not 20")
    _require(config.get("history_visual_views_per_query") == 4, "Config visual query does not pool four views")
    _require(
        config.get("history_visual_view_reduction") == "equal_weight_mean", "Config visual view reduction mismatch"
    )
    _require(config.get("raw_heatmap_logits_required") is True, "Config does not require raw heatmap logits")
    _require(config.get("model_pose_input") is None, "Config exposes a model pose input")
    _require(runtime.get("history_query_layer") == 20, "Runtime history query layer is not 20")
    _require(runtime.get("history_visual_views_per_query") == 4, "Runtime visual query does not pool four views")
    _require(runtime.get("matcher_uses_relative_pose") is False, "Runtime matcher uses relative pose")
    _require(
        runtime.get("raw_heatmap_logits_opt_in") == "return_heatmap_logits=True", "Runtime raw-logit contract mismatch"
    )
    _validate_selected_cell_contracts(reports)
    source_ids, source_hash = _validate_prediction_records(
        reports,
        width=width,
        source_samples=source_samples,
    )
    return {
        "passed": True,
        "report_schema": REPORT_SCHEMA,
        "report_protocol": REPORT_PROTOCOL,
        "strict_four_by_b1_qwen_forwards": True,
        "visual_history_query_source_verified": True,
        "pose_inputs_removed": True,
        "shared_warmup_head_bitwise_exact": True,
        "identity_control_causal_pair_verified": True,
        "all_intervention_bitwise_gates_passed": True,
        "single_anchor_swap_untargeted_routing_exact": True,
        "target_score_reconstruction_verified": True,
        "raw_panorama_view_marginal_verified": True,
        "raw_per_view_and_global_peak_reconstruction_verified": True,
        "source_samples": len(source_ids),
        "source_sample_identity_sha256": source_hash,
    }


def _circular_error(
    pred_view: int,
    pred_xy: np.ndarray,
    gt_view: int,
    gt_xy: np.ndarray,
    *,
    width: int,
) -> float:
    pred_x = pred_view * width + int(pred_xy[0])
    gt_x = gt_view * width + int(gt_xy[0])
    dx = abs(pred_x - gt_x)
    dx = min(dx, 4 * width - dx)
    return math.hypot(dx, int(pred_xy[1]) - int(gt_xy[1]))


def record_counts(
    record: Mapping[str, Any],
    *,
    width: int,
    slots: Iterable[int] | None = None,
) -> dict[str, list[float]]:
    logits = np.asarray(record["visibility_logits"], dtype=np.float64)
    gt_visibility = np.asarray(record["gt_visibility"], dtype=np.float64)
    pred_xy = np.asarray(record["pred_xy"], dtype=np.int64)
    global_pred = np.asarray(record["global_pred_view_xy"], dtype=np.int64)
    gt_xy = np.asarray(record["gt_xy"], dtype=np.int64)
    score = np.asarray(record["target_score_matrix"], dtype=np.float64)
    counts = {metric: [0.0, 0.0] for metric in METRICS}
    selected_slots = range(4) if slots is None else slots
    for history_slot in selected_slots:
        positives = np.flatnonzero(gt_visibility[history_slot] > 0.5)
        _require(positives.size == 1, "Compact metric requires exactly one visible view per target")
        gt_view = int(positives[0])
        predicted_view = int(np.argmax(logits[history_slot]))

        score_identity = int(np.argmax(score[history_slot]))
        counts["score_nearest_identity_accuracy"][0] += float(score_identity == history_slot)
        counts["score_nearest_identity_accuracy"][1] += 1.0

        conditional_error = _circular_error(
            gt_view,
            pred_xy[history_slot, gt_view],
            gt_view,
            gt_xy[history_slot, gt_view],
            width=width,
        )
        counts["conditional_pck8"][0] += float(conditional_error <= 8.0)
        counts["conditional_pck8"][1] += 1.0
        view_correct = predicted_view == gt_view
        counts["visible_view_accuracy"][0] += float(view_correct)
        counts["visible_view_accuracy"][1] += 1.0
        joint_error = _circular_error(
            predicted_view,
            pred_xy[history_slot, predicted_view],
            gt_view,
            gt_xy[history_slot, gt_view],
            width=width,
        )
        counts["true_joint_pck8"][0] += float(view_correct and joint_error <= 8.0)
        counts["true_joint_pck8"][1] += 1.0
        global_error = _circular_error(
            int(global_pred[history_slot, 0]),
            global_pred[history_slot, 1:],
            gt_view,
            gt_xy[history_slot, gt_view],
            width=width,
        )
        counts["global_map_joint_pck8"][0] += float(global_error <= 8.0)
        counts["global_map_joint_pck8"][1] += 1.0

        candidates: list[tuple[float, int]] = []
        for target_slot in range(4):
            target_views = np.flatnonzero(gt_visibility[target_slot] > 0.5)
            _require(target_views.size == 1, "Compact identity requires one target view")
            target_view = int(target_views[0])
            candidates.append(
                (
                    _circular_error(
                        predicted_view,
                        pred_xy[history_slot, predicted_view],
                        target_view,
                        gt_xy[target_slot, target_view],
                        width=width,
                    ),
                    target_slot,
                )
            )
        nearest = min(candidates)[1]
        counts["peak_nearest_identity_accuracy"][0] += float(nearest == history_slot)
        counts["peak_nearest_identity_accuracy"][1] += 1.0
    return counts


def build_contributions(
    records: list[dict[str, Any]],
    source_ids: list[str],
    *,
    width: int,
    targeted_swap: bool = False,
) -> dict[str, np.ndarray]:
    index = {sample_id: position for position, sample_id in enumerate(source_ids)}
    values = {metric: np.zeros((len(source_ids), 2), dtype=np.float64) for metric in METRICS}
    seen: dict[str, list[int]] = {sample_id: [] for sample_id in source_ids}
    for record in records:
        sample_id = str(record["sample_id"])
        _require(sample_id in index, f"Unknown source sample {sample_id}")
        target_slot = record.get("target_slot")
        if targeted_swap:
            _require(target_slot in range(4), f"Invalid swap target slot for {sample_id}")
            slots: Iterable[int] | None = [int(target_slot)]
            seen[sample_id].append(int(target_slot))
        else:
            slots = None
            seen[sample_id].append(-1)
        counts = record_counts(record, width=width, slots=slots)
        for metric, pair in counts.items():
            values[metric][index[sample_id]] += pair
    expected_seen = [0, 1, 2, 3] if targeted_swap else [-1]
    for sample_id, slots_seen in seen.items():
        _require(slots_seen == expected_seen, f"Contribution grouping mismatch for {sample_id}")
    return values


def all_same_peak_contribution(records: list[dict[str, Any]], source_ids: list[str]) -> np.ndarray:
    by_id = {str(record["sample_id"]): record for record in records}
    output = np.zeros((len(source_ids), 2), dtype=np.float64)
    for position, sample_id in enumerate(source_ids):
        record = by_id[sample_id]
        logits = np.asarray(record["visibility_logits"], dtype=np.float64)
        xy = np.asarray(record["pred_xy"], dtype=np.int64)
        peaks = []
        for slot in range(4):
            view = int(np.argmax(logits[slot]))
            peaks.append((view, int(xy[slot, view, 0]), int(xy[slot, view, 1])))
        output[position] = (float(len(set(peaks)) == 1), 1.0)
    return output


def _point(contribution: np.ndarray) -> float:
    denominator = float(contribution[:, 1].sum())
    _require(denominator > 0.0, "Metric has zero denominator")
    return float(contribution[:, 0].sum() / denominator)


def _bootstrap_distribution(contribution: np.ndarray, weights: np.ndarray) -> np.ndarray:
    numerator = weights @ contribution[:, 0]
    denominator = weights @ contribution[:, 1]
    _require(bool(np.all(denominator > 0.0)), "Bootstrap replicate has zero denominator")
    return numerator / denominator


def _interval(values: np.ndarray) -> list[float]:
    low, high = np.percentile(values, (2.5, 97.5))
    return [float(low), float(high)]


def metric_summary(contribution: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    return {
        "estimate": _point(contribution),
        "ci95": _interval(_bootstrap_distribution(contribution, weights)),
        "numerator": float(contribution[:, 0].sum()),
        "denominator": float(contribution[:, 1].sum()),
    }


def difference_summary(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    distribution = _bootstrap_distribution(left, weights) - _bootstrap_distribution(right, weights)
    interval = _interval(distribution)
    return {
        "estimate": _point(left) - _point(right),
        "ci95": interval,
        "ci95_excludes_zero": bool(interval[0] > 0.0 or interval[1] < 0.0),
    }


def _reported_metric(evaluation: Mapping[str, Any], metric: str) -> float:
    if metric == "peak_nearest_identity_accuracy":
        value = evaluation.get("anchor_identity", {}).get("accuracy")
    else:
        key = {
            "visible_view_accuracy": "visible_view_accuracy",
            "conditional_pck8": "pck8",
            "true_joint_pck8": "joint_pck8",
            "global_map_joint_pck8": "global_map_joint_pck8",
        }[metric]
        value = evaluation.get(key)
    _require(_finite_number(value), f"Evaluation is missing reported metric {metric}")
    return float(value)


def _validate_recomputed_metrics(
    reports: Mapping[str, Mapping[str, Any]],
    contributions: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> None:
    reported_metrics = tuple(metric for metric in METRICS if metric != "score_nearest_identity_accuracy")
    for cell in CELL_NAMES:
        evaluations = reports[cell]["evaluations"]
        for intervention in ("standard", "history-shuffle", "current-shuffle"):
            for metric in reported_metrics:
                actual = _point(contributions[cell][intervention][metric])
                reported = _reported_metric(evaluations[intervention], metric)
                _require(
                    math.isclose(actual, reported, rel_tol=0.0, abs_tol=1e-12),
                    f"Cell {cell} {intervention} compact/reported {metric} mismatch",
                )
        targeted = evaluations["single-anchor-swap"].get("targeted_slot_metrics")
        _require(isinstance(targeted, dict), f"Cell {cell} targeted swap metrics missing")
        for metric in reported_metrics:
            actual = _point(contributions[cell]["single-anchor-swap"][metric])
            reported = _reported_metric(targeted, metric)
            _require(
                math.isclose(actual, reported, rel_tol=0.0, abs_tol=1e-12),
                f"Cell {cell} targeted-swap compact/reported {metric} mismatch",
            )


def _check(value: float, threshold: float, operator: str) -> dict[str, Any]:
    if operator == ">=":
        passed = value >= threshold
    elif operator == "<":
        passed = value < threshold
    else:
        raise ValueError(f"Unsupported gate operator: {operator}")
    return {"passed": bool(passed), "value": value, "threshold": threshold, "operator": operator}


def _check_positive_ci(comparison: Mapping[str, Any]) -> dict[str, Any]:
    interval = comparison.get("ci95")
    _require(
        isinstance(interval, list) and len(interval) == 2 and all(_finite_number(value) for value in interval),
        "Comparison is missing a finite paired-bootstrap CI",
    )
    lower = float(interval[0])
    return {
        "passed": lower > 0.0,
        "ci95": [float(interval[0]), float(interval[1])],
        "threshold": 0.0,
        "operator": "ci95_lower>",
    }


def summarize(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    report_paths: Mapping[str, str | Path] | None = None,
    bootstrap_samples: int = 50_000,
    seed: int = 42,
    heatmap_width: int = 64,
) -> dict[str, Any]:
    _require(bootstrap_samples > 0, "bootstrap_samples must be positive")
    _require(heatmap_width > 0, "heatmap_width must be positive")
    contract = validate_contracts(reports, width=heatmap_width)
    source_ids = [str(record["sample_id"]) for record in reports["warmup-original"]["prediction_records"]["standard"]]
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(source_ids),
        np.full(len(source_ids), 1.0 / len(source_ids), dtype=np.float64),
        size=bootstrap_samples,
    ).astype(np.float64, copy=False)

    contributions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    collapse: dict[str, np.ndarray] = {}
    for cell in CELL_NAMES:
        records = reports[cell]["prediction_records"]
        contributions[cell] = {}
        for intervention in ("standard", "history-shuffle", "current-shuffle"):
            contributions[cell][intervention] = build_contributions(
                records[intervention], source_ids, width=heatmap_width
            )
        contributions[cell]["single-anchor-swap"] = build_contributions(
            records["single-anchor-swap"],
            source_ids,
            width=heatmap_width,
            targeted_swap=True,
        )
        collapse[cell] = all_same_peak_contribution(records["standard"], source_ids)
    _validate_recomputed_metrics(reports, contributions)

    cells: dict[str, Any] = {}
    causal_effects: dict[str, Any] = {}
    for cell in CELL_NAMES:
        cells[cell] = {
            "standard": {
                metric: metric_summary(contributions[cell]["standard"][metric], weights) for metric in METRICS
            },
            "all_same_peak_fraction": metric_summary(collapse[cell], weights),
        }
        causal_effects[cell] = {}
        for intervention in CAUSAL_INTERVENTIONS:
            causal_effects[cell][f"standard-minus-{intervention}"] = {
                "comparison_contract": (
                    "standard minus targeted swapped-slot outputs, with four replicas grouped by source"
                    if intervention == "single-anchor-swap"
                    else f"standard minus {intervention}, paired by source sample"
                ),
                "metrics": {
                    metric: difference_summary(
                        contributions[cell]["standard"][metric],
                        contributions[cell][intervention][metric],
                        weights,
                    )
                    for metric in METRICS
                },
            }

    comparisons: dict[str, Any] = {}
    for name, left, right in COMPARISONS:
        comparisons[name] = {
            "left": left,
            "right": right,
            "metrics": {
                metric: difference_summary(
                    contributions[left]["standard"][metric],
                    contributions[right]["standard"][metric],
                    weights,
                )
                for metric in METRICS
            },
        }

    identity_standard = cells["identity-trained"]["standard"]
    identity_control = comparisons["identity-minus-control"]["metrics"]
    panorama_control = cells["heatmap-control-trained"]["standard"]
    panorama_control_warmup = comparisons["control-minus-warmup"]["metrics"]
    identity_causal = causal_effects["identity-trained"]
    checks = {
        "identity_score_nearest_accuracy_at_least_0_45": _check(
            identity_standard["score_nearest_identity_accuracy"]["estimate"],
            SCORE_IDENTITY_MINIMUM,
            ">=",
        ),
        "identity_score_nearest_gain_vs_control_at_least_0_10": _check(
            identity_control["score_nearest_identity_accuracy"]["estimate"],
            SCORE_IDENTITY_VS_CONTROL_MINIMUM,
            ">=",
        ),
        "identity_peak_nearest_accuracy_at_least_0_35": _check(
            identity_standard["peak_nearest_identity_accuracy"]["estimate"],
            PEAK_IDENTITY_MINIMUM,
            ">=",
        ),
        "identity_peak_nearest_gain_vs_control_at_least_0_05": _check(
            identity_control["peak_nearest_identity_accuracy"]["estimate"],
            PEAK_IDENTITY_VS_CONTROL_MINIMUM,
            ">=",
        ),
        "identity_history_shuffle_score_identity_drop_at_least_0_05": _check(
            identity_causal["standard-minus-history-shuffle"]["metrics"]["score_nearest_identity_accuracy"]["estimate"],
            CAUSAL_SCORE_IDENTITY_DROP_MINIMUM,
            ">=",
        ),
        "identity_current_shuffle_score_identity_drop_at_least_0_05": _check(
            identity_causal["standard-minus-current-shuffle"]["metrics"]["score_nearest_identity_accuracy"]["estimate"],
            CAUSAL_SCORE_IDENTITY_DROP_MINIMUM,
            ">=",
        ),
        "identity_targeted_swap_score_identity_drop_at_least_0_05": _check(
            identity_causal["standard-minus-single-anchor-swap"]["metrics"]["score_nearest_identity_accuracy"][
                "estimate"
            ],
            CAUSAL_SCORE_IDENTITY_DROP_MINIMUM,
            ">=",
        ),
        "identity_all_same_peak_fraction_below_0_40": _check(
            cells["identity-trained"]["all_same_peak_fraction"]["estimate"],
            ALL_SAME_PEAK_FRACTION_MAXIMUM,
            "<",
        ),
        "identity_true_joint_pck8_not_more_than_0_03_below_control": _check(
            identity_control["true_joint_pck8"]["estimate"],
            TRUE_JOINT_PCK8_VS_CONTROL_MINIMUM,
            ">=",
        ),
        "identity_global_map_joint_pck8_not_more_than_0_03_below_control": _check(
            identity_control["global_map_joint_pck8"]["estimate"],
            GLOBAL_MAP_JOINT_PCK8_VS_CONTROL_MINIMUM,
            ">=",
        ),
        "panorama_control_raw_view_accuracy_at_least_0_40": _check(
            panorama_control["visible_view_accuracy"]["estimate"],
            PANORAMA_CONTROL_VIEW_ACCURACY_MINIMUM,
            ">=",
        ),
        "panorama_control_raw_view_gain_vs_warmup_at_least_0_10": _check(
            panorama_control_warmup["visible_view_accuracy"]["estimate"],
            PANORAMA_CONTROL_VIEW_GAIN_MINIMUM,
            ">=",
        ),
        "panorama_control_raw_view_gain_ci95_excludes_zero": _check_positive_ci(
            panorama_control_warmup["visible_view_accuracy"]
        ),
        "panorama_control_global_map_pck8_at_least_0_15": _check(
            panorama_control["global_map_joint_pck8"]["estimate"],
            PANORAMA_CONTROL_GLOBAL_MAP_PCK8_MINIMUM,
            ">=",
        ),
        "panorama_control_global_map_gain_vs_warmup_at_least_0_05": _check(
            panorama_control_warmup["global_map_joint_pck8"]["estimate"],
            PANORAMA_CONTROL_GLOBAL_MAP_GAIN_MINIMUM,
            ">=",
        ),
        "panorama_control_global_map_gain_ci95_excludes_zero": _check_positive_ci(
            panorama_control_warmup["global_map_joint_pck8"]
        ),
        "panorama_control_conditional_pck8_gain_vs_warmup_at_least_0_03": _check(
            panorama_control_warmup["conditional_pck8"]["estimate"],
            PANORAMA_CONTROL_CONDITIONAL_PCK8_GAIN_MINIMUM,
            ">=",
        ),
    }
    overall = all(check["passed"] for check in checks.values())

    if report_paths is None:
        inputs = {cell: {"path": None, "file_sha256": None} for cell in CELL_NAMES}
    else:
        _require(set(report_paths) == set(CELL_NAMES), "report_paths must bind all three cells")
        inputs = {
            cell: {
                "path": str(Path(report_paths[cell]).resolve()),
                "file_sha256": file_sha256(report_paths[cell]),
            }
            for cell in CELL_NAMES
        }
    return {
        "schema": SUMMARY_SCHEMA,
        "inputs": inputs,
        "contract_validation": contract,
        "bootstrap_contract": {
            "method": "paired_source_sample_percentile_bootstrap",
            "resampling_unit": "source_sample",
            "source_samples": len(source_ids),
            "replicates": bootstrap_samples,
            "seed": seed,
            "paired_across_cells_and_interventions": True,
            "four_history_outputs_remain_grouped": True,
            "single_anchor_swap_replicas_remain_grouped": True,
        },
        "metric_contract": {
            "score_nearest_identity": "row argmax of the registered 4x4 target-grounded score matrix",
            "peak_nearest_identity": "nearest of four GT panorama targets to the predicted output peak",
            "visible_view_accuracy": "argmax raw-logit spatial-logsumexp view equals the unique visible GT view",
            "conditional_pck8": "raw-logit peak error <=8 pixels in the ground-truth visible view",
            "true_joint_pck8": "raw marginal view is correct and its raw-logit peak error is <=8 pixels",
            "global_map_joint_pck8": "circular panorama error of raw argmax over all 4HW pixels is <=8 pixels",
            "learned_visibility_readout_used": False,
            "heatmap_view_width": heatmap_width,
        },
        "decision_threshold_contract": {
            "schema": SUMMARY_SCHEMA,
            "predeclared_and_not_cli_tunable": True,
            "score_identity_minimum": SCORE_IDENTITY_MINIMUM,
            "score_identity_vs_control_minimum": SCORE_IDENTITY_VS_CONTROL_MINIMUM,
            "peak_identity_minimum": PEAK_IDENTITY_MINIMUM,
            "peak_identity_vs_control_minimum": PEAK_IDENTITY_VS_CONTROL_MINIMUM,
            "history_shuffle_score_identity_drop_minimum": CAUSAL_SCORE_IDENTITY_DROP_MINIMUM,
            "current_shuffle_score_identity_drop_minimum": CAUSAL_SCORE_IDENTITY_DROP_MINIMUM,
            "targeted_swap_score_identity_drop_minimum": CAUSAL_SCORE_IDENTITY_DROP_MINIMUM,
            "all_same_peak_fraction_maximum_exclusive": ALL_SAME_PEAK_FRACTION_MAXIMUM,
            "true_joint_pck8_vs_control_minimum": TRUE_JOINT_PCK8_VS_CONTROL_MINIMUM,
            "global_map_joint_pck8_vs_control_minimum": GLOBAL_MAP_JOINT_PCK8_VS_CONTROL_MINIMUM,
            "panorama_control_view_accuracy_minimum": PANORAMA_CONTROL_VIEW_ACCURACY_MINIMUM,
            "panorama_control_view_gain_minimum": PANORAMA_CONTROL_VIEW_GAIN_MINIMUM,
            "panorama_control_global_map_pck8_minimum": PANORAMA_CONTROL_GLOBAL_MAP_PCK8_MINIMUM,
            "panorama_control_global_map_gain_minimum": PANORAMA_CONTROL_GLOBAL_MAP_GAIN_MINIMUM,
            "panorama_control_conditional_pck8_gain_minimum": PANORAMA_CONTROL_CONDITIONAL_PCK8_GAIN_MINIMUM,
            "panorama_control_gain_ci95_lower_must_exceed_zero": True,
        },
        "cells": cells,
        "causal_comparisons": comparisons,
        "causal_effects": causal_effects,
        "decision_gate": {
            "checks": checks,
            "overall_passed": overall,
            "overall_stage2_stage3_authorized": overall,
            "failure_action": (
                None
                if overall
                else "Do not authorize Stage2/Stage3: visual identity has not passed every preregistered causal gate."
            ),
        },
        "overall_stage2_stage3_authorized": overall,
    }


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = {
        "warmup-original": args.warmup_report,
        "identity-trained": args.identity_report,
        "heatmap-control-trained": args.control_report,
    }
    reports = {cell: load_json(path) for cell, path in paths.items()}
    summary = summarize(
        reports,
        report_paths=paths,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        heatmap_width=args.heatmap_width,
    )
    json_dump(Path(args.output), summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

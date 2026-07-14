#!/usr/bin/env python3
"""Fail-closed summary for the strict-B=1 Task-3.6c factorial evaluation.

The four required cells isolate the two trainable components:

* A: head-only head + the pinned Stage1-S2 LoRA;
* B: jointly trained head + jointly trained LoRA;
* C: jointly trained head + the pinned Stage1-S2 LoRA;
* D: head-only head + jointly trained LoRA.

Only compact prediction records embedded in the evaluation reports are read.
Bootstrap resampling is paired by source sample, never by history slot or by
the four single-anchor-swap replicas of a source sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

REPORT_SCHEMA = "task36c_pose_free_multihistory_report_v1"
SUMMARY_SCHEMA = "task36c_pose_free_b1_factorial_summary_v2"
CELL_NAMES = ("A", "B", "C", "D")
INTERVENTIONS = (
    "standard",
    "blank-images",
    "history-shuffle",
    "current-shuffle",
    "single-anchor-swap",
)
METRICS = (
    "anchor_identity_accuracy",
    "visible_view_accuracy",
    "conditional_pck4",
    "conditional_pck8",
    "true_joint_pck4",
    "true_joint_pck8",
)
FACTORIAL_COMPARISONS = (
    ("B-A", "B", "A"),
    ("B-C", "B", "C"),
    ("D-A", "D", "A"),
    ("D-C", "D", "C"),
)
CAUSAL_INTERVENTIONS = (
    "history-shuffle",
    "current-shuffle",
    "single-anchor-swap",
)

# These are decision thresholds, not CLI-tunable analysis choices.  Changing
# them requires changing the summary schema/version rather than post-hoc flags.
IDENTITY_MINIMUM = 0.35
HISTORY_SHUFFLE_IDENTITY_DROP_MINIMUM = 0.05
ALL_SAME_PEAK_FRACTION_MAXIMUM = 0.40
JOINT_VS_HEAD_IDENTITY_DELTA_MINIMUM = 0.05
LORA_RETENTION_IDENTITY_DELTA_MINIMUM = 0.05
CONDITIONAL_PCK8_MINIMUM = 0.30
TRUE_JOINT_PCK8_MINIMUM = 0.10
CURRENT_SHUFFLE_TRUE_JOINT_PCK8_DROP_MINIMUM = 0.05
SINGLE_SWAP_IDENTITY_DROP_MINIMUM = 0.05
PURE_LORA_IDENTITY_DELTA_MINIMUM = 0.05
TARGETED_OUTPUT_CHANGE_MINIMUM_EXCLUSIVE = 0.0
UNTARGETED_OUTPUT_CHANGE_EXACT = 0.0

PAIRED_SWAP_CONTRACT = "replace history i; compare output i against all output j!=i on the same current"
PAIRED_SWAP_METRICS = (
    "mean_heatmap_l1",
    "mean_visibility_l1",
    "mean_peak_displacement",
)

CELL_EXPECTATIONS = {
    "A": {
        "label": "head-only head + Stage1-S2 LoRA",
        "branch": "head-only",
        "eval_lora": "off",
        "head_branch": "head-only",
        "lora_branch": "stage1-s2",
        "head_override": False,
    },
    "B": {
        "label": "joint head + joint LoRA",
        "branch": "heatmap-lora",
        "eval_lora": "trained",
        "head_branch": "heatmap-lora",
        "lora_branch": "heatmap-lora",
        "head_override": False,
    },
    "C": {
        "label": "joint head + Stage1-S2 LoRA",
        "branch": "heatmap-lora",
        "eval_lora": "off",
        "head_branch": "heatmap-lora",
        "lora_branch": "stage1-s2",
        "head_override": False,
    },
    "D": {
        "label": "head-only head + joint LoRA",
        "branch": "heatmap-lora",
        "eval_lora": "trained",
        "head_branch": "head-only",
        "lora_branch": "heatmap-lora",
        "head_override": True,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-report", required=True)
    parser.add_argument("--b-report", required=True)
    parser.add_argument("--c-report", required=True)
    parser.add_argument("--d-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--heatmap-width",
        type=int,
        default=64,
        help="Width of one panorama view in compact pred_xy/gt_xy records.",
    )
    args = parser.parse_args(argv)
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if args.heatmap_width <= 0:
        parser.error("--heatmap-width must be positive")
    return args


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


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _same(reports: dict[str, dict[str, Any]], key: str) -> bool:
    values = [reports[cell].get(key) for cell in CELL_NAMES]
    return all(value == values[0] for value in values[1:])


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _validate_paired_swap_routing(
    evaluation: dict[str, Any],
    *,
    cell: str,
    source_samples: int,
) -> dict[str, Any]:
    paired = evaluation.get("paired_output_change_vs_standard")
    _require(isinstance(paired, dict), f"Cell {cell} has no paired single-swap routing metrics")
    _require(paired.get("contract") == PAIRED_SWAP_CONTRACT, f"Cell {cell} paired swap contract mismatch")

    expected_comparisons = {
        "targeted": source_samples * 4,
        "untargeted": source_samples * 4 * 3,
    }
    for route, expected in expected_comparisons.items():
        values = paired.get(route)
        _require(isinstance(values, dict), f"Cell {cell} paired swap {route} route missing")
        _require(
            values.get("comparisons") == expected,
            f"Cell {cell} paired swap {route} comparison count mismatch",
        )
        for metric in PAIRED_SWAP_METRICS:
            _require(
                _finite_nonnegative(values.get(metric)),
                f"Cell {cell} paired swap {route} {metric} is not finite/nonnegative",
            )

    untargeted = paired["untargeted"]
    for metric in PAIRED_SWAP_METRICS:
        _require(
            float(untargeted[metric]) == UNTARGETED_OUTPUT_CHANGE_EXACT,
            f"Cell {cell} single-swap changed untargeted route {metric}",
        )
    _require(
        paired.get("targeted_to_untargeted_heatmap_l1_ratio") is None,
        f"Cell {cell} paired swap ratio must be null when untargeted heatmap L1 is exactly zero",
    )
    return paired


def _validate_source(source: Any, *, cell: str, component: str, branch: str) -> dict[str, Any]:
    _require(isinstance(source, dict), f"Cell {cell} has no {component} checkpoint source")
    _require(source.get("branch") == branch, f"Cell {cell} {component} source branch mismatch")
    _require(isinstance(source.get("path"), str) and source["path"], f"Cell {cell} {component} source path missing")
    _require(_valid_sha256(source.get("file_sha256")), f"Cell {cell} {component} file hash invalid")
    _require(_valid_sha256(source.get("lora_state_sha256")), f"Cell {cell} {component} LoRA hash invalid")
    if component == "head":
        _require(_valid_sha256(source.get("head_state_sha256")), f"Cell {cell} head hash invalid")
    return source


def _record_key(record: dict[str, Any]) -> tuple[str, int | None]:
    return str(record.get("sample_id")), record.get("target_slot")


def _labels(record: dict[str, Any]) -> tuple[Any, Any]:
    return record.get("gt_visibility"), record.get("gt_xy")


def _validate_record_shapes(record: dict[str, Any], *, width: int, context: str) -> None:
    visibility = np.asarray(record.get("visibility_logits"), dtype=np.float64)
    gt_visibility = np.asarray(record.get("gt_visibility"), dtype=np.float64)
    pred_xy = np.asarray(record.get("pred_xy"), dtype=np.int64)
    gt_xy = np.asarray(record.get("gt_xy"), dtype=np.int64)
    _require(visibility.shape == (4, 4), f"{context}: visibility_logits must be [4,4]")
    _require(gt_visibility.shape == (4, 4), f"{context}: gt_visibility must be [4,4]")
    _require(pred_xy.shape == (4, 4, 2), f"{context}: pred_xy must be [4,4,2]")
    _require(gt_xy.shape == (4, 4, 2), f"{context}: gt_xy must be [4,4,2]")
    _require(np.isfinite(visibility).all(), f"{context}: non-finite visibility logit")
    for name, xy in (("pred_xy", pred_xy), ("gt_xy", gt_xy)):
        _require(bool(((xy[..., 0] >= 0) & (xy[..., 0] < width)).all()), f"{context}: {name} x outside width={width}")
        _require(bool((xy[..., 1] >= 0).all()), f"{context}: {name} y is negative")


def _validate_prediction_records(
    reports: dict[str, dict[str, Any]],
    *,
    width: int,
) -> tuple[list[str], str]:
    canonical_ids: list[str] | None = None
    canonical_labels: dict[str, tuple[Any, Any]] | None = None
    canonical_keys: dict[str, list[tuple[str, int | None]]] = {}

    for cell in CELL_NAMES:
        records_by_intervention = reports[cell].get("prediction_records")
        _require(isinstance(records_by_intervention, dict), f"Cell {cell} has no prediction_records")
        _require(
            set(records_by_intervention) == set(INTERVENTIONS),
            f"Cell {cell} prediction intervention set mismatch",
        )
        standard = records_by_intervention["standard"]
        _require(isinstance(standard, list) and standard, f"Cell {cell} standard records are empty")
        ids = [str(record.get("sample_id")) for record in standard]
        _require(len(ids) == len(set(ids)), f"Cell {cell} standard sample IDs are not unique")
        manifest_val = reports[cell]["manifest_contract"].get("val_samples")
        _require(manifest_val == len(ids), f"Cell {cell} manifest val_samples does not match records")

        labels: dict[str, tuple[Any, Any]] = {}
        for position, record in enumerate(standard):
            _validate_record_shapes(record, width=width, context=f"cell={cell} standard[{position}]")
            _require(record.get("target_slot") is None, f"Cell {cell} standard target_slot must be null")
            labels[ids[position]] = _labels(record)
            gt_visibility = np.asarray(record["gt_visibility"], dtype=np.float64)
            positive_views = (gt_visibility > 0.5).sum(axis=1)
            _require(
                bool(np.all(positive_views == 1)),
                "Compact identity bootstrap requires exactly one visible GT view per history; "
                f"cell={cell} sample={ids[position]} got {positive_views.tolist()}",
            )

        if canonical_ids is None:
            canonical_ids = ids
            canonical_labels = labels
        else:
            _require(ids == canonical_ids, f"Cell {cell} val source-sample order differs from cell A")
            _require(labels == canonical_labels, f"Cell {cell} GT labels differ from cell A")

        for intervention in INTERVENTIONS:
            records = records_by_intervention[intervention]
            _require(isinstance(records, list), f"Cell {cell} {intervention} records are not a list")
            expected_count = len(ids) * 4 if intervention == "single-anchor-swap" else len(ids)
            _require(len(records) == expected_count, f"Cell {cell} {intervention} record count mismatch")
            keys = [_record_key(record) for record in records]
            if intervention == "single-anchor-swap":
                expected_keys = [(sample_id, slot) for sample_id in ids for slot in range(4)]
            else:
                expected_keys = [(sample_id, None) for sample_id in ids]
            _require(keys == expected_keys, f"Cell {cell} {intervention} sample/slot order mismatch")
            for position, record in enumerate(records):
                _validate_record_shapes(
                    record,
                    width=width,
                    context=f"cell={cell} {intervention}[{position}]",
                )
                _require(
                    _labels(record) == labels[str(record["sample_id"])],
                    f"Cell {cell} {intervention} changes GT labels for {record['sample_id']}",
                )
            if intervention in canonical_keys:
                _require(keys == canonical_keys[intervention], f"Cell {cell} {intervention} keys differ")
            else:
                canonical_keys[intervention] = keys

    assert canonical_ids is not None
    identity_hash = hashlib.sha256("\n".join(canonical_ids).encode("utf-8")).hexdigest()
    return canonical_ids, identity_hash


def validate_contracts(
    reports: dict[str, dict[str, Any]],
    *,
    width: int,
) -> dict[str, Any]:
    _require(set(reports) == set(CELL_NAMES), "Exactly cells A/B/C/D are required")
    for cell in CELL_NAMES:
        report = reports[cell]
        expected = CELL_EXPECTATIONS[cell]
        _require(report.get("schema") == REPORT_SCHEMA, f"Cell {cell} report schema mismatch")
        _require(report.get("phase") == "eval", f"Cell {cell} is not an eval report")
        _require(report.get("branch") == expected["branch"], f"Cell {cell} branch mismatch")
        _require(report.get("eval_lora") == expected["eval_lora"], f"Cell {cell} eval_lora mismatch")
        _require(report.get("explicit_pose_inputs_removed") is True, f"Cell {cell} exposes pose input")
        _require(tuple(report.get("interventions", ())) == INTERVENTIONS, f"Cell {cell} interventions mismatch")

        sources = report.get("checkpoint_sources")
        _require(isinstance(sources, dict), f"Cell {cell} checkpoint_sources missing")
        head = _validate_source(sources.get("head"), cell=cell, component="head", branch=expected["head_branch"])
        lora = _validate_source(sources.get("lora"), cell=cell, component="lora", branch=expected["lora_branch"])
        pilot = report.get("pilot_checkpoint")
        _require(isinstance(pilot, dict), f"Cell {cell} pilot checkpoint contract missing")
        _require(isinstance(pilot.get("path"), str) and pilot["path"], f"Cell {cell} pilot path missing")
        _require(_valid_sha256(pilot.get("file_sha256")), f"Cell {cell} pilot file hash invalid")
        _require(
            _valid_sha256(pilot.get("checkpoint_head_state_sha256")),
            f"Cell {cell} pilot checkpoint head hash invalid",
        )
        _require(pilot.get("branch") == expected["branch"], f"Cell {cell} pilot branch mismatch")
        _require(pilot.get("eval_lora") == expected["eval_lora"], f"Cell {cell} pilot LoRA mode mismatch")
        _require(pilot.get("head_override") is expected["head_override"], f"Cell {cell} head override mismatch")
        _require(pilot.get("head_state_sha256") == head["head_state_sha256"], f"Cell {cell} active head mismatch")
        _require(pilot.get("active_lora_sha256") == lora["lora_state_sha256"], f"Cell {cell} active LoRA mismatch")
        _require(pilot.get("head_source_checkpoint") == head, f"Cell {cell} duplicated head source differs")
        _require(pilot.get("lora_source_checkpoint") == lora, f"Cell {cell} duplicated LoRA source differs")

        evaluations = report.get("evaluations", {})
        _require(isinstance(evaluations, dict), f"Cell {cell} evaluations missing")
        manifest_val_samples = int(report.get("manifest_contract", {}).get("val_samples", -1))
        for intervention in INTERVENTIONS:
            evaluation = evaluations.get(intervention)
            _require(isinstance(evaluation, dict), f"Cell {cell} {intervention} evaluation missing")
            expected_samples = manifest_val_samples * (4 if intervention == "single-anchor-swap" else 1)
            _require(
                evaluation.get("samples") == expected_samples,
                f"Cell {cell} {intervention} evaluation sample count mismatch",
            )
        _validate_paired_swap_routing(
            evaluations["single-anchor-swap"],
            cell=cell,
            source_samples=manifest_val_samples,
        )
        blank = evaluations.get("blank-images", {})
        history = evaluations.get("history-shuffle", {})
        _require(
            blank.get("blank_input_identity_gate", {}).get("passed") is True
            and blank.get("blank_input_identity_gate", {}).get("bitwise_exact") is True,
            f"Cell {cell} blank input identity gate did not pass exactly",
        )
        _require(
            blank.get("blank_output_identity_gate", {}).get("passed") is True
            and blank.get("blank_output_identity_gate", {}).get("bitwise_exact") is True
            and blank.get("blank_output_identity_gate", {}).get("maximum_abs_difference") == 0.0,
            f"Cell {cell} blank output identity gate did not pass exactly",
        )
        _require(
            history.get("permutation_equivariance_gate", {}).get("passed") is True
            and history.get("permutation_equivariance_gate", {}).get("bitwise_exact") is True
            and history.get("permutation_equivariance_gate", {}).get("maximum_abs_difference") == 0.0,
            f"Cell {cell} history permutation equivariance gate did not pass exactly",
        )

    for key in ("stage1_s2_contract", "manifest_contract", "pose_free_config_contract", "runtime_contract"):
        _require(_same(reports, key), f"A/B/C/D do not share the same {key}")

    config = reports["A"]["pose_free_config_contract"]
    runtime = reports["A"]["runtime_contract"]
    for name, contract in (("config", config), ("runtime", runtime)):
        _require(contract.get("isolated_pair_chains") is True, f"{name} does not isolate pair chains")
        _require(contract.get("histories_per_qwen_chain") == 1, f"{name} histories/chain is not one")
        _require(contract.get("qwen_forward_batch_size") == 1, f"{name} Qwen forward batch is not B=1")
        _require(contract.get("qwen_forwards_per_sample") == 4, f"{name} does not use four Qwen forwards")
    _require(config.get("model_pose_input") is None, "Config contract has a model pose input")
    _require(runtime.get("matcher_uses_relative_pose") is False, "Runtime matcher uses relative pose")

    sources = {cell: reports[cell]["checkpoint_sources"] for cell in CELL_NAMES}
    _require(sources["A"]["head"] == sources["D"]["head"], "A/D do not use the same head-only head")
    _require(sources["B"]["head"] == sources["C"]["head"], "B/C do not use the same joint head")
    _require(sources["A"]["lora"] == sources["C"]["lora"], "A/C do not use the same Stage1-S2 LoRA")
    _require(sources["B"]["lora"] == sources["D"]["lora"], "B/D do not use the same joint LoRA")

    stage1 = reports["A"]["stage1_s2_contract"]
    _require(
        sources["A"]["lora"]["path"] == stage1.get("path")
        and sources["A"]["lora"]["file_sha256"] == stage1.get("file_sha256")
        and sources["A"]["lora"]["lora_state_sha256"] == stage1.get("loaded_lora_sha256"),
        "Embedded Stage1-S2 LoRA source does not match the Stage1-S2 load contract",
    )
    _require(
        sources["A"]["head"]["lora_state_sha256"] == stage1.get("loaded_lora_sha256"),
        "Head-only checkpoint did not retain the pinned Stage1-S2 LoRA hash",
    )
    _require(
        sources["B"]["lora"]["lora_state_sha256"] != stage1.get("loaded_lora_sha256"),
        "Joint-trained LoRA source is bitwise identical to the Stage1-S2 LoRA",
    )

    for cell in ("A", "B", "C"):
        pilot = reports[cell]["pilot_checkpoint"]
        head = sources[cell]["head"]
        _require(
            (pilot["path"], pilot["file_sha256"]) == (head["path"], head["file_sha256"]),
            f"Cell {cell} head does not come from its declared base pilot checkpoint",
        )
    for cell in ("B", "D"):
        pilot = reports[cell]["pilot_checkpoint"]
        lora = sources[cell]["lora"]
        _require(
            (pilot["path"], pilot["file_sha256"]) == (lora["path"], lora["file_sha256"]),
            f"Cell {cell} trained LoRA does not come from its declared pilot checkpoint",
        )

    joint_base = {
        key: reports["B"]["pilot_checkpoint"].get(key)
        for key in ("path", "file_sha256", "branch", "checkpoint_head_state_sha256")
    }
    for cell in ("C", "D"):
        actual = {
            key: reports[cell]["pilot_checkpoint"].get(key)
            for key in ("path", "file_sha256", "branch", "checkpoint_head_state_sha256")
        }
        _require(actual == joint_base, f"Cell {cell} is not based on the exact joint pilot checkpoint")

    source_ids, source_hash = _validate_prediction_records(reports, width=width)
    return {
        "passed": True,
        "report_schema": REPORT_SCHEMA,
        "strict_b1_qwen_forwards": True,
        "pose_inputs_removed": True,
        "blank_identity_gates_passed": True,
        "history_permutation_equivariance_gates_passed": True,
        "single_anchor_swap_untargeted_routing_exact": True,
        "factorial_checkpoint_sources_cross_matched": True,
        "source_samples": len(source_ids),
        "source_sample_identity_sha256": source_hash,
    }


def _empty_counts() -> dict[str, list[float]]:
    return {metric: [0.0, 0.0] for metric in METRICS}


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
    record: dict[str, Any],
    *,
    width: int,
    slots: Iterable[int] | None = None,
) -> dict[str, list[float]]:
    """Return sufficient statistics matching the pilot's compact metrics."""
    logits = np.asarray(record["visibility_logits"], dtype=np.float64)
    gt_visibility = np.asarray(record["gt_visibility"], dtype=np.float64)
    pred_xy = np.asarray(record["pred_xy"], dtype=np.int64)
    gt_xy = np.asarray(record["gt_xy"], dtype=np.int64)
    counts = _empty_counts()
    selected_slots = range(4) if slots is None else slots

    for history_slot in selected_slots:
        positives = np.flatnonzero(gt_visibility[history_slot] > 0.5)
        if positives.size == 0:
            continue
        if positives.size != 1:
            raise RuntimeError("Compact identity summary requires exactly one positive view")
        gt_view = int(positives[0])
        predicted_view = int(np.argmax(logits[history_slot]))

        for threshold, metric in ((4.0, "conditional_pck4"), (8.0, "conditional_pck8")):
            error = _circular_error(
                gt_view,
                pred_xy[history_slot, gt_view],
                gt_view,
                gt_xy[history_slot, gt_view],
                width=width,
            )
            counts[metric][0] += float(error <= threshold)
            counts[metric][1] += 1.0

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
        for threshold, metric in ((4.0, "true_joint_pck4"), (8.0, "true_joint_pck8")):
            counts[metric][0] += float(view_correct and joint_error <= threshold)
            counts[metric][1] += 1.0

        candidates: list[tuple[float, int]] = []
        for target_slot in range(4):
            target_views = np.flatnonzero(gt_visibility[target_slot] > 0.5)
            if target_views.size == 0:
                continue
            if target_views.size != 1:
                raise RuntimeError("Compact identity summary requires exactly one target view")
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
        _require(bool(candidates), "No visible identity target in compact record")
        predicted_identity = min(candidates)[1]
        counts["anchor_identity_accuracy"][0] += float(predicted_identity == history_slot)
        counts["anchor_identity_accuracy"][1] += 1.0
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
        slots = [int(target_slot)] if targeted_swap else None
        if targeted_swap:
            _require(target_slot in range(4), f"Invalid swap target slot for {sample_id}")
            seen[sample_id].append(int(target_slot))
        else:
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
    _require(denominator > 0, "Metric has zero denominator")
    return float(contribution[:, 0].sum() / denominator)


def _bootstrap_distribution(contribution: np.ndarray, weights: np.ndarray) -> np.ndarray:
    numerator = weights @ contribution[:, 0]
    denominator = weights @ contribution[:, 1]
    _require(bool(np.all(denominator > 0)), "A source-sample bootstrap replicate has zero denominator")
    return numerator / denominator


def _interval(values: np.ndarray) -> list[float]:
    low, high = np.percentile(values, (2.5, 97.5))
    return [float(low), float(high)]


def metric_summary(contribution: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    distribution = _bootstrap_distribution(contribution, weights)
    return {
        "estimate": _point(contribution),
        "ci95": _interval(distribution),
        "numerator": float(contribution[:, 0].sum()),
        "denominator": float(contribution[:, 1].sum()),
    }


def difference_summary(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    distribution = _bootstrap_distribution(left, weights) - _bootstrap_distribution(right, weights)
    ci95 = _interval(distribution)
    return {
        "estimate": _point(left) - _point(right),
        "ci95": ci95,
        "ci95_excludes_zero": bool(ci95[0] > 0.0 or ci95[1] < 0.0),
    }


def _reported_metric(evaluation: dict[str, Any], metric: str) -> float:
    if metric == "anchor_identity_accuracy":
        value = evaluation.get("anchor_identity", {}).get("accuracy")
    else:
        key = {
            "visible_view_accuracy": "visible_view_accuracy",
            "conditional_pck4": "pck4",
            "conditional_pck8": "pck8",
            "true_joint_pck4": "joint_pck4",
            "true_joint_pck8": "joint_pck8",
        }[metric]
        value = evaluation.get(key)
    _require(isinstance(value, (int, float)), f"Evaluation is missing reported metric {metric}")
    return float(value)


def _validate_recomputed_metrics(
    reports: dict[str, dict[str, Any]],
    contributions: dict[str, dict[str, dict[str, np.ndarray]]],
) -> None:
    for cell in CELL_NAMES:
        evaluations = reports[cell]["evaluations"]
        for intervention in ("standard", "history-shuffle", "current-shuffle"):
            for metric in METRICS:
                actual = _point(contributions[cell][intervention][metric])
                reported = _reported_metric(evaluations[intervention], metric)
                _require(
                    math.isclose(actual, reported, rel_tol=0.0, abs_tol=1e-12),
                    f"Cell {cell} {intervention} compact/reported {metric} mismatch: "
                    f"compact={actual} report={reported}",
                )
        targeted = evaluations["single-anchor-swap"].get("targeted_slot_metrics")
        _require(isinstance(targeted, dict), f"Cell {cell} has no targeted swap metrics")
        for metric in METRICS:
            actual = _point(contributions[cell]["single-anchor-swap"][metric])
            reported = _reported_metric(targeted, metric)
            _require(
                math.isclose(actual, reported, rel_tol=0.0, abs_tol=1e-12),
                f"Cell {cell} targeted-swap compact/reported {metric} mismatch",
            )


def summarize(
    reports: dict[str, dict[str, Any]],
    *,
    report_paths: dict[str, str | Path] | None = None,
    bootstrap_samples: int = 50_000,
    seed: int = 42,
    heatmap_width: int = 64,
) -> dict[str, Any]:
    _require(bootstrap_samples > 0, "bootstrap_samples must be positive")
    contract = validate_contracts(reports, width=heatmap_width)
    source_ids = [str(record["sample_id"]) for record in reports["A"]["prediction_records"]["standard"]]
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(source_ids), 1.0 / len(source_ids), dtype=np.float64)
    weights = rng.multinomial(len(source_ids), probabilities, size=bootstrap_samples).astype(
        np.float64,
        copy=False,
    )

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
            "composition": CELL_EXPECTATIONS[cell]["label"],
            "standard": {
                metric: metric_summary(contributions[cell]["standard"][metric], weights) for metric in METRICS
            },
            "all_same_peak_fraction": metric_summary(collapse[cell], weights),
        }
        causal_effects[cell] = {}
        for intervention in CAUSAL_INTERVENTIONS:
            causal_effects[cell][f"standard_minus_{intervention}"] = {
                "comparison_contract": (
                    "standard minus the swapped target-slot outputs, grouped once per source sample"
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
    for name, left, right in FACTORIAL_COMPARISONS:
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

    b_identity = cells["B"]["standard"]["anchor_identity_accuracy"]["estimate"]
    b_conditional_pck8 = cells["B"]["standard"]["conditional_pck8"]["estimate"]
    b_true_joint_pck8 = cells["B"]["standard"]["true_joint_pck8"]["estimate"]
    b_history_drop = causal_effects["B"]["standard_minus_history-shuffle"]["metrics"]["anchor_identity_accuracy"][
        "estimate"
    ]
    b_current_joint_pck8_drop = causal_effects["B"]["standard_minus_current-shuffle"]["metrics"]["true_joint_pck8"][
        "estimate"
    ]
    b_single_swap_identity_drop = causal_effects["B"]["standard_minus_single-anchor-swap"]["metrics"][
        "anchor_identity_accuracy"
    ]["estimate"]
    b_collapse = cells["B"]["all_same_peak_fraction"]["estimate"]
    b_a_identity = comparisons["B-A"]["metrics"]["anchor_identity_accuracy"]["estimate"]
    b_a_identity_ci = comparisons["B-A"]["metrics"]["anchor_identity_accuracy"]["ci95"]
    b_c_identity = comparisons["B-C"]["metrics"]["anchor_identity_accuracy"]["estimate"]
    b_c_identity_ci = comparisons["B-C"]["metrics"]["anchor_identity_accuracy"]["ci95"]
    d_a_identity = comparisons["D-A"]["metrics"]["anchor_identity_accuracy"]["estimate"]
    d_a_identity_ci = comparisons["D-A"]["metrics"]["anchor_identity_accuracy"]["ci95"]
    b_swap_targeted = reports["B"]["evaluations"]["single-anchor-swap"]["paired_output_change_vs_standard"]["targeted"]
    b_targeted_heatmap_l1 = float(b_swap_targeted["mean_heatmap_l1"])
    b_targeted_peak_displacement = float(b_swap_targeted["mean_peak_displacement"])
    checks = {
        "B_standard_identity_at_least_0_35": {
            "passed": b_identity >= IDENTITY_MINIMUM,
            "value": b_identity,
            "threshold": IDENTITY_MINIMUM,
            "operator": ">=",
        },
        "B_standard_conditional_PCK8_at_least_0_30": {
            "passed": b_conditional_pck8 >= CONDITIONAL_PCK8_MINIMUM,
            "value": b_conditional_pck8,
            "threshold": CONDITIONAL_PCK8_MINIMUM,
            "operator": ">=",
        },
        "B_standard_true_joint_PCK8_at_least_0_10": {
            "passed": b_true_joint_pck8 >= TRUE_JOINT_PCK8_MINIMUM,
            "value": b_true_joint_pck8,
            "threshold": TRUE_JOINT_PCK8_MINIMUM,
            "operator": ">=",
        },
        "B_history_shuffle_identity_drop_at_least_0_05": {
            "passed": b_history_drop >= HISTORY_SHUFFLE_IDENTITY_DROP_MINIMUM,
            "value": b_history_drop,
            "threshold": HISTORY_SHUFFLE_IDENTITY_DROP_MINIMUM,
            "operator": ">=",
        },
        "B_current_shuffle_true_joint_PCK8_drop_at_least_0_05": {
            "passed": (b_current_joint_pck8_drop >= CURRENT_SHUFFLE_TRUE_JOINT_PCK8_DROP_MINIMUM),
            "value": b_current_joint_pck8_drop,
            "threshold": CURRENT_SHUFFLE_TRUE_JOINT_PCK8_DROP_MINIMUM,
            "operator": ">=",
        },
        "B_targeted_single_swap_identity_drop_at_least_0_05": {
            "passed": b_single_swap_identity_drop >= SINGLE_SWAP_IDENTITY_DROP_MINIMUM,
            "value": b_single_swap_identity_drop,
            "threshold": SINGLE_SWAP_IDENTITY_DROP_MINIMUM,
            "operator": ">=",
        },
        "B_targeted_single_swap_heatmap_L1_positive": {
            "passed": b_targeted_heatmap_l1 > TARGETED_OUTPUT_CHANGE_MINIMUM_EXCLUSIVE,
            "value": b_targeted_heatmap_l1,
            "threshold": TARGETED_OUTPUT_CHANGE_MINIMUM_EXCLUSIVE,
            "operator": ">",
        },
        "B_targeted_single_swap_peak_displacement_positive": {
            "passed": (b_targeted_peak_displacement > TARGETED_OUTPUT_CHANGE_MINIMUM_EXCLUSIVE),
            "value": b_targeted_peak_displacement,
            "threshold": TARGETED_OUTPUT_CHANGE_MINIMUM_EXCLUSIVE,
            "operator": ">",
        },
        "B_all_same_peak_fraction_below_0_40": {
            "passed": b_collapse < ALL_SAME_PEAK_FRACTION_MAXIMUM,
            "value": b_collapse,
            "threshold": ALL_SAME_PEAK_FRACTION_MAXIMUM,
            "operator": "<",
        },
        "B_minus_A_identity_material": {
            "passed": (b_a_identity >= JOINT_VS_HEAD_IDENTITY_DELTA_MINIMUM and b_a_identity_ci[0] > 0.0),
            "value": b_a_identity,
            "threshold": JOINT_VS_HEAD_IDENTITY_DELTA_MINIMUM,
            "operator": ">= and ci95 lower bound > 0",
            "ci95": b_a_identity_ci,
        },
        "B_minus_C_pure_LoRA_identity_gain": {
            "passed": (b_c_identity >= PURE_LORA_IDENTITY_DELTA_MINIMUM and b_c_identity_ci[0] > 0.0),
            "value": b_c_identity,
            "threshold": PURE_LORA_IDENTITY_DELTA_MINIMUM,
            "operator": ">= and ci95 lower bound > 0",
            "ci95": b_c_identity_ci,
        },
        "D_minus_A_retains_LoRA_identity_gain": {
            "passed": (d_a_identity >= LORA_RETENTION_IDENTITY_DELTA_MINIMUM and d_a_identity_ci[0] > 0.0),
            "value": d_a_identity,
            "threshold": LORA_RETENTION_IDENTITY_DELTA_MINIMUM,
            "operator": ">= and ci95 lower bound > 0",
            "ci95": d_a_identity_ci,
        },
    }
    overall = all(check["passed"] for check in checks.values())

    if report_paths is None:
        inputs = {cell: {"path": None, "file_sha256": None} for cell in CELL_NAMES}
    else:
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
            "single_anchor_swap_replicas_are_grouped_with_their_source_sample": True,
        },
        "metric_contract": {
            "conditional_pck": "peak error in the ground-truth visible view",
            "true_joint_pck": "predicted view must be visible and its peak must meet the pixel threshold",
            "anchor_identity": "nearest of the four GT panorama targets to each history-conditioned output peak",
            "heatmap_view_width": heatmap_width,
        },
        "decision_threshold_contract": {
            "schema": SUMMARY_SCHEMA,
            "predeclared_and_not_cli_tunable": True,
            "identity_minimum": IDENTITY_MINIMUM,
            "conditional_pck8_minimum": CONDITIONAL_PCK8_MINIMUM,
            "true_joint_pck8_minimum": TRUE_JOINT_PCK8_MINIMUM,
            "history_shuffle_identity_drop_minimum": HISTORY_SHUFFLE_IDENTITY_DROP_MINIMUM,
            "current_shuffle_true_joint_pck8_drop_minimum": (CURRENT_SHUFFLE_TRUE_JOINT_PCK8_DROP_MINIMUM),
            "single_swap_identity_drop_minimum": SINGLE_SWAP_IDENTITY_DROP_MINIMUM,
            "all_same_peak_fraction_maximum_exclusive": ALL_SAME_PEAK_FRACTION_MAXIMUM,
            "joint_vs_head_identity_delta_minimum": JOINT_VS_HEAD_IDENTITY_DELTA_MINIMUM,
            "pure_lora_identity_delta_minimum": PURE_LORA_IDENTITY_DELTA_MINIMUM,
            "lora_retention_identity_delta_minimum": LORA_RETENTION_IDENTITY_DELTA_MINIMUM,
            "targeted_output_change_minimum_exclusive": (TARGETED_OUTPUT_CHANGE_MINIMUM_EXCLUSIVE),
            "untargeted_output_change_exact": UNTARGETED_OUTPUT_CHANGE_EXACT,
            "single_swap_routing_contract": PAIRED_SWAP_CONTRACT,
        },
        "cells": cells,
        "factorial_comparisons": comparisons,
        "causal_effects": causal_effects,
        "decision_gate": {
            "checks": checks,
            "overall_passed": overall,
            "stage2_stage3_authorized_by_this_gate": overall,
            "failure_action": (
                None if overall else "Stop this anchor-token route; do not use this pilot as VLM-grounding evidence."
            ),
        },
    }


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = {
        "A": args.a_report,
        "B": args.b_report,
        "C": args.c_report,
        "D": args.d_report,
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
    print(json.dumps(summary["decision_gate"], ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
